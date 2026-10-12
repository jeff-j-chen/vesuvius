"""Audit real letter-finetune inputs and optionally test fixed-batch fitting.

Run from the repository root: python3 old/diagnose_finetune_letters.py --steps 100
This is a capacity probe, not a generalization test or the campaign's full objective.
Uses the loader's normal caches; no labels, volumes, or model checkpoints are written.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import campaign_finetune_letters as campaign
import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score

from utils.dataloader import DataManager
from utils.model import create_model
from utils.training_utils import create_optimizer_and_scheduler


def compare_focus():
    import types

    from PIL import Image, ImageDraw, ImageFont

    sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    output = campaign.ROOT / "output" / "letters_focus"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for tid in campaign.FOCUS_OVERRIDES:
        directories = list((campaign.ROOT / "runs_finetune_letters").glob(f"ftl_{tid}_*"))
        if not directories:
            raise FileNotFoundError(f"no run for {tid}")
        directory = max(directories, key=lambda path: path.stat().st_mtime)
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0, "images": 1, "histograms": 1}).Reload()
        tags = accumulator.Tags()["scalars"]
        wanted = ["Per_Scroll/PR_AUC_Train/pherc0846a", "Per_Scroll/PR_AUC_Train/pherc0009b",
                  "G_M/Loss/Train_Raw", "Learning_Rate",
                  *(f"R_M/s{sid}/Train/RecallAt1PctFPR" for sid in campaign.FOCUS_VIS_IDS)]
        report[tid] = {"run": str(directory.relative_to(campaign.ROOT)), "scalars": {}}
        for tag in wanted:
            if tag in tags:
                report[tid]["scalars"][tag] = [event.value for event in accumulator.Scalars(tag)]
        assert len(report[tid]["scalars"]["G_M/Loss/Train_Raw"]) == campaign.N_EPOCHS, tid
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    for sid in campaign.FOCUS_VIS_IDS:
        paths = [campaign.ROOT / result["run"] / "eval_figs" / f"eval_s{sid}_ep06.png"
                 for result in report.values()]
        with Image.open(paths[0]) as source:
            reference = np.asarray(source.convert("RGB"))
        half = reference[:reference.shape[0] // 2, :reference.shape[1] // 2]
        nonwhite = (half.min(axis=2) < 245).astype(np.uint8)
        contours, _ = cv2.findContours(nonwhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left, top, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        target_width = 460
        target_height = round(height * target_width / width)
        sheet = Image.new("RGB", (4 * target_width, 2 * (target_height + 45)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (tid, path) in enumerate(zip(report, paths)):
            with Image.open(path) as source:
                assert source.size == (reference.shape[1], reference.shape[0]), path
                panel = source.convert("RGB").crop((left, top, left + width, top + height))
                panel = panel.resize((target_width, target_height), Image.Resampling.LANCZOS)
            column, row = index % 4, index // 4
            origin = (column * target_width, row * (target_height + 45))
            draw.text((origin[0] + 8, origin[1] + 10), tid.removeprefix("focus_"), fill="black", font=font)
            sheet.paste(panel, (origin[0], origin[1] + 45))
        path = output / f"{sid}_comparison.png"
        sheet.save(path)
        print(path.relative_to(campaign.ROOT), sheet.size)
    print(output / "metrics.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=campaign.DIAGNOSTIC_BASE, choices=campaign.ARMS)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--init-weights", default=campaign.DEFAULT_INIT_WEIGHTS)
    parser.add_argument("--compare-focus", action="store_true", help="compare completed focused runs without training")
    args = parser.parse_args()
    if args.compare_focus:
        compare_focus()
        return
    if args.samples < 8 or args.steps < 0:
        parser.error("samples must be >=8 and steps must be >=0")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    config = campaign.build_config(campaign._test(args.arm, args.init_weights))
    config.data.vis_preload_persistent = False
    config.data.selective_chunk_preload = False
    config.dl.num_workers = 0
    fixed = []
    for index, scroll in enumerate(config.data.scrolls):
        scroll_id = int(scroll.scroll_id)
        manager = DataManager(config, scroll_id=scroll_id, character_namespace=index)
        dataset, valid = manager.get_datasets()
        coords = dataset._character_balanced_coords()[:args.samples]
        assert coords, f"no sampled coordinates: {scroll_id}"
        positive_targets, cell_counts, starts, ranges, saturation = [], [], [], [], []
        selected = {True: 0, False: 0}
        for z_off, y_off, x_off in coords:
            mask = dataset._fetch_mask(y_off, x_off)
            labels = dataset._fetch_label(y_off, x_off)
            if dataset._edge_soft_sigma > 0:
                labels = dataset._soften_edge_positives(labels, y_off, x_off)
            block, offset, shift, _ = dataset._fetch_block(z_off, y_off, x_off, allow_jitter=False)
            depth, confidence = dataset._fetch_surface_teacher(z_off, y_off, x_off, offset, shift)
            start = dataset.z_start + z_off + shift
            starts.append(start)
            assert np.isfinite(block).all() and block.std() > 0, (scroll_id, "invalid input")
            pad = (config.data.context_size - config.data.tile_size) // 2
            y0, x0 = dataset.y_start + y_off - pad, dataset.x_start + x_off - pad
            if y0 >= 0 and x0 >= 0:
                raw = np.asarray(manager.vol[start:start + config.data.depth,
                                             y0:y0 + config.data.context_size,
                                             x0:x0 + config.data.context_size], dtype=np.float32)
                if raw.shape == block.shape:
                    np.testing.assert_allclose(block, dataset._normalize_block(raw), atol=1e-6)
            active = mask > 0
            positive = (labels > 0.5) & active
            negative = (labels <= 0.5) & (labels >= -1.5) & active
            unlabeled = (labels < -1.5) & active
            positive_targets.extend(labels[positive].tolist())
            cell_counts.append([int(positive.sum()), int(negative.sum()), int(unlabeled.sum())])
            ranges.append([float(block.min()), float(block.max()), float(block.std())])
            saturation.append([float((block <= 0).mean()), float((block >= 1).mean())])
            has_ink = bool(positive.any())
            if scroll_id != campaign.PHERC0009B_ID and selected[has_ink] < 4 and (positive | negative).any():
                fixed.append((torch.from_numpy(block).unsqueeze(0), labels.clamp(min=0),
                              (positive | negative).float(), torch.from_numpy(depth), torch.from_numpy(confidence)))
                selected[has_ink] += 1
        report = {
            "scroll": scroll_id, "windows_per_epoch": len(dataset), "validation_windows": len(valid),
            "sampleable_characters": len(dataset._sampled_character_ids()),
            "unique_sampled_windows": len(set(coords)), "depth_starts": sorted(set(starts)),
            "mean_cells_pos_ring_far": np.mean(cell_counts, axis=0).tolist(),
            "positive_target_quantiles": np.quantile(positive_targets, [0, .25, .5, .75, 1]).tolist() if positive_targets else [],
            "mean_input_min_max_std": np.mean(ranges, axis=0).tolist(),
            "mean_clipped_low_high": np.mean(saturation, axis=0).tolist(),
        }
        print("AUDIT", json.dumps(report), flush=True)
    if not args.steps:
        return
    assert fixed, "no PHerc0846A windows for fitting"
    images, targets, masks, depths, confidences = [
        torch.stack(values).to(config.device) for values in zip(*fixed)
    ]
    print("FIXED", len(fixed), "windows", int((targets[masks > 0] > .5).sum()), "positive cells", flush=True)
    state = torch.load(args.init_weights, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {name.removeprefix("_orig_mod."): value for name, value in state.items()}
    for freeze in (True, False):
        trial = copy.deepcopy(config)
        trial.tra.freeze_prefixes = campaign.FREEZE_PREFIXES if freeze else ()
        torch.manual_seed(41)
        model, _ = create_model(trial)
        model.load_state_dict(state, strict=True)
        optimizer, _ = create_optimizer_and_scheduler(model, trial)
        print("FIT", "frozen" if freeze else "unfrozen", "trainable",
              sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), flush=True)
        for step in range(args.steps + 1):
            model.train()
            logits = model(images, teacher_surface_depth=depths, teacher_surface_confidence=confidences)
            loss = (functional.binary_cross_entropy_with_logits(logits, targets, reduction="none") * masks).sum() / masks.sum()
            if step in {0, args.steps // 2, args.steps}:
                with torch.no_grad():
                    active = masks > 0
                    truth = (targets[active] > .5).cpu().numpy()
                    train_ap = average_precision_score(truth, logits[active].detach().cpu().numpy())
                    model.eval()
                    eval_logits = model(images, teacher_surface_depth=depths, teacher_surface_confidence=confidences)
                    eval_ap = average_precision_score(truth, eval_logits[active].cpu().numpy())
                    print("FIT_STEP", json.dumps({"frozen": freeze, "step": step, "bce": float(loss.detach()),
                                                "train_ap": train_ap, "eval_ap": eval_ap}), flush=True)
            if step < args.steps:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), trial.tra.grad_norm)
                optimizer.step()
        del optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()