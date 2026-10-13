"""Compare completed overnight runs and regenerate affected figures without training."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

import cv2
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import campaign_letters_overnight as campaign

OUTPUT = ROOT / "output" / "letters_overnight_review"
OLD_RUN = ROOT / "runs_archs41" / "41_and_mask_11_03-15-22"


def scalars(folder):
    accumulator = EventAccumulator(str(folder), size_guidance={"scalars": 0, "images": 1, "histograms": 1}).Reload()
    return {tag: [{"step": event.step, "value": event.value} for event in accumulator.Scalars(tag)]
            for tag in accumulator.Tags()["scalars"]}


def audit_folds():
    from utils.dataloader import split_extra_labels

    inventory = campaign.verify_snapshot()
    report = {}
    for sid, components in inventory["components"].items():
        source = cv2.imread(str(campaign.INPUT_ROOT / "inklabels" / f"{sid}.png"), 0)
        regular, _ = split_extra_labels(source)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(regular.astype(np.uint8), 8)
        kept = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 8) + 1
        masks = [cv2.imread(str(campaign.INPUT_ROOT / "train_masks" / f"letters_fold_{fold}" / f"{sid}.png"), 0)
                 for fold in range(1, 6)]
        entries = []
        for component in components:
            pixels = labels == kept[component["component"] - 1]
            fractions = [float((mask[pixels] >= 240).mean()) for mask in masks]
            expected = [0.0 if fold == component["fold"] else 1.0 for fold in range(1, 6)]
            if fractions != expected:
                raise AssertionError((sid, component["component"], fractions, expected))
            entries.append({**component, "training_fraction_by_fold": fractions})
        report[sid] = entries
    return report


def collect():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {
        "supervised_old": scalars(OLD_RUN),
        "supervised_new": scalars(campaign.LOG_ROOT / "supervised" / "and_mask"),
        "finetunes_original": {tid: scalars(campaign.LOG_ROOT / "finetune" / tid) for tid in campaign.FINE_TUNES},
        "fold_audit": audit_folds(),
    }
    before = json.loads((OLD_RUN / "config.json").read_text())
    after = json.loads((campaign.LOG_ROOT / "supervised" / "and_mask" / "config.json").read_text())
    result["config_differences"] = {
        f"{section}.{key}": {"old": before[section].get(key), "new": after[section].get(key)}
        for section in ("data", "dl", "model", "tra")
        for key in set(before[section]) | set(after[section])
        if before[section].get(key) != after[section].get(key)
    }
    (OUTPUT / "metrics_and_folds.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[review] collected metrics and verified {sum(map(len, result['fold_audit'].values()))} components", flush=True)
    return result


def replay_config(tid):
    from utils.config import Config, ScrollConfig

    saved = json.loads((campaign.MODEL_ROOT / "finetune" / tid / "experiment.json").read_text())
    config = Config()
    for section, values in saved.items():
        if section not in ("data", "dl", "model", "tra", "hm"):
            setattr(config, section, values)
            continue
        for key, value in values.items():
            if section == "data" and key == "scrolls":
                value = [ScrollConfig(**entry) for entry in value]
            elif section == "data" and key == "probe_rois":
                value = {}
            elif key == "surface_window_offset_by_scroll":
                value = {int(sid): offset for sid, offset in value.items()}
            setattr(getattr(config, section), key, value)
    config.data.selective_chunk_preload = False
    config.data.vis_preload_persistent = False
    config.data.ram_safe_vis = True
    config.model.compile_model = False
    config.tra.log_dir = str(OUTPUT / "corrected")
    config.run_name = tid
    return config


def rerender(tid):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from utils.model import create_model
    from utils.visualizer import TensorboardVisualizer

    config = replay_config(tid)
    model, _ = create_model(config)
    model.load_state_dict(torch.load(config.save_final, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    targets = list(campaign.TRAIN_IDS) if tid.startswith("letters_fold_") else [campaign.TRAIN_IDS[0]]
    if tid.startswith("loo_"):
        targets = list(dict.fromkeys([*targets, int(tid.removeprefix("loo_"))]))
    folder = OUTPUT / "corrected" / tid
    folder.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(str(folder)) as writer:
        for sid in targets:
            print(f"[review] rendering {tid} / {sid}", flush=True)
            visualizer = TensorboardVisualizer(config, mode="train", scroll_id=sid, shared_writer=writer,
                                              tag_prefix=f"s{sid}/", load_test_frags=False)
            visualizer.load_visualization_volume()
            visualizer.add_evaluation_figures(config.tra.n_epochs - 1, model)
            visualizer.release_visualization_volume()
            del visualizer
            gc.collect()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scalars(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerender", action="store_true")
    parser.add_argument("--only", choices=campaign.FINE_TUNES)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    collect()
    if args.rerender:
        corrected = {}
        path = OUTPUT / "corrected_metrics.json"
        if path.is_file():
            corrected = json.loads(path.read_text())
        for tid in ([args.only] if args.only else campaign.FINE_TUNES):
            corrected[tid] = rerender(tid)
            path.write_text(json.dumps(corrected, indent=2) + "\n")
        print(f"[review] corrected figures and metrics saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()