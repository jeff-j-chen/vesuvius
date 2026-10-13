"""character_failures.py -- where does the current model fail? per-character weights for campaign 41.

Runs the given checkpoint over every campaign-41 training scroll (full frame, the same bounded-memory
predict_tiles path as eval_scrolls.ipynb), scores every 16 px target cell, and evaluates each character
against its own closed ring exactly as training defines them:

- characters: 8-connected components of the regular (full-brightness) ink in dilated_inklabels/<id>.png
  (>= 8 px), keyed by component id -- the same components training uses
- positive cells: cells holding any training ink (dilated_inklabels), assigned to the nearest character
- negative cells: closed-ring cells (ring c2g2s4 from the campaign-41 data config), nearest character

Per character it records recall / ring FPR / F1 at 0.5, AP against its ring, mean scores, cell counts,
split (train/valid/mixed by train_masks), bbox, and a sampling weight

    failure = 1 - (AP + recall) / 2      (1 - recall when the character has no ring cells)
    weight  = 1 + (max_weight - 1) * failure

written to campaign_archs_41.FORGETTING_PATH with a fingerprint of each label file's regular ink (assumed
half-brightness letters do not invalidate it), plus each scroll's
prediction map in --maps-dir (character_failures/maps/<id>.png full frame, black = ink; <id>.npy cell scores).
Campaign 41's `character_failures` arm is skipped without an up-to-date entry for every training scroll. The
file is rewritten after every scroll, so an interrupted run resumes where it stopped.

    python3 character_failures.py
    python3 character_failures.py --checkpoint models/archs40/holdout_n96_combined_surface_norm_nnpu/final.pth
    python3 character_failures.py --only 20260930144758,20260930144760 --force
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import distance_transform_edt, find_objects  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import campaign_archs_41 as campaign41  # noqa: E402
from utils.config import Config, ScrollConfig, resolve_run_config, startup_output  # noqa: E402
from utils.dataloader import DataManager, character_label_fingerprint, drawn_characters  # noqa: E402
from utils.model import create_model  # noqa: E402
from utils.norm import ensure_surface_anchors  # noqa: E402
from utils.visualizer import TensorboardVisualizer, group_by_depth, predict_tiles  # noqa: E402

DEFAULT_CHECKPOINT = "models/archs40/holdout_n96_combined_surface_norm_nnpu/final_best_character.pth"
_AUXILIARY_PREFIXES = ("supcon_head.", "domain_head.")
MIN_PIXELS = 8
CHARACTERS_FROM = "dilated_inklabels regular ink"
WEIGHT_RULE = "1 + (max_weight - 1) * (1 - (ap + recall) / 2); 1 - recall without ring cells"


def model_config(checkpoint: Path, stride: int) -> Config:
    """replay the checkpoint's saved run config, as eval_scrolls.ipynb does."""
    run_config, run_config_path = resolve_run_config(checkpoint)
    config = Config()
    for name, value in run_config["data"].items():
        if name == "scrolls":
            config.data.scrolls = [ScrollConfig(**item) for item in value]
        elif name != "probe_rois" and hasattr(config.data, name):
            setattr(config.data, name, value)
    for name, value in run_config["model"].items():
        if hasattr(config.model, name):
            setattr(config.model, name, value)
    config.data.eval_stride = int(stride)
    config.model.compile_model = False
    config.tra.supcon = False
    config.tra.dann = False
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[failures] run config {run_config_path}", flush=True)
    return config


def load_model(config: Config, checkpoint: Path):
    model, _ = create_model(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("_orig_mod.")
        if not key.startswith(_AUXILIARY_PREFIXES):
            cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return model.to(config.device).eval()


def data_config() -> Config:
    """campaign-41 label, ring and split definitions on the raw (untranslated) volumes the model saw."""
    config = campaign41.build_config(campaign41.TESTS[0])
    config.data.zarr_suffix = {}
    config.data.native_noise = ""
    config.data.extra_inklabels = False
    config.data.preload_volumes = False
    return config


def _cells(array: np.ndarray, unit: int, shape: tuple[int, int], reduce) -> np.ndarray:
    gh, gw = shape
    return reduce(np.asarray(array)[:gh * unit, :gw * unit].reshape(gh, unit, gw, unit), axis=(1, 3))


def _majority_grid(components: np.ndarray, unit: int, shape: tuple[int, int]) -> np.ndarray:
    gh, gw = shape
    grid = np.zeros(shape, dtype=np.int64)
    ys, xs = np.nonzero(components[:gh * unit, :gw * unit])
    if not ys.size:
        return grid
    ids = components[ys, xs].astype(np.int64)
    stride = int(ids.max()) + 1
    keys = ((ys // unit).astype(np.int64) * gw + xs // unit) * stride + ids
    unique, counts = np.unique(keys, return_counts=True)
    cells, comps = unique // stride, unique % stride
    order = np.lexsort((-counts, cells))
    first = np.concatenate(([True], cells[order][1:] != cells[order][:-1]))
    grid.flat[cells[order][first]] = comps[order][first]
    return grid


def predict_scroll(config: Config, model, manager: DataManager, mask01: np.ndarray, scroll_id: int) -> np.ndarray:
    height, width = mask01.shape
    fake = types.SimpleNamespace(c=config)
    coords = TensorboardVisualizer._gen_tile_coords(
        fake, (config.data.d_start, config.data.d_end), (0, height), (0, width), mask01, z_step=config.data.depth,
    )
    grouped = group_by_depth(coords)
    if len(grouped) != 1:
        raise RuntimeError(f"surface-relative inference expected one depth pass, got {len(grouped)}")
    offset = next(iter(grouped))
    surface_dir = Path(config.data.surface_label_dir) / str(scroll_id)
    mean, std, g_min, g_max = manager.norm_stats
    with torch.no_grad():
        return predict_tiles(
            config, model, manager.vol, mask01, grouped[offset], (0, height), (0, width),
            config.data.d_start + offset, f"failures_{scroll_id}", mean, std, g_min, g_max,
            surface_depth_map=np.load(surface_dir / "depth.npy", mmap_mode="r"),
            surface_confidence_map=np.load(surface_dir / "confidence.npy", mmap_mode="r"),
        )


def save_prediction_map(scores: np.ndarray, unit: int, shape: tuple[int, int], out_dir: Path, scroll_id: int) -> Path:
    """cell scores -> <out_dir>/<id>.npy and a full-frame PNG (black = ink, mid grey = not scored)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{scroll_id}.npy", scores.astype(np.float16))
    image = np.where(np.isfinite(scores), 255.0 * (1.0 - np.clip(np.nan_to_num(scores), 0.0, 1.0)), 160.0)
    image = np.repeat(np.repeat(image.astype(np.uint8), unit, axis=0), unit, axis=1)
    frame = np.full(shape, 160, dtype=np.uint8)
    h, w = min(shape[0], image.shape[0]), min(shape[1], image.shape[1])
    frame[:h, :w] = image[:h, :w]
    path = out_dir / f"{scroll_id}.png"
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"could not write {path}")
    return path


def evaluate_scroll(scroll_id: int, model_cfg: Config, model, data_cfg: Config, max_weight: float,
                    maps_dir: Path) -> dict:
    with startup_output():
        ensure_surface_anchors([scroll_id], data_cfg.data.zarr_path, surface_dir=data_cfg.data.surface_label_dir)
        manager = DataManager(data_cfg, scroll_id=scroll_id)
    labels = np.asarray(manager.labels) > 0.5
    mask01 = (np.asarray(manager.mask) > 0.5).astype(np.uint8)
    ring = np.asarray(manager._make_ring_mask()) > 0.5
    height, width = labels.shape
    unit = int(model_cfg.model.multitile_subtile)
    if unit != int(data_cfg.model.multitile_subtile):
        raise ValueError("checkpoint and campaign 41 disagree on the target cell size")
    shape = (height // unit, width // unit)

    started = time.time()
    scores = predict_scroll(model_cfg, model, manager, mask01, scroll_id)[:shape[0], :shape[1]]
    map_path = save_prediction_map(scores, unit, (height, width), maps_dir, scroll_id)
    print(f"[failures] {scroll_id}: prediction map -> {map_path}", flush=True)
    finite = np.isfinite(scores)
    papyrus = _cells(mask01 > 0, unit, shape, np.all)
    positive = _cells(labels, unit, shape, np.any) & papyrus & finite
    negative = _cells(ring, unit, shape, np.all) & ~_cells(labels, unit, shape, np.any) & papyrus & finite
    train_cells = (
        _cells(np.asarray(manager.manual_train_mask) > 0, unit, shape, np.all)
        if manager.manual_train_mask is not None else np.ones(shape, dtype=bool)
    )

    label_path = ROOT / campaign41.INKLABEL_DIR / f"{scroll_id}.png"
    components = drawn_characters(label_path, (height, width), MIN_PIXELS)
    boxes = find_objects(components)
    owner = _majority_grid(components, unit, shape)
    if owner.any():
        _, nearest = distance_transform_edt(owner == 0, return_indices=True)
        owner = owner[nearest[0], nearest[1]]

    characters = {}
    for component_id in range(1, int(components.max()) + 1):
        pos = scores[positive & (owner == component_id)]
        if pos.size == 0:
            continue
        neg_cells = negative & (owner == component_id)
        neg = scores[neg_cells]
        recall = float((pos >= 0.5).mean())
        fpr = float((neg >= 0.5).mean()) if neg.size else 0.0
        tp, fn, fp = int((pos >= 0.5).sum()), int((pos < 0.5).sum()), int((neg >= 0.5).sum())
        ap = float(average_precision_score(
            np.r_[np.ones(pos.size), np.zeros(neg.size)], np.r_[pos, neg]
        )) if neg.size else None
        failure = 1.0 - (0.5 * (ap + recall) if ap is not None else recall)
        cell_split = train_cells[positive & (owner == component_id)]
        box = boxes[component_id - 1]
        characters[str(component_id)] = {
            "weight": round(1.0 + (max_weight - 1.0) * min(max(failure, 0.0), 1.0), 4),
            "failure": round(failure, 4),
            "recall": round(recall, 4),
            "ring_fpr": round(fpr, 4),
            "f1": round(2.0 * tp / max(2 * tp + fp + fn, 1), 4),
            "ap": None if ap is None else round(ap, 4),
            "mean_score": round(float(pos.mean()), 4),
            "mean_ring_score": None if not neg.size else round(float(neg.mean()), 4),
            "positive_cells": int(pos.size),
            "ring_cells": int(neg.size),
            "split": "train" if cell_split.all() else "valid" if not cell_split.any() else "mixed",
            "bbox_yxyx": [int(box[0].start), int(box[1].start), int(box[0].stop), int(box[1].stop)],
        }
    failures = [stats["failure"] for stats in characters.values()]
    summary = {
        "characters": len(characters),
        "mean_failure": round(float(np.mean(failures)), 4) if failures else None,
        "upweighted_over_2": sum(stats["weight"] > 2.0 for stats in characters.values()),
        "seconds": round(time.time() - started, 1),
    }
    del manager, labels, mask01, ring, scores, components
    return {
        "label_fingerprint": character_label_fingerprint(label_path),
        "summary": summary,
        "characters": characters,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="per-character failure weights for campaign 41")
    parser.add_argument("--checkpoint", default=os.getenv("VESUVIUS_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default=campaign41.FORGETTING_PATH)
    parser.add_argument("--only", default=None, help="comma-separated scroll ids (default: all training scrolls)")
    parser.add_argument("--stride", type=int, default=32, help="inference window step in px (16 = densest)")
    parser.add_argument("--max-weight", type=float, default=4.0)
    parser.add_argument("--force", action="store_true", help="recompute scrolls that already have a current entry")
    parser.add_argument("--maps-dir", default="character_failures/maps",
                        help="prediction PNG/.npy per scroll; scrolls without a map are recomputed")
    args = parser.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    scroll_ids = campaign41.training_scroll_ids()
    if args.only:
        wanted = [int(value) for value in args.only.split(",") if value.strip()]
        unknown = sorted(set(wanted) - set(scroll_ids))
        if unknown:
            raise ValueError(f"not campaign-41 training scrolls: {unknown}")
        scroll_ids = wanted

    out = ROOT / args.out
    payload = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    if (payload.get("checkpoint") != str(checkpoint.relative_to(ROOT)) or payload.get("max_weight") != args.max_weight
            or payload.get("characters_from") != CHARACTERS_FROM):
        if payload:
            backup = out.with_suffix(".previous.json")
            os.replace(out, backup)
            print(f"[failures] incompatible {out.name} moved to {backup.name}", flush=True)
        payload = {}
    payload.update({
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "max_weight": args.max_weight,
        "min_pixels": MIN_PIXELS,
        "weight_rule": WEIGHT_RULE,
        "stride": args.stride,
        "characters_from": CHARACTERS_FROM,
    })
    payload.setdefault("scrolls", {})

    model_cfg = model_config(checkpoint, args.stride)
    model = load_model(model_cfg, checkpoint)
    data_cfg = data_config()
    maps_dir = ROOT / args.maps_dir
    for index, scroll_id in enumerate(scroll_ids, 1):
        label_path = ROOT / campaign41.INKLABEL_DIR / f"{scroll_id}.png"
        entry = payload["scrolls"].get(str(scroll_id))
        if (not args.force and entry is not None and label_path.is_file()
                and (maps_dir / f"{scroll_id}.png").is_file()
                and entry.get("label_fingerprint") == character_label_fingerprint(label_path)):
            print(f"[failures] {index}/{len(scroll_ids)} {scroll_id}: current entry kept", flush=True)
            continue
        print(f"[failures] {index}/{len(scroll_ids)} {scroll_id}: predicting", flush=True)
        payload["scrolls"][str(scroll_id)] = evaluate_scroll(
            scroll_id, model_cfg, model, data_cfg, args.max_weight, maps_dir,
        )
        payload["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write(out, payload)
        print(f"[failures] {scroll_id}: {payload['scrolls'][str(scroll_id)]['summary']}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[failures] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
