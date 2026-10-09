"""side-by-side depth-orientation test with one fixed model.

    A: normal zarr,  normal surface (surface_labels/: last papyrus slice before papyrus->air)
    B: normal zarr,  flipped surface direction (first papyrus slice after air->papyrus)
    C: depth-flipped zarr, normal surface detector run on the flipped zarr

B and C come from one detector run on the flipped zarr: its surface c maps to b = D-1-c in the
original frame. labelled scrolls get tile-level ROC/PR AUC (whole mask and manual-validation
cells); every scroll gets a figure under output/orientation_abc/.

    python3 crossres/orientation_abc.py [--ids 20260930144758,...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from sklearn.metrics import average_precision_score, roc_auc_score

import campaign_finetune as cf
from utils.model import create_model
from utils.visualizer import TensorboardVisualizer, group_by_depth, predict_tiles

MODEL_PATH = "/vesuvius/models/archs40/holdout_n96_combined_surface_norm_nnpu/final.pth"
TMP = ROOT / "_ves_tmp" / "orientation_abc"
OUT = ROOT / "output" / "orientation_abc"


def flipped_zarr(sid: int, raw: np.ndarray) -> Path:
    path = TMP / "zarrs" / f"{sid}.zarr"
    if not (path / ".zarray").is_file():
        store = zarr.open(str(path), mode="w", shape=raw.shape, chunks=(8, 64, 64),
                          dtype=raw.dtype, compressor=None)
        store[:] = raw[::-1]
    return path


def flipped_surface(sid: int, depth: int) -> tuple[np.ndarray, np.ndarray]:
    out_dir = TMP / "surface_flipped"
    folder = out_dir / str(sid)
    if not (folder / "metadata.json").is_file():
        meta = json.loads((ROOT / "surface_labels" / str(sid) / "metadata.json").read_text())
        subprocess.run([
            sys.executable, "generate_surface_supervision.py", "--scroll-id", str(sid),
            "--zarr-dir", str(TMP / "zarrs"), "--mask-dir", "masks",
            "--z-start", str(depth - int(meta["z_end"])), "--z-end", str(depth - int(meta["z_start"])),
            "--output-dir", str(out_dir), "--review-dir", str(TMP / "review"),
        ], check=True, stdout=subprocess.DEVNULL)
    return np.load(folder / "depth.npy"), np.load(folder / "confidence.npy")


def load_model(config):
    model, _ = create_model(config)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing, _ = model.load_state_dict(compatible, strict=False)
    print(f"[model] loaded {len(compatible)}/{len(state)} tensors, missing={len(missing)}", flush=True)
    return model.to(config.device).eval()


def grid(image: np.ndarray, shape: tuple[int, int], cell: int) -> np.ndarray:
    h, w = shape
    block = image[:h * cell, :w * cell].astype(np.float32)
    return block.reshape(h, cell, w, cell).mean(axis=(1, 3))


def scores(pred, labels, mask, valid):
    out = {}
    for name, region in (("all", mask), ("valid", mask & valid)):
        y, p = labels[region], pred[region]
        keep = np.isfinite(p)
        y, p = y[keep], p[keep]
        if y.size and 0 < y.mean() < 1:
            out[name] = {"roc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
                         "pos": float(y.mean())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    default_ids = [int(s.scroll_id) for s in cf.SCROLLS] + list(cf.LABEL_FREE_VIS_IDS)
    parser.add_argument("--ids", default=",".join(map(str, default_ids)))
    args = parser.parse_args()
    ids = [int(v) for v in args.ids.split(",") if v.strip()]

    config = cf.build_config(cf._test(MODEL_PATH))
    config.data.vis_scroll_ids = ids
    config.data.vis_preload_persistent = False
    config.tra.log_dir = str(TMP / "tb")
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config)
    cell = int(getattr(config.model, "multitile_subtile", 8))
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    for sid in ids:
        vis = TensorboardVisualizer(config, mode="train", scroll_id=sid, load_test_frags=False)
        raw = np.ascontiguousarray(zarr.open(f"{config.data.zarr_path}/{sid}.zarr", mode="r")[:])
        depth = int(raw.shape[0])
        flipped = np.ascontiguousarray(raw[::-1])
        flipped_zarr(sid, raw)
        c_depth, c_conf = flipped_surface(sid, depth)
        b_depth = np.where(c_depth != 255, depth - 1 - c_depth.astype(np.int16), 255).astype(np.uint8)
        a_depth = np.load(f"surface_labels/{sid}/depth.npy")
        a_conf = np.load(f"surface_labels/{sid}/confidence.npy")
        arms = {
            "A normal zarr / normal surface": (raw, a_depth, a_conf),
            "B normal zarr / flipped surface": (raw, b_depth, c_conf),
            "C flipped zarr / normal surface": (flipped, c_depth, c_conf),
        }
        full_y, full_x = vis.full_y_range, vis.full_x_range
        coords = vis._gen_tile_coords((config.data.d_start, config.data.d_end), full_y, full_x,
                                      vis.eval_mask, z_step=config.data.depth)
        grouped = group_by_depth(coords)
        (offset, tiles), = grouped.items()
        preds = {}
        with torch.no_grad():
            for name, (volume, s_depth, s_conf) in arms.items():
                preds[name] = predict_tiles(
                    config, model, volume, vis.eval_mask, tiles, full_y, full_x,
                    config.data.d_start + offset, f"{sid} {name[0]}", vis.global_mean, vis.global_std,
                    vis.global_min, vis.global_max, surface_depth_map=s_depth, surface_confidence_map=s_conf,
                )
        shape = next(iter(preds.values())).shape
        crop = (slice(full_y[0], full_y[1]), slice(full_x[0], full_x[1]))
        mask = grid(vis.eval_mask[crop] > 0, shape, cell) > 0.5
        labels = grid(np.asarray(vis.labels)[crop] > 0.5, shape, cell) > 0.25
        train = vis.manual_train_mask
        valid = (grid(np.asarray(train)[crop] == 0, shape, cell) > 0.5) if train is not None else np.ones(shape, bool)
        labelled = int(sid) not in set(cf.LABEL_FREE_VIS_IDS)
        surface_gap = (b_depth.astype(np.int16) - a_depth.astype(np.int16))[(a_conf > 0) & (c_conf > 0)]
        entry = {"surface_b_minus_a_median": float(np.median(surface_gap)) if surface_gap.size else None}
        for name, pred in preds.items():
            p = np.where(mask, pred, np.nan)
            entry[name] = {"mean_prob": float(np.nanmean(p)), "p95_prob": float(np.nanpercentile(p, 95))}
            if labelled:
                entry[name].update(scores(pred, labels, mask, valid))
        report[str(sid)] = entry
        print(f"[abc] {sid}: " + json.dumps(entry), flush=True)

        panels = list(preds.items()) + ([("labels", labels.astype(np.float32))] if labelled else [])
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5 * shape[0] / max(shape[1], 1) + 1))
        for ax, (name, image) in zip(axes, panels):
            ax.imshow(np.where(mask, image, np.nan), cmap="gray", vmin=0, vmax=1)
            metric = entry.get(name, {}).get("all")
            ax.set_title(name + (f"\nAP={metric['ap']:.3f} ROC={metric['roc']:.3f}" if metric else ""), fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{sid}", fontsize=11)
        fig.savefig(OUT / f"{sid}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        del vis, raw, flipped, preds
        torch.cuda.empty_cache()

    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"[abc] wrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
