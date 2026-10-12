"""Six-patch PHerc0846A MAE continuation, using private input aliases and caches.

This discovery initialization sees all six patches without ink labels. It is not a
strict held-out-patch initialization. The original MAE is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import campaign_archs_34 as campaign34
import campaign_archs_40 as campaign40

ROOT = Path(__file__).resolve().parent
NAME = "mae_0846a_six_flat13_m1_2k"
INIT_WEIGHTS = campaign34._pretrain_path(campaign40.SURFACE_FIBER_MAE_KEY)
PATCHES = (
    (20260927134931, "20260927134928", "masks", 20260927134931),
    (20261012065806, "20261012065803", "masks", 20261012065806),
    (20261012073246, "20261012073244", "masks", 20261012073246),
    (20261012081238, "20261012081237", "masks", 20261012081238),
    (20260927143929, "auto_grown_20260927143929299_abf", "masks/extras", None),
    (20260927154033, "auto_grown_20260927154033403_abf", "masks/extras", None),
)


def training_command(name=NAME, steps=2000, dry_run=False):
    command = [
        sys.executable, "-u", str(ROOT / "mae_pretrain_nnunet.py"),
        "--name", name, "--init-weights", str(INIT_WEIGHTS),
        *campaign34.PRETRAIN_SPECS[campaign40.SURFACE_FIBER_MAE_KEY]["args"],
        "--ctx", "96", "--ds", "1", "--depth", "8", "--d-start", "9", "--d-end", "17",
        "--scroll-ids", *(str(patch[0]) for patch in PATCHES), "--require-all-scrolls",
        "--steps", str(steps), "--batch-size", "24", "--lr", "1e-4", "--seed", "41",
        "--log-dir", str(ROOT / "runs_mae_letters"), "--log-int", "50", "--save-int", "500",
        "--no-figures",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def prepare_inputs(name):
    import cv2
    import numpy as np
    import zarr

    from utils.norm import compute_norm, compute_surface_anchors

    stage = ROOT / "_ves_tmp" / "mae_0846a" / name
    volumes, masks, surfaces = (stage / folder for folder in ("volumes", "masks", "surface_labels"))
    for folder in (volumes, masks, surfaces):
        folder.mkdir(parents=True, exist_ok=True)
    source_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    global_cache = json.loads((ROOT / "norm_cache.json").read_text())
    anchor_cache = json.loads((ROOT / "surface_anchor_cache.json").read_text())
    staged_norm, staged_anchors, sources = {}, {}, []
    for alias, source_id, mask_folder, norm_id in PATCHES:
        source = source_root / f"{source_id}.zarr"
        source_mask = ROOT / mask_folder / f"{source_id}.png"
        if not source.is_dir() or not source_mask.is_file():
            raise FileNotFoundError(f"Missing MAE input: {source} or {source_mask}")
        for link, target in ((volumes / f"{alias}.zarr", source), (masks / f"{alias}.png", source_mask)):
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    raise ValueError(f"Unexpected existing alias: {link}")
            elif link.exists():
                raise FileExistsError(link)
            else:
                link.symlink_to(target.resolve())
        volume = zarr.open(str(source), mode="r")
        mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != tuple(volume.shape[1:]) or not mask.any():
            raise ValueError(f"Invalid mask/frame for {source_id}")
        if volume.shape[0] < 17 or volume.attrs.get("layers_reversed_from") or volume.attrs.get("facing_reversed"):
            raise ValueError(f"Expected forward volume with slices 9-16: {source}")
        norm_key = str(norm_id if norm_id is not None else source_id)
        stats = global_cache.get(norm_key)
        if stats is None:
            values = compute_norm(alias, str(volumes), str(stage / "norm_cache.json"), mask_dir=str(masks))
            stats = dict(zip(("mean", "std", "min", "max"), values))
        staged_norm[str(alias)] = stats
        anchors = anchor_cache.get(str(norm_id)) if norm_id is not None else None
        if anchors is None:
            surface = surfaces / str(alias)
            surface.mkdir(exist_ok=True)
            np.save(surface / "depth.npy", np.where(mask > 0, 13, 255).astype(np.uint8))
            np.save(surface / "confidence.npy", np.where(mask > 0, 255, 0).astype(np.uint8))
            anchors = compute_surface_anchors(alias, str(volumes), surface_dir=str(surfaces), mask_dir=str(masks))
        if float(stats["std"]) <= 0 or float(anchors["papyrus"]) <= float(anchors["gap"]):
            raise ValueError(f"Invalid normalization for {source_id}")
        staged_anchors[str(alias)] = anchors
        sources.append({
            "alias": alias, "source_id": source_id, "zarr": str(source.resolve()),
            "mask": str(source_mask), "shape": list(volume.shape), "norm_id": norm_id,
            "flat_depth": 13, "global_norm": stats, "anchors": anchors,
        })
    (stage / "norm_cache.json").write_text(json.dumps(staged_norm, indent=2) + "\n")
    (stage / "surface_anchor_cache.json").write_text(json.dumps(staged_anchors, indent=2) + "\n")
    return stage, sources


def main():
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", "..") or args.steps <= 0:
        parser.error("name must be a filename stem and steps must be positive")
    if not INIT_WEIGHTS.is_file():
        raise FileNotFoundError(INIT_WEIGHTS)
    output = ROOT / "models" / f"{args.name}.pth"
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Preserving existing output; choose another --name: {output}")
    stage, sources = prepare_inputs(args.name)
    staged_output = stage / "models" / f"{args.name}.pth"
    if staged_output.exists() and not args.dry_run:
        raise FileExistsError(f"Preserving interrupted output; choose another --name: {staged_output}")
    command = training_command(args.name, args.steps, args.dry_run)
    environment = {**os.environ, "VESUVIUS_ZARR_PATH": str(stage / "volumes"),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    metadata = {
        "purpose": "six-patch discovery MAE continuation, not a held-out-patch initialization",
        "init_weights": str(INIT_WEIGHTS), "init_sha256": hashlib.sha256(INIT_WEIGHTS.read_bytes()).hexdigest(),
        "sources": sources, "command": command, "cwd": str(stage),
        "steps": args.steps, "supervised_labels_used": False, "surface_teacher_input": False,
        "surface_normalization_depth": 13, "window": [9, 17], "sampling": "equal patch round-robin",
        "output": str(output), "complete": False,
    }
    (stage / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[letters-mae] {'checking' if args.dry_run else 'adapting'} six patches from {INIT_WEIGHTS}", flush=True)
    subprocess.run(command, cwd=stage, env=environment, check=True)
    if args.dry_run:
        return
    import torch

    state = torch.load(staged_output, map_location="cpu", weights_only=True)
    initial = torch.load(INIT_WEIGHTS, map_location="cpu", weights_only=True)
    if set(state) != set(initial):
        raise ValueError("Adapted backbone keys differ from the matched MAE initialization")
    for key, value in state.items():
        if value.shape != initial[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapted tensor: {key}")
    for prefix in ("enc1.", "early2d_enc2.", "gated_cue_stem.", "fiber_coordinate_input."):
        if not any(not torch.equal(value, initial[key]) for key, value in state.items() if key.startswith(prefix)):
            raise ValueError(f"Adaptation did not update {prefix}")
    temporary = output.with_suffix(".tmp")
    shutil.copyfile(staged_output, temporary)
    temporary.replace(output)
    metadata["complete"] = True
    metadata["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[letters-mae] verified backbone saved to {output}; train a supervised ink model before inference", flush=True)


if __name__ == "__main__":
    main()