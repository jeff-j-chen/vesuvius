"""PHerc0175A MAE continuation: all 25 0175A quick patches plus the six PHerc0846A patches of the previous MAE.

starts from mae_0846a_six_flat13_m1_2k.pth and keeps training the same early-2D / gated / fiber backbone (native 96 px,
8 slices 9-16, surface-anchor norm) on unlabeled crops. every patch is read through its flat depth-13 alias. the two
physical scrolls are drawn half and half, and 75% of the crops are rotated by a uniform random angle about their
centre, because the 0175A fibers are not axis-aligned. the input MAE is never overwritten.

usage: python3 mae_pretrain_0175a.py [--name N] [--steps 4000] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "mae_0175a_plus_0846a_rot_4k"
INIT_WEIGHTS = ROOT / "models" / "mae_0846a_six_flat13_m1_2k.pth"
ARCHITECTURE_ARGS = ("--early-2d-unet", "--gated-stems", "--norm-mode", "ibn_full",
                     "--fiber-coordinate-branch", "--data-norm-mode", "surface_anchor")
# (alias, render id, mask folder, norm/anchor key): the six patches of the previous MAE
PATCHES_0846A = (
    (20260927134931, "20260927134928", "masks", 20260927134931),
    (20261012065811, "20261012065810", "masks", 20261012065811),  # straight re-trace replaces the retired 065803
    (20261012073246, "20261012073244", "masks", 20261012073246),
    (20261012081238, "20261012081237", "masks", 20261012081238),
    (20260927143929, "auto_grown_20260927143929299_abf", "masks/extras", None),
    (20260927154033, "auto_grown_20260927154033403_abf", "masks/extras", None),
)


def patches_0175a():
    from utils import pherc0175a

    return tuple((alias, name, "masks/extras", alias) for name, alias in pherc0175a.aliases().items())


def training_command(patches, name=NAME, steps=4000, rotate_prob=0.75, dry_run=False, prefilter="", mix=1.0,
                     cue=""):
    old = {p[0] for p in PATCHES_0846A}
    groups = {"pherc0846a": sorted(old), "pherc0175a": [p[0] for p in patches if p[0] not in old]}
    command = [
        sys.executable, "-u", str(ROOT / "old" / "mae_pretrain_nnunet.py"),
        "--name", name, "--init-weights", str(INIT_WEIGHTS), *ARCHITECTURE_ARGS,
        "--ctx", "96", "--ds", "1", "--depth", "8", "--d-start", "9", "--d-end", "17",
        "--scroll-ids", *(str(p[0]) for p in patches), "--require-all-scrolls",
        "--physical-round-robin", "--physical-groups", json.dumps(groups),
        "--rotate-prob", str(rotate_prob),
        *(["--input-prefilter", prefilter, "--input-prefilter-mix", str(mix)] if prefilter else []),
        *(["--input-cue", cue] if cue else []),
        "--steps", str(steps), "--batch-size", "24", "--lr", "1e-4", "--seed", "41",
        "--log-dir", str(ROOT / "runs_mae_0175a"), "--log-int", "50", "--save-int", "500", "--no-figures",
    ]
    return command + (["--dry-run"] if dry_run else [])


def prepare_inputs(name, patches):
    import cv2
    import numpy as np
    import zarr

    from utils.norm import compute_norm, compute_surface_anchors

    stage = ROOT / "_ves_tmp" / "mae_0175a" / name
    volumes, masks, surfaces = (stage / folder for folder in ("volumes", "masks", "surface_labels"))
    for folder in (volumes, masks, surfaces):
        folder.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / "ves_zarrs2"
    global_cache = json.loads((ROOT / "norm_cache.json").read_text())
    anchor_cache = json.loads((ROOT / "surface_anchor_cache.json").read_text())
    staged_norm, staged_anchors, sources = {}, {}, []
    for alias, source_id, mask_folder, norm_id in patches:
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
        sources.append({"alias": alias, "source_id": source_id, "zarr": str(source.resolve()), "mask": str(source_mask),
                        "shape": list(volume.shape), "norm_id": norm_id, "flat_depth": 13})
    (stage / "norm_cache.json").write_text(json.dumps(staged_norm, indent=2) + "\n")
    (stage / "surface_anchor_cache.json").write_text(json.dumps(staged_anchors, indent=2) + "\n")
    return stage, sources


def main():
    global INIT_WEIGHTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--rotate-prob", type=float, default=0.75)
    parser.add_argument("--init", default=str(INIT_WEIGHTS), help="MAE weights to continue from")
    parser.add_argument("--prefilter", default="", choices=("", "asf", "clahe"), help="fixed input filter (utils/prefilter.py)")
    parser.add_argument("--prefilter-mix", type=float, default=1.0)
    parser.add_argument("--input-cue", default="", choices=("", "asf", "clahe"),
                        help="fourth gated input head (the 3-cue init is expanded, new cue starts inert)")
    parser.add_argument("--only-0846a", action="store_true", help="train on the six 0846A patches only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    INIT_WEIGHTS = Path(args.init).resolve()
    if Path(args.name).name != args.name or args.name in (".", "..") or args.steps <= 0:
        parser.error("name must be a filename stem and steps must be positive")
    if not INIT_WEIGHTS.is_file():
        raise FileNotFoundError(INIT_WEIGHTS)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from utils import pherc0175a

    pherc0175a.prepare()
    patches = PATCHES_0846A if args.only_0846a else (*PATCHES_0846A, *patches_0175a())
    output = ROOT / "models" / f"{args.name}.pth"
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Preserving existing output; choose another --name: {output}")
    stage, sources = prepare_inputs(args.name, patches)
    staged_output = stage / "models" / f"{args.name}.pth"
    if staged_output.exists() and not args.dry_run:
        raise FileExistsError(f"Preserving interrupted output; choose another --name: {staged_output}")
    command = training_command(patches, args.name, args.steps, args.rotate_prob, args.dry_run,
                               args.prefilter, args.prefilter_mix, args.input_cue)
    environment = {**os.environ, "VESUVIUS_ZARR_PATH": str(stage / "volumes"), "PYTHONPATH": str(ROOT),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    metadata = {
        "purpose": "0175A + six 0846A patches MAE continuation with rotation augmentation; no ink labels",
        "init_weights": str(INIT_WEIGHTS), "init_sha256": hashlib.sha256(INIT_WEIGHTS.read_bytes()).hexdigest(),
        "sources": sources, "command": command, "cwd": str(stage), "steps": args.steps,
        "rotate_prob": args.rotate_prob, "supervised_labels_used": False, "surface_normalization_depth": 13,
        "window": [9, 17], "sampling": "half 0846A / half 0175A, patches round-robin inside each", "output": str(output),
        "complete": False,
    }
    (stage / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[0175a-mae] {'checking' if args.dry_run else 'adapting'} {len(patches)} patches from {INIT_WEIGHTS}", flush=True)
    subprocess.run(command, cwd=stage, env=environment, check=True)
    if args.dry_run:
        return
    import torch

    state = torch.load(staged_output, map_location="cpu", weights_only=True)
    initial = torch.load(INIT_WEIGHTS, map_location="cpu", weights_only=True)
    grown = bool(args.input_cue)
    if (set(initial) - set(state)) or (not grown and set(state) != set(initial)):
        raise ValueError("Adapted backbone keys differ from the matched MAE initialization")
    for key, value in state.items():
        if not torch.isfinite(value).all() or (key in initial and value.shape != initial[key].shape and not grown):
            raise ValueError(f"Invalid adapted tensor: {key}")
    for prefix in ("enc1.", "early2d_enc2.", "gated_cue_stem.", "fiber_coordinate_input."):
        if not any(key not in initial or value.shape != initial[key].shape or not torch.equal(value, initial[key])
                   for key, value in state.items() if key.startswith(prefix)):
            raise ValueError(f"Adaptation did not update {prefix}")
    temporary = output.with_suffix(".tmp")
    shutil.copyfile(staged_output, temporary)
    temporary.replace(output)
    metadata["complete"] = True
    metadata["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[0175a-mae] verified backbone saved to {output}", flush=True)


if __name__ == "__main__":
    main()
