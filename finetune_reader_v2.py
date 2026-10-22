"""Fine-tune reader-v2 (step 40000) on the 7 trained 0846A patches + PHerc0343 first letters, then render straightened2.

Labels follow finetune_v8in_0846a.py: half-brightness strokes are removed from the ink (split_extra_labels) and stay
supervised background; 343 trains on its train_masks region only. Volumes are cropped to 21 layers centred on each
patch's learned surface (the corpus layout: label plane = middle slice, 17-of-21 window jittered +-2).

  python3 finetune_reader_v2.py prep                                   # system python
  reader-v2/.venv/bin/python finetune_reader_v2.py train        # RR_SMOKE=1 for a short test run
  reader-v2/.venv/bin/python finetune_reader_v2.py infer
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RV2 = REPO / "reader-v2"
DATA = RV2 / "data" / "ft_0846a_343"
SMOKE = os.environ.get("RR_SMOKE") == "1"
V3 = os.environ.get("RR_V3") == "1"  # reader-v3 (384 px context) fine-tuned from its released checkpoint, same data and method as the v2 run
RV3 = REPO / "reader-v3"
EFFECTIVE_BATCH = 10  # the v3 recipe's batch (5 + 5 here); RR_MB < 10 splits it into accumulated micro-batches of the same effective size
MICRO_BATCH = int(os.environ.get("RR_MB", EFFECTIVE_BATCH))
ACC = EFFECTIVE_BATCH // MICRO_BATCH
HOLD = os.environ.get("RR_HOLD") == "1"  # from the base checkpoint again, with straightened2 (the old holdout) in the training set
CONT = os.environ.get("RR_CONT") == "1"  # second stage: weights of the first fine-tune, learning-rate schedule restarted
FIRST_RUN = RV2 / "runs" / "ft_0846a_343"
OUT_DIR = RV2 / "runs" / ("v3_smoke" if V3 and SMOKE else "v3_ft_0846a_343" if V3 else "ft_0846a_343_smoke" if SMOKE else "ft_0846a_343_cont" if CONT else "ft_with_holdout" if HOLD else "ft_0846a_343")
CHECKPOINT = RV3 / "reader-v3.pth" if V3 else FIRST_RUN / "ckpt_003200.pth" if CONT else RV2 / "reader-v2-step040000.pth"
TAG = os.environ.get("RR_TAG") or ("v3_finetuned" if V3 else "finetuned_cont" if CONT else "holdouttrained" if HOLD else "finetuned")
FIGURES = REPO / "output" / ("reader_v3" if V3 else "reader_v2")
DEPTH = 21  # crop depth; label plane = DEPTH // 2

# segment id (flat alias: label name) -> (render zarr id, surface layer of the 28-layer render, scroll, split-masked)
SEGMENTS = {
    "20260927134931": ("20260927134928", 13, "0846A", False),
    "20261012073246": ("20261012073244", 13, "0846A", False),
    "20261012065813": ("20261012065812", 13, "0846A", False),
    "20261012065817": ("20261012065816", 13, "0846A", False),
    "20261012065819": ("20261012065818", 13, "0846A", False),
    "20261012065821": ("20261012065820", 13, "0846A", False),
    "20261012065823": ("20261012065822", 13, "0846A", False),
    "20261007000000": ("20261007000000", 15, "0343", True),
}
if HOLD:
    SEGMENTS["20261012065815"] = ("20261012065814", 13, "0846A", False)
# the 0846A fragments of the final figures: alias -> (render zarr id, display name)
VIS = {
    "20260927134931": ("20260927134928", "auto_grown_20260927134928530_abf"),
    "20261012073246": ("20261012073244", "auto_grown_20261012073244544_abf"),
    "20261012065813": ("20261012065812", "auto_grown_20261012065803387_straight_abf_m28"),
    "20261012065817": ("20261012065816", "auto_grown_20261012065803387_straightened2_abf_in_1"),
    "20261012065819": ("20261012065818", "auto_grown_20261012065803387_straightened2_abf_out_1"),
    "20261012065821": ("20261012065820", "auto_grown_20261012065803387_straightened2_abf_out_2"),
    "20261012065823": ("20261012065822", "auto_grown_20261012065803387_straightened2_abf_out_3"),
    "20261012065815": ("20261012065814", "auto_grown_20261012065803387_straightened2_abf"),
}
BATCH = 32
ITERATIONS = 5000 if V3 else 10000 if HOLD else 9600 if CONT else 3200
LEARNING_RATE = 0.002 if CONT else 0.003


def _plane_zarr(path: Path, shape, chunks, dtype):
    import zarr

    group = zarr.open_group(str(path), mode="w")
    return group.create_dataset("0", shape=shape, chunks=chunks, dtype=dtype, fill_value=0, dimension_separator=".")


def prep() -> None:
    import cv2
    import numpy as np
    import zarr

    sys.path.insert(0, str(REPO))
    from utils.dataloader import split_extra_labels

    for segment, (zarr_id, surface, scroll, split_masked) in SEGMENTS.items():
        if (DATA / "volumes" / f"{segment}.zarr").exists() and (DATA / "labels" / segment / f"{segment}_inklabels.zarr").exists():
            print(f"[prep] {segment}: already prepared")
            continue
        source = zarr.open(str(REPO / "ves_zarrs2" / f"{zarr_id}.zarr"), mode="r")
        mask = cv2.imread(str(REPO / "masks" / f"{zarr_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{segment}.png"), cv2.IMREAD_GRAYSCALE)
        rows = min(source.shape[1], mask.shape[0], label.shape[0])
        cols = min(source.shape[2], mask.shape[1], label.shape[1])
        if (rows, cols) != tuple(source.shape[1:]):
            print(f"[prep] {segment}: cropping to {rows}x{cols} (volume {source.shape[1:]}, mask {mask.shape}, label {label.shape})")
        mask, label = mask[:rows, :cols], label[:rows, :cols]
        ink, _ = split_extra_labels(label)
        if split_masked:
            split = cv2.imread(str(REPO / "train_masks" / f"{segment}.png"), cv2.IMREAD_GRAYSCALE)[:rows, :cols]
            mask = mask & (split >= 240)
        z0 = surface - DEPTH // 2
        if z0 < 0 or z0 + DEPTH > source.shape[0]:
            raise ValueError(f"{segment}: surface {surface} leaves the volume")
        folder = DATA / "labels" / segment
        folder.mkdir(parents=True, exist_ok=True)
        (DATA / "volumes").mkdir(parents=True, exist_ok=True)
        shape = (DEPTH, rows, cols)
        volume = _plane_zarr(DATA / "volumes" / f"{segment}.zarr", shape, (DEPTH, 128, 128), source.dtype)
        for y in range(0, rows, 512):
            volume[:, y:y + 512, :] = source[z0:z0 + DEPTH, y:min(y + 512, rows), :cols]
        for kind, plane in (("inklabels", ink), ("supervision_mask", mask)):
            array = _plane_zarr(folder / f"{segment}_{kind}.zarr", shape, (DEPTH, 128, 128), np.uint8)
            array[DEPTH // 2, :, :] = plane.astype(np.uint8) * 255
        print(f"[prep] {segment} ({scroll}): {shape} ink={int(ink.sum())} supervised={int(mask.sum())}", flush=True)


def config() -> Path:
    base = json.loads(((RV3 if V3 else RV2) / "train_config.json").read_text())
    datasets = []
    for scroll in ("0846A", "0343"):
        segments = [s for s, v in SEGMENTS.items() if v[2] == scroll]
        family = f"native_9p36_{scroll}"
        datasets.append({
            "segments_path": str(DATA / "labels"),
            "segments": segments,
            "surface_volume_paths": {s: str(DATA / "volumes" / f"{s}.zarr") for s in segments},
            "volume_scale": 0,
            "source_family": family,
            "sampling_scroll": scroll,
            "sampling_physical_segment_keys": {s: f"{scroll}:{s}" for s in segments},
            "sampling_representation_keys": {s: f"{family}:{s}" for s in segments},
        })
    base["datasets"] = datasets
    batch = MICRO_BATCH if V3 else BATCH
    base["fixed_scroll_prior"]["target_batch_counts"] = {"0846A": batch // 2, "0343": batch // 2}
    iterations = (20 if SMOKE else ITERATIONS) * (ACC if V3 else 1)  # the loop counts micro-batches; optimizer steps = iterations / ACC
    base.update({
        "checkpoint": str(CHECKPOINT),
        "weights_only": True,
        "batch_size": batch,
        "grad_acc_steps": ACC if V3 else 1,
        "num_iterations": iterations,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": 5 if SMOKE else 200,
        "save_every": 10 if SMOKE else 1000 * ACC if V3 else 2000 if HOLD else 1600 if CONT else 800,
        "val_every": 10**9,
        "dataloader_workers": 6,
        "out_dir": str(OUT_DIR),
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CONT:  # same data, so reuse the patch index instead of re-scanning the volumes
        for index in FIRST_RUN.glob("flat_ink_patches*.json"):
            if not (OUT_DIR / index.name).exists():
                (OUT_DIR / index.name).write_bytes(index.read_bytes())
    if V3 and not SMOKE:  # the smoke run already scanned the volumes at this patch size
        for index in (RV2 / "runs" / "v3_smoke").glob("flat_ink_patches*.json"):
            if not (OUT_DIR / index.name).exists():
                (OUT_DIR / index.name).write_bytes(index.read_bytes())
    path = OUT_DIR / "config.json"
    path.write_text(json.dumps(base, indent=1))
    return path


def train() -> None:
    path = config()
    command = [sys.executable, "-m", "koine_machines.training.train", str(path)]
    print("launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=RV2, check=True)


def _final_checkpoint() -> Path:
    candidates = sorted(OUT_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint in {OUT_DIR}")
    print("checkpoints:", [c.name for c in candidates])
    return candidates[-1]


def infer() -> None:
    checkpoint = os.environ.get("RR_CKPT") or str(_final_checkpoint())
    FIGURES.mkdir(parents=True, exist_ok=True)
    tif = FIGURES / f"holdout_straightened2_{TAG}.tif"
    subprocess.run([sys.executable, "-m", "koine_machines.inference.infer", str(REPO / "ves_zarrs2" / "20261012065814.zarr"),
                    checkpoint, str(tif), "--overlap", "0.5", "--blend-mode", "hann", "--direction", "forward",
                    "--no-compile", "--batch-size", "1"], cwd=RV2, check=True)
    subprocess.run([str(REPO / "nader-fork" / ".venv" / "bin" / "python"), str(REPO / "_ves_tmp" / "reader_viz.py"), str(tif),
                    str(FIGURES / f"holdout_straightened2_{TAG}.png"),
                    f"straightened2 holdout | reader-v2 {TAG} on 0846A + 343 ({Path(checkpoint).name})"], check=True)


def viz() -> None:
    """every fragment of VIS with the final checkpoint: prediction | composite, then the inklabel in red."""
    checkpoint = os.environ.get("RR_CKPT") or str(_final_checkpoint())
    FIGURES.mkdir(parents=True, exist_ok=True)
    python = str(REPO / "nader-fork" / ".venv" / "bin" / "python")
    for alias, (zarr_id, name) in VIS.items():
        tif = FIGURES / f"{TAG}_{alias}.tif"
        subprocess.run([sys.executable, "-m", "koine_machines.inference.infer", str(REPO / "ves_zarrs2" / f"{zarr_id}.zarr"),
                        checkpoint, str(tif), "--overlap", "0.5", "--blend-mode", "hann", "--direction", "forward",
                        "--no-compile"], cwd=RV2, check=True)
        subprocess.run([python, str(REPO / "_ves_tmp" / "reader_viz.py"), str(tif), str(FIGURES / f"{TAG}_{alias}.png"),
                        f"reader-v2 {TAG} ({Path(checkpoint).name}), {alias}", "5", name, zarr_id, alias], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prep", "config", "train", "infer", "viz"))
    arguments = parser.parse_args()
    os.chdir(REPO)
    {"prep": prep, "config": config, "train": train, "infer": infer, "viz": viz}[arguments.command]()
