"""Fine-tune reader-v3 on the mix6 corpus (utils/mix6.py) for 2000 optimizer steps, from the released checkpoint.

one command does everything and can be re-run (finished steps are skipped):

  python3 finetune_readerv3.py              # setup -> prep -> train
  python3 finetune_readerv3.py --smoke      # the same with 20 steps into runs/mix6_smoke (data and environment are shared)
  python3 finetune_readerv3.py setup|prep|train

setup  clones DomRusso2/reader-v3 (train_config.json) and the koine_machines training code (ScrollPrize/villa, sparse) at the
       pinned commits into /vesuvius/reader-v3 and /vesuvius/villa-ink, builds reader-v3/.venv (uv, python 3.12, the locked
       requirements_readerv3.txt) and downloads the released checkpoint reader-v3/reader-v3.pth from Hugging Face.
prep   stages every mix6 segment the way the reader reads a corpus: a 21-layer crop of the 28-layer render centred on the surface
       layer (label plane = the middle slice; the trainer jitters a 17-layer window by +-2), the ink label and a supervision mask.
       Half-brightness strokes of dilated_inklabels are not ink (split_extra_labels, as train.py reads them) and stay supervised
       background in 0846A; every other scroll trains on its train_masks region only (255 trains, 119 = hard negative inside the
       papyrus mask, black = unused; half-brightness strokes unused). A segment is re-staged when its labels or masks change.
train  batch 10 (0846A 4, every other scroll 1; sgd, lr 0.003, 200 warmup steps), 384 px tiles, fp16; checkpoints every 1000 steps.

output: reader-v3/runs/mix6/ckpt_002000.pth (printed at the end); ensemble_eval.ipynb and 0846A_eval.ipynb read it from there.
Needs the mix6 data (python3 assemble_training_segments.py --mix6), git, uv and a GPU.
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

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from utils import mix6  # noqa: E402

READER = Path(os.environ.get("READER_V3_DIR", REPO / "reader-v3"))
VILLA = Path(os.environ.get("VILLA_DIR", REPO / "villa-ink"))
READER_URL, READER_COMMIT = "https://github.com/DomRusso2/reader-v3", "4a9f1c633d2f30d9b522922ec1cb52c082013183"
VILLA_URL, VILLA_COMMIT = "https://github.com/ScrollPrize/villa.git", "47a9c37f6f8e99afdab47bb46ad425aa726f5dc0"  # branch merge-ink-pipelines
HF_REPO, CHECKPOINT_FILE, CHECKPOINT_BYTES = "domenicor046/reader-v3", "reader-v3.pth", 138379659
PYTHON = READER / ".venv" / "bin" / "python"
CHECKPOINT = READER / CHECKPOINT_FILE
DATA = READER / "data" / "mix6"
DEPTH = 21  # crop depth; label plane = DEPTH // 2
STEPS, BATCH = 2000, 10
LEARNING_RATE, WARMUP_STEPS, SAVE_EVERY = 0.003, 200, 1000
# samples per batch of each scroll (fixed_scroll_prior sampling); 0846A has seven sheets, every other scroll one or two
BATCH_COUNTS = {"0846A": 4, "0343": 1, "0009B": 1, "1447": 1, "0841": 1, "0814": 1, "0172": 1}


# the torch wheels are several GB: keep uv's cache next to the repo (the container's home disk is small)
UV_ENV = dict(os.environ, UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", str(REPO / "_ves_tmp" / "uv_cache")))


def run(command, **kwargs) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def run_dir(smoke: bool) -> Path:
    return READER / "runs" / ("mix6_smoke" if smoke else "mix6")


def final_checkpoint(smoke: bool) -> Path:
    return run_dir(smoke) / f"ckpt_{(20 if smoke else STEPS):06d}.pth"  # the full run's is mix6.MODEL_PATHS["reader_v3"]


# ---- setup ---------------------------------------------------------------------------------------------------------------

def setup() -> None:
    if not (READER / ".git").exists():
        run(["git", "clone", READER_URL, READER])
        run(["git", "-C", READER, "checkout", "-q", READER_COMMIT])
    exclude = READER / ".git" / "info" / "exclude"
    known = exclude.read_text().split() if exclude.is_file() else []
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with open(exclude, "a", encoding="utf-8") as handle:
        handle.writelines(f"{name}\n" for name in (".venv/", "data/", "runs/", ".cache/") if name not in known)

    if not (VILLA / "ink-detection" / "koine_machines").is_dir():
        VILLA.mkdir(parents=True, exist_ok=True)
        if not (VILLA / ".git").exists():
            run(["git", "init", "-q", VILLA])
            run(["git", "-C", VILLA, "remote", "add", "origin", VILLA_URL])
        run(["git", "-C", VILLA, "fetch", "--depth", "1", "origin", VILLA_COMMIT])
        run(["git", "-C", VILLA, "sparse-checkout", "set", "--cone", "ink-detection", "vesuvius"])  # koine_machines + its vesuvius dependency
        run(["git", "-C", VILLA, "checkout", "-q", VILLA_COMMIT])

    if not PYTHON.exists():
        if shutil.which("uv") is None:
            raise SystemExit("uv is needed to build the reader environment: pip install uv")
        run(["uv", "venv", "--python", "3.12", READER / ".venv"], env=UV_ENV)
        run(["uv", "pip", "install", "--python", PYTHON, "-r", REPO / "requirements_readerv3.txt"], env=UV_ENV)
        run(["uv", "pip", "install", "--python", PYTHON, "--no-deps", "-e", VILLA / "vesuvius", "-e", VILLA / "ink-detection"], env=UV_ENV)
    run([PYTHON, "-c", "import koine_machines.training.train"])

    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size != CHECKPOINT_BYTES:
        run([PYTHON, "-c", f"from huggingface_hub import hf_hub_download; hf_hub_download({HF_REPO!r}, {CHECKPOINT_FILE!r}, local_dir={str(READER)!r})"])
    print(f"[setup] reader-v3 {READER}, koine_machines {VILLA}, checkpoint {CHECKPOINT}", flush=True)


# ---- prep ----------------------------------------------------------------------------------------------------------------

def _plane_zarr(path: Path, shape, chunks, dtype):
    import zarr

    group = zarr.open_group(str(path), mode="w")
    return group.create_dataset("0", shape=shape, chunks=chunks, dtype=dtype, fill_value=0, dimension_separator=".")


def _signature(alias: str) -> str:
    """hash of everything a staged segment is made from."""
    segment = mix6.SEGMENTS[alias]
    digest = hashlib.sha1(f"{segment.layer}:{DEPTH}:{segment.split_masked}".encode())
    paths = [REPO / "masks" / f"{segment.render}.png", REPO / "dilated_inklabels" / f"{alias}.png"]
    if segment.split_masked:
        paths.append(REPO / "train_masks" / f"{alias}.png")
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prep() -> None:
    import cv2
    import numpy as np
    import zarr

    from utils.dataloader import split_extra_labels

    for alias, segment in mix6.SEGMENTS.items():
        folder = DATA / "labels" / alias
        stamp = folder / "prep.json"
        signature = _signature(alias)
        staged = (DATA / "volumes" / f"{alias}.zarr").exists() and (folder / f"{alias}_inklabels.zarr").exists()
        # labels and masks are hashed; a regenerated render (new zarr metadata) is newer than the stamp
        rendered = mix6.zarr_mtime(mix6.zarr_dir() / f"{segment.render}.zarr")
        if staged and stamp.is_file() and json.loads(stamp.read_text()).get("signature") == signature and rendered <= stamp.stat().st_mtime:
            print(f"[prep] {alias}: up to date")
            continue
        for stale in (DATA / "volumes" / f"{alias}.zarr", folder):
            shutil.rmtree(stale, ignore_errors=True)
        source = zarr.open(str(mix6.zarr_dir() / f"{segment.render}.zarr"), mode="r")
        mask = cv2.imread(str(REPO / "masks" / f"{segment.render}.png"), cv2.IMREAD_GRAYSCALE) > 0
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        rows = min(source.shape[1], mask.shape[0], label.shape[0])
        cols = min(source.shape[2], mask.shape[1], label.shape[1])
        if (rows, cols) != tuple(source.shape[1:]):
            print(f"[prep] {alias}: cropping to {rows}x{cols} (volume {source.shape[1:]}, mask {mask.shape}, label {label.shape})")
        mask, label = mask[:rows, :cols], label[:rows, :cols]
        ink, extra = split_extra_labels(label)
        if segment.split_masked:
            split = cv2.imread(str(REPO / "train_masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)[:rows, :cols]
            mask = mask & ((split >= 240) | ((split >= 100) & (split <= 143))) & ~extra
        z0 = segment.layer - DEPTH // 2
        if z0 < 0 or z0 + DEPTH > source.shape[0]:
            raise ValueError(f"{alias}: surface layer {segment.layer} leaves the volume")
        folder.mkdir(parents=True, exist_ok=True)
        (DATA / "volumes").mkdir(parents=True, exist_ok=True)
        shape = (DEPTH, rows, cols)
        volume = _plane_zarr(DATA / "volumes" / f"{alias}.zarr", shape, (DEPTH, 128, 128), source.dtype)
        for y in range(0, rows, 512):
            volume[:, y:y + 512, :] = source[z0:z0 + DEPTH, y:min(y + 512, rows), :cols]
        for kind, plane in (("inklabels", ink), ("supervision_mask", mask)):
            array = _plane_zarr(folder / f"{alias}_{kind}.zarr", shape, (DEPTH, 128, 128), np.uint8)
            array[DEPTH // 2, :, :] = plane.astype(np.uint8) * 255
        stamp.write_text(json.dumps({"signature": signature}))
        print(f"[prep] {alias} ({segment.scroll}): {shape} ink={int(ink.sum())} supervised={int(mask.sum())}", flush=True)


# ---- train ---------------------------------------------------------------------------------------------------------------

def patch_cache(base: dict, out_dir: Path) -> Path:
    """one patch index per staged corpus, shared by the smoke and the full run and replaced whenever a segment is re-staged
    (the trainer would otherwise reuse the base config's single shared cache for any data)."""
    stamps = [DATA / "labels" / alias / "prep.json" for alias in mix6.SEGMENTS]
    size = "x".join(str(int(v)) for v in base["patch_size"][1:])
    if not all(stamp.is_file() for stamp in stamps):
        return out_dir / f"patches_{size}.json"
    digest = hashlib.sha1("".join(json.loads(stamp.read_text())["signature"] for stamp in stamps).encode()).hexdigest()[:10]
    return DATA / f"patches_{size}_{digest}.json"


def config(smoke: bool) -> Path:
    out_dir = run_dir(smoke)
    base = json.loads((READER / "train_config.json").read_text())
    datasets = []
    for scroll in dict.fromkeys(segment.scroll for segment in mix6.SEGMENTS.values()):
        aliases = [alias for alias, segment in mix6.SEGMENTS.items() if segment.scroll == scroll]
        family = f"native_9p36_{scroll}"
        datasets.append({
            "segments_path": str(DATA / "labels"),
            "segments": aliases,
            "surface_volume_paths": {alias: str(DATA / "volumes" / f"{alias}.zarr") for alias in aliases},
            "volume_scale": 0,
            "source_family": family,
            "sampling_scroll": scroll,
            "sampling_physical_segment_keys": {alias: f"{scroll}:{alias}" for alias in aliases},
            "sampling_representation_keys": {alias: f"{family}:{alias}" for alias in aliases},
        })
    base["datasets"] = datasets
    base["fixed_scroll_prior"]["target_batch_counts"] = dict(BATCH_COUNTS)
    base.update({
        "checkpoint": str(CHECKPOINT),
        "weights_only": True,
        "batch_size": BATCH,
        "grad_acc_steps": 1,
        "num_iterations": 20 if smoke else STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": 5 if smoke else WARMUP_STEPS,
        "save_every": 10 if smoke else SAVE_EVERY,
        "val_every": 10**9,
        "dataloader_workers": 6,
        "out_dir": str(out_dir),
        "patch_cache_filename": str(patch_cache(base, out_dir)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    path.write_text(json.dumps(base, indent=1))
    return path


def train(smoke: bool) -> Path:
    final = final_checkpoint(smoke)
    if final.is_file():
        print(f"[train] {final} exists already; delete {run_dir(smoke)} to train again", flush=True)
        return final
    if run_dir(smoke).exists():
        if smoke:
            shutil.rmtree(run_dir(smoke))
        elif list(run_dir(smoke).glob("ckpt_*.pth")):
            raise SystemExit(f"[train] {run_dir(smoke)} holds the checkpoints of an interrupted run; delete it to train again")
    path = config(smoke)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    run([PYTHON, "-m", "koine_machines.training.train", path], cwd=READER, env=env)
    if not final.is_file():
        raise SystemExit(f"[train] finished without {final}")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=("all", "setup", "prep", "train"))
    parser.add_argument("--smoke", action="store_true", help="20 steps into runs/mix6_smoke")
    args = parser.parse_args()
    os.chdir(REPO)
    final = None
    if args.command in ("all", "setup"):
        setup()
    if args.command in ("all", "prep"):
        problems = mix6.status([int(alias) for alias in mix6.SEGMENTS], derived=False)
        if problems:
            raise SystemExit("mix6 data is incomplete (python3 assemble_training_segments.py --mix6):\n  " + "\n  ".join(problems[:10]))
        prep()
    if args.command in ("all", "train"):
        final = train(args.smoke)
    if final is not None:
        print(f"\n[done] reader-v3 mix6 fine-tune ({'smoke, 20' if args.smoke else str(STEPS)} steps): {final}", flush=True)


if __name__ == "__main__":
    main()
