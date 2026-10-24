"""Fine-tune v8-in on the mix6 corpus (utils/mix6.py) for 5 epochs, from the released checkpoint.

one command does everything and can be re-run (finished steps are skipped):

  python3 finetune_v8in.py              # setup -> stage -> train -> export
  python3 finetune_v8in.py --smoke      # one epoch of 6 batches into runs/v8in_mix6_smoke (data and environment are shared)
  python3 finetune_v8in.py setup|stage|train

setup  clones jeff-j-chen/nader-fork at the pinned commit into /vesuvius/nader-fork (git lfs pulls the v8-in weights
       model.safetensors and the r3d50 backbone), builds nader-fork/.venv (uv, python 3.12, the locked requirements_v8in.txt).
stage  writes what v8-in reads, per segment, under nader-fork/runs/v8in_mix6_data/data/<alias>/: the central 24 layers of the
       render as 00.tif..23.tif (uint8, forward order), <alias>_inklabels.png and <alias>_mask.png. Half-brightness strokes of
       dilated_inklabels are not ink (split_extra_labels, as train.py reads them) and stay supervised background in 0846A; every
       other scroll trains on its train_masks region only (255 trains, 119 = hard negative inside the papyrus mask, black =
       unused; half-brightness strokes unused). Ink specks under 8 px are dropped in the three validation segments (the
       stitched validation metrics crash on them).
train  the PH1447 recipe of nader-fork (finetune_v8in_with_local_segment.py) on the staged corpus: 5 epochs, batch 4, NEG_WEIGHT 25,
       group-balanced sampling (0846A split into two groups, so it weighs 2 against 1 for every other scroll). The last epoch
       also scores the stitched validation sheets. The final checkpoint is exported to a model folder.

output: nader-fork/runs/v8in_mix6_5ep/model/ (model.safetensors + config.json, printed at the end; ensemble_eval.ipynb and
0846A_eval.ipynb read it from there); the Lightning checkpoints are nader-fork/runs/v8in_mix6_5ep/runs/*/checkpoints/.
Needs the mix6 data (python3 assemble_training_segments.py --mix6), git, git-lfs, uv and a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from utils import mix6  # noqa: E402

FORK = Path(os.environ.get("NADER_FORK_DIR", REPO / "nader-fork"))
FORK_URL, FORK_COMMIT = "https://github.com/jeff-j-chen/nader-fork.git", "8a811a86b751734223b3febff9adba24ecc549ad"
LFS_FILES = ("model.safetensors", "training/r3d50_KM_200ep.safetensors")
PYTHON = FORK / ".venv" / "bin" / "python"
DATA = FORK / "runs" / "v8in_mix6_data" / "data"
EPOCHS = 5
DEPTH = 24
NEG_WEIGHT = "25"
# the stitched validation sheets (trained segments, as in the author's recipe): two 0846A sheets and the 343 first letters
VAL = ("20261012065813", "20261012065823", "20261007000000")
# the container's home disk is small: keep uv's cache next to the repo
UV_ENV = dict(os.environ, UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", str(REPO / "_ves_tmp" / "uv_cache")))
# the sampler gives every group the same share: two 0846A groups make 0846A weigh 2 against 1 for each other scroll, as in mix6
GROUP_0846A = [alias for alias, segment in mix6.SEGMENTS.items() if segment.scroll == "0846A"]
GROUP = {alias: f"PHerc{segment.scroll}" for alias, segment in mix6.SEGMENTS.items()}
GROUP.update({alias: "PHerc0846A_b" for alias in GROUP_0846A[4:]})


def run(command, **kwargs) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def run_dir(smoke: bool) -> Path:
    return FORK / "runs" / ("v8in_mix6_smoke" if smoke else "v8in_mix6_5ep")


def model_dir(smoke: bool) -> Path:
    return run_dir(smoke) / "model"  # the full run's is mix6.MODEL_PATHS["v8in"]


# ---- setup ---------------------------------------------------------------------------------------------------------------

def setup() -> None:
    if not (FORK / ".git").exists():
        run(["git", "clone", FORK_URL, FORK])
        run(["git", "-C", FORK, "checkout", "-q", FORK_COMMIT])
    if shutil.which("git-lfs") is None:
        raise SystemExit("git-lfs is needed for the v8-in weights: apt install git-lfs && git lfs install")
    pointers = [name for name in LFS_FILES if not (FORK / name).is_file() or (FORK / name).stat().st_size < 1_000_000]
    if pointers:
        run(["git", "-C", FORK, "lfs", "pull", f"--include={','.join(pointers)}"])
    if not PYTHON.exists():
        if shutil.which("uv") is None:
            raise SystemExit("uv is needed to build the v8-in environment: pip install uv")
        run(["uv", "venv", "--python", "3.12", FORK / ".venv"], env=UV_ENV)
        run(["uv", "pip", "install", "--python", PYTHON, "-r", REPO / "requirements_v8in.txt"], env=UV_ENV)
    run([PYTHON, "-c", "import pytorch_lightning, segmentation_models_pytorch, safetensors"])
    print(f"[setup] nader-fork {FORK}, weights {FORK / 'model.safetensors'}", flush=True)


# ---- stage ---------------------------------------------------------------------------------------------------------------

def _layers(render: str, destination: Path) -> None:
    """the central DEPTH slices of the render as 00.tif..23.tif (uint8, forward order)."""
    import numpy as np
    import tifffile
    import zarr

    volume = zarr.open(str(mix6.zarr_dir() / f"{render}.zarr"), mode="r")
    start = (volume.shape[0] - DEPTH) // 2
    tiffs = [destination / f"{index:02d}.tif" for index in range(DEPTH)]
    # layers staged before the render was regenerated are dropped (a symlink to an older run's folder is unlinked, never written through)
    if all(path.is_file() for path in tiffs) and min(path.stat().st_mtime for path in tiffs) < mix6.zarr_mtime(mix6.zarr_dir() / f"{render}.zarr"):
        print(f"[stage] {render}: the render changed since its layers were staged; re-staging", flush=True)
        if destination.is_symlink():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(tiffs):
        if not path.is_file():
            tifffile.imwrite(path, np.clip(np.asarray(volume[start + index]), 0, 255).astype(np.uint8))


def stage() -> None:
    import cv2
    import numpy as np

    from utils.dataloader import split_extra_labels

    for alias, segment in mix6.SEGMENTS.items():
        folder = DATA / alias
        mask = cv2.imread(str(REPO / "masks" / f"{segment.render}.png"), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or label is None:
            raise FileNotFoundError(f"{alias}: mask or label missing (python3 assemble_training_segments.py --mix6)")
        mask = mask > 0
        if label.shape != mask.shape:
            print(f"[stage] {alias}: cropping label {label.shape} / mask {mask.shape} to the common shape", flush=True)
            rows, cols = min(label.shape[0], mask.shape[0]), min(label.shape[1], mask.shape[1])
            label, mask = label[:rows, :cols], mask[:rows, :cols]
        ink, extra = split_extra_labels(label)
        if alias in VAL:
            _, components, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
            tiny = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < 8) + 1
            if len(tiny):
                ink[np.isin(components, tiny)] = False
                print(f"[stage] {alias}: dropped {len(tiny)} ink specks under 8 px", flush=True)
        if segment.split_masked:
            split = cv2.imread(str(REPO / "train_masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)[:mask.shape[0], :mask.shape[1]]
            mask = mask & ((split >= 240) | ((split >= 100) & (split <= 143))) & ~extra
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{alias}_inklabels.png"), ink.astype(np.uint8) * 255)
        cv2.imwrite(str(folder / f"{alias}_mask.png"), mask.astype(np.uint8) * 255)
        _layers(segment.render, folder / "layers")
        print(f"[stage] {alias}: {mask.shape} ink={int(ink.sum())} mask={int(mask.sum())}", flush=True)


# ---- train ---------------------------------------------------------------------------------------------------------------

def recipe(smoke: bool) -> dict:
    sys.path.insert(0, str(FORK))
    import finetune_v8in_with_local_segment as fork

    config = json.loads(json.dumps(fork.PH1447_RECIPE))
    config["description"] = "v8-in finetune on the mix6 corpus."
    config["segments"] = {
        alias: {"base_path": GROUP[alias], "layer_range": [0, DEPTH], "reverse_layers": False} for alias in mix6.SEGMENTS
    }
    config["wandb"]["enabled"] = False
    config["wandb"]["tags"] = ["mix6", "v8in-finetune"]
    epochs = 1 if smoke else EPOCHS
    config["training"].update({
        "dataset_root": str(DATA),
        "train_segments": list(mix6.SEGMENTS),
        "val_segments": list(VAL),
        "init_ckpt_path": str((FORK / "model.safetensors").resolve()),
        # stitched metrics on the validation sheets once, at the last epoch
        "stitching_schedule": {"train_every_n_epochs": 1, "eval_every_n_epochs": epochs, "eval_every_n_epochs_plus_one": False},
    })
    config["training_hyperparameters"]["training"]["epochs"] = epochs
    return config


def last_checkpoint(smoke: bool) -> Path | None:
    """the final-epoch checkpoint (epochepoch=N.ckpt), not best-*.ckpt."""
    candidates = sorted((run_dir(smoke) / "runs").glob(f"{run_dir(smoke).name}*/**/epochepoch=*.ckpt"),
                        key=lambda path: int(path.stem.split("=")[-1]))
    final = (1 if smoke else EPOCHS) - 1
    return candidates[-1] if candidates and int(candidates[-1].stem.split("=")[-1]) == final else None


def train(smoke: bool) -> Path:
    existing = last_checkpoint(smoke)
    if existing is not None:
        print(f"[train] {existing} exists already; delete {run_dir(smoke)} to train again", flush=True)
        return existing
    if run_dir(smoke).exists():
        if smoke:
            shutil.rmtree(run_dir(smoke))
        elif list((run_dir(smoke) / "runs").glob("*/**/*.ckpt")):
            raise SystemExit(f"[train] {run_dir(smoke)} holds the checkpoints of an interrupted run; delete it to train again")
    run_dir(smoke).mkdir(parents=True, exist_ok=True)
    config_path = run_dir(smoke) / "config.json"
    config_path.write_text(json.dumps(recipe(smoke), indent=2) + "\n", encoding="utf-8")
    entry = FORK / "training" / "lib" / "train_resnet3d.py"
    # the author's finetune_v8in_with_local_segment.py settings
    env = dict(os.environ, INPUT_NORM_TYPE="instance", DECODER_TYPE="2d", NEG_WEIGHT=NEG_WEIGHT, VAL_STRIDE="64",
               SCHED_WORLD_SIZE="1", NO_ALBUMENTATIONS_UPDATE="1", WANDB_MODE="disabled",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    command = [PYTHON, entry, "--metadata_json", config_path, "--outputs_path", run_dir(smoke), "--devices", "1",
               "--accelerator", "gpu", "--precision", "16-mixed", "--run_name", run_dir(smoke).name]
    if smoke:
        command += ["--limit_train_batches", "6"]
    run(command, cwd=entry.parent, env=env)
    final = last_checkpoint(smoke)
    if final is None:
        raise SystemExit(f"[train] finished without the final-epoch checkpoint under {run_dir(smoke) / 'runs'}")
    return final


def export(smoke: bool, checkpoint: Path) -> Path:
    """model.safetensors + config.json of the final checkpoint, the folder InkDetector.from_pretrained reads."""
    target = model_dir(smoke)
    if not (target / "model.safetensors").is_file():
        run([PYTHON, FORK / "training" / "export_checkpoint.py", checkpoint, target])
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=("all", "setup", "stage", "train"))
    parser.add_argument("--smoke", action="store_true", help="one epoch of 6 batches into runs/v8in_mix6_smoke")
    args = parser.parse_args()
    os.chdir(REPO)
    exported = None
    if args.command in ("all", "setup"):
        setup()
    if args.command in ("all", "stage"):
        problems = mix6.status([int(alias) for alias in mix6.SEGMENTS], derived=False)
        if problems:
            raise SystemExit("mix6 data is incomplete (python3 assemble_training_segments.py --mix6):\n  " + "\n  ".join(problems[:10]))
        stage()
    if args.command in ("all", "train"):
        checkpoint = train(args.smoke)
        exported = export(args.smoke, checkpoint)
        print(f"\n[done] v8-in mix6 fine-tune ({'smoke, 1 epoch of 6 batches' if args.smoke else f'{EPOCHS} epochs'}): "
              f"model {exported}\n       Lightning checkpoint {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
