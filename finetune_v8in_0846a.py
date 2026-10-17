"""Fine-tune v8-in (nader-fork) on the PHerc0846A training corpus of train.py, then render the held-out sheets.

corpus = the 8 trained patches of train.py (ABF, 073246, m28, straightened2 in_1 / out_1 / out_2 / out_3, PHerc0009B);
straightened2, out_4 and out_5 and the three quick patches are never trained on (visualised only).
Half-brightness strokes of dilated_inklabels (assumed letters = hard negatives) are written as background, exactly
as train.py reads them (split_extra_labels with extra_inklabels off); PHerc0009B trains on its train_masks region.

  python3 finetune_v8in_0846a.py stage                     # system python: zarr -> 24 uint8 tif layers + labels + masks
  nader-fork/.venv/bin/python finetune_v8in_0846a.py train
  nader-fork/.venv/bin/python finetune_v8in_0846a.py visualize
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
FORK = REPO / "nader-fork"
RUN_DIR = FORK / "runs" / "v8in_0846a"
DATA = RUN_DIR / "data"
VIS_DATA = RUN_DIR / "data_vis"
FIGURES = REPO / "output" / "v8in_0846a"
RUN_NAME = "v8in_0846a_8ep"
EPOCHS = 8
DEPTH = 24

# segment id (the flat alias: label and cue name) -> (render zarr id, mask dir)
TRAIN = {
    "20260927134931": ("20260927134928", "masks"),
    "20261012073246": ("20261012073244", "masks"),
    "20261012065813": ("20261012065812", "masks"),
    "20261012065817": ("20261012065816", "masks"),
    "20261012065819": ("20261012065818", "masks"),
    "20261012065821": ("20261012065820", "masks"),
    "20261012065823": ("20261012065822", "masks"),
    "20250919125754": ("20250919125754", "masks"),
}
# the held-out sheets carry blank labels, so the monitored validation set (as in the author's recipe) is two trained patches
VAL = ("20261012065813", "20261012065823")
# held-out sheets rendered at the end: display name -> (zarr id, mask dir)
HOLDOUT = {
    "auto_grown_20261012065803387_straightened2_abf": ("20261012065814", "masks"),
    "auto_grown_20261012065803387_straightened2_abf_out_4": ("20261012065824", "masks"),
    "auto_grown_20261012065803387_straightened2_abf_out_5": ("20261012065826", "masks"),
    "auto_grown_20260926074031926_abf": ("auto_grown_20260926074031926_abf", "masks/extras"),
    "auto_grown_20260927143929299_abf": ("auto_grown_20260927143929299_abf", "masks/extras"),
    "auto_grown_20260927154033403_abf": ("auto_grown_20260927154033403_abf", "masks/extras"),
}
GROUP = {"20250919125754": "PHerc0009B"}  # every other segment is PHerc0846A


def _layers(zarr_id: str, destination: Path) -> None:
    """the central DEPTH slices of the render as 00.tif..23.tif (uint8, forward order)."""
    import numpy as np
    import tifffile
    import zarr

    volume = zarr.open(str(REPO / "ves_zarrs2" / f"{zarr_id}.zarr"), mode="r")
    start = (volume.shape[0] - DEPTH) // 2
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(DEPTH):
        path = destination / f"{index:02d}.tif"
        if not path.is_file():
            tifffile.imwrite(path, np.clip(np.asarray(volume[start + index]), 0, 255).astype(np.uint8))


def _mask(mask_dir: str, zarr_id: str):
    import cv2

    mask = cv2.imread(str(REPO / mask_dir / f"{zarr_id}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(REPO / mask_dir / f"{zarr_id}.png")
    return mask > 0


def stage() -> None:
    import cv2
    import numpy as np

    sys.path.insert(0, str(REPO))
    from utils.dataloader import split_extra_labels

    for segment, (zarr_id, mask_dir) in TRAIN.items():
        folder = DATA / segment
        mask = _mask(mask_dir, zarr_id)
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{segment}.png"), cv2.IMREAD_GRAYSCALE)
        if label is None or label.shape != mask.shape:
            raise ValueError(f"{segment}: label {None if label is None else label.shape} vs mask {mask.shape}")
        ink, _ = split_extra_labels(label)
        if segment == "20250919125754":
            split = cv2.imread(str(REPO / "train_masks" / f"{segment}.png"), cv2.IMREAD_GRAYSCALE)
            mask = mask & (split >= 240)
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{segment}_inklabels.png"), ink.astype(np.uint8) * 255)
        cv2.imwrite(str(folder / f"{segment}_mask.png"), mask.astype(np.uint8) * 255)
        _layers(zarr_id, folder / "layers")
        print(f"[stage] train {segment}: {mask.shape} ink={int(ink.sum())} mask={int(mask.sum())}", flush=True)
    for name, (zarr_id, mask_dir) in HOLDOUT.items():
        _layers(zarr_id, VIS_DATA / name / "layers")
        print(f"[stage] visualise {name}", flush=True)


def _recipe() -> dict:
    sys.path.insert(0, str(FORK))
    import finetune_v8in_with_local_segment as fork

    config = json.loads(json.dumps(fork.PH1447_RECIPE))
    segments = {
        segment: {"base_path": GROUP.get(segment, "PHerc0846A"), "layer_range": [0, DEPTH], "reverse_layers": False}
        for segment in TRAIN
    }
    config["description"] = "v8-in finetune on the PHerc0846A training corpus (+ PHerc0009B)."
    config["segments"] = segments
    config["wandb"]["enabled"] = False
    config["wandb"]["tags"] = ["0846a", "v8in-finetune"]
    training = config["training"]
    training.update({
        "dataset_root": str(DATA),
        "train_segments": list(TRAIN),
        "val_segments": list(VAL),
        "init_ckpt_path": str((FORK / "model.safetensors").resolve()),
        # stitched metrics on the held-out sheets once, at the last epoch
        "stitching_schedule": {"train_every_n_epochs": 1, "eval_every_n_epochs": EPOCHS,
                               "eval_every_n_epochs_plus_one": False},
    })
    config["training_hyperparameters"]["training"]["epochs"] = EPOCHS
    return config


def train() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config_path = RUN_DIR / "config.json"
    config_path.write_text(json.dumps(_recipe(), indent=2) + "\n", encoding="utf-8")
    entry = FORK / "training" / "lib" / "train_resnet3d.py"
    # the author's finetune_v8in_with_local_segment.py settings
    env = dict(os.environ, INPUT_NORM_TYPE="instance", DECODER_TYPE="2d", NEG_WEIGHT="25", VAL_STRIDE="64",
               SCHED_WORLD_SIZE="1", NO_ALBUMENTATIONS_UPDATE="1", WANDB_MODE="disabled")
    command = [sys.executable, str(entry), "--metadata_json", str(config_path), "--outputs_path", str(RUN_DIR),
               "--devices", "1", "--accelerator", "gpu", "--precision", "16-mixed", "--run_name", RUN_NAME]
    print("launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=entry.parent, env=env, check=True)


def _last_checkpoint() -> Path:
    candidates = sorted((RUN_DIR / "runs").glob(f"{RUN_NAME}*/**/*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint under {RUN_DIR / 'runs'}")
    return candidates[-1]


def visualize() -> None:
    import cv2
    import numpy as np
    import zarr
    from PIL import Image

    sys.path.insert(0, str(FORK))
    sys.path.insert(0, str(REPO))
    from ink8um import InkDetector, predict_surface
    from utils.fast_figure import colorize, gray_rgb, orient_display, save_panel_grid

    checkpoint = _last_checkpoint()
    model_dir = RUN_DIR / "model"
    subprocess.run([sys.executable, str(FORK / "training" / "export_checkpoint.py"), str(checkpoint), str(model_dir)],
                   check=True)
    print(f"[visualize] checkpoint {checkpoint}", flush=True)
    model = InkDetector.from_pretrained(str(model_dir))
    FIGURES.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    for name, (zarr_id, mask_dir) in HOLDOUT.items():
        prediction = predict_surface(model, VIS_DATA / name / "layers", reverse=model.reverse_layers)
        mask = _mask(mask_dir, zarr_id)
        volume = zarr.open(str(REPO / "ves_zarrs2" / f"{zarr_id}.zarr"), mode="r")
        start = (volume.shape[0] - DEPTH) // 2
        composite = np.clip(np.asarray(volume[start:start + DEPTH]).max(axis=0), 0, 255).astype(np.uint8) * mask
        shown = np.where(mask, prediction, np.nan)
        shown, composite = orient_display(shown, name), orient_display(composite, name)
        panels = [[
            (f"{name} | v8-in fine-tuned on 0846A, {EPOCHS} epochs",
             colorize(shown, "gray_r", (0.78, 0.86, 0.95)), cv2.INTER_AREA, None),
            (f"composite of layers {start}-{start + DEPTH - 1}", gray_rgb(composite), cv2.INTER_AREA, 9.362),
        ]]
        path = FIGURES / f"{name}.png"
        save_panel_grid(panels, composite.shape, str(path), gap_px=24)
        np.save(FIGURES / f"{name}.npy", prediction.astype(np.float16))
        print(f"[visualize] saved {path}", flush=True)


# trained patches shown as the notebook does: display name -> (render zarr id, flat alias of label/cue/staged layers)
TRAIN_VIS = {
    "auto_grown_20260927134928530_abf": ("20260927134928", "20260927134931"),
    "auto_grown_20261012073244544_abf": ("20261012073244", "20261012073246"),
    "auto_grown_20261012065803387_straight_abf_m28": ("20261012065812", "20261012065813"),
    "auto_grown_20261012065803387_straightened2_abf_in_1": ("20261012065816", "20261012065817"),
    "auto_grown_20261012065803387_straightened2_abf_out_1": ("20261012065818", "20261012065819"),
    "auto_grown_20261012065803387_straightened2_abf_out_2": ("20261012065820", "20261012065821"),
    "auto_grown_20261012065803387_straightened2_abf_out_3": ("20261012065822", "20261012065823"),
    "PHerc0009B_20250919125754": ("20250919125754", "20250919125754"),
}


def visualize_all() -> None:
    """notebook-style figures for every trained patch (2x2 with red inklabel overlay and asf_morph cue) and every
    held-out sheet (prediction | composite), reusing the predictions that `visualize` already saved."""
    import cv2
    import numpy as np
    import zarr
    from PIL import Image

    sys.path.insert(0, str(FORK))
    sys.path.insert(0, str(REPO))
    from ink8um import InkDetector, predict_surface
    from utils.fast_figure import colorize, gray_rgb, orient_display, red_overlay, save_panel_grid

    model_dir = RUN_DIR / "model"
    if not (model_dir / "model.safetensors").is_file():
        subprocess.run([sys.executable, str(FORK / "training" / "export_checkpoint.py"), str(_last_checkpoint()),
                        str(model_dir)], check=True)
    model = InkDetector.from_pretrained(str(model_dir))
    Image.MAX_IMAGE_PIXELS = None
    folder = FIGURES / "all"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(name, zarr_id, "masks", DATA / alias / "layers", alias) for name, (zarr_id, alias) in TRAIN_VIS.items()]
    jobs += [(name, zarr_id, mask_dir, VIS_DATA / name / "layers", None) for name, (zarr_id, mask_dir) in HOLDOUT.items()]
    for name, zarr_id, mask_dir, layers, alias in jobs:
        cached = FIGURES / f"{name}.npy"
        prediction = (np.load(cached).astype(np.float32) if cached.is_file()
                      else predict_surface(model, layers, reverse=model.reverse_layers))
        if not cached.is_file():
            np.save(cached, prediction.astype(np.float16))
        mask = _mask(mask_dir, zarr_id)
        volume = zarr.open(str(REPO / "ves_zarrs2" / f"{zarr_id}.zarr"), mode="r")
        start = (volume.shape[0] - DEPTH) // 2
        composite = np.clip(np.asarray(volume[start:start + DEPTH]).max(axis=0), 0, 255).astype(np.uint8) * mask
        shown = orient_display(np.where(mask, prediction, np.nan), name)
        composite = orient_display(composite, name)
        pred_rgb = colorize(shown, "gray_r", (0.78, 0.86, 0.95))
        title = f"{name} | v8-in fine-tuned on 0846A, {EPOCHS} epochs"
        composite_panel = (f"composite of layers {start}-{start + DEPTH - 1}", gray_rgb(composite), cv2.INTER_AREA, 9.362)
        if alias is None:
            panels = [[(title, pred_rgb, cv2.INTER_AREA, None), composite_panel]]
        else:
            label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE) > 0
            cue = np.load(REPO / "cue_asf_morph" / f"{alias}.npy")
            if label.shape != mask.shape or cue.shape != mask.shape:
                raise ValueError(f"{name}: mask {mask.shape}, label {label.shape}, cue {cue.shape} differ")
            overlay = red_overlay(cv2.resize(pred_rgb, (composite.shape[1], composite.shape[0]),
                                             interpolation=cv2.INTER_NEAREST), orient_display(label, name))
            panels = [
                [(title, pred_rgb, cv2.INTER_AREA, None), composite_panel],
                [(f"{name} | inklabel overlay (red)", overlay, cv2.INTER_AREA, None),
                 ("asf_morph cue", gray_rgb(orient_display(cue, name)), cv2.INTER_AREA, None)],
            ]
        path = folder / f"{name}.png"
        save_panel_grid(panels, composite.shape, str(path), gap_px=24)
        print(f"[visualize_all] saved {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("stage", "train", "visualize", "visualize_all"))
    arguments = parser.parse_args()
    os.chdir(REPO)
    {"stage": stage, "train": train, "visualize": visualize, "visualize_all": visualize_all}[arguments.command]()
