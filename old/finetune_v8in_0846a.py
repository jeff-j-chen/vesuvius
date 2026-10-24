"""Fine-tune v8-in (nader-fork, PH1447 recipe, base weights, 6 epochs) on PHerc0846A + PHerc0343, then render sheets.

corpus = the 7 trained 0846A patches (ABF, 073246, m28, straightened2 in_1 / out_1 / out_2 / out_3) and PHerc0343
first letters (20261007000000; not 343P); straightened2, out_4 and out_5 and the three quick patches are never
trained on (visualised only). Half-brightness strokes of dilated_inklabels (hard negatives in 0846A) are written as
background, exactly as train.py reads them (split_extra_labels with extra_inklabels off); 343 trains on its
train_masks region.

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

REPO = Path(__file__).resolve().parent.parent  # moved to old/: superseded by finetune_readerv3.py / finetune_v8in.py
FORK = REPO / "nader-fork"
CONT = os.environ.get("V8IN_CONT") == "1"  # 3 more epochs from the epoch-6 weights: lr 3e-5 -> 3e-6, NEG_WEIGHT 12
V5 = os.environ.get("V8IN_V5") == "1"  # base weights again on the v5 inklabels (re-staged over the old ones), 5 epochs, own run dir
X = os.environ.get("V8IN_X") == "1"  # v6 labels, 7 0846A + 343 first letters + 0009B (NOT 343P), 5 epochs, own run dir
V5 = V5 or X
M = os.environ.get("V8IN_M") == "1"  # v6 labels, the mix6 dataset (7 0846A + 0009B + 1447 + 0841 + 0814 + 0172 + 343 first letters, NOT 343P), 5 epochs, own data/run dirs
V5 = V5 or M
DATA = FORK / "runs" / ("v8in_v6m_data" if M else "v8in_0846a_343") / "data"
VIS_DATA = FORK / "runs" / "v8in_0846a_343" / "data_vis"
FIGURES = REPO / "output" / ("v8in_v6m_5ep" if M else "v8in_v6x_5ep" if X else "v8in_v5_5ep" if V5 else "v8in_0846a_343_cont3" if CONT else "v8in_0846a_343")
EPOCHS = 5 if V5 else 3 if CONT else 6  # 6 is the recipe's value
NEG_WEIGHT = "12" if CONT else "25"
TITLE_EPOCHS = "6+3 epochs (lr 3e-5, NEG_WEIGHT 12)" if CONT else f"{EPOCHS} epochs"
# the continuation only renders straightened2 and one training fragment
ONLY = {"auto_grown_20261012065803387_straightened2_abf", "auto_grown_20261012073244544_abf"} if CONT else None
SMOKE = os.environ.get("V8IN_SMOKE") == "1"  # a few batches per epoch, separate run dir
if M and SMOKE:
    EPOCHS = 1  # the last epoch carries the stitched evaluation, so one epoch exercises it
RUN_DIR = FORK / "runs" / ("v8in_v6m_smoke" if M and SMOKE else "v8in_0846a_343_smoke" if SMOKE else "v8in_v6m_5ep" if M else "v8in_v6x_5ep" if X else "v8in_v5_5ep" if V5 else "v8in_0846a_343_cont3" if CONT else "v8in_0846a_343")
RUN_NAME = "v8in_v6m_smoke" if M and SMOKE else "v8in_0846a_343_smoke" if SMOKE else "v8in_v6m_5ep" if M else "v8in_v6x_5ep" if X else "v8in_v5_5ep" if V5 else "v8in_0846a_343_ep6plus3" if CONT else "v8in_0846a_343_6ep"
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
    "20261007000000": ("20261007000000", "masks"),  # PHerc0343
}
SPLIT_MASKED = ("20261007000000",)  # trains on its train_masks region only
GROUP_0846A = tuple(TRAIN)[:7]
if X:
    TRAIN["20250919125754"] = ("20250919125754", "masks")  # PHerc0009B
    SPLIT_MASKED = ("20261007000000", "20250919125754")
GROUP = {"20261007000000": "PHerc0343", "20250919125754": "PHerc0009B"}  # every other segment is PHerc0846A
if M:
    # render id == flat alias for these; every one trains on its train_masks region (255 trains, 119 = hard negative, black = unused)
    for _segment in ("20250919125754", "20260930144758", "20260930144760", "20260221022814", "20260226000000", "20251111010954", "20251112000002"):
        TRAIN[_segment] = (_segment, "masks")
    SPLIT_MASKED = tuple(segment for segment in TRAIN if segment not in GROUP_0846A)
    GROUP.update({"20250919125754": "PHerc0009B", "20260930144758": "PHerc1447", "20260930144760": "PHerc1447",
                  "20260221022814": "PHerc0841", "20260226000000": "PHerc0814",
                  "20251111010954": "PHerc0172", "20251112000002": "PHerc0172"})
    # the sampler gives every group the same share; two 0846A groups make 0846A weigh 2 against 1 for each other domain, as in mix6
    GROUP.update({segment: "PHerc0846A_b" for segment in list(GROUP_0846A)[4:]})
# monitored validation set (as in the author's recipe: trained segments): two 0846A patches and 343
VAL = ("20261012065813", "20261012065823", "20261007000000")
# held-out sheets rendered at the end: display name -> (zarr id, mask dir)
HOLDOUT = {
    "auto_grown_20261012065803387_straightened2_abf": ("20261012065814", "masks"),
    "auto_grown_20261012065803387_straightened2_abf_out_4": ("20261012065824", "masks"),
    "auto_grown_20261012065803387_straightened2_abf_out_5": ("20261012065826", "masks"),
    "auto_grown_20260926074031926_abf": ("auto_grown_20260926074031926_abf", "masks/extras"),
    "auto_grown_20260927143929299_abf": ("auto_grown_20260927143929299_abf", "masks/extras"),
    "auto_grown_20260927154033403_abf": ("auto_grown_20260927154033403_abf", "masks/extras"),
}


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
        if label is None:
            raise FileNotFoundError(f"{segment}: no label")
        if label.shape != mask.shape:
            print(f"[stage] {segment}: cropping label {label.shape} / mask {mask.shape} to the common shape", flush=True)
            rows, cols = min(label.shape[0], mask.shape[0]), min(label.shape[1], mask.shape[1])
            label, mask = label[:rows, :cols], mask[:rows, :cols]
        ink, extra = split_extra_labels(label)
        if M and segment in VAL:
            # the stitched validation metrics crash on 1-2 px ink specks (empty Canny contour at 4x downsampling)
            _, components, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
            tiny = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < 8) + 1
            if len(tiny):
                ink[np.isin(components, tiny)] = False
                print(f"[stage] {segment}: dropped {len(tiny)} ink specks under 8 px", flush=True)
        if segment in SPLIT_MASKED:
            split = cv2.imread(str(REPO / "train_masks" / f"{segment}.png"), cv2.IMREAD_GRAYSCALE)
            split = split[:mask.shape[0], :mask.shape[1]]
            if M:
                # 255 trains, 119 is a hard negative (background inside the mask), black is unused; half-brightness inklabel strokes are unused letters
                mask = mask & ((split >= 240) | ((split >= 100) & (split <= 143))) & ~extra
            else:
                mask = mask & (split >= 240)
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{segment}_inklabels.png"), ink.astype(np.uint8) * 255)
        cv2.imwrite(str(folder / f"{segment}_mask.png"), mask.astype(np.uint8) * 255)
        old_layers = FORK / "runs" / "v8in_0846a_343" / "data" / segment / "layers"
        if M and not (folder / "layers").exists() and (old_layers / f"{DEPTH - 1:02d}.tif").is_file():
            (folder / "layers").symlink_to(old_layers)  # identical central slices, already staged
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
    config["description"] = "v8-in finetune on PHerc0846A + PHerc0343."
    config["segments"] = segments
    config["wandb"]["enabled"] = False
    config["wandb"]["tags"] = ["0846a", "v8in-finetune"]
    training = config["training"]
    training.update({
        "dataset_root": str(DATA),
        "train_segments": list(TRAIN),
        "val_segments": list(VAL),
        "init_ckpt_path": str(((FORK / "runs" / "v8in_0846a_343" / "model" / "model.safetensors") if CONT else (FORK / "model.safetensors")).resolve()),
        # stitched metrics on the held-out sheets once, at the last epoch
        "stitching_schedule": {"train_every_n_epochs": 1, "eval_every_n_epochs": EPOCHS,
                               "eval_every_n_epochs_plus_one": False},
    })
    config["training_hyperparameters"]["training"]["epochs"] = EPOCHS
    if CONT:
        config["training_hyperparameters"]["training"].update({"lr": 3e-5, "min_lr": 3e-6})
    return config


def train() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config_path = RUN_DIR / "config.json"
    config_path.write_text(json.dumps(_recipe(), indent=2) + "\n", encoding="utf-8")
    entry = FORK / "training" / "lib" / "train_resnet3d.py"
    # the author's finetune_v8in_with_local_segment.py settings
    env = dict(os.environ, INPUT_NORM_TYPE="instance", DECODER_TYPE="2d", NEG_WEIGHT=NEG_WEIGHT, VAL_STRIDE="64",
               SCHED_WORLD_SIZE="1", NO_ALBUMENTATIONS_UPDATE="1", WANDB_MODE="disabled")
    command = [sys.executable, str(entry), "--metadata_json", str(config_path), "--outputs_path", str(RUN_DIR),
               "--devices", "1", "--accelerator", "gpu", "--precision", "16-mixed", "--run_name", RUN_NAME]
    if SMOKE:
        command += ["--limit_train_batches", "6"]
    if os.environ.get("V8IN_RESUME"):
        command += ["--resume_from_ckpt", os.environ["V8IN_RESUME"]]
    print("launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=entry.parent, env=env, check=True)


def _last_checkpoint() -> Path:
    # the final-epoch checkpoint (epochepoch=N.ckpt), not best-*.ckpt
    candidates = sorted((RUN_DIR / "runs").glob(f"{RUN_NAME}*/**/epochepoch=*.ckpt"),
                        key=lambda path: int(path.stem.split("=")[-1]))
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
        if ONLY is not None and name not in ONLY:
            continue
        prediction = predict_surface(model, VIS_DATA / name / "layers", reverse=model.reverse_layers)
        mask = _mask(mask_dir, zarr_id)
        volume = zarr.open(str(REPO / "ves_zarrs2" / f"{zarr_id}.zarr"), mode="r")
        start = (volume.shape[0] - DEPTH) // 2
        composite = np.clip(np.asarray(volume[start:start + DEPTH]).max(axis=0), 0, 255).astype(np.uint8) * mask
        shown = np.where(mask, prediction, np.nan)
        shown, composite = orient_display(shown, name), orient_display(composite, name)
        panels = [[
            (f"{name} | v8-in fine-tuned on 0846A + 343, {TITLE_EPOCHS}",
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
    "PHerc0343_first_letters_20261007000000": ("20261007000000", "20261007000000"),
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
    only = ONLY
    if CONT and os.environ.get("V8IN_ALL") == "1":  # straightened2 plus the seven trained 0846A fragments
        only = {name for name in TRAIN_VIS if not name.startswith("PHerc0343")} | {"auto_grown_20261012065803387_straightened2_abf"}
    jobs = [job for job in jobs if only is None or job[0] in only]
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
        title = f"{name} | v8-in fine-tuned on 0846A + 343, {TITLE_EPOCHS}"
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
