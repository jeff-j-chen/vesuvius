"""the mix6 corpus: one definition shared by the downloaders, train.py, and the reader-v3 / v8-in fine-tunes.

mix6 = the 7 trained PHerc0846A sheets + PHerc0343 first letters (not 343P) + PHerc0009B + PHerc1447 w058/w060 + PHerc0841 +
PHerc0814 + PHerc0172 w068/w087, with the straightened2 sheet held out (rendered, never trained or validated on).

`python3 -m utils.mix6` prints the state of every artifact the three trainings read (read-only);
`--build` makes the missing derived ones (flat 1447 surfaces, surface anchors, 3 depths of asf cues).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent

# 0846A sheets: flat alias (the label / cue / norm / train-mask name) -> render zarr id and display name; the aliases are
# symlinks to the render (assemble_test_segments.py FLAT_ALIASES) and read a flat depth-13 surface
SHEETS_0846A = {
    20260927134931: (20260927134928, "auto_grown_20260927134928530_abf"),
    20261012073246: (20261012073244, "auto_grown_20261012073244544_abf"),
    20261012065813: (20261012065812, "auto_grown_20261012065803387_straight_abf_m28"),
    20261012065817: (20261012065816, "auto_grown_20261012065803387_straightened2_abf_in_1"),
    20261012065819: (20261012065818, "auto_grown_20261012065803387_straightened2_abf_out_1"),
    20261012065821: (20261012065820, "auto_grown_20261012065803387_straightened2_abf_out_2"),
    20261012065823: (20261012065822, "auto_grown_20261012065803387_straightened2_abf_out_3"),
}
TRAIN_0846A = tuple(SHEETS_0846A)
HOLDOUT = 20261012065815
HOLDOUT_RENDER = 20261012065814
HOLDOUT_NAME = "auto_grown_20261012065803387_straightened2_abf"
# straightened2 out_4 / out_5: whole-patch validation entries of the training config only, never downloaded for mix6
OUT_VIS = (20261012065825, 20261012065827)
FLAT_DEPTH_0846A = 13
WINDOW_OFFSET_0846A = -1

# train_scroll_dict domains after "pherc0846a" (training order); weights: 0846A 2, every other domain 1
DOMAINS = {
    "pherc0009b": [20250919125754],
    "pherc1447": [20260930144758, 20260930144760],
    "pherc0841": [20260221022814],
    "pherc0814": [20260226000000],
    "pherc0172": [20251111010954, 20251112000002],
    "pherc0343": [20261007000000],  # first letters, NOT 343P (20250511003658)
}


class Segment(NamedTuple):  # keyed by the alias: the label / cue / norm name
    render: str         # zarr / mask name
    scroll: str         # sampling group of the fine-tunes
    layer: int          # centre layer of the 28-layer render (median of the detected depth; flat for 0846A and 1447)
    split_masked: bool  # trains on its train_masks region only (255 trains, 119 = hard negative, black = unused)
    assembler: str      # name in assemble_training_segments.SEGMENTS ("" = assemble_test_segments.FRAGMENTS)


# fine-tune corpus order of reader-v3 and v8-in
SEGMENTS = {
    **{str(alias): Segment(str(render), "0846A", FLAT_DEPTH_0846A, False, "") for alias, (render, _) in SHEETS_0846A.items()},
    "20261007000000": Segment("20261007000000", "0343", 15, True, "p343_letters"),
    "20250919125754": Segment("20250919125754", "0009B", 15, True, "p9b_487"),
    "20260930144758": Segment("20260930144758", "1447", 14, True, "ph1447_w058"),
    "20260930144760": Segment("20260930144760", "1447", 15, True, "ph1447_w060"),
    "20260221022814": Segment("20260221022814", "0841", 15, True, "p841"),
    "20260226000000": Segment("20260226000000", "0814", 16, True, "seg46527"),
    "20251111010954": Segment("20251111010954", "0172", 15, True, "w068"),
    "20251112000002": Segment("20251112000002", "0172", 15, True, "w087"),
}
# PHerc1447 has no usable surface detection: its surface is flat at the layer with the highest mean intensity
FLAT_SURFACE = {20260930144758: 14, 20260930144760: 15}
TRAINING_NAMES = tuple(seg.assembler for seg in SEGMENTS.values() if seg.assembler)


# where the three trainings leave their model (the keys are utils/predict_models.KINDS); ensemble_eval.ipynb and 0846A_eval.ipynb read them here
CUE_DROPOUT_RUN = "cue_dropout_labelsv6_mix6"
MODEL_PATHS = {
    "cue_dropout": REPO / "models" / "runs" / CUE_DROPOUT_RUN / "best_model_loss.pth",   # train.py
    "reader_v3": REPO / "reader-v3" / "runs" / "mix6" / "ckpt_002000.pth",               # finetune_readerv3.py, step 2000
    "v8in": REPO / "nader-fork" / "runs" / "v8in_mix6_5ep" / "model",                    # finetune_v8in.py, epoch 5 exported
}


def train_ids() -> list[int]:
    """the 15 trained aliases in the order of the training config (0846A first, then the other domains)."""
    return [*TRAIN_0846A, *(sid for ids in DOMAINS.values() for sid in ids)]


def vis_ids() -> list[int]:
    return [*TRAIN_0846A, HOLDOUT]


def all_ids() -> list[int]:
    return [*train_ids(), HOLDOUT]


def render_id(alias) -> str:
    alias = int(alias)
    if alias == HOLDOUT:
        return str(HOLDOUT_RENDER)
    return str(SHEETS_0846A[alias][0]) if alias in SHEETS_0846A else str(alias)


def render_ids_0846a() -> list[str]:
    """render zarr ids assemble_test_segments.py has to produce for mix6 (7 trained sheets + the holdout)."""
    return [render_id(alias) for alias in (*TRAIN_0846A, HOLDOUT)]


def zarr_dir() -> Path:
    from utils.platform import get_zarr_dir

    return Path(get_zarr_dir())


def zarr_mtime(path) -> float:
    """newest change of a zarr's metadata: a re-render recreates it (rewriting chunks alone does not count)."""
    path = Path(path)
    return max(((path / name).stat().st_mtime for name in (".zarray", ".zattrs", ".zgroup") if (path / name).exists()), default=0.0)


# ---- surface -----------------------------------------------------------------------------------------------------------

def centre_layer(render: str) -> tuple[int, "object"]:
    """smoothed argmax of the mean intensity per layer over the sheet (one strided read); the first and last layer are skipped."""
    import cv2
    import numpy as np
    import zarr

    volume = zarr.open(str(zarr_dir() / f"{render}.zarr"), mode="r")
    mask = cv2.imread(str(REPO / "masks" / f"{render}.png"), cv2.IMREAD_GRAYSCALE)[::16, ::16] > 0
    stack = np.asarray(volume[:, ::16, ::16]).astype(np.float32)
    rows, cols = min(mask.shape[0], stack.shape[1]), min(mask.shape[1], stack.shape[2])
    profile = stack[:, :rows, :cols][:, mask[:rows, :cols]].mean(axis=1)
    valid = np.flatnonzero(profile > 0)  # the 24 released PHerc1447 layers cover layers 3 to 24 of the 28
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    inner = valid[1:-1]
    return int(inner[smooth[inner].argmax()]), profile


def is_flat_surface(alias) -> bool:
    meta = REPO / "surface_labels" / str(alias) / "metadata.json"
    return meta.is_file() and "flat_depth" in json.loads(meta.read_text())


def write_flat_surface(alias, force: bool = False) -> str:
    """surface_labels/<alias>: a flat surface at the measured centre layer (PHerc1447). a detected surface already there
    is kept under surface_labels_detected/<alias>; cached surface anchors are dropped."""
    import shutil

    import cv2
    import numpy as np

    alias = str(alias)
    if is_flat_surface(alias) and not force:
        return f"{alias}: flat surface exists"
    import zarr

    volume = zarr.open(str(zarr_dir() / f"{alias}.zarr"), mode="r")
    if not volume.attrs.get("depth_flipped", False):
        raise RuntimeError(f"{alias}: the stored stack is not the depth-reversed one (attrs depth_flipped missing)")
    depth, _ = centre_layer(alias)
    folder = REPO / "surface_labels" / alias
    backup = REPO / "surface_labels_detected" / alias
    if folder.exists() and not backup.exists() and not is_flat_surface(alias):
        backup.parent.mkdir(exist_ok=True)
        shutil.copytree(folder, backup)
    folder.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(REPO / "masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE) > 0
    np.save(folder / "depth.npy", np.where(mask, depth, 255).astype(np.uint8))
    np.save(folder / "confidence.npy", np.where(mask, 255, 0).astype(np.uint8))
    (folder / "metadata.json").write_text(json.dumps({
        "scroll_id": int(alias), "flat_depth": depth, "flat_rule": "smoothed argmax of the mean layer intensity",
        "invalid_depth_value": 255}, indent=2))
    cache = REPO / "surface_anchor_cache.json"
    if cache.is_file():
        data = json.loads(cache.read_text())
        if data.pop(alias, None) is not None:
            cache.write_text(json.dumps(data, indent=2))
    return f"{alias}: flat surface written at layer {depth}"


def flat_surface_problems(ids=None) -> list[str]:
    """PHerc1447 trains on a flat surface of its depth-reversed stack."""
    problems = []
    for alias in (ids or FLAT_SURFACE):
        if int(alias) in FLAT_SURFACE and not is_flat_surface(alias):
            problems.append(f"{alias}: surface is not the flat PHerc1447 one (surface_labels/{alias}/metadata.json); "
                            "run python3 -m utils.mix6 --build")
    return problems


def ensure_anchors(ids=None) -> None:
    """surface-anchor normalisation of every mix6 scroll (cached in surface_anchor_cache.json)."""
    from utils.norm import ensure_surface_anchors

    os.chdir(REPO)
    ensure_surface_anchors([str(sid) for sid in (ids or all_ids())], str(zarr_dir()), surface_dir="./surface_labels")


# ---- asf cues: depth shifts 0 (cue_dir), -1 and +1 (the windows depth jitter reads) -------------------------------------------

CUE_DIRS = {0: "cue_asf_morph", -1: "cue_asf_morph_dm1", 1: "cue_asf_morph_dp1"}


def window_offset(alias) -> int:
    return WINDOW_OFFSET_0846A if int(alias) in (*TRAIN_0846A, HOLDOUT) else 0


def cue_file(alias, delta: int = 0) -> Path:
    return REPO / CUE_DIRS[delta] / f"{alias}.npy"


def cue_config(alias, delta: int):
    from utils.config import Config

    config = Config()
    config.data.surface_window_offset_by_scroll = {int(alias): window_offset(alias) + delta}
    config.data.cue_dir = f"./{CUE_DIRS[delta]}"
    return config


def cue_shifts(alias) -> tuple[int, ...]:
    """trained scrolls read the depth-jittered windows too; the holdout is only rendered."""
    return (0,) if int(alias) == HOLDOUT else tuple(CUE_DIRS)


def _stale(path: Path, alias) -> bool:
    sources = [REPO / "surface_labels" / str(alias) / "depth.npy", REPO / "masks" / f"{alias}.png"]
    rendered = zarr_mtime(zarr_dir() / f"{render_id(alias)}.zarr")
    return rendered > path.stat().st_mtime or any(source.exists() and source.stat().st_mtime > path.stat().st_mtime for source in sources)


def cue_jobs(ids=None, force: bool = False) -> list[tuple[int, int, bool]]:
    """(alias, shift, force) for every missing cue map (every one with force)."""
    jobs = []
    for alias in (ids or all_ids()):
        for delta in cue_shifts(alias):
            if force or not cue_file(alias, delta).is_file():
                jobs.append((int(alias), delta, True))
    return jobs


def _build_cue(job) -> str:
    alias, delta, force = job
    os.chdir(REPO)
    for path in (str(REPO), str(REPO / "old")):  # nonml_ink.py lives in old/
        if path not in sys.path:
            sys.path.insert(0, path)
    from utils import precomputed_asf

    return f"{alias} shift {delta:+d}: " + precomputed_asf.build(alias, force=force, config=cue_config(alias, delta))


def ensure_cues(ids=None, workers: int = 4, force: bool = False) -> None:
    jobs = cue_jobs(ids, force)
    print(f"[mix6] {len(jobs)} asf cue map(s) to build", flush=True)
    if not jobs:
        return
    import multiprocessing

    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), mp_context=multiprocessing.get_context("spawn")) as pool:
        for line in pool.map(_build_cue, jobs):
            print("[mix6] asf cue", line, flush=True)


# ---- state -------------------------------------------------------------------------------------------------------------------

def stale_cues(ids=None) -> list[str]:
    """cue maps older than their render, surface labels or mask (rebuilt only with --rebuild-cues)."""
    return [f"{alias}: asf cue {cue_file(alias, delta)} is older than its render, surface labels or mask"
            for alias in (ids or all_ids()) for delta in cue_shifts(alias)
            if cue_file(alias, delta).is_file() and _stale(cue_file(alias, delta), alias)]


def layer_warnings(ids=None) -> list[str]:
    """the fine-tunes pin the surface layer of every scroll; flag a detected surface whose median differs."""
    import numpy as np

    warnings = []
    for alias in (ids or all_ids()):
        path = REPO / "surface_labels" / str(alias) / "depth.npy"
        if str(alias) not in SEGMENTS or not path.is_file():
            continue
        depth = np.asarray(np.load(path, mmap_mode="r")[::16, ::16])
        depth = depth[depth != 255]
        median = int(np.median(depth)) if depth.size else None
        if median != SEGMENTS[str(alias)].layer:
            warnings.append(f"{alias}: median surface layer {median}, the fine-tunes pin {SEGMENTS[str(alias)].layer}")
    return warnings


def status(ids=None, derived: bool = True) -> list[str]:
    """read-only: one line per missing artifact of the given mix6 aliases (default all 16).
    derived=False checks only the downloaded and tracked inputs (render, mask, label, train mask), not what is built from them."""
    zarrs = zarr_dir()
    norms = json.loads((REPO / "norm_cache.json").read_text()) if (REPO / "norm_cache.json").is_file() else {}
    anchors = json.loads((REPO / "surface_anchor_cache.json").read_text()) if (REPO / "surface_anchor_cache.json").is_file() else {}
    problems = []
    for alias in (ids or all_ids()):
        alias, render = str(alias), render_id(alias)
        paths = [zarrs / f"{render}.zarr", REPO / "masks" / f"{render}.png", REPO / "dilated_inklabels" / f"{alias}.png"]
        if int(alias) != HOLDOUT:
            paths.append(REPO / "train_masks" / f"{alias}.png")
        if derived:
            paths += [REPO / "surface_labels" / alias / "depth.npy", REPO / "surface_labels" / alias / "confidence.npy"]
            if int(alias) in SHEETS_0846A or int(alias) == HOLDOUT:
                paths += [zarrs / f"{alias}.zarr", REPO / "masks" / f"{alias}.png"]
        problems += [f"{alias}: missing {path}" for path in paths if not path.exists()]
        if not derived:
            continue
        for name, table in (("norm_cache.json", norms), ("surface_anchor_cache.json", anchors)):
            if alias not in table:
                problems.append(f"{alias}: no entry in {name}")
        for delta in cue_shifts(alias):
            if not cue_file(alias, delta).is_file():
                problems.append(f"{alias}: missing asf cue {cue_file(alias, delta)}")
        problems += flat_surface_problems([alias])
    return problems


def report(ids=None) -> list[str]:
    """print the read-only state of the given aliases (default all 16) and return the problems."""
    ids = ids or all_ids()
    stale = stale_cues(ids)
    if stale:
        print(f"  WARN: {len(stale)} asf cue map(s) are older than their render, surface labels or mask; they are kept "
              f"(--rebuild-cues rebuilds them): {', '.join(line.split(': ')[0] for line in stale)}")
    for line in layer_warnings(ids):
        print("  WARN:", line)
    problems = status(ids)
    for line in problems:
        print("  PROBLEM:", line)
    print(f"[mix6] {len(ids)} scrolls checked: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    return problems


def finalize(ids=None, workers: int = 4, rebuild_cues: bool = False) -> None:
    """flat PHerc1447 surfaces, surface anchors and the 3 depths of asf cues for the given aliases (default all 16)."""
    os.chdir(REPO)
    ids = [int(sid) for sid in (ids or all_ids())]
    for alias in ids:
        if alias in FLAT_SURFACE:
            print("[mix6]", write_flat_surface(alias), flush=True)
    ensure_anchors(ids)
    ensure_cues(ids, workers, force=rebuild_cues)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="make the missing derived artifacts (needs the zarrs and masks)")
    parser.add_argument("--workers", type=int, default=4, help="parallel asf cue builds")
    parser.add_argument("--rebuild-cues", action="store_true", help="with --build: rebuild every asf cue, not only the missing ones")
    args = parser.parse_args()
    os.chdir(REPO)
    if args.build:
        finalize(workers=args.workers, rebuild_cues=args.rebuild_cues)
    sys.exit(1 if report() else 0)


if __name__ == "__main__":
    main()
