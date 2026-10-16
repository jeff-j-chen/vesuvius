"""cross-validation arms of the best run (utils/config.py apply_best_run = iter2_asf_cue), 10 epochs each.

the seven trained PHerc0846A patches are ABF (20260927134931), 073246 (20261012073246), m28 (20261012065813) and the
straightened2 neighbours in_1 / out_1 / out_2 / out_3 (20261012065817 / 19 / 21 / 23); PHerc0009B always trains.

k-fold (cv_fold_1 .. cv_fold_3; the old 2-fold fold_1 / fold_2 runs are untouched): every PHerc0846A patch still trains, but in each fold a third of its letters (8-connected ink
    components, as the trainer's character grid defines them) and the papyrus nearest to them become validation. every
    letter is held out in exactly one fold.
leave-patches-out (loo_*): PHerc0009B plus the two named 0846A patches train; the other five 0846A patches are not
    trained on and are scored on the whole patch.
    loo_A_out2_out3: out_2 + out_3        loo_B_in1_out1: in_1 + out_1
    loo_C_073246_abf: 073246 + ABF        loo_D_m28_out3: m28 + out_3

every run renders the final eval figures for all seven 0846A patches, the winning sheet (straightened2), out_4, out_5 and
PHerc0009B. TensorBoard, figures and config.json go to runs_kfolds/<arm>, console logs to runs_kfolds/logs/<arm>.log,
checkpoints to models/runs_kfolds/<arm>, fold masks and previews to runs_kfolds/splits.

usage: python3 k_folds.py [--only cv_fold_1,loo_A_out2_out3] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
LOG_ROOT = REPO / "runs_kfolds"
MODEL_ROOT = REPO / "models" / "runs_kfolds"
SPLIT_ROOT = LOG_ROOT / "splits"
N_FOLDS = 3
N_EPOCHS = 10
FOLD_COLORS = ((230, 120, 60), (70, 170, 90), (60, 110, 230))  # BGR per fold

ABF, P073246, M28, IN_1, OUT_1, OUT_2, OUT_3 = 20260927134931, 20261012073246, 20261012065813, 20261012065817, \
    20261012065819, 20261012065821, 20261012065823
LOO_ARMS = {
    "loo_A_out2_out3": (OUT_2, OUT_3),
    "loo_B_in1_out1": (IN_1, OUT_1),
    "loo_C_073246_abf": (P073246, ABF),
    "loo_D_m28_out3": (M28, OUT_3),
}


def labeled_ids() -> list[int]:
    from utils.config import BEST_NEW_SET

    return [int(sid) for sid in BEST_NEW_SET]


def arm_names() -> list[str]:
    return [f"cv_fold_{k}" for k in range(1, N_FOLDS + 1)] + list(LOO_ARMS)


def fold_mask_dir(fold: int) -> Path:
    return SPLIT_ROOT / f"fold_{fold}" / "train_masks"


def character_grid(scroll_id: int, unit: int, min_pixels: int) -> tuple[np.ndarray, tuple[int, int]]:
    """the trainer's per-cell letter ids for this patch (regular ink only, extra_inklabels off), and its frame."""
    from utils.dataloader import DataManager, split_extra_labels

    raw = cv2.imread(str(REPO / "dilated_inklabels" / f"{scroll_id}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(REPO / "masks" / f"{scroll_id}.png"), cv2.IMREAD_GRAYSCALE)
    if raw is None or mask is None or raw.shape != mask.shape:
        raise ValueError(f"labels and mask of {scroll_id} are missing or differ in shape")
    regular, _ = split_extra_labels(raw)
    return DataManager._build_character_grid(regular.astype(np.float32), unit, min_pixels), mask.shape


def assign_folds(grid: np.ndarray) -> np.ndarray:
    """fold (0..N_FOLDS-1) per letter id; balanced by letter count, then by size."""
    ids, cells = np.unique(grid[grid > 0], return_counts=True)
    fold_of = np.full(int(grid.max()) + 1, -1, dtype=np.int32)
    counts, sizes = [0] * N_FOLDS, [0] * N_FOLDS
    for letter, size in sorted(zip(ids.tolist(), cells.tolist()), key=lambda item: -item[1]):
        fold = min(range(N_FOLDS), key=lambda f: (counts[f], sizes[f]))
        fold_of[letter] = fold
        counts[fold] += 1
        sizes[fold] += size
    return fold_of


def nearest_fold_cells(grid: np.ndarray, fold_of: np.ndarray) -> np.ndarray:
    """every target cell takes the fold of its nearest letter cell, so held-out letters keep their surroundings."""
    seeds = grid > 0
    _, nearest = cv2.distanceTransformWithLabels(
        (~seeds).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    seed_ids = grid[seeds]  # raster order matches DIST_LABEL_PIXEL numbering
    return fold_of[seed_ids[nearest - 1]]


def check_split(grid, fold_of, train_mask, fold, unit, scroll_id) -> None:
    """replay the loader's 16 px alignment and character majority rule; held-out letters must stay wholly valid."""
    from utils.dataloader import DataManager

    assignment = DataManager._align_manual_mask((train_mask >= 240).astype(np.uint8), unit)
    _, assignment = DataManager._exclude_characters_crossing_split(grid, assignment, unit)
    gh, gw = grid.shape
    cells = assignment[:gh * unit, :gw * unit].reshape(gh, unit, gw, unit).all(axis=(1, 3))
    for letter in np.unique(grid[grid > 0]):
        member = cells[grid == letter]
        expected_train = fold_of[letter] != fold
        if not (member.all() if expected_train else not member.any()):
            raise RuntimeError(f"fold {fold + 1}: letter {letter} of {scroll_id} crosses the split")


def build_splits() -> dict:
    """write one train-mask dir per fold (0846A masks rewritten, all other scrolls linked) and fold previews."""
    from utils.config import Config, apply_best_run
    from utils.fast_figure import orient_display

    config = Config()
    apply_best_run(config)
    unit = int(config.model.multitile_subtile)
    min_pixels = int(config.data.character_min_pixels)
    labeled = labeled_ids()
    summary = {}
    # generated files only; rebuilt from scratch so no mask from an older fold layout survives
    shutil.rmtree(SPLIT_ROOT, ignore_errors=True)
    for k in range(1, N_FOLDS + 1):
        folder = fold_mask_dir(k)
        folder.mkdir(parents=True, exist_ok=True)
        for scroll in config.data.scrolls:
            if int(scroll.scroll_id) in labeled:
                continue
            link = folder / f"{scroll.scroll_id}.png"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(REPO / "train_masks" / f"{scroll.scroll_id}.png")
    for sid in labeled:
        grid, shape = character_grid(sid, unit, min_pixels)
        fold_of = assign_folds(grid)
        cell_fold = nearest_fold_cells(grid, fold_of)
        pixel_fold = np.repeat(np.repeat(cell_fold, unit, axis=0), unit, axis=1)
        pixel_fold = np.pad(
            pixel_fold, ((0, shape[0] - pixel_fold.shape[0]), (0, shape[1] - pixel_fold.shape[1])), mode="edge"
        )
        for k in range(N_FOLDS):
            train_mask = np.where(pixel_fold == k, 0, 255).astype(np.uint8)
            check_split(grid, fold_of, train_mask, k, unit, sid)
            if not cv2.imwrite(str(fold_mask_dir(k + 1) / f"{sid}.png"), train_mask):
                raise RuntimeError(f"could not write the fold {k + 1} mask of {sid}")
        letters = np.unique(grid[grid > 0])
        summary[sid] = [int((fold_of[letters] == k).sum()) for k in range(N_FOLDS)]

        ink = cv2.imread(str(REPO / "dilated_inklabels" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE) >= 128
        papyrus = cv2.imread(str(REPO / "masks" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE) > 0
        preview = np.zeros((*shape, 3), np.uint8)
        for k, color in enumerate(FOLD_COLORS):
            preview[(pixel_fold == k) & papyrus] = np.array(color) // 3
            preview[(pixel_fold == k) & ink] = color
        preview = orient_display(preview, sid)
        scale = 1600 / max(preview.shape[:2])
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(preview, f"{sid}  held out per fold: {summary[sid]}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(SPLIT_ROOT / f"{sid}_folds.png"), preview)
    print(f"[splits] letters held out per fold (fold 1/2/3): {summary} -> {SPLIT_ROOT}", flush=True)
    return summary


def configure(config, arm: str) -> None:
    """the best run with the arm's split; its vis scrolls (7 trained 0846A patches, winning sheet, out_4, out_5, 0009B)
    are kept as they are."""
    from utils.config import apply_best_run

    apply_best_run(config)
    labeled = labeled_ids()
    config.run_name = config.exp_name = arm
    config.model_dir = os.path.join("models", "runs_kfolds", arm)
    config.tra.log_dir = "runs_kfolds"
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = N_EPOCHS
    if arm.startswith("cv_fold_"):
        config.data.train_mask_dir = str(fold_mask_dir(int(arm.rsplit("_", 1)[1])))
        # out of train-only, so the fold masks' held-out letters become validation
        config.data.train_only_scroll_ids = [
            sid for sid in config.data.train_only_scroll_ids if int(sid) not in labeled
        ]
    elif arm in LOO_ARMS:
        dropped = {sid for sid in labeled if sid not in LOO_ARMS[arm]}
        config.data.scrolls = [scroll for scroll in config.data.scrolls if int(scroll.scroll_id) not in dropped]
        config.data.train_scroll_dict = {
            name: [sid for sid in ids if int(sid) not in dropped]
            for name, ids in config.data.train_scroll_dict.items()
        }
        config.data.train_only_scroll_ids = [
            sid for sid in config.data.train_only_scroll_ids if int(sid) not in dropped
        ]
        config.data.extra_hard_negative_scroll_ids = [
            sid for sid in config.data.extra_hard_negative_scroll_ids if int(sid) not in dropped
        ]
        config.data.whole_patch_valid_scroll_ids = [*config.data.whole_patch_valid_scroll_ids, *sorted(dropped)]
    else:
        raise ValueError(f"unknown arm {arm}")


def run_worker(arm: str, dry_run: bool) -> None:
    import train

    sys.argv = ["train.py", "--run-name", arm, *(["--dry-run"] if dry_run else [])]
    train.main(configure=lambda config: configure(config, arm))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated arms (default: all seven)")
    parser.add_argument("--dry-run", action="store_true", help="build the splits and preflight every arm, no training")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    if args.worker:
        run_worker(args.worker, args.dry_run)
        return

    arms = [name.strip() for name in args.only.split(",") if name.strip()] or arm_names()
    unknown = sorted(set(arms) - set(arm_names()))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {arm_names()}")
    build_splits()
    (LOG_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    failed = []
    for index, arm in enumerate(arms, start=1):
        final = MODEL_ROOT / arm / "final.pth"
        if final.is_file() and not args.dry_run:
            print(f"[k_folds] {index}/{len(arms)} {arm}: done already ({final}), skipping", flush=True)
            continue
        log_path = LOG_ROOT / "logs" / f"{arm}{'_dry' if args.dry_run else ''}.log"
        command = [sys.executable, "-u", __file__, "--worker", arm, *(["--dry-run"] if args.dry_run else [])]
        print(f"[k_folds] {index}/{len(arms)} {arm} -> {log_path}", flush=True)
        started = time.time()
        # each arm in its own process, so RAM caches and CUDA state never carry over between runs
        with open(log_path, "a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = process.wait()
        status = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"[k_folds] {arm}: {status} after {(time.time() - started) / 60:.1f} min", flush=True)
        if code != 0:
            failed.append(arm)
    if failed:
        raise SystemExit(f"[k_folds] failed arms: {failed}")
    print("[k_folds] all arms finished", flush=True)


if __name__ == "__main__":
    main()
