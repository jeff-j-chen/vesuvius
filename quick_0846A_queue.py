#!/usr/bin/env python3
"""assemble the unassembled PHerc0846A quick patches in parallel and run the fine-tuned reader-v3 on each one as soon as its
render finishes (one GPU consumer thread), plus the already-assembled untrained sheets and holdouts.

figure per sheet: inverted, percentile-stretched prediction | stretched composite of the 17 layers reader-v3 reads.
usage: python quick_0846A_queue.py [--parallel 6] [--workers 32]
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import zarr

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import assemble_test_segments as ats  # noqa: E402
from utils import predict_models  # noqa: E402

CHECKPOINT = REPO / "reader-v3/runs/mix6/ckpt_002000.pth"
QUICK_ROOT = REPO / "tifxyz/quick_patches_from_D/PHerc0846A"
VOL_BASE, VOL_SHAPE = ats.EXTRA_VOLUMES["PHerc0846A"]
TRAINED_QUICK = {"auto_grown_20260927134928530_abf"}  # mix6 sheet (render 20260927134928)
# previous holdouts rendered under masks/: straightened2 prize sheet, out_4, out_5
HOLDOUT_RENDERS = {"20261012065814": "straightened2_abf", "20261012065824": "straightened2_abf_out_4",
                   "20261012065826": "straightened2_abf_out_5"}
OUT_DIR = REPO / "output/0846A" / f"reader_v3_{CHECKPOINT.parent.name}_{CHECKPOINT.stem}" / "quick_review"
CACHE = REPO / "_ves_tmp/0846A_inference" / f"reader_v3_{CHECKPOINT.parent.name}_{CHECKPOINT.stem}"


def stretch(values, valid, low=1.0, high=99.0):
    lo, hi = np.percentile(values[valid], (low, high)) if valid.any() else (0.0, 1.0)
    out = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.where(valid, np.rint(out * 255), 0).astype(np.uint8)


def infer(zid, mask_dir, label):
    zarr_path = REPO / "ves_zarrs2" / f"{zid}.zarr"
    mask = cv2.imread(str(Path(mask_dir) / f"{zid}.png"), cv2.IMREAD_GRAYSCALE) > 0
    prediction = predict_models.predict_reader(zarr_path, CHECKPOINT, CACHE / label / "reader_v3.tif", 4)
    framed = np.zeros(mask.shape, np.float32)
    rows, cols = min(mask.shape[0], prediction.shape[0]), min(mask.shape[1], prediction.shape[1])
    framed[:rows, :cols] = prediction[:rows, :cols]
    source = zarr.open(str(zarr_path), mode="r")
    start, end = predict_models.native_window("reader_v3", source.shape[0], CHECKPOINT)
    composite = np.asarray(source[start:end]).max(axis=0).astype(np.float32)
    left = stretch(1.0 - framed, mask)
    left[~mask] = 255
    right = stretch(composite, mask)
    # renders assembled before ROTATE_CCW existed still need the turn in the figure
    turns = (ats.ROTATE_CCW.get(zid, 0) - ats.applied_rotation(source.attrs)) % 4
    if turns:
        print(f"[orient] {label}: rotating figure {turns * 90} deg counter-clockwise", flush=True)
    left, right = np.rot90(left, turns), np.rot90(right, turns)
    gap = np.full((left.shape[0], 24), 255, np.uint8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{label}.png"
    cv2.imwrite(str(path), np.hstack([left, gap, right]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return path


def finished(label):
    return (OUT_DIR / f"{label}.png").is_file()


def consumer(jobs, failures):
    while True:
        job = jobs.get()
        if job is None:
            return
        zid, mask_dir, label = job
        try:
            print(f"[infer] {label} -> {infer(zid, mask_dir, label)}", flush=True)
        except Exception as error:
            traceback.print_exc()
            failures[label] = f"infer: {error}"


def render(name, workers):
    return ats.render_fragment(name, f"quick_patches_from_D/PHerc0846A/{name}", VOL_BASE, VOL_SHAPE, workers,
                               ats.ZARR_DIR, ats.SCRIPT_DIR, ats.DEFAULT_CHUNK_DEPTH, ats.DEFAULT_CHUNK_Y,
                               ats.DEFAULT_CHUNK_X, mask_dir=ats.EXTRAS_MASK_DIR, output_dtype=ats.EXTRAS_DTYPE,
                               drop_cache=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    ats._MIN_FREE_BYTES = int(10e9)

    names = [p.name for p in sorted(QUICK_ROOT.iterdir()) if (p / "x.tif").is_file() and p.name not in TRAINED_QUICK]
    done = [n for n in names if (Path(ats.ZARR_DIR) / f"{n}.zarr").is_dir() and (Path(ats.EXTRAS_MASK_DIR) / f"{n}.png").is_file()]
    todo = [n for n in names if n not in done]
    print(f"[queue] {len(done)} quick patches assembled, {len(todo)} to assemble", flush=True)

    jobs, failures = queue.Queue(), {}
    worker = threading.Thread(target=consumer, args=(jobs, failures), daemon=True)
    worker.start()
    for zid, label in HOLDOUT_RENDERS.items():
        if not finished(label):
            jobs.put((zid, ats.MASK_DIR, label))
    for name in done:
        if not finished(name):
            jobs.put((name, ats.EXTRAS_MASK_DIR, name))

    with ProcessPoolExecutor(args.parallel, mp_context=get_context("spawn"),
                             initializer=ats._set_min_free, initargs=(ats._MIN_FREE_BYTES,)) as pool:
        futures = {pool.submit(render, name, args.workers): name for name in todo}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _zid, status = future.result()
            except Exception as error:
                status = f"FAIL: {error}"
            print(f"[assemble] {name}: {status}", flush=True)
            if status.startswith("OK"):
                jobs.put((name, ats.EXTRAS_MASK_DIR, name))
            else:
                failures[name] = status
    jobs.put(None)
    worker.join()
    print(f"[queue] figures in {OUT_DIR}; failures: {failures or 'none'}", flush=True)


if __name__ == "__main__":
    main()
