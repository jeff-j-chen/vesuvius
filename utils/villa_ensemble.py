"""villa_ensemble.py -- four-model ink ensemble from Nieuwlaar/pherc343-first-letters, applied to our test surfaces.

Models (sha256-checked against the reference ink/models.tsv, stored in models/villa_ensemble/):
  dnative   huggingface.co/Nieuwlaar/ink9um-dense-native  (step 16000, rebuilt .pth via rebuild_dense_native_pth.py)
  v8in-1447 huggingface.co/YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062  (replaces ink_9um seed 42)
  ink9-43   huggingface.co/scrollprize/ink_9um  hybrid_3d2d-seed43/step-075000.pth
  hecate    huggingface.co/scrollprize/hecate   hecate_9.6um.pth + hecate.py

Inference runs in the isolated villa environment (VILLA_PY) through utils/villa_driver.py. Inputs follow the
reference contract: a uint8 zarr group with array "0" (Z,Y,X), layer k sampled at (k - (N-1)/2) voxels along the
mesh normal in the native voxel grid of the mesh's volume. Our 28-layer renders already have that geometry
(VC3D's standard 28-layer render = layers 19..46 of the reference 66-layer render), but are stored as uint16 bare
arrays, so ensure_villa_input writes a uint8 level-0 group copy. With 28 layers, depth d is limited to [-5, 5].

Ensemble (ink/scripts/ensemble_maps.py): villa maps v -> clip((v - 64) / 128, 0, 1), Hecate v / 255 resized
bilinearly to the render grid; per model mean/sd/p1/p99.5 over valid pixels (CT > 0 in all layers) of the reading
side at all computed depths; ensemble = mean of the four z-scores, stretched p1..p99.5 on an every-8th-pixel grid.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import zarr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VILLA_PY = os.getenv("VILLA_PY", "/opt/villa-venv/bin/python")
DRIVER = os.path.join(REPO, "utils", "villa_driver.py")
CKPT_DIR = os.path.join(REPO, "models", "villa_ensemble")
VILLA_ZARR_DIR = os.getenv("VILLA_ZARR_PATH", os.path.join(REPO, "ves_zarrs_villa"))
MAPS_ROOT = os.path.join(REPO, "output", "villa_ensemble_maps")

# (short name, driver model key); order matches ensemble_maps.py
MODELS = (("dnative", "dnative"), ("hecate", "hecate"),
          ("v8in-1447", "v8in-loo-w062"), ("ink9-43", "hybrid_3d2d-seed43"))
# alternatives for the third slot (MODELS[2])
SLOT3_CHOICES = {"ink9-42": ("ink9-42", "hybrid_3d2d-seed42"), "v8in-1447": ("v8in-1447", "v8in-loo-w062")}


def set_slot3(choice):
    """swap the third ensemble member in place (run_models/ensemble read MODELS at call time)"""
    global MODELS
    MODELS = (MODELS[0], MODELS[1], SLOT3_CHOICES[choice], MODELS[3])
    return MODELS
P_LO, P_HI = 1.0, 99.5
SUB = 8
# Hecate is compute-bound: bf16 is 1.65x faster than fp32 (batch size makes no difference); mean |diff| 0.13/255
HECATE_PRECISION = "bf16"
# v8in tile stride; None = its config (21). 32 is 2.3x faster: r=0.940 vs 0.954 (stride 21) against a stride-8 map
V8IN_STRIDE = None
# native voxel size of volumes that are not in assemble_test_segments.FRAGMENTS (native surface volumes)
EXTRA_NATIVE_UM = {"20260226000000": 9.362}


def native_um(scroll_id):
    """voxel size of the CT volume the segment was rendered from (= the render's pixel size)."""
    sid = str(scroll_id)
    if sid in EXTRA_NATIVE_UM:
        return EXTRA_NATIVE_UM[sid]
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from assemble_test_segments import FRAGMENTS, extra_fragments

    for zid, _mesh, vol_base, _shape, *_ in FRAGMENTS + extra_fragments():
        if zid == sid:
            return float(re.search(r"-(\d+\.\d+)um-", vol_base).group(1))
    raise KeyError(f"no native voxel size known for {sid}")


def villa_input_path(scroll_id):
    return os.path.join(VILLA_ZARR_DIR, f"{scroll_id}.zarr")


def _uint8_window(source, height):
    """None if the uint16 source already holds 8-bit values, else the (lo, hi) p0.5..p99.9 window of its nonzero voxels"""
    peak, samples = 0, []
    for y0 in range(0, height, 128 * 8):
        block = np.asarray(source[:, y0:y0 + 128, ::4])
        peak = max(peak, int(block.max(initial=0)))
        samples.append(block[block > 0][::16])
    if peak <= 255:
        return None
    lo, hi = np.percentile(np.concatenate(samples), [0.5, 99.9])
    return float(lo), float(hi)


def ensure_villa_input(scroll_id, zarr_dir):
    """<VILLA_ZARR_DIR>/<id>.zarr/0 in the models' format: a symlink for uint8 renders, else a uint8 copy; idempotent.
    full-range uint16 volumes (e.g. the 88 keV rescans) are windowed p0.5..p99.9 -> 1..255, keeping 0 as no data."""
    from tqdm import tqdm

    out = villa_input_path(scroll_id)
    if os.path.isfile(os.path.join(out, "0", ".zarray")) and zarr.open_group(out, mode="r").attrs.get("complete"):
        return out
    source_path = os.path.abspath(os.path.join(zarr_dir, f"{scroll_id}.zarr"))
    source = zarr.open(source_path, mode="r")
    depth, height, width = map(int, source.shape)
    if source.dtype == np.uint8:
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        with open(os.path.join(out, ".zgroup"), "w") as handle:
            handle.write('{"zarr_format": 2}')
        os.symlink(source_path, os.path.join(out, "0"))
        zarr.open_group(out, mode="r+").attrs.update({"linked": True, "complete": True})
        return out
    group = zarr.open_group(out, mode="w")
    window = _uint8_window(source, height)
    array = group.create_dataset("0", shape=(depth, height, width), chunks=(depth, 128, 128), dtype="u1",
                                 fill_value=0, dimension_separator="/", overwrite=True)
    for y0 in tqdm(range(0, height, 128), desc=f"uint8 copy {scroll_id}", unit="row", leave=False):
        block = np.asarray(source[:, y0:y0 + 128, :])
        if window is not None:
            lo, hi = window
            scaled = np.clip(np.rint(1 + (block.astype(np.float32) - lo) * (254.0 / max(hi - lo, 1.0))), 1, 255)
            array[:, y0:y0 + 128, :] = np.where(block > 0, scaled, 0).astype(np.uint8)
            continue
        if block.max(initial=0) > 255:
            raise ValueError(f"{scroll_id}: values above 255 cannot be stored as uint8")
        array[:, y0:y0 + 128, :] = block.astype(np.uint8)
    group.attrs.update({"depth_flipped": bool(source.attrs.get("depth_flipped", False)), "complete": True,
                        "uint16_window": window})
    return out


def villa_map_name(name, model, n, d, direction):
    return f"{name}__{model}__L{n}s{n // 2 + d - 8}{'_reverse' if direction == 'reverse' else ''}.tif"


def hecate_map_name(name, n, d, direction):
    return f"{name}__hecate__L{n}zc{(n - 1) / 2 + d:.2f}{'_reverse' if direction == 'reverse' else ''}.png"


def map_path(maps_dir, name, n, model_key, d, direction):
    if model_key == "hecate":
        return os.path.join(maps_dir, hecate_map_name(name, n, d, direction))
    if model_key == "v8in-loo-w062":
        suffix = "_reverse" if direction == "reverse" else ""
        tag = f"st{V8IN_STRIDE}" if V8IN_STRIDE else ""
        return os.path.join(maps_dir, f"{name}__v8in-loo-w062__L{n}s{(n - 24) // 2 + d}{tag}{suffix}.tif")
    return os.path.join(maps_dir, villa_map_name(name, model_key, n, d, direction))


def model_depths(model_key, n, depths, um):
    """the depths whose input window fits inside the n-layer render for this model"""
    if model_key == "v8in-loo-w062":
        return [d for d in depths if 0 <= (n - 24) // 2 + d and (n - 24) // 2 + d + 24 <= n]
    if model_key == "hecate":
        half = 7.5 * 9.6 / um
        return [d for d in depths if (n - 1) / 2 + d - half >= 0 and (n - 1) / 2 + d + half <= n - 1]
    return [d for d in depths if 0 <= n // 2 + d - 8 and n // 2 + d + 9 <= n]


def run_models(scroll_id, depths, directions, compile_model=True, villa_batch=4, hecate_batch=4):
    """run every missing (model, depth, direction) map in the villa env, with one fractional progress bar."""
    from tqdm import tqdm

    surface = villa_input_path(scroll_id)
    maps_dir = os.path.join(MAPS_ROOT, str(scroll_id))
    n = int(zarr.open_group(surface, mode="r")["0"].shape[0])
    um = native_um(scroll_id)
    missing = [key for _, key in MODELS for d in model_depths(key, n, depths, um) for r in directions
               if not os.path.isfile(map_path(maps_dir, str(scroll_id), n, key, d, r))]
    if not missing:
        print(f"[villa] {scroll_id}: all maps cached")
        return maps_dir
    # -P: don't put utils/ on sys.path (utils/platform.py would shadow the stdlib module)
    command = [VILLA_PY, "-P", DRIVER, "--surface", surface, "--out", maps_dir, "--ckpt-dir", CKPT_DIR,
               "--native-um", str(native_um(scroll_id)), "--depths", *map(str, depths),
               "--directions", *directions, "--villa-batch", str(villa_batch), "--hecate-batch", str(hecate_batch),
               "--hecate-precision", HECATE_PRECISION, "--models", *dict.fromkeys(missing)]
    if V8IN_STRIDE:
        command += ["--v8in-stride", str(V8IN_STRIDE)]
    if not compile_model:
        command.append("--no-compile")
    log_path = os.path.join(maps_dir, "driver.log")
    os.makedirs(maps_dir, exist_ok=True)
    bar = tqdm(total=len(missing), desc=str(scroll_id), unit="map", dynamic_ncols=True,
               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total} maps [{elapsed}<{remaining}] {postfix}")
    finished = 0
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=REPO) as proc:
        for line in proc.stdout:
            log.write(line)
            parts = line.split()
            if parts[:1] == ["PROGRESS"] and len(parts) == 4:
                done, total = int(parts[2]), max(1, int(parts[3]))
                bar.n = finished + done / total
                bar.set_postfix_str(f"{parts[1].split('__', 1)[-1]} {done}/{total} patches", refresh=False)
                bar.refresh()
            elif parts[:1] == ["TASK"]:
                finished += 1
                bar.n = finished
                bar.refresh()
        proc.wait()
    bar.close()
    if proc.returncode != 0:
        raise RuntimeError(f"villa driver failed (exit {proc.returncode}); see {log_path}")
    print(f"[villa] {scroll_id}: {finished} maps in {time.time() - t0:.0f}s -> {maps_dir}")
    return maps_dir


def read_map(path, shape, zero_is_missing=False):
    """ink map as float32 0..1 on the render grid (ensemble_maps.read_map); NaN where the model made no prediction."""
    import tifffile

    if path.lower().endswith(".png"):
        array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if array.ndim == 3:
            array = array[..., 0]
        array = array.astype(np.float32) / (65535.0 if array.dtype == np.uint16 else 255.0)
        if array.shape != tuple(shape):
            array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        array = tifffile.imread(path)
        if array.dtype == np.uint8:
            array = np.clip((array.astype(np.float32) - 64.0) / 128.0, 0, 1)
        elif array.dtype == np.uint16:
            array = array.astype(np.float32) / 65535.0
        else:
            array = array.astype(np.float32)
        if array.shape != tuple(shape):
            raise ValueError(f"{path}: shape {array.shape} != surface {tuple(shape)}")
    if zero_is_missing:
        array = np.where(array == 0, np.nan, array)
    return np.ascontiguousarray(array, dtype=np.float32)


def _pct_hist(hist, values, qs):
    """np.percentile (linear) from a histogram over discrete values (ensemble_maps.pct_hist)."""
    cumulative = np.cumsum(hist)
    count = int(cumulative[-1])
    out = []
    for q in qs:
        pos = q / 100.0 * (count - 1)
        lo = int(math.floor(pos))
        frac = pos - lo
        i_lo = int(np.searchsorted(cumulative, lo + 1))
        i_hi = int(np.searchsorted(cumulative, min(lo + 2, count))) if frac > 0 else i_lo
        out.append(float(values[i_lo] + frac * (values[i_hi] - values[i_lo])))
    return out


def valid_mask(scroll_id):
    """CT > 0 in every layer of the surface volume that holds data (fine-scan resamples leave outer layers empty)."""
    array = zarr.open_group(villa_input_path(scroll_id), mode="r")["0"]
    depth, height, width = map(int, array.shape)
    filled = np.zeros(depth, bool)
    for y0 in range(0, height, 128 * 8):
        filled |= (np.asarray(array[:, y0:y0 + 128, ::8]) > 0).any(axis=(1, 2))
    valid = np.zeros((height, width), bool)
    for y0 in range(0, height, 128):
        valid[y0:y0 + 128] = (np.asarray(array[:, y0:y0 + 128, :])[filled] > 0).all(0)
    return valid, depth


def ensemble(scroll_id, depths, direction, write_depth):
    """per-model stretched maps and the z-score ensemble at write_depth, all as float 0..1 (NaN = invalid).
    each model's stats pool over the depths it can reach; a model without a map at write_depth is left out."""
    maps_dir = os.path.join(MAPS_ROOT, str(scroll_id))
    name = str(scroll_id)
    valid, n = valid_mask(scroll_id)
    um = native_um(scroll_id)
    shape = valid.shape
    sub = np.zeros_like(valid)
    sub[SUB // 2::SUB, SUB // 2::SUB] = True
    sub &= valid
    stats, subs = {}, {}
    for short, key in MODELS:
        # v8in only predicts tiles fully inside its coverage mask and writes 0 elsewhere
        missing = key == "v8in-loo-w062"
        # villa maps are uint8 with 129 discrete levels after decoding; hecate/v8in are near-continuous
        villa = key not in ("hecate", "v8in-loo-w062")
        hist = np.zeros(129 if villa else 65536, np.int64)
        s1 = s2 = 0.0
        count = 0
        reach = model_depths(key, n, depths, um)
        samples = np.full((len(depths), int(sub.sum())), np.nan, np.float32)
        for i, d in enumerate(depths):
            if d not in reach:
                continue
            values = read_map(map_path(maps_dir, name, n, key, d, direction), shape, missing)
            v = values[valid]
            v = v[np.isfinite(v)]
            if villa:
                hist += np.bincount(np.rint(v * 128).astype(np.int64), minlength=129)[:129]
            else:
                hist += np.bincount(np.minimum((v * 65536).astype(np.int64), 65535), minlength=65536)
            s1 += float(v.sum(dtype=np.float64))
            s2 += float(np.square(v, dtype=np.float64).sum())
            count += v.size
            samples[i] = values[sub]
        grid = (np.arange(129) / 128.0) if villa else (np.arange(65536) / 65536.0)
        lo, hi = _pct_hist(hist, grid, [P_LO, P_HI])
        mean = s1 / count
        sd = math.sqrt(max(s2 / count - mean * mean, 0.0)) + 1e-6
        stats[short] = dict(p1=lo, p99_5=hi, mean=mean, sd=sd)
        subs[short] = samples
    # mean over the models that predicted each pixel
    z_grid = np.nanmean(np.stack([(subs[short] - stats[short]["mean"]) / stats[short]["sd"] for short, _ in MODELS]), 0)
    ens_lo, ens_hi = (float(x) for x in np.nanpercentile(z_grid.ravel(), [P_LO, P_HI]))
    stretched = {}
    z_sum, z_count = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    for short, key in MODELS:
        if write_depth not in model_depths(key, n, [write_depth], um):
            stretched[short] = np.full(shape, np.nan, np.float32)
            continue
        values = read_map(map_path(maps_dir, name, n, key, write_depth, direction), shape, key == "v8in-loo-w062")
        s = stats[short]
        stretched[short] = np.where(valid, np.clip((values - s["p1"]) / max(s["p99_5"] - s["p1"], 1e-9), 0, 1), np.nan)
        z = (values - s["mean"]) / s["sd"]
        z_sum += np.nan_to_num(z)
        z_count += np.isfinite(z)
    ens = np.where(valid & (z_count > 0),
                   np.clip((z_sum / np.maximum(z_count, 1) - ens_lo) / max(ens_hi - ens_lo, 1e-9), 0, 1), np.nan)
    stats["ensemble"] = dict(p1=ens_lo, p99_5=ens_hi)
    return stretched, ens.astype(np.float32), stats


def composite(scroll_id, d0=10, d1=18):
    """raw max projection of layers [d0, d1) as 0..1 (NaN = invalid); same as test_inference.ipynb's composite."""
    array = zarr.open_group(villa_input_path(scroll_id), mode="r")["0"]
    valid, _ = valid_mask(scroll_id)
    image = np.zeros(valid.shape, np.uint8)
    for depth in range(d0, min(d1, int(array.shape[0]))):
        np.maximum(image, np.asarray(array[depth]), out=image)
    return np.where(valid, image / np.float32(255), np.nan).astype(np.float32)


def save_ensemble_figure(scroll_id, ensemble, title, out_path):
    """ensemble over the fiber composite (with 1 cm bar), both at full resolution"""
    panels = [(title, ensemble), ("composite view", composite(scroll_id), native_um(scroll_id))]
    save_grid(panels, 1, out_path)


def save_grid(panels, n_cols, out_path, gap_px=24):
    """pixel-exact grid of (title, image 0..1 with NaN invalid[, um_per_px for a 1 cm bar]) panels at
    full render resolution; not shown."""
    from utils.fast_figure import colorize, save_panel_grid

    height, width = panels[0][1].shape
    cells = [(title, colorize(image, "gray", (0.0, 0.0, 0.0)), cv2.INTER_NEAREST, bar[0] if bar else None)
             for title, image, *bar in panels]
    rows = [cells[i:i + n_cols] for i in range(0, len(cells), n_cols)]
    save_panel_grid(rows, (height, width), out_path, gap_px=gap_px)
