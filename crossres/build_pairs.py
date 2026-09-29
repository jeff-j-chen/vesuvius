"""step 3: cut paired training tiles: native training-frame input + co-registered high-resolution target.

inputs
  - the local training zarr (ves_zarrs2/<zid>.zarr): the exact input domain fine-tuning sees, so the
    9.36 um side is never downloaded again (0009B's resampled frame and 0841's crop come for free)
  - crossres/sanity/registration.json from sanity_midslice.py (low-native px -> high level-0 px, depth offset)
  - the ~2.4 um render streamed from S3 at --high-level, only for the chunks under the selected tiles
    (surface-volume chunks are [all layers, 128, 128], so a level-L chunk covers 128 * 2^L high px)

target geometry (the two plans in PLAN.md)
  - depth super-resolution (plan 1): --high-level 2 --xy-scale 1 --depth-factor 4. the 2.4 um render
    downscaled 4x in x/y (~9.6 um, resampled onto the training grid) keeps every one of its ~109 layers,
    so each training slice gets 4 sub-slices: the 9.36 um column "unsquished"
  - 3D super-resolution (plan 2): --high-level 1 --xy-scale 2 --depth-factor 4, 4x the download

tile selection
  - training scrolls (and 0009B, paired but still held out from fine-tuning): every tile inside the ink
    labels' bounding box grown by --margin that is mostly footprint, so papyrus between and around the
    text is sampled as densely as ink (--region near keeps only tiles touching ink, to save disk)
  - 0841, the strict holdout: off unless --include-holdouts; random footprint tiles, never its labels

per tile: depth offset refined against the input's layer profile (sub-slice steps), residual x/y shift
re-estimated against the input midslice, tiles whose midslice NCC stays below --min-ncc dropped.

output: crossres/pairs/<plan>/<zid>/pairs.zarr with input (N, Z, T, T) uint16,
target (N, Z*F, S*T, S*T) uint8, valid (N, S*T, S*T) bool, origin (N, 2), plus meta.json.

    python crossres/build_pairs.py --count-only                       # tiles, download and disk per pair
    python crossres/build_pairs.py --zarr-dir ves_zarrs2 --plan depth # plan 1
    python crossres/build_pairs.py --zarr-dir ves_zarrs2 --plan xyz   # plan 2
    python crossres/build_pairs.py --zarr-dir ves_zarrs2 --plan degrade --names w044 ... seg46527  # plan R
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pairs import PAIRS, TRAIN_FRAME_UM, high_to_frame, url  # noqa: E402

PLANS = {
    "depth": {"high_level": 2, "xy_scale": 1, "depth_factor": 4},
    "xyz": {"high_level": 1, "xy_scale": 2, "depth_factor": 4},
    # plan R: the degrader is applied to all 28 slices, so it trains on all of them (same chunks as depth)
    "degrade": {"high_level": 2, "xy_scale": 1, "depth_factor": 4, "z0": 0, "z1": 28},
}
HIGH_LAYERS_GUESS = 109  # 28 x 9.362 / 2.4; the real count is read from the zarr when building


def _select_tiles(pair, frame_shape, args, rng):
    height, width = frame_shape
    tile = args.tile
    if pair["role"] == "train":
        label = cv2.imread(str(Path(args.repo) / "dilated_inklabels" / f"{pair['zid']}.png"), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"dilated_inklabels/{pair['zid']}.png")
        label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST) > 0
        mask = cv2.imread(str(Path(args.repo) / "masks" / f"{pair['zid']}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
        ys, xs = np.nonzero(label)
        if args.region == "bbox":
            y_lo, y_hi = max(0, ys.min() - args.margin), min(height, ys.max() + args.margin)
            x_lo, x_hi = max(0, xs.min() - args.margin), min(width, xs.max() + args.margin)
            area = np.zeros_like(label)
            area[y_lo:y_hi, x_lo:x_hi] = True
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.margin + 1, 2 * args.margin + 1))
            area = cv2.dilate(label.astype(np.uint8), kernel) > 0
        tiles = [(y, x) for y in range(0, height - tile + 1, tile) for x in range(0, width - tile + 1, tile)
                 if area[y:y + tile, x:x + tile].mean() > 0.5 and mask[y:y + tile, x:x + tile].mean() > 0.5]
        print(f"[{pair['name']}] ink bbox y[{ys.min()},{ys.max()}] x[{xs.min()},{xs.max()}] "
              f"({(ys.max() - ys.min()) * (xs.max() - xs.min()) / (height * width):.0%} of the frame) "
              f"-> {len(tiles)} tiles ({args.region}, margin {args.margin}px)", flush=True)
        return tiles
    mask = cv2.imread(str(Path(args.repo) / "masks" / f"{pair['zid']}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    tiles = [(y, x) for y in range(0, height - tile + 1, tile) for x in range(0, width - tile + 1, tile)
             if mask[y:y + tile, x:x + tile].mean() > 0.9]
    rng.shuffle(tiles)
    print(f"[{pair['name']}] held out: {min(len(tiles), args.holdout_tiles)} random footprint tiles "
          f"(labels not used)", flush=True)
    return tiles[:args.holdout_tiles]


def _depth_weights(frame_slices, frame_depth, factor, high_depth, frame_um, high_um, offset_um):
    """(Z*F, Dh) box weights: each training slice split into F sub-slices, each averaging the high layers
    it overlaps. both renders centre on the fitted surface; offset_um shifts the high render."""
    width = frame_um / factor
    layer_centres = (np.arange(high_depth) - (high_depth - 1) / 2) * high_um
    weights = np.zeros((len(frame_slices) * factor, high_depth), np.float32)
    for row, k in enumerate(frame_slices):
        for sub in range(factor):
            centre = (k - (frame_depth - 1) / 2 + (sub + 0.5) / factor - 0.5) * frame_um + offset_um
            lo, hi = centre - width / 2, centre + width / 2
            overlap = np.clip(np.minimum(layer_centres + high_um / 2, hi)
                              - np.maximum(layer_centres - high_um / 2, lo), 0, None)
            if overlap.sum() > 0:
                weights[row * factor + sub] = overlap / overlap.sum()
    return weights


def _map_coords(pair, low_to_high0, level, y0, x0, size, scale, shift=(0.0, 0.0)):
    """high level-`level` coordinates of the scale-x output grid over one training-frame tile."""
    grid = (np.arange(scale * size, dtype=np.float64) + 0.5) / scale - 0.5
    xt, yt = np.meshgrid(x0 + grid + shift[0], y0 + grid + shift[1])
    frame_to_native = 1.0 if pair["frame"] == "identity" else TRAIN_FRAME_UM / pair["low_um"]
    xn, yn = (xt + 0.5) * frame_to_native - 0.5, (yt + 0.5) * frame_to_native - 0.5
    a = np.asarray(low_to_high0, np.float64)
    u0 = a[0, 0] * xn + a[0, 1] * yn + a[0, 2]
    v0 = a[1, 0] * xn + a[1, 1] * yn + a[1, 2]
    return (u0 + 0.5) / 2 ** level - 0.5, (v0 + 0.5) / 2 ** level - 0.5


def _load_block(high, u, v, pad=4, reverse_depth=False):
    ux0, ux1 = max(0, int(np.floor(u.min())) - pad), min(high.shape[2], int(np.ceil(u.max())) + pad)
    vy0, vy1 = max(0, int(np.floor(v.min())) - pad), min(high.shape[1], int(np.ceil(v.max())) + pad)
    if ux1 <= ux0 or vy1 <= vy0:
        return None
    for attempt in range(6):
        try:
            block = np.asarray(high[:, vy0:vy1, ux0:ux1], dtype=np.float32)
            break
        except Exception as error:  # S3 returns sporadic 500s
            if attempt == 5:
                raise
            print(f"  read failed ({error}); retry {attempt + 1}", flush=True)
            time.sleep(2 ** attempt)
    if reverse_depth:
        block = np.ascontiguousarray(block[::-1])
    return block, (u - ux0).astype(np.float32), (v - vy0).astype(np.float32)


def _render(block, map_x, map_y, weights):
    pooled = np.tensordot(weights, block, axes=(1, 0))
    target = np.stack([cv2.remap(plane, map_x, map_y, cv2.INTER_LINEAR, borderValue=0) for plane in pooled])
    present = (block > 0).any(axis=0).astype(np.float32)
    valid = cv2.remap(present, map_x, map_y, cv2.INTER_NEAREST, borderValue=0) > 0.5
    return target, valid


def _ncc(a, b, valid):
    if valid.sum() < 500:
        return -1.0
    x, y = a[valid].astype(np.float64), b[valid].astype(np.float64)
    return float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9))


def _coarse(target, factor, scale, size):
    """target pooled back onto the input's slices and pixels, for comparison with the input."""
    slices = target.reshape(-1, factor, *target.shape[1:]).mean(axis=1)
    if scale == 1:
        return slices
    return np.stack([cv2.resize(plane, (size, size), interpolation=cv2.INTER_AREA) for plane in slices])


def _fetch(address, out_path):
    """one chunk to disk; 404 is an all-fill chunk and leaves no file."""
    if out_path.exists():
        return "cached"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(address, timeout=120) as response:
                data = response.read()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # unique per writer: parallel builds share this cache
            tmp = out_path.with_name(f"{out_path.name}.{os.getpid()}.{threading.get_ident()}.part")
            tmp.write_bytes(data)
            tmp.replace(out_path)
            return "ok"
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return "fill"
            if attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 5:
                raise
        time.sleep(2 ** attempt)


def _prefetch_high(pair, registration, tiles, args):
    """download, in parallel, every high chunk any tile (plus its residual-shift margin) can read."""
    base = url(pair, "high", args.high_level)
    local = Path(args.cache_dir) / pair["name"] / str(args.high_level)
    local.mkdir(parents=True, exist_ok=True)
    meta_path = local / ".zarray"
    if not meta_path.exists():
        with urllib.request.urlopen(f"{base}/.zarray", timeout=60) as response:
            meta_path.write_bytes(response.read())
    meta = json.loads(meta_path.read_text())
    sep = meta.get("dimension_separator", ".")
    _, height, width = meta["shape"]
    _, cy_size, cx_size = meta["chunks"]
    pad = int(args.max_residual) + 8
    keys = set()
    for y0, x0 in tiles:
        u, v = _map_coords(pair, registration["low_to_high0"], args.high_level, y0 - pad, x0 - pad,
                           int(args.tile + 2 * pad), 1)
        cy0, cy1 = max(0, int(v.min()) // cy_size), min((height - 1) // cy_size, int(np.ceil(v.max())) // cy_size)
        cx0, cx1 = max(0, int(u.min()) // cx_size), min((width - 1) // cx_size, int(np.ceil(u.max())) // cx_size)
        keys.update((cy, cx) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1))
    jobs = [(f"{base}/0{sep}{cy}{sep}{cx}", local / "0" / str(cy) / str(cx) if sep == "/" else local / f"0.{cy}.{cx}")
            for cy, cx in sorted(keys)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done, _ in enumerate(pool.map(lambda job: _fetch(*job), jobs), 1):
            if done % 200 == 0 or done == len(jobs):
                print(f"[{pair['name']}] prefetched {done}/{len(jobs)} chunks ({time.time() - started:.0f}s)",
                      flush=True)
    return zarr.open_array(str(local), mode="r")


def _chunk_budget(pair, tiles, args):
    """unique high chunks under the tiles (registration ignored: a +-1 chunk estimate)."""
    side = 128 * 2 ** args.high_level / high_to_frame(pair)
    chunks = set()
    for y0, x0 in tiles:
        for cy in range(int(y0 // side), int((y0 + args.tile) // side) + 1):
            for cx in range(int(x0 // side), int((x0 + args.tile) // side) + 1):
                chunks.add((cy, cx))
    return len(chunks) * HIGH_LAYERS_GUESS * 128 * 128


def build_pair(pair, registration, args, rng):
    factor, scale, size = args.depth_factor, args.xy_scale, args.tile
    depth = args.z1 - args.z0
    if args.count_only:
        source = "dilated_inklabels" if pair["role"] == "train" else "masks"
        shape = cv2.imread(str(Path(args.repo) / source / f"{pair['zid']}.png"), cv2.IMREAD_GRAYSCALE).shape
        tiles = _select_tiles(pair, shape, args, rng)
        per_tile = depth * size ** 2 * 2 + depth * factor * (scale * size) ** 2 + (scale * size) ** 2
        download = _chunk_budget(pair, tiles, args)
        print(f"[{pair['name']}] {len(tiles)} tiles: download ~{download / 1e9:.1f} GB at level "
              f"{args.high_level}, disk ~{len(tiles) * per_tile / 1e9:.1f} GB uncompressed", flush=True)
        return download, len(tiles) * per_tile

    frame = zarr.open(str(Path(args.zarr_dir) / f"{pair['zid']}.zarr"), mode="r")
    tiles = _select_tiles(pair, tuple(frame.shape[1:]), args, rng)
    high = _prefetch_high(pair, registration, tiles, args)
    reverse = bool(registration.get("depth_reversed", False))
    slices = list(range(args.z0, args.z1))
    frame_depth = int(frame.shape[0])
    mid = slices.index(frame_depth // 2) if frame_depth // 2 in slices else len(slices) // 2
    frame_um = pair["low_um"] if pair["frame"] == "identity" else TRAIN_FRAME_UM
    base_offset = float(registration.get("depth_offset_low_slices") or 0.0) * pair["low_um"]
    offsets = base_offset + np.arange(-args.depth_search, args.depth_search + 1e-6, 0.125) * frame_um
    weights_by_offset = [_depth_weights(slices, frame_depth, factor, int(high.shape[0]), frame_um,
                                        pair["high_um"], offset) for offset in offsets]
    out_dir = ROOT / "pairs" / args.plan / str(pair["zid"])
    out_dir.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(out_dir / "pairs.zarr"), mode="w")
    big = scale * size
    inputs = group.create_dataset("input", shape=(0, depth, size, size), chunks=(1, depth, 128, 128), dtype="u2")
    targets = group.create_dataset("target", shape=(0, depth * factor, big, big),
                                   chunks=(1, depth * factor, 128 * scale, 128 * scale), dtype="u1")
    valids = group.create_dataset("valid", shape=(0, big, big), chunks=(1, 128 * scale, 128 * scale), dtype=bool)
    origins, dropped, nccs, depth_shifts = [], [], [], []
    for index, (y0, x0) in enumerate(tiles):
        source = np.asarray(frame[args.z0:args.z1, y0:y0 + size, x0:x0 + size]).astype(np.float32)
        if not source.any():
            continue
        u, v = _map_coords(pair, registration["low_to_high0"], args.high_level, y0, x0, size, scale)
        loaded = _load_block(high, u, v, reverse_depth=reverse)
        if loaded is None:
            dropped.append([y0, x0, "outside high render"])
            continue
        block, map_x, map_y = loaded
        # depth first: the profile through the sheet must line up before x/y matters
        profile = np.array([layer[layer > 0].mean() if (layer > 0).any() else 0.0 for layer in source])
        scores = []
        for weights in weights_by_offset:
            target, valid = _render(block, map_x, map_y, weights)
            keep = valid[::scale, ::scale].ravel()
            candidate = _coarse(target, factor, scale, size).reshape(depth, -1)[:, keep].mean(axis=1) \
                if keep.any() else np.zeros(depth)
            scores.append(np.corrcoef(profile, candidate)[0, 1] if candidate.std() > 0 else -1.0)
        pick = int(np.nanargmax(scores))
        target, valid = _render(block, map_x, map_y, weights_by_offset[pick])
        coarse = _coarse(target, factor, scale, size)
        best = (_ncc(source[mid], coarse[mid], valid[::scale, ::scale] & (source[mid] > 0)), target, valid)
        # then a residual x/y shift from the midslices; both signs, phaseCorrelate's is ambiguous
        (dx, dy), _ = cv2.phaseCorrelate(coarse[mid].astype(np.float32), source[mid])
        if 0.25 < np.hypot(dx, dy) < args.max_residual:
            for sign in (1.0, -1.0):
                u2, v2 = _map_coords(pair, registration["low_to_high0"], args.high_level, y0, x0, size,
                                     scale, (sign * dx, sign * dy))
                loaded2 = _load_block(high, u2, v2, reverse_depth=reverse)
                if loaded2 is None:
                    continue
                target2, valid2 = _render(*loaded2, weights_by_offset[pick])
                coarse2 = _coarse(target2, factor, scale, size)
                score2 = _ncc(source[mid], coarse2[mid], valid2[::scale, ::scale] & (source[mid] > 0))
                if score2 > best[0]:
                    best = (score2, target2, valid2)
        score, target, valid = best
        if valid.mean() < 0.5:
            dropped.append([y0, x0, "outside high render"])
            continue
        if score < args.min_ncc:
            dropped.append([y0, x0, f"ncc {score:.2f}"])
            continue
        n = inputs.shape[0]
        inputs.resize((n + 1, depth, size, size))
        targets.resize((n + 1, depth * factor, big, big))
        valids.resize((n + 1, big, big))
        inputs[n] = source.astype(np.uint16)
        targets[n] = np.clip(np.rint(target), 0, 255).astype(np.uint8)
        valids[n] = valid
        origins.append([y0, x0])
        nccs.append(score)
        depth_shifts.append(float((offsets[pick] - base_offset) / frame_um))
        if (index + 1) % 20 == 0:
            print(f"[{pair['name']}] {index + 1}/{len(tiles)} tiles, kept {len(origins)}", flush=True)
    group.create_dataset("origin", data=np.asarray(origins, np.int32).reshape(-1, 2))
    meta = {
        "pair": pair, "registration": registration, "plan": args.plan, "tile": size,
        "z_range": [args.z0, args.z1], "high_level": args.high_level, "xy_scale": scale,
        "depth_factor": factor, "kept": len(origins), "dropped": dropped,
        "ncc_median": float(np.median(nccs)) if nccs else None,
        "tile_depth_shift_slices": depth_shifts, "tile_ncc": [round(float(v), 4) for v in nccs],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"[{pair['name']}] kept {len(origins)} tiles (median midslice NCC {meta['ncc_median']}, "
          f"median depth shift {np.median(depth_shifts) if depth_shifts else 0:+.2f} slices), "
          f"dropped {len(dropped)} -> {out_dir}", flush=True)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=tuple(PLANS), default="depth")
    parser.add_argument("--zarr-dir", default=None, help="directory holding the training zarrs (ves_zarrs2)")
    parser.add_argument("--repo", default=str(ROOT.parent), help="repo root with dilated_inklabels/ and masks/")
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--include-holdouts", action="store_true")
    parser.add_argument("--holdout-tiles", type=int, default=64)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--region", choices=("bbox", "near"), default="bbox",
                        help="bbox: the whole ink bounding box; near: only tiles touching ink")
    parser.add_argument("--margin", type=int, default=256, help="px the ink bbox (or ink) is grown by")
    parser.add_argument("--z0", type=int, default=8, help="first training-frame slice kept")
    parser.add_argument("--z1", type=int, default=20, help="the 8-slice MAE window moves within [z0, z1)")
    parser.add_argument("--depth-search", type=float, default=0.5, help="+- training slices searched per tile")
    parser.add_argument("--min-ncc", type=float, default=0.3)
    parser.add_argument("--max-residual", type=float, default=12.0, help="largest residual shift applied, px")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32, help="parallel S3 chunk downloads")
    parser.add_argument("--cache-dir", default=str(ROOT.parent / "_ves_tmp" / "crossres_high"),
                        help="local copy of the streamed high-resolution chunks")
    parser.add_argument("--count-only", action="store_true",
                        help="print tiles, download and disk per pair from the label / mask pngs, then exit")
    args = parser.parse_args()
    for key, value in PLANS[args.plan].items():
        setattr(args, key, value)
    if not args.count_only and not args.zarr_dir:
        parser.error("--zarr-dir is required unless --count-only")
    rng = np.random.default_rng(args.seed)
    roles = ("train", "holdout") if args.include_holdouts else ("train",)
    selected = [pair for pair in PAIRS
                if (not args.names or pair["name"] in args.names) and pair["role"] in roles]
    if args.count_only:
        totals = [build_pair(pair, {}, args, rng) for pair in selected]
        print(f"plan {args.plan}: download ~{sum(t[0] for t in totals) / 1e9:.1f} GB, "
              f"disk ~{sum(t[1] for t in totals) / 1e9:.1f} GB uncompressed")
        return
    registration = json.loads((ROOT / "sanity" / "registration.json").read_text())
    for pair in selected:
        entry = registration.get(pair["name"])
        if not entry or "low_to_high0" not in entry:
            print(f"[{pair['name']}] no registration; run sanity_midslice.py first", flush=True)
            continue
        if entry.get("verdict") == "INVESTIGATE" and not args.names:
            print(f"[{pair['name']}] registration verdict INVESTIGATE; skipped until resolved", flush=True)
            continue
        build_pair(pair, entry, args, rng)


if __name__ == "__main__":
    main()
