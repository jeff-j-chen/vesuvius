"""text-line layout from the ink labels: where writing lines run, with and without the neighbouring windings.

own:   bands are the stretches along the line-normal axis where this patch's own ink labels sit (smoothed row profile).
cross: consecutive windings of one scroll share the line grid, so the row profiles of the labelled neighbour sheets are
       registered to each other by 1-D cross-correlation and their bands pooled; a patch then also gets the bands of lines
       whose letters are invisible on it but visible one winding over.

a band is only claimed over the patch's text-column extent (its own ink extent along the line direction plus a margin).
cells are `unit` px squares on the patch's pixel grid.
"""
from __future__ import annotations

import functools
import os

import cv2
import numpy as np

LABEL_DIR = "./dilated_inklabels"
BIN = 32            # px, profile resolution for axis choice and registration
SMOOTH_PX = 25.0    # gaussian sigma of the row profile, px
BAND_THRESHOLD = 0.12
BAND_PAD_PX = 24
MERGE_GAP_PX = 24
EXTENT_MARGIN_PX = 320
GAP_SHRINK_PX = 16
MIN_GAP_PX = 32
MAX_SHIFT_BINS = 40
# upright straightened windings sharing one line grid (flat aliases); lines run along the image x axis
FAMILY = (20261012065813, 20261012065817, 20261012065819, 20261012065821, 20261012065823)
REFERENCE = 20261012065819


@functools.lru_cache(maxsize=None)
def ink_map(scroll_id: int) -> np.ndarray:
    path = os.path.join(LABEL_DIR, f"{int(scroll_id)}.png")
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return image >= 200


def _profile(ink: np.ndarray, axis: int) -> np.ndarray:
    """ink pixels per line-normal position; axis 0 = rows (text lines run along x)."""
    return ink.sum(axis=1 - axis).astype(np.float64)


def _smooth(profile: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(3 * sigma) + 1
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return np.convolve(profile, kernel / kernel.sum(), mode="same")


# raw arrays whose writing lines run along y (the display rotates them a quarter turn); every other patch has lines along x
LINES_ALONG_Y = (20261012073246, 20261012073250)


def line_axis(scroll_id: int) -> int:
    """0 when the writing lines run along x (rows are the line-normal axis), 1 when they run along y."""
    return 1 if int(scroll_id) in LINES_ALONG_Y else 0


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(edges == 1)[0].tolist(), np.nonzero(edges == -1)[0].tolist()))


def _bands(profile: np.ndarray) -> list[tuple[int, int]]:
    smooth = _smooth(profile, SMOOTH_PX)
    if smooth.max() <= 0:
        return []
    raw = _intervals(smooth > BAND_THRESHOLD * smooth.max())
    padded = [(max(0, a - BAND_PAD_PX), min(len(profile), b + BAND_PAD_PX)) for a, b in raw]
    merged: list[tuple[int, int]] = []
    for a, b in padded:
        if merged and a - merged[-1][1] < MERGE_GAP_PX:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


@functools.lru_cache(maxsize=None)
def _binned(scroll_id: int) -> np.ndarray:
    profile = _profile(ink_map(scroll_id), 0)
    n = len(profile) // BIN
    return _smooth(profile[:n * BIN].reshape(n, BIN).sum(1), 1.0)


@functools.lru_cache(maxsize=None)
def registration_shift_px(scroll_id: int) -> int:
    """rows of `scroll_id` -> rows of the reference winding (add the returned px)."""
    scroll_id = int(scroll_id)
    if scroll_id == REFERENCE:
        return 0
    ref, own = _binned(REFERENCE), _binned(scroll_id)
    best, best_shift = -2.0, 0
    for shift in range(-MAX_SHIFT_BINS, MAX_SHIFT_BINS + 1):
        a = ref[shift:] if shift >= 0 else ref[:len(ref) + shift]
        b = own[:len(a)] if shift >= 0 else own[-shift:]
        n = min(len(a), len(b))
        if n < 20 or a[:n].std() < 1e-9 or b[:n].std() < 1e-9:
            continue
        corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if corr > best:
            best, best_shift = corr, shift
    return best_shift * BIN


def _extent(ink: np.ndarray, axis: int) -> tuple[int, int]:
    along = ink.any(axis=axis)  # positions along the line direction that hold ink
    where = np.nonzero(along)[0]
    return (int(where[0]), int(where[-1])) if where.size else (0, ink.shape[1 - axis] - 1)


@functools.lru_cache(maxsize=None)
def bands_px(scroll_id: int, source: str) -> tuple[tuple[int, int], ...]:
    """text-line bands along the line-normal axis, in this patch's pixel coordinates."""
    scroll_id = int(scroll_id)
    ink = ink_map(scroll_id)
    axis = line_axis(scroll_id)
    own = _bands(_profile(ink, axis))
    if source == "own" or scroll_id not in FAMILY:
        return tuple(own)
    if source != "cross":
        raise ValueError(f"unknown line prior source {source!r}")
    total = ink.shape[axis]
    to_ref = registration_shift_px(scroll_id)
    pooled = np.zeros(total, dtype=bool)
    for other in FAMILY:
        shift = registration_shift_px(other) - to_ref  # other's rows -> this patch's rows
        for a, b in _bands(_profile(ink_map(other), 0)):
            lo, hi = max(0, a + shift), min(total, b + shift)
            if hi > lo:
                pooled[lo:hi] = True
    bands = _intervals(pooled)
    return tuple(bands)


def _unit_axis_mask(intervals, length_px: int, unit: int) -> np.ndarray:
    """per-unit flag along one axis: the unit's centre lies inside an interval."""
    count = length_px // unit
    centres = np.arange(count) * unit + unit // 2
    flag = np.zeros(count, dtype=bool)
    for a, b in intervals:
        flag |= (centres >= a) & (centres < b)
    return flag


def line_cells(scroll_id: int, shape_hw: tuple[int, int], unit: int, source: str):
    """(band cells, gap cells) on the unit grid of a (h, w) patch; True where a cell sits in a text line / between two lines."""
    scroll_id = int(scroll_id)
    h, w = shape_hw
    ink = ink_map(scroll_id)
    axis = line_axis(scroll_id)
    bands = bands_px(scroll_id, source)
    lo, hi = _extent(ink, axis)
    extent = [(max(0, lo - EXTENT_MARGIN_PX), hi + EXTENT_MARGIN_PX + 1)]
    along_len, normal_len = (w, h) if axis == 0 else (h, w)
    band_flag = _unit_axis_mask(bands, normal_len, unit)
    extent_flag = _unit_axis_mask(extent, along_len, unit)
    # gaps are always the stretches between this patch's own visible lines; pooled neighbour bands never erase them
    own = bands_px(scroll_id, "own")
    gaps = []
    for (_, end), (start, _) in zip(own[:-1], own[1:]):
        if start - end >= MIN_GAP_PX + 2 * GAP_SHRINK_PX:
            gaps.append((end + GAP_SHRINK_PX, start - GAP_SHRINK_PX))
    gap_flag = _unit_axis_mask(gaps, normal_len, unit)
    if axis == 0:
        band_cells = band_flag[:, None] & extent_flag[None, :]
        gap_cells = gap_flag[:, None] & extent_flag[None, :]
    else:
        band_cells = extent_flag[:, None] & band_flag[None, :]
        gap_cells = extent_flag[:, None] & gap_flag[None, :]
    return band_cells & ~gap_cells, gap_cells
