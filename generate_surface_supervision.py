"""Generate full-resolution papyrus-air surface pseudo-labels and review overlays.

The detector selects one papyrus-to-air transition per valid spatial column from
an explicitly chosen depth window. It uses relative occupancy because assembled
surface zarrs can retain a positive reconstruction floor in nominal air.

Example:
    python generate_surface_supervision.py --scroll-id 20240304141531 --z-start 4 --z-end 28
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import zarr


def _depth_quantiles_linear(x: np.ndarray, quantiles: tuple[float, ...]) -> list[np.ndarray]:
    """exact NumPy-linear quantiles along a short depth axis using one partial partition."""
    depth = x.shape[0]
    positions = [float(q) * (depth - 1) for q in quantiles]
    lower = [int(np.floor(position)) for position in positions]
    upper = [int(np.ceil(position)) for position in positions]
    partitioned = np.partition(x, sorted(set(lower + upper)), axis=0)
    out = []
    for position, lo, hi in zip(positions, lower, upper):
        if lo == hi:
            out.append(partitioned[lo])
        else:
            weight = np.float32(position - lo)
            out.append(partitioned[lo] * (1.0 - weight) + partitioned[hi] * weight)
    return out


def _unit_intensity(raw: np.ndarray) -> np.ndarray:
    """convert integer reconstruction values to a stable [0, 1] scale."""
    if np.issubdtype(raw.dtype, np.integer):
        scale = float(np.iinfo(raw.dtype).max)
        return raw.astype(np.float32) / max(scale, 1.0)
    return np.clip(raw.astype(np.float32), 0.0, 1.0)


def _detect_strip(
    raw: np.ndarray,
    threshold_frac: float,
    min_contrast: float,
    min_peak: float,
    min_margin: float,
    spatial_sigma: float,
    coarse_sigma: float,
    coarse_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """return relative depth, confidence, and validity for one DHW strip."""
    x = _unit_intensity(raw)
    padded = np.pad(x, ((1, 1), (0, 0), (0, 0)), mode="edge")
    smooth = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) * 0.25

    low, high = _depth_quantiles_linear(smooth, (0.10, 0.90))
    contrast = high - low
    threshold = low + threshold_frac * contrast
    tau = np.maximum(0.08 * contrast, 0.01)
    occupancy = 1.0 / (1.0 + np.exp(np.clip(-(smooth - threshold[None]) / tau[None], -30.0, 30.0)))

    transition = np.maximum(occupancy[:-1] - occupancy[1:], 0.0)
    # OpenCV blurs each channel independently. Treating depth as channels reduces
    # 2*D Python/OpenCV dispatches to two vectorized calls with identical results.
    transition_hwd = np.ascontiguousarray(np.moveaxis(transition, 0, -1))
    fine = cv2.GaussianBlur(
        transition_hwd,
        ksize=(0, 0),
        sigmaX=spatial_sigma,
        sigmaY=spatial_sigma,
    )
    coarse = cv2.GaussianBlur(
        transition_hwd,
        ksize=(0, 0),
        sigmaX=coarse_sigma,
        sigmaY=coarse_sigma,
    )
    transition = np.moveaxis(
        (1.0 - coarse_weight) * fine + coarse_weight * coarse,
        -1,
        0,
    )

    depth_index = transition.argmax(axis=0).astype(np.uint8)
    partitioned = np.partition(transition, -2, axis=0)
    peak = partitioned[-1]
    margin = partitioned[-1] - partitioned[-2]
    valid = (contrast >= min_contrast) & (peak >= min_peak) & (margin >= min_margin)

    # rescue isolated weak detections without bridging broad low-evidence regions
    neighbor_count = cv2.filter2D(
        valid.astype(np.uint8),
        ddepth=cv2.CV_16U,
        kernel=np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    isolated = (
        ~valid
        & (neighbor_count >= 7)
        & (contrast >= 0.70 * min_contrast)
        & (peak >= 0.70 * min_peak)
        & (margin >= 0.50 * min_margin)
    )
    valid |= isolated
    confidence = np.clip(peak, 0.0, 1.0)
    return depth_index, confidence, valid


def _write_overlays(
    volume,
    depth_map: np.ndarray,
    confidence: np.ndarray,
    z_start: int,
    z_end: int,
    output_dir: Path,
    output_height: int = 900,
    overlay_alpha: float = 0.25,
    ink_volume: np.ndarray | None = None,
    ink_alpha: float = 0.75,
) -> list[Path]:
    """write fixed-height layer views: guessed surface in red, optional 3D ink in white."""
    height, width = depth_map.shape
    if output_height <= 0:
        raise ValueError("output_height must be positive")
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be in [0, 1]")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_size = (max(1, int(round(width * output_height / height))), output_height)
    valid = confidence > 0
    paths = []

    for depth in range(z_start, z_end):
        raw_layer = np.asarray(volume[depth]).astype(np.float32)
        # display-only stretch: some uint16 zarrs hold 0..255 values and render black
        nonzero = raw_layer[::8, ::8][raw_layer[::8, ::8] > 0]
        display_max = float(np.percentile(nonzero, 99.5)) if nonzero.size else 1.0
        layer = np.clip(raw_layer * (255.0 / max(display_max, 1.0)), 0, 255).astype(np.uint8)
        gray = cv2.resize(layer, out_size, interpolation=cv2.INTER_AREA)
        selected = ((depth_map == depth) & valid).astype(np.uint8)
        selected = cv2.resize(selected, out_size, interpolation=cv2.INTER_NEAREST) > 0
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb[selected] = (
            (1.0 - overlay_alpha) * rgb[selected].astype(np.float32)
            + overlay_alpha * np.array([0.0, 0.0, 255.0], dtype=np.float32)
        ).astype(np.uint8)
        lines = [f"depth {depth:02d}  red=guessed surface ({100.0 * overlay_alpha:.0f}%)"]
        if ink_volume is not None:
            # alpha scales with ink probability, reaching ink_alpha at full confidence
            ink = cv2.resize(ink_volume[depth], out_size, interpolation=cv2.INTER_AREA)
            alpha = (ink_alpha * ink.astype(np.float32) / 255.0)[..., None]
            rgb = (rgb.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha).astype(np.uint8)
            lines.append(f"white=3D ink ({100.0 * ink_alpha:.0f}%)")
        lines.append(f"valid surface pixels: {int(selected.sum()):,}")
        for index, text in enumerate(lines):
            cv2.putText(
                rgb,
                text,
                (24, 46 + 40 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0 if index == 0 else 0.8,
                (255, 255, 255),
                3 if index == 0 else 2,
                cv2.LINE_AA,
            )
        path = output_dir / f"depth_{depth:02d}.jpg"
        if not cv2.imwrite(str(path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write layer view: {path}")
        paths.append(path)
    return paths


def _ink_surface_alignment(
    ink_volume: np.ndarray,
    depth_map: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """ink mass as a function of layer offset from the guessed surface."""
    depth = ink_volume.shape[0]
    valid = (confidence > 0) & mask & (depth_map != 255)
    surface = depth_map[valid].astype(np.int16)
    mass = np.zeros(2 * depth - 1, dtype=np.float64)
    for layer in range(depth):
        values = ink_volume[layer][valid].astype(np.float64)
        mass += np.bincount(layer - surface + depth - 1, weights=values, minlength=2 * depth - 1)
    total = float(mass.sum())
    offsets = np.arange(-(depth - 1), depth)
    share = mass / max(total, 1e-12)
    peak = int(offsets[int(mass.argmax())]) if total > 0 else 0
    absolute = np.array([float(ink_volume[layer][valid].astype(np.float64).sum()) for layer in range(depth)])
    return {
        "offset_semantics": "ink layer minus guessed surface layer",
        "total_ink_mass": total,
        "peak_offset": peak,
        "share_at_surface": float(share[depth - 1]),
        "share_within_1": float(share[depth - 2:depth + 1].sum()),
        "share_within_2": float(share[depth - 3:depth + 2].sum()),
        "share_below_surface": float(share[:depth - 1].sum()),
        "share_above_surface": float(share[depth:].sum()),
        "share_by_offset": {str(int(o)): float(s) for o, s in zip(offsets, share) if s > 0},
        "share_by_absolute_layer": {
            str(layer): float(value / max(absolute.sum(), 1e-12)) for layer, value in enumerate(absolute)
        },
    }


def _review_layer_views(paths: list[Path], start_index: int = 0) -> None:
    """review pre-rendered layers in one persistent OpenCV window."""
    if not paths:
        raise ValueError("no layer views were provided")
    if os.name == "posix" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "OpenCV review requires DISPLAY or WAYLAND_DISPLAY; run --review-only "
            "from a graphical shell"
        )
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"could not load layer view: {path}")
        frames.append(frame)

    index = min(max(int(start_index), 0), len(frames) - 1)
    window = "surface layer review | up/down: layer | q/esc: close"
    up_keys = {82, 2490368, 65362}
    down_keys = {84, 2621440, 65364}
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            cv2.imshow(window, frames[index])
            key = cv2.waitKeyEx(0)
            if key in (27, ord("q"), ord("Q")):
                break
            if key in up_keys:
                index = min(index + 1, len(frames) - 1)
            elif key in down_keys:
                index = max(index - 1, 0)
    finally:
        cv2.destroyWindow(window)


def _regularize_surface_map(
    depth_map: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    row_block: int = 512,
    halo: int = 24,
) -> dict[str, int]:
    """remove isolated depth spikes and fill only tiny surrounded confidence holes."""
    height, width = depth_map.shape
    source_depth = np.asarray(depth_map)
    source_confidence = np.asarray(confidence)
    regularized_depth = np.empty((height, width), dtype=np.uint8)
    regularized_confidence = np.empty((height, width), dtype=np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled_total = 0
    outlier_total = 0

    for y0 in range(0, height, row_block):
        y1 = min(height, y0 + row_block)
        ys = max(0, y0 - halo)
        ye = min(height, y1 + halo)
        local_mask = mask[ys:ye]
        local_depth = source_depth[ys:ye]
        local_confidence = source_confidence[ys:ye]
        local_valid = (local_confidence > 0) & local_mask

        support = cv2.GaussianBlur(local_valid.astype(np.float32), (0, 0), 3.0)
        weighted_depth = cv2.GaussianBlur(
            np.where(local_valid, local_depth, 0).astype(np.float32),
            (0, 0),
            3.0,
        )
        prior = np.rint(weighted_depth / np.maximum(support, 1e-4)).clip(0, 254).astype(np.uint8)
        closed = cv2.morphologyEx(local_valid.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel) > 0
        fill = closed & ~local_valid & local_mask & (support >= 0.9)

        working_depth = np.where(local_valid, local_depth, prior).astype(np.uint8)
        local_median = cv2.medianBlur(working_depth, 5)
        outlier = (
            local_valid
            & (support >= 0.8)
            & (np.abs(local_depth.astype(np.int16) - local_median.astype(np.int16)) > 1)
        )

        local_out_depth = local_depth.copy()
        local_out_confidence = local_confidence.copy()
        local_out_depth[fill] = prior[fill]
        local_out_depth[outlier] = local_median[outlier]
        neighbor_confidence = cv2.GaussianBlur(local_confidence.astype(np.float32), (0, 0), 3.0)
        local_out_confidence[fill] = np.clip(neighbor_confidence[fill] * 0.5, 1, 255).astype(np.uint8)
        local_out_confidence[outlier] = np.minimum(
            local_out_confidence[outlier],
            np.clip(neighbor_confidence[outlier], 1, 255).astype(np.uint8),
        )
        local_out_depth[~local_mask] = 255
        local_out_confidence[~local_mask] = 0

        keep = slice(y0 - ys, y1 - ys)
        regularized_depth[y0:y1] = local_out_depth[keep]
        regularized_confidence[y0:y1] = local_out_confidence[keep]
        filled_total += int(fill[keep].sum())
        outlier_total += int(outlier[keep].sum())

    # confidence is thresholded per pixel, so isolated failures remain even after
    # score smoothing. fill only components of at most four pixels; larger holes
    # retain their invalid status because they commonly correspond to weak scans.
    topology_depth = regularized_depth.copy()
    topology_confidence = regularized_confidence.copy()
    component_filled = 0
    depth_islands_replaced = 0
    component_halo = 16
    for y0 in range(0, height, row_block):
        y1 = min(height, y0 + row_block)
        for x0 in range(0, width, row_block):
            x1 = min(width, x0 + row_block)
            ys = max(0, y0 - component_halo)
            ye = min(height, y1 + component_halo)
            xs = max(0, x0 - component_halo)
            xe = min(width, x1 + component_halo)
            local_mask = mask[ys:ye, xs:xe]
            local_valid = (regularized_confidence[ys:ye, xs:xe] > 0) & local_mask
            invalid = (local_mask & ~local_valid).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(invalid, 8)
            if count <= 1:
                continue
            small_lookup = np.zeros(count, dtype=bool)
            small_lookup[1:] = stats[1:, cv2.CC_STAT_AREA] <= 16
            small = small_lookup[labels]
            if not small.any():
                continue

            support = cv2.GaussianBlur(local_valid.astype(np.float32), (0, 0), 3.0)
            weighted_depth = cv2.GaussianBlur(
                np.where(local_valid, regularized_depth[ys:ye, xs:xe], 0).astype(np.float32),
                (0, 0),
                3.0,
            )
            prior = np.rint(weighted_depth / np.maximum(support, 1e-4)).clip(0, 254).astype(np.uint8)
            neighbor_confidence = cv2.GaussianBlur(
                regularized_confidence[ys:ye, xs:xe].astype(np.float32),
                (0, 0),
                3.0,
            )
            core = (slice(y0 - ys, y1 - ys), slice(x0 - xs, x1 - xs))
            core_small = small[core]
            core_depth = topology_depth[y0:y1, x0:x1]
            core_confidence = topology_confidence[y0:y1, x0:x1]
            core_depth[core_small] = prior[core][core_small]
            core_confidence[core_small] = np.clip(
                neighbor_confidence[core][core_small] * 0.5,
                1,
                255,
            ).astype(np.uint8)
            component_filled += int(core_small.sum())

    # remove small connected depth islands even when they border an invalid hole.
    # larger components are retained because they can represent real folds.
    final_valid = (topology_confidence > 0) & mask
    for y0 in range(0, height, row_block):
        y1 = min(height, y0 + row_block)
        for x0 in range(0, width, row_block):
            x1 = min(width, x0 + row_block)
            ys = max(0, y0 - component_halo)
            ye = min(height, y1 + component_halo)
            xs = max(0, x0 - component_halo)
            xe = min(width, x1 + component_halo)
            local_valid = final_valid[ys:ye, xs:xe]
            local_depth = topology_depth[ys:ye, xs:xe]
            support = cv2.GaussianBlur(local_valid.astype(np.float32), (0, 0), 3.0)
            weighted_depth = cv2.GaussianBlur(
                np.where(local_valid, local_depth, 0).astype(np.float32),
                (0, 0),
                3.0,
            )
            prior = np.rint(weighted_depth / np.maximum(support, 1e-4)).clip(0, 254).astype(np.uint8)
            local_median = cv2.medianBlur(np.where(local_valid, local_depth, prior).astype(np.uint8), 5)
            discrepant = (
                local_valid
                & (np.abs(local_depth.astype(np.int16) - local_median.astype(np.int16)) > 1)
            ).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(discrepant, 8)
            if count <= 1:
                continue
            small_lookup = np.zeros(count, dtype=bool)
            small_lookup[1:] = stats[1:, cv2.CC_STAT_AREA] <= 16
            small = small_lookup[labels]
            core = (slice(y0 - ys, y1 - ys), slice(x0 - xs, x1 - xs))
            core_small = small[core]
            core_depth = topology_depth[y0:y1, x0:x1]
            core_depth[core_small] = local_median[core][core_small]
            depth_islands_replaced += int(core_small.sum())

    depth_map[:] = topology_depth
    confidence[:] = topology_confidence
    return {
        "surrounded_holes_filled": filled_total,
        "small_components_filled": component_filled,
        "depth_outliers_replaced": outlier_total,
        "small_depth_islands_replaced": depth_islands_replaced,
    }


def _elastic_surface_map(
    depth_map: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    initial_depth: float = 12.0,
    grid_step: int = 16,
    iterations: int = 80,
    smooth_sigma: float = 1.5,
    data_strength: float = 0.28,
    smooth_strength: float = 0.45,
    confidence_floor: float = 0.08,
    max_update: float = 0.25,
    max_gradient: float = 0.55,
    fill_radius: int = 8,
) -> dict[str, float | int]:
    """fit a confidence-weighted elastic sheet and preserve broad invalid holes."""
    height, width = depth_map.shape
    observed_valid = (confidence > 0) & mask & (depth_map != 255)
    step = max(1, int(grid_step))
    coarse_width = max(1, int(np.ceil(width / step)))
    coarse_height = max(1, int(np.ceil(height / step)))

    weights = np.where(observed_valid, confidence.astype(np.float32) / 255.0, 0.0)
    weighted_depth = np.where(observed_valid, depth_map, 0).astype(np.float32) * weights
    coarse_weight = cv2.resize(weights, (coarse_width, coarse_height), interpolation=cv2.INTER_AREA)
    coarse_weighted_depth = cv2.resize(
        weighted_depth,
        (coarse_width, coarse_height),
        interpolation=cv2.INTER_AREA,
    )
    observed = coarse_weighted_depth / np.maximum(coarse_weight, 1e-6)
    data_weight = np.clip(coarse_weight, 0.0, 1.0)
    data_weight[data_weight < float(confidence_floor)] = 0.0
    coarse_mask = cv2.resize(
        mask.astype(np.uint8),
        (coarse_width, coarse_height),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    surface = np.full((coarse_height, coarse_width), float(initial_depth), dtype=np.float32)
    if np.any(data_weight > 0):
        seed = cv2.GaussianBlur(
            observed * data_weight,
            (0, 0),
            max(float(smooth_sigma) * 2.0, 1.0),
        )
        seed_weight = cv2.GaussianBlur(
            data_weight,
            (0, 0),
            max(float(smooth_sigma) * 2.0, 1.0),
        )
        seeded = seed / np.maximum(seed_weight, 1e-5)
        surface = np.where(seed_weight > 1e-3, seeded, surface).astype(np.float32)

    for _ in range(max(0, int(iterations))):
        smooth = cv2.GaussianBlur(
            surface,
            (0, 0),
            max(float(smooth_sigma), 0.1),
            borderType=cv2.BORDER_REPLICATE,
        )
        update = (
            float(data_strength) * data_weight * (observed - surface)
            + float(smooth_strength) * (smooth - surface)
        )
        update *= np.clip(coarse_mask, 0.0, 1.0)
        surface += np.clip(update, -float(max_update), float(max_update))

        # alternate directional projections to cap local slope without flattening folds
        limit = max(float(max_gradient), 0.0)
        if limit > 0:
            for _projection in range(2):
                surface[:, 1:] = np.clip(
                    surface[:, 1:], surface[:, :-1] - limit, surface[:, :-1] + limit
                )
                surface[:, :-1] = np.clip(
                    surface[:, :-1], surface[:, 1:] - limit, surface[:, 1:] + limit
                )
                surface[1:, :] = np.clip(
                    surface[1:, :], surface[:-1, :] - limit, surface[:-1, :] + limit
                )
                surface[:-1, :] = np.clip(
                    surface[:-1, :], surface[1:, :] - limit, surface[1:, :] + limit
                )

    full_surface = cv2.resize(surface, (width, height), interpolation=cv2.INTER_LINEAR)
    full_surface = np.clip(full_surface, 0.0, 254.0)

    # iterative pinhole repair is intentionally stricter than the larger close below
    patched_valid = observed_valid.copy()
    pinholes = np.zeros_like(observed_valid)
    for _ in range(3):
        neighbor_count = cv2.filter2D(
            patched_valid.astype(np.uint8),
            ddepth=cv2.CV_16U,
            kernel=np.ones((3, 3), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT,
        )
        new_pinhole = mask & ~patched_valid & (neighbor_count >= 7)
        if not new_pinhole.any():
            break
        patched_valid |= new_pinhole
        pinholes |= new_pinhole

    fill_radius = max(0, int(fill_radius))
    if fill_radius > 0:
        kernel_size = 2 * fill_radius + 1
        fill_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        closed_valid = cv2.morphologyEx(
            patched_valid.astype(np.uint8),
            cv2.MORPH_CLOSE,
            fill_kernel,
        ) > 0
        fill = closed_valid & ~observed_valid & mask
    else:
        fill = pinholes

    output_valid = observed_valid | fill
    fitted_depth = np.rint(full_surface).astype(np.uint8)
    original_confidence = confidence.copy()
    depth_map[output_valid] = fitted_depth[output_valid]
    depth_map[~output_valid] = 255
    confidence[observed_valid] = np.maximum(original_confidence[observed_valid], 1)
    if fill.any():
        nearby_confidence = cv2.GaussianBlur(
            original_confidence.astype(np.float32),
            (0, 0),
            max(float(fill_radius) / 2.0, 1.0),
        )
        confidence[fill] = np.clip(nearby_confidence[fill] * 0.5, 1, 255).astype(np.uint8)
    confidence[~output_valid] = 0

    valid_depth = full_surface[output_valid]
    grad_y, grad_x = np.gradient(full_surface)
    gradient = np.hypot(grad_y, grad_x)[output_valid]
    return {
        "initial_depth": float(initial_depth),
        "grid_step": int(step),
        "iterations": int(iterations),
        "smooth_sigma": float(smooth_sigma),
        "data_strength": float(data_strength),
        "smooth_strength": float(smooth_strength),
        "confidence_floor": float(confidence_floor),
        "max_update": float(max_update),
        "max_gradient": float(max_gradient),
        "fill_radius": int(fill_radius),
        "isolated_pinholes_filled": int(pinholes.sum()),
        "confidence_gaps_filled": int(fill.sum()),
        "valid_depth_mean": float(valid_depth.mean()) if valid_depth.size else 0.0,
        "mean_gradient": float(gradient.mean()) if gradient.size else 0.0,
        "p99_gradient": float(np.quantile(gradient, 0.99)) if gradient.size else 0.0,
    }


def _write_depth_overview(
    depth_map: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    z_start: int,
    z_end: int,
    output_path: Path,
    downscale: int = 2,
) -> None:
    """write one full-resolution grayscale map encoding all predicted depths."""
    valid = confidence > 0
    scale = 255.0 / max(1, z_end - z_start)
    gray = np.clip((depth_map.astype(np.float32) - z_start) * scale, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb[~valid] = (80, 0, 80)
    resize_scale = 1.0 / max(1, int(downscale))
    if resize_scale < 1.0:
        rgb = cv2.resize(
            rgb,
            (int(round(rgb.shape[1] * resize_scale)), int(round(rgb.shape[0] * resize_scale))),
            interpolation=cv2.INTER_NEAREST,
        )

    legend_width = 520
    canvas = np.zeros((rgb.shape[0], rgb.shape[1] + legend_width, 3), dtype=np.uint8)
    canvas[:, :rgb.shape[1]] = rgb
    x0 = rgb.shape[1] + 55
    cv2.putText(canvas, "SURFACE DEPTH", (x0, 110), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "black = depth 4", (x0, 185), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "gray = depth 16", (x0, 245), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "white scale endpoint = depth 28", (x0, 305), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "purple = invalid", (x0, 365), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "latest detectable transition = depth 26", (x0, 415),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    bar_y0, bar_y1 = 500, min(canvas.shape[0] - 100, 2500)
    bar = np.linspace(0, 255, bar_y1 - bar_y0, dtype=np.uint8)[:, None]
    canvas[bar_y0:bar_y1, x0:x0 + 100] = cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)
    for depth in (z_start, (z_start + z_end) // 2, z_end):
        y = int(round(bar_y0 + (depth - z_start) / max(1, z_end - z_start) * (bar_y1 - bar_y0 - 1)))
        value = int(round((depth - z_start) * scale))
        cv2.line(canvas, (x0 + 105, y), (x0 + 135, y), (255, 255, 255), 3)
        cv2.putText(canvas, f"depth {depth}: {value}", (x0 + 150, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    frame_coverage = float(valid.mean())
    mask_coverage = float((valid & mask).sum() / max(int(mask.sum()), 1))
    cv2.putText(canvas, f"frame coverage: {100.0 * frame_coverage:.2f}%", (x0, bar_y1 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"scroll-mask coverage: {100.0 * mask_coverage:.2f}%", (x0, bar_y1 + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])


def main() -> None:
    parser = argparse.ArgumentParser(description="generate papyrus-air surface pseudo-labels")
    parser.add_argument("--scroll-id", type=int, required=True)
    parser.add_argument("--zarr-dir", default="./ves_zarrs2")
    parser.add_argument("--mask-dir", default="./masks")
    parser.add_argument("--output-dir", default="./surface_labels")
    parser.add_argument("--review-dir", default="./output/surface_review")
    parser.add_argument("--z-start", type=int, default=4)
    parser.add_argument("--z-end", type=int, default=28)
    parser.add_argument("--row-block", type=int, default=128)
    parser.add_argument("--halo", type=int, default=8)
    parser.add_argument("--threshold-frac", type=float, default=0.35)
    parser.add_argument("--min-contrast", type=float, default=0.08)
    parser.add_argument("--min-peak", type=float, default=0.08)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--spatial-sigma", type=float, default=1.25)
    parser.add_argument("--coarse-sigma", type=float, default=8.0)
    parser.add_argument("--coarse-weight", type=float, default=0.65)
    parser.add_argument("--elastic-initial-depth", type=float, default=12.0)
    parser.add_argument("--elastic-grid-step", type=int, default=16)
    parser.add_argument("--elastic-iterations", type=int, default=80)
    parser.add_argument("--elastic-sigma", type=float, default=1.5)
    parser.add_argument("--elastic-data-strength", type=float, default=0.28)
    parser.add_argument("--elastic-smooth-strength", type=float, default=0.45)
    parser.add_argument("--elastic-confidence-floor", type=float, default=0.08)
    parser.add_argument("--elastic-max-update", type=float, default=0.25)
    parser.add_argument("--elastic-max-gradient", type=float, default=0.55)
    parser.add_argument("--elastic-fill-radius", type=int, default=8)
    parser.add_argument("--prefetch-blocks", type=int, default=2,
                        help="bounded background zarr reads to overlap I/O with detection")
    parser.add_argument("--view-all-layers", action="store_true",
                        help="also write per-depth layer jpgs (default: only the depth overview)")
    parser.add_argument("--review", action="store_true",
                        help="open generated layer views in a persistent OpenCV window (implies --view-all-layers)")
    parser.add_argument("--review-only", action="store_true",
                        help="open existing layer views without rebuilding supervision")
    parser.add_argument("--review-height", type=int, default=900)
    parser.add_argument("--overlay-alpha", type=float, default=0.25)
    parser.add_argument("--ink3d-zarr", default=None,
                        help="28-layer 3D ink zarr (extract_ink3d_patch.py) drawn in white")
    parser.add_argument("--ink-alpha", type=float, default=0.75)
    parser.add_argument("--overlays-only", action="store_true",
                        help="re-render layer views from existing depth.npy/confidence.npy")
    args = parser.parse_args()

    # review needs the per-depth frames
    args.view_all_layers = args.view_all_layers or args.review
    scroll_id = str(args.scroll_id)
    review_root = Path(args.review_dir)
    if args.review_only:
        direct_paths = sorted(review_root.glob("depth_*.jpg"))
        nested_paths = sorted((review_root / scroll_id).glob("depth_*.jpg"))
        _review_layer_views(direct_paths or nested_paths)
        return

    volume = zarr.open(os.path.join(args.zarr_dir, f"{scroll_id}.zarr"), mode="r")
    depth, height, width = map(int, volume.shape)
    if not (0 <= args.z_start < args.z_end <= depth):
        raise ValueError(f"invalid depth window [{args.z_start}, {args.z_end}) for depth {depth}")

    mask = cv2.imread(os.path.join(args.mask_dir, f"{scroll_id}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (height, width):
        raise ValueError(f"missing or mismatched scroll mask for {scroll_id}")
    mask = mask > 0

    output_dir = Path(args.output_dir) / scroll_id
    review_dir = review_root / scroll_id
    ink_volume = None
    if args.ink3d_zarr:
        ink_array = zarr.open(args.ink3d_zarr, mode="r")
        if tuple(ink_array.shape) != (depth, height, width):
            raise ValueError(f"ink zarr shape {ink_array.shape} != volume {(depth, height, width)}")
        ink_volume = np.asarray(ink_array[:])

    if args.overlays_only:
        depth_map = np.load(output_dir / "depth.npy", mmap_mode="r")
        confidence = np.load(output_dir / "confidence.npy", mmap_mode="r")
        review_dir.mkdir(parents=True, exist_ok=True)
        for stale_overlay in review_dir.glob("depth_*.jpg"):
            stale_overlay.unlink()
        overlay_paths = []
        if args.view_all_layers:
            overlay_paths = _write_overlays(
                volume, depth_map, confidence, args.z_start, args.z_end, review_dir,
                output_height=args.review_height, overlay_alpha=args.overlay_alpha,
                ink_volume=ink_volume, ink_alpha=args.ink_alpha,
            )
        _write_depth_overview(
            depth_map, confidence, mask, args.z_start, args.z_end,
            review_dir / "surface_depth_overview.jpg", downscale=2,
        )
        if ink_volume is not None:
            alignment = _ink_surface_alignment(ink_volume, depth_map, confidence, mask)
            with open(review_dir / "ink_surface_alignment.json", "w", encoding="utf-8") as handle:
                json.dump(alignment, handle, indent=2)
            print("[surface] ink vs surface: " + json.dumps(
                {k: v for k, v in alignment.items() if not isinstance(v, dict)}))
        print(f"[surface] depth overview -> {review_dir / 'surface_depth_overview.jpg'}")
        if overlay_paths:
            print(f"[surface] layer views -> {review_dir / 'depth_*.jpg'}")
        if args.review:
            _review_layer_views(overlay_paths)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    for stale_overlay in review_dir.glob("depth_*.jpg"):
        stale_overlay.unlink()
    depth_path = output_dir / "depth.npy"
    confidence_path = output_dir / "confidence.npy"
    depth_map = np.lib.format.open_memmap(depth_path, mode="w+", dtype=np.uint8, shape=(height, width))
    confidence = np.lib.format.open_memmap(
        confidence_path,
        mode="w+",
        dtype=np.uint8,
        shape=(height, width),
    )
    depth_map[:] = 255
    confidence[:] = 0

    block_specs = []
    for y0 in range(0, height, args.row_block):
        y1 = min(height, y0 + args.row_block)
        ys = max(0, y0 - args.halo)
        ye = min(height, y1 + args.halo)
        block_specs.append((y0, y1, ys, ye))

    def read_block(spec):
        _y0, _y1, ys, ye = spec
        return np.asarray(volume[args.z_start:args.z_end, ys:ye, :])

    def prefetched_blocks():
        prefetch = max(0, int(args.prefetch_blocks))
        if prefetch == 0:
            for spec in block_specs:
                yield spec, read_block(spec)
            return
        with ThreadPoolExecutor(max_workers=prefetch) as pool:
            pending = deque()
            next_index = 0
            while next_index < min(prefetch, len(block_specs)):
                spec = block_specs[next_index]
                pending.append((spec, pool.submit(read_block, spec)))
                next_index += 1
            while pending:
                spec, future = pending.popleft()
                raw = future.result()
                if next_index < len(block_specs):
                    next_spec = block_specs[next_index]
                    pending.append((next_spec, pool.submit(read_block, next_spec)))
                    next_index += 1
                yield spec, raw

    detection_started = time.perf_counter()
    for (y0, y1, ys, ye), raw in prefetched_blocks():
        rel_depth, conf, valid = _detect_strip(
            raw,
            threshold_frac=args.threshold_frac,
            min_contrast=args.min_contrast,
            min_peak=args.min_peak,
            min_margin=args.min_margin,
            spatial_sigma=args.spatial_sigma,
            coarse_sigma=args.coarse_sigma,
            coarse_weight=args.coarse_weight,
        )
        keep = slice(y0 - ys, y1 - ys)
        valid = valid[keep] & mask[y0:y1]
        absolute_depth = rel_depth[keep].astype(np.uint16) + args.z_start
        depth_map[y0:y1] = np.where(valid, absolute_depth, 255).astype(np.uint8)
        confidence[y0:y1] = np.where(valid, np.clip(conf[keep] * 255.0, 1, 255), 0).astype(np.uint8)
        print(f"[surface] rows {y0}:{y1}/{height}", flush=True)
    print(f"[surface] detection: {time.perf_counter() - detection_started:.1f}s", flush=True)

    depth_map.flush()
    confidence.flush()
    regularization_started = time.perf_counter()
    regularization = _regularize_surface_map(depth_map, confidence, mask)
    print(
        f"[surface] topology regularization: "
        f"{time.perf_counter() - regularization_started:.1f}s",
        flush=True,
    )
    elastic_started = time.perf_counter()
    elastic = _elastic_surface_map(
        depth_map,
        confidence,
        mask,
        initial_depth=args.elastic_initial_depth,
        grid_step=args.elastic_grid_step,
        iterations=args.elastic_iterations,
        smooth_sigma=args.elastic_sigma,
        data_strength=args.elastic_data_strength,
        smooth_strength=args.elastic_smooth_strength,
        confidence_floor=args.elastic_confidence_floor,
        max_update=args.elastic_max_update,
        max_gradient=args.elastic_max_gradient,
        fill_radius=args.elastic_fill_radius,
    )
    print(f"[surface] elastic fit: {time.perf_counter() - elastic_started:.1f}s", flush=True)
    depth_map.flush()
    confidence.flush()
    valid_count = int((confidence > 0).sum())
    mask_count = int(mask.sum())
    histogram = {
        str(depth_index): int(((depth_map == depth_index) & (confidence > 0)).sum())
        for depth_index in range(args.z_start, args.z_end)
    }
    metadata = {
        "scroll_id": int(args.scroll_id),
        "volume_shape": [depth, height, width],
        "z_start": args.z_start,
        "z_end": args.z_end,
        "depth_semantics": "last papyrus-like slice before strongest papyrus-to-air transition",
        "invalid_depth_value": 255,
        "valid_pixels": valid_count,
        "mask_pixels": mask_count,
        "valid_fraction_inside_mask": valid_count / max(mask_count, 1),
        "threshold_frac": args.threshold_frac,
        "min_contrast": args.min_contrast,
        "min_peak": args.min_peak,
        "min_margin": args.min_margin,
        "spatial_sigma": args.spatial_sigma,
        "coarse_sigma": args.coarse_sigma,
        "coarse_weight": args.coarse_weight,
        "regularization": regularization,
        "elastic": elastic,
        "histogram": histogram,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    overlay_paths = []
    if args.view_all_layers:
        overlay_paths = _write_overlays(
            volume,
            depth_map,
            confidence,
            args.z_start,
            args.z_end,
            review_dir,
            output_height=args.review_height,
            overlay_alpha=args.overlay_alpha,
            ink_volume=ink_volume,
            ink_alpha=args.ink_alpha,
        )
    if ink_volume is not None:
        alignment = _ink_surface_alignment(ink_volume, depth_map, confidence, mask)
        with open(review_dir / "ink_surface_alignment.json", "w", encoding="utf-8") as handle:
            json.dump(alignment, handle, indent=2)
    _write_depth_overview(
        depth_map,
        confidence,
        mask,
        args.z_start,
        args.z_end,
        review_dir / "surface_depth_overview.jpg",
        downscale=2,
    )
    print(f"[surface] depth labels -> {depth_path}")
    print(f"[surface] confidence -> {confidence_path}")
    if overlay_paths:
        print(f"[surface] layer views -> {review_dir / 'depth_*.jpg'}")
    print(f"[surface] depth overview -> {review_dir / 'surface_depth_overview.jpg'}")
    print(f"[surface] valid inside mask: {100.0 * metadata['valid_fraction_inside_mask']:.2f}%")
    if args.review:
        mode_depth = max(histogram, key=histogram.get)
        _review_layer_views(
            overlay_paths,
            start_index=int(mode_depth) - args.z_start,
        )


if __name__ == "__main__":
    main()
