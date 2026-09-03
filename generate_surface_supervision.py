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
from pathlib import Path

import cv2
import numpy as np
import zarr


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
    x = raw.astype(np.float32) / 255.0
    padded = np.pad(x, ((1, 1), (0, 0), (0, 0)), mode="edge")
    smooth = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) * 0.25

    low = np.quantile(smooth, 0.10, axis=0)
    high = np.quantile(smooth, 0.90, axis=0)
    contrast = high - low
    threshold = low + threshold_frac * contrast
    tau = np.maximum(0.08 * contrast, 0.01)
    occupancy = 1.0 / (1.0 + np.exp(np.clip(-(smooth - threshold[None]) / tau[None], -30.0, 30.0)))

    transition = np.maximum(occupancy[:-1] - occupancy[1:], 0.0)
    for depth in range(transition.shape[0]):
        fine = cv2.GaussianBlur(
            transition[depth],
            ksize=(0, 0),
            sigmaX=spatial_sigma,
            sigmaY=spatial_sigma,
        )
        coarse = cv2.GaussianBlur(
            transition[depth],
            ksize=(0, 0),
            sigmaX=coarse_sigma,
            sigmaY=coarse_sigma,
        )
        transition[depth] = (1.0 - coarse_weight) * fine + coarse_weight * coarse

    depth_index = transition.argmax(axis=0).astype(np.uint8)
    partitioned = np.partition(transition, -2, axis=0)
    peak = partitioned[-1]
    margin = partitioned[-1] - partitioned[-2]
    valid = (contrast >= min_contrast) & (peak >= min_peak) & (margin >= min_margin)
    confidence = np.clip(peak, 0.0, 1.0)
    return depth_index, confidence, valid


def _write_overlays(
    volume,
    depth_map: np.ndarray,
    confidence: np.ndarray,
    z_start: int,
    z_end: int,
    output_dir: Path,
    max_side: int,
) -> None:
    """write one downscaled red surface overlay for every input depth layer."""
    height, width = depth_map.shape
    scale = min(1.0, float(max_side) / max(height, width))
    out_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    valid = confidence > 0

    for depth in range(z_start, z_end):
        layer = np.asarray(volume[depth], dtype=np.uint8)
        gray = cv2.resize(layer, out_size, interpolation=cv2.INTER_AREA)
        selected = ((depth_map == depth) & valid).astype(np.uint8)
        selected = cv2.resize(selected, out_size, interpolation=cv2.INTER_NEAREST) > 0
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rgb[selected] = (
            0.20 * rgb[selected].astype(np.float32)
            + 0.80 * np.array([0.0, 0.0, 255.0], dtype=np.float32)
        ).astype(np.uint8)
        cv2.putText(
            rgb,
            f"depth {depth:02d}  red=predicted papyrus-to-air surface",
            (24, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            rgb,
            f"valid surface pixels: {int(selected.sum()):,}",
            (24, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / f"depth_{depth:02d}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])


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
            small_lookup[1:] = stats[1:, cv2.CC_STAT_AREA] <= 4
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


def _write_depth_overview(
    depth_map: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    z_start: int,
    z_end: int,
    output_path: Path,
    max_side: int = 4000,
) -> None:
    """write one full-resolution grayscale map encoding all predicted depths."""
    valid = confidence > 0
    scale = 255.0 / max(1, z_end - z_start)
    gray = np.clip((depth_map.astype(np.float32) - z_start) * scale, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb[~valid] = (80, 0, 80)
    resize_scale = min(1.0, float(max_side) / max(rgb.shape[:2]))
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
    parser.add_argument("--max-review-side", type=int, default=2200)
    args = parser.parse_args()

    scroll_id = str(args.scroll_id)
    volume = zarr.open(os.path.join(args.zarr_dir, f"{scroll_id}.zarr"), mode="r")
    depth, height, width = map(int, volume.shape)
    if not (0 <= args.z_start < args.z_end <= depth):
        raise ValueError(f"invalid depth window [{args.z_start}, {args.z_end}) for depth {depth}")

    mask = cv2.imread(os.path.join(args.mask_dir, f"{scroll_id}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (height, width):
        raise ValueError(f"missing or mismatched scroll mask for {scroll_id}")
    mask = mask > 0

    output_dir = Path(args.output_dir)
    review_dir = Path(args.review_dir) / scroll_id
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    depth_path = output_dir / f"{scroll_id}_depth.npy"
    confidence_path = output_dir / f"{scroll_id}_confidence.npy"
    depth_map = np.lib.format.open_memmap(depth_path, mode="w+", dtype=np.uint8, shape=(height, width))
    confidence = np.lib.format.open_memmap(
        confidence_path,
        mode="w+",
        dtype=np.uint8,
        shape=(height, width),
    )
    depth_map[:] = 255
    confidence[:] = 0

    for y0 in range(0, height, args.row_block):
        y1 = min(height, y0 + args.row_block)
        ys = max(0, y0 - args.halo)
        ye = min(height, y1 + args.halo)
        raw = np.asarray(volume[args.z_start:args.z_end, ys:ye, :])
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

    depth_map.flush()
    confidence.flush()
    regularization = _regularize_surface_map(depth_map, confidence, mask)
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
        "histogram": histogram,
    }
    with open(output_dir / f"{scroll_id}_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    _write_overlays(
        volume,
        depth_map,
        confidence,
        args.z_start,
        args.z_end,
        review_dir,
        args.max_review_side,
    )
    _write_depth_overview(
        depth_map,
        confidence,
        mask,
        args.z_start,
        args.z_end,
        review_dir / "surface_depth_overview.jpg",
    )
    print(f"[surface] depth labels -> {depth_path}")
    print(f"[surface] confidence -> {confidence_path}")
    print(f"[surface] 24 review images -> {review_dir}")
    print(f"[surface] depth overview -> {review_dir / 'surface_depth_overview.jpg'}")
    print(f"[surface] valid inside mask: {100.0 * metadata['valid_fraction_inside_mask']:.2f}%")


if __name__ == "__main__":
    main()
