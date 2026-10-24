"""fast full-resolution figure output: pixel-exact panel grids composed with numpy/OpenCV instead of matplotlib.

matplotlib rasterizes a ~150 Mpx grid twice (draw + savefig) at ~5 s each; composing the same layout directly
and encoding once takes ~1-2 s. layout constants match the notebooks' former matplotlib _save_figure/save_grid.
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
# matplotlib/agg and jpeg both cap each image dimension below 2**16 px
MAX_FIGURE_PX = 65000
FIGURE_DPI = 100
RED = (255, 0, 0)
# savefig(pil_kwargs=FAST_PNG): zlib level 1 instead of 6, same pixels, ~20% faster on huge figures
FAST_PNG = {"compress_level": 1}

DISPLAY_ROTATIONS = {
    "20260927134931": 2, "20260927134928": 2, "auto_grown_20260927134928530_abf": 2,
    "20261012073246": 1, "20261012073244": 1, "auto_grown_20261012073244544_abf": 1,
    "20261012073250": 3, "20261012073249": 3, "auto_grown_20261012073244544_abf_p29": 3,
}


def display_rotation(scroll_id):
    """Counterclockwise quarter-turns for presentation only, including source-volume aliases."""
    return DISPLAY_ROTATIONS.get(str(scroll_id), 0)


def orient_display(image, scroll_id):
    """Rotate spatial image axes without changing source data or prediction values."""
    if image is None:
        return None
    return np.ascontiguousarray(np.rot90(image, display_rotation(scroll_id), axes=(0, 1)))


@lru_cache(maxsize=32)
def _font(size_px: int):
    import matplotlib.font_manager as font_manager

    try:
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), max(1, int(size_px)))
    except OSError:
        return ImageFont.load_default()


def colorize(values, cmap="gray_r", bad_rgb=(0.78, 0.86, 0.95), vmin=0.0, vmax=1.0):
    """float map (NaN = invalid) -> uint8 RGB at the map's own resolution through a 256-entry colormap LUT."""
    import matplotlib

    lut = (matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8)
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    index = np.clip((np.nan_to_num(values, nan=vmin) - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgb = lut[np.rint(index * 255).astype(np.uint8)]
    rgb[~valid] = np.round(np.asarray(bad_rgb) * 255).astype(np.uint8)
    return rgb


def gray_rgb(image):
    """uint8 grayscale (e.g. a composite) -> uint8 RGB."""
    image = np.asarray(image, dtype=np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


def red_overlay(rgb, label, alpha=0.4):
    """uint8 RGB with the boolean/uint8 `label` (any resolution) blended in red; edge pixels blend by label coverage."""
    height, width = rgb.shape[:2]
    coverage = (np.asarray(label) > 0).astype(np.float32)
    if coverage.shape != (height, width):
        coverage = cv2.resize(coverage, (width, height), interpolation=cv2.INTER_AREA)
    weight = (alpha * coverage)[..., None]
    out = rgb.astype(np.float32) * (1 - weight) + np.asarray(RED, dtype=np.float32) * weight
    return out.round().astype(np.uint8)


def crop_panel_pair(path, cell_hw, gap_px=24):
    """(left, right) RGB panels of a one-row, two-column save_panel_grid figure, found from its cell size."""
    cell_height, cell_width = cell_hw
    gap = max(gap_px, cell_width // 40)
    title_h = max(60, cell_height // 14)
    fig_w = 2 * cell_width + 3 * gap
    fig_h = title_h + cell_height + 2 * gap
    scale = min(1.0, MAX_FIGURE_PX / fig_w, MAX_FIGURE_PX / fig_h)
    panel_w, panel_h = max(1, int(cell_width * scale)), max(1, int(cell_height * scale))
    top = int((gap + title_h) * scale)
    with Image.open(path) as figure:
        if figure.size != (max(1, int(fig_w * scale)), max(1, int(fig_h * scale))):
            raise ValueError(f"{path} is {figure.size}, not the layout expected for cells {cell_hw}")
        image = np.asarray(figure.convert("RGB"))
    return tuple(image[top:top + panel_h, int((gap + c * (cell_width + gap)) * scale):][:, :panel_w].copy()
                 for c in range(2))


def _text(canvas, xy, text, size_px, fill, anchor):
    """draw text in place, touching only the patch it covers (the canvas can be ~500 MB)"""
    font = _font(size_px)
    left, top, right, bottom = font.getbbox(text, anchor=anchor)
    margin = max(2, size_px // 4)
    x0 = max(0, int(xy[0] + left) - margin)
    y0 = max(0, int(xy[1] + top) - margin)
    x1 = min(canvas.shape[1], int(xy[0] + right) + margin)
    y1 = min(canvas.shape[0], int(xy[1] + bottom) + margin)
    if x0 >= x1 or y0 >= y1:
        return
    patch = Image.fromarray(np.ascontiguousarray(canvas[y0:y1, x0:x1]))
    ImageDraw.Draw(patch).text((xy[0] - x0, xy[1] - y0), text, fill=fill, font=font, anchor=anchor)
    canvas[y0:y1, x0:x1] = np.asarray(patch)


def _scale_bar(panel, px_per_cm, font_px):
    """red 1 cm bar in the panel's bottom-right corner, drawn in place (matplotlib layout of the old helper)."""
    height, width = panel.shape[:2]
    thickness = max(2.0, min(height, width) / 250)
    pad = 0.05 * min(height, width) + thickness
    t = max(1, int(round(thickness)))
    if width >= height:
        x1, y = int(width - pad), int(height - pad)
        x0 = int(max(0.0, x1 - px_per_cm))
        cv2.rectangle(panel, (x0, y - t), (x1, y), RED, -1)
        _text(panel, (x0, int(y - 3 * thickness)), "1 cm", font_px, RED, "lb")
    else:
        y1, x = int(height - pad), int(width - pad)
        y0 = int(max(0.0, y1 - px_per_cm))
        cv2.rectangle(panel, (x - t, y0), (x, y1), RED, -1)
        _text(panel, (x, int(y0 - 2 * thickness)), "1 cm", font_px, RED, "rb")


def _dotted_outline(panel, grid):
    """red dotted boundary of a boolean grid (any resolution) drawn in place, like ax.contour(levels=[0.5], ':')."""
    height, width = panel.shape[:2]
    up = cv2.resize(np.asarray(grid, dtype=np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    # matplotlib drew max(1, width / 500) points
    line = max(1, int(round(max(1.0, width / 500) * FIGURE_DPI / 72)))
    edge = cv2.morphologyEx(up, cv2.MORPH_GRADIENT, np.ones((line, line), np.uint8)) > 0
    yy, xx = np.indices(edge.shape)
    period = 4 * line
    panel[edge & (((yy + xx) // period) % 2 == 0)] = RED


def save_panel_grid(rows, cell_hw, out_path, gap_px=24, quality=92):
    """save a grid of panels at full resolution without displaying it.

    rows: list of rows; each panel is (title, rgb uint8 image of any size, interpolation, um_per_px or None
    [, boolean outline grid]). every panel is resized to the cell (cell_hw at full resolution, scaled down only when
    the whole figure would exceed MAX_FIGURE_PX); um_per_px adds a red 1 cm bar, an outline grid a dotted boundary.
    layout and title sizes match the former matplotlib grid.
    """
    cell_height, cell_width = cell_hw
    n_rows, n_cols = len(rows), max(len(row) for row in rows)
    gap = max(gap_px, cell_width // 40)
    title_h = max(60, cell_height // 14)
    fig_w = n_cols * cell_width + (n_cols + 1) * gap
    fig_h = n_rows * (title_h + cell_height) + (n_rows + 1) * gap
    scale = min(1.0, MAX_FIGURE_PX / fig_w, MAX_FIGURE_PX / fig_h)
    out_w, out_h = max(1, int(fig_w * scale)), max(1, int(fig_h * scale))
    panel_w, panel_h = max(1, int(cell_width * scale)), max(1, int(cell_height * scale))
    # matplotlib title size was title_h * scale * 0.45 points at FIGURE_DPI -> same height in pixels
    font_px = max(1, int(round(title_h * scale * 0.45)))
    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    for r, row in enumerate(rows):
        top = int((gap + r * (title_h + cell_height + gap) + title_h) * scale)
        for c, (title, image, interpolation, um_per_px, *outline) in enumerate(row):
            left = int((gap + c * (cell_width + gap)) * scale)
            panel = cv2.resize(np.ascontiguousarray(image), (panel_w, panel_h), interpolation=interpolation)
            if outline and outline[0] is not None and outline[0].any() and not outline[0].all():
                _dotted_outline(panel, outline[0])
            if um_per_px:
                _scale_bar(panel, 10000.0 / um_per_px * panel_w / cell_width, font_px)
            canvas[top:top + panel_h, left:left + panel_w] = panel
            title_pad = int(round(font_px * 0.4))
            title_rgb = (0, 0, 0)
            if isinstance(title, tuple):  # (text, rgb)
                title, title_rgb = title
            _text(canvas, (left + panel_w // 2, top - title_pad), str(title), font_px, title_rgb, "mb")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if str(out_path).lower().endswith((".jpg", ".jpeg")):
        Image.fromarray(canvas).save(out_path, quality=quality)
    else:
        # fast lossless PNG: zlib level 1 instead of the slow default
        cv2.imwrite(str(out_path), canvas[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return out_path
