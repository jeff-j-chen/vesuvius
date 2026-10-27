"""discard the parts of the trained PHerc0846A sheets that come within RADIUS_VOX of the straightened2 prize sheet.

the neighbour sheets were traced next to straightened2 and in places run onto it (in_1 / out_1 share papyrus with it), so
their renders and labels would leak the prize sheet into training. For every trained 0846A sheet this measures, at each
mesh point, the 3D distance to the prize-sheet mesh and writes holdout_exclusion/<alias>.png in the render frame
(255 = discard). train.py, finetune_readerv3.py and finetune_v8in.py drop supervision there; figures go to
output/holdout_exclusion/.

  python3 -m utils.holdout_exclusion [--force]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXCLUSION_DIR = REPO / "holdout_exclusion"
FIGURE_DIR = REPO / "output" / "holdout_exclusion"
RADIUS_VOX = 14.0
HOLDOUT_DENSE = 8   # prize-sheet mesh sampled every 20/8 = 2.5 voxels
SHEET_DENSE = 4     # trained-sheet mesh sampled every 5 voxels (= render pixels)
GRID_PX = 20        # render pixels per tifxyz grid cell (scale 0.05, 9.362 um volume)
VERSION = 1


def _mesh(name: str, factor: int):
    """dense mesh points of a 0846A sheet on its render's cropped grid: points (H,W,3), valid (H,W), render shape."""
    from assemble_test_segments import ROTATE_180, load_mesh
    from assemble_training_segments import MESHES_0846A

    render, _alias, mesh = MESHES_0846A[name]
    X, Y, Z, valid = load_mesh(str(REPO / "tifxyz" / mesh))
    if render in ROTATE_180:
        X, Y, Z, valid = (np.ascontiguousarray(grid[::-1, ::-1]) for grid in (X, Y, Z, valid))
    # the renderer's crop: valid bounding box + 8 cells
    ys, xs = np.where(valid)
    y0, x0 = max(0, ys.min() - 8), max(0, xs.min() - 8)
    y1, x1 = min(valid.shape[0], ys.max() + 9), min(valid.shape[1], xs.max() + 9)
    X, Y, Z, valid = (grid[y0:y1, x0:x1] for grid in (X, Y, Z, valid))
    gh, gw = X.shape
    shape = ((gh - 1) * GRID_PX + 1, (gw - 1) * GRID_PX + 1)
    height, width = (gh - 1) * factor + 1, (gw - 1) * factor + 1
    # corner-aligned bilinear upsampling, so dense index i sits exactly at grid row i / factor
    rows, cols = np.linspace(0, gh - 1, height), np.linspace(0, gw - 1, width)
    r0 = np.minimum(rows.astype(int), gh - 2)
    c0 = np.minimum(cols.astype(int), gw - 2)
    fr, fc = (rows - r0)[:, None], (cols - c0)[None, :]

    def up(grid):
        grid = grid.astype(np.float64)
        top = grid[r0][:, c0] * (1 - fc) + grid[r0][:, c0 + 1] * fc
        bottom = grid[r0 + 1][:, c0] * (1 - fc) + grid[r0 + 1][:, c0 + 1] * fc
        return top * (1 - fr) + bottom * fr

    points = np.stack([up(X), up(Y), up(Z)], axis=-1)
    dense_valid = up(valid.astype(np.float32)) > 0.999
    return points, dense_valid, shape


def compute(name: str, holdout_tree) -> tuple[np.ndarray, np.ndarray]:
    """(distance to the prize sheet in voxels on the render frame, exclusion mask on the render frame)."""
    points, valid, shape = _mesh(name, SHEET_DENSE)
    distance = np.full(valid.shape, np.inf)
    nearest, _ = holdout_tree.query(points[valid], distance_upper_bound=200.0)
    distance[valid] = nearest
    # nearest prize-sheet sample overestimates the surface distance by at most half its 2.5 voxel spacing
    excluded = distance <= RADIUS_VOX + 0.5 * GRID_PX / HOLDOUT_DENSE
    # one dense step of margin (5 px) covers the nearest-neighbour mapping to render pixels
    excluded = cv2.dilate(excluded.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    step = GRID_PX // SHEET_DENSE
    rows = np.minimum(np.rint(np.arange(shape[0]) / step).astype(int), valid.shape[0] - 1)
    cols = np.minimum(np.rint(np.arange(shape[1]) / step).astype(int), valid.shape[1] - 1)
    distance_px = distance[rows][:, cols].astype(np.float32)
    excluded_px = excluded[rows][:, cols]
    return distance_px, excluded_px


def exclusion_path(alias) -> Path:
    return EXCLUSION_DIR / f"{int(alias)}.png"


def load(alias, shape=None) -> np.ndarray | None:
    """bool discard mask of a trained alias, or None when it has none; checked against the frame shape."""
    path = exclusion_path(alias)
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    if shape is not None and tuple(mask.shape) != tuple(shape):
        raise ValueError(f"{path}: shape {mask.shape} differs from the frame {tuple(shape)}")
    return mask


def expected_aliases() -> list[int]:
    from utils import mix6

    return list(mix6.TRAIN_0846A)


def missing() -> list[str]:
    return [str(exclusion_path(alias)) for alias in expected_aliases() if not exclusion_path(alias).is_file()]


def _figure(alias, name, render, distance, excluded, papyrus, ink, summary_row) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.fast_figure import orient_display

    def orient(image):
        return orient_display(image, render)

    rgb = np.full(papyrus.shape + (3,), 255, np.uint8)
    rgb[papyrus] = (205, 205, 205)
    rgb[papyrus & ink] = (40, 40, 40)
    rgb[papyrus & excluded] = (240, 120, 120)
    rgb[papyrus & excluded & ink] = (150, 0, 0)
    near = np.where(papyrus, np.minimum(distance, 60.0), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    axes[0].imshow(orient(rgb), interpolation="nearest")
    axes[0].set_title("kept papyrus (grey, ink black) / DISCARDED within "
                      f"{RADIUS_VOX:.0f} vox of the prize sheet (red, ink dark red)")
    image = axes[1].imshow(orient(near), cmap="magma", vmin=0, vmax=60, interpolation="nearest")
    axes[1].contour(orient(np.nan_to_num(near, nan=999.0)), levels=[RADIUS_VOX], colors="cyan", linewidths=1.0)
    axes[1].set_title(f"3D distance to the prize-sheet mesh (voxels, capped at 60); cyan = {RADIUS_VOX:.0f}")
    fig.colorbar(image, ax=axes[1], shrink=0.7)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{name.replace('auto_grown_', '')} (alias {alias}): discarded {summary_row['discarded_cm2']:.2f} of "
        f"{summary_row['papyrus_cm2']:.2f} cm2 ({100 * summary_row['discarded_fraction']:.0f}%), ink "
        f"{summary_row['discarded_ink_cm2']:.3f} of {summary_row['ink_cm2']:.3f} cm2", fontsize=14)
    fig.savefig(FIGURE_DIR / f"{alias}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def _overview(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    fig, axes = plt.subplots(2, 4, figsize=(28, 13))
    for ax, alias in zip(axes.flat, names):
        ax.imshow(cv2.cvtColor(cv2.imread(str(FIGURE_DIR / f"{alias}.png")), cv2.COLOR_BGR2RGB))
        ax.axis("off")
    table = axes.flat[-1]
    table.axis("off")
    rows = [["sheet", "discarded cm2", "%", "ink discarded cm2", "ink %"]]
    for alias, row in summary.items():
        rows.append([row["name"].replace("auto_grown_20261012065803387_", "").replace("auto_grown_", ""),
                     f"{row['discarded_cm2']:.2f}", f"{100 * row['discarded_fraction']:.0f}",
                     f"{row['discarded_ink_cm2']:.3f}", f"{100 * row['discarded_ink_fraction']:.0f}"])
    cells = table.table(cellText=rows, loc="center", cellLoc="left")
    cells.auto_set_font_size(False)
    cells.set_fontsize(10)
    cells.scale(1, 2.0)
    fig.suptitle(f"trained PHerc0846A sheets: supervision discarded within {RADIUS_VOX:.0f} voxels of the straightened2 prize sheet",
                 fontsize=16)
    fig.savefig(FIGURE_DIR / "overview.png", dpi=70, bbox_inches="tight")
    plt.close(fig)


def build(force: bool = False) -> dict:
    from scipy.spatial import cKDTree

    from utils import mix6
    from utils.dataloader import split_extra_labels

    EXCLUSION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EXCLUSION_DIR / "summary.json"
    if not force and not missing() and summary_path.is_file() \
            and json.loads(summary_path.read_text()).get("version") == VERSION:
        print(f"[exclusion] {EXCLUSION_DIR} is up to date")
        return json.loads(summary_path.read_text())["sheets"]
    holdout_points, holdout_valid, _ = _mesh(mix6.HOLDOUT_NAME, HOLDOUT_DENSE)
    tree = cKDTree(holdout_points[holdout_valid])
    pixel_cm2 = (9.362e-4) ** 2
    summary = {}
    for alias, (render, name) in mix6.SHEETS_0846A.items():
        distance, excluded = compute(name, tree)
        mask_path = REPO / "masks" / f"{render}.png"
        papyrus = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if papyrus is None:
            raise FileNotFoundError(f"{mask_path} (python3 assemble_training_segments.py --mix6)")
        if papyrus.shape != excluded.shape:
            raise ValueError(f"{name}: render mask {papyrus.shape} differs from the mesh frame {excluded.shape}")
        papyrus = papyrus > 0
        label = cv2.imread(str(REPO / "dilated_inklabels" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE)
        ink = split_extra_labels(label)[0] if label is not None else np.zeros_like(papyrus)
        cv2.imwrite(str(exclusion_path(alias)), (excluded & papyrus).astype(np.uint8) * 255)
        row = {
            "name": name, "render": int(render),
            "papyrus_cm2": float(papyrus.sum() * pixel_cm2),
            "discarded_cm2": float((papyrus & excluded).sum() * pixel_cm2),
            "ink_cm2": float((papyrus & ink).sum() * pixel_cm2),
            "discarded_ink_cm2": float((papyrus & ink & excluded).sum() * pixel_cm2),
            "min_distance_vox": float(np.min(distance[papyrus])) if papyrus.any() else None,
        }
        row["discarded_fraction"] = row["discarded_cm2"] / max(row["papyrus_cm2"], 1e-9)
        row["discarded_ink_fraction"] = row["discarded_ink_cm2"] / max(row["ink_cm2"], 1e-9)
        summary[str(alias)] = row
        _figure(alias, name, render, distance, excluded, papyrus, ink, row)
        print(f"[exclusion] {alias} {name}: discard {row['discarded_cm2']:.2f}/{row['papyrus_cm2']:.2f} cm2 "
              f"({100 * row['discarded_fraction']:.0f}%), ink {row['discarded_ink_cm2']:.3f}/{row['ink_cm2']:.3f} cm2, "
              f"min distance {row['min_distance_vox']:.1f} vox", flush=True)
    _overview(summary)
    summary_path.write_text(json.dumps({"version": VERSION, "radius_vox": RADIUS_VOX, "sheets": summary}, indent=1))
    print(f"[exclusion] masks in {EXCLUSION_DIR}, figures in {FIGURE_DIR}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    build(force=args.force)


if __name__ == "__main__":
    main()
