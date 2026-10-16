"""PHerc0175A quick patches (tifxyz/quick_patches_from_D/PHerc0175A) under numeric, flat depth-13 aliases.

training needs integer scroll ids, so every rendered patch <name> gets the alias <timestamp>[+variant] and is read like
the PHerc0846A aliases: <alias>.zarr and masks/<alias>.png link to the render, surface_labels/<alias>/ is a constant
depth 13 over the mask, and the surface window offset is -1 (slices 9-16). the patches have no ink labels; a black
dilated_inklabels/<alias>.png and a white train_masks/<alias>.png keep loaders that insist on those files working.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MESH_DIR = REPO / "tifxyz" / "quick_patches_from_D" / "PHerc0175A"
FLAT_DEPTH = 13
WINDOW_OFFSET = -1
_VARIANT = {"": 0, "a": 1, "b": 2, "c": 3}


def alias_of(name: str) -> int:
    """auto_grown_20261010071652255_a_abf -> 20261010071653 (14-digit timestamp plus the a/b/c variant)."""
    match = re.fullmatch(r"auto_grown_(\d{14})\d*(?:_([abc]))?_abf", name)
    if match is None:
        raise ValueError(f"unexpected 0175A segment name {name!r}")
    return int(match.group(1)) + _VARIANT[match.group(2) or ""]


def segment_names() -> list[str]:
    """every 0175A mesh whose render and mask exist, sorted."""
    names = sorted(path.name for path in MESH_DIR.iterdir() if path.is_dir())
    return [n for n in names if (REPO / "ves_zarrs2" / f"{n}.zarr").is_dir() and (REPO / "masks" / "extras" / f"{n}.png").is_file()]


def aliases() -> dict[str, int]:
    out = {name: alias_of(name) for name in segment_names()}
    if len(set(out.values())) != len(out):
        raise ValueError("0175A aliases collide")
    return out


def alias_ids() -> list[int]:
    return list(aliases().values())


def _link(link: Path, target: str) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) != target:
            raise FileExistsError(f"{link} already points to {os.readlink(link)}, expected {target}")
        return
    link.symlink_to(target)


def prepare() -> dict[str, int]:
    """create or verify every alias artifact (links, flat surface, black label, norm and anchor cache entries)."""
    from utils.norm import compute_norm, ensure_surface_anchors, load_cached_norm

    zarr_dir = REPO / "ves_zarrs2"
    norm_cache = REPO / "norm_cache.json"
    done = aliases()
    for name, alias in done.items():
        alias_zarr = zarr_dir / f"{alias}.zarr"
        if alias_zarr.exists() and not alias_zarr.is_symlink():
            raise FileExistsError(f"{alias_zarr} is a real directory, not the alias of {name}")
        _link(alias_zarr, f"{name}.zarr")
        _link(REPO / "masks" / f"{alias}.png", f"extras/{name}.png")
        mask = cv2.imread(str(REPO / "masks" / f"{alias}.png"), cv2.IMREAD_GRAYSCALE) > 0
        label = REPO / "dilated_inklabels" / f"{alias}.png"
        if not label.exists():
            cv2.imwrite(str(label), np.zeros(mask.shape, np.uint8))
        train_mask = REPO / "train_masks" / f"{alias}.png"
        if not train_mask.exists():
            cv2.imwrite(str(train_mask), np.full(mask.shape, 255, np.uint8))
        surface = REPO / "surface_labels" / str(alias)
        depth_path = surface / "depth.npy"
        if not depth_path.is_file() or np.load(depth_path, mmap_mode="r").shape != mask.shape:
            surface.mkdir(parents=True, exist_ok=True)
            np.save(depth_path, np.where(mask, FLAT_DEPTH, 255).astype(np.uint8))
            np.save(surface / "confidence.npy", np.where(mask, 255, 0).astype(np.uint8))
            (surface / "metadata.json").write_text(json.dumps(
                {"scroll_id": alias, "flat_from": name, "flat_depth": FLAT_DEPTH, "invalid_depth_value": 255}, indent=2))
        if load_cached_norm(alias, str(norm_cache)) is None:
            cached = json.loads(norm_cache.read_text()).get(name)
            if isinstance(cached, dict) and all(k in cached for k in ("mean", "std", "min", "max")):
                data = json.loads(norm_cache.read_text())
                data[str(alias)] = {k: cached[k] for k in ("mean", "std", "min", "max")}
                temporary = norm_cache.with_suffix(".tmp")
                temporary.write_text(json.dumps(data, indent=4))
                temporary.replace(norm_cache)
            else:
                compute_norm(alias, str(zarr_dir), cache_path=str(norm_cache))
    ensure_surface_anchors(list(done.values()), str(zarr_dir))
    return done


if __name__ == "__main__":
    os.chdir(REPO)
    print(json.dumps(prepare(), indent=1))
