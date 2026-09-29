"""in-place depth reversal of rendered 28-layer zarrs whose mesh normal points the opposite way to training.

the renders sample symmetric offsets (k - 13.5) along the normal, so a reversed normal is exactly a depth
flip (layer k <-> D-1-k). the list lives in depth_reversed.json (tracked); assemble_test_segments.py renders
those ids with reversed offsets and flips any existing, unmarked zarr in place.

a corrected zarr carries attrs {"depth_flipped": true}, so nothing is ever flipped twice. the flip works one
chunk row at a time (~100 MB) with an on-disk backup of the current strip, so it resumes safely after a crash.

    python -m utils.depth_flip            # flip every listed zarr in $VESUVIUS_ZARR_PATH / ves_zarrs2
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parent.parent
LIST_PATH = REPO / "depth_reversed.json"


def reversed_ids() -> set[str]:
    return set(json.loads(LIST_PATH.read_text(encoding="utf-8"))["zids"])


def is_flipped(zarr_path) -> bool:
    return bool(zarr.open(str(zarr_path), mode="r").attrs.get("depth_flipped", False))


def flip_zarr_depth(zarr_path, backup_dir=REPO / "_ves_tmp") -> bool:
    """reverse the depth axis in place; returns False if the zarr was already flipped."""
    zarr_path = Path(zarr_path)
    array = zarr.open(str(zarr_path), mode="r+")
    if array.attrs.get("depth_flipped", False):
        return False
    _, height, _ = array.shape
    step = int(array.chunks[1])
    start = int(array.attrs.get("depth_flip_next_row", 0))
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    for y0 in range(start, height, step):
        y1 = min(height, y0 + step)
        backup = Path(backup_dir) / f"depth_flip_{zarr_path.stem}_{y0}.npy"
        if backup.exists():
            block = np.load(backup)          # original strip from an interrupted write
        else:
            block = np.asarray(array[:, y0:y1, :])
            partial = backup.with_suffix(".partial.npy")
            np.save(partial, block)
            os.replace(partial, backup)
        array[:, y0:y1, :] = block[::-1]
        array.attrs["depth_flip_next_row"] = y1
        backup.unlink()
    attrs = dict(array.attrs)
    attrs.pop("depth_flip_next_row", None)
    attrs["depth_flipped"] = True
    array.attrs.put(attrs)
    return True


def main() -> None:
    zarr_dir = Path(os.getenv("VESUVIUS_ZARR_PATH", REPO / "ves_zarrs2"))
    for zid in sorted(reversed_ids()):
        path = zarr_dir / f"{zid}.zarr"
        if not path.is_dir():
            print(f"{zid}: no zarr, skipped")
            continue
        print(f"{zid}: {'flipped' if flip_zarr_depth(path) else 'already flipped'}", flush=True)


if __name__ == "__main__":
    main()
