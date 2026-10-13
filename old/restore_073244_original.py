"""Restore the committed-mesh base while retaining the rebuilt offset and separate label sources."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BACKUP = ROOT / "_ves_tmp" / "rebuild_073244_pair" / "previous"
WORK = ROOT / "_ves_tmp" / "restore_073244_original"
MAIN, FLAT_MAIN, OFFSET, FLAT_OFFSET = 20261012073244, 20261012073246, 20261012073249, 20261012073250
MESH_NAME = "auto_grown_20261012073244544_abf"


def preserve(path, destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if path.exists() or path.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))


def fingerprint(folder):
    return {str(path.relative_to(folder)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in folder.rglob("*") if path.is_file()}


def main():
    if (WORK / "complete.json").exists():
        raise FileExistsError("Restore already completed")
    WORK.mkdir(parents=True, exist_ok=True)
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    tracked = ROOT / "tifxyz" / "3_new" / MESH_NAME
    restored_mesh = WORK / "committed_mesh"
    restored_mesh.mkdir(exist_ok=True)
    for filename in ("x.tif", "y.tif", "z.tif", "meta.json"):
        relative = (tracked / filename).relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        materialized = (tracked / filename).read_bytes()
        if committed.startswith(b"version https://git-lfs.github.com/spec/v1"):
            digest = next(line.split("sha256:")[1] for line in committed.decode().splitlines() if line.startswith("oid "))
            if hashlib.sha256(materialized).hexdigest() != digest:
                raise ValueError(f"Working mesh differs from committed LFS data: {relative}")
        elif committed != materialized:
            raise ValueError(f"Working mesh differs from git HEAD: {relative}")
        shutil.copyfile(tracked / filename, restored_mesh / filename)
    retained = {str(sid): fingerprint(ROOT / "surface_labels" / str(sid)) for sid in (OFFSET, FLAT_OFFSET)}
    retained_volume = fingerprint(zarr_root / f"{OFFSET}.zarr")
    retained_mask = hashlib.sha256((ROOT / "masks" / f"{OFFSET}.png").read_bytes()).hexdigest()
    caches = {filename: json.loads((ROOT / filename).read_text())
              for filename in ("norm_cache.json", "surface_anchor_cache.json")}
    old_shape = tuple(zarr.open(str(BACKUP / "volumes" / f"{MAIN}.zarr"), mode="r").shape[1:])
    offset_shape = tuple(zarr.open(str(zarr_root / f"{OFFSET}.zarr"), mode="r").shape[1:])
    staged_labels = {}
    for folder in ("dilated_inklabels", "train_masks"):
        base = cv2.imread(str(ROOT / folder / f"{MAIN}.png"), 0)
        offset = cv2.imread(str(BACKUP / folder / f"{OFFSET}.png"), 0)
        if base is None or offset is None:
            raise ValueError(f"Missing separate label source in {folder}")
        if base.shape == old_shape[::-1]:
            base = np.ascontiguousarray(np.rot90(base, 3))
        if offset.shape == offset_shape[::-1]:
            offset = np.ascontiguousarray(np.rot90(offset, 1))
        if base.shape != old_shape or offset.shape != offset_shape:
            raise ValueError(f"Label alignment failed for {folder}")
        staged_labels[folder] = (base, offset)
    edited = ROOT / "tifxyz" / MESH_NAME
    offset_source = ROOT / "tifxyz" / f"{MESH_NAME}_offset_source"
    preserve(edited, offset_source)
    shutil.copytree(restored_mesh, edited)
    preserve(zarr_root / f"{MAIN}.zarr", WORK / "replaced" / "volumes" / f"{MAIN}.zarr")
    shutil.copytree(BACKUP / "volumes" / f"{MAIN}.zarr", zarr_root / f"{MAIN}.zarr")
    preserve(ROOT / "masks" / f"{MAIN}.png", WORK / "replaced" / "masks" / f"{MAIN}.png")
    shutil.copyfile(BACKUP / "masks" / f"{MAIN}.png", ROOT / "masks" / f"{MAIN}.png")
    for sid in (MAIN, FLAT_MAIN):
        preserve(ROOT / "surface_labels" / str(sid), WORK / "replaced" / "surface_labels" / str(sid))
        shutil.copytree(BACKUP / "surface_labels" / str(sid), ROOT / "surface_labels" / str(sid))
    for folder, (base, offset) in staged_labels.items():
        for sid, image in ((MAIN, base), (OFFSET, offset)):
            path = ROOT / folder / f"{sid}.png"
            preserve(path, WORK / "replaced" / folder / path.name)
            if not cv2.imwrite(str(path), image):
                raise OSError(path)
    for filename, cache in caches.items():
        previous = json.loads((BACKUP / filename).read_text())
        for sid in (MAIN, FLAT_MAIN):
            cache[str(sid)] = previous[str(sid)]
        temporary = ROOT / f"{filename}.restore.tmp"
        temporary.write_text(json.dumps(cache, indent=2) + "\n")
        temporary.replace(ROOT / filename)
    for sid in (MAIN, FLAT_MAIN):
        for path in (ROOT / "output" / "composite_cache").glob(f"{sid}_*.png"):
            preserve(path, WORK / "replaced" / "composites" / path.name)
    for sid in (OFFSET, FLAT_OFFSET):
        if fingerprint(ROOT / "surface_labels" / str(sid)) != retained[str(sid)]:
            raise ValueError("Offset surface artifacts changed")
    if fingerprint(zarr_root / f"{OFFSET}.zarr") != retained_volume:
        raise ValueError("Offset volume changed")
    if hashlib.sha256((ROOT / "masks" / f"{OFFSET}.png").read_bytes()).hexdigest() != retained_mask:
        raise ValueError("Offset mask changed")
    for sid in (MAIN, FLAT_MAIN, OFFSET, FLAT_OFFSET):
        shape = tuple(zarr.open(str(zarr_root / f"{sid}.zarr"), mode="r").shape[1:])
        if cv2.imread(str(ROOT / "dilated_inklabels" / f"{sid}.png"), 0).shape != shape:
            raise ValueError(f"Final label/volume mismatch: {sid}")
    if (ROOT / "dilated_inklabels" / f"{OFFSET}.png").is_symlink():
        raise ValueError("Offset must retain its own independent label file")
    result = {"complete": True, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
              "base_mesh": str(edited), "offset_source_mesh": str(offset_source),
              "base_shape": list(old_shape), "offset_shape": list(offset_shape),
              "offset_volume_and_surfaces_unchanged": True, "separate_label_sources": [MAIN, OFFSET]}
    (WORK / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[restore] complete", json.dumps(result), flush=True)


def restore_offset():
    work = ROOT / "_ves_tmp" / "restore_073249_original"
    if work.exists():
        raise FileExistsError(work)
    mesh = ROOT / "tifxyz" / "3_new_offset" / f"{MESH_NAME}_p29"
    for filename in ("x.tif", "y.tif", "z.tif", "meta.json"):
        relative = (mesh / filename).relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        saved = (BACKUP / "offset_mesh" / filename).read_bytes()
        if committed.startswith(b"version https://git-lfs.github.com/spec/v1"):
            matches = f"oid sha256:{hashlib.sha256(saved).hexdigest()}" in committed.decode()
        else:
            matches = committed == saved
        if not matches:
            raise ValueError(f"Backup mesh differs from git HEAD: {relative}")
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    base_before = fingerprint(zarr_root / f"{MAIN}.zarr")
    shape = tuple(zarr.open(str(BACKUP / "volumes" / f"{OFFSET}.zarr"), mode="r").shape[1:])
    staged = {}
    for folder in ("dilated_inklabels", "train_masks"):
        image = cv2.imread(str(ROOT / folder / f"{OFFSET}.png"), 0)
        if image is None:
            raise ValueError(f"Missing independent offset source: {folder}")
        if image.shape == shape[::-1]:
            image = np.ascontiguousarray(np.rot90(image, 3))
        if image.shape != shape:
            raise ValueError(f"Offset label frame mismatch: {folder}")
        staged[folder] = image
    work.mkdir(parents=True)
    preserve(mesh, work / "replaced" / "offset_mesh")
    shutil.copytree(BACKUP / "offset_mesh", mesh)
    volume = zarr_root / f"{OFFSET}.zarr"
    preserve(volume, work / "replaced" / "volumes" / volume.name)
    shutil.copytree(BACKUP / "volumes" / volume.name, volume)
    mask = ROOT / "masks" / f"{OFFSET}.png"
    preserve(mask, work / "replaced" / "masks" / mask.name)
    shutil.copyfile(BACKUP / "masks" / mask.name, mask)
    for sid in (OFFSET, FLAT_OFFSET):
        surface = ROOT / "surface_labels" / str(sid)
        preserve(surface, work / "replaced" / "surface_labels" / str(sid))
        shutil.copytree(BACKUP / "surface_labels" / str(sid), surface)
    for folder, image in staged.items():
        path = ROOT / folder / f"{OFFSET}.png"
        preserve(path, work / "replaced" / folder / path.name)
        if not cv2.imwrite(str(path), image):
            raise OSError(path)
    for filename in ("norm_cache.json", "surface_anchor_cache.json"):
        path = ROOT / filename
        shutil.copyfile(path, work / filename)
        cache = json.loads(path.read_text())
        previous = json.loads((BACKUP / filename).read_text())
        for sid in (OFFSET, FLAT_OFFSET):
            cache[str(sid)] = previous[str(sid)]
        temporary = path.with_suffix(".restore.tmp")
        temporary.write_text(json.dumps(cache, indent=2) + "\n")
        temporary.replace(path)
    for sid in (OFFSET, FLAT_OFFSET):
        for path in (ROOT / "output" / "composite_cache").glob(f"{sid}_*.png"):
            preserve(path, work / "replaced" / "composites" / path.name)
        assert tuple(zarr.open(str(zarr_root / f"{sid}.zarr"), mode="r").shape[1:]) == shape
        for folder in ("masks", "dilated_inklabels", "train_masks"):
            assert cv2.imread(str(ROOT / folder / f"{sid}.png"), 0).shape == shape
        for filename in ("depth.npy", "confidence.npy"):
            assert np.load(ROOT / "surface_labels" / str(sid) / filename).shape == shape
    assert fingerprint(zarr_root / f"{MAIN}.zarr") == base_before
    result = {"complete": True, "mesh_matches_git": True, "shape": list(shape),
              "base_volume_unchanged": True, "separate_label_source": OFFSET}
    (work / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[restore-offset]", json.dumps(result), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", action="store_true")
    arguments = parser.parse_args()
    restore_offset() if arguments.offset else main()