"""Rebuild the updated 073244 mesh and its +29 normal-offset partner, preserving old artifacts."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2
import numpy as np
import zarr

import assemble_test_segments as assembler
from utils.norm import compute_norm, ensure_surface_anchors

WORK = ROOT / "_ves_tmp" / "rebuild_073244_pair"
MAIN, OFFSET = 20261012073244, 20261012073249
FLAT = {MAIN: 20261012073246, OFFSET: 20261012073250}


def preserve(path, destination):
    if path.exists() or path.is_symlink():
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to replace backup: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))


def main():
    if (WORK / "complete.json").exists():
        raise FileExistsError("This rebuild already completed; preserve its history before repeating")
    volumes, masks, surfaces = (WORK / name for name in ("volumes", "masks", "surface_labels"))
    for folder in (volumes, masks, surfaces):
        folder.mkdir(parents=True, exist_ok=True)
    backup = WORK / "previous"
    backup.mkdir(exist_ok=True)
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    mesh = ROOT / "tifxyz" / "auto_grown_20261012073244544_abf"
    offset_mesh = WORK / "auto_grown_20261012073244544_abf_p29"
    subprocess.run([sys.executable, str(ROOT / "_ves_tmp" / "offset_mesh.py"), str(mesh), str(offset_mesh), "29"], check=True)
    original_valid = assembler.load_mesh(str(mesh))[3]
    offset_valid = assembler.load_mesh(str(offset_mesh))[3]
    if not np.array_equal(original_valid, offset_valid):
        raise ValueError("Offset mesh footprint differs from source mesh")
    fragment = next(entry for entry in assembler.FRAGMENTS if entry[0] == str(MAIN))
    for sid, source_mesh in ((MAIN, mesh), (OFFSET, offset_mesh)):
        assembler.render_surface_volume(
            mesh_dir=str(source_mesh), cache_dir=str(ROOT / "_ves_tmp" / f"render_{sid}"),
            vol_base=fragment[2], vol_shape=tuple(map(int, fragment[3].split(","))),
            layers=28, normal_step=1.0, upsample_factor=20.0, workers=32,
            out_zarr=str(volumes / f"{sid}.zarr"), out_id=str(sid),
            chunk_depth=8, chunk_y=64, chunk_x=64, mask_dir=str(masks), output_dtype="<u2",
        )
    mask_main = cv2.imread(str(masks / f"{MAIN}.png"), 0)
    mask_offset = cv2.imread(str(masks / f"{OFFSET}.png"), 0)
    if mask_main.shape != mask_offset.shape:
        raise ValueError("Paired render shapes differ")
    common = np.where((mask_main > 0) & (mask_offset > 0), 255, 0).astype(np.uint8)
    if not common.any():
        raise ValueError("Paired render has no common footprint")
    print(f"[pair] shared footprint {common.shape}, {int((common > 0).sum())} pixels", flush=True)
    for sid in (MAIN, OFFSET):
        if not cv2.imwrite(str(masks / f"{sid}.png"), common):
            raise OSError("Could not write common footprint")
        subprocess.run([
            sys.executable, str(ROOT / "generate_surface_supervision.py"), "--scroll-id", str(sid),
            "--zarr-dir", str(volumes), "--mask-dir", str(masks), "--output-dir", str(surfaces),
            "--review-dir", str(WORK / "surface_review"),
        ], check=True)
    label_transforms = {}
    for folder in ("dilated_inklabels", "train_masks", "inklabels"):
        source = ROOT / folder / f"{MAIN}.png"
        if not source.is_file():
            if folder == "inklabels":
                continue
            raise FileNotFoundError(source)
        image = cv2.imread(str(source), 0)
        if image.shape == common.shape[::-1]:
            aligned = np.ascontiguousarray(np.rot90(image))
            label_transforms[folder] = "90 degrees CCW"
        elif image.shape == common.shape:
            aligned = image
            label_transforms[folder] = "already in updated frame"
        else:
            raise ValueError(f"Cannot align {source}: {image.shape} vs {common.shape}")
        staged = WORK / folder
        staged.mkdir(exist_ok=True)
        if not cv2.imwrite(str(staged / f"{MAIN}.png"), aligned):
            raise OSError(source)
    for filename in ("norm_cache.json", "surface_anchor_cache.json"):
        destination = backup / filename
        if not destination.exists():
            shutil.copyfile(ROOT / filename, destination)
    for sid in (MAIN, OFFSET):
        preserve(zarr_root / f"{sid}.zarr", backup / "volumes" / f"{sid}.zarr")
        shutil.move(str(volumes / f"{sid}.zarr"), str(zarr_root / f"{sid}.zarr"))
        preserve(ROOT / "masks" / f"{sid}.png", backup / "masks" / f"{sid}.png")
        shutil.move(str(masks / f"{sid}.png"), str(ROOT / "masks" / f"{sid}.png"))
        for surface_id in (sid, FLAT[sid]):
            preserve(ROOT / "surface_labels" / str(surface_id), backup / "surface_labels" / str(surface_id))
        shutil.move(str(surfaces / str(sid)), str(ROOT / "surface_labels" / str(sid)))
        flat_dir = ROOT / "surface_labels" / str(FLAT[sid])
        flat_dir.mkdir(parents=True)
        np.save(flat_dir / "depth.npy", np.where(common > 0, 13, 255).astype(np.uint8))
        np.save(flat_dir / "confidence.npy", common)
        (flat_dir / "metadata.json").write_text(json.dumps({"flat_from": sid, "flat_depth": 13}))
        for filename in ("norm_cache.json", "surface_anchor_cache.json"):
            for key in (sid, FLAT[sid]):
                assembler._drop_json_key(str(ROOT / filename), str(key))
        for cache_root in (ROOT / "output" / "composite_cache", ROOT / "_ves_tmp" / "0846A_inference"):
            for key in (str(sid), str(FLAT[sid]), "auto_grown_20261012073244544_abf"):
                for path in cache_root.glob(f"{key}_*.png"):
                    preserve(path, backup / "composites" / path.name)
    for folder in label_transforms:
        for sid in (MAIN, OFFSET):
            preserve(ROOT / folder / f"{sid}.png", backup / folder / f"{sid}.png")
        shutil.move(str(WORK / folder / f"{MAIN}.png"), str(ROOT / folder / f"{MAIN}.png"))
        (ROOT / folder / f"{OFFSET}.png").symlink_to(f"{MAIN}.png")
    published_offset = ROOT / "tifxyz" / "3_new_offset" / offset_mesh.name
    preserve(published_offset, backup / "offset_mesh")
    shutil.copytree(offset_mesh, published_offset)
    for sid in (MAIN, OFFSET, *FLAT.values()):
        compute_norm(sid, str(zarr_root))
    ensure_surface_anchors((MAIN, OFFSET, *FLAT.values()), str(zarr_root))
    for sid in (MAIN, OFFSET, *FLAT.values()):
        if tuple(zarr.open(str(zarr_root / f"{sid}.zarr"), mode="r").shape) != (28, *common.shape):
            raise ValueError(f"Incorrect final volume shape: {sid}")
        if not np.array_equal(cv2.imread(str(ROOT / "masks" / f"{sid}.png"), 0), common):
            raise ValueError(f"Incorrect final footprint: {sid}")
    result = {"main": MAIN, "offset": OFFSET, "offset_voxels": 29, "shape": [28, *common.shape],
              "common_mask_pixels": int((common > 0).sum()), "label_transforms": label_transforms,
              "source_mesh": str(mesh), "backup": str(backup), "complete": True}
    (WORK / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[pair] rebuild complete", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()