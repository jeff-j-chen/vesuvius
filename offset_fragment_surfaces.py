"""offset_fragment_surfaces.py -- render the separated fragments 29 layers (9.362 um) deeper into the papyrus and read them
with reader-v3. Not training segments: the renders only exist to judge readability.

  PHerc0500P2 (front, -1, -2): S3 9.362 um tifxyz meshes moved 29 voxels along the renderer normal (old/offset_mesh.py,
      like auto_grown_20261012065803387_straight_abf_m28), rendered from the 9.362 um volume
  Fr143, Fr47, Cr4 Fr8, Fr34 (88 keV rescans) and Cr1 Fr3 (70 keV): the exposed-surface PPM stack shifted 29 output layers
      (29 * 9.362 / 3.24 = 84 source voxels) along the PPM normal (utils/fragment_rescan.render depth_shift)

The sign is chosen per fragment from a normal-direction intensity profile through the raw scan: towards the denser side,
away from the air. Output ids are <zid>_in29 (ves_zarrs2/, masks/); reader-v3 maps and figures go to output/offset29/.

usage: python3 offset_fragment_surfaces.py [--only 20250628074500,...] [--checkpoint reader-v3/reader-v3.pth]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

import assemble_training_segments as ats

OFFSET_LAYERS = 29
SUFFIX = f"_in{OFFSET_LAYERS}"
OUT_DIR = os.path.join("output", "offset29")
P500_VOLUME = (f"{ats.BUCKET}/PHerc0500P2/volumes/20250820143440-9.362um-1.2m-113keV-masked.zarr/0", (7057, 4196, 4196))
MESHES = {  # zid -> mesh dir (9.362 um, downloaded from the segment's mesh/ folder)
    "20250628074500": "tifxyz/PHerc0500P2/500P2_front",
    "20250825181859": "tifxyz/PHerc0500P2/500P2_m1",
    "20250825181917": "tifxyz/PHerc0500P2/500P2_m2",
}
CR1FR3_VOLUME = "https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/volumes_zarr/70keV_3.24um_.zarr/0"
PPMS = {  # zid -> assembler name (PPM, mask, volume and affine come from assemble_training_segments)
    "20230301213755": "paris2_fr143", "20230205142449": "paris2_fr47", "20231205222200": "scroll6_fr8",
    "20230301213423": "paris1_fr34", "20231201215900": "cr1fr3",
}
PROBE_POINTS = 24
PROBE_LAYERS = np.arange(-40, 41, 2)


def _denser_sign(profile):
    """+1 if the scan is denser on the +normal side; profile is the mean intensity at PROBE_LAYERS."""
    plus = profile[PROBE_LAYERS >= 10].mean()
    minus = profile[PROBE_LAYERS <= -10].mean()
    return (1 if plus > minus else -1), float(plus), float(minus)


def _probe(points, normals, step, read_box):
    """mean intensity along each normal at PROBE_LAYERS * step voxels; points/normals (N, 3) xyz."""
    from scipy.ndimage import map_coordinates

    rows = []
    for point, normal in zip(points, normals):
        pts = point[None] + (PROBE_LAYERS * step)[:, None] * normal[None]
        zyx = pts[:, ::-1]
        lo = np.floor(zyx.min(0)).astype(int) - 2
        hi = np.ceil(zyx.max(0)).astype(int) + 3
        box = read_box(lo, hi).astype(np.float32)
        rows.append(map_coordinates(box, (zyx - lo).T, order=1, mode="nearest"))
    return np.mean(rows, axis=0)


def _pick(valid, count, seed=0):
    ys, xs = np.nonzero(valid)
    keep = np.random.default_rng(seed).choice(len(ys), size=min(count, len(ys)), replace=False)
    return ys[keep], xs[keep]


def render_mesh(zid, workers):
    import zarr
    from scipy.ndimage import distance_transform_edt

    import assemble_test_segments as test_assembler

    out_id = f"{zid}{SUFFIX}"
    out_zarr = os.path.join(ats.ZARR_DIR, f"{out_id}.zarr")
    info_path = os.path.join(OUT_DIR, out_id, "offset.json")
    if os.path.isdir(out_zarr) and os.path.isfile(f"masks/{out_id}.png") and os.path.isfile(info_path):
        return out_id
    vol_url, vol_shape = P500_VOLUME
    src = MESHES[zid]
    X, Y, Z, valid = test_assembler.load_mesh(src)
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    normals = test_assembler.compute_normals(X[iy, ix], Y[iy, ix], Z[iy, ix])
    inner = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = _pick(inner & (np.linalg.norm(normals, axis=-1) > 0.5), PROBE_POINTS)
    volume = zarr.open(vol_url, mode="r")

    def read_box(lo, hi):
        lo_c, hi_c = np.maximum(lo, 0), np.minimum(hi, vol_shape)
        out = np.zeros(tuple(hi - lo), np.uint8)
        out[tuple(slice(a - l, b - l) for a, b, l in zip(lo_c, hi_c, lo))] = \
            volume[tuple(slice(a, b) for a, b in zip(lo_c, hi_c))]
        return out

    points = np.stack([X[ys, xs], Y[ys, xs], Z[ys, xs]], 1)
    profile = _probe(points, normals[ys, xs], 1.0, read_box)
    sign, plus, minus = _denser_sign(profile)
    dst = f"{src}{SUFFIX}"
    subprocess.run([sys.executable, "old/offset_mesh.py", src, dst, str(sign * OFFSET_LAYERS)], check=True)
    test_assembler.render_surface_volume(
        mesh_dir=dst, cache_dir=os.path.join(ats.TMP, f"render_{out_id}"), vol_base=vol_url, vol_shape=vol_shape,
        layers=ats.TARGET_DEPTH, normal_step=1.0, upsample_factor=20.0, workers=workers, out_zarr=out_zarr, out_id=out_id,
        chunk_depth=ats.DEFAULT_CHUNK_DEPTH, chunk_y=ats.DEFAULT_CHUNK_Y, chunk_x=ats.DEFAULT_CHUNK_X, mask_dir="masks",
        output_dtype="|u1", source_um=ats.TARGET_VOXEL_UM)
    shutil.rmtree(os.path.join(ats.TMP, f"render_{out_id}"), ignore_errors=True)
    _write_info(info_path, {"source": src, "mesh": dst, "offset_voxels": sign * OFFSET_LAYERS, "voxel_um": ats.TARGET_VOXEL_UM,
                            "profile_layers": PROBE_LAYERS.tolist(), "profile": np.round(profile, 2).tolist(),
                            "mean_plus_side": plus, "mean_minus_side": minus})
    return out_id


def render_ppm(zid):
    from utils import fragment_rescan

    name = PPMS[zid]
    out_id = f"{zid}{SUFFIX}"
    out_zarr = os.path.join(ats.ZARR_DIR, f"{out_id}.zarr")
    info_path = os.path.join(OUT_DIR, out_id, "offset.json")
    if os.path.isdir(out_zarr) and os.path.isfile(f"masks/{out_id}.png") and os.path.isfile(info_path):
        return out_id
    opts = ats.FRAG_OPTS[name]
    base = opts["dlash_surface_base"].rstrip("/")
    rescan = ats.RESCAN_88KEV.get(name, {})
    volume_url = rescan.get("volume_url", CR1FR3_VOLUME)
    affine = np.asarray(rescan.get("affine", np.eye(4)[:3]), float)
    cache_dir = os.path.join(ats.TMP, f"offset_{zid}")
    mask_path = os.path.join(cache_dir, "mask.png")
    ats._download_once(f"{base}/mask.png", mask_path)
    _, height, width = map(int, opts["surface_expected_shape"])
    frame = fragment_rescan.out_shape(height, width)
    footprint = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), frame[::-1], interpolation=cv2.INTER_NEAREST) > 0
    xyz, nrm, valid = fragment_rescan.read_ppm_grid(f"{base}/result.ppm", os.path.join(cache_dir, "ppm_grid.npz"))
    inner = cv2.erode((valid & footprint).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    ys, xs = _pick(inner, PROBE_POINTS)
    volume = fragment_rescan.RemoteZarr(volume_url, os.path.join(cache_dir, "chunks"), cap=None)
    # probe in the target scan: map the points and the normal through the fitted affine
    points = xyz[ys, xs] @ affine[:, :3].T + affine[:, 3]
    normals = nrm[ys, xs] @ affine[:, :3].T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    profile = _probe(points, normals, fragment_rescan.S, volume.box)
    sign, plus, minus = _denser_sign(profile)
    field_path = rescan.get("offset_field")
    field = np.load(field_path) if field_path and os.path.isfile(field_path) else None
    if field_path and field is None:
        print(f"[offset] {zid}: {field_path} missing -> affine only (residual up to ~10 source voxels)", flush=True)
    partial = out_zarr + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    mask = fragment_rescan.render(
        partial, f"{base}/result.ppm", volume_url, affine.tolist(), cache_dir, footprint=footprint,
        tif_url=rescan.get("tif_url"), tif_digits=int(rescan.get("tif_digits", 4)),
        chunks=(ats.DEFAULT_CHUNK_DEPTH, ats.DEFAULT_CHUNK_Y, ats.DEFAULT_CHUNK_X), offset_field=field,
        depth_shift=sign * OFFSET_LAYERS)
    shutil.rmtree(out_zarr, ignore_errors=True)
    os.replace(partial, out_zarr)
    cv2.imwrite(f"masks/{out_id}.png", mask.astype(np.uint8) * 255)
    shutil.rmtree(os.path.join(cache_dir, "chunks"), ignore_errors=True)
    _write_info(info_path, {"source": f"{base}/result.ppm", "volume": volume_url, "depth_shift_layers": sign * OFFSET_LAYERS,
                            "source_voxels": round(sign * OFFSET_LAYERS * fragment_rescan.S, 2),
                            "profile_layers": PROBE_LAYERS.tolist(), "profile": np.round(profile, 2).tolist(),
                            "mean_plus_side": plus, "mean_minus_side": minus})
    return out_id


def _write_info(path, info):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=1)


def read_and_show(out_id, checkpoint):
    import zarr

    from utils.predict_models import predict_reader

    folder = os.path.join(OUT_DIR, out_id)
    zarr_path = os.path.join(ats.ZARR_DIR, f"{out_id}.zarr")
    prediction = predict_reader(zarr_path, checkpoint, os.path.join(folder, "reader_v3.tif"))
    mask = cv2.imread(f"masks/{out_id}.png", cv2.IMREAD_GRAYSCALE) > 0
    middle = np.asarray(zarr.open(zarr_path, mode="r")[ats.TARGET_DEPTH // 2]).astype(np.float32)
    lo, hi = np.percentile(middle[mask], [1, 99]) if mask.any() else (0, 1)
    middle = np.clip((middle - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    ink = np.clip(prediction * 255, 0, 255).astype(np.uint8)
    ink[~mask] = 0
    cv2.imwrite(os.path.join(folder, "reader_v3.png"), ink)
    panel = np.concatenate([np.where(mask, middle, 0), np.full((middle.shape[0], 16), 255, np.uint8), ink], axis=1)
    scale = min(1.0, 3000 / panel.shape[1])
    cv2.imwrite(os.path.join(folder, "midslice_and_reader_v3.jpg"),
                cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    print(f"[offset] {out_id}: {folder}/reader_v3.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--checkpoint", default="reader-v3/reader-v3.pth")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-predict", action="store_true")
    args = ap.parse_args()
    # 500P2 offsets are optional (-1/-2 are the canonical neighbouring sheets): only with --only
    want = set(args.only.split(",")) if args.only else set(PPMS)
    args.checkpoint = os.path.abspath(args.checkpoint)
    results = {}
    for zid in [*MESHES, *PPMS]:
        if zid not in want:
            continue
        try:
            out_id = render_mesh(zid, args.workers) if zid in MESHES else render_ppm(zid)
            if not args.no_predict:
                read_and_show(out_id, args.checkpoint)
            results[zid] = "OK"
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[zid] = f"FAIL: {exc}"
    for zid, status in results.items():
        print(f"  {zid}{SUFFIX}: {status}")


if __name__ == "__main__":
    main()
