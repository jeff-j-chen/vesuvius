"""one-time (2026-10-28): starter labels + blank train masks for the segments added to assemble_training_segments.py.

Every label source lives on the segment's fine-scan render (2.2-2.4 um), a different mesh from our 9.362 um frame, so the
level-2 surface midslice is registered onto our volume midslice (old/assemble_researcher_inklabels.surface_maps: SIFT affine +
smoothed displacement field) and the source map is warped with it.

  inklabels/<zid>.png, dilated_inklabels/<zid>.png  official ink-labels/ zarr (level 2) when present, else the ink-detection tif
                                                   area-downsampled to level 2; thresholded at 0.5 (to be corrected by hand)
  inklabels/2_4um/<zid>.png                        the warped ink-detection map itself (soft, uint8)
  train_masks/<zid>.png                            all zero
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import tifffile
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assemble_researcher_inklabels import BUCKET, ZARR_DIR, open_remote, read, surface_maps  # noqa: E402

# zid -> (segment, fine surface volume, ink-labels volume dir or None, ink-detection tif)
SOURCES = {
    "20250825181859": ("PHerc0500P2/segments/20250825181859--1", "2.215um-0.4m-111keV-volume-20250526151718.zarr",
                       "2.215um-volume-20250526151718",
                       "PHerc0500P2-20250825181859-2.215um-0.4m-111keV-volume-20250526151718-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"),
    "20250825181917": ("PHerc0500P2/segments/20250825181917--2", "2.215um-0.4m-111keV-volume-20250526151718.zarr", None,
                       "PHerc0500P2-20250825181917-2.215um-0.4m-111keV-volume-20250526151718-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"),
    "20250902170441": ("PHerc0343P/segments/20250902170441--4_b2", "2.215um-0.4m-111keV-volume-20260304131111.zarr", None,
                       "PHerc0343P-20250902170441-2.215um-0.4m-111keV-volume-20260304131111-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"),
    "20250919132115": ("PHerc0009B/segments/20250919132115-auto_grown_20250919062115592_inp_hr",
                       "2.401um-0.35m-77keV-volume-20250820154339.zarr", None,
                       "PHerc0009B-20250919132115-2.401um-0.35m-77keV-volume-20250820154339-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"),
    "20260220213127": ("PHerc0841/segments/20260220213127-w00", "2.403um-0.22m-77keV-volume-20260319124803.zarr",
                       "2.403um-volume-20260319124803",
                       "PHerc0841-20260220213127-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"),
    "20250925222237": ("PHerc0814/segments/20250925222237-auto_grown_20250925222237459",
                       "2.399um-0.22m-78keV-volume-20260309142202.zarr", None,
                       "PHerc0814-20250925222237-2.399um-0.22m-78keV-volume-20260309142202-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"),
    "20260422000000": ("PHerc0139/segments/20260422000000-title_2026042222_zmid_flatboi",
                       "2.399um-0.22m-78keV-volume-20260102150214.zarr", None,
                       "PHerc0139-20260422000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"),
}
MANIFEST = os.path.join("output", "new_segments_20261028", "labels_manifest.json")


def detection_level2(seg, tif, shape):
    path = os.path.join("_ves_tmp", "detect", tif)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        subprocess.run(["curl", "-s", "--fail", "--retry", "3", "-o", path, f"{BUCKET}/{seg}/ink-detection/{tif}"], check=True)
    image = tifffile.imread(path)
    if image.ndim == 3:
        image = image[..., 0]
    if image.dtype != np.uint8:
        image = np.clip(image.astype(np.float32) * (255.0 if image.max() <= 1 else 1.0), 0, 255).astype(np.uint8)
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    want = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else set(SOURCES)
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    for zid, (seg, surface, label_dir, tif) in SOURCES.items():
        if zid not in want:
            continue
        target = tuple(int(v) for v in zarr.open(os.path.join(ZARR_DIR, f"{zid}.zarr"), mode="r").shape[1:])
        mx, my, src_shape, info = surface_maps(zid, seg, surface, 2)
        detection = detection_level2(seg, tif, src_shape)
        soft = cv2.remap(detection, mx, my, cv2.INTER_LINEAR, borderValue=0)
        if label_dir:
            label = read(open_remote(f"{BUCKET}/{seg}/ink-labels/{label_dir}/20260918/inklabels.zarr/2")).astype(np.float32)
            label /= max(float(label.max()), 1.0)
            if label.shape != tuple(src_shape):
                label = cv2.resize(label, (src_shape[1], src_shape[0]), interpolation=cv2.INTER_AREA)
            prob, source = cv2.remap(label, mx, my, cv2.INTER_LINEAR, borderValue=0), "ink-labels"
        else:
            prob, source = soft / 255.0, "ink-detection"
        assert prob.shape == target, (zid, prob.shape, target)
        binary = (prob >= 0.5).astype(np.uint8) * 255
        for folder in ("inklabels", "dilated_inklabels"):
            cv2.imwrite(os.path.join(folder, f"{zid}.png"), binary)
        os.makedirs(os.path.join("inklabels", "2_4um"), exist_ok=True)
        cv2.imwrite(os.path.join("inklabels", "2_4um", f"{zid}.png"), np.clip(np.rint(soft), 0, 255).astype(np.uint8))
        train_mask = os.path.join("train_masks", f"{zid}.png")
        if not os.path.exists(train_mask):
            cv2.imwrite(train_mask, np.zeros(target, np.uint8))
        info.update({"source": source, "shape": list(target), "positive_frac": round(float((binary > 0).mean()), 5)})
        manifest[zid] = info
        print(f"[labels] {zid}: {info}", flush=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as handle:
        json.dump(manifest, handle, indent=1)


if __name__ == "__main__":
    main()
