"""export_training_ensembles.py -- temporary: export the training-segment ensemble results by scroll id.

Run after villa_ensemble_training_c40.ipynb has finished (system python, from /vesuvius):
    python3 export_training_ensembles.py            # every segment whose figures exist
    python3 export_training_ensembles.py w030 w044  # only these segments
    python3 export_training_ensembles.py --force    # overwrite existing exports

For each segment <name> with scroll id <id>, writes to output/test_visualizations/villa_ensemble/training_c40_by_id/:
    <id>.png          the ensemble at the inklabel size (uint8 grayscale, 0 outside the valid surface). It is
                      rebuilt from the cached model maps (output/villa_ensemble_maps/<id>/), which live on the
                      zarr grid, instead of being cut out of the titled figure; it is resized only if the
                      inklabel differs from the zarr frame.
    <id>_4models.jpg  the 2x2 model grid figure as JPEG (no crop/resize)
The original figures in training_c40/ are left untouched (copied, not moved). Everything in training_c40_by_id/ is
then zipped to output/test_visualizations/villa_ensemble/training_c40_by_id.zip (skip with --no-zip).
"""
import argparse
import json
import os
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REPO = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(REPO, "villa_ensemble_training_c40.ipynb")


def notebook_setup():
    """the training notebook's setup cell: segment list, model set, v8in stride, render pixel sizes"""
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    cells = json.load(open(NOTEBOOK, encoding="utf-8"))["cells"]
    namespace = {}
    exec("".join(next(c for c in cells if c["cell_type"] == "code")["source"]), namespace)
    return namespace


def label_shape(zid, fallback):
    for directory in ("inklabels", "dilated_inklabels", "masks"):
        image = cv2.imread(os.path.join(REPO, directory, f"{zid}.png"), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image.shape, directory
    return fallback, "zarr"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="segment names (default: all)")
    parser.add_argument("--force", action="store_true", help="overwrite existing exports")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--no-zip", action="store_true", help="skip zipping the export folder")
    args = parser.parse_args()

    ns = notebook_setup()
    ve, fragments, tag = ns["ve"], ns["FRAGMENTS"], ns["TAG"]
    out_dir = ns["OUT_DIR"] + "_by_id"
    os.makedirs(out_dir, exist_ok=True)
    unknown = sorted(set(args.names) - set(fragments))
    if unknown:
        raise SystemExit(f"unknown segment names {unknown}; known: {sorted(fragments)}")

    pending = []
    for name in args.names or fragments:
        zid = fragments[name]
        grid_figure, ensemble_figure = ns["outputs"](name)
        if not os.path.isfile(ensemble_figure):
            pending.append(name)
            continue
        ensemble_path = os.path.join(out_dir, f"{zid}.png")
        if args.force or not os.path.isfile(ensemble_path):
            _, ensemble, _ = ve.ensemble(zid, ns["DEPTHS"], ns["DIRECTION"], ns["WRITE_DEPTH"])
            image = np.rint(np.nan_to_num(ensemble, nan=0.0) * 255).astype(np.uint8)
            del ensemble
            shape, source = label_shape(zid, image.shape)
            if image.shape != shape:
                print(f"  {name}: resizing {image.shape} -> {shape} ({source})")
                image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(ensemble_path, image):
                raise OSError(f"could not write {ensemble_path}")
            print(f"[ensemble] {name} -> {ensemble_path} {image.shape[1]}x{image.shape[0]} (matches {source})",
                  flush=True)
        grid_path = os.path.join(out_dir, f"{zid}_4models.jpg")
        if os.path.isfile(grid_figure) and (args.force or not os.path.isfile(grid_path)):
            with Image.open(grid_figure) as figure:
                figure.convert("RGB").save(grid_path, quality=args.jpeg_quality, optimize=True)
            print(f"[4models] {name} -> {grid_path}", flush=True)
    print(f"done -> {out_dir}" + (f"; not rendered yet: {pending}" if pending else ""))
    if not args.no_zip:
        zip_path = out_dir + ".zip"
        files = sorted(f for f in os.listdir(out_dir) if f.endswith((".png", ".jpg")))
        # png/jpg are already compressed: store them
        with zipfile.ZipFile(zip_path + ".partial", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for filename in files:
                archive.write(os.path.join(out_dir, filename), f"{os.path.basename(out_dir)}/{filename}")
        os.replace(zip_path + ".partial", zip_path)
        print(f"zipped {len(files)} files -> {zip_path} ({os.path.getsize(zip_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
