"""Download researcher ink references into inklabels/2_4um without creating training labels."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr

TMP_ROOT = Path("/data/extra/tmp" if os.path.isdir("/data/extra") else "_ves_tmp")
OUTPUT_DIR = Path("inklabels/2_4um")

SOURCES = {
    "p343": {
        "id": "20250511003658",
        "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0343P/segments/20250511003658-tifxyz/ink-detection/PHerc0343P-20250511003658-2.215um-0.4m-111keV-volume-20260304131111-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif",
        "source_shape": (13420, 8020),
        "surface_shape": (3440, 2060),
    },
    "cr1fr3": {
        "id": "20231201215900",
        "url": "https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/inklabels.png",
        "source_shape": (7309, 4560),
        "surface_shape": (7309, 4560),
    },
    "p841": {
        "id": "20260221022814",
        "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/segments/20260221022814-auto_grown_20260220174252405/ink-detection/PHerc0841-20260221022814-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif",
        "source_shape": (84080, 47780),
        "surface_shape": (21560, 12260),
        "surface_crop": (14656, 21560, 0, 5248),
    },
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    result = subprocess.run(
        [
            "curl", "-L", "--fail", "--connect-timeout", "20", "--max-time", "600",
            "--retry", "3", "--retry-delay", "2", "--show-error",
            "-o", str(destination), url,
        ]
    )
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}")


def _read_image(path: Path) -> tuple[np.ndarray, object | None]:
    if path.suffix.lower() == ".png":
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to read {path}")
        return image, None
    store = tifffile.imread(str(path), aszarr=True)
    image = zarr.open(store, mode="r")
    return image, store


def archive(name: str) -> Path:
    spec = SOURCES[name]
    scroll_id = str(spec["id"])
    suffix = Path(str(spec["url"])).suffix
    local = TMP_ROOT / f"research_ink_{scroll_id}{suffix}"
    _download(str(spec["url"]), local)
    source, handle = _read_image(local)
    try:
        if source.ndim == 3:
            source = source[..., 0]
        source_shape = tuple(map(int, source.shape))
        if source_shape != tuple(spec["source_shape"]):
            raise RuntimeError(f"{name}: source shape {source_shape} != {spec['source_shape']}")
        zarr_path = Path("ves_zarrs2") / f"{scroll_id}.zarr"
        target = zarr.open(str(zarr_path), mode="r")
        target_shape = tuple(map(int, target.shape[1:]))
        y0, y1, x0, x1 = spec.get(
            "surface_crop",
            (0, spec["surface_shape"][0], 0, spec["surface_shape"][1]),
        )
        source_y0 = int(round(y0 * source_shape[0] / spec["surface_shape"][0]))
        source_y1 = int(round(y1 * source_shape[0] / spec["surface_shape"][0]))
        source_x0 = int(round(x0 * source_shape[1] / spec["surface_shape"][1]))
        source_x1 = int(round(x1 * source_shape[1] / spec["surface_shape"][1]))
        cropped = np.asarray(source[source_y0:source_y1, source_x0:source_x1], dtype=np.uint8)
        resized = cv2.resize(cropped, target_shape[::-1], interpolation=cv2.INTER_AREA)
        mask = cv2.imread(f"masks/{scroll_id}.png", cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != target_shape:
            raise RuntimeError(f"{name}: missing or mismatched mask")
        resized[mask == 0] = 0
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIR / f"{scroll_id}.png"
        temporary = output.with_suffix(".tmp.png")
        if not cv2.imwrite(str(temporary), resized):
            raise RuntimeError(f"failed to write {temporary}")
        temporary.replace(output)
        print(
            f"{name}: source={source_shape} crop={(source_y0, source_y1, source_x0, source_x1)} "
            f"-> {target_shape} output={output}"
        )
        return output
    finally:
        if handle is not None:
            close = getattr(handle, "close", None)
            if close is not None:
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=tuple(SOURCES))
    args = parser.parse_args()
    for name in args.names:
        archive(name)


if __name__ == "__main__":
    main()
