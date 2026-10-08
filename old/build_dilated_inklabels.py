"""build_dilated_inklabels.py -- drawn labels, dilated, restricted to high-resolution ink predictions.

per segment (every inklabels/<id>.png):
  1. load inklabels/1_1um/<id>.png and inklabels/2_4um/<id>.png (all zeros if missing)
  2. binarize both at >= THRESHOLD
  3. high_res = 1_1um OR 2_4um, closed CLOSE_ITERS x with a CLOSE_KERNEL^2 kernel (except NO_CLOSE_IDS)
  4. drawn = inklabels/<id>.png dilated DILATE_ITERS x with a DILATE_KERNEL^2 kernel, then AND high_res
     (EXTRA_DILATE_PASSES adjusts passes; DRAWN_ONLY_IDS skip the high-res step)
  5. write dilated_inklabels/<id>.png (uint8 0/255, same shape as the drawn label)

High-resolution labels are aligned top-left to the drawn label (cropped or zero-padded at the
bottom/right), matching the dataloader's top-left common-crop convention.

usage:
  python build_dilated_inklabels.py --only 20231210121321,20231201215900
  python build_dilated_inklabels.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DRAWN_DIR = ROOT / "inklabels"
HIGH_RES_DIRS = (DRAWN_DIR / "1_1um", DRAWN_DIR / "2_4um")
OUT_DIR = ROOT / "dilated_inklabels"
THRESHOLD = 167
DILATE_KERNEL = 5
DILATE_ITERS = 4
# morphological close on the thresholded 1.1um/2.4um detections before they gate the drawn labels
CLOSE_KERNEL = 3
CLOSE_ITERS = 2
# Paris1 Fr34, Paris2 Fr143, Cr1 Fr3, Cr4 Fr8: detections used unclosed
NO_CLOSE_IDS = {"20230301213423", "20230301213755", "20231201215900", "20231205222200"}
# dilation passes added to (or removed from) DILATE_ITERS for specific segments
EXTRA_DILATE_PASSES = {
    "20240304141531": 2, "20240304144031": 2, "20250919125754": 2, "20260221022814": 2,
    "20250511003658": -3, "20250628074500": -3,
}
# extra close passes on top of CLOSE_ITERS
EXTRA_CLOSE_PASSES = {"20260221022814": 1}
# segments whose detections are ignored: the dilated drawn label is used as-is
DRAWN_ONLY_IDS = {
    "20251111010954", "20251112000002", "20250511003658", "20250628074500",
}
# researcher-labelled segments copied unchanged, never dilated:
# w030, w043, w045, w040, w041, w039 (PHerc0139), Paris4, w018, w044, PHerc0841, Paris2 Fr47
COPY_IDS = {
    "20250108000005", "20260112000000", "20260126000000",
    "20250831000000", "20260108000000", "20260302000000",
    "20231210121321", "20240304144031", "20260115000000",
    "20260221022814", "20230205142449",
}


def _read_gray(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    return image


def _fit_top_left(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=image.dtype)
    h, w = min(shape[0], image.shape[0]), min(shape[1], image.shape[1])
    out[:h, :w] = image[:h, :w]
    return out


def build(scroll_id: str) -> None:
    drawn = _read_gray(DRAWN_DIR / f"{scroll_id}.png")
    if drawn is None:
        raise FileNotFoundError(DRAWN_DIR / f"{scroll_id}.png")
    shape = drawn.shape
    high_res = np.zeros(shape, dtype=bool)
    sources = []
    for directory in HIGH_RES_DIRS:
        image = _read_gray(directory / f"{scroll_id}.png")
        if image is None:
            sources.append(f"{directory.name}=missing")
            continue
        sources.append(f"{directory.name}={image.shape}")
        high_res |= _fit_top_left(image, shape) >= THRESHOLD
    if scroll_id not in NO_CLOSE_IDS:
        close_iters = CLOSE_ITERS + EXTRA_CLOSE_PASSES.get(scroll_id, 0)
        close_kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), dtype=np.uint8)
        high_res = cv2.morphologyEx(
            high_res.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=close_iters
        ) > 0
        sources.append(f"close={CLOSE_KERNEL}x{CLOSE_KERNEL}x{close_iters}")

    kernel = np.ones((DILATE_KERNEL, DILATE_KERNEL), dtype=np.uint8)
    iterations = 0 if scroll_id in COPY_IDS else DILATE_ITERS + EXTRA_DILATE_PASSES.get(scroll_id, 0)
    dilated = drawn > 0
    if iterations:
        dilated = cv2.dilate(dilated.astype(np.uint8), kernel, iterations=iterations) > 0
    if scroll_id in COPY_IDS:
        combine, final = "copy", dilated
    elif scroll_id in DRAWN_ONLY_IDS:
        combine, final = "drawn_only", dilated
    else:
        combine, final = "and", dilated & high_res

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{scroll_id}.png"
    if not cv2.imwrite(str(out_path), final.astype(np.uint8) * 255):
        raise RuntimeError(f"failed to write {out_path}")
    print(
        f"{scroll_id}: shape={shape} {' '.join(sources)} | dilate x{iterations} combine={combine} "
        f"drawn={int((drawn > 0).sum())} "
        f"dilated={int(dilated.sum())} high_res={int(high_res.sum())} final={int(final.sum())} "
        f"-> {out_path.relative_to(ROOT)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", type=str, default=None, help="comma-separated segment ids")
    args = parser.parse_args()
    if args.only:
        scroll_ids = [value.strip() for value in args.only.split(",") if value.strip()]
    else:
        scroll_ids = sorted(path.stem for path in DRAWN_DIR.glob("*.png"))
    for scroll_id in scroll_ids:
        build(scroll_id)


if __name__ == "__main__":
    main()
