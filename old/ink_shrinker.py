from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def transform_ink_labels(
    input_folder="eroded_inklabels",
    output_folder="eroded2_inklabels",
    mode="shrink",
    radius=3,
    mask_folder=None,
    only=None,
):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    if not input_path.is_dir():
        raise FileNotFoundError(f"input folder does not exist: {input_path}")
    if mode not in {"shrink", "dilate"}:
        raise ValueError(f"unsupported mode: {mode}")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    files = sorted(input_path.glob("*.png"))
    if only:
        filename = only if str(only).lower().endswith(".png") else f"{only}.png"
        files = [path for path in files if path.name == filename]
    if not files:
        raise FileNotFoundError(f"no PNG files found in {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    mask_path = Path(mask_folder) if mask_folder else None
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    operation = cv2.erode if mode == "shrink" else cv2.dilate

    print(f"processing {len(files)} labels: mode={mode} radius={radius}")
    for index, source in enumerate(files, 1):
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to read {source}")
        if mask_path is not None:
            mask_file = mask_path / source.name
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"mask not found: {mask_file}")
            if mask.shape != image.shape:
                source_shape = image.shape
                image = cv2.resize(
                    image,
                    mask.shape[::-1],
                    interpolation=cv2.INTER_NEAREST,
                )
                print(
                    f"[resample] {source.name}: label={source_shape} -> {image.shape}"
                )
        transformed = operation(image, kernel, iterations=1) if radius else image.copy()
        if mask_path is not None:
            transformed[mask == 0] = 0

        destination = output_path / source.name
        temporary = destination.with_suffix(".tmp.png")
        if not cv2.imwrite(str(temporary), transformed):
            raise RuntimeError(f"failed to write {temporary}")
        temporary.replace(destination)
        print(f"[{index}/{len(files)}] {source.name}")


def erode_ink_labels(
    input_folder="eroded_inklabels",
    output_folder="eroded2_inklabels",
    erosion_size=3,
    iterations=3,
):
    radius = iterations * (erosion_size // 2)
    transform_ink_labels(input_folder, output_folder, mode="shrink", radius=radius)


def main():
    parser = argparse.ArgumentParser(description="shrink or dilate binary ink labels")
    parser.add_argument("mode", choices=("shrink", "dilate"))
    parser.add_argument("--input-folder", default=None)
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--mask-folder", default=None)
    parser.add_argument("--only", default=None, help="process one PNG filename or stem")
    args = parser.parse_args()
    input_folder = args.input_folder or (
        "eroded_inklabels" if args.mode == "shrink" else "inklabels"
    )
    output_folder = args.output_folder or (
        "eroded2_inklabels" if args.mode == "shrink" else "dilated_inklabels"
    )
    radius = args.radius if args.radius is not None else (3 if args.mode == "shrink" else 2)
    transform_ink_labels(
        input_folder=input_folder,
        output_folder=output_folder,
        mode=args.mode,
        radius=radius,
        mask_folder=args.mask_folder,
        only=args.only,
    )


if __name__ == "__main__":
    main()