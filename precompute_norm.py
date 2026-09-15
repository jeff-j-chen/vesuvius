#!/usr/bin/env python3
"""Precompute cached normalization statistics for one or more surface zarrs."""
from __future__ import annotations

import argparse
import os

from utils.norm import UNIFIED_CACHE_PATH, compute_norm


def main() -> None:
    parser = argparse.ArgumentParser(description="precompute Vesuvius zarr normalization")
    parser.add_argument(
        "--scroll-id",
        action="append",
        required=True,
        help="segment id; repeat the option to process several segments",
    )
    parser.add_argument(
        "--zarr-path",
        default=os.getenv("VESUVIUS_ZARR_PATH", "/vesuvius/ves_zarrs2"),
    )
    parser.add_argument("--cache-path", default=UNIFIED_CACHE_PATH)
    parser.add_argument("--mask-dir", default="./masks")
    parser.add_argument("--y-block", type=int, default=512)
    args = parser.parse_args()

    for scroll_id in args.scroll_id:
        compute_norm(
            scroll_id,
            args.zarr_path,
            cache_path=args.cache_path,
            y_block=args.y_block,
            mask_dir=args.mask_dir,
        )


if __name__ == "__main__":
    main()
