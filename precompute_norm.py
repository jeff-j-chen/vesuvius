"""precompute_norm.py -- CLI wrapper around utils/norm.compute_norm.

usage:
  python precompute_norm.py --scroll-id 20260115000000
  python precompute_norm.py --scroll-id 20250223000000 --scroll-id 20260206000001

writes results to norm_cache.json (used by training automatically).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.norm import compute_norm, UNIFIED_CACHE_PATH

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll-id", action="append", required=True, dest="scroll_ids")
    ap.add_argument("--zarr-path", default=os.getenv(
        "VESUVIUS_ZARR_PATH",
        "/vesuvius/ves_zarrs2" if os.name == "posix"
        else r"C:\Users\ChenJeff\Documents\ves_zarrs2"))
    ap.add_argument("--cache", default=UNIFIED_CACHE_PATH)
    args = ap.parse_args()
    for sid in args.scroll_ids:
        compute_norm(sid, args.zarr_path, args.cache)

if __name__ == "__main__":
    main()
