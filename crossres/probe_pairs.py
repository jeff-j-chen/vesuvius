"""step 1: confirm every pair exists and that the two renders' sizes agree with their voxel sizes.

reads only .zarray metadata (a few KB per segment). writes crossres/pairs_manifest.json.

    python crossres/probe_pairs.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import PAIRS, TRAIN_FRAME_UM, high_to_low, url  # noqa: E402


def _json(address: str):
    try:
        with urllib.request.urlopen(address, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def _levels(base: str) -> dict:
    levels = {}
    for level in range(8):
        meta = _json(f"{base}/{level}/.zarray")
        if meta is None:
            break
        levels[level] = {"shape": meta["shape"], "chunks": meta["chunks"], "dtype": meta["dtype"]}
    return levels


def main() -> None:
    manifest = []
    for pair in PAIRS:
        low, high = _levels(url(pair, "low")), _levels(url(pair, "high"))
        entry = {**pair, "low_levels": low, "high_levels": high}
        if not low or not high:
            print(f"{pair['name']:12s} MISSING low={bool(low)} high={bool(high)}")
            manifest.append(entry)
            continue
        low_shape, high_shape = low[0]["shape"], high[0]["shape"]
        expected = high_to_low(pair)
        ratio_y, ratio_x = high_shape[1] / low_shape[1], high_shape[2] / low_shape[2]
        # thickness in um: identical renders around one surface should cover the same physical depth
        thickness = (low_shape[0] * pair["low_um"], high_shape[0] * pair["high_um"])
        entry.update({
            "expected_ratio": expected, "ratio_y": ratio_y, "ratio_x": ratio_x,
            "thickness_um": thickness,
            "ratio_ok": abs(ratio_y / expected - 1) < 0.03 and abs(ratio_x / expected - 1) < 0.03,
        })
        flag = "ok" if entry["ratio_ok"] else "CHECK (different crop, mesh or pixel spacing)"
        print(f"{pair['name']:12s} low={low_shape} high={high_shape} chunks={high[0]['chunks']} "
              f"ratio y/x={ratio_y:.3f}/{ratio_x:.3f} expected={expected:.3f} "
              f"thickness low/high={thickness[0]:.0f}/{thickness[1]:.0f} um  {flag}")
        if pair["frame"] == "resample":
            out_h = round(low_shape[1] * pair["low_um"] / TRAIN_FRAME_UM)
            out_w = round(low_shape[2] * pair["low_um"] / TRAIN_FRAME_UM)
            print(f"{'':12s} training frame after resampling: (28, {out_h}, {out_w})")
        manifest.append(entry)
    path = Path(__file__).resolve().parent / "pairs_manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
