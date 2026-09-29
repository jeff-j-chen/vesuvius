"""audit the depth orientation and centring of every 28-layer training / test render.

a flattened render shows the inked sheet as an intensity peak near the centre. with the normals used in
training, the rise into the sheet is gradual and the fall after it (into the gap) is sharp. a render with
its normal reversed shows the mirror image; a render with a mis-fitted surface has its peak off-centre.
the surface-relative window corrects the centring during fine-tuning but nothing corrects a flip.

writes crossres/depth_orientation.json: {zid: {peak, offset, asymmetry, verdict, profile}}. verdicts
are a heuristic: confirm "reversed" by eye before flipping a scroll.

    python crossres/depth_orientation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def profile(zid: int, tiles: int = 60, seed: int = 0) -> np.ndarray:
    volume = zarr.open(str(ROOT.parent / "ves_zarrs2" / f"{zid}.zarr"), mode="r")
    depth, height, width = map(int, volume.shape)
    mask = cv2.imread(str(ROOT.parent / "masks" / f"{zid}.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    rng = np.random.default_rng(seed)
    total, count = np.zeros(depth), 0
    for _ in range(tiles * 10):
        y, x = int(rng.integers(0, height - 64)), int(rng.integers(0, width - 64))
        if mask[y:y + 64, x:x + 64].mean() < 0.95:
            continue
        block = np.asarray(volume[:, y:y + 64, x:x + 64], np.float32).reshape(depth, -1)
        block[block == 0] = np.nan
        total += np.nan_to_num(np.nanmean(block, axis=1))
        count += 1
        if count >= tiles:
            break
    return total / max(count, 1)


def describe(values: np.ndarray) -> dict:
    smooth = np.convolve(values, np.ones(3) / 3, mode="same")
    centre = (len(values) - 1) / 2
    peak = int(6 + np.argmax(smooth[6:len(values) - 6]))
    # steepest rise into the peak versus steepest fall out of it, within 4 slices
    rise = max(smooth[i + 1] - smooth[i] for i in range(max(0, peak - 4), peak))
    fall = max(smooth[i] - smooth[i + 1] for i in range(peak, min(len(values) - 1, peak + 4)))
    asymmetry = float((fall - rise) / (fall + rise + 1e-9))
    return {"peak": peak, "offset": float(peak - centre), "asymmetry": asymmetry,
            "profile": [round(float(v), 2) for v in values]}


def main() -> None:
    import campaign_archs_33 as campaign33
    from utils.config import DEFAULT_TEST_SCROLL_IDS
    train_ids = [int(s) for s in campaign33.CAMPAIGN33_SCROLL_IDS]
    ids = list(dict.fromkeys(train_ids + [int(s) for s in DEFAULT_TEST_SCROLL_IDS]))
    report = {zid: describe(profile(zid)) for zid in ids}
    reference = np.median([report[zid]["asymmetry"] for zid in train_ids])
    for zid, entry in report.items():
        flipped = np.sign(entry["asymmetry"]) != np.sign(reference) and abs(entry["asymmetry"]) > 0.2
        off = abs(entry["offset"]) >= 2.5
        entry["verdict"] = "reversed?" if flipped else "off-centre" if off else "ok"
        entry["role"] = "train" if zid in train_ids else "test"
        print(f"{zid} {entry['role']:5s} peak={entry['peak']:2d} offset={entry['offset']:+5.1f} "
              f"asymmetry={entry['asymmetry']:+.2f} {entry['verdict']}")
    print(f"training median asymmetry {reference:+.2f} (sign = the training orientation)")
    path = ROOT / "depth_orientation.json"
    path.write_text(json.dumps({str(k): v for k, v in report.items()}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
