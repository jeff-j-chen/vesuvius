"""where the papyrus sheet sits along depth: histogram of the per-block intensity-profile peak.

The scroll is densely wound with weak contrast, so the global masked mean is flat. Each 64x64 block's depth profile
still peaks on the sheet; the prominence-weighted histogram of those peaks has one mode at the sheet centre.

    python3 _ves_tmp/depth_peaks.py <zarr id> [<zarr id> ...]
"""
import sys

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter1d
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
BLOCK = 64
SIGMA_FINE, SIGMA_COARSE = 1.5, 5.0  # band-pass in layers: drops the monotone trends that peak at the volume edges
SEARCH = (3, 24)  # inclusive layers searched, so every 8-slice window around a peak stays inside the volume


def block_profiles(zid):
    volume = zarr.open(f"/vesuvius/ves_zarrs2/{zid}.zarr", mode="r")
    mask = np.asarray(Image.open(f"/vesuvius/masks/{zid}.png").convert("L")) > 0
    _, height, width = volume.shape
    rows, cols = height // BLOCK, width // BLOCK
    out = []
    for row in range(rows):
        strip = np.asarray(volume[:, row * BLOCK:(row + 1) * BLOCK, :cols * BLOCK], dtype=np.float32)
        frame = mask[row * BLOCK:(row + 1) * BLOCK, :cols * BLOCK]
        for col in range(cols):
            if frame[:, col * BLOCK:(col + 1) * BLOCK].mean() < 0.99:
                continue
            out.append(strip[:, :, col * BLOCK:(col + 1) * BLOCK].mean((1, 2)))
    return np.array(out)


def summarize(zid):
    profiles = block_profiles(zid)
    band = (gaussian_filter1d(profiles, SIGMA_FINE, axis=1, mode="nearest")
            - gaussian_filter1d(profiles, SIGMA_COARSE, axis=1, mode="nearest"))
    lo, hi = SEARCH
    peaks = band[:, lo:hi + 1].argmax(1) + lo
    prominence = band[:, lo:hi + 1].max(1) - np.median(band, 1)
    histogram = np.bincount(peaks, weights=np.clip(prominence, 0, None), minlength=profiles.shape[1])
    histogram /= histogram.sum()
    local = np.convolve(np.pad(histogram, 1), np.ones(3), mode="valid")
    mode = int(local.argmax())
    window = np.arange(max(mode - 2, 0), min(mode + 3, len(histogram)))
    centre = float((window * histogram[window]).sum() / histogram[window].sum())
    print(f"{zid}: {len(profiles)} blocks; peak-histogram mode {mode} (weighted centre {centre:.2f}); "
          f"median block peak {np.median(peaks):.1f}")
    print("  depth :", " ".join(f"{d:4d}" for d in range(len(histogram))))
    print("  hist% :", " ".join(f"{100 * v:4.1f}" for v in histogram))
    return mode, centre


if __name__ == "__main__":
    for zid in sys.argv[1:]:
        summarize(zid)
