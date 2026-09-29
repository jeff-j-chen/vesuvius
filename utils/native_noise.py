"""plan R.3: real 113 keV / 1.2 m scan noise, added to training crops of translated (or pooled) fragments.

the bank (crossres/build_noise_bank.py) holds residuals raw - denoiser(raw) from native volumes, in
normalised units. when its sidecar says the noise depends on intensity, the patches are stored at unit
std and rescaled here by sigma(local clean intensity) = polyval(coeffs, block).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


class NativeNoise:
    def __init__(self, path: str, scale: float = 1.0):
        self.path, self.scale = str(path), float(scale)
        self.bank = np.load(self.path, mmap_mode="r")
        meta = json.loads(Path(self.path).with_suffix(".json").read_text(encoding="utf-8"))
        self.coeffs = np.asarray(meta["sigma_coeffs"], np.float32) if meta.get("intensity_scaled") else None
        self.sigma_floor = float(meta.get("sigma_floor", 0.0))

    def add(self, block: np.ndarray) -> np.ndarray:
        """(D, H, W) normalised block -> the same with a fresh random noise patch; zeros stay zero."""
        depth, height, width = block.shape
        count, bank_depth, bank_h, bank_w = self.bank.shape
        if depth > bank_depth or height > bank_h or width > bank_w:
            raise ValueError(f"noise bank patches {self.bank.shape[1:]} are smaller than the block {block.shape}")
        z = random.randint(0, bank_depth - depth)
        y = random.randint(0, bank_h - height)
        x = random.randint(0, bank_w - width)
        patch = np.asarray(self.bank[random.randrange(count), z:z + depth, y:y + height, x:x + width], np.float32)
        patch = np.rot90(patch, k=random.randint(0, 3), axes=(1, 2))
        if random.random() < 0.5:
            patch = patch[:, :, ::-1]
        if self.coeffs is not None:
            patch = patch * np.maximum(np.polyval(self.coeffs, block), self.sigma_floor)
        noisy = np.clip(block + self.scale * patch, 0.0, 1.0)
        return np.ascontiguousarray(np.where(block > 0, noisy, 0.0), dtype=np.float32)
