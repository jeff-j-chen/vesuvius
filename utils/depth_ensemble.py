"""depth ensembles for the evaluation notebooks: run predict_tiles at several surface offsets and combine the maps.

the methods and defaults are those of the depth-composite cell in 0846A_eval.ipynb. combined maps are returned on
a 0..1 scale (z-score is divided by its 99.5th percentile and clipped) so the usual figures can show them unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

METHODS = ("z-score", "max", "mid-boost", "mean")
DEFAULT_METHOD = "mid-boost"
TOPK_MEAN = 2               # "mean": mean of the k highest probabilities
ZSCORE_TOPK = 2             # "z-score": strongest per-depth z-scores averaged per pixel
CORROB_K = 3                # "mid-boost": a cell counts as corroborated when its k-th strongest depth is also high
CORROB_GAMMA = 0.5
CORROB_SIGMA = 1.0          # spatial agreement, in prediction cells
CORROB_RAMP = (0.15, 0.45)


def resolve(depth_ensemble, ensemble_method) -> tuple[list[int], str]:
    """offsets to run and the combine method; an empty depth_ensemble means the single usual depth."""
    method = ensemble_method or DEFAULT_METHOD
    if method not in METHODS:
        raise ValueError(f"ensemble_method must be one of {METHODS} (or '' for {DEFAULT_METHOD}), got {method!r}")
    if not depth_ensemble:
        return [0], method
    if len(depth_ensemble) != 2 or int(depth_ensemble[0]) > int(depth_ensemble[1]):
        raise ValueError(f"depth_ensemble must be [] or [low, high] with low <= high, got {depth_ensemble!r}")
    low, high = int(depth_ensemble[0]), int(depth_ensemble[1])
    return list(range(low, high + 1)), method


def describe(offsets, method) -> str:
    """figure-title fragment; empty for a single depth."""
    if len(offsets) <= 1:
        return ""
    detail = {"z-score": f"top-{ZSCORE_TOPK}", "mean": f"top-{TOPK_MEAN}", "mid-boost": f"k={CORROB_K}"}.get(method)
    name = f"{method} ({detail})" if detail else method
    return f"{name} ensemble of {len(offsets)} depths d{offsets[0]:+d}..{offsets[-1]:+d}"


def combine(stack: np.ndarray, method: str) -> np.ndarray:
    """(depths, h, w) probability maps -> one 0..1 map; cells invalid at any depth stay NaN."""
    valid = np.isfinite(stack).all(axis=0)
    probability = np.clip(np.nan_to_num(stack, nan=0.5), 1e-4, 1 - 1e-4)
    ranked = np.sort(probability, axis=0)[::-1]
    if method == "max":
        result = ranked[0]
    elif method == "mean":
        result = ranked[:min(TOPK_MEAN, len(ranked))].mean(axis=0)
    elif method == "mid-boost":
        k = min(CORROB_K, len(ranked))
        weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), CORROB_SIGMA)
        support = cv2.GaussianBlur((ranked[k - 1] * valid).astype(np.float32), (0, 0), CORROB_SIGMA)
        support /= np.maximum(weight, 1e-6)
        agree = np.clip((support - CORROB_RAMP[0]) / (CORROB_RAMP[1] - CORROB_RAMP[0]), 0, 1)
        agree = agree * agree * (3 - 2 * agree)
        result = ranked[:k].mean(axis=0) ** (1 - (1 - CORROB_GAMMA) * agree)
    elif method == "z-score":
        # each depth on its own noise scale (median/MAD of its logits), so a noisy depth cannot dominate
        logits = np.log(probability) - np.log1p(-probability)
        z = np.empty_like(logits)
        for index, depth_logits in enumerate(logits):
            values = depth_logits[valid] if valid.any() else depth_logits.ravel()
            median = np.median(values)
            z[index] = (depth_logits - median) / (1.4826 * np.median(np.abs(values - median)) + 1e-6)
        z_top = np.sort(z, axis=0)[-min(ZSCORE_TOPK, len(z)):].mean(axis=0)
        upper = float(np.percentile(z_top[valid], 99.5)) if valid.any() else 1.0
        result = np.clip(z_top / max(upper, 1e-6), 0, 1)
    else:
        raise ValueError(f"unknown ensemble method {method!r}")
    return np.where(valid, result, np.nan).astype(np.float32)


def ensemble_predict(predict, offsets, method):
    """wrap a predict_tiles-style callable: run it at the caller's surface_depth_offset plus each offset and combine.

    a single [0] offset returns predict itself; tuple results (also_tta) are combined element-wise.
    """
    if list(offsets) == [0]:
        return predict

    def wrapped(*args, surface_depth_offset=0, **kwargs):
        results = [predict(*args, surface_depth_offset=int(surface_depth_offset) + offset, **kwargs)
                   for offset in offsets]
        if isinstance(results[0], tuple):
            return tuple(combine(np.stack([result[i] for result in results]), method) for i in range(len(results[0])))
        return combine(np.stack(results), method)

    return wrapped
