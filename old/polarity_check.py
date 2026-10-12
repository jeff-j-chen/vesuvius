"""zero-training polarity check: score a checkpoint on published training volumes with the local contrast sign
flipped (x' = low(x) - (x - low(x)), masked in-plane gaussian low-pass), which keeps papyrus/air geometry.

    python3 old/polarity_check.py <checkpoint> [--sigma 8] [--ids 20260221022814,20250919125754]
"""
import json
import os
import sys

import cv2
import numpy as np

REPO = "/vesuvius"
sys.path.insert(0, os.path.join(REPO, "old"))
sys.path.insert(0, REPO)
os.chdir(REPO)
import test_eval  # noqa: E402
from p841_render_check import recall_at_fpr  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def flip_local_contrast(vol, sigma):
    out = np.empty_like(vol)
    for k in range(vol.shape[0]):
        layer = vol[k].astype(np.float32)
        valid = (layer > 0).astype(np.float32)
        low = cv2.GaussianBlur(layer * valid, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(valid, (0, 0), sigma), 1e-3)
        flipped = np.clip(2.0 * low - layer, 1, np.iinfo(vol.dtype).max) * valid
        out[k] = flipped.astype(vol.dtype)
    return out


def score(pred, label):
    mh, mw = pred.shape
    ink = cv2.resize(label.astype(np.float32), (mw, mh), interpolation=cv2.INTER_AREA) > 0
    keep = np.isfinite(pred)
    s, y = pred[keep], ink[keep]
    return {"recall@1%fpr": recall_at_fpr(s, y), "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)), "mean_score": float(s.mean()), "cells": int(keep.sum())}


def main():
    checkpoint = sys.argv[1]
    sigma = float(sys.argv[sys.argv.index("--sigma") + 1]) if "--sigma" in sys.argv else 8.0
    ids = (sys.argv[sys.argv.index("--ids") + 1] if "--ids" in sys.argv else "20260221022814,20250919125754").split(",")
    ns = test_eval.notebook_namespace(checkpoint)
    results = {}
    for zid in ids:
        mask = cv2.imread(ns["mask_path_for"](int(zid)), cv2.IMREAD_GRAYSCALE) > 0
        label = cv2.imread(f"inklabels/{zid}.png", cv2.IMREAD_GRAYSCALE) > 0
        inputs = ns["load_inputs"](ns["C"], int(zid), (0, mask.shape[0], 0, mask.shape[1]))
        base = np.asarray(ns["predict_map"](ns["MODEL"], ns["C"], inputs), np.float32)
        inputs["vol"] = flip_local_contrast(inputs["vol"], sigma)
        flipped = np.asarray(ns["predict_map"](ns["MODEL"], ns["C"], inputs), np.float32)
        keep = np.isfinite(base) & np.isfinite(flipped)
        corr = float(np.corrcoef(base[keep], flipped[keep])[0, 1])
        results[zid] = {"original": score(base, label), "flipped": score(flipped, label), "map_corr": corr}
        for name in ("original", "flipped"):
            row = results[zid][name]
            print(f"[polarity] {zid} {name:8s} " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "cells"),
                  flush=True)
        print(f"[polarity] {zid} map corr original vs flipped {corr:.3f}", flush=True)
        del inputs
    out = os.path.join("_ves_tmp", f"polarity_{os.path.basename(os.path.dirname(checkpoint))}.json")
    with open(out, "w") as handle:
        json.dump({"checkpoint": checkpoint, "sigma": sigma, "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
