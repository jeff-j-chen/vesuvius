"""split the 0841 self-render drop into volume / surface map / normalization: score our render with every
combination of its own and the published surface maps and norm stats."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/vesuvius/old")
sys.path.insert(0, "/vesuvius")
os.chdir("/vesuvius")
import test_eval  # noqa: E402
from p841_render_check import recall_at_fpr  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

PUB, OURS = 20260221022814, 20260221022891
ns = test_eval.notebook_namespace(sys.argv[1])
mask = cv2.imread(ns["mask_path_for"](OURS), cv2.IMREAD_GRAYSCALE) > 0
bounds = (0, mask.shape[0], 0, mask.shape[1])
pub_inputs = ns["load_inputs"](ns["C"], PUB, bounds)
ours = ns["load_inputs"](ns["C"], OURS, bounds)
print(f"[norm] published {np.round(pub_inputs['norm'], 4).tolist()}  ours {np.round(ours['norm'], 4).tolist()}")
saved = np.load("_ves_tmp/p841check/maps.npz")
common, ink = saved["common"], saved["ink"]
results = {}
for surface in ("ours", "published"):
    for norm in ("ours", "published"):
        inputs = dict(ours)
        if surface == "published":
            inputs["surface_depth"] = pub_inputs["surface_depth"]
            inputs["surface_confidence"] = pub_inputs["surface_confidence"]
        if norm == "published":
            inputs["norm"] = pub_inputs["norm"]
        pred = np.asarray(ns["predict_map"](ns["MODEL"], ns["C"], inputs), np.float32)
        s, y = pred[common], ink[common]
        key = f"surface={surface} norm={norm}"
        results[key] = {"recall@1%fpr": recall_at_fpr(s, y), "pr_auc": float(average_precision_score(y, s)),
                        "mean_score": float(s.mean())}
        print(f"[factor] {key}: " + "  ".join(f"{k}={v:.4f}" for k, v in results[key].items()), flush=True)
with open("_ves_tmp/p841check/factor_check.json", "w") as handle:
    json.dump(results, handle, indent=2)
