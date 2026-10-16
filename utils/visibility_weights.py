"""per-character visibility of the drawn letters, as static character draw weights for the trainer.

visibility = how far a letter's ink sits above the papyrus around it in the max over the 8 slices the model reads
(flat depth 13, offset -1), in units of that papyrus's own spread. letters the eye separates easily get the lowest
draw weight, faint ones the highest, so the stream of training characters is no longer dominated by what is already
obvious. written in the schema of character_failures.py (tra.character_forgetting_static).

python3 -m utils.visibility_weights [--out models/runs_iter2/visibility_weights.json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

WINDOW = (9, 17)       # slices 9-16
RING_OUTER_PX = 41
RING_INNER_PX = 13
MARGIN_PX = 80
LOW, HIGH = 0.5, 3.0   # draw weight of the most / least visible letter
NEW_SET = (20260927134931, 20261012073246, 20261012065813,
           20261012065817, 20261012065819, 20261012065821, 20261012065823)
PLAIN = (20250919125754,)  # trained with unit weights


def _ellipse(diameter: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def scroll_visibility(scroll_id: int, min_pixels: int = 8) -> tuple[np.ndarray, str, np.ndarray]:
    import zarr

    from utils.dataloader import character_label_fingerprint, drawn_characters

    path = Path("dilated_inklabels") / f"{scroll_id}.png"
    mask = cv2.imread(f"masks/{scroll_id}.png", cv2.IMREAD_GRAYSCALE) > 0
    volume = zarr.open(os.path.join("ves_zarrs2", f"{scroll_id}.zarr"), mode="r")
    height, width = min(mask.shape[0], volume.shape[1]), min(mask.shape[1], volume.shape[2])
    composite = np.asarray(volume[WINDOW[0]:WINDOW[1], :height, :width], dtype=np.float32).max(axis=0)
    composite = cv2.GaussianBlur(composite, (0, 0), 2.0)
    components = drawn_characters(path, (height, width), min_pixels)
    ink_all = components > 0
    keepout = cv2.dilate(ink_all.astype(np.uint8), _ellipse(RING_INNER_PX)) > 0
    papyrus = mask[:height, :width]
    ids, scores = [], []
    for cid in range(1, int(components.max()) + 1):
        ys, xs = np.nonzero(components == cid)
        if ys.size == 0:
            continue
        y0, y1 = max(0, ys.min() - MARGIN_PX), min(height, ys.max() + MARGIN_PX + 1)
        x0, x1 = max(0, xs.min() - MARGIN_PX), min(width, xs.max() + MARGIN_PX + 1)
        own = (components[y0:y1, x0:x1] == cid).astype(np.uint8)
        ring = ((cv2.dilate(own, _ellipse(RING_OUTER_PX)) > 0) & ~keepout[y0:y1, x0:x1] & papyrus[y0:y1, x0:x1])
        patch = composite[y0:y1, x0:x1]
        if ring.sum() < 50:
            continue
        ids.append(cid)
        scores.append(float((patch[own > 0].mean() - patch[ring].mean()) / (patch[ring].std() + 1e-6)))
    return np.asarray(ids), character_label_fingerprint(path), np.asarray(scores)


def build(out_path: str) -> None:
    from utils.dataloader import character_label_fingerprint, drawn_characters

    per_scroll = {sid: scroll_visibility(sid) for sid in NEW_SET}
    pooled = np.concatenate([scores for _, _, scores in per_scroll.values()])
    order = pooled.argsort().argsort() / max(len(pooled) - 1, 1)  # 0 = faintest letter
    entries, start = {}, 0
    for sid, (ids, fingerprint, scores) in per_scroll.items():
        rank = order[start:start + len(scores)]
        start += len(scores)
        weights = LOW + (HIGH - LOW) * (1.0 - rank)
        entries[str(sid)] = {"label_fingerprint": fingerprint,
                             "characters": {str(int(i)): {"weight": float(w), "visibility": float(s)}
                                            for i, w, s in zip(ids, weights, scores)}}
        print(f"[visibility] {sid}: {len(ids)} letters, visibility {scores.min():.2f}..{scores.max():.2f} "
              f"(median {np.median(scores):.2f}), weights {weights.min():.2f}..{weights.max():.2f}", flush=True)
    for sid in PLAIN:
        path = Path("dilated_inklabels") / f"{sid}.png"
        mask = cv2.imread(f"masks/{sid}.png", cv2.IMREAD_GRAYSCALE)
        components = drawn_characters(path, mask.shape, 8)
        entries[str(sid)] = {"label_fingerprint": character_label_fingerprint(path),
                             "characters": {str(i): {"weight": 1.0} for i in range(1, int(components.max()) + 1)}}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"min_pixels": 8, "scrolls": entries}, indent=1), encoding="utf-8")
    print(f"[visibility] wrote {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="models/runs_iter2/visibility_weights.json")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    build(args.out)


if __name__ == "__main__":
    main()
