"""build figures/ring_data.js: the closed-ring supervision + edge softening for one PHerc0009B letter.

Mirrors utils/dataloader.py (_make_ring_mask with ring_label_source='closed', _fetch_mask_mt with
pos_only + ring gate, _soften_edge_positives) at the campaign-40 settings.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SID = 20250919125754
T = 16            # tile_size
SUB = 16          # multitile_subtile
GRID = 4          # multitile_grid
CTX = 96          # context_size
CLOSE_R, GAP_R, SHELL_R = 2, 2, 4
SIGMA, FLOOR = 8.0, 0.55
MARGIN_TILES = 10


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def rect(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))


def tile_any(a: np.ndarray, n_ty: int, n_tx: int) -> np.ndarray:
    return a[:n_ty * T, :n_tx * T].reshape(n_ty, T, n_tx, T).any(axis=(1, 3)).astype(np.uint8)


def main() -> None:
    labels = cv2.imread(str(ROOT / "dilated_inklabels" / f"{SID}.png"), 0) > 127
    drawn = cv2.imread(str(ROOT / "inklabels" / f"{SID}.png"), 0) > 127
    scroll = cv2.imread(str(ROOT / "masks" / f"{SID}.png"), 0) > 127
    manual = cv2.imread(str(ROOT / "train_masks" / f"{SID}.png"), 0)
    split = manual >= 240
    forced_neg = (manual >= 100) & (manual <= 143)
    h, w = labels.shape
    n_ty, n_tx = h // T, w // T

    ink_tile = tile_any(labels, n_ty, n_tx)
    mask_tile = tile_any(scroll, n_ty, n_tx)
    ring_seed = labels & ~forced_neg
    seed_tile = tile_any(ring_seed, n_ty, n_tx)
    closed = cv2.erode(cv2.dilate(seed_tile, rect(CLOSE_R)), rect(CLOSE_R)) & mask_tile
    exclusion = cv2.dilate(closed, rect(GAP_R)) & mask_tile
    ring = ((cv2.dilate(exclusion, rect(SHELL_R)) - exclusion) > 0).astype(np.uint8) & mask_tile
    supervision_tile = (ink_tile | ring).astype(np.uint8)
    supervision = np.kron(supervision_tile, np.ones((T, T), np.uint8)).astype(bool)
    supervision = np.pad(supervision, ((0, h - supervision.shape[0]), (0, w - supervision.shape[1])))

    # the Delta: drawn component with a ~395x281 bbox
    n, lab, st, _ = cv2.connectedComponentsWithStats(drawn.astype(np.uint8), 8)
    comp = min(range(1, n), key=lambda i: abs(st[i, 2] - 395) + abs(st[i, 3] - 281))
    x, y, bw, bh = (int(v) for v in st[comp, :4])
    ty0 = max(0, y // T - MARGIN_TILES)
    tx0 = max(0, x // T - MARGIN_TILES)
    ty1 = min(n_ty, (y + bh) // T + 1 + MARGIN_TILES)
    tx1 = min(n_tx, (x + bw) // T + 1 + MARGIN_TILES)
    py0, px0, py1, px1 = ty0 * T, tx0 * T, ty1 * T, tx1 * T

    blurred = cv2.GaussianBlur(labels.astype(np.float32)[py0 - 64:py1 + 64, px0 - 64:px1 + 64],
                               (0, 0), SIGMA, borderType=cv2.BORDER_REPLICATE)[64:-64, 64:-64]

    def window(cy: int, cx: int) -> dict:
        """one multitile training sample whose 16px target tile has top-left (cy, cx)."""
        y0 = cy + (T - GRID * SUB) // 2
        x0 = cx + (T - GRID * SUB) // 2
        cells = []
        for iy in range(GRID):
            for ix in range(GRID):
                ys, xs = y0 + iy * SUB, x0 + ix * SUB
                lbl = float(labels[ys:ys + SUB, xs:xs + SUB].any())
                valid = bool(
                    scroll[ys:ys + SUB, xs:xs + SUB].all()
                    and split[ys:ys + SUB, xs:xs + SUB].all()
                    and supervision[ys:ys + SUB, xs:xs + SUB].all()
                )
                reason = "valid" if valid else (
                    "outside supervision (gap / hole)" if scroll[ys:ys + SUB, xs:xs + SUB].all() else "off-scroll")
                if valid and lbl == 0:
                    py, px = (ys // T) * T, (xs // T) * T
                    if labels[py:py + T, px:px + T].any():
                        valid, reason = False, "non-ink cell in a positive tile"
                peak = float(blurred[ys - py0:ys - py0 + SUB, xs - px0:xs - px0 + SUB].max()) if lbl else 0.0
                target = lbl if (lbl == 0 or peak >= 0.98) else max(FLOOR, peak)
                cells.append({"iy": iy, "ix": ix, "label": lbl, "valid": valid, "reason": reason,
                              "peak": round(peak, 3), "target": round(target, 3)})
        mixed = any(c["valid"] and c["label"] > 0 for c in cells)
        for c in cells:
            c["dropped"] = bool(mixed and c["valid"] and c["label"] == 0)
            c["used"] = bool(c["valid"] and not c["dropped"])
        return {"cy": cy - py0, "cx": cx - px0, "y0": y0 - py0, "x0": x0 - px0, "cells": cells}

    # candidate samples on the 16px training lattice inside the crop
    best_mixed, best_ring = None, None
    for cy in range(py0 + 3 * T, py1 - 4 * T, T):
        for cx in range(px0 + 3 * T, px1 - 4 * T, T):
            wd = window(cy, cx)
            cs = wd["cells"]
            soft = sum(c["used"] and 0 < c["target"] < 1 for c in cs)
            full = sum(c["used"] and c["target"] == 1 for c in cs)
            drop = sum(c["dropped"] for c in cs)
            gap = sum(c["reason"].startswith("outside") for c in cs)
            neg = sum(c["used"] and c["label"] == 0 for c in cs)
            score_m = min(soft, 4) * 3 + min(full, 3) * 2 + min(drop, 3) * 2 + min(gap, 3)
            if soft and (best_mixed is None or score_m > best_mixed[0]):
                best_mixed = (score_m, wd)
            score_r = min(neg, 10) * 2 + min(gap, 4) * 2
            if neg and not any(c["label"] for c in cs) and (best_ring is None or score_r > best_ring[0]):
                best_ring = (score_r, wd)

    crop = lambda a: a[py0:py1, px0:px1]
    lab_img = np.zeros((py1 - py0, px1 - px0, 4), np.uint8)
    lab_img[crop(labels)] = (255, 170, 90, 255)  # BGRA
    lab_img[crop(drawn)] = (255, 255, 255, 255)
    blur_img = np.zeros((py1 - py0, px1 - px0, 4), np.uint8)
    blur_img[..., 3] = (np.clip(blurred, 0, 1) * 255).astype(np.uint8)
    blur_img[..., :3] = 255

    sl = lambda a: a[ty0:ty1, tx0:tx1].astype(int).tolist()
    data = {
        "scroll": "PHerc0009B",
        "scroll_id": SID,
        "origin": [int(py0), int(px0)],
        "tile": T, "sub": SUB, "grid": GRID, "ctx": CTX,
        "close_r": CLOSE_R, "gap_r": GAP_R, "shell_r": SHELL_R,
        "sigma": SIGMA, "floor": FLOOR,
        "label_png": png_b64(lab_img),
        "blur_png": png_b64(blur_img),
        "tiles": {
            "ink": sl(ink_tile), "seed": sl(seed_tile), "closed": sl(closed),
            "exclusion": sl(exclusion), "ring": sl(ring), "mask": sl(mask_tile),
            "split": sl(tile_any(split, n_ty, n_tx) & ~tile_any(~split, n_ty, n_tx)),
        },
        "counts": {
            "ink_tiles": int(ink_tile[ty0:ty1, tx0:tx1].sum()),
            "closed_tiles": int(closed[ty0:ty1, tx0:tx1].sum()),
            "exclusion_tiles": int(exclusion[ty0:ty1, tx0:tx1].sum()),
            "ring_tiles": int(ring[ty0:ty1, tx0:tx1].sum()),
            "scroll_ink_tiles": int(ink_tile.sum()),
            "scroll_ring_tiles": int(ring.sum()),
        },
        "mixed": best_mixed[1] if best_mixed else None,
        "ringwin": best_ring[1] if best_ring else None,
    }
    out = ROOT / "figures" / "ring_data.js"
    out.write_text("window.RING = " + json.dumps(data) + ";\n")
    print(f"crop px y[{py0},{py1}) x[{px0},{px1}) tiles {ty1 - ty0}x{tx1 - tx0}", data["counts"])
    for key in ("mixed", "ringwin"):
        wd = data[key]
        print(key, wd and [(c["label"], c["target"], c["used"], c["dropped"], c["reason"][:8]) for c in wd["cells"]])


if __name__ == "__main__":
    main()
