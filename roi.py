"""roi.py -- interactive probe-ROI picker for the eroded_inklabels maps.

click the TOP-LEFT of where you want a 576x576 probe window; it expands down+right.
  '1' -> easy mode (green)   default
  '2' -> hard mode (red)
  click again in the same mode -> replaces that mode's box
  space -> save this scroll's boxes to the unified probe_rois.json and advance
  q     -> save the current scroll and quit
  esc   -> abort (current image NOT saved)

the dotted yellow line is the train/valid split read from utils.config (per-scroll axis/frac),
so it matches the dataloader's split exactly.

boxes are stored in FULL-RES pixel coords (top-left x,y + size) in ONE cache file
(probe_rois.json, keyed by scroll id -- like norm_cache.json), shared with the visualizer.
the top-left is snapped to a 96px grid = LCM(16,32,48) so the window aligns to whatever model
tile grid is used (tile 16, context 32, or context 48). 576 = 96*6 is divisible by all three.

usage: python roi.py
"""
from __future__ import annotations
import cv2
import glob
import json
import os

import numpy as np

ERODED_DIR = "eroded_inklabels"
CACHE = "probe_rois.json"   # single unified probe-ROI cache (keyed by scroll id)
ROI_SIZE = 576          # full-res probe side; divisible by 16/32/48
GRID = 96               # LCM(16,32,48): snap top-left so the roi aligns to any model grid
MAX_W = 1600            # display downscale cap (screen width)
MAX_H = 900             # display downscale cap (screen height)
BORDER = 2              # rect line thickness, drawn ON the roi boundary (no size bloat)
INK_OVERLAY_DIR = os.path.join("inklabels", "2_4um")  # raw 2.4um labels shown faintly behind ROIs
INK_OVERLAY_ALPHA = 0.2                                # 1/5 intensity


def _box_color(mode: str, is_new: bool):
    """BGR color for a box: FULL for ones drawn this session, HALF for json-loaded ones."""
    if mode == "easy":
        return (0, 255, 0) if is_new else (0, 128, 0)   # green
    return (0, 0, 255) if is_new else (0, 0, 128)        # red


VAL_COLOR = (0, 255, 255)   # BGR yellow: the train/valid split line


def _dotted_vline(img, x, h, color, dash=9, gap=6, t=1):
    y = 0
    while y < h:
        cv2.line(img, (x, y), (x, min(y + dash, h)), color, t)
        y += dash + gap


def _dotted_hline(img, y, w, color, dash=9, gap=6, t=1):
    x = 0
    while x < w:
        cv2.line(img, (x, y), (min(x + dash, w), y), color, t)
        x += dash + gap


def _load_split_map():
    """scroll_id -> (axis, frac) from the training Config so the drawn validation line
    matches the dataloader's train/valid split exactly. returns (map, tile_size)."""
    try:
        from utils.config import Config
    except Exception as e:
        print(f"(config unavailable, skipping validation line: {e})")
        return {}, 16
    c = Config()
    return ({int(sc.scroll_id): (str(sc.split_axis).lower(), float(sc.train_split_frac))
             for sc in c.data.scrolls}, int(c.data.tile_size))


def _snap(v: float) -> int:
    """snap a full-res coordinate down to the 96px grid"""
    return (int(v) // GRID) * GRID


def _load_cache() -> dict:
    """read the whole unified probe-ROI cache ({sid_str: {easy,hard}}); {} if absent."""
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    """write the whole unified probe-ROI cache back to disk (sorted keys for stable diffs)."""
    with open(CACHE, "w") as f:
        json.dump({k: cache[k] for k in sorted(cache)}, f, indent=2)


def _render(base_bgr, boxes: dict, scale: float, mode: str, name: str, new_modes: set,
            vline=None, hline=None):
    """draw the train/valid split line + all stored boxes + a status header onto a fresh
    copy of the display image. boxes drawn this session (in new_modes) render FULL color;
    json-loaded ones render HALF."""
    img = base_bgr.copy()
    # dotted validation line (train/valid split from config); None when scroll not in config
    if vline is not None:
        _dotted_vline(img, vline, img.shape[0], VAL_COLOR)
    if hline is not None:
        _dotted_hline(img, hline, img.shape[1], VAL_COLOR)
    for m, b in boxes.items():
        if m not in ("easy", "hard"):
            continue
        x = int(b["x"] * scale)
        y = int(b["y"] * scale)
        s = int(b["size"] * scale)
        # line straddles the roi boundary [x, x+s] so the box shows the true 576 extent
        cv2.rectangle(img, (x, y), (x + s, y + s), _box_color(m, m in new_modes), BORDER)
    header = f"{name}  mode={mode}  (1=easy 2=hard  space=save+next  q=save+quit  esc=abort)"
    cv2.putText(img, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _on_click(event, x, y, flags, params):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    scale = params["scale"]
    W, H = params["full_wh"]
    # click = top-left in DISPLAY px -> divide by scale to full-res, snap to the 96 grid.
    # the in-frame clamp bound is ALSO snapped so an edge click never stores an off-grid x/y
    # (keeps roi.py's drawn box byte-consistent with the visualizer's aligned window).
    max_x = max(0, ((W - ROI_SIZE) // GRID) * GRID)
    max_y = max(0, ((H - ROI_SIZE) // GRID) * GRID)
    fx = min(max_x, _snap(x / scale))
    fy = min(max_y, _snap(y / scale))
    mode = params["mode"]
    params["boxes"][mode] = {"x": int(fx), "y": int(fy), "size": ROI_SIZE}
    params["new_modes"].add(mode)   # drawn this session -> full color
    params["dirty"] = True
    print(f"  [{mode}] roi top-left=({fx},{fy}) size={ROI_SIZE}")
    cv2.imshow("roi", _render(params["base"], params["boxes"], scale, mode, params["name"],
                              params["new_modes"], params["vline"], params["hline"]))


def main():
    pngs = sorted(glob.glob(os.path.join(ERODED_DIR, "*.png")))
    if not pngs:
        print(f"no pngs found in {ERODED_DIR}/")
        return
    print(f"found {len(pngs)} scroll(s) in {ERODED_DIR}/")

    cv2.namedWindow("roi", cv2.WINDOW_AUTOSIZE)
    split_map, T = _load_split_map()
    cache = _load_cache()   # unified probe_rois.json, mutated in place and rewritten on save

    for png in pngs:
        name = os.path.basename(png)
        gray = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skipping unreadable {name}")
            continue
        H, W = gray.shape[:2]
        stem = os.path.splitext(name)[0]
        # weakly overlay the raw 2.4um inklabel (scaled to this map's size) at 1/5 intensity, so
        # the fuller labels show faintly behind the tight (eroded) 'confident' labels
        ink = cv2.imread(os.path.join(INK_OVERLAY_DIR, f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if ink is not None:
            if ink.shape[:2] != (H, W):
                ink = cv2.resize(ink, (W, H), interpolation=cv2.INTER_AREA)
            gray = np.clip(gray.astype(np.float32) + INK_OVERLAY_ALPHA * ink.astype(np.float32),
                           0, 255).astype(np.uint8)
        scale = min(1.0, MAX_W / float(W), MAX_H / float(H))
        disp_w, disp_h = int(W * scale), int(H * scale)
        base = cv2.cvtColor(cv2.resize(gray, (disp_w, disp_h)), cv2.COLOR_GRAY2BGR)

        sid_str = os.path.splitext(name)[0]
        boxes = dict(cache.get(sid_str, {}))   # copy so edits only commit on save
        mode = "easy"

        # train/valid split line from config (full-res split snapped to tile -> display px).
        # none when the scroll is not in the training set (e.g. the holdout).
        vline = hline = None
        try:
            sid = int(os.path.splitext(name)[0])
        except ValueError:
            sid = None
        info = split_map.get(sid)
        if info:
            axis, frac = info
            if axis == "x":
                split = (int(((W // T) * T) * frac) // T) * T
                vline = int(split * scale)
            else:
                split = (int(((H // T) * T) * frac) // T) * T
                hline = int(split * scale)

        print(f"\n{name}  ({W}x{H} -> {disp_w}x{disp_h}, scale={scale:.3f})"
              f"  existing={list(boxes.keys())}  "
              f"split={('x@'+str(vline)) if vline is not None else (('y@'+str(hline)) if hline is not None else 'none')}")

        params = {
            "base": base, "boxes": boxes, "scale": scale, "mode": mode,
            "full_wh": (W, H), "name": name, "dirty": False, "new_modes": set(),
            "vline": vline, "hline": hline,
        }
        cv2.imshow("roi", _render(base, boxes, scale, mode, name, params["new_modes"], vline, hline))
        cv2.setMouseCallback("roi", _on_click, params)

        action = "next"
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("1"):
                params["mode"] = "easy"
                print("  mode -> easy")
                cv2.imshow("roi", _render(base, params["boxes"], scale, "easy", name, params["new_modes"], vline, hline))
            elif key == ord("2"):
                params["mode"] = "hard"
                print("  mode -> hard")
                cv2.imshow("roi", _render(base, params["boxes"], scale, "hard", name, params["new_modes"], vline, hline))
            elif key == ord(" "):
                action = "next"; break
            elif key == ord("q"):
                action = "quit"; break
            elif key == 27:
                action = "abort"; break

        # esc discards the current image's edits and stops entirely
        if action == "abort":
            print("aborted (current image not saved).")
            break

        # space and q both persist the current image first
        if params["dirty"]:
            cache[sid_str] = params["boxes"]
            _save_cache(cache)
            print(f"  saved {CACHE} [{sid_str}]: {params['boxes']}")
        else:
            print("  no changes; nothing written")

        if action == "quit":
            print("quit (progress saved).")
            break

    cv2.destroyAllWindows()
    print("\ndone.")


if __name__ == "__main__":
    main()
