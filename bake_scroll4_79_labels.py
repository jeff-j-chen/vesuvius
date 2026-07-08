"""bake_scroll4_79_labels.py — produce the 7.91um (w023, flipped) inklabels + eroded labels
from the 2.4um ink prediction using the validated colored-dot TPS warp.

this is a TWO-STAGE tool so you can hand-correct the warped ink before it is finalized.
note: NOTHING here touches the 2.4um *volume* — we only read the pre-saved 2.4 ink prediction
tif and the pre-saved dot-warp maps. no 2.4 volume download / reconstruction happens.

STAGE A  (default, `python bake_scroll4_79_labels.py`):
  1. load the saved dot-warp maps (target 7.91 @ 3600 width -> source 2.4 @ 3600 width).
  2. downscale the 2.4 ink tif to the source working frame, binarize with a LOW threshold
     (captures faint strokes), then clean it up with morphology (open kills speckle, close
     fills pinholes / bridges tiny gaps inside letters).
  3. remap (warp) the cleaned ink into the flipped-7.91 working frame.
  4. write the working-res warped map to EDIT_PATH for manual correction (small + editable).
     -> paint white to ADD ink, black to REMOVE. keep it strictly 0/255 grayscale.

STAGE B  (`python bake_scroll4_79_labels.py --shrink`):
  5. read the (hand-corrected) EDIT_PATH, UPSCALE to the full 7.91 frame (13303, 31674)
     which matches the reconstructed zarr exactly -> inklabels/<id>.png.
  6. erode -> eroded_inklabels/<id>.png (the "shrunk" labels the trainer consumes).
  7. sanity: report coverage + that dims match the zarr.

the mask is produced by reconstruct_scroll4_7um.py (papyrus signal, already flipped).
"""
import os
import argparse
import numpy as np
import cv2
import tifffile

TMP = r"C:\Users\ChenJeff\Documents\_ves_tmp"
ZARR_ROOT = os.getenv("VESUVIUS_ZARR_PATH", r"C:\Users\ChenJeff\Documents\ves_zarrs2")


def _frame_from_zarr(out_id, override_h=None, override_w=None):
    """read (H,W) from the target zarr so the baked labels match it exactly."""
    if override_h and override_w:
        return int(override_h), int(override_w)
    import json
    za = os.path.join(ZARR_ROOT, f"{out_id}.zarr", ".zarray")
    d = json.load(open(za))
    return int(d["shape"][1]), int(d["shape"][2])


def _load_maps(cfg):
    mapx = np.load(os.path.join(cfg.tmp, f"{cfg.tag}_dotwarp_mapx.npy"))   # (H79h,WW) target->src x
    mapy = np.load(os.path.join(cfg.tmp, f"{cfg.tag}_dotwarp_mapy.npy"))
    H79h, WW = mapx.shape
    # recover the source (2.4) working height at WW from the marked source PNG aspect
    src_png = cv2.imread(cfg.src_mark, cv2.IMREAD_GRAYSCALE)
    if src_png is None:
        raise FileNotFoundError(f"marked source png not found: {cfg.src_mark}")
    H24_mark, W_mark = src_png.shape
    sc = WW / W_mark
    H24h = int(round(H24_mark * sc))
    print(f"[maps] target {H79h}x{WW}  source frame {H24h}x{WW} (sc={sc:.3f})")
    return mapx, mapy, H24h, WW


def warp_stage(cfg):
    """stage A: threshold low + morph clean the 2.4 ink, warp, write editable working-res png"""
    mapx, mapy, H24h, WW = _load_maps(cfg)

    ink = tifffile.imread(cfg.ink_tif)
    if ink.ndim == 3:
        ink = ink[..., 0]
    ink_ws = cv2.resize(ink, (WW, H24h), interpolation=cv2.INTER_AREA)
    inkb = (ink_ws > cfg.ink_thr).astype(np.uint8) * 255
    print(f"[ink] thr({cfg.ink_thr}) source frac {float((inkb>0).mean()):.3f}")

    # morphological cleanup: open (despeckle) then close (fill pinholes / bridge gaps)
    inkb = cv2.morphologyEx(inkb, cv2.MORPH_OPEN,
                            np.ones((cfg.open_k, cfg.open_k), np.uint8), iterations=cfg.open_iters)
    inkb = cv2.morphologyEx(inkb, cv2.MORPH_CLOSE,
                            np.ones((cfg.close_k, cfg.close_k), np.uint8), iterations=cfg.close_iters)
    print(f"[clean] after open/close source frac {float((inkb>0).mean()):.3f}")

    ink_warp = cv2.remap(inkb, mapx, mapy, cv2.INTER_NEAREST, borderValue=0)
    ink_warp = (ink_warp > 127).astype(np.uint8) * 255
    print(f"[warp] working-res warped frac {float((ink_warp>0).mean()):.3f}")

    os.makedirs(cfg.tmp, exist_ok=True)
    cv2.imwrite(cfg.edit_path, ink_warp)
    print(f"[edit] wrote {cfg.edit_path}  ({ink_warp.shape[1]}x{ink_warp.shape[0]}, 0/255 grayscale)")
    print("[next] hand-correct that file (white=add ink, black=remove), then run the same")
    print(f"       command with --shrink")


def shrink_stage(cfg):
    """stage B: read corrected working-res png, upscale to full frame, erode -> final labels"""
    warp = cv2.imread(cfg.edit_path, cv2.IMREAD_GRAYSCALE)
    if warp is None:
        raise FileNotFoundError(f"edit file not found at {cfg.edit_path}; run stage A first")
    warp = (warp > 127).astype(np.uint8) * 255
    print(f"[edit] loaded {cfg.edit_path}  ({warp.shape[1]}x{warp.shape[0]}) "
          f"frac {float((warp>0).mean()):.3f}")

    FRAME_H, FRAME_W = _frame_from_zarr(cfg.out_id, cfg.frame_h, cfg.frame_w)
    ink_full = cv2.resize(warp, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
    ink_full = (ink_full > 127).astype(np.uint8) * 255

    os.makedirs("inklabels", exist_ok=True)
    os.makedirs("eroded_inklabels", exist_ok=True)
    cv2.imwrite(f"inklabels/{cfg.out_id}.png", ink_full)
    eroded = cv2.erode(ink_full, np.ones((cfg.erode_k, cfg.erode_k), np.uint8), iterations=cfg.erode_iters)
    cv2.imwrite(f"eroded_inklabels/{cfg.out_id}.png", eroded)
    print(f"[save] inklabels frac {float((ink_full>0).mean()):.3f}  "
          f"eroded frac {float((eroded>0).mean()):.3f}  dims {ink_full.shape} (want {(FRAME_H, FRAME_W)})")
    assert ink_full.shape == (FRAME_H, FRAME_W)
    print(f"[done] {cfg.out_id} inklabels + eroded written")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shrink", action="store_true",
                    help="stage B: consume the hand-corrected edit file -> final + eroded labels")
    ap.add_argument("--out-id", default="20240304161941", help="segment id (label + zarr filename)")
    ap.add_argument("--tag", default=None, help="dot-warp map tag (default: derived from out-id)")
    ap.add_argument("--tmp", default=TMP)
    ap.add_argument("--ink-tif", default=os.path.join(TMP, "w023_ink_full.tif"),
                    help="2.4um ink prediction tif (label source)")
    ap.add_argument("--src-mark", default=r"C:\Users\ChenJeff\Documents\warp_MARK_2p4_source.png",
                    help="marked 2.4um source png (used to recover source aspect)")
    ap.add_argument("--edit-path", default=None,
                    help="editable working-res warped png (default: <tmp>/<tag>_warp_edit.png)")
    ap.add_argument("--frame-h", type=int, default=None, help="override frame H (else read from zarr)")
    ap.add_argument("--frame-w", type=int, default=None, help="override frame W (else read from zarr)")
    ap.add_argument("--ink-thr", type=int, default=99)
    ap.add_argument("--open-k", type=int, default=3); ap.add_argument("--open-iters", type=int, default=2)
    ap.add_argument("--close-k", type=int, default=3); ap.add_argument("--close-iters", type=int, default=2)
    ap.add_argument("--erode-k", type=int, default=3); ap.add_argument("--erode-iters", type=int, default=6)
    cfg = ap.parse_args()
    # derived defaults
    if cfg.tag is None:
        cfg.tag = "w023" if cfg.out_id == "20240304161941" else cfg.out_id
    if cfg.edit_path is None:
        cfg.edit_path = os.path.join(cfg.tmp, f"{cfg.tag}_warp_edit.png")

    if cfg.shrink:
        shrink_stage(cfg)
    else:
        warp_stage(cfg)


if __name__ == "__main__":
    main()
