"""sweep_scroll4_full.py — run a trained dense_unet across ENTIRE scroll fragments.

pure inference (no training, no backprop, no memorization possible outside the tiny
training crop). renders, per scroll:
  - <id>_pred.png          prediction-only depth-MAX composite (magma)
  - <id>_pred_overlay.png  prediction with eroded inklabels overlaid in gold
                           (same style as the eval_int figures)

reuses the exact fully-convolutional Hann-free chunk inference from
utils/visualizer.add_dense_evaluation_figure: read the volume in CH-sized blocks,
sigmoid the model, immediately downsample each block into a ~canvas-max canvas so the
whole ~14k x 30k frame never lives in memory at full res. depth-MAX over all depth
blocks gives the best-depth response per pixel.

MEMORY SAFETY:
  - only a downsampled canvas (~1.3k x 2.8k float32, tens of MB) is held per depth block
  - each raw block read is D x CH x CH (~18 MB at CH=768, D=8)
  - nothing full-res is ever materialized

usage:
  .venv\\Scripts\\python.exe sweep_scroll4_full.py --checkpoint models/best_model_f1.pth \\
      --scroll-ids 20240304161941 20240304144031 --arch dense_unet --depth 8 \\
      --d-start 0 --d-end 64 --out-dir sweeps
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import cv2
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config
from utils.model import create_model
from utils.training_utils import load_model

try:
    from torch.amp import autocast
    def _autocast(dev):
        return autocast(dev if isinstance(dev, str) else str(dev))
except Exception:
    from torch.cuda.amp import autocast as _cuda_autocast
    def _autocast(dev):
        return _cuda_autocast()

NORM_CACHE = "./norm_cache.json"


def load_norm(scroll_id):
    """read mean/std/min/max from norm_cache.json for this scroll id"""
    with open(NORM_CACHE, "r") as f:
        cache = json.load(f)
    e = cache.get(str(scroll_id))
    if not (isinstance(e, dict) and all(k in e for k in ("mean", "std", "min", "max"))):
        raise RuntimeError(f"no norm stats for {scroll_id} — run precompute_norm.py first")
    return float(e["mean"]), float(e["std"]), float(e["min"]), float(e["max"])


def sweep_one(model, cfg, scroll_id, out_dir, canvas_max, chunk, thermal_ms):
    import zarr, time
    dev = cfg.device
    D = int(cfg.data.depth)
    zf0 = int(cfg.data.d_start)

    zarr_dir = os.path.join(cfg.data.zarr_path, f"{scroll_id}.zarr")
    vol = zarr.open(zarr_dir, mode="r")
    Z, H, W = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
    zf1 = min(int(cfg.data.d_end), Z)

    g_mean, g_std, g_min, g_max = load_norm(scroll_id)

    # mask (full-res, binary). resize if the png resolution differs from the zarr frame.
    mk = cv2.imread(f"./masks/{scroll_id}.png", cv2.IMREAD_GRAYSCALE)
    if mk is None:
        raise FileNotFoundError(f"masks/{scroll_id}.png not found")
    if mk.shape != (H, W):
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
    # eroded inklabels for the overlay (optional — may be absent for inference-only frames)
    lab = cv2.imread(f"./eroded_inklabels/{scroll_id}.png", cv2.IMREAD_GRAYSCALE)
    if lab is not None and lab.shape != (H, W):
        lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)

    # downsample factor so the whole frame fits a canvas_max-px canvas
    DS = max(1, int(np.ceil(max(H, W) / float(canvas_max))))
    Hc, Wc = H // DS, W // DS

    # chunk size: divisible by 8 (3-pool u-net) AND by DS (so downsampled placement tiles)
    CH = (int(chunk) // 8) * 8
    while CH % DS != 0:
        CH -= 8
    CH = max(CH, 8 * DS)

    # depth blocks: full inference depth swept in blocks of D (depth-max composite)
    stride_z = max(1, D)
    z_starts = list(range(zf0, max(zf0 + 1, zf1 - D + 1), stride_z))
    if not z_starts:
        z_starts = [zf0]
    if z_starts[-1] != zf1 - D and (zf1 - D) >= zf0:
        z_starts.append(zf1 - D)

    def _norm(blk):
        blk = (blk - g_mean) / g_std
        return np.clip((blk - g_min) / (g_max - g_min + 1e-12), 0, 1)

    def _pad8(a):
        _, h, w = a.shape
        ph = (-h) % 8; pw = (-w) % 8
        if ph or pw:
            a = np.pad(a, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        return a, h, w

    mask_ds = cv2.resize((mk > 0).astype(np.float32), (Wc, Hc), interpolation=cv2.INTER_AREA)
    mask_ds = (mask_ds > 0.5).astype(np.float32)

    ys = list(range(0, H, CH))
    xs = list(range(0, W, CH))
    total = len(z_starts) * len(ys) * len(xs)
    done = 0
    t0 = time.time()

    model.eval()
    composite = np.zeros((Hc, Wc), np.float32)
    with torch.no_grad():
        for z0 in z_starts:
            canvas = np.zeros((Hc, Wc), np.float32)
            for yy in ys:
                ch = min(CH, H - yy)
                for xx in xs:
                    cw = min(CH, W - xx)
                    done += 1
                    blk = np.asarray(vol[z0:z0 + D, yy:yy + ch, xx:xx + cw]).astype(np.float32)
                    if blk.shape[0] != D:
                        continue
                    # skip fully-masked-out (air) chunks — cheap + avoids wasted forward passes
                    myy, mxx = (yy // DS), (xx // DS)
                    mhh, mww = (ch // DS), (cw // DS)
                    if mhh >= 1 and mww >= 1 and mask_ds[myy:myy + mhh, mxx:mxx + mww].max() < 0.5:
                        continue
                    blk = _norm(blk)
                    blk, oh, ow = _pad8(blk)
                    bt = torch.from_numpy(blk).unsqueeze(0).unsqueeze(0).float().to(dev)
                    with _autocast(dev):
                        p = torch.sigmoid(model(bt))[0, 0, :oh, :ow].float().cpu().numpy()
                    cyd, cxd = (yy // DS), (xx // DS)
                    chd, cwd = oh // DS, ow // DS
                    if chd < 1 or cwd < 1:
                        continue
                    pd = cv2.resize(p, (cwd, chd), interpolation=cv2.INTER_AREA)
                    canvas[cyd:cyd + chd, cxd:cxd + cwd] = pd
                    if thermal_ms > 0:
                        time.sleep(thermal_ms / 1000.0)
            canvas *= mask_ds
            composite = np.maximum(composite, canvas)
            el = time.time() - t0
            print(f"  [{scroll_id}] depth z{z0}-{z0+D} done  ({done}/{total} chunks, {el:.0f}s)")

    os.makedirs(out_dir, exist_ok=True)

    # 1) prediction-only (magma)
    pred_u8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
    pred_rgb = cv2.applyColorMap(pred_u8, cv2.COLORMAP_MAGMA)  # BGR
    out_pred = os.path.join(out_dir, f"{scroll_id}_pred.png")
    cv2.imwrite(out_pred, pred_rgb)

    # 2) prediction + eroded inklabels overlay (gold, semi-transparent) — eval-style
    out_overlay = None
    if lab is not None:
        gt_ds = cv2.resize((lab > 127).astype(np.float32), (Wc, Hc), interpolation=cv2.INTER_AREA)
        gt_ds = (gt_ds > 0.3).astype(np.float32)
        overlay = pred_rgb.copy()
        gold = np.array([0, 200, 255], dtype=np.float32)  # BGR gold
        m = gt_ds > 0.5
        overlay[m] = (0.45 * overlay[m].astype(np.float32) + 0.55 * gold).astype(np.uint8)
        out_overlay = os.path.join(out_dir, f"{scroll_id}_pred_overlay.png")
        cv2.imwrite(out_overlay, overlay)

    print(f"[sweep] {scroll_id}: canvas={Hc}x{Wc} DS={DS} CH={CH} "
          f"depth-blocks={z_starts}\n    -> {out_pred}"
          + (f"\n    -> {out_overlay}" if out_overlay else ""))
    return out_pred, out_overlay


def main():
    ap = argparse.ArgumentParser(description="full-scroll inference sweep")
    ap.add_argument("--checkpoint", required=True, help="path to .pth state dict")
    ap.add_argument("--scroll-ids", nargs="+", required=True, type=str)
    ap.add_argument("--arch", default="dense_unet")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--d-start", type=int, default=0)
    ap.add_argument("--d-end", type=int, default=64)
    ap.add_argument("--out-dir", default="sweeps")
    ap.add_argument("--canvas-max", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=768)
    ap.add_argument("--thermal-ms", type=float, default=0.0,
                    help="sleep between chunks (ms) for thermal relief")
    args = ap.parse_args()

    cfg = Config()
    cfg.model.arch = args.arch
    cfg.data.depth = args.depth
    cfg.data.d_start = args.d_start
    cfg.data.d_end = args.d_end
    cfg.data.tile_size = 32

    print(f"[sweep] device={cfg.device} arch={args.arch} ckpt={args.checkpoint}")
    model, _ = create_model(cfg)
    load_model(model, args.checkpoint)
    model.eval()

    for sid in args.scroll_ids:
        print(f"\n=== sweeping {sid} ===")
        sweep_one(model, cfg, sid, args.out_dir, args.canvas_max, args.chunk, args.thermal_ms)

    print("\n[sweep] all done")


if __name__ == "__main__":
    main()
