"""mae_pretrain_twostage.py -- masked-autoencoder pretraining for the v15 two-stage backbone.

WHY (lever 1): every supervised run learns the papyrus representation AND the ink boundary
at once from a tiny, noisy label set -- the worst case for a faint signal. MAE splits the
problems: first learn "what 9.4um papyrus/fiber/depth texture looks like" from MILLIONS of
UNLABELED crops (reconstruct masked voxels), then fine-tune with the ink head on top of an
already-rich feature space. this regularizes by INITIALIZATION, not by cutting capacity --
the escape from the "overfits but any regularizer kills it" paradox.

WHAT IT PRETRAINS: the shared stage-1 backbone of the two-stage model (per_slice stem +
depth positional encoding + depth_mix + voxel head). the checkpoint keys are prefixed
`stage1.*`, so train.py's `--init-weights` / config.init_weights warm-start transfers them
straight into the two-stage model (stage2 fusion stays fresh). the MAE decoder is dropped.

WHAT IT DOES:
  - samples random tile_size x tile_size x depth (8-slice) crops from the raw zarrs, ONLY
    from each scroll's TRAINING region so the held-out area stays genuinely unseen
  - the z-window start is randomized within [train_d_start, train_d_end-depth]; the matching
    absolute depth offset is fed to stage1.encode so depth_pe is pretrained across the band
  - masks ~mask_frac of each crop as patch blocks (filled with the visible mean), encodes,
    decodes back to the raw crop, MSE loss on the MASKED voxels only (standard MAE)
  - a SCATTERED (not geographic) train/monitor split lets MAE/recon_mse_heldout flag
    memorization on the same texture distribution

TRANSFER (fine-tune):
  python train.py -n ink_from_mae            # then in a campaign set:
  c.init_weights = "models/mae_twostage.pth"  # OR add --init-weights if you wire a CLI flag
  the loader is strict=False: stage1.* transfers, decoder.* is skipped, stage2.* is fresh.

WHAT TO EXPECT: MAE/recon_mse_train falls then plateaus (it only means it learned to inpaint
papyrus texture -- NOT an ink metric). heldout should track train (a big gap = memorizing;
lower mask_frac or add scrolls). the verdict is the FINE-TUNE: does warm-start give sharper
held-out letters than from scratch?
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import Config
from utils.model import create_model

try:
    from torch.amp import autocast as _ac, GradScaler as _GS
    def _autocast(dev): return _ac(dev if isinstance(dev, str) else str(dev))
    def _scaler(dev): return _GS(dev if isinstance(dev, str) else str(dev))
except Exception:
    from torch.cuda.amp import autocast as _acc, GradScaler as _GSS
    def _autocast(dev): return _acc()
    def _scaler(dev): return _GSS()

NORM_CACHE = "./norm_cache.json"


def load_norm(scroll_id):
    with open(NORM_CACHE) as f:
        e = json.load(f).get(str(scroll_id))
    if not (isinstance(e, dict) and all(k in e for k in ("mean", "std", "min", "max"))):
        raise RuntimeError(f"no norm for {scroll_id}; run precompute_norm.py first")
    return float(e["mean"]), float(e["std"]), float(e["min"]), float(e["max"])


class CropSampler:
    """samples normalized (D,T,T) raw crops from one scroll, with a scattered train/monitor
    split so the recon-mse gap reflects memorization (same texture distribution), not a
    geographic confound. returns (batch, z_start) so the caller can pass the matching
    absolute depth offset into stage1.encode (pretrains depth_pe across the real band)."""
    def __init__(self, cfg, scroll_id, y_lo, y_hi, x_lo, x_hi, role="train",
                 holdout_frac=0.1, block_px=512, min_cover=0.6):
        import zarr, cv2
        self.vol = zarr.open(os.path.join(cfg.data.zarr_path, f"{scroll_id}.zarr"), mode="r")
        self.T = int(cfg.data.tile_size)
        self.D = int(cfg.data.depth)
        self.z0 = int(cfg.data.train_d_start)
        self.z1 = int(cfg.data.train_d_end)
        self.y_lo, self.y_hi = y_lo, y_hi
        self.x_lo, self.x_hi = x_lo, x_hi
        self.role = role
        self.holdout_mod = max(2, int(round(1.0 / max(holdout_frac, 1e-6))))
        self.block_px = int(block_px)
        self.min_cover = min_cover
        m = cv2.imread(f"./masks/{scroll_id}.png", cv2.IMREAD_GRAYSCALE)
        H, W = int(self.vol.shape[1]), int(self.vol.shape[2])
        if m is None:
            m = np.full((H, W), 255, np.uint8)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        self.mask = (m > 0).astype(np.uint8)
        self.g_mean, self.g_std, self.g_min, self.g_max = load_norm(scroll_id)

    def _norm(self, blk):
        blk = (blk - self.g_mean) / self.g_std
        return np.clip((blk - self.g_min) / (self.g_max - self.g_min + 1e-12), 0, 1)

    def _block_is_holdout(self, yy, xx):
        by, bx = yy // self.block_px, xx // self.block_px
        return ((by * 73856093) ^ (bx * 19349663)) % self.holdout_mod == 0

    def batch(self, n, rng):
        T, D = self.T, self.D
        # random z-window start; z0 is the matching absolute depth offset for depth_pe
        z = self.z0 if (self.z1 - self.z0) <= D else int(rng.integers(self.z0, self.z1 - D + 1))
        out, tries = [], 0
        want_holdout = (self.role == "monitor")
        while len(out) < n and tries < n * 40:
            tries += 1
            yy = int(rng.integers(self.y_lo, self.y_hi - T))
            xx = int(rng.integers(self.x_lo, self.x_hi - T))
            if self._block_is_holdout(yy, xx) != want_holdout:
                continue
            if self.mask[yy:yy + T, xx:xx + T].mean() < self.min_cover:
                continue
            blk = np.asarray(self.vol[z:z + D, yy:yy + T, xx:xx + T]).astype(np.float32)
            if blk.shape != (D, T, T):
                continue
            out.append(self._norm(blk))
        if not out:
            return None, z
        arr = np.stack(out)                                  # (n,D,T,T)
        return torch.from_numpy(arr).unsqueeze(1).float(), z  # (n,1,D,T,T), z_start


class MultiSampler:
    """round-robin over per-scroll CropSamplers so each batch mixes scrolls. all sub-samplers
    share the drawn z (depth offset) for the batch so a single offset is well-defined."""
    def __init__(self, samplers):
        self.samplers = [s for s in samplers if s is not None]

    def batch(self, n, rng):
        if not self.samplers:
            return None, 0
        parts, z_used = [], 0
        per = max(1, n // len(self.samplers))
        for s in self.samplers:
            b, z = s.batch(per, rng)
            if b is not None:
                parts.append(b); z_used = z
        if not parts:
            return None, 0
        return torch.cat(parts, dim=0), z_used


def make_mask(B, T, patch, frac, dev, rng):
    """block-random mask over a TxT image: (B,1,T,T), 1 = MASKED (hidden)."""
    g = T // patch
    n = g * g
    k = max(1, int(round(frac * n)))
    m = torch.zeros(B, n, device=dev)
    for b in range(B):
        idx = torch.from_numpy(rng.choice(n, size=k, replace=False)).to(dev)
        m[b, idx] = 1.0
    m = m.view(B, 1, g, g)
    return torch.nn.functional.interpolate(m, scale_factor=patch, mode="nearest")


def apply_mask(xb, mask):
    """fill masked voxels with each sample's VISIBLE mean (mask-token proxy). zero-fill would
    create hard data edges the conv learns to trace; the visible mean removes that edge and
    leaks no hidden info. mask (B,1,T,T) 1=hidden; xb (B,1,D,T,T)."""
    m = mask.unsqueeze(2)
    vis = xb * (1.0 - m)
    denom = (1.0 - m).sum(dim=(1, 2, 3, 4), keepdim=True) + 1e-8
    fill = vis.sum(dim=(1, 2, 3, 4), keepdim=True) / denom
    return xb * (1.0 - m) + fill * m


class TwoStageMAE(nn.Module):
    """stage-1 backbone (encoder) + a light conv decoder that reconstructs the raw crop.
    only stage1.* is kept for warm-start; the decoder is throwaway."""
    def __init__(self, stage1):
        super().__init__()
        self.stage1 = stage1
        # encode() returns (B,256,D,T/2,T/2); depth_mix pooled H,W by 2 -> upsample back.
        self.decoder = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1, bias=False), nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
            nn.Conv3d(128, 64, 3, padding=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, 1, bias=True),
        )

    def forward(self, x, depth_offset=0):
        feat = self.stage1.encode(x, depth_offset=depth_offset)  # (B,256,D,T/2,T/2)
        rec = self.decoder(feat)                                  # (B,1,D,T,T)
        return rec.squeeze(1)                                     # (B,D,T,T)


def _save_fig(writer, step, target, xin, pred, mask):
    """log a (target | masked-input | reconstruction) triptych for the depth-mean image."""
    try:
        t = target[0].mean(0).detach().float().cpu().numpy()
        i = xin[0, 0].mean(0).detach().float().cpu().numpy()
        p = pred[0].mean(0).detach().float().cpu().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a, im, ti in zip(ax, [t, i, p], ["target", "masked input", "reconstruction"]):
            a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(ti); a.axis("off")
        plt.tight_layout()
        writer.add_figure("MAE/recon", fig, step)
        plt.close(fig)
    except Exception as e:
        print(f"[mae] fig failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="MAE pretraining for the v15 two-stage backbone")
    ap.add_argument("-n", "--name", default="mae_twostage")
    ap.add_argument("--scroll-ids", type=int, nargs="+", default=None,
                    help="scroll ids to sample UNLABELED crops from (default: all DEFAULT_SCROLLS)")
    ap.add_argument("--arch", default="v15_twostage_wide_zgrad",
                    help="two-stage arch whose stage-1 backbone to pretrain")
    ap.add_argument("--tile-size", type=int, default=64,
                    help="pretrain crop size; larger than the 16px fine-tune tile gives more "
                         "inpainting context. the backbone is fully-conv so weights transfer to 16px.")
    ap.add_argument("--depth", type=int, default=8, help="crop depth = stage-1 window (8)")
    ap.add_argument("--train-d-start", type=int, default=4)
    ap.add_argument("--train-d-end", type=int, default=28)
    ap.add_argument("--train-frac", type=float, default=0.75,
                    help="sample crops only from this leading fraction of each frame (its train "
                         "region) along the scroll's split axis, so the held-out area stays unseen")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--holdout-block", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--mask-frac", type=float, default=0.75)
    ap.add_argument("--mask-patch", type=int, default=8)
    ap.add_argument("--log-int", type=int, default=50)
    ap.add_argument("--fig-int", type=int, default=500)
    ap.add_argument("--save-int", type=int, default=1000)
    ap.add_argument("--step-cooldown-ms", type=float, default=0.0)
    ap.add_argument("--cooldown-secs", type=float, default=0.0)
    ap.add_argument("--cooldown-int", type=int, default=200)
    ap.add_argument("--log-dir", default="runs_mae")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config()
    cfg.model.arch = args.arch
    cfg.data.tile_size = args.tile_size
    cfg.data.depth = args.depth
    cfg.data.train_d_start = args.train_d_start
    cfg.data.train_d_end = args.train_d_end
    dev = cfg.device

    # resolve scroll ids + per-scroll split (so crops come from the TRAIN region only)
    from utils.config import DEFAULT_SCROLLS
    split_by_id = {int(s.scroll_id): (s.split_axis, float(s.train_split_frac)) for s in DEFAULT_SCROLLS}
    scroll_ids = args.scroll_ids if args.scroll_ids else [int(s.scroll_id) for s in DEFAULT_SCROLLS]

    import zarr
    T = args.tile_size
    rng = np.random.default_rng(args.seed)
    train_samplers, held_samplers = [], []
    for sid in scroll_ids:
        try:
            vol = zarr.open(os.path.join(cfg.data.zarr_path, f"{sid}.zarr"), mode="r")
        except Exception as e:
            print(f"[mae] skip {sid}: {e}"); continue
        H, W = int(vol.shape[1]), int(vol.shape[2])
        axis, frac = split_by_id.get(int(sid), ("x", args.train_frac))
        # restrict crops to the TRAIN region along the scroll's split axis
        if axis == "y":   # horizontal split: train = top frac rows
            y0, y1 = 0, (int(H * frac) // T) * T
            x0, x1 = 0, (W // T) * T
        else:             # vertical split: train = left frac cols
            y0, y1 = 0, (H // T) * T
            x0, x1 = 0, (int(W * frac) // T) * T
        if (y1 - y0) < T or (x1 - x0) < T:
            print(f"[mae] skip {sid}: train region too small ({y1-y0}x{x1-x0})"); continue
        print(f"[mae] {sid} frame {H}x{W} axis={axis} frac={frac} -> train y[{y0},{y1}] x[{x0},{x1}]")
        train_samplers.append(CropSampler(cfg, sid, y0, y1, x0, x1, role="train",
                                          holdout_frac=args.holdout_frac, block_px=args.holdout_block))
        held_samplers.append(CropSampler(cfg, sid, y0, y1, x0, x1, role="monitor",
                                         holdout_frac=args.holdout_frac, block_px=args.holdout_block))
    if not train_samplers:
        raise RuntimeError("no usable scrolls for MAE pretraining")

    train_s, held_s = MultiSampler(train_samplers), MultiSampler(held_samplers)

    # build the two-stage model, steal its stage-1 backbone as the MAE encoder
    full, _ = create_model(cfg)
    if not hasattr(full, "stage1") or not hasattr(full.stage1, "encode"):
        raise RuntimeError(f"arch {args.arch} has no stage1.encode(); use a v15_twostage_* arch")
    model = TwoStageMAE(full.stage1).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = _scaler(dev)

    warmup = max(1, int(args.warmup_frac * args.steps))
    min_lr = args.min_lr_frac
    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, args.steps - warmup)
        return min_lr + (1.0 - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)

    from torch.utils.tensorboard import SummaryWriter
    run_dir = os.path.join(args.log_dir, f"{args.name}_{time.strftime('%m%d_%H-%M-%S')}")
    writer = SummaryWriter(run_dir)
    os.makedirs("models", exist_ok=True)
    print(f"[mae] arch={args.arch} scrolls={len(train_samplers)} tile={T} depth={args.depth} "
          f"-> logging {run_dir}")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb, z = train_s.batch(args.batch_size, rng)
        if xb is None:
            continue
        xb = xb.to(dev)
        target = xb[:, 0]                                             # (B,D,T,T) raw target
        mask = make_mask(xb.shape[0], T, args.mask_patch, args.mask_frac, dev, rng)
        xin = apply_mask(xb, mask)

        opt.zero_grad(set_to_none=True)
        with _autocast(dev):
            pred = model(xin, depth_offset=z)                        # (B,D,T,T)
            diff = (pred - target) ** 2
            loss = (diff * mask).sum() / (mask.sum() * target.shape[1] + 1e-8)  # masked-only MSE
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.log_int == 0:
            writer.add_scalar("MAE/recon_mse_train", float(loss.item()), step)
            writer.add_scalar("MAE/lr", float(opt.param_groups[0]["lr"]), step)
            model.eval()
            with torch.no_grad():
                hb, hz = held_s.batch(args.batch_size, rng)
                if hb is not None:
                    hb = hb.to(dev)
                    ht = hb[:, 0]
                    hm = make_mask(hb.shape[0], T, args.mask_patch, args.mask_frac, dev, rng)
                    hi = apply_mask(hb, hm)
                    with _autocast(dev):
                        hp = model(hi, depth_offset=hz)
                        hloss = (((hp - ht) ** 2) * hm).sum() / (hm.sum() * ht.shape[1] + 1e-8)
                    writer.add_scalar("MAE/recon_mse_heldout", float(hloss.item()), step)
            model.train()
            print(f"[mae] step {step}/{args.steps}  mse={loss.item():.5f}  ({time.time()-t0:.0f}s)")

        if step % args.fig_int == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(), _autocast(dev):
                pv = model(xin, depth_offset=z)
            _save_fig(writer, step, target, xin, pv, mask)
            model.train()

        if step % args.save_int == 0 or step == args.steps:
            path = os.path.join("models", f"{args.name}.pth")
            torch.save(model.state_dict(), path)   # keys: stage1.* (warm-start) + decoder.* (dropped)
            print(f"[mae] saved {path} @ step {step}")

        if args.step_cooldown_ms > 0:
            time.sleep(args.step_cooldown_ms / 1000.0)
        if args.cooldown_secs > 0 and step % args.cooldown_int == 0:
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize()
            print(f"[mae] cooldown {args.cooldown_secs:.0f}s @ step {step}")
            time.sleep(args.cooldown_secs)

    writer.close()
    print(f"[mae] done. warm-start fine-tune by setting  c.init_weights = 'models/{args.name}.pth'  "
          f"in your campaign (stage1.* transfers; stage2 stays fresh).")


if __name__ == "__main__":
    main()
