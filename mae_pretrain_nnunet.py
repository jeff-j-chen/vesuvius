"""mae_pretrain_nnunet.py -- MAE pretraining for the nnunet3d_lcndz backbone.

WHY: ink detection is label-scarce and signal-faint. warm-starting from a model that
already understands papyrus texture (depth profiles, fiber patterns, layer interfaces)
gives the supervised head a richer feature space to separate ink from papyrus -- the
same lever that made MAE pretraining necessary for the older two-stage architecture.

WHAT IS PRETRAINED: the full nnunet3d_lcndz encoder + decoder -- everything except
the final binary out_head. a throwaway reconstruction head (1 conv) is appended for
MAE. at fine-tune time, train.py loads the saved checkpoint with strict=False; the
backbone weights transfer and the recon head is silently ignored.

APPROACH: block-masked 3D inpainting.
  - sample (D, ctx, ctx) crops from each scroll's TRAIN region (no label leakage)
  - block-mask a fraction of the spatial HxW positions (whole depth-column, matching
    the papyrus-layer structure)
  - fill masked positions with the visible-mean (mask-token proxy)
  - forward through encoder-decoder; apply recon head to get (B, 1, D, H/ds, W/ds)
  - MSE loss on masked positions in the ds-downsampled target
  - a scattered train/monitor split detects memorization without a geographic confound

TRANSFER: set config.init_weights = "models/mae_nnunet.pth" in a campaign or pass
  --init-weights on the command line.

    python mae_pretrain_nnunet.py --steps 6000 --ctx 96 --ds 2
    python mae_pretrain_nnunet.py --steps 6000 --ctx 48 --ds 2
    python mae_pretrain_nnunet.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import Config, DEFAULT_SCROLLS
from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm, compute_norm

try:
    from torch.amp import autocast as _ac, GradScaler as _GS
    def _autocast(dev): return _ac(dev if isinstance(dev, str) else str(dev))
    def _scaler(dev): return _GS(dev if isinstance(dev, str) else str(dev))
except Exception:
    from torch.cuda.amp import autocast as _acc, GradScaler as _GSS
    def _autocast(dev): return _acc()
    def _scaler(dev): return _GSS()


# ── crop sampler ─────────────────────────────────────────────────────────────

class CropSampler:
    """samples normalized (D, ctx, ctx) crops from one scroll's train region.

    uses a scattered hold-out (block-hash mod) so the recon_mse_monitor gap
    measures memorization on the same texture distribution, not a geographic split.
    """

    def __init__(self, scroll_id, zarr_path, cfg, y0, y1, x0, x1,
                 role="train", holdout_frac=0.1, block_px=512):
        import zarr
        self.vol = zarr.open(os.path.join(zarr_path, f"{scroll_id}.zarr"), mode="r")
        self.T = cfg.data.tile_size       # center tile size (for coord grid)
        self.ctx = int(getattr(cfg.data, "context_size", 48) or 48)
        self.D = cfg.data.depth
        self.z0 = cfg.data.train_d_start
        self.z1 = cfg.data.train_d_end
        self.y0, self.y1 = y0, y1
        self.x0, self.x1 = x0, x1
        self.role = role
        self.holdout_mod = max(2, int(round(1.0 / max(holdout_frac, 1e-6))))
        self.block_px = int(block_px)

        import cv2
        mask_img = cv2.imread(f"./masks/{scroll_id}.png", cv2.IMREAD_GRAYSCALE)
        H, W = int(self.vol.shape[1]), int(self.vol.shape[2])
        if mask_img is None:
            mask_img = np.full((H, W), 255, np.uint8)
        if mask_img.shape != (H, W):
            mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
        self.mask = (mask_img > 0).astype(np.uint8)

        norm = load_cached_norm(str(scroll_id), UNIFIED_CACHE_PATH)
        if norm is None:
            print(f"[mae] computing norm for {scroll_id}...")
            norm = compute_norm(str(scroll_id), zarr_path, UNIFIED_CACHE_PATH)
        self.mean, self.std, self.g_min, self.g_max = norm

    def _is_holdout(self, yy, xx):
        by, bx = yy // self.block_px, xx // self.block_px
        return ((by * 73856093) ^ (bx * 19349663)) % self.holdout_mod == 0

    def _norm(self, blk):
        b = (blk - self.mean) / self.std
        return np.clip((b - self.g_min) / (self.g_max - self.g_min + 1e-12), 0, 1)

    def sample(self, n, rng):
        """return (n, 1, D, ctx, ctx) float32 tensor or None if too few valid tiles."""
        ctx, D = self.ctx, self.D
        z_range = max(0, self.z1 - self.z0 - D)
        want_hold = (self.role == "monitor")
        out, tries = [], 0
        while len(out) < n and tries < n * 40:
            tries += 1
            yy = int(rng.integers(self.y0, max(self.y0 + 1, self.y1 - ctx)))
            xx = int(rng.integers(self.x0, max(self.x0 + 1, self.x1 - ctx)))
            if self._is_holdout(yy, xx) != want_hold:
                continue
            # require at least 30% mask coverage inside the crop
            mc = self.mask[yy:yy + ctx, xx:xx + ctx]
            if mc.size == 0 or mc.mean() < 0.30:
                continue
            z = self.z0 if z_range == 0 else int(rng.integers(self.z0, self.z0 + z_range + 1))
            try:
                blk = np.array(self.vol[z:z + D, yy:yy + ctx, xx:xx + ctx], dtype=np.float32)
            except Exception:
                continue
            if blk.shape != (D, ctx, ctx):
                continue
            out.append(self._norm(blk))
        if not out:
            return None
        arr = np.stack(out).astype(np.float32)        # (n, D, ctx, ctx)
        return torch.from_numpy(arr).unsqueeze(1)     # (n, 1, D, ctx, ctx)


class MultiSampler:
    """round-robins over per-scroll samplers."""

    def __init__(self, samplers):
        self.samplers = [s for s in samplers if s is not None]

    def sample(self, n, rng):
        if not self.samplers:
            return None
        per = max(1, n // len(self.samplers))
        parts = [s.sample(per, rng) for s in self.samplers]
        parts = [p for p in parts if p is not None]
        return torch.cat(parts, dim=0) if parts else None


# ── spatial block mask ────────────────────────────────────────────────────────

def _make_spatial_mask(B, ctx, ds, patch, frac, dev, rng):
    """block mask over (ctx/ds) x (ctx/ds) feature map; 1 = masked.

    masking whole depth-columns matches the papyrus depth-layer structure:
    the model must infer what the hidden column contains from lateral context,
    pushing it to learn fiber patterns and layer boundaries.
    """
    H = ctx // ds
    gH = H // patch
    n = gH * gH
    k = max(1, int(round(frac * n)))
    m = torch.zeros(B, n, device=dev)
    for b in range(B):
        idx = torch.from_numpy(rng.choice(n, size=k, replace=False)).to(dev)
        m[b, idx] = 1.0
    m = m.view(B, 1, 1, gH, gH)                   # (B, 1, 1, gH, gH)
    return F.interpolate(m.view(B, 1, gH, gH), scale_factor=patch, mode="nearest").view(B, 1, 1, H, H)


def _apply_mask(x, mask):
    """replace masked positions with the visible-mean per sample."""
    vis = x * (1.0 - mask)
    denom = (1.0 - mask).sum(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)
    fill = vis.sum(dim=(1, 2, 3, 4), keepdim=True) / denom
    return vis + fill * mask


# ── MAE wrapper ───────────────────────────────────────────────────────────────

class NnUnetMAE(nn.Module):
    """nnunet3d_lcndz backbone + throwaway reconstruction head.

    the backbone's _encode_decode runs on the masked input.
    the recon head maps the decoded features back to the ds-downsampled raw volume.
    only backbone.* keys are meaningful at fine-tune time.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # dec1 has 32 channels; reconstruct at (D, H/ds, W/ds) resolution
        self.recon_head = nn.Conv3d(32, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.recon_head.weight)
        nn.init.zeros_(self.recon_head.bias)

    def forward(self, x_masked):
        _, dec1 = self.backbone._encode_decode(x_masked)
        return self.recon_head(dec1)   # (B, 1, D, H/ds, W/ds)


# ── logging ───────────────────────────────────────────────────────────────────

def _log_fig(writer, step, target_ds, masked_ds, pred):
    """triptych: target | masked input | reconstruction (mean over D)."""
    try:
        def _show(t):
            return t[0, 0].mean(0).detach().float().cpu().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for a, im, title in zip(ax,
                                [_show(target_ds), _show(masked_ds), _show(pred)],
                                ["target (ds)", "masked input", "reconstruction"]):
            a.imshow(im, cmap="gray", vmin=0, vmax=1)
            a.set_title(title, fontsize=9)
            a.axis("off")
        plt.tight_layout()
        writer.add_figure("MAE/recon", fig, step)
        plt.close(fig)
    except Exception as e:
        print(f"[mae] fig failed: {e}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MAE pretraining for nnunet3d_lcndz")
    ap.add_argument("-n", "--name", default="mae_nnunet")
    ap.add_argument("--scroll-ids", type=int, nargs="+", default=None,
                    help="scroll ids to sample from (default: all DEFAULT_SCROLLS)")
    ap.add_argument("--ctx", type=int, default=96,
                    help="context window size in pixels (should match campaign ctx)")
    ap.add_argument("--ds", type=int, default=2,
                    help="context_downsample (must match campaign setting)")
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--d-start", type=int, default=4)
    ap.add_argument("--d-end", type=int, default=28)
    ap.add_argument("--mask-frac", type=float, default=0.65,
                    help="fraction of spatial positions to mask")
    ap.add_argument("--mask-patch", type=int, default=4,
                    help="patch size for block masking in ds-space (4 = 4x4 blocks)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--log-int", type=int, default=50)
    ap.add_argument("--fig-int", type=int, default=500)
    ap.add_argument("--save-int", type=int, default=1000)
    ap.add_argument("--log-dir", default="runs_mae")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # build a minimal config matching the campaign settings
    cfg = Config()
    cfg.model.arch = "nnunet3d_lcndz"
    cfg.model.attn_mil = False         # no MIL during pretraining
    cfg.model.learned_surface = False
    cfg.tra.supcon = False
    cfg.data.tile_size = 16
    cfg.data.depth = args.depth
    cfg.data.train_d_start = args.d_start
    cfg.data.train_d_end = args.d_end
    cfg.data.context_size = args.ctx
    cfg.data.context_downsample = args.ds

    dev = cfg.device
    rng = np.random.default_rng(args.seed)

    # build samplers from the configured scrolls
    scroll_ids = args.scroll_ids or [int(s.scroll_id) for s in DEFAULT_SCROLLS]
    split_by_id = {int(s.scroll_id): (s.split_axis, float(s.train_split_frac))
                   for s in DEFAULT_SCROLLS}

    import zarr
    zarr_path = cfg.data.zarr_path
    train_samplers, mon_samplers = [], []
    for sid in scroll_ids:
        try:
            vol = zarr.open(os.path.join(zarr_path, f"{sid}.zarr"), mode="r")
        except Exception as e:
            print(f"[mae] skip {sid}: {e}"); continue
        H, W = int(vol.shape[1]), int(vol.shape[2])
        axis, frac = split_by_id.get(sid, ("x", 0.75))
        ctx = args.ctx
        if axis == "y":
            y0, y1 = 0, (int(H * frac) // ctx) * ctx
            x0, x1 = 0, (W // ctx) * ctx
        else:
            y0, y1 = 0, (H // ctx) * ctx
            x0, x1 = 0, (int(W * frac) // ctx) * ctx
        if (y1 - y0) < ctx or (x1 - x0) < ctx:
            print(f"[mae] skip {sid}: train region too small"); continue
        print(f"[mae] {sid} ({H}x{W}) axis={axis} frac={frac} -> "
              f"y[{y0},{y1}] x[{x0},{x1}]")
        sc = CropSampler(sid, zarr_path, cfg, y0, y1, x0, x1, "train",
                         holdout_frac=args.holdout_frac)
        mc = CropSampler(sid, zarr_path, cfg, y0, y1, x0, x1, "monitor",
                         holdout_frac=args.holdout_frac)
        train_samplers.append(sc)
        mon_samplers.append(mc)

    if not train_samplers:
        raise RuntimeError("no usable scrolls")

    train_s = MultiSampler(train_samplers)
    mon_s = MultiSampler(mon_samplers)

    if args.dry_run:
        xb = train_s.sample(4, rng)
        print(f"[mae] dry-run OK: sampled batch {xb.shape if xb is not None else None}")
        return

    from utils.model import create_model
    backbone, _ = create_model(cfg)
    model = NnUnetMAE(backbone).to(dev)
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
    save_path = os.path.join("models", f"{args.name}.pth")

    print(f"[mae] nnunet3d backbone  ctx={args.ctx} ds={args.ds} depth={args.depth} "
          f"mask_frac={args.mask_frac}  steps={args.steps}  log={run_dir}")
    print(f"[mae] save -> {save_path}   use as:  c.init_weights = '{save_path}'")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb = train_s.sample(args.batch_size, rng)
        if xb is None:
            continue
        xb = xb.to(dev)

        # target: ds-downsampled raw crop (what the backbone's decoder sees)
        if args.ds > 1:
            target_ds = F.avg_pool3d(
                xb,
                kernel_size=(1, args.ds, args.ds),
                stride=(1, args.ds, args.ds),
            )
        else:
            target_ds = xb

        mask = _make_spatial_mask(xb.shape[0], args.ctx, args.ds,
                                  args.mask_patch, args.mask_frac, dev, rng)
        xb_masked = _apply_mask(xb, F.interpolate(
            mask.squeeze(2), scale_factor=args.ds, mode="nearest").unsqueeze(2))

        opt.zero_grad(set_to_none=True)
        with _autocast(dev):
            pred = model(xb_masked)              # (B, 1, D, H/ds, W/ds)
            diff = (pred - target_ds) ** 2
            # MSE on masked positions only
            loss = (diff * mask).sum() / (mask.sum() * target_ds.shape[2] + 1e-8)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.log_int == 0:
            writer.add_scalar("MAE/recon_mse_train", float(loss.item()), step)
            writer.add_scalar("MAE/lr", float(opt.param_groups[0]["lr"]), step)
            model.eval()
            with torch.no_grad():
                mb = mon_s.sample(args.batch_size, rng)
                if mb is not None:
                    mb = mb.to(dev)
                    if args.ds > 1:
                        mt = F.avg_pool3d(mb, kernel_size=(1, args.ds, args.ds),
                                          stride=(1, args.ds, args.ds))
                    else:
                        mt = mb
                    mm = _make_spatial_mask(mb.shape[0], args.ctx, args.ds,
                                            args.mask_patch, args.mask_frac, dev, rng)
                    mm_up = F.interpolate(mm.squeeze(2), scale_factor=args.ds,
                                          mode="nearest").unsqueeze(2)
                    mb_m = _apply_mask(mb, mm_up)
                    with _autocast(dev):
                        mp = model(mb_m)
                        ml = ((mp - mt) ** 2 * mm).sum() / (mm.sum() * mt.shape[2] + 1e-8)
                    writer.add_scalar("MAE/recon_mse_monitor", float(ml.item()), step)
            model.train()
            elapsed = time.time() - t0
            print(f"[mae] step {step}/{args.steps}  "
                  f"mse={loss.item():.5f}  ({elapsed:.0f}s)", flush=True)

        if step % args.fig_int == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(), _autocast(dev):
                pv = model(xb_masked)
            _log_fig(writer, step, target_ds, xb_masked if args.ds == 1 else F.avg_pool3d(
                xb_masked, kernel_size=(1, args.ds, args.ds),
                stride=(1, args.ds, args.ds)), pv)
            model.train()

        if step % args.save_int == 0 or step == args.steps:
            # save ONLY backbone.* (recon_head is dropped at fine-tune time)
            backbone_state = {k[len("backbone."):]: v
                              for k, v in model.state_dict().items()
                              if k.startswith("backbone.")}
            torch.save(backbone_state, save_path)
            print(f"[mae] saved {save_path} @ step {step} "
                  f"({len(backbone_state)} keys)", flush=True)

    writer.close()
    print(f"[mae] done.  warm-start fine-tune:  c.init_weights = '{save_path}'")


if __name__ == "__main__":
    main()
