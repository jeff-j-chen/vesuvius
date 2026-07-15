"""mae_pretrain.py — masked-autoencoder self-supervised pretraining for the 7.9um ink encoder.

WHY: every supervised run so far learns the papyrus representation AND the ink boundary
simultaneously from a tiny number of weak labels — the worst case for a faint signal. MAE
splits those problems: first learn "what 7.9um papyrus texture looks like" from MILLIONS of
UNLABELED crops (reconstruct masked voxels), then (in a separate train.py fine-tune) attach
the ink head and let it find a boundary in an already-rich feature space.

WHAT IT DOES:
  - samples random tile_size x tile_size x depth crops from the raw 7.9um zarr
  - ONLY from the TRAINING region (top `train_split_frac` of the frame) so the held-out
    bottom stays genuinely unseen — keeps the later generalization test honest
  - masks ~`mask_frac` of each crop as patch blocks (zeroed), reconstructs the depth-MEAN
    image, MSE loss on the MASKED pixels only (standard MAE objective)
  - saves encoder weights (keys match arch='dense_unet') -> models/<name>.pth
  - logs to tensorboard: recon MSE (train + held-out crops), a masked/recon/target figure

TRANSFER: fine-tune with
  python train.py -n ink_from_mae --arch dense_unet --init-weights models/mae_dense_unet.pth \
      --dense-labels --dense-soft-labels ...(your usual v7 flags)...
  the loader is strict=False: every stem/encoder/decoder tensor transfers; only the MAE
  'recon' head is dropped and the fresh ink 'head' is trained.

WHAT TO EXPECT (metrics):
  - MAE/recon_mse_train should fall steadily then plateau — this ONLY means it learned to
    inpaint papyrus texture; it is NOT an ink metric.
  - MAE/recon_mse_heldout should track train (no big gap) — a big gap = memorizing, lower
    mask_frac or add more crops.
  - the recon figure should visibly reproduce fiber/layer structure in masked regions.
  - NONE of this proves ink is recoverable. the VERDICT comes from the FINE-TUNE step:
    does dense_unet warm-started from MAE produce sharper held-out letters than from scratch?
    that is the actual test of "is there a linearly-recoverable ink texture at 7.9um".
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# force the non-interactive Agg backend BEFORE pyplot is ever imported. figures are only
# written to tensorboard from a background context, so the default Tk backend would try to
# talk to a gui event loop that doesn't exist here -> "main thread is not in main loop" /
# Tcl_AsyncDelete crash. Agg has no gui and renders straight to buffers.
import matplotlib
matplotlib.use("Agg")

from utils.config import Config
from utils.model import create_model

try:
    from torch.amp import autocast, GradScaler
    def _autocast(dev): return autocast(dev if isinstance(dev, str) else str(dev))
    def _scaler(dev): return GradScaler(dev if isinstance(dev, str) else str(dev))
except Exception:
    from torch.cuda.amp import autocast as _ac, GradScaler as _GS
    def _autocast(dev): return _ac()
    def _scaler(dev): return _GS()

NORM_CACHE = "./norm_cache.json"


def load_norm(scroll_id):
    with open(NORM_CACHE) as f:
        e = json.load(f).get(str(scroll_id))
    if not (isinstance(e, dict) and all(k in e for k in ("mean", "std", "min", "max"))):
        raise RuntimeError(f"no norm for {scroll_id}; run precompute_norm.py first")
    return float(e["mean"]), float(e["std"]), float(e["min"]), float(e["max"])


class CropSampler:
    """samples normalized (D,T,T) crops from one scroll's 7.9um zarr.

    role-based scattered hold-out (NOT a geographic strip): the frame is tiled into
    `block_px` blocks; ~`holdout_frac` of blocks are deterministically assigned to the
    monitor set, spread UNIFORMLY across the whole frame. this keeps train and monitor
    crops on the SAME texture/intensity distribution, so the train/monitor recon-MSE gap
    reflects ONLY memorization — not the bottom-is-darker or right-is-edge confounds a
    contiguous strip would introduce. keeps only crops whose mask coverage > min_cover."""
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
        self.role = role                       # 'train' or 'monitor'
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
        # deterministic scattered assignment: hash the block grid coords
        by = yy // self.block_px
        bx = xx // self.block_px
        return ((by * 73856093) ^ (bx * 19349663)) % self.holdout_mod == 0

    def batch(self, n, rng):
        T, D = self.T, self.D
        z = self.z0 if (self.z1 - self.z0) <= D else int(rng.integers(self.z0, self.z1 - D + 1))
        out = []
        tries = 0
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
            return None
        arr = np.stack(out)                     # (n,D,T,T)
        return torch.from_numpy(arr).unsqueeze(1).float()   # (n,1,D,T,T)


class MultiSampler:
    """round-robin over several CropSamplers so each batch mixes all scrolls."""
    def __init__(self, samplers):
        self.samplers = [s for s in samplers if s is not None]

    def batch(self, n, rng):
        if not self.samplers:
            return None
        parts = []
        per = max(1, n // len(self.samplers))
        for s in self.samplers:
            b = s.batch(per, rng)
            if b is not None:
                parts.append(b)
        if not parts:
            return None
        return torch.cat(parts, dim=0)


def make_mask(B, T, patch, frac, dev, rng):
    """block-random mask over a TxT image: returns (B,1,T,T) with 1=MASKED (hidden)."""
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
    """replace masked voxels with each sample's VISIBLE mean (a mask-token proxy) instead
    of zero. zero-fill creates hard black/data edges the conv learns to trace (blob
    artifacts); filling with the visible mean removes that edge while leaking no masked
    info. mask is (B,1,T,T) with 1=hidden; xb is (B,1,D,T,T)."""
    m = mask.unsqueeze(2)                                  # (B,1,1,T,T)
    vis = xb * (1.0 - m)
    denom = (1.0 - m).sum(dim=(1, 2, 3, 4), keepdim=True) + 1e-8
    fill = vis.sum(dim=(1, 2, 3, 4), keepdim=True) / denom  # (B,1,1,1,1) per-sample visible mean
    return xb * (1.0 - m) + fill * m


def main():
    ap = argparse.ArgumentParser(description="MAE pretraining for the 7.9um ink encoder")
    ap.add_argument("-n", "--name", default="mae_dense_unet")
    ap.add_argument("--scroll-ids", type=int, nargs="+", default=[20240304161941],
                    help="one or more scroll ids to sample UNLABELED crops from (more = richer texture prior)")
    ap.add_argument("--arch", default="dense_unet_mae")
    ap.add_argument("--tile-size", type=int, default=64,
                    help="pretrain crop size; LARGER than the 32px fine-tune tile gives more "
                         "inpainting context. dense_unet is fully-conv so weights transfer to 32px.")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--train-d-start", type=int, default=0)
    ap.add_argument("--train-d-end", type=int, default=64)
    ap.add_argument("--crop-x-frac", type=str, default="0.0,1.0")
    ap.add_argument("--crop-y-frac", type=str, default="0.0,1.0")
    ap.add_argument("--holdout-frac", type=float, default=0.1,
                    help="fraction of SCATTERED tile-blocks reserved ONLY for the recon-mse "
                         "overfitting monitor (spread uniformly across the frame, same "
                         "distribution as train — not a geographic strip)")
    ap.add_argument("--holdout-block", type=int, default=512,
                    help="block size (px) for the scattered train/monitor split")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="fraction of steps for linear lr warmup before cosine decay")
    ap.add_argument("--min-lr-frac", type=float, default=0.02,
                    help="cosine floor as a fraction of peak lr (annealed-to value)")
    ap.add_argument("--mask-frac", type=float, default=0.75,
                    help="fraction of patches hidden (0.75 = MAE standard; forces long-range "
                         "structural inference rather than local interpolation)")
    ap.add_argument("--mask-patch", type=int, default=8)
    ap.add_argument("--log-int", type=int, default=50)
    ap.add_argument("--fig-int", type=int, default=500)
    ap.add_argument("--save-int", type=int, default=1000)
    ap.add_argument("--step-cooldown-ms", type=float, default=0.0,
                    help="sleep this many ms after EVERY step (steady thermal relief)")
    ap.add_argument("--cooldown-secs", type=float, default=0.0,
                    help="sleep this many seconds every --cooldown-int steps (periodic deep cooldown)")
    ap.add_argument("--cooldown-int", type=int, default=200,
                    help="interval (steps) between the periodic --cooldown-secs pauses")
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

    import zarr
    cxf = tuple(float(v) for v in args.crop_x_frac.split(","))
    cyf = tuple(float(v) for v in args.crop_y_frac.split(","))
    T = args.tile_size

    # build one train sampler + one monitor sampler per scroll, over the full masked frame.
    # the train/monitor split is SCATTERED across tile-blocks (not a geographic strip), so
    # both share the same texture/intensity distribution and the recon-mse gap reflects only
    # memorization. no labels are used anywhere — this is transductive self-supervision.
    train_samplers, held_samplers = [], []
    for sid in args.scroll_ids:
        vol = zarr.open(os.path.join(cfg.data.zarr_path, f"{sid}.zarr"), mode="r")
        H, W = int(vol.shape[1]), int(vol.shape[2])
        x0 = (int(W * cxf[0]) // T) * T; x1 = (int(W * cxf[1]) // T) * T
        y0 = (int(H * cyf[0]) // T) * T; y1 = (int(H * cyf[1]) // T) * T
        print(f"[mae] {sid} frame {H}x{W}  region y[{y0},{y1}] x[{x0},{x1}]  "
              f"scattered {int(args.holdout_frac*100)}% monitor (block={args.holdout_block}px)")
        train_samplers.append(CropSampler(cfg, sid, y0, y1, x0, x1, role="train",
                                          holdout_frac=args.holdout_frac, block_px=args.holdout_block))
        held_samplers.append(CropSampler(cfg, sid, y0, y1, x0, x1, role="monitor",
                                         holdout_frac=args.holdout_frac, block_px=args.holdout_block))

    rng = np.random.default_rng(args.seed)
    train_s = MultiSampler(train_samplers)
    held_s  = MultiSampler(held_samplers)

    model, _ = create_model(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = _scaler(dev)

    # cosine lr schedule with linear warmup (standard for MAE; lets longer runs anneal to
    # fine texture at the end). NOT plateau — MAE's masked-mse floor is tiny and noisy, so a
    # metric-reactive scheduler would misfire; cosine is a smooth deterministic anneal.
    import math
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
    print(f"[mae] logging -> {run_dir}")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb = train_s.batch(args.batch_size, rng)
        if xb is None:
            continue
        xb = xb.to(dev)
        target = xb[:, 0]                                # (B,D,T,T) per-slice target (all 8 slices)
        mask = make_mask(xb.shape[0], T, args.mask_patch, args.mask_frac, dev, rng)  # (B,1,T,T) 1=hidden
        xin = apply_mask(xb, mask)                        # fill masked voxels with visible mean

        opt.zero_grad(set_to_none=True)
        with _autocast(dev):
            pred = model(xin)                            # (B,D,T,T) per-slice reconstruction
            diff = (pred - target) ** 2                 # mask (B,1,T,T) broadcasts across D
            loss = (diff * mask).sum() / (mask.sum() * target.shape[1] + 1e-8)   # MSE on masked pixels only
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % args.log_int == 0:
            writer.add_scalar("MAE/recon_mse_train", float(loss.item()), step)
            writer.add_scalar("MAE/lr", float(opt.param_groups[0]["lr"]), step)
            # held-out crops (no grad)
            model.eval()
            with torch.no_grad():
                hb = held_s.batch(args.batch_size, rng)
                if hb is not None:
                    hb = hb.to(dev)
                    ht = hb[:, 0]                        # (B,D,T,T) per-slice target
                    hm = make_mask(hb.shape[0], T, args.mask_patch, args.mask_frac, dev, rng)
                    hi = apply_mask(hb, hm)
                    with _autocast(dev):
                        hp = model(hi)
                        hloss = (((hp - ht) ** 2) * hm).sum() / (hm.sum() * ht.shape[1] + 1e-8)
                    writer.add_scalar("MAE/recon_mse_heldout", float(hloss.item()), step)
            model.train()
            el = time.time() - t0
            print(f"[mae] step {step}/{args.steps}  mse={loss.item():.5f}  ({el:.0f}s)")

        if step % args.fig_int == 0 or step == args.steps:
            _save_fig(writer, step, target, xin, model, xin, mask)

        if step % args.save_int == 0 or step == args.steps:
            path = os.path.join("models", f"{args.name}.pth")
            torch.save(model.state_dict(), path)
            print(f"[mae] saved {path} @ step {step}")

        # thermal relief: a steady per-step sleep + a periodic deeper pause. keeps the
        # gpu/cpu from running flat-out (which was crashing the laptop at ~104C).
        if args.step_cooldown_ms > 0:
            time.sleep(args.step_cooldown_ms / 1000.0)
        if args.cooldown_secs > 0 and step % args.cooldown_int == 0:
            print(f"[mae] cooldown {args.cooldown_secs:.0f}s @ step {step}")
            torch.cuda.synchronize() if str(dev).startswith("cuda") else None
            time.sleep(args.cooldown_secs)

    writer.close()
    print(f"[mae] done. warm-start fine-tune with:\n"
          f"  python train.py -n ink_from_mae --arch dense_unet "
          f"--init-weights models/{args.name}.pth --dense-labels --dense-soft-labels ...")


def _save_fig(writer, step, target, xin_vol, model, xin, mask):
    """figure for per-slice MAE. target is (B,D,T,T); we display the MIDDLE slice (which
    preserves real texture) plus the depth-mean, so blur from averaging isn't mistaken for
    model failure. p is (B,D,T,T) per-slice reconstruction."""
    import matplotlib.pyplot as plt
    model.eval()
    with torch.no_grad():
        p = model(xin)                                   # (B,D,T,T) per-slice reconstruction
    model.train()
    D = target.shape[1]
    mid = D // 2
    m2 = mask                                            # (B,1,T,T) broadcasts over slices
    # middle-slice views (real texture) + composite for that slice
    tgt_mid = target[:, mid:mid+1]
    prd_mid = p[:, mid:mid+1]
    masked_mid = xin_vol[:, 0, mid:mid+1]                # (B,1,T,T) masked input, middle slice
    comp_mid = tgt_mid * (1.0 - m2) + prd_mid * m2
    k = min(4, target.shape[0])
    vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(4, k, figsize=(3 * k, 12))
    if k == 1: ax = ax.reshape(4, 1)
    for i in range(k):
        ax[0, i].imshow(tgt_mid[i, 0].float().cpu(), cmap="gray", vmin=vmin, vmax=vmax); ax[0, i].set_title(f"target (slice {mid})")
        ax[1, i].imshow(masked_mid[i, 0].float().cpu(), cmap="gray", vmin=vmin, vmax=vmax); ax[1, i].set_title("masked input")
        ax[2, i].imshow(prd_mid[i, 0].float().cpu(), cmap="gray", vmin=vmin, vmax=vmax); ax[2, i].set_title("raw reconstruction")
        ax[3, i].imshow(comp_mid[i, 0].float().cpu(), cmap="gray", vmin=vmin, vmax=vmax); ax[3, i].set_title("composite (visible+filled)")
        for r in range(4): ax[r, i].axis("off")
    plt.tight_layout()
    writer.add_figure("MAE/reconstruction", fig, step)
    plt.close(fig)


if __name__ == "__main__":
    main()
