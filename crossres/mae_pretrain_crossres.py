"""step 4: cross-resolution MAE, continued from the production native-96 MAE (three plans, see PLAN.md).

every batch mixes two kinds of sample (--paired-frac):
  - standard: a 9.36 um crop from any of the 37 pretraining volumes (training, held-out and test scrolls,
    unlabelled), masked; loss = MSE on the masked 9.36 um voxels (the existing MAE objective)
  - paired: a 9.36 um crop from crossres/pairs/<plan>, masked the same way; the same 9.36 um loss plus a
    super-resolution loss: a throwaway head predicts the co-registered high-resolution target
    (plan depth: 4 sub-slices per slice, same x/y; plan xyz: also 2x in x/y), masked positions
    weighted 1 and visible ones --visible-weight, only where the target is valid

the SR head predicts the residual over the trilinear upsampling of the (unmasked) input, so the loss
measures only what the 9.36 um voxels do not already show. targets are intensity-matched per segment by
quantile mapping (pooled target -> normalized input), applied as a 256-entry LUT to the uint8 target.

--slab-mask-prob also hides whole runs of 1-3 slices (every pixel) on top of the column mask, so the
backbone must rebuild layers from the layers around them. plan none (no pairs) trains only that.

health check (fixed monitor crops from tiles never trained on, logged every --log-int steps and at step 0):
  monitor_sr          head on the unmasked input, uniform weight over valid target voxels
  monitor_sr_trilinear  trilinear upsampling of the input (= the head at initialisation)
  monitor_sr_linear3  a least-squares 3x3x3 linear filter (+ bias) on the upsampled input, fitted per segment
                      on training tiles; the head must beat this, not just trilinear
  monitor_rec         column-masked 9.36 um reconstruction on standard crops (the production MAE's objective)

only backbone.* is saved, with the production key names, so the checkpoint drops into init_weights
exactly like the current MAE. the fine-tune input stays a 96 x 96 x 8 crop at 9.36 um.

plan U (--upsampler, with --plan none): every 8 x 96 x 96 crop is masked on its native grid, then passed
through a frozen utils/upsampler.LearnedUpsampler (a train_upsampler.py checkpoint, or 'trilinear') into a
16 x 192 x 192 network input; the target is the upsampled unmasked crop and the loss mask is the native
mask upsampled (nearest). masking before upsampling means interpolation cannot leak hidden voxels.

    python crossres/mae_pretrain_crossres.py --plan depth --name mae_crossres_depth \
        --init-weights models/mae_nnunet_192_campaign34_early_gated_native96_2k.pth --dry-run
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from mae_pretrain_nnunet import (  # noqa: E402
    CropSampler, NnUnetMAE, _apply_mask, _autocast, _make_spatial_mask, _scaler,
    make_physical_sampler,
)
from pairs import PAIRS  # noqa: E402
from utils.config import Config, DEFAULT_SCROLLS  # noqa: E402
from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm  # noqa: E402
from utils.upsampler import load_upsampler  # noqa: E402


def upsample(x, factor, scale):
    return F.interpolate(x, scale_factor=(factor, scale, scale), mode="trilinear", align_corners=False)


class CrossResMAE(NnUnetMAE):
    """the production MAE wrapper plus a residual super-resolution head (depth_factor x depth, scale x x/y)."""

    def __init__(self, backbone, depth: int, scale: int = 1, depth_factor: int = 4):
        super().__init__(backbone, depth)
        if not self.early_2d or backbone._overlapping_depth_windows:
            raise ValueError("the cross-resolution head is written for the early-2D decoder")
        self.scale, self.factor = int(scale), int(depth_factor)
        channels = int(backbone.early2d_head.in_channels)
        self.sr_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, depth * depth_factor * scale * scale, 1),
            nn.PixelShuffle(scale) if scale > 1 else nn.Identity(),
        )
        nn.init.zeros_(self.sr_head[2].weight)
        nn.init.zeros_(self.sr_head[2].bias)

    def forward(self, x_in, x_full=None):
        """x_full: the unmasked input of the last len(x_full) samples, whose SR prediction is returned."""
        _, dec1 = self.backbone._encode_decode_early_2d(x_in, None, None)
        recon = self.recon_head(dec1).unsqueeze(1)
        if x_full is None:
            return recon, None
        residual = self.sr_head(dec1[-x_full.shape[0]:]).unsqueeze(1)
        return recon, residual.float() + upsample(x_full.float(), self.factor, self.scale)


class PairedSegment:
    """one segment's pairs.zarr held in RAM: normalized input, quantile-matched target, validity."""

    def __init__(self, pair: dict, cfg, norm_mode: str, plan: str):
        import zarr
        self.zid, self.scroll, self.name = int(pair["zid"]), pair["scroll"], pair["name"]
        folder = ROOT / "pairs" / plan / str(self.zid)
        self.meta = json.loads((folder / "meta.json").read_text())
        self.scale, self.factor = int(self.meta["xy_scale"]), int(self.meta["depth_factor"])
        group = zarr.open_group(str(folder / "pairs.zarr"), mode="r")
        self.input, self.target, self.valid = group["input"][:], group["target"][:], group["valid"][:]
        self.ctx, self.depth = cfg.data.context_size, cfg.data.depth
        z0, z1 = self.meta["z_range"]
        # window starts allowed by both the stored slices and the MAE depth range, relative to z0
        self.starts = np.arange(max(cfg.data.train_d_start, z0),
                                min(cfg.data.train_d_end, z1) - self.depth + 1) - z0
        if len(self.starts) == 0:
            raise ValueError(f"{self.zid}: stored slices {self.meta['z_range']} do not cover the MAE window")
        tiles = np.arange(self.input.shape[0])
        # every tenth tile is kept for monitoring, never trained on
        self.tiles = {"train": tiles[tiles % 10 != 0], "monitor": tiles[tiles % 10 == 0]}
        if len(self.tiles["train"]) == 0 or len(self.tiles["monitor"]) == 0:
            raise ValueError(f"{self.zid}: {len(tiles)} tiles cannot form train and monitor splits")
        self.mean, self.std, self.g_min, self.g_max = load_cached_norm(str(self.zid), UNIFIED_CACHE_PATH,
                                                                       mode=norm_mode)
        self.lut = self._quantile_lut()

    def _norm(self, block):
        b = (block - self.mean) / self.std
        return np.clip((b - self.g_min) / (self.g_max - self.g_min + 1e-12), 0, 1)

    def _windows(self, start):
        return (slice(start, start + self.depth),
                slice(start * self.factor, (start + self.depth) * self.factor))

    def _quantile_lut(self):
        """uint8 target -> normalized-input intensity, from the quantiles of the target pooled onto the input's
        voxels vs the input itself (co-located, valid voxels of training tiles)."""
        s, f = self.scale, self.factor
        window, target_window = self._windows(int(self.starts[len(self.starts) // 2]))
        ins, tgs = [], []
        for tile in self.tiles["train"][:: max(1, len(self.tiles["train"]) // 24)]:
            source = self._norm(self.input[tile, window].astype(np.float32))
            target = self.target[tile, target_window].astype(np.float32)
            n, h, w = target.shape
            coarse = target.reshape(n // f, f, h // s, s, w // s, s).mean(axis=(1, 3, 5))
            present = (target > 0).reshape(n // f, f, h // s, s, w // s, s).all(axis=(1, 3, 5))
            valid = self.valid[tile].reshape(h // s, s, w // s, s).all(axis=(1, 3))
            keep = present & valid[None] & (self.input[tile, window] > 0)
            ins.append(source[keep])
            tgs.append(coarse[keep])
        ins, tgs = np.concatenate(ins), np.concatenate(tgs)
        levels = np.linspace(0.005, 0.995, 199)
        in_q, tg_q = np.quantile(ins, levels), np.quantile(tgs, levels)
        tg_q = np.maximum.accumulate(tg_q) + np.arange(len(tg_q)) * 1e-6
        values = np.arange(256, dtype=np.float64)
        lut = np.interp(values, tg_q, in_q)
        # linear extrapolation past the fitted range: the fine target has wider tails than its pooled version
        edge = len(levels) // 20
        lo_slope = (in_q[edge] - in_q[0]) / max(tg_q[edge] - tg_q[0], 1e-6)
        hi_slope = (in_q[-1] - in_q[-1 - edge]) / max(tg_q[-1] - tg_q[-1 - edge], 1e-6)
        lut = np.where(values < tg_q[0], in_q[0] + (values - tg_q[0]) * lo_slope, lut)
        lut = np.where(values > tg_q[-1], in_q[-1] + (values - tg_q[-1]) * hi_slope, lut)
        return lut.astype(np.float32)

    def sample(self, role, n, rng):
        ctx, s = self.ctx, self.scale
        size = int(self.input.shape[-1])
        xs, ts, vs = [], [], []
        for _ in range(n):
            window, target_window = self._windows(int(rng.choice(self.starts)))
            for _ in range(20):
                tile = int(rng.choice(self.tiles[role]))
                y, x = (int(v) for v in rng.integers(0, size - ctx + 1, size=2))
                valid = self.valid[tile, s * y:s * (y + ctx), s * x:s * (x + ctx)]
                if valid.mean() >= 0.9:
                    break
            source = self.input[tile, window, y:y + ctx, x:x + ctx].astype(np.float32)
            target = self.target[tile, target_window, s * y:s * (y + ctx), s * x:s * (x + ctx)]
            xs.append(self._norm(source))
            ts.append(self.lut[target])
            vs.append(valid & (target > 0).all(axis=0))
        to = lambda items: torch.from_numpy(np.stack(items).astype(np.float32)).unsqueeze(1)
        return to(xs), to(ts), to(vs).unsqueeze(2)


class PairedGroups:
    """scrolls drawn with weight sqrt(training tiles), then segments within a scroll by their tile count,
    so a small segment is not repeated far more often than a large one."""

    def __init__(self, segments, role):
        self.role = role
        groups = {}
        for segment in segments:
            groups.setdefault(segment.scroll, []).append(segment)
        self.groups = list(groups.values())
        train_tiles = np.array([sum(len(s.tiles["train"]) for s in g) for g in self.groups], np.float64)
        self.scroll_p = np.sqrt(train_tiles) / np.sqrt(train_tiles).sum()
        self.segment_p = [np.array([len(s.tiles[role]) for s in g], np.float64) for g in self.groups]
        self.segment_p = [p / p.sum() for p in self.segment_p]

    def describe(self):
        return {g[0].scroll: round(float(p), 3) for g, p in zip(self.groups, self.scroll_p)}

    def sample(self, n, rng):
        parts, zids = [], []
        for group_index in rng.choice(len(self.groups), size=n, p=self.scroll_p):
            group = self.groups[group_index]
            segment = group[int(rng.choice(len(group), p=self.segment_p[group_index]))]
            parts.append(segment.sample(self.role, 1, rng))
            zids.append(segment.zid)
        return (*(torch.cat(items) for items in zip(*parts)), np.asarray(zids))


def _neighbourhood(volume):
    """(B, 27, D, H, W) replicate-padded 3x3x3 neighbours of a (B, 1, D, H, W) volume."""
    padded = F.pad(volume, (1, 1, 1, 1, 1, 1), mode="replicate")
    d, h, w = volume.shape[2:]
    return torch.cat([padded[:, :, dz:dz + d, dy:dy + h, dx:dx + w]
                      for dz, dy, dx in itertools.product(range(3), repeat=3)], dim=1)


def fit_linear3(segment, factor, scale, dev, rng, crops=64, chunk=8, keep_frac=0.25):
    """least-squares 3x3x3 filter + bias from the trilinear-upsampled input to the target, on training tiles."""
    xtx = torch.zeros(28, 28, dtype=torch.float64, device=dev)
    xty = torch.zeros(28, dtype=torch.float64, device=dev)
    for _ in range(crops // chunk):
        x, target, valid = (t.to(dev) for t in segment.sample("train", chunk, rng))
        feats = _neighbourhood(upsample(x, factor, scale))
        mask = valid.expand_as(target)[:, 0] > 0
        mask &= torch.rand(mask.shape, device=dev) < keep_frac
        design = feats.permute(0, 2, 3, 4, 1)[mask].double()
        design = torch.cat([design, torch.ones_like(design[:, :1])], dim=1)
        xtx += design.T @ design
        xty += design.T @ target[:, 0][mask].double()
    solution = torch.linalg.solve(xtx + 1e-6 * torch.eye(28, dtype=torch.float64, device=dev), xty)
    return solution[:27].float().view(1, 1, 3, 3, 3), solution[27].float()


def apply_linear3(x, kernel, bias, factor, scale):
    up = upsample(x, factor, scale)
    return F.conv3d(F.pad(up, (1, 1, 1, 1, 1, 1), mode="replicate"), kernel) + bias


def build_config(args) -> Config:
    """the production early-gated native-96 MAE architecture (EARLY_GATED_ARGS in campaign_archs_34)."""
    cfg = Config()
    model = cfg.model
    model.arch = "nnunet3d_lcndz"
    model.attn_mil = False
    model.learned_surface = False
    model.surface_teacher_input = False
    model.use_ibn = True
    model.norm_mode = "ibn_full"
    model.multitile = True
    model.early_2d_unet = True
    model.early_2d_channels_mult = 1.0
    model.mid_2d_unet = False
    model.mid_2d_channels_mult = 1.0
    model.residual_2d_unet = False
    model.two_d_block_depth = 2
    model.two_d_bottleneck_channels = 0
    model.two_d_extra_levels = 0
    model.two_d_extra_channels = ()
    model.two_d_strided_down = False
    model.channels_mult = 1.0
    model.raw_only_stem = False
    model.gated_stems = True
    model.explicit_depth_channels = False
    model.overlapping_depth_windows = False
    model.planar_early_convs = False
    model.depth_antialias = False
    model.factorized_2plus1d = False
    model.divided_attention = False
    model.divided_attention_spatial = False
    model.divided_attention_heads = 4
    model.divided_attention_window = 8
    model.mednext_adapters = False
    model.mednext_kernel = 5
    model.mednext_expansion = 2
    model.fiber_coordinate_branch = bool(args.fiber_coordinate_branch)
    model.input_denoise_sigma = 0.0
    model.input_denoiser = ""
    cfg.data.norm_mode = args.data_norm_mode
    cfg.tra.supcon = False
    cfg.data.tile_size = 16
    cfg.data.depth = args.depth
    cfg.data.train_d_start = args.d_start
    cfg.data.train_d_end = args.d_end
    cfg.data.context_size = args.ctx
    cfg.data.context_downsample = 1
    return cfg


def standard_samplers(cfg, scroll_ids, holdout_frac):
    import zarr
    split_by_id = {int(s.scroll_id): (s.split_axis, float(s.train_split_frac)) for s in DEFAULT_SCROLLS}
    train, monitor = [], []
    for sid in scroll_ids:
        volume = zarr.open(os.path.join(cfg.data.zarr_path, f"{sid}.zarr"), mode="r")
        height, width, ctx = int(volume.shape[1]), int(volume.shape[2]), cfg.data.context_size
        axis, frac = split_by_id.get(sid, ("x", 1.0))
        if axis == "y":
            box = (0, (int(height * frac) // ctx) * ctx, 0, (width // ctx) * ctx)
        else:
            box = (0, (height // ctx) * ctx, 0, (int(width * frac) // ctx) * ctx)
        crop = CropSampler(sid, cfg.data.zarr_path, cfg, *box, "train", holdout_frac=holdout_frac)
        train.append(crop)
        monitor.append(CropSampler(sid, cfg.data.zarr_path, cfg, *box, "monitor",
                                   holdout_frac=holdout_frac, shared=crop))
    return make_physical_sampler(train)[0], make_physical_sampler(monitor)[0]


def main():
    ap = argparse.ArgumentParser(description="cross-resolution MAE continued from the production MAE")
    ap.add_argument("--name", required=True)
    ap.add_argument("--plan", choices=("depth", "xyz", "none"), default="depth",
                    help="which crossres/pairs/<plan> tiles to use; none = no pairs (slab-masked MAE only)")
    ap.add_argument("--init-weights", required=True, help="the production MAE this continues from")
    ap.add_argument("--scroll-ids", type=int, nargs="*", default=None,
                    help="standard-branch volumes (default: the 13 paired segments in pairs.PAIRS)")
    ap.add_argument("--pair-names", nargs="*", default=None, help="default: every built pair")
    ap.add_argument("--exclude-holdout-pairs", action="store_true",
                    help="drop pairs whose role is holdout (0841 and 0009B)")
    ap.add_argument("--paired-frac", type=float, default=0.5)
    ap.add_argument("--sr-weight", type=float, default=1.0)
    ap.add_argument("--visible-weight", type=float, default=0.25)
    ap.add_argument("--fiber-coordinate-branch", action="store_true")
    ap.add_argument("--data-norm-mode", default="global", choices=("global", "surface_anchor"))
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--d-start", type=int, default=8,
                    help="the 8-slice window moves within [d-start, d-end); the production MAE fixed it at 10-17")
    ap.add_argument("--d-end", type=int, default=20)
    ap.add_argument("--mask-frac", type=float, default=0.65)
    ap.add_argument("--mask-patch", type=int, default=4)
    ap.add_argument("--slab-mask-prob", type=float, default=0.5,
                    help="share of samples that also lose a run of whole slices")
    ap.add_argument("--slab-max", type=int, default=3, help="longest run of hidden slices")
    ap.add_argument("--unmasked-paired-frac", type=float, default=0.25,
                    help="paired samples seen whole, as the generator sees them at inference (hallucination route)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum-steps", type=int, default=1, help="micro-batches per step (batch-size is the total)")
    ap.add_argument("--upsampler", default="",
                    help="plan U: train_upsampler.py checkpoint or 'trilinear'; the network sees the upsampled crop")
    ap.add_argument("--lr", type=float, default=1.5e-4, help="half the from-scratch MAE lr: this is a warm start")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="the production MAE used 1e-4; 0 by default here")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--head-warmup-steps", type=int, default=200,
                    help="backbone frozen while the zero-initialised heads fit (the init checkpoint has no heads)")
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--log-int", type=int, default=50)
    ap.add_argument("--save-int", type=int, default=500)
    ap.add_argument("--snapshot-steps", type=int, nargs="*", default=[1000],
                    help="also keep models/<name>.step<N>.pth at these steps")
    ap.add_argument("--monitor-crops", type=int, default=96, help="fixed monitor crops per branch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = build_config(args)
    dev = cfg.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.batch_size % args.accum_steps:
        ap.error("--batch-size must be divisible by --accum-steps")

    up, model_cfg, net_depth, net_ctx = None, cfg, args.depth, args.ctx
    if args.upsampler:
        if args.plan != "none":
            ap.error("--upsampler pretrains the standard MAE only; use --plan none")
        up = load_upsampler(args.upsampler).to(dev).eval()
        for parameter in up.parameters():
            parameter.requires_grad_(False)
        net_depth = args.depth * up.depth_factor // up.depth_pool
        net_ctx = args.ctx * up.scale
        model_cfg = copy.deepcopy(cfg)
        model_cfg.data.depth, model_cfg.data.context_size = net_depth, net_ctx
        print(f"[crossres] plan U: {args.upsampler} {up.config()} -> network input {net_depth} x {net_ctx} x {net_ctx}",
              flush=True)

    def net_view(x, mask):
        """(masked network input, reconstruction target, loss mask) on the grid the network sees."""
        if up is None:
            return _apply_mask(x, mask), x, mask
        full = mask.expand(-1, 1, args.depth, args.ctx, args.ctx)
        with torch.no_grad():
            return (up.network_input(_apply_mask(x, full)), up.network_input(x),
                    F.interpolate(full, size=(net_depth, net_ctx, net_ctx), mode="nearest"))

    def make_mask(batch, n_paired=0, mask_rng=rng, slab_prob=None, unmasked_frac=None):
        """(B, 1, D, H, W): hidden columns, plus on some samples a run of whole hidden slices; a share of
        the paired samples (the last n_paired) is left whole."""
        slab_prob = args.slab_mask_prob if slab_prob is None else slab_prob
        unmasked_frac = args.unmasked_paired_frac if unmasked_frac is None else unmasked_frac
        columns = _make_spatial_mask(batch, args.ctx, 1, args.mask_patch, args.mask_frac, dev, mask_rng)
        mask = columns.expand(batch, 1, args.depth, args.ctx, args.ctx).clone()
        for b in range(batch):
            if b >= batch - n_paired and mask_rng.random() < unmasked_frac:
                mask[b] = 0.0
                continue
            if mask_rng.random() < slab_prob:
                length = int(mask_rng.integers(1, args.slab_max + 1))
                start = int(mask_rng.integers(0, args.depth - length + 1))
                mask[b, :, start:start + length] = 1.0
        return mask

    if args.scroll_ids:
        scroll_ids = list(args.scroll_ids)
    else:
        # 0841 stays here unlabelled even when --exclude-holdout-pairs drops its pair
        scroll_ids = [int(p["zid"]) for p in PAIRS]
    missing = [sid for sid in scroll_ids if not os.path.isdir(os.path.join(cfg.data.zarr_path, f"{sid}.zarr"))]
    if missing:
        raise SystemExit(f"the standard branch needs every pretraining zarr; missing {missing}")
    std_train, std_monitor = standard_samplers(cfg, scroll_ids, holdout_frac=0.1)

    wanted = [] if args.plan == "none" else [
        p for p in PAIRS if (ROOT / "pairs" / args.plan / str(p["zid"]) / "meta.json").exists()
        and (not args.pair_names or p["name"] in args.pair_names)
        and not (args.exclude_holdout_pairs and p["role"] == "holdout")
    ]
    if args.plan != "none" and not wanted:
        raise SystemExit(f"no built pairs under crossres/pairs/{args.plan}; run build_pairs.py --plan {args.plan}")
    segments = []
    for pair in wanted:
        segment = PairedSegment(pair, cfg, args.data_norm_mode, args.plan)
        segments.append(segment)
        print(f"[crossres] {pair['name']} ({pair['role']}): {len(segment.tiles['train'])} train / "
              f"{len(segment.tiles['monitor'])} monitor tiles, LUT 0/128/255 -> "
              f"{segment.lut[0]:.3f}/{segment.lut[128]:.3f}/{segment.lut[255]:.3f}", flush=True)
    by_zid = {segment.zid: segment for segment in segments}
    scale = segments[0].scale if segments else 1
    factor = segments[0].factor if segments else 1
    paired = PairedGroups(segments, "train") if segments else None
    paired_monitor = PairedGroups(segments, "monitor") if segments else None
    if paired:
        print(f"[crossres] paired scroll weights (sqrt tiles): {paired.describe()}", flush=True)
    n_pair = max(1, int(round(args.batch_size * args.paired_frac))) if paired else 0
    n_std = args.batch_size - n_pair
    if n_pair % args.accum_steps or n_std % args.accum_steps:
        ap.error("both branches of the batch must split evenly into --accum-steps micro-batches")

    # fixed monitor sets so the curves are comparable step to step (and to step 0)
    monitor_rng = np.random.default_rng(args.seed + 1)
    mon_std = torch.cat([std_monitor.sample(min(32, args.monitor_crops - i), monitor_rng)
                         for i in range(0, args.monitor_crops, 32)])
    mon_std_cols = make_mask(len(mon_std), 0, monitor_rng, slab_prob=0.0)
    mon_std_slab = make_mask(len(mon_std), 0, monitor_rng, slab_prob=1.0)
    if paired_monitor:
        mon_x, mon_t, mon_v, mon_z = paired_monitor.sample(args.monitor_crops, monitor_rng)
        mon_pair_mask = make_mask(len(mon_x), len(mon_x), monitor_rng, unmasked_frac=0.0)

    from utils.model import create_model
    backbone, _ = create_model(model_cfg)
    model = CrossResMAE(backbone, net_depth, scale=scale, depth_factor=factor).to(dev)
    state = torch.load(args.init_weights, map_location=dev, weights_only=True)
    state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
    own = model.backbone.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing_keys = model.backbone.load_state_dict(compatible, strict=False).missing_keys
    print(f"[crossres] warm start from {args.init_weights}: loaded {len(compatible)}/{len(own)} "
          f"(missing {len(missing_keys)})", flush=True)
    if len(compatible) < 0.95 * len(own):
        raise SystemExit("the init checkpoint does not match this architecture (check --fiber-coordinate-branch)")

    linear3 = {}
    for segment in segments:
        linear3[segment.zid] = fit_linear3(segment, factor, scale, dev, np.random.default_rng(args.seed + 2))

    def rec_loss(recon, x, mask):
        full = mask.expand_as(x)
        return ((recon.float() - x) ** 2 * full).sum() / (full.sum() + 1e-8)

    def sr_weight(mask, valid):
        """training weights on the target grid: masked 1, visible --visible-weight, whole samples 1."""
        full = mask.expand(-1, 1, args.depth, args.ctx, args.ctx)
        fine = full[:, 0].repeat_interleave(factor, dim=1)
        if scale > 1:
            fine = F.interpolate(fine, scale_factor=scale, mode="nearest")
        fine = fine.unsqueeze(1)
        whole = (full.flatten(1).amax(dim=1) == 0).float().view(-1, 1, 1, 1, 1)
        visible = args.visible_weight + (1 - args.visible_weight) * whole
        return (fine + visible * (1 - fine)) * valid

    def weighted_mse(pred, target, weight):
        weight = weight.expand_as(target)
        return ((pred - target) ** 2 * weight).sum() / (weight.sum() + 1e-8)

    def train_losses(x, target, valid, n_paired):
        mask = make_mask(x.shape[0], n_paired)
        x_in, x_target, loss_mask = net_view(x, mask)
        with _autocast(dev):
            recon, sr = model(x_in, x[-n_paired:] if n_paired else None)
        rec = rec_loss(recon, x_target, loss_mask)
        if not n_paired:
            return rec, torch.zeros((), device=dev)
        return rec, weighted_mse(sr, target, sr_weight(mask[-n_paired:], valid))

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        sums = {"rec": 0.0, "rec_slab": 0.0}
        for i in range(0, len(mon_std), 32):
            x = mon_std[i:i + 32].to(dev)
            for key, masks in (("rec", mon_std_cols), ("rec_slab", mon_std_slab)):
                x_in, x_target, loss_mask = net_view(x, masks[i:i + 32])
                with _autocast(dev):
                    recon, _ = model(x_in)
                sums[key] += float(rec_loss(recon, x_target, loss_mask)) * len(x)
        out["monitor_rec"] = sums["rec"] / len(mon_std)
        out["monitor_rec_slab"] = sums["rec_slab"] / len(mon_std)
        if paired_monitor:
            totals = {k: [0.0, 0.0] for k in ("sr", "sr_trilinear", "sr_linear3", "sr_masked")}
            for i in range(0, len(mon_x), 16):
                x, t, v = (a[i:i + 16].to(dev) for a in (mon_x, mon_t, mon_v))
                zids = mon_z[i:i + 16]
                uniform = v.expand_as(t)
                with _autocast(dev):
                    _, sr = model(x, x)
                    _, sr_masked = model(_apply_mask(x, mon_pair_mask[i:i + 16]), x)
                lin = torch.cat([apply_linear3(x[j:j + 1], *linear3[int(z)], factor, scale)
                                 for j, z in enumerate(zids)])
                preds = {"sr": sr, "sr_trilinear": upsample(x, factor, scale), "sr_linear3": lin}
                for key, pred in preds.items():
                    totals[key][0] += float(((pred - t) ** 2 * uniform).sum())
                    totals[key][1] += float(uniform.sum())
                weight = sr_weight(mon_pair_mask[i:i + 16], v).expand_as(t)
                totals["sr_masked"][0] += float(((sr_masked - t) ** 2 * weight).sum())
                totals["sr_masked"][1] += float(weight.sum())
            for key, (num, den) in totals.items():
                out[f"monitor_{key}"] = num / max(den, 1e-8)
            out["monitor_sr_gain_vs_linear3"] = 1 - out["monitor_sr"] / out["monitor_sr_linear3"]
        model.train()
        return out

    if args.dry_run:
        if dev != "cpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        x_std = std_train.sample(max(1, n_std // args.accum_steps), rng)
        print(f"[crossres] dry-run standard {tuple(x_std.shape)}")
        target = valid = None
        n_pair_micro = n_pair // args.accum_steps
        if paired:
            x_pair, target, valid, zids = paired.sample(n_pair_micro, rng)
            print(f"[crossres] dry-run paired {tuple(x_pair.shape)} target {tuple(target.shape)} "
                  f"valid_frac={valid.float().mean():.3f} zids={sorted(set(zids.tolist()))} "
                  f"(scale {scale}, depth x{factor})")
            x_std = torch.cat([x_std, x_pair])
            target, valid = target.to(dev), valid.to(dev)
        rec, srl = train_losses(x_std.to(dev), target, valid, n_pair_micro)
        (rec + args.sr_weight * srl).backward()
        grads = sum(p.grad is not None for p in model.parameters())
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"[crossres] dry-run forward/backward rec={float(rec):.5f} sr={float(srl):.5f} "
              f"params with grad={grads} micro-batch={len(x_std)} peak GPU {peak:.1f} GB", flush=True)
        print(f"[crossres] dry-run monitor {json.dumps({k: round(v, 6) for k, v in evaluate().items()})}")
        return

    model.train()
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    opt = torch.optim.AdamW([{"params": backbone_params}, {"params": head_params}],
                            lr=args.lr, weight_decay=args.weight_decay)
    scaler = _scaler(dev)
    warmup = max(1, int(args.warmup_frac * args.steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    def backbone_lr_at(step):
        return lr_at(step) * min(1.0, max(0.0, (step - args.head_warmup_steps) / warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[backbone_lr_at, lr_at])
    from torch.utils.tensorboard import SummaryWriter
    run_dir = os.path.join("runs_mae", f"{args.name}_{time.strftime('%m%d_%H-%M-%S')}")
    writer = SummaryWriter(run_dir)
    save_path = os.path.join("models", f"{args.name}.pth")
    os.makedirs("models", exist_ok=True)
    history = []

    def log(step, values):
        for tag, value in values.items():
            writer.add_scalar(f"CrossRes/{tag}", float(value), step)
        history.append({"step": step, **{k: float(v) for k, v in values.items()}})
        (Path(run_dir) / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
        print(f"[crossres] step {step}/{args.steps} " + " ".join(f"{k}={float(v):.5f}" for k, v in values.items())
              + f" {time.time() - started:.0f}s", flush=True)

    def save(path):
        backbone_state = {k[len("backbone."):]: v for k, v in model.state_dict().items() if k.startswith("backbone.")}
        torch.save(backbone_state, path)
        if up is not None:
            # fine-tuning must put this exact upsampler in front of the model
            Path(path).with_suffix(".json").write_text(json.dumps({
                "upsampler": args.upsampler, "upsampler_config": up.config(),
                "native_input": [args.depth, args.ctx, args.ctx], "network_input": [net_depth, net_ctx, net_ctx],
                "init_weights": args.init_weights}, indent=1) + "\n")
        if segments:
            # the whole network incl. the SR head: the generator for the hallucination route
            torch.save({"state_dict": model.state_dict(), "xy_scale": scale, "depth_factor": factor,
                        "depth": args.depth, "ctx": args.ctx, "residual_over": "trilinear",
                        "target_lut": {s.zid: s.lut.tolist() for s in segments}},
                       path.replace(".pth", ".generator.pth"))
        else:
            # the whole network incl. the reconstruction head, for inspection (crossres/visualize_tests.ipynb)
            torch.save({"state_dict": model.state_dict(), "xy_scale": 1, "depth_factor": 1,
                        "depth": net_depth, "ctx": net_ctx, "upsampler": args.upsampler or None},
                       path.replace(".pth", ".full.pth"))
        print(f"[crossres] saved {path} ({len(backbone_state)} backbone keys)", flush=True)

    started = time.time()
    log(0, evaluate())
    running = {"train_rec": 0.0, "train_sr": 0.0, "n": 0}
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum_steps):
            x_std = std_train.sample(n_std // args.accum_steps, rng) if n_std else None
            if paired:
                x_pair, target, valid, _ = paired.sample(n_pair // args.accum_steps, rng)
                target, valid = target.to(dev), valid.to(dev)
            else:
                x_pair = target = valid = None
            x = torch.cat([t for t in (x_std, x_pair) if t is not None]).to(dev)
            rec, srl = train_losses(x, target, valid, n_pair // args.accum_steps)
            scaler.scale((rec + args.sr_weight * srl) / args.accum_steps).backward()
            running["train_rec"] += float(rec) / args.accum_steps
            running["train_sr"] += float(srl) / args.accum_steps
        scaler.step(opt)
        scaler.update()
        sched.step()
        running["n"] += 1
        if step % args.log_int == 0:
            values = {"train_rec": running["train_rec"] / running["n"], "lr": opt.param_groups[0]["lr"],
                      "lr_head": opt.param_groups[1]["lr"]}
            if paired:
                values["train_sr"] = running["train_sr"] / running["n"]
            running = {"train_rec": 0.0, "train_sr": 0.0, "n": 0}
            log(step, {**values, **evaluate()})
        if step % args.save_int == 0 or step == args.steps:
            save(save_path)
        if step in args.snapshot_steps:
            save(save_path.replace(".pth", f".step{step}.pth"))
    writer.close()


if __name__ == "__main__":
    main()
