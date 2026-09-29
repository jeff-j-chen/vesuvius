"""step 4: cross-resolution MAE, continued from the production native-96 MAE (three plans, see PLAN.md).

every batch mixes two kinds of sample (--paired-frac):
  - standard: a 9.36 um crop from any of the 37 pretraining volumes (training, held-out and test scrolls,
    unlabelled), masked; loss = MSE on the masked 9.36 um voxels (the existing MAE objective)
  - paired: a 9.36 um crop from crossres/pairs/<plan>, masked the same way; the same 9.36 um loss plus a
    super-resolution loss: a throwaway head predicts the co-registered high-resolution target
    (plan depth: 4 sub-slices per slice, same x/y; plan xyz: also 2x in x/y), masked positions
    weighted 1 and visible ones --visible-weight, only where the target is valid

--slab-mask-prob also hides whole runs of 1-3 slices (every pixel) on top of the column mask, so the
backbone must rebuild layers from the layers around them. plan none (no pairs) trains only that.

only backbone.* is saved, with the production key names, so the checkpoint drops into init_weights
exactly like the current MAE. the reconstruction and super-resolution heads are discarded; the fine-tune
input stays a 96 x 96 x 8 crop at 9.36 um.

UNTESTED: written without running (no disk on the authoring machine). run --dry-run first.

    python crossres/mae_pretrain_crossres.py --plan depth --name mae_crossres_depth \
        --init-weights models/mae_nnunet_192_campaign34_early_gated_native96_2k.pth --dry-run
"""
from __future__ import annotations

import argparse
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


class CrossResMAE(NnUnetMAE):
    """the production MAE wrapper plus a super-resolution head (depth_factor x depth, scale x x/y)."""

    def __init__(self, backbone, depth: int, scale: int = 1, depth_factor: int = 4):
        super().__init__(backbone, depth)
        if not self.early_2d or backbone._overlapping_depth_windows:
            raise ValueError("the cross-resolution head is written for the early-2D decoder")
        channels = int(backbone.early2d_head.in_channels)
        self.sr_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, depth * depth_factor * scale * scale, 1),
            nn.PixelShuffle(scale) if scale > 1 else nn.Identity(),
        )
        nn.init.zeros_(self.sr_head[2].weight)
        nn.init.zeros_(self.sr_head[2].bias)

    def forward(self, x_masked):
        _, dec1 = self.backbone._encode_decode_early_2d(x_masked, None, None)
        return self.recon_head(dec1).unsqueeze(1), self.sr_head(dec1).unsqueeze(1)


class PairedTiles:
    """crops from one segment's pairs.zarr: normalized input, intensity-matched target, validity."""

    def __init__(self, zid: int, cfg, role: str, norm_mode: str, plan: str):
        import zarr
        self.zid = int(zid)
        folder = ROOT / "pairs" / plan / str(zid)
        self.meta = json.loads((folder / "meta.json").read_text())
        self.scale, self.factor = int(self.meta["xy_scale"]), int(self.meta["depth_factor"])
        group = zarr.open_group(str(folder / "pairs.zarr"), mode="r")
        self.input, self.target, self.valid = group["input"], group["target"], group["valid"]
        self.ctx, self.depth = cfg.data.context_size, cfg.data.depth
        z0, z1 = self.meta["z_range"]
        # window starts allowed by both the stored slices and the MAE depth range, relative to z0
        self.starts = np.arange(max(cfg.data.train_d_start, z0),
                                min(cfg.data.train_d_end, z1) - self.depth + 1) - z0
        if len(self.starts) == 0:
            raise ValueError(f"{zid}: stored slices {self.meta['z_range']} do not cover the MAE window")
        tiles = np.arange(self.input.shape[0])
        # every tenth tile monitors reconstruction on tiles the model never trains on
        self.tiles = tiles[tiles % 10 != 0] if role == "train" else tiles[tiles % 10 == 0]
        if len(self.tiles) == 0:
            raise ValueError(f"{zid}: no {role} tiles")
        self.mean, self.std, self.g_min, self.g_max = load_cached_norm(str(zid), UNIFIED_CACHE_PATH, mode=norm_mode)
        self._match = self._intensity_match()

    def _norm(self, block):
        b = (block - self.mean) / self.std
        return np.clip((b - self.g_min) / (self.g_max - self.g_min + 1e-12), 0, 1)

    def _windows(self, start):
        return (slice(start, start + self.depth),
                slice(start * self.factor, (start + self.depth) * self.factor))

    def _intensity_match(self):
        """robust affine that maps the target onto the normalized input's intensity distribution."""
        window, target_window = self._windows(int(self.starts[len(self.starts) // 2]))
        ins, tgs = [], []
        for tile in self.tiles[:: max(1, len(self.tiles) // 16)]:
            source = self._norm(np.asarray(self.input[tile, window], np.float32))
            target = np.asarray(self.target[tile, target_window], np.float32)
            valid = np.asarray(self.valid[tile])
            ins.append(source[source > 0])
            tgs.append(target[:, valid][target[:, valid] > 0])
        ins, tgs = np.concatenate(ins), np.concatenate(tgs)
        in_med, tg_med = np.median(ins), np.median(tgs)
        in_mad = np.median(np.abs(ins - in_med)) + 1e-6
        tg_mad = np.median(np.abs(tgs - tg_med)) + 1e-6
        return float(in_mad / tg_mad), float(in_med - tg_med * in_mad / tg_mad)

    def sample(self, n, rng):
        ctx, s = self.ctx, self.scale
        size = int(self.input.shape[-1])
        xs, ts, vs = [], [], []
        while len(xs) < n:
            window, target_window = self._windows(int(rng.choice(self.starts)))
            for _ in range(20):
                tile = int(rng.choice(self.tiles))
                y, x = (int(v) for v in rng.integers(0, size - ctx + 1, size=2))
                valid = np.asarray(self.valid[tile, s * y:s * (y + ctx), s * x:s * (x + ctx)])
                if valid.mean() >= 0.9:
                    break
            source = np.asarray(self.input[tile, window, y:y + ctx, x:x + ctx], np.float32)
            target = np.asarray(self.target[tile, target_window, s * y:s * (y + ctx), s * x:s * (x + ctx)],
                                np.float32)
            gain, bias = self._match
            xs.append(self._norm(source))
            ts.append(target * gain + bias)
            vs.append(valid & (target > 0).all(axis=0))
        to = lambda items: torch.from_numpy(np.stack(items).astype(np.float32)).unsqueeze(1)
        return to(xs), to(ts), to(vs).unsqueeze(2)


class PairedGroups:
    """round robin over physical scrolls, then over each scroll's segments."""

    def __init__(self, samplers):
        groups = {}
        for sampler in samplers:
            groups.setdefault(sampler.scroll, []).append(sampler)
        self.groups = list(groups.values())
        self.cursor = 0

    def sample(self, n, rng):
        parts = []
        for index in range(n):
            group = self.groups[(self.cursor + index) % len(self.groups)]
            parts.append(group[int(rng.integers(len(group)))].sample(1, rng))
        self.cursor = (self.cursor + n) % len(self.groups)
        return tuple(torch.cat(items) for items in zip(*parts))


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


def standard_samplers(cfg, scroll_ids, holdout_frac, flip_ids=()):
    import zarr
    split_by_id = {int(s.scroll_id): (s.split_axis, float(s.train_split_frac)) for s in DEFAULT_SCROLLS}
    flip_ids = {int(i) for i in flip_ids}
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
        crops = (crop, CropSampler(sid, cfg.data.zarr_path, cfg, *box, "monitor",
                                   holdout_frac=holdout_frac, shared=crop))
        if int(sid) in flip_ids:
            for sampler in crops:
                # renders whose normal points the other way are read in the training orientation
                sampler.sample = _depth_flipped(sampler.sample)
        train.append(crops[0])
        monitor.append(crops[1])
    return make_physical_sampler(train)[0], make_physical_sampler(monitor)[0]


def _depth_flipped(sample):
    def wrapped(n, rng):
        crops = sample(n, rng)
        return None if crops is None else torch.flip(crops, dims=(2,))
    return wrapped


def main():
    ap = argparse.ArgumentParser(description="cross-resolution MAE continued from the production MAE")
    ap.add_argument("--name", required=True)
    ap.add_argument("--plan", choices=("depth", "xyz", "none"), default="depth",
                    help="which crossres/pairs/<plan> tiles to use; none = no pairs (slab-masked MAE only)")
    ap.add_argument("--init-weights", required=True, help="the production MAE this continues from")
    ap.add_argument("--scroll-ids", type=int, nargs="*", default=None,
                    help="standard-branch volumes (default: campaign33.PRETRAIN_SCROLL_IDS, all 37)")
    ap.add_argument("--pair-names", nargs="*", default=None, help="default: every built pair")
    ap.add_argument("--exclude-holdout-pairs", action="store_true",
                    help="drop pairs whose role is holdout (0841, the strict measure); 0009B's pair stays in")
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
    ap.add_argument("--lr", type=float, default=1.5e-4, help="half the from-scratch MAE lr: this is a warm start")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="the production MAE used 1e-4; 0 by default here")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--log-int", type=int, default=50)
    ap.add_argument("--save-int", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--input-upsampler", default="",
                    help="train_upsampler.py checkpoint applied to every crop first (v8-in's upsample-before-"
                         "network geometry); requires --plan none")
    ap.add_argument("--upsampler-depth-pool", type=int, default=2,
                    help="average this many predicted sub-slices: x4 depth pooled 2 -> 16 slices")
    ap.add_argument("--flip-depth-ids", type=int, nargs="*", default=(),
                    help="scrolls whose renders are reversed in depth (crossres/depth_orientation.py); "
                         "their crops are flipped. the stored window 8-19 is symmetric about the centre")
    args = ap.parse_args()
    if args.input_upsampler and args.plan != "none":
        ap.error("--input-upsampler trains the standard MAE on upsampled crops; use --plan none")

    cfg = build_config(args)
    dev = cfg.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    upsampler = None
    model_ctx, model_depth, model_patch = args.ctx, args.depth, args.mask_patch
    model_cfg = cfg
    if args.input_upsampler:
        import copy
        from utils.upsampler import load_upsampler
        upsampler = load_upsampler(args.input_upsampler, dev)
        up_scale, up_factor = upsampler.xy_scale, upsampler.depth_factor
        if up_factor % args.upsampler_depth_pool:
            ap.error("--upsampler-depth-pool must divide the upsampler's depth factor")
        model_ctx = args.ctx * up_scale
        model_depth = args.depth * up_factor // args.upsampler_depth_pool
        # the same physical mask blocks on the finer grid
        model_patch = args.mask_patch * up_scale
        model_cfg = copy.deepcopy(cfg)
        model_cfg.data.context_size, model_cfg.data.depth = model_ctx, model_depth
        print(f"[crossres] input upsampler {args.input_upsampler}: {args.depth}x{args.ctx}^2 crops -> "
              f"{model_depth}x{model_ctx}^2 network input", flush=True)

    def prepare(x):
        """crops as the network sees them: upsampled (and depth-pooled) when an upsampler is set."""
        x = x.to(dev)
        if upsampler is None:
            return x
        with torch.no_grad():
            x = upsampler(x)
            pool = args.upsampler_depth_pool
            return F.avg_pool3d(x, kernel_size=(pool, 1, 1)) if pool > 1 else x

    def make_mask(batch, n_paired=0):
        """(B, 1, D, H, W): hidden columns, plus on some samples a run of whole hidden slices; a share of
        the paired samples (the last n_paired) is left whole."""
        columns = _make_spatial_mask(batch, model_ctx, 1, model_patch, args.mask_frac, dev, rng)
        mask = columns.expand(batch, 1, model_depth, model_ctx, model_ctx).clone()
        slab_max = args.slab_max * max(1, model_depth // args.depth)
        for b in range(batch):
            if b >= batch - n_paired and rng.random() < args.unmasked_paired_frac:
                mask[b] = 0.0
                continue
            if rng.random() < args.slab_mask_prob:
                length = int(rng.integers(1, slab_max + 1))
                start = int(rng.integers(0, model_depth - length + 1))
                mask[b, :, start:start + length] = 1.0
        return mask

    if args.scroll_ids:
        scroll_ids = list(args.scroll_ids)
    else:
        import campaign_archs_33 as campaign33
        scroll_ids = [int(s) for s in campaign33.PRETRAIN_SCROLL_IDS]
    std_train, std_monitor = standard_samplers(cfg, scroll_ids, holdout_frac=0.1, flip_ids=args.flip_depth_ids)

    wanted = [] if args.plan == "none" else [
        p for p in PAIRS if (ROOT / "pairs" / args.plan / str(p["zid"]) / "meta.json").exists()
        and (not args.pair_names or p["name"] in args.pair_names)
        and not (args.exclude_holdout_pairs and p["role"] == "holdout")
    ]
    if args.plan != "none" and not wanted:
        raise SystemExit(f"no built pairs under crossres/pairs/{args.plan}; run build_pairs.py --plan {args.plan}")
    pair_train, pair_monitor = [], []
    for pair in wanted:
        for role, bucket in (("train", pair_train), ("monitor", pair_monitor)):
            tiles = PairedTiles(pair["zid"], cfg, role, args.data_norm_mode, args.plan)
            tiles.scroll = pair["scroll"]
            bucket.append(tiles)
        print(f"[crossres] {pair['name']} ({pair['role']}): {len(pair_train[-1].tiles)} train / "
              f"{len(pair_monitor[-1].tiles)} monitor tiles, target gain/bias={pair_train[-1]._match}", flush=True)
    scale = pair_train[0].scale if pair_train else 1
    factor = pair_train[0].factor if pair_train else 1
    paired = PairedGroups(pair_train) if pair_train else None
    paired_monitor = PairedGroups(pair_monitor) if pair_monitor else None
    n_pair = max(1, int(round(args.batch_size * args.paired_frac))) if paired else 0
    n_std = args.batch_size - n_pair

    if args.dry_run:
        x_std = prepare(std_train.sample(max(1, n_std), rng))
        print(f"[crossres] dry-run standard {tuple(x_std.shape)} mask {tuple(make_mask(x_std.shape[0]).shape)}")
        if paired:
            x_pair, target, valid = paired.sample(n_pair, rng)
            print(f"[crossres] dry-run paired {tuple(x_pair.shape)} target {tuple(target.shape)} "
                  f"valid {tuple(valid.shape)} valid_frac={valid.float().mean():.3f} (scale {scale}, depth x{factor})")
        return

    from utils.model import create_model
    backbone, _ = create_model(model_cfg)
    model = CrossResMAE(backbone, model_depth, scale=scale, depth_factor=factor).to(dev)
    state = torch.load(args.init_weights, map_location=dev, weights_only=True)
    state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
    own = model.backbone.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing = model.backbone.load_state_dict(compatible, strict=False).missing_keys
    print(f"[crossres] warm start from {args.init_weights}: loaded {len(compatible)}/{len(own)} "
          f"(missing {len(missing)})", flush=True)
    if len(compatible) < 0.95 * len(own):
        raise SystemExit("the init checkpoint does not match this architecture (check --fiber-coordinate-branch)")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = _scaler(dev)
    warmup = max(1, int(args.warmup_frac * args.steps))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join("runs_mae", f"{args.name}_{time.strftime('%m%d_%H-%M-%S')}"))
    save_path = os.path.join("models", f"{args.name}.pth")
    os.makedirs("models", exist_ok=True)

    def losses(x, target, valid, n_paired):
        mask = make_mask(x.shape[0], n_paired)
        with _autocast(dev):
            recon, sr = model(_apply_mask(x, mask))
            full = mask.expand_as(x)
            rec = ((recon - x) ** 2 * full).sum() / (full.sum() + 1e-8)
            if not n_paired:
                return rec, torch.zeros((), device=dev)
            # the mask, stretched onto the target's sub-slices and pixels
            fine = full[-n_paired:, 0].repeat_interleave(factor, dim=1)
            if scale > 1:
                fine = F.interpolate(fine, scale_factor=scale, mode="nearest")
            fine = fine.unsqueeze(1)
            # whole (unmasked) samples train every position at full weight, like generation
            whole = (full[-n_paired:].flatten(1).amax(dim=1) == 0).float().view(-1, 1, 1, 1, 1)
            visible = args.visible_weight + (1 - args.visible_weight) * whole
            weight = (fine + visible * (1 - fine)) * valid
            srl = ((sr[-n_paired:] - target) ** 2 * weight).sum() / (weight.sum() + 1e-8)
        return rec, srl

    started = time.time()
    for step in range(1, args.steps + 1):
        x_std = prepare(std_train.sample(n_std, rng)) if n_std else None
        if paired:
            x_pair, target, valid = paired.sample(n_pair, rng)
            target, valid = target.to(dev), valid.to(dev)
        else:
            x_pair = target = valid = None
        x = torch.cat([t for t in (x_std, x_pair) if t is not None]).to(dev)
        rec, srl = losses(x, target, valid, n_pair)
        loss = rec + args.sr_weight * srl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % args.log_int == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                xm_std = prepare(std_monitor.sample(max(1, n_std), rng))
                if paired_monitor:
                    xm_pair, tm, vm = (t.to(dev) for t in paired_monitor.sample(n_pair, rng))
                    mon_rec, mon_sr = losses(torch.cat([xm_std.to(dev), xm_pair]), tm, vm, n_pair)
                    # the head must beat plain trilinear upsampling of the input, or it learned nothing
                    upsampled = F.interpolate(xm_pair, scale_factor=(factor, scale, scale),
                                              mode="trilinear", align_corners=False)
                    full_vm = vm.expand_as(tm)
                    baseline = ((upsampled - tm) ** 2 * full_vm).sum() / (full_vm.sum() + 1e-8)
                else:
                    mon_rec, mon_sr = losses(xm_std.to(dev), None, None, 0)
                    baseline = torch.zeros(())
            model.train()
            for tag, value in (("train_rec", rec), ("train_sr", srl), ("monitor_rec", mon_rec),
                               ("monitor_sr", mon_sr), ("monitor_sr_trilinear", baseline)):
                writer.add_scalar(f"CrossRes/{tag}", float(value), step)
            print(f"[crossres] step {step}/{args.steps} rec={float(rec):.5f} sr={float(srl):.5f} "
                  f"monitor rec={float(mon_rec):.5f} sr={float(mon_sr):.5f} "
                  f"(trilinear {float(baseline):.5f}) {time.time() - started:.0f}s", flush=True)
        if step % args.save_int == 0 or step == args.steps:
            backbone_state = {k[len("backbone."):]: v for k, v in model.state_dict().items()
                              if k.startswith("backbone.")}
            torch.save(backbone_state, save_path)
            if pair_train:
                # the whole network incl. the SR head: the generator for the hallucination route
                torch.save({"state_dict": model.state_dict(), "xy_scale": scale, "depth_factor": factor,
                            "depth": args.depth, "ctx": args.ctx, "target_gain_bias":
                            {int(t.zid): t._match for t in pair_train}},
                           save_path.replace(".pth", ".generator.pth"))
            print(f"[crossres] saved {save_path} ({len(backbone_state)} backbone keys)", flush=True)


if __name__ == "__main__":
    main()
