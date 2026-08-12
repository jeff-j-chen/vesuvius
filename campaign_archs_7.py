"""campaign_archs_7.py -- villa-inspired improvements + radical archs on PHerc0500P2 (2026-08-11).

SINGLE-SCROLL ISOLATION: Train ONLY on PHerc0500P2 (20250628074500) - the fragment with
crystal-clear 2.215um inklabels. This isolates label quality from multi-scroll domain diversity
to test villa-inspired architectural improvements AND radical vision architectures without
confounding label noise.

13 TESTS:
1-3: Baseline + archs5 sanity checks (GCE, 5-window)
4-7: Villa-inspired (depth attention, MAD norm, GroupNorm, InstanceNorm)
8-13: Radical architectures from archs6 (ViT, Swin, ConvNeXt, XCiT, nnU-Net, SlotAttn)

BASELINE: depsc_high (depth SupCon λ=0.3) from archs5, proven best on 15-scroll sweep.

CONFIGURATION:
- 10 epochs only (faster iteration, PHerc0500P2 is small)
- eval_int=10 (fires once at end), probe_int=999, test_int=999
- fast_eval_figure=True (left 40% of valid region only)
- soft augmentation (single-scroll regime from archs5/w044_soft)

  python campaign_archs_7.py --dry-run
  python campaign_archs_7.py --only baseline
  python campaign_archs_7.py
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, ScrollConfig
from utils.platform import get_zarr_dir, get_default_batch_size, get_default_eval_bs, get_default_workers, get_default_lr

MAE_CKPT = "models/mae_twostage.pth"
LOG_DIR = "./runs_archs7"
N_EP = 10  # shorter for single-scroll iteration


def _base_config(exp_name: str) -> Config:
    c = Config()
    on_linux = (os.name == "posix")
    c.exp_name = exp_name
    c.model.arch = "v16_arch_ctx"
    
    c.data.zarr_path = get_zarr_dir()
    
    c.data.tile_size     = 16
    c.data.depth         = 24
    c.data.train_d_start = 4
    c.data.train_d_end   = 28
    c.data.d_start = 4
    c.data.d_end   = 28
    c.data.context_size        = 48
    c.data.context_downsample  = 2
    c.model.conv1_drop = 0.15
    c.model.conv2_drop = 0.15
    c.model.head_drop  = 0.4
    c.tra.n_epochs     = N_EP
    c.tra.eval_int     = 10   # fires once at end
    c.tra.test_int     = 999  # disabled
    c.tra.probe_int    = 999  # disabled
    c.tra.save_int     = 5
    c.tra.log_dir      = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = get_default_lr()
    c.data.eval_infer_bs = get_default_eval_bs()
    c.tra.eval_int_scrolls = 1
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda    = 0.0
    c.dl.batch_size    = get_default_batch_size()
    c.dl.num_workers   = get_default_workers()
    c.dl.data_aug      = True
    c.data.mask_memmap       = True
    c.data.mask_bitpack      = True
    c.data.ring_negatives    = True
    c.data.ring_close_r      = 3
    c.data.ring_gap_r        = 3
    c.data.ring_shell_r      = 2
    c.tra.ranking_lambda     = 0.5
    c.tra.ranking_neg_frac   = 1.0
    c.tra.fast_eval_figure   = True  # left 40% only for speed
    # soft augmentation (single-scroll regime from archs5)
    c.dl.flip_prob           = 0.25
    c.dl.rotation_prob       = 0.25
    c.dl.noise_prob          = 0.15
    c.dl.brightness_prob     = 0.15
    c.dl.contrast_prob       = 0.15
    c.dl.cutout_prob         = 0.1
    c.dl.cutout_max_frac     = 0.1
    c.dl.cutout_n_patches    = 1
    c.dl.depth_mask_prob     = 0.0
    c.tra.epoch_cooldown_secs   = 0 
    c.tra.val_cooldown_secs     = 0 
    c.tra.eval_cooldown_secs    = 0 
    c.tra.fig_chunk_cooldown_ms = 0 
    c.tra.dann_n_domains = 1  # single scroll
    return c


# baseline: depsc_high from archs5 (winner) + sc15 curriculum supcon + attn_mil
_BASE7 = dict(
    init_weights=MAE_CKPT,
    scrolls=[ScrollConfig(20250628074500, split_axis="y", train_split_frac=0.75)],  # PHerc0500P2 only
    dann=False,
    supcon=True, supcon_temp=0.07,
    supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5, supcon_curriculum_epochs=10,
    attn_mil=True,
    attn_entropy_weight=0.03,
    depth_supcon=True, depth_supcon_lambda=0.3,  # winner from archs5
    mean_teacher=False, test_consistency=False,
)


def _mk7(tid, tag, **overrides):
    d = dict(_BASE7); d.update(overrides); d["tid"] = tid; d["tag"] = tag
    return d


TESTS = [
    # BASELINE: depsc_high on PHerc0500P2
    _mk7("baseline", "depsc_high_p500p2"),
    
    # ARCHS5 CARRYOVERS: quick sanity checks
    _mk7("gce_noise", "gce_q09",
         gce_q=0.9),
    
    _mk7("five_win", "5_depth_windows",
         n_depth_windows=5),
    
    # VILLA-INSPIRED IMPROVEMENTS
    # 4. Hybrid depth attention per window (attention+max per 8-slice window, NOT global collapse)
    _mk7("villa_depth_attn", "hybrid_depth_attn_per_window",
         depth_attention_mode="hybrid_per_window"),  # NEW flag
    
    # 5. Robust-MAD normalization (per-patch median/MAD instead of global z-score)
    _mk7("villa_mad_norm", "robust_mad_normalization",
         normalization_mode="robust_mad"),  # NEW flag
    
    # 6. GroupNorm + LeakyReLU (batch-independent, stable for small tiles)
    _mk7("villa_groupnorm", "groupnorm_leakyrelu",
         normalization_layer="group", activation="leaky"),  # NEW flags
    
    # 7. InstanceNorm + LeakyReLU (villa's choice, test despite small-tile risk)
    _mk7("villa_instancenorm", "instancenorm_leakyrelu",
         normalization_layer="instance", activation="leaky"),  # NEW flags
    
    # ARCHS6 RADICAL ARCHITECTURES (from campaign_archs_6)
    # 8. 3D Vision Transformer (pure self-attention)
    _mk7("vit3d", "vision_transformer_3d",
         arch="vit3d", init_weights=None),
    
    # 9. Swin Transformer 3D (hierarchical shifted windows)
    _mk7("swin3d", "swin_transformer_3d",
         arch="swin3d", init_weights=None),
    
    # 10. ConvNeXt 3D (modernized CNN with 7x7 kernels)
    _mk7("convnext3d", "convnext_3d",
         arch="convnext3d", init_weights=MAE_CKPT),
    
    # 11. XCiT 3D (cross-covariance attention)
    _mk7("xcit3d", "xcit_3d",
         arch="xcit3d", init_weights=None),
    
    # 12. nnU-Net 3D (encoder-decoder with deep supervision)
    _mk7("nnunet3d", "nnunet_3d",
         arch="nnunet3d", init_weights=None,
         attn_mil=False),  # deep supervision replaces MIL
    
    # 13. Slot Attention 3D (object-centric representation)
    _mk7("slot3d", "slot_attention_3d",
         arch="slot3d", init_weights=None,
         attn_mil=False),  # slot attention replaces MIL
]


_OVERRIDES = {
    # model
    "arch":                  ("model", "arch"),
    "attn_mil":              ("model", "attn_mil"),
    "attn_entropy_weight":   ("model", "attn_entropy_weight"),
    "physics_stem":          ("model", "physics_stem"),
    "physics_stem_depthmax": ("model", "physics_stem_depthmax"),
    "surface_stem":          ("model", "surface_stem"),
    "surface_stem_withdog":  ("model", "surface_stem_withdog"),
    "learned_surface":       ("model", "learned_surface"),
    "n_depth_windows":       ("model", "n_depth_windows"),
    "depth_attention_mode":  ("model", "depth_attention_mode"),  # NEW
    "normalization_layer":   ("model", "normalization_layer"),   # NEW
    "activation":            ("model", "activation"),            # NEW
    "conv1_drop":            ("model", "conv1_drop"),
    "conv2_drop":            ("model", "conv2_drop"),
    "head_drop":             ("model", "head_drop"),
    # training (init_weights handled specially in build_config, not here)
    "n_epochs":              ("tra", "n_epochs"),
    "eval_int":              ("tra", "eval_int"),
    "probe_int":             ("tra", "probe_int"),
    "l1":                    ("tra", "l1_lambda"),
    "weight_decay":          ("tra", "weight_decay"),
    "ranking_lambda":        ("tra", "ranking_lambda"),
    "tv_lambda":             ("tra", "tv_lambda"),
    "depth_supcon":          ("tra", "depth_supcon"),
    "depth_supcon_lambda":   ("tra", "depth_supcon_lambda"),
    "tta_consistency":       ("tra", "tta_consistency"),
    "tta_consistency_lambda":("tra", "tta_consistency_lambda"),
    "gce_q":                 ("tra", "gce_q"),
    "loss_type":             ("tra", "loss_type"),
    "focal_gamma":           ("tra", "focal_gamma"),
    "label_smooth_pos":      ("tra", "label_smooth_pos"),
    "label_smooth_neg":      ("tra", "label_smooth_neg"),
    # data
    "normalization_mode":    ("data", "normalization_mode"),  # NEW
    "context_size":          ("data", "context_size"),
    "context_downsample":    ("data", "context_downsample"),
    "ring_label_source":     ("data", "ring_label_source"),
    # augmentation
    "flip_prob":             ("dl", "flip_prob"),
    "rotation_prob":         ("dl", "rotation_prob"),
    "noise_prob":            ("dl", "noise_prob"),
    "brightness_prob":       ("dl", "brightness_prob"),
    "contrast_prob":         ("dl", "contrast_prob"),
    "cutout_prob":           ("dl", "cutout_prob"),
    "cutout_max_frac":       ("dl", "cutout_max_frac"),
    "cutout_n_patches":      ("dl", "cutout_n_patches"),
    "depth_mask_prob":       ("dl", "depth_mask_prob"),
    # supcon
    "supcon":                ("tra", "supcon"),
    "supcon_lambda":         ("tra", "supcon_lambda"),
    "supcon_temp":           ("tra", "supcon_temp"),
    "supcon_curriculum":     ("tra", "supcon_curriculum"),
    "supcon_lambda_start":   ("tra", "supcon_lambda_start"),
    "supcon_lambda_end":     ("tra", "supcon_lambda_end"),
    "supcon_curriculum_epochs": ("tra", "supcon_curriculum_epochs"),
    "batch_size":            ("dl", "batch_size"),
}


def build_config(t: dict) -> Config:
    tid = t["tid"]; tag = t["tag"]
    c = _base_config(f"cmp_archs7_{tid}_{tag}")
    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            try:
                setattr(getattr(c, sec), attr, t[k])
            except AttributeError:
                print(f"[WARNING] {tid}: {sec}.{attr} does not exist")
    if "scrolls" in t:
        c.data.scrolls = t["scrolls"]
    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[archs7] init_weights '{iw}' not found -- {tid} trains from scratch")
    c.dl.data_aug = any([c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
                         c.dl.brightness_prob, c.dl.contrast_prob,
                         c.dl.cutout_prob, c.dl.depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0
    os.makedirs("models/archs7", exist_ok=True)
    c.save_final = f"models/archs7/{tid}_{tag}_final.pth"
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[archs7] {c.exp_name}\n{'='*70}", flush=True)
    n_scrolls = len(c.data.scrolls)
    print(f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
          f"  entropy={c.model.attn_entropy_weight}")
    print(f"  SINGLE-SCROLL: PHerc0500P2 only (clean labels)")
    print(f"  n_epochs={c.tra.n_epochs}  fast_eval_figure={c.tra.fast_eval_figure}")
    print(f"  depth_supcon={c.tra.depth_supcon}  depth_supcon_lam={c.tra.depth_supcon_lambda}")
    print(f"  gce_q={c.tra.gce_q}  n_depth_windows={c.model.n_depth_windows}")
    norm_mode = getattr(c.data, "normalization_mode", "zscore")
    norm_layer = getattr(c.model, "normalization_layer", "batch")
    activation = getattr(c.model, "activation", "relu")
    depth_attn = getattr(c.model, "depth_attention_mode", "none")
    print(f"  norm_mode={norm_mode}  norm_layer={norm_layer}  activation={activation}")
    print(f"  depth_attention={depth_attn}")
    print(f"  aug: flip={c.dl.flip_prob} rot={c.dl.rotation_prob} noise={c.dl.noise_prob}"
          f" cutout={c.dl.cutout_prob}")
    if dry_run:
        print("  [DRY RUN] skipping")
        return True
    from train import Trainer
    try:
        trainer = Trainer(c)
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return False


def main():
    ap = argparse.ArgumentParser(description="campaign_archs_7: villa improvements on PHerc0500P2")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--from", dest="from_id", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        selected = [t for t in TESTS if t["tid"] in want]
        missing = want - {t["tid"] for t in selected}
        if missing:
            print(f"[ABORT] --only id(s) {sorted(missing)} not found; valid: {[t['tid'] for t in TESTS]}")
            return
    elif args.from_id:
        ids = [t["tid"] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[archs7] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[archs7] SINGLE-SCROLL ISOLATION: PHerc0500P2 (crystal-clear 2.215um labels)")
    print(f"[archs7] Baseline: depsc_high (depth SupCon λ=0.3)")
    print(f"[archs7] Tests 1-3: baseline + archs5 sanity checks")
    print(f"[archs7] Tests 4-7: villa-inspired (depth attn, MAD, GroupNorm, InstanceNorm)")
    print(f"[archs7] Tests 8-13: radical archs from archs6 (ViT, Swin, ConvNeXt, XCiT, nnUNet, Slot)")

    results = {}
    for t in selected:
        tid = t["tid"]
        c = build_config(t)
        ok = run_test(c, args.dry_run)
        results[tid] = "OK" if ok else "FAIL"
        if not args.dry_run:
            del c; gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\n{'='*70}\n[archs7] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
