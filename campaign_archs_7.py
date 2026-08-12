"""campaign_archs_7.py -- comprehensive architecture exploration on w044 (2026-08-11).

SINGLE-SCROLL ISOLATION: Train on w044 (20260115000000) to isolate architecture effects
from multi-scroll domain diversity. Tests 49 architectural variations systematically.

ORGANIZATION (49 TESTS):
1-2:   Sanity checks (w044 baseline, w044 no-aug overfit test)
3-6:   Context ablation (64×64, 96×96, 128×128 at ds2; 96×96 foveated)
7-10:  Dual-stream depth (parallel squashed + non-squashed, 4 fusion strategies)
11-14: Hybrid depth attention (Villa-inspired, 4 variants: per-window/global/triple/gated)
15-20: Multi-scale & efficient (pyramid, depth-SE, depthwise-sep, mixed-windows, octave, efficient)
21-26: Attention mechanisms (non-local, coordinate, deformable, progressive, dual, axial)
27-32: Advanced fusion (FPN, bi-FPN, ghost, inverted-residual, ResNeXt, depth-shift)
33-34: Archs5 proven (GCE q=0.9, 5-window)
35-38: Villa normalization (robust-MAD, GroupNorm, InstanceNorm, LayerNorm)
39-45: Radical archs from archs6 (ViT, Swin, ConvNeXt, XCiT, nnU-Net, SlotAttn, nnU-Net+soft-aug)
46-49: nnU-Net-inspired baseline upgrades (late collapse 32/64, shallow U-Net fusion, nnU-Net ds2)

BASELINE: depsc_high (depth SupCon λ=0.3, ctx=48 ds=2) from archs5.

DESIGN PRINCIPLES (from past campaigns):
✅ WORKS: depth preservation, large context, wide head, zgrad, lcn, attn-MIL+entropy, supcon
❌ FAILS: early depth squashing, dense pixel-level architectures

CONFIGURATION:
- 12 epochs (w044 is small, fast iteration)
- eval_int=12 (end only), probe/test disabled
- fast_eval_figure=True (bottom-left 40%×40% = 16% area)
- NO augmentation (overfitting test - can the model learn at all?)
- batch=32, eval_bs=64, workers=4

  python campaign_archs_7.py --dry-run
  python campaign_archs_7.py --only w044_sanity
  python campaign_archs_7.py --only dual_early,dual_late,dual_gated,dual_asym
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
    c.tra.n_epochs     = 12
    c.tra.eval_int     = 12
    c.tra.test_int     = 999  # disabled
    c.tra.probe_int    = 999  # disabled
    c.tra.save_int     = 5
    c.tra.log_dir      = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = get_default_lr()
    c.data.eval_infer_bs = 64
    c.tra.eval_int_scrolls = 1
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda    = 0.0
    c.dl.batch_size    = 32
    c.dl.num_workers   = 4
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
    # c.dl.flip_prob           = 0.25
    # c.dl.rotation_prob       = 0.25
    # c.dl.noise_prob          = 0.15
    # c.dl.brightness_prob     = 0.15
    # c.dl.contrast_prob       = 0.15
    # c.dl.cutout_prob         = 0.1
    # c.dl.cutout_max_frac     = 0.1
    # c.dl.cutout_n_patches    = 1
    c.dl.flip_prob           = 0
    c.dl.rotation_prob       = 0
    c.dl.noise_prob          = 0
    c.dl.brightness_prob     = 0
    c.dl.contrast_prob       = 0
    c.dl.cutout_prob         = 0
    c.dl.cutout_max_frac     = 0
    c.dl.cutout_n_patches    = 0
    
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
    # scrolls=[ScrollConfig(20250628074500, split_axis="y", train_split_frac=0.75)],  # PHerc0500P2 only
    scrolls=[ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055)], #w044 only
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
    # _mk7("baseline", "depsc_high_p500p2"),

    # BASELINE: despc high on w044
    _mk7("w044_sanity", "w044_solo_check",
     scrolls=[ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055)]),

    # BASELINE: w044 with NO AUGS, basically an overfitting test
    _mk7("w044_noaug", "w044_no_augs",
     scrolls=[ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055)],
     flip_prob=0.0, rotation_prob=0.0, noise_prob=0.0,
     brightness_prob=0.0, contrast_prob=0.0,
     cutout_prob=0.0, cutout_max_frac=0.0, cutout_n_patches=0,
     depth_mask_prob=0.0),
    
    # CONTEXT SIZE ABLATION (villa uses 128x128, our smaller size was used due to a prior competition limit)
    # Both downsample to ~32x32 effective resolution, predict center 16x16 tile
    _mk7("ctx96_ds2", "context_96x96_ds2",
         context_size=96, context_downsample=2),
    
    _mk7("ctx64_ds2", "context_64x64_ds2",
         context_size=64, context_downsample=2),

    # _mk7("ctx128_ds2", "context_128x128_ds2",
    #      context_size=128, context_downsample=2),

    # Foveated: 64x64 context with center 16x16 at full-res, surround ds2
    _mk7("ctx64_fovea", "context_64x64_foveated",
         arch="v16_arch_ctx_fovea", context_size=64, context_downsample=2),
    
    # ==================================================================
    # DUAL-STREAM DEPTH (squashed + non-squashed parallel processing)
    # ==================================================================
    # 7. Early fusion: fuse after stems, before deep layers
    _mk7("dual_early", "dual_stream_early_fusion",
         arch="v16_dual_stream_early"),
    
    # 8. Late fusion: ensemble-like combination at logits level
    _mk7("dual_late", "dual_stream_late_fusion",
         arch="v16_dual_stream_late"),
    
    # 9. Gated fusion: learnable attention weights between streams
    _mk7("dual_gated", "dual_stream_gated_fusion",
         arch="v16_dual_stream_gated"),
    
    # 10. Asymmetric: lightweight squashed (context), heavy non-squashed (detail)
    _mk7("dual_asym", "dual_stream_asymmetric",
         arch="v16_dual_stream_asym"),
    
    # ==================================================================
    # HYBRID DEPTH ATTENTION (Villa-style but adapted to our needs)
    # ==================================================================
    # 11. Per-window attention (Villa approach per 8-slice window)
    _mk7("hybrid_win", "hybrid_attn_per_window",
         arch="v16_hybrid_depth_per_window"),
    
    # 12. Global attention (single attention over all 24 slices, risky)
    _mk7("hybrid_global", "hybrid_attn_global",
         arch="v16_hybrid_depth_global"),
    
    # 13. Triple-branch (attention + max + mean for robustness)
    _mk7("hybrid_triple", "hybrid_attn_triple_branch",
         arch="v16_hybrid_depth_triple"),
    
    # 14. Gated hybrid (learnable mix of attention vs max per window)
    _mk7("hybrid_gated", "hybrid_attn_gated_mix",
         arch="v16_hybrid_depth_gated"),
    
    # ==================================================================
    # MULTI-SCALE & EFFICIENT ARCHITECTURES
    # ==================================================================
    # 15. Multi-scale pyramid: process at 3 scales (1x, 0.5x, 0.25x), fuse
    _mk7("multiscale", "multiscale_pyramid_fusion",
         arch="v16_multiscale_pyramid"),
    
    # 16. Depth squeeze-excitation: channel attention on depth dimension
    _mk7("depth_se", "depth_squeeze_excitation",
         arch="v16_depth_se"),
    
    # 17. Depthwise separable 3D: lighter, preserves depth separation
    _mk7("depthsep", "depthwise_separable_3d",
         arch="v16_depthwise_sep"),
    
    # 18. Mixed-depth windows: 3 non-overlapping + 2 at seams simultaneously
    _mk7("mixed_win", "mixed_depth_windows",
         arch="v16_mixed_depth_windows"),
    
    # 19. Octave convolutions: high/low frequency separate paths
    _mk7("octave", "octave_conv_hf_lf",
         arch="v16_octave_conv"),
    
    # 20. EfficientNet compound scaling: balanced width/depth/resolution
    _mk7("efficient", "efficientnet_compound_scale",
         arch="v16_efficientnet_scale"),
    
    # ==================================================================
    # ATTENTION MECHANISMS (modern, proven effective)
    # ==================================================================
    # 21. Non-local blocks: self-attention within depth dimension
    _mk7("nonlocal", "nonlocal_depth_attention",
         arch="v16_nonlocal_depth"),
    
    # 22. Coordinate attention: channel + spatial awareness (CVPR 2021)
    _mk7("coord_attn", "coordinate_attention",
         arch="v16_coord_attention"),
    
    # 23. Deformable convolutions: learn offset patterns for ink
    _mk7("deform", "deformable_conv_3d",
         arch="v16_deformable_conv"),
    
    # 24. Progressive depth refinement: coarse-to-fine depth processing
    _mk7("progressive", "progressive_depth_refine",
         arch="v16_progressive_depth"),
    
    # 25. Dual attention: channel + spatial applied together
    _mk7("dual_attn", "dual_channel_spatial_attn",
         arch="v16_dual_attention"),
    
    # 26. Axial attention: attention along H, W, D axes separately (cheaper)
    _mk7("axial", "axial_attention_3d",
         arch="v16_axial_attention"),
    
    # ==================================================================
    # ADVANCED FUSION & REFINEMENT
    # ==================================================================
    # 27. Feature Pyramid Network: multi-level feature fusion
    _mk7("fpn", "feature_pyramid_network",
         arch="v16_fpn"),
    
    # 28. Bi-directional FPN: top-down + bottom-up paths
    _mk7("bifpn", "bidirectional_fpn",
         arch="v16_bifpn"),
    
    # 29. Ghost convolutions: cheaper feature generation (Huawei 2020)
    _mk7("ghost", "ghost_convolutions",
         arch="v16_ghost_conv"),
    
    # 30. Inverted residuals: MobileNetV2-style bottlenecks
    _mk7("inverted", "inverted_residuals",
         arch="v16_inverted_residual"),
    
    # 31. ResNeXt grouped convolutions: cardinality over depth
    _mk7("resnext", "resnext_grouped_conv",
         arch="v16_resnext_groups"),
    
    # 32. Temporal shift adapted for depth: shift features across depth
    _mk7("depth_shift", "depth_shift_module",
         arch="v16_depth_shift"),
    
    # ==================================================================
    # ARCHS5 CARRYOVERS: quick sanity checks
    # ==================================================================
    # 33. GCE with q=0.9 (proven noise-robust)
    _mk7("gce_noise", "gce_q09",
         gce_q=0.9),
    
    # 34. 5 depth windows (proven effective)
    _mk7("five_win", "5_depth_windows",
         n_depth_windows=5),
    
    # ==================================================================
    # VILLA-INSPIRED NORMALIZATION (depth attention covered above)
    # ==================================================================
    # 35. Robust-MAD normalization (per-patch median/MAD instead of global z-score)
    _mk7("villa_mad_norm", "robust_mad_normalization",
         normalization_mode="robust_mad"),
    
    # 36. GroupNorm + LeakyReLU (batch-independent, stable for small tiles)
    _mk7("villa_groupnorm", "groupnorm_leakyrelu",
         normalization_layer="group", activation="leaky"),
    
    # 37. InstanceNorm + LeakyReLU (villa's choice, test despite small-tile risk)
    _mk7("villa_instancenorm", "instancenorm_leakyrelu",
         normalization_layer="instance", activation="leaky"),
    
    # 38. LayerNorm (transformer-style, per-sample channel-wise)
    _mk7("villa_layernorm", "layernorm_leakyrelu",
         normalization_layer="layer", activation="leaky"),
    
    # ==================================================================
    # ARCHS6 RADICAL ARCHITECTURES (from campaign_archs_6)
    # ==================================================================
    # 39. 3D Vision Transformer (pure self-attention)
    _mk7("vit3d", "vision_transformer_3d",
         arch="vit3d", init_weights=None),
    
    # 40. Swin Transformer 3D (hierarchical shifted windows)
    _mk7("swin3d", "swin_transformer_3d",
         arch="swin3d", init_weights=None),
    
    # 41. ConvNeXt 3D (modernized CNN with 7x7 kernels)
    _mk7("convnext3d", "convnext_3d",
         arch="convnext3d", init_weights=MAE_CKPT),
    
    # 42. XCiT 3D (cross-covariance attention)
    _mk7("xcit3d", "xcit_3d",
         arch="xcit3d", init_weights=None),
    
    # 43. nnU-Net 3D (encoder-decoder with deep supervision)
    _mk7("nnunet3d", "nnunet_3d",
         arch="nnunet3d", init_weights=None,
         attn_mil=False),  # deep supervision replaces MIL
    
    # 44. Slot Attention 3D (object-centric representation)
    _mk7("slot3d", "slot_attention_3d",
         arch="slot3d", init_weights=None,
         attn_mil=False),  # slot attention replaces MIL

    # 45. nnU-Net 3D with the archs5 soft augmentation recipe restored
    _mk7("nnunet3d_softaug", "nnunet_3d_softaug",
         arch="nnunet3d", init_weights=None,
         attn_mil=False,
         flip_prob=0.25, rotation_prob=0.25, noise_prob=0.15,
         brightness_prob=0.15, contrast_prob=0.15,
         cutout_prob=0.1, cutout_max_frac=0.1, cutout_n_patches=1,
         depth_mask_prob=0.0),

    # ==================================================================
    # NNUNET-INSPIRED BASELINE UPGRADES
    # ==================================================================
    # 46. Delay voxel collapse: keep 32 channels/window through stage-2
    _mk7("latecollapse32", "late_collapse_32ch",
         arch="v16_latecollapse32"),

    # 47. Delay voxel collapse harder: keep 64 channels/window through stage-2
    _mk7("latecollapse64", "late_collapse_64ch",
         arch="v16_latecollapse64"),

    # 48. Shallow U-Net fusion over rich window features, crop only at final MIL
    _mk7("late_unet", "late_feature_unet_fusion",
         arch="v16_late_unet"),

    # 49. nnU-Net with context_downsample=2 honored explicitly
    _mk7("nnunet3d_ds2", "nnunet_3d_ds2",
         arch="nnunet3d_ds", init_weights=None,
         attn_mil=False, context_downsample=2),
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
     print(f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
            f"  entropy={c.model.attn_entropy_weight}")
     print(f"  SINGLE-SCROLL: w044 only")
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
     ap = argparse.ArgumentParser(description="campaign_archs_7: villa-inspired architecture sweep on w044")
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
     print(f"[archs7] SINGLE-SCROLL w044: comprehensive 49-test architecture exploration")
     print(f"[archs7] Baseline: depsc_high (depth SupCon lambda=0.3, ctx=48 ds=2)")
     print(f"[archs7] Groups: 1-2 sanity | 3-6 context | 7-10 dual-stream | 11-14 hybrid-depth")
     print(f"[archs7]         15-20 multi-scale | 21-26 attention | 27-32 fusion")
     print(f"[archs7]         33-34 archs5 | 35-38 villa-norm | 39-45 radical | 46-49 nnunet-lifts")
     print(f"[archs7] Config: 12ep, mostly no-aug except explicit aug tests, fast_eval (16% area)")

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
