"""campaign_archs_6.py -- RADICAL architecture search based on cutting-edge vision papers (2026-08-08).

MOTIVATION: we've plateaued with incremental CNN variations. time to throw the kitchen sink at it.
six architectures from six different paradigms, all adapted to the 3D ink detection problem.

WHAT WE KNOW WORKS (carry forward):
  - MAE pretraining on papyrus texture
  - LCN preprocessing (removes bulk-density baseline at 113 keV)
  - zgrad (dI/dz highlights ink-layer interfaces)
  - context=48px (competition limit, helps with spatial context)
  - depth=24 slices (captures full ring signature)
  - supcon curriculum + attn_mil + entropy regularization
  - ring negatives with closed label source

SIX NEW ARCHITECTURES (all use the proven training stack above):
  1. **3D-ViT** (Dosovitskiy et al. 2020) - pure self-attention on 3D patches. NO convolutions.
     hypothesis: global receptive field from layer 1 can learn long-range ink correlations CNNs miss.
  
  2. **Swin3D** (Liu et al. 2021) - shifted-window 3D transformer with hierarchical pooling.
     hypothesis: local attention windows are more parameter-efficient than global, and shifting
     them lets info propagate globally. hierarchical features (like CNN) but with attention.
  
  3. **ConvNeXt3D** (Liu et al. 2022) - modernized CNN: depthwise separable, LayerNorm, GELU, 7x7 kernels.
     hypothesis: CNNs can match transformers IF designed correctly. larger kernels capture more context.
  
  4. **XCiT3D** (El-Nouby et al. 2021) - cross-covariance image transformers.
     hypothesis: XCA (cross-covariance attention) is more efficient than dot-product attention and
     learns better feature interactions for fine-grained discrimination (ink vs papyrus texture).
  
  5. **nnU-Net3D** (Isensee et al. 2021) - self-configuring U-Net with deep supervision.
     hypothesis: multi-scale features + skip connections let the model see both local ink texture
     and global spatial layout. deep supervision enforces good intermediate representations.
  
  6. **SlotAttention3D** (Locatello et al. 2020) - object-centric iterative attention.
     hypothesis: treat each potential ink region as an "object slot". slot attention learns to
     bind features to ink regions without explicit localization supervision. interpretable.

EACH TEST RUNS 15 EPOCHS WITH THE PROVEN BASELINE STACK. models are NOT comparable in params
(that's intentional -- we're testing paradigms, not tuning FLOPs). verdict = valid F1 / AUC.

  python campaign_archs_6.py --dry-run
  python campaign_archs_6.py --only vit3d
  python campaign_archs_6.py
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, ScrollConfig
from utils.platform import is_high_perf, get_zarr_dir

MAE_CKPT = "models/mae_twostage.pth"
LOG_DIR = "./runs_archs2"
N_EP = 15


def _base_config(exp_name: str, arch: str) -> Config:
    """baseline config: proven training stack + the new architecture"""
    c = Config()
    c.exp_name = exp_name
    c.model.arch = arch
    
    # Set platform-aware zarr path
    c.data.zarr_path = get_zarr_dir()
    
    # Use all 17 downloaded training fragments (multi-scroll mode)
    # The scrolls list is already configured with proper train/val splits in config.py DEFAULT_SCROLLS
    # No need to override unless doing single-scroll pilot tests
    
    # data: 48px context, 24-slice depth, 16px tiles, depth window 4-28
    c.data.tile_size     = 16
    c.data.depth         = 24
    c.data.train_d_start = 4
    c.data.train_d_end   = 28
    c.data.d_start = 4
    c.data.d_end   = 28
    c.data.context_size        = 48
    c.data.context_downsample  = 2
    
    # regularization
    c.model.conv1_drop = 0.15
    c.model.conv2_drop = 0.15
    c.model.head_drop  = 0.4
    
    # training
    c.tra.n_epochs     = N_EP
    c.tra.eval_int     = 999
    c.tra.test_int     = 999
    c.tra.probe_int    = N_EP
    c.tra.save_int     = 5
    c.tra.log_dir      = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = 1.5e-4 if is_high_perf() else 1.0e-4
    c.tra.weight_decay = 3e-1
    c.tra.ranking_lambda = 0.5
    c.tra.ranking_neg_frac = 1.0
    
    # dataloader
    c.dl.data_aug      = True
    c.dl.flip_prob     = 0.6
    c.dl.rotation_prob = 0.6
    c.dl.noise_prob    = 0.3
    c.dl.brightness_prob = 0.6
    c.dl.contrast_prob = 0.6
    c.dl.cutout_prob   = 0.4
    c.dl.cutout_max_frac = 0.2
    c.dl.cutout_n_patches = 2
    c.dl.depth_mask_prob = 0.0
    
    # ring negatives
    c.data.ring_negatives    = True
    c.data.ring_label_source = "closed"
    c.data.ring_close_r      = 3
    c.data.ring_gap_r        = 3
    c.data.ring_shell_r      = 2
    
    # efficiency
    c.data.eval_infer_bs = 256 if is_high_perf() else 32
    c.tra.eval_int_scrolls = 1
    c.data.mask_memmap = True
    c.data.mask_bitpack = True
    c.tra.epoch_cooldown_secs = 0 if is_high_perf() else 18
    c.tra.val_cooldown_secs = 0 if is_high_perf() else 24
    c.tra.eval_cooldown_secs = 0 if is_high_perf() else 120
    c.tra.fig_chunk_cooldown_ms = 0 if is_high_perf() else 120
    c.tra.dann_n_domains = 16
    
    return c


def _mk6(tid, tag, arch, **train_overrides):
    """create a test dict: tid, tag, arch (required), plus optional training config overrides"""
    return dict(tid=tid, tag=tag, arch=arch, **train_overrides)


TESTS = [
    # ==============================================================================
    # 1. 3D VISION TRANSFORMER (ViT3D)
    # ==============================================================================
    # pure self-attention, no convolutions. patchify 3D input -> linear project -> transformer.
    # hypothesis: global receptive field from layer 1 lets it learn long-range correlations
    # that CNNs need many layers to capture. downside: quadratic complexity in sequence length.
    # architecture: patch_size=4 (48/4=12 patches per dim -> 12*12*6=864 tokens at ds=2),
    # 6 transformer layers, 8 heads, dim=256. MLP ratio=4. droppath=0.1.
    _mk6("vit3d", "vision_transformer_3d", "vit3d",
         init_weights=None,  # no MAE checkpoint for ViT (different architecture)
         supcon=True, supcon_temp=0.07,
         supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5,
         supcon_curriculum_epochs=10,
         attn_mil=True, attn_entropy_weight=0.03),

    # ==============================================================================
    # 2. SWIN TRANSFORMER 3D (Swin3D)
    # ==============================================================================
    # hierarchical transformer with shifted windows: local attention (efficient) + shifting
    # (global info propagation). 4 stages with 2/2/6/2 blocks, window_size=4, patch_size=2.
    # hypothesis: local windows are enough for ink detection (ink is spatially local), and
    # hierarchical pooling (like CNNs) gives multi-scale features critical for this task.
    _mk6("swin3d", "swin_transformer_3d", "swin3d",
         init_weights=None,
         supcon=True, supcon_temp=0.07,
         supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5,
         supcon_curriculum_epochs=10,
         attn_mil=True, attn_entropy_weight=0.03),

    # ==============================================================================
    # 3. CONVNEXT 3D
    # ==============================================================================
    # "A ConvNet for the 2020s": depthwise separable 7x7 kernels, LayerNorm, GELU, inverted
    # bottlenecks (like transformers). NO BatchNorm, NO ReLU. 4 stages: [32, 64, 128, 256]ch.
    # hypothesis: modern CNN design principles (larger kernels, better normalization) can match
    # transformers for this task, with fewer parameters and better inductive bias for 3D.
    _mk6("convnext3d", "convnext_3d", "convnext3d",
         init_weights=MAE_CKPT,  # MAE checkpoint can transfer to stage 1 if arch compatible
         supcon=True, supcon_temp=0.07,
         supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5,
         supcon_curriculum_epochs=10,
         attn_mil=True, attn_entropy_weight=0.03),

    # ==============================================================================
    # 4. XCIT 3D (Cross-Covariance Image Transformer)
    # ==============================================================================
    # XCA (cross-covariance attention) instead of dot-product: O(d^2) not O(N^2), where d=embed_dim.
    # for large N (864 tokens here), XCA is much cheaper. 12 layers, dim=256, 8 heads.
    # hypothesis: cross-covariance explicitly models feature interactions (not just token sim),
    # which is better for fine-grained discrimination (ink texture vs papyrus texture).
    _mk6("xcit3d", "xcit_3d", "xcit3d",
         init_weights=None,
         supcon=True, supcon_temp=0.07,
         supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5,
         supcon_curriculum_epochs=10,
         attn_mil=True, attn_entropy_weight=0.03),

    # ==============================================================================
    # 5. nnU-Net 3D (self-configuring U-Net with deep supervision)
    # ==============================================================================
    # encoder-decoder with skip connections, multi-scale feature fusion. 5 stages down, 4 up.
    # deep supervision: auxiliary heads at each decoder stage (weighted sum of losses).
    # hypothesis: skip connections preserve fine spatial detail lost in downsampling; multi-scale
    # features let the model see both local ink texture AND global sheet layout. deep supervision
    # enforces good intermediate features (prevents gradient vanishing in deep net).
    _mk6("nnunet3d", "nnunet_3d", "nnunet3d",
         init_weights=None,  # U-Net has encoder-decoder, MAE is encoder-only
         supcon=True, supcon_temp=0.07,
         supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5,
         supcon_curriculum_epochs=10,
         attn_mil=False,  # deep supervision IS the multi-scale aggregation
         attn_entropy_weight=0.0),

    # ==============================================================================
    # 6. SLOT ATTENTION 3D (object-centric representation)
    # ==============================================================================
    # iterative attention mechanism: K slots compete to explain the input. each slot binds to
    # one "object" (here: one ink region). 4 iterations, 8 slots, dim=128. then pool slots.
    # hypothesis: ink regions are spatially discrete "objects". slot attention learns to segment
    # them without localization labels. HIGHLY interpretable: can visualize what each slot attends
    # to. if it works: we get ink localization + classification from tile-level labels alone.
    _mk6("slot3d", "slot_attention_3d", "slot3d",
         init_weights=None,
         supcon=True, supcon_temp=0.07,
         supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5,
         supcon_curriculum_epochs=10,
         attn_mil=False,  # slot attention IS the aggregation mechanism
         attn_entropy_weight=0.0),
]


# ==============================================================================
# overrides map: {test_key: (config_path, attr_name)} for trainer.py compat
# ==============================================================================
_OVERRIDES = {
    "arch": ("model", "arch"),
    "init_weights": ("tra", "init_weights"),
    "supcon": ("tra", "supcon"),
    "supcon_temp": ("tra", "supcon_temp"),
    "supcon_curriculum": ("tra", "supcon_curriculum"),
    "supcon_lambda_start": ("tra", "supcon_lambda_start"),
    "supcon_lambda_end": ("tra", "supcon_lambda_end"),
    "supcon_curriculum_epochs": ("tra", "supcon_curriculum_epochs"),
    "attn_mil": ("model", "attn_mil"),
    "attn_entropy_weight": ("model", "attn_entropy_weight"),
}


def _apply_overrides(c, test_dict):
    """apply test-specific overrides to config instance"""
    for k, v in test_dict.items():
        if k in ("tid", "tag", "arch"):
            continue
        if k not in _OVERRIDES:
            print(f"[WARN] unknown override key '{k}' in test {test_dict.get('tag', '?')}")
            continue
        path, attr = _OVERRIDES[k]
        parent = getattr(c, path)
        setattr(parent, attr, v)
    return c


def run_campaign(tests_to_run=None, dry_run=False):
    """run the architecture campaign: 6 tests, 15 epochs each"""
    tests = TESTS if tests_to_run is None else [t for t in TESTS if t["tid"] in tests_to_run]
    
    print(f"\n{'='*80}\n[campaign_archs_6] {len(tests)} test(s), {N_EP} epochs each\n{'='*80}")
    if dry_run:
        for t in tests:
            print(f"  {t['tid']:12s}  {t['tag']:40s}  arch={t['arch']}")
        print("\n[dry-run] exiting without training")
        return
    
    from train import Trainer
    results = []
    
    for i, test in enumerate(tests, 1):
        tid, tag, arch = test["tid"], test["tag"], test["arch"]
        exp = f"c6_{tid}"
        print(f"\n{'='*80}\n=== {i}/{len(tests)}: {tid} ({tag}) ===\n{'='*80}")
        
        try:
            c = _base_config(exp, arch)
            c = _apply_overrides(c, test)
            
            trainer = Trainer(c)
            trainer.run()
            
            # extract final metrics
            best_f1 = getattr(trainer, "best_f1", 0.0)
            best_auc = getattr(trainer, "best_auc", 0.0)
            results.append((tid, tag, arch, best_f1, best_auc, "OK"))
            
            # cleanup
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\n[{tid}] DONE  F1={best_f1:.4f}  AUC={best_auc:.4f}")
        
        except Exception as e:
            print(f"\n[{tid}] FAILED: {e}")
            traceback.print_exc()
            results.append((tid, tag, arch, 0.0, 0.0, f"FAIL: {e}"))
        
        # cooldown between tests
        if i < len(tests):
            time.sleep(5)
    
    # summary
    print(f"\n{'='*80}\n[campaign_archs_6] SUMMARY\n{'='*80}")
    print(f"{'TID':<12} {'TAG':<40} {'ARCH':<15} {'F1':>8} {'AUC':>8} {'STATUS':<20}")
    print("-" * 110)
    for tid, tag, arch, f1, auc, status in results:
        print(f"{tid:<12} {tag:<40} {arch:<15} {f1:>8.4f} {auc:>8.4f} {status:<20}")
    print("=" * 110)


def main():
    ap = argparse.ArgumentParser(description="campaign_archs_6: radical architecture search")
    ap.add_argument("--dry-run", action="store_true", help="print tests without running")
    ap.add_argument("--only", type=str, help="run only the test with this tid (e.g. --only vit3d)")
    args = ap.parse_args()
    
    tests_to_run = [args.only] if args.only else None
    run_campaign(tests_to_run, args.dry_run)


if __name__ == "__main__":
    main()
