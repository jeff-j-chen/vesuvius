"""campaign_archs_19.py -- standalone c32 feature/surface/character baseline.

No previous campaign is imported. Every effective setting is assigned in
base_config() so the experiment can be audited without following inheritance.
Noise, brightness/contrast, and FDA are removed after campaign 18 showed no
benefit; synchronized flips and rotations remain as exact symmetries.
Every arm is capped at 20,000 training windows per epoch so larger centers do
not receive extra optimizer steps merely because they overlap more ring cells.

Tests:
- baseline: c32 center, 4x4 8px targets
- mae192_ibn: baseline with matched 192px/ds2 + IBN MAE checkpoint
- c64_t8: c64 center, 8x8 8px targets
- c64_t16: c64 center, 4x4 16px targets
- surface_strong: baseline with surface loss 0.10 -> 0.20
- context_cutout: baseline with center-protected context cutout
- context_jitter: baseline with target-aware +/-32px context jitter
- context_replace: feathered, surface-aligned real-papyrus context replacement
- bce_soft: BCE with mild positive/negative target smoothing
- gce_q03/q07/q09: hard-label GCE sweep
- gce_q03_soft: one GCE x soft-label interaction

  python campaign_archs_19.py --dry-run
  python campaign_archs_19.py --only baseline
  python campaign_archs_19.py --only c64_t8,c64_t16
  python campaign_archs_19.py
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, DEFAULT_SCROLLS
from utils.platform import get_zarr_dir

LOG_DIR = "./runs_archs19"
_W013_ID = 20240304141531
_W013 = [scroll for scroll in DEFAULT_SCROLLS if int(scroll.scroll_id) == _W013_ID]


def base_config(exp_name: str) -> Config:
    """construct the complete campaign-19 operating point without campaign inheritance."""
    config = Config()
    config.exp_name = exp_name
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.model_dir = "models"
    config.init_weights = "models/mae_nnunet_96.pth"

    config.data.zarr_path = get_zarr_dir()
    config.data.scrolls = list(_W013)
    config.data.tile_size = 16
    config.data.depth = 24
    config.data.train_d_start = 4
    config.data.train_d_end = 28
    config.data.d_start = 4
    config.data.d_end = 28
    config.data.context_size = 192
    config.data.context_downsample = 2
    config.data.ctx_jitter = 0
    config.data.target_aware_ctx_jitter = False
    config.data.depth_jitter = 4
    config.data.simple_split = False
    config.data.train_mask_dir = "./train_masks"
    config.data.mask_memmap = True
    config.data.mask_bitpack = True
    config.data.ring_negatives = True
    config.data.ring_label_source = "closed"
    config.data.ring_close_r = 3
    config.data.ring_gap_r = 3
    config.data.ring_shell_r = 2
    config.data.inklabel_dir = "./eroded_inklabels"
    config.data.dot_inklabel_dir = ""
    config.data.dot_scroll_whitelist = []
    config.data.multitile_train_step = 16
    config.data.multitile_pos_only = True
    config.data.character_balanced_sampling = True
    config.data.character_balance_scrolls = True
    config.data.character_min_pixels = 8
    config.data.max_samples_per_epoch = 20_000
    config.data.eval_infer_bs = 64
    config.data.eval_prefetch = 3
    config.data.tta_mode = "light"
    config.data.vis_scroll_ids = [_W013_ID]

    config.model.arch = "nnunet3d_lcndz"
    config.model.multitile = True
    config.model.multitile_subtile = 8
    config.model.multitile_grid = 4
    config.model.attn_mil = False
    config.model.feature_attn_mil = True
    config.model.attn_entropy_weight = 0.03
    config.model.learned_surface = True
    config.model.new_learned_surface = True
    config.model.surface_guided_mil = False
    config.model.surface_guided_mix = 0.5
    config.model.surface_band_sigma = 1.5
    config.model.use_ibn = True
    config.model.conv1_drop = 0.05
    config.model.conv2_drop = 0.05
    config.model.head_drop = 0.10
    config.model.skip_drop = 0.20
    config.model.no_dz = False
    config.model.channels_mult = 1.0

    config.tra.n_epochs = 15
    config.tra.lr = 1e-4
    config.tra.weight_decay = 0.3
    config.tra.l1_lambda = 0.0
    config.tra.grad_norm = 0.5
    config.tra.patience = 5
    config.tra.lr_decay = 0.5
    config.tra.save_int = 15
    config.tra.log_dir = LOG_DIR
    config.tra.eval_int = 15
    config.tra.eval_int_scrolls = 1
    config.tra.test_int = 999
    config.tra.probe_int = 999
    config.tra.fast_eval_figure = False
    config.tra.test_on_final = False
    config.tra.loss_type = "bce"
    config.tra.gce_q = 0.7
    config.tra.label_smooth_pos = 0.0
    config.tra.label_smooth_neg = 0.0
    config.tra.tile_pos_weight = 0.0
    config.tra.tile_pos_weight_auto = True
    config.tra.tta_consistency = True
    config.tra.tta_consistency_lambda = 0.3
    config.tra.tta_consistency_mode = "flips"
    config.tra.tta_consistency_prob = 1.0
    config.tra.spill_reduction = True
    config.tra.spill_lambda = 0.5
    config.tra.spill_min_depth_var = 0.8
    config.tra.spill_depth_threshold = 0.35
    config.tra.spill_active_depth_tau = 0.08
    config.tra.spill_max_active_depth_frac = 0.35
    config.tra.spill_prob = False
    config.tra.spill_entropy = False
    config.tra.supcon = True
    config.tra.supcon_lambda = 0.1
    config.tra.supcon_temp = 0.07
    config.tra.supcon_proj_dim = 128
    config.tra.supcon_hidden_dim = 256
    config.tra.supcon_curriculum = True
    config.tra.supcon_lambda_start = 0.05
    config.tra.supcon_lambda_end = 0.5
    config.tra.supcon_curriculum_epochs = 8
    config.tra.supcon_cross_frag = False
    config.tra.dann = False
    config.tra.dann_lambda = 0.0
    config.tra.dann_n_domains = 1
    config.tra.dann_grl_anneal = False
    config.tra.new_surface_lambda = 0.1
    config.tra.new_surface_smooth_lambda = 0.02
    config.tra.character_macro_metrics = True
    config.tra.character_score_threshold = 0.5
    config.tra.character_recall_target = 0.5
    config.tra.character_max_ring_fpr = 0.1
    config.tra.character_checkpoint_metric = "character_ap_macro"
    config.tra.aug_start_epoch = 0
    config.tra.deterministic = False
    config.tra.epoch_cooldown_secs = 0
    config.tra.val_cooldown_secs = 0
    config.tra.eval_cooldown_secs = 0
    config.tra.fig_chunk_cooldown_ms = 0

    config.dl.batch_size = 32
    config.dl.num_workers = 8
    config.dl.data_aug = True
    config.dl.flip_prob = 0.6
    config.dl.rotation_prob = 0.6
    config.dl.noise_prob = 0.0
    config.dl.brightness_prob = 0.0
    config.dl.contrast_prob = 0.0
    config.dl.brightness_delta = 0.15
    config.dl.contrast_delta = 0.15
    config.dl.noise_std_min = 0.001
    config.dl.noise_std_max = 0.005
    config.dl.fda_prob = 0.0
    config.dl.fda_beta = 0.05
    config.dl.elastic_prob = 0.0
    config.dl.cutout_prob = 0.0
    config.dl.cutout_max_frac = 0.0
    config.dl.cutout_n_patches = 0
    config.dl.cutout_protect_center = False
    config.dl.depth_mask_prob = 0.0
    config.dl.depth_warp_prob = 0.0
    config.dl.surface_atten_prob = 0.0
    config.dl.acquisition_blur_prob = 0.0
    config.dl.correlated_noise_prob = 0.0
    config.dl.context_replace_prob = 0.0
    config.dl.context_replace_keep_size = 0
    config.dl.context_replace_margin = 16
    config.dl.context_replace_feather = 16
    config.dl.context_replace_min_mask_frac = 0.8
    config.dl.context_replace_surface_align = True

    config.hm.enabled = False
    return config


TESTS = [
    {"tid": "baseline", "tag": "19_baseline"},
    {
        "tid": "mae192_ibn",
        "tag": "19_mae192_ibn",
        "init_weights": "models/mae_nnunet_192_ibn.pth",
    },
    {"tid": "c64_t8", "tag": "19_c64_t8", "multitile_subtile": 8, "multitile_grid": 8},
    {"tid": "c64_t16", "tag": "19_c64_t16", "multitile_subtile": 16, "multitile_grid": 4},
    {"tid": "surface_strong", "tag": "19_surface_strong", "new_surface_lambda": 0.2},
    {
        "tid": "context_cutout",
        "tag": "19_context_cutout",
        "cutout_prob": 0.5,
        "cutout_max_frac": 0.12,
        "cutout_n_patches": 2,
        "cutout_protect_center": True,
    },
    {
        "tid": "context_jitter",
        "tag": "19_context_jitter",
        "target_aware_ctx_jitter": True,
        "ctx_jitter": 32,
    },
    {
        "tid": "context_replace",
        "tag": "19_context_replace",
        "context_replace_prob": 0.5,
        "context_replace_keep_size": 0,
        "context_replace_margin": 16,
        "context_replace_feather": 16,
        "context_replace_min_mask_frac": 0.8,
    },
    {
        "tid": "bce_soft",
        "tag": "19_bce_soft",
        "loss_type": "bce",
        "label_smooth_pos": 0.10,
        "label_smooth_neg": 0.05,
    },
    {"tid": "gce_q03", "tag": "19_gce_q03", "loss_type": "gce", "gce_q": 0.3},
    {"tid": "gce_q07", "tag": "19_gce_q07", "loss_type": "gce", "gce_q": 0.7},
    {"tid": "gce_q09", "tag": "19_gce_q09", "loss_type": "gce", "gce_q": 0.9},
    {
        "tid": "gce_q03_soft",
        "tag": "19_gce_q03_soft",
        "loss_type": "gce",
        "gce_q": 0.3,
        "label_smooth_pos": 0.10,
        "label_smooth_neg": 0.05,
    },
]

_OVERRIDES = {
    "multitile_subtile": ("model", "multitile_subtile"),
    "multitile_grid": ("model", "multitile_grid"),
    "new_surface_lambda": ("tra", "new_surface_lambda"),
    "cutout_prob": ("dl", "cutout_prob"),
    "cutout_max_frac": ("dl", "cutout_max_frac"),
    "cutout_n_patches": ("dl", "cutout_n_patches"),
    "cutout_protect_center": ("dl", "cutout_protect_center"),
    "target_aware_ctx_jitter": ("data", "target_aware_ctx_jitter"),
    "ctx_jitter": ("data", "ctx_jitter"),
    "context_replace_prob": ("dl", "context_replace_prob"),
    "context_replace_keep_size": ("dl", "context_replace_keep_size"),
    "context_replace_margin": ("dl", "context_replace_margin"),
    "context_replace_feather": ("dl", "context_replace_feather"),
    "context_replace_min_mask_frac": ("dl", "context_replace_min_mask_frac"),
    "loss_type": ("tra", "loss_type"),
    "gce_q": ("tra", "gce_q"),
    "label_smooth_pos": ("tra", "label_smooth_pos"),
    "label_smooth_neg": ("tra", "label_smooth_neg"),
}


def build_config(test: dict) -> Config:
    config = base_config(str(test["tag"]))
    for key, (section, attr) in _OVERRIDES.items():
        if key in test:
            setattr(getattr(config, section), attr, test[key])
    if "init_weights" in test:
        config.init_weights = str(test["init_weights"])
        if not os.path.exists(config.init_weights):
            print(f"[archs19] WARNING checkpoint not found yet: {config.init_weights}")
    os.makedirs("models/archs19", exist_ok=True)
    setattr(config, "save_final", f"models/archs19/{test['tid']}_final.pth")
    return config


def run_test(config: Config, dry_run: bool) -> bool:
    center = int(config.model.multitile_subtile) * int(config.model.multitile_grid)
    targets = int(config.model.multitile_grid) ** 2
    print(f"\n{'=' * 70}\n[archs19] {config.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  scrolls={[s.scroll_id for s in config.data.scrolls]} manual_split={not config.data.simple_split}"
        f"  character_balanced={config.data.character_balanced_sampling}"
    )
    print(
        f"  context={config.data.context_size}/ds{config.data.context_downsample}"
        f"  center={center} subtile={config.model.multitile_subtile}"
        f"  grid={config.model.multitile_grid}x{config.model.multitile_grid} targets={targets}"
    )
    print(
        f"  feature_attn={config.model.feature_attn_mil} surface={config.model.new_learned_surface}"
        f"  surface_lambda={config.tra.new_surface_lambda}"
        f"  ctx_jitter={config.data.ctx_jitter if config.data.target_aware_ctx_jitter else 0}"
        f"  cutout={config.dl.cutout_prob} context_replace={config.dl.context_replace_prob}"
    )
    print(
        f"  epochs={config.tra.n_epochs} batch={config.dl.batch_size} lr={config.tra.lr:.2e}"
        f"  samples/epoch={config.data.max_samples_per_epoch}"
        f"  loss={config.tra.loss_type} q={config.tra.gce_q}"
        f"  smooth=({config.tra.label_smooth_pos},{config.tra.label_smooth_neg})"
        f"  init={config.init_weights}"
    )
    if dry_run:
        print("  [DRY RUN] skipping")
        return True

    from train import Trainer

    try:
        trainer = Trainer(config)
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign_archs_19: standalone improved w013 baseline")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = TESTS
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [test for test in TESTS if str(test["tid"]) in wanted]
        missing = wanted - {str(test["tid"]) for test in selected}
        if missing:
            print(f"[ABORT] --only id(s) {sorted(missing)} not found; valid: "
                  f"{[str(test['tid']) for test in TESTS]}")
            return
    elif args.from_id:
        ids = [str(test["tid"]) for test in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[archs19] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print("[archs19] standalone c32 feature-attn + surface + character-balanced baseline")

    results = {}
    for test in selected:
        tid = str(test["tid"])
        config = build_config(test)
        results[tid] = "OK" if run_test(config, args.dry_run) else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print(f"\n{'=' * 70}\n[archs19] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
