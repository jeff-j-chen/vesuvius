"""campaign_archs_20.py -- combined campaign-19 findings.

Future arms use the campaign-19/20 operating point selected from metrics and figures:
- matched MAE initialization
- 192px context at ds2
- c64 center with 4x4 16px targets
- protected context cutout and target-aware context jitter
- weakened surface-aligned real-context replacement
- BCE with positive/negative label smoothing
- strong surface supervision

Tests:
- future_baseline: corrected ctx128 operating point for all future comparisons
- bce_soft: matched ctx192 BCE with positive/negative label smoothing
- context_consistency: explicit target-logit invariance to distant context replacement
- character_bag_rank: rank each character bag above its assigned local ring
- character_groupdro: persistent worst-character reweighting
- character_cvar: optimize the worst quartile of characters per batch
- surface_canonical: align the 24-slice ink input to the physical surface
- surface_slice8: classify from eight surface-relative slices after 24-slice localization
- jepa192: initialize from 3D JEPA instead of voxel-reconstruction MAE
- multi3_control: balanced three-scroll control without domain adaptation
- dann_0025/005/01/02: weak annealed three-scroll DANN sweep
- context_consistency_strong: context consistency lambda 0.3
- character_bag_rank_strong: character ranking lambda 0.4
- character_groupdro_fast: GroupDRO eta 0.1
- character_cvar_half: optimize the worst half of represented characters
- surface_slice12: classify from twelve surface-relative slices

  python campaign_archs_20.py --dry-run
    python campaign_archs_20.py --only future_baseline_5090
    python campaign_archs_20.py --only multi3_control_5090,dann_0025_5090
  python campaign_archs_20.py
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, DEFAULT_SCROLLS
from utils.platform import get_zarr_dir

LOG_DIR = "./runs_archs20_5090"
_W013_ID = 20240304141531
_W013 = [scroll for scroll in DEFAULT_SCROLLS if int(scroll.scroll_id) == _W013_ID]
_THREE_MASK_IDS = {20240304141531, 20250628074500, 20260115000000}
_THREE_SCROLLS = [
    scroll for scroll in DEFAULT_SCROLLS
    if int(scroll.scroll_id) in _THREE_MASK_IDS
]
if len(_THREE_SCROLLS) != 3:
    raise RuntimeError(f"expected three train-mask scrolls, found {_THREE_SCROLLS}")


def base_config(exp_name: str) -> Config:
    """construct the complete campaign-20 operating point."""
    config = Config()
    config.exp_name = exp_name
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.model_dir = "models"
    config.init_weights = "models/mae_nnunet_192_ibn.pth"

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
    config.data.ctx_jitter = 32
    config.data.target_aware_ctx_jitter = True
    config.data.depth_jitter = 4
    config.data.simple_split = False
    config.data.train_mask_dir = "./train_masks"
    config.data.mask_memmap = True
    config.data.mask_bitpack = True
    config.data.preload_volumes = True
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
    config.data.eval_infer_bs = 96
    config.data.eval_prefetch = 3
    config.data.eval_chunk_gb = 3.0
    config.data.tta_mode = "light"
    config.data.vis_scroll_ids = [_W013_ID]

    config.model.arch = "nnunet3d_lcndz"
    config.model.compile_model = True
    config.model.multitile = True
    config.model.multitile_subtile = 16
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
    config.tra.lr = 1.2e-4
    config.tra.warmup_epochs = 5
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
    config.tra.gce_q = 0.9
    config.tra.label_smooth_pos = 0.10
    config.tra.label_smooth_neg = 0.05
    config.tra.tile_pos_weight = 1
    config.tra.tile_pos_weight_auto = False
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
    config.tra.new_surface_lambda = 0.2
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

    config.dl.batch_size = 48
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
    config.dl.cutout_prob = 0.65
    config.dl.cutout_max_frac = 0.16
    config.dl.cutout_n_patches = 3
    config.dl.cutout_protect_center = True
    config.dl.depth_mask_prob = 0.0
    config.dl.depth_warp_prob = 0.0
    config.dl.surface_atten_prob = 0.0
    config.dl.acquisition_blur_prob = 0.0
    config.dl.correlated_noise_prob = 0.0
    config.dl.context_replace_prob = 0.35
    config.dl.context_replace_keep_size = 0
    config.dl.context_replace_margin = 20
    config.dl.context_replace_feather = 20
    config.dl.context_replace_min_mask_frac = 0.8
    config.dl.context_replace_surface_align = True

    config.hm.enabled = False
    return config


TESTS = [
    # {
    #     "tid": "future_baseline",
    #     "tag": "20_future_baseline",
    #     "sanity_guard_epoch": 3,
    #     "sanity_min_character_ap": 0.55,
    #     "sanity_min_specificity": 0.25,
    # },
    # {
    #     "tid": "bce_soft_noweight",
    #     "tag": "20_bce_soft_noweight",
    #     "loss_type": "bce",
    #     "label_smooth_pos": 0.10,
    #     "label_smooth_neg": 0.05,
    # },
    {
        "tid": "context_consistency",
        "tag": "20_context_consistency_soft",
        # paired forwards need extra activation headroom
        # sqrt scaling from b48 gives 1.2e-4 * sqrt(32/48) ~= 9.8e-5
        "batch_size": 32,
        "lr": 1.0e-4,
        "compile_model": False,
        "context_replace_prob": 0.0,
        "context_consistency": True,
        "context_consistency_prob": 0.25,
        "context_consistency_lambda": 0.1,
    },
    {
        "tid": "character_bag_rank",
        "tag": "20_character_bag_rank_soft",
        "character_bag_ranking": True,
        "character_bag_margin": 0.5,
        "character_bag_topk_frac": 0.5,
        "character_bag_lambda": 0.2,
    },
    {
        "tid": "character_groupdro",
        "tag": "20_character_groupdro_soft",
        "character_groupdro": True,
        "character_groupdro_eta": 0.05,
        "character_groupdro_max_ratio": 3.0,
    },
    {
        "tid": "character_cvar",
        "tag": "20_character_cvar_soft",
        "n_epochs": 25,
        "eval_int": 25,
        "character_cvar": True,
        "character_cvar_alpha": 0.25,
    },
    {
        "tid": "surface_canonical",
        "tag": "20_surface_canonical_soft",
        "surface_canonicalize": True,
        "surface_canonical_depth": 24,
        "compile_model": False,
    },
    {
        "tid": "surface_slice8",
        "tag": "20_surface_slice8_soft",
        "surface_canonicalize": True,
        "surface_canonical_depth": 8,
        "compile_model": False,
    },
    {
        "tid": "jepa192",
        "tag": "20_jepa192_soft",
        "init_weights": "models/jepa_nnunet_192_ibn.pth",
    },
    {
        "tid": "multi3_control",
        "tag": "20_multi3_control_soft",
        "scrolls": list(_THREE_SCROLLS),
        "max_samples_per_epoch": 6667,
        "vis_scroll_ids": sorted(_THREE_MASK_IDS),
        "eval_int_scrolls": 3,
        "fast_eval_figure": True,
        "dann": False,
        "dann_n_domains": 3,
    },
    {
        "tid": "dann_0025",
        "tag": "20_dann_0025_soft",
        "batch_size": 32,
        "lr": 1.0e-4,
        "scrolls": list(_THREE_SCROLLS),
        "max_samples_per_epoch": 6667,
        "vis_scroll_ids": sorted(_THREE_MASK_IDS),
        "eval_int_scrolls": 3,
        "fast_eval_figure": True,
        "dann": True,
        "dann_lambda": 0.0025,
        "dann_n_domains": 3,
        "dann_grl_anneal": True,
    },
    {
        "tid": "dann_005",
        "tag": "20_dann_005_soft",
        "batch_size": 32,
        "lr": 1.0e-4,
        "scrolls": list(_THREE_SCROLLS),
        "max_samples_per_epoch": 6667,
        "vis_scroll_ids": sorted(_THREE_MASK_IDS),
        "eval_int_scrolls": 3,
        "fast_eval_figure": True,
        "dann": True,
        "dann_lambda": 0.005,
        "dann_n_domains": 3,
        "dann_grl_anneal": True,
    },
    {
        "tid": "dann_01",
        "tag": "20_dann_01_soft",
        "batch_size": 32,
        "lr": 1.0e-4,
        "scrolls": list(_THREE_SCROLLS),
        "max_samples_per_epoch": 6667,
        "vis_scroll_ids": sorted(_THREE_MASK_IDS),
        "eval_int_scrolls": 3,
        "fast_eval_figure": True,
        "dann": True,
        "dann_lambda": 0.01,
        "dann_n_domains": 3,
        "dann_grl_anneal": True,
    },
    {
        "tid": "dann_02",
        "tag": "20_dann_02_soft",
        "batch_size": 32,
        "lr": 1.0e-4,
        "scrolls": list(_THREE_SCROLLS),
        "max_samples_per_epoch": 6667,
        "vis_scroll_ids": sorted(_THREE_MASK_IDS),
        "eval_int_scrolls": 3,
        "fast_eval_figure": True,
        "dann": True,
        "dann_lambda": 0.02,
        "dann_n_domains": 3,
        "dann_grl_anneal": True,
    },
    {
        "tid": "context_consistency_strong",
        "tag": "20_context_consistency_strong_soft",
        "batch_size": 32,
        "lr": 1.0e-4,
        "compile_model": False,
        "context_replace_prob": 0.0,
        "context_consistency": True,
        "context_consistency_prob": 0.25,
        "context_consistency_lambda": 0.3,
    },
    {
        "tid": "character_bag_rank_strong",
        "tag": "20_character_bag_rank_strong_soft",
        "character_bag_ranking": True,
        "character_bag_margin": 0.5,
        "character_bag_topk_frac": 0.5,
        "character_bag_lambda": 0.4,
    },
    {
        "tid": "character_groupdro_fast",
        "tag": "20_character_groupdro_fast_soft",
        "character_groupdro": True,
        "character_groupdro_eta": 0.1,
        "character_groupdro_max_ratio": 3.0,
    },
    {
        "tid": "character_cvar_half",
        "tag": "20_character_cvar_half_soft",
        "n_epochs": 25,
        "eval_int": 25,
        "character_cvar": True,
        "character_cvar_alpha": 0.5,
    },
    {
        "tid": "surface_slice12",
        "tag": "20_surface_slice12_soft",
        "surface_canonicalize": True,
        "surface_canonical_depth": 12,
        "compile_model": False,
    },
]

for _test in TESTS:
    _test["tid"] = f"{_test['tid']}_5090"
    _test["tag"] = f"{_test['tag']}_5090"

_OVERRIDES = {
    "batch_size": ("dl", "batch_size"),
    "lr": ("tra", "lr"),
    "n_epochs": ("tra", "n_epochs"),
    "eval_int": ("tra", "eval_int"),
    "compile_model": ("model", "compile_model"),
    "context_size": ("data", "context_size"),
    "scrolls": ("data", "scrolls"),
    "max_samples_per_epoch": ("data", "max_samples_per_epoch"),
    "vis_scroll_ids": ("data", "vis_scroll_ids"),
    "eval_int_scrolls": ("tra", "eval_int_scrolls"),
    "fast_eval_figure": ("tra", "fast_eval_figure"),
    "context_replace_prob": ("dl", "context_replace_prob"),
    "context_replace_margin": ("dl", "context_replace_margin"),
    "context_replace_feather": ("dl", "context_replace_feather"),
    "loss_type": ("tra", "loss_type"),
    "label_smooth_pos": ("tra", "label_smooth_pos"),
    "label_smooth_neg": ("tra", "label_smooth_neg"),
    "sanity_guard_epoch": ("tra", "sanity_guard_epoch"),
    "sanity_min_character_ap": ("tra", "sanity_min_character_ap"),
    "sanity_min_specificity": ("tra", "sanity_min_specificity"),
    "dann": ("tra", "dann"),
    "dann_lambda": ("tra", "dann_lambda"),
    "dann_n_domains": ("tra", "dann_n_domains"),
    "dann_grl_anneal": ("tra", "dann_grl_anneal"),
    "context_consistency": ("tra", "context_consistency"),
    "context_consistency_prob": ("tra", "context_consistency_prob"),
    "context_consistency_lambda": ("tra", "context_consistency_lambda"),
    "character_bag_ranking": ("tra", "character_bag_ranking"),
    "character_bag_margin": ("tra", "character_bag_margin"),
    "character_bag_topk_frac": ("tra", "character_bag_topk_frac"),
    "character_bag_lambda": ("tra", "character_bag_lambda"),
    "character_groupdro": ("tra", "character_groupdro"),
    "character_groupdro_eta": ("tra", "character_groupdro_eta"),
    "character_groupdro_max_ratio": ("tra", "character_groupdro_max_ratio"),
    "character_cvar": ("tra", "character_cvar"),
    "character_cvar_alpha": ("tra", "character_cvar_alpha"),
    "surface_canonicalize": ("model", "surface_canonicalize"),
    "surface_canonical_depth": ("model", "surface_canonical_depth"),
}


def build_config(test: dict) -> Config:
    config = base_config(str(test["tag"]))
    for key, (section, attr) in _OVERRIDES.items():
        if key in test:
            setattr(getattr(config, section), attr, test[key])
    if "init_weights" in test:
        config.init_weights = str(test["init_weights"])
    if not os.path.exists(config.init_weights):
        print(f"[archs20] WARNING checkpoint not found yet: {config.init_weights}")
    os.makedirs("models/archs20", exist_ok=True)
    setattr(config, "save_final", f"models/archs20/{test['tid']}_final.pth")
    return config


def run_test(config: Config, dry_run: bool) -> bool:
    center = int(config.model.multitile_subtile) * int(config.model.multitile_grid)
    targets = int(config.model.multitile_grid) ** 2
    protected = center + 2 * int(config.dl.context_replace_margin)
    print(f"\n{'=' * 70}\n[archs20] {config.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  context={config.data.context_size}/ds{config.data.context_downsample}"
        f" center={center} subtile={config.model.multitile_subtile}"
        f" grid={config.model.multitile_grid}x{config.model.multitile_grid} targets={targets}"
    )
    print(
        f"  scrolls={[int(scroll.scroll_id) for scroll in config.data.scrolls]}"
        f" balance_scrolls={config.data.character_balance_scrolls}"
        f" samples/scroll={config.data.max_samples_per_epoch}"
        f" eval_scrolls={config.tra.eval_int_scrolls}"
    )
    print(
        f"  cutout={config.dl.cutout_prob} ctx_jitter={config.data.ctx_jitter}"
        f" context_replace={config.dl.context_replace_prob}"
        f" protected={protected} feather={config.dl.context_replace_feather}"
    )
    print(
        f"  loss={config.tra.loss_type} q={config.tra.gce_q}"
        f" pos_weight={config.tra.tile_pos_weight} auto={config.tra.tile_pos_weight_auto}"
        f" surface_lambda={config.tra.new_surface_lambda} init={config.init_weights}"
    )
    print(
        f"  context_consistency={config.tra.context_consistency}"
        f" bag_rank={config.tra.character_bag_ranking}"
        f" groupdro={config.tra.character_groupdro} cvar={config.tra.character_cvar}"
        f" surface_canonical={config.model.surface_canonicalize}"
        f" canonical_depth={config.model.surface_canonical_depth}"
    )
    print(
        f"  dann={config.tra.dann} lambda={config.tra.dann_lambda}"
        f" domains={config.tra.dann_n_domains} anneal={config.tra.dann_grl_anneal}"
    )
    print(
        f"  epochs={config.tra.n_epochs} batch={config.dl.batch_size} lr={config.tra.lr:.2e}"
        f" workers={config.dl.num_workers} compile={config.model.compile_model}"
        f" eval_bs={config.data.eval_infer_bs} eval_prefetch={config.data.eval_prefetch}"
        f" eval_chunk_gb={config.data.eval_chunk_gb}"
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


def ensure_pretraining(selected: list[dict], dry_run: bool) -> None:
    """create prerequisite self-supervised checkpoints before campaign training."""
    if not any(str(test["tid"]) == "jepa192_5090" for test in selected):
        return
    root = Path(__file__).resolve().parent
    checkpoint = root / "models/jepa_nnunet_192_ibn.pth"
    if checkpoint.exists():
        print(f"[archs20] JEPA checkpoint ready: {checkpoint}")
        return
    command = [
        sys.executable,
        str(root / "jepa_pretrain_nnunet.py"),
        "--name", "jepa_nnunet_192_ibn",
        "--ctx", "192",
        "--ds", "2",
        "--depth", "24",
        "--d-start", "4",
        "--d-end", "28",
        "--steps", "1000",
        "--batch-size", "8",
        "--accum-steps", "4",
        "--require-all-scrolls",
    ]
    if dry_run:
        print(f"[archs20] PREFLIGHT would run: {' '.join(command)}")
        return
    print("[archs20] JEPA checkpoint missing; running pretraining before campaign", flush=True)
    subprocess.run(command, cwd=root, check=True)
    if not checkpoint.exists():
        raise RuntimeError(f"JEPA pretraining completed without creating {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign_archs_20: combined c19 findings")
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

    ensure_pretraining(selected, args.dry_run)
    print(f"[archs20] {len(selected)} test(s) queued (log -> {LOG_DIR})")
    print("[archs20] c64_t16 + context augmentations + soft BCE + strong surface")

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
        if tid == "future_baseline_5090" and results[tid] == "FAIL":
            print("[archs20] guarded baseline failed; aborting remaining tests")
            break

    print(f"\n{'=' * 70}\n[archs20] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
