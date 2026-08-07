"""campaign_archs_4.py -- follow-up from campaign_archs_3 results (2026-08-07).

ANALYSIS FROM ARCHS_2 + ARCHS_3:
  - ds=2 == ds=1: fine spatial detail is irrelevant; signal is in the DEPTH profile
  - surface_dist: strongest learner of any test, but overfit; depth alignment is real
  - learn01 (learned depth attention): best at epoch 15, not converged; needs more epochs
  - DoG/spatial physics: no improvement (wrong abstraction level for rings of 100px+)
  - attn entropy=0.06: too sparse; 0.03 is the sweet spot
  - Higher SupCon proj dims: no effect
  - emb01/02, surf_dog: no improvement over baseline
  - Grand combo: noisy; conflicting gradients
  - surf_reg01/02: correct direction but augmentation is the wrong tool; the issue is
    generalization not dropout -- this is a data cleanliness problem with no easy fix

TWO NEW IDEAS:
  1. TV (total variation) regularizer on the per-voxel logit map:
     forces adjacent voxels to predict similarly -> spatial coherence -> READABILITY.
     the missing piece for "metrics good but letters unreadable."
     applied to the voxel map before MIL aggregation, at no extra forward pass cost.

  2. Depth Profile SupCon (new):
     contrastive learning on the RAW MEAN DEPTH PROFILE at the center tile,
     BEFORE any spatial convolution. completely independent of the spatial SupCon.
     motivated by: depth signal >> spatial signal across all experiments.
     if ink creates a distinctive depth signature (shifted peak, different width, etc.),
     this should cluster ink depth profiles and separate them from papyrus.

ALSO:
  - learn01 for 30 epochs (it was converging!)
  - GCE q=0.9 (more noise-robust; current q=0.7; q=1 = MAE, fully noise-robust)
  - single_w044: train ONLY on w044 (cleanest labels, user's most-cleaned scroll).
    hypothesis: if 15-scroll label diversity hurts more than it helps due to label
    noise from cross-scroll projection, the single-scroll model should generalize BETTER
    on w044 validation (not worse). this tells us if label noise or data diversity matters.

6 tests total.

  python campaign_archs_4.py --dry-run
  python campaign_archs_4.py --only tv01
  python campaign_archs_4.py
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, DEFAULT_SCROLLS, ScrollConfig

MAE_CKPT = "models/mae_twostage.pth"
LOG_DIR = "./runs_archs4"
N_EP = 15


def _base_config(exp_name: str) -> Config:
    """base config for archs4 -- identical hardware setup to archs3."""
    c = Config()
    on_linux = (os.name == "posix")
    c.exp_name = exp_name
    c.model.arch = "v16_arch_ctx"
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
    c.tra.eval_int     = 999
    c.tra.test_int     = 999
    c.tra.probe_int    = N_EP
    c.tra.save_int     = 5
    c.tra.log_dir      = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = 1.5e-4 if on_linux else 1.0e-4
    c.data.eval_infer_bs = 256 if on_linux else 32
    c.tra.eval_int_scrolls = 1
    c.tra.weight_decay = 3e-1
    c.data.ring_label_source = "closed"
    c.tra.tta_consistency = False
    c.tra.l1_lambda    = 0.0
    c.dl.batch_size    = 96 if on_linux else 32
    c.dl.num_workers   = 12 if on_linux else 0
    c.dl.data_aug      = True
    c.data.mask_memmap       = True
    c.data.mask_bitpack      = True
    c.data.ring_negatives    = True
    c.data.ring_close_r      = 3
    c.data.ring_gap_r        = 3
    c.data.ring_shell_r      = 2
    c.tra.ranking_lambda     = 0.5
    c.tra.ranking_neg_frac   = 1.0
    c.dl.flip_prob           = 0.6
    c.dl.rotation_prob       = 0.6
    c.dl.noise_prob          = 0.3
    c.dl.brightness_prob     = 0.6
    c.dl.contrast_prob       = 0.6
    c.dl.cutout_prob         = 0.4
    c.dl.cutout_max_frac     = 0.2
    c.dl.cutout_n_patches    = 2
    c.dl.depth_mask_prob     = 0.0
    c.tra.epoch_cooldown_secs   = 0 if on_linux else 18
    c.tra.val_cooldown_secs     = 0 if on_linux else 24
    c.tra.eval_cooldown_secs    = 0 if on_linux else 120
    c.tra.fig_chunk_cooldown_ms = 0 if on_linux else 120
    c.tra.dann_n_domains = 15
    return c


# archs4 baseline: sc15 curriculum supcon + attn_mil + entropy=0.03
_BASE4 = dict(
    init_weights=MAE_CKPT,
    dann=False,
    supcon=True, supcon_temp=0.07,
    supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5, supcon_curriculum_epochs=10,
    attn_mil=True,
    attn_entropy_weight=0.03,
    mean_teacher=False, test_consistency=False,
)


def _mk4(tid, tag, **overrides):
    """test with archs4 baseline (sc15 supcon curriculum + attn_mil + entropy=0.03)."""
    d = dict(_BASE4); d.update(overrides); d["tid"] = tid; d["tag"] = tag
    return d


TESTS = [
    # ==============================================================================
    # 1. TV REGULARIZER: spatial coherence -> letter readability
    # ==============================================================================
    # the key missing piece: metrics are decent but letters are unreadable because
    # adjacent tile predictions are spatially incoherent (tiles are classified independently).
    # TV loss on the per-voxel logit map penalizes high-frequency spatial variation
    # within the context window -> encourages letter-shape coherence.
    # no extra forward pass: reuses the existing last_voxel_map from forward_with_extras.
    #
    # tv01: lambda=0.2 (moderate -- suppress spatial noise but keep real boundaries)
    # tv02: lambda=0.5 (stronger -- force more coherence; risk smoothing real boundaries)
    _mk4("tv01", "tv_lam02", tv_lambda=0.2),
    _mk4("tv02", "tv_lam05", tv_lambda=0.5),

    # ==============================================================================
    # 2. DEPTH PROFILE SUPCON: contrastive on raw depth signatures
    # ==============================================================================
    # all experiments show depth >> spatial: surface_dist best learner, ds=2==ds=1.
    # conclusion: the ink signal is in HOW CT INTENSITY VARIES WITH DEPTH at ink sites,
    # not in spatial texture within a slice.
    # depth profile supcon: contrastive learning on raw mean depth profiles at the center
    # tile. uses a tiny DepthProfileHead (2-layer MLP, proj_dim=32) completely SEPARATE
    # from the spatial supcon head. pulls ink depth profiles together across scrolls.
    #
    # depsc01: lambda=0.1 (light touch; the depth profile is a weak feature)
    # depsc02: lambda=0.2 + combined with spatial supcon (both depth and spatial contrastive)
    _mk4("depsc01", "depth_supcon_lam01", depth_supcon=True, depth_supcon_lambda=0.1),
    _mk4("depsc02", "depth_supcon_lam02", depth_supcon=True, depth_supcon_lambda=0.2),

    # ==============================================================================
    # 3. LEARNED SURFACE ATTENTION WITH MORE EPOCHS
    # ==============================================================================
    # learn01 was the best archs2 performer at 15 epochs but clearly not converged.
    # the DepthSurfaceAttn module (~320 params) learns which depth slices are surface-
    # proximal and amplifies those features. running for 30 epochs gives it time.
    _mk4("learn01_30ep", "learned_surf_30ep", learned_surface=True, n_epochs=25, probe_int=25),

    # ==============================================================================
    # 4. LABEL NOISE TEST: single-scroll focused training
    # ==============================================================================
    # the labels are the best available (1.1µm model + hand cleaning) but are fundamentally
    # noisy due to resolution downsampling from 1.1µm to 9.4µm.
    # hypothesis: training on ALL 15 scrolls compounds label noise from cross-scroll
    # projection. if training on ONLY w044 (original fragment, most hand-cleaned) gives
    # BETTER validation on w044, then diversity is HURTING (noisy labels from other scrolls
    # contaminate the gradient).
    # if it performs WORSE, diversity IS needed despite noise.
    # this is a diagnostic test to understand whether to focus on label quality or diversity.
    _mk4("single_w044", "w044_only",
         # override scrolls: w044 only, with standard split
         scrolls=[ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055)]),
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
    "conv1_drop":            ("model", "conv1_drop"),
    "conv2_drop":            ("model", "conv2_drop"),
    "head_drop":             ("model", "head_drop"),
    # training
    "n_epochs":              ("tra", "n_epochs"),
    "eval_int":              ("tra", "eval_int"),
    "probe_int":             ("tra", "probe_int"),
    "l1":                    ("tra", "l1_lambda"),
    "weight_decay":          ("tra", "weight_decay"),
    "ranking_lambda":        ("tra", "ranking_lambda"),
    "label_smooth":          ("tra", "label_smooth"),
    "tv_lambda":             ("tra", "tv_lambda"),
    "depth_supcon":          ("tra", "depth_supcon"),
    "depth_supcon_lambda":   ("tra", "depth_supcon_lambda"),
    # augmentation magnitudes
    "cutout_prob":           ("dl", "cutout_prob"),
    "cutout_max_frac":       ("dl", "cutout_max_frac"),
    "cutout_n_patches":      ("dl", "cutout_n_patches"),
    "brightness_delta":      ("dl", "brightness_delta"),
    "contrast_delta":        ("dl", "contrast_delta"),
    "noise_prob":            ("dl", "noise_prob"),
    "noise_std_max":         ("dl", "noise_std_max"),
    # SupCon
    "supcon":                ("tra", "supcon"),
    "supcon_lambda":         ("tra", "supcon_lambda"),
    "supcon_temp":           ("tra", "supcon_temp"),
    "supcon_curriculum":     ("tra", "supcon_curriculum"),
    "supcon_lambda_start":   ("tra", "supcon_lambda_start"),
    "supcon_lambda_end":     ("tra", "supcon_lambda_end"),
    "supcon_curriculum_epochs": ("tra", "supcon_curriculum_epochs"),
    "supcon_proj_dim":       ("tra", "supcon_proj_dim"),
    "supcon_hidden_dim":     ("tra", "supcon_hidden_dim"),
    # mean teacher
    "mean_teacher":              ("tra", "mean_teacher"),
    "mean_teacher_alpha":        ("tra", "mean_teacher_alpha"),
    "mean_teacher_lambda":       ("tra", "mean_teacher_lambda"),
    "mean_teacher_ramp_epochs":  ("tra", "mean_teacher_ramp_epochs"),
    "verified_neg_lambda":       ("tra", "verified_neg_lambda"),
    # data
    "context_size":          ("data", "context_size"),
    "context_downsample":    ("data", "context_downsample"),
    "ring_label_source":     ("data", "ring_label_source"),
    # dataloader
    "batch_size":            ("dl", "batch_size"),
}


def build_config(t: dict) -> Config:
    tid = t["tid"]; tag = t["tag"]
    c = _base_config(f"cmp_archs4_{tid}_{tag}")

    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            try:
                setattr(getattr(c, sec), attr, t[k])
            except AttributeError:
                print(f"[WARNING] {tid}: {sec}.{attr} does not exist yet")

    # special: scrolls override (for single-scroll training test)
    if "scrolls" in t:
        c.data.scrolls = t["scrolls"]

    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[archs4] init_weights '{iw}' not found -- {tid} trains from scratch")

    c.dl.data_aug = any([c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
                         c.dl.brightness_prob, c.dl.contrast_prob,
                         c.dl.cutout_prob, c.dl.depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0
    os.makedirs("models/archs4", exist_ok=True)
    c.save_final = f"models/archs4/{tid}_{tag}_final.pth"
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[archs4] {c.exp_name}\n{'='*70}", flush=True)
    n_scrolls = len(c.data.scrolls)
    print(f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
          f"  attn_mil={c.model.attn_mil}  entropy={c.model.attn_entropy_weight}")
    print(f"  n_scrolls={n_scrolls}  n_epochs={c.tra.n_epochs}  wd={c.tra.weight_decay:.1e}")
    print(f"  supcon={c.tra.supcon} T={c.tra.supcon_temp}")
    print(f"  tv_lambda={c.tra.tv_lambda}  depth_supcon={c.tra.depth_supcon}"
          f"  depth_supcon_lam={c.tra.depth_supcon_lambda}")
    print(f"  surface_stem={c.model.surface_stem}  learned_surface={c.model.learned_surface}")
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
    ap = argparse.ArgumentParser(description="campaign_archs_4: spatial coherence + depth profile")
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
            print(f"[ABORT] --only id(s) {sorted(missing)} not found;"
                  f" valid: {[t['tid'] for t in TESTS]}")
            return
    elif args.from_id:
        ids = [t["tid"] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[archs4] {len(selected)} test(s) queued  (log -> {LOG_DIR})")

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

    print(f"\n{'='*70}\n[archs4] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
