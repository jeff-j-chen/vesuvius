"""campaign_archs_5.py -- follow-up from campaign_archs_4 results (2026-08-08).

KEY FINDINGS FROM ARCHS_4:
  - TV loss (both lambda values): zero effect. voxel logit map is 7x7 at ds=2 resolution --
    too coarse for TV to meaningfully enforce spatial coherence. dropped entirely.
  - Depth SupCon lambda=0.2 (depsc02): marginal but real improvement. depth profile carries
    some ink-discriminative signal. worth pushing higher and combining.
  - Learned surface (30 epochs): peaked at epoch 16, then overfit. sweet spot is ~16 epochs.
  - Single-scroll (single_w044): FAILED due to augmentation conflict.
    the augmentation magnitude was tuned for 15-scroll natural domain diversity.
    at 1 scroll, augmentation noise exceeds the training signal -> model can't learn.
    FIX: scale augmentation down proportionally to dataset size.

NEW HYPOTHESES:
  1. 1-scroll with SOFT augmentation: test if the signal emerges with halved aug strength.
  2. 2-scroll with clean labels (w044 + P500P2): modest diversity without noisy label transfer.
  3. Depth SupCon at lambda=0.3: 0.2 was marginal; push higher to test ceiling.
  4. Depth SupCon + Learned surface: best two depth signals combined.
  5. Learned surface at correct epoch count (16 not 30): avoid the overfit tail.

5 tests total.

  python campaign_archs_5.py --dry-run
  python campaign_archs_5.py --only w044_soft
  python campaign_archs_5.py
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, ScrollConfig

MAE_CKPT = "models/mae_twostage.pth"
LOG_DIR = "./runs_archs5"
N_EP = 15


def _base_config(exp_name: str) -> Config:
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
    # standard 15-scroll augmentation
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
    c.tra.dann_n_domains = 16  # updated to 16 (P500P2 added)
    return c


# baseline: sc15 curriculum supcon + attn_mil + entropy=0.03 (proven base)
_BASE5 = dict(
    init_weights=MAE_CKPT,
    dann=False,
    supcon=True, supcon_temp=0.07,
    supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5, supcon_curriculum_epochs=10,
    attn_mil=True,
    attn_entropy_weight=0.03,
    mean_teacher=False, test_consistency=False,
)


def _mk5(tid, tag, **overrides):
    d = dict(_BASE5); d.update(overrides); d["tid"] = tid; d["tag"] = tag
    return d


TESTS = [
    # BASELINE WITH 16 SCROLLS: compare directly to 15 scrolls
    _mk5("baseline", "baseline"),
    # ==============================================================================
    # 1. SINGLE-SCROLL WITH SOFT AUGMENTATION
    # ==============================================================================
    # single_w044 failed because 15-scroll augmentation magnitude exceeds the SNR of
    # 1 scroll's training signal. fix: scale aug proportionally to dataset size.
    # halved flip/rotation/noise/brightness/contrast probabilities; removed cutout entirely.
    # the model should be able to learn the ink signal with less noise disruption.
    # if this works: label diversity is less important than label quality at small scale.
    _mk5("w044_soft", "w044_softaug",
         scrolls=[ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055)],
         flip_prob=0.2, rotation_prob=0.2, noise_prob=0.1,
         brightness_prob=0.1, contrast_prob=0.1,
         cutout_prob=0.0),

    # ==============================================================================
    # 2. TWO CLEAN-LABEL SCROLLS
    # ==============================================================================
    # instead of 1 scroll (too little data) or 15 scrolls (too much label noise),
    # try the 2 scrolls with the best labels: w044 (original, user hand-cleaned)
    # and P500P2 (crystal-clear 2.215um inklabels).
    # this gives: scroll domain diversity WITHOUT cross-scroll label projection noise.
    # if this beats 15-scroll: noisy label transfer from other scrolls is net negative.
    _mk5("two_clean", "w044_p500p2",
         scrolls=[
             ScrollConfig(20260115000000, split_axis="y", train_split_frac=0.8055),  # w044
             ScrollConfig(20250628074500, split_axis="x", train_split_frac=0.75),    # P500P2
         ]),

    # ==============================================================================
    # 3. DEPTH SUPCON HIGHER LAMBDA
    # ==============================================================================
    # depsc02 (lambda=0.2) showed marginal but real improvement over baseline.
    # push to lambda=0.3 to check if the depth profile contrastive signal can be
    # amplified further without destabilizing the spatial supcon.
    _mk5("depsc_high", "depth_supcon_lam03",
         depth_supcon=True, depth_supcon_lambda=0.3),

    # ==============================================================================
    # 4. DEPTH SUPCON + LEARNED SURFACE (best two depth signals together)
    # ==============================================================================
    # depth supcon: contrastive on raw depth profiles (depsc02 winner)
    # learned surface: DepthSurfaceAttn amplifies surface-proximal features (learn01 winner)
    # hypothesis: the two signals are complementary -- depth supcon pulls ink depth profiles
    # together across scrolls; learned surface focuses the model on the surface-proximal slices
    # where ink actually is.
    _mk5("depsc_surf", "depth_supcon_learned_surf",
         depth_supcon=True, depth_supcon_lambda=0.2,
         learned_surface=True),

    # ==============================================================================
    # RADICAL IDEAS
    # ==============================================================================

    # 5. FOCAL LOSS gamma=1.5 (Lin et al. 2017, RetinaNet)
    # focal loss = (1 - p_t)^gamma * CE. exponentially down-weights EASY examples
    # (tiles the model is already very confident about) and concentrates gradient
    # on HARD examples (ambiguous boundary tiles near ink strokes).
    # our problem: the model is over-confident about clear papyrus tiles (easy negatives
    # constitute most of the gradient). focal loss suppresses these so the gradient is
    # dominated by the tiles that actually contain information: ring boundaries.
    # previously tested ONLY with old architectures (pre-v14, pre-supcon, pre-attn).
    # with the current baseline (supcon+attn_mil+entropy=0.03), the interaction may differ.
    # gamma=1.5: moderate (1.0=mild, 2.0=aggressive in original paper).
    _mk5("focal_loss", "focal_gamma15",
         loss_type="focal", focal_gamma=1.5,
         # gce_q doesn't apply when loss_type="focal"
         ),

    # 6. ASYMMETRIC LABEL SMOOTHING (physics-motivated)
    # standard label_smooth=0.1 smooths SYMMETRICALLY: 1->0.9, 0->0.1.
    # but label noise is ASYMMETRIC in this problem:
    #   - ink labels (1): projected from 1.1um to 9.4um, inherently noisy due to
    #     partial volume, misalignment, and uncertain boundary placement.
    #     many tiles labeled ink contain mostly papyrus at 9.4um.
    #   - papyrus labels (0): ring negatives are geometrically defined and conservative.
    #     a tile labeled non-ink almost certainly has no ink at 9.4um.
    # asymmetric smoothing: ink=0.25 (more uncertainty), papyrus=0.02 (very confident).
    # this reflects the actual label noise structure and was not previously testable
    # (label_smooth_pos / label_smooth_neg are new config params, just implemented).
    _mk5("asym_smooth", "asym_label_smooth",
         label_smooth_pos=0.25, label_smooth_neg=0.02),

    # 7. STOCHASTIC DEPTH MASKING (depth_mask_prob=0.3)
    # randomly zero out individual depth slices during training. forces the model to
    # recognize the ring pattern even with incomplete depth coverage.
    # physics motivation: the ring appears at DIFFERENT absolute depths per tile due to
    # papyrus undulation. if the model can't see all depths, it must learn depth-invariant
    # ink features rather than memorizing 'ink appears at absolute depth 11'.
    # analogous to SpecAugment (time/frequency masking in speech). risk: could destroy
    # the ring signal if too aggressive. 0.3 means each slice has a 30% chance of zeroing.
    _mk5("depth_drop", "depth_slice_masking",
         depth_mask_prob=0.3),

    # 8. GCE q=0.9 (Zhang & Sabuncu, NeurIPS 2018) -- NEVER RUN at q=0.9 before
    # current baseline uses q=0.7. q=0.9 is near-MAE: treats mislabeled tiles as outliers
    # rather than hard examples to overfit. the gradient of GCE saturates when the model
    # is confidently wrong, preventing mislabeled tiles from dominating the update.
    # this is fundamentally different from focal loss (which focuses on HARD examples);
    # GCE-q focuses on CONSISTENT examples and ignores the noisy/inconsistent ones.
    _mk5("gce_noise", "gce_q09",
         gce_q=0.9),

    # 9. FIVE OVERLAPPING DEPTH WINDOWS
    # the 3-window model has SEAMS at absolute depths 12 and 20:
    #   window 0: abs 4-12   window 1: abs 12-20   window 2: abs 20-28
    # if ink peaks at depth 12 or 20, it sits at the EDGE of two windows.
    # stage2 can partially compensate via its 3x3x3 convs across windows, but the signal
    # is split: window 0 sees slice 7 (depth 12) as its LAST slice; window 1 sees it as
    # its FIRST slice. the windows may not integrate the signal across that boundary.
    # fix: add 2 intermediate windows centered on the seams:
    #   window 0:   abs  4-12   (unchanged)
    #   window 0.5: abs  8-16   NEW -- depth 12 is in the MIDDLE of this window
    #   window 1:   abs 12-20   (unchanged)
    #   window 1.5: abs 16-24   NEW -- depth 20 is in the MIDDLE of this window
    #   window 2:   abs 20-28   (unchanged)
    # stage1 backbone is TIED across all 5 windows (no additional parameters).
    # stage2 grows slightly: Conv3d(5,32) instead of Conv3d(3,32) = +576 params.
    # MAE warm-start still loads stage1 cleanly; stage2 is randomly initialized anyway.
    _mk5("five_win", "5_depth_windows",
         n_depth_windows=5),
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
    # tta consistency
    "tta_consistency":       ("tra", "tta_consistency"),
    "tta_consistency_lambda":("tra", "tta_consistency_lambda"),
    # noise-robust loss
    "gce_q":                 ("tra", "gce_q"),
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
    # mean teacher
    "mean_teacher":              ("tra", "mean_teacher"),
    "verified_neg_lambda":       ("tra", "verified_neg_lambda"),
    # data
    "context_size":          ("data", "context_size"),
    "context_downsample":    ("data", "context_downsample"),
    "ring_label_source":     ("data", "ring_label_source"),
    "batch_size":            ("dl", "batch_size"),
}


def build_config(t: dict) -> Config:
    tid = t["tid"]; tag = t["tag"]
    c = _base_config(f"cmp_archs5_{tid}_{tag}")
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
        print(f"[archs5] init_weights '{iw}' not found -- {tid} trains from scratch")
    c.dl.data_aug = any([c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
                         c.dl.brightness_prob, c.dl.contrast_prob,
                         c.dl.cutout_prob, c.dl.depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0
    os.makedirs("models/archs5", exist_ok=True)
    c.save_final = f"models/archs5/{tid}_{tag}_final.pth"
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[archs5] {c.exp_name}\n{'='*70}", flush=True)
    n_scrolls = len(c.data.scrolls)
    print(f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
          f"  entropy={c.model.attn_entropy_weight}")
    print(f"  n_scrolls={n_scrolls}  n_epochs={c.tra.n_epochs}")
    print(f"  depth_supcon={c.tra.depth_supcon}  depth_supcon_lam={c.tra.depth_supcon_lambda}"
          f"  learned_surface={c.model.learned_surface}")
    print(f"  gce_q={c.tra.gce_q}  tta_cons={c.tra.tta_consistency}  depth_mask={c.dl.depth_mask_prob}")
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
    ap = argparse.ArgumentParser(description="campaign_archs_5: aug scaling + depth signals")
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

    print(f"[archs5] {len(selected)} test(s) queued  (log -> {LOG_DIR})")

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

    print(f"\n{'='*70}\n[archs5] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
