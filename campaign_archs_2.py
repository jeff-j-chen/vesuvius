"""campaign_archs_2.py -- second-generation architecture sweep.

Based on analysis of campaign_archs results:
- SupCon T=0.07 is the clear winner (consistent gains, stable)
- DANN+SupCon combo (dann_sc1) was the top scorer (PR_AUC=0.6224)
- Attention-MIL works but is sparse (needs coverage regularization)
- MeanTeacher verified-neg works; test-consistency failed (domain shift)

This campaign focuses on:
1. SupCon optimization (lambda tuning, curriculum, embedding dimension)
2. DANN+SupCon stability and tuning (since it's the top scorer)
3. Attention-MIL coverage fixes
4. MeanTeacher on same-scroll unlabeled (not cross-scroll)

Key differences from campaign_archs:
- Logs to runs_archs2 (separate from original)
- Only renders 1 scroll during eval (eval_int_scrolls=1) for speed
- Longer runs (20 epochs) for top candidates
- More focused on winners from round 1
- ALL missing features now implemented (SupCon curriculum, attention entropy, etc.)
- Expanded MeanTeacher tests (11 variations vs 3 in original)

Total: 34 tests (vs 23 in draft, 15 in campaign_archs)

  python campaign_archs_2.py --dry-run
  python campaign_archs_2.py --only sc10
  python campaign_archs_2.py --only sc10,sc11,sc12
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config
from utils.platform import get_zarr_dir

MAE_CKPT = "models/mae_twostage.pth"
LOG_DIR = "./runs_archs2"


def _base_config(exp_name: str) -> Config:
    """base config for archs2 -- same as campaign_archs but logs to runs_archs2 and only renders 1 scroll."""
    c = Config()
    on_linux = (os.name == "posix")
    c.exp_name = exp_name
    c.model.arch = "v16_arch_ctx"
    
    # Set platform-aware zarr path
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
    c.tra.n_epochs     = 15
    c.tra.eval_int     = 999
    c.tra.test_int     = 999
    c.tra.probe_int    = 15
    c.tra.save_int     = 5
    c.tra.log_dir      = LOG_DIR
    c.tra.deterministic = False
    c.tra.lr = get_default_lr()
    c.data.eval_infer_bs = get_default_eval_bs()
    
    # OPTIMIZATION: only render 1 scroll during eval (much faster)
    c.tra.eval_int_scrolls = 1
    
    # winners from campaign_archs
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
    c.dl.flip_prob           = 0.6
    c.dl.rotation_prob       = 0.6
    c.dl.noise_prob          = 0.3
    c.dl.brightness_prob     = 0.6
    c.dl.contrast_prob       = 0.6
    c.dl.cutout_prob         = 0.4
    c.dl.cutout_max_frac     = 0.2
    c.dl.cutout_n_patches    = 2
    c.dl.depth_mask_prob     = 0.0
    c.tra.epoch_cooldown_secs   = 0 if on_linux else 9 * 2
    c.tra.val_cooldown_secs     = 0 if on_linux else 12 * 2
    c.tra.eval_cooldown_secs    = 0 if on_linux else 60 * 2
    c.tra.fig_chunk_cooldown_ms = 0 if on_linux else 60 * 2
    c.tra.dann_n_domains = 15
    return c


_BASE = dict(
    init_weights=MAE_CKPT,
    dann=False, supcon=False, attn_mil=False,
    mean_teacher=False, test_consistency=False,
)

# base for all tests from attn13 onward: sc15 curriculum SupCon + basic attention-MIL
# DANN is dropped (ds13-ds15 debunked it). entropy_weight not yet proven (attn10-12 pending).
_BASE2 = dict(
    init_weights=MAE_CKPT,
    dann=False,
    supcon=True, supcon_temp=0.07,
    supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5, supcon_curriculum_epochs=10,
    attn_mil=True, attn_entropy_weight=0.03,
    mean_teacher=False, test_consistency=False,
)


def _mk(tid, tag, **overrides):
    d = dict(_BASE); d.update(overrides); d["tid"] = tid; d["tag"] = tag
    return d


def _mk2(tid, tag, **overrides):
    """like _mk but with sc15 curriculum SupCon + attn_mil as assumed base."""
    d = dict(_BASE2); d.update(overrides); d["tid"] = tid; d["tag"] = tag
    return d


TESTS = [
    # ==============================================================================
    # SUPCON OPTIMIZATION (Priority 1)
    # ==============================================================================
    # sc10-12: Lambda interpolation around the winner (0.3 was best, test 0.25, 0.35, 0.4)
    _mk("sc10", "supcon_t007_lam025", supcon=True, supcon_temp=0.07, supcon_lambda=0.25),
    _mk("sc12", "supcon_t007_lam040", supcon=True, supcon_temp=0.07, supcon_lambda=0.40),
    
    # sc14-15: Lambda curriculum (progressive transfer learning)
    # Start low (focus on ink detection), gradually increase (focus on cross-scroll transfer)
    _mk("sc14", "supcon_t007_curriculum_slow", 
        supcon=True, supcon_temp=0.07, supcon_lambda=0.3,
        supcon_curriculum=True, supcon_lambda_start=0.1, supcon_lambda_end=0.4, supcon_curriculum_epochs=15),
    _mk("sc15", "supcon_t007_curriculum_fast",
        supcon=True, supcon_temp=0.07, supcon_lambda=0.3,
        supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5, supcon_curriculum_epochs=10),
    
    # ==============================================================================
    # DANN+SUPCON COMBOS (Top scorer from round 1, optimize it)
    # ==============================================================================
    
    # ds13-14: Vary SupCon lambda (0.2, 0.3) while keeping DANN fixed at sweet spot
    _mk("ds13", "dann03_sc_t007_lam02",
        dann=True, dann_lambda=0.3, dann_ramp_epochs=8,
        supcon=True, supcon_temp=0.07, supcon_lambda=0.2),
    _mk("ds14", "dann03_sc_t007_lam03",
        dann=True, dann_lambda=0.3, dann_ramp_epochs=8,
        supcon=True, supcon_temp=0.07, supcon_lambda=0.3),
    
    # ds15: Progressive DANN lambda (start gentle, ramp up slowly)
    _mk("ds15", "dann_progressive_sc",
        dann=True, dann_lambda=0.4, dann_ramp_epochs=15,  # longer ramp
        supcon=True, supcon_temp=0.07, supcon_lambda=0.2),
    
    # ==============================================================================
    # ATTENTION-MIL WITH COVERAGE FIXES
    # ==============================================================================
    # attn10: Pure attention (rerun baseline for 20 epochs)
    _mk("attn10", "attentionmil_e20", attn_mil=True),
    
    # attn11-12: Attention + entropy regularizer (force coverage spread)
    # NOTE: These require model.py changes to add entropy_weight parameter
    _mk("attn12", "attn_entropy03", attn_mil=True, attn_entropy_weight=0.03),
    
    # ==============================================================================
    # FROM HERE: all tests use _mk2 base = sc15 curriculum SupCon + attn_mil_e20
    # DANN is gone (debunked). entropy_weight stays off until attn11-12 results confirm it.
    # Each test varies EXACTLY ONE additional feature on top of the validated base.
    # ==============================================================================

    # attn13: the baseline with new base (sc15 supcon curriculum + attn_mil)
    # question: is the combination better than sc15 alone or attn10 alone?
    # attn14: test if attention entropy helps now that supcon curriculum is the baseline
    # (attn11/12 tested entropy without supcon; this tests it WITH the full base)

    # ==============================================================================
    # PHYSICS STEM: DoG ring detection + gradient magnitude sharpness
    # All use sc15 supcon curriculum + attn_mil base.
    # ==============================================================================
    # phy01: per-slice DoG (ring detector: bright annulus at ink-papyrus boundary)
    # vs attn13: does physics ring channel improve over the validated base?
    _mk2("phy01", "physics_dog", physics_stem=True),

    # phy02: depth-max DoG (wavy-papyrus fix: max ring response over 8-slice window)
    # vs phy01: does broadcasting the peak ring response help?
    _mk2("phy02", "physics_dmax", physics_stem_depthmax=True),

    # ==============================================================================
    # SURFACE DETECTION: per-(y,x) depth alignment for wavy papyrus
    # surface_dist: signed distance to papyrus surface in [-1,+1]
    # surface_attn: softmax(|dI/dz|) peaked at the papyrus-air boundary
    # ==============================================================================
    # surf01: surface channels only (5-ch stem)
    # vs attn13: does knowing 'where is the surface in this tile?' help?
    _mk2("surf01", "surface_dist", surface_stem=True),

    # surf02: surface + DoG ring detection (6-ch stem: all fixed physics)
    # vs phy01 and surf01: do ring + surface signals complement each other?
    _mk2("surf02", "surface_dog", surface_stem_withdog=True),

    # ==============================================================================
    # LEARNED SURFACE ATTENTION: tiny 1D-depth conv (~320 params) that learns
    # which depth slices are surface-proximal from the training signal itself.
    # advantage over fixed physics: can learn which SIDE has ink, handle flaking.
    # ==============================================================================
    # learn01: learned surface attention, standard 3-ch stem
    # vs surf01: can the model LEARN a better surface detector than the fixed |dz| peak?
    _mk2("learn01", "learned_surf", learned_surface=True),

    # learn02: learned surface attention + physics DoG channels (best of learned + fixed)
    # vs learn01 and phy01: do physics channels help the learned surface finder?
    _mk2("learn02", "learned_surf_dog", learned_surface=True, physics_stem=True),

    # ==============================================================================
    # MEANTEACHER: with sc15 supcon curriculum + attn_mil as base
    # mt01-03 are the best representatives from the mt10-mt20 design,
    # now with a stronger base. each tests ONE additional signal type.
    # ==============================================================================
    # mt01: verified-neg (2.4um hard negatives reinforce papyrus boundary)
    # best single MeanTeacher signal from round 1 (mt_vn1 was 4th place overall)
    _mk2("mt01", "mt_vneg",
        mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.2,
        mean_teacher_ramp_epochs=3, verified_neg_lambda=0.3),

    # mt02: same-scroll pseudo-labeling (high-confidence teacher on validation split)
    # the domain-shift fix: avoids cross-scroll pseudo-labels that caused collapse
    _mk2("mt02", "mt_pseudo",
        mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.2,
        mean_teacher_ramp_epochs=3, verified_neg_lambda=0.3,
        pseudo_label_same_scroll=True, pseudo_label_threshold=0.95),

    # mt03: consistency on labeled tiles (original MeanTeacher, Tarvainen 2017)
    # student/teacher see different augmentations of the same tile: rotation-invariance
    _mk2("mt03", "mt_consistency",
        mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.3,
        mean_teacher_ramp_epochs=3, consistency_on_labeled=True),

    # ==============================================================================
    # GRAND COMBO: the richest configuration of all validated components
    # ==============================================================================
    # grand1: sc15 supcon curriculum + attn_mil + learned surface + physics DoG + verified-neg MT
    # if this works: we have a strong multi-signal model for the final campaign
    _mk2("grand1", "grand_surf_dog_mt",
        physics_stem=True, learned_surface=True,
        mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.2,
        mean_teacher_ramp_epochs=3, verified_neg_lambda=0.3),
]
# dict-key -> (config-section, attribute)
_OVERRIDES = {
    "arch":                ("model", "arch"),
    "attn_mil":            ("model", "attn_mil"),
    "attn_entropy_weight": ("model", "attn_entropy_weight"),
    "physics_stem":        ("model", "physics_stem"),
    "physics_stem_depthmax": ("model", "physics_stem_depthmax"),
    "surface_stem":          ("model", "surface_stem"),
    "surface_stem_withdog":  ("model", "surface_stem_withdog"),
    "learned_surface":       ("model", "learned_surface"),
    "n_epochs":            ("tra", "n_epochs"),
    "eval_int":            ("tra", "eval_int"),
    "probe_int":           ("tra", "probe_int"),
    "l1":                  ("tra", "l1_lambda"),
    "weight_decay":        ("tra", "weight_decay"),
    "ranking_lambda":      ("tra", "ranking_lambda"),
    "label_smooth":        ("tra", "label_smooth"),
    # DANN
    "dann":                ("tra", "dann"),
    "dann_lambda":         ("tra", "dann_lambda"),
    "dann_ramp_epochs":    ("tra", "dann_ramp_epochs"),
    "dann_n_domains":      ("tra", "dann_n_domains"),
    # SupCon
    "supcon":              ("tra", "supcon"),
    "supcon_lambda":       ("tra", "supcon_lambda"),
    "supcon_temp":         ("tra", "supcon_temp"),
    "supcon_curriculum":   ("tra", "supcon_curriculum"),
    "supcon_lambda_start": ("tra", "supcon_lambda_start"),
    "supcon_lambda_end":   ("tra", "supcon_lambda_end"),
    "supcon_curriculum_epochs": ("tra", "supcon_curriculum_epochs"),
    # mean teacher
    "mean_teacher":           ("tra", "mean_teacher"),
    "mean_teacher_alpha":     ("tra", "mean_teacher_alpha"),
    "mean_teacher_lambda":    ("tra", "mean_teacher_lambda"),
    "mean_teacher_ramp_epochs": ("tra", "mean_teacher_ramp_epochs"),
    "verified_neg_lambda":    ("tra", "verified_neg_lambda"),
    "test_consistency":       ("tra", "test_consistency"),
    "test_consistency_lambda":("tra", "test_consistency_lambda"),
    "pseudo_label_same_scroll": ("tra", "pseudo_label_same_scroll"),
    "pseudo_label_threshold":   ("tra", "pseudo_label_threshold"),
    "consistency_on_labeled":   ("tra", "consistency_on_labeled"),
    # data
    "context_size":        ("data", "context_size"),
    "context_downsample":  ("data", "context_downsample"),
    "ring_label_source":   ("data", "ring_label_source"),
    # dataloader
    "batch_size":          ("dl", "batch_size"),
}


def build_config(t: dict) -> Config:
    tid = t["tid"]; tag = t["tag"]
    c = _base_config(f"cmp_archs2_{tid}_{tag}")

    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            # Some attributes might not exist yet (new features) - skip with warning
            try:
                setattr(getattr(c, sec), attr, t[k])
            except AttributeError:
                print(f"[WARNING] {tid}: attribute {sec}.{attr} does not exist yet (feature not implemented)")

    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[archs2] init_weights '{iw}' not found -- {tid} trains from scratch")

    c.dl.data_aug = any([c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
                         c.dl.brightness_prob, c.dl.contrast_prob,
                         c.dl.cutout_prob, c.dl.depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0
    os.makedirs("models/archs2", exist_ok=True)
    c.save_final = f"models/archs2/{tid}_{tag}_final.pth"
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[archs2] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
          f"  attn_mil={c.model.attn_mil}")
    print(f"  ring={c.data.ring_label_source}  n_epochs={c.tra.n_epochs}"
          f"  wd={c.tra.weight_decay:.1e}  l1={c.tra.l1_lambda:.1e}")
    print(f"  dann={c.tra.dann} lam={c.tra.dann_lambda}"
          f"  supcon={c.tra.supcon} lam={c.tra.supcon_lambda} T={c.tra.supcon_temp}")
    print(f"  mt={c.tra.mean_teacher} lam={c.tra.mean_teacher_lambda} alpha={c.tra.mean_teacher_alpha}")
    print(f"  physics_stem={c.model.physics_stem}  physics_stem_depthmax={c.model.physics_stem_depthmax}")
    print(f"  scrolls={len(c.data.scrolls)}  eval_int_scrolls={c.tra.eval_int_scrolls}")
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
    ap = argparse.ArgumentParser(description="campaign_archs_2: second-gen architecture sweep")
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

    print(f"[archs2] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print(f"[archs2] NOTE: Some tests use features not yet implemented (will skip with warnings)")

    results = {}
    for i, t in enumerate(selected):
        tid = t["tid"]
        c = build_config(t)
        ok = run_test(c, args.dry_run)
        results[tid] = "OK" if ok else "FAIL"
        if not args.dry_run:
            del c; gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\n{'='*70}\n[archs2] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
