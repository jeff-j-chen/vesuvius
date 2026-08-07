"""campaign_archs_3.py -- follow-up from campaign_archs_2 results (2026-08-07).

Key findings from archs2:
  - attn12 (entropy=0.03) is the clear winner across all runs; more entropy = better
  - surf01 (surface_dist) shows by far the fastest and strongest training convergence
    but overfits severely: the surface channels provide real signal but need more reg.
  - surf02 (surface_dog) solid performance, worth re-testing on updated baseline
  - phy01/phy02 (DoG only) both below average; deprioritized
  - DANN: already debunked in archs1/archs2

New baseline vs archs2:
  + attn_entropy_weight=0.03 (winner of attn12, now baked into _BASE3)
  everything else identical to archs2 base (sc15 supcon curriculum + attn_mil)

6 tests = 4 required + 2 tuning variants:
  ent01:      entropy=0.06         (explore: more entropy -> better?)
  surf_reg01: surface + wd=0.5     (mild extra reg to address surf01 overfit)
  surf_reg02: surface + wd=0.8, higher dropout (aggressive reg)
  surf_dog01: surface+dog          (re-test on updated baseline with entropy=0.03)
  emb01:      proj_dim=256         (2x SupCon embedding dimension)
  emb02:      proj_dim=512         (4x SupCon embedding dimension)

  python campaign_archs_3.py --dry-run
  python campaign_archs_3.py --only ent01
  python campaign_archs_3.py
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config

MAE_CKPT = "models/mae_twostage.pth"
LOG_DIR = "./runs_archs3"
N_EP = 15


def _base_config(exp_name: str) -> Config:
    """base config for archs3 -- same hardware/data setup as archs2, logs to runs_archs3."""
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
    # default dropout (may be overridden per-test for surf_reg variants)
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


# new baseline: sc15 curriculum supcon + attn_mil + attn_entropy_weight=0.03 (archs2 winner)
_BASE3 = dict(
    init_weights=MAE_CKPT,
    dann=False,
    supcon=True, supcon_temp=0.07,
    supcon_curriculum=True, supcon_lambda_start=0.05, supcon_lambda_end=0.5, supcon_curriculum_epochs=10,
    attn_mil=True,
    attn_entropy_weight=0.03,   # attn12 winner, now baked in as default
    mean_teacher=False, test_consistency=False,
)


def _mk3(tid, tag, **overrides):
    """test with archs3 baseline (sc15 supcon curriculum + attn_mil + entropy=0.03)."""
    d = dict(_BASE3); d.update(overrides); d["tid"] = tid; d["tag"] = tag
    return d


TESTS = [
    # ==============================================================================
    # 1. ENTROPY TUNING
    # ==============================================================================
    # attn12 (entropy=0.03) won; baseline now baked in. explore if 0.06 is better.
    # prior data: attn10=0.0, attn11=0.01, attn12=0.03 (winner). trend says more = better.
    # if 0.06 wins, we'd then try 0.1 in archs4.
    _mk3("ent01", "attn_entropy06", attn_entropy_weight=0.06),

    # ==============================================================================
    # 2. SURFACE DIST WITH STRONGER REGULARIZATION
    # ==============================================================================
    # surf01 (surface_dist) had by far the fastest/strongest training -- but overfit.
    # the surface channels provide real signal. the model is learning TOO fast from them.
    # approach: keep the surface channels, increase regularization pressure.
    #
    # surf_reg01: mild increase -- stronger dropout only
    # weight_decay and l1 are useless for this architecture (proven). levers that work:
    # dropout (conv channels, head) and augmentation magnitude/probability.
    _mk3("surf_reg01", "surface_dist_highdrop",
         surface_stem=True,
         conv1_drop=0.25, conv2_drop=0.25, head_drop=0.55),

    # surf_reg02: aggressive -- higher dropout + stronger augmentation
    # cutout patches destroy spatial structure, forcing the model to not rely on
    # any single region. higher brightness/contrast variation forces contrast-invariance.
    _mk3("surf_reg02", "surface_dist_drop_aug",
         surface_stem=True,
         conv1_drop=0.35, conv2_drop=0.35, head_drop=0.6,
         cutout_prob=0.65, cutout_max_frac=0.3, cutout_n_patches=3,
         brightness_delta=0.25, contrast_delta=0.25,
         noise_prob=0.5, noise_std_max=0.008),

    # ==============================================================================
    # 3. SURFACE + DOG ON UPDATED BASELINE
    # ==============================================================================
    # surf02 (surface_dog) in archs2 performed solidly (started slower, ended high).
    # archs2 ran it WITHOUT entropy=0.03 in the base (base was entropy=0.0 back then).
    # re-run with updated baseline including entropy=0.03.
    # note: DoG sigma is now corrected to (8,20) -- ring-boundary scale, not fiber scale.
    _mk3("surf_dog01", "surface_dog_e03_base",
         surface_stem_withdog=True),

    # ==============================================================================
    # 4. HIGHER SUPCON EMBEDDING DIMENSION
    # ==============================================================================
    # The SupCon projection head was implemented with configurable dims (proj_dim, hidden_dim)
    # but has never been tested beyond the default (proj_dim=128, hidden_dim=256).
    # hypothesis: a higher-capacity projection head gives the model more room to separate
    # ink/papyrus/scroll-texture into orthogonal subspaces in embedding space.
    # using the full new baseline: sc15 supcon + attn_mil + entropy=0.03.
    #
    # emb01: proj_dim=256, hidden=512 (2x baseline)
    # emb02: proj_dim=512, hidden=1024 (4x baseline)
    _mk3("emb01", "supcon_proj256",
         supcon_proj_dim=256, supcon_hidden_dim=512),

    _mk3("emb02", "supcon_proj512",
         supcon_proj_dim=512, supcon_hidden_dim=1024),
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
    # dropout (for surface regularization tests)
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
    # augmentation magnitudes (proven levers for regularization)
    "cutout_prob":           ("dl", "cutout_prob"),
    "cutout_max_frac":       ("dl", "cutout_max_frac"),
    "cutout_n_patches":      ("dl", "cutout_n_patches"),
    "brightness_delta":      ("dl", "brightness_delta"),
    "contrast_delta":        ("dl", "contrast_delta"),
    "noise_prob":            ("dl", "noise_prob"),
    "noise_std_max":         ("dl", "noise_std_max"),
    "flip_prob":             ("dl", "flip_prob"),
    "rotation_prob":         ("dl", "rotation_prob"),
    # SupCon (including embedding dimension -- new in archs3)
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
    "test_consistency":          ("tra", "test_consistency"),
    "pseudo_label_same_scroll":  ("tra", "pseudo_label_same_scroll"),
    "pseudo_label_threshold":    ("tra", "pseudo_label_threshold"),
    "consistency_on_labeled":    ("tra", "consistency_on_labeled"),
    # data
    "context_size":          ("data", "context_size"),
    "context_downsample":    ("data", "context_downsample"),
    "ring_label_source":     ("data", "ring_label_source"),
    # dataloader
    "batch_size":            ("dl", "batch_size"),
}


def build_config(t: dict) -> Config:
    tid = t["tid"]; tag = t["tag"]
    c = _base_config(f"cmp_archs3_{tid}_{tag}")

    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            try:
                setattr(getattr(c, sec), attr, t[k])
            except AttributeError:
                print(f"[WARNING] {tid}: {sec}.{attr} does not exist yet")

    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[archs3] init_weights '{iw}' not found -- {tid} trains from scratch")

    c.dl.data_aug = any([c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
                         c.dl.brightness_prob, c.dl.contrast_prob,
                         c.dl.cutout_prob, c.dl.depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0
    os.makedirs("models/archs3", exist_ok=True)
    c.save_final = f"models/archs3/{tid}_{tag}_final.pth"
    return c


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[archs3] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
          f"  attn_mil={c.model.attn_mil}  entropy_weight={c.model.attn_entropy_weight}")
    print(f"  ring={c.data.ring_label_source}  n_epochs={c.tra.n_epochs}"
          f"  wd={c.tra.weight_decay:.1e}")
    print(f"  supcon={c.tra.supcon} T={c.tra.supcon_temp}"
          f"  proj_dim={c.tra.supcon_proj_dim} hidden={c.tra.supcon_hidden_dim}")
    print(f"  surface_stem={c.model.surface_stem}  surface_stem_withdog={c.model.surface_stem_withdog}")
    print(f"  conv_drop=({c.model.conv1_drop},{c.model.conv2_drop})  head_drop={c.model.head_drop}")
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
    ap = argparse.ArgumentParser(description="campaign_archs_3: follow-up targeted sweep")
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

    print(f"[archs3] {len(selected)} test(s) queued  (log -> {LOG_DIR})")

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

    print(f"\n{'='*70}\n[archs3] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
