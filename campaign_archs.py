"""campaign_archs.py -- architecture research sweep for the two-stage MIL detector.

tests DANN (domain-adversarial), SupCon (supervised contrastive), Attention-MIL, mean-teacher
verified-negative supervision from 2.4um inklabels, and test-scroll EMA consistency, plus
combinations and ablations. run over ~4 days.

baseline = closed ctx48/ds2, 15 epochs, arch=v16_arch_ctx.

IMPORTANT: the baseline dict uses SENTINEL values for regularization parameters that MUST be
overridden before running. they are set to 999/"BROKEN" so the campaign ABORTS loudly if you
forget to substitute the winners from the current wd/TTA campaigns.

  python campaign_archs.py --dry-run          # verify all configs before launching
  python campaign_archs.py --only c0base      # single test
  python campaign_archs.py --only dann1,sc2   # comma-separated, runs in TESTS order

expected batch sequences:
  batch A (baseline + DANN sweep):         c0base,dann1,dann2,dann3
  batch B (SupCon sweep):                  sc1,sc2,sc3,sc4
  batch C (combos + attention-MIL):        dann_sc1,dann_sc2,attn1,dann_attn,sc_attn,dann_sc_attn
  batch D (mean-teacher verified-negs):    mt_vn1,mt_vn2,mt_vn3
  batch E (test-scroll consistency):       mt_tc1,mt_full
  batch F (attention-MIL combos + final):  attn_dann,attn_sc,attn_mt,attn_all
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config
from utils.platform import get_zarr_dir, get_default_batch_size, get_default_eval_bs, get_default_workers, get_default_lr

INTER_RUN_COOLDOWN_SECS = 120
MAE_CKPT = "models/mae_twostage.pth"
LOG_DIR = "./runs_archs"
N_EP = 15
EVAL_INT = 15
PROBE_INT = 5


def _base_config(exp_name: str) -> Config:
    """fresh config for the architecture sweep — same data/aug stack as campaign_ts_ctx
    ctx48/ds2 baseline, but logging to ./runs_archs and using arch=v16_arch_ctx."""
    c = Config()
    on_linux = (os.name == "posix")
    c.exp_name = exp_name
    c.model.arch = "v16_arch_ctx"     # InkDetectorArch: ctx48/ds2 + optional DANN/SupCon/AttnMIL
    
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
    c.tra.n_epochs     = N_EP
    c.tra.eval_int     = EVAL_INT
    c.tra.test_int     = 999
    c.tra.probe_int    = PROBE_INT
    c.tra.save_int     = 2
    c.tra.log_dir      = LOG_DIR
    c.tra.deterministic = False  # disabled to reduce memory overhead (TPM errors)
    c.tra.lr = get_default_lr()
    c.data.eval_infer_bs = get_default_eval_bs()

    # ---- SENTINEL VALUES: MUST be replaced with winners before running ----
    # set weight_decay to the best-performing value from the wd sweep.
    c.tra.weight_decay = 3e-1
    # set ring_label_source to "eroded" or "closed" based on c8/c9 comparison.
    c.data.ring_label_source = "closed"
    # set tta_consistency_lambda and mode based on cons sweep results.
    # conclusion: tta consistency adds significant time for little benefit
    c.tra.tta_consistency = False 
    c.tra.tta_consistency_lambda = 0.5
    c.tra.tta_consistency_mode   = "dihedral"
    # ---- END SENTINELS ----

    c.tra.l1_lambda    = 0.0           # proven inert in Adam -- keep off
    c.dl.batch_size    = get_default_batch_size()
    c.dl.num_workers   = get_default_workers()
    c.dl.data_aug      = True          # set by build_config below based on aug probs
    c.data.mask_memmap       = True
    c.data.mask_bitpack      = True    # bit-packing: 1 bit/pixel (8x smaller, saves 6GB)
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
    # DANN: 15 scrolls in the training set
    c.tra.dann_n_domains = 15
    # seg46527 (20260226000000) intentionally kept in for this whole campaign
    return c


# shared baseline knobs for the arch sweep (ctx48/ds2, no new arch features enabled)
_BASE = dict(
    init_weights=MAE_CKPT,
    n_epochs=15,
    eval_int=15,
    dann=False,    supcon=False,    attn_mil=False,
    mean_teacher=False, test_consistency=False,
)


def _mk(tid, tag, **overrides):
    """build a test entry. always starts from _BASE so missing fields stay False/default."""
    d = dict(_BASE); d.update(overrides); d["tid"] = tid; d["tag"] = tag
    return d


TESTS = [
    # ==============================================================================
    # BATCH A: baseline + DANN sweep (domain-adversarial, 3 lambda values)
    # ==============================================================================
    # c0: BASELINE — all new features OFF. establishes the v16_arch_ctx reference curve
    # with the sentinel wd/ring/tta filled in. every other test is a delta off this.
    # _mk("c0base",  "ctx48_baseline_arch_closed", n_epochs=5),

    # dann1-3: domain-adversarial network. backbone is penalized for predicting which scroll
    # a tile comes from -> forced to learn scroll-invariant (transferable) ink features.
    # lambda ramps 0->dann_lambda over dann_ramp_epochs=5. three sane values tested.
    # _mk("dann1",   "ctx48_dann_lam01",  dann=True, dann_lambda=0.1,  dann_ramp_epochs=5),
    # _mk("dann2",   "ctx48_dann_lam04",  dann=True, dann_lambda=0.4,  dann_ramp_epochs=5),
    # _mk("dann3",   "ctx48_dann_lam05",  dann=True, dann_lambda=0.5,  dann_ramp_epochs=5),

    # ==============================================================================
    # BATCH B: SupCon sweep (supervised contrastive, temp and lambda)
    # ==============================================================================
    # sc1-4: projection head + cross-scroll supervised contrastive loss pulls ink embeddings
    # together and pushes papyrus apart -> transferable boundary geometry. tested over
    # (temperature, lambda) combinations. temp=0.07 is the standard (Khosla 2020); higher
    # temp softens the distribution. lambda sets the tradeoff vs the primary BCE/ranking.
    # _mk("sc1",    "ctx48_supcon_t007_lam01",  supcon=True, supcon_temp=0.07,  supcon_lambda=0.1),
    _mk("sc2",    "ctx48_supcon_t007_lam03",  supcon=True, supcon_temp=0.07,  supcon_lambda=0.3),
    _mk("sc3",    "ctx48_supcon_t02_lam01",   supcon=True, supcon_temp=0.2,   supcon_lambda=0.1),
    _mk("sc4",    "ctx48_supcon_t02_lam03",   supcon=True, supcon_temp=0.2,   supcon_lambda=0.3),

    # ==============================================================================
    # BATCH C: DANN + SupCon combined; then Attention-MIL alone
    # ==============================================================================
    # dann_sc1/2: do DANN and SupCon reinforce each other? DANN removes scroll cues,
    # SupCon builds the shared ink cluster -> should be additive in principle.
    # use the best values from batches A+B (update before running from results).
    _mk("dann_sc1", "ctx48_dann03_sc_t007_lam01",
        dann=True, dann_lambda=0.4,
        supcon=True, supcon_temp=0.07, supcon_lambda=0.1),
    _mk("dann_sc2", "ctx48_dann05_sc_t007_lam03",
        dann=True, dann_lambda=0.5,
        supcon=True, supcon_temp=0.07, supcon_lambda=0.3),

    # attn1: attention-MIL replaces LSE. learns per-voxel attention weights -> model
    # learns WHERE the ink is (sub-tile soft segmentation emerges for free from tile labels).
    # also improves SNR on faint strokes (concentrates gradient on signal voxels).
    _mk("attn1",    "ctx48_attentionmil", attn_mil=True),

    # dann+attn, sc+attn, dann+sc+attn: does attention-MIL compose well with the invariance/
    # contrastive regularizers?
    _mk("dann_attn",    "ctx48_dann04_attnmil",
        dann=True, dann_lambda=0.4, attn_mil=True),
    _mk("sc_attn",      "ctx48_sc_t007_lam01_attnmil",
        supcon=True, supcon_temp=0.07, supcon_lambda=0.1, attn_mil=True),
    _mk("dann_sc_attn", "ctx48_dann04_sc_t007_attnmil",
        dann=True, dann_lambda=0.4,
        supcon=True, supcon_temp=0.07, supcon_lambda=0.1, attn_mil=True),

    # ==============================================================================
    # BATCH D: mean teacher with 2.4um verified-negative supervision
    # ==============================================================================
    # mt_vn1-3: mean-teacher EMA + extra supervised negatives from 2.4um inklabels.
    # tiles where 2.4um label < verified_neg_threshold are trusted papyrus ->
    # extra BCE supervision on those hard tiles. three lambda values tested.
    # NOTE: test_consistency=False here -- isolates the verified-neg effect.
    _mk("mt_vn1", "ctx48_mt_vn_lam01",
        mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.1,
        mean_teacher_ramp_epochs=3, verified_neg_lambda=0.2, test_consistency=False),
    _mk("mt_vn3", "ctx48_mt_vn_lam03",
        mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.3,
        mean_teacher_ramp_epochs=3, verified_neg_lambda=0.4, test_consistency=False),

    # ==============================================================================
    # BATCH E: test-scroll consistency (EMA + unlabeled test fragments)
    # ==============================================================================
    # mt_tc1: add test-scroll consistency on top of verified-neg supervision. teacher
    # predictions on unlabeled test-scroll tiles provide soft targets for the student ->
    # adapts the model toward the actual test-domain WITHOUT ever asserting a class.
    _mk("mt_tc1",  "ctx48_mt_vn_tc",
        mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.2,
        mean_teacher_ramp_epochs=3, verified_neg_lambda=0.3,
        test_consistency=True, test_consistency_lambda=0.1),

    # mt_full: mean-teacher + DANN + SupCon all together (kitchen-sink on the best arch
    # variant so far from batch C). update arch flags from batch C winner.
    # _mk("mt_full", "ctx48_mt_dann_sc",
    #     mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.2,
    #     mean_teacher_ramp_epochs=3, verified_neg_lambda=0.3,
    #     test_consistency=True, test_consistency_lambda=0.1,
    #     dann=True, dann_lambda=0.4,
    #     supcon=True, supcon_temp=0.07, supcon_lambda=0.1),

    # ==============================================================================
    # BATCH F: attention-MIL combos + grand finale
    # ==============================================================================
    # attn+mean-teacher verified-neg: the two most orthogonal levers (attention = where-to-look;
    # verified-neg = what-counts-as-papyrus). should be cleanly additive.
    # _mk("attn_mt",   "ctx48_attnmil_mt_vn",
    #     attn_mil=True,
    #     mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.2,
    #     mean_teacher_ramp_epochs=3, verified_neg_lambda=0.3, test_consistency=False),

    # # attn+DANN: does attention make domain-adversarial more effective (attention focuses
    # # on ink voxels; DANN removes scroll-identity -> cleaner invariant ink detection)?
    # _mk("attn_dann",  "ctx48_attnmil_dann04",
    #     attn_mil=True, dann=True, dann_lambda=0.4),

    # # grand finale: all proven components combined. use the winners from A-F.
    # _mk("grand",     "ctx48_grand_all",
    #     attn_mil=True,
    #     dann=True, dann_lambda=0.4,
    #     supcon=True, supcon_temp=0.07, supcon_lambda=0.1,
    #     mean_teacher=True, mean_teacher_alpha=0.999, mean_teacher_lambda=0.2,
    #     mean_teacher_ramp_epochs=3, verified_neg_lambda=0.3,
    #     test_consistency=True, test_consistency_lambda=0.1),
]

# dict-key -> (config-section, attribute)
_OVERRIDES = {
    "arch":                ("model", "arch"),
    "attn_mil":            ("model", "attn_mil"),
    "n_epochs":            ("tra", "n_epochs"),
    "eval_int":            ("tra", "eval_int"),
    "probe_int":           ("tra", "probe_int"),
    "l1":                  ("tra", "l1_lambda"),
    "weight_decay":        ("tra", "weight_decay"),
    "tta_consistency":     ("tra", "tta_consistency"),
    "tta_cons_lambda":     ("tra", "tta_consistency_lambda"),
    "tta_cons_mode":       ("tra", "tta_consistency_mode"),
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
    # mean teacher
    "mean_teacher":           ("tra", "mean_teacher"),
    "mean_teacher_alpha":     ("tra", "mean_teacher_alpha"),
    "mean_teacher_lambda":    ("tra", "mean_teacher_lambda"),
    "mean_teacher_ramp_epochs": ("tra", "mean_teacher_ramp_epochs"),
    "verified_neg_lambda":    ("tra", "verified_neg_lambda"),
    "test_consistency":       ("tra", "test_consistency"),
    "test_consistency_lambda":("tra", "test_consistency_lambda"),
    # data
    "context_size":        ("data", "context_size"),
    "context_downsample":  ("data", "context_downsample"),
    "ring_label_source":   ("data", "ring_label_source"),
    # dataloader
    "batch_size":          ("dl", "batch_size"),
}


def build_config(t: dict) -> Config:
    tid = t["tid"]; tag = t["tag"]
    c = _base_config(f"cmp_archs_{tid}_{tag}")

    for k, (sec, attr) in _OVERRIDES.items():
        if k in t:
            setattr(getattr(c, sec), attr, t[k])

    iw = t.get("init_weights")
    if iw and os.path.exists(iw):
        c.init_weights = iw
    elif iw:
        print(f"[archs] init_weights '{iw}' not found -- {tid} trains from scratch")

    c.dl.data_aug = any([c.dl.flip_prob, c.dl.rotation_prob, c.dl.noise_prob,
                         c.dl.brightness_prob, c.dl.contrast_prob,
                         c.dl.cutout_prob, c.dl.depth_mask_prob])
    c.dl.channel_mixing_prob = 0.0
    os.makedirs("models/archs", exist_ok=True)
    c.save_final = f"models/archs/{tid}_{tag}_final.pth"

    # validate sentinels
    _bad = []
    if c.tra.weight_decay == 999.0:
        _bad.append("weight_decay=999 (SENTINEL — set from wd sweep winner)")
    if c.data.ring_label_source == "BROKEN":
        _bad.append("ring_label_source=BROKEN (SENTINEL — set from c8/c9 comparison)")
    if c.tra.tta_consistency_lambda == 999.0:
        _bad.append("tta_consistency_lambda=999 (SENTINEL — set from cons sweep winner)")
    if c.tra.tta_consistency_mode == "BROKEN":
        _bad.append("tta_consistency_mode=BROKEN (SENTINEL — set 'flips' or 'dihedral')")
    if _bad:
        raise ValueError(f"[archs] {tid}: SENTINEL values NOT replaced:\n  " + "\n  ".join(_bad))
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[archs] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch={c.model.arch}  ctx={c.data.context_size} ds={c.data.context_downsample}"
          f"  attn_mil={c.model.attn_mil}")
    print(f"  ring={c.data.ring_label_source}  n_epochs={c.tra.n_epochs}"
          f"  wd={c.tra.weight_decay:.1e}  l1={c.tra.l1_lambda:.1e}")
    print(f"  dann={c.tra.dann} lam={c.tra.dann_lambda}"
          f"  supcon={c.tra.supcon} lam={c.tra.supcon_lambda} T={c.tra.supcon_temp}")
    print(f"  mt={c.tra.mean_teacher} lam={c.tra.mean_teacher_lambda} alpha={c.tra.mean_teacher_alpha}"
          f"  tc={c.tra.test_consistency} lam={c.tra.test_consistency_lambda}")
    print(f"  tta_cons={c.tra.tta_consistency} lam={c.tra.tta_consistency_lambda}"
          f"  mode={c.tra.tta_consistency_mode}")
    print(f"  scrolls={len(c.data.scrolls)}")
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
        # force GPU cleanup on failure
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return False


def main():
    ap = argparse.ArgumentParser(description="campaign_archs: architectural regularization sweep")
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

    print(f"[archs] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print("[archs] NOTE: sentinels (wd/ring/tta) must be set to winners before a real run!")

    results = {}
    for i, t in enumerate(selected):
        tid = t["tid"]
        try:
            c = build_config(t)
        except ValueError as e:
            print(f"\n[ABORT] {e}")
            return
        ok = run_test(c, args.dry_run)
        results[tid] = "OK" if ok else "FAIL"
        if not args.dry_run:
            del c; gc.collect()
            # force GPU memory release between tests
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if i < len(selected) - 1 and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, f"after {tid}")

    print(f"\n{'='*70}\n[archs] SUMMARY\n{'='*70}")
    for tid, status in results.items():
        tag = next(t["tag"] for t in TESTS if t["tid"] == tid)
        print(f"  {tid} ({tag}): {status}")


if __name__ == "__main__":
    main()
