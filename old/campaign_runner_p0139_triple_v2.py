"""campaign_runner_p0139_triple_v2.py -- config-driven 11-test triple-scroll sweep.

uses the NEW system: each test instantiates Config() and mutates fields directly,
then passes the config object to Trainer. no CLI args passed to train.py.

TESTS (11, skipping t01_base_t16 which already completed OK):
  t02  base     tile=24  depth=8   range=0-28
  t03  zgrad    tile=16  depth=8   range=0-28   (variation 1: depth-gradient input)
  t04  lcn      tile=16  depth=8   range=0-28   (variation 2: LCN + depth PE)
  t05  zgrad    tile=24  depth=8   range=0-28
  t06  lcn      tile=24  depth=8   range=0-28
  t07  base     tile=16  depth=4   range=0-28
  t08  base     tile=24  depth=4   range=0-28
  t09  base     tile=16  depth=8   range=8-16
  t10  base     tile=24  depth=8   range=8-16
  t11  base     tile=16  depth=4   range=8-16
  t12  base     tile=24  depth=4   range=8-16

all runs share:
  - triple scroll: w044 + w059 + w047  (per-scroll splits baked into DEFAULT_SCROLLS)
  - 20 epochs, eval at end (eval_int=20), no test figure (test_int=9999)
  - ring eroded negatives, l1=3e-7, dropout (0.05, 0.075), batch=64, num_workers=0
  - long thermal cooldowns: epoch 90s, val 120s, eval 600s, fig_chunk 600ms
  - 420s inter-run cooldown

run all:   python campaign_runner_p0139_triple_v2.py
run one:   python campaign_runner_p0139_triple_v2.py --only t04
dry-run:   python campaign_runner_p0139_triple_v2.py --dry-run
"""
from __future__ import annotations
import argparse, copy, os, sys, time, traceback
from dataclasses import replace
from pathlib import Path
from typing import List

# add repo root to path so utils imports work when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, ModelConfig

INTER_RUN_COOLDOWN_SECS = 420

# ---------------------------------------------------------------------------
# test definitions: each entry overrides specific config fields
# ---------------------------------------------------------------------------
def _base_config(exp_name: str) -> Config:
    """return a fresh config with all triple-campaign defaults."""
    c = Config()
    c.exp_name = exp_name
    # use default scrolls (w044 + w059 + w047 with correct per-scroll splits)
    # training range and epochs
    c.data.train_d_start = 0
    c.data.train_d_end   = 28
    c.data.d_start       = 0
    c.data.d_end         = 28
    c.tra.n_epochs  = 20
    c.tra.eval_int  = 20    # one eval figure at the very end
    c.tra.test_int  = 9999  # no test figure (expensive, only on demand)
    c.tra.probe_int          = 5     # probe ROI figures every 5 epochs
    c.tra.log_dir   = "./runs_p0139_triple"
    # regularisation
    c.tra.l1_lambda   = 3e-7
    c.model.conv1_drop = 0.05
    c.model.conv2_drop = 0.075
    # data loading
    c.dl.batch_size  = 64
    c.dl.num_workers = 0
    c.dl.data_aug    = False
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "eroded"
    # thermal cooldowns (idle sleeps, not compute)
    c.tra.epoch_cooldown_secs   = 15
    c.tra.val_cooldown_secs     = 15
    c.tra.eval_cooldown_secs    = 60
    c.tra.fig_chunk_cooldown_ms = 60
    return c


TESTS = [
    # (test_id,  arch,            tile, depth, d_start, d_end, extra_mutators)
    ("t02", "v14_mil_deep",   24,  8,  0,  28, {}),
    ("t03", "v14b_mil_zgrad", 16,  8,  0,  28, {}),
    ("t04", "v14c_mil_lcn",   16,  8,  0,  28, {}),
    ("t05", "v14b_mil_zgrad", 24,  8,  0,  28, {}),
    ("t06", "v14c_mil_lcn",   24,  8,  0,  28, {}),
    ("t07", "v14_mil_deep",   16,  4,  0,  28, {}),
    ("t08", "v14_mil_deep",   24,  4,  0,  28, {}),
    ("t09", "v14_mil_deep",   16,  8,  8,  16, {}),
    ("t10", "v14_mil_deep",   24,  8,  8,  16, {}),
    ("t11", "v14_mil_deep",   16,  4,  8,  16, {}),
    ("t12", "v14_mil_deep",   24,  4,  8,  16, {}),
    # t13: same as t02 (base tile=24) but with all augmentations turned on
    ("t13", "v14_mil_deep",   24,  8,  0,  28, {"aug": True}),
]


def build_config(tid: str, arch: str, tile: int, depth: int,
                 d_start: int, d_end: int, extra: dict) -> Config:
    campaign = "p0139_triple_v2_2026_07_17"
    c = _base_config(f"cmp_{campaign}_{tid}_{arch}_t{tile}_d{depth}_r{d_start}to{d_end}")
    c.model.arch       = arch
    c.data.tile_size   = tile
    c.data.depth       = depth
    c.data.d_start     = d_start
    c.data.d_end       = d_end
    c.data.train_d_start = d_start
    c.data.train_d_end   = d_end
    # per-test checkpoint path (stored as a freeform attribute on Config)
    c.save_final = f"models/triple/{tid}_{arch}_t{tile}_d{depth}_r{d_start}_{d_end}_final.pth"
    for k, v in extra.items():
        if k == "aug":
            c.dl.data_aug = bool(v)
        else:
            setattr(c, k, v)
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    """run one training configuration. returns True on success."""
    print(f"\n{'='*70}\n[triple_v2] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch={c.model.arch}  tile={c.data.tile_size}  depth={c.data.depth}"
          f"  range={c.data.d_start}-{c.data.d_end}"
          f"  aug={c.dl.data_aug}  probe_int={c.tra.probe_int}")
    if dry_run:
        print("  [DRY RUN] skipping")
        return True
    # import Trainer here so the campaign runner itself has no GPU deps at import time
    from train import Trainer
    try:
        trainer = Trainer(c)
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(description="triple-scroll config-driven sweep v2")
    ap.add_argument("--only", type=str, default=None,
                    help="run only the test whose id starts with this string (e.g. t04)")
    ap.add_argument("--from", dest="from_id", type=str, default=None,
                    help="start from the test with this id, skipping all earlier ones (e.g. t03)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build configs and print descriptions without training")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        selected = [(tid, *rest) for tid, *rest in TESTS if tid.startswith(args.only)]
        if not selected:
            print(f"[ABORT] --only '{args.only}' matched nothing; valid ids: {[t[0] for t in TESTS]}")
            return
    elif args.from_id:
        ids = [t[0] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid ids: {ids}")
            return
        idx = ids.index(args.from_id)
        selected = TESTS[idx:]

    print(f"\n[triple_v2] {len(selected)} test(s) queued")
    results: list[str] = []

    for i, (tid, arch, tile, depth, d_start, d_end, extra) in enumerate(selected, 1):
        c = build_config(tid, arch, tile, depth, d_start, d_end, extra)
        ok = run_test(c, args.dry_run)
        status = "OK" if ok else "FAIL"
        results.append(f"  {tid} ({arch} t{tile} d{depth} r{d_start}-{d_end}): {status}")
        print(f"[triple_v2] done {tid} -> {status}", flush=True)
        if i < len(selected) and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, "inter-run")

    print(f"\n{'='*70}\n[triple_v2] SUMMARY\n{'='*70}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
