"""campaign_runner_lcn.py -- LCN refinement sweep (4-scroll, quad PHerc0139).

winner from triple_v2: v14c_mil_lcn (t06/t04) — extreme overfitter, need tighter reg.
this campaign probes tile size and depth stack with higher l1 (7e-6) to rein it in.

SCROLLS (4):
  w044  20260115000000  split y 0.8055
  w059  20250223000000  split x 0.75
  w047  20260206000001  split x 0.75
  w056  20260115000001  split y 0.50   (new, top=train / bottom=val)

TESTS:
  t01  v14c_mil_lcn  tile=16  depth=8  range=8-16
  t02  v14c_mil_lcn  tile=16  depth=4  range=8-16
  t03  v14c_mil_lcn  tile=8   depth=4  range=8-16

shared:
  - 4-scroll: w044 + w059 + w047 + w056 (DEFAULT_SCROLLS)
  - l1_lambda = 7e-6  (ramped up from 3e-7 to combat overfitting)
  - 20 epochs, eval at end, probes every 5 epochs
  - ring eroded negatives, batch=64, num_workers=0
  - cooldowns: epoch 90s / val 120s / eval 600s / fig_chunk 600ms
  - 420s inter-run cooldown

run all:   python campaign_runner_lcn.py
run one:   python campaign_runner_lcn.py --only t02
dry-run:   python campaign_runner_lcn.py --dry-run
"""
from __future__ import annotations
import argparse, os, sys, time, traceback
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config

INTER_RUN_COOLDOWN_SECS = 420


def _base_config(exp_name: str) -> Config:
    """fresh config with all lcn-campaign defaults (4-scroll, high l1)."""
    c = Config()
    c.exp_name = exp_name
    # DEFAULT_SCROLLS already includes all 4 fragments; nothing to override here
    c.data.d_start       = 8
    c.data.d_end         = 16
    c.data.train_d_start = 8
    c.data.train_d_end   = 16
    c.tra.n_epochs  = 20
    c.tra.eval_int  = 20
    c.tra.test_int  = 9999
    c.tra.probe_int = 5
    c.tra.log_dir   = "./runs_lcn"
    # higher l1 to counter the overfitting tendency of lcn
    c.tra.l1_lambda    = 7e-6
    c.model.arch       = "v14c_mil_lcn"
    c.model.conv1_drop = 0.05
    c.model.conv2_drop = 0.075
    c.dl.batch_size  = 64
    c.dl.num_workers = 0
    c.dl.data_aug    = False
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "eroded"
    # thermal cooldowns
    c.tra.epoch_cooldown_secs   = 90
    c.tra.val_cooldown_secs     = 120
    c.tra.eval_cooldown_secs    = 600
    c.tra.fig_chunk_cooldown_ms = 600
    return c


# (test_id, tile, depth)  — all lcn, range 8-16
TESTS = [
    ("t01", 16, 8),
    ("t02", 16, 4),
    ("t03",  8, 4),
]


def build_config(tid: str, tile: int, depth: int) -> Config:
    campaign = "lcn_2026_07_19"
    c = _base_config(f"cmp_{campaign}_{tid}_v14c_mil_lcn_t{tile}_d{depth}_r8to16")
    c.data.tile_size = tile
    c.data.depth     = depth
    c.save_final = f"models/lcn/{tid}_v14c_mil_lcn_t{tile}_d{depth}_r8_16_final.pth"
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_test(c: Config, dry_run: bool) -> bool:
    print(f"\n{'='*70}\n[lcn] {c.exp_name}\n{'='*70}", flush=True)
    print(f"  arch={c.model.arch}  tile={c.data.tile_size}  depth={c.data.depth}"
          f"  range={c.data.d_start}-{c.data.d_end}"
          f"  l1={c.tra.l1_lambda:.1e}  probe_int={c.tra.probe_int}")
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
        return False


def main():
    ap = argparse.ArgumentParser(description="lcn refinement sweep")
    ap.add_argument("--only", type=str, default=None,
                    help="run only the test whose id starts with this string (e.g. t02)")
    ap.add_argument("--from", dest="from_id", type=str, default=None,
                    help="start from this test id, skipping earlier ones (e.g. t02)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print configs without training")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        selected = [(tid, *rest) for tid, *rest in TESTS if tid.startswith(args.only)]
        if not selected:
            print(f"[ABORT] --only '{args.only}' matched nothing; valid: {[t[0] for t in TESTS]}")
            return
    elif args.from_id:
        ids = [t[0] for t in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"\n[lcn] {len(selected)} test(s) queued  (l1=7e-6, 4 scrolls incl. w056)")
    results: List[str] = []

    for i, (tid, tile, depth) in enumerate(selected, 1):
        c = build_config(tid, tile, depth)
        ok = run_test(c, args.dry_run)
        status = "OK" if ok else "FAIL"
        results.append(f"  {tid} (v14c_mil_lcn t{tile} d{depth} r8-16): {status}")
        print(f"[lcn] done {tid} -> {status}", flush=True)
        if i < len(selected) and not args.dry_run:
            cooldown(INTER_RUN_COOLDOWN_SECS, "inter-run")

    print(f"\n{'='*70}\n[lcn] SUMMARY\n{'='*70}")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
