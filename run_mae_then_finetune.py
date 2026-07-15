"""run_mae_then_finetune.py — chained MAE pretrain (50% mask, 6000 steps) -> ink fine-tune.

runs the two stages in sequence so you can launch once:
  STAGE 1: mae_pretrain.py  (6000 steps, mask-frac 0.50, both scrolls, thermal cooldowns)
           -> models/mae_dense_unet.pth
  STAGE 2: train.py fine-tune, warm-started from that MAE checkpoint (--init-weights),
           v7 config (top-75% crop, y-split 0.75, blur soft labels), 20 epochs, eval at 20.

stage 2 only starts if stage 1 exits cleanly (rc==0).
"""
from __future__ import annotations
import os, subprocess, sys, time

PY = sys.executable
REPO = os.path.dirname(os.path.abspath(__file__))

MAE_CKPT = "models/mae_dense_unet_resenc.pth"

MAE_CMD = [
    PY, "mae_pretrain.py",
    "-n", "mae_dense_unet_resenc",
    "--arch", "dense_unet_resenc_mae",   # residual-encoder MAE twin
    "--scroll-ids", "20240304161941", "20240304144031",
    "--tile-size", "64",
    "--steps", "6000",
    "--batch-size", "64",
    "--mask-frac", "0.50",          # 50% (75% was too aggressive)
    "--step-cooldown-ms", "120",    # steady per-step relief (was 40)
    "--cooldown-secs", "90",        # deeper periodic pause (was 30)
    "--cooldown-int", "100",        # more frequent (was 200)
]

FINETUNE_CMD = [
    PY, "train.py",
    "-n", "ink_from_mae_resenc",
    "--arch", "dense_unet_resenc",
    "--init-weights", MAE_CKPT,
    "--scroll-id", "20240304161941",
    "--dense-labels", "--dense-soft-labels",
    "--tile-size", "32", "--depth", "8",
    "--train-d-start", "0", "--train-d-end", "64",   # CRITICAL: full depth window (was the bug)
    "--d-start", "0", "--d-end", "64",
    "--epochs", "20", "--eval-int", "20", "--test-int", "60",
    "--ring-negatives", "--ring-label-source", "eroded",
    "--crop-x-frac", "0.6,1.0", "--crop-y-frac", "0.0,0.75",
    "--split-axis", "y", "--train-split-frac", "0.75",
    "--batch-size", "1024", "--lr", "2e-4",
    "--log-dir", "runs_scroll4_79um",
    "--num-workers", "2", "--mask-memmap",
    "--no-hard-mining", "--no-probe-rois", "--test-scroll2-only",
    "--eval-cooldown", "240", "--val-cooldown", "90", "--fig-chunk-cooldown", "200",
]


def run(cmd, label):
    print(f"\n{'='*78}\n[chain] START {label}\n  {' '.join(cmd)}\n{'='*78}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO)
    dt = time.time() - t0
    print(f"[chain] {label} exited rc={rc} after {dt/60:.1f} min", flush=True)
    return rc


def main():
    rc = run(MAE_CMD, "STAGE 1: MAE pretrain (6000 steps, 50% mask)")
    if rc != 0:
        print(f"[chain] ABORT — MAE stage failed (rc={rc}); not starting fine-tune.")
        return
    if not os.path.exists(os.path.join(REPO, MAE_CKPT)):
        print(f"[chain] ABORT — {MAE_CKPT} not found after MAE stage.")
        return
    run(FINETUNE_CMD, "STAGE 2: ink fine-tune from MAE")
    print("\n[chain] ALL DONE. compare runs_scroll4_79um/dense_figs/dense_eval_ep20.png "
          "(ink_from_mae) vs the from-scratch v7 figure.")


if __name__ == "__main__":
    main()
