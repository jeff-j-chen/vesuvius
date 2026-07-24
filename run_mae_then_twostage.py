"""run_mae_then_twostage.py -- one-shot pipeline: MAE-pretrain the two-stage backbone, then
run the two-stage campaign warm-started from that MAE checkpoint.

sequential + fail-fast: if MAE fails or the checkpoint is missing, we abort BEFORE fine-tuning
so we never warm-start from a half-written/absent file. one MAE checkpoint warm-starts the
fine-tune(s) (the backbone is fully-convolutional and spatially size-agnostic, so the same
stage1.* weights transfer regardless of the fine-tune crop size).

current campaign (campaign_runner_twostage.py TESTS):
  tsJd  32px context window (label 16px) + very-strong reg + MAE

usage:  python run_mae_then_twostage.py                 # MAE 32px x4000 -> campaign
        python run_mae_then_twostage.py --mae-steps 6000
        python run_mae_then_twostage.py --skip-mae      # reuse existing models/mae_twostage.pth
"""
from __future__ import annotations
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MAE_CKPT = os.path.join("models", "mae_twostage.pth")


def run(cmd, label):
    print(f"\n{'='*72}\n[orchestrator] {label}\n  {' '.join(cmd)}\n{'='*72}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        print(f"[orchestrator] {label} FAILED (exit {r.returncode}); aborting.", flush=True)
        sys.exit(r.returncode)
    print(f"[orchestrator] {label} done.", flush=True)


def main():
    ap = argparse.ArgumentParser(description="MAE pretrain -> two-stage campaign")
    ap.add_argument("--mae-steps", type=int, default=4000)
    ap.add_argument("--mae-tile", type=int, default=32,
                    help="MAE crop size; matched to the 32px context fine-tune (backbone is "
                         "fully-conv so weights transfer to any fine-tune crop size)")
    ap.add_argument("--skip-mae", action="store_true",
                    help="reuse an existing models/mae_twostage.pth instead of re-running MAE")
    args = ap.parse_args()

    if args.skip_mae:
        if not os.path.exists(MAE_CKPT):
            print(f"[orchestrator] --skip-mae but {MAE_CKPT} not found; aborting."); sys.exit(1)
        print(f"[orchestrator] reusing existing {MAE_CKPT}")
    else:
        run([PY, "mae_pretrain_twostage.py", "-n", "mae_twostage",
             "--arch", "v15_twostage_wide_zgrad",
             "--tile-size", str(args.mae_tile), "--steps", str(args.mae_steps)],
            "STAGE 1/2: MAE pretraining")
        if not os.path.exists(MAE_CKPT):
            print(f"[orchestrator] expected {MAE_CKPT} not found after MAE; aborting."); sys.exit(1)

    run([PY, "campaign_runner_twostage.py"],
        "STAGE 2/2: two-stage campaign (4 runs, warm-started from MAE)")
    print("\n[orchestrator] ALL DONE.", flush=True)


if __name__ == "__main__":
    main()
