"""campaign_runner_scroll4_79um_v5.py — 30-epoch original config rerun.

exact reproduction of cmp_scroll4_79um_2026_07_09_t01_dense_unet_10_00-11-06,
the best-performing run (valid pixel AUC 0.5548 at epoch 15, visually superior).

changes from that run:
  - 30 epochs instead of 15 (to see if the signal continues improving)
  - eval_int=30 (fires once at the very end)
  - thermal cooldowns retained (val_cooldown=45s, eval_cooldown=120s, fig_chunk_cooldown=100ms)
    — these were not present in the original but are needed to prevent laptop crashes

everything else is EXACTLY as the original:
  - arch: dense_unet  WITH BatchNorm (reverted from InstanceNorm)
  - dense_soft_labels=False  (hard eroded labels, no Gaussian blur)
  - tile_size=32, depth=8
  - batch_size=1024, num_workers=2
  - no residuals, no attention gates, no architecture changes
  - ring negatives (eroded source)
  - same crop: right 40% x, top 75% y, y-split 75/25
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCROLL4_79_ID = 20240304161941
EPOCHS        = 30

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

BASE: Dict[str, Any] = {
    "scroll-id":          SCROLL4_79_ID,
    "tile-size":          32,
    "depth":              8,
    "train-d-start":      0,
    "train-d-end":        64,
    "d-start":            0,
    "d-end":              64,
    "epochs":             EPOCHS,
    "eval-int":           EPOCHS,          # eval figure once at the very end
    "probe-int":          5,
    "test-int":           60,              # never fires (> epochs)
    "test-scroll2-only":  True,
    "lr":                 2e-4,
    "l1-lambda":          0.0,
    "conv1-drop":         0.0,
    "conv2-drop":         0.0,
    "fc1-drop":           0.0,
    "fc2-drop":           0.0,
    "data-aug":           0,
    "channel-mixing-prob": 0.0,
    "ring-negatives":     True,
    "ring-label-source":  "eroded",
    "crop-x-frac":        "0.6,1.0",
    "crop-y-frac":        "0.0,0.75",
    "split-axis":         "y",
    "train-split-frac":   0.75,
    "dense-labels":       True,
    # NO dense-soft-labels — hard eroded labels only (original config)
    "batch-size":         1024,
    "num-workers":        2,
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    "ranking-lambda":     0.0,
    # thermal cooldowns (not in original, but needed to prevent crashes)
    "eval-cooldown":      120,
    "val-cooldown":       45,
    "fig-chunk-cooldown": 100,
    # arch: dense_unet with BatchNorm — THE ORIGINAL CONFIGURATION
    "arch":               "dense_unet",
}

RUN_SPECS: List[Dict[str, Any]] = [
    {
        "name": "t01_original_30ep",
        # no overrides — all BASE settings are already the original config
    },
]


def dict_to_cli_args(d: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in d.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def build_cmd(python_exe: str, runs_dir: Path, campaign_id: str, spec: Dict[str, Any]):
    merged = dict(BASE)
    for k, v in spec.items():
        if k != "name":
            merged[k] = v
    exp_name = f"cmp_{campaign_id}_{spec['name']}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def labels_exist(sid: int) -> bool:
    return (Path("eroded_inklabels") / f"{sid}.png").exists()


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=120.0):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env,
                                stdout=lf, stderr=lf)
    last_progress = time.time()
    last_epoch    = 0
    while proc.poll() is None:
        time.sleep(20)
        try:
            lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        except Exception:
            continue
        tail = "".join(lines[-80:])
        for sig in CRASH_SIGNALS:
            if sig in tail:
                print(f"\n[MONITOR] CRASH -- '{sig}'\n" + "".join(lines[-15:]))
                try: proc.kill()
                except Exception: pass
                proc.wait()
                return proc.returncode or 1, True
        for line in lines[-80:]:
            if "--- Epoch" in line:
                try:
                    ep = int(line.strip().split("/")[0].split()[-1])
                    if ep > last_epoch:
                        last_epoch = ep; last_progress = time.time()
                        print(f"[MONITOR] {line.strip()}")
                except Exception:
                    pass
        if time.time() - last_progress > stall_minutes * 60:
            print(f"\n[MONITOR] STALL — no progress in {stall_minutes:.0f} min")
            try: proc.kill()
            except Exception: pass
            proc.wait()
            return 1, True
    proc.wait()
    rc = proc.returncode
    print(f"[MONITOR] {'OK' if rc == 0 else f'exited rc={rc}'}")
    return rc, False


def main():
    ap = argparse.ArgumentParser(description="v5: original config 30-epoch rerun")
    ap.add_argument("--campaign-id",   type=str,   default="scroll4_79um_v5_2026_07_12")
    ap.add_argument("--python-exe",    type=str,   default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=120.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_scroll4_79um"
    runs_dir.mkdir(exist_ok=True)
    log_dir   = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    print("\n" + "=" * 78)
    print(f"[v5] ORIGINAL CONFIG RERUN — BatchNorm + hard labels + dense_unet")
    print(f"  {EPOCHS} epochs | tile=32 | depth=8 | batch=1024 | no soft labels")
    print("=" * 78)

    if not labels_exist(SCROLL4_79_ID):
        print(f"[ABORT] eroded_inklabels/{SCROLL4_79_ID}.png not found")
        return

    for spec in RUN_SPECS:
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"\n   exp: {exp_name}")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                          stall_minutes=args.stall_minutes)
        print(f"[v5] done  rc={rc}  crashed={crashed}")

    if args.dry_run:
        print("\n[v5] dry-run only")


if __name__ == "__main__":
    main()
