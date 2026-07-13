"""campaign_runner_scroll4_79um_v7.py — train exclusively on the hand-corrected labels.

reproduces the winning a1_blur configuration EXACTLY, on the new manually-written
labels for scroll 20240304161941:
  - eroded_inklabels/20240304161941.png  (hand-written, 2026-07-13)
  - soft_inklabels/20240304161941.png    (hand-written, 2026-07-13)

winning config (from cmp_scroll4_79um_v6_2026_07_12_a1_blur):
  - arch: dense_unet (BatchNorm)
  - dense_soft_labels=True   (blurred per-pixel target)
  - ring_negatives=True, ring_label_source='eroded'
  - tile=32, depth=8, batch=1024, lr=2e-4
  - crop right 40% x / top 75% y, y-split 75/25

CRITICAL: this runner does NOT generate or touch soft_inklabels — the user's
hand-written labels are used as-is and must not be overwritten. the training
pipeline (DataManager) only READS the label PNGs, never writes them.

20 epochs, eval_int=20 (fires once at end), test_int=60 (never), same folder.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCROLL4_79_ID = 20240304161941
EPOCHS        = 20

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
    "eval-int":           EPOCHS,
    "probe-int":          5,
    "test-int":           60,          # never fires
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
    "ring-label-source":  "eroded",    # ring boundary from eroded labels (winning config)
    "crop-x-frac":        "0.6,1.0",
    "crop-y-frac":        "0.0,0.75",
    "split-axis":         "y",
    "train-split-frac":   0.75,
    "dense-labels":       True,
    "dense-soft-labels":  True,         # blurred per-pixel target (hand-written soft labels)
    "batch-size":         1024,
    "num-workers":        2,
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    "ranking-lambda":     0.0,
    "eval-cooldown":      120,
    "val-cooldown":       45,
    "fig-chunk-cooldown": 100,
    "arch":               "dense_unet",
}

RUN_SPECS: List[Dict[str, Any]] = [
    {"name": "t01_handlabels_20ep"},   # no overrides — BASE is the winning config
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


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=120.0):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=lf, stderr=lf)
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
    ap = argparse.ArgumentParser(description="v7: train on hand-corrected labels")
    ap.add_argument("--campaign-id",   type=str,   default="scroll4_79um_v7_2026_07_13")
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
    print("[v7] HAND-LABEL RUN — winning a1_blur config on hand-written labels")
    print(f"  {EPOCHS} epochs | dense_unet BatchNorm | blur target | ring eroded")
    print("=" * 78)

    # verify the hand-written labels exist (do NOT generate or modify them)
    for sub in ("eroded_inklabels", "soft_inklabels"):
        p = Path(sub) / f"{SCROLL4_79_ID}.png"
        if not p.exists():
            print(f"[ABORT] {p} not found"); return
        print(f"  using {p}  (mtime {time.ctime(p.stat().st_mtime)})")

    for spec in RUN_SPECS:
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"\n   exp: {exp_name}")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path, args.stall_minutes)
        print(f"[v7] done  rc={rc}  crashed={crashed}")

    if args.dry_run:
        print("\n[v7] dry-run only")


if __name__ == "__main__":
    main()
