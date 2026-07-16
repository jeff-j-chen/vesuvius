"""campaign_runner_p0139_double.py -- the winning arch, now DOUBLE-SCROLL.

runs the arch28 winner (v14_mil_deep -- the first arch to actually learn ink on this
dataset) on BOTH PHerc0139 fragments at once:
  w044  20260115000000   (9.4um, existing)
  w059  20250223000000   (9.4um, newly assembled; restricted to the 1.1um overlap band)

WINNER RECAP: v14_mil_deep is multiple-instance learning -- a per-slice texture stem +
3D depth-mix + a PER-VOXEL logit head, aggregated over all voxels by log-sum-exp (a soft,
learnable-hardness max). unlike v1 / asym_pool (which GLOBAL-average away space), MIL lets
a handful of high-confidence voxels drive the tile logit, so sparse ink survives. tile
output (BCE 32x32 -> 1 bit), so NO --dense-labels here.

CONFIG (per user): epochs 20, eval-int 20 (one figure at the end), l1-lambda 6e-7,
tiny dropout at the last layers (conv1_drop 0.10, conv2_drop 0.15 -> the two Dropout3d
in v14's depth_mix, conv2_drop being the very last before the voxel head).
probe/test OFF, ring eroded negatives, no aug, no hard mining, long thermal cooldowns.

double-scroll: --scroll-ids merges both fragments into one training stream; the visualizer
renders one eval figure PER scroll (namespaced s<id>/), each now with the extra
"MAX across all depths" row + gold overlay.

NOT started automatically -- run this file when the inklabels are ready:
  eroded_inklabels/20250223000000.png  (required for ring negatives on w059; user-made)
  soft_inklabels/20250223000000.png    (only needed if --dense-labels; not used here)
  w044's eroded/soft already exist.

  python campaign_runner_p0139_double.py            # launch
  python campaign_runner_p0139_double.py --dry-run  # print the command only
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

W044 = 20260115000000
W059 = 20250223000000
CKPT = "models/double/v14_mil_deep_double_final.pth"

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

ARGS: Dict[str, Any] = {
    "scroll-ids":         f"{W044},{W059}",   # double-scroll merged stream
    "tile-size":          32,
    "train-d-start":      0,
    "train-d-end":        28,
    "d-start":            0,
    "d-end":              28,
    "epochs":             20,
    "eval-int":           20,                 # one eval figure at the very end
    "probe-int":          9999,               # probes OFF
    "test-int":           9999,               # test figures OFF
    "ring-negatives":     True,
    "ring-label-source":  "eroded",
    "crop-x-frac":        "0.0,1.0",
    "crop-y-frac":        "0.0,1.0",
    "split-axis":         "y",
    "train-split-frac":   0.8055,
    "batch-size":         64,                 # v14_mil_deep memory footprint
    "lr":                 2e-4,
    "l1-lambda":          6e-7,               # per user
    "conv1-drop":         0.10,               # tiny dropout, depth_mix mid
    "conv2-drop":         0.15,               # tiny dropout, very last before voxel head
    "data-aug":           0,
    "num-workers":        2,
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    "arch":               "v14_mil_deep",
    "depth":              8,
    "save-final":         CKPT,
    # thermal cooldowns (idle sleeps)
    "epoch-cooldown":     90,
    "val-cooldown":       120,
    "eval-cooldown":      600,
    "fig-chunk-cooldown": 600,
    # NOTE: no --dense-labels -> tile BCE (v14_mil_deep outputs one logit per tile)
}


def dict_to_cli_args(d: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        else:
            args.extend([f"--{k}", str(v)])
    return args


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=180.0):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=lf, stderr=lf)
    last_progress = time.time(); last_epoch = 0
    while proc.poll() is None:
        time.sleep(20)
        try:
            lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        except Exception:
            continue
        tail = "".join(lines[-80:])
        for sig in CRASH_SIGNALS:
            if sig in tail:
                print(f"\n[MONITOR] CRASH -- '{sig}'\n" + "".join(lines[-12:]))
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
            print(f"\n[MONITOR] STALL -- no progress in {stall_minutes:.0f} min")
            try: proc.kill()
            except Exception: pass
            proc.wait()
            return 1, True
    proc.wait()
    rc = proc.returncode
    print(f"[MONITOR] {'OK' if rc == 0 else f'exited rc={rc}'}")
    return rc, False


def main():
    ap = argparse.ArgumentParser(description="double-scroll v14_mil_deep run")
    ap.add_argument("--python-exe",    type=str, default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=180.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_p0139_double"
    runs_dir.mkdir(exist_ok=True)
    log_dir   = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    (repo_root / "models" / "double").mkdir(parents=True, exist_ok=True)

    exp_name = "cmp_p0139_double_v14_mil_deep"
    cmd = [args.python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(ARGS)

    print("\n" + "=" * 78)
    print("[double] PHerc0139 w044 + w059 -- v14_mil_deep (arch28 winner), double-scroll")
    print(f"         epochs=20 eval-int=20 l1=6e-7 drop=(0.10,0.15) batch=64")
    print("=" * 78)
    print(f"   scrolls: {W044} (w044) + {W059} (w059, 1.1um overlap band)")
    print(f"   cmd: {' '.join(str(c) for c in cmd)}")

    # preflight: w059 needs eroded inklabels for ring negatives
    need = repo_root / "eroded_inklabels" / f"{W059}.png"
    if not need.exists():
        print(f"\n   [WARN] {need} missing -- ring negatives on w059 will fail.")
        print(f"          create the eroded inklabel for w059 before launching.")

    if args.dry_run:
        print("\n[double] dry-run only (not launched)")
        return

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    log_path = log_dir / f"{exp_name}.log"
    rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path, args.stall_minutes)
    print(f"[double] done  rc={rc}  crashed={crashed}")


if __name__ == "__main__":
    main()
