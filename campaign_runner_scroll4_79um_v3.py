"""campaign_runner_scroll4_79um_v3.py — tile=32, back to basics, 2 targeted runs.

following the v2 result (tile=48 regressed from 0.5548 to 0.5066), this campaign
returns to tile=32 and tests two specific hypotheses at the correct tile scale:

  A) t01_blur_d8      — dense_unet, depth=8, soft labels (sigma=15)
                         isolates whether blurred labels help vs hard labels
                         (v1 30ep run used blurred labels; this gives a clean 10ep baseline)

  C) t02_blur_res_d8  — dense_unet_res_attn, depth=8, soft labels (sigma=15)
                         tests residual blocks + attention gates at the correct tile scale

all runs:
  - tile_size=32  (proven best scale)
  - InstanceNorm (new default in all dense archs)
  - soft inklabels sigma=15
  - 10 epochs, eval_int=10 (fires once at end), test_int=30 (never fires)
  - same thermal cooldowns as v2 (val_cooldown=45s, eval_cooldown=120s)
  - logs -> runs_scroll4_79um (same tensorboard dir as v1/v2)
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCROLL4_79_ID = 20240304161941
EPOCHS        = 10

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

BASE: Dict[str, Any] = {
    "scroll-id":          SCROLL4_79_ID,
    "tile-size":          32,           # proven optimal scale
    "train-d-start":      0,
    "train-d-end":        64,
    "d-start":            0,
    "d-end":              64,
    "epochs":             EPOCHS,
    "eval-int":           EPOCHS,
    "probe-int":          5,
    "test-int":           30,
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
    "dense-soft-labels":  True,         # sigma=15 blurred labels for both runs
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    "ranking-lambda":     0.0,
    "eval-cooldown":      120,
    "val-cooldown":       45,
    "fig-chunk-cooldown": 100,
    "num-workers":        2,
}

RUN_SPECS: List[Dict[str, Any]] = [
    {
        "name":       "t01_blur_d8",
        "arch":       "dense_unet",
        "depth":      8,
        "batch-size": 1024,
        # measures blurred-label benefit vs v1 30ep run (which also used blurred labels
        # but ran 30 epochs). gives a clean 10ep blurred-label baseline at tile=32
        # with InstanceNorm, for direct comparison with the res+attn run below.
    },
    {
        "name":       "t02_blur_res_d8",
        "arch":       "dense_unet_res_attn",
        "depth":      8,
        "batch-size": 1024,
        # residual blocks + attention gates + blurred labels at tile=32.
        # same data regime as t01; any metric difference is attributable to the arch.
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


def soft_labels_exist(sid: int) -> bool:
    return (Path("soft_inklabels") / f"{sid}.png").exists()


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
    ap = argparse.ArgumentParser(description="scroll4 79um v3: tile=32 blur vs res+attn")
    ap.add_argument("--campaign-id",   type=str,   default="scroll4_79um_v3_2026_07_10")
    ap.add_argument("--python-exe",    type=str,   default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=120.0)
    ap.add_argument("--run",           type=str,   default=None)
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
    print(f"[scroll4 79um v3] id={SCROLL4_79_ID}  tile=32  blur+soft_labels")
    print(f"  runs: {[s['name'] for s in RUN_SPECS]}")
    print("=" * 78)

    if not labels_exist(SCROLL4_79_ID):
        print(f"[ABORT] eroded_inklabels/{SCROLL4_79_ID}.png not found")
        return
    if not soft_labels_exist(SCROLL4_79_ID):
        print(f"[ABORT] soft_inklabels/{SCROLL4_79_ID}.png not found")
        return

    specs = RUN_SPECS if not args.run else [s for s in RUN_SPECS if s["name"] == args.run]
    if not specs:
        print(f"[ABORT] unknown run '{args.run}'")
        return

    for spec in specs:
        print(f"\n{'='*78}")
        print(f"[v3] run: {spec['name']}  arch={spec['arch']}  depth={spec['depth']}  batch={spec['batch-size']}")
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"   exp: {exp_name}")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                          stall_minutes=args.stall_minutes)
        print(f"[v3] {spec['name']} done  rc={rc}  crashed={crashed}")
        if crashed:
            print("[v3] aborting remaining runs after crash")
            break

    if args.dry_run:
        print("\n[v3] dry-run only")


if __name__ == "__main__":
    main()
