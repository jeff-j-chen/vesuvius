"""campaign_runner_scroll4_79um_v2.py — tile=48, InstanceNorm, 3 architectural variants.

following from the v1 campaign (dense_unet tile=32) which found AUC=0.5548, this
campaign tests three hypotheses at tile_size=48 (more spatial context per tile):

  A) t01_tile48_d8   — dense_unet, depth=8   (baseline: same arch, larger tile)
  B) t02_tile48_d4   — dense_unet, depth=4   (more z-windows via half-step)
  C) t03_tile48_res  — dense_unet_res_attn,  (residual blocks + attention gates)
                       depth=8

all runs:
  - tile_size=48  (48×48px ≈ 379×379μm — covers a full letter stroke + context)
  - InstanceNorm throughout (new default in dense archs)
  - soft inklabels sigma=15
  - 10 epochs, eval_int=10 (fires once at end), test_int=30 (never fires)
  - longer cooldowns to protect thermals (val_cooldown=45s, eval_cooldown=120s)
  - logs -> runs_scroll4_79um (same tensorboard dir as v1)

the probe size auto-snaps to tile multiples, so 512px probe windows become
 (512//48)*48 = 10*48 = 480px grids for readability metric computation.
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

# shared base — only things that differ from the training default
BASE: Dict[str, Any] = {
    "scroll-id":          SCROLL4_79_ID,
    "tile-size":          48,
    "train-d-start":      0,
    "train-d-end":        64,
    "d-start":            0,
    "d-end":              64,
    "epochs":             EPOCHS,
    "eval-int":           EPOCHS,      # full eval only at the final epoch
    "probe-int":          5,           # probe figure every 5 epochs
    "test-int":           30,          # > epochs, never fires
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
    "dense-soft-labels":  True,
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    "ranking-lambda":     0.0,
    # thermal management — longer than v1
    "eval-cooldown":      120,   # 2 min rest after probe/eval epoch
    "val-cooldown":       45,    # 45s between train and validation each epoch
    "fig-chunk-cooldown": 100,   # 100ms between eval figure chunks
    "num-workers":        2,
}

RUN_SPECS: List[Dict[str, Any]] = [
    {
        "name":       "t01_tile48_d8",
        "arch":       "dense_unet",
        "depth":      8,
        "batch-size": 512,
        # tile=48, depth=8: z_step=4 -> ~15 z-windows/tile
        # (1,8,48,48) fp16 ≈ 18KB/item; B=512 ≈ 9MB input, well within 24GB
    },
    {
        "name":       "t02_tile48_d4",
        "arch":       "dense_unet",
        "depth":      4,
        "batch-size": 1024,
        # depth=4: z_step=2 -> ~29 z-windows/tile (~2x more training data vs d8)
        # each item is half the size, so double the batch fits in same VRAM
    },
    {
        "name":       "t03_tile48_res",
        "arch":       "dense_unet_res_attn",
        "depth":      8,
        "batch-size": 512,
        # residual blocks + attention gates — same memory footprint as t01
        # attention gates add ~0.3M params (negligible) but focus decoder
        # on ink-boundary locations
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


def build_cmd(python_exe: str, runs_dir: Path, campaign_id: str,
              spec: Dict[str, Any]):
    merged = dict(BASE)
    for k, v in spec.items():
        if k != "name":
            merged[k] = v
    exp_name = f"cmp_{campaign_id}_{spec['name']}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def labels_exist(scroll_id: int) -> bool:
    return (Path("eroded_inklabels") / f"{scroll_id}.png").exists()


def soft_labels_exist(scroll_id: int) -> bool:
    return (Path("soft_inklabels") / f"{scroll_id}.png").exists()


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
    ap = argparse.ArgumentParser(description="scroll4 79um v2: tile=48 arch variants")
    ap.add_argument("--campaign-id",   type=str,   default="scroll4_79um_v2_2026_07_10")
    ap.add_argument("--python-exe",    type=str,   default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=120.0)
    ap.add_argument("--run",           type=str,   default=None,
                    help="run only a specific spec by name")
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
    print(f"[scroll4 79um v2] id={SCROLL4_79_ID}  tile=48  InstanceNorm")
    print(f"  runs: {[s['name'] for s in RUN_SPECS]}")
    print("=" * 78)

    if not labels_exist(SCROLL4_79_ID):
        print(f"[ABORT] eroded_inklabels/{SCROLL4_79_ID}.png not found")
        return
    if not soft_labels_exist(SCROLL4_79_ID):
        print(f"[ABORT] soft_inklabels/{SCROLL4_79_ID}.png not found — "
              f"run campaign_runner_scroll4_79um_dense.py dry-run to generate it")
        return

    specs = RUN_SPECS if not args.run else [s for s in RUN_SPECS if s["name"] == args.run]
    if not specs:
        print(f"[ABORT] unknown run name '{args.run}'")
        return

    for spec in specs:
        print(f"\n{'='*78}")
        print(f"[v2] run: {spec['name']}  arch={spec['arch']}  "
              f"depth={spec['depth']}  batch={spec['batch-size']}")
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"   exp: {exp_name}")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                          stall_minutes=args.stall_minutes)
        print(f"[v2] {spec['name']} done  rc={rc}  crashed={crashed}")
        if crashed:
            print("[v2] aborting remaining runs after crash")
            break

    if args.dry_run:
        print("\n[v2] dry-run only")


if __name__ == "__main__":
    main()
