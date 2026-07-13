"""campaign_runner_scroll4_79um_v4.py — weekend campaign: 6 runs, tile=32, 15 epochs.

standard baseline: tile=32, depth=8, InstanceNorm, soft labels σ=15, dense_unet.
best result so far: 0.5548 AUC (15ep, hard labels, BatchNorm).

six hypotheses, each isolating one variable:

  t01_d4      depth=4,  tile=32: z_step=2 → ~29 z-windows/tile (2× data vs depth=8)
  t02_t24     depth=8,  tile=24: smaller tile → more tiles at boundaries, larger ring count
  t03_res     depth=8,  tile=32: dense_unet_res_attn (residual blocks + attention gates)
  t04_wide    depth=8,  tile=32: dense_unet_wide (32 stem channels vs 16 — richer texture)
  t05_ms      depth=8,  tile=32: dense_unet_multiscale (fine+coarse dilated per-slice stem)
  t06_d4_res  depth=4,  tile=32: dense_unet_res_attn + depth=4 (best data × best arch)

all runs log to runs_scroll4_79um (same tensorboard dir as all previous campaigns).
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

# new standard baseline: InstanceNorm + soft labels already baked into all dense archs
BASE: Dict[str, Any] = {
    "scroll-id":          SCROLL4_79_ID,
    "tile-size":          32,
    "train-d-start":      0,
    "train-d-end":        64,
    "d-start":            0,
    "d-end":              64,
    "epochs":             EPOCHS,
    "eval-int":           EPOCHS,      # eval figure once at the very end
    "probe-int":          5,
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
    "dense-soft-labels":  True,        # sigma=15 blurred labels throughout
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    "ranking-lambda":     0.0,
    "eval-cooldown":      180,   # 3 min rest after probe/eval epoch
    "val-cooldown":       60,    # 60s between train and validation each epoch
    "fig-chunk-cooldown": 150,   # 150ms between eval figure chunks (heavy eval at 20ep)
    "num-workers":        2,
}

RUN_SPECS: List[Dict[str, Any]] = [
    {   # t01: depth=4 — 2× z-windows per epoch via z_step=2 (more data)
        "name":       "t01_d4",
        "arch":       "dense_unet",
        "depth":      4,
        "batch-size": 1024,   # 2048 hits WinError 1455 (shared mem commit limit); 1024 safe
    },
    {   # t02: tile=24 — higher boundary fraction per tile
        "name":       "t02_t24",
        "arch":       "dense_unet",
        "tile-size":  24,
        "depth":      8,
        "batch-size": 1024,
    },
    {   # t03: residual encoder + attention gates on skips
        "name":       "t03_res_attn",
        "arch":       "dense_unet_res_attn",
        "depth":      8,
        "batch-size": 1024,
    },
    {   # t04: 2× wider stem (32 channels) + wider decoder
        "name":       "t04_wide",
        "arch":       "dense_unet_wide",
        "depth":      8,
        "batch-size": 512,
    },
    {   # t05: dual-scale per-slice stem (fine 3×3 + dilated 3×3)
        "name":       "t05_ms",
        "arch":       "dense_unet_multiscale",
        "depth":      8,
        "batch-size": 512,
    },
    {   # t06: residual+attn × depth=4 (best arch + 2× data)
        "name":       "t06_d4_res",
        "arch":       "dense_unet_res_attn",
        "depth":      4,
        "batch-size": 1024,
    },
    {   # t07: asymmetric deep encoder (4 down) + shallow bilinear decoder (2 up).
        # more encoding capacity focused on the texture where signal lives;
        # decoder relies on bilinear upsampling rather than learned transposed-conv.
        "name":       "t07_asym",
        "arch":       "dense_unet_asym",
        "depth":      8,
        "batch-size": 512,
    },
    {   # t08: Laplacian edge pre-emphasis concatenated to raw per-slice input.
        # provides the stem with an explicit second-derivative edge channel per slice
        # targeting the ink-papyrus morphological boundary transition.
        "name":       "t08_lap",
        "arch":       "dense_unet_lap",
        "depth":      8,
        "batch-size": 1024,
    },
    {   # t09: deeper per-slice stem (4 conv layers instead of 2).
        # adds depth (not breadth) to the primary texture-detection component.
        # stem output is 32-channel; same decoder as standard dense_unet.
        "name":       "t09_deep",
        "arch":       "dense_unet_deep",
        "depth":      8,
        "batch-size": 1024,
    },
    {   # t10: multiscale stem + depth=4 (richer per-slice features + more data windows).
        "name":       "t10_ms_d4",
        "arch":       "dense_unet_multiscale",
        "depth":      4,
        "batch-size": 512,
    },
    {   # t11: Laplacian stem + depth=4 (edge emphasis + more z-windows per epoch).
        # combines the explicit boundary prior with the data-quantity advantage of depth=4.
        "name":       "t11_lap_d4",
        "arch":       "dense_unet_lap",
        "depth":      4,
        "batch-size": 1024,  # 2048 hits WinError 1455 shared memory limit
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
    ap = argparse.ArgumentParser(description="scroll4 79um v4: weekend 6-run campaign")
    ap.add_argument("--campaign-id",   type=str,   default="scroll4_79um_v4_2026_07_11")
    ap.add_argument("--python-exe",    type=str,   default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=120.0)
    ap.add_argument("--run",           type=str,   default=None,
                    help="run only a specific spec by name (e.g. t01_d4)")
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
    print(f"[scroll4 79um v4] weekend campaign  id={SCROLL4_79_ID}")
    print(f"  {EPOCHS} epochs | InstanceNorm | soft-labels σ=15 | tile=32 baseline")
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
        tile = spec.get("tile-size", BASE.get("tile-size", 32))
        print(f"\n{'='*78}")
        print(f"[v4] {spec['name']}  arch={spec['arch']}  tile={tile}  "
              f"depth={spec['depth']}  batch={spec['batch-size']}")
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"   exp: {exp_name}")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                          stall_minutes=args.stall_minutes)
        print(f"[v4] {spec['name']} done  rc={rc}  crashed={crashed}")
        if crashed:
            print("[v4] aborting remaining runs after crash")
            break

    if args.dry_run:
        print("\n[v4] dry-run only")


if __name__ == "__main__":
    main()
