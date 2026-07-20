"""campaign_runner_p0139_arch10.py -- 10-architecture ink-detection sweep (PHerc0139 9.3um).

CONTEXT: every architecture tried so far (v1 d4, dense_unet d4/d28, asym_attn_pool d8)
produces a SATURATED / uniform prediction on the pred-only panel = wrong-architecture
failure (identical to scroll1's failure mode). the signal EXISTS; we need an architecture
that HARNESSES it. two levers dominate:
  (1) PRESERVE SPATIAL RESOLUTION  -> per-pixel dense U-Net outputs, never global-pool
  (2) LOCALIZE THE INK IN DEPTH    -> soft depth-attention / per-voxel MIL, not hard-max
plus one data lever the user flagged:
  (3) SLIDING DEPTH WINDOWS (d4/d8) across the full 28-layer stack multiply the training
      samples ~21-25x vs a single d28 window -> "adds more data", historically helpful.

PHYSICS RECAP (why these arches, not global-pool ones):
  9.36um voxels, 113keV -> low absorption contrast. ink = a thin carbon layer at the
  papyrus SURFACE that perturbs fiber MORPHOLOGY/density at one depth band, NOT an
  in-plane brightness offset. so the discriminative feature is localized-in-depth +
  spatially-textured. any arch that spatial-avg-pools (v1, asym) destroys the morphology
  and can only answer "is this valid papyrus" -> saturates. we test spatial-preserving,
  depth-aware arches only.

  z-window math: z_range = train_d_end - train_d_start - depth + 1
    depth 28 -> 1 window  (whole stack, 1 sample per tile)
    depth 8  -> 21 windows (21x samples)
    depth 4  -> 25 windows (25x samples)

TESTS
  t01 dense_unet_depth  d28  warm-init MAE  -- soft depth-attention + per-pixel (top pick)
  t02 dense_unet_depth  d28  from scratch   -- ablates the MAE prior (does it cause saturation?)
  t03 dense_unet_resenc d28  from scratch   -- residual encoder, strided-conv, better grad flow
  t04 dense_unet        d8   from scratch   -- proven arch + sliding windows (tests lever 3 vs t02)
  t05 dense_unet_depth  d8   warm-init MAE  -- best arch + more data (levers 1+2+3 combined)
  t06 dense_unet_lap    d8   from scratch   -- fixed Laplacian edge channel: ink = morphological edge
  t07 dense_unet_multiscale d8 from scratch -- fine+dilated stem: boundary texture + broader context
  t08 dense_unet_res_attn   d8 from scratch -- residual enc + attn gates suppress blank papyrus
  t09 v14_mil_deep      d8   from scratch   -- MIL: per-voxel logits + LSE localize WHERE ink is
  t10 dense_unet_deep   d4   from scratch   -- deepest per-slice stem + thinnest/most-data window

CONFIG (per user): full-depth range, eroded ring negatives, blurred (soft) inklabels,
20 epochs, probe-int OFF (9999), eval-int==epochs (one final figure), test-int OFF (9999),
no data aug, no l1, no hard mining. split-axis y, train-split-frac 0.8055, lr 2e-4.

NOTE: dense_* arches output (B,1,H,W) -> --dense-labels --dense-soft-labels.
      v14_mil_deep outputs (B,1) tile logit -> NO --dense-labels.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SID       = 20260115000000                       # PHerc0139 w044 training scroll
EPOCHS    = 20
MAE_CKPT  = "models/mae_p0139_dense_unet.pth"    # self-supervised MAE twin (dense_unet body)

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

BASE: Dict[str, Any] = {
    "scroll-id":          SID,
    "tile-size":          32,
    "train-d-start":      0,
    "train-d-end":        28,           # full stack; windows slide within this range
    "d-start":            0,
    "d-end":              28,
    "epochs":             EPOCHS,
    "eval-int":           EPOCHS,        # dense figure fires once, at the final epoch
    "probe-int":          9999,          # probes OFF (none set up)
    "test-int":           9999,          # test figures OFF
    "ring-negatives":     True,
    "ring-label-source":  "eroded",      # eroded ring training set
    "crop-x-frac":        "0.0,1.0",
    "crop-y-frac":        "0.0,1.0",
    "split-axis":         "y",
    "train-split-frac":   0.8055,
    "batch-size":         128,           # default for d4/d8 dense; overridden to 64 for d28/mil
    "lr":                 2e-4,
    "l1-lambda":          0.0,           # no l1
    "data-aug":           0,             # no augmentation
    "num-workers":        2,
    "mask-memmap":        True,
    "no-hard-mining":     True,          # no hard mining
    "no-probe-rois":      True,
    # THERMAL SAFETY (hot summer day, laptop has crashed repeatedly): long cooldowns
    # everywhere. sleeps are wall-clock only (gpu/cpu idle) so they cost time, not compute.
    "epoch-cooldown":     90,            # sleep 90s after EVERY epoch
    "val-cooldown":       120,           # sleep 120s between train and validation each epoch
    "eval-cooldown":      600,           # sleep 600s after the heavy final eval-figure epoch
    "fig-chunk-cooldown": 600,           # sleep 600ms between spatial chunks during figure inference
}

# seconds to idle between training runs so the machine fully cools before the next arch
INTER_RUN_COOLDOWN_SECS = 420

# soft (blurred) dense labels apply to every dense_* arch (per-pixel output)
DENSE = {"dense-labels": True, "dense-soft-labels": True}

RUN_SPECS: List[Dict[str, Any]] = [
    # -- tests 1-3: full-depth (d28) spatial-preserving + depth-aware --
    {"name": "t01_dense_unet_depth_d28_warminit", "arch": "dense_unet_depth",
     "depth": 28, "batch-size": 64, "init-weights": MAE_CKPT, **DENSE},
    {"name": "t02_dense_unet_depth_d28_scratch",  "arch": "dense_unet_depth",
     "depth": 28, "batch-size": 64, **DENSE},
    {"name": "t03_dense_unet_resenc_d28_scratch", "arch": "dense_unet_resenc",
     "depth": 28, "batch-size": 64, **DENSE},

    # -- tests 4-10: sliding depth windows (d8/d4) -> more data --
    {"name": "t04_dense_unet_d8_scratch",         "arch": "dense_unet",
     "depth": 8, **DENSE},
    {"name": "t05_dense_unet_depth_d8_warminit",  "arch": "dense_unet_depth",
     "depth": 8, "init-weights": MAE_CKPT, **DENSE},
    {"name": "t06_dense_unet_lap_d8_scratch",     "arch": "dense_unet_lap",
     "depth": 8, **DENSE},
    {"name": "t07_dense_unet_multiscale_d8_scratch", "arch": "dense_unet_multiscale",
     "depth": 8, **DENSE},
    {"name": "t08_dense_unet_res_attn_d8_scratch", "arch": "dense_unet_res_attn",
     "depth": 8, **DENSE},
    {"name": "t09_v14_mil_deep_d8_scratch",       "arch": "v14_mil_deep",
     "depth": 8, "batch-size": 64},   # tile output -> NO dense-labels
    {"name": "t10_dense_unet_deep_d4_scratch",    "arch": "dense_unet_deep",
     "depth": 4, **DENSE},
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


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=180.0):
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
    ap = argparse.ArgumentParser(description="10-architecture 9.3um ink sweep (PHerc0139)")
    ap.add_argument("--campaign-id",   type=str,   default="p0139_arch10_2026_07_15")
    ap.add_argument("--python-exe",    type=str,   default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--only",          type=str,   default=None,
                    help="run only specs whose name contains this substring (e.g. t05)")
    ap.add_argument("--stall-minutes", type=float, default=180.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_p0139_full_depth"
    runs_dir.mkdir(exist_ok=True)
    log_dir   = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    print("\n" + "=" * 78)
    print("[arch10] PHerc0139 9.3um -- 10-architecture spatial-preserving + depth-aware sweep")
    print("=" * 78)

    specs = RUN_SPECS
    if args.only:
        specs = [s for s in RUN_SPECS if args.only in s["name"]]
        if not specs:
            print(f"[ABORT] --only '{args.only}' matched no spec"); return

    results: List[str] = []
    for i, spec in enumerate(specs, 1):
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"\n[{i}/{len(specs)}] exp: {exp_name}  (arch={spec['arch']}, depth={spec['depth']})")
        iw = spec.get("init-weights")
        print(f"   init-weights: {iw + ' (partial MAE transfer, strict=False)' if iw else 'none (from scratch)'}")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path, args.stall_minutes)
        status = "OK" if rc == 0 and not crashed else f"FAIL(rc={rc},crashed={crashed})"
        results.append(f"   {exp_name}: {status}")
        print(f"[arch10] done  {exp_name}  rc={rc}  crashed={crashed}")

        # thermal cooldown between runs (skip after the last one)
        if i < len(specs) and INTER_RUN_COOLDOWN_SECS > 0:
            print(f"[COOLDOWN] inter-run pause {INTER_RUN_COOLDOWN_SECS}s before next arch...")
            time.sleep(INTER_RUN_COOLDOWN_SECS)

    if args.dry_run:
        print("\n[arch10] dry-run only (no training launched)")
    else:
        print("\n" + "=" * 78)
        print("[arch10] campaign complete -- summary")
        print("=" * 78)
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
