"""campaign_runner_p0139_full_depth.py — full-depth (28-layer) architecture sweep.

MOTIVATION: depth-4 windows on PHerc0139 9.3um/113keV returned val AUC~0.47 for both
dense_unet and v1 (chance). hypothesis: at this energy, the ink signal is a thin
density peak at the papyrus-ink INTERFACE — a low-frequency depth-profile feature that
a depth-4 window misses if it doesn't happen to land at the right layer. feeding the
full 28-layer stack (all depths at once) lets the model see the complete profile.

t01: dense_unet_depth @ depth 28 — soft depth-attention + per-pixel output.
     warm-started from MAE (per_slice + full U-Net body transfer; depth_mix+score fresh).
t02: v12_asym_attn_pool @ depth 28 — soft depth-attention, tile-level output, from scratch.
     (MAE arch is incompatible with asym_pool; no warm-start.)

NOTES:
  - depth=28 -> z_range_size=max(0,28-0-28+1)=1 -> only ONE depth window (d=0).
    this IS the whole stack; the model sees the full profile every sample.
  - tile caps not needed (training set is small with one z-window; fast epochs).
  - scroll3-id wired to PHerc0191 patch (20260715114436) so test_int=5 shows us
    the model's response on the prize scroll every 5 epochs without a separate run.
  - dense_unet_depth outputs (B,1,H,W) -> use --dense-labels (a1_blur config).
  - v12_asym_attn_pool outputs (B,1) -> NO --dense-labels.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SID           = 20260115000000    # PHerc0139 w044 training scroll
TEST_SCROLL3  = 20260715114436    # PHerc0191 96-layer inference target
EPOCHS        = 20
MAE_CKPT      = "models/ink_p0139_9um_dense_unet_final.pth"

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

BASE: Dict[str, Any] = {
    "scroll-id":         SID,
    "tile-size":         32,
    "train-d-start":     0,
    "train-d-end":       28,
    "d-start":           0,
    "d-end":             28,
    "epochs":            EPOCHS,
    "eval-int":          EPOCHS,        # dense eval fires once, at the final epoch (== epochs)
    "probe-int":         9999,          # no probes (none set up)
    "test-int":          9999,          # test figures OFF (won't trigger this run)
    "ring-negatives":    True,
    "ring-label-source": "eroded",
    "crop-x-frac":       "0.0,1.0",
    "crop-y-frac":       "0.0,1.0",
    "split-axis":        "y",
    "train-split-frac":  0.8055,
    "batch-size":        256,          # 1024 OOMs at depth=8 (asym Conv3d 128→256 @ 1024B)
    "lr":                2e-4,
    "l1-lambda":         0.0,
    "data-aug":          0,
    "num-workers":       2,
    "mask-memmap":       True,
    "no-hard-mining":    True,
    "no-probe-rois":     True,
    "eval-cooldown":     240,
    "val-cooldown":      90,
    "fig-chunk-cooldown": 200,
}

RUN_SPECS: List[Dict[str, Any]] = [
    {
        "name":             "t01_asym_attn_pool_d8",
        "arch":             "v12_asym_attn_pool",  # C14/C15 winner; soft depth-attn, tile output
        "depth":            8,                     # campaign-15 depth (config default it ran at)
        # no init-weights (arch incompatible w/ dense_unet MAE); no --dense-labels (tile output)
    },
    {
        "name":             "t02_dense_unet_d28_warminit",
        "arch":             "dense_unet",          # full-depth: 28 slices -> per-slice stem -> depth-max
        "depth":            28,                    # entire stack fed at once (one window)
        "batch-size":       64,                    # depth-28 stems are memory-heavy: (B,16,28,32,32) tensor
        "init-weights":     MAE_CKPT,              # FULL match to MAE (same arch) -> complete warm-start
        "dense-labels":     True,
        "dense-soft-labels": True,
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
    ap = argparse.ArgumentParser(description="full-depth 9.3um architecture sweep")
    ap.add_argument("--campaign-id",   type=str,   default="p0139_9um_fulldepth_2026_07_15")
    ap.add_argument("--python-exe",    type=str,   default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--only",          type=str,   default=None,
                    help="run only the spec whose name contains this substring (e.g. t01)")
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
    # no VESUVIUS_MAX_SAMPLES_PER_EPOCH: epochs are fast with one z-window

    print("\n" + "=" * 78)
    print("[full-depth] 9.3um PHerc0139: asym_attn_pool(d8) then dense_unet(d28, warm-init)")
    print("=" * 78)

    specs = RUN_SPECS
    if args.only:
        specs = [s for s in RUN_SPECS if args.only in s["name"]]
        if not specs:
            print(f"[ABORT] --only '{args.only}' matched no spec"); return

    for spec in specs:
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"\n   exp: {exp_name}  (arch={spec['arch']})")
        iw = spec.get("init-weights")
        if iw:
            print(f"   init-weights: {iw} (partial MAE transfer, strict=False)")
        else:
            print(f"   init-weights: none (from scratch)")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path, args.stall_minutes)
        print(f"[full-depth] done  {exp_name}  rc={rc}  crashed={crashed}")

    if args.dry_run:
        print("\n[full-depth] dry-run only")


if __name__ == "__main__":
    main()
