"""campaign_runner_scroll4_2p4um.py — train on the NATIVE 2.399um (78keV) scroll4 data.

trains two models, in order, on the native no-warp 2.4um volume + native
ink-detection labels (scroll 20251217075048):
  1. a1_blur winner        -> arch dense_unet          (BatchNorm, blur soft target)
  2. researcher's copy      -> arch dense_unet_resenc   (residual encoder)

DATA (built by build_scroll_zarr.py s3patch, NO warp):
  - ves_zarrs2/20251217075048.zarr        shape (109, 32512, 41344) uint8
  - inklabels/20251217075048.png          native ink-detection labels (padded to 41344)
  - eroded_inklabels/20251217075048.png    eroded (ring source)
  - soft_inklabels/20251217075048.png      dilate 1px + gaussian blur sigma=15 (a1_blur target)
  - masks/20251217075048.png               valid-data mask (89.8% valid)

SECTION: the built zarr already IS the specified section — build_scroll_zarr cropped
native x[61056:102400] y[0:32512], i.e. exactly the a1_blur crop region (right ~40% x,
top ~75% y of the full native volume). so training uses the FULL zarr (crop 0..1) and
applies the same 75/25 y-split for train/val.

DEPTH: native volume is 109 deep (we care about the depth dimension), so the training
window spans the full depth d[0:109] with depth-8 sub-windows (a1_blur used d[0:64] on
the 64-deep 7.9um teacher; here we use all 109 native slices).

reproduces the winning a1_blur configuration otherwise EXACTLY:
  - dense_soft_labels=True, ring_negatives=True, ring_label_source='eroded'
  - tile=32, depth=8, batch=1024, lr=2e-4
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCROLL_2P4_ID = 20251217075048
EPOCHS        = 20

# native 2.4um is disk-I/O bound (~5 s/it streaming the 293 GB uncompressed zarr).
# a full epoch is ~6533 iters (~9 hrs). cap tiles/epoch so each epoch is ~2 hrs; each
# epoch draws a FRESH random subset, so coverage stays broad over 20 epochs. full depth
# d[0:109] is preserved. ~1.5M tiles / 1024 batch = ~1465 iters/epoch.
MAX_SAMPLES_PER_EPOCH = 1_500_000

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

BASE: Dict[str, Any] = {
    "scroll-id":          SCROLL_2P4_ID,
    "tile-size":          32,
    "depth":              8,
    "train-d-start":      0,
    "train-d-end":        109,        # full native depth
    "d-start":            0,
    "d-end":              109,        # full native depth
    "epochs":             EPOCHS,
    "eval-int":           EPOCHS,       # full-region eval only once at the end (expensive)
    "probe-int":          5,            # cheap 512px ROI probes every 5 epochs
    "test-int":           1000,        # never fires (no cross-scroll test)
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
    "crop-x-frac":        "0.0,1.0",   # crop already baked into the zarr
    "crop-y-frac":        "0.0,1.0",
    "split-axis":         "y",
    "train-split-frac":   0.75,        # top 75% train, bottom 25% val
    "dense-labels":       True,
    "dense-soft-labels":  True,         # blurred per-pixel target (a1_blur)
    "batch-size":         1024,
    "num-workers":        2,
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    "ranking-lambda":     0.0,
    "eval-cooldown":      120,
    "val-cooldown":       45,
    "fig-chunk-cooldown": 100,
}

RUN_SPECS: List[Dict[str, Any]] = [
    {"name": "t01_a1_blur_dense_unet",     "arch": "dense_unet"},          # winning model
    {"name": "t02_researcher_resenc",       "arch": "dense_unet_resenc"},   # researcher's copy
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
    ap = argparse.ArgumentParser(description="train a1_blur + researcher-resenc on native 2.4um")
    ap.add_argument("--campaign-id",   type=str,   default="scroll4_2p4um_2026_07_15")
    ap.add_argument("--python-exe",    type=str,   default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--only",          type=str,   default=None,
                    help="run only the spec whose name contains this substring")
    ap.add_argument("--stall-minutes", type=float, default=180.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_scroll4_2p4um"
    runs_dir.mkdir(exist_ok=True)
    log_dir   = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env["VESUVIUS_MAX_SAMPLES_PER_EPOCH"] = str(MAX_SAMPLES_PER_EPOCH)

    print("\n" + "=" * 78)
    print("[2p4um] NATIVE 2.4um RUN — a1_blur (dense_unet) then researcher (dense_unet_resenc)")
    print(f"  scroll {SCROLL_2P4_ID} | {EPOCHS} epochs each | full depth d[0:109] | ring eroded")
    print("=" * 78)

    # verify data + labels exist
    import zarr
    zpath = repo_root / "ves_zarrs2" / f"{SCROLL_2P4_ID}.zarr"
    if not zpath.exists():
        # zarr-path may differ; the training reads via its own config, just warn
        print(f"[warn] {zpath} not found under repo; train.py uses its configured zarr-path")
    for sub in ("inklabels", "eroded_inklabels", "soft_inklabels", "masks"):
        p = Path(sub) / f"{SCROLL_2P4_ID}.png"
        if not p.exists():
            print(f"[ABORT] {p} not found"); return
        print(f"  using {p}")

    specs = RUN_SPECS
    if args.only:
        specs = [s for s in RUN_SPECS if args.only in s["name"]]
        if not specs:
            print(f"[ABORT] --only '{args.only}' matched no spec"); return

    for spec in specs:
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"\n   exp: {exp_name}  (arch={spec['arch']})")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path, args.stall_minutes)
        print(f"[2p4um] done  {exp_name}  rc={rc}  crashed={crashed}")

    if args.dry_run:
        print("\n[2p4um] dry-run only")


if __name__ == "__main__":
    main()
