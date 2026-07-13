"""campaign_runner_scroll4_79um_dense.py — dense U-Net probes on the 7.9um scroll4 w023.

four sequential runs, all targeting the same 7.9um/53keV zarr (20240304161941) with:
  - tile_size=32  (native resolution; NOT the 106px teacher scale)
  - dense_labels=True (per-pixel BCE against the warped eroded ink labels)
  - no l1, no dropout, no augmentation — clean capability test
  - probe figure every 5 epochs (auto-detected ink-rich window)
  - full eval figure at epoch 15 only (eval_int == epochs)
  - all runs log to the SAME runs_scroll4_79um tensorboard directory

runs (in order):
  t01_dense_unet            — per-slice stem, depth-max collapse, no depth info
  t02_dense_unet_depth      — per-slice stem + depth-mixing + learned depth attention (depth=64)
  t03_dense_unet_depth_blur — same arch + soft/blurred inklabels (dilated 1px, blurred sigma=2)
  t04_dense_unet_depth_mod  — same arch with depth=16, z-stepped by 8 (0-16, 8-24, ..., 48-64)

VRAM budget: 24 GB   RAM budget: 32 GB
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCROLL4_79_ID = 20240304161941   # w023 flipped 7.91um/53keV full sheet
EPOCHS        = 30

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

# ---- shared baseline -------------------------------------------------------
# everything that is identical across all four runs.
# the key goal: no regularisation, no augmentation, no hard mining
# so we can see whether the 7.9um signal is LEARNABLE at all.
BASE: Dict[str, Any] = {
    # scroll + tile setup
    "scroll-id":        SCROLL4_79_ID,
    "tile-size":        32,
    # full depth for training: the 7.9um ink band is unknown, sweep everything
    "train-d-start":    0,
    "train-d-end":      64,
    # eval / inference figure also sweeps full depth
    "d-start":          0,
    "d-end":            64,
    # epoch / logging schedule
    "epochs":           EPOCHS,
    "eval-int":         15,          # full eval at ep15 and ep30
    "probe-int":        5,           # cheap probe ROIs every 5 epochs
    "test-int":         30,          # test figures (scroll2 + scroll3) at ep30 only
    # skip the huge training-scroll test figure; scroll2 + scroll3 are the targets.
    # test_scroll2_only skips training-scroll Test figure but KEEPS both scroll2 and scroll3.
    "test-scroll2-only": True,
    # optimiser — no regularisation
    "lr":               2e-4,
    "l1-lambda":        0.0,
    # dropout — fully off so the model shows peak raw capacity
    "conv1-drop":       0.0,
    "conv2-drop":       0.0,
    "fc1-drop":         0.0,
    "fc2-drop":         0.0,
    # augmentation — all off
    "data-aug":         0,
    "channel-mixing-prob": 0.0,
    # ring negatives: eroded boundary (consistent with every winning campaign run)
    "ring-negatives":   True,
    "ring-label-source": "eroded",
    # region crop: right 40% x, top 75% y  →  y-split 75/25 within that
    "crop-x-frac":      "0.6,1.0",
    "crop-y-frac":      "0.0,0.75",
    "split-axis":       "y",
    "train-split-frac": 0.75,
    # dense per-pixel supervision (the point of this entire campaign)
    "dense-labels":     True,
    # misc — num-workers is set per-run (each run has different tile/batch sizes)
    "mask-memmap":      True,
    "no-hard-mining":   True,
    "no-probe-rois":    True,        # old scroll1 probe ROIs off; dense probe auto-activates
    "ranking-lambda":   0.0,
    # test-int > epochs -> test figures never fire (we only want train + probe + eval)
}

# ---- per-run overrides -----------------------------------------------------
# depth and batch size vary by architecture:
#   dense_unet  (depth=8):  many z-windows across full volume; large batch
#   dense_unet_depth (depth=64): one full-volume z-window per tile; small batch
#   dense_unet_depth_mod (depth=16): stepped z-windows (0-16, 8-24, …); medium batch
#
# batch sizes are conservative for 24 GB VRAM: the dense eval figure runs one
# forward pass at chunk size ~760x760 which briefly peaks ~3-4 GB for depth=64.

RUN_SPECS: List[Dict[str, Any]] = [
    {
        "name":       "t01_dense_unet_30ep",
        "arch":       "dense_unet",
        "depth":      8,
        "batch-size": 1024,
        "dense-soft-labels": True,
        "num-workers": 2,
        "eval-cooldown": 90,    # post probe/eval epoch rest
        "val-cooldown":  30,    # train -> val pause every epoch
        "fig-chunk-cooldown": 50,  # 50ms between eval figure chunks
    },
]


# ---- soft-label generation -------------------------------------------------

_SOFT_LABEL_SCRIPT = """\
import sys, pathlib
import cv2, numpy as np

scroll_id = int(sys.argv[1])
repo_root = pathlib.Path(sys.argv[2])

src     = repo_root / "eroded_inklabels" / f"{scroll_id}.png"
dst_dir = repo_root / "soft_inklabels"
dst_dir.mkdir(exist_ok=True)
dst     = dst_dir / f"{scroll_id}.png"

if dst.exists():
    print(f"[soft_labels] {dst.name} already exists, skipping")
    sys.exit(0)

img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"[soft_labels] ERROR: could not read {src}")
    sys.exit(1)

# dilate 1px (3x3 ellipse kernel) then gaussian blur sigma=5
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated = cv2.dilate(img, kernel, iterations=1)
blurred = cv2.GaussianBlur(dilated, (0, 0), sigmaX=15.0, sigmaY=15.0)
cv2.imwrite(str(dst), blurred)
print(f"[soft_labels] generated {dst}  mean={blurred.mean():.2f}  max={blurred.max()}")
"""


def generate_soft_labels(scroll_id: int, repo_root: Path,
                         python_exe: str = sys.executable) -> bool:
    """generate soft_inklabels/<id>.png via the training Python (which has cv2).

    runs a small inline script as a subprocess so the campaign runner itself
    doesn't need cv2 installed in whatever shell environment it lives in.
    output: eroded labels dilated 1px + gaussian blur sigma=2 -> uint8 PNG [0,255].
    """
    dst = repo_root / "soft_inklabels" / f"{scroll_id}.png"
    if dst.exists():
        print(f"[soft_labels] {dst.name} already exists, skipping")
        return True

    result = subprocess.run(
        [python_exe, "-c", _SOFT_LABEL_SCRIPT, str(scroll_id), str(repo_root)],
        capture_output=True, text=True, cwd=str(repo_root)
    )
    for line in (result.stdout + result.stderr).splitlines():
        print(f"  {line}")
    if result.returncode != 0:
        print(f"[soft_labels] FAILED (rc={result.returncode})")
        return False
    return True


# ---- CLI builder -----------------------------------------------------------

def dict_to_cli_args(d: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in d.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            # False booleans that are flags: skip (they are store_true / default-off)
        else:
            args.extend([f"--{key}", str(value)])
    return args


def build_cmd(python_exe: str, runs_dir: Path, campaign_id: str,
              spec: Dict[str, Any]) -> tuple[list[str], str]:
    merged = dict(BASE)
    # apply run-specific overrides
    for k, v in spec.items():
        if k != "name":
            merged[k] = v

    exp_name = f"cmp_{campaign_id}_{spec['name']}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


# ---- runner ----------------------------------------------------------------

def labels_exist(scroll_id: int) -> bool:
    return (Path("eroded_inklabels") / f"{scroll_id}.png").exists()


def run_with_monitoring(cmd: list, repo_root: Path, env: dict,
                        log_path: Path, stall_minutes: float = 60.0):
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
                        last_epoch = ep
                        last_progress = time.time()
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
    ap = argparse.ArgumentParser(
        description="dense U-Net capability probe on the 7.9um scroll4 w023")
    ap.add_argument("--campaign-id",    type=str,   default="scroll4_79um_2026_07_09")
    ap.add_argument("--python-exe",     type=str,   default=sys.executable)
    ap.add_argument("--dry-run",        action="store_true")
    ap.add_argument("--stall-minutes",  type=float, default=60.0)
    ap.add_argument("--run",            type=str,   default=None,
                    help="run only a specific run by name (e.g. t01_dense_unet)")
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
    print(f"[scroll4 79um dense] id={SCROLL4_79_ID}  (w023 flipped 7.91um/53keV)")
    print(f"  goal: determine whether dense U-Net can learn ink on 7.9um scroll4")
    print(f"  log dir: {runs_dir}")
    print("=" * 78)

    if not labels_exist(SCROLL4_79_ID):
        print(f"[ABORT] eroded_inklabels/{SCROLL4_79_ID}.png not found — bake labels first")
        return

    specs_to_run = RUN_SPECS
    if args.run:
        specs_to_run = [s for s in RUN_SPECS if s["name"] == args.run]
        if not specs_to_run:
            print(f"[ABORT] unknown run name '{args.run}'; valid: {[s['name'] for s in RUN_SPECS]}")
            return

    for spec in specs_to_run:
        print(f"\n{'='*78}")
        print(f"[scroll4 79um] run: {spec['name']}  arch={spec['arch']}  depth={spec['depth']}")

        # run 3 requires soft labels — generate them now if missing
        if spec.get("dense-soft-labels"):
            print("[scroll4 79um] generating soft labels for blurred-target run...")
            ok = generate_soft_labels(SCROLL4_79_ID, repo_root, python_exe=args.python_exe)
            if not ok:
                print("[scroll4 79um] soft label generation failed — skipping this run")
                continue

        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"   exp: {exp_name}")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")

        if args.dry_run:
            continue

        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                          stall_minutes=args.stall_minutes)
        print(f"[scroll4 79um] {spec['name']} done  rc={rc}  crashed={crashed}")
        if crashed:
            print("[scroll4 79um] aborting remaining runs after crash")
            break

    if args.dry_run:
        print("\n[scroll4 79um] dry-run only — no processes were launched")


if __name__ == "__main__":
    main()
