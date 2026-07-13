"""campaign_runner_scroll4_79um_v6.py — chained campaign after v5.

runs sequentially after the v5 30-epoch BatchNorm baseline:

PHASE A: label-config comparison (20 epochs each)
  A1: dense_unet, BatchNorm, BLUR labels (soft labels sigma=15)

PHASE B: architecture search, BatchNorm throughout (20 epochs each)
  B1: dense_unet depth=4   (2x z-windows via z_step=2)
  B2: dense_unet tile=24   (higher boundary fraction)
  B3: dense_unet_res_attn  (residual + attention gates)
  B4: dense_unet_wide      (32 stem channels)
  B5: dense_unet_multiscale (fine + coarse dilated stem)
  B6: dense_unet_res_attn depth=4 (best arch x best data)

PHASE C: novel archs (20 epochs each), using the better label config from A vs v5
  C1: dense_unet_asym   (4-down encoder + 2-up bilinear decoder)
  C2: dense_unet_lap    (Laplacian edge pre-emphasis)
  C3: dense_unet_coord  (CoordConv within-tile position channels)

After phase B, a comparison step reads the epoch-20 valid F1 from v5 (BatchNorm, hard)
vs A1 (BatchNorm, blur) and picks the better label config for phase C.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCROLL4_79_ID = 20240304161941
EPOCHS        = 20

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

# shared base — BatchNorm is now default in all dense archs
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
    "test-int":           60,         # never fires
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

PHASE_A = [
    {
        "name":              "a1_blur",
        "dense-soft-labels": True,   # sigma=15 blurred labels
    },
]

PHASE_B = [
    {"name": "b1_d4",     "depth": 4,  "batch-size": 1024},
    {"name": "b2_t24",    "tile-size": 24},
    {"name": "b3_res",    "arch": "dense_unet_res_attn"},
    {"name": "b4_wide",   "arch": "dense_unet_wide",        "batch-size": 512},
    {"name": "b5_ms",     "arch": "dense_unet_multiscale",  "batch-size": 512},
    {"name": "b6_d4_res", "arch": "dense_unet_res_attn",    "depth": 4, "batch-size": 1024},
]

# phase C specs — label config inherited from the comparison step
PHASE_C_ARCH_ONLY = [
    {"name": "c1_asym",  "arch": "dense_unet_asym",  "batch-size": 512},
    {"name": "c2_lap",   "arch": "dense_unet_lap"},
]


# ── helpers ──────────────────────────────────────────────────────────────────

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
              spec: Dict[str, Any]) -> tuple[list, str]:
    merged = dict(BASE)
    for k, v in spec.items():
        if k != "name":
            merged[k] = v
    exp_name = f"cmp_{campaign_id}_{spec['name']}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def read_best_f1(log_path: Path) -> float:
    """scan a training log for the highest 'New best F1' line and return that F1."""
    best = 0.0
    if not log_path.exists():
        return best
    pat = re.compile(r"New best F1 model saved.*Val F1:\s*([\d.]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            best = max(best, float(m.group(1)))
    return best


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


def run_spec(spec, python_exe, runs_dir, log_dir, campaign_id, env, repo_root,
             stall_minutes, dry_run) -> tuple[str, bool]:
    """build and optionally launch one training run. returns (exp_name, crashed)."""
    cmd, exp_name = build_cmd(python_exe, runs_dir, campaign_id, spec)
    tile = spec.get("tile-size", BASE.get("tile-size", 32))
    print(f"\n{'='*78}\n[v6] {spec['name']}  arch={spec.get('arch', BASE['arch'])}  "
          f"tile={tile}  depth={spec.get('depth', BASE.get('depth',8))}  "
          f"blur={'--dense-soft-labels' in cmd}")
    print(f"   exp: {exp_name}")
    if dry_run:
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        return exp_name, False
    log_path = log_dir / f"{exp_name}.log"
    rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes)
    print(f"[v6] {spec['name']} done  rc={rc}  crashed={crashed}")
    return exp_name, crashed


def main():
    ap = argparse.ArgumentParser(description="v6: chained campaign after v5")
    ap.add_argument("--campaign-id",   type=str,   default="scroll4_79um_v6_2026_07_12")
    ap.add_argument("--python-exe",    type=str,   default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=120.0)
    ap.add_argument("--v5-log",        type=str,   default=None,
                    help="path to v5 log file for F1 comparison (auto-detected if omitted)")
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
    print("[v6] chained campaign: A(blur baseline) → B(arch search) → compare → C(novel)")
    print("=" * 78)

    if not (Path("eroded_inklabels") / f"{SCROLL4_79_ID}.png").exists():
        print("[ABORT] eroded_inklabels not found"); return
    if not (Path("soft_inklabels") / f"{SCROLL4_79_ID}.png").exists():
        print("[ABORT] soft_inklabels not found"); return

    # ── PHASE A: blur label baseline ─────────────────────────────────────────
    print("\n=== PHASE A: blur label baseline (20 epochs) ===")
    a1_name, a1_crashed = run_spec(
        PHASE_A[0], args.python_exe, runs_dir, log_dir,
        args.campaign_id, env, repo_root, args.stall_minutes, args.dry_run)
    if a1_crashed:
        print("[v6] phase A crashed — aborting"); return

    # ── COMPARISON: blur (A1) vs non-blur (v5) — done IMMEDIATELY after A ──
    # the winner propagates to ALL remaining runs (Phase B and Phase C).
    print("\n=== COMPARISON: blur (A1) vs non-blur (v5) ===")
    a1_log = log_dir / f"cmp_{args.campaign_id}_a1_blur.log"
    a1_f1  = read_best_f1(a1_log)

    v5_log_path = args.v5_log
    if not v5_log_path:
        candidates = sorted(log_dir.glob("*t01_original_30ep*.log"), key=lambda p: p.stat().st_mtime)
        v5_log_path = str(candidates[-1]) if candidates else None
    v5_f1 = read_best_f1(Path(v5_log_path)) if v5_log_path else 0.0

    print(f"  v5 (hard labels) best F1 : {v5_f1:.4f}  log={v5_log_path}")
    print(f"  A1 (blur labels) best F1 : {a1_f1:.4f}  log={a1_log}")

    use_blur = True   # user override: blur confirmed winner visually
    winner   = "BLUR"
    print(f"  WINNER: BLUR (user override — visually confirmed)  all Phase B+C runs will use BLUR labels")

    # ── PHASE B: architecture search with winning label config ────────────────
    print(f"\n=== PHASE B: architecture search (20 epochs, {winner} labels) ===")
    b_crashed = False
    for spec in PHASE_B:
        full_spec = dict(spec)
        if use_blur:
            full_spec["dense-soft-labels"] = True
        _, crashed = run_spec(
            full_spec, args.python_exe, runs_dir, log_dir,
            args.campaign_id, env, repo_root, args.stall_minutes, args.dry_run)
        if crashed:
            print(f"[v6] phase B crashed at {spec['name']} — aborting"); b_crashed = True; break
    if b_crashed:
        return

    # ── PHASE C: novel archs with winning label config ────────────────────────
    print(f"\n=== PHASE C: novel archs (20 epochs, {winner} labels) ===")
    for spec in PHASE_C_ARCH_ONLY:
        full_spec = dict(spec)
        if use_blur:
            full_spec["dense-soft-labels"] = True
        _, crashed = run_spec(
            full_spec, args.python_exe, runs_dir, log_dir,
            args.campaign_id, env, repo_root, args.stall_minutes, args.dry_run)
        if crashed:
            print(f"[v6] phase C crashed at {spec['name']} — aborting"); return

    print("\n[v6] ALL PHASES COMPLETE")

    if args.dry_run:
        print("[v6] dry-run only — no processes were launched")


if __name__ == "__main__":
    main()
