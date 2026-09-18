"""campaign_runner_p0139_arch18.py -- automated 18-arch ink sweep with early stopping.

PHerc0139 9.3um. 18 runs = 8 survivors from arch10 (both dense_unet_depth d28 dropped:
they learned nothing) + 10 NEW physics-motivated dense architectures.

AUTOMATED EARLY-STOP PROTOCOL (per user, 2026-07-15):
  every run trains 10 epochs (phase 1), saving a resumable checkpoint at epoch 10.
  then, from the phase-1 train_loss curve, we decide:
     CUT if  (a) STAGNATION: improvement from epoch 5 -> 10 is < STAGNATION_EPS
         OR  (b) NEVER-LEARNED: train_loss never dropped below LOSS_FLOOR (0.8)
     else CONTINUE: resume from the epoch-10 checkpoint for 10 more epochs (phase 2),
          rendering the final dense eval figure at the end.
  (validation is ignored for the gate; the gate is purely on training loss.)

THERMAL SAFETY (hot day, repeated crashes): long cooldowns everywhere -
  per-epoch 90s, train->val 120s, post-eval-figure 600s, fig-chunk 600ms,
  inter-phase 300s, inter-run 420s. these are idle sleeps (no compute).

reads train.py's "[METRICS] epoch=N train_loss=X ..." console line (added 2026-07-15).
resume uses train.py --save-final (phase1) then --init-weights (phase2), weights-only
(optimizer/scheduler restart; acceptable for a "does it keep improving" probe).
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SID                     = 20260115000000     # PHerc0139 w044 training scroll
MAE_CKPT                = "models/mae_p0139_dense_unet.pth"
EPOCHS_PER_PHASE        = 10
LOSS_FLOOR              = 0.8                 # never below this -> never learned
STAGNATION_EPS          = 0.02               # improvement ep5->ep10 below this -> stagnated
INTER_RUN_COOLDOWN_SECS = 420
INTER_PHASE_COOLDOWN_SECS = 300
CKPT_DIR                = "models/arch18"

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

BASE: Dict[str, Any] = {
    "scroll-id":          SID,
    "tile-size":          32,
    "train-d-start":      0,
    "train-d-end":        28,
    "d-start":            0,
    "d-end":              28,
    "probe-int":          9999,          # probes OFF
    "test-int":           9999,          # test figures OFF
    "ring-negatives":     True,
    "ring-label-source":  "eroded",
    "crop-x-frac":        "0.0,1.0",
    "crop-y-frac":        "0.0,1.0",
    "split-axis":         "y",
    "train-split-frac":   0.8055,
    "batch-size":         128,           # per-spec override for heavy archs
    "lr":                 2e-4,
    "l1-lambda":          0.0,
    "data-aug":           0,
    "num-workers":        2,
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    # thermal cooldowns (idle sleeps)
    "epoch-cooldown":     90,
    "val-cooldown":       120,
    "eval-cooldown":      600,
    "fig-chunk-cooldown": 600,
}

DENSE = {"dense-labels": True, "dense-soft-labels": True}

# 18 runs. depth 8 default (sliding windows -> ~21x more samples than d28).
RUN_SPECS: List[Dict[str, Any]] = [
    # --- 8 survivors from arch10 (dense_unet_depth d28 x2 dropped) ---
    {"name": "s01_dense_unet_resenc_d28",    "arch": "dense_unet_resenc",    "depth": 28, "batch-size": 64,  **DENSE},
    {"name": "s02_dense_unet_d8",            "arch": "dense_unet",           "depth": 8,  **DENSE},
    {"name": "s03_dense_unet_depth_d8_warm", "arch": "dense_unet_depth",     "depth": 8,  "init-weights": MAE_CKPT, **DENSE},
    {"name": "s04_dense_unet_lap_d8",        "arch": "dense_unet_lap",       "depth": 8,  **DENSE},
    {"name": "s05_dense_unet_multiscale_d8", "arch": "dense_unet_multiscale","depth": 8,  **DENSE},
    {"name": "s06_dense_unet_res_attn_d8",   "arch": "dense_unet_res_attn",  "depth": 8,  **DENSE},
    {"name": "s07_v14_mil_deep_d8",          "arch": "v14_mil_deep",         "depth": 8,  "batch-size": 64},  # TILE output
    {"name": "s08_dense_unet_deep_d4",       "arch": "dense_unet_deep",      "depth": 4,  **DENSE},
    # --- 10 NEW physics-motivated dense archs ---
    {"name": "n01_zconv1d_d8",               "arch": "dense_unet_zconv1d",   "depth": 8,  **DENSE},
    {"name": "n02_zgrad_d8",                 "arch": "dense_unet_zgrad",     "depth": 8,  **DENSE},
    {"name": "n03_zpe_attn_d8",              "arch": "dense_unet_zpe_attn",  "depth": 8,  **DENSE},
    {"name": "n04_bandsplit_d8",             "arch": "dense_unet_bandsplit", "depth": 8,  **DENSE},
    {"name": "n05_3denc_d8",                 "arch": "dense_unet_3denc",     "depth": 8,  "batch-size": 64, **DENSE},
    {"name": "n06_lcn_d8",                   "arch": "dense_unet_lcn",       "depth": 8,  **DENSE},
    {"name": "n07_gabor_d8",                 "arch": "dense_unet_gabor",     "depth": 8,  **DENSE},
    {"name": "n08_bottattn_d8",              "arch": "dense_unet_bottattn",  "depth": 8,  **DENSE},
    {"name": "n09_aspp_d8",                  "arch": "dense_unet_aspp",      "depth": 8,  **DENSE},
    {"name": "n10_hr_d8",                    "arch": "dense_unet_hr",        "depth": 8,  **DENSE},
]

_METRICS_RE = re.compile(r"\[METRICS\] epoch=(\d+) train_loss=([\d.]+)")


def dict_to_cli_args(d: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in d.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def build_cmd(python_exe, runs_dir, campaign_id, spec, phase, epochs, extra):
    """build a train.py command for one phase of one spec."""
    merged = dict(BASE)
    for k, v in spec.items():
        if k != "name":
            merged[k] = v
    merged["epochs"] = epochs
    merged.update(extra)
    exp_name = f"cmp_{campaign_id}_{spec['name']}_p{phase}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def parse_train_losses(log_path: Path) -> List[float]:
    """extract per-epoch train_loss values from a run log, in epoch order."""
    losses: Dict[int, float] = {}
    try:
        for line in open(log_path, encoding="utf-8", errors="replace"):
            m = _METRICS_RE.search(line)
            if m:
                losses[int(m.group(1))] = float(m.group(2))
    except Exception:
        return []
    return [losses[e] for e in sorted(losses)]


def decide(losses: List[float]) -> Tuple[bool, str]:
    """return (continue?, reason). gate purely on the phase-1 train_loss curve."""
    if len(losses) < EPOCHS_PER_PHASE:
        return False, f"incomplete ({len(losses)}/{EPOCHS_PER_PHASE} epochs) -> treat as failed"
    l5, l10 = losses[4], losses[9]
    improvement = l5 - l10
    min_loss = min(losses)
    never = min_loss >= LOSS_FLOOR
    stagnated = improvement < STAGNATION_EPS
    if never and stagnated:
        return False, f"CUT: never<{LOSS_FLOOR} (min={min_loss:.3f}) AND stagnant (d5->10={improvement:+.3f})"
    if never:
        return False, f"CUT: never dropped below {LOSS_FLOOR} (min={min_loss:.3f})"
    if stagnated:
        return False, f"CUT: stagnant ep5->10 (improvement={improvement:+.3f} < {STAGNATION_EPS})"
    return True, f"CONTINUE: min={min_loss:.3f}, ep5->10 improvement={improvement:+.3f}"


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


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} pause {secs}s...")
        time.sleep(secs)


def main():
    ap = argparse.ArgumentParser(description="automated 18-arch early-stopping sweep")
    ap.add_argument("--campaign-id",   type=str, default="p0139_arch18_2026_07_15")
    ap.add_argument("--python-exe",    type=str, default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--only",          type=str, default=None,
                    help="run only specs whose name contains this substring")
    ap.add_argument("--stall-minutes", type=float, default=180.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_p0139_arch18"
    runs_dir.mkdir(exist_ok=True)
    log_dir   = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    (repo_root / CKPT_DIR).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    print("\n" + "=" * 78)
    print("[arch18] PHerc0139 9.3um -- 18-arch automated early-stopping sweep")
    print(f"         gate: CUT if never<{LOSS_FLOOR} OR ep5->10 improvement<{STAGNATION_EPS}")
    print("=" * 78)

    specs = RUN_SPECS
    if args.only:
        specs = [s for s in RUN_SPECS if args.only in s["name"]]
        if not specs:
            print(f"[ABORT] --only '{args.only}' matched no spec"); return

    results: List[str] = []
    for i, spec in enumerate(specs, 1):
        print(f"\n{'#'*78}\n[{i}/{len(specs)}] {spec['name']}  (arch={spec['arch']}, depth={spec['depth']})\n{'#'*78}")

        p1_ckpt = f"{CKPT_DIR}/{spec['name']}_p1.pth"
        p1_extra = {"eval-int": 9999, "save-final": p1_ckpt}   # no figure in phase 1
        cmd1, exp1 = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec, 1, EPOCHS_PER_PHASE, p1_extra)
        print(f"   PHASE 1 ({EPOCHS_PER_PHASE} ep): {' '.join(str(c) for c in cmd1)}")
        if args.dry_run:
            # also show what phase 2 would look like
            p2_extra = {"eval-int": EPOCHS_PER_PHASE, "init-weights": p1_ckpt,
                        "save-final": f"{CKPT_DIR}/{spec['name']}_p2.pth"}
            cmd2, _ = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec, 2, EPOCHS_PER_PHASE, p2_extra)
            print(f"   PHASE 2 (if continue): {' '.join(str(c) for c in cmd2)}")
            continue

        log1 = log_dir / f"{exp1}.log"
        rc1, crashed1 = run_with_monitoring(cmd1, repo_root, env, log1, args.stall_minutes)
        losses = parse_train_losses(log1)
        print(f"   phase1 train_loss curve: {[f'{l:.3f}' for l in losses]}")
        if crashed1 or rc1 != 0:
            results.append(f"   {spec['name']}: PHASE1-FAIL(rc={rc1},crashed={crashed1})  losses={[f'{l:.3f}' for l in losses]}")
            cooldown(INTER_RUN_COOLDOWN_SECS if i < len(specs) else 0, "inter-run")
            continue

        cont, reason = decide(losses)
        print(f"   DECISION: {reason}")
        if not cont:
            results.append(f"   {spec['name']}: {reason}")
            cooldown(INTER_RUN_COOLDOWN_SECS if i < len(specs) else 0, "inter-run")
            continue

        # phase 2: resume from p1 checkpoint, 10 more epochs, render final figure
        cooldown(INTER_PHASE_COOLDOWN_SECS, "inter-phase")
        p2_ckpt = f"{CKPT_DIR}/{spec['name']}_p2.pth"
        p2_extra = {"eval-int": EPOCHS_PER_PHASE, "init-weights": p1_ckpt, "save-final": p2_ckpt}
        cmd2, exp2 = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec, 2, EPOCHS_PER_PHASE, p2_extra)
        print(f"   PHASE 2 ({EPOCHS_PER_PHASE} ep, resume): {' '.join(str(c) for c in cmd2)}")
        log2 = log_dir / f"{exp2}.log"
        rc2, crashed2 = run_with_monitoring(cmd2, repo_root, env, log2, args.stall_minutes)
        losses2 = parse_train_losses(log2)
        p2_status = "OK" if rc2 == 0 and not crashed2 else f"FAIL(rc={rc2},crashed={crashed2})"
        results.append(f"   {spec['name']}: CONTINUED->phase2 {p2_status}  "
                       f"p1={[f'{l:.3f}' for l in losses]}  p2={[f'{l:.3f}' for l in losses2]}")
        cooldown(INTER_RUN_COOLDOWN_SECS if i < len(specs) else 0, "inter-run")

    print("\n" + "=" * 78)
    print("[arch18] campaign complete -- summary")
    print("=" * 78)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
