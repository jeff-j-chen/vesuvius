"""campaign_runner_p0139_arch28.py -- automated arch sweep, round 2 (resume + 10 new).

PHerc0139 9.3um. logs into the SAME folder as arch18 (runs_p0139_arch18) using the same
two-phase early-cancel mechanism. this round:
  - RESUMES the arch18 winner s07_v14_mil_deep straight to phase 2 (it completed phase 1
    at train_loss 0.814 -- best of the whole sweep -- but the 0.8 gate cut it, so we never
    saw its eval figure). uses its saved phase-1 checkpoint.
  - FINISHES the arch18 leftovers the crash never completed: n06_lcn (crashed at ep8, no
    ckpt), n07_gabor, n08_bottattn, n09_aspp, n10_hr.
  - adds 10 NEW physics-motivated archs (r01-r10) built on the two winners:
      s07_v14_mil_deep -> MIL per-voxel logits + LSE localization
      n06_lcn          -> local contrast normalization removes the 113keV bulk-density baseline

GATE (kept aggressive at 0.8, per user): in arch18 NO run's train_loss dropped below 0.8
(best s07=0.814), so a 0.8 floor cuts every NEW phase-1 run that can't cross it fast --
which is the intent: anything that can't cross 0.8 quickly is implicitly a bust. the
s07 resume below is a MANUAL override that bypasses the gate (goes straight to phase 2),
so it is unaffected by the floor. arch18 min-losses for reference:
  s07 0.814, s05 0.912, n06 0.916, s04 0.935, s06 0.939, s08 0.949, s03 0.970, s02 0.990,
  n03 1.014, n02 1.018, n01 1.026, n04 1.035, n05 1.041, s01 1.156.

THERMAL SAFETY: long idle cooldowns everywhere (per-epoch 90s, train->val 120s, post-eval
600s, fig-chunk 600ms, inter-phase 300s, inter-run 420s).
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SID                       = 20260115000000
MAE_CKPT                  = "models/mae_p0139_dense_unet.pth"
EPOCHS_PER_PHASE          = 10
LOSS_FLOOR                = 0.8              # aggressive gate: cant cross 0.8 fast -> bust
STAGNATION_EPS            = 0.02             # improvement ep5->ep10 below this -> stagnated
INTER_RUN_COOLDOWN_SECS   = 420
INTER_PHASE_COOLDOWN_SECS = 300
CKPT_DIR                  = "models/arch28"
ARCH18_CKPT_DIR           = "models/arch18"  # source of s07's phase-1 checkpoint

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
    "probe-int":          9999,
    "test-int":           9999,
    "ring-negatives":     True,
    "ring-label-source":  "eroded",
    "crop-x-frac":        "0.0,1.0",
    "crop-y-frac":        "0.0,1.0",
    "split-axis":         "y",
    "train-split-frac":   0.8055,
    "batch-size":         128,
    "lr":                 2e-4,
    "l1-lambda":          0.0,
    "data-aug":           0,
    "num-workers":        2,
    "mask-memmap":        True,
    "no-hard-mining":     True,
    "no-probe-rois":      True,
    "epoch-cooldown":     90,
    "val-cooldown":       120,
    "eval-cooldown":      600,
    "fig-chunk-cooldown": 600,
}

DENSE = {"dense-labels": True, "dense-soft-labels": True}

# arch18 winner to resume straight to phase 2 (has a saved phase-1 checkpoint).
RESUME_SPECS: List[Dict[str, Any]] = [
    {"name": "s07_v14_mil_deep_d8", "arch": "v14_mil_deep", "depth": 8, "batch-size": 64,
     "resume-ckpt": f"{ARCH18_CKPT_DIR}/s07_v14_mil_deep_d8_p1.pth"},   # TILE output (no dense)
]

# full phase1->gate->phase2 runs: 5 arch18 leftovers the crash never finished + 10 new.
RUN_SPECS: List[Dict[str, Any]] = [
    # --- finish arch18 leftovers (crash happened during n06) ---
    {"name": "n06_lcn_d8",        "arch": "dense_unet_lcn",      "depth": 8, **DENSE},
    {"name": "n07_gabor_d8",      "arch": "dense_unet_gabor",    "depth": 8, **DENSE},
    {"name": "n08_bottattn_d8",   "arch": "dense_unet_bottattn", "depth": 8, **DENSE},
    {"name": "n09_aspp_d8",       "arch": "dense_unet_aspp",     "depth": 8, **DENSE},
    {"name": "n10_hr_d8",         "arch": "dense_unet_hr",       "depth": 8, **DENSE},
    # --- 10 NEW archs, built on winners LCN(n06) + MIL/LSE(s07) ---
    {"name": "r01_lcnmil_d8",     "arch": "dense_unet_lcnmil",    "depth": 8, **DENSE},
    {"name": "r02_zscore_d8",     "arch": "dense_unet_zscore",    "depth": 8, **DENSE},
    {"name": "r03_moments_d8",    "arch": "dense_unet_moments",   "depth": 8, **DENSE},
    {"name": "r04_lcnms_d8",      "arch": "dense_unet_lcnms",     "depth": 8, **DENSE},
    {"name": "r05_tophat_d8",     "arch": "dense_unet_tophat",    "depth": 8, **DENSE},
    {"name": "r06_coherence_d8",  "arch": "dense_unet_coherence", "depth": 8, **DENSE},
    {"name": "r07_dog_d8",        "arch": "dense_unet_dog",       "depth": 8, **DENSE},
    {"name": "r08_lse_d8",        "arch": "dense_unet_lse",       "depth": 8, **DENSE},
    {"name": "r09_milhead_d8",    "arch": "dense_unet_milhead",   "depth": 8, **DENSE},
    {"name": "r10_attn_lcn_d8",   "arch": "dense_unet_attn_lcn",  "depth": 8, **DENSE},
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
    """build a train.py command for one phase of one spec (drops runner-only keys)."""
    merged = dict(BASE)
    for k, v in spec.items():
        if k not in ("name", "resume-ckpt"):
            merged[k] = v
    merged["epochs"] = epochs
    merged.update(extra)
    exp_name = f"cmp_{campaign_id}_{spec['name']}_p{phase}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def parse_train_losses(log_path: Path) -> List[float]:
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
    """gate purely on the phase-1 train_loss curve. aggressive by design."""
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


def run_phase2(python_exe, runs_dir, log_dir, campaign_id, spec, init_ckpt, repo_root, env, stall):
    """run one phase-2 (10 ep + eval figure), resuming from init_ckpt. returns result str."""
    cooldown(INTER_PHASE_COOLDOWN_SECS, "inter-phase")
    p2_ckpt = f"{CKPT_DIR}/{spec['name']}_p2.pth"
    p2_extra = {"eval-int": EPOCHS_PER_PHASE, "init-weights": init_ckpt, "save-final": p2_ckpt}
    cmd2, exp2 = build_cmd(python_exe, runs_dir, campaign_id, spec, 2, EPOCHS_PER_PHASE, p2_extra)
    print(f"   PHASE 2 ({EPOCHS_PER_PHASE} ep, resume from {init_ckpt}): {' '.join(str(c) for c in cmd2)}")
    log2 = log_dir / f"{exp2}.log"
    rc2, crashed2 = run_with_monitoring(cmd2, repo_root, env, log2, stall)
    losses2 = parse_train_losses(log2)
    status = "OK" if rc2 == 0 and not crashed2 else f"FAIL(rc={rc2},crashed={crashed2})"
    return f"phase2 {status}  p2={[f'{l:.3f}' for l in losses2]}"


def main():
    ap = argparse.ArgumentParser(description="arch28: resume winner + finish leftovers + 10 new")
    ap.add_argument("--campaign-id",   type=str, default="p0139_arch28_2026_07_16")
    ap.add_argument("--python-exe",    type=str, default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--only",          type=str, default=None,
                    help="run only specs whose name contains this substring")
    ap.add_argument("--skip-resume",   action="store_true", help="skip the s07 phase-2 resume")
    ap.add_argument("--stall-minutes", type=float, default=180.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_p0139_arch18"     # SAME folder as arch18 (per user)
    runs_dir.mkdir(exist_ok=True)
    log_dir   = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    (repo_root / CKPT_DIR).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    print("\n" + "=" * 78)
    print("[arch28] PHerc0139 9.3um -- resume winner + finish leftovers + 10 new archs")
    print(f"         gate: CUT if never<{LOSS_FLOOR} OR ep5->10 improvement<{STAGNATION_EPS}")
    print("=" * 78)

    results: List[str] = []

    # ---- phase-2 resumes of arch18 winners ----
    if not args.skip_resume:
        for spec in RESUME_SPECS:
            if args.only and args.only not in spec["name"]:
                continue
            ckpt = spec["resume-ckpt"]
            print(f"\n{'#'*78}\n[RESUME] {spec['name']}  (arch={spec['arch']}) -> phase 2 from {ckpt}\n{'#'*78}")
            if not (repo_root / ckpt).exists():
                print(f"   [SKIP] checkpoint missing: {ckpt}")
                results.append(f"   {spec['name']}: RESUME-SKIP (no ckpt {ckpt})")
                continue
            if args.dry_run:
                _, exp2 = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec, 2,
                                    EPOCHS_PER_PHASE, {"eval-int": EPOCHS_PER_PHASE,
                                    "init-weights": ckpt, "save-final": f"{CKPT_DIR}/{spec['name']}_p2.pth"})
                print(f"   would run phase 2 -> {exp2}")
                continue
            res = run_phase2(args.python_exe, runs_dir, log_dir, args.campaign_id, spec,
                             ckpt, repo_root, env, args.stall_minutes)
            results.append(f"   {spec['name']}: RESUME->{res}")
            cooldown(INTER_RUN_COOLDOWN_SECS, "inter-run")

    # ---- full phase1 -> gate -> phase2 runs ----
    specs = RUN_SPECS
    if args.only:
        specs = [s for s in RUN_SPECS if args.only in s["name"]]

    for i, spec in enumerate(specs, 1):
        print(f"\n{'#'*78}\n[{i}/{len(specs)}] {spec['name']}  (arch={spec['arch']}, depth={spec['depth']})\n{'#'*78}")
        p1_ckpt = f"{CKPT_DIR}/{spec['name']}_p1.pth"
        p1_extra = {"eval-int": 9999, "save-final": p1_ckpt}
        cmd1, exp1 = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec, 1, EPOCHS_PER_PHASE, p1_extra)
        print(f"   PHASE 1 ({EPOCHS_PER_PHASE} ep): {' '.join(str(c) for c in cmd1)}")
        if args.dry_run:
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
        res = run_phase2(args.python_exe, runs_dir, log_dir, args.campaign_id, spec,
                         p1_ckpt, repo_root, env, args.stall_minutes)
        results.append(f"   {spec['name']}: CONTINUED->{res}  p1={[f'{l:.3f}' for l in losses]}")
        cooldown(INTER_RUN_COOLDOWN_SECS if i < len(specs) else 0, "inter-run")

    print("\n" + "=" * 78)
    print("[arch28] campaign complete -- summary")
    print("=" * 78)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
