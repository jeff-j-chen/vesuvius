"""campaign_runner_11.py — conv-stem hybrids vs full dataset, no ring.

Key learnings from C10:
  - Ring negatives hurt hard generalization (LocalContrast). Disabled entirely.
  - Full dataset (pos_weight=7.66) + no ring outperforms ring on valid F1 and easy contrast
  - Training range 28-44 preferred (40-48 is weak signal; evaluation still uses full 28-48)
  - Overfitting = better train metrics but WORSE hard probe. Regularization matters.
  - L1 lambda=7e-6 was previously tuned for v1 (1.3M params); scale linearly for other sizes
  - Hard mining only useful when model can reach the decision boundary on hard examples
  - Attention alone fails; conv stem first, then attention on features = plausible hybrid

C11 strategy:
  - 30 epochs, probe_int=5 for frequent readability snapshots
  - test_int=100 (never runs; expensive scroll-level inference not useful at arch-search scale)
  - No ring training in any run (t03 is the ring-ablation control)
  - t01: copy of C10 t10 (v1, no ring, 20 epochs) — already done, symlinked in
  - t02-t16: 8 architecture × {no-reg, regularized} pairs + ring ablation + alternating control
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:
    event_accumulator = None


@dataclass(frozen=True)
class RunSpec:
    run_id: int
    name: str
    axis: str
    overrides: Dict[str, Any]
    why: str


SMALL_SCROLL_ID = 20230827161847
SCROLL4_ID      = 20231210132040

# L1 lambda scaled by parameter count relative to v1 baseline (1.3M, tuned lambda=7e-6)
# lambda_arch = 7e-6 * (1.3M / param_count_arch)  [actually: scale proportionally]
# we scale so that total L1 penalty budget is roughly equal across architectures
L1_V1           = 7e-6    # 1.3M params
L1_UNET         = 1.7e-6  # 5.4M params → 7e-6 * (1.3/5.4) = 1.69e-6
L1_DEEP_CBAM    = 2.5e-6  # 3.6M params → 7e-6 * (1.3/3.6) = 2.53e-6
L1_ASYM_POOL    = 7.5e-6  # 1.2M params → 7e-6 * (1.3/1.2) = 7.58e-6
# v11 hybrids: estimated param counts filled in after first run prints them
L1_V11_DEPTH    = 6.5e-6  # ~1.4M est
L1_V11_SE       = 7e-6    # ~1.3M est (same structure as v1, SE replaces CBAM)
L1_V11_NONLOCAL = 6.5e-6  # ~1.4M est

# BASE: no ring, depth window 28-44 for training (evaluation unchanged at 28-48)
BASE: Dict[str, Any] = {
    "epochs": 30,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 64,
    "num-workers": 2,
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 100,
    "eval-cooldown": 45,          # 45s pause after each probe/eval epoch to cool hardware
    "no-hard-mining": True,       # default off; overridden per run
    # no ring — disabled entirely based on C10 findings
    "train-d-start": 28,
    "train-d-end": 44,            # tighter window; 44-48 has weak signal
    "l1-lambda": 0.0,             # default off; overridden per run
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "conv1-drop": 0.0,
    "channel-mixing-prob": 0.0,
}

RUN_SPECS: List[RunSpec] = [

    # ── t01: reference — C10 t10 result copied in (v1, no ring, 20 epochs, no reg)
    # NOT scheduled to run. copy_c10_t10() handles it before the loop starts.

    # ── baseline v1 ──────────────────────────────────────────────────────────

    # 2. v1, no lambda, no hm — matches C10 t10 conditions but with 30 epochs + d-end=44
    RunSpec(2, "t02_v1_noreg", "baseline",
        {"arch": "v10_v1_full_reg"},
        why="v1 baseline under C11 standard conditions (no ring, d-end=44, 30 epochs). "
            "establishes the capacity ceiling with zero regularization; expected to overfit "
            "past epoch 20 now that ring negatives no longer constrain it."),

    # 3. v1, lambda, hm — the 'properly regularized' v1
    RunSpec(3, "t03_v1_reg", "baseline",
        {"arch": "v10_v1_full_reg",
         "l1-lambda": L1_V1,
         "conv1-drop": 0.0, "conv2-drop": 0.05,
         "fc1-drop": 0.2, "fc2-drop": 0.1},
        why="v1 with full regularization (L1=7e-6, dropout matching original config defaults, "
            "hard mining enabled). this is the closest run to the original tuned baseline. "
            "hard mining is re-enabled; without ring, the full dataset boundary is broader "
            "and hard mining is less likely to conflict with the label set."),

    # 4. v1, ring set, lambda, hm — ring as an ablation (should underperform t03)
    RunSpec(4, "t04_v1_ring_reg", "ring_ablation",
        {"arch": "v10_v1_full_reg",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_V1,
         "conv1-drop": 0.0, "conv2-drop": 0.05,
         "fc1-drop": 0.2, "fc2-drop": 0.1},
        why="v1 RING training with full regularization — the C10 'correct' baseline we never ran. "
            "ring_negatives=True uses eroded labels. this is the direct ablation: "
            "C11-standard reg vs ring sampling. expected to underperform t03 based on C10."),

    # 5. v1, alternating ring, lambda, hm — novel: alternate full/ring per epoch
    RunSpec(5, "t05_v1_alt_ring", "alternating",
        {"arch": "v10_v1_full_reg",
         "alternating-ring": True,
         "l1-lambda": L1_V1,
         "conv1-drop": 0.0, "conv2-drop": 0.05,
         "fc1-drop": 0.2, "fc2-drop": 0.1},
        why="alternating ring: odd epochs train on ring set (boundary tiles), "
            "even epochs train on full set. hard mining active on ring epochs only. "
            "hypothesis: ring set epochs boost LocalContrast by training near the decision boundary; "
            "full epochs prevent the model from collapsing to boundary-only features."),

    # ── 3D U-Net ─────────────────────────────────────────────────────────────

    # 6. unet, no lambda, no hm — isolates arch quality from regularization
    RunSpec(6, "t06_unet_noreg", "unet",
        {"arch": "v10_3d_unet"},
        why="3D U-Net without regularization, no ring. C10 unet trained on ring (t02) and "
            "massively overfit (train_f1=0.999). this run tests unet on the full dataset "
            "to isolate whether the overfitting was architecture-driven or ring-driven."),

    # 7. unet, scaled lambda, hm
    RunSpec(7, "t07_unet_reg", "unet",
        {"arch": "v10_3d_unet",
         "l1-lambda": L1_UNET},
        why="3D U-Net with L1 scaled for 5.4M params (lambda=1.7e-6). "
            "L1 was designed to prevent memorization; at the right lambda, "
            "the U-Net's skip connections may actually help generalization rather than hurt it. "
            "hard mining enabled to push training signal toward the decision boundary."),

    # ── deep 3D CBAM ─────────────────────────────────────────────────────────

    # 8. deep_3d_cbam, no lambda, no hm
    RunSpec(8, "t08_deep_cbam_noreg", "deep_cnn",
        {"arch": "v10_deep_3d_cbam"},
        why="4-block deep 3D CBAM without regularization, no ring. C10 t05 (ring) overfit to "
            "train_f1=0.969 with hard=0.340. full dataset + no reg should reveal whether "
            "deep CBAM overfit was ring-driven or is intrinsic to the architecture."),

    # 9. deep_3d_cbam, scaled lambda, hm
    RunSpec(9, "t09_deep_cbam_reg", "deep_cnn",
        {"arch": "v10_deep_3d_cbam",
         "l1-lambda": L1_DEEP_CBAM},
        why="deep 3D CBAM with L1 scaled for 3.6M params (lambda=2.5e-6). "
            "CBAM at every block should maintain spatial focus; L1 prevents the channel "
            "capacity from memorizing easy ink morphology."),

    # ── asymmetric pool ───────────────────────────────────────────────────────

    # 10. asymmetric_pool, no lambda, no hm
    RunSpec(10, "t10_asym_pool_noreg", "subvoxel",
        {"arch": "v10_asymmetric_pool"},
        why="asymmetric pool (spatial avg + depth max) without regularization. "
            "C10 t14 on ring achieved hard=0.452 — third best. now tested on full dataset. "
            "depth-max is the key hypothesis: ink absorption spike at 1-2 slices, not spread."),

    # 11. asymmetric_pool, scaled lambda, hm
    RunSpec(11, "t11_asym_pool_reg", "subvoxel",
        {"arch": "v10_asymmetric_pool",
         "l1-lambda": L1_ASYM_POOL},
        why="asymmetric pool with full regularization. 1.2M params, lambda=7.5e-6. "
            "if depth-max correctly isolates the ink signal, regularization should prevent "
            "it from memorizing the spatial texture of easy ink and generalize to hard cases."),

    # ── v11 hybrid 1: conv stem + cross-depth attention ─────────────────────

    # 12. conv_stem_depth_attn, no lambda, no hm
    RunSpec(12, "t12_stem_depth_attn_noreg", "conv_attn",
        {"arch": "v11_conv_stem_depth_attn"},
        why="v1 3D conv stem (CBAM blocks 1-2+pool) → project to 64d → single Transformer "
            "layer over the post-pool feature tokens. conv provides inductive bias; attention "
            "resolves cross-depth correlations in feature space where S/N is already high. "
            "specifically designed to address the attention-fails-on-raw-voxels failure of C10."),

    # 13. conv_stem_depth_attn, lambda, hm
    RunSpec(13, "t13_stem_depth_attn_reg", "conv_attn",
        {"arch": "v11_conv_stem_depth_attn",
         "l1-lambda": L1_V11_DEPTH},
        why="conv stem + depth attention with regularization (L1=6.5e-6, HM enabled). "
            "the attention layer is small (1 Transformer layer, d=64) and should be stable "
            "to train with L1. hard mining provides boundary-region signal."),

    # ── v11 hybrid 2: SE-only channel attention ──────────────────────────────

    # 14. conv_stem_se, no lambda, no hm
    RunSpec(14, "t14_stem_se_noreg", "se_attn",
        {"arch": "v11_conv_stem_se"},
        why="v1 structure with SE (squeeze-excitation) replacing CBAM. "
            "SE does channel recalibration only — no spatial attention. "
            "hypothesis: CBAM's spatial attention destabilizes under noisy ring labels; "
            "SE-only is more stable. tests on full dataset which is noisier spatially."),

    # 15. conv_stem_se, lambda, hm
    RunSpec(15, "t15_stem_se_reg", "se_attn",
        {"arch": "v11_conv_stem_se",
         "l1-lambda": L1_V11_SE,
         "conv1-drop": 0.0, "conv2-drop": 0.05,
         "fc1-drop": 0.2, "fc2-drop": 0.1},
        why="SE-based v1 with full regularization matching C11 t03 (v1 reg) conditions. "
            "direct comparison: SE vs CBAM under identical hyperparameters. "
            "if SE wins, we should replace CBAM in the baseline for C12."),

    # ── v11 hybrid 3: non-local block after conv ──────────────────────────────

    # 16. conv_stem_nonlocal, no lambda, no hm
    RunSpec(16, "t16_stem_nonlocal_noreg", "nonlocal",
        {"arch": "v11_conv_stem_nonlocal"},
        why="v1 CBAM conv blocks → non-local means block at (B, 256, ~2, 7, 7) → pool + classifier. "
            "non-local: every feature position attends to every other position globally. "
            "unlike attention on raw voxels (C10 failures), this operates on CBAM-filtered features "
            "where ink signal is already concentrated. N~98 tokens at this resolution."),

    # 17. conv_stem_nonlocal, lambda, hm
    RunSpec(17, "t17_stem_nonlocal_reg", "nonlocal",
        {"arch": "v11_conv_stem_nonlocal",
         "l1-lambda": L1_V11_NONLOCAL,
         "conv1-drop": 0.0, "conv2-drop": 0.05,
         "fc1-drop": 0.2, "fc2-drop": 0.1},
        why="non-local v1 with full regularization. the non-local block adds long-range "
            "spatial context that pure convolution cannot express. with L1 preventing "
            "memorization, the model is forced to generalize — ideally to the hard ink signal."),
]


CRASH_SIGNALS = [
    "Traceback (most recent call last)",
    "CUDA error:",
    "RuntimeError:",
    "OSError: [Errno",
    "pickle data was truncated",
    "_pickle.UnpicklingError",
    "CUDA out of memory",
    "forrtl: error",
    "WinError 1455",
]


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=90):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
    last_progress = time.time(); last_epoch = 0
    while proc.poll() is None:
        time.sleep(15)
        try: lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        except Exception: continue
        tail = "".join(lines[-40:])
        for sig in CRASH_SIGNALS:
            if sig in tail:
                print(f"\n[MONITOR] CRASH -- '{sig}'")
                print("[MONITOR] last output:\n" + "".join(lines[-15:]))
                try: proc.kill()
                except Exception: pass
                proc.wait(); return proc.returncode or 1, True
        for line in lines[-40:]:
            if "--- Epoch" in line:
                try:
                    ep = int(line.strip().split("/")[0].split()[-1])
                    if ep > last_epoch:
                        last_epoch = ep; last_progress = time.time()
                        print(f"[MONITOR] {line.strip()}")
                except Exception: pass
        if time.time() - last_progress > stall_minutes * 60:
            print(f"\n[MONITOR] STALL -- no progress in {stall_minutes} min")
            try: proc.kill()
            except Exception: pass
            proc.wait(); return 1, True
    proc.wait(); rc = proc.returncode
    if rc != 0:
        try:
            tail = open(log_path, encoding="utf-8", errors="replace").readlines()[-20:]
            print("[MONITOR] last output:\n" + "".join(tail))
        except Exception: pass
    print(f"[MONITOR] {'completed successfully' if rc == 0 else f'exited rc={rc}'}")
    return rc, False


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def dict_to_cli_args(overrides):
    args = []
    for key, value in overrides.items():
        if key in ("no-ring-negatives", "alternating-ring"):
            pass  # handled specially in run_one
        elif isinstance(value, bool):
            if value: args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def find_run_dir(runs_dir, exp_name, start_ts):
    matches = [p for p in runs_dir.glob(f"{exp_name}_*") if p.is_dir()]
    if not matches: return None
    matches.sort(key=lambda p: p.stat().st_mtime)
    for p in reversed(matches):
        if p.stat().st_mtime >= start_ts - 5: return p
    return matches[-1]


def extract_metrics(run_dir):
    m = {"valid_f1_last": None, "probe_easy_last": None, "probe_hard_last": None}
    if run_dir is None or event_accumulator is None: return m
    evts = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not evts: return m
    ea = event_accumulator.EventAccumulator(str(evts[-1]), size_guidance={"scalars": 0})
    ea.Reload(); avail = set(ea.Tags().get("scalars", []))
    for key, tag in [("valid_f1", "P_M/F1_Score/Valid"),
                     ("probe_easy", "R_M/Probe/Easy/ReadabilityComposite"),
                     ("probe_hard", "R_M/Probe/Hard/ReadabilityComposite")]:
        if tag in avail:
            vals = [e.value for e in ea.Scalars(tag)]; m[f"{key}_last"] = vals[-1]
    return m


def quality_score(m):
    return float(m.get("valid_f1_last") or 0) + float(m.get("probe_easy_last") or 0)


def print_summary(completed):
    if not completed: return
    print("\n+-- campaign 11 results (ranked by hard probe) ----------------------")
    print(f"|  {'run':<48} {'hard':>5} {'easy':>5} {'f1':>5} {'qual':>6}")
    print("|  " + "-" * 69)
    for r in sorted(completed,
                    key=lambda r: (r.get("metrics") or {}).get("probe_hard_last") or 0,
                    reverse=True):
        m = r.get("metrics") or {}
        hard = f"{m.get('probe_hard_last',0.0):.3f}" if m.get("probe_hard_last") is not None else "?"
        easy = f"{m.get('probe_easy_last',0.0):.3f}" if m.get("probe_easy_last") is not None else "?"
        f1   = f"{m.get('valid_f1_last',0.0):.3f}"  if m.get("valid_f1_last")   is not None else "?"
        print(f"|  {r['name'][-48:]:<48} {hard:>5} {easy:>5} {f1:>5} {quality_score(m):>6.3f}")
    print("+--" + "-" * 71 + "\n")


def choose_next(pending, completed):
    """strict sequential ordering by run_id."""
    return sorted(pending, key=lambda s: s.run_id)[0]


def copy_c10_t10(runs_dir: Path, state: dict, campaign_id: str):
    """link/copy the C10 t10 result into runs_campaign11 as the t01 reference."""
    c10_run = Path("runs_campaign10/cmp_c10_2026_06_15_t10_v1_no_ring_full_reg_16_13-08-32")
    target  = runs_dir / "ref_c10_t10_v1_no_ring"

    # check if already done
    if any(r.get("run_id") == 1 for r in state.get("completed", [])):
        print("[t01] C10 t10 reference already recorded, skipping copy")
        return

    if not c10_run.exists():
        print(f"[t01] WARNING: C10 t10 dir not found at {c10_run}, skipping reference copy")
        return

    if not target.exists():
        shutil.copytree(str(c10_run), str(target))
        print(f"[t01] copied C10 t10 → {target}")
    else:
        print(f"[t01] reference dir already exists at {target}")

    # record in state with frozen metrics from C10 state JSON
    c10_state_path = Path("runs_campaign10/campaign_logs/c10_2026_06_15_state.json")
    c10_metrics = {"valid_f1_last": 0.293, "probe_easy_last": 0.660, "probe_hard_last": 0.361}
    if c10_state_path.exists():
        try:
            c10_state = json.loads(c10_state_path.read_text(encoding="utf-8"))
            for r in c10_state.get("completed", []):
                if r.get("run_id") == 10:
                    c10_metrics = r.get("metrics", c10_metrics)
                    break
        except Exception:
            pass

    rec = {
        "run_id": 1,
        "name": f"cmp_{campaign_id}_t01_ref_c10_t10",
        "axis": "reference",
        "note": "copy of C10 t10 (v1, no ring, 20 epochs, no reg). baseline reference.",
        "run_dir": str(target),
        "metrics": c10_metrics,
        "ended_at": now_utc(),
    }
    state.setdefault("completed", []).append(rec)
    print(f"[t01] recorded reference: hard={c10_metrics.get('probe_hard_last'):.3f}")


def main():
    parser = argparse.ArgumentParser(description="campaign 11 -- conv-stem hybrids, full dataset, no ring")
    parser.add_argument("--campaign-id", type=str, default="c11_2026_06_17")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=120.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_campaign11"
    runs_dir.mkdir(exist_ok=True)
    state_dir = runs_dir / "campaign_logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{args.campaign_id}_state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"campaign_id": args.campaign_id, "created_at": now_utc(),
                 "completed": [], "failed": []}

    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "TF_ENABLE_ONEDNN_OPTS": "0",
                "TF_CPP_MIN_LOG_LEVEL": "3"})

    # copy C10 t10 as the t01 reference before starting the loop
    if not args.dry_run:
        copy_c10_t10(runs_dir, state, args.campaign_id)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    base = dict(BASE)
    base["log-dir"] = str(runs_dir)

    done_ids = {int(r["run_id"]) for r in state.get("completed", []) + state.get("failed", [])}
    pending  = [s for s in RUN_SPECS if s.run_id not in done_ids]
    completed_records = state.get("completed", [])

    while pending:
        print_summary(completed_records)
        spec = choose_next(pending, completed_records)

        merged = dict(base)
        merged.update(spec.overrides)

        # handle boolean flags that need special treatment
        if spec.overrides.get("alternating-ring"):
            merged["alternating-ring"] = True
        # hard mining: enabled only for regularized runs (l1 > 0)
        # BASE has no-hard-mining=True; remove it for runs that should use HM
        if "no-hard-mining" not in spec.overrides:
            if merged.get("l1-lambda", 0.0) == 0.0:
                merged["no-hard-mining"] = True   # no-reg run: keep HM off
            else:
                merged.pop("no-hard-mining", None) # reg run: enable HM by removing the flag
        merged["log-dir"] = str(runs_dir)

        exp_name = f"cmp_{args.campaign_id}_{spec.name}"
        log_path = state_dir / f"{exp_name}.log"
        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)
        # inject boolean flags not handled by dict_to_cli_args
        if merged.get("alternating-ring"):
            cmd.append("--alternating-ring")

        print(f"\n{'='*62}")
        print(f"  run {spec.run_id:02d}/17: {spec.name}  [{spec.axis}]")
        print(f"  overrides: { {k:v for k,v in spec.overrides.items()} }")
        print(f"  {spec.why[:120]}")
        print(f"{'='*62}")

        start_ts = time.time()
        rc, crashed = (0, False) if args.dry_run else run_with_monitoring(
            cmd, repo_root, env, str(log_path), args.stall_minutes)

        run_dir = find_run_dir(runs_dir, exp_name, start_ts)
        metrics = extract_metrics(run_dir)
        hard = metrics.get("probe_hard_last"); easy = metrics.get("probe_easy_last")
        f1   = metrics.get("valid_f1_last")
        print(f"\n  RESULT: hard={hard}  easy={easy}  f1={f1}  quality={quality_score(metrics):.3f}")

        rec = {"run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
               "overrides": merged, "run_dir": str(run_dir) if run_dir else None,
               "metrics": metrics, "ended_at": now_utc()}
        if rc == 0:
            state.setdefault("completed", []).append(rec)
            completed_records = state["completed"]
        else:
            rec.update({"return_code": rc, "crashed_early": crashed})
            state.setdefault("failed", []).append(rec)

        if not args.dry_run:
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        done_ids.add(spec.run_id)
        pending = [s for s in RUN_SPECS if s.run_id not in done_ids]

        # cooldown between runs
        if pending and not args.dry_run:
            print("[COOLDOWN] waiting 90s for GPU to cool before next run...")
            time.sleep(90)

    print_summary(state.get("completed", []))
    print("campaign 11 finished")


if __name__ == "__main__":
    main()
