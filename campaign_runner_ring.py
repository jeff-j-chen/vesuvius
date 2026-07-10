"""campaign_runner_ring.py — re-run best models from C3-C7 with corrected ring negatives.

ring negatives: training uses only tiles that are ink OR are ring-adjacent to ink tiles.
tile-level dilation to ~1:1 pos/neg ratio ensures clean labels and balanced classes.
pos_weight is re-computed from actual training set (not cached) → no focal loss needed.
no hard mining: ring already ensures clean, balanced samples.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
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

# all ring runs use num-workers=0 (safe on Windows) and ring-negatives=True
# no focal loss, no hard mining — ring provides balanced 1:1 training
RING_BASE: Dict[str, Any] = {
    "epochs": 20,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 512,
    "num-workers": 0,
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 45,
    "no-hard-mining": True,
    "ring-negatives": True,
    "ring-label-source": "closed",  # CLOSE+GAP morphology, 0% contamination, best sanity result
    "channel-mixing-prob": 0.0,
    "conv1-drop": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "l1-lambda": 0.0,
}

RUN_SPECS: List[RunSpec] = [
    # C3 / baseline CNN
    RunSpec(1, "r01_v1_ring", "baseline_cnn",
        {"arch": "v1", "batch-size": 64},
        why="v1 original 3D CBAM CNN. already tested (t21) gave hard=0.440 easy=0.557, "
            "the best easy probe across all campaigns. baseline sanity check."),

    RunSpec(2, "r02_v3preact_ring", "baseline_cnn",
        {"arch": "v3_preact_baseline", "batch-size": 64},
        why="C3 best: pre-activation ResNet baseline. with ring should outperform v1 "
            "due to better gradient flow in deeper CNN."),

    # C5 best sequential
    RunSpec(3, "r03_depth_transformer_ring", "c5_best",
        {"arch": "v5_depth_profile_transformer"},
        why="C5 best: depth transformer (hard=0.372). ring should improve "
            "by removing unlabeled-ink contamination from negatives."),

    RunSpec(4, "r04_depth_profile_1d_ring", "c5_best",
        {"arch": "v5_depth_profile_1d"},
        why="C5: depth profile 1D CNN (hard=0.360). clean ring training."),

    RunSpec(5, "r05_depth_variance_ring", "c5_best",
        {"arch": "v5_depth_variance_2d"},
        why="C5: depth variance 2D CNN — captured variance signal (easy=0.506). "
            "spatial variance at each depth may benefit from clean ring."),

    # C6 best sequential
    RunSpec(6, "r06_lstm_ring", "c6_best",
        {"arch": "v6_lstm_slices"},
        why="C6 BEST non-ring (hard=0.445). already tested with ring (t23=0.433). "
            "re-running with corrected pos_weight (ring gives 1:1, old run used cached 7.66). "
            "expect improvement over previous ring test."),

    RunSpec(7, "r07_bigru_ring", "c6_best",
        {"arch": "v6_bigru_slices"},
        why="C6: BiGRU (hard=0.419, ring t22=0.436). corrected pos_weight re-run."),

    RunSpec(8, "r08_pixel_local_attn_ring", "c6_best",
        {"arch": "v6_pixel_local_attn", "batch-size": 64},
        why="C6: pixel_local_attn (hard=0.414, easy=0.532). local window attention. "
            "ring + correct pos_weight may significantly improve easy probe."),

    RunSpec(9, "r09_fulldepth_gru_ring", "c6_best",
        {"arch": "v6_fulldepth_gru", "input-mode": "fulldepth", "batch-size": 64},
        why="C6: fulldepth_gru (hard=0.436). full 64-depth GRU + ring."),

    RunSpec(10, "r10_fulldepth_1d_ring", "c6_best",
        {"arch": "v6_fulldepth_1d", "input-mode": "fulldepth", "batch-size": 64},
        why="C6: fulldepth_1d 1D CNN on full 64-depth profile (hard=0.345). "
            "clean ring training may improve."),

    # C6 spatial
    RunSpec(11, "r11_slice_attention_ring", "c6_best",
        {"arch": "v6_slice_attention"},
        why="C6: slice_attention Transformer over depth slices (hard=0.373). "
            "multi-head attention over 8 depth slices."),

    # C7 best novel features
    RunSpec(12, "r12_pairwise_ring", "c7_best",
        {"arch": "v7_pairwise_depth"},
        why="C7: pairwise_depth (hard=0.414). scale-invariant pairwise comparisons. "
            "best non-ring C7 approach after focal BiGRU."),

    RunSpec(13, "r13_percentile_ring", "c7_best",
        {"arch": "v7_percentile_depth"},
        why="C7: percentile_depth (hard=0.406). robust to sparse ink pixels. "
            "already tested with ring (t24=0.419, corrected pos_weight re-run)."),

    RunSpec(14, "r14_bigru_percentile_ring", "c7_best",
        {"arch": "v7_bigru_percentile"},
        why="C7: bigru_percentile — BiGRU on percentile sequences (hard=0.389). "
            "combines percentile robustness with sequential depth modeling."),

    RunSpec(15, "r15_ae_bottleneck_ring", "c7_best",
        {"arch": "v7_ae_bottleneck"},
        why="C7: ae_bottleneck (hard=0.393). AE reconstruction error as anomaly score. "
            "ring may help by removing unlabeled-ink from reconstruction training."),
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


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=60):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env,
                                stdout=lf, stderr=None)
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
        if isinstance(value, bool):
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
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    for key, tag in [("valid_f1", "P_M/F1_Score/Valid"),
                     ("probe_easy", "R_M/Probe/Easy/ReadabilityComposite"),
                     ("probe_hard", "R_M/Probe/Hard/ReadabilityComposite")]:
        if tag in avail:
            vals = [e.value for e in ea.Scalars(tag)]
            m[f"{key}_last"] = vals[-1]
    return m


def print_summary(completed):
    if not completed: return
    print("\n+-- ring re-run results (ranked by hard probe) -------")
    print(f"|  {'run':<40} {'hard':>5} {'easy':>5}")
    print("|  " + "-" * 50)
    for r in sorted(completed,
                    key=lambda r: (r.get("metrics") or {}).get("probe_hard_last") or 0,
                    reverse=True):
        m = r.get("metrics") or {}
        hard = f"{m.get('probe_hard_last',0.0):.3f}" if m.get("probe_hard_last") is not None else "?"
        easy = f"{m.get('probe_easy_last',0.0):.3f}" if m.get("probe_easy_last") is not None else "?"
        print(f"|  {r['name'][-40:]:<40} {hard:>5} {easy:>5}")
    print("+--" + "-" * 54 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ring re-run of best C3-C7 models")
    parser.add_argument("--campaign-id", type=str, default="ring_2026_06_14")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=60.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_ring"
    runs_dir.mkdir(exist_ok=True)
    state_dir = runs_dir / "campaign_logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{args.campaign_id}_state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"campaign_id": args.campaign_id, "created_at": now_utc(),
                 "completed": [], "failed": []}

    if args.retry_failed and state.get("failed"):
        state["failed"] = []
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    base = dict(RING_BASE)
    base["log-dir"] = str(runs_dir)

    while True:
        completed_records = state.get("completed", [])
        completed_ids = {int(r["run_id"]) for r in completed_records}
        failed_ids    = {int(r["run_id"]) for r in state.get("failed", [])}
        pending = [s for s in RUN_SPECS
                   if s.run_id not in completed_ids and s.run_id not in failed_ids]
        if not pending or len(completed_records) >= args.max_runs: break

        print_summary(completed_records)
        spec = sorted(pending, key=lambda s: s.run_id)[0]
        merged = dict(base); merged.update(spec.overrides)
        exp_name = f"cmp_{args.campaign_id}_{spec.name}"
        log_path = state_dir / f"{exp_name}.log"

        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)
        print(f"\n{'='*60}")
        print(f"  run {spec.run_id:02d}/{args.max_runs}: {spec.name}  [{spec.axis}]")
        print(f"  overrides: {spec.overrides}")
        print(f"{'='*60}")

        start_ts = time.time()
        env = os.environ.copy()
        env.update({"MPLBACKEND": "Agg", "TF_ENABLE_ONEDNN_OPTS": "0",
                    "TF_CPP_MIN_LOG_LEVEL": "3"})

        rc, crashed = (0, False) if args.dry_run else run_with_monitoring(
            cmd, repo_root, env, str(log_path), args.stall_minutes)

        run_dir = find_run_dir(runs_dir, exp_name, start_ts)
        metrics = extract_metrics(run_dir)

        hard = metrics.get("probe_hard_last")
        easy = metrics.get("probe_easy_last")
        print(f"\n  RESULT: hard={hard}  easy={easy}")

        rec = {"run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
               "overrides": merged, "run_dir": str(run_dir) if run_dir else None,
               "metrics": metrics, "ended_at": now_utc()}
        if rc == 0:
            state.setdefault("completed", []).append(rec)
        else:
            rec.update({"return_code": rc, "crashed_early": crashed})
            state.setdefault("failed", []).append(rec)
        if not args.dry_run:
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print_summary(state.get("completed", []))
    print("ring campaign finished")


if __name__ == "__main__":
    main()
