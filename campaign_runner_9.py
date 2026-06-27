"""campaign_runner_9.py — Campaign 9: sanity-gated ring selection + full v8 suite.

Structure:
  Tests 1-3  (sanity): v1 CNN with eroded / original / closed ring — 20 epochs each.
             After all 3 complete, the best ring source is auto-selected based on
             F1 + PR-AUC (valid_f1 + probe_easy composite score).
  Tests 4-21 (v8 suite): all remaining v8 architectures using the winning ring source.
             Results from C8 t01/t04/t05/t06 already exist; those archs are skipped.

Quality metric: higher is better for (valid_f1_last + probe_easy_last).
Target: all sanity tests should outperform C7 t21 (easy≈0.557, hard≈0.440, f1≈0.203).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# base overrides — ring-label-source filled in dynamically after sanity tests
BASE: Dict[str, Any] = {
    "epochs": 20,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 512,
    "num-workers": 2,
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 45,
    "no-hard-mining": True,
    "ring-negatives": True,
    "channel-mixing-prob": 0.0,
    "conv1-drop": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "l1-lambda": 0.0,
}

SANITY_SOURCES = ["eroded", "original", "closed"]

# v8 architectures not yet run in C8 (t01/t04/t05/t06 already complete)
V8_RUNS: List[RunSpec] = [
    RunSpec(4,  "t04_percentile_bigru",       "percentile_seq",   {"arch": "v8_percentile_bigru"},
        why="4-layer BiGRU on percentile-feature sequences. combines percentile robustness with deep recurrent modeling."),
    RunSpec(5,  "t05_pairwise_bigru",          "percentile_seq",   {"arch": "v8_pairwise_bigru"},
        why="3-layer BiGRU on pairwise depth differences. scale-invariant + sequential."),
    RunSpec(6,  "t06_robust_zscore",           "physics",          {"arch": "v8_robust_zscore"},
        why="IQR z-score normalization before 1D CNN. shift- and scale-invariant depth shape."),
    RunSpec(7,  "t07_absorption_ratio",        "physics",          {"arch": "v8_absorption_ratio"},
        why="Centered profile + depth/mean ratio. physics-motivated scale-invariant features."),
    RunSpec(8,  "t08_spatial_contrast",        "spatial_depth",    {"arch": "v8_spatial_contrast"},
        why="Within-tile quadrant contrast → BiGRU. finds spatial heterogeneity at each depth."),
    RunSpec(9,  "t09_tile_entropy",            "spatial_depth",    {"arch": "v8_tile_entropy"},
        why="Soft spatial entropy per depth slice → BiGRU. ink = spatially heterogeneous tile."),
    RunSpec(10, "t10_superpixel_bigru",        "spatial_depth",    {"arch": "v8_superpixel_bigru"},
        why="4×4 superpixel profiles in raster order → BiGRU. captures ink stroke continuity."),
    RunSpec(11, "t11_multiscale_percentile",   "spatial_depth",    {"arch": "v8_multiscale_percentile"},
        why="Percentile features at 3 spatial scales: full tile + quadrants + 4×4 cells."),
    RunSpec(12, "t12_residual_spatial_depth",  "spatial_depth",    {"arch": "v8_residual_spatial_depth"},
        why="Per-slice 2D CNN + depth attention. learns spatial features per depth then combines."),
    RunSpec(13, "t13_deep_bigru",              "deep_sequential",  {"arch": "v8_deep_bigru"},
        why="6-layer 512-hidden BiGRU on 8-depth profile. deepest sequential model tried."),
    RunSpec(14, "t14_fulldepth_transformer16", "deep_sequential",  {"arch": "v8_fulldepth_transformer16", "input-mode": "fulldepth", "batch-size": 64},
        why="16-layer transformer on full 64-depth profile. deepest transformer + full curve."),
    RunSpec(15, "t15_full64_pct_bigru",        "deep_sequential",  {"arch": "v8_full64_pct_bigru", "input-mode": "fulldepth", "batch-size": 64},
        why="Percentile-sequence BiGRU on full 64-depth profile. fullest combination."),
    RunSpec(16, "t16_matched_filter_diff",     "combo",            {"arch": "v8_matched_filter", "input-mode": "diff"},
        why="Matched filter templates on differential signal. baseline removal + template matching."),
    RunSpec(17, "t17_percentile_bigru_diff",   "combo",            {"arch": "v8_percentile_bigru", "input-mode": "diff"},
        why="Percentile BiGRU on differential input. baseline removal + percentile + sequential."),
    RunSpec(18, "t18_pairwise_bigru_fulldepth","combo",            {"arch": "v8_pairwise_bigru", "input-mode": "fulldepth", "batch-size": 64},
        why="Pairwise BiGRU on full 64-depth. scale-invariant pairwise + full absorption curve."),
    RunSpec(19, "t19_spatial_contrast_30ep",   "combo",            {"arch": "v8_spatial_contrast", "epochs": 30},
        why="Spatial contrast with 30 epochs for fuller convergence."),
    RunSpec(20, "t20_v1_cnn_no_ring",          "baseline",         {"arch": "v1", "batch-size": 64, "no-ring-negatives": True},
        why="v1 CNN WITHOUT ring negatives — full-mask training as ablation baseline."),
    RunSpec(21, "t21_fulldepth_gru_best",      "best_rerun",       {"arch": "v6_fulldepth_gru", "input-mode": "fulldepth", "batch-size": 64},
        why="C ring best: fulldepth_gru (hard=0.479 in ring campaign). rerun with winning ring."),

    # ── Extended depth tests (d_start=28, d_end=48 — full pre+ink+post window) ──
    # original arch_search (runs/) trained on d_start=28/d_end=48 with z_step=depth//2=4
    # giving windows 28-36, 32-40, 36-44, 40-48 — much more diverse depth coverage.
    # compare to: arch_search t01 (f1=0.345, easy=0.503), t05 (f1=0.336, easy=0.538)
    # and campaign 9 sanity tests (eroded qual=0.850, original qual=0.858, closed qual=0.839)
    RunSpec(22, "t22_v1_full_depth_eroded",    "full_depth_ring",
        {"arch": "v1", "batch-size": 64, "ring-label-source": "eroded",
         "train-d-start": 28, "train-d-end": 48},
        why="v1 CNN, eroded ring, full depth 28-48. step=4 gives windows 28-36/32-40/36-44/40-48 "
            "— same depth diversity as original arch_search (f1~0.34, easy~0.54). "
            "does wider depth coverage recover arch_search performance?"),

    RunSpec(23, "t23_v1_full_depth_original",  "full_depth_ring",
        {"arch": "v1", "batch-size": 64, "ring-label-source": "original",
         "train-d-start": 28, "train-d-end": 48},
        why="v1 CNN, original ring (best sanity qual=0.858), full depth 28-48. "
            "combining cleanest ring with diverse depth stepping. expect best of all variants."),

    RunSpec(24, "t24_v1_full_depth_closed",    "full_depth_ring",
        {"arch": "v1", "batch-size": 64, "ring-label-source": "closed",
         "train-d-start": 28, "train-d-end": 48},
        why="v1 CNN, closed ring (0% contamination, air gap), full depth 28-48. "
            "closed had lowest sanity quality but best balance. "
            "full depth diversity may compensate for the smaller ring."),
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
            print(f"\n[MONITOR] STALL")
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
        if key == "no-ring-negatives":
            pass  # handled below
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
    m = {"valid_f1_last": None, "probe_easy_last": None, "probe_hard_last": None,
         "valid_pauc_last": None}
    if run_dir is None or event_accumulator is None: return m
    evts = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not evts: return m
    ea = event_accumulator.EventAccumulator(str(evts[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    tag_map = {
        "valid_f1":    "P_M/F1_Score/Valid",
        "probe_easy":  "R_M/Probe/Easy/ReadabilityComposite",
        "probe_hard":  "R_M/Probe/Hard/ReadabilityComposite",
        "valid_pauc":  "P_M/PAUC/Valid",
    }
    for key, tag in tag_map.items():
        if tag in avail:
            vals = [e.value for e in ea.Scalars(tag)]
            m[f"{key}_last"] = vals[-1]
    return m


def quality_score(metrics: dict) -> float:
    """combined quality: F1 + probe_easy (both in [0,1], higher is better)"""
    f1   = metrics.get("valid_f1_last")   or 0.0
    easy = metrics.get("probe_easy_last") or 0.0
    return float(f1) + float(easy)


def print_summary(completed, label="campaign 9"):
    if not completed: return
    print(f"\n+-- {label} results (ranked by quality = F1 + easy probe) ---")
    print(f"|  {'run':<44} {'hard':>5} {'easy':>5} {'f1':>5} {'qual':>6}")
    print("|  " + "-" * 65)
    for r in sorted(completed,
                    key=lambda r: quality_score(r.get("metrics") or {}),
                    reverse=True):
        m = r.get("metrics") or {}
        hard = f"{m.get('probe_hard_last',0.0):.3f}" if m.get("probe_hard_last") is not None else "?"
        easy = f"{m.get('probe_easy_last',0.0):.3f}" if m.get("probe_easy_last") is not None else "?"
        f1   = f"{m.get('valid_f1_last',0.0):.3f}"  if m.get("valid_f1_last")   is not None else "?"
        qual = f"{quality_score(m):.3f}"
        print(f"|  {r['name'][-44:]:<44} {hard:>5} {easy:>5} {f1:>5} {qual:>6}")
    print("+--" + "-" * 67 + "\n")


def main():
    parser = argparse.ArgumentParser(description="campaign 9 — sanity-gated ring + full v8")
    parser.add_argument("--campaign-id",    type=str, default="c9_2026_06_14")
    parser.add_argument("--python-exe",     type=str, default=sys.executable)
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--stall-minutes",  type=float, default=60.0)
    parser.add_argument("--default-ring",   type=str,  default="closed",
                        choices=["eroded","original","closed"],
                        help="ring source to use if sanity tests haven't run yet")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_campaign9"
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
    base = dict(BASE)
    base["log-dir"] = str(runs_dir)

    def is_done(run_id):
        done_ids = {int(r["run_id"]) for r in state.get("completed", [])}
        fail_ids = {int(r["run_id"]) for r in state.get("failed", [])}
        return run_id in done_ids or run_id in fail_ids

    def run_one(run_id, name, overrides, axis=""):
        exp_name = f"cmp_{args.campaign_id}_{name}"
        log_path = state_dir / f"{exp_name}.log"
        merged   = dict(base); merged.update(overrides)
        # handle no-ring-negatives flag — rebuild base without ring params
        if overrides.get("no-ring-negatives"):
            merged = {k: v for k, v in base.items()
                      if k not in ("ring-negatives", "ring-label-source")}
            merged.update({k: v for k, v in overrides.items()
                           if k != "no-ring-negatives"})
        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)
        print(f"\n{'='*60}")
        print(f"  run {run_id:02d}: {name}  [{axis}]")
        print(f"  overrides: {overrides}")
        print(f"{'='*60}")
        start_ts = time.time()
        rc, crashed = (0, False) if args.dry_run else run_with_monitoring(
            cmd, repo_root, env, str(log_path), args.stall_minutes)
        run_dir = find_run_dir(runs_dir, exp_name, start_ts)
        metrics = extract_metrics(run_dir)
        rec = {"run_id": run_id, "name": exp_name, "axis": axis,
               "overrides": merged, "run_dir": str(run_dir) if run_dir else None,
               "metrics": metrics, "ended_at": now_utc()}
        hard = metrics.get("probe_hard_last"); easy = metrics.get("probe_easy_last")
        f1   = metrics.get("valid_f1_last")
        print(f"\n  RESULT: hard={hard}  easy={easy}  f1={f1}  quality={quality_score(metrics):.3f}")
        if rc == 0:
            state.setdefault("completed", []).append(rec)
        else:
            rec.update({"return_code": rc, "crashed_early": crashed})
            state.setdefault("failed", []).append(rec)
        if not args.dry_run:
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return metrics, rc == 0

    # ── Phase 1: sanity tests (runs 1-3) ──────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1: Sanity tests — v1 CNN with 3 ring sources (20 epochs)")
    print("  Target: outperform C7 t21 (easy=0.557, hard=0.440, f1=0.203)")
    print("="*60)

    sanity_results: Dict[str, dict] = {}
    for idx, src in enumerate(SANITY_SOURCES, start=1):
        if is_done(idx):
            rec = next((r for r in state.get("completed", []) + state.get("failed", [])
                        if r["run_id"] == idx), None)
            if rec:
                sanity_results[src] = rec.get("metrics", {})
                print(f"  [skip] sanity {src} already done — quality={quality_score(rec.get('metrics',{})):.3f}")
            continue
        m, ok = run_one(idx, f"sanity_v1_{src}_ring", {
            "arch": "v1", "batch-size": 64, "ring-label-source": src,
        }, axis="sanity")
        sanity_results[src] = m
        print_summary([r for r in state.get("completed", []) if r["run_id"] <= 3],
                      label="sanity results so far")

    # pick best ring source
    best_ring = max(sanity_results, key=lambda s: quality_score(sanity_results[s]),
                    default=args.default_ring)
    best_qual = quality_score(sanity_results.get(best_ring, {}))
    print(f"\n*** Best ring source: '{best_ring}'  (quality={best_qual:.3f})")
    print(f"  → all remaining runs will use ring-label-source={best_ring}\n")

    # C7 t21 reference for comparison
    C7_QUAL = 0.557 + 0.203  # easy + f1 from C7 t21
    if best_qual < C7_QUAL:
        print(f"  ⚠ best sanity ({best_qual:.3f}) < C7 t21 ({C7_QUAL:.3f}) — proceeding anyway")

    # ── Phase 2: v8 architecture suite ────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  PHASE 2: v8 architecture suite  [ring-label-source={best_ring}]")
    print("="*60)

    # inject winning ring into base overrides
    base["ring-label-source"] = best_ring

    for spec in V8_RUNS:
        run_id = spec.run_id + 3  # offset: 1-3 are sanity, 4+ are v8
        if is_done(run_id):
            print(f"  [skip] {spec.name} already done")
            continue
        print(f"\n  {spec.why}")
        run_one(run_id, spec.name, dict(spec.overrides), axis=spec.axis)
        print_summary(state.get("completed", []))

    print_summary(state.get("completed", []), label="campaign 9 final")
    print("campaign 9 finished")


if __name__ == "__main__":
    main()
