"""campaign_runner_8.py — novel architectures, all trained with ring negatives.

key insights from C5-C7:
- ring negatives (tile-level, 1:1 ratio) give cleanest training signal
- pos_weight recomputed fresh per run (not cached) → no overbright predictions
- focal loss and hard mining unnecessary with balanced ring dataset
- LSTM (hard=0.445), BiGRU (0.436), ring_v1_cnn (easy=0.557) are best so far
- hard examples show NO signal in any model — sub-voxel physics limit

C8 strategy: 16 novel architectures + 4 combinations.
all use ring negatives + recomputed pos_weight. no focal loss. no hard mining.
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
    expect: str


SMALL_SCROLL_ID = 20230827161847
SCROLL4_ID      = 20231210132040

BASE_OVERRIDES: Dict[str, Any] = {
    "epochs": 20,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 512,
    "num-workers": 0,          # safe on Windows
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 45,
    "no-hard-mining": True,    # ring provides balanced training; no hard mining needed
    "ring-negatives": True,    # tile-level ring, 1:1 pos/neg, corrected pos_weight
    "ring-label-source": "closed",  # CLOSE+GAP morphology, 0% contamination, best sanity result
    "channel-mixing-prob": 0.0,
    "conv1-drop": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "l1-lambda": 0.0,
}

RUN_SPECS: List[RunSpec] = [

    # A: matched filter / template approaches
    RunSpec(1, "t01_matched_filter", "matched_filter",
        {"arch": "v8_matched_filter"},
        why="K=64 learned ink profile templates. cosine similarity to all templates → MLP. "
            "the optimal detector for a known signal shape in Gaussian noise. "
            "learns exactly what ink depth curves look like and scores by match quality.",
        expect="strong if ink has a consistent depth profile shape; diagnostic for template learning"),

    # B: percentile-enhanced sequential
    RunSpec(2, "t02_percentile_bigru", "percentile_seq",
        {"arch": "v8_percentile_bigru"},
        why="4-layer BiGRU on percentile-feature sequences (D×5 features). "
            "combines C7's best feature (percentile robustness for sparse ink pixels) "
            "with deeper recurrent modeling. ring removes contamination from negatives.",
        expect="best of C7+C8; should outperform v7_bigru_percentile with ring + deeper GRU"),

    RunSpec(3, "t03_pairwise_bigru", "percentile_seq",
        {"arch": "v8_pairwise_bigru"},
        why="3-layer BiGRU on pairwise depth differences. scale-invariant comparisons "
            "fed to a sequential model. v7_pairwise scored 0.414 (non-ring). "
            "adds sequential depth modeling to scale-invariant features.",
        expect="improvement over v7_pairwise; sequential integration of scale-invariant features"),

    # C: physics-based signal processing
    RunSpec(4, "t04_diff_of_gaussians", "physics",
        {"arch": "v8_diff_of_gaussians"},
        why="DoG filter applied to depth profile at 3 scales. DoG is the mathematically "
            "optimal blob detector. ink creates a bump (bell curve) in the depth profile — "
            "exactly what DoG is designed to find. multi-scale captures bumps of different widths.",
        expect="unique signal; if ink bump is detectable, DoG should find it optimally"),

    RunSpec(5, "t05_laplacian_depth", "physics",
        {"arch": "v8_laplacian_depth"},
        why="Laplacian (2nd derivative) of depth profile. maximally sensitive to curvature. "
            "ink bump → large negative Laplacian at peak + biphasic flanking pattern. "
            "flat background → near-zero Laplacian. different from 1st derivative (C6).",
        expect="strong signal at ink depths; Laplacian amplifies the bump shape"),

    RunSpec(6, "t06_wavelet_depth", "physics",
        {"arch": "v8_wavelet_depth"},
        why="Haar wavelet decomposition: approximation (smooth) + detail (edge) coefficients. "
            "ink absorption edges (rise at z=32, fall at z=40) are high-frequency transients — "
            "exactly what wavelet detail coefficients detect. unlike FFT, wavelets are localized.",
        expect="if ink edge is detectable, wavelets will find it; novel signal representation"),

    RunSpec(7, "t07_robust_zscore", "physics",
        {"arch": "v8_robust_zscore"},
        why="robust z-score: (profile - median) / IQR applied before 1D CNN. "
            "makes model shift- and scale-invariant. IQR is robust to outliers. "
            "the z-scored profile encodes depth SHAPE ONLY, removing brightness confounds.",
        expect="cleaner depth shape signal; normalization may reveal subtle ink bumps"),

    RunSpec(8, "t08_absorption_ratio", "physics",
        {"arch": "v8_absorption_ratio"},
        why="physics-motivated: centered profile (subtract tile mean) + ratio (depth/mean). "
            "two explicit scale-invariant representations of the depth absorption curve. "
            "combined in a deep MLP. directly measures what ink physically does.",
        expect="strong physics alignment; combined features should be discriminative"),

    # D: spatial-depth hybrid
    RunSpec(9, "t09_spatial_contrast", "spatial_depth",
        {"arch": "v8_spatial_contrast"},
        why="tile's depth profile contrasted against its 4 quadrants. ink is spatially "
            "localized — quadrants containing ink differ from empty quadrants. "
            "BiGRU over the 5-feature (tile+4 quadrants) contrast sequence through depth. "
            "finds spatial heterogeneity within the 32x32 tile at each depth.",
        expect="captures within-tile spatial patterns that pure depth-average misses"),

    RunSpec(10, "t10_tile_entropy", "spatial_depth",
        {"arch": "v8_tile_entropy"},
        why="soft spatial entropy at each depth slice. ink tiles are spatially heterogeneous "
            "(only ~5-10% pixels hit ink) → HIGH entropy. background is uniform → LOW entropy. "
            "entropy + mean profile → BiGRU. explicitly measures spatial heterogeneity "
            "that sub-voxel ink particles should create.",
        expect="entropy directly measures the spatial anomaly ink creates; novel signal"),

    RunSpec(11, "t11_superpixel_bigru", "spatial_depth",
        {"arch": "v8_superpixel_bigru"},
        why="4×4 superpixel profiles → 3-layer BiGRU in raster order. a letter stroke "
            "spans ~2-4 superpixels; the GRU accumulates the contiguous run of elevated "
            "superpixel profiles as it scans left-to-right, top-to-bottom. "
            "different from t12 (transformer); recurrent captures ordered spatial pattern.",
        expect="captures spatial continuity of ink strokes at the superpixel scale"),

    RunSpec(12, "t12_multiscale_percentile", "spatial_depth",
        {"arch": "v8_multiscale_percentile"},
        why="percentile features at 3 spatial scales: full tile + 4 quadrants + 16 cells. "
            "(1+4+16)×5×D features. ink patch size varies; different scales capture "
            "ink at the scale that matches the patch. most information-rich percentile model.",
        expect="robust to variable ink patch sizes; most complete percentile representation"),

    RunSpec(13, "t13_residual_spatial_depth", "spatial_depth",
        {"arch": "v8_residual_spatial_depth"},
        why="per-slice 2D CNN (32→64 channels, global pool to 4×4) + depth attention. "
            "encodes SPATIAL features at each depth independently, then attention-weights "
            "across depths. different from 3D CNNs: learns pure 2D spatial features first, "
            "then combines. better separation of spatial and depth signals.",
        expect="richer spatial features per depth than 3D conv; attention finds key depths"),

    # E: deep sequential
    RunSpec(14, "t14_deep_bigru", "deep_sequential",
        {"arch": "v8_deep_bigru"},
        why="6-layer BiGRU with 512 hidden on 8-depth profile. C6 1-layer 256-hidden "
            "BiGRU got 0.419; ring improved to 0.436. 6× deeper with 2× wider. "
            "deep recurrent models can represent exponentially more complex patterns. "
            "25M params — the largest sequential model tried.",
        expect="best sequential result; depth of model should capture subtle depth patterns"),

    RunSpec(15, "t15_fulldepth_transformer16", "deep_sequential",
        {"arch": "v8_fulldepth_transformer16", "input-mode": "fulldepth", "batch-size": 64},
        why="16-layer transformer on full 64-depth spatial-mean profile. "
            "combines (1) full absorption curve, (2) very deep attention, "
            "(3) ring clean training, (4) recomputed pos_weight. 12.6M params. "
            "deepest transformer tried on depth profiles.",
        expect="may find complex multi-depth relationships invisible to shallower models"),

    RunSpec(16, "t16_full64_pct_bigru", "deep_sequential",
        {"arch": "v8_full64_pct_bigru", "input-mode": "fulldepth", "batch-size": 64},
        why="percentile-sequence 3-layer BiGRU on FULL 64-depth profile. "
            "combines: (1) complete absorption curve, (2) percentile robustness, "
            "(3) sequential depth modeling, (4) ring clean training. "
            "most information-rich approach that also handles sparse ink pixels robustly.",
        expect="potentially best overall: full curve + percentile + sequential + clean ring"),

    # F: best combinations with diff input (no focal needed with ring)
    RunSpec(17, "t17_matched_filter_diff", "combo",
        {"arch": "v8_matched_filter", "input-mode": "diff"},
        why="matched filter templates on differential signal (ink - pre_band). "
            "differential removes scroll baseline; matched filter then detects ink-shaped "
            "bumps in the baseline-subtracted profile. two-stage signal enhancement.",
        expect="cleaner template matching when baseline removed"),

    RunSpec(18, "t18_percentile_bigru_diff", "combo",
        {"arch": "v8_percentile_bigru", "input-mode": "diff"},
        why="percentile BiGRU on differential input. removes baseline, then "
            "percentile-robust sequential modeling of the residual ink signal.",
        expect="cleanest possible combination: baseline removal + percentile + sequential"),

    RunSpec(19, "t19_pairwise_bigru_fulldepth", "combo",
        {"arch": "v8_pairwise_bigru", "input-mode": "fulldepth",
         "batch-size": 64},
        why="pairwise BiGRU on all C(64,2) pairwise depth differences in full 64-depth. "
            "2016 pairwise features (vs 28 for 8-depth) → BiGRU. full absorption curve "
            "in scale-invariant representation.",
        expect="pairwise scale-invariance + full absorption curve; novel combination"),

    RunSpec(20, "t20_spatial_contrast_ring_30ep", "combo",
        {"arch": "v8_spatial_contrast", "epochs": 30},
        why="spatial_contrast (within-tile quadrant deviation) with 30 epochs instead of 20. "
            "spatial contrast is a fundamentally novel signal that may need more epochs "
            "to converge. ring negatives ensure stable balanced training throughout.",
        expect="extended training may reveal subtle spatial-depth patterns"),

    # SANITY CHECKS: v1 CNN with eroded ring (bug) vs original ring (fix)
    # if the original-ring run shows letters and eroded-ring does not, bug is confirmed
    RunSpec(21, "t21_sanity_v1_eroded_ring", "sanity",
        {"arch": "v1", "batch-size": 64, "ring-label-source": "eroded"},
        why="SANITY: v1 3D CBAM CNN with ERODED ring (old buggy behavior). "
            "20.9% of ring tiles contain original ink but labeled negative. "
            "expect: model may predict entire scroll as ink (washed out).",
        expect="reproduces the washed-out bug if eroded ring is the cause"),

    RunSpec(22, "t22_sanity_v1_original_ring", "sanity",
        {"arch": "v1", "batch-size": 64, "ring-label-source": "original"},
        why="SANITY: v1 3D CBAM CNN with ORIGINAL ring (proposed fix). "
            "ring negatives have 0% contamination from original ink. "
            "should show clear letter contrast similar to C7 t21.",
        expect="clear letters visible, validates original-ring fix"),

    RunSpec(23, "t23_sanity_v1_closed_ring", "sanity",
        {"arch": "v1", "batch-size": 64, "ring-label-source": "closed"},
        why="SANITY: v1 3D CBAM CNN with CLOSED ring (dil+erode to close letter holes, air gap). "
            "Large dilation (r=5 tiles=160px) closes letter interior holes, "
            "then erosion restores boundary with guaranteed air gap before ring starts. "
            "most conservative: ring tiles are far from any ink edge.",
        expect="best contrast if air gap prevents any boundary confusion"),
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
    print("\n+-- campaign 8 results (ranked by hard probe) ----------------------")
    print(f"|  {'run':<42} {'hard':>5} {'easy':>5} {'f1':>5}")
    print("|  " + "-" * 58)
    for r in sorted(completed,
                    key=lambda r: (r.get("metrics") or {}).get("probe_hard_last") or 0,
                    reverse=True):
        m = r.get("metrics") or {}
        hard = f"{m.get('probe_hard_last',0.0):.3f}" if m.get("probe_hard_last") is not None else "?"
        easy = f"{m.get('probe_easy_last',0.0):.3f}" if m.get("probe_easy_last") is not None else "?"
        f1   = f"{m.get('valid_f1_last',0.0):.3f}"  if m.get("valid_f1_last")   is not None else "?"
        print(f"|  {r['name'][-42:]:<42} {hard:>5} {easy:>5} {f1:>5}")
    print("+--" + "-" * 62 + "\n")


def choose_next(pending, completed):
    if len(pending) == 1: return pending[0]
    axis_score: Dict[str, float] = {}
    axis_count: Dict[str, int] = {}
    for rec in completed:
        m = rec.get("metrics") or {}
        h = m.get("probe_hard_last")
        a = rec.get("axis")
        if a is None or h is None: continue
        axis_score[a] = axis_score.get(a, 0.0) + float(h)
        axis_count[a] = axis_count.get(a, 0) + 1
    for a in list(axis_score): axis_score[a] /= max(axis_count[a], 1)
    return sorted(pending, key=lambda s: (-(axis_score.get(s.axis, 0.3)), s.run_id))[0]


def main():
    parser = argparse.ArgumentParser(description="campaign 8 -- sub-voxel ink push with ring negatives")
    parser.add_argument("--campaign-id", type=str, default="c8_2026_06_14")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=23)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=60.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_campaign8"
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

    base = dict(BASE_OVERRIDES)
    base["log-dir"] = str(runs_dir)
    target = min(args.max_runs, len(RUN_SPECS))

    while True:
        completed_records = state.get("completed", [])
        completed_ids = {int(r["run_id"]) for r in completed_records}
        failed_ids    = {int(r["run_id"]) for r in state.get("failed", [])}
        pending = [s for s in RUN_SPECS
                   if s.run_id not in completed_ids and s.run_id not in failed_ids]
        if not pending or len(completed_records) >= target: break

        print_summary(completed_records)
        spec = choose_next(pending, completed_records)
        merged = dict(base); merged.update(spec.overrides)
        exp_name = f"cmp_{args.campaign_id}_{spec.name}"
        log_path = state_dir / f"{exp_name}.log"

        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)
        print(f"\n{'='*60}")
        print(f"  run {spec.run_id:02d}/20: {spec.name}  [{spec.axis}]")
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
        pending_after = [s for s in pending if s.run_id != spec.run_id]
        next_spec = choose_next(pending_after, completed_records + [
            {"run_id": spec.run_id, "axis": spec.axis, "metrics": metrics}
        ]) if pending_after else None
        next_name = "none" if next_spec is None else f"{next_spec.run_id:02d}:{next_spec.name}"

        hard = metrics.get("probe_hard_last")
        easy = metrics.get("probe_easy_last")
        f1   = metrics.get("valid_f1_last")
        print(f"\n  RESULT: hard={hard}  easy={easy}  f1={f1}")
        print(f"  next -> {next_name}")

        if rc == 0:
            state.setdefault("completed", []).append({
                "run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
                "overrides": merged, "run_dir": str(run_dir) if run_dir else None,
                "metrics": metrics, "ended_at": now_utc(), "next_planned": next_name,
            })
        else:
            state.setdefault("failed", []).append({
                "run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
                "overrides": merged, "ended_at": now_utc(), "return_code": rc,
                "metrics": metrics, "crashed_early": crashed,
            })
        if not args.dry_run:
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print_summary(state.get("completed", []))
    print("campaign 8 finished")


if __name__ == "__main__":
    main()
