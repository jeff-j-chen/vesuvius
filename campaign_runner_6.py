"""campaign_runner_6.py — depth profile architecture push

C5 key finding: depth profile 1D CNN (hard=0.360) and Transformer (hard=0.372)
beat ALL 3D/MIL/attention approaches. the absorption curve shape through depth
is the ink signal. campaign 6 exploits this radically further:

  - full 64-depth profiles (8x more info than campaign 5's 8-depth)
  - per-pixel depth profiles (no spatial averaging — find exact ink pixels)
  - sequential models (LSTM/GRU/Transformer) designed for curve shape
  - physics transforms (Beer-Lambert, derivative)
  - multi-scale aggregation and PCA decomposition

20 architectures, monitored one at a time. runs adapt based on results.
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
    expect: str


SMALL_SCROLL_ID = 20230827161847
SCROLL4_ID      = 20231210132040

BASE_OVERRIDES: Dict[str, Any] = {
    "epochs": 20,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 512,   # fulldepth models are tiny (150K params); large batch amortizes 8x zarr IO
    "num-workers": 4,    # more parallel zarr readers to keep GPU fed
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 45,
    "hm-frac": 0.05,
    "channel-mixing-prob": 0.0,
    "conv1-drop": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "l1-lambda": 0.0,
    "no-hard-mining": True,
}

RUN_SPECS: List[RunSpec] = [

    # A: per-pixel profiles (8-depth) — no spatial averaging
    RunSpec(1, "t01_perpixel_1d", "perpixel",
        {"arch": "v6_perpixel_1d"},
        why="per-pixel 8-depth 1D CNN + MIL attention. unlike C5 depth profile models "
            "that average 32x32 spatially first, this processes each pixel independently. "
            "if ink covers 5% of the tile, this finds those 5% of pixels.",
        expect="hard probe improvement over C5 depth profile 1D; spatial precision matters"),
    RunSpec(2, "t02_perpixel_gated", "perpixel",
        {"arch": "v6_perpixel_gated"},
        why="per-pixel profiles + gated MIL (Ilse 2018). more stable training than vanilla "
            "softmax attention under the extreme positive-class rarity of ink tiles.",
        expect="most stable per-pixel result; comparable or better than t01"),
    RunSpec(3, "t03_perpixel_max", "perpixel",
        {"arch": "v6_perpixel_max"},
        why="per-pixel profiles + HARD max: the single most ink-shaped pixel drives the prediction. "
            "no averaging or attention weighting. if ONE pixel has ink, tile = ink.",
        expect="high recall, potentially noisy; tests extreme of the per-pixel hypothesis"),
    RunSpec(4, "t04_perpixel_local_sub", "perpixel",
        {"arch": "v6_perpixel_local_sub"},
        why="per-pixel RESIDUAL profiles: each pixel minus the tile spatial mean. "
            "removes the shared scroll background; residual = pure per-pixel deviation. "
            "amplifies the small ink signal above the local baseline.",
        expect="cleanest per-pixel ink signal; best hard probe of perpixel tier if residual helps"),

    # B: full 64-depth profile baseline
    RunSpec(5, "t05_fulldepth_1d", "fulldepth",
        {"arch": "v6_fulldepth_1d", "input-mode": "fulldepth", "batch-size": 64},
        why="1D CNN on the FULL 64-depth spatial-mean absorption profile. "
            "C5 used only 8 of 64 available depth slices. the full profile reveals "
            "the complete ink absorption bell curve (baseline -> rise -> peak -> fall -> baseline).",
        expect="substantially better hard probe than C5 1D CNN; the full curve shape is the key signal"),

    # C: spatial self-attention (NEW t06/t07 — inserted after t05)
    #
    # motivation: ink at 7.91um is sub-voxel, but ink STROKES are 12-63 voxels wide.
    # many adjacent pixels each carry a weak partial-voxel absorption signal. MIL
    # treats pixels independently; self-attention lets each pixel ask "do my neighbors
    # also look anomalous?" — the right inductive bias for detecting correlated
    # sub-voxel ink traces across a letter stroke.
    RunSpec(6, "t06_pixel_spatial_attn", "spatial_attn",
        {"arch": "v6_pixel_spatial_attn", "batch-size": 64},
        why="full spatial self-attention over per-pixel depth profiles. each of 1024 pixels "
            "in the 32x32 tile gets its 8-depth profile (z=32-40, peak ink band, one zarr block) "
            "encoded by a weight-shared MLP into a 512-dim token. 8-layer 8-head transformer lets all "
            "pixels compare with each other — learns that a CLUSTER of mildly-elevated pixels "
            "is ink (stroke), not noise. d=512 fills ~8-9GB VRAM via flash attention.",
        expect="hard probe improvement if correlated multi-pixel signal is the key ink feature"),
    RunSpec(7, "t07_pixel_local_attn", "spatial_attn",
        {"arch": "v6_pixel_local_attn", "batch-size": 64},
        why="local window attention: 32x32 tile split into 16 non-overlapping 8x8 windows. "
            "4-layer local attention within each window (64 tokens = ~63um, ink stroke width). "
            "4-layer global attention across 16 window summaries. d=512 throughout. "
            "uses peak ink band z=32-40 (8 slices, one zarr block). ~6-7 GB VRAM at batch=64.",
        expect="similar or better than t06; local context is more focused on relevant spatial scale"),

    # C: full 64-depth profiles
    RunSpec(8, "t08_fulldepth_transformer", "fulldepth",
        {"arch": "v6_fulldepth_transformer", "input-mode": "fulldepth", "batch-size": 64},
        why="Transformer over 64 depth positions. C5 Transformer over 8 positions scored 0.372; "
            "with 64 tokens the self-attention can span the full ink absorption profile. "
            "4 layers and 8 heads can learn the characteristic rise-peak-fall pattern.",
        expect="best hard probe of the campaign if curve shape is the key feature"),
    RunSpec(9, "t09_fulldepth_gru", "fulldepth",
        {"arch": "v6_fulldepth_gru", "input-mode": "fulldepth", "batch-size": 64},
        why="bidirectional GRU over 64 depth positions. bidirectional because ink absorption "
            "is symmetric (entry edge + exit edge equally informative). lighter than transformer.",
        expect="strong; may outperform transformer on this sequential task"),
    RunSpec(10, "t10_fulldepth_perpixel", "fulldepth",
        {"arch": "v6_fulldepth_perpixel", "input-mode": "fulldepth", "batch-size": 32},
        why="per-pixel FULL 64-depth profiles + gated MIL. most information-rich approach: "
            "every pixel's complete absorption curve through all 64 depths, with attention "
            "selecting the ink-like pixels. if ink is at even 1 pixel, its 64-depth profile "
            "should be unambiguously distinguishable from background.",
        expect="potentially the best result of all campaigns; combines per-pixel locality with full depth"),
    RunSpec(11, "t11_fulldepth_deriv", "fulldepth",
        {"arch": "v6_fulldepth_deriv", "input-mode": "fulldepth", "batch-size": 64},
        why="first derivative of full 64-depth profile. with 64 values, the ink absorption "
            "edge (rise then fall) is clearly resolved as a biphasic derivative signal. "
            "background profiles have near-zero derivative; ink profiles have distinct peaks.",
        expect="may be more discriminative than raw profile; derivative highlights edges"),

    # D: physics-motivated transforms
    RunSpec(12, "t12_depth_derivative", "physics",
        {"arch": "v6_depth_derivative"},
        why="depth derivative of 8-slice spatial-mean profile. ink creates absorption edges; "
            "derivative converts bell curve to biphasic (positive + negative) edge signal.",
        expect="different failure mode than raw profile; may help hard ROI"),
    RunSpec(13, "t13_beer_lambert", "physics",
        {"arch": "v6_beer_lambert"},
        why="Beer-Lambert log transform: -log(I) converts relative intensities to linear "
            "attenuation coefficients where material absorption differences are more linear. "
            "ink vs papyrus may separate more cleanly in log-attenuation space.",
        expect="moderate improvement if the log transform linearizes the ink/background boundary"),
    RunSpec(14, "t14_robust_stats", "physics",
        {"arch": "v6_robust_stats"},
        why="2-channel profile: spatial mean + spatial std at each depth. "
            "std captures: at ink depths, absorption is spatially heterogeneous "
            "(some voxels have carbon, others don't). mean + std together should be more discriminative.",
        expect="improved hard probe if spatial heterogeneity is a useful ink indicator"),

    # E: sequential models over 8 depth slices
    RunSpec(15, "t15_lstm_slices", "sequential",
        {"arch": "v6_lstm_slices"},
        why="LSTM over 8 depth slices, each encoded by 2D conv. the LSTM captures: "
            "'how does absorption change as we go deeper?' ink creates a systematic "
            "increase-then-decrease pattern. the recurrent state accumulates this pattern.",
        expect="captures cross-depth temporal pattern; different from pure 1D profile on mean"),
    RunSpec(16, "t16_bigru_slices", "sequential",
        {"arch": "v6_bigru_slices"},
        why="bidirectional GRU over depth slices. captures both entry and exit absorption edges. "
            "lighter than LSTM, same sequential pattern-finding capability.",
        expect="comparable to LSTM with faster training; bidirectional may help symmetric ink profile"),
    RunSpec(17, "t17_slice_attention", "sequential",
        {"arch": "v6_slice_attention"},
        why="multi-head self-attention across 8 depth slices. unlike LSTM which is sequential, "
            "attention directly compares any two depths simultaneously. can learn: "
            "'this depth is high relative to depths 2 and 3 slices away' = ink peak pattern.",
        expect="similar to C5 depth transformer but on full 2D slice features vs spatial mean"),

    # F: multi-resolution and PCA
    RunSpec(18, "t18_triple_scale", "multiscale",
        {"arch": "v6_triple_scale"},
        why="profiles at 3 spatial granularities: whole tile + 2x2 quadrants + 4x4 blocks. "
            "if ink covers only part of a tile, coarser/finer spatial averaging reveals "
            "the ink signal at the scale that matches the ink patch size.",
        expect="improved hard probe if ink patch size varies; multi-scale is robust"),
    RunSpec(19, "t19_profile_pca", "pca",
        {"arch": "v6_profile_pca"},
        why="learnable PCA basis for depth profiles: N basis vectors learned jointly with "
            "classification. ink profiles project differently than background onto learned basis. "
            "much smaller model (1345 params) - forced to be maximally efficient with the signal.",
        expect="diagnostic: if tiny model achieves good hard probe, the profile is highly discriminative"),

    # G: best combinations with diff input
    RunSpec(20, "t20_perpixel_gated_diff", "combo",
        {"arch": "v6_perpixel_gated", "input-mode": "diff"},
        why="per-pixel gated MIL on diff input (ink - pre_band). two independent signal improvements: "
            "diff removes scroll baseline from each pixel's profile, gated MIL finds the most ink-like pixels. "
            "combines physics preprocessing with spatial localization.",
        expect="strong if both diff input and per-pixel analysis are complementary"),
    RunSpec(21, "t21_fulldepth_1d_diff", "combo",
        {"arch": "v6_fulldepth_1d", "input-mode": "diff", "batch-size": 64},
        why="full 64-depth profile of the DIFFERENTIAL signal (ink - pre_band). "
            "if the ink absorption is small relative to scroll baseline, "
            "seeing the full baseline-subtracted curve may reveal a clearer ink peak.",
        expect="potentially strongest result: full context + background removal"),
    RunSpec(22, "t22_robust_stats_diff", "combo",
        {"arch": "v6_robust_stats", "input-mode": "diff"},
        why="mean+std depth profile on differential signal. the std of (ink-pre) highlights "
            "positions where the depth absorption variation is different from background - "
            "a very specific physical signature of differential ink absorption.",
        expect="novel signal combination; tests whether diff heterogeneity is detectable"),
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
]


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=45):
    """run subprocess, tail the log, detect crashes, report progress."""
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
    last_progress = time.time()
    last_epoch = 0
    while proc.poll() is None:
        time.sleep(15)
        try:
            lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        except Exception:
            continue
        tail_text = "".join(lines[-40:])
        for sig in CRASH_SIGNALS:
            if sig in tail_text:
                print(f"\n[MONITOR] CRASH -- '{sig}'")
                print("[MONITOR] last output:\n" + "".join(lines[-15:]))
                try: proc.kill()
                except Exception: pass
                proc.wait()
                return proc.returncode or 1, True
        for line in lines[-40:]:
            if "--- Epoch" in line:
                try:
                    ep = int(line.strip().split("/")[0].split()[-1])
                    if ep > last_epoch:
                        last_epoch = ep
                        last_progress = time.time()
                        print(f"[MONITOR] {line.strip()}")
                except Exception: pass
        if time.time() - last_progress > stall_minutes * 60:
            print(f"\n[MONITOR] STALL -- no epoch progress in {stall_minutes} min")
            try: proc.kill()
            except Exception: pass
            proc.wait()
            return 1, True
    proc.wait()
    rc = proc.returncode
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
    m = {"valid_f1_best": None, "valid_f1_last": None,
         "readability_best": None, "readability_last": None,
         "probe_easy_last": None, "probe_hard_last": None}
    if run_dir is None or event_accumulator is None: return m
    evts = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not evts: return m
    ea = event_accumulator.EventAccumulator(str(evts[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    for key, tag in [("valid_f1", "P_M/F1_Score/Valid"),
                     ("readability", "R_M/ReadabilityComposite"),
                     ("probe_easy", "R_M/Probe/Easy/ReadabilityComposite"),
                     ("probe_hard", "R_M/Probe/Hard/ReadabilityComposite")]:
        if tag in avail:
            vals = [e.value for e in ea.Scalars(tag)]
            m[f"{key}_best"] = max(vals); m[f"{key}_last"] = vals[-1]
    return m


def append_text(path, text):
    with path.open("a", encoding="utf-8") as f: f.write(text)


def print_summary(completed):
    if not completed: return
    print("\n+-- campaign 6 results (ranked by hard probe) ---------------------")
    print(f"|  {'run':<40} {'hard':>5} {'easy':>5} {'f1':>5}")
    print("|  " + "-" * 56)
    for r in sorted(completed, key=lambda r: (r.get("metrics") or {}).get("probe_hard_last") or 0, reverse=True):
        m = r.get("metrics") or {}
        hard = f"{m.get('probe_hard_last',0.0):.3f}" if m.get("probe_hard_last") is not None else "?"
        easy = f"{m.get('probe_easy_last',0.0):.3f}" if m.get("probe_easy_last") is not None else "?"
        f1   = f"{m.get('valid_f1_last',0.0):.3f}"  if m.get("valid_f1_last")   is not None else "?"
        print(f"|  {r['name'][-40:]:<40} {hard:>5} {easy:>5} {f1:>5}")
    print("+--" + "-" * 62 + "\n")


def choose_next(pending, completed):
    if len(pending) == 1: return pending[0]
    baseline = next((r for r in completed if r.get("run_id") == 5), None)  # fulldepth_1d as baseline
    if baseline is None:
        return sorted(pending, key=lambda s: s.run_id)[0]
    base_hard = (baseline.get("metrics") or {}).get("probe_hard_last") or 0.0
    axis_score: Dict[str, float] = {}
    axis_count: Dict[str, int]   = {}
    for rec in completed:
        m = rec.get("metrics") or {}
        h = m.get("probe_hard_last")
        a = rec.get("axis")
        if a is None or h is None: continue
        axis_score[a] = axis_score.get(a, 0.0) + float(h) - float(base_hard)
        axis_count[a] = axis_count.get(a, 0) + 1
    for a in list(axis_score): axis_score[a] /= max(axis_count[a], 1)
    return sorted(pending, key=lambda s: (-(axis_score.get(s.axis, 0.0)), s.run_id))[0]


def main():
    parser = argparse.ArgumentParser(description="campaign 6 -- depth profile push")
    parser.add_argument("--campaign-id", type=str, default="c6_2026_06_11")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=23)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=45.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_campaign6"
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
        print(f"retrying {len(state['failed'])} failed: {[r['name'] for r in state['failed']]}")
        state["failed"] = []
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    base = dict(BASE_OVERRIDES)
    base["log-dir"] = str(runs_dir)
    target = min(args.max_runs, len(RUN_SPECS))

    while True:
        completed_records = state.get("completed", [])
        completed_ids = {int(r["run_id"]) for r in completed_records}
        failed_ids    = {int(r["run_id"]) for r in state.get("failed", [])}
        pending = [s for s in RUN_SPECS if s.run_id not in completed_ids and s.run_id not in failed_ids]
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
        env.update({"MPLBACKEND": "Agg", "TF_ENABLE_ONEDNN_OPTS": "0", "TF_CPP_MIN_LOG_LEVEL": "3"})

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
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print_summary(state.get("completed", []))
    print("campaign 6 finished")


if __name__ == "__main__":
    main()
