"""campaign_runner_7.py — sensitivity push: finding sub-voxel ink traces

C6 finding: depth-sequential models (LSTM/BiGRU) outperform all spatial approaches.
best hard probe: t15_lstm=0.445, t09_bigru_full64=0.436. but NO model showed
visible letter strokes in the hard probe region.

physical insight: ink at 7.91um/voxel is sub-voxel (ink particles 1-5um).
the 4x resolution improvement (3.7um scan) immediately revealed ink on scroll 4 —
meaning the ink signal at 7.91um is tiny but may still be statistically detectable.

C7 strategy:
  - re-enable hard mining + focal loss to force attention on hard examples
  - robust statistics: percentiles, centered profiles, pairwise diffs (not fooled by noise)
  - deeper sequential models: 12-block ResNet, 12-layer Transformer, 4-layer BiGRU
  - anomaly detection: pixel deviation from tile mean, AE bottleneck reconstruction error
  - multi-scale depth: dilated convs, inception-style, spectral features
  - spatial-depth hybrid at the right scale: 8x8 superpixels (~63um, ink stroke width)
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
    "num-workers": 2,    # 4 causes Windows pipe truncation on spawn; 2 is safe
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
    # hard mining ENABLED in C7 (was disabled in C6)
    # focal loss OFF by default — override per-run where beneficial
}

RUN_SPECS: List[RunSpec] = [

    # A: training strategy: re-enable hard mining + focal on C6-best arches
    RunSpec(1, "t01_focal_bigru", "training",
        {"arch": "v6_bigru_slices", "focal-gamma": 5.0, "hm-frac": 0.1,
         "hn-cutoff": 0.8, "hp-cutoff": 0.4},
        why="C6 best: BiGRU over depth slices (hard=0.419). C6 disabled hard mining. "
            "focal gamma=5 + hard mining should force the model to focus on hard examples "
            "instead of memorizing easy ink patterns. simple architecture change — if this "
            "works, training strategy was the bottleneck.",
        expect="hard probe improvement over C6 t16; if focal+mining helps, training was the issue"),

    RunSpec(2, "t02_focal_lstm", "training",
        {"arch": "v6_lstm_slices", "focal-gamma": 5.0, "hm-frac": 0.1,
         "hn-cutoff": 0.8, "hp-cutoff": 0.4},
        why="C6 best: LSTM (hard=0.445). same focal+mining treatment as t01. "
            "LSTM captures sequential depth pattern; forcing it to focus on hard examples "
            "might reveal the subtle depth patterns in hard-to-read regions.",
        expect="best result of campaign if training strategy was the C6 bottleneck"),

    RunSpec(3, "t03_focal_fulldepth", "training",
        {"arch": "v6_fulldepth_1d", "input-mode": "fulldepth", "batch-size": 64,
         "focal-gamma": 5.0, "hm-frac": 0.1, "hn-cutoff": 0.8, "hp-cutoff": 0.4},
        why="full 64-depth 1D CNN (C6 t05, hard=0.345) with focal+mining. "
            "the full absorption curve is the most information-rich single signal. "
            "hard mining should force the model toward the hard examples it was ignoring.",
        expect="moderate hard probe improvement; tests if full-depth + training strategy works"),

    # B: robust statistical features — not fooled by per-pixel noise
    RunSpec(4, "t04_percentile_depth", "robust_stats",
        {"arch": "v7_percentile_depth"},
        why="5 percentiles (10/25/50/75/90) per depth slice → deep MLP. "
            "if ink covers 5-10% of pixels at a given depth, the 90th percentile is elevated "
            "even when the mean is dominated by background. far more sensitive than mean+std. "
            "40-dim features: completely different input representation from all prior models.",
        expect="better hard probe than mean-based models; percentiles capture rare ink pixels"),

    RunSpec(5, "t05_centered_depth", "robust_stats",
        {"arch": "v7_centered_depth"},
        why="depth profile minus per-tile mean → 1D CNN. removes scroll-wide brightness "
            "variation, leaving only the RELATIVE depth pattern. background tiles → flat "
            "centered profile. ink tiles → positive bump at ink depths (32-40). "
            "explicitly removes the main source of confounding variation across tiles.",
        expect="cleaner depth signature; may reveal subtle ink bumps hidden by baseline"),

    RunSpec(6, "t06_pairwise_depth", "robust_stats",
        {"arch": "v7_pairwise_depth"},
        why="all 28 pairwise depth differences for D=8 → deep MLP. "
            "explicitly encodes: is depth D_ink higher than D_pre? D_post? explicitly "
            "scale-invariant to per-tile brightness. ink: D_32>D_20, D_32>D_40 systematically. "
            "different inductive bias from any prior approach — direct comparison not regression.",
        expect="high specificity; pairwise comparisons are noise-resistant"),

    RunSpec(7, "t07_prototype_depth", "robust_stats",
        {"arch": "v7_prototype_depth"},
        why="K=32 learned prototype depth profiles; score by cosine similarity → MLP. "
            "like a matched filter bank. prototypes initialize randomly and learn to represent "
            "common depth profile shapes. ink detection = high similarity to an ink prototype. "
            "explicitly learns WHAT ink profiles look like rather than discriminating features.",
        expect="diagnostic: if good prototypes exist, this will find them"),

    # C: multi-scale depth feature extraction
    RunSpec(8, "t08_multiscale_depth", "multiscale",
        {"arch": "v7_multiscale_depth"},
        why="dilated conv1d at rates 1, 2, 4 concatenated on depth profile. "
            "rate-1: adjacent depth differences (ink peak edge detection). "
            "rate-2: 2-step differences (broader bell curve shape). "
            "rate-4: global shape (baseline-to-peak ratio). "
            "simultaneously captures local and global depth patterns.",
        expect="richer depth features than single-scale 1D CNN"),

    RunSpec(9, "t09_inception_depth", "multiscale",
        {"arch": "v7_inception_depth"},
        why="parallel conv1d kernels 1, 3, 5, 7 concatenated → mixing stage → head. "
            "inception-style: pointwise, local context, medium context, broader context. "
            "then deeper mixing of multi-scale features. stronger than multiscale dilated "
            "because the second stage can learn cross-scale interactions.",
        expect="best multi-scale result; inception-style mixing should help"),

    # D: deeper sequential models on depth
    RunSpec(10, "t10_deep_resnet_depth", "deep_arch",
        {"arch": "v7_deep_resnet_depth"},
        why="12-block 1D ResNet on 8-depth profile — 24 conv layers with skip connections. "
            "far deeper than any prior depth-profile model. even on just 8 values, a very deep "
            "ResNet can learn extremely subtle pattern distinctions through many refinement passes. "
            "skip connections prevent gradient vanishing; BatchNorm stabilizes.",
        expect="captures depth patterns that shallow models miss; depth=quality in CNNs"),

    RunSpec(11, "t11_deep_transformer_depth", "deep_arch",
        {"arch": "v7_deep_transformer_depth"},
        why="12-layer Transformer on D=8 depth positions with d=256. "
            "when N=8, attention is trivially cheap (64 values/head/layer). "
            "12 layers = 3-6x deeper than any C5/C6 depth transformer. "
            "deep transformers can represent exponentially more complex functions; "
            "ink might require complex multi-way depth interactions to detect.",
        expect="best sequential result if depth relationships are complex and non-local"),

    # E: spatial-depth hybrid at the right scale
    RunSpec(12, "t12_superpixel_attn", "spatial_depth",
        {"arch": "v7_superpixel_attn", "batch-size": 64},
        why="4×4=16 superpixels of 8×8 each, each with a mean depth profile as a token. "
            "6-layer transformer over 16 tokens: superpixels compare with each other. "
            "8×8 pixels at 7.91um = ~63um — matches typical ink letter stroke width. "
            "provides just enough spatial structure to detect correlated multi-superpixel "
            "absorption patterns without per-pixel noise. better scale than t06/t07 (which used 32x32).",
        expect="captures spatial correlations at the right scale for ink stroke detection"),

    RunSpec(13, "t13_pixel_deviation", "spatial_depth",
        {"arch": "v7_pixel_deviation"},
        why="per-pixel depth profile minus tile mean → gated MIL. "
            "explicitly finds SPATIAL OUTLIERS in the tile: pixels whose depth profile "
            "deviates from the tile average. ink pixels should deviate at specific depths. "
            "background: all pixels similar → near-zero residuals. ink: anomalous bump at ink depths. "
            "directly targets the hypothesis that ink = spatially localized absorption anomaly.",
        expect="strong if ink appears as isolated anomalous pixels; tests spatial outlier detection"),

    # F: full 64-depth sequential
    RunSpec(14, "t14_full64_deep_bigru", "fulldepth",
        {"arch": "v7_full64_deep_bigru", "input-mode": "fulldepth", "batch-size": 64},
        why="4-layer BiGRU on full 64-depth spatial-mean profile (256 hidden). "
            "C6 t09 1-layer 64-depth BiGRU got hard=0.436 — C6 best. "
            "4 layers × 256 hidden = much more sequential modeling capacity. "
            "the full absorption curve from baseline through peak through return "
            "is the richest single signal; deep BiGRU should extract its full pattern.",
        expect="best result of campaign; builds on the strongest C6 architecture"),

    # G: novel feature representations
    RunSpec(15, "t15_spectral_depth", "novel_features",
        {"arch": "v7_spectral_depth"},
        why="FFT of depth profile → real+imag components → MLP. "
            "the frequency spectrum of the depth absorption encodes the profile shape. "
            "ink bell curve → specific frequency ratio. scale-invariant to overall brightness. "
            "completely different feature space from all time-domain models.",
        expect="unique failure mode; spectral features may separate where spatial don't"),

    RunSpec(16, "t16_bigru_percentile", "novel_features",
        {"arch": "v7_bigru_percentile"},
        why="3-layer BiGRU on the sequence of (5-percentile vector) across D depth positions. "
            "at each depth d: [10th,25th,50th,75th,90th] percentile across 32x32 pixels. "
            "BiGRU processes this as a time series of distribution-snapshots through depth. "
            "captures HOW the spatial distribution EVOLVES through depth — ink shifts the "
            "distribution at specific depths, which a BiGRU can detect sequentially.",
        expect="combines percentile robustness with sequential depth modeling — potentially best"),

    RunSpec(17, "t17_ae_bottleneck", "novel_features",
        {"arch": "v7_ae_bottleneck"},
        why="joint AE + classifier: encoder compresses depth profile D→16→reconstruct→D. "
            "classifier sees bottleneck (16) + per-dim reconstruction errors (D). "
            "ink tiles with unusual depth profiles should have high reconstruction error. "
            "the reconstruction difficulty is itself a discriminative feature. "
            "anomaly detection principle: hard to reconstruct = anomalous = ink.",
        expect="reconstruction error is a novel signal; tests anomaly detection approach"),

    # H: differential signal + robust stats
    RunSpec(18, "t18_diff_percentile", "differential",
        {"arch": "v7_diff_percentile", "input-mode": "diff"},
        why="percentile features on (ink_band - pre_band) differential signal. "
            "combines two independently effective ideas: differential removes scroll baseline, "
            "percentiles are robust to per-pixel noise. "
            "the 90th percentile of (ink-pre) should be elevated at ink pixels. "
            "addresses both the noise problem and the baseline problem simultaneously.",
        expect="strong combination; differential + percentile is most noise-resistant approach"),

    RunSpec(19, "t19_centered_diff", "differential",
        {"arch": "v7_centered_depth", "input-mode": "diff"},
        why="centered depth profile on differential input: subtract-mean normalization "
            "applied to an already baseline-corrected signal. "
            "double baseline removal: differential removes scroll-wide baseline, "
            "centering removes tile-specific residual baseline. leaves only the "
            "depth-varying ink absorption shape at the cleanest possible level.",
        expect="cleanest signal; two-stage baseline removal may reveal very faint ink patterns"),

    RunSpec(20, "t20_percentile_focal", "training",
        {"arch": "v7_percentile_depth", "focal-gamma": 5.0, "hm-frac": 0.1,
         "hn-cutoff": 0.8, "hp-cutoff": 0.4},        why="percentile features (t04) combined with hard mining + focal loss. "
            "t04 uses the most robust feature representation; t01-t02 use best training strategy. "
            "combining both may give the best hard example detection: "
            "percentiles survive noise, focal loss + mining forces attention on hard tiles.",
        expect="best hard probe if both robust features AND training strategy matter"),

    # I: ring-negatives — train only on known non-ink adjacent to labeled ink
    #
    # hypothesis: unlabeled ink in the scroll is being trained as negatives,
    # actively suppressing the hard signal. ring negatives ensure every '0' label
    # is a tile that is ADJACENT to confirmed ink and therefore certainly non-ink.
    # dilation radius chosen automatically to achieve ~1:1 positive/negative ratio.
    RunSpec(21, "t21_ring_v1_cnn", "ring_negatives",
        {"arch": "v1", "ring-negatives": True, "num-workers": 0,
         "batch-size": 64, "no-hard-mining": True},
        why="v1: original baseline 3D CBAM CNN (commit 002a007 era architecture). "
            "num-workers=0 avoids Windows pipe error. batch=64 avoids OOM. "
            "ring negatives: only ring-adjacent-to-ink tiles used as negatives. "
            "this model should 100% learn easy examples if ring is working correctly.",
        expect="high easy probe (>0.6) if ring negatives are clean; validates ring implementation"),

    RunSpec(22, "t22_ring_bigru", "ring_negatives",
        {"arch": "v6_bigru_slices", "ring-negatives": True, "num-workers": 0,
         "no-hard-mining": True},
        why="C6 best sequential model (hard=0.419) + ring negatives + num-workers=0 (pipe-safe). "
            "BiGRU over 8 depth slices was consistently among the strongest C6 arches. "
            "combining the best architecture with clean negatives tests both hypotheses together.",
        expect="if ring negatives help, this combines cleanest training with best arch"),

    RunSpec(23, "t23_ring_lstm", "ring_negatives",
        {"arch": "v6_lstm_slices", "ring-negatives": True, "num-workers": 0,
         "no-hard-mining": True},
        why="C6 BEST (hard=0.445) + ring negatives + num-workers=0. LSTM captured the most hard signal "
            "in C6. ring negatives may allow it to pick up even fainter hard ink signals "
            "that were being suppressed by unlabeled-ink contamination in the negatives.",
        expect="best result if LSTM architecture was already close and negatives were the issue"),

    RunSpec(24, "t24_ring_percentile", "ring_negatives",
        {"arch": "v7_percentile_depth", "ring-negatives": True, "num-workers": 0,
         "no-hard-mining": True},
        why="robust percentile features + ring negatives. num-workers=0 for pipe safety. "
            "percentiles are the most noise-resistant feature; ring negatives are the "
            "cleanest negative selection. together: best noise resistance AND best label quality.",
        expect="strong combination; tests if the problem is feature noise OR label noise"),

    RunSpec(25, "t25_ring_focal_lstm", "ring_negatives",
        {"arch": "v6_lstm_slices", "ring-negatives": True, "num-workers": 0,
         "focal-gamma": 5.0, "hm-frac": 0.1, "hn-cutoff": 0.8, "hp-cutoff": 0.4},
        why="everything combined: ring negatives (clean labels) + focal loss + hard mining "
            "(forces attention on hard tiles) + LSTM (best C6 arch) + num-workers=0 (pipe-safe). "
            "attacks all three hypotheses simultaneously: label contamination, "
            "training focus, and architecture.",
        expect="strongest test; if any of these factors is the bottleneck this should reveal it"),
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
         "probe_easy_last": None, "probe_hard_last": None}
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
            m[f"{key}_best"] = max(vals); m[f"{key}_last"] = vals[-1]
    return m


def print_summary(completed):
    if not completed: return
    print("\n+-- campaign 7 results (ranked by hard probe) ----------------------")
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
    # prioritize axes that have shown the best improvement
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
    parser = argparse.ArgumentParser(description="campaign 7 -- sensitivity push")
    parser.add_argument("--campaign-id", type=str, default="c7_2026_06_13")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=60.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_campaign7"
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
    print("campaign 7 finished")


if __name__ == "__main__":
    main()
