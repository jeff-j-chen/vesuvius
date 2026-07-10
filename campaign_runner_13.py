"""campaign_runner_13.py — depth-profile focus: inter-layer signal, diverse inputs.

Key learnings from C11 + C12:
  - ring training is required to prevent easy-texture collapse; every run uses ring
  - LocalContrast is the honest metric: coverage_recall saturates trivially under ring
  - ReadabilityComposite now weights LocalContrast 2× and coverage 0.5× (fixed in visualizer)
  - spatial attention (CBAM spatial, non-local, 2D attention) is consistently useless
  - inter-layer depth attention on conv features (not raw voxels) is the only mechanism
    that has shown any selectivity development (t04, C12)
  - the ink signal is likely sub-tile scale, < 2 voxels at 7.91um — a depth profile spike
  - full depth (64 slices) vs 8 slices: never tested with ring; should reveal the full curve
  - diff/triple inputs tested in old campaigns on bad architectures; re-test with ring + depth focus
  - no reg on/off pairs: each slot tests a qualitatively different idea

C13 strategy:
  - 30 epochs, probe_int=5, test_int=100
  - all runs use ring training (ring+eroded) — the only reliable anti-collapse constraint
  - each run tests a structurally different approach to the depth-profile hypothesis
  - NO duplicate ablation pairs: every slot is a new architecture or input modality
  - t01: asym_pool + ring + full depth (64 slices, fulldepth mode, spatial reduce first)
  - t02: depth_diff_conv — asym_pool on the differential (ink minus pre-band clip)
  - t03: triple_band_stem — factorized 3D conv on pre+ink+post (24 slices, cross-band)
  - t04: depth_delta_transformer — per-slice 2D CNN embed → depth transformer → max
  - t05: depth_profile_mil — per-pixel depth profile + gated MIL (which pixels show spike?)
  - t06: depth_contrast_conv — asym_pool on double input (ink||pre, 16 slices)
  - t07: cross_depth_attn — 3D stem → query from max-depth position → attend all depths
  - t08: depth_gradient — depth first-derivative input (sharp transitions = ink boundaries)
  - t09: slice_contrast_bigru — per-slice spatial-max minus running mean → BiGRU deviation
  - t10: asym_pool + ring + double input (ink||pre, reuse proven backbone on paired input)
  - t11: depth_delta_transformer + full depth (fulldepth mode, more curve context)
  - t12: v10_asymmetric_pool + ring + diff input (proven backbone, differential absorption)
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

L1_ASYM         = 7.5e-6   # 1.2M params (asym_pool backbone)
# for small models: scale L1 so total penalty budget ≈ v1 baseline
# formula: L1 = 7e-6 * (1.3M / param_count)
L1_113K         = 8.0e-5   # 113K params (fulldepth_conv1d)
L1_88K          = 1.0e-4   # 88K  params (depth_delta_transformer)
L1_116K         = 7.5e-5   # 116K params (slice_contrast_bigru)
L1_233K         = 4.0e-5   # 233K params (triple_band_stem)
L1_286K         = 3.2e-5   # 286K params (depth_gradient)
L1_296K         = 3.1e-5   # 296K params (cross_depth_attn)
L1_12K          = 3.0e-4   # 12K  params (depth_profile_mil) — capped conservatively

BASE: Dict[str, Any] = {
    "epochs": 30,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 64,
    "num-workers": 2,
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 100,
    "eval-cooldown": 45,
    "no-hard-mining": True,    # ring provides hard-example focus; HM adds instability
    "train-d-start": 28,
    "train-d-end": 44,
    "l1-lambda": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.3,           # applied to all v13 heads; prevents memorization
    "fc2-drop": 0.0,
    "conv1-drop": 0.0,
    "channel-mixing-prob": 0.0,
    # all runs use ring training
    "ring-negatives": True,
    "ring-label-source": "eroded",
}

RUN_SPECS: List[RunSpec] = [

    # ── t01: asym_pool + fulldepth (64 slices) ───────────────────────────────
    RunSpec(1, "t01_asym_pool_fulldepth", "fulldepth",
        {"arch": "v13_fulldepth_conv1d_ring",
         "input-mode": "fulldepth",
         "l1-lambda": L1_113K},
        why="the most information-rich depth experiment: full 64-slice volume → spatial "
            "reduce (1x3x3 conv + adaptive avg) → 1D CNN along depth → adaptive max pool. "
            "if ink shows as a bump in the depth profile, 64 slices should reveal the "
            "full bell curve shape instead of catching just the peak in 8 slices. "
            "depth-max ensures the spike is captured even if it spans 1-2 slices. "
            "ring constrains to boundary tiles; fulldepth provides full curve context."),

    # ── t02: depth_diff_conv — differential absorption ─────────────────────
    RunSpec(2, "t02_depth_diff_conv", "diff_input",
        {"arch": "v13_depth_diff_conv",
         "input-mode": "diff",
         "l1-lambda": L1_ASYM},
        why="differential input: clip(ink_band - pre_band, 0). non-ink pixels should "
            "have near-zero diff; ink pixels show positive absorption anomaly. "
            "asym_pool backbone (4x conv3d, spatial-avg, depth-max) then isolates the "
            "strongest differential depth position. tests: does subtracting the baseline "
            "scroll structure reveal hard ink above detection threshold?"),

    # ── t03: triple_band_stem — cross-band factorized conv ──────────────────
    RunSpec(3, "t03_triple_band_stem", "triple_input",
        {"arch": "v13_triple_band_stem",
         "input-mode": "triple",
         "l1-lambda": L1_233K},
        why="triple input: [pre_band | ink_band | post_band] concatenated = 24 slices. "
            "factorized (3,1,1)+(1,3,3) convs process depth-then-spatial; the 24-slice "
            "axis spans all three bands so cross-band transitions are explicitly visible. "
            "if ink causes absorption in the ink_band but not pre or post, the factorized "
            "depth convs should detect that specific pattern even in hard tiles."),

    # ── t04: depth_delta_transformer — slice embed → depth attention ─────────
    RunSpec(4, "t04_depth_delta_transformer", "depth_transformer",
        {"arch": "v13_depth_delta_transformer",
         "l1-lambda": L1_88K},
        why="per-slice 2D CNN (shared weights) embeds each depth slice to a 64-d token, "
            "then a 2-layer transformer attends over the depth sequence with learned "
            "positional embeddings. output = max over depth tokens (anomalous position). "
            "specifically addresses C11 lesson: 'attention on conv features, not raw voxels'. "
            "the 2D CNN provides spatial context per slice; transformer resolves cross-depth. "
            "ring training prevents spatial texture memorization."),

    # ── t05: depth_profile_mil — per-pixel gated MIL ────────────────────────
    RunSpec(5, "t05_depth_profile_mil", "depth_mil",
        {"arch": "v13_depth_profile_mil",
         "l1-lambda": L1_12K},
        why="per-pixel 1D CNN encodes each pixel's 8-slice depth profile; gated MIL "
            "(attention-weighted bag aggregation over 32x32=1024 pixel instances) asks: "
            "'which pixels in this tile show an ink-like depth spike?' "
            "the gate is driven by depth profile features, not spatial texture. "
            "smallest param model (12K) — if the depth profile is the signal, "
            "a tiny model should suffice and generalize best."),

    # ── t06: depth_contrast_conv — asym_pool on double (ink||pre) ───────────
    RunSpec(6, "t06_depth_contrast_conv", "double_input",
        {"arch": "v13_depth_contrast_conv",
         "input-mode": "double",
         "l1-lambda": L1_ASYM},
        why="double input: concatenate([ink_band, pre_band], depth_axis) = 16 slices. "
            "asym_pool backbone sees both the ink and pre band jointly; the 3D conv "
            "learns to compare them. depth-max then picks the most discriminative "
            "depth position across the 16-slice paired volume. "
            "unlike diff mode (which clips and subtracts), this preserves both "
            "absolute values and lets the model learn any comparison function."),

    # ── t07: cross_depth_attn — query-from-max cross-depth attention ─────────
    RunSpec(7, "t07_cross_depth_attn", "depth_attention",
        {"arch": "v13_cross_depth_attn",
         "l1-lambda": L1_296K},
        why="3D conv stem reduces spatial → (B, 128, D); the strongest depth position "
            "(max L2-norm) becomes the query and attends to all others as keys/values. "
            "'given the most activated depth slice, do other depths confirm it as ink?' "
            "implements cross-depth verification: if neighboring slices also activate "
            "coherently, that's a strong signal. if only one slice fires, it may be noise. "
            "directly addresses inter-layer relationship hypothesis from prior campaigns."),

    # ── t08: depth_gradient — first-derivative depth input ──────────────────
    RunSpec(8, "t08_depth_gradient", "gradient_input",
        {"arch": "v13_depth_gradient",
         "l1-lambda": L1_286K},
        why="compute x[:,1:] - x[:,:-1] along depth before the network: first-order "
            "gradient of the depth profile. ink absorption creates a sharp rise then fall; "
            "the gradient turns this into a positive spike followed by a negative spike — "
            "potentially easier to detect than the absolute peak. "
            "same asym_pool-style backbone on the gradient volume. "
            "if the issue is that small ink signals are buried in baseline, "
            "the gradient removes that baseline completely."),

    # ── t09: slice_contrast_bigru — deviation-from-running-mean BiGRU ───────
    RunSpec(9, "t09_slice_contrast_bigru", "recurrent_depth",
        {"arch": "v13_slice_contrast_bigru",
         "l1-lambda": L1_116K},
        why="per-slice: spatial max-pool → embed (32-d). deviation sequence: "
            "subtract running mean of previous slices (online baseline). "
            "BiGRU processes the deviation sequence; the hidden state at the max-deviation "
            "depth position is classified. spatial max (not mean) ensures a single "
            "ink pixel's signal can propagate through. "
            "the running-mean baseline removal is a learnable normalization-free way "
            "to implement 'this slice is anomalous relative to its context'."),

    # ── t10: asym_pool + double input (proven backbone, paired bands) ────────
    RunSpec(10, "t10_asym_pool_double", "double_input",
        {"arch": "v10_asymmetric_pool",
         "input-mode": "double",
         "l1-lambda": L1_ASYM},
        why="reuse the proven asym_pool backbone (C11 t10/t11 best performer) with "
            "double input (ink||pre, 16 slices). the 4-block 3D CNN backbone already "
            "shows depth-max selectivity; giving it paired bands may let it learn to "
            "detect the differential pattern without the clipping artifact of diff mode. "
            "direct comparison to t06 (same backbone, same input format, different init)."),

    # ── t11: depth_delta_transformer + fulldepth (64 tokens) ─────────────────
    RunSpec(11, "t11_depth_delta_transformer_full", "depth_transformer",
        {"arch": "v13_depth_delta_transformer",
         "input-mode": "fulldepth",
         "l1-lambda": L1_88K},
        why="t04's architecture (per-slice CNN → depth transformer) with 64 depth slices "
            "instead of 8. 64 tokens give the transformer the full absorption curve "
            "shape: baseline → ascent → peak → descent → baseline. "
            "the transformer should learn to recognize that specific shape pattern "
            "vs the flat curve of non-ink. max over 64 depth tokens picks the most "
            "anomalous position in the full depth profile."),

    # ── t12: asym_pool + diff + ring (asym_pool baseline on differential) ────
    RunSpec(12, "t12_asym_pool_diff", "diff_input",
        {"arch": "v10_asymmetric_pool",
         "input-mode": "diff",
         "l1-lambda": L1_ASYM},
        why="proven best backbone (asym_pool, C11/C12 consistently ~0.44 hard RC) "
            "with differential input mode. tests whether the clip(ink-pre) "
            "preprocessing reveals signal that the raw 8-slice input cannot. "
            "baseline comparison: same backbone, same ring, only input differs from "
            "t01/t10 (C12). if hard LC jumps, diff preprocessing is the missing key."),
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
    print("\n+-- campaign 13 results (ranked by hard probe) ----------------------")
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
    return sorted(pending, key=lambda s: s.run_id)[0]


def main():
    parser = argparse.ArgumentParser(description="campaign 13 -- depth-profile focus")
    parser.add_argument("--campaign-id", type=str, default="c13_2026_06_23")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=120.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_campaign13"
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

    done_ids = {int(r["run_id"]) for r in state.get("completed", []) + state.get("failed", [])}
    pending  = [s for s in RUN_SPECS if s.run_id not in done_ids]
    completed_records = state.get("completed", [])

    while pending:
        print_summary(completed_records)
        spec = choose_next(pending, completed_records)

        merged = dict(base)
        merged.update(spec.overrides)
        merged["log-dir"] = str(runs_dir)

        exp_name = f"cmp_{args.campaign_id}_{spec.name}"
        log_path = state_dir / f"{exp_name}.log"
        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)

        print(f"\n{'='*62}")
        print(f"  run {spec.run_id:02d}/12: {spec.name}  [{spec.axis}]")
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

        pending = [s for s in RUN_SPECS if s.run_id not in
                   {int(r["run_id"]) for r in state.get("completed", []) + state.get("failed", [])}]

    print_summary(state.get("completed", []))
    print("\n[campaign 13] all runs complete.")


if __name__ == "__main__":
    main()
