"""campaign_runner_15.py — best C14 performers under DOUBLE-scroll (small + big) training.

QUESTION
  C14 ranked architectures on the SINGLE small scroll. The open question: which of those
  architectures actually benefits from a LARGER, more diverse dataset? C15 retrains the top
  C14 performers on BOTH scroll1 fragments simultaneously (small 20230827161847 + big
  20230702185753, integrated batches) and watches how each one's hard-probe readability and
  scroll2 transfer change relative to its single-scroll C14 result.

LEARNINGS FROM C14 (carried over)
  - Depth window 28-44 beat 32-40 on every metric -> 28-44 is the fixed training window here.
  - Ring negatives (eroded) dominate the honest hard-probe ranking across 157 past runs -> ring ON.
  - No hard mining (most top runs ran without it; also auto-disabled in multiscroll anyway).
  - L1 scaled by parameter count; v1-family dropout for >=1M-param archs; L1=0 for the tiny
    depth-pool archs (they scored best historically with no L1).
  - C14 hard-probe order: v12_asym_attn_pool 0.366 > v1+dilation 0.360 > v2_no_cbam 0.356 >
    v10_topk_depth 0.351 > v1 0.338. Roster below is that head plus the v1 anchor.

ROSTER (5 runs)
  t01 v1                       anchor; directly comparable to C14 t02 (single-scroll v1 28-44,
                               hard 0.338) AND to the earlier multiscroll_v1_baseline (val F1 0.42).
  t02 v12_asym_attn_pool       C14 overall winner.
  t03 v1 + conv3-dilation=2    C14 2nd; the 'smaller features' dilated backbone.
  t04 v2_no_cbam               C14 3rd; strong plain CNN.
  t05 v10_topk_depth           C14 4th; depth-pooling family representative (tiny, L1=0).

COST CONTROL (multiscroll is expensive — read before editing)
  - eval-int == EPOCHS: the FULL-scroll evaluation figure (the big-fragment one is ~1.5-2 h with
    eval_aggregate's 4 averaging passes) fires only ONCE, at the final epoch.
  - probe-int 5: cheap 608px ROI probes give frequent monitoring. With both scrolls active this is
    now 8 ROIs (small easy/medium/hard + big easy/medium/hard + scroll4 + scroll2) per the new
    per-scroll probe generation.
  - test-int == EPOCHS + --test-scroll2-only: the goal-scroll2 transfer figure (full fragment,
    ~5 min) renders once at the end; the multi-hour big-fragment "Test" figure is skipped.
  - 30 epochs (extended from the c14 length of 20; the larger double-scroll dataset is still
    learning meaningfully past epoch 20).
  - batch 64 / workers 2 (only stable combo on this Win/Blackwell box), eval-cooldown 45 s.

CRASH SAFETY
  - best_model_f1/loss are saved BEFORE any figure code each epoch; probe/eval/test calls are all
    wrapped in try/except (probes individually). A figure failure cannot lose trained weights.
  - watchdog stall window is 240 min so the long final big-scroll eval is never mistaken for a hang.
  - resumable: completed run_ids are skipped on restart.
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


SMALL_SCROLL_ID = 20230827161847
BIG_SCROLL_ID   = 20230702185753
SCROLL4_ID      = 20231210132040
GOAL_SCROLL2_ID = 20230709155141

EPOCHS = 30

BASE: Dict[str, Any] = {
    "epochs": EPOCHS,
    "scroll-ids": f"{SMALL_SCROLL_ID},{BIG_SCROLL_ID}",   # DOUBLE-scroll integrated batches
    "scroll4-id": SCROLL4_ID,
    "batch-size": 64,
    "num-workers": 2,
    "probe-int": 5,
    "eval-int": EPOCHS,          # full eval figure once, at the end (big-scroll eval is ~2h)
    "test-int": EPOCHS,          # scroll2 transfer figure once, at the end
    "test-scroll2-only": True,
    "eval-cooldown": 45,
    "no-hard-mining": True,      # also auto-disabled in multiscroll, set explicitly for clarity
    "ring-negatives": True,
    "ring-label-source": "eroded",
    "train-d-start": 28,
    "train-d-end": 44,
    "channel-mixing-prob": 0.0,
    "mask-memmap": True,         # back mask/labels with on-disk memmap: pickle-as-path avoids the
                                 # intermittent multiscroll+ring spawn crash (OSError[Errno22]) seen in c15 t04
}

V1_DROPOUT = {"conv1-drop": 0.0, "conv2-drop": 0.05, "fc1-drop": 0.2, "fc2-drop": 0.1}
NO_DROPOUT = {"conv1-drop": 0.0, "conv2-drop": 0.0, "fc1-drop": 0.0, "fc2-drop": 0.0}


@dataclass(frozen=True)
class RunSpec:
    run_id: int
    name: str
    axis: str
    overrides: Dict[str, Any]
    why: str


RUN_SPECS: List[RunSpec] = [
    RunSpec(1, "t01_v1_anchor", "anchor",
        {"arch": "v1", "l1-lambda": 7e-6, **V1_DROPOUT},
        why="v1 anchor on both scrolls. directly comparable to C14 t02 (single-scroll v1 28-44, "
            "hard 0.338) and to multiscroll_v1_baseline (val F1 0.42). establishes the 'more data' "
            "delta for the reference architecture and validates the full multiscroll + new-probe "
            "+ new-figure pipeline first."),

    RunSpec(2, "t02_asym_attn_pool", "best_perf",
        {"arch": "v12_asym_attn_pool", "l1-lambda": 7.65e-6, **V1_DROPOUT},
        why="C14 overall winner (hard 0.366). asymmetric attention pooling; test whether its edge "
            "widens or narrows with 2x more, more-diverse data."),

    RunSpec(3, "t03_v1_dilated", "smaller_features",
        {"arch": "v1", "l1-lambda": 7e-6, "conv3-dilation": 2, **V1_DROPOUT},
        why="C14 2nd (hard 0.360). dilated final stage keeps fine input-grid detail while widening "
            "context — the 'ink features smaller than 7.9um' hypothesis, now with more data to learn "
            "those fine features from."),

    RunSpec(4, "t04_no_cbam", "best_perf",
        {"arch": "v2_no_cbam", "l1-lambda": 9.08e-6, **V1_DROPOUT},
        why="C14 3rd (hard 0.356). strong plain CNN with the most capacity in the roster; larger "
            "datasets typically help high-capacity models most — a key data-scaling datapoint."),

    RunSpec(5, "t05_topk_depth", "smaller_features",
        {"arch": "v10_topk_depth", "l1-lambda": 0.0, **NO_DROPOUT},
        why="C14 4th (hard 0.351). depth-pooling family (tiny, 286k, L1=0): averages the few "
            "strongest depth slices so thin ink is not diluted. tests whether a small model can "
            "exploit the extra data or saturates."),
]


CRASH_SIGNALS = [
    "Traceback (most recent call last)",
    "CUDA error:",
    "CUDA out of memory",
    "OSError: [Errno",
    "pickle data was truncated",
    "_pickle.UnpicklingError",
    "forrtl: error",
    "WinError 1455",
]
# bare "RuntimeError:" is NOT fatal: caught figure tracebacks print it but weights are safe.


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def dict_to_cli_args(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def build_cmd(python_exe: str, spec: RunSpec, runs_dir: Path, campaign_id: str):
    merged = dict(BASE)
    merged.update(spec.overrides)
    exp_name = f"cmp_{campaign_id}_{spec.name}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def find_run_dir(runs_dir: Path, exp_name: str, start_ts: float):
    # multiscroll creates the metrics run (exp_name_*) plus per-scroll figure runs
    # (exp_name_*_s<sid>). we want the metrics run: exclude the _s<sid> suffixes.
    suffixes = tuple(f"_s{sid}" for sid in (SMALL_SCROLL_ID, BIG_SCROLL_ID, SCROLL4_ID, GOAL_SCROLL2_ID))
    matches = [p for p in runs_dir.glob(f"{exp_name}_*")
               if p.is_dir() and not p.name.endswith(suffixes)]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime)
    for p in reversed(matches):
        if p.stat().st_mtime >= start_ts - 5:
            return p
    return matches[-1]


def extract_metrics(runs_dir: Path, exp_name: str, start_ts: float):
    """metrics live on the PER-SCROLL figure runs (probes/eval), not the metrics-only run.
    pull hard-probe composites for both scrolls plus valid F1 from the metrics run."""
    m = {"valid_f1_last": None,
         "small_hard_last": None, "small_easy_last": None,
         "big_hard_last": None, "big_easy_last": None}
    if event_accumulator is None:
        return m

    def last_scalar(run_dir, tag):
        evts = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
        if not evts:
            return None
        ea = event_accumulator.EventAccumulator(str(evts[-1]), size_guidance={"scalars": 0})
        ea.Reload()
        if tag in set(ea.Tags().get("scalars", [])):
            vals = [e.value for e in ea.Scalars(tag)]
            if vals:
                return vals[-1]
        return None

    # metrics-only run (valid F1)
    metrics_run = find_run_dir(runs_dir, exp_name, start_ts)
    if metrics_run is not None:
        m["valid_f1_last"] = last_scalar(metrics_run, "P_M/F1_Score/Valid")

    # per-scroll figure runs hold the probe composites (rendered once on the primary svis,
    # which writes to whichever per-scroll run rendered them). search all matching per-scroll runs.
    for p in runs_dir.glob(f"{exp_name}_*"):
        if not p.is_dir():
            continue
        if p.name.endswith(f"_s{SMALL_SCROLL_ID}"):
            m["small_hard_last"] = last_scalar(p, "R_M/Probe/Hard/ReadabilityComposite") or m["small_hard_last"]
            m["small_easy_last"] = last_scalar(p, "R_M/Probe/Easy/ReadabilityComposite") or m["small_easy_last"]
            m["big_hard_last"]   = last_scalar(p, "R_M/Probe/BigHard/ReadabilityComposite") or m["big_hard_last"]
            m["big_easy_last"]   = last_scalar(p, "R_M/Probe/BigEasy/ReadabilityComposite") or m["big_easy_last"]

    # unified_vis_dir layout: no per-scroll folders exist; probe ROIs are global tags
    # written into the single (metrics) run folder. fall back to reading them there.
    if metrics_run is not None and m["small_hard_last"] is None:
        m["small_hard_last"] = last_scalar(metrics_run, "R_M/Probe/Hard/ReadabilityComposite")
        m["small_easy_last"] = last_scalar(metrics_run, "R_M/Probe/Easy/ReadabilityComposite")
        m["big_hard_last"]   = last_scalar(metrics_run, "R_M/Probe/BigHard/ReadabilityComposite")
        m["big_easy_last"]   = last_scalar(metrics_run, "R_M/Probe/BigEasy/ReadabilityComposite")
    return m


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=240):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env,
                                stdout=lf, stderr=None)
    last_progress = time.time()
    last_epoch = 0
    while proc.poll() is None:
        time.sleep(20)
        try:
            lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        except Exception:
            continue
        tail = "".join(lines[-80:])
        for sig in CRASH_SIGNALS:
            if sig in tail:
                print(f"\n[MONITOR] CRASH -- '{sig}'")
                print("[MONITOR] last output:\n" + "".join(lines[-15:]))
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.wait()
                return proc.returncode or 1, True
        # progress = new epoch OR any heavy-IO marker (eval/test/probe predict + tqdm reads)
        for line in lines[-80:]:
            if "--- Epoch" in line:
                try:
                    ep = int(line.strip().split("/")[0].split()[-1])
                    if ep > last_epoch:
                        last_epoch = ep
                        last_progress = time.time()
                        print(f"[MONITOR] {line.strip()}")
                except Exception:
                    pass
        for marker in ("Starting evaluation figure generation", "Starting test figure generation",
                       "Logging probe-region figures", "Predict train", "Read train",
                       "Read valid", "Predict valid", "Read Scroll2", "Predict Scroll2",
                       "Read BigEasy", "Read BigHard"):
            if marker in tail:
                last_progress = time.time()
                break
        if time.time() - last_progress > stall_minutes * 60:
            print(f"\n[MONITOR] STALL -- no progress in {stall_minutes} min")
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()
            return 1, True
    proc.wait()
    rc = proc.returncode
    if rc != 0:
        try:
            tail = open(log_path, encoding="utf-8", errors="replace").readlines()[-20:]
            print("[MONITOR] last output:\n" + "".join(tail))
        except Exception:
            pass
    print(f"[MONITOR] {'completed successfully' if rc == 0 else f'exited rc={rc}'}")
    return rc, False


def print_summary(completed):
    if not completed:
        return
    print("\n+-- campaign 15 results (double-scroll; ranked by small-scroll hard probe) ----")
    print(f"|  {'run':<26} {'s_hard':>7} {'s_easy':>7} {'b_hard':>7} {'b_easy':>7} {'f1':>6}")
    print("|  " + "-" * 66)
    def k(r):
        mm = r.get("metrics") or {}
        return mm.get("small_hard_last") or 0
    for r in sorted(completed, key=k, reverse=True):
        mm = r.get("metrics") or {}
        def f(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "  ?  "
        print(f"|  {r['name'][-26:]:<26} {f(mm.get('small_hard_last')):>7} "
              f"{f(mm.get('small_easy_last')):>7} {f(mm.get('big_hard_last')):>7} "
              f"{f(mm.get('big_easy_last')):>7} {f(mm.get('valid_f1_last')):>6}")
    print("+--" + "-" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="campaign 15 -- C14 winners under double-scroll training")
    parser.add_argument("--campaign-id", type=str, default="c15_2026_06_29")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=240.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir = repo_root / "runs_campaign15"
    runs_dir.mkdir(exist_ok=True)
    state_dir = runs_dir / "campaign_logs"
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / f"{args.campaign_id}_state.json"

    state = {"completed": []}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {"completed": []}
    done_ids = {r.get("run_id") for r in state.get("completed", [])}

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    pending = [s for s in RUN_SPECS if s.run_id not in done_ids]
    pending.sort(key=lambda s: s.run_id)
    print(f"[campaign15] {len(done_ids)} done, {len(pending)} pending")

    for spec in pending:
        cmd, exp_name = build_cmd(args.python_exe, spec, runs_dir, args.campaign_id)
        print("\n" + "=" * 78)
        print(f"[RUN {spec.run_id}] {exp_name}  (axis={spec.axis})")
        print(f"   why: {spec.why}")
        print(f"   cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue
        start_ts = time.time()
        log_path = state_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                          stall_minutes=args.stall_minutes)
        metrics = extract_metrics(runs_dir, exp_name, start_ts)
        rec = {
            "run_id": spec.run_id,
            "name": exp_name,
            "axis": spec.axis,
            "overrides": {**BASE, **spec.overrides},
            "metrics": metrics,
            "return_code": rc,
            "crashed": crashed,
            "ended_at": now_utc(),
        }
        state.setdefault("completed", []).append(rec)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[RUN {spec.run_id}] done rc={rc} crashed={crashed} "
              f"small_hard={metrics.get('small_hard_last')}")
        print_summary(state["completed"])

    print("\n[campaign15] all runs processed.")
    print_summary(state.get("completed", []))


if __name__ == "__main__":
    main()
