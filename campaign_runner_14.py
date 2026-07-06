"""campaign_runner_14.py — best past performers vs the GOAL scroll (scroll2) transfer test.

PURPOSE
  Re-run the strongest architectures from campaigns 9-12 under one controlled regime and,
  at the final epoch of each, render a FULL-FRAGMENT inference figure on scroll2 (our goal
  scroll). We want to see which model — trained only on scroll1 ink labels — transfers any
  readable ink signal onto scroll2, which has no labels of its own.

WHAT CHANGED IN THE CODE (this task)
  - visualizer._load_scroll2_region now covers the ENTIRE scroll2 fragment (was a 2048x1024
    crop), full depth. Mask-gated, so only real papyrus tiles are read (~292k tile reads,
    ~35-50 min for one figure).
  - new flag --test-scroll2-only: the test figure renders ONLY scroll2 and skips the
    ~5-6 h full training-scroll "Test" figure. Used here so the end-of-training transfer
    check is affordable on every run.
  - crash safety (already true, reconfirmed): best_model_f1/loss are saved in
    _periodic_model_save BEFORE _log_epoch, and every test-figure call is wrapped in
    try/except. A failed/expensive test figure can never destroy trained weights.

DESIGN CHOICES AND WHY  (read before editing)
  Goal-scroll insight: scroll4 showed NO text at 7.9um but ink appeared immediately at 3.7um.
  Our scrolls (1 + goal scroll2) are 7.9um, so the ink features we want are SMALLER than the
  network's usual receptive field assumes. The suite therefore favours architectures that
  preserve fine detail and pick the strongest LOCAL depth response rather than averaging it
  away:
    - depth-pooling family (v10_max_depth_pool, v10_topk_depth): take the max / top-k ink
      response across depth instead of mean — keeps thin ink layers from being diluted.
    - multi-scale receptive field (v10_multiscale_3d): parallel kernels capture both fine and
      coarse structure.
    - a dilated-conv variant of the v1 baseline (conv3-dilation=2): widens context WITHOUT
      extra spatial downsampling, so fine detail at the input grid is retained.

  Training regime (held constant so the comparison is clean):
    - SINGLE scroll: the small goal-adjacent scroll 20230827161847. This matches the exact
      conditions under which every top-ranked past run was trained, and keeps each run ~2 h so
      the whole suite fits in ~a day. (Multi-scroll already proven to help; not the variable here.)
    - RING negatives ON (ring-label-source=eroded). EVIDENCE: across c9-c12 essentially every
      top-hard-probe run used ring negatives (v6_fulldepth_gru 0.479, v12_asym_attn_pool 0.466,
      v10_3d_unet 0.462, v10_max_depth_pool 0.458, v1 0.452 ...). C11's "ring hurts" claim was
      not borne out by the ranked results; c12 returned to ring and topped the board. Ring is
      the reliable anti-collapse / hard-example mechanism.
    - NO hard mining: most top runs ran without it, and it only helps once a model already
      reaches the decision boundary on hard tiles. Disabling it also removes a crash surface
      for an unattended multi-hour campaign.
    - Regularization to prevent memorization: L1 scaled by parameter count
      (lambda = 7e-6 * 1.3M / params, clamped) for the >=1M-param archs; the tiny depth-pool
      archs (87k-286k params) scored best historically with L1=0, so they keep L1=0 and rely on
      their small capacity. v1-family dropout conv2=0.05 / fc1=0.2 / fc2=0.1.
    - Depth window 28-44 for TRAINING (3 windows: 28-36,32-40,36-44 at step 4) — preferred in
      c11/c12; evaluation still spans 28-48. t01 vs t02 below is the explicit A/B that justifies
      this (32-40, the previous multiscroll baseline, vs 28-44).
    - 30 epochs, batch 64 / workers 2 (only stable combo on this Win/Blackwell box),
      eval-cooldown 45 s (thermal), probe-int 5, eval-int 10.
    - TEST: --test-scroll2-only with test-int == epochs, so the full scroll2 transfer figure is
      rendered exactly once, at the very end, on every run.

  Ranking proxy: scroll2 has no labels, so transfer quality is judged VISUALLY from the scroll2
  figures in TensorBoard. The numeric ranking still uses the honest hard-probe ReadabilityComposite
  on the small scroll (valid F1 is known to be a useless selector here).

RUN ORDER
  t01,t02 : v1 depth A/B (32-40 vs 28-44) — confirms the depth choice. Run first.
  t03-t08 : best performers on 28-44 (the evidence-based prior; if the A/B upset it, the user can
            see both and we adjust — 28-44 is the safe default so the suite is not blocked).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:
    event_accumulator = None


SMALL_SCROLL_ID = 20230827161847   # training scroll (has ink labels)
SCROLL4_ID      = 20231210132040   # loaded by visualizer init; not tested here
GOAL_SCROLL2_ID = 20230709155141   # transfer target (full-fragment test figure)

EPOCHS = 30

# shared regime applied to every run; per-run overrides win
BASE: Dict[str, Any] = {
    "epochs": EPOCHS,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 64,
    "num-workers": 2,
    "probe-int": 5,
    "eval-int": 10,
    "test-int": EPOCHS,          # test fires once, at the final epoch
    "test-scroll2-only": False,   # render ONLY the full goal-scroll2 transfer figure
    "eval-cooldown": 45,
    "no-hard-mining": True,
    "ring-negatives": True,
    "ring-label-source": "eroded",
    "train-d-start": 28,
    "train-d-end": 44,
    "channel-mixing-prob": 0.0,
}

# v1-family regularization head (shared InkDetector-style head)
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
    # ── depth-window A/B on the v1 baseline (only the train window differs) ──────────
    RunSpec(1, "t01_v1_d32_40", "depth_ab",
        {"arch": "v1", "l1-lambda": 7e-6, **V1_DROPOUT,
         "train-d-start": 32, "train-d-end": 40},
        why="v1 trained on the single 32-40 window — replicates the previous multiscroll "
            "baseline's depth setting under this campaign's single-scroll ring+reg regime. "
            "the 'less data' arm of the depth A/B."),

    RunSpec(2, "t02_v1_d28_44", "depth_ab",
        {"arch": "v1", "l1-lambda": 7e-6, **V1_DROPOUT,
         "train-d-start": 28, "train-d-end": 44},
        why="v1 trained on 28-44 (windows 28-36,32-40,36-44 ~ 3x the depth data). the 'more "
            "data' arm. if this beats t01 on hard probe + scroll2 transfer, 28-44 is confirmed "
            "as the going-forward default (already the c11/c12 preference)."),

    # ── best past performers (28-44), judged on scroll2 transfer ────────────────────
    RunSpec(3, "t03_asym_attn_pool", "best_perf",
        {"arch": "v12_asym_attn_pool", "l1-lambda": 7.65e-6, **V1_DROPOUT},
        why="top regularized performer overall (c12 t03, hard 0.466). asymmetric attention "
            "pooling weights informative spatial locations instead of flat-averaging — useful "
            "when ink occupies a small fraction of a tile."),

    RunSpec(4, "t04_no_cbam", "best_perf",
        {"arch": "v2_no_cbam", "l1-lambda": 9.08e-6, **V1_DROPOUT},
        why="simple strong CNN, no attention (c12 t10, hard 0.462). a clean high-capacity "
            "baseline; tests whether plain convolution already captures the fine ink signal."),

    RunSpec(5, "t05_max_depth_pool", "smaller_features",
        {"arch": "v10_max_depth_pool", "l1-lambda": 0.0, **NO_DROPOUT},
        why="max-over-depth pooling (c10 t12, hard 0.458). directly targets the 'features are "
            "smaller/thinner than expected' insight: take the STRONGEST ink response across "
            "depth so a thin ink layer is not diluted by mean pooling. tiny (286k) -> L1=0."),

    RunSpec(6, "t06_topk_depth", "smaller_features",
        {"arch": "v10_topk_depth", "l1-lambda": 0.0, **NO_DROPOUT},
        why="top-k depth pooling (c10 t13, hard 0.449). softer cousin of max-depth: averages "
            "the few strongest depth slices, robust to single-slice noise while still "
            "emphasising the concentrated ink response. tiny (286k) -> L1=0."),

    RunSpec(7, "t07_multiscale_3d", "smaller_features",
        {"arch": "v10_multiscale_3d", "l1-lambda": 0.0, **NO_DROPOUT},
        why="parallel multi-scale 3D kernels (c10 t08, hard 0.450). captures fine AND coarse "
            "structure simultaneously — the multi-resolution view motivated by scroll4's "
            "signal only appearing at finer voxel size. tiny (87k) -> L1=0."),

    RunSpec(8, "t08_v1_dilated", "smaller_features",
        {"arch": "v1", "l1-lambda": 7e-6, "conv3-dilation": 2, **V1_DROPOUT},
        why="v1 with a dilated final conv stage (novel here). widens spatial context WITHOUT "
            "extra downsampling, so the fine input-grid detail is preserved while still seeing "
            "neighbourhood context — a direct probe of the 'smaller features' hypothesis on the "
            "most reliable backbone."),
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
# NOTE: we deliberately do NOT treat bare "RuntimeError:" as fatal, because the test-figure
# code prints a caught "[ERROR] ... figure failed" traceback that does not threaten the
# already-saved weights. We match the harder signals above instead.


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


def build_cmd(python_exe: str, spec: RunSpec, runs_dir: Path, campaign_id: str) -> tuple[list[str], str]:
    merged = dict(BASE)
    merged.update(spec.overrides)
    exp_name = f"cmp_{campaign_id}_{spec.name}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def find_run_dir(runs_dir: Path, exp_name: str, start_ts: float):
    matches = [p for p in runs_dir.glob(f"{exp_name}_*") if p.is_dir()]
    # exclude the per-scroll figure suffixes if any; we want the primary run dir
    matches = [p for p in matches if not p.name.endswith(tuple(f"_s{sid}" for sid in
               (SMALL_SCROLL_ID, GOAL_SCROLL2_ID, SCROLL4_ID)))]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime)
    for p in reversed(matches):
        if p.stat().st_mtime >= start_ts - 5:
            return p
    return matches[-1]


def extract_metrics(run_dir):
    m = {"valid_f1_last": None, "probe_easy_last": None, "probe_hard_last": None}
    if run_dir is None or event_accumulator is None:
        return m
    evts = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not evts:
        return m
    ea = event_accumulator.EventAccumulator(str(evts[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    for key, tag in [("valid_f1", "P_M/F1_Score/Valid"),
                     ("probe_easy", "R_M/Probe/Easy/ReadabilityComposite"),
                     ("probe_hard", "R_M/Probe/Hard/ReadabilityComposite")]:
        if tag in avail:
            vals = [e.value for e in ea.Scalars(tag)]
            if vals:
                m[f"{key}_last"] = vals[-1]
    return m


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=90):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
    last_progress = time.time()
    last_epoch = 0
    saw_test = False
    while proc.poll() is None:
        time.sleep(15)
        try:
            lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        except Exception:
            continue
        tail = "".join(lines[-60:])
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
        # progress = new epoch line, OR entering the final test (which then has no epoch line
        # for ~40 min while it reads scroll2 — must not be flagged as a stall)
        for line in lines[-60:]:
            if "--- Epoch" in line:
                try:
                    ep = int(line.strip().split("/")[0].split()[-1])
                    if ep > last_epoch:
                        last_epoch = ep
                        last_progress = time.time()
                        print(f"[MONITOR] {line.strip()}")
                except Exception:
                    pass
        if (not saw_test) and ("Starting test figure generation" in tail or "Read Scroll2" in tail):
            saw_test = True
            last_progress = time.time()
            print("[MONITOR] final scroll2 transfer figure started")
        # while the scroll2 test is running, keep the watchdog alive on tqdm read updates
        if saw_test and ("Read Scroll2" in tail or "Predict Scroll2" in tail):
            last_progress = time.time()
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
    print("\n+-- campaign 14 results (ranked by hard probe) ---------------------------")
    print(f"|  {'run':<30} {'hard':>6} {'easy':>6} {'f1':>6}")
    print("|  " + "-" * 54)
    for r in sorted(completed,
                    key=lambda r: (r.get("metrics") or {}).get("probe_hard_last") or 0,
                    reverse=True):
        m = r.get("metrics") or {}
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "  ?  "
        print(f"|  {r['name'][-30:]:<30} {fmt(m.get('probe_hard_last')):>6} "
              f"{fmt(m.get('probe_easy_last')):>6} {fmt(m.get('valid_f1_last')):>6}")
    print("+--" + "-" * 56 + "\n")


def main():
    parser = argparse.ArgumentParser(description="campaign 14 -- best performers vs scroll2 transfer")
    parser.add_argument("--campaign-id", type=str, default="c14_2026_06_28")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=90.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir = repo_root / "runs_campaign14"
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
    print(f"[campaign14] {len(done_ids)} done, {len(pending)} pending")

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
        run_dir = find_run_dir(runs_dir, exp_name, start_ts)
        metrics = extract_metrics(run_dir)
        rec = {
            "run_id": spec.run_id,
            "name": exp_name,
            "axis": spec.axis,
            "overrides": {**BASE, **spec.overrides},
            "run_dir": str(run_dir) if run_dir else None,
            "metrics": metrics,
            "return_code": rc,
            "crashed": crashed,
            "ended_at": now_utc(),
        }
        state.setdefault("completed", []).append(rec)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        h = metrics.get("probe_hard_last")
        print(f"[RUN {spec.run_id}] done rc={rc} crashed={crashed} "
              f"hard={h if h is None else round(h,3)}")
        print_summary(state["completed"])

    print("\n[campaign14] all runs processed.")
    print_summary(state.get("completed", []))


if __name__ == "__main__":
    main()
