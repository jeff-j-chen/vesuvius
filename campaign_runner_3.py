"""campaign_runner_3.py — architecture search campaign 3

builds on campaign 2 learnings:
  - preact_res and residual_no_cbam were the strongest readability performers
  - t11_deeper (4 blocks) had the best hard-probe score
  - CBAM consistently hurts; ECA is neutral
  - deeper > wider for hard probe sensitivity

this campaign explores:
  A) clean preact baselines (control + deeper + wider)
  B) depth-axis specialization (depth attention, depth squeeze)
  C) multi-scale and non-local context
  D) pooling variants on the preact backbone
  E) structural variants (asym first, dilated, bottleneck)
  F) normalization alternatives (instance norm)

fixed controls identical across all runs:
  - no hard mining
  - no dropout, no l1
  - channel_mixing_prob=0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
SCROLL4_ID = 20231210132040

BASE_OVERRIDES: Dict[str, Any] = {
    "epochs": 20,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 96,
    "num-workers": 2,
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 30,
    "hm-frac": 0.05,
    "channel-mixing-prob": 0.0,
    # eliminate all non-arch variables
    "conv1-drop": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "l1-lambda": 0.0,
    "no-hard-mining": True,
}


RUN_SPECS: List[RunSpec] = [

    # ── A: preact baselines (control + scale) ────────────────────────────────
    RunSpec(
        run_id=1,
        name="t01_preact_baseline",
        axis="preact_scale",
        overrides={"arch": "v3_preact_baseline"},
        why="clean re-run of campaign 2 winner with all bug fixes (no hooks, correct cuDNN); establishes true control",
        expect="improved readability scores vs campaign 2 t08 due to bug fixes",
    ),
    RunSpec(
        run_id=2,
        name="t02_linear_head",
        axis="simplification",
        overrides={"arch": "v3_linear_head"},
        why="most aggressive head simplification: pool → single Linear(256,1), no intermediate layers; "
            "t01_slim_head (2-layer) was visually best in campaign 2 — does 1-layer go further? "
            "fewer head parameters = less per-tile discrimination = coarser, more coherent outputs",
        expect="lower F1 but improved coherence score; prediction map looks less scattered",
    ),
    RunSpec(
        run_id=3,
        name="t03_depth_project_deep",
        axis="simplification",
        overrides={"arch": "v3_depth_project_deep"},
        why="deeper 2D CNN treating depth as channels (64→256→512→512, 3rd conv block); "
            "t18_depth_project was 2nd best visually in campaign 2 — adding depth may help further; "
            "fully decouples depth selection from spatial pattern recognition",
        expect="best visual coherence in the campaign; improved coverage_recall",
    ),
    RunSpec(
        run_id=4,
        name="t04_smooth_sigma1",
        axis="smoothing",
        overrides={"arch": "v3_preact_baseline", "smooth-sigma": 1.0},
        why="test-time Gaussian blur (sigma=1 tile) on prediction maps; no training change; "
            "directly tests whether scattered predictions are inherently coherent but display as noise; "
            "if coherence metric improves substantially: the model already knows the right regions",
        expect="improved coherence and visual readability; slight loss of topk precision",
    ),
    RunSpec(
        run_id=5,
        name="t05_smooth_sigma2",
        axis="smoothing",
        overrides={"arch": "v3_preact_baseline", "smooth-sigma": 2.0},
        why="stronger Gaussian blur (sigma=2 tiles); tests how much spatial integration helps; "
            "if sigma=1 improves hard ROI more than sigma=2: predictions are locally structured "
            "but not globally structured — different conclusion than sigma=1 < sigma=2",
        expect="higher coherence than t04 but possible loss of local contrast",
    ),

    # ── B: depth-axis specialization ─────────────────────────────────────────
    RunSpec(
        run_id=6,
        name="t06_depth_attn",
        axis="depth_axis",
        overrides={"arch": "v3_depth_attn"},
        why="1D attention over depth slices before second pool; learns which depth windows carry ink signal",
        expect="improved hard probe; more stable across depth-variable scrolls",
    ),
    RunSpec(
        run_id=7,
        name="t07_depth_squeeze",
        axis="depth_axis",
        overrides={"arch": "v3_depth_squeeze"},
        why="compress depth axis first via learned conv, then process spatially with 2D CNN; "
            "explicit separation: which depth has ink, then what does ink look like spatially",
        expect="different failure modes; may capture depth profile better than joint 3D conv",
    ),

    # ── C: multi-scale and non-local context ─────────────────────────────────
    RunSpec(
        run_id=8,
        name="t08_fpn",
        axis="multiscale",
        overrides={"arch": "v3_fpn"},
        why="feature pyramid: pool features from stride-1, -2, -4 and concat; "
            "ink may be easier to detect at a different scale depending on region",
        expect="improved hard probe if hard-region ink appears at a different spatial frequency",
    ),
    RunSpec(
        run_id=9,
        name="t09_multiscale_pool",
        axis="multiscale",
        overrides={"arch": "v3_multiscale_pool"},
        why="spatial pyramid pooling (1x1, 2x2, 4x4); retains some spatial layout info lost by global pool",
        expect="complementary to fpn; preserves coarse spatial positions which global pool discards",
    ),
    RunSpec(
        run_id=10,
        name="t10_nonlocal",
        axis="multiscale",
        overrides={"arch": "v3_nonlocal"},
        why="non-local means block for long-range spatial context; "
            "an ink tile near other ink tiles should score higher — conv alone cannot capture this",
        expect="improved local contrast metric; may help hard probe if hard ink is clustered",
    ),

    # ── D: pooling variants on preact backbone ────────────────────────────────
    RunSpec(
        run_id=11,
        name="t11_spatial_attn_pool",
        axis="pooling",
        overrides={"arch": "v3_spatial_attn_pool"},
        why="learned spatial attention weight map for global pooling instead of uniform average; "
            "in hard regions ink is spatially localized — uniform avg dilutes it",
        expect="improved hard probe precision; sharper response on ink-tile locations",
    ),
    RunSpec(
        run_id=12,
        name="t12_preact_gem",
        axis="pooling",
        overrides={"arch": "v3_preact_gem"},
        why="preact backbone + geometric mean pooling; emphasizes peak responses over uniform average",
        expect="better at detecting sparse/faint ink signal than avg pool",
    ),
    RunSpec(
        run_id=13,
        name="t13_preact_dual_pool",
        axis="pooling",
        overrides={"arch": "v3_preact_dual_pool"},
        why="concat avg+max pool; avg captures mean level, max captures peak signal — both useful for faint ink",
        expect="improved score separation between hard-positive and hard-negative tiles",
    ),

    # ── E: structural variants ────────────────────────────────────────────────
    RunSpec(
        run_id=14,
        name="t14_preact_asym",
        axis="structural",
        overrides={"arch": "v3_preact_asym"},
        why="preact backbone + (1,3,3) first conv (spatial before depth coupling); "
            "t13 in campaign 2 showed this helps — now combined with proven preact backbone",
        expect="improved spatial feature quality; marginal readability improvement",
    ),
    RunSpec(
        run_id=15,
        name="t15_dilated_preact",
        axis="structural",
        overrides={"arch": "v3_dilated_preact"},
        why="dilation=2 in 3rd conv block; larger receptive field without extra parameters; "
            "faint/diffuse ink patterns may be better captured at larger scale",
        expect="improved recall@1%fpr; better at diffuse low-contrast ink",
    ),
    RunSpec(
        run_id=16,
        name="t16_preact_bottleneck",
        axis="structural",
        overrides={"arch": "v3_preact_bottleneck"},
        why="preact with bottleneck residuals (1x1→3x3→1x1); more layers at same cost → richer hierarchy",
        expect="competitive with preact_baseline with lower parameter count",
    ),

    # ── F: attention on proven backbone ──────────────────────────────────────
    RunSpec(
        run_id=17,
        name="t17_preact_eca",
        axis="attention",
        overrides={"arch": "v3_preact_eca"},
        why="preact residuals + ECA channel attention after each block; "
            "ECA was least harmful in campaign 2 — does it help on top of preact?",
        expect="marginal improvement over preact_baseline; ECA adds minimal overhead",
    ),

    # ── G: normalization alternatives ────────────────────────────────────────
    RunSpec(
        run_id=18,
        name="t18_focal_gamma1",
        axis="focal",
        overrides={"arch": "v3_preact_baseline", "focal-gamma": 1.0},
        why="focal loss gamma=1 on preact baseline; mild down-weighting of easy background tiles; "
            "directly tests whether the training signal (not architecture) is the bottleneck for hard ROI",
        expect="lower overall F1 but improved hard probe; broader, less conservative predictions",
    ),

    # ── H: wider model ───────────────────────────────────────────────────────
    RunSpec(
        run_id=19,
        name="t19_focal_gamma2",
        axis="focal",
        overrides={"arch": "v3_preact_baseline", "focal-gamma": 2.0},
        why="focal loss gamma=2 (standard focal loss setting); stronger suppression of easy negatives; "
            "classic medical imaging setting for rare/subtle positive detection",
        expect="further improvement in hard probe recall; possible F1 drop as model becomes more sensitive",
    ),

    # ── I: rerun campaign 2 residual_no_cbam (best readability) with bug fixes ──
    RunSpec(
        run_id=20,
        name="t20_focal_gamma3",
        axis="focal",
        overrides={"arch": "v3_preact_baseline", "focal-gamma": 3.0},
        why="focal loss gamma=3; aggressive suppression of easy negatives; "
            "tests whether even stronger focus on hard examples improves hard ROI at cost of easy metrics",
        expect="highest hard probe recall if focal down-weighting is the key; "
               "may degrade easy ROI significantly",
    ),
]


def now_utc() -> str:
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


def find_run_dir(runs_dir: Path, exp_name: str, start_ts: float) -> Optional[Path]:
    matches = [p for p in runs_dir.glob(f"{exp_name}_*") if p.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime)
    for p in reversed(matches):
        if p.stat().st_mtime >= start_ts - 5:
            return p
    return matches[-1]


def _metric_best_last(events: List[Any]) -> Dict[str, Optional[float]]:
    if not events:
        return {"best": None, "last": None}
    vals = [float(ev.value) for ev in events]
    return {"best": max(vals), "last": vals[-1]}


def extract_metrics(run_dir: Optional[Path]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "valid_f1_best": None,
        "valid_f1_last": None,
        "readability_best": None,
        "readability_last": None,
        "probe_easy_last": None,
        "probe_hard_last": None,
        "probe_scroll4_last": None,
    }
    if run_dir is None or event_accumulator is None:
        return metrics

    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return metrics

    event_files.sort(key=lambda p: p.stat().st_mtime)
    ea = event_accumulator.EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    tag_map = {
        "valid_f1":     "P_M/F1_Score/Valid",
        "readability":  "R_M/ReadabilityComposite",
        "probe_easy":   "R_M/Probe/Easy/ReadabilityComposite",
        "probe_hard":   "R_M/Probe/Hard/ReadabilityComposite",
        "probe_scroll4": "R_M/Probe/Scroll4_Pi/ReadabilityComposite",
    }
    available = set(ea.Tags().get("scalars", []))
    if tag_map["valid_f1"] in available:
        stats = _metric_best_last(ea.Scalars(tag_map["valid_f1"]))
        metrics["valid_f1_best"] = stats["best"]
        metrics["valid_f1_last"] = stats["last"]
    if tag_map["readability"] in available:
        stats = _metric_best_last(ea.Scalars(tag_map["readability"]))
        metrics["readability_best"] = stats["best"]
        metrics["readability_last"] = stats["last"]
    for key in ("probe_easy", "probe_hard", "probe_scroll4"):
        if tag_map[key] in available:
            ev = ea.Scalars(tag_map[key])
            metrics[f"{key}_last"] = float(ev[-1].value) if ev else None
    return metrics


def append_text(path: Path, text: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def ensure_sections(runs_md: Path, future_md: Path, campaign_id: str):
    run_marker = f"## Automated Campaign {campaign_id}"
    future_marker = f"## Automated Campaign Log ({campaign_id})"
    runs_content = runs_md.read_text(encoding="utf-8") if runs_md.exists() else ""
    if run_marker not in runs_content:
        append_text(runs_md, f"\n\n{run_marker}\n\narcitecture search campaign 3 — builds on preact_res and residual_no_cbam.\n")
    future_content = future_md.read_text(encoding="utf-8") if future_md.exists() else ""
    if future_marker not in future_content:
        append_text(future_md, f"\n\n## {future_marker}\n\n- campaign 3 started\n")


def choose_next_spec(pending: List[RunSpec], completed: List[Dict[str, Any]]) -> RunSpec:
    if len(pending) == 1:
        return pending[0]
    baseline = next((r for r in completed if r.get("run_id") == 1), None)
    if baseline is None:
        return sorted(pending, key=lambda s: s.run_id)[0]
    baseline_r = (baseline.get("metrics") or {}).get("readability_last")
    axis_score: Dict[str, float] = {}
    axis_count: Dict[str, int] = {}
    if baseline_r is not None:
        for rec in completed:
            m = rec.get("metrics") or {}
            current = m.get("readability_last")
            axis = rec.get("axis")
            if axis is None or current is None:
                continue
            delta = float(current) - float(baseline_r)
            axis_score[axis] = axis_score.get(axis, 0.0) + delta
            axis_count[axis] = axis_count.get(axis, 0) + 1
        for axis in list(axis_score):
            axis_score[axis] /= max(axis_count.get(axis, 1), 1)
    return sorted(pending, key=lambda s: (-(axis_score.get(s.axis, 0.0)), s.run_id))[0]


def main():
    parser = argparse.ArgumentParser(description="architecture search campaign 3")
    parser.add_argument("--campaign-id", type=str, default="arch_search3_2026_06_10")
    parser.add_argument("--sleep-hours", type=float, default=0.0)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="move failed runs back to pending so they are retried")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir = repo_root / "runs"
    runs_md = repo_root / "runs.md"
    future_md = repo_root / "FUTURE.md"
    state_dir = runs_dir / "campaign_logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{args.campaign_id}_state.json"

    ensure_sections(runs_md, future_md, args.campaign_id)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"campaign_id": args.campaign_id, "created_at": now_utc(), "completed": [], "failed": []}

    if args.retry_failed and state.get("failed"):
        print(f"retrying {len(state['failed'])} failed run(s): {[r['name'] for r in state['failed']]}")
        state["failed"] = []
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    target_runs = min(args.max_runs, len(RUN_SPECS))

    while True:
        completed_records = state.get("completed", [])
        completed_ids = {int(r["run_id"]) for r in completed_records}
        failed_ids = {int(r["run_id"]) for r in state.get("failed", [])}
        pending = [s for s in RUN_SPECS if s.run_id not in completed_ids and s.run_id not in failed_ids]

        if not pending or len(completed_records) >= target_runs:
            break

        spec = choose_next_spec(pending, completed_records)
        merged = dict(BASE_OVERRIDES)
        merged.update(spec.overrides)
        exp_name = f"cmp_{args.campaign_id}_{spec.name}"

        append_text(runs_md,
            f"\n\n### Test {spec.run_id:02d}: {exp_name}\n"
            f"- started_at: {now_utc()}\n"
            f"- status: started\n"
            f"- arch: {merged.get('arch', 'v1')}\n"
            f"- axis: {spec.axis}\n"
            f"- why: {spec.why}\n"
            f"- expected: {spec.expect}\n")

        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)
        print("Running", " ".join(cmd))

        start_ts = time.time()
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"

        rc = 0 if args.dry_run else subprocess.run(cmd, cwd=str(repo_root), env=env, check=False).returncode

        run_dir = find_run_dir(runs_dir, exp_name, start_ts)
        metrics = extract_metrics(run_dir)

        pending_after = [s for s in pending if s.run_id != spec.run_id]
        next_spec = choose_next_spec(pending_after, completed_records + [{"run_id": spec.run_id, "axis": spec.axis, "metrics": metrics}]) if pending_after else None
        next_name = "none" if next_spec is None else f"{next_spec.run_id:02d}:{next_spec.name}"

        if rc == 0:
            record = {"run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
                      "overrides": merged, "why": spec.why, "expect": spec.expect,
                      "run_dir": str(run_dir) if run_dir else None, "metrics": metrics,
                      "ended_at": now_utc(), "next_planned": next_name}
            state.setdefault("completed", []).append(record)
            append_text(runs_md,
                f"- status: completed\n- run_dir: {record['run_dir']}\n"
                f"- results: valid_f1_last={metrics.get('valid_f1_last')}, "
                f"readability_last={metrics.get('readability_last')}, "
                f"probe_easy_last={metrics.get('probe_easy_last')}, "
                f"probe_hard_last={metrics.get('probe_hard_last')}\n"
                f"- next_planned_based_on_results: {next_name}\n")
            append_text(future_md,
                f"- {record['ended_at']} | arch={merged.get('arch')} | "
                f"readability_last={metrics.get('readability_last')} | "
                f"probe_hard={metrics.get('probe_hard_last')} | next={next_name}\n")
        else:
            fail_record = {"run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
                           "overrides": merged, "ended_at": now_utc(), "return_code": rc,
                           "run_dir": str(run_dir) if run_dir else None, "metrics": metrics,
                           "next_planned": next_name}
            state.setdefault("failed", []).append(fail_record)
            append_text(runs_md, f"- status: failed\n- return_code: {rc}\n- next_planned_based_on_results: {next_name}\n")
            append_text(future_md, f"- {fail_record['ended_at']} | arch={merged.get('arch')} failed rc={rc} | next={next_name}\n")

        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        sleep_seconds = max(0.0, args.sleep_hours * 3600.0)
        if not args.dry_run and sleep_seconds > 0:
            print(f"sleeping {sleep_seconds:.0f}s before next run")
            time.sleep(sleep_seconds)

    print("campaign finished or reached target run count")


if __name__ == "__main__":
    main()
