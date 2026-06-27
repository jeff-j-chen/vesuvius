"""campaign_runner_4.py — drastic input/loss redesign

key hypotheses under test:
  - differential input (ink_band - pre_band) directly encodes carbon absorption signal
  - triple-band input (pre + ink + post) lets model learn contrast implicitly
  - pairwise ranking loss forces local score separation (fixes abstention failure mode)
  - soft depth labels teach the model the ink-signal gradient across depth
  - self-supervised band-identity pretraining bootstraps ink representations

fixed controls:
  no hard mining, no dropout, no l1, channel_mixing=0, sigma=1.5 on combination runs
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
    "batch-size": 32,
    "num-workers": 0,   # windows pipe buffer too small to pickle zarr dataset with workers>0
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 30,
    "hm-frac": 0.05,
    "channel-mixing-prob": 0.0,
    "conv1-drop": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "l1-lambda": 0.0,
    "no-hard-mining": True,
    "arch": "v3_preact_baseline",  # best clean baseline from campaign 3
}


RUN_SPECS: List[RunSpec] = [

    # ── A: input representation ───────────────────────────────────────────────
    RunSpec(
        run_id=1,
        name="t01_diff_input",
        axis="input",
        overrides={"input-mode": "diff"},
        why="subtract pre-ink band (20-28) from ink band (32-40); removes baseline scroll absorption, "
            "leaving only differential carbon absorption — the direct physical signature of ink",
        expect="cleaner ink signal without scroll-body noise; improved hard probe if ink features are subtle",
    ),
    RunSpec(
        run_id=2,
        name="t02_triple_input",
        axis="input",
        overrides={"input-mode": "triple"},
        why="concatenate pre(20-28) + ink(32-40) + post(40-48) as 24 depth channels; "
            "model sees full band context and must learn the contrast pattern implicitly",
        expect="model learns to compare bands; different failure mode from single-band; "
               "may detect faint ink through band-relative comparison",
    ),
    RunSpec(
        run_id=3,
        name="t03_diff_sigma15",
        axis="input",
        overrides={"input-mode": "diff", "smooth-sigma": 1.5},
        why="differential input + spatial smoothing (sigma=1.5, the identified sweet spot from campaign 3); "
            "combines better physics encoding with inference-time coherence improvement",
        expect="best hard probe and coherence of the input-only tier",
    ),
    RunSpec(
        run_id=4,
        name="t04_triple_sigma15",
        axis="input",
        overrides={"input-mode": "triple", "smooth-sigma": 1.5},
        why="triple band + sigma=1.5 smoothing; tests whether 3-band context + coherence fix stack",
        expect="strong if triple input is learning useful cross-band patterns",
    ),
    RunSpec(
        run_id=5,
        name="t05_diff_depth_project",
        axis="input",
        overrides={"input-mode": "diff", "arch": "v3_depth_project_deep"},
        why="differential input + 2D CNN treating depth as channels; "
            "the 2D CNN is well-suited to the 8-channel diff tensor (each channel = one depth-diff slice); "
            "was 2nd best visually in campaign 2 (as v2_depth_project)",
        expect="depth_project_deep may be better suited to differential input than 3D conv",
    ),
    RunSpec(
        run_id=6,
        name="t06_pretrain5_diff",
        axis="input",
        overrides={"pretrain-epochs": 5, "input-mode": "diff"},
        why="5 epochs of band-identity pretraining (can model tell ink band from flanking band?) "
            "followed by BCE fine-tuning on diff input; "
            "backbone learns differential absorption representation before ink classification",
        expect="better generalization to faint ink; contrastive pre-training encodes ink-specific features",
    ),

    # ── B: loss function ─────────────────────────────────────────────────────
    RunSpec(
        run_id=7,
        name="t07_ranking_01",
        axis="loss",
        overrides={"ranking-lambda": 0.1},
        why="pairwise ranking loss (lambda=0.1): every positive tile must outscore every negative "
            "in the batch by margin=0.3; directly attacks the abstention failure mode — "
            "the model cannot minimize loss by predicting everything as background",
        expect="improved recall, broader predictions; score distribution widens",
    ),
    RunSpec(
        run_id=8,
        name="t08_ranking_03",
        axis="loss",
        overrides={"ranking-lambda": 0.3},
        why="stronger ranking pressure (lambda=0.3); tests whether heavier ranking regularization "
            "further improves hard probe at cost of easy precision",
        expect="more aggressive recall; may reduce easy ROI performance",
    ),
    RunSpec(
        run_id=9,
        name="t09_focal2_ranking01",
        axis="loss",
        overrides={"focal-gamma": 2.0, "ranking-lambda": 0.1},
        why="focal loss (down-weight easy examples) + ranking loss (force separation); "
            "focal stops abstention by de-emphasizing confident background tiles, "
            "ranking forces positive tiles to stand out — both attack the same problem from different angles",
        expect="strongest improvement in hard probe of the loss-only tier; possible F1 trade-off",
    ),
    RunSpec(
        run_id=10,
        name="t10_diff_ranking01",
        axis="loss",
        overrides={"input-mode": "diff", "ranking-lambda": 0.1},
        why="better physics encoding (diff input) + training pressure toward separation (ranking); "
            "tests whether the two best single-axis improvements stack",
        expect="additive improvement from both; strong hard probe candidate",
    ),
    RunSpec(
        run_id=11,
        name="t11_diff_focal_ranking",
        axis="loss",
        overrides={"input-mode": "diff", "focal-gamma": 2.0, "ranking-lambda": 0.1},
        why="triple combination: differential physics encoding + focal down-weighting + ranking pressure",
        expect="potentially best hard probe in the campaign; kitchen sink for loss tier",
    ),

    # ── C: soft depth labels ──────────────────────────────────────────────────
    RunSpec(
        run_id=12,
        name="t12_soft_labels_03",
        axis="soft_labels",
        overrides={"soft-label-prob": 0.3, "soft-label-value": 0.3},
        why="30% of the time, labeled ink tiles are sampled from the flanking band instead, "
            "with label=0.3 (weak positive); model learns that ink fades gradually at depth edges — "
            "teaches it to be less binary at ambiguous depths",
        expect="improved calibration; model less overconfident; may help hard probe indirectly",
    ),
    RunSpec(
        run_id=13,
        name="t13_soft_labels_01",
        axis="soft_labels",
        overrides={"soft-label-prob": 0.3, "soft-label-value": 0.1},
        why="softer flanking label (0.1 instead of 0.3); very weak signal that the flanking bands "
            "contain trace ink; tests sensitivity of the label strength hyperparameter",
        expect="more conservative improvement than t12; calibration effect at flanking depths",
    ),
    RunSpec(
        run_id=14,
        name="t14_diff_soft03",
        axis="soft_labels",
        overrides={"input-mode": "diff", "soft-label-prob": 0.3, "soft-label-value": 0.3},
        why="diff input + soft labels; the diff naturally handles depth-band variation "
            "but soft labels additionally teach the model that weak differential signals are still ink",
        expect="synergy between physics encoding and label smoothing",
    ),
    RunSpec(
        run_id=15,
        name="t15_triple_soft03",
        axis="soft_labels",
        overrides={"input-mode": "triple", "soft-label-prob": 0.3, "soft-label-value": 0.3},
        why="triple input + soft labels; the model sees all three bands and learns that "
            "ink tiles in flanking bands should score at 0.3",
        expect="improved hard probe if triple input is learning meaningful cross-band patterns",
    ),

    # ── D: best combinations ─────────────────────────────────────────────────
    RunSpec(
        run_id=16,
        name="t16_diff_rank_soft_sig",
        axis="combo",
        overrides={"input-mode": "diff", "ranking-lambda": 0.1,
                   "soft-label-prob": 0.3, "soft-label-value": 0.3, "smooth-sigma": 1.5},
        why="diff input + ranking(0.1) + soft_labels(0.3) + sigma=1.5; "
            "stacks input physics, training separation pressure, label smoothing, and inference coherence",
        expect="best hard probe in the campaign if all four axes are complementary",
    ),
    RunSpec(
        run_id=17,
        name="t17_triple_rank_soft_sig",
        axis="combo",
        overrides={"input-mode": "triple", "ranking-lambda": 0.1,
                   "soft-label-prob": 0.3, "soft-label-value": 0.3, "smooth-sigma": 1.5},
        why="same as t16 but with triple input instead of diff; "
            "tests whether the implicit cross-band comparison outperforms the explicit diff",
        expect="strong; useful to compare vs t16 to determine best input mode",
    ),
    RunSpec(
        run_id=18,
        name="t18_diff_focal_rank_soft_sig",
        axis="combo",
        overrides={"input-mode": "diff", "focal-gamma": 2.0, "ranking-lambda": 0.1,
                   "soft-label-prob": 0.1, "soft-label-value": 0.1, "smooth-sigma": 1.5},
        why="full kitchen sink: diff + focal + ranking + soft labels + smoothing; "
            "softer labels (0.1) to avoid conflicting with focal's down-weighting of easy examples",
        expect="highest possible recall; may trade off easy ROI precision",
    ),
    RunSpec(
        run_id=19,
        name="t19_pretrain_diff_rank_sig",
        axis="combo",
        overrides={"pretrain-epochs": 5, "input-mode": "diff", "ranking-lambda": 0.1,
                   "smooth-sigma": 1.5},
        why="pretraining + diff + ranking + smoothing; tests whether contrastive initialization "
            "helps when combined with the best loss and inference fixes",
        expect="improved hard probe from better feature initialization; novel test point",
    ),
    RunSpec(
        run_id=20,
        name="t20_triple_focal_rank_soft_sig",
        axis="combo",
        overrides={"input-mode": "triple", "focal-gamma": 2.0, "ranking-lambda": 0.3,
                   "soft-label-prob": 0.3, "soft-label-value": 0.3, "smooth-sigma": 1.5},
        why="triple input kitchen sink: stronger ranking (0.3) to compensate for triple's implicit "
            "rather than explicit band comparison; all other improvements stacked",
        expect="complements t18; highest parameter count for ablation purposes",
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
        "valid_f1_best": None, "valid_f1_last": None,
        "readability_best": None, "readability_last": None,
        "probe_easy_last": None, "probe_hard_last": None, "probe_scroll4_last": None,
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
        append_text(runs_md, f"\n\n{run_marker}\n\ninput/loss redesign — differential absorption + ranking loss + soft labels.\n")
    future_content = future_md.read_text(encoding="utf-8") if future_md.exists() else ""
    if future_marker not in future_content:
        append_text(future_md, f"\n\n## {future_marker}\n\n- campaign 4 started\n")


def choose_next_spec(pending: List[RunSpec], completed: List[Dict[str, Any]]) -> RunSpec:
    if len(pending) == 1:
        return pending[0]
    baseline = next((r for r in completed if r.get("run_id") == 1), None)
    if baseline is None:
        return sorted(pending, key=lambda s: s.run_id)[0]
    baseline_r = (baseline.get("metrics") or {}).get("probe_hard_last") or 0.0
    axis_score: Dict[str, float] = {}
    axis_count: Dict[str, int] = {}
    for rec in completed:
        m = rec.get("metrics") or {}
        current = m.get("probe_hard_last")
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
    parser = argparse.ArgumentParser(description="campaign 4 — input/loss redesign")
    parser.add_argument("--campaign-id", type=str, default="arch_search4_2026_06_11")
    parser.add_argument("--sleep-hours", type=float, default=0.0)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir = repo_root / "runs_campaign4"
    runs_dir.mkdir(exist_ok=True)
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

    # override log_dir so runs land in runs_campaign4
    base_with_logdir = dict(BASE_OVERRIDES)
    base_with_logdir["log-dir"] = str(runs_dir)

    target_runs = min(args.max_runs, len(RUN_SPECS))

    while True:
        completed_records = state.get("completed", [])
        completed_ids = {int(r["run_id"]) for r in completed_records}
        failed_ids = {int(r["run_id"]) for r in state.get("failed", [])}
        pending = [s for s in RUN_SPECS if s.run_id not in completed_ids and s.run_id not in failed_ids]

        if not pending or len(completed_records) >= target_runs:
            break

        spec = choose_next_spec(pending, completed_records)
        merged = dict(base_with_logdir)
        merged.update(spec.overrides)
        exp_name = f"cmp_{args.campaign_id}_{spec.name}"

        append_text(runs_md,
            f"\n\n### Test {spec.run_id:02d}: {exp_name}\n"
            f"- started_at: {now_utc()}\n- status: started\n"
            f"- axis: {spec.axis}\n- why: {spec.why}\n- expected: {spec.expect}\n")

        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)
        print("Running", " ".join(cmd))

        start_ts = time.time()
        env = os.environ.copy()
        env.update({"MPLBACKEND": "Agg", "TF_ENABLE_ONEDNN_OPTS": "0", "TF_CPP_MIN_LOG_LEVEL": "3"})

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
                f"- results: f1={metrics.get('valid_f1_last')}, "
                f"probe_hard={metrics.get('probe_hard_last')}, "
                f"probe_easy={metrics.get('probe_easy_last')}\n"
                f"- next_planned_based_on_results: {next_name}\n")
            append_text(future_md,
                f"- {record['ended_at']} | {spec.name} | "
                f"probe_hard={metrics.get('probe_hard_last')} | "
                f"f1={metrics.get('valid_f1_last')} | next={next_name}\n")
        else:
            fail_record = {"run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
                           "overrides": merged, "ended_at": now_utc(), "return_code": rc,
                           "run_dir": str(run_dir) if run_dir else None, "metrics": metrics,
                           "next_planned": next_name}
            state.setdefault("failed", []).append(fail_record)
            append_text(runs_md, f"- status: failed\n- return_code: {rc}\n- next_planned: {next_name}\n")
            append_text(future_md, f"- {fail_record['ended_at']} | {spec.name} failed rc={rc} | next={next_name}\n")

        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        sleep_seconds = max(0.0, args.sleep_hours * 3600.0)
        if not args.dry_run and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print("campaign finished or reached target run count")


if __name__ == "__main__":
    main()
