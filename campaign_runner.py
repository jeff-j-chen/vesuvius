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

# reduce tensorflow/tensorboard startup noise in runner and children
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
}


RUN_SPECS: List[RunSpec] = [
    RunSpec(
        run_id=1,
        name="t01_baseline_probe1",
        axis="baseline",
        overrides={},
        why="baseline with per-epoch probe metrics for fast readability monitoring",
        expect="stable baseline for readability composite and probe trends",
    ),
    RunSpec(
        run_id=2,
        name="t02_no_channel_mix",
        axis="channel_mix",
        overrides={"channel-mixing-prob": 0.0},
        why="remove depth permutation which can break physical depth cues",
        expect="better local contrast and less spill compared with baseline",
    ),
    RunSpec(
        run_id=3,
        name="t03_low_channel_mix",
        axis="channel_mix",
        overrides={"channel-mixing-prob": 0.1},
        why="test partial channel mixing as lighter regularization",
        expect="middle ground between baseline and no channel mixing",
    ),
    RunSpec(
        run_id=4,
        name="t04_pool_max",
        axis="pooling",
        overrides={"pooling": "max"},
        why="test sparse-evidence pooling instead of averaging",
        expect="sharper positives and stronger local ranking",
    ),
    RunSpec(
        run_id=5,
        name="t05_pool_gem_p3",
        axis="pooling",
        overrides={"pooling": "gem", "gem-p": 3.0},
        why="test soft sparse pooling with learnable GeM behavior",
        expect="improved readability composite with controlled spill",
    ),
    RunSpec(
        run_id=6,
        name="t06_pool_gem_p4",
        axis="pooling",
        overrides={"pooling": "gem", "gem-p": 4.0},
        why="test stronger GeM emphasis on high-response regions",
        expect="higher contrast but possible stability tradeoff",
    ),
    RunSpec(
        run_id=7,
        name="t07_no_mix_gem",
        axis="combo",
        overrides={"channel-mixing-prob": 0.0, "pooling": "gem", "gem-p": 3.0},
        why="combine top two structural hypotheses from FUTURE notes",
        expect="best readability among early tests if hypotheses are right",
    ),
    RunSpec(
        run_id=8,
        name="t08_conv3_dil2",
        axis="receptive",
        overrides={"conv3-dilation": 2},
        why="increase within-tile receptive field while keeping 32x32 input",
        expect="better weak-stroke coverage with similar compute",
    ),
    RunSpec(
        run_id=9,
        name="t09_conv3_dil2_gem",
        axis="receptive",
        overrides={"conv3-dilation": 2, "pooling": "gem", "gem-p": 3.0},
        why="pair receptive-field increase with sparse pooling",
        expect="higher weak-region recall and improved composite",
    ),
    RunSpec(
        run_id=10,
        name="t10_hm_off",
        axis="hard_mining",
        overrides={"hm-frac": 0.0},
        why="test whether hard mining currently reinforces spill behavior",
        expect="potentially cleaner maps with lower aggressive positives",
    ),
    RunSpec(
        run_id=11,
        name="t11_hm_frac_005",
        axis="hard_mining",
        overrides={"hm-frac": 0.05},
        why="reduce hard-mined sample pressure relative to current default",
        expect="less over-brightening than full hard-mining fraction",
    ),
    RunSpec(
        run_id=12,
        name="t12_hn_cut_090",
        axis="hard_mining",
        overrides={"hn-cutoff": 0.9},
        why="mine only very confident hard negatives",
        expect="fewer but cleaner hard negatives and more stable training",
    ),
    RunSpec(
        run_id=13,
        name="t13_hn_cut_070",
        axis="hard_mining",
        overrides={"hn-cutoff": 0.7},
        why="mine broader hard-negative set for stronger suppression",
        expect="higher background suppression with possible recall hit",
    ),
    RunSpec(
        run_id=14,
        name="t14_hp_cut_035",
        axis="hard_mining",
        overrides={"hp-cutoff": 0.35},
        why="focus hard-positive mining on severe misses only",
        expect="less noisy hard-positive injection",
    ),
    RunSpec(
        run_id=15,
        name="t15_hp_cut_055",
        axis="hard_mining",
        overrides={"hp-cutoff": 0.55},
        why="mine broader hard-positive errors to boost weak recall",
        expect="higher weak recall with spill risk",
    ),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def dict_to_cli_args(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            value = int(value)
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
        "valid_f1": "P_M/F1_Score/Valid",
        "readability": "R_M/ReadabilityComposite",
        "probe_easy": "R_M/Probe/Easy/ReadabilityComposite",
        "probe_hard": "R_M/Probe/Hard/ReadabilityComposite",
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

    if tag_map["probe_easy"] in available:
        ev = ea.Scalars(tag_map["probe_easy"])
        metrics["probe_easy_last"] = float(ev[-1].value) if ev else None

    if tag_map["probe_hard"] in available:
        ev = ea.Scalars(tag_map["probe_hard"])
        metrics["probe_hard_last"] = float(ev[-1].value) if ev else None

    if tag_map["probe_scroll4"] in available:
        ev = ea.Scalars(tag_map["probe_scroll4"])
        metrics["probe_scroll4_last"] = float(ev[-1].value) if ev else None

    return metrics


def fmt_overrides(overrides: Dict[str, Any]) -> str:
    if not overrides:
        return "none"
    return ", ".join(f"{k}={v}" for k, v in overrides.items())


def append_text(path: Path, text: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def ensure_sections(runs_md: Path, future_md: Path, campaign_id: str):
    run_marker = f"## Automated Campaign {campaign_id}"
    future_marker = f"## 20. Automated Campaign Log ({campaign_id})"

    runs_content = runs_md.read_text(encoding="utf-8") if runs_md.exists() else ""
    if run_marker not in runs_content:
        append_text(
            runs_md,
            "\n\n"
            f"{run_marker}\n\n"
            "This section is auto-updated by campaign_runner.py.\n"
            "Each test entry includes what changed, why, expected result, observed result, and next planned run based on results.\n",
        )

    future_content = future_md.read_text(encoding="utf-8") if future_md.exists() else ""
    if future_marker not in future_content:
        append_text(
            future_md,
            "\n\n"
            f"{future_marker}\n\n"
            "- campaign started with automated sequential 30-epoch tests on small scroll 20230827161847\n",
        )


def choose_next_spec(pending: List[RunSpec], completed: List[Dict[str, Any]]) -> RunSpec:
    if len(pending) == 1:
        return pending[0]

    baseline = None
    for rec in completed:
        if rec.get("run_id") == 1:
            baseline = rec
            break

    if baseline is None:
        pending_sorted = sorted(pending, key=lambda s: s.run_id)
        return pending_sorted[0]

    baseline_r = baseline.get("metrics", {}).get("readability_last")
    axis_score: Dict[str, float] = {}
    axis_count: Dict[str, int] = {}

    if baseline_r is not None:
        for rec in completed:
            m = rec.get("metrics", {})
            current = m.get("readability_last")
            axis = rec.get("axis")
            if axis is None or current is None:
                continue
            delta = float(current) - float(baseline_r)
            axis_score[axis] = axis_score.get(axis, 0.0) + delta
            axis_count[axis] = axis_count.get(axis, 0) + 1

    for axis, total in list(axis_score.items()):
        axis_score[axis] = total / max(axis_count.get(axis, 1), 1)

    pending_sorted = sorted(
        pending,
        key=lambda s: (-(axis_score.get(s.axis, 0.0)), s.run_id),
    )
    return pending_sorted[0]


def main():
    parser = argparse.ArgumentParser(description="run scheduled readability campaign")
    parser.add_argument("--campaign-id", type=str, default="readability_2026_06_08")
    parser.add_argument("--sleep-hours", type=float, default=0.0)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
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
        state = {
            "campaign_id": args.campaign_id,
            "created_at": now_utc(),
            "completed": [],
            "failed": [],
        }

    completed_ids = {int(rec["run_id"]) for rec in state.get("completed", [])}
    failed_ids = {int(rec["run_id"]) for rec in state.get("failed", [])}
    remaining = [s for s in RUN_SPECS if s.run_id not in completed_ids and s.run_id not in failed_ids]

    if not remaining:
        print("No remaining runs")
        return

    target_runs = min(args.max_runs, len(RUN_SPECS))

    while len(state.get("completed", [])) < target_runs:
        completed_records = state.get("completed", [])
        completed_ids = {int(rec["run_id"]) for rec in completed_records}
        failed_ids = {int(rec["run_id"]) for rec in state.get("failed", [])}
        pending = [s for s in RUN_SPECS if s.run_id not in completed_ids and s.run_id not in failed_ids]

        if not pending:
            break

        spec = choose_next_spec(pending, completed_records)

        merged = dict(BASE_OVERRIDES)
        merged.update(spec.overrides)
        exp_name = f"cmp_{args.campaign_id}_{spec.name}"

        append_text(
            runs_md,
            "\n\n"
            f"### Test {spec.run_id:02d}: {exp_name}\n"
            f"- started_at: {now_utc()}\n"
            f"- status: started\n"
            f"- changed: {fmt_overrides(merged)}\n"
            f"- why: {spec.why}\n"
            f"- expected: {spec.expect}\n"
            "- next_planned_based_on_results: pending completion\n",
        )

        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)
        print("Running", " ".join(cmd))

        start_ts = time.time()
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"

        if args.dry_run:
            rc = 0
        else:
            proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
            rc = int(proc.returncode)

        run_dir = find_run_dir(runs_dir, exp_name, start_ts)
        metrics = extract_metrics(run_dir)

        completed_records = state.get("completed", [])
        pending_after_current = [s for s in pending if s.run_id != spec.run_id]
        next_spec = choose_next_spec(pending_after_current, completed_records + [{
            "run_id": spec.run_id,
            "name": exp_name,
            "axis": spec.axis,
            "metrics": metrics,
        }]) if pending_after_current else None
        next_name = "none" if next_spec is None else f"{next_spec.run_id:02d}:{next_spec.name}"

        if rc == 0:
            record = {
                "run_id": spec.run_id,
                "name": exp_name,
                "axis": spec.axis,
                "overrides": merged,
                "why": spec.why,
                "expect": spec.expect,
                "run_dir": str(run_dir) if run_dir else None,
                "metrics": metrics,
                "ended_at": now_utc(),
                "next_planned": next_name,
            }
            state.setdefault("completed", []).append(record)

            append_text(
                runs_md,
                "\n"
                "- status: completed\n"
                f"- run_dir: {record['run_dir']}\n"
                f"- results: valid_f1_last={metrics.get('valid_f1_last')}, readability_last={metrics.get('readability_last')}, probe_easy_last={metrics.get('probe_easy_last')}, probe_hard_last={metrics.get('probe_hard_last')}\n"
                f"- next_planned_based_on_results: {next_name}\n",
            )

            append_text(
                future_md,
                "\n"
                f"- {record['ended_at']} | test {spec.run_id:02d} {spec.name} | readability_last={metrics.get('readability_last')} | probe_easy={metrics.get('probe_easy_last')} | probe_hard={metrics.get('probe_hard_last')} | next={next_name}\n",
            )
        else:
            fail_record = {
                "run_id": spec.run_id,
                "name": exp_name,
                "axis": spec.axis,
                "overrides": merged,
                "ended_at": now_utc(),
                "return_code": rc,
                "run_dir": str(run_dir) if run_dir else None,
                "metrics": metrics,
                "next_planned": next_name,
            }
            state.setdefault("failed", []).append(fail_record)

            append_text(
                runs_md,
                "\n"
                "- status: failed\n"
                f"- return_code: {rc}\n"
                f"- run_dir: {fail_record['run_dir']}\n"
                f"- next_planned_based_on_results: {next_name}\n",
            )

            append_text(
                future_md,
                "\n"
                f"- {fail_record['ended_at']} | test {spec.run_id:02d} {spec.name} failed rc={rc} | next={next_name}\n",
            )

        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        if len(state.get("completed", [])) >= target_runs:
            break

        completed_ids = {int(rec["run_id"]) for rec in state.get("completed", [])}
        failed_ids = {int(rec["run_id"]) for rec in state.get("failed", [])}
        pending = [s for s in RUN_SPECS if s.run_id not in completed_ids and s.run_id not in failed_ids]
        if not pending:
            break

        sleep_seconds = max(0.0, args.sleep_hours * 3600.0)
        print(f"sleeping for {sleep_seconds:.0f} seconds before next run")
        if not args.dry_run and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print("campaign finished or reached target run count")


if __name__ == "__main__":
    main()
