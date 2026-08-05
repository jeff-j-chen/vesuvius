"""compare a throughput run against the archived arch baseline via tensorboard scalars"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


DEFAULT_BASELINE = Path("runs_archs/cmp_archs_c0base_ctx48_baseline_arch_closed_31_08-39-48")
DEFAULT_TAGS = [
    "P_M/F1_Score/Train",
    "P_M/F1_Score/Valid",
    "AUC/PR_AUC/Train",
    "AUC/PR_AUC/Valid",
    "G_M/Loss/Train",
    "G_M/Loss/Valid",
    "Learning_Rate",
    "Time_Elapsed",
]


def _latest_event(run_dir: Path) -> Path:
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"no tensorboard events found under {run_dir}")
    return events[-1]


def _load_scalars(run_dir: Path):
    event_path = _latest_event(run_dir)
    acc = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    return event_path, {tag: acc.Scalars(tag) for tag in acc.Tags().get("scalars", [])}


def _series_by_step(series):
    return {int(item.step): float(item.value) for item in series}


def compare_runs(baseline_dir: Path, candidate_dir: Path, max_step: int, tags: list[str]):
    b_event, b_scalars = _load_scalars(baseline_dir)
    c_event, c_scalars = _load_scalars(candidate_dir)

    print(f"baseline:  {baseline_dir}")
    print(f"  event:   {b_event.name}")
    print(f"candidate: {candidate_dir}")
    print(f"  event:   {c_event.name}")
    print("")

    for tag in tags:
        b_series = b_scalars.get(tag, [])
        c_series = c_scalars.get(tag, [])
        if not b_series or not c_series:
            print(f"[{tag}] missing")
            continue

        b_steps = _series_by_step([item for item in b_series if item.step <= max_step])
        c_steps = _series_by_step([item for item in c_series if item.step <= max_step])
        shared_steps = sorted(set(b_steps) & set(c_steps))
        if not shared_steps:
            print(f"[{tag}] no shared steps <= {max_step}")
            continue

        deltas = []
        print(f"[{tag}]")
        for step in shared_steps:
            b_val = b_steps[step]
            c_val = c_steps[step]
            delta = c_val - b_val
            deltas.append(abs(delta))
            print(f"  step={step} baseline={b_val:.6f} candidate={c_val:.6f} delta={delta:+.6f}")

        print(f"  shared_steps={len(shared_steps)} mean_abs_delta={sum(deltas) / len(deltas):.6f} max_abs_delta={max(deltas):.6f}")


def main():
    parser = argparse.ArgumentParser(description="compare throughput run scalars against archived baseline")
    parser.add_argument("candidate", type=Path, help="candidate run directory under runs_archs")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--max-step", type=int, default=4, help="maximum epoch step to compare")
    parser.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    args = parser.parse_args()

    compare_runs(args.baseline, args.candidate, args.max_step, args.tags)


if __name__ == "__main__":
    main()