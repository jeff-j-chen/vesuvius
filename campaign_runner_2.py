"""campaign_runner_2.py — architecture search campaign

sweeps 20 model architecture variants while holding all other training
hyperparameters fixed at values confirmed by campaign 1:
  - channel-mixing-prob 0.0 (no depth permutation)
  - small scroll 20230827161847 for fast iteration
  - 20-epoch runs with probe every 5 epochs

each run differs only in --arch, so metric differences are directly attributable
to architectural choices.
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
    # confirmed best from campaign 1: no depth channel permutation
    "channel-mixing-prob": 0.0,
    # zero all dropout and l1 so architecture is the only variable
    "conv1-drop": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "l1-lambda": 0.0,
    # disable hard mining so the model is the only determinant
    "no-hard-mining": True,
}


RUN_SPECS: List[RunSpec] = [
    # ── head ablations ────────────────────────────────────────────────────────
    RunSpec(
        run_id=1,
        name="t01_slim_head",
        axis="head",
        overrides={"arch": "v2_slim_head"},
        why="replace 5-layer MLP head with 2-layer (256→64→1); deep heads may memorize",
        expect="comparable F1, improved readability from less head overfitting",
    ),

    # ── attention ablations ───────────────────────────────────────────────────
    RunSpec(
        run_id=2,
        name="t02_no_cbam",
        axis="attention",
        overrides={"arch": "v2_no_cbam"},
        why="remove CBAM entirely; test if attention actually helps on 32×32 tiles",
        expect="faster training, potentially cleaner features without attention noise",
    ),
    RunSpec(
        run_id=3,
        name="t03_se_only",
        axis="attention",
        overrides={"arch": "v2_se_only"},
        why="SE blocks (channel-only attention); removes spatial CBAM component",
        expect="lighter attention with channel recalibration; simpler than CBAM",
    ),
    RunSpec(
        run_id=4,
        name="t04_eca",
        axis="attention",
        overrides={"arch": "v2_eca"},
        why="efficient channel attention (1D conv over channels, zero FC overhead)",
        expect="minimal parameter overhead with cross-channel recalibration",
    ),

    # ── residual / skip connections ───────────────────────────────────────────
    RunSpec(
        run_id=5,
        name="t05_residual",
        axis="skip_conn",
        overrides={"arch": "v2_residual"},
        why="add ResBlock3D after each CBAM conv stage; identity bypass for gradient flow",
        expect="more stable training curves, potentially better readability",
    ),
    RunSpec(
        run_id=6,
        name="t06_residual_no_cbam",
        axis="skip_conn",
        overrides={"arch": "v2_residual_no_cbam"},
        why="pure residual backbone with no attention; isolates residual benefit",
        expect="separates skip-connection effect from attention effect",
    ),
    RunSpec(
        run_id=7,
        name="t07_bottleneck",
        axis="skip_conn",
        overrides={"arch": "v2_bottleneck"},
        why="bottleneck residual (1×1 reduce→3×3→1×1 expand + skip); ResNet-50 style",
        expect="parameter efficiency with residual flow; less overfitting",
    ),
    RunSpec(
        run_id=8,
        name="t08_preact_res",
        axis="skip_conn",
        overrides={"arch": "v2_preact_res"},
        why="pre-activation residual (BN→ReLU→conv + skip); ResNet-v2 style",
        expect="cleaner skip-path gradient; better generalization in deeper nets",
    ),

    # ── backbone width / depth ────────────────────────────────────────────────
    RunSpec(
        run_id=9,
        name="t09_wider_shallow",
        axis="depth_width",
        overrides={"arch": "v2_wider_shallow"},
        why="2 conv blocks (1→64→256), fewer abstraction levels; less spatial compression",
        expect="better readability if 3 pooling stages over-compresses 32×32 input",
    ),
    RunSpec(
        run_id=10,
        name="t10_slim_all",
        axis="depth_width",
        overrides={"arch": "v2_slim_all"},
        why="narrow backbone (1→16→64→128) + slim head; tests overparameterization",
        expect="less overfitting; improved probe scores if model is too large",
    ),
    RunSpec(
        run_id=11,
        name="t11_deeper",
        axis="depth_width",
        overrides={"arch": "v2_deeper"},
        why="4-block backbone (32→128→256→384) with 3 MaxPool stages",
        expect="more abstraction capacity; useful if current 3-level model under-fits",
    ),

    # ── factorization and kernel geometry ────────────────────────────────────
    RunSpec(
        run_id=12,
        name="t12_factorized_depth",
        axis="factorized",
        overrides={"arch": "v2_factorized_depth"},
        why="each conv block replaced by (3,1,1) depth-conv + (1,3,3) spatial-conv in sequence; "
            "models depth and spatial axes independently; matches the depth-ordering insight",
        expect="improved readability by respecting scroll geometry structure",
    ),
    RunSpec(
        run_id=13,
        name="t13_asymmetric_first",
        axis="factorized",
        overrides={"arch": "v2_asymmetric_first"},
        why="first conv is (1,3,3) — spatial only, no depth mixing; "
            "depth mixing begins at layer 2; delays depth-spatial coupling",
        expect="cleaner first-layer spatial features before depth integration",
    ),

    # ── pooling variants ──────────────────────────────────────────────────────
    RunSpec(
        run_id=14,
        name="t14_strided_conv",
        axis="pooling",
        overrides={"arch": "v2_strided_conv"},
        why="replace MaxPool3d with strided Conv3d; learnable downsampling "
            "may preserve weak ink signals that max-pool discards",
        expect="better weak-signal retention; improved hard-probe scores",
    ),
    RunSpec(
        run_id=15,
        name="t15_dual_pool",
        axis="pooling",
        overrides={"arch": "v2_dual_pool"},
        why="concat global avg + global max pool (512-dim input to head); "
            "avg captures mean activation, max captures peak ink evidence",
        expect="complementary pooling signals; improved score separation",
    ),

    # ── normalization variants ────────────────────────────────────────────────
    RunSpec(
        run_id=16,
        name="t16_group_norm",
        axis="normalization",
        overrides={"arch": "v2_group_norm"},
        why="GroupNorm(8, ch) instead of BatchNorm3d; batch-size independent statistics; "
            "more stable with highly variable ink/background ratio per batch",
        expect="more consistent training, better cross-scroll generalization",
    ),
    RunSpec(
        run_id=17,
        name="t17_no_norm_drop",
        axis="normalization",
        overrides={"arch": "v2_no_norm_drop"},
        why="no BatchNorm at all, heavier dropout instead; "
            "BN creates statistical coupling between samples that may hurt generalization",
        expect="interesting baseline; slower convergence but possibly better calibration",
    ),

    # ── fundamentally different architectures ─────────────────────────────────
    RunSpec(
        run_id=18,
        name="t18_depth_project",
        axis="architecture",
        overrides={"arch": "v2_depth_project"},
        why="reshape (B,1,D,H,W)→(B,D,H,W) and use a 2D CNN; "
            "treats 8 depth slices as independent channels (like RGB); "
            "removes depth-spatial entanglement entirely",
        expect="different failure modes; worth inspecting depth-channel weight patterns",
    ),
    RunSpec(
        run_id=19,
        name="t19_two_stream",
        axis="architecture",
        overrides={"arch": "v2_two_stream"},
        why="parallel depth-stream (1D conv on spatial-averaged signal) + "
            "spatial-stream (2D conv on depth-averaged signal), merged before head; "
            "explicit decomposition of depth profile vs spatial texture",
        expect="each stream specializes; merged representation may be more discriminative",
    ),
    RunSpec(
        run_id=20,
        name="t20_inception_first",
        axis="architecture",
        overrides={"arch": "v2_inception_first"},
        why="inception-style entry: parallel (1,3,3) spatial + (3,1,1) depth + (1,1,1) pointwise; "
            "multi-scale feature extraction from layer 1 without committing to one kernel",
        expect="richer early features; may help with diverse ink signal morphology",
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
            # False booleans are simply omitted (flag not set)
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

    for key in ("probe_easy", "probe_hard", "probe_scroll4"):
        if tag_map[key] in available:
            ev = ea.Scalars(tag_map[key])
            metrics[f"{key}_last"] = float(ev[-1].value) if ev else None

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
    future_marker = f"## Automated Campaign Log ({campaign_id})"

    runs_content = runs_md.read_text(encoding="utf-8") if runs_md.exists() else ""
    if run_marker not in runs_content:
        append_text(
            runs_md,
            "\n\n"
            f"{run_marker}\n\n"
            "architecture search campaign — 20 variants, all other settings fixed.\n"
            "channel-mixing-prob=0.0 throughout (confirmed best from campaign 1).\n",
        )

    future_content = future_md.read_text(encoding="utf-8") if future_md.exists() else ""
    if future_marker not in future_content:
        append_text(
            future_md,
            "\n\n"
            f"## {future_marker}\n\n"
            "- architecture search campaign started\n",
        )


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
    parser = argparse.ArgumentParser(description="architecture search campaign")
    parser.add_argument("--campaign-id", type=str, default="arch_search_2026_06_10")
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
        state = {
            "campaign_id": args.campaign_id,
            "created_at": now_utc(),
            "completed": [],
            "failed": [],
        }

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

        append_text(
            runs_md,
            "\n\n"
            f"### Test {spec.run_id:02d}: {exp_name}\n"
            f"- started_at: {now_utc()}\n"
            f"- status: started\n"
            f"- arch: {merged.get('arch', 'v1')}\n"
            f"- axis: {spec.axis}\n"
            f"- why: {spec.why}\n"
            f"- expected: {spec.expect}\n",
        )

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
                "- status: completed\n"
                f"- run_dir: {record['run_dir']}\n"
                f"- results: valid_f1_last={metrics.get('valid_f1_last')}, "
                f"readability_last={metrics.get('readability_last')}, "
                f"probe_easy_last={metrics.get('probe_easy_last')}, "
                f"probe_hard_last={metrics.get('probe_hard_last')}\n"
                f"- next_planned_based_on_results: {next_name}\n",
            )
            append_text(
                future_md,
                f"- {record['ended_at']} | arch={merged.get('arch')} | "
                f"readability_last={metrics.get('readability_last')} | "
                f"probe_easy={metrics.get('probe_easy_last')} | "
                f"probe_hard={metrics.get('probe_hard_last')} | next={next_name}\n",
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
