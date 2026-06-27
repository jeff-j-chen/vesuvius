"""campaign_runner_5.py — fundamental signal detection redesign

core insight: global average pooling dilutes ink signal (covering <1% of tile
volume) by 160-1600×. no architecture we've tried can detect a signal that
gets averaged away before classification.

this campaign tests signal-detection approaches that bypass global pooling:
  MIL attention: explicitly learn WHICH positions matter, weight them up
  local normalization: remove the scroll baseline to amplify small ink deviations
  depth profile: treat absorption profile as a 1D time-series (bell-shaped = ink)
  spectral: ink creates characteristic frequency patterns in FFT domain
  per-voxel: force model to localize ink at sub-tile level via MIL max loss
  siamese: compare ink band to pre-band embedding to learn what makes it different
  autoencoder: ink = anomaly = high reconstruction error

monitoring: each run writes stdout+stderr to a log file; the runner polls every
15 seconds for crash indicators and epoch progress, kills stalled processes,
and prints real-time status. runs are tracked and can be retried individually.
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
SCROLL4_ID      = 20231210132040

BASE_OVERRIDES: Dict[str, Any] = {
    "epochs": 20,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 64,
    "num-workers": 2,       # zarr opened lazily inside each worker — no pickle limit
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
}

RUN_SPECS: List[RunSpec] = [

    # ── A: MIL attention (6 runs) ─────────────────────────────────────────────
    RunSpec(
        1, "t01_mil_attention", "mil",
        overrides={"arch": "v5_mil_attention"},
        why="MIL attention pooling: weights spatial positions by learned relevance; "
            "the ONLY pooling that doesn't dilute a signal occupying <1% of tile volume",
        expect="substantial hard probe improvement vs any campaign 2-4 result",
    ),
    RunSpec(
        2, "t02_mil_gated", "mil",
        overrides={"arch": "v5_mil_gated"},
        why="gated MIL (Ilse 2018): two-branch gate prevents attention collapse "
            "under class imbalance — more stable training than vanilla MIL attention",
        expect="comparable or better than t01 with more stable training curves",
    ),
    RunSpec(
        3, "t03_local_norm_mil", "mil",
        overrides={"arch": "v5_local_norm_mil"},
        why="local tile normalization + MIL attention: first amplify the small ink "
            "deviation from scroll baseline, then find where it occurs; two-stage fix",
        expect="best MIL result; local norm makes ink stand out before attention finds it",
    ),
    RunSpec(
        4, "t04_local_norm_mil_gated", "mil",
        overrides={"arch": "v5_local_norm_mil_gated"},
        why="local norm + gated MIL: stable two-branch gate on locally normalized signal",
        expect="most reliable training; should match t03 with less variance",
    ),
    RunSpec(
        5, "t05_mil_attention_diff", "mil",
        overrides={"arch": "v5_mil_attention", "input-mode": "diff"},
        why="MIL + differential absorption (ink - pre_band): physics encodes the signal, "
            "attention locates it; both independently attack the dilution problem",
        expect="additive improvement if physics encoding and attention are complementary",
    ),
    RunSpec(
        6, "t06_mil_gated_diff", "mil",
        overrides={"arch": "v5_mil_gated", "input-mode": "diff"},
        why="gated MIL + diff input: stable attention on a physically-clean signal; "
            "top expected performer of the campaign",
        expect="best hard probe score overall; most principled combination",
    ),

    # ── B: local normalization (2 runs) ───────────────────────────────────────
    RunSpec(
        7, "t07_local_norm_preact", "local_norm",
        overrides={"arch": "v5_local_norm_preact"},
        why="ablation: local norm alone with standard global avg pool; "
            "isolates whether the baseline removal helps with standard pooling",
        expect="moderate improvement; not as strong as MIL but a useful ablation",
    ),
    RunSpec(
        8, "t08_local_norm_diff", "local_norm",
        overrides={"arch": "v5_local_norm_preact", "input-mode": "diff"},
        why="local norm applied to differential signal: double detrending "
            "(diff removes global scroll baseline, local norm removes local tile baseline)",
        expect="stronger than t07; tests whether double detrending is additive",
    ),

    # ── C: depth profile approaches (4 runs) ──────────────────────────────────
    RunSpec(
        9, "t09_depth_profile_1d", "depth_profile",
        overrides={"arch": "v5_depth_profile_1d"},
        why="1D CNN over depth axis after spatial averaging: treat absorption profile as "
            "a time-series. ink creates a characteristic bell-shaped absorption peak "
            "across depth — this directly classifies the peak shape, not voxel values",
        expect="very different failure mode; may find patterns invisible to 3D conv",
    ),
    RunSpec(
        10, "t10_depth_transformer", "depth_profile",
        overrides={"arch": "v5_depth_profile_transformer"},
        why="transformer over depth positions: self-attention learns inter-depth "
            "relationships (e.g. absorption rises before depth 36, falls after); "
            "more expressive than 1D CNN for asymmetric profiles",
        expect="captures non-local depth patterns; may outperform 1D CNN on hard ROI",
    ),
    RunSpec(
        11, "t11_depth_variance_2d", "depth_profile",
        overrides={"arch": "v5_depth_variance_2d"},
        why="depth variance map: at ink positions, absorption varies strongly across "
            "depth (rise then fall); background is flat. zero-parameter physics feature — "
            "model classifies the variance pattern, not raw voxel values",
        expect="strong for easy ROI; unknown for hard; novel approach worth testing",
    ),
    RunSpec(
        12, "t12_depth_var_diff", "depth_profile",
        overrides={"arch": "v5_depth_variance_2d", "input-mode": "diff"},
        why="depth variance of differential signal: variance of (ink - pre) highlights "
            "positions where the ink-to-background contrast varies with depth",
        expect="may capture subtle ink absorption profiles better than single-band variance",
    ),

    # ── D: spectral features (2 runs) ─────────────────────────────────────────
    RunSpec(
        13, "t13_spectral_3d", "spectral",
        overrides={"arch": "v5_spectral_3d"},
        why="FFT magnitude spectrum per depth slice: ink creates characteristic spatial "
            "frequency patterns (absorption edges). at 7.91um, ink features below voxel "
            "size still show up as elevated high-frequency energy in the FFT",
        expect="different failure mode; may detect sub-voxel ink edge patterns",
    ),
    RunSpec(
        14, "t14_spectral_diff", "spectral",
        overrides={"arch": "v5_spectral_3d", "input-mode": "diff"},
        why="spectral features of differential signal: FFT of (ink - pre) removes "
            "low-frequency scroll background; residual high-frequency components = ink edges",
        expect="cleaner spectral features; better signal-to-noise in frequency domain",
    ),

    # ── E: per-voxel MIL (2 runs) ─────────────────────────────────────────────
    RunSpec(
        15, "t15_per_voxel_mil", "per_voxel",
        overrides={"arch": "v5_per_voxel_mil"},
        why="output 32×32 spatial heatmap + MIL max loss: model is explicitly trained "
            "to find WHERE in the tile the ink is; max(heatmap) = tile score. "
            "gradient flows back to specific voxels, not a tile average",
        expect="forces spatial localization; may find ink positions invisible to global pool",
    ),
    RunSpec(
        16, "t16_per_voxel_diff", "per_voxel",
        overrides={"arch": "v5_per_voxel_mil", "input-mode": "diff"},
        why="per-voxel localization on differential signal: find WHERE the differential "
            "absorption is highest, not just whether the overall tile has high diff",
        expect="sharpest spatial localization; most direct at finding ink positions",
    ),

    # ── F: siamese comparison (1 run) ─────────────────────────────────────────
    RunSpec(
        17, "t17_siamese_double", "siamese",
        overrides={"arch": "v5_siamese", "input-mode": "double"},
        why="siamese: encode ink_band and pre_band independently via shared backbone, "
            "classify on the embedding DIFFERENCE. learns what makes ink depth different "
            "from its reference, bypassing absolute value sensitivity",
        expect="robust to overall brightness variation; novel cross-band comparison",
    ),

    # ── G: autoencoder anomaly (1 run) ────────────────────────────────────────
    RunSpec(
        18, "t18_ae_anomaly", "anomaly",
        overrides={"arch": "v5_ae_anomaly"},
        why="autoencoder reconstruction error = ink score: model learns to reconstruct "
            "normal scroll patterns well; ink tiles are anomalous and reconstruct poorly. "
            "high reconstruction error -> ink. trained end-to-end with BCE on error->logit",
        expect="no reliance on positive label quality; novel signal; may detect faint ink",
    ),

    # ── H: best combinations (2 runs) ─────────────────────────────────────────
    RunSpec(
        19, "t19_best_combo", "combo",
        overrides={"arch": "v5_local_norm_mil_gated", "input-mode": "diff"},
        why="gated MIL + local norm + diff input: stacks all three independent improvements "
            "that each address the sub-voxel dilution problem from different angles",
        expect="highest hard probe of the campaign if axes are complementary",
    ),
    RunSpec(
        20, "t20_mil_ranking", "combo",
        overrides={"arch": "v5_mil_gated", "ranking-lambda": 0.2},
        why="gated MIL + pairwise ranking loss (lambda=0.2): attention finds the ink "
            "positions, ranking loss enforces that those positions score above background; "
            "two independent mechanisms both fighting the abstention failure mode",
        expect="improved score separation; ranking loss ensures attention positions matter",
    ),
]


# ── monitoring ────────────────────────────────────────────────────────────────

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
    """run a training subprocess, monitoring the log for crashes and progress.
    returns (return_code: int, crashed_early: bool)."""
    print(f"[MONITOR] log -> {log_path}")

    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(repo_root), env=env,
            stdout=lf, stderr=subprocess.STDOUT,
        )

    last_progress = time.time()
    last_epoch = 0

    while proc.poll() is None:
        time.sleep(15)

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            continue

        tail = lines[-40:]
        tail_text = "".join(tail)

        # crash detection
        for sig in CRASH_SIGNALS:
            if sig in tail_text:
                print(f"\n[MONITOR] CRASH DETECTED — '{sig}'")
                print("[MONITOR] last output:")
                print("".join(tail[-15:]))
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.wait()
                return proc.returncode or 1, True

        # epoch progress
        for line in tail:
            if "--- Epoch" in line:
                try:
                    ep = int(line.strip().split("/")[0].split()[-1])
                    if ep > last_epoch:
                        last_epoch = ep
                        last_progress = time.time()
                        print(f"[MONITOR] {line.strip()}")
                except Exception:
                    pass
            elif "Training completed" in line:
                last_progress = time.time()

        # stall detection
        if time.time() - last_progress > stall_minutes * 60:
            print(f"\n[MONITOR] STALL — no epoch progress in {stall_minutes} min")
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()
            return 1, True

    proc.wait()
    rc = proc.returncode
    if rc == 0:
        print(f"[MONITOR] completed successfully")
    else:
        print(f"[MONITOR] exited with rc={rc}")
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-20:]
            print("[MONITOR] last output:\n" + "".join(tail))
        except Exception:
            pass
    return rc, False


# ── helpers (unchanged from campaign runners 2-4) ─────────────────────────────

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


def find_run_dir(runs_dir: Path, exp_name: str, start_ts: float) -> Optional[Path]:
    matches = [p for p in runs_dir.glob(f"{exp_name}_*") if p.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime)
    for p in reversed(matches):
        if p.stat().st_mtime >= start_ts - 5:
            return p
    return matches[-1]


def _best_last(events):
    if not events:
        return None, None
    vals = [float(e.value) for e in events]
    return max(vals), vals[-1]


def extract_metrics(run_dir: Optional[Path]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "valid_f1_best": None, "valid_f1_last": None,
        "readability_best": None, "readability_last": None,
        "probe_easy_last": None, "probe_hard_last": None,
    }
    if run_dir is None or event_accumulator is None:
        return metrics
    event_files = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return metrics
    ea = event_accumulator.EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    avail = set(ea.Tags().get("scalars", []))
    tag_map = {
        "valid_f1":    "P_M/F1_Score/Valid",
        "readability": "R_M/ReadabilityComposite",
        "probe_easy":  "R_M/Probe/Easy/ReadabilityComposite",
        "probe_hard":  "R_M/Probe/Hard/ReadabilityComposite",
    }
    for key, tag in tag_map.items():
        if tag in avail:
            best, last = _best_last(ea.Scalars(tag))
            metrics[f"{key}_best"] = best
            metrics[f"{key}_last"] = last
    return metrics


def append_text(path: Path, text: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def ensure_sections(runs_md: Path, future_md: Path, campaign_id: str):
    marker = f"## Automated Campaign {campaign_id}"
    content = runs_md.read_text(encoding="utf-8") if runs_md.exists() else ""
    if marker not in content:
        append_text(runs_md, f"\n\n{marker}\n\ncampaign 5 — MIL, depth profile, spectral, per-voxel, siamese, AE approaches.\n")
    fm = future_md.read_text(encoding="utf-8") if future_md.exists() else ""
    if campaign_id not in fm:
        append_text(future_md, f"\n\n## campaign log ({campaign_id})\n\n- campaign 5 started\n")


def choose_next_spec(pending: List[RunSpec], completed: List[Dict]) -> RunSpec:
    if len(pending) == 1:
        return pending[0]
    baseline = next((r for r in completed if r.get("run_id") == 1), None)
    if baseline is None:
        return sorted(pending, key=lambda s: s.run_id)[0]
    base_hard = (baseline.get("metrics") or {}).get("probe_hard_last") or 0.0
    axis_score: Dict[str, float] = {}
    axis_count: Dict[str, int] = {}
    for rec in completed:
        m = rec.get("metrics") or {}
        h = m.get("probe_hard_last")
        axis = rec.get("axis")
        if axis is None or h is None:
            continue
        delta = float(h) - float(base_hard)
        axis_score[axis] = axis_score.get(axis, 0.0) + delta
        axis_count[axis] = axis_count.get(axis, 0) + 1
    for a in list(axis_score):
        axis_score[a] /= max(axis_count[a], 1)
    return sorted(pending, key=lambda s: (-(axis_score.get(s.axis, 0.0)), s.run_id))[0]


def print_run_summary(completed: List[Dict]):
    if not completed:
        return
    print("\n┌─ campaign 5 progress ──────────────────────────────────────────")
    print(f"│  {'run':<35} {'hard':>5} {'easy':>5} {'f1':>5}")
    print("│  " + "─" * 54)
    for r in completed:
        m = r.get("metrics") or {}
        hard = f"{m.get('probe_hard_last',0.0):.3f}" if m.get("probe_hard_last") is not None else "?"
        easy = f"{m.get('probe_easy_last',0.0):.3f}" if m.get("probe_easy_last") is not None else "?"
        f1   = f"{m.get('valid_f1_last',0.0):.3f}" if m.get("valid_f1_last") is not None else "?"
        print(f"│  {r['name'][-35:]:<35} {hard:>5} {easy:>5} {f1:>5}")
    print("└" + "─" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="campaign 5 — MIL + depth + spectral + per-voxel")
    parser.add_argument("--campaign-id", type=str, default="c5_2026_06_11")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=45.0)
    args = parser.parse_args()

    repo_root   = Path(__file__).resolve().parent
    runs_dir    = repo_root / "runs_campaign5"
    runs_dir.mkdir(exist_ok=True)
    runs_md     = repo_root / "runs.md"
    future_md   = repo_root / "FUTURE.md"
    state_dir   = runs_dir / "campaign_logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path  = state_dir / f"{args.campaign_id}_state.json"

    ensure_sections(runs_md, future_md, args.campaign_id)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"campaign_id": args.campaign_id, "created_at": now_utc(), "completed": [], "failed": []}

    if args.retry_failed and state.get("failed"):
        print(f"retrying {len(state['failed'])} failed run(s): {[r['name'] for r in state['failed']]}")
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

        if not pending or len(completed_records) >= target:
            break

        print_run_summary(completed_records)

        spec = choose_next_spec(pending, completed_records)
        merged = dict(base)
        merged.update(spec.overrides)
        exp_name = f"cmp_{args.campaign_id}_{spec.name}"
        log_path = state_dir / f"{exp_name}.log"

        append_text(runs_md,
            f"\n\n### {exp_name}\n"
            f"- started_at: {now_utc()}\n- status: started\n"
            f"- axis: {spec.axis}\n- why: {spec.why}\n- expected: {spec.expect}\n")

        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)
        print(f"\n{'='*60}")
        print(f"  run {spec.run_id:02d}/20: {spec.name}")
        print(f"  axis: {spec.axis}")
        print(f"  cmd: {' '.join(cmd[-6:])}")   # last 6 args for readability
        print(f"{'='*60}")

        start_ts = time.time()
        env = os.environ.copy()
        env.update({"MPLBACKEND": "Agg", "TF_ENABLE_ONEDNN_OPTS": "0", "TF_CPP_MIN_LOG_LEVEL": "3"})

        if args.dry_run:
            rc, crashed = 0, False
        else:
            rc, crashed = run_with_monitoring(cmd, repo_root, env, str(log_path), args.stall_minutes)

        run_dir = find_run_dir(runs_dir, exp_name, start_ts)
        metrics = extract_metrics(run_dir)
        pending_after = [s for s in pending if s.run_id != spec.run_id]
        next_spec = choose_next_spec(pending_after, completed_records + [{"run_id": spec.run_id, "axis": spec.axis, "metrics": metrics}]) if pending_after else None
        next_name = "none" if next_spec is None else f"{next_spec.run_id:02d}:{next_spec.name}"

        hard = metrics.get("probe_hard_last")
        easy = metrics.get("probe_easy_last")
        f1   = metrics.get("valid_f1_last")
        print(f"\n  RESULT: hard={hard}  easy={easy}  f1={f1}  ->  next={next_name}")

        if rc == 0:
            record = {
                "run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
                "overrides": merged, "why": spec.why, "expect": spec.expect,
                "run_dir": str(run_dir) if run_dir else None,
                "metrics": metrics, "ended_at": now_utc(), "next_planned": next_name,
            }
            state.setdefault("completed", []).append(record)
            append_text(runs_md,
                f"- status: completed\n- probe_hard={hard}  probe_easy={easy}  f1={f1}\n"
                f"- next_planned: {next_name}\n")
            append_text(future_md,
                f"- {record['ended_at']} | {spec.name} | hard={hard} | easy={easy} | f1={f1} | next={next_name}\n")
        else:
            fail_record = {
                "run_id": spec.run_id, "name": exp_name, "axis": spec.axis,
                "overrides": merged, "ended_at": now_utc(), "return_code": rc,
                "run_dir": str(run_dir) if run_dir else None,
                "metrics": metrics, "next_planned": next_name,
                "crashed_early": crashed,
            }
            state.setdefault("failed", []).append(fail_record)
            append_text(runs_md, f"- status: failed  rc={rc}  crashed_early={crashed}\n- next_planned: {next_name}\n")
            append_text(future_md, f"- {fail_record['ended_at']} | {spec.name} FAILED rc={rc} | next={next_name}\n")

        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print_run_summary(state.get("completed", []))
    print("campaign 5 finished or reached target run count")


if __name__ == "__main__":
    main()
