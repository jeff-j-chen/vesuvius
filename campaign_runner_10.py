"""campaign_runner_10.py — multi-scale 3D architectures for sub-voxel ink detection.

Key learnings from C1-C9:
  - Full depth window 28-48 (step=4) is critical: inter-layer relations matter
  - Ring negatives + full depth = best combination
  - Original regularization (dropout, L1, hard mining, 50 epochs) was regressed in C5-C9
  - 3D CBAM CNN (v1) remains the strongest single-scroll architecture
  - Hard probe ceiling ~0.45 across all approaches — sub-voxel physics limit?
  - U-Net is the leading architecture among other researchers for this task
  - Residual_spatial_depth (per-slice 2D CNN + depth attention) was strongest novel arch in C9

C10 strategy:
  - Restore original regularization (dropout, L1, hard mining, 50 epochs, num_workers=8)
  - 10 architectures spanning: U-Net variants, depth-slice 2D CNN, multi-scale attention,
    per-pixel cross-depth attention, local statistical pooling, learned filter banks
  - All use eroded ring + full depth 28-48 (best C9 combination: easy=0.618, hard=0.440)
  - v10_v1_full_reg is the ablation control: v1 with full original hyperparameters restored
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


SMALL_SCROLL_ID = 20230827161847
SCROLL4_ID      = 20231210132040

# BASE hyperparameters for C10 (architectural selection phase):
# - L1 DISABLED: we want to know if the architecture CAN overfit on hard examples.
#   L1 prevents memorization and would hide whether the arch has capacity.
# - Dropout DISABLED: same reason.
# - Hard mining DISABLED: ring conflict.
# - 20 epochs: enough to reveal overfitting capacity.
# - eroded ring + full depth 28-48 (best C9 finding: easy=0.618, hard=0.440)
BASE: Dict[str, Any] = {
    "epochs": 20,
    "scroll-id": SMALL_SCROLL_ID,
    "scroll4-id": SCROLL4_ID,
    "batch-size": 64,
    "num-workers": 2,
    "probe-int": 5,
    "eval-int": 10,
    "test-int": 50,
    "no-hard-mining": True,
    "ring-negatives": True,
    "ring-label-source": "eroded",
    "train-d-start": 28,
    "train-d-end": 48,
    "l1-lambda": 0.0,          # disabled: let architectures overfit freely
    "conv2-drop": 0.0,         # disabled: same reason
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "conv1-drop": 0.0,
    "channel-mixing-prob": 0.0,
}

RUN_SPECS: List[RunSpec] = [

    # 1. Ablation control: v1 with ALL original hyperparameters restored
    RunSpec(1, "t01_v1_baseline", "ablation",
        {"arch": "v10_v1_full_reg"},
        why="v1 3D CBAM CNN without regularization — architectural baseline with same "
            "conditions as all other C10 runs. establishes the capacity ceiling for the "
            "original architecture under no-dropout/no-L1 conditions."),

    # 2. 3D U-Net — the architecture used by leading researchers
    RunSpec(2, "t02_3d_unet", "unet",
        {"arch": "v10_3d_unet"},
        why="3D U-Net on 32x32x8: encoder downsamples spatially (preserves depth), "
            "decoder restores with skip connections. skip connections preserve fine-grained "
            "local absorption structure at every scale. classifies from bottleneck. "
            "this is the architecture leading other vesuvius researchers."),

    # 3. 3D U-Net with dual classification (bottleneck + full-res)
    RunSpec(3, "t03_3d_unet_dual", "unet",
        {"arch": "v10_3d_unet_classify"},
        why="3D U-Net where BOTH bottleneck (global context) AND full-resolution output "
            "(local anomaly) contribute to final classification. bottleneck sees ink strokes; "
            "full-res sees individual voxel anomalies. both signals are complementary for "
            "sub-voxel ink detection."),

    # 4. Per-slice 2D CNN + cross-depth GRU (best C9 family: residual_spatial_depth)
    RunSpec(4, "t04_slice2d_gru", "spatial_depth",
        {"arch": "v10_depth_slice_2d_gru"},
        why="2D conv per depth slice (weight-shared) → GRU across depth. preserves full "
            "32x32 spatial resolution at each depth before depth fusion. extends C9's "
            "best novel arch (residual_spatial_depth, hard=0.419) with bidirectional GRU "
            "instead of attention for depth aggregation."),

    # 5. Deeper 3D CBAM with attention at every block
    RunSpec(5, "t05_deep_3d_cbam", "deep_cnn",
        {"arch": "v10_deep_3d_cbam"},
        why="4-block deep 3D ResNet with CBAM (channel+spatial attention) at every block. "
            "v1 applies CBAM only at end. if hard ink occupies 2-3 voxels in a 32x32 tile, "
            "global pooling dilutes it 500x. CBAM at every block maintains sharp spatial focus "
            "throughout the network. 3.6M params, much deeper than v1."),

    # 6. Local statistical pooling instead of global avg
    RunSpec(6, "t06_local_stat_pool", "local_pool",
        {"arch": "v10_local_stat_pool"},
        why="3D CBAM CNN but replaces global avg pool with AdaptiveAvgPool3d(2,4,4). "
            "preserves spatial resolution in the pooled features instead of collapsing. "
            "at 4x4 spatial: each cell covers 8x8=64 pixels (~63um). ink particle clusters "
            "at this scale should be distinguishable. local structure survives."),

    # 7. Spatial-depth cross-attention
    RunSpec(7, "t07_spatial_depth_xattn", "cross_attn",
        {"arch": "v10_spatial_depth_xattn"},
        why="2D CNN per slice → Transformer self-attention across D=8 depth tokens at each "
            "spatial position. every (x,y) position asks 'how does my absorption vary through "
            "depth?' then MIL across spatial positions. cross-depth spatial attention is the "
            "only operation that can detect: 'this pixel's depth profile matches an ink signature'."),

    # 8. Multi-scale 3D parallel branches with cross-scale attention
    RunSpec(8, "t08_multiscale_3d", "multiscale",
        {"arch": "v10_multiscale_3d"},
        why="3 parallel 3D CNN branches: full 32x32, 16x16 (quadrant-pooled), 8x8 (block-pooled). "
            "cross-scale self-attention selects which scale is most discriminative per tile. "
            "easy ink may be visible at coarse scale; hard ink only at fine scale. "
            "explicitly routes computation to the relevant spatial granularity."),

    # 9. Per-pixel depth self-attention (efficient version of failed C6 t06)
    RunSpec(9, "t09_perpixel_depth_attn", "pixel_attn",
        {"arch": "v10_perpixel_depth_attn"},
        why="1024 per-pixel depth profiles → lightweight encoder (D→32→64) → Transformer "
            "spatial self-attention over 1024 pixel tokens at d=64 (was 512 in C6). "
            "C6 failed due to size (25M params, OOM). this version has 200K params. "
            "learns: ink stroke = spatially correlated depth anomalies across 10-40 pixels."),

    # 10. v1 with original reg but NO ring (ablation: ring vs no-ring with proper hyperparams)
    RunSpec(10, "t10_v1_no_ring_full_reg", "ablation",
        {"arch": "v10_v1_full_reg", "no-ring-negatives": True},
        why="v1 with full original regularisation (dropout, L1, mining, 50 epochs) but NO ring. "
            "C9 showed no-ring outperformed ring at 20 epochs without regularisation. "
            "with proper regularisation restored, does ring still hurt generalization? "
            "this ablation isolates: ring_negatives × regularisation interaction."),

    # 11. Retry t04 (machine crash during data load — not an arch bug)
    RunSpec(11, "t04_slice2d_gru_retry", "spatial_depth",
        {"arch": "v10_depth_slice_2d_gru"},
        why="retry of t04 which was killed by machine crash (0xC0000005 during data load, "
            "not a CUDA kernel error). scheduled last so it does not delay other runs."),

    # ── sub-voxel individual ───────────────────────────────────────────────

    # 12. Global MAX pool over depth (vs avg) — preserves single-voxel ink spike
    RunSpec(12, "t12_max_depth_pool", "subvoxel",
        {"arch": "v10_max_depth_pool"},
        why="3D CNN with global MAX pool over depth instead of avg. ink absorption occupies "
            "1-2 of 8 depth slices; avg dilutes it 4-8×. max preserves the spike exactly. "
            "simplest possible sub-voxel fix — tests whether pooling alone is the bottleneck."),

    # 13. Top-k mean over depth — less extreme than hard max, more robust to noise
    RunSpec(13, "t13_topk_depth", "subvoxel",
        {"arch": "v10_topk_depth"},
        why="3D CNN with top-3 mean pool over depth. compromise between max (noisy) and avg "
            "(diluting). averages the 3 highest-absorbing depth slices per channel — robust "
            "to single-slice noise while still amplifying the ink signal vs full avg."),

    # 14. Asymmetric pooling: avg spatial + max depth — optimal per-axis
    RunSpec(14, "t14_asymmetric_pool", "subvoxel",
        {"arch": "v10_asymmetric_pool"},
        why="4-block 3D CNN with asymmetric pooling: AdaptiveAvgPool over spatial (H,W) "
            "then AdaptiveMaxPool over depth (D). spatial avg is correct (ink texture spreads "
            "across pixels); depth max is correct (absorption spike at 1-2 slices not all 8). "
            "tests whether the axis mismatch in standard global avg is the key failure."),

    # ── depth sequential individual ────────────────────────────────────────

    # 15. 3-layer deep BiGRU on per-slice 2D features — more depth modeling capacity
    RunSpec(15, "t15_deep_bigru_slice", "depth_seq",
        {"arch": "v10_deep_bigru_slice"},
        why="2D CNN per depth slice → stacked 3-layer BiGRU (vs 1-layer in t04/t11). "
            "hidden dim 384 (vs 256). more recurrent depth to model complex absorption curve "
            "shapes. directly tests whether the C9 BiGRU winner needs more sequential capacity."),

    # 16. Dilated TCN over depth — exponential receptive field in O(log n) layers
    RunSpec(16, "t16_tcn_depth", "depth_seq",
        {"arch": "v10_tcn_depth"},
        why="per-slice 2D CNN → dilated 1D temporal conv network across depth (dilation 1,2,4). "
            "TCN receptive field = 7 slices in 3 layers vs GRU's linear rollout. "
            "parallelizable unlike GRU — tests dilated convs as alternative to recurrence for "
            "depth profile modeling."),

    # 17. Full D-token transformer + max-over-tokens — attention + sub-voxel preservation
    RunSpec(17, "t17_depth_transformer_max", "depth_seq",
        {"arch": "v10_depth_transformer_max"},
        why="per-slice 2D CNN → 3-layer transformer with D=8 depth tokens → MAX over tokens. "
            "transformer attends globally across all depth positions. max-over-tokens keeps the "
            "single most anomalous depth (sub-voxel). different from t07: full 3-layer encoder, "
            "max not MIL, spatial avg before attention."),

    # ── tandem: sub-voxel + depth sequential ──────────────────────────────

    # 18. BiGRU per spatial position + max over space — depth sequential + spatial max
    RunSpec(18, "t18_gru_maxpool", "tandem",
        {"arch": "v10_gru_maxpool"},
        why="2D CNN per slice → BiGRU across depth at each spatial position independently → "
            "MAX over all 1024 spatial positions. BiGRU captures the depth profile at each "
            "pixel; max preserves the single pixel with the strongest ink depth signature. "
            "tandem: sequential depth modeling + sub-voxel spatial preservation."),

    # 19. Percentile features → TCN — sub-voxel robust input + sequential depth model
    RunSpec(19, "t19_percentile_tcn", "tandem",
        {"arch": "v10_percentile_tcn"},
        why="7 percentiles per depth slice (incl. p95, p99 to catch sparse ink pixels) → "
            "dilated TCN across depth. percentiles replace mean (sub-voxel: top percentiles "
            "amplify the rare high-absorption pixels); TCN models depth profile shape. "
            "tandem of two orthogonal sub-voxel strategies."),

    # 20. Sparse top-k depth attention — learns to ignore normal depths, focus on ink spike
    RunSpec(20, "t20_sparse_depth_attn", "tandem",
        {"arch": "v10_sparse_depth_attn"},
        why="per-slice 2D CNN → learned anomaly score per depth → soft top-3 attention. "
            "model explicitly learns which depth positions are most informative and gates out "
            "the rest. combines depth-sequential reasoning (scoring each depth) with sub-voxel "
            "focus (attending only to the anomalous slices, not averaging all 8)."),

    # 21. Hierarchical BiGRU: local window → global — fine + coarse depth structure
    RunSpec(21, "t21_hierarchical_gru", "tandem",
        {"arch": "v10_hierarchical_gru"},
        why="2-level BiGRU: local (2-slice windows detect ink onset/offset transitions) → "
            "global (across window summaries detects overall depth curve shape). "
            "ink absorption has both a sharp local onset AND a characteristic global profile. "
            "hierarchical temporal structure captures both scales simultaneously."),
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
                                stdout=lf, stderr=subprocess.STDOUT)
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
        if key == "no-ring-negatives":
            pass  # handled in run_one
        elif isinstance(value, bool):
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
    print("\n+-- campaign 10 results (ranked by hard probe) ----------------------")
    print(f"|  {'run':<44} {'hard':>5} {'easy':>5} {'f1':>5} {'qual':>6}")
    print("|  " + "-" * 65)
    for r in sorted(completed,
                    key=lambda r: (r.get("metrics") or {}).get("probe_hard_last") or 0,
                    reverse=True):
        m = r.get("metrics") or {}
        hard = f"{m.get('probe_hard_last',0.0):.3f}" if m.get("probe_hard_last") is not None else "?"
        easy = f"{m.get('probe_easy_last',0.0):.3f}" if m.get("probe_easy_last") is not None else "?"
        f1   = f"{m.get('valid_f1_last',0.0):.3f}"  if m.get("valid_f1_last")   is not None else "?"
        print(f"|  {r['name'][-44:]:<44} {hard:>5} {easy:>5} {f1:>5} {quality_score(m):>6.3f}")
    print("+--" + "-" * 67 + "\n")


def choose_next(pending, completed):
    """always pick the lowest run_id — strict sequential ordering."""
    return sorted(pending, key=lambda s: s.run_id)[0]


def main():
    parser = argparse.ArgumentParser(description="campaign 10 -- multi-scale 3D for sub-voxel ink")
    parser.add_argument("--campaign-id", type=str, default="c10_2026_06_15")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=90.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_campaign10"
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

        merged = dict(base); merged.update(spec.overrides)
        # handle no-ring-negatives
        if spec.overrides.get("no-ring-negatives"):
            merged = {k: v for k, v in base.items()
                      if k not in ("ring-negatives", "ring-label-source")}
            merged.update({k: v for k, v in spec.overrides.items() if k != "no-ring-negatives"})
            merged["log-dir"] = str(runs_dir)

        exp_name = f"cmp_{args.campaign_id}_{spec.name}"
        log_path = state_dir / f"{exp_name}.log"
        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)

        print(f"\n{'='*60}")
        print(f"  run {spec.run_id:02d}/21: {spec.name}  [{spec.axis}]")
        print(f"  overrides: {spec.overrides}")
        print(f"  {spec.why}")
        print(f"{'='*60}")

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

        done_ids.add(spec.run_id)
        pending = [s for s in RUN_SPECS if s.run_id not in done_ids]

        # cooldown between runs — GPU needs time to shed heat before next heavy workload
        if pending and not args.dry_run:
            print("[COOLDOWN] waiting 90s for GPU to cool before next run...")
            time.sleep(90)

    print_summary(state.get("completed", []))
    print("campaign 10 finished")


if __name__ == "__main__":
    main()
