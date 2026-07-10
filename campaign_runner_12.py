"""campaign_runner_12.py — deep dive into ring + depth-pooling combinations.

Key learnings from C11:
  - ring training (t04) is the MOST RELIABLE mechanism against easy collapse:
    hard RC stays at 0.44-0.45 for all 30 epochs; train F1 capped at 0.685
  - asymmetric pool (t10/t11) is STRUCTURALLY INCAPABLE of full overfitting:
    train F1 tops at 0.49 via depth-max ceiling; hard RC 0.37-0.44
  - asym_pool + ring = NEVER TESTED — the highest-priority gap
  - t13 (stem_depth_attn + full + reg + HM) crashed at ep13; likely HM injection
    caused it; the arch itself matched t04 at ep9 before crash
  - U-Net/deep_CBAM skip connections ACTIVELY HURT hard probe (fully overfit,
    hard CR collapses to 0.05-0.11)
  - valid F1 (0.22-0.37 range) is noise — hard probe is the only signal
  - ink features are sub-voxel/1-2 slice scale (scroll-4 voxel insight);
    depth-max is correct instinct; narrower depth window may remove noise slices

C12 strategy:
  - 30 epochs, probe_int=5, test_int=100 (never runs)
  - ring-label-source=eroded throughout (matches C11 t04 which was best)
  - t01-t02: asym_pool × ring (the #1 untested cross; two reg/noreg ablations)
  - t03: new v12_asym_attn_pool: soft 1D depth attention instead of hard max
  - t04-t05: stem_depth_attn crash investigation (ring+no-HM vs full+no-HM)
  - t06: focal loss on top of asym_pool+ring (down-weight easy boundary tiles)
  - t07-t08: narrow depth window (d_end=40 vs 44): remove outer crust slices
  - t09-t10: alternative architectures with ring (factorized depth, no_cbam)
  - t11: alternating ring on asym_pool (full epochs every other for breadth)
  - t12: stem_SE + ring (t15's arch gained with ring; channel-only attn + ring)
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

# L1 lambdas scaled per param count relative to v1 baseline (1.3M, tuned=7e-6)
L1_V1           = 7e-6    # 1.3M  params
L1_ASYM_POOL    = 7.5e-6  # 1.2M  params → 7e-6 * (1.3/1.2) ≈ 7.58e-6
L1_ASYM_ATTN    = 7.5e-6  # ~1.3M params (same backbone as asym_pool + small attn head)
L1_DEPTH_ATTN   = 6.5e-6  # 1.06M params → 7e-6 * (1.3/1.06) ≈ 8.6e-6 (conservative)
L1_FACTORIZED   = 8e-6    # ~0.87M params → 7e-6 * (1.3/0.87) ≈ 10.5e-6 (conservative)
L1_NO_CBAM      = 9e-6    # ~1.0M params → 7e-6 * (1.3/1.0) = 9.1e-6
L1_STEM_SE      = 7e-6    # ~1.3M params (same as v1_full_reg backbone, SE vs CBAM)

# BASE settings: ring disabled by default (overridden per run), d-window same as C11
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
    "no-hard-mining": True,      # default off; overridden for reg runs
    "train-d-start": 28,
    "train-d-end": 44,           # C11 standard range
    "l1-lambda": 0.0,
    "conv2-drop": 0.0,
    "fc1-drop": 0.0,
    "fc2-drop": 0.0,
    "conv1-drop": 0.0,
    "channel-mixing-prob": 0.0,
}

# shared dropout for regularized runs (matches C11 t03/t04 convention)
REG_DROP = {"conv1-drop": 0.0, "conv2-drop": 0.05, "fc1-drop": 0.2, "fc2-drop": 0.1}

RUN_SPECS: List[RunSpec] = [

    # ── t01: asym_pool + ring + reg  [PRIORITY 1] ────────────────────────────

    RunSpec(1, "t01_asym_pool_ring_reg", "asym_ring",
        {"arch": "v10_asymmetric_pool",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_ASYM_POOL,
         **REG_DROP},
        why="the #1 untested cross from C11: asym_pool structural ceiling (depth-max prevents "
            "easy overfitting) combined with ring sampling (boundary tiles only). "
            "ring cuts epoch time 2.5x vs full dataset; L1 and dropout provide additional "
            "regularization. asym_pool capped train_f1 at 0.49; ring at 0.68 — combined "
            "should hard-cap even lower while preserving hard probe signal."),

    # ── t02: asym_pool + ring + noreg  [ablation of reg] ─────────────────────

    RunSpec(2, "t02_asym_pool_ring_noreg", "asym_ring",
        {"arch": "v10_asymmetric_pool",
         "ring-negatives": True,
         "ring-label-source": "eroded"},
        why="ablation: ring + asym_pool without any regularization. isolates whether the "
            "structural mechanisms alone (depth-max + ring sampling) are sufficient without "
            "L1/dropout. if hard RC matches t01, reg is superfluous on this architecture."),

    # ── t03: v12_asym_attn_pool + ring + reg  [new arch] ─────────────────────

    RunSpec(3, "t03_asym_attn_pool_ring_reg", "new_arch",
        {"arch": "v12_asym_attn_pool",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_ASYM_ATTN,
         **REG_DROP},
        why="new architecture: soft 1D depth attention (learned softmax weights over depth) "
            "instead of hard depth-max. motivation: ink absorption may be a smooth 2-3 slice "
            "bump not a sharp spike — soft attention learns the aggregation shape. "
            "same backbone as asym_pool; only the depth aggregation changes. "
            "with ring, expected to resist easy overfitting as much as t01."),

    # ── t04: stem_depth_attn + ring + reg + no HM  [t13 crash fix #1] ────────

    RunSpec(4, "t04_depth_attn_ring_reg", "t13_fix",
        {"arch": "v11_conv_stem_depth_attn",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_DEPTH_ATTN,
         "no-hard-mining": True},
        why="t13 (full + reg + HM) crashed at ep13. fix attempt #1: replace full dataset "
            "with ring (ring naturally focuses on hard boundary tiles, making HM redundant "
            "and removing the likely HM-injection crash trigger). at ep9 t13 was matching "
            "t04 at hard RC=0.44 — this architecture was the most promising in C11 before crash. "
            "ring also cuts epoch time ~2.5x."),

    # ── t05: stem_depth_attn + full + reg + no HM  [t13 crash fix #2] ────────

    RunSpec(5, "t05_depth_attn_full_reg_nohm", "t13_fix",
        {"arch": "v11_conv_stem_depth_attn",
         "l1-lambda": L1_DEPTH_ATTN,
         "no-hard-mining": True},
        why="t13 crash fix #2: same arch + full dataset + reg but with HM completely disabled. "
            "if t13 crashed due to stale hard mining files being injected mid-training, "
            "disabling HM should allow the run to complete. "
            "gives a direct answer: was the architecture itself fine and only HM caused crash?"),

    # ── t06: asym_pool + ring + reg + focal  [focal emphasis] ────────────────

    RunSpec(6, "t06_asym_pool_ring_focal", "focal",
        {"arch": "v10_asymmetric_pool",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_ASYM_POOL,
         "focal-gamma": 2.0,
         **REG_DROP},
        why="ring already restricts to boundary tiles (first-level hard-example focus); "
            "focal loss (gamma=2) further down-weights easy-to-classify tiles within the ring "
            "set, focusing gradient on the most ambiguous boundary examples. "
            "double-level hard-example selection: ring spatially, focal probabilistically."),

    # ── t07: v1 + ring + reg + narrow d_end=40  [narrow depth window] ────────

    RunSpec(7, "t07_v1_ring_reg_narrow", "depth_window",
        {"arch": "v10_v1_full_reg",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_V1,
         "train-d-end": 40,   # remove slices 40-44; only 28-40 used for training
         **REG_DROP},
        why="voxel-scale insight: ink features are smaller than expected at 7.91um. "
            "the outer depth slices (40-44) may be crust/carbonized material with no ink signal. "
            "narrowing training to d_start=28, d_end=40 removes 2 depth starting positions "
            "but focuses on the core ink band. v1+ring is the proven best combo from C11; "
            "this tests whether restricting depth further improves hard probe."),

    # ── t08: asym_pool + ring + reg + narrow d_end=40 ────────────────────────

    RunSpec(8, "t08_asym_pool_ring_narrow", "depth_window",
        {"arch": "v10_asymmetric_pool",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_ASYM_POOL,
         "train-d-end": 40,
         **REG_DROP},
        why="combine narrow depth window with t01 (asym_pool + ring + reg). "
            "two-axis restriction: ring restricts tiles spatially (boundary only), "
            "narrow window restricts depth temporally (core ink band only). "
            "if t01 already saturates the hard probe, narrowing depth should be neutral or better; "
            "if t01 still noisy, narrowing depth may be the missing ingredient."),

    # ── t09: factorized_depth + ring + reg  [explicit depth separation] ──────

    RunSpec(9, "t09_factorized_ring_reg", "arch_variant",
        {"arch": "v2_factorized_depth",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_FACTORIZED,
         **REG_DROP},
        why="v2_factorized_depth uses (3,1,1) depth-only convolutions followed by (1,3,3) "
            "spatial-only convolutions at every stage — depth and spatial axes are never mixed "
            "in the same kernel. this mirrors the depth-max insight structurally: treat depth "
            "as a distinct signal axis from spatial texture. "
            "with ring, the model can't memorize easy letter spatial texture."),

    # ── t10: no_cbam + ring + reg  [remove all spatial attention] ────────────

    RunSpec(10, "t10_no_cbam_ring_reg", "arch_variant",
        {"arch": "v2_no_cbam",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_NO_CBAM,
         **REG_DROP},
        why="CBAM's spatial attention may be the mechanism by which models route easy-letter "
            "spatial texture and overfit. v2_no_cbam removes all CBAM blocks (channel + spatial "
            "attention disabled). with ring removing boundary tiles, spatial attention may be "
            "net-negative since there are no coarse easy-letter centers to attend to. "
            "tests: is conv alone + ring sufficient to learn ink features?"),

    # ── t11: asym_pool + alternating_ring + reg  [alternating for breadth] ───

    RunSpec(11, "t11_asym_pool_altring_reg", "alternating",
        {"arch": "v10_asymmetric_pool",
         "alternating-ring": True,
         "l1-lambda": L1_ASYM_POOL,
         **REG_DROP},
        why="alternating ring: odd epochs train on ring set (boundary tiles), even epochs "
            "train on full set (all tiles). asym_pool's depth-max ceiling limits full-epoch "
            "overfitting; ring epochs reinforce boundary discrimination. "
            "compared to pure ring (t01): alternating exposes the model to the full spatial "
            "distribution every other epoch, potentially improving generalization to hard tiles "
            "that are not near ink boundaries but have faint ink signal."),

    # ── t12: stem_SE + ring + reg  [channel-only attention + ring] ───────────

    RunSpec(12, "t12_stem_se_ring_reg", "arch_variant",
        {"arch": "v11_conv_stem_se",
         "ring-negatives": True,
         "ring-label-source": "eroded",
         "l1-lambda": L1_STEM_SE,
         **REG_DROP},
        why="t15 (stem_SE + reg, no ring) slowed easy collapse to train_f1=0.55 at ep29 "
            "and maintained hard RC at ~0.42-0.43 through ep14. SE removes CBAM's spatial "
            "attention which accelerates easy memorization. adding ring on top of SE should "
            "combine structural resistance (no spatial attention) with sampling constraint "
            "(no interior letter tiles). fastest arch in C11 (~2min/epoch)."),
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
        if key in ("alternating-ring",):
            pass  # handled specially in main
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
    print("\n+-- campaign 12 results (ranked by hard probe) ----------------------")
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
    """strict sequential ordering by run_id"""
    return sorted(pending, key=lambda s: s.run_id)[0]


def main():
    parser = argparse.ArgumentParser(description="campaign 12 -- ring + depth-pooling deep dive")
    parser.add_argument("--campaign-id", type=str, default="c12_2026_06_19")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=120.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_campaign12"
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

        # hard mining: enabled for reg runs unless explicitly forced off
        if spec.overrides.get("no-hard-mining"):
            merged["no-hard-mining"] = True          # explicitly disabled for this run
        elif merged.get("l1-lambda", 0.0) == 0.0:
            merged["no-hard-mining"] = True          # noreg runs: always off
        else:
            merged.pop("no-hard-mining", None)       # reg runs: enable HM

        merged["log-dir"] = str(runs_dir)

        exp_name = f"cmp_{args.campaign_id}_{spec.name}"
        log_path = state_dir / f"{exp_name}.log"
        cmd = [args.python_exe, "train.py", "-n", exp_name] + dict_to_cli_args(merged)

        # inject boolean flags not handled by dict_to_cli_args
        if merged.get("alternating-ring"):
            cmd.append("--alternating-ring")

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
    print("\n[campaign 12] all runs complete.")


if __name__ == "__main__":
    main()
