"""campaign_runner_scroll4_1d.py — 2.4um scroll4 w023: per-slice 2+1D architecture search.

PROBLEM CONTEXT
  Scroll1 ink is detectable at 7.9um/54keV because the ink contains heavy metals (likely
  lead) with ~30x higher X-ray absorption than carbon. A simple tile classifier with global
  average pooling learns this as a scalar intensity signal.

  Scrolls 2/3/4 use carbon-based ink, elementally identical to carbonized papyrus. At
  7.9um/54keV there is no absorption contrast — the signal is undetectable by any model
  including the researchers' heavy U-Net.

  At 2.4um/78keV (scroll4 w023 only), the ink becomes readable. Why? Because at this
  resolution individual papyrus fiber gaps (~4-20 voxels) are resolved. Ink FILLS these
  gaps, making the local material more uniform — lower local variance — vs blank papyrus
  which shows the alternating fiber-gap-fiber texture. The signal is SPATIAL TEXTURE, not
  mean absorption.

ARCHITECTURE HYPOTHESIS (from campaign experiments)
  v14_mil_deep showed the most promise in preliminary runs: its first stage uses
  Conv3d(kernel=(1,3,3)) — extracting per-slice 2D texture features before ANY depth mixing.
  This is consistent with the texture hypothesis: ink leaves a detectable pattern in
  individual 2D cross-sections, not a depth-axis absorption profile.

  This campaign tests three architectures that build on this insight:
    t01_v14    — v14_mil_deep baseline (per-slice → 3D CBAM → MIL)
    t02_v17    — v17_2p1d_maxattn: strict 2+1D, spatial max per slice, 1D depth attention
                 (pure separation of texture extraction from depth selection)
    t03_v18    — v18_2p1d_lv: same as v17 but with LOCAL VARIANCE as additional input channel
                 (explicitly feeds the physics-predicted discriminative statistic to the model:
                 ink regions have LOWER local variance than fiber+air-gap papyrus)

SETTINGS
  - scroll4 w023 2.4um teacher zarr (20251217075048): right 30% / top 40% of frame
  - tile_size=106 (one stroke = ~4 tiles at 2.4um; ring dataset feasible at this size)
  - depth=16 (sweep full 0-64 range with stride 8 to find ink-bearing slices)
  - ring negatives, eroded source (best-performing ring config from campaigns 1-15)
  - y-split 75/25 (avoids x-axis label concentration confound)
  - data_aug=False (signal is too weak; augmentation noise ≈ signal)
  - l1_lambda=0 (no regularization fighting the weak signal)
  - lr=2e-4, 15 epochs (exploration setting; single eval at end to save time)
  - eval OOM fix: infer_bs scales as 256*(32/T)^2*(8/D) — keeps eval batch under ~1GB

logs -> runs_campaign_scroll4_1d/logs/<exp_name>.log
TB   -> runs_campaign_scroll4_1d/
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCROLL4_24_ID = 20251217075048   # 2.4um teacher zarr (w023, right 30% / top 40%)
EPOCHS = 15

RUN_SPECS = [
    {
        "name": "t01_v14",
        "arch": "v14_mil_deep",
        "comment": "baseline: per-slice (1,3,3) → 3D CBAM → LSE MIL",
    },
    {
        "name": "t02_v17",
        "arch": "v17_2p1d_maxattn",
        "comment": "strict 2+1D: per-slice backbone → spatial max → 1D depth attention",
    },
    {
        "name": "t03_v18",
        "arch": "v18_2p1d_lv",
        "comment": "2+1D + local variance channel: explicit texture-energy input",
    },
]

BASE: Dict[str, Any] = {
    "scroll-id": SCROLL4_24_ID,
    "tile-size": 106,
    "depth": 16,
    # training and validation depth window: 24-40 confirmed as ink-bearing from
    # activation analysis. training/val tiles are drawn only from this range.
    "train-d-start": 24,
    "train-d-end": 40,
    # eval figure (add_evaluation_figures) sweeps the FULL 0->64 depth so the
    # scroll-level prediction map shows the complete depth response. the trainer
    # passes d_start/d_end to the visualizer for figure inference.
    "d-start": 0,
    "d-end": 64,
    "epochs": EPOCHS,
    "eval-int": EPOCHS,       # eval only at end — expensive on T=106/D=16
    "test-int": EPOCHS + 1,   # > epochs -> no test figure
    "probe-int": EPOCHS + 1,  # no probe
    "batch-size": 16,         # T=106/D=16 with CBAM: OOM above B=16 on 80GB H100
    "num-workers": 4,
    "lr": 2e-4,
    "l1-lambda": 0,
    "data-aug": 0,            # all augmentation off: noise std ≈ signal strength
    "ring-negatives": True,
    "ring-label-source": "eroded",  # tightest ring, proven best in campaigns 1-15
    "split-axis": "y",
    "train-split-frac": 0.75,
    "no-hard-mining": True,
    "mask-memmap": True,
    "channel-mixing-prob": 0.0,
}

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "forrtl: error",
]


def dict_to_cli_args(d: Dict[str, Any]) -> List[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v: out.append(f"--{k}")
        else:
            out.extend([f"--{k}", str(v)])
    return out


def build_cmd(python_exe: str, runs_dir: Path, campaign_id: str, spec: Dict[str, Any]):
    merged = dict(BASE)
    merged["arch"] = spec["arch"]
    exp_name = f"cmp_{campaign_id}_{spec['name']}_s4_24um"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def labels_exist() -> bool:
    return (Path("eroded_inklabels") / f"{SCROLL4_24_ID}.png").exists()


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=120):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=lf, stderr=lf)
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
                print(f"\n[MONITOR] CRASH -- '{sig}'\n" + "".join(lines[-15:]))
                try: proc.kill()
                except Exception: pass
                proc.wait()
                return proc.returncode or 1, True
        for line in lines[-80:]:
            if "--- Epoch" in line:
                try:
                    ep = int(line.strip().split("/")[0].split()[-1])
                    if ep > last_epoch:
                        last_epoch = ep; last_progress = time.time()
                        print(f"[MONITOR] {line.strip()}")
                except Exception:
                    pass
        if time.time() - last_progress > stall_minutes * 60:
            print(f"\n[MONITOR] STALL -- no progress in {stall_minutes} min")
            try: proc.kill()
            except Exception: pass
            proc.wait()
            return 1, True
    proc.wait()
    rc = proc.returncode
    print(f"[MONITOR] {'completed successfully' if rc == 0 else f'exited rc={rc}'}")
    return rc, False


def main():
    ap = argparse.ArgumentParser(description="scroll4 2.4um per-slice 2+1D campaign")
    ap.add_argument("--campaign-id", type=str, default="scroll4_24_2026_07_09")
    ap.add_argument("--python-exe", type=str, default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=120.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir = repo_root / "runs_campaign_scroll4_1d"
    runs_dir.mkdir(exist_ok=True)
    log_dir = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    print("\n" + "=" * 78)
    print(f"[s4-24um 2+1D] scroll_id={SCROLL4_24_ID}  (2.4um w023 aligned teacher zarr)")
    if not labels_exist():
        print(f"[s4-24um 2+1D] ABORT — eroded_inklabels/{SCROLL4_24_ID}.png not found")
        return

    for spec in RUN_SPECS:
        print(f"\n{'='*78}")
        print(f"[s4-24um 2+1D] run: {spec['name']}  arch={spec['arch']}")
        print(f"   hypothesis: {spec['comment']}")
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        print(f"   cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                          stall_minutes=args.stall_minutes)
        print(f"[s4-24um 2+1D] {spec['name']} done rc={rc} crashed={crashed}")
        if crashed:
            print("[s4-24um 2+1D] aborting remaining runs after crash")
            break

    if args.dry_run:
        print("\n[s4-24um 2+1D] dry-run only.")
    else:
        print("\n[s4-24um 2+1D] campaign complete.")
        print(f"   tensorboard: tensorboard --logdir={runs_dir}")


if __name__ == "__main__":
    main()
