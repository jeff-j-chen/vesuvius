"""campaign_runner_scroll4.py — scroll4 7.91um v1 BASELINE (the real feasibility run).

the question: can the v1 baseline (the C15 winner that nailed the scroll1 artifacts) detect
ink on scroll4's OWN held-out 7.91um/53keV region -- the SAME modality as scroll3, our goal
scroll? labels come from the 2.4um scan warped into the flipped-7.91 frame.

single scan only: 7.91um w023 (id 20240304161941). the 2.4um scan is deliberately NOT touched.

DELIBERATE SETTINGS (clean full baseline, no crutches)
  - v1 baseline arch.
  - region crop for speed: train only the RIGHT 60% (x) and TOP 75% (y) of the frame.
  - within that crop, HORIZONTAL split -> top 75% train / bottom 25% valid (split-axis y).
  - eval-int 20  -> eval-on-self figure (cropped region, horizontal split) renders at epoch 20.
  - test-int 30  -> > epochs, so the scroll2/scroll3 transfer figure never fires this run.
  - probe ROIs OFF (coords pinned to old scrolls).
  - pr-auc surrogate (pairwise ranking loss) OFF -> ranking-lambda 0.0.
  - ring dataset ON (ring-label-source eroded) — proven more robust + faster than full-mask
    across campaigns 10-15. no hard mining.

logs -> runs_scroll4/ (a fresh campaign set).
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

EPOCHS = 10

# v1 baseline pipeline (single-scroll, eval-on-self, test disabled, probes + surrogate off)
BASE: Dict[str, Any] = {
    "epochs": EPOCHS,
    "arch": "v1",
    "l1-lambda": 7e-6,
    "conv1-drop": 0.0, "conv2-drop": 0.05, "fc1-drop": 0.2, "fc2-drop": 0.1,
    "batch-size": 512,
    "num-workers": 8,
    "eval-int": 10,          # eval-on-self figure (cropped, horizontal split) at epoch 20
    "test-int": 30,          # > epochs -> scroll2/scroll3 transfer figure never renders
    "no-probe-rois": True,   # probe ROIs pinned to old scrolls
    "ranking-lambda": 0.0,   # pr-auc surrogate (pairwise ranking loss) OFF
    "no-hard-mining": True,
    # ring dataset: proven across campaigns 10-15 to be more robust + faster to train than
    # the full-mask dataset. negatives restricted to a dilated ring around the ink (~1:1),
    # ring boundary from 'eroded' labels (the setting every recent winning campaign used).
    "ring-negatives": True,
    "ring-label-source": "eroded",
    # ink layer is UNKNOWN for this scroll (train_d 28-44 was honed on scroll1) -> sweep the
    # FULL depth for both training tiles AND the eval/inference visualization (0..64).
    "train-d-start": 0,
    "train-d-end": 64,
    "d-start": 0,
    "d-end": 64,
    "channel-mixing-prob": 0.0,
    "mask-memmap": True,
    # region + split: train only the right 60% (x) and top 75% (y); within that,
    # split HORIZONTALLY -> top 75% train / bottom 25% valid.
    "crop-x-frac": "0.4,1.0",
    "crop-y-frac": "0.0,0.75",
    "split-axis": "y",
    "train-split-frac": 0.75,
}

SCROLL4_79_ID = 20240304161941   # w023 flipped 7.91um/53keV full sheet


CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]


def dict_to_cli_args(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def build_cmd(python_exe: str, runs_dir: Path, campaign_id: str):
    merged = dict(BASE)
    merged["scroll-id"] = SCROLL4_79_ID
    exp_name = f"cmp_{campaign_id}_s4_79um"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def labels_exist(scroll_id: int) -> bool:
    return (Path("eroded_inklabels") / f"{scroll_id}.png").exists()


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=90):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
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
    ap = argparse.ArgumentParser(description="scroll4 7.91um v1 baseline feasibility run")
    ap.add_argument("--campaign-id", type=str, default="scroll4_2026_07_06")
    ap.add_argument("--python-exe", type=str, default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=90.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir = repo_root / "runs_scroll4"
    runs_dir.mkdir(exist_ok=True)
    log_dir = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    print("\n" + "=" * 78)
    print(f"[scroll4 79um] id={SCROLL4_79_ID}  (w023 flipped 7.91um/53keV; scroll3-modality)")
    if not labels_exist(SCROLL4_79_ID):
        print(f"[scroll4 79um] ABORT — eroded_inklabels/{SCROLL4_79_ID}.png not found")
        return
    cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id)
    print(f"   cmd: {' '.join(cmd)}")
    if args.dry_run:
        print("\n[scroll4] dry-run only.")
        return
    log_path = log_dir / f"{exp_name}.log"
    rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                      stall_minutes=args.stall_minutes)
    print(f"[scroll4 79um] done rc={rc} crashed={crashed}")


if __name__ == "__main__":
    main()
