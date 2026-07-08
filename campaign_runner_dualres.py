"""campaign_runner_dualres.py — scroll4 dual-resolution SANITY / FEASIBILITY diagnostic.

PURPOSE (not a performance campaign — a plumbing + feasibility check)
  We now have the SAME scroll4 w018 text region reconstructed at two scan densities:
    - 2.4um  (78keV, "straight" S3 surface)   id 20240304144031   -- has ink labels
    - 7.91um (53keV, "arched" volpkg flatboi)  id 20231117161658   -- labels drawn separately
  This runner trains the v1 baseline on ONE scan at a time (single-scroll) to confirm:
    1. the reconstructed zarr loads and the model can learn off it,
    2. the ink labels sit in the correct location (visible in the eval figure),
    3. ink detection is feasible at each scan density (the core 7.91-vs-2.4 question).

DELIBERATE SETTINGS (per the sanity-test spec)
  - v1 baseline only (the simpler of the two C15 winners; v12 can be added later).
  - eval-int == epochs -> the eval figure (whole-region inference, EXPENSIVE) renders ONCE at
    the final epoch, on the SCROLL BEING TRAINED (eval-on-self).
  - test-int > epochs -> the scroll2/scroll4 transfer "Test" figure NEVER fires.
  - probe ROIs OFF (their coords are pinned to the old scrolls; re-enabled later).
  - no hard mining, no ring — keep the pipeline minimal for a clean read.

THE ACTUAL EXPERIMENT is the 7.91um run: scroll3 (our goal) only exists at 7.91um/53keV, and
scroll4-7.91um is the same modality. the question is whether v1 can detect ink on scroll4's OWN
held-out 7.91um region at all. the 2.4um run is NOT the experiment (we already know 2.4um shows
ink — it's the label source); it is only an optional pipeline smoke-check, so run it with --only
2.4um if you want to re-validate plumbing, otherwise default to 7.91um.

NOTE: the 7.91um run is skipped automatically until its eroded inklabel exists
  (eroded_inklabels/20231117161658.png). the 2.4um run is ready now.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

EPOCHS = 20

# shared v1 baseline pipeline (single-scroll, eval-on-self, test disabled, probes off)
BASE: Dict[str, Any] = {
    "epochs": EPOCHS,
    "arch": "v1",
    "l1-lambda": 7e-6,
    "conv1-drop": 0.0, "conv2-drop": 0.05, "fc1-drop": 0.2, "fc2-drop": 0.1,
    "batch-size": 64,
    "num-workers": 2,
    "eval-int": 20,         # eval figure (on the training scroll) at epoch 20
    "test-int": 30,         # > epochs -> scroll2/scroll4 transfer figure never renders
    "no-probe-rois": True,  # probe ROIs pinned to old scrolls; off for this diagnostic
    "no-hard-mining": True,
    "train-d-start": 28,
    "train-d-end": 44,
    "channel-mixing-prob": 0.0,
    "mask-memmap": True,
    # region + split: train only the right 60% (x) and top 75% (y) of the frame; within that,
    # split HORIZONTALLY -> top 75% train / bottom 25% valid.
    "crop-x-frac": "0.4,1.0",
    "crop-y-frac": "0.0,0.75",
    "split-axis": "y",
    "train-split-frac": 0.75,
}


@dataclass(frozen=True)
class ScanSpec:
    key: str          # cli selector
    name: str         # experiment name suffix
    scroll_id: int    # zarr / label id
    note: str


SCANS: List[ScanSpec] = [
    ScanSpec("7.91um", "s4_79um", 20240304161941,
             "w023 flipped 7.91um/53keV full sheet; SAME modality as the scroll3 target"),
    ScanSpec("2.4um",  "s4_24um", 20240304144031,
             "straight 2.4um/78keV region; the label source, richest detail"),
]


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


def build_cmd(python_exe: str, spec: ScanSpec, runs_dir: Path, campaign_id: str):
    merged = dict(BASE)
    merged["scroll-id"] = spec.scroll_id
    exp_name = f"cmp_{campaign_id}_{spec.name}"
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
    ap = argparse.ArgumentParser(description="scroll4 dual-resolution sanity diagnostic")
    ap.add_argument("--campaign-id", type=str, default="dualres_2026_07_06")
    ap.add_argument("--python-exe", type=str, default=sys.executable)
    ap.add_argument("--only", type=str, default=None,
                    help="run only this scan key (7.91um or 2.4um); default runs all with labels")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stall-minutes", type=float, default=90.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir = repo_root / "runs_dualres"
    runs_dir.mkdir(exist_ok=True)
    log_dir = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    selected = [s for s in SCANS if (args.only is None or s.key == args.only)]
    if not selected:
        print(f"[dualres] no scan matches --only {args.only}; options: {[s.key for s in SCANS]}")
        return

    for spec in selected:
        print("\n" + "=" * 78)
        print(f"[SCAN {spec.key}] id={spec.scroll_id}  ({spec.note})")
        if not labels_exist(spec.scroll_id):
            print(f"[SCAN {spec.key}] SKIP — eroded_inklabels/{spec.scroll_id}.png not found "
                  f"(draw/warp labels first)")
            continue
        cmd, exp_name = build_cmd(args.python_exe, spec, runs_dir, args.campaign_id)
        print(f"   cmd: {' '.join(cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path,
                                          stall_minutes=args.stall_minutes)
        print(f"[SCAN {spec.key}] done rc={rc} crashed={crashed}")

    print("\n[dualres] all selected scans processed.")


if __name__ == "__main__":
    main()
