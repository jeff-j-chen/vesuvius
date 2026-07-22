"""campaign_runner_p0139_triple.py -- 7-test tile/architecture sweep on the TRIPLE scroll.

trains v14_mil_deep (the arch that first found ink) and two NEW physics-influenced MIL
variations across tile sizes 16 and 24, plus an augmentation test. all runs:
  - TRIPLE scroll: w044 (20260115000000) + w059 (20250223000000) + w047 (20260206000001)
  - 20 epochs, eval-int 20 (ONE eval figure at the end; NO test figure -> test-int 9999)
  - w059 + w047 split VERTICALLY (left 75% train / right 25% valid); w044 horizontal 80.55
  - ring eroded negatives, l1 3e-7, tiny dropout (0.05, 0.075), batch 64, num-workers 0
  - long thermal cooldowns everywhere (hot day / flaky hardware)
  - YlGnBu scroll colormap (set in visualizer)

PHYSICS VARIATIONS (v14_mil_deep is per-slice stem -> 3D depth-mix -> per-voxel logits ->
LSE bag aggregation, tile BCE):
  v14b_mil_zgrad : + depth-gradient input channel [raw, dI/dz]. ink between layers is a
                   DISCONTINUITY in the depth profile; dI/dz peaks at the interface and is
                   invariant to the bulk papyrus baseline (113keV -> low absorption contrast).
  v14c_mil_lcn   : + local-contrast-normalization front-end [raw, lcn] + learnable depth
                   positional encoding. LCN removes the bulk-density baseline that dominates
                   absolute intensity at 113keV; depth-PE lets the model key on the absolute
                   depth band where ink sits (depth = the dominant variable).

TESTS (7):
  t01 base   tile16   v14_mil_deep
  t02 base   tile24   v14_mil_deep
  t03 zgrad  tile16   v14b_mil_zgrad   (16x16 variation 1)
  t04 lcn    tile16   v14c_mil_lcn     (16x16 variation 2)
  t05 zgrad  tile24   v14b_mil_zgrad   (24x24 variation 1)
  t06 lcn    tile24   v14c_mil_lcn     (24x24 variation 2)
  t07 base   tile16   v14_mil_deep  + ALL data augmentations on (default tuned probs)

logs -> runs_p0139_triple. checkpoints -> models/triple/<name>_final.pth.

  python campaign_runner_p0139_triple.py            # run all 7 in sequence
  python campaign_runner_p0139_triple.py --dry-run  # print commands only
  python campaign_runner_p0139_triple.py --only t03 # run one
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

W044 = 20260115000000
W059 = 20250223000000
W047 = 20260206000001
INTER_RUN_COOLDOWN_SECS = 420

CRASH_SIGNALS = [
    "Traceback (most recent call last)", "CUDA error:", "CUDA out of memory",
    "OSError: [Errno", "pickle data was truncated", "_pickle.UnpicklingError",
    "forrtl: error", "WinError 1455",
]

BASE: Dict[str, Any] = {
    "scroll-ids":         f"{W044},{W059},{W047}",
    "train-d-start":      0,
    "train-d-end":        28,
    "d-start":            0,
    "d-end":              28,
    "epochs":             20,
    "eval-int":           20,             # ONE eval figure at the end
    "probe-int":          10,           # probes OFF
    "test-int":           9999,           # test figure OFF (per user)
    "ring-negatives":     True,
    "ring-label-source":  "eroded",
    "crop-x-frac":        "0.0,1.0",
    "crop-y-frac":        "0.0,1.0",
    "split-axis":         "y",
    "train-split-frac":   0.8055,
    "split-override":     [f"{W059}:x:0.75", f"{W047}:x:0.75"],
    "batch-size":         256,
    "lr":                 2e-4,
    "l1-lambda":          3e-7,
    "conv1-drop":         0.05,
    "conv2-drop":         0.075,
    "data-aug":           0,
    "num-workers":        2,
    "mask-memmap":        True,
    "depth":              8,
    # long thermal cooldowns
    "epoch-cooldown":     90,
    "val-cooldown":       120,
    "eval-cooldown":      600,
    "fig-chunk-cooldown": 600,
}

RUN_SPECS: List[Dict[str, Any]] = [
    # --- tile / architecture sweep (depth 8, full range 0-28) ---
    {"name": "t01_base_t16",      "arch": "v14_mil_deep",   "tile-size": 16},
    {"name": "t02_base_t24",      "arch": "v14_mil_deep",   "tile-size": 24},
    {"name": "t03_zgrad_t16",     "arch": "v14b_mil_zgrad", "tile-size": 16},
    {"name": "t04_lcn_t16",       "arch": "v14c_mil_lcn",   "tile-size": 16},
    {"name": "t05_zgrad_t24",     "arch": "v14b_mil_zgrad", "tile-size": 24},
    {"name": "t06_lcn_t24",       "arch": "v14c_mil_lcn",   "tile-size": 24},
    {"name": "t07_base_t16_aug",  "arch": "v14_mil_deep",   "tile-size": 16, "data-aug": 1},
    # --- depth-4 variants (thinner depth window, full range 0-28) ---
    {"name": "t08_base_t16_d4",   "arch": "v14_mil_deep",   "tile-size": 16, "depth": 4},
    {"name": "t09_base_t24_d4",   "arch": "v14_mil_deep",   "tile-size": 24, "depth": 4},
    # --- training-range 8-16 only (depth 8): one window straddling the mid-stack ---
    {"name": "t10_base_t16_d8_r8to16", "arch": "v14_mil_deep", "tile-size": 16, "depth": 8,
     "train-d-start": 8, "train-d-end": 16, "d-start": 8, "d-end": 16},
    {"name": "t11_base_t24_d8_r8to16", "arch": "v14_mil_deep", "tile-size": 24, "depth": 8,
     "train-d-start": 8, "train-d-end": 16, "d-start": 8, "d-end": 16},
    # --- training-range 8-16 only, depth 4: sliding d4 windows within 8-16 ---
    {"name": "t12_base_t16_d4_r8to16", "arch": "v14_mil_deep", "tile-size": 16, "depth": 4,
     "train-d-start": 8, "train-d-end": 16, "d-start": 8, "d-end": 16},
    {"name": "t13_base_t24_d4_r8to16", "arch": "v14_mil_deep", "tile-size": 24, "depth": 4,
     "train-d-start": 8, "train-d-end": 16, "d-start": 8, "d-end": 16},
]


def dict_to_cli_args(d: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        elif isinstance(v, list):
            for item in v:
                args.extend([f"--{k}", str(item)])
        else:
            args.extend([f"--{k}", str(v)])
    return args


def build_cmd(python_exe, runs_dir, campaign_id, spec):
    merged = dict(BASE)
    for k, v in spec.items():
        if k != "name":
            merged[k] = v
    merged["save-final"] = f"models/triple/{spec['name']}_final.pth"
    exp_name = f"cmp_{campaign_id}_{spec['name']}"
    cmd = [python_exe, "train.py", "-n", exp_name, "--log-dir", str(runs_dir)]
    cmd += dict_to_cli_args(merged)
    return cmd, exp_name


def run_with_monitoring(cmd, repo_root, env, log_path, stall_minutes=180.0):
    print(f"[MONITOR] log -> {log_path}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=lf, stderr=lf)
    last_progress = time.time(); last_epoch = 0
    while proc.poll() is None:
        time.sleep(20)
        try:
            lines = open(log_path, encoding="utf-8", errors="replace").readlines()
        except Exception:
            continue
        tail = "".join(lines[-80:])
        for sig in CRASH_SIGNALS:
            if sig in tail:
                print(f"\n[MONITOR] CRASH -- '{sig}'\n" + "".join(lines[-12:]))
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
            print(f"\n[MONITOR] STALL -- no progress in {stall_minutes:.0f} min")
            try: proc.kill()
            except Exception: pass
            proc.wait()
            return 1, True
    proc.wait()
    rc = proc.returncode
    print(f"[MONITOR] {'OK' if rc == 0 else f'exited rc={rc}'}")
    return rc, False


def main():
    ap = argparse.ArgumentParser(description="7-test triple-scroll tile/arch sweep")
    ap.add_argument("--campaign-id",   type=str, default="p0139_triple_2026_07_16")
    ap.add_argument("--python-exe",    type=str, default=sys.executable)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--only",          type=str, default=None)
    ap.add_argument("--stall-minutes", type=float, default=180.0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_dir  = repo_root / "runs_p0139_triple"
    runs_dir.mkdir(exist_ok=True)
    log_dir   = runs_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    (repo_root / "models" / "triple").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    print("\n" + "=" * 78)
    print("[triple] PHerc0139 w044+w059+w047 -- 7-test tile/arch sweep (v14_mil_deep + 2 physics vars)")
    print("         20 epochs, eval-int 20 (eval figs only), YlGnBu, triple scroll")
    print("=" * 78)

    specs = RUN_SPECS
    if args.only:
        specs = [s for s in RUN_SPECS if args.only in s["name"]]
        if not specs:
            print(f"[ABORT] --only '{args.only}' matched no spec"); return

    # preflight: all three scrolls need eroded inklabels for ring negatives
    for sid, nm in ((W044, "w044"), (W059, "w059"), (W047, "w047")):
        p = repo_root / "eroded_inklabels" / f"{sid}.png"
        if not p.exists():
            print(f"   [WARN] eroded_inklabels/{sid}.png ({nm}) missing -- ring negatives will fail.")

    results: List[str] = []
    for i, spec in enumerate(specs, 1):
        cmd, exp_name = build_cmd(args.python_exe, runs_dir, args.campaign_id, spec)
        aug = spec.get("data-aug", 0)
        print(f"\n{'#'*78}\n[{i}/{len(specs)}] {spec['name']}  (arch={spec['arch']}, tile={spec['tile-size']}, aug={aug})\n{'#'*78}")
        print(f"   cmd: {' '.join(str(c) for c in cmd)}")
        if args.dry_run:
            continue
        log_path = log_dir / f"{exp_name}.log"
        rc, crashed = run_with_monitoring(cmd, repo_root, env, log_path, args.stall_minutes)
        results.append(f"   {spec['name']}: {'OK' if rc==0 and not crashed else f'FAIL(rc={rc},crashed={crashed})'}")
        print(f"[triple] done  {exp_name}  rc={rc}  crashed={crashed}")
        if i < len(specs) and INTER_RUN_COOLDOWN_SECS > 0:
            print(f"[COOLDOWN] inter-run pause {INTER_RUN_COOLDOWN_SECS}s...")
            time.sleep(INTER_RUN_COOLDOWN_SECS)

    if not args.dry_run:
        print("\n" + "=" * 78)
        print("[triple] campaign complete -- summary")
        print("=" * 78)
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
