"""campaign_archs_16.py -- axis split versus hand-annotated split on w013.

The baseline is constructed directly from campaign 15's c32_t8 control dictionary so every
training, model, data, and evaluation setting remains identical. The only experimental variable
is data.simple_split:

  baseline      True   legacy x-axis split at 75/25
  manual_split  False  train_masks/20240304141531.png partitions ring targets

Both arms use the campaign-15 geometry: 32px center, 8px sub-tiles, 4x4=16 targets/window.
Both also use the current corrected pos_only implementation shared with campaign 15. Archived
runs produced before that correction are not numerically comparable to a rerun of either file.

  python campaign_archs_16.py --dry-run
  python campaign_archs_16.py --only baseline
  python campaign_archs_16.py
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config
from campaign_archs_15 import TESTS as _TESTS15
from campaign_archs_15 import _OVERRIDES as _OVERRIDES15
from campaign_archs_15 import _base_config as _base_config15

LOG_DIR = "./runs_archs16"
_W013_ID = 20240304141531

# copy the complete resolved campaign-dictionary input for the c15 control. this is safer than
# reconstructing its inheritance chain and accidentally dropping an inherited setting.
_C15_CONTROL = next(t for t in _TESTS15 if str(t["tid"]) == "c16_t8")
_BASE16 = {k: v for k, v in _C15_CONTROL.items() if k not in {"tid", "tag"}}

_OVERRIDES = dict(_OVERRIDES15)


def _mk16(tid: str, tag: str, **overrides: object) -> dict:
    test = dict(_BASE16)
    test.update(overrides)
    test["tid"] = tid
    test["tag"] = tag
    return test


TESTS = [
    # exact c15 c16t8 control. simple_split=True is the new explicit name for the unchanged
    # axis/fraction behavior and is also the DataConfig default.
    _mk16(
        "baseline", "16_baseline",
        simple_split=True,
        multitile_pos_only=True,
    ),

    # only variable changed from the control: split eligible ink/ring targets by the hand mask.
    _mk16(
        "manual_split", "16_manual_split",
        simple_split=False,
        multitile_pos_only=True,
    ),
]


def build_config(test: dict) -> Config:
    tid = str(test["tid"])
    config = _base_config15(str(test["tag"]))
    config.tra.log_dir = LOG_DIR
    for key, (section, attr) in _OVERRIDES.items():
        if key not in test:
            continue
        try:
            setattr(getattr(config, section), attr, test[key])
        except AttributeError:
            print(f"[WARNING] {tid}: {section}.{attr} does not exist")

    init_weights = test.get("init_weights")
    if init_weights and os.path.exists(init_weights):
        config.init_weights = str(init_weights)
    elif init_weights:
        print(f"[archs16] init_weights '{init_weights}' not found -- {tid} trains from scratch")

    config.dl.data_aug = any([
        config.dl.flip_prob,
        config.dl.rotation_prob,
        config.dl.noise_prob,
        config.dl.brightness_prob,
        config.dl.contrast_prob,
        config.dl.cutout_prob,
        config.dl.depth_mask_prob,
        getattr(config.dl, "elastic_prob", 0.0),
    ])
    os.makedirs("models/archs16", exist_ok=True)
    setattr(config, "save_final", f"models/archs16/{tid}_final.pth")
    return config


def run_test(config: Config, dry_run: bool) -> bool:
    subtile = int(config.model.multitile_subtile)
    grid = int(config.model.multitile_grid)
    scroll_ids = [int(scroll.scroll_id) for scroll in config.data.scrolls]

    print(f"\n{'=' * 70}\n[archs16] {config.exp_name}\n{'=' * 70}", flush=True)
    print(
        f"  arch={config.model.arch}  scrolls={scroll_ids}  simple_split={config.data.simple_split}"
        f"  train_mask_dir={config.data.train_mask_dir}"
    )
    print(
        f"  center={subtile * grid}px  subtile={subtile}px  grid={grid}x{grid}={grid * grid} targets"
        f"  step={config.data.multitile_train_step}px  pos_only={config.data.multitile_pos_only}"
    )
    print(
        f"  epochs={config.tra.n_epochs}  batch={config.dl.batch_size}  lr={config.tra.lr:.2e}"
        f"  eval_int={config.tra.eval_int}  eval_bs={config.data.eval_infer_bs}"
    )
    print(
        f"  ring={config.data.ring_label_source}"
        f" close={config.data.ring_close_r} gap={config.data.ring_gap_r}"
        f" shell={config.data.ring_shell_r}  labels={config.data.inklabel_dir}"
    )
    if dry_run:
        print("  [DRY RUN] skipping")
        return True

    from train import Trainer

    try:
        trainer = Trainer(config)
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign_archs_16: manual split A/B on w013")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = TESTS
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [test for test in TESTS if str(test["tid"]) in wanted]
        missing = wanted - {str(test["tid"]) for test in selected}
        if missing:
            print(f"[ABORT] --only id(s) {sorted(missing)} not found; valid: "
                  f"{[str(test['tid']) for test in TESTS]}")
            return
    elif args.from_id:
        ids = [str(test["tid"]) for test in TESTS]
        if args.from_id not in ids:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {ids}")
            return
        selected = TESTS[ids.index(args.from_id):]

    print(f"[archs16] {len(selected)} test(s) queued  (log -> {LOG_DIR})")
    print("[archs16] exact c15 c32_t8 control versus manual train-mask split")

    results = {}
    for test in selected:
        tid = str(test["tid"])
        config = build_config(test)
        results[tid] = "OK" if run_test(config, args.dry_run) else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\n{'=' * 70}\n[archs16] SUMMARY\n{'=' * 70}")
    for tid, status in results.items():
        test = next(item for item in TESTS if str(item["tid"]) == tid)
        print(f"  {tid} ({test['tag']}) simple_split={test['simple_split']}: {status}")


if __name__ == "__main__":
    main()
