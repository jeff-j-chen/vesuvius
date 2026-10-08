"""finetune.py -- fine-tune the train.py checkpoint (mirrors ft_all_pioychi_h in campaign_finetune_96.py).

Trains on PHerc1447 w060 x2 + PHerc0009B + PHerc0139 w044/w035 + PHerc0500P2 + PHerc0211.

Starts from the model written by `python3 train.py` (models/train/holdout_n96_combined_surface_norm/final.pth);
if that file is missing it falls back to models/archs40/holdout_n96_combined_surface_norm/final.pth.

enc1/enc2 are frozen, enc3 + bottleneck train at 1e-6, decoder + head at 5e-5, for 8 epochs.
PHerc0211 gets domain weight 2, the rest 1. Train/val splits come from train_masks/<scroll_id>.png.

Usage:
    python3 finetune.py
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import DEFAULT_SCROLLS, Config, ScrollConfig


TRAINED_WEIGHTS = "models/train/holdout_n96_combined_surface_norm/final.pth"
FALLBACK_WEIGHTS = "models/archs40/holdout_n96_combined_surface_norm/final.pth"
LOG_DIR = "./runs_finetune"
MODEL_DIR = "models/finetune/ft_all_pioychi_h"
EXP_NAME = "ft_all_pioychi_h"

PHERC1447_IDS = (20260930144758, 20260930144760)
PHERC0211_ID = 20260928000003
TRAIN_SCROLL_DICT = {
    "pherc1447": list(PHERC1447_IDS),
    "pherc0009b": [20250919125754],
    "pherc0139": [20260115000000, 20260317000000],  # w044, w035
    "pherc0500p2": [20250628074500],
    "pherc0211": [PHERC0211_ID],
}
TRAIN_SCROLL_WEIGHTS = [1, 1, 1, 1, 2]
_DEFAULT_SCROLL_BY_ID = {int(scroll.scroll_id): scroll for scroll in DEFAULT_SCROLLS}
SCROLLS = [
    ScrollConfig(PHERC1447_IDS[0], split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC1447_IDS[1], split_axis="x", train_split_frac=0.75),
    *(_DEFAULT_SCROLL_BY_ID[scroll_id] for domain in ("pherc0009b", "pherc0139", "pherc0500p2")
      for scroll_id in TRAIN_SCROLL_DICT[domain]),
    ScrollConfig(PHERC0211_ID, split_axis="x", train_split_frac=1.0),
]
VIS_SCROLL_IDS = [PHERC0211_ID, PHERC1447_IDS[1]]

N_EPOCHS = 8
FREEZE_PREFIXES = ("enc1.", "enc2.")
ENCODER_LR = 1e-6
TASK_LR = 5e-5


def resolve_init_weights() -> str:
    for path in (TRAINED_WEIGHTS, FALLBACK_WEIGHTS):
        full = os.path.join(REPO_ROOT, path)
        if os.path.isfile(full):
            if path == FALLBACK_WEIGHTS:
                print(f"[finetune] {TRAINED_WEIGHTS} not found; falling back to {FALLBACK_WEIGHTS}", flush=True)
            return full
    raise FileNotFoundError(f"no starting checkpoint: neither {TRAINED_WEIGHTS} nor {FALLBACK_WEIGHTS} exists")


def build_config() -> Config:
    config = Config()
    config.exp_name = EXP_NAME
    config.init_weights = resolve_init_weights()
    config.tra.log_dir = os.path.normpath(os.path.join(REPO_ROOT, LOG_DIR))
    config.model_dir = os.path.normpath(os.path.join(REPO_ROOT, MODEL_DIR))
    config.save_final = os.path.join(config.model_dir, "final.pth")

    config.data.scrolls = list(SCROLLS)
    config.data.train_scroll_dict = {domain: list(ids) for domain, ids in TRAIN_SCROLL_DICT.items()}
    config.data.train_scroll_weights = list(TRAIN_SCROLL_WEIGHTS)
    config.data.vis_scroll_ids = list(VIS_SCROLL_IDS)

    config.tra.n_epochs = N_EPOCHS
    config.tra.eval_int = N_EPOCHS
    config.tra.save_int = 1
    config.tra.eval_int_scrolls = len(VIS_SCROLL_IDS)
    config.tra.dann_n_domains = len(TRAIN_SCROLL_DICT)
    config.tra.lr = TASK_LR
    config.tra.encoder_lr_scale = 1.0
    config.tra.encoder_freeze_epochs = 0
    # read by utils/training_utils.py via getattr; not TrainingConfig fields
    config.tra.freeze_prefixes = FREEZE_PREFIXES
    config.tra.encoder_lr = ENCODER_LR
    config.tra.task_lr = TASK_LR
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="fine-tune the train.py checkpoint (ft_all_pioychi_h recipe)")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and print the config, then exit")
    args = parser.parse_args()
    # data, label, and cache paths in the config are repo-relative
    os.chdir(REPO_ROOT)
    config = build_config()

    from train import Trainer, preflight_inputs

    preflight_inputs(config)
    if args.dry_run:
        print(
            f"[dry-run] scrolls={config.scroll_ids()} "
            f"domains={dict(zip(config.data.train_scroll_dict, config.data.train_scroll_weights))} "
            f"vis={config.data.vis_scroll_ids} epochs={config.tra.n_epochs} freeze={config.tra.freeze_prefixes} "
            f"encoder_lr={config.tra.encoder_lr} task_lr={config.tra.task_lr} "
            f"init={config.init_weights} -> {config.save_final}",
            flush=True,
        )
        return

    from utils.norm import ensure_surface_anchors

    ensure_surface_anchors(
        dict.fromkeys([*config.scroll_ids(), *config.data.vis_scroll_ids]),
        config.data.zarr_path,
        surface_dir=config.data.surface_label_dir,
    )
    print(f"[finetune] init={config.init_weights} -> {config.save_final}", flush=True)

    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
