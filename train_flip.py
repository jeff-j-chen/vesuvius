"""train_flip: the default run's recipe (iter2_asf_cue) trained from the MAE on straightened2 ALONE for 6 epochs.

straightened2 is normally the held-out sheet; here it is the only trained patch, so the seven usual trained patches,
out_4 and out_5 become whole-patch-scored vis scrolls (PHerc0009B is left out entirely). at the end every one of them, plus straightened2
itself, is rendered (the same figures as the default run).

needs a hand-drawn label for straightened2 (flat alias 20261012065815, render 20261012065814):
    dilated_inklabels/20261012065815.png   8-bit grayscale PNG, 2561 rows x 4101 columns (height x width), the mask's size
    255 = ink, 127 = half-brightness hard negative (as in the other 0846A labels), 0 = background
the committed file is blank; a real run refuses to start until it holds ink.

usage: python3 train_flip.py [--dry-run]
logs/figures runs_iter2/iter2_flip, checkpoints models/runs_iter2/iter2_flip.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# train.py reads this before it imports torch
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import train_iter2 as base  # noqa: E402

STRAIGHTENED2 = base.HOLDOUT
RUN_NAME = "iter2_flip"
EPOCHS = 6
WARMUP_EPOCHS = 2
LABEL = REPO / "dilated_inklabels" / f"{STRAIGHTENED2}.png"


def configure(config) -> None:
    from utils.config import ScrollConfig

    config.run_name = config.exp_name = RUN_NAME
    config.model_dir = os.path.join("models", "runs_iter2", RUN_NAME)
    config.tra.log_dir = base.LOG_ROOT.name
    base._asf_cue(config)
    data = config.data
    data.scrolls = [ScrollConfig(STRAIGHTENED2, split_axis="x", train_split_frac=1.0)]
    data.train_scroll_dict = {"pherc0846a": [STRAIGHTENED2]}
    data.train_scroll_weights = [1]
    data.train_only_scroll_ids = [STRAIGHTENED2]
    data.extra_hard_negative_scroll_ids = [STRAIGHTENED2]
    # everything that used to train, and the other held-out sheets, is only rendered, over the whole patch
    data.whole_patch_valid_scroll_ids = [*base.NEW_SET, *base.OUT_VIS]
    data.label_free_vis_scroll_ids = []
    data.vis_scroll_ids = [*base.NEW_SET, base.HOLDOUT, *base.OUT_VIS]
    config.tra.eval_int_scrolls = len(data.vis_scroll_ids)
    config.tra.dann_n_domains = 1
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = EPOCHS
    config.tra.warmup_epochs = WARMUP_EPOCHS


def label_has_ink() -> bool:
    import cv2

    image = cv2.imread(str(LABEL), cv2.IMREAD_GRAYSCALE)
    return image is not None and bool((image >= 128).any())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="check inputs and print the config, no training")
    args = parser.parse_args()
    os.chdir(REPO)
    if not label_has_ink():
        message = f"{LABEL} has no ink yet; draw straightened2's label there before training"
        if not args.dry_run:
            raise SystemExit(message)
        print(f"[flip] WARNING {message}", flush=True)
    import train

    sys.argv = ["train.py", "--run-name", RUN_NAME, *(["--dry-run"] if args.dry_run else [])]
    train.main(configure=configure)


if __name__ == "__main__":
    main()
