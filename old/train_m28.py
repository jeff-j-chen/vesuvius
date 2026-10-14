"""two default train.py runs (combo_patch_bag_strict) that each train one 0846A straight-mesh sheet in place of p29.

m28:      trains m28 (auto_grown_20261012065803387_straight_abf_m28, flat alias 20261012065813);
          the straight sheet (20261012065811) stays label-free vis-only.
straight: trains the straight sheet (auto_grown_20261012065803387_straight_abf, flat alias 20261012065811);
          m28 becomes label-free vis-only.
in both, p29 (auto_grown_20261012073244544_abf_p29, flat alias 20261012073250) leaves training and becomes
vis-only, scored on the whole patch, and the final eval figures cover all five 0846A patches. the trained sheet
takes p29's slot, so both arms run in one process and the second reuses every prepared dataset except that sheet.
labels: dilated_inklabels/<flat alias>.png. TensorBoard and figures go to runs/<arm>, checkpoints to models/runs/<arm>.

usage: python3 train_m28.py [--only m28|straight]
"""
import argparse
import gc
import sys

import torch

import train
from utils.config import ScrollConfig

STRAIGHT_ID = 20261012065811
M28_ID = 20261012065813
P29_ID = 20261012073250
ARMS = {"straight": (STRAIGHT_ID, M28_ID), "m28": (M28_ID, STRAIGHT_ID)}  # arm -> (trained, vis-only)


def configure(config, arm: str) -> None:
    trained, vis_only = ARMS[arm]
    config.run_name = config.exp_name = arm
    config.model_dir = f"models/runs/{arm}"
    index = next(i for i, scroll in enumerate(config.data.scrolls) if int(scroll.scroll_id) == P29_ID)
    config.data.scrolls = [
        ScrollConfig(trained, split_axis="x", train_split_frac=1.0) if i == index else scroll
        for i, scroll in enumerate(config.data.scrolls)
    ]
    config.data.train_scroll_dict = {
        name: [trained if int(sid) == P29_ID else sid for sid in ids]
        for name, ids in config.data.train_scroll_dict.items()
    }
    config.data.train_only_scroll_ids = [
        trained if int(sid) == P29_ID else sid for sid in config.data.train_only_scroll_ids
    ]
    config.data.label_free_vis_scroll_ids = [vis_only]
    config.data.whole_patch_valid_scroll_ids = [*config.data.whole_patch_valid_scroll_ids, P29_ID]
    config.data.vis_scroll_ids = [20260927134931, 20261012073246, trained, P29_ID, vis_only]
    config.tra.eval_int_scrolls = len(config.data.vis_scroll_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(ARMS), help="run one arm (default: m28, then straight)")
    args = parser.parse_args()
    for arm in [args.only] if args.only else list(ARMS):
        print(f"[train_m28] arm {arm}: trains {ARMS[arm][0]}, vis-only {ARMS[arm][1]} and {P29_ID}", flush=True)
        sys.argv = ["train.py"]
        train.main(configure=lambda config, arm=arm: configure(config, arm))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
