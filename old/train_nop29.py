"""the default train.py run (combo_patch_bag_strict) without the PHerc0846A +29 offset sheet.

auto_grown_20261012073244544_abf_p29 (flat alias 20261012073250) leaves the training corpus and becomes vis-only,
scored on the whole patch. the final eval figures cover all five 0846A patches: ABF 134931 and 073246 (trained),
073250, 065811 and 065813 (vis-only). TensorBoard and figures go to runs/nop29, checkpoints to models/runs/nop29.

usage: python3 train_nop29.py [--dry-run]
       python3 train_nop29.py --resume models/runs/nop29/final_best_character_ep4.pth --start-epoch 4
           continues to epoch 8 in runs/nop29_resume, saving models/runs/nop29_resume/model_epoch_<n>.pth each epoch
"""
import argparse
import sys

import train

P29_ID = 20261012073250
VIS_IDS = [20260927134931, 20261012073246, P29_ID, 20261012065811, 20261012065813]


def configure(config, resume: str = "", start_epoch: int = 0) -> None:
    config.run_name = config.exp_name = "nop29_resume" if resume else "nop29"
    config.model_dir = f"models/runs/{config.run_name}"
    if resume:
        config.init_weights = resume
        config.tra.start_epoch = start_epoch
        config.tra.save_int = 1
    config.data.scrolls = [scroll for scroll in config.data.scrolls if int(scroll.scroll_id) != P29_ID]
    config.data.train_scroll_dict = {
        name: [sid for sid in ids if int(sid) != P29_ID] for name, ids in config.data.train_scroll_dict.items()
    }
    config.data.train_only_scroll_ids = [sid for sid in config.data.train_only_scroll_ids if int(sid) != P29_ID]
    config.data.whole_patch_valid_scroll_ids = sorted({*config.data.whole_patch_valid_scroll_ids, P29_ID})
    config.data.vis_scroll_ids = list(VIS_IDS)
    config.tra.eval_int_scrolls = len(VIS_IDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="", help="weights saved by an interrupted nop29 run")
    parser.add_argument("--start-epoch", type=int, default=0, help="epochs already completed by those weights")
    args, rest = parser.parse_known_args()
    if bool(args.resume) != (args.start_epoch > 0):
        raise SystemExit("--resume and --start-epoch go together")
    sys.argv = [sys.argv[0], *rest]
    train.main(configure=lambda config: configure(config, args.resume, args.start_epoch))
