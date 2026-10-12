"""Sequential campaign-41 AND-mask retraining and PHerc0846A transfer experiments."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
import campaign_archs_41 as campaign41
import campaign_finetune_letters as letters
import mae_pretrain_letters as mae

ROOT = Path(__file__).resolve().parent
JOB = "letters_mae_overnight"
MODEL_ROOT = ROOT / "models" / JOB
LOG_ROOT = ROOT / f"runs_{JOB}"
INPUT_ROOT = ROOT / "_ves_tmp" / JOB
MAE_WEIGHTS = ROOT / "models" / f"{mae.NAME}.pth"
SUPERVISED_WEIGHTS = MODEL_ROOT / "and_mask" / "final.pth"
TRAIN_IDS = tuple(letters.FLAT_OFFSETEXTRA)
EXTRA_SOURCE = "auto_grown_20261002131432623_b_abf"
EXTRA_ID = 20261002131432
VIS_IDS = [*TRAIN_IDS, letters.FLAT_FORWARD[3], EXTRA_ID]


def supervised_config():
    test = campaign41._test("and_mask", campaign41.AND_MASK)
    config = campaign41.build_config(test)
    config.init_weights = str(MAE_WEIGHTS)
    config.model_dir = str(SUPERVISED_WEIGHTS.parent)
    config.save_final = str(SUPERVISED_WEIGHTS)
    config.exp_name = "and_mask_mae_0846a"
    config.tra.log_dir = str(LOG_ROOT / "supervised")
    config.data.vis_scroll_ids = []
    config.data.label_free_vis_scroll_ids = []
    config.data.vis_preload_persistent = False
    config.tra.eval_int = config.tra.test_int = config.tra.probe_int = 999999
    config.tra.eval_int_scrolls = 0
    config.tra.test_on_final = False
    config.tra.save_vis = False
    config.tra.disable_figures = True
    return config