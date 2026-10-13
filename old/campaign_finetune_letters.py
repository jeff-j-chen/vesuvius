"""campaign_finetune_letters.py -- single-scroll decoder fine-tunes from campaign 41, for letter hunting.

The supervised source is campaign 41 `and_mask`, not its `baseline` arm. Historical fine-tunes start
from models/archs41/and_mask/final.pth; --init-weights changes that source. The local fine-tune disables
AND-mask and uses the loss/augmentation settings below. campaign_letters_overnight.py rebuilds the
supervised AND-mask model from the updated MAE in a separate directory before its fine-tunes.

The default queue is refine_baseline_m1, refine_depth_m2, refine_encoder_3e6: frozen/no-PU, close 2 /
gap 2 / shell 4 with positive-only windows, 5:1 domain sampling, and six epochs. The baseline uses
fixed slices 9-16; its matched controls use slices 8-15 or encoder LR 3e-6 (decoder LR remains 1e-4).
Only 20261012073246 and 20261012065806 render. Historical arms require --only or --from.

Historical arms freeze the encoders (enc1-3 + bottleneck, the early-2D branch encoder, the gated cue stem)
and train the decoders + heads for 6 epochs. The diag_* arms below test fitting constraints from the same
checkpoint, without MAE retraining. Figures are drawn after the final epoch of each arm.

| tid        | trains on                                                         | vis-only                          |
|------------|-------------------------------------------------------------------|-----------------------------------|
| pherc0009b | pherc0009b (20250919125754), train_masks split                    | abf, 3_new x3                     |
| abf_reversed | auto_grown_20260927134928530_abf layer-reversed (20260927134929), | pherc0009b, 3_new x3            |
|            | surface detected on the reversed volume; train-only, labels/masks |                                   |
|            | symlinked to 20260927134928's; rendered the same way              |                                   |
| abf_rev_plus2 | abf_reversed's scroll + 3_new 065803 as 20261012065804 (regular  | pherc0009b, 3_new 081237          |
|            | volume, surface from the reversed volume mirrored back) + 073244  |                                   |
|            | as rendered; all train-only, labels link to the source ids        |                                   |
| rev_plus2_0009b | abf_rev_plus2 + pherc0009b (train_masks split); domain       | 3_new 081237                      |
|            | weights pherc0846a 5 : pherc0009b 1                               |                                   |
| flat10_18_0009b | rev_plus2_0009b's PHerc0846A volumes with flat depth-13      | 3_new 081237 (flat)               |
|            | surfaces (fixed slices 10-17 in training and rendering); pherc0009b |                                 |
|            | keeps its detected surface; weights 5 : 1                         |                                   |
| flat_10_18_0009b_noaug_forward | flat slices 10-17 on abf / 065803 / 073244 as     | 3_new 081237 (flat, forward)      |
|            | rendered (forward) + pherc0009b untouched; weights 5 : 1; every   |                                   |
|            | augmentation, dropout, RSC, native noise and patch GroupDRO off   |                                   |
| flat_10_18_0009b_noaug_backward | same on the layer-reversed volumes of all four  | 3_new 081237 (flat, reversed)     |
| surface_0009b_noaug_forward | as noaug_forward, but rendered volumes with their own  | 3_new 081237 (rendered surface)   |
|            | detected surfaces (abf 20260927134928, 065803, 073244)            |                                   |
| surface_0009b_noaug_forward_offsetextra | surface_0009b_noaug_forward + 073244's next   | 3_new 081237 (rendered surface)   |
|            | sheet (mesh +29 along the normal, 20261012073249), train-only     |                                   |

3_new = the PHerc0846A patches in tifxyz/3_new (20261012065803, 20261012073244, 20261012081237), assembled by
assemble_test_segments.py. Label-free vis scrolls render with empty labels and are never trained on.
An arm whose training label file has no full-brightness (255) ink is skipped: the character split needs letters.

Usage:
    python3 campaign_finetune_letters.py --dry-run
    python3 campaign_finetune_letters.py --smoke
    python3 campaign_finetune_letters.py
    python3 campaign_finetune_letters.py --only diag_unfreeze,diag_no_pu,diag_unfreeze_no_pu
    python3 campaign_finetune_letters.py --only refine_depth_m2

See old/FINETUNE_LETTERS_DIAGNOSIS.md for measured evidence and comparison groups.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
import campaign_archs_40 as campaign40
import campaign_archs_41 as campaign41
from utils.config import ScrollConfig, startup_output
from utils.norm import ensure_surface_anchors


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_finetune_letters"
MODEL_DIR = "models/finetune_letters"
SMOKE_LOG_DIR = "./runs_finetune_letters_smoke"
SMOKE_MODEL_DIR = "models/finetune_letters_smoke"
INKLABEL_DIR = campaign41.INKLABEL_DIR
DEFAULT_INIT_WEIGHTS = "/vesuvius/models/archs41/and_mask/final.pth"
N_EPOCHS = 6
LR = 1e-4
FREEZE_PREFIXES = (
    "enc1.", "enc2.", "enc3.", "bottleneck.",
    "early2d_enc2.", "early2d_enc3.", "early2d_bottleneck.", "gated_cue_stem.",
)

PHERC0009B_ID = campaign41.PHERC0009B_ID
ABF_ID = 20260927134928  # auto_grown_20260927134928530_abf
# ABF_ID with layers reversed and surface labels detected on the reversed volume; label files link to ABF_ID's
ABF_REV_ID = 20260927134929
NEW_IDS = [20261012065803, 20261012073244, 20261012081237]  # tifxyz/3_new
# 20261012073244's mesh moved +29 voxels along its render normal (tifxyz/3_new_offset/..._abf_p29): the next sheet
OFFSET_ID = 20261012073249
ASSEMBLED_IDS = [ABF_ID, *NEW_IDS, OFFSET_ID]
# regular volume (zarr symlinked), surface detected on the layer-reversed volume and mirrored back (27 - d);
# masks / labels / train masks link to the source id's files
REVSURF_IDS = {20261012065803: 20261012065804}
NEW_073244_ID = NEW_IDS[1]  # rendered volume + rendered surface
# same volume as the key id, but a flat surface at depth 13 (confidence 255): the surface-relative 8-slice window
# becomes the fixed slices 10-17 everywhere, in training and rendering; masks / labels link to the key id's
FLAT_DEPTH = 13
FLAT_IDS = {
    ABF_REV_ID: 20260927134930,
    20261012065803: 20261012065806,
    NEW_073244_ID: 20261012073246,
    NEW_IDS[2]: 20261012081238,
}
# full layer-reversed copies (no surface detection; only used under flat surfaces); label files link to the source
REVERSED_IDS = {20261012065803: 20261012065807, NEW_073244_ID: 20261012073247, NEW_IDS[2]: 20261012081239}
FLAT_IDS.update({ABF_ID: 20260927134931, **{rev: rev + 1 for rev in REVERSED_IDS.values()}})
FLAT_IDS[OFFSET_ID] = 20261012073250
# flat slices 10-17 on all four PHerc0846A patches, as rendered / layer-reversed: (abf, 065803, 073244, 081237)
FLAT_FORWARD = [FLAT_IDS[ABF_ID], FLAT_IDS[20261012065803], FLAT_IDS[NEW_073244_ID], FLAT_IDS[NEW_IDS[2]]]
FLAT_BACKWARD = [FLAT_IDS[ABF_REV_ID], *(FLAT_IDS[REVERSED_IDS[sid]] for sid in NEW_IDS)]
FLAT_OFFSETEXTRA = [FLAT_IDS[sid] for sid in (ABF_ID, *NEW_IDS[:2], OFFSET_ID)]

PHERC0009B_SCROLL = next(s for s in campaign40.SCROLLS_BASE if int(s.scroll_id) == PHERC0009B_ID)
ABF_REV_SCROLL = ScrollConfig(ABF_REV_ID, split_axis="x", train_split_frac=1.0)
REVSURF_SCROLLS = [ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in REVSURF_IDS.values()]
NEW_073244_SCROLL = ScrollConfig(NEW_073244_ID, split_axis="x", train_split_frac=1.0)
PLUS2_IDS = [ABF_REV_ID, *REVSURF_IDS.values(), NEW_073244_ID]
PLUS2_SCROLLS = [ABF_REV_SCROLL, *REVSURF_SCROLLS, NEW_073244_SCROLL]
FLAT_TRAIN_IDS = [FLAT_IDS[ABF_REV_ID], FLAT_IDS[20261012065803], FLAT_IDS[NEW_073244_ID]]
FLAT_SCROLLS = [ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in FLAT_TRAIN_IDS]


def _flat_noaug_arm(flat_ids: list[int]) -> tuple:
    """first three flat ids train (train-only) alongside pherc0009b; all four render, the fourth label-free."""
    train = flat_ids[:3]
    return (
        {"pherc0846a": list(train), "pherc0009b": [PHERC0009B_ID]},
        [*(ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in train), PHERC0009B_SCROLL],
        list(train), [*train, PHERC0009B_ID, flat_ids[3]], [flat_ids[3]],
    )


# every augmentation and training extra off: transforms, cutout, context swaps, jitter, native noise, RSC,
# dropout, patch GroupDRO (nnPU, soft edges and bag ranking define the loss and stay)
NO_AUG = {
    "dl.data_aug": False,
    "dl.rotation_prob": 0.0,
    "dl.flip_prob": 0.0,
    "dl.cutout_prob": 0.0,
    "dl.context_replace_prob": 0.0,
    "data.ctx_jitter": 0,
    "data.depth_jitter": 0,
    "data.native_noise": "",
    "data.native_noise_ids": [],
    "tra.rsc_prob": 0.0,
    "tra.physical_patch_groupdro": False,
    "model.conv1_drop": 0.0,
    "model.conv2_drop": 0.0,
    "model.head_drop": 0.0,
    "model.skip_drop": 0.0,
}

# tid -> (train_scroll_dict, scrolls, train-only ids, vis ids (trained first), label-free vis ids)
ARMS = {
    # "pherc0009b": (
    #     {"pherc0009b": [PHERC0009B_ID]}, [PHERC0009B_SCROLL], [],
    #     [PHERC0009B_ID, ABF_ID, *NEW_IDS], [ABF_ID, *NEW_IDS],
    # ),
    # "abf_reversed": (
    #     {"pherc0846a": [ABF_REV_ID]}, [ABF_REV_SCROLL], [ABF_REV_ID],
    #     [ABF_REV_ID, PHERC0009B_ID, *NEW_IDS], list(NEW_IDS),
    # ),
    # "abf_rev_plus2": (
    #     {"pherc0846a": list(PLUS2_IDS)}, list(PLUS2_SCROLLS), list(PLUS2_IDS),
    #     [*PLUS2_IDS, PHERC0009B_ID, NEW_IDS[2]], [NEW_IDS[2]],
    # ),
    # "rev_plus2_0009b": (
    #     {"pherc0846a": list(PLUS2_IDS), "pherc0009b": [PHERC0009B_ID]},
    #     [*PLUS2_SCROLLS, PHERC0009B_SCROLL], list(PLUS2_IDS),
    #     [*PLUS2_IDS, PHERC0009B_ID, NEW_IDS[2]], [NEW_IDS[2]],
    # ),
    # # PHerc0846A on fixed slices 10-17 (flat surfaces); pherc0009b keeps its detected surface
    # "flat10_18_0009b": (
    #     {"pherc0846a": list(FLAT_TRAIN_IDS), "pherc0009b": [PHERC0009B_ID]},
    #     [*FLAT_SCROLLS, PHERC0009B_SCROLL], list(FLAT_TRAIN_IDS),
    #     [*FLAT_TRAIN_IDS, PHERC0009B_ID, FLAT_IDS[NEW_IDS[2]]], [FLAT_IDS[NEW_IDS[2]]],
    # ),
    # "flat_10_18_0009b_noaug_forward": _flat_noaug_arm(FLAT_FORWARD),
    # "flat_10_18_0009b_noaug_backward": _flat_noaug_arm(FLAT_BACKWARD),
    # rendered volumes with their own detected surfaces (abf, 065803, 073244; 081237 render-only)
    # "surface_0009b_noaug_forward": _flat_noaug_arm([ABF_ID, *NEW_IDS]),
    # "surface_0009b_noaug_forward_offsetextra": (
    #     {"pherc0846a": [ABF_ID, *NEW_IDS[:2], OFFSET_ID], "pherc0009b": [PHERC0009B_ID]},
    #     [*(ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in (ABF_ID, *NEW_IDS[:2], OFFSET_ID)),
    #      PHERC0009B_SCROLL],
    #     [ABF_ID, *NEW_IDS[:2], OFFSET_ID],
    #     [ABF_ID, *NEW_IDS[:2], OFFSET_ID, PHERC0009B_ID, NEW_IDS[2]], [NEW_IDS[2]],
    # ),
    # "flat_10_18_0009b_noaug_forward_offsetextra": (
    #     {"pherc0846a": list(FLAT_OFFSETEXTRA), "pherc0009b": [PHERC0009B_ID]},
    #     [*(ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in FLAT_OFFSETEXTRA), PHERC0009B_SCROLL],
    #     list(FLAT_OFFSETEXTRA),
    #     [*FLAT_OFFSETEXTRA, PHERC0009B_ID, FLAT_IDS[NEW_IDS[2]]], [FLAT_IDS[NEW_IDS[2]]],
    # ),
    # same as above with ring close 0 / gap 1 / shell 4 and negatives also supervised in ink windows
    "flat_10_18_0009b_noaug_forward_offsetextra_ring": (
        {"pherc0846a": list(FLAT_OFFSETEXTRA), "pherc0009b": [PHERC0009B_ID]},
        [*(ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in FLAT_OFFSETEXTRA), PHERC0009B_SCROLL],
        list(FLAT_OFFSETEXTRA),
        [*FLAT_OFFSETEXTRA, PHERC0009B_ID, FLAT_IDS[NEW_IDS[2]]], [FLAT_IDS[NEW_IDS[2]]],
    ),
}
# per-test domain weight multipliers on campaign 40's regime weights (domains are drawn round-robin)
ARM_WEIGHT_SCALE = {
    tid: {"pherc0846a": 5} for tid in ("rev_plus2_0009b", "flat10_18_0009b", "flat_10_18_0009b_noaug_forward",
                                       "flat_10_18_0009b_noaug_backward", "surface_0009b_noaug_forward",
                                       "surface_0009b_noaug_forward_offsetextra",
                                       "flat_10_18_0009b_noaug_forward_offsetextra",
                                       "flat_10_18_0009b_noaug_forward_offsetextra_ring")
}
# per-test config overrides ("section.field": value), applied last
ARM_OVERRIDES = {tid: NO_AUG for tid in ("flat_10_18_0009b_noaug_forward", "flat_10_18_0009b_noaug_backward",
                                         "surface_0009b_noaug_forward", "surface_0009b_noaug_forward_offsetextra",
                                         "flat_10_18_0009b_noaug_forward_offsetextra")}
ARM_OVERRIDES["flat_10_18_0009b_noaug_forward_offsetextra_ring"] = {
    **NO_AUG, "data.ring_close_r": 0, "data.ring_gap_r": 1, "data.ring_shell_r": 4, "data.multitile_pos_only": False,
}

DIAGNOSTIC_BASE = "flat_10_18_0009b_noaug_forward_offsetextra_ring"
UNFREEZE = {"tra.freeze_prefixes": ()}
UNFREEZE_NO_PU = {"tra.freeze_prefixes": (), "tra.pu_lambda": 0.0, "data.far_negative_share": 0.0}
DIAGNOSTIC_OVERRIDES = {
    "diag_unfreeze": UNFREEZE,
    "diag_no_pu": {"tra.pu_lambda": 0.0, "data.far_negative_share": 0.0},
    "diag_unfreeze_no_pu": UNFREEZE_NO_PU,
    "diag_long": {"tra.n_epochs": 18, "tra.patience": 18},
    "diag_unfreeze_long": {**UNFREEZE, "tra.n_epochs": 18, "tra.patience": 18},
    "diag_stem": {
        "tra.freeze_prefixes": tuple(prefix for prefix in FREEZE_PREFIXES
                                     if prefix not in ("enc1.", "gated_cue_stem.")),
    },
    "diag_no_bag": {**UNFREEZE, "tra.character_bag_ranking": False},
    "diag_weight20": UNFREEZE,
    "diag_0846_only": UNFREEZE,
    "diag_depth_m2": {
        **UNFREEZE,
        "data.surface_window_offset_by_scroll": {sid: -2 for sid in (*FLAT_OFFSETEXTRA, FLAT_FORWARD[3])},
    },
    "diag_depth_p2": {
        **UNFREEZE,
        "data.surface_window_offset_by_scroll": {sid: 2 for sid in (*FLAT_OFFSETEXTRA, FLAT_FORWARD[3])},
    },
    "diag_tiles8": {**UNFREEZE, "model.multitile_subtile": 8, "model.multitile_grid": 8},
    "diag_loose_ring": {
        **UNFREEZE, "data.ring_close_r": 2, "data.ring_gap_r": 2, "data.multitile_pos_only": True,
    },
}
for _tid, _overrides in DIAGNOSTIC_OVERRIDES.items():
    ARMS[_tid] = ARMS[DIAGNOSTIC_BASE]
    ARM_WEIGHT_SCALE[_tid] = dict(ARM_WEIGHT_SCALE[DIAGNOSTIC_BASE])
    ARM_OVERRIDES[_tid] = {**ARM_OVERRIDES[DIAGNOSTIC_BASE], **_overrides}
ARM_WEIGHT_SCALE["diag_weight20"] = {"pherc0846a": 20}
ARMS["diag_0846_only"] = (
    {"pherc0846a": list(FLAT_OFFSETEXTRA)},
    [ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in FLAT_OFFSETEXTRA],
    list(FLAT_OFFSETEXTRA), list(ARMS[DIAGNOSTIC_BASE][3]), list(ARMS[DIAGNOSTIC_BASE][4]),
)

FOCUS_VIS_IDS = [20261012073246, 20261012065806]
FOCUS_OVERRIDES = {
    "focus_control": {},
    "focus_loose_ring": {
        "data.ring_close_r": 2, "data.ring_gap_r": 2, "data.multitile_pos_only": True,
    },
    "focus_ratio1": {},
    "focus_depth_m1": {
        "data.surface_window_offset_by_scroll": {sid: -1 for sid in FLAT_OFFSETEXTRA},
    },
    "focus_depth_p1": {
        "data.surface_window_offset_by_scroll": {sid: 1 for sid in FLAT_OFFSETEXTRA},
    },
    "focus_encoder_1e6": {
        "tra.freeze_prefixes": (), "tra.encoder_prefixes": FREEZE_PREFIXES, "tra.encoder_lr": 1e-6,
    },
    "focus_encoder_1e5": {
        "tra.freeze_prefixes": (), "tra.encoder_prefixes": FREEZE_PREFIXES, "tra.encoder_lr": 1e-5,
    },
}
for _tid, _overrides in FOCUS_OVERRIDES.items():
    ARMS[_tid] = ARMS[DIAGNOSTIC_BASE]
    ARM_WEIGHT_SCALE[_tid] = dict(ARM_WEIGHT_SCALE[DIAGNOSTIC_BASE])
    ARM_OVERRIDES[_tid] = {**ARM_OVERRIDES["diag_no_pu"], **_overrides}
ARM_WEIGHT_SCALE["focus_ratio1"] = {"pherc0846a": 1}
CURRENT_BASELINE = {
    **ARM_OVERRIDES["diag_no_pu"],
    "data.ring_close_r": 2, "data.ring_gap_r": 2, "data.ring_shell_r": 4,
    "data.multitile_pos_only": True,
    "data.surface_window_offset_by_scroll": {sid: -1 for sid in FLAT_OFFSETEXTRA},
}
REFINE_OVERRIDES = {
    "refine_baseline_m1": {},
    "refine_depth_m2": {
        "data.surface_window_offset_by_scroll": {sid: -2 for sid in FLAT_OFFSETEXTRA},
    },
    "refine_encoder_3e6": {
        "tra.freeze_prefixes": (), "tra.encoder_prefixes": FREEZE_PREFIXES, "tra.encoder_lr": 3e-6,
    },
}
for _tid, _overrides in REFINE_OVERRIDES.items():
    ARMS[_tid] = ARMS[DIAGNOSTIC_BASE]
    ARM_WEIGHT_SCALE[_tid] = dict(ARM_WEIGHT_SCALE[DIAGNOSTIC_BASE])
    ARM_OVERRIDES[_tid] = {**CURRENT_BASELINE, **_overrides}
DEFAULT_ARMS = tuple(REFINE_OVERRIDES)


def _test(tid: str, init_weights: str) -> dict:
    train_scroll_dict, scrolls, train_only, vis, label_free = ARMS[tid]
    test = campaign40._test(
        tid,
        campaign40.SURFACE_FIBER_MAE_KEY,
        train_scroll_dict,
        scrolls,
        changes={**campaign40.NNPU, **campaign41.DEGRADED},
        init_weights=init_weights,
        batch_size=campaign40.SCRATCH_TRAINING["batch_size"],
        lr=LR,
    )
    test["tag"] = f"ftl_{tid}"
    test["train_only"] = list(train_only)
    test["vis"] = list(FOCUS_VIS_IDS if tid in FOCUS_OVERRIDES or tid in REFINE_OVERRIDES else vis)
    test["label_free"] = list(label_free)
    return test


@contextlib.contextmanager
def _campaign40_paths():
    saved = campaign40.LOG_DIR, campaign40.MODEL_DIR
    campaign40.LOG_DIR, campaign40.MODEL_DIR = LOG_DIR, MODEL_DIR
    try:
        yield
    finally:
        campaign40.LOG_DIR, campaign40.MODEL_DIR = saved


def build_config(test: dict):
    with _campaign40_paths():
        config = campaign40.build_config(test)
    config.data.scrolls = list(test["scrolls"])
    config.data.train_scroll_dict = {domain: list(ids) for domain, ids in test["train_scroll_dict"].items()}
    config.data.train_scroll_weights = [
        weight * ARM_WEIGHT_SCALE.get(test["tid"], {}).get(domain, 1)
        for domain, weight in zip(config.data.train_scroll_dict,
                                  campaign40._train_scroll_weights(config.data.train_scroll_dict))
    ]
    config.data.simple_split = False
    config.data.train_mask_dir = "./train_masks"
    config.data.inklabel_dir = INKLABEL_DIR
    config.data.train_only_scroll_ids = list(test["train_only"])
    config.data.vis_scroll_ids = list(test["vis"])
    config.data.vis_preload_persistent = True
    config.data.label_free_vis_scroll_ids = list(test["label_free"])
    config.tra.n_epochs = N_EPOCHS
    config.tra.lr = LR
    config.tra.warmup_epochs = 1
    config.tra.freeze_prefixes = FREEZE_PREFIXES
    config.tra.encoder_freeze_epochs = 0
    config.tra.encoder_lr_scale = 1.0
    config.tra.encoder_prefixes = ()
    config.tra.encoder_lr = None
    config.tra.task_lr = None
    config.tra.eval_int = N_EPOCHS
    config.tra.eval_int_scrolls = len(test["vis"])
    config.tra.test_int = 999
    config.tra.probe_int = 999
    config.tra.fast_eval_figure = False
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    trained = {int(sid) for ids in config.data.train_scroll_dict.values() for sid in ids}
    trained |= {int(scroll.scroll_id) for scroll in config.data.scrolls}
    leaked = trained & set(test["label_free"])
    if leaked:
        raise ValueError(f"vis-only scrolls {sorted(leaked)} must not be trained")
    for key, value in ARM_OVERRIDES.get(test["tid"], {}).items():
        section, name = key.split(".")
        if not hasattr(getattr(config, section), name):
            raise AttributeError(f"unknown config field {key}")
        setattr(getattr(config, section), name, value)
    config.tra.eval_int = config.tra.n_epochs
    return config


def _write_diagnostic_manifest(test: dict, config) -> None:
    """Preserve dynamic settings omitted by the shared dataclass config serializer."""
    path = ROOT / config.model_dir / "letters_experiment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tid": test["tid"], "init_weights": config.init_weights,
        "exp_name": config.exp_name, "log_dir": config.tra.log_dir,
        "data": vars(config.data), "dl": vars(config.dl),
        "model": vars(config.model), "tra": vars(config.tra),
    }
    path.write_text(json.dumps(payload, indent=2, default=lambda value: vars(value)) + "\n")


def _smoke_config(config) -> None:
    """one tiny epoch, then the end-of-training figure on the trained scroll only."""
    config.tra.n_epochs = 1
    config.tra.eval_int = 1
    config.dl.num_workers = 2
    config.data.max_samples_per_epoch = 2 * config.dl.num_workers * config.dl.batch_size
    config.data.vis_scroll_ids = config.data.vis_scroll_ids[:1]
    config.data.vis_preload_persistent = False
    config.tra.eval_int_scrolls = 1


def _missing(selected: list[dict]) -> list[str]:
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    problems = []
    for scroll_id in dict.fromkeys(sid for test in selected for sid in test["vis"]):
        labelled = not all(scroll_id in test["label_free"] for test in selected if scroll_id in test["vis"])
        required = [
            zarr_root / f"{scroll_id}.zarr",
            ROOT / "masks" / f"{scroll_id}.png",
            ROOT / "surface_labels" / str(scroll_id) / "depth.npy",
            ROOT / "surface_labels" / str(scroll_id) / "confidence.npy",
        ]
        if labelled:
            required += [ROOT / INKLABEL_DIR / f"{scroll_id}.png", ROOT / "train_masks" / f"{scroll_id}.png"]
        missing = [str(path) for path in required if not path.exists()]
        if not missing:
            missing = campaign41._frame_mismatches(zarr_root / f"{scroll_id}.zarr", scroll_id)
            if labelled:
                missing += _label_mismatches(zarr_root / f"{scroll_id}.zarr", scroll_id)
        if missing:
            problems.append(f"{scroll_id}: {', '.join(missing)}")
    return problems


def _label_mismatches(volume_path: Path, scroll_id: int) -> list[str]:
    import cv2
    import zarr

    height, width = (int(value) for value in zarr.open(str(volume_path), mode="r").shape[-2:])
    out = []
    for name in (f"{INKLABEL_DIR.strip('./')}/{scroll_id}.png", f"train_masks/{scroll_id}.png"):
        shape = cv2.imread(str(ROOT / name), cv2.IMREAD_GRAYSCALE).shape
        if tuple(shape) != (height, width):
            out.append(f"{name} is {shape[0]}x{shape[1]} but the zarr is {height}x{width}")
    return out


def ensure_reversed_abf(dry_run: bool) -> None:
    """ABF_REV_ID: ABF_ID's zarr with layers reversed, surface labels detected on it, label files symlinked."""
    ensure_reversed(ABF_ID, ABF_REV_ID, dry_run, detect_surface=True)


def ensure_reversed(src_id: int, dst_id: int, dry_run: bool, detect_surface: bool = False) -> None:
    """dst_id: src_id's zarr with layers reversed; mask / label / train-mask files symlinked to src_id's."""
    import numpy as np
    import zarr

    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    out = zarr_root / f"{dst_id}.zarr"
    surface = ROOT / "surface_labels" / str(dst_id)
    built = out.is_dir() and zarr.open(str(out), mode="r").attrs.get("complete")
    if dry_run:
        print(f"[letters] reversed {src_id} -> {dst_id}: zarr={'ok' if built else 'would build'}"
              + (f" surface={'ok' if (surface / 'depth.npy').is_file() else 'would generate'}" if detect_surface else ""),
              flush=True)
        return
    for folder in ("masks", INKLABEL_DIR.strip("./"), "train_masks"):
        link = ROOT / folder / f"{dst_id}.png"
        if (ROOT / folder / f"{src_id}.png").exists() and not link.is_symlink():
            link.unlink(missing_ok=True)
            link.symlink_to(f"{src_id}.png")
    if not built:
        src = zarr.open(str(zarr_root / f"{src_id}.zarr"), mode="r")
        dst = zarr.open(str(out), mode="w", shape=src.shape, chunks=src.chunks, dtype=src.dtype)
        for y0 in range(0, src.shape[1], 512):
            dst[:, y0:y0 + 512] = np.asarray(src[:, y0:y0 + 512])[::-1]
        dst.attrs.update({**dict(src.attrs), "layers_reversed_from": src_id, "complete": True})
        print(f"[letters] wrote layer-reversed {out}", flush=True)
    if detect_surface and not (surface / "depth.npy").is_file():
        subprocess.run([sys.executable, str(ROOT / "generate_surface_supervision.py"),
                        "--scroll-id", str(dst_id)], cwd=ROOT, check=True)


def ensure_revsurf(src_id: int, dst_id: int, dry_run: bool) -> None:
    """dst_id: src_id's volume (symlink) with the surface found on the layer-reversed volume, mirrored back."""
    import cv2
    import numpy as np
    import zarr

    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    surface = ROOT / "surface_labels" / str(dst_id)
    if dry_run:
        print(f"[letters] revsurf {src_id} -> {dst_id}: surface={'ok' if (surface / 'depth.npy').is_file() else 'would build'}",
              flush=True)
        return
    shape = cv2.imread(str(ROOT / "masks" / f"{src_id}.png"), cv2.IMREAD_GRAYSCALE).shape
    # blank label / all-train mask on the source id unless already drawn
    for folder, value in ((INKLABEL_DIR.strip("./"), 0), ("train_masks", 255)):
        path = ROOT / folder / f"{src_id}.png"
        if not path.exists():
            cv2.imwrite(str(path), np.full(shape, value, np.uint8))
    for folder in ("masks", INKLABEL_DIR.strip("./"), "train_masks"):
        link = ROOT / folder / f"{dst_id}.png"
        if not link.is_symlink():
            link.unlink(missing_ok=True)
            link.symlink_to(f"{src_id}.png")
    volume_link = zarr_root / f"{dst_id}.zarr"
    if not volume_link.is_symlink():
        volume_link.symlink_to(f"{src_id}.zarr")
    if (surface / "depth.npy").is_file():
        return
    tmp = ROOT / "_ves_tmp" / "facing"
    reversed_dir, revzarr = tmp / "ves_zarrs_reversed", tmp / "surface_revzarr" / str(src_id)
    out = reversed_dir / f"{src_id}.zarr"
    if not (out.is_dir() and zarr.open(str(out), mode="r").attrs.get("complete")):
        src = zarr.open(str(zarr_root / f"{src_id}.zarr"), mode="r")
        dst = zarr.open(str(out), mode="w", shape=src.shape, chunks=src.chunks, dtype=src.dtype)
        for y0 in range(0, src.shape[1], 512):
            dst[:, y0:y0 + 512] = np.asarray(src[:, y0:y0 + 512])[::-1]
        dst.attrs.update({**dict(src.attrs), "facing_reversed": True, "complete": True})
    if not (revzarr / "depth.npy").is_file():
        subprocess.run([sys.executable, str(ROOT / "generate_surface_supervision.py"), "--scroll-id", str(src_id),
                        "--zarr-dir", str(reversed_dir), "--mask-dir", str(ROOT / "masks"),
                        "--output-dir", str(revzarr.parent), "--review-dir", str(tmp / "surface_review")],
                       cwd=ROOT, check=True)
    layers = int(zarr.open(str(zarr_root / f"{src_id}.zarr"), mode="r").shape[0])
    depth = np.load(revzarr / "depth.npy")
    # 255 = no surface fitted
    mirrored = np.where(depth == 255, depth, (layers - 1) - depth.astype(np.int16)).astype(depth.dtype)
    surface.mkdir(parents=True, exist_ok=True)
    np.save(surface / "depth.npy", mirrored)
    np.save(surface / "confidence.npy", np.load(revzarr / "confidence.npy"))
    if (revzarr / "metadata.json").is_file():
        shutil.copy(revzarr / "metadata.json", surface / "metadata.json")
    print(f"[letters] wrote mirrored reversed-volume surface {surface}", flush=True)


def ensure_flat(src_id: int, dst_id: int, dry_run: bool) -> None:
    """dst_id: src_id's volume (symlink) with a flat FLAT_DEPTH surface, so every window is slices 10-17."""
    import json

    import cv2
    import numpy as np

    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    surface = ROOT / "surface_labels" / str(dst_id)
    if dry_run:
        print(f"[letters] flat {src_id} -> {dst_id}: surface={'ok' if (surface / 'depth.npy').is_file() else 'would build'}",
              flush=True)
        return
    for folder in ("masks", INKLABEL_DIR.strip("./"), "train_masks"):
        link = ROOT / folder / f"{dst_id}.png"
        if (ROOT / folder / f"{src_id}.png").exists() and not link.is_symlink():
            link.unlink(missing_ok=True)
            link.symlink_to(f"{src_id}.png")
    volume_link = zarr_root / f"{dst_id}.zarr"
    if not volume_link.is_symlink():
        volume_link.symlink_to(f"{src_id}.zarr")
    if (surface / "depth.npy").is_file():
        return
    mask = cv2.imread(str(ROOT / "masks" / f"{src_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
    surface.mkdir(parents=True, exist_ok=True)
    np.save(surface / "depth.npy", np.where(mask, FLAT_DEPTH, 255).astype(np.uint8))
    np.save(surface / "confidence.npy", np.where(mask, 255, 0).astype(np.uint8))
    (surface / "metadata.json").write_text(json.dumps(
        {"scroll_id": dst_id, "flat_from": src_id, "flat_depth": FLAT_DEPTH, "invalid_depth_value": 255}, indent=2))
    print(f"[letters] wrote flat depth-{FLAT_DEPTH} surface {surface}", flush=True)


def preflight_inputs(selected: list[dict], dry_run: bool) -> None:
    selected = [
        {**test, "vis": list(dict.fromkeys([*(int(scroll.scroll_id) for scroll in test["scrolls"]), *test["vis"]]))}
        for test in selected
    ]
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    unassembled = [sid for sid in ASSEMBLED_IDS if not (zarr_root / f"{sid}.zarr").is_dir()
                   or not (ROOT / "masks" / f"{sid}.png").is_file()]
    if unassembled:
        command = [sys.executable, str(ROOT / "assemble_test_segments.py"),
                   *(arg for sid in unassembled for arg in ("--only", str(sid)))]
        print(f"[letters] {'would assemble' if dry_run else 'assembling'}: {' '.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    for sid in ASSEMBLED_IDS:
        if (zarr_root / f"{sid}.zarr").is_dir() and not (ROOT / "surface_labels" / str(sid) / "depth.npy").is_file():
            command = [sys.executable, str(ROOT / "generate_surface_supervision.py"), "--scroll-id", str(sid)]
            print(f"[letters] {'would generate' if dry_run else 'generating'} surface labels: {' '.join(command)}",
                  flush=True)
            if not dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
    if any({ABF_REV_ID, FLAT_IDS[ABF_REV_ID]} & set(test["vis"]) for test in selected):
        ensure_reversed_abf(dry_run)
    for src_id, dst_id in REVSURF_IDS.items():
        if any(dst_id in test["vis"] for test in selected):
            ensure_revsurf(src_id, dst_id, dry_run)
    for src_id, dst_id in REVERSED_IDS.items():
        if any(FLAT_IDS[dst_id] in test["vis"] for test in selected):
            ensure_reversed(src_id, dst_id, dry_run)
    for src_id, dst_id in FLAT_IDS.items():
        if any(dst_id in test["vis"] for test in selected):
            ensure_flat(src_id, dst_id, dry_run)
    problems = _missing(selected)
    external = [path for test in selected for path in campaign40._external_files(test)
                if not (ROOT / path).is_file() and not Path(path).is_file()]
    if problems or external:
        text = "\n  ".join(problems + ([f"external files missing: {sorted(set(external))}"] if external else []))
        if not dry_run:
            raise FileNotFoundError(f"fine-tune inputs missing:\n  {text}")
        print(f"[letters] WARNING fine-tune inputs missing:\n  {text}", flush=True)


def arm_problems(test: dict) -> list[str]:
    """training scrolls whose label file holds no full-brightness ink (the character split needs letters)."""
    import cv2

    empty = []
    for scroll in test["scrolls"]:
        label = cv2.imread(str(ROOT / INKLABEL_DIR / f"{int(scroll.scroll_id)}.png"), cv2.IMREAD_GRAYSCALE)
        # extra_inklabels is off, so half-brightness (~128) strokes are erased by the loader
        if label is None or not (label >= 192).any():
            empty.append(f"{INKLABEL_DIR}/{int(scroll.scroll_id)}.png has no letters (draw them at 255)")
    return empty


def _ensure_global_norm(scroll_ids) -> None:
    """surface-anchor normalisation also needs each scroll's global stats in norm_cache.json."""
    import json

    cache_path = ROOT / "norm_cache.json"
    cached = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    missing = [str(sid) for sid in scroll_ids if str(sid) not in cached]
    if missing:
        command = [sys.executable, str(ROOT / "precompute_norm.py"), *(a for sid in missing for a in ("--scroll-id", sid))]
        print(f"[letters] computing global norm stats: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="letter fine-tunes and fitting diagnostics from campaign 41")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_tid", choices=tuple(ARMS), help="start at this arm in the queue")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="one tiny epoch + final vis on the trained scroll, separate log/model dirs")
    parser.add_argument("--init-weights", default=DEFAULT_INIT_WEIGHTS)
    args = parser.parse_args()
    if args.only and args.from_tid:
        parser.error("--only and --from are mutually exclusive")
    if args.smoke:
        global LOG_DIR, MODEL_DIR
        LOG_DIR, MODEL_DIR = SMOKE_LOG_DIR, SMOKE_MODEL_DIR

    candidates = ARMS if args.only or args.from_tid else DEFAULT_ARMS
    selected = [_test(tid, args.init_weights) for tid in candidates]
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [test for test in selected if test["tid"] in wanted]
        missing = wanted - {test["tid"] for test in selected}
        if missing:
            raise ValueError(f"unknown test ids: {sorted(missing)}")
    if args.from_tid:
        selected = selected[next(index for index, test in enumerate(selected) if test["tid"] == args.from_tid):]

    with startup_output():
        preflight_inputs(selected, args.dry_run)
        campaign29.preflight_train_masks(
            [scroll for test in selected for scroll in test["scrolls"]],
            inklabel_dir=ROOT / INKLABEL_DIR, strict=not args.dry_run,
        )
        for test in selected:
            if build_config(test).data.native_noise:
                campaign41.ensure_noise_bank(test, args.dry_run)
        if not args.dry_run:
            input_ids = tuple(dict.fromkeys(
                sid for test in selected for sid in [*(int(scroll.scroll_id) for scroll in test["scrolls"]), *test["vis"]]
            ))
            _ensure_global_norm(input_ids)
            ensure_surface_anchors(input_ids, str(ROOT / "ves_zarrs2"))
        print(f"[letters] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    for test in selected:
        for problem in arm_problems(test):
            print(f"[letters] WARNING {test['tid']} not ready yet: {problem}", flush=True)

    results = {}
    for test in selected:
        problems = arm_problems(test)
        if problems and not args.dry_run:
            print(f"[letters] SKIP {test['tid']}: {problems[0]}", flush=True)
            results[test["tid"]] = f"SKIPPED ({problems[0]})"
            continue
        config = build_config(test)
        if args.smoke:
            _smoke_config(config)
        with startup_output():
            print(
                f"[letters] {test['tid']}: domains={list(config.data.train_scroll_dict)} "
                f"train_only={config.data.train_only_scroll_ids} vis={config.data.vis_scroll_ids} "
                f"label_free={config.data.label_free_vis_scroll_ids} freeze={config.tra.freeze_prefixes} "
                f"batch={config.dl.batch_size} lr={config.tra.lr} epochs={config.tra.n_epochs} "
                f"weights={config.data.train_scroll_weights} pu={config.tra.pu_lambda} "
                f"encoder_lr={config.tra.encoder_lr} encoder_prefixes={config.tra.encoder_prefixes} "
                f"bag_rank={config.tra.character_bag_ranking} subtile={config.model.multitile_subtile} "
                f"depth_offsets={config.data.surface_window_offset_by_scroll} "
                f"and_mask={config.tra.and_mask} init={config.init_weights}",
                flush=True,
            )
        if not args.dry_run:
            _write_diagnostic_manifest(test, config)
        if args.dry_run:
            success = campaign31.run_test(config, True)
        elif args.smoke:
            success = campaign31.run_test_isolated(config)
        else:
            campaign29.prewarm_data_cache(config)
            success = campaign31.run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        del config
        gc.collect()

    print(f"\n{'=' * 78}\n[letters] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
