# python3 assemble_training_segments.py --only w044,w035,p9b_487,500P2_front,p841,ph1447_w058,ph1447_w060 && python3 assemble_test_segments.py --only 20260928000003 --only 20260928000004

"""campaign_finetune_96.py -- fine-tune the archs40 native-96 combined-surface model.

This starts from the campaign-40 `holdout_n96_combined_surface_norm_nnpu` checkpoint and fine-tunes on
the low-res / high-energy target domain only, keeping its nnPU objective (`nnpu=True`):

- PHerc1447: 20260930144758 / 20260930144760 (manual train/val split from train_masks)
- PHerc0211: 20260928000003 (manual train/val split from train_masks, like every other scroll)
- PHerc0211_z5520_w040_abf: 20260928000004 -- visualization only (RAM-preloaded, no labels, never trained)
- auto_grown_20260717193517: 20260717193517 -- PHerc0211 merged test segment, visualization only

Runs (tids), all ring (2,2,4) with enc1/enc2 frozen, enc3+bottleneck lr=1e-6, decoder+head lr=5e-5:
- ft_six             PHerc1447 x2 + PHerc0139 w044/w035 + PHerc0009B + PHerc0500P2 + PHerc0841, all
                     weight 1; both PHerc0211 fragments vis-only
- ft_six_0211_pioy   ft_six plus PHerc0211 (20260928000003) in training at weight 3; both PHerc0211
                     fragments still visualized
- ft_1447_0211       PHerc1447 + PHerc0211 (PHerc0211 weight 2, PHerc1447 weight 1); figures only for
                     the two PHerc0211 fragments (20260928000003, 20260928000004)
- ft_six_vis0211     PHerc1447 x2 + PHerc0139 w044/w035 + PHerc0009B + PHerc0500P2 + PHerc0841 (no
                     PHerc0343P), all weight 1; both PHerc0211 fragments vis-only; 20 epochs, eval at 10 and 20
- ft_six_vis0211_nofreeze  same, nothing frozen (whole encoder at 1e-6)
- ft_6_base          ft_six baseline (TRAIN_SIX_0841); PHerc0211 vis-only
- ft_7_0172          ft_six + PHerc0172 w068/w087 (native resampled zarrs, not degraded); PHerc0211 vis-only
- ft_7_paris4        ft_six + PHercParis4 from its degrader-translated sibling; PHerc0211 vis-only
- ft_6_16depth       ft_six with the frozen crossres D generator in front of the model: 8 real + 32
                     predicted slices, mixed by a learned per-voxel layer to 16 network slices; the
                     3D stem (enc1) is unfrozen so it can adapt to the doubled depth sampling
  every run also visualizes PHerc0841
  translated sibling: python3 assemble_training_segments.py --only paris4 --degrader models/c39_degrader.pth
Older (commented out):
- ft_all_pioychi_h   PHerc1447 + PHerc0009B + PHerc0139 w044/w035 + PHerc0500P2 + PHerc0211
                     (PHerc0211 weight 2, the rest weight 1); visualized on PHerc0211 and PHerc1447 w060
- ft_all             PHerc1447 + PHerc0009B + PHerc0139 w044/w035 + PHerc0500P2 (the six-scroll set
                     minus PHerc0343P / PHerc0841), all weight 1; PHerc0211 is held out and is the
                     only visualized scroll
- ft_all_pcgrad      same, with full PCGrad (one task per physical domain). PCGrad replaces patch
                     GroupDRO, which the trainer treats as mutually exclusive.
- ft_1447_<scroll>   PHerc1447 + one added scroll (both weight 1), visualized on the added scroll
                     and PHerc0211 (vis-only, preloaded to RAM). <scroll> is one of
                     pherc0841, pherc0009b, pherc0139_w044, pherc0139_w035, pherc0343p, pherc0500p2.

Each run's log folder `runs_finetune/<tid>_<timestamp>` has the same name as its checkpoint folder
`models/finetune/<tid>_<timestamp>`, and config.json is saved in both.

Usage:
    python3 campaign_finetune_96.py --dry-run
    python3 campaign_finetune_96.py --only ft_all_pcgrad
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
import campaign_archs_40 as campaign40
from utils.config import DEFAULT_SCROLLS, ScrollConfig, startup_output
from utils.norm import ensure_surface_anchors


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_finetune2"
MODEL_DIR = "models/finetune"
INKLABEL_DIR = "./dilated_inklabels"
DEFAULT_INIT_WEIGHTS = "/vesuvius/models/archs40/holdout_n96_combined_surface_norm_nnpu/final.pth"
PHERC0211_ID = 20260928000003
# PHerc0211_z5520_w040_abf: visualized every run, never trained
PHERC0211_VIS_ONLY_ID = 20260928000004
# auto_grown_20260717193517: PHerc0211 merged test segment; vis-only unless a run trains it (positives only)
PHERC0211_MERGED_TEST_ID = 20260717193517
PHERC1447_IDS = (20260930144758, 20260930144760)
PHERC0841_ID = 20260221022814
PHERC0172_IDS = (20251111010954, 20251112000002)
PARIS4_ID = 20231210121321
GENERATOR = "models/c39_mae_crossres_depth.generator.pth"
PHERC0211_DOMAIN = campaign40.NEW_DOMAIN
PHERC1447_DOMAIN = "pherc1447"
ENCODER_LR_SCALE = 0.25
VIS_SCROLL_IDS = [PHERC0211_ID, PHERC0211_VIS_ONLY_ID, PHERC0211_MERGED_TEST_ID, PHERC0841_ID]
LABEL_FREE_VIS_IDS = [PHERC0211_VIS_ONLY_ID, PHERC0211_MERGED_TEST_ID]
# (tid suffix, domain, scroll id) for the PHerc1447 + one-scroll tests
ADDED_SCROLLS = (
    ("pherc0841", "pherc0841", PHERC0841_ID),
    ("pherc0009b", "pherc0009b", 20250919125754),
    ("pherc0139_w044", "pherc0139", 20260115000000),
    ("pherc0139_w035", "pherc0139", 20260317000000),
    ("pherc0343p", "pherc0343p", 20250511003658),
    ("pherc0500p2", "pherc0500p2", 20250628074500),
)
SCROLL_WEIGHT_BY_ID = {
    PHERC0211_ID: 5,
    **{scroll_id: 1 for _, _, scroll_id in ADDED_SCROLLS},
    **{scroll_id: 1 for scroll_id in (*PHERC0172_IDS, PARIS4_ID)},
}
# both PHerc0211 segments share one domain, so they must carry the same weight
SCROLL_WEIGHT_BY_ID[PHERC0211_MERGED_TEST_ID] = SCROLL_WEIGHT_BY_ID[PHERC0211_ID]

# test 3: absolute learning rates. enc1/enc2 are fully frozen (requires_grad=False and excluded
# from the optimizer); enc3 + bottleneck are the "lower encoder" group; everything else
# (decoder + head) is the task group.
FREEZE12_LAYER_LR = {
    "freeze_prefixes": ("enc1.", "enc2."),
    "encoder_lr": 1e-6,
    "task_lr": 5e-5,
}
NOFREEZE_LAYER_LR = {**FREEZE12_LAYER_LR, "freeze_prefixes": ()}
# the stem must re-learn depth at 2x sampling, so it trains at 1e-5 instead of staying frozen
DEPTH16_LAYER_LR = {"freeze_prefixes": ("enc2.",), "encoder_lr": 1e-5, "task_lr": 5e-5}

SCROLL_CONFIGS = {
    PHERC0211_ID: ScrollConfig(PHERC0211_ID, split_axis="x", train_split_frac=1.0),
    PHERC0211_MERGED_TEST_ID: ScrollConfig(PHERC0211_MERGED_TEST_ID, split_axis="x", train_split_frac=1.0),
    PHERC1447_IDS[0]: ScrollConfig(PHERC1447_IDS[0], split_axis="x", train_split_frac=0.75),
    PHERC1447_IDS[1]: ScrollConfig(PHERC1447_IDS[1], split_axis="x", train_split_frac=0.75),
    **{
        int(scroll.scroll_id): scroll
        for scroll in DEFAULT_SCROLLS
        if int(scroll.scroll_id) in {scroll_id for _, _, scroll_id in ADDED_SCROLLS} | {*PHERC0172_IDS, PARIS4_ID}
    },
}
TRAIN_1447 = {PHERC1447_DOMAIN: list(PHERC1447_IDS)}
TRAIN_1447_0211 = {PHERC1447_DOMAIN: list(PHERC1447_IDS), PHERC0211_DOMAIN: [PHERC0211_ID]}
# the six-scroll set minus PHerc0343P and PHerc0841; PHerc0211 stays held out
TRAIN_ALL = {
    PHERC1447_DOMAIN: list(PHERC1447_IDS),
    "pherc0009b": [20250919125754],
    "pherc0139": [20260115000000, 20260317000000],
    "pherc0500p2": [20250628074500],
}
TRAIN_ALL_0211 = {**TRAIN_ALL, PHERC0211_DOMAIN: [PHERC0211_ID]}
# both PHerc1447, both PHerc0139, PHerc0009B, PHerc0500P2, PHerc0841; no PHerc0343P or PHerc0211
TRAIN_SIX_0841 = {**TRAIN_ALL, "pherc0841": [PHERC0841_ID]}
TRAIN_SIX_0841_0211 = {**TRAIN_SIX_0841, PHERC0211_DOMAIN: [PHERC0211_ID]}
TRAIN_SIX_0841_0211_MERGED = {**TRAIN_SIX_0841, PHERC0211_DOMAIN: [PHERC0211_ID, PHERC0211_MERGED_TEST_ID]}
PHERC0211_X3_WEIGHTS = {**SCROLL_WEIGHT_BY_ID, PHERC0211_ID: 3, PHERC0211_MERGED_TEST_ID: 3}
VIS_0211_ONLY = [PHERC0211_ID]
TRAIN_SEVEN_0172 = {**TRAIN_SIX_0841, "pherc0172": list(PHERC0172_IDS)}
TRAIN_SEVEN_PARIS4 = {**TRAIN_SIX_0841, "phercparis4": [PARIS4_ID]}
ALL_0211_IDS = [PHERC0211_ID, PHERC0211_VIS_ONLY_ID, PHERC0211_MERGED_TEST_ID]


def _train_scroll_weights(
    train_scroll_dict: dict[str, list[int]],
    overrides: dict[int, int] = SCROLL_WEIGHT_BY_ID,
) -> list[int]:
    """domain weights derived from per-scroll overrides, falling back to campaign-40 defaults."""
    weights = []
    for domain, scroll_ids in train_scroll_dict.items():
        default = int(campaign40.REGIME_WEIGHT_BY_DOMAIN.get(domain, 1))
        chosen = {
            int(overrides.get(int(scroll_id), default))
            for scroll_id in scroll_ids
        }
        if len(chosen) != 1:
            raise ValueError(f"domain {domain} has inconsistent per-scroll weights: {sorted(chosen)}")
        weight = chosen.pop()
        if weight < 1:
            raise ValueError(f"domain {domain} has non-positive weight {weight}")
        weights.append(weight)
    return weights


def _check_weights(config, test: dict) -> None:
    """post-condition: the weights on the final config match the exposed per-scroll/domain weights."""
    domains = list(config.data.train_scroll_dict)
    weights = [int(w) for w in config.data.train_scroll_weights]
    expected = _train_scroll_weights(config.data.train_scroll_dict, test["weight_overrides"])
    if len(weights) != len(domains):
        raise ValueError(f"{test['tid']}: {len(weights)} weights for {len(domains)} domains")
    if weights != expected:
        raise ValueError(f"{test['tid']}: train_scroll_weights {weights} != expected {expected} for {domains}")
    if [int(w) for w in test["train_scroll_weights"]] != expected:
        raise ValueError(f"{test['tid']}: test weights drifted from expected {expected}")
    if int(config.tra.dann_n_domains) != len(domains):
        raise ValueError(f"{test['tid']}: dann_n_domains {config.tra.dann_n_domains} != {len(domains)} domains")


def _test(
    tid: str,
    train_scroll_dict: dict[str, list[int]],
    init_weights: str,
    close_r: int,
    gap_r: int,
    shell_r: int,
    *,
    multitile_pos_only: bool | None = None,
    layer_lr: dict | None = None,
    pcgrad: bool = False,
    nnpu: bool = False,
    vis_scroll_ids: list[int] | None = None,
    vis_only_ids: list[int] | None = None,
    n_epochs: int = 8,
    eval_int: int | None = None,
    scroll_configs: dict[int, ScrollConfig] = SCROLL_CONFIGS,
    weight_overrides: dict[int, int] = SCROLL_WEIGHT_BY_ID,
    translated_ids: tuple[int, ...] = (),
    input_generator: str | None = None,
) -> dict:
    scrolls = [scroll_configs[int(sid)] for sid in campaign40._scroll_ids_from_dict(train_scroll_dict)]
    test = campaign40._test(
        tid,
        campaign40.SURFACE_FIBER_MAE_KEY,
        train_scroll_dict,
        scrolls,
        init_weights=init_weights,
        changes={
            "data.ring_close_r": int(close_r),
            "data.ring_gap_r": int(gap_r),
            "data.ring_shell_r": int(shell_r),
            **(campaign40.NNPU if nnpu else {}),
            **({"data.zarr_suffix": {str(int(sid)): campaign40.TRANSLATED_SUFFIX for sid in translated_ids}}
               if translated_ids else {}),
        },
        **campaign40.SCRATCH_TRAINING,
    )
    test["tag"] = tid
    test["ring_close_r"] = int(close_r)
    test["ring_gap_r"] = int(gap_r)
    test["ring_shell_r"] = int(shell_r)
    test["weight_overrides"] = dict(weight_overrides)
    test["train_scroll_weights"] = _train_scroll_weights(train_scroll_dict, weight_overrides)
    # None -> leave whatever campaign40 / DataConfig default is in place
    test["multitile_pos_only"] = None if multitile_pos_only is None else bool(multitile_pos_only)
    test["layer_lr"] = dict(layer_lr) if layer_lr else None
    test["pcgrad"] = bool(pcgrad)
    test["vis_scroll_ids"] = list(vis_scroll_ids or VIS_SCROLL_IDS)
    # visualized scrolls that must never be trained, on top of LABEL_FREE_VIS_IDS
    test["vis_only_ids"] = list(vis_only_ids or [])
    test["n_epochs"] = int(n_epochs)
    test["eval_int"] = int(eval_int if eval_int is not None else n_epochs)
    test["input_generator"] = input_generator
    return test


def _selected_scrolls(selected: list[dict]) -> list[ScrollConfig]:
    ordered: dict[int, ScrollConfig] = {}
    for test in selected:
        for scroll in test["scrolls"]:
            ordered[int(scroll.scroll_id)] = scroll
    return list(ordered.values())


def _surface_anchor_ids(selected: list[dict]) -> tuple:
    return tuple(dict.fromkeys(
        [int(scroll.scroll_id) for scroll in _selected_scrolls(selected)]
        + [int(scroll_id) for test in selected for scroll_id in test.get("vis_scroll_ids", [])]
        # translated siblings carry their own anchors under <id><suffix>
        + [f"{sid}{suffix}" for test in selected
           for sid, suffix in (test["config"].get("data.zarr_suffix") or {}).items()]
    ))


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
    # campaign31 pre-creates MODEL_DIR/<tid>; checkpoints go to MODEL_DIR/<run_name> instead
    with contextlib.suppress(OSError):
        (ROOT / config.model_dir).rmdir()
    run_name = f"{test['tid']}_{datetime.now():%d_%H-%M-%S}"
    config.exp_name = config.run_name = run_name
    config.model_dir = os.path.join(MODEL_DIR, run_name)
    config.save_final = os.path.join(config.model_dir, "final.pth")
    if config.tra.character_forgetting:
        config.tra.character_forgetting_path = os.path.join(config.model_dir, "character_forgetting.json")
    config.data.scrolls = list(test["scrolls"])
    config.data.train_scroll_dict = {
        domain: list(ids) for domain, ids in test["train_scroll_dict"].items()
    }
    config.data.train_scroll_weights = list(test["train_scroll_weights"])
    config.data.simple_split = False
    config.data.train_mask_dir = "./train_masks"
    config.data.inklabel_dir = INKLABEL_DIR
    config.data.vis_scroll_ids = list(test["vis_scroll_ids"])
    config.data.vis_preload_persistent = True
    trained_ids = {int(sid) for ids in config.data.train_scroll_dict.values() for sid in ids}
    config.data.label_free_vis_scroll_ids = [sid for sid in LABEL_FREE_VIS_IDS if sid not in trained_ids]
    config.data.train_only_scroll_ids = []
    config.data.ring_close_r = int(test["ring_close_r"])
    config.data.ring_gap_r = int(test["ring_gap_r"])
    config.data.ring_shell_r = int(test["ring_shell_r"])
    # PHerc0211 letters are best guesses: their neighbours may be unlabelled ink, not ring negatives
    config.data.ring_free_scroll_ids = [PHERC0211_ID, PHERC0211_MERGED_TEST_ID]

    # per-test override of DataConfig.multitile_pos_only (only set when the test asks for it)
    if test.get("multitile_pos_only") is not None:
        if not hasattr(config.data, "multitile_pos_only"):
            raise AttributeError("DataConfig has no attribute 'multitile_pos_only'")
        config.data.multitile_pos_only = bool(test["multitile_pos_only"])

    config.tra.n_epochs = int(test["n_epochs"])
    config.tra.eval_int = int(test["eval_int"])
    config.tra.save_int = 1
    config.tra.eval_int_scrolls = len(test["vis_scroll_ids"])
    config.tra.test_int = 999
    config.tra.fast_eval_figure = False
    config.tra.encoder_lr_scale = ENCODER_LR_SCALE
    config.tra.freeze_prefixes = ()
    config.tra.encoder_lr = None
    config.tra.task_lr = None
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)

    layer_lr = test.get("layer_lr")
    if layer_lr:
        # absolute per-group LRs; the optimizer turns these into lr_scale vs config.tra.lr
        config.tra.freeze_prefixes = tuple(layer_lr["freeze_prefixes"])
        config.tra.encoder_lr = float(layer_lr["encoder_lr"])
        config.tra.task_lr = float(layer_lr["task_lr"])
        config.tra.lr = config.tra.task_lr          # base lr == decoder+head lr
        config.tra.encoder_lr_scale = 1.0           # superseded by encoder_lr
        config.tra.encoder_freeze_epochs = 0

    if test.get("pcgrad"):
        # full PCGrad: one task gradient per physical domain, projected against every other domain
        config.tra.pcgrad = True
        config.tra.pcgrad_groups = 0
        config.tra.pcgrad_lite = False
        config.tra.pcgrad_gram = False
        # mutually exclusive with PCGrad in the trainer
        config.tra.physical_patch_groupdro = False
        config.tra.physical_domain_groupdro = False

    if test.get("input_generator"):
        config.model.input_generator = str(test["input_generator"])

    _check_weights(config, test)
    trained = {int(sid) for ids in config.data.train_scroll_dict.values() for sid in ids}
    trained |= {int(scroll.scroll_id) for scroll in config.data.scrolls}
    leaked = trained & (
        set(config.data.label_free_vis_scroll_ids) | {int(sid) for sid in test.get("vis_only_ids", [])}
    )
    if leaked:
        raise ValueError(f"{test['tid']}: vis-only scrolls {sorted(leaked)} must not be trained")
    return config


def preflight_inputs(selected: list[dict], dry_run: bool) -> None:
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    failures = []
    for scroll in _selected_scrolls(selected):
        scroll_id = int(scroll.scroll_id)
        required = (
            zarr_root / f"{scroll_id}.zarr",
            ROOT / "masks" / f"{scroll_id}.png",
            ROOT / INKLABEL_DIR.strip("./") / f"{scroll_id}.png",
            ROOT / "train_masks" / f"{scroll_id}.png",
            ROOT / "surface_labels" / str(scroll_id) / "depth.npy",
            ROOT / "surface_labels" / str(scroll_id) / "confidence.npy",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            failures.append(f"{scroll_id}: {', '.join(missing)}")
    for scroll_id in LABEL_FREE_VIS_IDS:
        required = (
            zarr_root / f"{scroll_id}.zarr",
            ROOT / "masks" / f"{scroll_id}.png",
            ROOT / "surface_labels" / str(scroll_id) / "depth.npy",
            ROOT / "surface_labels" / str(scroll_id) / "confidence.npy",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            failures.append(f"{scroll_id} (vis-only): {', '.join(missing)}")
    external = []
    for test in selected:
        for path in campaign40._external_files(test) + ([test["input_generator"]] if test.get("input_generator") else []):
            if not (ROOT / path).is_file() and not Path(path).is_file():
                external.append(path)
    translation = sorted({problem for test in selected for problem in campaign40._translation_problems(test)})
    if failures or external or translation:
        message = []
        if failures:
            message.append("fine-tune inputs missing:\n  " + "\n  ".join(failures))
        if external:
            message.append(f"external files missing: {sorted(set(external))}")
        if translation:
            message.append("translated siblings not ready:\n  " + "\n  ".join(translation))
        text = "\n".join(message)
        if not dry_run:
            raise FileNotFoundError(text)
        print(f"[campaign_finetune] WARNING {text}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="fine-tune archs40 native-96 combined-surface model on PHerc1447 (+ PHerc0211)")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--init-weights", default=DEFAULT_INIT_WEIGHTS,
                        help="starting checkpoint (default: archs40 holdout_n96_combined_surface_norm final.pth)")
    args = parser.parse_args()

    selected = [
        # _test("ft_six_03_txcylmxpixmmwncpio_17_wceicoyopit", TRAIN_SIX_0841_0211_MERGED, args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR,
        #       nnpu=True, vis_scroll_ids=VIS_SCROLL_IDS, vis_only_ids=[PHERC0211_VIS_ONLY_ID]),
        # PHerc0211 is visualized but never trained in the tests below
        _test("ft_6_base", TRAIN_SIX_0841, args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR,
              nnpu=True, vis_only_ids=ALL_0211_IDS),
        _test("ft_7_0172", TRAIN_SEVEN_0172, args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR,
              nnpu=True, vis_only_ids=ALL_0211_IDS),
        _test("ft_7_paris4", TRAIN_SEVEN_PARIS4, args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR,
              nnpu=True, vis_only_ids=ALL_0211_IDS, translated_ids=(PARIS4_ID,)),
        _test("ft_6_16depth", TRAIN_SIX_0841, args.init_weights, 2, 2, 4, layer_lr=DEPTH16_LAYER_LR,
              nnpu=True, vis_only_ids=ALL_0211_IDS, input_generator=GENERATOR),
        # needs letters drawn in dilated_inklabels/20260717193517.png first: a ring-free scroll with no positives has
        # nothing to train on
        # _test("ft_six_0211_merged", TRAIN_SIX_0841_0211_MERGED, args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR,
        #       nnpu=True, vis_scroll_ids=VIS_SCROLL_IDS, vis_only_ids=[PHERC0211_VIS_ONLY_ID],
        #       weight_overrides=PHERC0211_X3_WEIGHTS),
        # _test("ft_1447", TRAIN_1447, args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR),
        # _test("ft_all_pioychi_h", TRAIN_ALL_0211, args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR),
        # PHerc0211 visualized but not trained
        # *(
        #     _test(tid, TRAIN_ALL, args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR, pcgrad=pcgrad,
        #           vis_scroll_ids=VIS_0211_ONLY)
        #     for tid, pcgrad in (("ft_all", False), ("ft_all_pcgrad", True))
        # ),
    ]
    # for suffix, domain, scroll_id in ADDED_SCROLLS:
    #     test = _test(
    #         f"ft_1447_{suffix}",
    #         {PHERC1447_DOMAIN: list(PHERC1447_IDS), domain: [scroll_id]},
    #         args.init_weights, 2, 2, 4, layer_lr=FREEZE12_LAYER_LR,
    #     )
    #     test["vis_scroll_ids"] = [scroll_id, PHERC0211_ID]
    #     selected.append(test)
    for test in selected:
        test.setdefault("vis_scroll_ids", list(VIS_SCROLL_IDS))
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [test for test in selected if test["tid"] in wanted]
        missing = wanted - {test["tid"] for test in selected}
        if missing:
            raise ValueError(f"unknown test ids: {sorted(missing)}")

    with startup_output():
        preflight_inputs(selected, args.dry_run)
        if not args.dry_run:
            ensure_surface_anchors(_surface_anchor_ids(selected), str(ROOT / "ves_zarrs2"))
        print(f"[campaign_finetune] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        with startup_output():
            print(
                f"[campaign_finetune] {test['tid']}: ctx={config.data.context_size} "
                f"domains={list(config.data.train_scroll_dict)} "
                f"weights={dict(zip(config.data.train_scroll_dict, config.data.train_scroll_weights))} "
                f"simple_split={config.data.simple_split} "
                f"ring=({config.data.ring_close_r},{config.data.ring_gap_r},{config.data.ring_shell_r}) "
                f"multitile_pos_only={getattr(config.data, 'multitile_pos_only', None)} "
                f"fast_eval={config.tra.fast_eval_figure} vis={config.data.vis_scroll_ids} "
                f"batch={config.dl.batch_size} lr={config.tra.lr} epochs={config.tra.n_epochs} "
                f"encoder_lr_scale={config.tra.encoder_lr_scale} "
                f"freeze={config.tra.freeze_prefixes} encoder_lr={config.tra.encoder_lr} "
                f"task_lr={config.tra.task_lr} pcgrad={config.tra.pcgrad} "
                f"pu={config.tra.pu_lambda}/{config.tra.pu_prior} far_share={config.data.far_negative_share} "
                f"patch_groupdro={config.tra.physical_patch_groupdro} "
                f"run={config.run_name} init={config.init_weights}",
                flush=True,
            )
        if args.dry_run:
            success = campaign31.run_test(config, True)
        else:
            campaign29.prewarm_data_cache(config)
            success = campaign31.run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        del config
        gc.collect()

    print(f"\n{'=' * 78}\n[campaign_finetune] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()