"""campaign 32: researcher inklabels and ring geometry

Baseline is Campaign 31 `early_gated_patch_groupdro` with model EMA. Only the 8 training
patches with official researcher ink labels (2026-09-18) are trained on, grouped into
the same physical domains as Campaign 29. Labels AND rings come from
`researcher_inklabels/`; forced positives/negatives in `train_masks` never seed a ring.

Arms sweep the ring exclusion gap and pos_only:
  pos_only=True : unlabelled regions are unsupervised; windows containing ink emit only
                  positives and hard (forced) negatives.
  pos_only=False: every sub-tile inside the ring mask is supervised, so positives and
                  ring negatives may touch.

Usage:
    python3 campaign_archs_32.py --dry-run
    python3 campaign_archs_32.py --only baseline
    python3 campaign_archs_32.py --from c0g0_nopos
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
from utils.config import startup_output


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs32"
MODEL_DIR = "models/archs32"
INKLABEL_DIR = "./researcher_inklabels"
VIS_SCROLL_ID = 20260317000000  # w035

RESEARCHER_SCROLL_IDS = (
    20260115000000,  # w044
    20260317000000,  # w035
    20240304141531,  # w013
    20240304144031,  # w018
    20250919125754,  # PHerc0009B 487
    20231210121321,  # PHercParis4
    20260226000000,  # PHerc0814
    20260221022814,  # PHerc0841
)
_WEIGHT_BY_DOMAIN = dict(zip(campaign29.CAMPAIGN28_SCROLL_DICT, campaign29.CAMPAIGN28_SCROLL_WEIGHTS))
CAMPAIGN32_SCROLL_DICT = {
    domain: [scroll_id for scroll_id in scroll_ids if scroll_id in RESEARCHER_SCROLL_IDS]
    for domain, scroll_ids in campaign29.CAMPAIGN28_SCROLL_DICT.items()
}
CAMPAIGN32_SCROLL_DICT = {domain: ids for domain, ids in CAMPAIGN32_SCROLL_DICT.items() if ids}
CAMPAIGN32_SCROLL_WEIGHTS = [_WEIGHT_BY_DOMAIN[domain] for domain in CAMPAIGN32_SCROLL_DICT]
CAMPAIGN32_SCROLL_IDS = tuple(
    scroll_id for scroll_ids in CAMPAIGN32_SCROLL_DICT.values() for scroll_id in scroll_ids
)
assert set(CAMPAIGN32_SCROLL_IDS) == set(RESEARCHER_SCROLL_IDS)
CAMPAIGN32_SCROLLS = [campaign29._SCROLLS_BY_ID[scroll_id] for scroll_id in CAMPAIGN32_SCROLL_IDS]


def _test(tid: str, close_r: int, gap_r: int, shell_r: int, pos_only: bool, subtile: int = 16):
    test = campaign31._test(
        tid,
        "early_gated_c30",
        early_2d_unet=True,
        mid_2d_unet=False,
        physical_patch_groupdro=True,
        model_ema=True,
    )
    test.update({
        "tag": f"32_{tid}",
        "ring_close_r": close_r,
        "ring_gap_r": gap_r,
        "ring_shell_r": shell_r,
        "multitile_pos_only": pos_only,
        "multitile_subtile": subtile,
        # keep the 64px prediction center fixed
        "multitile_grid": 64 // subtile,
    })
    return test


TESTS = [
    _test("baseline", close_r=2, gap_r=2, shell_r=4, pos_only=True),
    _test("c1g1_pos", close_r=1, gap_r=1, shell_r=4, pos_only=True),
    _test("c0g0_pos", close_r=0, gap_r=0, shell_r=4, pos_only=True),
    _test("c1g1_nopos", close_r=1, gap_r=1, shell_r=4, pos_only=False),
    _test("c0g0_nopos", close_r=0, gap_r=0, shell_r=4, pos_only=False),
    _test("c0g0_nopos_8px", close_r=0, gap_r=0, shell_r=4, pos_only=False, subtile=8),
]


def build_config(test: dict):
    config = campaign31.build_config(test)
    config.exp_name = test["tag"]
    config.tra.log_dir = LOG_DIR

    config.data.scrolls = list(CAMPAIGN32_SCROLLS)
    config.data.train_scroll_dict = {
        domain: list(ids) for domain, ids in CAMPAIGN32_SCROLL_DICT.items()
    }
    config.data.train_scroll_weights = list(CAMPAIGN32_SCROLL_WEIGHTS)
    config.tra.dann_n_domains = len(CAMPAIGN32_SCROLL_DICT)

    config.data.inklabel_dir = INKLABEL_DIR
    config.data.ring_negatives = True
    config.data.ring_label_source = "closed"
    config.data.ring_from_inklabel_dir = True
    config.data.multitile_ring_gate = True
    config.data.ring_close_r = int(test["ring_close_r"])
    config.data.ring_gap_r = int(test["ring_gap_r"])
    config.data.ring_shell_r = int(test["ring_shell_r"])
    config.data.multitile_pos_only = bool(test["multitile_pos_only"])
    config.model.multitile_subtile = int(test["multitile_subtile"])
    config.model.multitile_grid = int(test["multitile_grid"])

    # one fast eval figure per run, on w035 only, from a full in-RAM copy of the volume
    config.data.vis_scroll_ids = [VIS_SCROLL_ID]
    config.data.ram_safe_vis = False
    config.tra.fast_eval_figure = True
    config.tra.eval_int = int(config.tra.n_epochs)
    config.tra.eval_int_scrolls = 1
    config.data.eval_chunk_gb = 3.0
    config.data.eval_prefetch = 3

    checkpoint_dir = os.path.join(MODEL_DIR, test["tid"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    config.tra.character_forgetting_path = (
        os.path.join(checkpoint_dir, "character_forgetting.json")
        if config.tra.character_forgetting else ""
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 32: researcher inklabels and ring geometry")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--from", dest="from_id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = TESTS
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [test for test in TESTS if test["tid"] in wanted]
        missing = wanted - {test["tid"] for test in selected}
        if missing:
            raise ValueError(f"unknown test ids: {sorted(missing)}")
    elif args.from_id:
        ids = [test["tid"] for test in TESTS]
        if args.from_id not in ids:
            raise ValueError(f"unknown --from {args.from_id!r}; valid={ids}")
        selected = TESTS[ids.index(args.from_id):]

    with startup_output():
        campaign29.preflight_train_masks(
            CAMPAIGN32_SCROLLS,
            inklabel_dir=ROOT / INKLABEL_DIR,
            strict=not args.dry_run,
        )
        campaign31.preflight_pretraining(selected, args.dry_run)
        print(f"[campaign32] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        with startup_output():
            print(
                f"[campaign32] {config.exp_name}: scrolls={len(config.data.scrolls)} "
                f"labels={config.data.inklabel_dir} ring=c{config.data.ring_close_r}"
                f"/g{config.data.ring_gap_r}/s{config.data.ring_shell_r} "
                f"pos_only={config.data.multitile_pos_only} "
                f"target={config.model.multitile_subtile}px grid={config.model.multitile_grid}",
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

    print(f"\n{'=' * 78}\n[campaign32] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
