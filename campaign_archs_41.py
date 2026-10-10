"""campaign 41: degraded fine scans + PHerc1447 on the archs40 nnPU recipe, then cross-scroll and label-targeting arms

Every arm is campaign 40's `holdout_n96_combined_surface_norm_nnpu` (scratch campaign-40 MAE, native 96 px
field, ring c2g2s4, sigma-8 soft edges, bag ranking, fiber branch, regime weights, RSC, dropout/augs,
surface-anchor norm, nnPU far background, patch GroupDRO, batch 95 / lr 1.5e-4, 10 epochs) with:

- the fine-scan fragments (pherc1667 w013/w018/cr1fr3, paris4, paris2 fr143/fr47, paris1 fr34, 51cr4 fr8)
  read from their degrader-translated `.translated` siblings (models/c39_degrader.pth, synthetic 9.36 um)
  plus real 113 keV / 1.2 m scan noise on every training crop (build_noise_bank.py)
- PHerc1447 (20260930144758 / 20260930144760) added as its own physical domain (train_masks split)
- figures only after the final epoch, for five scrolls: the vis-only p343_letters (20261007000000),
  PHerc0211 (20260928000003) and PHerc0813 (20260928000007), which are preloaded but never trained on,
  plus the trained pherc0500p2 (20250628074500) and pherc0814 seg46527 (20260226000000)

| tid                 | change vs baseline                                                         |
|---------------------|----------------------------------------------------------------------------|
| baseline            | none                                                                       |
| and_mask            | AND-mask: per step, 4 random groups of physical domains run separately and |
|                     | only weights whose gradient sign >= 3 of 4 groups share are updated        |
| divdis              | DivDis: a second dec1 + head route on the shared trunk; both fit the       |
|                     | labels, while on unlabeled PHerc0211/PHerc0813 crops their predictions     |
|                     | must carry no mutual information and keep a 0.15 ink rate. The eval head   |
|                     | is the better one on PHerc1447 validation (character AP)                   |
| character_failures  | static character forgetting: characters this repo's current model fails on |
|                     | are drawn more often (character_failures.py, weight 1-4)                   |
| new_letters         | assumed letters drawn at half brightness (value ~128) in dilated_inklabels |
|                     | train as their own characters and fill 30% of all training draws           |

`character_failures` needs character_failures.py to have written FORGETTING_PATH for every training
scroll from the current ./inklabels; `new_letters` needs at least one assumed letter in a training split.
Both are checked when the arm starts (they run last) and skipped if still not ready. Assumed letters get
the same ring/close/gap/softening as every other letter, follow train_masks, are kept out of patch
GroupDRO / CVaR (they re-enter as a plain cell-weighted mean) and are scored separately (Character/Extra/*,
Character/Patch/<id>/F1/*_Extra, Per_Scroll/*_Extra/*). The other arms erase half-brightness strokes, so
they read the label files exactly as campaign 40 did.

Usage:
    python3 campaign_archs_41.py --dry-run
    python3 character_failures.py            # required before character_failures
    python3 campaign_archs_41.py
    python3 campaign_archs_41.py --only baseline,new_letters,and_mask,divdis
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
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
from utils.config import ScrollConfig, startup_output
from utils.norm import ensure_surface_anchors


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs41"
MODEL_DIR = "models/archs41"
INKLABEL_DIR = "./dilated_inklabels"
N_EPOCHS = 10
PHERC1447_DOMAIN = "pherc1447"
PHERC1447_IDS = (20260930144758, 20260930144760)
P343_LETTERS_ID = 20261007000000
PHERC0211_ID = 20260928000003
PHERC0813_ID = 20260928000007
VIS_ONLY_IDS = [P343_LETTERS_ID, PHERC0211_ID, PHERC0813_ID]
TRAINED_VIS_IDS = [20250628074500, 20260226000000]  # pherc0500p2, pherc0814 seg46527
VIS_SCROLL_IDS = [*VIS_ONLY_IDS, *TRAINED_VIS_IDS]
FORGETTING_PATH = "character_failures/c41_character_failures.json"
DENOISER = "models/denoiser_n2v_block3_4k.pth"
EXTRA_LABEL_SHARE = 0.3
# missing zarrs and the command that assembles each one
ASSEMBLY = {
    20260930144758: ("assemble_training_segments.py", "ph1447_w058"),
    20260930144760: ("assemble_training_segments.py", "ph1447_w060"),
    P343_LETTERS_ID: ("assemble_training_segments.py", "p343_letters"),
    PHERC0211_ID: ("assemble_test_segments.py", str(PHERC0211_ID)),
    PHERC0813_ID: ("assemble_test_segments.py", str(PHERC0813_ID)),
}

TRAIN_SCROLL_DICT = {
    **campaign40.BASE_SCROLL_DICT,
    PHERC1447_DOMAIN: list(PHERC1447_IDS),
}
SCROLLS = [
    *campaign40.SCROLLS_BASE,
    ScrollConfig(PHERC1447_IDS[0], split_axis="x", train_split_frac=0.75),
    ScrollConfig(PHERC1447_IDS[1], split_axis="x", train_split_frac=0.75),
]
DEGRADED = {**campaign40.DOWNSAMPLED, **campaign40.NATIVE_NOISE}
CHARACTER_FAILURES = {
    "tra.character_forgetting": True,
    "tra.character_forgetting_static": True,
    "tra.character_forgetting_path": FORGETTING_PATH,
    "tra.character_forgetting_max_weight": 4.0,
}
NEW_LETTERS = {
    "data.extra_inklabels": True,
    "data.extra_label_share": EXTRA_LABEL_SHARE,
}
# sign agreement across 4 random groups of physical domains: a coordinate moves only if >= 3 of 4 agree
AND_MASK = {
    "tra.and_mask": True,
    "tra.and_mask_groups": 4,
    "tra.and_mask_threshold": 0.5,
}
# unlabeled target crops: the label-free vis scrolls, never p343 (the held-out judge)
DIVDIS_TARGET_IDS = [PHERC0211_ID, PHERC0813_ID]
DIVDIS = {
    "model.divdis_heads": 2,
    "data.divdis_target_scroll_ids": DIVDIS_TARGET_IDS,
    "tra.divdis_mi_lambda": 1.0,
    "tra.divdis_reg_lambda": 1.0,
    "tra.divdis_target_prior": 0.15,
    "tra.divdis_batch_size": 32,
    "tra.divdis_select_domains": [PHERC1447_DOMAIN],
}


def _test(tid: str, changes: dict | None = None) -> dict:
    test = campaign40._test(
        tid,
        campaign40.SURFACE_FIBER_MAE_KEY,
        TRAIN_SCROLL_DICT,
        SCROLLS,
        changes={**campaign40.NNPU, **DEGRADED, **(changes or {})},
        **campaign40.SCRATCH_TRAINING,
    )
    test["tag"] = f"41_{tid}"
    return test


# arms that need hand-made inputs run last, so letters can still be drawn while the others train
TESTS = [
    _test("baseline"),
    _test("and_mask", AND_MASK),
    _test("divdis", DIVDIS),
    _test("character_failures", CHARACTER_FAILURES),
    _test("new_letters", NEW_LETTERS),
]


def training_scroll_ids() -> list[int]:
    return [int(scroll.scroll_id) for scroll in SCROLLS]


def _label_free_vis_ids() -> list[int]:
    """vis-only scrolls without a label or train mask render with empty labels."""
    return [
        scroll_id for scroll_id in VIS_ONLY_IDS
        if not (ROOT / INKLABEL_DIR / f"{scroll_id}.png").is_file()
        or not (ROOT / "train_masks" / f"{scroll_id}.png").is_file()
    ]


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
    config.data.train_only_scroll_ids = []
    config.data.vis_scroll_ids = list(VIS_SCROLL_IDS)
    config.data.vis_preload_persistent = True
    config.data.label_free_vis_scroll_ids = _label_free_vis_ids()
    config.tra.n_epochs = N_EPOCHS
    config.tra.eval_int = N_EPOCHS
    config.tra.eval_int_scrolls = len(VIS_SCROLL_IDS)
    config.tra.test_int = 999
    config.tra.probe_int = 999
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    trained = {int(scroll.scroll_id) for scroll in config.data.scrolls}
    leaked = trained & set(VIS_ONLY_IDS)
    if leaked:
        raise ValueError(f"vis-only scrolls {sorted(leaked)} must not be trained")
    return config


def _missing_inputs() -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """(missing zarrs by scroll, other missing files by scroll) for training and vis scrolls."""
    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    zarrs, others = {}, {}
    label_free = set(_label_free_vis_ids())
    for scroll_id in [*training_scroll_ids(), *VIS_ONLY_IDS]:
        if not (zarr_root / f"{scroll_id}.zarr").is_dir():
            zarrs[scroll_id] = [str(zarr_root / f"{scroll_id}.zarr")]
        required = [
            ROOT / "masks" / f"{scroll_id}.png",
            ROOT / "surface_labels" / str(scroll_id) / "depth.npy",
            ROOT / "surface_labels" / str(scroll_id) / "confidence.npy",
        ]
        if scroll_id not in label_free:
            required += [ROOT / INKLABEL_DIR / f"{scroll_id}.png", ROOT / "train_masks" / f"{scroll_id}.png"]
        missing = [str(path) for path in required if not path.exists()]
        if not missing and scroll_id not in zarrs:
            missing = _frame_mismatches(zarr_root / f"{scroll_id}.zarr", scroll_id)
        if missing:
            others[scroll_id] = missing
    return zarrs, others


def _frame_mismatches(volume_path: Path, scroll_id: int) -> list[str]:
    """mask / surface maps whose frame differs from the zarr (they would crash the data loader)."""
    import cv2
    import numpy as np
    import zarr

    height, width = (int(value) for value in zarr.open(str(volume_path), mode="r").shape[-2:])
    shapes = {
        f"masks/{scroll_id}.png": cv2.imread(str(ROOT / "masks" / f"{scroll_id}.png"), cv2.IMREAD_GRAYSCALE).shape,
        f"surface_labels/{scroll_id}/depth.npy": np.load(
            ROOT / "surface_labels" / str(scroll_id) / "depth.npy", mmap_mode="r",
        ).shape,
    }
    return [
        f"{name} is {shape[0]}x{shape[1]} but the zarr is {height}x{width}"
        for name, shape in shapes.items() if tuple(shape) != (height, width)
    ]


def preflight_inputs(dry_run: bool) -> None:
    """assemble missing training/vis zarrs, then require every mask, label and surface map."""
    zarrs, _ = _missing_inputs()
    unknown = sorted(set(zarrs) - set(ASSEMBLY))
    if unknown:
        message = f"campaign 41 zarrs missing with no assembly recipe: {unknown}"
        if not dry_run:
            raise FileNotFoundError(message)
        print(f"[campaign41] WARNING {message}", flush=True)
    by_script: dict[str, list[str]] = {}
    for scroll_id in sorted(set(zarrs) & set(ASSEMBLY)):
        script, name = ASSEMBLY[scroll_id]
        by_script.setdefault(script, []).append(name)
    for script, names in by_script.items():
        if script == "assemble_test_segments.py":
            command = [sys.executable, str(ROOT / script), *(arg for name in names for arg in ("--only", name))]
        else:
            command = [sys.executable, str(ROOT / script), "--only", ",".join(names)]
        if dry_run:
            print(f"[campaign41] would assemble missing zarrs with: {' '.join(command)}", flush=True)
            continue
        print(f"[campaign41] assembling missing zarrs: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    zarrs, others = _missing_inputs()
    if zarrs and not dry_run:
        raise FileNotFoundError(f"campaign 41 zarrs still missing after assembly: {sorted(zarrs)}")
    for scroll_id, paths in zarrs.items():
        others[scroll_id] = paths + others.get(scroll_id, [])
    if others:
        message = "campaign 41 inputs missing:\n  " + "\n  ".join(
            f"{scroll_id}: {', '.join(paths)}" for scroll_id, paths in sorted(others.items())
        )
        if not dry_run:
            raise FileNotFoundError(message)
        print(f"[campaign41] WARNING {message}", flush=True)


def ensure_noise_bank(test: dict, dry_run: bool) -> None:
    bank = test["config"].get("data.native_noise")
    if not bank:
        return
    paths = [ROOT / str(bank), (ROOT / str(bank)).with_suffix(".json")]
    if all(path.is_file() for path in paths):
        return
    if not (ROOT / DENOISER).is_file():
        message = f"campaign 41 noise bank needs the pretrained denoiser {DENOISER}"
        if not dry_run:
            raise FileNotFoundError(message)
        print(f"[campaign41] WARNING {message}", flush=True)
    command = [sys.executable, str(ROOT / "build_noise_bank.py"), "--out", str(bank), "--denoiser", DENOISER]
    if dry_run:
        print(f"[campaign41] would build the native noise bank with: {' '.join(command)}", flush=True)
        return
    print(f"[campaign41] building the native noise bank: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"campaign 41 noise bank still missing after build: {missing}")


def forgetting_problems() -> list[str]:
    """why FORGETTING_PATH cannot drive the character_failures arm (empty when it can)."""
    from utils.dataloader import inklabel_fingerprint

    path = ROOT / FORGETTING_PATH
    if not path.is_file():
        return [f"{FORGETTING_PATH} is missing"]
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{FORGETTING_PATH} is unreadable: {exc}"]
    problems = []
    for scroll_id in training_scroll_ids():
        entry = stored.get("scrolls", {}).get(str(scroll_id))
        label = ROOT / "inklabels" / f"{scroll_id}.png"
        if entry is None:
            problems.append(f"{scroll_id}: no entry")
        elif not label.is_file():
            problems.append(f"{scroll_id}: {label} missing")
        elif entry.get("inklabel_fingerprint") != inklabel_fingerprint(str(label)):
            problems.append(f"{scroll_id}: ./inklabels changed since the file was written")
    return problems


def extra_letter_counts() -> dict[int, int]:
    """assumed-letter pixels per training scroll label file."""
    from utils.dataloader import imread_gray, split_extra_labels

    counts = {}
    for scroll_id in training_scroll_ids():
        raw = imread_gray(str(ROOT / INKLABEL_DIR / f"{scroll_id}.png"))
        if raw is None:
            continue
        _, extra = split_extra_labels(raw)
        if extra.any():
            counts[scroll_id] = int(extra.sum())
    return counts


def arm_problems(test: dict) -> list[str]:
    """hand-made inputs this arm still lacks (empty when it can run)."""
    if test["tid"] == "character_failures":
        failures = forgetting_problems()
        return ["needs `python3 character_failures.py` first: " + "; ".join(failures)] if failures else []
    if test["tid"] == "new_letters" and not extra_letter_counts():
        return [f"needs assumed letters drawn at half brightness (~128) in {INKLABEL_DIR}/<id>.png"]
    return []


def preflight_arms(selected: list[dict]) -> None:
    """warn at startup; each arm is checked again when it starts and skipped if still not ready."""
    for test in selected:
        problems = arm_problems(test)
        if problems:
            print(f"[campaign41] WARNING {test['tid']} not ready yet: {problems[0]}", flush=True)
        elif test["tid"] == "character_failures":
            print(f"[campaign41] {FORGETTING_PATH} covers all {len(SCROLLS)} training scrolls", flush=True)
        elif test["tid"] == "new_letters":
            print(f"[campaign41] assumed-letter pixels by scroll: {extra_letter_counts()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 41: degraded fine scans + PHerc1447, label-targeting arms")
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
        preflight_arms(selected)
        preflight_inputs(args.dry_run)
        campaign29.preflight_train_masks(SCROLLS, inklabel_dir=ROOT / INKLABEL_DIR, strict=not args.dry_run)
        campaign40.preflight_external(selected, args.dry_run)
        campaign40._preflight_pretraining(selected, args.dry_run)
        for test in selected:
            campaign40._ensure_downsampled_inputs(test, args.dry_run)
            ensure_noise_bank(test, args.dry_run)
        if not args.dry_run:
            ensure_surface_anchors(
                tuple(dict.fromkeys([*training_scroll_ids(), *VIS_SCROLL_IDS])), str(ROOT / "ves_zarrs2"),
            )
        print(f"[campaign41] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        problems = arm_problems(test)
        if problems:
            print(f"[campaign41] SKIP {test['tid']}: {problems[0]}", flush=True)
            results[test["tid"]] = f"SKIPPED ({problems[0]})"
            continue
        config = build_config(test)
        with startup_output():
            print(
                f"[campaign41] {test['tid']}: ctx={config.data.context_size} "
                f"domains={list(config.data.train_scroll_dict)} weights={config.data.train_scroll_weights} "
                f"translated={len(config.data.zarr_suffix)} noise={config.data.native_noise or 'none'} "
                f"vis={config.data.vis_scroll_ids} label_free={config.data.label_free_vis_scroll_ids} "
                f"batch={config.dl.batch_size} lr={config.tra.lr} epochs={config.tra.n_epochs} "
                f"init={config.init_weights} forgetting={config.tra.character_forgetting} "
                f"extra_letters={config.data.extra_inklabels}/{config.data.extra_label_share} "
                f"and_mask={config.tra.and_mask} divdis_heads={config.model.divdis_heads} "
                f"divdis_targets={config.data.divdis_target_scroll_ids}",
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

    print(f"\n{'=' * 78}\n[campaign41] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
