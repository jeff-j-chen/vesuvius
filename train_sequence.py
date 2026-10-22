"""train_sequence: controlled single-purpose runs, one process each (logs ./runs_sequence, checkpoints models/runs_sequence).

default queue (add-one domain transfer to PHerc0846A), all cue_dropout recipe (asf cue + 30% flat-cue dropout, depth jitter +-1
reading the matching shifted cue map), the labels as they are on disk:
  cue_dropout_labelsv4               pure PHerc0846A baseline (the 7 trained patches, no PHerc0009B)
  cue_dropout_labelsv4_add_<key>     PHerc0846A + one added domain, domain weights 1:1; the added patches are rendered too
the 7 trained 0846A patches draw hard negatives from the half-brightness strokes drawn on their inklabels
(data.extra_hard_negative_scroll_ids); every other scroll's hard negatives are the half-brightness (119) regions of its train mask
(always on) and its half-brightness inklabel strokes stay erased (extra letters, unused).
data.equalize_domain keeps the 0846A draws per epoch identical in every run (add-one epochs are twice as long: the added
domain gets the other half), so runs differ only in what is added.

PHerc1447 trains on a flat surface at its measured centre layer instead of a detected one (_ves_tmp/flatten_1447.py), reading the
stored depth-reversed stack (attrs depth_flipped=true: the whole layer order reversed, as the reverse option of v8-in and
reader-v2 does); missing_inputs refuses to start otherwise.

python3 train_sequence.py --fragments runs the older one-fragment-alone runs (seq_<id>: train_flip's recipe, 6 epochs).

usage: python3 train_sequence.py [--only name,name] [--fragments] [--list] [--dry-run] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# train.py reads this before it imports torch
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import train_iter2 as base  # noqa: E402

LOG_ROOT = REPO / "runs_sequence"
MODEL_ROOT = REPO / "models" / "runs_sequence"
FRAGMENTS = base.NEW_SET
ALL_PATCHES = (*base.NEW_SET, base.HOLDOUT, *base.OUT_VIS)
EPOCHS = 6
WARMUP_EPOCHS = 2

BASELINE = "cue_dropout_labelsv4"
# key -> (domain name, patch ids); queue order: likely helpful first, then the mixed/harmful ones
DOMAIN_RUNS = {
    "0009b": ("pherc0009b", [20250919125754]),
    "0172": ("pherc0172", [20251111010954, 20251112000002]),
    "0841_0814": ("pherc0841_0814", [20260221022814, 20260226000000]),  # 0814 is tiny, so the two are one domain
    "0500p2": ("pherc0500p2", [20250628074500]),
    "1447": ("pherc1447", [20260930144758, 20260930144760]),
    "0343": ("pherc0343", [20261007000000]),  # PHerc0343 first letters
    "0139_w059": ("pherc0139", [20250223000000]),
    "0343p": ("pherc0343p", [20250511003658]),  # PHerc0343P, not 0343
    "paris2_fr143": ("phercparis2_fr143", [20230301213755]),
    "paris1_fr34": ("phercparis1_fr34", [20230301213423]),
    "0139_w035": ("pherc0139", [20260317000000]),
}
DOMAIN_RUN_NAMES = [BASELINE, *(f"{BASELINE}_add_{key}" for key in DOMAIN_RUNS)]
# mixed run: 0846A at weight 2 plus six domains at weight 1; the _shell5 duplicate uses the ring parameters close_r 0, gap_r 0, shell_r 5
MIX_DOMAINS = {
    "pherc0009b": [20250919125754],
    "pherc1447": [20260930144758, 20260930144760],
    "pherc0841": [20260221022814],
    "pherc0814": [20260226000000],
    "pherc0172": [20251111010954, 20251112000002],
    "pherc0343": [20261007000000],  # PHerc0343 first letters, not 343P
}
MIX_NAME = f"{BASELINE}_mix6"
MIX_RUNS = {MIX_NAME: None, f"{MIX_NAME}_shell5": (0, 0, 5)}  # name -> (ring_close_r, ring_gap_r, ring_shell_r)
MIX_PATCH_IDS = [sid for ids in MIX_DOMAINS.values() for sid in ids]
DOMAIN_PATCH_IDS = sorted({sid for _, ids in DOMAIN_RUNS.values() for sid in ids})
# PHerc1447 has no surface detection: surface_labels/<id> is a flat surface at the centre layer measured from the layer
# intensities (_ves_tmp/flatten_1447.py), and the stored zarr is already the fully depth-reversed stack (depth_flipped=true)
FLAT_SURFACE_IDS = [str(sid) for sid in DOMAIN_RUNS["1447"][1]]
TRAINED_SHIFTS = ("-1", "1")  # depth jitter +-1; shift 0 is data.cue_dir


def fragment_name(fragment: int) -> str:
    return f"seq_{fragment}"


def configure_fragment(config, fragment: int) -> None:
    from utils.config import ScrollConfig

    config.run_name = config.exp_name = fragment_name(fragment)
    config.model_dir = os.path.join("models", MODEL_ROOT.name, config.run_name)
    config.tra.log_dir = LOG_ROOT.name
    base._asf_cue(config)
    data = config.data
    others = [sid for sid in ALL_PATCHES if sid != fragment]
    data.scrolls = [ScrollConfig(fragment, split_axis="x", train_split_frac=1.0)]
    data.train_scroll_dict = {"pherc0846a": [fragment]}
    data.train_scroll_weights = [1]
    data.train_only_scroll_ids = [fragment]
    data.extra_hard_negative_scroll_ids = [fragment]
    data.whole_patch_valid_scroll_ids = others
    data.label_free_vis_scroll_ids = []
    data.vis_scroll_ids = others
    config.tra.eval_int_scrolls = len(others)
    config.tra.dann_n_domains = 1
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = EPOCHS
    config.tra.warmup_epochs = WARMUP_EPOCHS


def configure_domain(config, name: str, groups: dict[str, list[int]], weight_0846a: int = 1) -> None:
    """cue_dropout recipe on the 7 trained 0846A patches (weight_0846a) plus added domains (weight 1 each)."""
    ids = [sid for group in groups.values() for sid in group]
    base._base_recipe(config, old_set=True)  # whole-corpus config (scroll configs, native-noise bank), pruned below
    data = config.data
    keep = [*base.NEW_SET, *ids]
    scroll_by_id = {int(scroll.scroll_id): scroll for scroll in data.scrolls}
    missing = [sid for sid in keep if sid not in scroll_by_id]
    if missing:
        raise ValueError(f"{name}: no scroll config for {missing}")
    data.scrolls = [scroll_by_id[sid] for sid in keep]
    data.train_scroll_dict = {"pherc0846a": list(base.NEW_SET)}
    data.train_scroll_weights = [1]
    data.train_scroll_weights[0] = weight_0846a
    for domain, group in groups.items():
        data.train_scroll_dict[domain] = list(group)
        data.train_scroll_weights.append(1)
    data.train_only_scroll_ids = [sid for sid in data.train_only_scroll_ids if int(sid) in keep]
    data.equalize_domain = "pherc0846a"  # every run draws the same 0846A examples per epoch as the baseline
    # 0846A: half-brightness strokes drawn on the inklabels; added scrolls: half-brightness of their train masks (loader default)
    data.extra_hard_negative_scroll_ids = list(base.NEW_SET)
    # the added domains' patches are only rendered by the add-one runs; the mixed runs show the 0846A fragments only
    data.vis_scroll_ids = [*base.NEW_SET, base.HOLDOUT, *base.OUT_VIS, *(ids if len(groups) == 1 else [])]
    config.tra.eval_int_scrolls = len(data.vis_scroll_ids)
    config.tra.dann_n_domains = len(data.train_scroll_dict)
    config.model.input_cue = "pre_clahe"
    data.cue_dir = base.CUE_ASF_DIR
    data.cue_dropout_prob = base.CUE_DROPOUT
    data.depth_cue_dirs = {shift: base.DEPTH_CUE_DIRS[shift] for shift in TRAINED_SHIFTS}
    config.init_weights = str(base.INIT_CUE4.relative_to(REPO))
    config.tra.cue_head_prefixes = (f"gated_cue_stem.branches.{base.CUE_HEAD}.",)
    config.tra.cue_head_lr_scale = base.t10.CUE_HEAD_LR_SCALE
    config.run_name = config.exp_name = name
    config.model_dir = os.path.join("models", MODEL_ROOT.name, name)
    config.tra.log_dir = LOG_ROOT.name
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = base.EPOCHS


def configure(config, name: str, smoke: bool = False) -> None:
    if name in MIX_RUNS:
        configure_domain(config, name, MIX_DOMAINS, weight_0846a=2)
        if MIX_RUNS[name]:
            config.data.ring_close_r, config.data.ring_gap_r, config.data.ring_shell_r = MIX_RUNS[name]
    elif name in DOMAIN_RUN_NAMES:
        if name == BASELINE:
            configure_domain(config, name, {})
        else:
            domain, ids = DOMAIN_RUNS[name.removeprefix(f"{BASELINE}_add_")]
            configure_domain(config, name, {domain: ids})
    elif name.startswith("seq_"):
        configure_fragment(config, int(name.removeprefix("seq_")))
    else:
        raise ValueError(f"unknown run {name}")
    if smoke:
        data = config.data
        config.model_dir = os.path.join("models", "runs_sequence_smoke", name)
        config.tra.log_dir = "runs_sequence_smoke"
        config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = 1
        data.max_samples_per_epoch = int(os.environ.get("SEQ_SMOKE_SAMPLES", 64))
        config.dl.batch_size = int(os.environ.get("SEQ_SMOKE_BATCH", 8))
        config.dl.num_workers = 4
        data.selective_chunk_preload = False
        data.vis_preload_persistent = False
        data.ram_safe_vis = True
        added = [sid for ids in data.train_scroll_dict.values() for sid in ids if sid not in base.NEW_SET]
        data.vis_scroll_ids = [base.HOLDOUT, *added[:1]]
        config.tra.eval_int_scrolls = len(data.vis_scroll_ids)


def flat_surface_problems(sid: int) -> list[str]:
    """PHerc1447 must read the depth-reversed stack on a flat surface, with cue maps built from that surface."""
    import json

    import zarr

    problems = []
    volume = REPO / "ves_zarrs2" / f"{sid}.zarr"
    if volume.exists() and not zarr.open(str(volume), mode="r").attrs.get("depth_flipped", False):
        problems.append(f"{volume}: not the depth-reversed stack (attrs depth_flipped)")
    depth = REPO / "surface_labels" / str(sid) / "depth.npy"
    meta = depth.with_name("metadata.json")
    if not meta.exists() or "flat_depth" not in json.loads(meta.read_text()):
        problems.append(f"{meta}: no flat surface (run _ves_tmp/flatten_1447.py)")
    for directory in ("cue_asf_morph", "cue_asf_morph_dm1", "cue_asf_morph_dp1"):
        cue = REPO / directory / f"{sid}.npy"
        if cue.exists() and depth.exists() and cue.stat().st_mtime < depth.stat().st_mtime:
            problems.append(f"{cue}: older than the flat surface (rerun _ves_tmp/build_domain_cues.py)")
    return problems


def missing_inputs(name: str) -> list[str]:
    """label, train mask, volume, surface and cue files a run reads that are not there yet."""
    from utils.config import Config

    config = Config()
    configure(config, name)
    data = config.data
    trained = [int(sid) for ids in data.train_scroll_dict.values() for sid in ids]
    suffix = data.zarr_suffix or {}
    problems = []
    for sid in dict.fromkeys([*trained, *(int(s) for s in data.vis_scroll_ids)]):
        volume_id = f"{sid}{suffix.get(str(sid), '')}"
        paths = [Path(data.zarr_path) / f"{volume_id}.zarr", REPO / "masks" / f"{sid}.png"]
        if sid in trained:
            paths += [REPO / data.inklabel_dir / f"{sid}.png", REPO / data.train_mask_dir / f"{sid}.png",
                      REPO / data.surface_label_dir / str(sid) / "depth.npy"]
        if config.model.input_cue == "pre_clahe":
            dirs = [data.cue_dir, *(data.depth_cue_dirs.values() if sid in trained else [])]
            paths += [REPO / directory / f"{volume_id}.npy" for directory in dirs]
        problems += [str(path) for path in paths if not path.exists()]
        if str(sid) in FLAT_SURFACE_IDS:
            problems += flat_surface_problems(sid)
    return problems


def run_worker(name: str, dry_run: bool, smoke: bool) -> None:
    import train

    sys.argv = ["train.py", "--run-name", name, *(["--dry-run"] if dry_run else [])]
    train.main(configure=lambda config: configure(config, name, smoke))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated run names, in the given order (default: the domain queue)")
    parser.add_argument("--fragments", action="store_true", help="queue the one-fragment-alone runs (seq_<id>) instead")
    parser.add_argument("--list", action="store_true", help="print the run names and exit")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and print each config, no training")
    parser.add_argument("--smoke", action="store_true", help="one tiny epoch + one added-patch figure per run (own output dirs)")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    if args.worker:
        run_worker(args.worker, args.dry_run, args.smoke)
        return
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    if not names:
        names = [fragment_name(f) for f in FRAGMENTS] if args.fragments else list(DOMAIN_RUN_NAMES)
    if args.list:
        print("\n".join(names))
        return
    known = {*DOMAIN_RUN_NAMES, *MIX_RUNS, *(fragment_name(f) for f in FRAGMENTS)}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown runs {unknown}; choose from {sorted(known)}")
    if not base.INIT_CUE4.is_file():
        base.build_cue_init()
    problems = {n: missing_inputs(n) for n in names}
    problems = {n: p for n, p in problems.items() if p}
    if problems:
        text = "\n".join(f"  {n}: {p[:6]}{' ...' if len(p) > 6 else ''}" for n, p in problems.items())
        raise SystemExit(f"[sequence] missing inputs (build cues with _ves_tmp/build_domain_cues.py):\n{text}")
    (LOG_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    suffix = "_dry" if args.dry_run else "_smoke" if args.smoke else ""
    failed = []
    for index, name in enumerate(names, start=1):
        if not (args.dry_run or args.smoke) and (MODEL_ROOT / name / "final.pth").is_file():
            print(f"[sequence] {index}/{len(names)} {name}: done already, skipping", flush=True)
            continue
        log_path = LOG_ROOT / "logs" / f"{name}{suffix}.log"
        command = [sys.executable, "-u", __file__, "--worker", name, *(["--dry-run"] if args.dry_run else []),
                   *(["--smoke"] if args.smoke else [])]
        print(f"[sequence] {index}/{len(names)} {name} -> {log_path}", flush=True)
        started = time.time()
        with open(log_path, "a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = process.wait()
        print(f"[sequence] {name}: {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - started) / 60:.1f} min",
              flush=True)
        if code != 0:
            failed.append(name)
    if failed:
        raise SystemExit(f"[sequence] failed runs: {failed}")


if __name__ == "__main__":
    main()
