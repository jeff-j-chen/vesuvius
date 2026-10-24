"""PHerc0175A experiments: the default train.py run plus one twist per arm, every arm with any-angle rotation.

none of the 25 PHerc0175A patches has ink labels, so every arm trains exactly the default corpus (the two labeled
PHerc0846A patches included) and tries to make that model read the unlabeled scroll more cleanly. all arms start
from the 0175A MAE (mae_pretrain_0175a.py) and add arbitrary-angle rotation about the prediction centre
(dl.arbitrary_rotation_prob): the 0175A fibers are not axis-aligned, and the 64 px / 4x4 cell targets are re-derived
on the rotated cell lattice for every sample (labels, validity and letter ids included), not rotated as a grid.

arms (the twist is the only difference to rot_base):
  rot_base         control: rotation augmentation only; 0175A is vis-only and label-free
  rot_pu_target    the 25 patches join training as blank train-only scrolls: all their papyrus is nnPU-unlabeled
                   (label -2), so the ink-vs-unlabeled risk is estimated on 0175A texture too and the noisy
                   texture-as-ink false positives get pushed down; domain weight --target-weight
  rot_consistency  label-free 0175A crops turned by 90/180/270 degrees must give the same cell probabilities
                   (squared difference, tra.target_consistency_lambda); targets the orientation-dependent speckle
  rot_divdis       DivDis: 3 heads fit the same labels but are pushed to disagree (mutual information) on 0175A
                   crops, so the head whose reading of 0175A is the sparsest and most stroke-like can be chosen
  rot_denoise_ema  frozen blind-spot (Noise2Void) input denoiser in front of the backbone and EMA weights: a cleaner
                   image in, smoother cell probabilities out
  rot_mae_anchor   L2-SP towards the 0175A MAE on the stem/early encoders plus a 10x slower encoder: the supervised
                   0846A labels cannot erase what the MAE learned about 0175A fibers
  rot_combo        pu_target + consistency + denoiser/EMA
  rot_asf          fixed asf_morph filter (alternating open/close, radii 2-12 px, utils/prefilter.py) on the max over
                   the 8 slices; its change to that map is added to every slice in front of the backbone; init is
                   the 0175A MAE continued through the same filter (mae_0175a_asf_2k)
  rot_clahe        the same with CLAHE (3x3 tiles, clip 2.5) on the max over slices; init is
                   mae_0175a_clahe_2k

every arm renders the final eval figures (tta --tta) of the two trained 0846A patches and the chosen 0175A patches.
TensorBoard and figures go to runs_0175a/<arm>, console logs to runs_0175a/logs/<arm>.log, checkpoints to
models/runs_0175a/<arm>.

usage: python3 train_0175A.py [--only rot_base,rot_pu_target] [--vis review|all|none] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG_ROOT = REPO / "runs_0175a"
MODEL_ROOT = REPO / "models" / "runs_0175a"
DEFAULT_INIT = "models/mae_0175a_plus_0846a_rot_4k.pth"
# the vis-only 0175A patches rendered at the end of training
REVIEW_IDS = (20261009171909, 20261010151349, 20261010144621, 20261010151014)
DENOISER = "models/denoiser_n2v_block3_4k.pth"
ENCODER_PREFIXES = ("enc1.", "early2d_enc", "early2d_bottleneck.", "early_depth_", "gated_cue_stem.",
                    "fiber_coordinate_input.")

ARMS = {
    "rot_base": "rotation augmentation only",
    "rot_pu_target": "0175A as blank train-only nnPU-unlabeled scrolls",
    "rot_consistency": "dihedral consistency on label-free 0175A crops",
    "rot_divdis": "3 DivDis heads disagreeing on 0175A",
    "rot_denoise_ema": "blind-spot input denoiser + EMA weights",
    "rot_mae_anchor": "L2-SP to the 0175A MAE + slow early encoder",
    "rot_combo": "pu_target + consistency + denoiser/EMA",
    "rot_asf": "asf_morph of the slice max added to every slice, filter-matched MAE",
    "rot_clahe": "CLAHE of the slice max added to every slice, filter-matched MAE",
}
# arms whose input filter needs an MAE continued through that same filter (mae_pretrain_0175a.py --prefilter)
ARM_INIT = {"rot_asf": "models/mae_0175a_asf_2k.pth", "rot_clahe": "models/mae_0175a_clahe_2k.pth"}


def twists(arm: str) -> set[str]:
    return {
        "rot_pu_target": {"pu"}, "rot_consistency": {"consistency"}, "rot_divdis": {"divdis"},
        "rot_denoise_ema": {"denoise"}, "rot_mae_anchor": {"anchor"},
        "rot_combo": {"pu", "consistency", "denoise"}, "rot_asf": {"asf"}, "rot_clahe": {"clahe"},
    }.get(arm, set())


def configure(config, arm: str, opts) -> None:
    """the default run config with the 0175A patches, rotation augmentation and the arm's twist."""
    from utils import pherc0175a
    from utils.config import RUN_0846A_IDS, RUN_DOMAIN_WEIGHTS, RUN_LABEL_FREE_VIS_IDS, RUN_P29_ID, ScrollConfig

    ids = pherc0175a.alias_ids()
    vis = {"review": [i for i in REVIEW_IDS if i in ids], "all": ids, "none": []}[opts.vis]
    wanted = twists(arm)
    config.run_name = config.exp_name = arm
    config.model_dir = os.path.join("models", "runs_0175a", arm)
    config.tra.log_dir = "runs_0175a"
    config.init_weights = opts.init if opts.init != DEFAULT_INIT else ARM_INIT.get(arm, DEFAULT_INIT)
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = opts.epochs
    config.data.tta_mode = opts.tta
    config.dl.arbitrary_rotation_prob = opts.rotation
    config.dl.arbitrary_rotation_max_deg = 180.0

    # p29 and the two label-free 0846A sheets are not part of these runs
    dropped = {RUN_P29_ID, *RUN_LABEL_FREE_VIS_IDS}
    config.data.whole_patch_valid_scroll_ids = [s for s in config.data.whole_patch_valid_scroll_ids if int(s) not in dropped]
    config.data.surface_window_offset_by_scroll = {
        **{s: o for s, o in config.data.surface_window_offset_by_scroll.items() if int(s) not in dropped},
        **{i: pherc0175a.WINDOW_OFFSET for i in ids},
    }
    config.data.vis_scroll_ids = [*(int(s) for s in RUN_0846A_IDS), *vis]
    config.tra.eval_int_scrolls = len(config.data.vis_scroll_ids)
    config.data.label_free_vis_scroll_ids = [] if "pu" in wanted else list(vis)

    if "pu" in wanted:
        config.data.scrolls = [*config.data.scrolls, *(ScrollConfig(i, split_axis="x", train_split_frac=1.0) for i in ids)]
        config.data.train_scroll_dict = {**config.data.train_scroll_dict, "pherc0175a": list(ids)}
        weights = {**RUN_DOMAIN_WEIGHTS, "pherc0175a": opts.target_weight}
        config.data.train_scroll_weights = [weights.get(name, 1) for name in config.data.train_scroll_dict]
        config.data.train_only_scroll_ids = [*config.data.train_only_scroll_ids, *ids]
        config.data.blank_scroll_samples_per_epoch = opts.blank_samples
    if wanted & {"consistency", "divdis"}:
        config.data.divdis_target_scroll_ids = list(ids)
        config.tra.divdis_batch_size = 24
    if "consistency" in wanted:
        config.tra.target_consistency_lambda = opts.consistency
    if "divdis" in wanted:
        config.model.divdis_heads = 3
        config.tra.divdis_target_prior = 0.05
        # DivDis steps are plain full-batch steps; domain-group AND-masking is incompatible with it
        config.tra.and_mask = False
    if "denoise" in wanted:
        config.model.input_denoiser = DENOISER
        config.tra.model_ema = True
        config.tra.model_ema_decay = 0.995
    if "asf" in wanted or "clahe" in wanted:
        config.model.input_prefilter = "asf" if "asf" in wanted else "clahe"
        config.model.input_prefilter_mix = 1.0
    if "anchor" in wanted:
        config.tra.mae_anchor_lambda = opts.anchor
        config.tra.encoder_prefixes = ENCODER_PREFIXES
        config.tra.encoder_lr_scale = 0.1


def run_worker(arm: str, opts) -> None:
    import train

    sys.argv = ["train.py", "--run-name", arm, *(["--dry-run"] if opts.dry_run else [])]
    train.main(configure=lambda config: configure(config, arm, opts))


def add_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--init", default=DEFAULT_INIT, help="MAE (or other) init weights")
    parser.add_argument("--vis", choices=("review", "all", "none"), default="review",
                        help="0175A patches rendered at the end of every arm (review = %d patches)" % len(REVIEW_IDS))
    parser.add_argument("--tta", choices=("none", "light", "flips", "dihedral"), default="none")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--rotation", type=float, default=0.8, help="probability of an any-angle rotation per sample")
    parser.add_argument("--target-weight", type=int, default=3, help="rot_pu_target/rot_combo domain weight (default run: 28 total)")
    parser.add_argument("--blank-samples", type=int, default=400, help="windows per epoch drawn from each blank 0175A patch")
    parser.add_argument("--consistency", type=float, default=2.0)
    parser.add_argument("--anchor", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="preflight every arm and print its config, no training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated arms (default: all)")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    add_options(parser)
    opts = parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from utils import pherc0175a

    pherc0175a.prepare()
    if opts.worker:
        run_worker(opts.worker, opts)
        return

    arms = [name.strip() for name in opts.only.split(",") if name.strip()] or list(ARMS)
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
    (LOG_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    passthrough = [
        "--init", opts.init, "--vis", opts.vis, "--tta", opts.tta, "--epochs", str(opts.epochs),
        "--rotation", str(opts.rotation), "--target-weight", str(opts.target_weight),
        "--blank-samples", str(opts.blank_samples), "--consistency", str(opts.consistency),
        "--anchor", str(opts.anchor), *(["--dry-run"] if opts.dry_run else []),
    ]
    failed = []
    for index, arm in enumerate(arms, start=1):
        if (MODEL_ROOT / arm / "final.pth").is_file() and not opts.dry_run:
            print(f"[0175A] {index}/{len(arms)} {arm}: done already, skipping", flush=True)
            continue
        log_path = LOG_ROOT / "logs" / f"{arm}{'_dry' if opts.dry_run else ''}.log"
        command = [sys.executable, "-u", __file__, "--worker", arm, *passthrough]
        print(f"[0175A] {index}/{len(arms)} {arm} ({ARMS[arm]}) -> {log_path}", flush=True)
        started = time.time()
        # each arm in its own process, so RAM caches and CUDA state never carry over between runs
        with open(log_path, "a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = process.wait()
        print(f"[0175A] {arm}: {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - started) / 60:.1f} min",
              flush=True)
        if code != 0:
            failed.append(arm)
    if failed:
        raise SystemExit(f"[0175A] failed arms: {failed}")
    print("[0175A] all arms finished", flush=True)


if __name__ == "__main__":
    main()
