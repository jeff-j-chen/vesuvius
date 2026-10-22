"""iteration 2: a 192 px baseline on the new PHerc0846A set plus eleven single-change arms (logs ./runs_iter2).

common recipe (every arm unless stated): 192 px field (batch 32, lr 1e-4, 192 MAE init), dual_scale + nnpu_strong +
text_region, ring close 2 / gap 2 / shell 4, half-brightness regions of the new-set labels as hard negatives, PHerc0846A
domain weight 3, native-noise bank off, 10 epochs.
  new set  = ABF 20260927134931, 073246, m28 (20261012065813) and straightened2 in_1/out_1/out_2/out_3
             (aliases 20261012065817/819/821/823), trained together with PHerc0009B (weights 3:1)
  held out = straightened2 (20261012065815), out_4 (...825) and out_5 (...827): rendered, never trained or validated on
  figures  = every trained 0846A patch, the three held-out sheets, then PHerc0009B

  base                       the recipe above
  ring0_posfalse             base with ring close 0 / gap 0 / shell 5 and multitile_pos_only off
  asf_cue                    base plus the precomputed nonml asf_morph map as a fourth input head (cue_asf_morph/)
  old_plus_new               base trained on the whole default corpus plus the new set (PHerc0846A 3, PHerc0009B 2, PHerc0172 2),
                             192 px native-noise bank (models/native_noise_bank_192.npy), hard negatives from both labelling systems
  stroke_floor_strongneg     base plus the stroke floor, hard negatives drawn in 35% of the negative draws, and the stretches
                             between a patch's own text lines as certain negatives
  heavy_photometric          base with brightness x0.7..1.5, contrast +-35%, gaussian + low-frequency noise; cutout and context replace off
  idea1_visibility_weights   letters drawn in proportion to how faint they are (utils/visibility_weights.py)
  idea2_depth_mil            positives explained by the better of two depth views (+-2 slices), negatives must hold in both
  idea3_line_llp             text-line bands of each patch (own labels): the in-line unlabelled cells carry 20% ink as a population
  idea4_line_nnpu_own        the same cells under an nnPU risk with prior 0.2 (far background keeps 0.05), text-region target covers the band
  idea5_line_nnpu_cross      idea4 with bands pooled over the neighbouring windings (utils/line_grid.py cross-registers them)
  idea6_layout_prior         idea5 plus the stretches between a patch's own lines as certain negatives
  combo_asf_heavy_linepu_ring0  asf_cue + heavy_photometric + idea4 (own-band line nnPU) + ring0_posfalse, hard negatives on, new set only

  asf_cue_labelsv2           asf_cue rerun unchanged, for the revised inklabels
  depth_mil_labelsv2         idea2_depth_mil with +-1 slice pair views instead of +-2, for the revised inklabels
  no_nnpu                    asf_cue with nnPU off entirely (pu_lambda 0, no far-background unlabelled cells)
  cue_pair_mil               asf_cue + depth-MIL from the MAE init; every view reads the cue of the window it actually shows
                             (jitter +-1 and the pair view +-2: cue_asf_morph_dm2/_dm1/_dp1/_dp2)
  cue_pair_mil_strict        cue_pair_mil + a negative is charged its worse depth view + view consistency (0.2) over every target cell
  cue_dropout                asf_cue where 30% of training samples see a flat cue (128 over the papyrus): the raw slices must carry the ink
  cue_dropout_agree          cue_dropout, rendered as the cellwise minimum of the real-cue and flat-cue prediction (eval only)
  halo_penalty               asf_cue + unsupervised cells within 2 cells (32 px) of labelled ink are pushed to 0, weighted by how ink-like the cue is
  depth_peak_smooth          idea2_depth_mil + neighbouring ink-like cells must prefer the same depth view (TV of the soft view choice)
  cross_gap_negatives        idea6_layout_prior with the between-line negatives bounded by the pooled lines of every winding
  small_loss_negatives       asf_cue + the 20% highest-loss ordinary negative cells leave the loss (noisy-label small-loss selection)
  slice_masking              asf_cue + 2 consecutive depth slices replaced by the mean of the rest in half the samples (speech time masking)
  cue_dropout_labelsv3       cue_dropout rerun unchanged, for the rebuilt inklabels (v3)
  cue_dropout_labelsv3_seed17  cue_dropout_labelsv3 with every seed changed to 17 (training seed, coordinate-hash split seed, cue-head init)
  depth_peak_smooth_labelsv3 depth_peak_smooth rerun unchanged, for the v3 inklabels
  small_loss_negatives_labelsv3  asf_cue + the 20% highest-loss ordinary negative AND positive cells leave the loss, v3 inklabels

usage: python3 train_iter2.py [--arm a,b | --after ARM] [--dry-run] [--smoke] [--full-vis-smoke]
logs/figures: runs_iter2/iter2_<arm>, console log runs_iter2/logs/<arm>.log, checkpoints models/runs_iter2/iter2_<arm>.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "old"))  # train_tests10 lives in old/
import train_tests10 as t10  # noqa: E402  (recipe helpers; no heavy imports at module level)

LOG_ROOT = REPO / "runs_iter2"
MODEL_ROOT = REPO / "models" / "runs_iter2"
ABF, P073246, M28 = t10.ABF, t10.P073246, t10.STRAIGHT_M28
NEIGHBORS = t10.NEIGHBORS
NEW_SET = (ABF, P073246, M28, *NEIGHBORS)
HOLDOUT = t10.HOLDOUT_S2
OUT_VIS = t10.OUT_VIS
PHERC0009B = t10.PHERC0009B
EPOCHS = 10
WEIGHT_0846A = 3
CUE_ASF_DIR = "./cue_asf_morph"
INIT_CUE4 = MODEL_ROOT / "init_mae192_cue4.pth"
NOISE_BANK_192 = "models/native_noise_bank_192.npy"
VIS_WEIGHTS = "models/runs_iter2/visibility_weights.json"
CUE_HEAD = 3
HEAD_INIT_SCALE = 0.5
LINE_PU_LAMBDA = 1.0
LINE_PRIOR = 0.2
# cue maps for the windows depth jitter (+-1) and the depth-MIL pair view (+-2) read; shift 0 is CUE_ASF_DIR
DEPTH_CUE_DIRS = {"-2": "./cue_asf_morph_dm2", "-1": "./cue_asf_morph_dm1",
                  "1": "./cue_asf_morph_dp1", "2": "./cue_asf_morph_dp2"}
PAIR_CONSISTENCY = 0.2
CUE_DROPOUT = 0.3
HALO_LAMBDA = 0.3
HALO_RADIUS = 2
DEPTH_PEAK_LAMBDA = 1.0
NEGATIVE_FORGET_RATE = 0.2
POSITIVE_FORGET_RATE = 0.2
SLICE_MASK_PROB = 0.5
RESEED = 17
INIT_CUE4_RESEED = MODEL_ROOT / f"init_mae192_cue4_seed{RESEED}.pth"


def _weights(config, **named) -> None:
    data = config.data
    table = {name: weight for name, weight in zip(data.train_scroll_dict, data.train_scroll_weights)}
    table.update(named)
    data.train_scroll_weights = [table[name] for name in data.train_scroll_dict]


def _base_recipe(config, old_set: bool) -> None:
    from utils.config import add_0846a_train_patch

    data = config.data
    for scroll_id in (M28, *NEIGHBORS):
        add_0846a_train_patch(config, scroll_id)
    for scroll_id in (*NEW_SET, HOLDOUT, *OUT_VIS):
        data.surface_window_offset_by_scroll[scroll_id] = -1
    t10._ctx192(config)
    if not old_set:
        t10._only_domains(config, ("pherc0846a", "pherc0009b"))
        data.native_noise = ""
    else:
        data.native_noise = NOISE_BANK_192
    t10._dual_scale(config)
    t10._nnpu_strong(config)
    t10._text_region(config)
    _weights(config, pherc0846a=WEIGHT_0846A)
    data.extra_hard_negative_scroll_ids = list(NEW_SET)
    data.whole_patch_valid_scroll_ids = [HOLDOUT, *OUT_VIS]
    data.label_free_vis_scroll_ids = []
    data.vis_scroll_ids = [*NEW_SET, HOLDOUT, *OUT_VIS, PHERC0009B]
    config.tra.eval_int_scrolls = len(data.vis_scroll_ids)


def _base(config):
    _base_recipe(config, old_set=False)


def _ring0_posfalse(config):
    _base(config)
    config.data.ring_close_r = 0
    config.data.ring_gap_r = 0
    config.data.ring_shell_r = 5
    config.data.multitile_pos_only = False


def _asf_cue(config):
    _base(config)
    config.model.input_cue = "pre_clahe"
    config.data.cue_dir = CUE_ASF_DIR
    config.init_weights = str(INIT_CUE4.relative_to(REPO))
    config.tra.cue_head_prefixes = (f"gated_cue_stem.branches.{CUE_HEAD}.",)
    config.tra.cue_head_lr_scale = t10.CUE_HEAD_LR_SCALE


def _old_plus_new(config):
    _base_recipe(config, old_set=True)
    _weights(config, pherc0009b=2, pherc0172=2)


def _stroke_floor_strongneg(config):
    _base(config)
    t10._stroke_floor(config)
    config.data.explicit_negative_share = 0.35
    config.data.line_gap_negative_scroll_ids = list(NEW_SET)
    config.data.line_prior_source = "own"


def _apply_heavy_photometric(config):
    dl = config.dl
    dl.brightness_prob, dl.brightness_low, dl.brightness_delta = 0.9, 0.30, 0.50
    dl.contrast_prob, dl.contrast_delta = 0.7, 0.35
    dl.noise_prob, dl.noise_std_min, dl.noise_std_max = 0.7, 0.01, 0.04
    dl.correlated_noise_prob, dl.correlated_noise_min, dl.correlated_noise_max, dl.correlated_noise_sigma = 0.5, 0.01, 0.05, 6.0
    dl.cutout_prob = 0.0
    dl.context_replace_prob = 0.0


def _heavy_photometric(config):
    _base(config)
    _apply_heavy_photometric(config)


def _idea1_visibility_weights(config):
    _base(config)
    config.tra.character_forgetting = True
    config.tra.character_forgetting_static = True
    config.tra.character_forgetting_path = VIS_WEIGHTS
    config.tra.character_forgetting_max_weight = 4.0


def _apply_depth_mil(config, offset: int = 2):
    tra = config.tra
    tra.depth_view_consistency = True
    tra.depth_view_consistency_lambda = 0.0
    tra.depth_view_consistency_offset = offset
    tra.depth_view_consistency_prob = 0.5
    tra.depth_view_mil = True


def _idea2_depth_mil(config):
    _base(config)
    _apply_depth_mil(config)


def _depth_mil_labelsv2(config):
    _base(config)
    _apply_depth_mil(config, offset=1)


def _line_prior(config, source: str):
    _base(config)
    config.data.line_prior_scroll_ids = list(NEW_SET)
    config.data.line_prior_source = source


def _idea3_line_llp(config):
    _line_prior(config, "own")
    config.tra.line_llp_lambda = 1.0
    config.tra.line_llp_prior = LINE_PRIOR


def _idea4_line_nnpu_own(config):
    _line_prior(config, "own")
    config.tra.line_pu_lambda = LINE_PU_LAMBDA
    config.tra.line_pu_prior = LINE_PRIOR


def _idea5_line_nnpu_cross(config):
    _line_prior(config, "cross")
    config.tra.line_pu_lambda = LINE_PU_LAMBDA
    config.tra.line_pu_prior = LINE_PRIOR


def _idea6_layout_prior(config):
    _idea5_line_nnpu_cross(config)
    config.data.line_gap_negative_scroll_ids = list(NEW_SET)


def _combo_asf_heavy_linepu_ring0(config):
    _ring0_posfalse(config)
    config.model.input_cue = "pre_clahe"
    config.data.cue_dir = CUE_ASF_DIR
    config.init_weights = str(INIT_CUE4.relative_to(REPO))
    config.tra.cue_head_prefixes = (f"gated_cue_stem.branches.{CUE_HEAD}.",)
    config.tra.cue_head_lr_scale = t10.CUE_HEAD_LR_SCALE
    _apply_heavy_photometric(config)
    config.data.line_prior_scroll_ids = list(NEW_SET)
    config.data.line_prior_source = "own"
    config.tra.line_pu_lambda = LINE_PU_LAMBDA
    config.tra.line_pu_prior = LINE_PRIOR


def _cue_pair_mil(config):
    _asf_cue(config)
    _apply_depth_mil(config)
    config.data.depth_cue_dirs = dict(DEPTH_CUE_DIRS)


def _cue_pair_mil_strict(config):
    _cue_pair_mil(config)
    config.tra.depth_view_mil_negative = "max"
    config.tra.depth_view_consistency_lambda = PAIR_CONSISTENCY
    config.tra.depth_view_consistency_all_cells = True


def _no_nnpu(config):
    _asf_cue(config)
    config.tra.pu_lambda = 0.0
    config.data.far_negative_share = 0.0


def _cue_dropout(config):
    _asf_cue(config)
    config.data.cue_dropout_prob = CUE_DROPOUT


def _cue_dropout_agree(config):
    _cue_dropout(config)
    config.data.eval_cue_agree = True


def _halo_penalty(config):
    _asf_cue(config)
    config.tra.halo_penalty_lambda = HALO_LAMBDA
    config.tra.halo_radius = HALO_RADIUS


def _depth_peak_smooth(config):
    _idea2_depth_mil(config)
    config.tra.depth_peak_lambda = DEPTH_PEAK_LAMBDA


def _cross_gap_negatives(config):
    _idea6_layout_prior(config)
    config.data.line_gap_source = "cross"


def _small_loss_negatives(config):
    _asf_cue(config)
    config.tra.negative_forget_rate = NEGATIVE_FORGET_RATE
    config.tra.negative_forget_ramp_epochs = 2


def _slice_masking(config):
    _asf_cue(config)
    config.tra.slice_mask_prob = SLICE_MASK_PROB
    config.tra.slice_mask_width = 2


def _small_loss_cells(config):
    _small_loss_negatives(config)
    config.tra.positive_forget_rate = POSITIVE_FORGET_RATE


def _cue_dropout_reseeded(config):
    _cue_dropout(config)
    config.tra.seed = RESEED
    config.data.coordinate_hash_seed = RESEED
    config.init_weights = str(INIT_CUE4_RESEED.relative_to(REPO))


# run order is the user's: runs 1-6 as listed, then the six ideas; never reorder (the whole-corpus arm is run 4)
# the pair-cue arms were appended afterwards; asf_cue_labelsv2 sits just before them, depth_mil_labelsv2 and the rest follow
ARMS = {
    "base": _base,
    "ring0_posfalse": _ring0_posfalse,
    "asf_cue": _asf_cue,
    "old_plus_new": _old_plus_new,
    "stroke_floor_strongneg": _stroke_floor_strongneg,
    "heavy_photometric": _heavy_photometric,
    "idea1_visibility_weights": _idea1_visibility_weights,
    "idea2_depth_mil": _idea2_depth_mil,
    "idea3_line_llp": _idea3_line_llp,
    "idea4_line_nnpu_own": _idea4_line_nnpu_own,
    "idea5_line_nnpu_cross": _idea5_line_nnpu_cross,
    "idea6_layout_prior": _idea6_layout_prior,
    "combo_asf_heavy_linepu_ring0": _combo_asf_heavy_linepu_ring0,
    "asf_cue_labelsv2": _asf_cue,
    "cue_pair_mil": _cue_pair_mil,
    "cue_pair_mil_strict": _cue_pair_mil_strict,
    "depth_mil_labelsv2": _depth_mil_labelsv2,
    "no_nnpu": _no_nnpu,
    "cue_dropout": _cue_dropout,
    "cue_dropout_agree": _cue_dropout_agree,
    "halo_penalty": _halo_penalty,
    "depth_peak_smooth": _depth_peak_smooth,
    "cross_gap_negatives": _cross_gap_negatives,
    "small_loss_negatives": _small_loss_negatives,
    "slice_masking": _slice_masking,
    "cue_dropout_labelsv3": _cue_dropout,
    "cue_dropout_labelsv3_seed17": _cue_dropout_reseeded,
    "depth_peak_smooth_labelsv3": _depth_peak_smooth,
    "small_loss_negatives_labelsv3": _small_loss_cells,
}


def configure(config, arm: str, smoke: bool = False, full_vis: bool = False) -> None:
    name = f"iter2_{arm}"
    config.run_name = config.exp_name = name
    config.model_dir = os.path.join("models", "runs_iter2", name)
    config.tra.log_dir = LOG_ROOT.name
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = EPOCHS
    ARMS[arm](config)
    if smoke:
        _smoke(config, name, arm, full_vis)


def _smoke(config, name: str, arm: str, full_vis: bool) -> None:
    """a few windows, one epoch and one figure per arm; the whole-corpus arm keeps three extra old domains."""
    data = config.data
    config.model_dir = os.path.join("models", "runs_iter2_smoke", name)
    config.tra.log_dir = "runs_iter2_smoke"
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = 1
    data.max_samples_per_epoch = int(os.environ.get("ITER2_SMOKE_SAMPLES", 16))
    config.dl.batch_size = int(os.environ.get("ITER2_SMOKE_BATCH", 8))
    config.dl.num_workers = 4
    data.selective_chunk_preload = False
    data.vis_preload_persistent = False
    data.ram_safe_vis = True
    if not full_vis:
        data.vis_scroll_ids = [HOLDOUT]
    config.tra.eval_int_scrolls = len(data.vis_scroll_ids)
    keep = ("pherc0846a", "pherc0009b", *(("pherc0172", "pherc1667") if arm == "old_plus_new" else ()))
    weights = dict(zip(data.train_scroll_dict, data.train_scroll_weights))
    data.train_scroll_dict = {k: v for k, v in data.train_scroll_dict.items() if k in keep}
    data.train_scroll_weights = [weights[k] for k in data.train_scroll_dict]
    kept = {int(s) for ids in data.train_scroll_dict.values() for s in ids}
    data.scrolls = [s for s in data.scrolls if int(s.scroll_id) in kept]
    config.tra.dann_n_domains = len(data.train_scroll_dict)


def build_cue_init(seed: int = 41, path: Path = INIT_CUE4) -> None:
    """grow the 3-head 192 MAE to four heads for the asf_cue arm; the new head's enc1 weights start at HEAD_INIT_SCALE x the others' rms."""
    import torch

    from utils.config import Config
    from utils.cue_expand import expand_cue_state
    from utils.model import create_model

    config = Config()
    config.device = "cpu"
    configure(config, "asf_cue")
    config.model.compile_model = False
    model, _ = create_model(config)
    model_state = model.state_dict()
    state = expand_cue_state(torch.load(REPO / t10.MAE192, map_location="cpu", weights_only=True), model_state)
    bad = [k for k, v in state.items() if k in model_state and v.shape != model_state[k].shape]
    if bad:
        raise SystemExit(f"expanded checkpoint does not fit the 4-head model: {bad[:5]}")
    weight = state["enc1.net.0.weight"]
    rms = [float(weight[:, i].pow(2).mean().sqrt()) for i in range(CUE_HEAD)]
    fresh = torch.randn(weight[:, CUE_HEAD].shape, generator=torch.Generator().manual_seed(seed))
    weight[:, CUE_HEAD] = fresh * (HEAD_INIT_SCALE * sum(rms) / CUE_HEAD) / fresh.pow(2).mean().sqrt()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[iter2] 4-head init {path} (seed {seed}): head enc1 rms {HEAD_INIT_SCALE * sum(rms) / CUE_HEAD:.4f}", flush=True)


def run_worker(arm: str, dry_run: bool, smoke: bool, full_vis: bool) -> None:
    import train

    sys.argv = ["train.py", "--run-name", f"iter2_{arm}", *(["--dry-run"] if dry_run else [])]
    train.main(configure=lambda config: configure(config, arm, smoke, full_vis))


def main() -> None:
    global LOG_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default=",".join(ARMS), help="comma-separated arms, run in the given order")
    parser.add_argument("--after", default="", help="run every arm listed after this one, in order (overrides --arm)")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and print each config, no training")
    parser.add_argument("--smoke", action="store_true", help="one tiny epoch + one figure per arm (own output dirs)")
    parser.add_argument("--full-vis-smoke", action="store_true", help="with --smoke: render every figure scroll, not just one")
    parser.add_argument("--log-dir", default=LOG_ROOT.name, help="TensorBoard/figure/console-log folder under the repo")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    LOG_ROOT = REPO / args.log_dir
    if args.worker:
        run_worker(args.worker, args.dry_run, args.smoke, args.full_vis_smoke)
        return
    arms = [name.strip() for name in args.arm.split(",") if name.strip()]
    if args.after:
        if args.after not in ARMS:
            raise SystemExit(f"unknown arm {args.after!r}; choose from {list(ARMS)}")
        arms = list(ARMS)[list(ARMS).index(args.after) + 1:]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
    if "asf_cue" in arms and not INIT_CUE4.is_file():
        build_cue_init()
    if "cue_dropout_labelsv3_seed17" in arms and not INIT_CUE4_RESEED.is_file():
        build_cue_init(RESEED, INIT_CUE4_RESEED)
    if "idea1_visibility_weights" in arms and not (REPO / VIS_WEIGHTS).is_file():
        subprocess.run([sys.executable, "-m", "utils.visibility_weights", "--out", VIS_WEIGHTS], cwd=REPO, check=True)
    (LOG_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    suffix = "_dry" if args.dry_run else "_smoke" if args.smoke else ""
    failed = []
    for index, arm in enumerate(arms, start=1):
        if not (args.dry_run or args.smoke) and (MODEL_ROOT / f"iter2_{arm}" / "final.pth").is_file():
            print(f"[iter2] {index}/{len(arms)} {arm}: done already, skipping", flush=True)
            continue
        log_path = LOG_ROOT / "logs" / f"{arm}{suffix}.log"
        command = [sys.executable, "-u", __file__, "--worker", arm, "--log-dir", args.log_dir,
                   *(["--dry-run"] if args.dry_run else []), *(["--smoke"] if args.smoke else []),
                   *(["--full-vis-smoke"] if args.full_vis_smoke else [])]
        print(f"[iter2] {index}/{len(arms)} {arm} -> {log_path}", flush=True)
        started = time.time()
        with open(log_path, "a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = process.wait()
        print(f"[iter2] {arm}: {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - started) / 60:.1f} min",
              flush=True)
        if code != 0:
            failed.append(arm)
    if failed:
        raise SystemExit(f"[iter2] failed arms: {failed}")


if __name__ == "__main__":
    main()
