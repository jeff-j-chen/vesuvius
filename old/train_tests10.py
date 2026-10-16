"""ten method tests on the full multi-scroll pipeline (10 epochs each), against a matched control.

corpus of every arm = the default run (python3 train.py: AND-mask over patches, bag ranking, nnPU, patch GroupDRO, six-patch
0846A MAE init) with ONE change to the data: 20261012065811 is the only held-out 0846A patch (labels, no train region, scored
over the whole patch) and 20261012065813 trains with the other 0846A patches. only the four main 0846A patches are rendered
(at the end of epoch 10: eval_int = save_int = n_epochs = 10): 20260927134931, 20261012073246, 20261012065811, 20261012065813.
judge every arm on 20261012065811.

arms (each changes one thing relative to `base`):
  base               control
  corrob_head        fourth gated input head fed by the precomputed corroborated_s15 map (cue_corrob_s15/, nonml_ink.py priors);
                     3-head init grown to four heads with non-zero head weights, head branch at 5x lr
  noise_n2v          noise reduction: frozen blind-spot (Noise2Void) 3D denoiser, then an in-plane gaussian of sigma 1.0 px, in
                     front of the backbone. strokes are >= 20 px wide, so the removed band is noise plus fibre grain
  stroke_floor       0846A strokes: the weakest 30% of every 0846A stroke's cells must reach a mean logit of +2 (weight 1);
                     fills gaps inside strokes that the ordinary losses leave, at the cost of noise
  ring_pu            ring cells of the three trained 0846A patches are nnPU-unlabelled instead of negatives
  pu_rank            pairwise ranking of ink cells above unlabelled cells of the same scroll (weight 0.3); hidden ink does not
                     bias PU AUC
  dual_scale         independent 64 px local expert added to the score (mix 0.5)
  text_region        coarse text-region head gates the ink logits (layout prior: text vs far background)
  nnpu_strong        nnPU turned up: weight 3, far-background share 0.5, far background from 96 px away from ink (was 160)
  target_consistency label-free crops of the held-out patch and p29 must give the same cell probabilities when turned by
                     90/180/270 degrees (weight 1); uses held-out pixels, never its labels
  ctx192             192 px field at native resolution (batch 32, lr 1e-4), initialised from
                     mae_nnunet_192_campaign34_early_gated_fiber_native96_surfacenorm_campaign40_2k (a size-agnostic conv MAE)
                     (crashed: the native-noise bank patches are smaller than 192 px)
  ctx192_4scroll     ctx192 trained only on PHerc0009B and the three trained 0846A patches (default domain weights 5:1),
                     native-noise bank off; 20261012065811 stays held out
  ctx192_4scroll_dual_nnpu_region
                     ctx192_4scroll plus dual_scale, nnpu_strong and text_region (same recipe, trained from the same MAE)
  ctx192_4scroll_dual_nnpu_region_ring0_hardneg
                     that run with ring close 0 / gap 0 / shell 5, multitile_pos_only off, and the half-brightness regions of the
                     trained 0846A labels as hard negatives
  ctx96_5scroll_dual_nnpu_region_hardneg
                     96 px field (default init), PHerc0009B plus the four trained 0846A patches (adds 20261014084611, flat
                     window offset -1), dual_scale + nnpu_strong + text_region, default ring close 2 / gap 2 / shell 4,
                     half-brightness regions as hard negatives; 20261012065811 stays held out, 20261014084611 is also rendered

  ctx96_neighbors_dual_nnpu_region
                     96 px, dual_scale + nnpu_strong + text_region, default ring 2/2/4, hard negatives off; PHerc0009B plus
                     ABF, 073246, m28 and straightened2 in_1/out_1/out_2/out_3 (aliases ...817/819/821/823); the only holdout is
                     straightened2 (alias 20261012065815, blank labels), visualized last

  ctx196_neighbors_dual_nnpu_region_ringneg_off / _ringneg_on
                     the ctx96_neighbors run at a 196 px field (ctx192 recipe: batch 32, lr 1e-4, 192 px MAE init, native-noise
                     bank off), with ring_negatives off / on

  ctx196_neighbors_dual_nnpu_region_hardneg_off / _hardneg_on
                     the same 196 px recipe with default ring negatives and the half-brightness regions of the trained 0846A
                     labels as hard negatives off / on; vis also covers out_4, out_5 and PHerc0009B

logs/figures: runs_t10/t10_<arm> (one TensorBoard root), console log runs_t10/logs/<arm>.log, checkpoints models/runs_t10/t10_<arm>.

usage: python3 train_tests10.py [--arm base,corrob_head,...] [--dry-run] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG_ROOT = REPO / "runs_t10"
MODEL_ROOT = REPO / "models" / "runs_t10"
ABF, P073246, P29, STRAIGHT, STRAIGHT_M28 = 20260927134931, 20261012073246, 20261012073250, 20261012065811, 20261012065813
VIS = (ABF, P073246, STRAIGHT, STRAIGHT_M28)
TRAINED_0846A = (ABF, P073246, STRAIGHT_M28)
NEW_STRAIGHT = 20261014084611
# straightened2 neighbors in_1, out_1..out_3 (train) and the straightened2 sheet itself (sole holdout, blank labels)
NEIGHBORS = (20261012065817, 20261012065819, 20261012065821, 20261012065823)
HOLDOUT_S2 = 20261012065815
OUT_VIS = (20261012065825, 20261012065827)  # straightened2 out_4, out_5 (vis only)
PHERC0009B = 20250919125754
EPOCHS = 10
DEFAULT_INIT = "models/mae_0846a_six_flat13_m1_2k.pth"
INIT_CORROB = MODEL_ROOT / "init_corrob_head.pth"
MAE192 = "models/mae_nnunet_192_campaign34_early_gated_fiber_native96_surfacenorm_campaign40_2k.pth"
DENOISER = "models/denoiser_n2v_block3_4k.pth"
CUE_DIR = "./cue_corrob_s15"
CUE_HEAD = 3
CUE_HEAD_LR_SCALE = 5.0
HEAD_INIT_SCALE = 0.5


def _corrob_head(config):
    config.model.input_cue = "pre_clahe"
    config.data.cue_dir = CUE_DIR
    config.init_weights = str(INIT_CORROB.relative_to(REPO))
    config.tra.cue_head_prefixes = (f"gated_cue_stem.branches.{CUE_HEAD}.",)
    config.tra.cue_head_lr_scale = CUE_HEAD_LR_SCALE


def _noise_n2v(config):
    config.model.input_denoiser = DENOISER
    config.model.input_denoise_sigma = 1.0


def _stroke_floor(config):
    config.tra.stroke_floor_lambda = 1.0
    config.tra.stroke_floor_fraction = 0.3
    config.tra.stroke_floor_margin = 2.0


def _ring_pu(config):
    config.data.ring_pu_scroll_ids = list(TRAINED_0846A)


def _pu_rank(config):
    config.tra.pu_rank_lambda = 0.3


def _dual_scale(config):
    config.model.dual_scale = True
    config.model.dual_scale_mix = 0.5


def _text_region(config):
    config.model.text_region_gate = True


def _nnpu_strong(config):
    config.tra.pu_lambda = 3.0
    config.data.far_negative_share = 0.5
    config.data.far_negative_min_dist = 96


def _target_consistency(config):
    config.data.divdis_target_scroll_ids = [STRAIGHT, P29]
    config.tra.target_consistency_lambda = 1.0


def _ctx192(config):
    config.init_weights = MAE192
    config.data.context_size = 192
    config.dl.batch_size = 32
    config.tra.lr = 1e-4
    config.data.eval_infer_bs = 64
    config.data.eval_chunk_gb = 2.0
    config.data.ctx_jitter = 20
    config.dl.context_replace_margin = 14
    config.dl.context_replace_feather = 26


def _only_domains(config, names) -> None:
    data = config.data
    weights = dict(zip(data.train_scroll_dict, data.train_scroll_weights))
    data.train_scroll_dict = {k: v for k, v in data.train_scroll_dict.items() if k in names}
    data.train_scroll_weights = [weights[k] for k in data.train_scroll_dict]
    kept = {int(s) for ids in data.train_scroll_dict.values() for s in ids}
    data.scrolls = [s for s in data.scrolls if int(s.scroll_id) in kept]
    config.tra.dann_n_domains = len(data.train_scroll_dict)


def _ctx192_4scroll(config):
    _ctx192(config)
    _only_domains(config, ("pherc0846a", "pherc0009b"))
    config.data.native_noise = ""


def _ctx192_4scroll_combo(config):
    _ctx192_4scroll(config)
    _dual_scale(config)
    _nnpu_strong(config)
    _text_region(config)


def _ctx192_4scroll_combo_ring0_hardneg(config):
    _ctx192_4scroll_combo(config)
    config.data.ring_close_r = 0
    config.data.ring_gap_r = 0
    config.data.ring_shell_r = 5
    config.data.multitile_pos_only = False
    config.data.extra_hard_negative_scroll_ids = list(TRAINED_0846A)


def _ctx96_5scroll_combo_hardneg(config):
    from utils.config import add_0846a_train_patch

    add_0846a_train_patch(config, NEW_STRAIGHT)
    config.data.surface_window_offset_by_scroll[NEW_STRAIGHT] = -1
    _only_domains(config, ("pherc0846a", "pherc0009b"))
    _dual_scale(config)
    _nnpu_strong(config)
    _text_region(config)
    config.data.extra_hard_negative_scroll_ids = [*TRAINED_0846A, NEW_STRAIGHT]
    config.data.vis_scroll_ids = [*VIS, NEW_STRAIGHT]
    config.tra.eval_int_scrolls = len(config.data.vis_scroll_ids)


def _ctx192_5scroll_combo_hardneg(config):
    _ctx192(config)
    config.data.native_noise = ""
    _ctx96_5scroll_combo_hardneg(config)


def _ctx192_5scroll_combo(config):
    _ctx192_5scroll_combo_hardneg(config)
    config.data.extra_hard_negative_scroll_ids = []


def _ctx96_neighbors_combo(config):
    from utils.config import add_0846a_train_patch

    for sid in NEIGHBORS:
        add_0846a_train_patch(config, sid)
        config.data.surface_window_offset_by_scroll[sid] = -1
    config.data.surface_window_offset_by_scroll[HOLDOUT_S2] = -1
    _only_domains(config, ("pherc0846a", "pherc0009b"))
    _dual_scale(config)
    _nnpu_strong(config)
    _text_region(config)
    config.data.extra_hard_negative_scroll_ids = []
    config.data.whole_patch_valid_scroll_ids = [HOLDOUT_S2]
    config.data.vis_scroll_ids = [*TRAINED_0846A, *NEIGHBORS, HOLDOUT_S2]
    config.tra.eval_int_scrolls = len(config.data.vis_scroll_ids)


def _ctx196_neighbors(config, ring_negatives=None, hard_negatives=False):
    _ctx192(config)
    config.data.context_size = 196
    config.data.native_noise = ""
    _ctx96_neighbors_combo(config)
    if ring_negatives is not None:
        config.data.ring_negatives = ring_negatives
    if hard_negatives:
        config.data.extra_hard_negative_scroll_ids = [*TRAINED_0846A, *NEIGHBORS]
    # vis-only at the end: unlabeled out_4/out_5 (like the holdout) and PHerc0009B
    for sid in OUT_VIS:
        config.data.surface_window_offset_by_scroll[sid] = -1
    config.data.whole_patch_valid_scroll_ids = [HOLDOUT_S2, *OUT_VIS]
    config.data.vis_scroll_ids = [*config.data.vis_scroll_ids, *OUT_VIS, PHERC0009B]
    config.tra.eval_int_scrolls = len(config.data.vis_scroll_ids)


# run order: the 192 px arm is by far the slowest, so it goes last
ARMS = {
    "base": lambda config: None,
    "corrob_head": _corrob_head,
    "noise_n2v": _noise_n2v,
    "stroke_floor": _stroke_floor,
    "ring_pu": _ring_pu,
    "pu_rank": _pu_rank,
    "dual_scale": _dual_scale,
    "text_region": _text_region,
    "nnpu_strong": _nnpu_strong,
    "target_consistency": _target_consistency,
    "ctx192": _ctx192,
    "ctx192_4scroll": _ctx192_4scroll,
    "ctx192_4scroll_dual_nnpu_region": _ctx192_4scroll_combo,
    "ctx192_4scroll_dual_nnpu_region_ring0_hardneg": _ctx192_4scroll_combo_ring0_hardneg,
    "ctx96_5scroll_dual_nnpu_region_hardneg": _ctx96_5scroll_combo_hardneg,
    "ctx96_neighbors_dual_nnpu_region": _ctx96_neighbors_combo,
    "ctx196_neighbors_dual_nnpu_region_ringneg_off": lambda config: _ctx196_neighbors(config, False),
    "ctx196_neighbors_dual_nnpu_region_ringneg_on": lambda config: _ctx196_neighbors(config, True),
    "ctx196_neighbors_dual_nnpu_region_hardneg_off": lambda config: _ctx196_neighbors(config),
    "ctx196_neighbors_dual_nnpu_region_hardneg_on": lambda config: _ctx196_neighbors(config, hard_negatives=True),
    "ctx192_5scroll_dual_nnpu_region_hardneg": _ctx192_5scroll_combo_hardneg,
    "ctx192_5scroll_dual_nnpu_region": _ctx192_5scroll_combo,
}


def configure(config, arm: str, smoke: bool = False) -> None:
    from utils.config import add_0846a_train_patch

    name = f"t10_{arm}"
    add_0846a_train_patch(config, STRAIGHT_M28)
    data = config.data
    data.whole_patch_valid_scroll_ids = [STRAIGHT]
    data.label_free_vis_scroll_ids = []
    data.vis_scroll_ids = list(VIS)
    config.run_name = config.exp_name = name
    config.model_dir = os.path.join("models", "runs_t10", name)
    config.tra.log_dir = "runs_t10"
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = EPOCHS
    config.tra.eval_int_scrolls = len(VIS)
    ARMS[arm](config)
    if smoke:
        _smoke(config, name)


def _smoke(config, name: str) -> None:
    """two domains, a handful of windows, one epoch, one figure: exercises every code path of the arm."""
    data = config.data
    config.model_dir = os.path.join("models", "runs_t10_smoke", name)
    config.tra.log_dir = "runs_t10_smoke"
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = 1
    data.max_samples_per_epoch = 16
    config.dl.batch_size = 8
    config.dl.num_workers = 4
    data.selective_chunk_preload = False
    data.vis_preload_persistent = False
    data.ram_safe_vis = True
    data.vis_scroll_ids = [STRAIGHT]
    config.tra.eval_int_scrolls = 1
    weights = dict(zip(data.train_scroll_dict, data.train_scroll_weights))
    data.train_scroll_dict = {k: v for k, v in data.train_scroll_dict.items() if k in ("pherc0846a", "pherc0009b")}
    data.train_scroll_weights = [weights[k] for k in data.train_scroll_dict]
    kept = {int(s) for ids in data.train_scroll_dict.values() for s in ids}
    data.scrolls = [s for s in data.scrolls if int(s.scroll_id) in kept]
    config.tra.dann_n_domains = len(data.train_scroll_dict)


def build_corrob_init() -> None:
    """grow the default 3-head init to four heads; the new head's enc1 weights are random at HEAD_INIT_SCALE x the others' rms."""
    import torch

    from utils.config import Config
    from utils.cue_expand import expand_cue_state
    from utils.model import create_model

    config = Config()
    config.device = "cpu"
    configure(config, "corrob_head")
    config.model.compile_model = False
    model, _ = create_model(config)
    model_state = model.state_dict()
    state = expand_cue_state(torch.load(REPO / DEFAULT_INIT, map_location="cpu", weights_only=True), model_state)
    bad = [k for k, v in state.items() if k in model_state and v.shape != model_state[k].shape]
    if bad:
        raise SystemExit(f"expanded checkpoint does not fit the 4-head model: {bad[:5]}")
    weight = state["enc1.net.0.weight"]
    rms = [float(weight[:, i].pow(2).mean().sqrt()) for i in range(CUE_HEAD)]
    fresh = torch.randn(weight[:, CUE_HEAD].shape, generator=torch.Generator().manual_seed(41))
    weight[:, CUE_HEAD] = fresh * (HEAD_INIT_SCALE * sum(rms) / CUE_HEAD) / fresh.pow(2).mean().sqrt()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(state, INIT_CORROB)
    print(f"[t10] 4-head init {INIT_CORROB}: head enc1 rms {HEAD_INIT_SCALE * sum(rms) / CUE_HEAD:.4f} (others {[round(r, 4) for r in rms]})",
          flush=True)


def run_worker(arm: str, dry_run: bool, smoke: bool) -> None:
    import train

    sys.argv = ["train.py", "--run-name", f"t10_{arm}", *(["--dry-run"] if dry_run else [])]
    train.main(configure=lambda config: configure(config, arm, smoke))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default=",".join(ARMS), help="comma-separated arms, run in the given order")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and print each config, no training")
    parser.add_argument("--smoke", action="store_true", help="one tiny epoch + one figure per arm (own output dirs)")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    if args.worker:
        run_worker(args.worker, args.dry_run, args.smoke)
        return
    arms = [name.strip() for name in args.arm.split(",") if name.strip()]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
    if "corrob_head" in arms:
        if not (args.dry_run or args.smoke):
            subprocess.run([sys.executable, "-m", "utils.precomputed_corrob", "--all", "--workers", "6"], cwd=REPO, check=True)
        build_corrob_init()
    (LOG_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    suffix = "_dry" if args.dry_run else "_smoke" if args.smoke else ""
    failed = []
    for index, arm in enumerate(arms, start=1):
        if not (args.dry_run or args.smoke) and (MODEL_ROOT / f"t10_{arm}" / "final.pth").is_file():
            print(f"[t10] {index}/{len(arms)} {arm}: done already, skipping", flush=True)
            continue
        log_path = LOG_ROOT / "logs" / f"{arm}{suffix}.log"
        command = [sys.executable, "-u", __file__, "--worker", arm,
                   *(["--dry-run"] if args.dry_run else []), *(["--smoke"] if args.smoke else [])]
        print(f"[t10] {index}/{len(arms)} {arm} -> {log_path}", flush=True)
        started = time.time()
        with open(log_path, "a", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = process.wait()
        print(f"[t10] {arm}: {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - started) / 60:.1f} min",
              flush=True)
        if code != 0:
            failed.append(arm)
    if failed:
        raise SystemExit(f"[t10] failed arms: {failed}")


if __name__ == "__main__":
    main()
