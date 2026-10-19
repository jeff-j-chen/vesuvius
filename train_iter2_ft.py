"""iteration-2 fine-tunes from the old_plus_new checkpoint, on the new set (ABF, 073246, m28, straightened2 in_1/out_1-3 + PHerc0009B).

every arm starts from models/runs_iter2/iter2_old_plus_new_2/final.pth, trains 4 epochs (lr 5e-5, 1 warmup epoch), logs to
./runs_iter2 like train_iter2.py, and renders only the held-out straightened2 sheet at the end.

  ft_mil               depth-MIL (idea2: positives explained by the better of two depth views, negatives hold in both)
  ft_asf_mil           old_plus_new + the asf_morph cue as a fourth input head (new head starts at non-zero weights) + depth-MIL
  ft_stroke_strongmil  stroke floor + strong-negative sampling + strong depth-MIL (every window paired, plus a consistency term)

the next four start from new-set-only iter2 checkpoints and train SHORT_EPOCHS (3) epochs:
  ft_cue_mil           iter2_asf_cue (cue head already trained) + depth-MIL
  ft_cue_mil_heavy     ft_cue_mil + heavy photometric augmentation
  ft_mil_heavy         iter2_idea2_depth_mil + the iter2 heavy_photometric recipe
  ft_cue_mil_layout    iter2_asf_cue + depth-MIL + the layout prior (idea6: cross-winding line nnPU + gaps between lines as negatives)

usage: python3 train_iter2_ft.py [--arm a,b] [--dry-run]     (arms run in the given order)
checkpoints models/runs_iter2/iter2_<arm>, console logs runs_iter2/logs/<arm>.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# the controller must stay CUDA-free (see prewarm), and train.py reads this before it imports torch
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import train_iter2 as base  # noqa: E402

OLD_PLUS_NEW = base.MODEL_ROOT / "iter2_old_plus_new_2" / "final.pth"
INIT_CUE4 = base.MODEL_ROOT / "init_old_plus_new_cue4.pth"
ASF_CUE = base.MODEL_ROOT / "iter2_asf_cue" / "final.pth"
IDEA2_MIL = base.MODEL_ROOT / "iter2_idea2_depth_mil" / "final.pth"
EPOCHS = 4
SHORT_EPOCHS = 3
WARMUP_EPOCHS = 1
LR = 5e-5
STRONG_MIL_CONSISTENCY = 0.2


def _schedule(config, init: Path, epochs: int = EPOCHS) -> None:
    """short fine-tune schedule from `init`; only the straightened2 sheet is visualised."""
    config.init_weights = str(init.relative_to(REPO))
    config.tra.lr = LR
    config.tra.warmup_epochs = WARMUP_EPOCHS
    config.tra.n_epochs = config.tra.eval_int = config.tra.save_int = epochs
    config.data.vis_scroll_ids = [base.HOLDOUT]
    config.tra.eval_int_scrolls = 1
    # no probe or test figures: nothing but the final straightened2 evaluation figure
    config.tra.probe_int = config.tra.test_int = 10**6


def _finetune(config) -> None:
    """new-set depth-MIL recipe from old_plus_new."""
    base._idea2_depth_mil(config)
    _schedule(config, OLD_PLUS_NEW)


def _ft_mil(config) -> None:
    _finetune(config)


def _asf_head(config) -> None:
    config.model.input_cue = "pre_clahe"
    config.data.cue_dir = base.CUE_ASF_DIR
    config.init_weights = str(INIT_CUE4.relative_to(REPO))
    config.tra.cue_head_prefixes = (f"gated_cue_stem.branches.{base.CUE_HEAD}.",)
    config.tra.cue_head_lr_scale = base.t10.CUE_HEAD_LR_SCALE


def _ft_asf_mil(config) -> None:
    _finetune(config)
    _asf_head(config)


def _ft_stroke_strongmil(config) -> None:
    base._stroke_floor_strongneg(config)
    base._apply_depth_mil(config)
    _schedule(config, OLD_PLUS_NEW)
    tra = config.tra
    tra.depth_view_consistency_prob = 1.0
    tra.depth_view_consistency_lambda = STRONG_MIL_CONSISTENCY


def _trained_cue(config) -> None:
    """the iter2_asf_cue checkpoint already carries its trained asf_morph head, so no head init or extra head lr."""
    config.model.input_cue = "pre_clahe"
    config.data.cue_dir = base.CUE_ASF_DIR


def _ft_cue_mil(config) -> None:
    base._idea2_depth_mil(config)
    _trained_cue(config)
    _schedule(config, ASF_CUE, SHORT_EPOCHS)


def _ft_cue_mil_heavy(config) -> None:
    _ft_cue_mil(config)
    base._apply_heavy_photometric(config)


def _ft_mil_heavy(config) -> None:
    base._base(config)
    base._apply_heavy_photometric(config)
    _schedule(config, IDEA2_MIL, SHORT_EPOCHS)


def _ft_cue_mil_layout(config) -> None:
    base._idea6_layout_prior(config)
    base._apply_depth_mil(config)
    _trained_cue(config)
    _schedule(config, ASF_CUE, SHORT_EPOCHS)


ARMS = {
    "ft_mil": _ft_mil,
    "ft_asf_mil": _ft_asf_mil,
    "ft_stroke_strongmil": _ft_stroke_strongmil,
    "ft_cue_mil": _ft_cue_mil,
    "ft_cue_mil_heavy": _ft_cue_mil_heavy,
    "ft_mil_heavy": _ft_mil_heavy,
    "ft_cue_mil_layout": _ft_cue_mil_layout,
}


def configure(config, arm: str) -> None:
    name = f"iter2_{arm}"
    config.run_name = config.exp_name = name
    config.model_dir = os.path.join("models", "runs_iter2", name)
    config.tra.log_dir = base.LOG_ROOT.name
    ARMS[arm](config)


def build_cue_init() -> None:
    """old_plus_new grown to four input heads; the new head starts at base.HEAD_INIT_SCALE x the others' rms (not zero)."""
    import torch

    from utils.config import Config
    from utils.cue_expand import expand_cue_state
    from utils.model import create_model

    config = Config()
    config.device = "cpu"
    configure(config, "ft_asf_mil")
    config.model.compile_model = False
    model, _ = create_model(config)
    model_state = model.state_dict()
    state = expand_cue_state(torch.load(OLD_PLUS_NEW, map_location="cpu", weights_only=True), model_state)
    bad = [k for k, v in state.items() if k in model_state and v.shape != model_state[k].shape]
    if bad:
        raise SystemExit(f"expanded checkpoint does not fit the 4-head model: {bad[:5]}")
    weight = state["enc1.net.0.weight"]
    head = base.CUE_HEAD
    rms = [float(weight[:, i].pow(2).mean().sqrt()) for i in range(head)]
    fresh = torch.randn(weight[:, head].shape, generator=torch.Generator().manual_seed(41))
    weight[:, head] = fresh * (base.HEAD_INIT_SCALE * sum(rms) / head) / fresh.pow(2).mean().sqrt()
    torch.save(state, INIT_CUE4)
    print(f"[iter2-ft] 4-head init {INIT_CUE4}: head enc1 rms {base.HEAD_INIT_SCALE * sum(rms) / head:.4f}", flush=True)


def run_worker(arm: str, dry_run: bool) -> None:
    import train

    sys.argv = ["train.py", "--run-name", f"iter2_{arm}", *(["--dry-run"] if dry_run else [])]
    train.main(configure=lambda config: configure(config, arm))


class _Tee:
    """stdout/stderr that also appends to an arm's console log."""

    def __init__(self, stream, handle):
        self._stream, self._handle = stream, handle

    def write(self, text):
        self._stream.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _run_logged(arm: str, dry_run: bool, log_path: Path) -> int:
    """one arm in this (forked) process, console output also appended to its log."""
    stdout, stderr = sys.stdout, sys.stderr
    code = 0
    with open(log_path, "a", encoding="utf-8") as log:
        sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
        try:
            run_worker(arm, dry_run)
        except SystemExit as exit_request:
            code = exit_request.code if isinstance(exit_request.code, int) else int(bool(exit_request.code))
        except Exception:
            traceback.print_exc()
            code = 1
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    return code


def prewarm(arm: str) -> None:
    """the CUDA-free controller fills the prepared-dataset and chunk caches the forked arm then reuses (as campaigns did)."""
    sys.path.insert(0, str(REPO / "old"))
    import campaign_archs_29 as campaign29

    from utils.config import Config
    from utils.norm import ensure_surface_anchors

    config = Config()
    configure(config, arm)
    os.chdir(REPO)
    if config.data.norm_mode in ("surface_anchor", "surface_anchor_knee"):
        ensure_surface_anchors(
            dict.fromkeys([*config.scroll_ids(), *(config.data.vis_scroll_ids or [])]),
            config.data.zarr_path, surface_dir=config.data.surface_label_dir,
        )
    campaign29.prewarm_data_cache(config)


def run_forked(arm: str, dry_run: bool, log_path: Path) -> int:
    """campaign-style isolation: this parent never touches CUDA and keeps the data caches warm; each arm is a forked
    child that inherits them copy-on-write and takes its GPU state, workers and any OOM kill with it."""
    if not dry_run:
        prewarm(arm)
    pid = os.fork()
    if pid == 0:
        try:
            sys.stdout.reconfigure(line_buffering=True)
            code = _run_logged(arm, dry_run, log_path)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    print(f"[iter2-ft] {arm}: child killed by signal {os.WTERMSIG(status)}", flush=True)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default=",".join(ARMS), help="comma-separated arms, run in the given order")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and print each config, no training")
    parser.add_argument("--isolate", action="store_true",
                        help="one fresh subprocess per arm (reloads the data every arm); default forks from a warm controller")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    if args.worker:
        run_worker(args.worker, args.dry_run)
        return
    arms = [name.strip() for name in args.arm.split(",") if name.strip()]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
    missing = [str(path) for path in (OLD_PLUS_NEW, ASF_CUE, IDEA2_MIL) if not path.is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    if "ft_asf_mil" in arms and not INIT_CUE4.is_file():
        build_cue_init()
    log_root = base.LOG_ROOT
    (log_root / "logs").mkdir(parents=True, exist_ok=True)
    failed = []
    for index, arm in enumerate(arms, start=1):
        if not args.dry_run and (base.MODEL_ROOT / f"iter2_{arm}" / "final.pth").is_file():
            print(f"[iter2-ft] {index}/{len(arms)} {arm}: done already, skipping", flush=True)
            continue
        log_path = log_root / "logs" / f"{arm}{'_dry' if args.dry_run else ''}.log"
        print(f"[iter2-ft] {index}/{len(arms)} {arm} -> {log_path}", flush=True)
        started = time.time()
        if not args.isolate:
            code = run_forked(arm, args.dry_run, log_path)
        else:
            command = [sys.executable, "-u", __file__, "--worker", arm, *(["--dry-run"] if args.dry_run else [])]
            with open(log_path, "a", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                code = process.wait()
        print(f"[iter2-ft] {arm}: {'ok' if code == 0 else f'FAILED (exit {code})'} after {(time.time() - started) / 60:.1f} min",
              flush=True)
        if code != 0:
            failed.append(arm)
    if failed:
        raise SystemExit(f"[iter2-ft] failed arms: {failed}")


if __name__ == "__main__":
    main()
