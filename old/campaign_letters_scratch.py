"""From-MAE supervised AND-mask variants, judged by eye on two vis-only PHerc0846A sheets.

Every arm trains campaign 41 `and_mask` from an MAE initialization (no supervised checkpoint, no
fine-tune) on the full campaign 40/41 corpus (every campaign-40 scroll incl. PHerc0009B, plus campaign 41's
PHerc1447 x2 and p343_letters: 27 patches, 13 domains) and the three labeled PHerc0846A patches (domain
weight 5, train-only, flat slices 9-16). Two label-free sheets are never trained, only rendered after the last
epoch: 20261012065811 (flat-13 alias of the straightened re-trace 20261012065810) and 20261012065813 (flat-13
alias of that mesh moved -28 voxels along the normal, 20261012065812). They carry no labels, so arms are compared
by the copied figures in output/letters_scratch_v1/.

| arm                   | change vs baseline                                                         |
|-----------------------|----------------------------------------------------------------------------|
| baseline              | none                                                                       |
| ring_pu_0846a         | ring cells around 0846A letters are nnPU-unlabelled, not negatives: faint  |
|                       | unlabelled letters beside visible ones stop being taught as papyrus        |
| unlabeled_0846a_pu    | four unlabeled PHerc0846A patches (>= 1 mm from the holdout) train as a    |
|                       | weight-3 domain whose cells are all nnPU-unlabelled                        |
| pu_rank               | pairwise ranking of ink cells above same-domain nnPU-unlabelled cells     |
|                       | (lambda 0.3); PU AUC is affine in PN AUC, so hidden ink does not bias it   |
| and_mask_patches      | AND-mask groups are sets of segments instead of physical domains           |
| nnpu_strict           | nnPU prior 0.05 (from 0.2) and far-background share 0.30 (from 0.15)       |
| conflict_weighted     | domain-gradient conflict weighting (cosine 0.8, strength 8) replaces patch |
|                       | GroupDRO                                                                   |
| bag_rank_strong       | bag ranking lambda 0.6, margin 1.0, top 25% (from 0.2 / 0.5 / 50%)         |
| analog_heavy          | domain weights PHerc0846A 15, PHerc0009B 5 (visible + hidden ink analogs)  |
| native_regime_corpus  | the eight resampled 2.4 / 3.24 um fragments are dropped                    |
| arch_native192_deep2d | (arch) native 192 px field and a fifth 2D U-Net level; own MAE (corpus,    |
|                       | then the six-patch PHerc0846A continuation), batch 32 / lr 1e-4            |
| flat_low_scrolls      | eight low-performing low-res patches use a fixed flat window (as 0846A)   |
|                       | instead of their detected surface; depths from old/surface_window_audit.py |
| combo_patch_bag_strict| and_mask_patches + bag_rank_strong + nnpu_strict together                  |

flat_low_scrolls and combo_patch_bag_strict train 8 epochs.

Arms whose patch count differs from the baseline scale max_samples_per_epoch so an epoch keeps about
the same number of optimizer steps. Arms are independent: a failed arm is recorded and the queue moves on.

Usage:
    python3 campaign_letters_scratch.py --dry-run
    python3 campaign_letters_scratch.py --smoke [--only baseline,ring_pu_0846a]
    python3 campaign_letters_scratch.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import campaign_archs_34 as campaign34
import campaign_archs_40 as campaign40
import campaign_archs_41 as campaign41
import campaign_letters_overnight as overnight
import mae_pretrain_letters as mae
from utils.config import ScrollConfig

ROOT = Path(__file__).resolve().parent
RUN = "letters_scratch_v1"
INPUT_ROOT = ROOT / "_ves_tmp" / RUN
MODEL_ROOT = ROOT / "models" / RUN
DEFAULT_LOG_ROOT = ROOT / "runs_letters_transfer_probes_v2"
OUTPUT_ROOT = ROOT / "output" / RUN
MAE_WEIGHTS = mae.ROOT / "models" / f"{mae.NAME}.pth"
TRAIN_0846A = (20260927134931, 20261012073246, 20261012073250)
PHERC0009B_ID = campaign41.PHERC0009B_ID
PHERC0009B_SCROLL = next(s for s in campaign40.SCROLLS_BASE if int(s.scroll_id) == PHERC0009B_ID)
# flat-13 aliases of unlabeled PHerc0846A meshes, all >= 80 voxels from the holdout mesh
UNLABELED_0846A = (20261012081238, 20261002131432, 20260927200859, 20261002075023)
FINE_SCAN_DOMAINS = ("pherc1667", "phercparis4", "phercparis2_fr143", "pherc51cr4_fr8", "phercparis1_fr34")
N_EPOCHS = 12
BASE_PATCHES = len(campaign41.SCROLLS) + len(TRAIN_0846A) + 1
MAE192_BASE = "mae_letters_scratch_n192x1_2k"
MAE192_SIX = "mae_letters_scratch_n192x1_0846a_six_2k"
ARCH_ARMS = ("arch_native192_deep2d",)
ARMS = (
    "baseline", "ring_pu_0846a", "unlabeled_0846a_pu", "pu_rank", "and_mask_patches", "nnpu_strict",
    "conflict_weighted", "bag_rank_strong", "analog_heavy", "native_regime_corpus", *ARCH_ARMS,
    "flat_low_scrolls", "combo_patch_bag_strict",
)
# flat-13 aliases of the straightened 0846A re-trace and of its -28 normal offset; never trained, unlabelled
STRAIGHT_HOLDOUT = 20261012065811
STRAIGHT_M28 = 20261012065813
EXTRA_VIS = (STRAIGHT_HOLDOUT, STRAIGHT_M28)
BOTH_HOLDOUT_ARMS = ("flat_low_scrolls", "combo_patch_bag_strict")
BOTH_HOLDOUT_EPOCHS = 8
# flat surface depth per patch (window depth-3 .. depth+4): the central depth 9-16 with the largest ink-vs-
# nearby-papyrus contrast, used only where it beats the detected-surface window by >= 15%
# (CPU audit _ves_tmp/surface_window_audit.json; detected |d| -> flat |d|)
FLAT_DEPTHS = {
    20260930144758: 16,  # PHerc1447 w058, reversed normal: 0.70 -> 0.91
    20260930144760: 9,   # PHerc1447 w060, reversed normal, ink on the opposite face to w058: 0.86 -> 1.39
    20250628074500: 16,  # PHerc0500P2 front: 0.23 -> 0.54
    20261007000000: 12,  # PHerc0343P p343_letters: 0.85 -> 1.17
    20260115000000: 10,  # PHerc0139 w044: 0.73 -> 1.09
    20250108000005: 13,  # PHerc0139 w030: 0.56 -> 0.73
    20250831000000: 10,  # PHerc0139 w040: 0.54 -> 0.76
    20260108000000: 9,   # PHerc0139 w041: 0.37 -> 0.66
}
FLAT_SURFACE_DIR = INPUT_ROOT / "surface_labels_flat"


def log_root():
    settings = INPUT_ROOT / "run_settings.json"
    if settings.is_file():
        return Path(json.loads(settings.read_text())["log_dir"])
    return DEFAULT_LOG_ROOT


def run_name(tid):
    return f"scratch_{tid}"


def mae_paths(smoke=False):
    suffix = "_smoke" if smoke else ""
    return ROOT / "models" / f"{MAE192_BASE}{suffix}.pth", ROOT / "models" / f"{MAE192_SIX}{suffix}.pth"


def _match_epoch(config):
    """keep about the baseline's optimizer steps per epoch when the patch count changes."""
    config.data.max_samples_per_epoch = int(round(6667 * BASE_PATCHES / len(config.data.scrolls)))


def build_config(tid, smoke=False):
    if tid not in ARMS:
        raise ValueError(f"Unknown arm: {tid}")
    config = campaign41.build_config(campaign41._test("and_mask", campaign41.AND_MASK))
    config.init_weights = str(MAE_WEIGHTS)
    root = MODEL_ROOT.with_name(RUN + "_smoke") if smoke else MODEL_ROOT
    config.model_dir = str(root / tid)
    config.save_final = str(root / tid / "final.pth")
    config.exp_name = config.run_name = run_name(tid)
    config.tra.log_dir = str(ROOT / f"runs_{RUN}_smoke" if smoke else log_root())
    config.data.inklabel_dir = str(INPUT_ROOT / "inklabels")
    config.data.train_mask_dir = str(INPUT_ROOT / "train_masks")
    config.data.scrolls = [
        *config.data.scrolls,
        *(ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in TRAIN_0846A),
        PHERC0009B_SCROLL,
    ]
    config.data.train_scroll_dict = {
        **config.data.train_scroll_dict, "pherc0009b": [PHERC0009B_ID], "pherc0846a": list(TRAIN_0846A),
    }
    config.data.train_scroll_weights = [*config.data.train_scroll_weights, 1, 5]
    config.data.train_only_scroll_ids = list(TRAIN_0846A)
    config.data.holdout_domains = []
    config.data.validation_scroll_ids = None
    config.data.surface_window_offset_by_scroll = {
        sid: -1 for sid in (*TRAIN_0846A, *UNLABELED_0846A, *EXTRA_VIS)
    }
    config.data.vis_scroll_ids = list(EXTRA_VIS)
    config.data.label_free_vis_scroll_ids = list(EXTRA_VIS)
    config.tra.n_epochs = N_EPOCHS
    if tid in BOTH_HOLDOUT_ARMS:
        config.tra.n_epochs = BOTH_HOLDOUT_EPOCHS
    config.data.vis_preload_persistent = True
    config.tra.eval_int = config.tra.n_epochs
    config.tra.eval_int_scrolls = len(config.data.vis_scroll_ids)
    config.tra.test_int = config.tra.probe_int = 999
    config.tra.disable_figures = False
    weights = dict(zip(config.data.train_scroll_dict, config.data.train_scroll_weights))

    if tid == "ring_pu_0846a":
        config.data.ring_pu_scroll_ids = list(TRAIN_0846A)
    elif tid == "unlabeled_0846a_pu":
        config.data.scrolls += [ScrollConfig(sid, split_axis="x", train_split_frac=1.0) for sid in UNLABELED_0846A]
        config.data.train_scroll_dict["pherc0846a_unlabeled"] = list(UNLABELED_0846A)
        weights["pherc0846a_unlabeled"] = 3
        config.data.train_only_scroll_ids += list(UNLABELED_0846A)
        _match_epoch(config)
    elif tid == "pu_rank":
        config.tra.pu_rank_lambda = 0.3
    elif tid == "and_mask_patches":
        config.tra.and_mask_unit = "patch"
    elif tid == "nnpu_strict":
        config.tra.pu_prior = 0.05
        config.data.far_negative_share = 0.30
    elif tid == "conflict_weighted":
        config.tra.physical_patch_groupdro = False
        config.tra.domain_gradient_mode = "conflict_weighted"
        config.tra.domain_gradient_threshold = 0.8
        config.tra.domain_gradient_strength = 8.0
    elif tid == "bag_rank_strong":
        config.tra.character_bag_lambda = 0.6
        config.tra.character_bag_margin = 1.0
        config.tra.character_bag_topk_frac = 0.25
    elif tid == "combo_patch_bag_strict":
        config.tra.and_mask_unit = "patch"
        config.tra.character_bag_lambda = 0.6
        config.tra.character_bag_margin = 1.0
        config.tra.character_bag_topk_frac = 0.25
        config.tra.pu_prior = 0.05
        config.data.far_negative_share = 0.30
    elif tid == "analog_heavy":
        weights.update({"pherc0846a": 15, "pherc0009b": 5})
    elif tid == "native_regime_corpus":
        dropped = {sid for domain in FINE_SCAN_DOMAINS for sid in config.data.train_scroll_dict.pop(domain)}
        config.data.scrolls = [scroll for scroll in config.data.scrolls if int(scroll.scroll_id) not in dropped]
        _match_epoch(config)
    elif tid == "arch_native192_deep2d":
        config.init_weights = str(mae_paths(smoke)[1])
        config.data.context_size = 192
        config.model.two_d_extra_levels = 1
        config.dl.batch_size = 32
        config.tra.lr = 1e-4
        config.data.eval_infer_bs = 64
        config.data.eval_chunk_gb = 2.0
        config.data.ctx_jitter = 20
        config.dl.context_replace_margin = 14
        config.dl.context_replace_feather = 26
    elif tid == "flat_low_scrolls":
        config.data.surface_label_dir = str(FLAT_SURFACE_DIR)
    config.data.train_scroll_weights = [weights[domain] for domain in config.data.train_scroll_dict]
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    trained = {int(scroll.scroll_id) for scroll in config.data.scrolls}
    held_out = {*EXTRA_VIS, 20261012065810, 20261012065812}
    if held_out & (trained | {sid for ids in config.data.train_scroll_dict.values() for sid in ids}):
        raise ValueError("the holdout patches must never be trained")
    if trained != {int(sid) for ids in config.data.train_scroll_dict.values() for sid in ids}:
        raise ValueError(f"{tid}: scroll list and domain dictionary disagree")
    return config


def _snapshot_ids():
    return sorted({*campaign41.training_scroll_ids(), *TRAIN_0846A, PHERC0009B_ID, *UNLABELED_0846A})


def prepare_inputs():
    import cv2
    import numpy as np

    manifest = INPUT_ROOT / "inputs.json"
    if manifest.is_file():
        return verify_snapshot()
    for sid in UNLABELED_0846A:
        if not (ROOT / "masks" / f"{sid}.png").is_file():
            raise FileNotFoundError(f"unlabeled alias {sid} is not prepared")
    files, counts = {}, {}
    for folder in ("inklabels", "train_masks"):
        (INPUT_ROOT / folder).mkdir(parents=True, exist_ok=True)
    for sid in _snapshot_ids():
        label_path = INPUT_ROOT / "inklabels" / f"{sid}.png"
        split_path = INPUT_ROOT / "train_masks" / f"{sid}.png"
        mask = cv2.imread(str(ROOT / "masks" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"masks/{sid}.png")
        if sid in UNLABELED_0846A:
            if not (cv2.imwrite(str(label_path), np.zeros_like(mask))
                    and cv2.imwrite(str(split_path), np.full_like(mask, 255))):
                raise OSError(f"could not write blank inputs for {sid}")
        else:
            shutil.copyfile(ROOT / campaign41.INKLABEL_DIR / f"{sid}.png", label_path)
            shutil.copyfile(ROOT / "train_masks" / f"{sid}.png", split_path)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        counts[str(sid)] = int((label >= 192).sum())
        for path in (label_path, split_path):
            files[str(path.relative_to(INPUT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(counts[str(sid)] == 0 for sid in TRAIN_0846A):
        raise ValueError("0846A label snapshot is missing letters")
    overnight._atomic_json(manifest, {
        "run": RUN, "arms": list(ARMS), "files": files, "ink_pixels": counts,
        "label_source": str(ROOT / campaign41.INKLABEL_DIR),
        "note": "unlabeled 0846A patches are blank",
    })
    return verify_snapshot()


def verify_snapshot():
    record = json.loads((INPUT_ROOT / "inputs.json").read_text())
    for relative, expected in record["files"].items():
        if hashlib.sha256((INPUT_ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"snapshot changed: {relative}")
    return record


def _mae_complete(path):
    marker = path.with_suffix(".complete.json")
    return path.is_file() and marker.is_file() and json.loads(marker.read_text()).get("complete")


def _verify_mae(path, reference=None):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not any(key.startswith("early2d_extra_encoders.") for key in state):
        raise ValueError(f"{path} lacks the fifth 2D level")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"non-finite MAE tensors: {path}")
    if reference is not None:
        initial = torch.load(reference, map_location="cpu", weights_only=True)
        if set(state) != set(initial) or any(value.shape != initial[key].shape for key, value in state.items()):
            raise ValueError("continued MAE keys differ from the corpus MAE")


def run_mae(smoke=False):
    """corpus MAE for the 192 px deep-2D arm, then the six-patch PHerc0846A continuation."""
    base_path, six_path = mae_paths(smoke)
    steps = "3" if smoke else "2000"
    arch_args = [*campaign34.PRETRAIN_SPECS[campaign40.SURFACE_FIBER_MAE_KEY]["args"], "--two-d-extra-levels", "1"]
    if not _mae_complete(base_path):
        scroll_ids = campaign34.PRETRAIN_SPECS[campaign40.SURFACE_FIBER_MAE_KEY]["scroll_ids"]
        command = [
            sys.executable, "-u", str(ROOT / "mae_pretrain_nnunet.py"), "--name", base_path.stem,
            "--scroll-ids", *map(str, scroll_ids), "--require-all-scrolls", "--physical-round-robin",
            "--ctx", "192", "--ds", "1", "--depth", "8", "--d-start", "10", "--d-end", "18",
            "--steps", steps, "--batch-size", "32", "--accum-steps", "1", "--lr", "3e-4", "--seed", "41",
            "--log-dir", str(ROOT / "runs_mae_letters"), "--no-figures", *arch_args,
        ]
        print(f"[scratch] MAE corpus pretraining: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        _verify_mae(base_path)
        overnight._atomic_json(base_path.with_suffix(".complete.json"), {
            "complete": True, "command": command, "scroll_ids": list(map(int, scroll_ids)),
            "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        })
    if not _mae_complete(six_path):
        stage, sources = mae.prepare_inputs(six_path.stem)
        staged = stage / "models" / f"{six_path.stem}.pth"
        if staged.exists():
            staged.unlink()
        command = [
            sys.executable, "-u", str(ROOT / "mae_pretrain_nnunet.py"), "--name", six_path.stem,
            "--init-weights", str(base_path), *arch_args,
            "--ctx", "192", "--ds", "1", "--depth", "8", "--d-start", "9", "--d-end", "17",
            "--scroll-ids", *(str(patch[0]) for patch in mae.PATCHES), "--require-all-scrolls",
            "--steps", steps, "--batch-size", "24", "--lr", "1e-4", "--seed", "41",
            "--log-dir", str(ROOT / "runs_mae_letters"), "--log-int", "50", "--save-int", "500", "--no-figures",
        ]
        environment = {**os.environ, "VESUVIUS_ZARR_PATH": str(stage / "volumes")}
        print(f"[scratch] MAE six-patch continuation: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=stage, env=environment, check=True)
        _verify_mae(staged, base_path)
        temporary = six_path.with_suffix(".tmp")
        shutil.copyfile(staged, temporary)
        temporary.replace(six_path)
        overnight._atomic_json(six_path.with_suffix(".complete.json"), {
            "complete": True, "command": command, "init": str(base_path), "sources": sources,
            "sha256": hashlib.sha256(six_path.read_bytes()).hexdigest(),
        })


SMOKE_DOMAINS = ("pherc0814", "pherc0172", "pherc0500p2", "pherc0009b", "pherc0846a", "pherc0846a_unlabeled")


def _smoke(config, full=False):
    config.tra.n_epochs = config.tra.eval_int = 1
    config.data.max_samples_per_epoch = 4
    config.dl.num_workers = 0
    config.dl.batch_size = min(config.dl.batch_size, 16)
    config.data.selective_chunk_preload = False
    config.data.vis_preload_persistent = False
    config.data.ram_safe_vis = True
    if not full:
        weights = dict(zip(config.data.train_scroll_dict, config.data.train_scroll_weights))
        config.data.train_scroll_dict = {
            domain: ids for domain, ids in config.data.train_scroll_dict.items() if domain in SMOKE_DOMAINS
        }
        config.data.train_scroll_weights = [weights[domain] for domain in config.data.train_scroll_dict]
        kept = {int(sid) for ids in config.data.train_scroll_dict.values() for sid in ids}
        config.data.scrolls = [scroll for scroll in config.data.scrolls if int(scroll.scroll_id) in kept]
        config.tra.dann_n_domains = len(config.data.train_scroll_dict)


def run_worker(tid, smoke=False, retry_incomplete=False, smoke_full=False):
    verify_snapshot()
    if tid in ARCH_ARMS:
        run_mae(smoke)
    config = build_config(tid, smoke)
    if smoke:
        _smoke(config, smoke_full)
    overnight.run_config(config, tid, smoke=smoke, allow_mae=True, retry_incomplete=retry_incomplete)
    if not smoke:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        epochs = config.tra.n_epochs
        figures = Path(config.tra.log_dir) / config.run_name / "eval_figs"
        for sid, suffix in ((STRAIGHT_HOLDOUT, "straight"), (STRAIGHT_M28, "straight_m28")):
            shutil.copyfile(figures / f"eval_s{sid}_ep{epochs:02d}.png", OUTPUT_ROOT / f"{tid}_{suffix}.png")


def ensure_flat_surfaces():
    """private surface-label dir: originals linked, FLAT_DEPTHS patches replaced by flat maps."""
    import cv2
    import numpy as np

    FLAT_SURFACE_DIR.mkdir(parents=True, exist_ok=True)
    for sid in (*_snapshot_ids(), *EXTRA_VIS):
        target = FLAT_SURFACE_DIR / str(sid)
        if sid not in FLAT_DEPTHS:
            if not target.is_symlink():
                target.symlink_to((ROOT / "surface_labels" / str(sid)).resolve())
            continue
        depth = np.load(ROOT / "surface_labels" / str(sid) / "depth.npy", mmap_mode="r")
        mask = cv2.imread(str(ROOT / "masks" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        valid = np.zeros(depth.shape, dtype=bool)
        h, w = min(mask.shape[0], depth.shape[0]), min(mask.shape[1], depth.shape[1])
        valid[:h, :w] = mask[:h, :w] > 0
        flat = np.where(valid, FLAT_DEPTHS[sid], 255).astype(np.uint8)
        if (target / "depth.npy").is_file() and np.array_equal(np.load(target / "depth.npy"), flat):
            continue
        target.mkdir(exist_ok=True)
        np.save(target / "depth.npy", flat)
        np.save(target / "confidence.npy", np.where(valid, 255, 0).astype(np.uint8))
        (target / "metadata.json").write_text(json.dumps(
            {"scroll_id": sid, "flat_depth": FLAT_DEPTHS[sid], "source": "old/surface_window_audit.py"}) + "\n")


def preflight(selected, smoke=False):
    prepare_inputs()
    if "flat_low_scrolls" in selected:
        ensure_flat_surfaces()
    if not MAE_WEIGHTS.is_file():
        raise FileNotFoundError(MAE_WEIGHTS)
    for tid in selected:
        config = build_config(tid, smoke)
        overnight.check_inputs(config)
        if tid not in ARCH_ARMS:
            overnight.checkpoint_check(MAE_WEIGHTS, config, allow_mae=True)
    if shutil.disk_usage(ROOT).free < 20 * 1024 ** 3:
        raise RuntimeError("need at least 20 GiB free")
    print(f"[scratch] preflight passed: {len(selected)} arm(s), vis-only {list(EXTRA_VIS)}", flush=True)


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="one tiny epoch + the vis-only figures per arm")
    parser.add_argument("--smoke-full", action="store_true", help="smoke on the whole corpus, not a domain subset")
    parser.add_argument("--only", help="comma-separated arm subset")
    parser.add_argument("--log-dir", help="TensorBoard root (remembered); default runs_letters_transfer_probes_v2")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        run_worker(args.worker, args.smoke, args.retry_incomplete, args.smoke_full)
        return
    selected = list(ARMS)
    if args.only:
        wanted = set(args.only.split(","))
        if wanted - set(ARMS):
            parser.error(f"unknown arms: {sorted(wanted - set(ARMS))}")
        selected = [tid for tid in ARMS if tid in wanted]
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (INPUT_ROOT / ("smoke.lock" if args.smoke else "runner.lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.log_dir:
            overnight._atomic_json(INPUT_ROOT / "run_settings.json", {"log_dir": str(Path(args.log_dir).resolve())})
        preflight(selected, args.smoke)
        if args.dry_run:
            print("[scratch] QUEUE", selected, flush=True)
            return
        status_path = MODEL_ROOT.with_name(RUN + "_smoke" if args.smoke else RUN) / "status.json"
        previous = json.loads(status_path.read_text()).get("results", {}) if status_path.is_file() else {}
        earlier = {tid: state for tid, state in previous.items() if tid not in selected}
        results = {}
        for tid in selected:
            overnight._atomic_json(status_path, {"state": "running", "stage": tid, "results": {**earlier, **results}})
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", tid]
            command += ["--smoke"] if args.smoke else []
            command += ["--smoke-full"] if args.smoke_full else []
            command += ["--retry-incomplete"] if args.retry_incomplete else []
            results[tid] = "complete" if subprocess.run(command, cwd=ROOT).returncode == 0 else "failed"
            print(f"[scratch] {tid}: {results[tid]}", flush=True)
        failed = [tid for tid, state in results.items() if state != "complete"]
        overnight._atomic_json(status_path, {"state": "failed" if failed else "complete",
                                             "results": {**earlier, **results}})
        print(f"[scratch] DONE: {results}", flush=True)
        if failed:
            raise SystemExit(f"failed arms: {failed}")


if __name__ == "__main__":
    main()
