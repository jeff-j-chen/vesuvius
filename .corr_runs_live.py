"""corr_runs.py -- patch-to-patch gradient correlation study over every training segment (not only mix6).

For every training PATCH s (one label id; PHerc0172 gives two runs, w068 and w087) the current train.py recipe
(utils/config.py apply_mix6_run) trains on s alone for --epochs (3) at --lr (1e-4) without warmup and is then frozen.
The frozen model probes every patch t of the plan (s itself and the other patches of its scroll included) with --batches
training batches drawn exactly as training draws them (augmentations on). Each target's batches are materialised once and
reused for every source model, so all models see identical probe samples. Each probe gradient is compared with
  g_last / g_prev  gradient of the last / second-to-last real optimizer step on s
  g_ema            EMA (0.9) of the unit step gradients of the run (a less noisy "where training was heading")
  u_last           parameter update the last step applied (AdamW-preconditioned; <G_t, u_last> = first-order loss change on t)
  G_s              the next training gradient of s: the mean of s's own probe batches (per-batch cosines to G_s come from
                   count sketches of every batch gradient; for s itself the leave-one-out mean)
Gradient means are kept per half (even / odd batches), so every cosine also has an unbiased cross-half version whose self
value is the split-half reliability. The frozen model is also scored in eval mode on every target (validation region,
or the train region of train-only patches). A baseline run "init" (no training, the init weights) is probed the same way.

The scheduler (default action) runs three phases, each job in its own subprocess (RAM caches and CUDA state never carry
over; no figures): (1) while the downloads run, build cues / norm / anchors of newly complete patches and train each
eligible patch (runs/<s>/trained.pt); (2) once the downloads ended and everything is trained, freeze plan.json and probe
target by target under every model (grads/<t>/<s>.pt); (3) per source model, Gram matrices and tables, then the report.
Every job is resumable and retried up to --max-failures times; a restarted scheduler adopts still-running jobs.

runs_corr/
  daemon.log, daemon_state.json   scheduler console and retry state
  plan.json, patches.csv      frozen patch plan (targets, source models; skipped patches with the reason)
  logs/<mode>_<run>.log       console of each job (train / probe / gram)
  grads/<t>/<s>.pt            half-mean gradients (bf16), per-batch rows and sketches, eval metrics of target t under s
  runs/<run>/
    trained.pt, trained.json  frozen weights, last step gradients / update, AdamW preconditioner
    config.json, tb/          run config, TensorBoard scalars of the training
    train_steps.csv           per optimizer step: loss scale, pre-clip / applied grad norm, cos to previous step and g_ema
    train_losses.csv          per step loss / raw loss;  train_epochs.csv: per-epoch training metrics
    probe_batches.csv         per probe batch: loss, label stats, grad norm, cosine to every reference and to G_s
    targets.csv               per target: every summary statistic (reliability, cross-half / disattenuated / Adam cosines,
                              sign agreement, first-order loss changes, noise scale, batch-level tests, eval metrics)
    target_groups.csv         per target and parameter group (top-level module): the same alignment split by layer
    pairs.csv                 every target pair (t, u) under this run's model
    gram.npz                  raw / Adam-preconditioned / sign Gram matrices per parameter group (re-analysis)
    done.json                 written last by the gram job
  report/                     aggregate of every finished run: matrices, tests, clustering, report.md

usage:
  nohup python3 corr_runs.py > runs_corr/daemon.log 2>&1 &   # the whole study, unattended
  python3 corr_runs.py --dry-run             # eligibility only
  python3 corr_runs.py --no-wait             # do not wait for assemble_training_segments.py to finish
  python3 corr_runs.py --only w068,w087      # a subset of source models (the targets stay every eligible patch)
  python3 corr_runs.py --report              # rebuild the report from the finished runs
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import functools
import gc
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO / "runs_corr"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
REPORT = ROOT / "report"
PLAN = ROOT / "plan.json"
BASELINE = "init"
REFS = ("g_last", "g_prev", "g_ema", "u_last")
GRAM_CHUNK = 1 << 22
NAN = float("nan")
MAX_SAMPLES = None  # --max-samples: overrides data.max_samples_per_epoch (smoke tests)
SKETCH_DIM = 1 << 14  # count-sketch size of every probe batch gradient (per-batch cosines to G_s, computed later)
GRADS = ROOT / "grads"


def set_root(root: str) -> None:
    global ROOT, RUNS, LOGS, REPORT, PLAN, GRADS
    ROOT = (REPO / root).resolve()
    RUNS, LOGS, REPORT, PLAN = ROOT / "runs", ROOT / "logs", ROOT / "report", ROOT / "plan.json"
    GRADS = ROOT / "grads"


# ---- patches and configs -------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Patch:
    pid: int          # label / mask / cue id (the flat alias for PHerc0846A)
    name: str         # assemble_training_segments.SEGMENTS name
    label: str        # run and column label
    scroll: str       # physical scroll (train_scroll_dict domain name)
    fine_scan: bool   # resampled from a finer scan, trained on its .matched zarr
    source_um: float  # native voxel size before resampling to 9.362 um


def _label(name: str) -> str:
    if not name.startswith("auto_grown_"):
        return name
    for marker in ("straightened2_abf_", "straight_abf_"):
        if marker in name:
            return "0846a_" + name.split(marker, 1)[1]
    return "0846a_" + name.split("_")[2][:14]


def _scroll_of(pid: int, segment: str) -> str:
    from utils import mix6
    from utils.config import RUN_TRAIN_SCROLL_DICT

    if pid in mix6.SHEETS_0846A:
        return "pherc0846a"
    for name, ids in RUN_TRAIN_SCROLL_DICT.items():
        if pid in {int(sid) for sid in ids}:
            return name
    return (segment.split("/")[0] or "unknown").lower()


def all_patches() -> list[Patch]:
    """every training segment of assemble_training_segments.SEGMENTS except the holdouts (w055, straightened2)."""
    import assemble_training_segments as ats
    from utils import mix6
    from utils.config import RUN_FINE_SCAN_IDS

    patches = []
    holdouts = set(getattr(ats, "EVAL_0846A", ()))  # held-out 0846A evaluation segments, never trained
    for name, segment, zid in ats.SEGMENTS:
        opts = ats.FRAG_OPTS.get(name, {})
        if opts.get("holdout") or name == mix6.HOLDOUT_NAME or name in holdouts:
            continue
        pid = int(ats.MESHES_0846A[name][1]) if name in ats.MESHES_0846A else int(zid)
        if opts.get("hf_pherc1447"):
            source_um = ats.PHERC1447_SOURCE_UM
        elif opts.get("pooled_special") or opts.get("w013_special"):
            source_um = 2.4
        else:
            source_um = float(opts.get("resample_um") or opts.get("source_um") or ats.TARGET_VOXEL_UM)
        patches.append(Patch(pid, name, _label(name), _scroll_of(pid, segment), pid in RUN_FINE_SCAN_IDS, source_um))
    for patch in list(patches):
        if patch.name in FLAT_FRAGMENTS:
            patches.append(dataclasses.replace(patch, pid=FLAT_FRAGMENTS[patch.name], label=f"{patch.label}_flat"))
    patches = [p for p in patches if p.label not in EXCLUDED]
    return sorted(patches, key=lambda p: (p.scroll, p.label))


# CORR_EXCLUDE=label,label keeps patches out of the study (inherited by every child job)
EXCLUDED = {name.strip() for name in os.environ.get("CORR_EXCLUDE", "").split(",") if name.strip()}


# detached fragments re-run on a flat surface at their intensity-peak layer (like the 0846A depth-13 aliases):
# segment name -> alias id; the alias symlinks the fragment's zarrs / mask / labels
FLAT_FRAGMENTS = {"cr1fr3": 20231201215977, "scroll6_fr8": 20231205222277, "paris1_fr34": 20230301213500,
                  "paris2_fr143": 20230301213832, "paris2_fr47": 20230205142526}
FLAT_LINK_DIRS = ("masks", "inklabels", "dilated_inklabels", "train_masks")


def _intensity_peak_layer(volume_path: str, mask_path: str) -> int:
    """smoothed argmax of the mean intensity per layer under the mask (strided), first and last valid layer skipped."""
    import cv2
    import zarr

    volume = zarr.open(volume_path, mode="r")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[::16, ::16] > 0
    stack = np.asarray(volume[:, ::16, ::16]).astype(np.float32)
    rows, cols = min(mask.shape[0], stack.shape[1]), min(mask.shape[1], stack.shape[2])
    profile = stack[:, :rows, :cols][:, mask[:rows, :cols]].mean(axis=1)
    valid = np.flatnonzero(profile > 0)[1:-1]
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    return int(valid[smooth[valid].argmax()])


def _merge_norm_entries(entries: dict) -> None:
    from utils.norm import UNIFIED_CACHE_PATH

    for attempt in range(10):
        try:
            cache = json.loads(Path(UNIFIED_CACHE_PATH).read_text())
            break
        except ValueError:
            if attempt == 9:
                raise
            time.sleep(1.0)
    cache.update(entries)
    _atomic_json(cache, Path(UNIFIED_CACHE_PATH))


def ensure_flat_fragments(patches) -> None:
    """set up every flat fragment alias whose source fragment is complete (idempotent)."""
    from utils.norm import UNIFIED_CACHE_PATH, load_cached_norm

    config = base_config()
    zarrs = Path(config.data.zarr_path)
    sources = {p.name: p for p in patches if not p.label.endswith("_flat")}
    for name, alias in FLAT_FRAGMENTS.items():
        source = sources.get(name)
        if source is None or basic_problems(source, config) or norm_problems(source, config):
            continue
        src, alias = str(source.pid), str(alias)
        try:
            for suffix in ("", ".matched"):
                link = zarrs / f"{alias}{suffix}.zarr"
                if not link.exists():
                    link.symlink_to(f"{src}{suffix}.zarr")
            for folder in FLAT_LINK_DIRS:
                target = REPO / folder / f"{src}.png"
                link = REPO / folder / f"{alias}.png"
                if target.is_file() and not link.exists():
                    link.symlink_to(f"{src}.png")
            surface = REPO / config.data.surface_label_dir / alias
            if not (surface / "metadata.json").is_file():
                import cv2

                mask_path = str(REPO / "masks" / f"{src}.png")
                depth = _intensity_peak_layer(str(zarrs / f"{src}.matched.zarr"), mask_path)
                detected = np.load(REPO / config.data.surface_label_dir / src / "depth.npy", mmap_mode="r")[::16, ::16]
                detected = detected[detected != 255]
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
                surface.mkdir(parents=True, exist_ok=True)
                np.save(surface / "depth.npy", np.where(mask, depth, 255).astype(np.uint8))
                np.save(surface / "confidence.npy", np.where(mask, 255, 0).astype(np.uint8))
                _atomic_json({"scroll_id": int(alias), "flat_from": int(src), "flat_depth": depth,
                              "flat_rule": "smoothed argmax of the mean layer intensity (matched volume)",
                              "detected_median": int(np.median(detected)) if detected.size else None,
                              "detected_p10_p90": [int(np.percentile(detected, q)) for q in (10, 90)] if detected.size else None,
                              "invalid_depth_value": 255}, surface / "metadata.json")
                print(f"[corr] flat alias {alias} of {name}: layer {depth} (detected median "
                      f"{int(np.median(detected)) if detected.size else '?'})", flush=True)
            entries = {}
            for suffix in ("", ".matched"):
                if load_cached_norm(f"{alias}{suffix}", UNIFIED_CACHE_PATH) is None:
                    stats = load_cached_norm(f"{src}{suffix}", UNIFIED_CACHE_PATH)
                    if stats is not None:  # same voxels, same global statistics
                        entries[f"{alias}{suffix}"] = dict(zip(("mean", "std", "min", "max"), stats))
            if entries:
                _merge_norm_entries(entries)
        except Exception as exc:
            print(f"[corr] flat alias of {name} failed: {exc}", flush=True)


def _build_alias_cue(job) -> str:
    """asf cue of a flat alias, read from its matched volume around its flat surface."""
    alias, delta, force = job
    os.chdir(REPO)
    for path in (str(REPO), str(REPO / "old")):  # nonml_ink.py lives in old/
        if path not in sys.path:
            sys.path.insert(0, path)
    from utils import mix6, precomputed_asf

    config = base_config()
    config.data.cue_dir = f"./{mix6.CUE_DIRS[delta]}"
    config.data.surface_window_offset_by_scroll = {int(alias): delta}
    return f"{alias} shift {delta:+d}: " + precomputed_asf.build(alias, force=force, config=config)


def base_config():
    """the train.py default run (apply_mix6_run); its per-scroll dicts (suffixes, offsets, train-only ids) cover every id."""
    from utils.config import Config, apply_mix6_run

    config = Config()
    apply_mix6_run(config)
    for alias in FLAT_FRAGMENTS.values():
        config.data.zarr_suffix[str(alias)] = ".matched"
        config.data.native_noise_ids = [*config.data.native_noise_ids, alias]
    config.data.train_only_scroll_ids = [*config.data.train_only_scroll_ids, *TRAIN_ONLY_EXTRA]
    return config


# labelled segments whose train mask is empty: the whole patch is training data (eval uses unaugmented train windows)
TRAIN_ONLY_EXTRA = (20250825181859, 20260220213127)  # 500P2_m1, p841_w00


def _scroll_entry(pid: int):
    from utils import mix6
    from utils.config import DEFAULT_SCROLLS, RUN_SCROLLS, TRAIN_SCROLLS, ScrollConfig

    known = {int(s.scroll_id): s for s in (*DEFAULT_SCROLLS, *TRAIN_SCROLLS, *RUN_SCROLLS)}
    if pid in known:
        return dataclasses.replace(known[pid])
    return ScrollConfig(pid, split_axis="x", train_split_frac=1.0 if pid in mix6.SHEETS_0846A or pid in TRAIN_ONLY_EXTRA else 0.75)


def single_patch_config(patch: Patch, run_dir: Path | None = None, epochs: int = 3, lr: float = 1e-4):
    """the current recipe restricted to one patch: constant lr, no warmup, no figures, no checkpoints."""
    config = base_config()
    data, tra = config.data, config.tra
    data.scrolls = [_scroll_entry(patch.pid)]
    data.train_scroll_dict = {patch.scroll: [patch.pid]}
    data.train_scroll_weights = [1]
    data.equalize_domain = ""
    data.vis_scroll_ids = []
    data.validation_scroll_ids = None
    data.holdout_exclusion_dir = ""  # pure per-patch diagnostics: no prize-sheet discard on the 0846A sheets
    tra.eval_int_scrolls = 0
    tra.dann_n_domains = 1
    tra.n_epochs = int(epochs)
    tra.lr = float(lr)
    tra.warmup_epochs = 0
    tra.disable_figures = True
    tra.eval_int = tra.test_int = tra.probe_int = tra.save_int = 10 ** 9
    tra.test_on_final = False
    config.save_final = None
    if MAX_SAMPLES:
        data.max_samples_per_epoch = int(MAX_SAMPLES)
    if os.environ.get("CORR_SMOKE_BATCH"):  # smoke tests next to a running job on the same GPU
        config.dl.batch_size = int(os.environ["CORR_SMOKE_BATCH"])
    if run_dir is not None:
        config.model_dir = str(run_dir)
        tra.log_dir = str(run_dir)
        config.run_name = config.exp_name = f"corr_{run_dir.name}"
    return config


def basic_problems(patch: Patch, config) -> list[str]:
    from utils import mix6

    suffix = (config.data.zarr_suffix or {}).get(str(patch.pid), "")
    surface = Path(config.data.surface_label_dir) / str(patch.pid)
    paths = [
        Path(config.data.zarr_path) / f"{patch.pid}{suffix}.zarr",
        Path("masks") / f"{patch.pid}.png",
        Path(config.data.inklabel_dir) / f"{patch.pid}.png",
        Path(config.data.train_mask_dir) / f"{patch.pid}.png",
        surface / "depth.npy",
        surface / "confidence.npy",
        surface / "metadata.json",  # written after depth / confidence are complete
    ]
    problems = [f"missing {path}" for path in paths if not path.exists()]
    if suffix == ".matched" and not problems:
        from utils.spectral_match import is_matched

        if not is_matched(str(paths[0])):
            problems.append(f"{paths[0]} is not a finished matched zarr")
    if not problems and patch.pid in mix6.FLAT_SURFACE:
        problems += mix6.flat_surface_problems([patch.pid])
    return problems


def norm_problems(patch: Patch, config) -> list[str]:
    """global norm stats and surface anchors of the volume training reads."""
    from utils.norm import load_cached_norm

    volume = f"{patch.pid}{(config.data.zarr_suffix or {}).get(str(patch.pid), '')}"
    try:
        if load_cached_norm(volume, mode=str(config.data.norm_mode)) is None:
            return [f"no norm stats for {volume}"]
    except Exception as exc:
        return [f"{volume}: {exc}"]
    return []


def _missing_cues(patches, config) -> list[tuple[int, int, bool]]:
    suffixes = config.data.zarr_suffix or {}
    directories = {0: config.data.cue_dir, **{int(k): v for k, v in (config.data.depth_cue_dirs or {}).items()}}
    return [
        (p.pid, delta, False)
        for p in patches for delta, directory in directories.items()
        if not (Path(directory) / f"{p.pid}{suffixes.get(str(p.pid), '')}.npy").is_file()
    ]


def prepare(patches, workers: int) -> int:
    """surface anchors and the 3 depths of asf cue maps for every patch whose downloaded inputs are present;
    returns the number of cue maps it tried to build."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from utils import mix6
    from utils.norm import UNIFIED_CACHE_PATH, compute_norm, ensure_surface_anchors, load_cached_norm

    config = base_config()
    ensure_flat_fragments(patches)
    ready = [p for p in patches if not basic_problems(p, config)]
    suffixes = config.data.zarr_suffix or {}
    ids = [str(p.pid) for p in ready]
    for sid in [*ids, *(f"{i}{suffixes[i]}" for i in ids if suffixes.get(i))]:
        try:
            if load_cached_norm(sid, UNIFIED_CACHE_PATH) is None:
                compute_norm(sid, config.data.zarr_path, cache_path=UNIFIED_CACHE_PATH, mask_id=sid.split(".")[0])
            ensure_surface_anchors([sid], config.data.zarr_path, surface_dir=config.data.surface_label_dir)
        except Exception as exc:
            print(f"[corr] norm / surface anchors of {sid} failed: {exc}", flush=True)
    jobs = _missing_cues(ready, config)
    print(f"[corr] {len(ready)}/{len(patches)} patches have their inputs; {len(jobs)} asf cue map(s) to build", flush=True)
    if not jobs:
        return 0
    aliases = set(FLAT_FRAGMENTS.values())
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_build_alias_cue if job[0] in aliases else mix6._build_cue, job): job for job in jobs}
        for future, job in futures.items():
            try:
                print("[corr] asf cue", future.result(), flush=True)
            except Exception as exc:
                print(f"[corr] asf cue {job[0]} shift {job[1]:+d} FAILED: {exc}", flush=True)
    return len(jobs)


def eligibility(patches, build_init: bool) -> tuple[list[Patch], dict[int, str]]:
    """basic file checks plus train.preflight_inputs on the single-patch config of every patch."""
    import train
    from utils.config import MIX6_INIT

    if build_init and not os.path.isfile(MIX6_INIT):
        from utils.cue_expand import build_mix6_init

        build_mix6_init(MIX6_INIT)
    config = base_config()
    eligible, skipped = [], {}
    for patch in patches:
        problems = basic_problems(patch, config) or norm_problems(patch, config)
        if not problems:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    train.preflight_inputs(single_patch_config(patch), build_init=False)
            except FileNotFoundError as exc:
                problems = [line.strip() for line in str(exc).splitlines() if line.strip()][1:]
        if problems:
            skipped[patch.pid] = "; ".join(problems)[:400]
        else:
            eligible.append(patch)
    return eligible, skipped


def load_plan() -> dict:
    return json.loads(PLAN.read_text())


def plan_patches(plan: dict) -> list[Patch]:
    return [Patch(**entry) for entry in plan["patches"]]


def write_patches_csv(patches, eligible, skipped) -> None:
    chosen = {p.pid for p in eligible}
    with open(ROOT / "patches.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pid", "label", "name", "scroll", "fine_scan", "source_um", "status", "reason"])
        for p in patches:
            writer.writerow([p.pid, p.label, p.name, p.scroll, int(p.fine_scan), p.source_um,
                             "eligible" if p.pid in chosen else "skipped", skipped.get(p.pid, "")])


# ---- worker helpers ------------------------------------------------------------------------------------------------------

class _TbOnly:
    """stands in for TensorboardVisualizer: a SummaryWriter, no figure volumes."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.log_path = log_dir
        self.writer = SummaryWriter(log_dir)

    def close(self) -> None:
        self.writer.close()


def _make_trainer(config):
    """train.Trainer with TensorBoard scalars only (no figure visualizers), logging under config.model_dir/tb."""
    import train

    class CorrTrainer(train.Trainer):
        def _init_visualizers(self):
            self.vis = _TbOnly(os.path.join(str(self.c.model_dir), "tb"))
            self.scroll_vis = None

    return CorrTrainer(config)


def _atomic_save(obj, path: Path) -> None:
    import torch

    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def _atomic_json(obj, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    temporary.write_text(json.dumps(obj, indent=2))
    os.replace(temporary, path)


def _flat_params(params):
    import torch

    return torch.cat([p.detach().reshape(-1).float() for p in params])


def _flat_grads(params):
    import torch

    return torch.cat([
        p.grad.detach().reshape(-1).float() if p.grad is not None
        else torch.zeros(p.numel(), device=p.device) for p in params
    ])


def _param_groups(named) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = {}
    offset = 0
    for name, parameter in named:
        groups.setdefault(name.split(".")[0], []).append((offset, offset + parameter.numel()))
        offset += parameter.numel()
    return groups


def _seed(pid: int) -> int:
    return int((int(pid) * 2654435761) % (2 ** 31 - 1))


def _seed_all(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def _loader(dataset, batch_size: int, workers: int, seed: int, drop_last: bool):
    from torch.utils.data import DataLoader

    from utils.dataloader import _worker_init

    kwargs = {"batch_size": batch_size, "num_workers": workers, "pin_memory": True, "drop_last": drop_last}
    if workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = functools.partial(_worker_init, base_seed=seed)
    return DataLoader(dataset, **kwargs)


def _close_iterator(iterator) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def parse_batch(c, batch):
    """the field layout train.Trainer.train_epoch reads."""
    from utils.dataloader import needs_domain_ids, needs_patch_ids

    images, labels, mask = batch[:3]
    index = 3

    def take(flag, count=1):
        nonlocal index
        if not flag:
            return [None] * count
        values = [batch[index + i] if len(batch) > index + i else None for i in range(count)]
        index += count
        return values

    (domain_ids,) = take(needs_domain_ids(c))
    (patch_ids,) = take(needs_patch_ids(c))
    (character_ids,) = take(bool(getattr(c.tra, "character_macro_metrics", False)))
    (target_offsets,) = take(bool(getattr(c.data, "target_aware_ctx_jitter", False)))
    surface_maps = any(bool(getattr(c.model, name, False))
                       for name in ("new_learned_surface", "better_surface", "surface_teacher_input"))
    surface_depth, surface_confidence = take(surface_maps, 2)
    (depth_shift,) = take(bool(getattr(c.tra, "depth_shift_aux", False)))
    context_pair, context_pair_active = take(bool(getattr(c.tra, "context_consistency", False)), 2)
    depth_pair, depth_pair_surface, depth_pair_confidence, depth_pair_active = take(
        bool(getattr(c.tra, "depth_view_consistency", False)), 4)
    return images, labels, mask, {
        "domain_ids": domain_ids, "patch_ids": patch_ids, "character_ids": character_ids,
        "target_offsets": target_offsets, "surface_depth": surface_depth, "surface_confidence": surface_confidence,
        "depth_shift": depth_shift, "context_pair": context_pair, "context_pair_active": context_pair_active,
        "depth_pair": depth_pair, "depth_pair_surface": depth_pair_surface,
        "depth_pair_confidence": depth_pair_confidence, "depth_pair_active": depth_pair_active,
    }


class _StepRecorder:
    """wraps optimizer.step: keeps the real step gradients / update and logs per-step alignment."""

    def __init__(self, trainer, params, clip_state):
        self.trainer, self.params, self.clip_state = trainer, params, clip_state
        self.optimizer = trainer.optimizer
        self._step = self.optimizer.step
        self.optimizer.step = self.step
        self.g_last = self.g_prev = self.g_ema = self.u_last = None
        self.rows: list[dict] = []
        self.epoch = 0
        self.skipped = 0

    def detach(self) -> None:
        self.optimizer.step = self._step

    def step(self, *args, **kwargs):
        import torch

        found_inf = getattr(self.optimizer, "found_inf", None)
        before = _flat_params(self.params)
        grad = _flat_grads(self.params)
        result = self._step(*args, **kwargs)
        if (found_inf is not None and float(found_inf) > 0) or not bool(torch.isfinite(grad).all()):
            self.skipped += 1
            return result
        update = _flat_params(self.params).sub_(before)
        norm = float(grad.norm())
        unit = grad / max(norm, 1e-30)
        cos_prev = float(unit.dot(self.g_last) / max(float(self.g_last.norm()), 1e-30)) if self.g_last is not None else NAN
        cos_ema = float(unit.dot(self.g_ema) / max(float(self.g_ema.norm()), 1e-30)) if self.g_ema is not None else NAN
        self.g_prev, self.g_last, self.u_last = self.g_last, grad, update
        self.g_ema = unit.clone() if self.g_ema is None else self.g_ema.mul_(0.9).add_(unit, alpha=0.1)
        scale = float(self.trainer.scaler.get_scale()) if self.trainer.scaler.is_enabled() else 1.0
        pre_clip = self.clip_state.get("last")
        self.rows.append({
            "step": len(self.rows) + 1, "epoch": self.epoch + 1, "loss_scale": scale,
            "grad_norm_preclip_unscaled": float(pre_clip) / scale if pre_clip is not None else NAN,
            "grad_norm_applied_unscaled": norm / scale,
            "cos_prev_step": cos_prev, "cos_ema": cos_ema, "update_norm": float(update.norm()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        })
        return result


def _adam_preconditioner(optimizer, params):
    """lr_scale / (sqrt(v_hat) + eps) per coordinate (the diagonal AdamW applies), and the share of coordinates in the
    eps-dominated (SGD-like) regime."""
    import torch

    lookup = {}
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0)) if group.get("lr_enabled", True) else 0.0
        for p in group["params"]:
            lookup[id(p)] = (scale, float(group["betas"][1]), float(group["eps"]))
    parts, below, counted = [], 0, 0
    for p in params:
        state = optimizer.state.get(p, {})
        if "exp_avg_sq" not in state or id(p) not in lookup:
            parts.append(torch.zeros(p.numel(), device=p.device))
            continue
        scale, beta2, eps = lookup[id(p)]
        step = float(state["step"])
        root = (state["exp_avg_sq"].float() / max(1.0 - beta2 ** step, 1e-12)).sqrt().reshape(-1)
        below += int((root < eps).sum())
        counted += root.numel()
        parts.append(scale / (root + eps))
    return torch.cat(parts), below / max(counted, 1)


def _batch_stats(values) -> dict:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return {"mean": NAN, "std": NAN, "sem": NAN, "ci_lo": NAN, "ci_hi": NAN, "frac_pos": NAN, "t": NAN, "p": NAN}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else NAN
    sem = std / math.sqrt(values.size) if values.size > 1 else NAN
    t = p = NAN
    if values.size > 2 and std > 0:
        from scipy import stats

        t = mean / sem
        p = float(2 * stats.t.sf(abs(t), values.size - 1))
    return {"mean": mean, "std": std, "sem": sem, "ci_lo": mean - 1.96 * sem, "ci_hi": mean + 1.96 * sem,
            "frac_pos": float((values > 0).mean()), "t": t, "p": p}


def _score_metrics(prefix, labels, scores, chars) -> dict:
    from utils.training_utils import calculate_character_metrics, calculate_metrics

    if not len(labels):
        return {}
    labels = np.concatenate(labels).astype(int)
    scores = np.concatenate(scores).astype(np.float64)
    chars = np.concatenate(chars).astype(np.int64)
    base = calculate_metrics(labels, (scores > 0.5).astype(int), scores)
    character = calculate_character_metrics(labels, scores, chars)
    return {
        f"{prefix}cells": int(labels.size), f"{prefix}pos_frac": float((labels == 1).mean()),
        f"{prefix}roc_auc": float(base["roc_auc"]), f"{prefix}pr_auc": float(base["pr_auc"]),
        f"{prefix}f1": float(base["f1"]), f"{prefix}balanced_accuracy": float(base["balanced_accuracy"]),
        f"{prefix}char_ap": float(character["character_ap_macro"]),
        f"{prefix}char_f1": float(character["character_f1_macro"]),
        f"{prefix}char_count": int(character["character_count"]),
    }


def _collect_batches(c, dataset, n, seed, workers, drop_last=True) -> list:
    """materialise n loader batches once (augmentations as configured on the dataset); every source model then sees
    exactly these windows."""
    batches, passes = [], 0
    while len(batches) < n and passes < 8:
        _seed_all(seed + passes)
        iterator = iter(_loader(dataset, int(c.dl.batch_size), workers, seed + passes, drop_last=drop_last))
        produced = 0
        try:
            for batch in iterator:
                produced += 1
                batches.append(parse_batch(c, batch))
                if len(batches) >= n:
                    break
        finally:
            _close_iterator(iterator)
            del iterator
        if produced == 0 or not drop_last:
            break
        passes += 1
    return batches


def _clone(value):
    return value.clone() if hasattr(value, "clone") else value


def _batch_gradient(trainer, params, batch, epoch_arg, stats):
    """training gradient of one batch at the frozen weights (no optimizer step), unscaled; (grad, result) or None."""
    import torch
    import train

    images, labels, mask, kwargs = batch
    # an fp16 overflow redoes the same batch at half the loss scale, so no batch is dropped selectively
    for _attempt in range(4):
        trainer._defer_optimizer_step = True
        try:
            pending = trainer._train_batch(_clone(images), _clone(labels), _clone(mask), epoch=epoch_arg,
                                           **{key: _clone(value) for key, value in kwargs.items()})
        finally:
            trainer._defer_optimizer_step = False
        if not isinstance(pending, train._PendingGroup):
            stats["empty"] += 1
            return None
        result, _ = trainer._finish_pending(pending)
        scale = float(trainer.scaler.get_scale()) if trainer.scaler.is_enabled() else 1.0
        grad = _flat_grads(params).div_(scale)
        trainer.optimizer.zero_grad(set_to_none=True)
        if bool(torch.isfinite(grad).all()):
            return grad, result
        if not trainer.scaler.is_enabled() or not math.isfinite(float(result[3])) or scale <= 1.0:
            break  # a non-finite loss is not an overflow
        stats["retries"] += 1
        trainer.scaler.update(max(scale / 2.0, 1.0))
    stats["nonfinite"] += 1
    return None


def _probe_source(trainer, params, n_params, batches, epoch_arg, ref_matrix, ref_norms, sketch) -> dict:
    """gradients of the stored batches under the currently loaded model: half means, per-batch rows and sketches."""
    import torch

    device = trainer.c.device
    sums = [torch.zeros(n_params, device=device), torch.zeros(n_params, device=device)]
    counts, sumsq, rows, sketches = [0, 0], 0.0, [], []
    scores, labels, chars = [], [], []
    stats = {"empty": 0, "nonfinite": 0, "retries": 0}
    for index, batch in enumerate(batches):
        out = _batch_gradient(trainer, params, batch, epoch_arg, stats)
        if out is None:
            continue
        grad, result = out
        done = counts[0] + counts[1]
        sums[done % 2] += grad
        counts[done % 2] += 1
        norm2 = float(grad.dot(grad))
        sumsq += norm2
        norm = math.sqrt(norm2)
        batch_labels = np.asarray(result[1])
        row = {"batch": index, "loss": float(result[3]), "raw_loss": float(result[4]),
               "cells": int(batch_labels.size),
               "pos_frac": float((batch_labels == 1).mean()) if batch_labels.size else NAN, "grad_norm": norm}
        if ref_matrix is not None:
            dots = (ref_matrix @ grad).tolist()
            for name, dot, ref_norm in zip(REFS, dots, ref_norms):
                row[f"cos_{name}"] = dot / (norm * ref_norm) if ref_norm > 0 and norm > 0 else NAN
            row["dot_u_last"] = dots[REFS.index("u_last")]
        rows.append(row)
        sketches.append(sketch(grad).cpu())
        scores.append(np.asarray(result[0]))
        labels.append(batch_labels)
        chars.append(np.asarray(result[2]))
    return {
        "A": (sums[0] / max(counts[0], 1)).to(torch.bfloat16).cpu(),
        "B": (sums[1] / max(counts[1], 1)).to(torch.bfloat16).cpu(),
        "nA": counts[0], "nB": counts[1], "sumsq": sumsq, "rows": rows,
        "sketch": torch.stack(sketches) if sketches else torch.zeros(0, SKETCH_DIM),
        "loss_scale": float(trainer.scaler.get_scale()) if trainer.scaler.is_enabled() else 1.0,
        "train_mode": _score_metrics("trainmode_", labels, scores, chars), **stats,
    }


def _eval_dataset(train_set, valid_set, batches, batch_size, seed):
    """a fixed random subset of the target's validation windows (train windows, without augmentation, for
    train-only patches); (dataset, split) or (None, 'none')."""
    dataset, split = valid_set, "valid"
    coords = getattr(valid_set, "block_coords", None)
    if coords is None or len(coords) == 0:
        dataset, split = train_set, "train"
        dataset._character_balanced = False
        dataset.shuffle = False
        coords = dataset.block_coords
    if coords is None or len(coords) == 0:
        return None, "none"
    order = np.random.default_rng(seed).permutation(len(coords))[: batches * batch_size]
    dataset.block_coords = [coords[i] for i in order]
    dataset.samples_per_epoch = len(dataset.block_coords)
    dataset.apply_transforms = False
    return dataset, split


def _eval_batches(trainer, batches, split) -> dict:
    """the loaded model in eval mode on the stored eval batches."""
    import torch
    from torch.amp.autocast_mode import autocast

    if not batches:
        return {"eval_split": split}
    device = trainer.c.device
    model = trainer.model
    model.eval()
    loss_sum, n_cells, scores, labels, chars = 0.0, 0, [], [], []
    try:
        with torch.no_grad(), autocast(device, enabled=device == "cuda"):
            for images, batch_labels, mask, kwargs in batches:
                images = images.to(device, non_blocking=True)
                depth, confidence = kwargs["surface_depth"], kwargs["surface_confidence"]
                if depth is not None:
                    depth = depth.to(device, non_blocking=True).float()
                    confidence = confidence.to(device, non_blocking=True).float()
                outputs = model(images, target_offsets=kwargs["target_offsets"],
                                teacher_surface_depth=depth, teacher_surface_confidence=confidence)
                if outputs.dim() == 4:
                    outputs = outputs.flatten(1).max(dim=1, keepdim=True).values
                outputs = outputs.float()
                count = images.size(0)
                batch_labels = batch_labels.to(device).view(count, -1).float()
                mask = mask.to(device).view(count, -1)
                valid = (mask > 0) if mask.shape == batch_labels.shape else (mask.sum(dim=1, keepdim=True) > 0)
                # -1 = explicit negative; -2 / -3 (PU / line unlabelled) carry no label
                keep = valid & (batch_labels >= -1)
                target = (batch_labels > 0.5).float()
                loss_sum += float((trainer.criterion(outputs, target) * keep).sum())
                n_cells += int(keep.sum())
                scores.append(torch.sigmoid(outputs)[keep].cpu().numpy())
                labels.append(target[keep].cpu().numpy().astype(int))
                ids = kwargs["character_ids"]
                chars.append(ids.view(count, -1).to(device)[keep].cpu().numpy() if ids is not None
                             else np.zeros(int(keep.sum()), dtype=np.int64))
    finally:
        model.train()
    out = {"eval_split": split, "eval_loss": loss_sum / max(n_cells, 1)}
    out.update(_score_metrics("eval_", labels, scores, chars))
    return out


def _grams(rows_host, n_targets, groups, precond):
    """per parameter group: raw and Adam-preconditioned Gram of all rows, sign Gram of the target means and refs."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(groups)
    m = len(rows_host)
    s = n_targets + len(REFS)
    raw = np.zeros((len(names), m, m))
    pre = np.full((len(names), m, m), NAN) if precond is None else np.zeros((len(names), m, m))
    sign = np.zeros((len(names), s, s))
    coords = np.zeros(len(names), dtype=np.int64)
    for gi, name in enumerate(names):
        for a, b in groups[name]:
            for c0 in range(a, b, GRAM_CHUNK):
                c1 = min(b, c0 + GRAM_CHUNK)
                x = torch.stack([row[c0:c1] for row in rows_host]).to(device).double()
                raw[gi] += (x @ x.T).cpu().numpy()
                if precond is not None:
                    xp = x * precond[c0:c1].to(device).double().sqrt()
                    pre[gi] += (xp @ xp.T).cpu().numpy()
                    del xp
                means = x[0:2 * n_targets:2] + x[1:2 * n_targets:2]
                signs = torch.sign(torch.cat([means, x[2 * n_targets:]]))
                sign[gi] += (signs @ signs.T).cpu().numpy()
                coords[gi] += c1 - c0
                del x, means, signs
    return names, raw, pre, sign, coords


def _cos(h, i, j) -> float:
    denominator = h[i, i] * h[j, j]
    return float(h[i, j] / math.sqrt(denominator)) if denominator > 0 else NAN


def _cross(h, t, u) -> float:
    """cosine between independent halves: unbiased by batch noise; the self value is the split-half reliability."""
    if t == u:
        return _cos(h, 2 * t, 2 * t + 1)
    return 0.5 * (_cos(h, 2 * t, 2 * u + 1) + _cos(h, 2 * t + 1, 2 * u))


def _cross_dot(h, t, u) -> float:
    if t == u:
        return float(h[2 * t, 2 * t + 1])
    return 0.5 * float(h[2 * t, 2 * u + 1] + h[2 * t + 1, 2 * u])


def _means_gram(h, weights_a):
    """Gram of the full target means (and refs) from the half Gram."""
    n_targets = len(weights_a)
    m = h.shape[0]
    w = np.zeros((n_targets + len(REFS), m))
    for t, wa in enumerate(weights_a):
        w[t, 2 * t], w[t, 2 * t + 1] = wa, 1.0 - wa
    for r in range(len(REFS)):
        w[n_targets + r, 2 * n_targets + r] = 1.0
    return w @ h @ w.T


def _safe_div(a, b) -> float:
    return float(a / b) if b and np.isfinite(b) and b != 0 else NAN


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


# ---- worker --------------------------------------------------------------------------------------------------------------

def _trained_ok(run: str, args) -> bool:
    path = RUNS / run / "trained.json"
    if not path.is_file() or not (RUNS / run / "trained.pt").is_file():
        return False
    info = json.loads(path.read_text())
    return (info.get("epochs") == args.epochs and abs(float(info.get("lr", -1)) - args.lr) < 1e-12
            and int(info.get("max_samples", 0)) == int(args.max_samples or 0))


def _save_trained(run_dir: Path, named, trainer, recorder, precond, adam_eps_share, info: dict) -> None:
    """frozen weights plus everything the probe phase needs from the end of training (atomic)."""
    import torch

    dro = trainer._patch_groupdro_state
    payload = {
        "model": {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
        "scaler": trainer.scaler.state_dict(),
        "param_names": [name for name, _ in named],
        "refs": {name: (getattr(recorder, name) if getattr(recorder, name) is not None
                        else torch.zeros_like(recorder.g_last)).cpu() for name in REFS},
        "precond": precond.cpu(), "adam_eps_share": adam_eps_share, "steps": len(recorder.rows),
        "dro_state": None if dro is None else tuple(t.detach().cpu() for t in dro),
        "dro_slots": dict(trainer._patch_groupdro_slots),
        "log_weights": {name: dict(getattr(trainer, name)) for name in (
            "_character_groupdro_log_weights", "_physical_domain_groupdro_log_weights",
            "_physical_patch_groupdro_log_weights")},
    }
    temporary = run_dir / f"trained.pt.tmp{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, run_dir / "trained.pt")
    (run_dir / "trained.json").write_text(json.dumps(info, indent=2))


def train_worker(run: str, args) -> None:
    """train one source patch (the current recipe on that patch alone) and save the frozen model to trained.pt."""
    import torch

    os.chdir(REPO)
    matches = [p for p in all_patches() if p.label == run]
    if not matches:
        raise SystemExit(f"[corr] unknown patch {run}")
    source = matches[0]
    run_dir = RUNS / run
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs)
    config = single_patch_config(source, run_dir, epochs, args.lr)
    started = time.time()
    timings = {}

    clip_state: dict = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(*clip_args, **clip_kwargs):
        clip_state["last"] = original_clip(*clip_args, **clip_kwargs)
        return clip_state["last"]

    trainer = _make_trainer(config)
    model = trainer.model
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    groups = _param_groups(named)
    n_params = sum(p.numel() for p in params)
    timings["setup_s"] = time.time() - started

    torch.nn.utils.clip_grad_norm_ = recording_clip
    recorder = _StepRecorder(trainer, params, clip_state)
    losses, epoch_rows = [], []
    original_finish = trainer._finish_pending

    def finishing(pending):
        result, tail = original_finish(pending)
        losses.append({"index": len(losses) + 1, "epoch": recorder.epoch + 1,
                       "loss": float(result[3]), "raw_loss": float(result[4])})
        return result, tail

    trainer._finish_pending = finishing
    writer = trainer.vis.writer
    t0 = time.time()
    for epoch in range(epochs):
        recorder.epoch = epoch
        epoch_start = time.time()
        steps_before = len(recorder.rows)
        trainer._update_encoder_optimization(epoch)
        trainer.train_dataset.apply_transforms = bool(
            epoch >= int(getattr(config.tra, "aug_start_epoch", 0)) and config.dl.data_aug)
        metrics = trainer.train_epoch(None, epoch=epoch)
        row = {"epoch": epoch + 1, "seconds": time.time() - epoch_start, "steps": len(recorder.rows) - steps_before}
        row.update({key: float(value) for key, value in metrics.items()
                    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)})
        epoch_rows.append(row)
        for key in ("loss", "raw_loss", "roc_auc", "pr_auc", "character_ap_macro", "pu_loss", "text_region_loss"):
            if key in row:
                writer.add_scalar(f"train/{key}", row[key], epoch + 1)
        print(f"[corr] {run} epoch {epoch + 1}/{epochs}: loss={row.get('loss', NAN):.4f} "
              f"pr_auc={row.get('pr_auc', NAN):.4f} steps={row['steps']} ({row['seconds']:.0f}s)", flush=True)
    for row in recorder.rows:
        for key in ("grad_norm_applied_unscaled", "cos_prev_step", "cos_ema"):
            if np.isfinite(row[key]):
                writer.add_scalar(f"step/{key}", row[key], row["step"])
    for row in losses:
        writer.add_scalar("step/loss", row["loss"], row["index"])
    writer.flush()
    timings["train_s"] = time.time() - t0
    del trainer._finish_pending
    recorder.detach()
    torch.nn.utils.clip_grad_norm_ = original_clip
    _write_csv(run_dir / "train_steps.csv", recorder.rows)
    _write_csv(run_dir / "train_losses.csv", losses)
    _write_csv(run_dir / "train_epochs.csv", epoch_rows)
    if recorder.g_last is None:
        raise RuntimeError(f"{run}: no optimizer step completed")
    precond, adam_eps_share = _adam_preconditioner(trainer.optimizer, params)
    timings["total_s"] = time.time() - started
    _save_trained(run_dir, named, trainer, recorder, precond, adam_eps_share, {
        "run": run, "source": source.pid, "epochs": epochs, "lr": float(args.lr),
        "max_samples": int(args.max_samples or 0), "steps": len(recorder.rows),
        "skipped_steps": recorder.skipped, "n_params": int(n_params), "groups": groups, "timings": timings,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"[corr] {run} trained in {timings['total_s'] / 60:.1f} min -> {run_dir / 'trained.pt'}", flush=True)
    with contextlib.suppress(Exception):
        trainer.close()


def _pair_meta(args) -> dict:
    return {"batches": int(args.batches), "eval_batches": int(args.eval_batches), "epochs": int(args.epochs),
            "lr": float(args.lr), "max_samples": int(args.max_samples or 0)}


def _pair_ok(target_label: str, source_label: str, args) -> bool:
    """the (source model, target patch) probe file exists and was made with the current settings."""
    meta = GRADS / target_label / f"{source_label}.json"
    if not meta.is_file() or not (GRADS / target_label / f"{source_label}.pt").is_file():
        return False
    try:
        info = json.loads(meta.read_text())
    except ValueError:
        return False
    return all(info.get(key) == value for key, value in _pair_meta(args).items())


def _model_state(trainer, source_label, run_dir, init_state, device):
    """(state to load, refs, extras) of a source: the init weights for the baseline, else trained.pt."""
    import torch

    if source_label == BASELINE:
        return init_state, None
    payload = torch.load(run_dir / "trained.pt", map_location="cpu", weights_only=False)
    names = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
    if payload["param_names"] != names:
        raise RuntimeError(f"{source_label}: trained.pt parameters do not match the model")
    state = {
        "model": payload["model"], "scaler": payload["scaler"],
        "dro_state": None if payload["dro_state"] is None else tuple(t.to(device) for t in payload["dro_state"]),
        "dro_slots": payload["dro_slots"], "log_weights": payload["log_weights"],
    }
    return state, [payload["refs"][name].to(device) for name in REFS]


def _load_state(trainer, state) -> None:
    trainer.model.load_state_dict(state["model"])
    trainer.scaler.load_state_dict(state["scaler"])
    trainer._patch_groupdro_state = None if state["dro_state"] is None else tuple(t.clone() for t in state["dro_state"])
    trainer._patch_groupdro_slots = dict(state["dro_slots"])
    for name, value in state["log_weights"].items():
        setattr(trainer, name, dict(value))
    trainer.optimizer.zero_grad(set_to_none=True)


def probe_worker(target_label: str, args) -> None:
    """target-outer probe: build the target's probe and eval batches once, then compute them under every source model
    of the plan (resumable per pair) and save grads/<target>/<source>.pt."""
    import torch
    from utils import dataloader as data_module

    os.chdir(REPO)
    plan = load_plan()
    patches = plan_patches(plan)
    target = {p.label: p for p in patches}[target_label]
    sources = list(plan["sources"])
    out_dir = GRADS / target.label
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in sources if not _pair_ok(target.label, s, args)]
    print(f"[corr] probe target {target.label}: {len(pending)}/{len(sources)} source model(s) to run", flush=True)
    if not pending:
        return
    started = time.time()
    config = single_patch_config(target, ROOT / "probe_tmp" / target.label, max(int(args.epochs), 1), args.lr)
    config.data.selective_chunk_preload = False
    trainer = _make_trainer(config)
    model = trainer.model
    device = config.device
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    n_params = sum(p.numel() for p in params)
    if not (GRADS / "_meta.json").is_file():
        _atomic_json({"n_params": int(n_params), "groups": _param_groups(named),
                      "param_names": [name for name, _ in named]}, GRADS / "_meta.json")
    init_state = {
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "scaler": trainer.scaler.state_dict(), "dro_state": None, "dro_slots": {},
        "log_weights": {name: {} for name in ("_character_groupdro_log_weights",
                                              "_physical_domain_groupdro_log_weights",
                                              "_physical_patch_groupdro_log_weights")},
    }
    trainer.optimizer.state.clear()
    trainer._shutdown_data_workers()
    trainer.train_loader = trainer.valid_loader = trainer.train_dataset = None
    trainer._scroll_dms, trainer._scroll_train_sets, trainer._train_children = {}, None, []
    data_module._PREPARED_DATASET_CACHE.clear()
    gc.collect()

    manager = data_module.DataManager(config, scroll_id=target.pid, domain_id=0)
    train_set, valid_set = manager.get_datasets()
    train_tiles = len(train_set)
    seed = _seed(target.pid)
    workers = int(args.probe_workers)
    train_set.apply_transforms = bool(config.dl.data_aug)
    batches = _collect_batches(config, train_set, int(args.batches), seed, workers)
    eval_set, split = _eval_dataset(train_set, valid_set, int(args.eval_batches), int(config.dl.batch_size), seed + 7919)
    eval_batches = _collect_batches(config, eval_set, int(args.eval_batches), seed + 7919, workers, drop_last=False) \
        if eval_set is not None else []
    del manager, train_set, valid_set, eval_set
    data_module._PREPARED_DATASET_CACHE.clear()
    data_module.cleanup_mmap_files()
    gc.collect()
    print(f"[corr] {target.label}: {len(batches)} probe + {len(eval_batches)} eval batches materialised in "
          f"{time.time() - started:.0f}s", flush=True)

    generator = torch.Generator().manual_seed(12345)
    buckets = torch.randint(SKETCH_DIM, (n_params,), generator=generator).to(device)
    signs = torch.randint(0, 2, (n_params,), generator=generator).float().mul_(2).sub_(1).to(device)

    def sketch(grad):
        return torch.zeros(SKETCH_DIM, device=device).index_add_(0, buckets, grad * signs)

    failures = {}
    for position, source_label in enumerate(pending, start=1):
        source_start = time.time()
        try:
            state, refs = _model_state(trainer, source_label, RUNS / source_label, init_state, device)
            ref_matrix = torch.stack(refs) if refs is not None else None
            ref_norms = [float(r.norm()) for r in refs] if refs is not None else None
            _load_state(trainer, state)
            probe = _probe_source(trainer, params, n_params, batches, int(args.epochs), ref_matrix, ref_norms, sketch)
            _load_state(trainer, state)  # the probe forwards moved the running statistics
            probe["eval"] = _eval_batches(trainer, eval_batches, split)
            probe.update({"train_tiles": train_tiles, "seconds": time.time() - source_start, **_pair_meta(args)})
            _atomic_save(probe, out_dir / f"{source_label}.pt")
            _atomic_json({**_pair_meta(args), "nA": probe["nA"], "nB": probe["nB"], "seconds": probe["seconds"]},
                         out_dir / f"{source_label}.json")
            n = probe["nA"] + probe["nB"]
            print(f"[corr] {target.label} <- {source_label} ({position}/{len(pending)}): {n} batches "
                  f"(retries={probe['retries']} nonfinite={probe['nonfinite']}), eval {split} "
                  f"pr_auc={probe['eval'].get('eval_pr_auc', NAN):.3f} ({probe['seconds']:.0f}s)", flush=True)
            del state, refs, ref_matrix, probe
        except Exception as exc:
            import traceback

            traceback.print_exc()
            failures[source_label] = f"{type(exc).__name__}: {exc}"[:400]
            trainer.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        trainer.close()
    print(f"[corr] probe target {target.label} finished in {(time.time() - started) / 60:.1f} min; "
          f"failed sources: {failures or 'none'}", flush=True)
    if failures:
        raise SystemExit(1)


def gram_worker(run: str, args) -> None:
    """Gram matrices and every per-run table of one source model from its probe files over all plan targets."""
    import torch

    os.chdir(REPO)
    plan = load_plan()
    patches = plan_patches(plan)
    source = None if run == BASELINE else {p.label: p for p in patches}[run]
    run_dir = RUNS / run
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    meta = json.loads((GRADS / "_meta.json").read_text())
    groups = {name: [tuple(span) for span in spans] for name, spans in meta["groups"].items()}
    n_params = int(meta["n_params"])
    precond, adam_eps_share, n_steps, refs = None, NAN, 0, [torch.zeros(n_params) for _ in REFS]
    if source is not None:
        payload = torch.load(run_dir / "trained.pt", map_location="cpu", weights_only=False)
        precond, adam_eps_share, n_steps = payload["precond"], payload["adam_eps_share"], int(payload["steps"])
        refs = [payload["refs"][name].float() for name in REFS]
        del payload

    halves, per_target, sketches, target_errors = {}, {}, {}, {}
    for target in patches:
        path = GRADS / target.label / f"{run}.pt"
        try:
            probe = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            target_errors[target.label] = f"{type(exc).__name__}: {exc}"[:300]
            probe = {"A": torch.zeros(n_params), "B": torch.zeros(n_params), "nA": 0, "nB": 0, "sumsq": 0.0,
                     "rows": [], "sketch": torch.zeros(0, SKETCH_DIM), "loss_scale": NAN, "train_mode": {},
                     "eval": {"eval_split": "error", "error": target_errors[target.label]}, "train_tiles": 0,
                     "seconds": NAN, "empty": 0, "nonfinite": 0, "retries": 0}
        halves[target.pid] = (probe.pop("A").float(), probe.pop("B").float())
        sketches[target.pid] = probe.pop("sketch").double()
        per_target[target.pid] = probe

    # per-batch cosine to G_s from the count sketches (linear, so mean sketches are sketches of the means);
    # the source's own batches use the leave-one-out mean
    if source is not None and sketches[source.pid].shape[0] > 1:
        own = sketches[source.pid]
        total = own.sum(dim=0)
        mean = total / own.shape[0]
        for target in patches:
            rows, rows_sketch = per_target[target.pid]["rows"], sketches[target.pid]
            for k, row in enumerate(rows):
                vector = rows_sketch[k]
                reference = (total - vector) / (own.shape[0] - 1) if target.pid == source.pid else mean
                denominator = float(vector.norm() * reference.norm())
                row["cos_src"] = float(vector.dot(reference)) / denominator if denominator > 0 else NAN
    batch_rows = [{"target": t.pid, "target_label": t.label, **row} for t in patches for row in per_target[t.pid]["rows"]]
    _write_csv(run_dir / "probe_batches.csv", batch_rows)

    rows_host = [v for p in patches for v in halves[p.pid]] + refs
    names, raw, pre, sign, coords = _grams(rows_host, len(patches), groups, precond)
    del rows_host, halves
    np.savez_compressed(
        run_dir / "gram.npz", raw=raw, pre=pre, sign=sign, coords=coords, groups=np.array(names),
        rows=np.array([f"{p.label}|{h}" for p in patches for h in "AB"] + list(REFS)),
        targets=np.array([p.pid for p in patches]),
        n_a=np.array([per_target[p.pid]["nA"] for p in patches]),
        n_b=np.array([per_target[p.pid]["nB"] for p in patches]),
    )
    batch_size = int(base_config().dl.batch_size)
    _derive(run, source, patches, per_target, names, raw, pre, sign, coords, batch_size, int(args.batches), run_dir)
    _atomic_json({
        "run": run, "source": source.pid if source else None, "targets": [p.pid for p in patches],
        **_pair_meta(args), "epochs": 0 if source is None else int(args.epochs), "batch_size": batch_size,
        "n_params": n_params, "steps": n_steps, "target_errors": target_errors,
        "adam_eps_dominated_share": adam_eps_share, "timings": {"gram_s": time.time() - started},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, run_dir / "done.json")
    print(f"[corr] {run} gram + tables done in {(time.time() - started) / 60:.1f} min "
          f"(target errors: {target_errors or 'none'})", flush=True)


def _derive(run, source, patches, per_target, names, raw, pre, sign, coords, batch_size, batches, run_dir) -> None:
    """targets.csv, target_groups.csv and pairs.csv from the Gram matrices and the per-batch rows."""
    n = len(patches)
    index = {p.pid: i for i, p in enumerate(patches)}
    si = index[source.pid] if source is not None else None
    weights_a = [_safe_div(per_target[p.pid]["nA"], per_target[p.pid]["nA"] + per_target[p.pid]["nB"]) for p in patches]
    weights_a = [w if np.isfinite(w) else 0.5 for w in weights_a]
    h, hp = raw.sum(axis=0), pre.sum(axis=0)
    hg = _means_gram(h, weights_a)
    s_total, n_coords = sign.sum(axis=0), max(int(coords.sum()), 1)
    ref = {name: n + r for r, name in enumerate(REFS)}
    rel = [_cross(h, t, t) for t in range(n)]

    rows = []
    for t, patch in enumerate(patches):
        info = per_target[patch.pid]
        count = info["nA"] + info["nB"]
        ab = _cross_dot(h, t, t)
        mean_sq = info["sumsq"] / max(count, 1)
        rel_full = 2 * rel[t] / (1 + rel[t]) if np.isfinite(rel[t]) and rel[t] > -1 else NAN
        row = {
            "run": run, "source": source.pid if source else "", "source_label": source.label if source else BASELINE,
            "source_scroll": source.scroll if source else "", "target": patch.pid, "target_label": patch.label,
            "target_scroll": patch.scroll, "target_fine_scan": int(patch.fine_scan),
            "same_patch": int(source is not None and patch.pid == source.pid),
            "same_scroll": int(source is not None and patch.scroll == source.scroll),
            "same_regime": int(source is not None and patch.fine_scan == source.fine_scan),
            "train_tiles": info["train_tiles"], "n_batches": count, "n_nonfinite": info["nonfinite"],
            "n_empty": info["empty"], "n_scale_retries": info["retries"], "probe_loss_scale": info["loss_scale"],
            "probe_seconds": info["seconds"],
            "loss_mean": float(np.mean([r["loss"] for r in info["rows"]])) if info["rows"] else NAN,
            "loss_sem": float(np.std([r["loss"] for r in info["rows"]], ddof=1) / math.sqrt(count)) if count > 1 else NAN,
            "raw_loss_mean": float(np.mean([r["raw_loss"] for r in info["rows"]])) if info["rows"] else NAN,
            "batch_grad_norm_rms": math.sqrt(mean_sq),
            "mean_grad_norm_unbiased": math.sqrt(ab) if ab > 0 else NAN,
            "noise_ratio": _safe_div(mean_sq - ab, ab),
            # McCandlish simple noise scale B_simple = tr(Sigma) / |G|^2 in samples
            "b_simple": _safe_div(batch_size * (mean_sq - ab), ab),
            "reliability_half": rel[t], "reliability_full": rel_full,
            "batches_for_rel90": _safe_div(batches * 0.9 * (1 - rel_full), rel_full * 0.1) if rel_full > 0 else NAN,
        }
        if si is not None:
            src_cross = _cross(h, t, si)
            row.update({
                "cos_src_cross": src_cross,
                "cos_src_full": _cos(hg, t, si),
                "cos_src_disatt": _safe_div(src_cross, math.sqrt(rel[t] * rel[si])) if rel[t] > 0 and rel[si] > 0 else NAN,
                "dot_src": _cross_dot(h, t, si),
                "dot_src_rel": _safe_div(_cross_dot(h, t, si), _cross_dot(h, si, si)),
                "cos_src_cross_adam": _cross(hp, t, si),
                "dot_src_rel_adam": _safe_div(_cross_dot(hp, t, si), _cross_dot(hp, si, si)),
                "sign_src": s_total[t, si] / n_coords,
            })
            for name in REFS:
                row[f"cos_{name}"] = _cos(hg, t, ref[name])
                row[f"sign_{name}"] = s_total[t, ref[name]] / n_coords
            # first-order loss change on t from the last real update (negative = the update helped t)
            row["dloss_u_last"] = float(hg[t, ref["u_last"]])
            row["dloss_u_last_rel"] = _safe_div(hg[t, ref["u_last"]], abs(hg[si, ref["u_last"]]))
            for name, column in (("cos_src", "batch_cos_src"), ("cos_g_last", "batch_cos_g_last"),
                                 ("cos_u_last", "batch_cos_u_last")):
                stats = _batch_stats([r.get(name, NAN) for r in info["rows"]])
                row.update({f"{column}_{key}": value for key, value in stats.items()})
        row.update(info["train_mode"])
        row.update(info["eval"])
        rows.append(row)
    _write_csv(run_dir / "targets.csv", rows)

    group_rows = []
    for gi, name in enumerate(names):
        hgi = _means_gram(raw[gi], weights_a)
        for t, patch in enumerate(patches):
            row = {"run": run, "target": patch.pid, "target_label": patch.label, "group": name,
                   "n_params": int(coords[gi]), "grad_share": _safe_div(hgi[t, t], hg[t, t]),
                   "reliability_half": _cross(raw[gi], t, t)}
            if si is not None:
                row.update({"cos_src_cross": _cross(raw[gi], t, si), "cos_g_last": _cos(hgi, t, ref["g_last"]),
                            "cos_u_last": _cos(hgi, t, ref["u_last"]),
                            "sign_src": sign[gi][t, si] / max(int(coords[gi]), 1)})
            group_rows.append(row)
    _write_csv(run_dir / "target_groups.csv", group_rows)

    pair_rows = []
    for t in range(n):
        for u in range(t, n):
            pair_rows.append({
                "run": run, "t": patches[t].pid, "u": patches[u].pid, "t_label": patches[t].label,
                "u_label": patches[u].label, "cos_cross": _cross(h, t, u), "cos_full": _cos(hg, t, u),
                "cos_cross_adam": _cross(hp, t, u), "sign": s_total[t, u] / n_coords, "dot": _cross_dot(h, t, u),
            })
    _write_csv(run_dir / "pairs.csv", pair_rows)


# ---- report --------------------------------------------------------------------------------------------------------------

def _finished_runs(plan) -> list[str]:
    pids = [p["pid"] for p in plan["patches"]]
    runs = []
    for run in [BASELINE, *(p["label"] for p in plan["patches"])]:
        path = RUNS / run / "done.json"
        if path.is_file() and json.loads(path.read_text()).get("targets") == pids:
            runs.append(run)
    return runs


def _group_contrast(s, groups, rng, n_perm=5000) -> dict:
    """mean off-diagonal similarity of same-group minus different-group pairs; label-permutation p-value (one-sided)."""
    groups = np.asarray(groups)
    iu = np.triu_indices(len(groups), 1)
    values = s[iu]
    ok = np.isfinite(values)

    def diff(g):
        same = (g[iu[0]] == g[iu[1]]) & ok
        other = (g[iu[0]] != g[iu[1]]) & ok
        if not same.any() or not other.any():
            return NAN, NAN, NAN
        return values[same].mean() - values[other].mean(), values[same].mean(), values[other].mean()

    observed, same_mean, other_mean = diff(groups)
    if not np.isfinite(observed):
        return {"n_same": 0, "mean_same": NAN, "mean_diff": NAN, "contrast": NAN, "p_perm": NAN}
    null = np.array([diff(rng.permutation(groups))[0] for _ in range(n_perm)])
    same_pairs = int(((groups[iu[0]] == groups[iu[1]]) & ok).sum())
    return {"n_same": same_pairs, "mean_same": same_mean, "mean_diff": other_mean, "contrast": observed,
            "p_perm": float((1 + np.sum(null >= observed)) / (1 + n_perm))}


def _distance(s):
    s = np.where(np.isfinite(s), s, 0.0)
    s = 0.5 * (s + s.T)
    d = np.clip((1.0 - s) / 2.0, 0.0, 1.0)
    np.fill_diagonal(d, 0.0)
    return d


def _clusters(s, k):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    return fcluster(linkage(squareform(_distance(s), checks=False), "average"), k, "maxclust")


def _cluster_table(s, stack, kmax, rng, n_null=200, n_boot=200) -> list[dict]:
    """per k: silhouette, within/between similarity, null p (symmetric shuffles of the matrix), bootstrap stability
    over source runs (ARI), and the normalised-Laplacian eigengap."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    n = s.shape[0]
    d = _distance(s)
    affinity = np.clip(np.where(np.isfinite(s), 0.5 * (s + s.T), 0.0), 0.0, None)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    inv = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    eig = np.sort(np.linalg.eigvalsh(np.eye(n) - inv[:, None] * affinity * inv[None, :]))
    iu = np.triu_indices(n, 1)
    rows = []
    for k in range(2, min(kmax, n - 1) + 1):
        labels = _clusters(s, k)
        if len(set(labels)) < 2:
            continue
        silhouette = float(silhouette_score(d, labels, metric="precomputed"))
        same = labels[iu[0]] == labels[iu[1]]
        values = 0.5 * (s + s.T)[iu]
        null = []
        for _ in range(n_null):
            shuffled = np.zeros_like(s)
            shuffled[iu] = rng.permutation(values)
            shuffled = shuffled + shuffled.T
            null_labels = _clusters(shuffled, k)
            if len(set(null_labels)) > 1:
                null.append(silhouette_score(_distance(shuffled), null_labels, metric="precomputed"))
        ari = []
        if stack is not None and stack.shape[0] > 2:
            for _ in range(n_boot):
                sample = np.nanmean(stack[rng.integers(0, stack.shape[0], stack.shape[0])], axis=0)
                ari.append(adjusted_rand_score(labels, _clusters(sample, k)))
        rows.append({
            "k": k, "silhouette": silhouette, "silhouette_null_mean": float(np.mean(null)) if null else NAN,
            "p_null": float((1 + np.sum(np.asarray(null) >= silhouette)) / (1 + len(null))) if null else NAN,
            "within_mean": float(np.nanmean(values[same])) if same.any() else NAN,
            "between_mean": float(np.nanmean(values[~same])) if (~same).any() else NAN,
            "bootstrap_ari_mean": float(np.mean(ari)) if ari else NAN,
            "bootstrap_ari_p10": float(np.percentile(ari, 10)) if ari else NAN,
            "eigengap": float(eig[k] - eig[k - 1]) if k < n else NAN,
            "sizes": "/".join(str(int((labels == c).sum())) for c in sorted(set(labels))),
        })
    return rows


def _bh(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    q = np.full(p.shape, NAN)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * ok.sum() / (np.arange(ok.sum()) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    values = np.empty(ok.sum())
    values[order] = ranked
    q[ok] = values
    return q


def _heatmap(matrix, rows, cols, path, title, order_rows=None, order_cols=None, center=0.0) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = np.asarray(matrix, dtype=np.float64)
    if order_rows is not None:
        m, rows = m[order_rows], [rows[i] for i in order_rows]
    if order_cols is not None:
        m, cols = m[:, order_cols], [cols[i] for i in order_cols]
    span = np.nanmax(np.abs(m - center)) if np.isfinite(m).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(cols) + 3), max(5, 0.32 * len(rows) + 2)))
    image = ax.imshow(m, cmap="RdBu_r", vmin=center - span, vmax=center + span, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _md_table(rows: list[dict], columns: list[str], digits: int = 3) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def build_report() -> None:
    import pandas as pd
    from scipy import stats

    plan = load_plan()
    patches = plan_patches(plan)
    labels = [p.label for p in patches]
    scroll = {p.label: p.scroll for p in patches}
    regime = {p.label: "fine" if p.fine_scan else "native" for p in patches}
    n = len(patches)
    runs = _finished_runs(plan)
    sources = [r for r in runs if r != BASELINE]
    REPORT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = [f"# runs_corr report ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    lines.append(f"- plan: {n} patches over {len(set(scroll.values()))} scrolls; {len(plan.get('skipped', {}))} skipped "
                 f"(see patches.csv)")
    lines.append(f"- finished runs: {len(sources)}/{n} source runs, baseline {'yes' if BASELINE in runs else 'no'}")
    if not runs:
        (REPORT / "report.md").write_text("\n".join(lines) + "\n\nno finished runs yet\n")
        print("[report] no finished runs yet", flush=True)
        return
    done = {r: json.loads((RUNS / r / "done.json").read_text()) for r in runs}
    any_done = done[runs[-1]]
    lines.append(f"- probe: {any_done['batches']} batches x {any_done['batch_size']} windows per target, eval "
                 f"{any_done['eval_batches']} batches; training {any_done['epochs']} epochs at lr {any_done['lr']}")
    eps_shares = [done[r]["adam_eps_dominated_share"] for r in sources]
    if eps_shares:
        lines.append(f"- AdamW sqrt(v_hat) < eps on a median {np.median(eps_shares):.1%} of coordinates at the end of "
                     "training (high = AdamW acts like SGD there; gradients are clipped while loss-scaled)")
    lines.append("")

    targets = pd.concat([pd.read_csv(RUNS / r / "targets.csv") for r in runs], ignore_index=True)
    targets.to_csv(REPORT / "all_targets.csv", index=False)
    groups_df = pd.concat([pd.read_csv(RUNS / r / "target_groups.csv") for r in runs], ignore_index=True)
    pairs = {r: pd.read_csv(RUNS / r / "pairs.csv") for r in runs}

    def matrix(column, run_list=None):
        run_list = sources if run_list is None else run_list
        out = np.full((len(run_list), n), NAN)
        for i, r in enumerate(run_list):
            frame = targets[targets["run"] == r].set_index("target_label").reindex(labels)
            if column in frame:
                out[i] = frame[column].to_numpy(dtype=np.float64)
        return out

    def pair_matrix(r, column="cos_cross"):
        out = np.full((n, n), NAN)
        position = {label: i for i, label in enumerate(labels)}
        frame = pairs[r]
        for t, u, v in zip(frame["t_label"], frame["u_label"], frame[column]):
            out[position[t], position[u]] = out[position[u], position[t]] = v
        return out

    # consensus affinity: target-pair cosine averaged over every trained model
    stack = np.stack([pair_matrix(r) for r in sources]) if sources else None
    consensus = np.nanmean(stack, axis=0) if stack is not None else pair_matrix(BASELINE)
    pd.DataFrame(consensus, index=labels, columns=labels).to_csv(REPORT / "consensus_cos_cross.csv")
    if stack is not None and len(sources) > 1:
        pd.DataFrame(np.nanstd(stack, axis=0), index=labels, columns=labels).to_csv(REPORT / "consensus_cos_cross_std.csv")
    if BASELINE in runs:
        pd.DataFrame(pair_matrix(BASELINE), index=labels, columns=labels).to_csv(REPORT / "init_cos_cross.csv")

    leaf_order = list(range(n))
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        leaf_order = list(leaves_list(linkage(squareform(_distance(consensus), checks=False), "average")))
    except Exception:
        pass
    _heatmap(consensus, labels, labels, REPORT / "consensus_cos_cross.png",
             "consensus cross-half gradient cosine (mean over trained models)", leaf_order, leaf_order)

    directional = {}
    if sources:
        for column in ("cos_src_cross", "cos_src_disatt", "cos_src_full", "cos_src_cross_adam", "dot_src_rel",
                       "sign_src", "cos_g_last", "cos_g_ema", "cos_u_last", "dloss_u_last_rel",
                       "eval_roc_auc", "eval_pr_auc", "eval_char_ap", "eval_loss"):
            m = matrix(column)
            directional[column] = m
            pd.DataFrame(m, index=sources, columns=labels).to_csv(REPORT / f"matrix_{column}.csv")
        if BASELINE in runs:
            for column in ("eval_roc_auc", "eval_pr_auc", "eval_char_ap", "eval_loss"):
                delta = directional[column] - matrix(column, [BASELINE])
                directional[f"delta_{column}"] = delta
                pd.DataFrame(delta, index=sources, columns=labels).to_csv(REPORT / f"matrix_delta_{column}.csv")
        row_order = [sources.index(labels[i]) for i in leaf_order if labels[i] in sources]
        for column, title in (("cos_src_cross", "cos(G_t, G_s) cross-half; row = source model"),
                              ("cos_g_last", "cos(G_t, last training step of s)"),
                              ("delta_eval_char_ap", "eval character AP of s-model on t minus init model")):
            if column in directional:
                _heatmap(directional[column], sources, labels, REPORT / f"matrix_{column}.png", title, row_order, leaf_order)

    # reliability: is --batches enough?
    lines.append("## Probe reliability")
    rel = targets.groupby("target_label")[["reliability_half", "reliability_full", "batches_for_rel90", "b_simple"]].median()
    rel.to_csv(REPORT / "reliability_by_target.csv")
    lines.append(f"split-half reliability of a target mean gradient (median over runs and targets): "
                 f"{targets['reliability_half'].median():.3f} (full-length {targets['reliability_full'].median():.3f}); "
                 f"batches needed for 0.9: median {targets['batches_for_rel90'].median():.0f}, "
                 f"90th percentile {targets['batches_for_rel90'].quantile(0.9):.0f}. "
                 f"targets below 0.5 full reliability: "
                 f"{', '.join(rel.index[rel['reliability_full'] < 0.5]) or 'none'}.")
    lines.append(f"median simple noise scale B_simple: {targets['b_simple'].median():.0f} windows")
    lines.append("")

    cross = targets[(targets["run"] != BASELINE) & (targets["same_patch"] == 0)].copy() if sources else targets.iloc[0:0]
    if len(cross):
        cross["batch_q"] = _bh(cross["batch_cos_src_p"].to_numpy())
        cross.to_csv(REPORT / "pairs_significance.csv", index=False)
        helpful = cross[(cross["batch_cos_src_mean"] > 0) & (cross["batch_q"] < 0.05)]
        harmful = cross[(cross["batch_cos_src_mean"] < 0) & (cross["batch_q"] < 0.05)]
        lines.append("## Directional alignment (source model s, target t != s)")
        lines.append(f"{len(cross)} (s, t) pairs; per-batch cos(g_t, G_s) significantly positive (BH q<0.05): "
                     f"{len(helpful)}, significantly negative: {len(harmful)}.")
        same = cross[cross["same_scroll"] == 1]
        other = cross[cross["same_scroll"] == 0]
        if len(same) and len(other):
            test = stats.mannwhitneyu(same["cos_src_cross"].dropna(), other["cos_src_cross"].dropna(), alternative="greater")
            lines.append(f"cos_src_cross same-scroll mean {same['cos_src_cross'].mean():.3f} (n={len(same)}) vs "
                         f"other-scroll {other['cos_src_cross'].mean():.3f} (n={len(other)}), Mann-Whitney p={test.pvalue:.3g}")
        selfs = targets[targets["same_patch"] == 1].set_index("source_label")
        dominance = []
        for r in sources:
            frame = cross[cross["source_label"] == r]
            if r not in selfs.index or not len(frame):
                continue
            self_value = selfs.loc[r, "cos_src_cross"]
            values = frame["cos_src_cross"].dropna().to_numpy()
            dominance.append({"source": r, "self_cross": self_value, "best_other": values.max() if values.size else NAN,
                              "best_other_target": frame.loc[frame["cos_src_cross"].idxmax(), "target_label"] if values.size else "",
                              "mean_other": values.mean() if values.size else NAN,
                              "self_rank": int(1 + np.sum(values > self_value))})
        if dominance:
            pd.DataFrame(dominance).to_csv(REPORT / "self_dominance.csv", index=False)
            lines.append(f"self (split-half) beats every other target in {sum(d['self_rank'] == 1 for d in dominance)}"
                         f"/{len(dominance)} runs (self_dominance.csv)")
        columns = ["source_label", "target_label", "source_scroll", "target_scroll", "cos_src_cross", "cos_src_disatt",
                   "batch_cos_src_mean", "batch_q", "cos_g_last", "dloss_u_last_rel"]
        if "eval_char_ap" in cross:
            columns.append("eval_char_ap")
        ranked = cross.sort_values("cos_src_cross")
        lines.append("\nmost conflicting pairs:\n")
        lines.append(_md_table(ranked.head(15).to_dict("records"), columns))
        lines.append("\nmost aligned cross-patch pairs:\n")
        lines.append(_md_table(ranked.tail(15).iloc[::-1].to_dict("records"), columns))
        lines.append("")

    # hypothesis: patch similarity has scroll / regime structure
    lines.append("## Group structure tests (label permutation, one-sided: same group more similar)")
    test_rows = []
    matrices = {"consensus_cos_cross": (consensus, labels)}
    if sources and len(sources) > 1:
        idx = [labels.index(r) for r in sources]
        for column in ("cos_src_cross", "cos_g_last", "dot_src_rel", "delta_eval_char_ap", "delta_eval_pr_auc"):
            if column in directional:
                square = directional[column][:, idx]
                matrices[f"sym_{column}"] = (0.5 * (square + square.T), sources)
    for name, (m, names) in matrices.items():
        for grouping, mapping in (("scroll", scroll), ("regime", regime)):
            result = _group_contrast(m, [mapping[x] for x in names], rng)
            test_rows.append({"matrix": name, "grouping": grouping, **result})
    pd.DataFrame(test_rows).to_csv(REPORT / "group_tests.csv", index=False)
    lines.append(_md_table(test_rows, ["matrix", "grouping", "n_same", "mean_same", "mean_diff", "contrast", "p_perm"]))
    lines.append("")

    if sources and len(sources) > 2:
        idx = [labels.index(r) for r in sources]
        square = directional["cos_src_cross"][:, idx]
        off = ~np.eye(len(idx), dtype=bool)
        a, b = square[off], square.T[off]
        ok = np.isfinite(a) & np.isfinite(b)
        lines.append("## Consistency checks")
        if ok.sum() > 3:
            lines.append(f"- asymmetry: Spearman(cos_src_cross[s,t], [t,s]) = {stats.spearmanr(a[ok], b[ok])[0]:.3f}")
        for proxy in ("cos_src_cross", "cos_g_last", "dot_src_rel", "dloss_u_last_rel"):
            for outcome in ("delta_eval_char_ap", "delta_eval_pr_auc", "delta_eval_loss"):
                if proxy in directional and outcome in directional:
                    x, y = directional[proxy].ravel(), directional[outcome].ravel()
                    mask = np.isfinite(x) & np.isfinite(y)
                    self_mask = np.array([[labels[j] == s for j in range(n)] for s in sources]).ravel()
                    mask &= ~self_mask
                    if mask.sum() > 3:
                        rho, p = stats.spearmanr(x[mask], y[mask])
                        lines.append(f"- gradient proxy vs transfer: Spearman({proxy}, {outcome}) = {rho:.3f} (p={p:.2g}, "
                                     f"n={int(mask.sum())})")
        lines.append("")

    # clustering of the consensus affinity
    lines.append("## Clustering of the consensus affinity (average linkage on (1 - cos) / 2)")
    table = _cluster_table(consensus, stack, kmax=min(10, n - 1), rng=rng) if n > 3 else []
    pd.DataFrame(table).to_csv(REPORT / "clusters_consensus.csv", index=False)
    if table:
        lines.append(_md_table(table, ["k", "sizes", "silhouette", "silhouette_null_mean", "p_null", "within_mean",
                                       "between_mean", "bootstrap_ari_mean", "bootstrap_ari_p10", "eigengap"]))
        assignments = {f"k{row['k']}": _clusters(consensus, row["k"]) for row in table}
        frame = pd.DataFrame(assignments, index=labels)
        frame.insert(0, "scroll", [scroll[x] for x in labels])
        frame.insert(1, "regime", [regime[x] for x in labels])
        frame.to_csv(REPORT / "cluster_assignments.csv")
        best = max(table, key=lambda row: row["silhouette"])
        lines.append(f"\nbest silhouette at k={best['k']}; members:\n")
        for c in sorted(set(assignments[f"k{best['k']}"])):
            members = [x for x, a in zip(labels, assignments[f"k{best['k']}"]) if a == c]
            lines.append(f"- cluster {c}: " + ", ".join(f"{x} ({scroll[x]})" for x in members))
    lines.append("")

    # layer view: where do cross-patch gradients agree or conflict
    if sources and "cos_src_cross" in groups_df:
        merged = groups_df.merge(targets[["run", "target", "same_patch", "same_scroll"]], on=["run", "target"])
        merged = merged[merged["run"] != BASELINE]
        merged["relation"] = np.where(merged["same_patch"] == 1, "self",
                                      np.where(merged["same_scroll"] == 1, "same_scroll", "other_scroll"))
        layer = merged.groupby(["group", "relation"])["cos_src_cross"].mean().unstack()
        layer["n_params"] = merged.groupby("group")["n_params"].first()
        layer["grad_share"] = merged.groupby("group")["grad_share"].mean()
        layer = layer.sort_values("grad_share", ascending=False)
        layer.to_csv(REPORT / "layer_alignment.csv")
        lines.append("## Alignment by parameter group (mean cross-half cosine)")
        lines.append(_md_table(layer.reset_index().head(20).to_dict("records"),
                               ["group", "n_params", "grad_share", "self", "same_scroll", "other_scroll"]))
        lines.append("")

    skipped = plan.get("skipped", {})
    if skipped:
        lines.append("## Skipped patches")
        lines += [f"- {pid}: {reason}" for pid, reason in skipped.items()]
    (REPORT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[report] {REPORT / 'report.md'} ({len(sources)} source runs)", flush=True)


# ---- orchestration -------------------------------------------------------------------------------------------------------

def _run_done(run: str, pids: list[int], args) -> bool:
    path = RUNS / run / "done.json"
    if not path.is_file():
        return False
    done = json.loads(path.read_text())
    return (done.get("targets") == pids and done.get("batches") == args.batches
            and done.get("eval_batches") == args.eval_batches
            and int(done.get("max_samples", 0)) == int(args.max_samples or 0)
            and done.get("epochs") == (0 if run == BASELINE else args.epochs)
            and abs(float(done.get("lr", -1)) - args.lr) < 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated source run labels (default: init + every patch)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batches", type=int, default=48, help="probe batches per target (gradients)")
    parser.add_argument("--eval-batches", type=int, default=16, help="eval-mode batches per target")
    parser.add_argument("--probe-workers", type=int, default=8)
    parser.add_argument("--cue-workers", type=int, default=4)
    parser.add_argument("--no-baseline", action="store_true", help="skip the untrained init run")
    parser.add_argument("--dry-run", action="store_true", help="check eligibility and print the plan, no training")
    parser.add_argument("--report", action="store_true", help="only rebuild the report")
    parser.add_argument("--root", default="runs_corr", help="output folder (default runs_corr)")
    parser.add_argument("--max-samples", type=int, default=0, help="override windows per epoch (smoke tests only)")
    parser.add_argument("--daemon", action="store_true", help="accepted for compatibility; the scheduler is the default")
    parser.add_argument("--no-wait", action="store_true", help="daemon: do not wait for the downloads to end")
    parser.add_argument("--poll", type=float, default=60.0, help="daemon: seconds between scheduler passes")
    parser.add_argument("--max-failures", type=int, default=3, help="daemon: attempts per run before giving up")
    parser.add_argument("--train-timeout", type=float, default=8.0, help="daemon: hours before a training run is killed")
    parser.add_argument("--probe-timeout", type=float, default=16.0, help="daemon: hours before a probe run is killed")
    parser.add_argument("--prepare-once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="probe", choices=("train", "probe", "gram"), help=argparse.SUPPRESS)
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(REPO)
    global MAX_SAMPLES
    MAX_SAMPLES = args.max_samples or None
    set_root(args.root)
    if args.worker:
        {"train": train_worker, "probe": probe_worker, "gram": gram_worker}[args.mode](args.worker, args)
        return
    if args.report:
        build_report()
        return
    if args.prepare_once:
        prepare(all_patches(), args.cue_workers)
        return
    if args.dry_run:
        patches = all_patches()
        eligible, skipped = eligibility(patches, build_init=False)
        print(f"[corr] {len(eligible)}/{len(patches)} patches eligible:", flush=True)
        for p in eligible:
            print(f"  {p.label:<28} {p.pid}  {p.scroll}{'  (fine scan)' if p.fine_scan else ''}")
        for pid, reason in skipped.items():
            print(f"  SKIP {pid}: {reason}")
        return
    daemon(args)


def _worker_command(run: str, mode: str, args) -> list[str]:
    return [sys.executable, "-u", __file__, "--worker", run, "--mode", mode, "--epochs", str(args.epochs),
            "--lr", repr(args.lr), "--batches", str(args.batches), "--eval-batches", str(args.eval_batches),
            "--probe-workers", str(args.probe_workers), "--root", str(ROOT), "--max-samples", str(args.max_samples)]


def _run_child(run: str, mode: str, args, wait: bool):
    """one process per run: the model, CUDA context, chunk caches and memmaps are all released on exit."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{mode}_{run}.log", "a", encoding="utf-8")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {mode} {run} =====\n")
    log.flush()
    process = subprocess.Popen(_worker_command(run, mode, args), stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    if not wait:
        return process, log
    code = process.wait()
    log.close()
    return code


def _log_tail(path: Path, lines: int = 25) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-lines:])
    except OSError:
        return ""


def _downloads_running() -> bool:
    result = subprocess.run(["pgrep", "-f", "assemble_training_segments.py|generate_surface_supervision.py"],
                            capture_output=True, text=True)
    return bool(result.stdout.strip())


def _kill_group(process) -> None:
    import signal

    with contextlib.suppress(Exception):
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    with contextlib.suppress(Exception):
        os.killpg(process.pid, signal.SIGKILL)


def _freeze_plan(eligible, skipped, sources) -> None:
    _atomic_json({"patches": [dataclasses.asdict(p) for p in eligible], "sources": list(sources),
                  "skipped": {str(k): v for k, v in skipped.items()},
                  "created": time.strftime("%Y-%m-%d %H:%M:%S")}, PLAN)


def _external(pattern: str) -> bool:
    """a matching corr_runs child that this daemon did not start (left over from a restarted daemon)."""
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return bool(result.stdout.strip())


def daemon(args) -> None:
    """long-running scheduler. While patches arrive: build their cues / norm / anchors (CPU child) and train each
    eligible patch as soon as it is ready (GPU child). Once the downloads have ended and every eligible patch is
    trained, freeze the plan, probe every trained model (and the init baseline) against all of it, then report.
    Failed children are retried up to --max-failures times; the daemon itself never exits on a child error."""
    import traceback

    for folder in (RUNS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / "daemon_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    state.setdefault("failures", {})
    state.setdefault("errors", {})
    state.setdefault("prep_attempts", {})

    def save_state():
        temporary = state_path.with_suffix(f".tmp{os.getpid()}")
        temporary.write_text(json.dumps(state, indent=2))
        os.replace(temporary, state_path)

    def log(message):
        print(f"[daemon {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    timeouts = {"train": args.train_timeout * 3600, "probe": args.probe_timeout * 3600, "gram": 3 * 3600,
                "report": 3600}
    prep = gpu = None  # (process, log handle, key, started)
    stable = 0
    last_status = ""
    if PLAN.is_file() and not state.get("plan_frozen"):
        log("an older plan.json exists; it is replaced when the probe phase starts")
    while True:
        try:
            if prep is not None and prep[0].poll() is not None:
                prep[1].close()
                log(f"prepare finished (exit {prep[0].returncode})")
                prep = None
            if gpu is not None:
                process, handle, key, started = gpu
                code = process.poll()
                if code is None and time.time() - started > timeouts[key.split(":")[0]]:
                    log(f"{key} exceeded its timeout; killing it")
                    _kill_group(process)
                    code = process.poll() if process.poll() is not None else -9
                if code is not None:
                    handle.close()
                    mode, run = key.split(":", 1)
                    minutes = (time.time() - started) / 60
                    if code == 0:
                        state["failures"].pop(key, None)
                        if mode == "report":
                            state["reported"] = True
                        log(f"{key} ok ({minutes:.1f} min)")
                    else:
                        state["failures"][key] = state["failures"].get(key, 0) + 1
                        state["errors"][key] = _log_tail(LOGS / f"{mode}_{run}.log")
                        log(f"{key} FAILED (exit {code}, attempt {state['failures'][key]}/{args.max_failures}, "
                            f"{minutes:.1f} min); see {LOGS / f'{mode}_{run}.log'}")
                    save_state()
                    gpu = None
                    gc.collect()

            patches = all_patches()
            config = base_config()
            ready = [p for p in patches if not basic_problems(p, config)]
            prep_todo = sorted({str(job[0]) for job in _missing_cues(ready, config)}
                               | {str(p.pid) for p in ready if norm_problems(p, config)})
            prep_key = ",".join(prep_todo)
            # a frozen plan needs no new inputs; leave newly downloaded segments to their assembler
            if prep_todo and prep is None and not state.get("plan_frozen") \
                    and state["prep_attempts"].get(prep_key, 0) < 3 \
                    and not _external("corr_runs.*py .*--prepare-once"):
                state["prep_attempts"][prep_key] = state["prep_attempts"].get(prep_key, 0) + 1
                save_state()
                handle = open(LOGS / "prepare.log", "a", encoding="utf-8")
                handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} prepare {prep_key} =====\n")
                handle.flush()
                process = subprocess.Popen(
                    [sys.executable, "-u", __file__, "--prepare-once", "--cue-workers", str(args.cue_workers),
                     "--root", str(ROOT)], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                prep = (process, handle, "prepare", time.time())
                log(f"prepare started for {len(prep_todo)} patch(es): {prep_key}")

            eligible, skipped = eligibility(patches, build_init=False)
            write_patches_csv(patches, eligible, skipped)
            labels = [p.label for p in eligible if not wanted or p.label in wanted]
            trained = [label for label in labels if _trained_ok(label, args)]

            orphan = gpu is None and _external("corr_runs.*py --worker ")
            if orphan:
                stable = 0
            if gpu is None and not orphan and not state.get("plan_frozen"):
                todo = [label for label in labels if label not in trained
                        and state["failures"].get(f"train:{label}", 0) < args.max_failures]
                if todo:
                    process, handle = _run_child(todo[0], "train", args, wait=False)
                    gpu = (process, handle, f"train:{todo[0]}", time.time())
                    log(f"train {todo[0]} started ({len(trained)} trained, {len(todo)} to train, "
                        f"{len(eligible)}/{len(patches)} eligible)")
                else:
                    quiet = (args.no_wait or not _downloads_running()) and prep is None and not (
                        prep_todo and state["prep_attempts"].get(prep_key, 0) < 3)
                    stable = stable + 1 if quiet else 0
                    # a trained patch that lost eligibility (e.g. a damaged cache) gets 3 h to be repaired first
                    eligible_labels = {p.label for p in eligible}
                    stranded = [p.label for p in patches if p.label not in eligible_labels
                                and (not wanted or p.label in wanted) and _trained_ok(p.label, args)]
                    if stranded and stable == 3:
                        log(f"waiting before freezing: trained but now ineligible: {stranded}")
                    if stable >= 3 and len(eligible) >= 2 and (not stranded or stable * args.poll >= 3 * 3600):
                        sources = ([] if args.no_baseline else [BASELINE]) + trained
                        _freeze_plan(eligible, skipped, sources)
                        state["plan_frozen"] = True
                        state["reported"] = False
                        save_state()
                        log(f"downloads ended and every eligible patch is trained: plan frozen with {len(eligible)} "
                            f"targets and {len(sources)} source models; probe phase begins")

            if gpu is None and not orphan and state.get("plan_frozen"):
                plan = load_plan()
                pids = [p["pid"] for p in plan["patches"]]
                sources = list(plan["sources"])
                targets = [p["label"] for p in plan["patches"]]
                probe_todo = [t for t in targets if state["failures"].get(f"probe:{t}", 0) < args.max_failures
                              and not all(_pair_ok(t, s, args) for s in sources)]
                gram_todo = [s for s in sources if state["failures"].get(f"gram:{s}", 0) < args.max_failures
                             and not _run_done(s, pids, args)]
                if probe_todo:
                    process, handle = _run_child(probe_todo[0], "probe", args, wait=False)
                    gpu = (process, handle, f"probe:{probe_todo[0]}", time.time())
                    log(f"probe target {probe_todo[0]} started ({len(targets) - len(probe_todo)}/{len(targets)} "
                        f"targets probed under {len(sources)} models)")
                elif gram_todo:
                    process, handle = _run_child(gram_todo[0], "gram", args, wait=False)
                    gpu = (process, handle, f"gram:{gram_todo[0]}", time.time())
                    log(f"gram {gram_todo[0]} started ({len(sources) - len(gram_todo)}/{len(sources)} done)")
                elif not state.get("reported") and state["failures"].get("report:all", 0) < args.max_failures:
                    handle = open(LOGS / "report_all.log", "a", encoding="utf-8")
                    process = subprocess.Popen([sys.executable, "-u", __file__, "--report", "--root", str(ROOT)],
                                               stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    gpu = (process, handle, "report:all", time.time())
                    log("building the report")
                elif prep is None:
                    log(f"all done; report at {REPORT / 'report.md'}")
                    return

            status = (f"eligible {len(eligible)}/{len(patches)}, trained {len(trained)}, gpu "
                      f"{gpu[2] if gpu else 'adopted job' if orphan else 'idle'}, prepare {'running' if prep else 'idle'}, "
                      f"downloads {'running' if _downloads_running() else 'ended'}, "
                      f"plan {'frozen' if state.get('plan_frozen') else 'open'}")
            if status != last_status:
                log(status)
                last_status = status
        except Exception:
            log("scheduler error (continuing):\n" + traceback.format_exc())
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
