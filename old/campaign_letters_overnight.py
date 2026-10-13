"""Sequential campaign-41 AND-mask retraining and PHerc0846A transfer experiments."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import campaign_archs_29 as campaign29
import campaign_archs_31 as campaign31
import campaign_archs_41 as campaign41
import campaign_finetune_letters as letters
import mae_pretrain_letters as mae

ROOT = Path(__file__).resolve().parent
JOB = "letters_mae_overnight"
MODEL_ROOT = ROOT / "models" / JOB
LOG_ROOT = ROOT / f"runs_{JOB}"
INPUT_ROOT = ROOT / "_ves_tmp" / JOB
MAE_WEIGHTS = ROOT / "models" / f"{mae.NAME}.pth"
SUPERVISED_WEIGHTS = MODEL_ROOT / "and_mask" / "final.pth"
TRAIN_IDS = tuple(letters.FLAT_OFFSETEXTRA)
EXTRA_SOURCE = "auto_grown_20261002131432623_b_abf"
EXTRA_ID = 20261002131432
VIS_IDS = [*TRAIN_IDS, letters.FLAT_FORWARD[3], EXTRA_ID]
COMPONENT_FOLDS = 5
FINE_TUNES = (
    "baseline", *(f"loo_{sid}" for sid in TRAIN_IDS), "encoder_1e6", "geometric", "bundled",
    *(f"letters_fold_{index + 1}" for index in range(COMPONENT_FOLDS)),
)


def supervised_config():
    test = campaign41._test("and_mask", campaign41.AND_MASK)
    config = campaign41.build_config(test)
    config.init_weights = str(MAE_WEIGHTS)
    config.model_dir = str(SUPERVISED_WEIGHTS.parent)
    config.save_final = str(SUPERVISED_WEIGHTS)
    config.exp_name = "and_mask_mae_0846a"
    config.tra.log_dir = str(LOG_ROOT / "supervised")
    config.data.vis_scroll_ids = []
    config.data.label_free_vis_scroll_ids = []
    config.data.vis_preload_persistent = False
    config.tra.eval_int = config.tra.test_int = config.tra.probe_int = 999999
    config.tra.eval_int_scrolls = 0
    config.tra.test_on_final = False
    config.tra.save_vis = False
    config.tra.disable_figures = True
    config.data.inklabel_dir = str(INPUT_ROOT / "inklabels")
    config.data.train_mask_dir = str(INPUT_ROOT / "train_masks" / "supervised")
    config.run_name = "and_mask"
    return config


def finetune_config(tid):
    if tid not in FINE_TUNES:
        raise ValueError(f"Unknown fine-tune: {tid}")
    config = letters.build_config(letters._test("refine_baseline_m1", str(SUPERVISED_WEIGHTS)))
    config.model_dir = str(MODEL_ROOT / "finetune" / tid)
    config.save_final = str(Path(config.model_dir) / "final.pth")
    config.exp_name = config.run_name = tid
    config.tra.log_dir = str(LOG_ROOT / "finetune")
    config.data.inklabel_dir = str(INPUT_ROOT / "inklabels")
    config.data.train_mask_dir = str(INPUT_ROOT / "train_masks" / tid)
    config.data.vis_scroll_ids = list(VIS_IDS)
    config.data.label_free_vis_scroll_ids = [letters.FLAT_FORWARD[3], EXTRA_ID]
    config.data.surface_window_offset_by_scroll = {sid: -1 for sid in VIS_IDS}
    config.data.vis_preload_persistent = False
    config.data.ram_safe_vis = True
    config.tra.eval_int_scrolls = len(VIS_IDS)
    config.tra.disable_figures = False
    config.tra.patience = config.tra.n_epochs
    config.data.validation_scroll_ids = None
    if tid.startswith("loo_"):
        heldout = int(tid.removeprefix("loo_"))
        domain = f"heldout_{heldout}"
        config.data.train_scroll_dict = {
            "pherc0846a": [sid for sid in TRAIN_IDS if sid != heldout],
            "pherc0009b": [letters.PHERC0009B_ID], domain: [heldout],
        }
        config.data.train_scroll_weights = [5, 1, 1]
        config.data.holdout_domains = [domain]
        config.data.train_only_scroll_ids = [sid for sid in TRAIN_IDS if sid != heldout]
        config.data.validation_scroll_ids = [heldout]
    elif tid.startswith("letters_fold_"):
        config.data.train_only_scroll_ids = []
        config.data.validation_scroll_ids = list(TRAIN_IDS)
    elif tid == "encoder_1e6":
        config.tra.freeze_prefixes = ()
        config.tra.encoder_prefixes = letters.FREEZE_PREFIXES
        config.tra.encoder_lr = 1e-6
    if tid in ("geometric", "bundled"):
        config.dl.data_aug = True
        config.dl.flip_prob = config.dl.rotation_prob = 0.5
    if tid == "bundled":
        for key, value in letters.campaign40.DROPOUT_AUGS.items():
            section, name = key.split(".")
            setattr(getattr(config, section), name, value)
        config.tra.rsc_prob = 0.33
        config.data.native_noise = campaign41.DEGRADED["data.native_noise"]
        config.data.native_noise_ids = [*TRAIN_IDS, letters.PHERC0009B_ID]
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    return config


def _link(link, source):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise ValueError(f"Existing alias points elsewhere: {link}")
    elif link.exists():
        raise FileExistsError(link)
    else:
        link.symlink_to(source.resolve())


def prepare_extra(source_name=EXTRA_SOURCE, alias_id=EXTRA_ID):
    import cv2
    import numpy as np
    import zarr
    from utils.norm import compute_norm, ensure_surface_anchors, load_cached_norm

    zarr_root = Path(os.getenv("VESUVIUS_ZARR_PATH", str(ROOT / "ves_zarrs2")))
    source = zarr_root / f"{source_name}.zarr"
    source_mask = ROOT / "masks" / "extras" / f"{source_name}.png"
    if not source.is_dir() or not source_mask.is_file():
        raise FileNotFoundError(f"Missing extra inference input: {source}")
    _link(zarr_root / f"{alias_id}.zarr", source)
    _link(ROOT / "masks" / f"{alias_id}.png", source_mask)
    mask = cv2.imread(str(source_mask), 0)
    if mask is None or mask.shape != tuple(zarr.open(str(source), mode="r").shape[1:]):
        raise ValueError("Extra mask and volume frames differ")
    surface = ROOT / "surface_labels" / str(alias_id)
    if not surface.exists():
        surface.mkdir(parents=True)
        np.save(surface / "depth.npy", np.where(mask > 0, 13, 255).astype(np.uint8))
        np.save(surface / "confidence.npy", np.where(mask > 0, 255, 0).astype(np.uint8))
        (surface / "metadata.json").write_text(json.dumps({"flat_depth": 13, "source": source_name}))
    else:
        depth = np.load(surface / "depth.npy")
        if depth.shape != mask.shape or not np.all(depth[mask > 0] == 13):
            raise ValueError("Existing extra alias does not have the expected flat surface")
    if load_cached_norm(alias_id) is None:
        compute_norm(alias_id, str(zarr_root))
    ensure_surface_anchors([alias_id], str(zarr_root))


def component_assignments(label, unit=16, folds=COMPONENT_FOLDS, start=0):
    import cv2
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from utils.dataloader import DataManager, split_extra_labels

    regular, _ = split_extra_labels(label)
    count, raw_components, stats, centroids = cv2.connectedComponentsWithStats(regular.astype(np.uint8), 8)
    kept = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 8) + 1
    if len(kept) < 2:
        raise ValueError("A component-fold patch must have at least two retained components")
    grid = DataManager._build_character_grid(regular.astype(np.float32), unit, 8)
    present = set(np.unique(grid)) - {0}
    if present != set(range(1, len(kept) + 1)):
        raise ValueError("Some components disappear on the target grid; inspect labels before folding")
    order = np.random.default_rng(41 + start).permutation(len(kept)) + 1
    fold_by_component = np.full(len(kept) + 1, -1, dtype=np.int32)
    for index, component in enumerate(order):
        fold_by_component[component] = (start + index) % folds
    nearest = distance_transform_edt(grid == 0, return_distances=False, return_indices=True)
    owners = grid[nearest[0], nearest[1]]
    owner_folds = fold_by_component[owners]
    masks, inventory = [], []
    height, width = label.shape
    for fold in range(folds):
        assignment = np.zeros_like(label, dtype=np.uint8)
        cells = (owner_folds != fold).astype(np.uint8)
        assignment[:grid.shape[0] * unit, :grid.shape[1] * unit] = np.repeat(np.repeat(cells, unit, 0), unit, 1) * 255
        _, aligned = DataManager._exclude_characters_crossing_split(grid, assignment > 0, unit)
        if not np.array_equal(assignment > 0, aligned):
            raise ValueError("Component partition was changed by the loader's split alignment")
        for component, raw_id in enumerate(kept, start=1):
            pixels = raw_components == raw_id
            if fold_by_component[component] == fold and np.any(assignment[pixels] > 0):
                raise ValueError("Withheld component shares a target cell with training ink")
        masks.append(assignment)
    for component, raw_id in enumerate(kept, start=1):
        inventory.append({"component": component, "fold": int(fold_by_component[component]) + 1,
                          "pixels": int(stats[raw_id, cv2.CC_STAT_AREA]),
                          "bbox": stats[raw_id, :4].tolist(), "centroid": centroids[raw_id].tolist()})
    return masks, inventory


def prepare_inputs():
    import cv2
    import numpy as np

    if (INPUT_ROOT / "inputs.json").is_file():
        verify_snapshot()
        return
    prepare_extra()
    ink_dir = INPUT_ROOT / "inklabels"
    ink_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    ids = set(campaign41.training_scroll_ids()) | set(TRAIN_IDS) | {letters.PHERC0009B_ID}
    for sid in sorted(ids):
        source = ROOT / letters.INKLABEL_DIR / f"{sid}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = ink_dir / source.name
        shutil.copyfile(source, destination)
        sources[str(destination.relative_to(INPUT_ROOT))] = hashlib.sha256(destination.read_bytes()).hexdigest()
    for tid in ("supervised", *FINE_TUNES):
        folder = INPUT_ROOT / "train_masks" / tid
        folder.mkdir(parents=True, exist_ok=True)
        selected = campaign41.training_scroll_ids() if tid == "supervised" else [*TRAIN_IDS, letters.PHERC0009B_ID]
        for sid in selected:
            source = ROOT / "train_masks" / f"{sid}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copyfile(source, folder / source.name)
        if tid.startswith("loo_"):
            heldout = int(tid.removeprefix("loo_"))
            label = cv2.imread(str(ink_dir / f"{heldout}.png"), 0)
            if not cv2.imwrite(str(folder / f"{heldout}.png"), np.zeros_like(label)):
                raise OSError(f"Could not write heldout split for {heldout}")
    inventory = {}
    start = 0
    for sid in TRAIN_IDS:
        label = cv2.imread(str(ink_dir / f"{sid}.png"), 0)
        masks, components = component_assignments(label, start=start)
        inventory[str(sid)] = components
        start += len(components)
        for index, mask in enumerate(masks):
            destination = INPUT_ROOT / "train_masks" / f"letters_fold_{index + 1}" / f"{sid}.png"
            if not cv2.imwrite(str(destination), mask):
                raise OSError(destination)
    for path in (INPUT_ROOT / "train_masks").glob("*/*.png"):
        sources[str(path.relative_to(INPUT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"files": sources, "components": inventory, "component_count": start,
               "mae_exposure": "all-six discovery MAE; supervised holdouts only", "fine_tunes": list(FINE_TUNES)}
    (INPUT_ROOT / "inputs.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[overnight] snapshotted labels and {start} components into {COMPONENT_FOLDS} folds", flush=True)


def verify_snapshot():
    payload = json.loads((INPUT_ROOT / "inputs.json").read_text())
    for relative, expected in payload["files"].items():
        if hashlib.sha256((INPUT_ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Overnight snapshot changed: {relative}")
    return payload


def _config_payload(config):
    return json.loads(json.dumps(vars(config), default=vars))


def checkpoint_check(path, config, allow_mae=False):
    import torch
    from utils.model import create_model

    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    clean = {key.removeprefix("module.").removeprefix("_orig_mod."): value for key, value in state.items()}
    cpu_config = copy.deepcopy(config)
    cpu_config.device = "cpu"
    cpu_config.model.compile_model = False
    model, _ = create_model(cpu_config)
    missing, unexpected = model.load_state_dict(clean, strict=False)
    allowed = {"new_surface_input.weight"} if allow_mae else set()
    if set(missing) - allowed or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if not all(torch.isfinite(value).all() for value in clean.values()):
        raise ValueError(f"Non-finite weights: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs(config):
    import numpy as np
    import zarr
    from utils.norm import load_cached_norm

    label_free = set(config.data.label_free_vis_scroll_ids)
    train_ids = set(config.scroll_ids())
    ids = train_ids | set(config.data.vis_scroll_ids or [])
    for sid in ids:
        suffix = (config.data.zarr_suffix or {}).get(str(sid), "")
        volume_id = f"{sid}{suffix}"
        volume_path = Path(config.data.zarr_path) / f"{volume_id}.zarr"
        volume = zarr.open(str(volume_path), mode="r")
        required = [ROOT / "masks" / f"{sid}.png"]
        if sid not in label_free:
            required.append(Path(config.data.inklabel_dir) / f"{sid}.png")
            required.append(Path(config.data.train_mask_dir) / f"{sid}.png")
        for name in ("depth.npy", "confidence.npy"):
            path = ROOT / config.data.surface_label_dir / str(sid) / name
            values = np.load(path, mmap_mode="r")
            if tuple(values.shape) != tuple(volume.shape[1:]):
                raise ValueError(f"Surface/volume frame mismatch: {path}")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        if load_cached_norm(volume_id, mode=config.data.norm_mode) is None:
            raise ValueError(f"Missing normalization: {volume_id}")
    if config.data.native_noise and not (ROOT / config.data.native_noise).is_file():
        raise FileNotFoundError(config.data.native_noise)


def preflight():
    prepare_inputs()
    write_fold_guide(verify_snapshot())
    supervised = supervised_config()
    checkpoint_check(MAE_WEIGHTS, supervised, allow_mae=True)
    for config in (supervised, *(finetune_config(tid) for tid in FINE_TUNES)):
        check_inputs(config)
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    if free < 8:
        raise RuntimeError(f"Only {free:.1f} GiB free; need at least 8 GiB for overnight outputs")
    print(f"[overnight] preflight passed: AND-mask + {len(FINE_TUNES)} fine-tunes, disk free {free:.1f} GiB", flush=True)


def write_fold_guide(payload):
    import cv2
    import numpy as np

    colors = ((230, 140, 30), (70, 180, 60), (40, 80, 230), (180, 80, 160), (40, 180, 200))
    panels = []
    for sid, components in payload["components"].items():
        label = cv2.imread(str(INPUT_ROOT / "inklabels" / f"{sid}.png"), 0)
        scale = 700 / max(label.shape)
        panel = cv2.cvtColor(cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        for component in components:
            left, top, width, height = component["bbox"]
            start = (round(left * scale), round(top * scale))
            end = (round((left + width) * scale), round((top + height) * scale))
            color = colors[component["fold"] - 1]
            cv2.rectangle(panel, start, end, color, 2)
            cv2.putText(panel, f"{component['component']}:F{component['fold']}",
                        (start[0], max(18, start[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        canvas = np.full((750, 720, 3), 245, np.uint8)
        cv2.putText(canvas, sid, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        canvas[40:40 + panel.shape[0], 10:10 + panel.shape[1]] = panel
        panels.append(canvas)
    guide = np.concatenate((np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)), axis=0)
    if not cv2.imwrite(str(INPUT_ROOT / "component_folds.png"), guide):
        raise OSError("Could not write component fold guide")


def verify_outputs(config):
    import math
    import types
    sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    digest = checkpoint_check(Path(config.save_final), config)
    log_path = Path(config.tra.log_dir) / config.run_name
    events = EventAccumulator(str(log_path), size_guidance={"scalars": 0, "images": 1}).Reload()
    losses = events.Scalars("G_M/Loss/Train_Raw")
    if [event.step for event in losses] != list(range(config.tra.n_epochs)):
        raise RuntimeError(f"Incomplete training event history: {log_path}")
    if not all(math.isfinite(event.value) for event in losses):
        raise RuntimeError(f"Non-finite training loss history: {log_path}")
    if not config.tra.disable_figures:
        for sid in config.data.vis_scroll_ids:
            figure = log_path / "eval_figs" / f"eval_s{sid}_ep{config.tra.n_epochs:02d}.png"
            if not figure.is_file() or figure.stat().st_size == 0:
                raise FileNotFoundError(f"Final figure missing: {figure}")
    elif events.Tags().get("images"):
        raise RuntimeError("Supervised no-figure run unexpectedly logged images")
    return digest


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_worker(stage, smoke=False):
    from utils.dataloader import cleanup_mmap_files

    verify_snapshot()
    config = supervised_config() if stage == "and_mask" else finetune_config(stage)
    if smoke:
        config.model_dir = str(MODEL_ROOT / "smoke" / stage)
        config.save_final = str(Path(config.model_dir) / "final.pth")
        config.tra.log_dir = str(LOG_ROOT / "smoke")
        config.tra.n_epochs = 1
        config.tra.eval_int = 999999 if stage == "and_mask" else 1
        config.data.max_samples_per_epoch = 4
        config.dl.num_workers = 0
        config.dl.batch_size = 8
        if stage != "and_mask":
            config.init_weights = str(MODEL_ROOT / "smoke" / "and_mask" / "final.pth")
            config.data.vis_scroll_ids = [int(stage.removeprefix("loo_"))] if stage.startswith("loo_") else [EXTRA_ID]
            config.tra.eval_int_scrolls = 1
        config.data.selective_chunk_preload = False
    return run_config(config, stage, smoke=smoke, allow_mae=stage == "and_mask")


def archive_incomplete(config):
    import uuid

    model_dir = Path(config.model_dir)
    if (model_dir / "complete.json").exists():
        raise ValueError(f"Refusing to archive a completed stage: {model_dir}")
    saved_path = model_dir / "experiment.json"
    if not saved_path.is_file():
        raise ValueError(f"Cannot identify incomplete stage: {model_dir}")
    saved = json.loads(saved_path.read_text())
    if Path(saved["model_dir"]).resolve() != model_dir.resolve():
        raise ValueError("Incomplete stage metadata belongs to another model directory")
    log_paths = {
        Path(saved["tra"]["log_dir"]) / saved["run_name"],
        Path(config.tra.log_dir) / config.run_name,
    }
    for log_path in log_paths:
        if log_path.exists():
            record = json.loads((log_path / "config.json").read_text())
            if Path(record["model_dir"]).resolve() != model_dir.resolve():
                raise ValueError(f"Log directory belongs to another experiment: {log_path}")
    archive = model_dir.parent / "_failed_attempts" / f"{model_dir.name}_{uuid.uuid4().hex[:12]}"
    archive.mkdir(parents=True)
    model_dir.rename(archive / "model")
    for index, log_path in enumerate(sorted(log_paths)):
        if log_path.exists():
            log_path.rename(archive / f"logs_{index}")
    print(f"[overnight] archived incomplete attempt to {archive}", flush=True)
    return archive


def run_config(config, stage, smoke=False, allow_mae=False, retry_incomplete=False):
    from utils.dataloader import cleanup_mmap_files

    check_inputs(config)
    init_hash = checkpoint_check(Path(config.init_weights), config, allow_mae=allow_mae)
    payload = _config_payload(config)
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    marker = Path(config.model_dir) / "complete.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["config_sha256"] != signature or previous["init_sha256"] != init_hash:
            raise ValueError(f"Completed stage configuration changed: {stage}")
        if verify_outputs(config) != previous["checkpoint_sha256"]:
            raise ValueError(f"Completed checkpoint changed: {stage}")
        print(f"[overnight] verified completed stage, skip {stage}", flush=True)
        return
    model_dir = Path(config.model_dir)
    if model_dir.exists() and any(model_dir.iterdir()):
        if not retry_incomplete:
            raise FileExistsError(f"Unverified existing stage outputs: {model_dir}; use an explicit incomplete-run retry")
        archive_incomplete(config)
    _atomic_json(model_dir / "experiment.json", payload)
    print(f"[overnight] START {stage}: init={config.init_weights}, output={config.save_final}", flush=True)
    try:
        if smoke:
            from itertools import islice
            from train import Trainer

            trainer = Trainer(config)
            try:
                trainer.valid_loader = list(islice(trainer.valid_loader, 2))
                trainer.run()
            finally:
                trainer.close()
        else:
            campaign29.prewarm_data_cache(config)
            if not campaign31.run_test(config, False):
                raise RuntimeError(f"Training failed: {stage}")
        output_hash = verify_outputs(config)
        _atomic_json(marker, {"stage": stage, "complete": True, "config_sha256": signature,
                              "init_sha256": init_hash, "checkpoint_sha256": output_hash})
        print(f"[overnight] COMPLETE {stage}", flush=True)
    finally:
        cleanup_mmap_files()


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", help="comma-separated fine-tune subset (AND-mask always precedes it)")
    parser.add_argument("--worker", choices=("and_mask", *FINE_TUNES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        run_worker(args.worker, args.smoke)
        return
    selected = list(FINE_TUNES)
    if args.only:
        wanted = set(args.only.split(","))
        if wanted - set(FINE_TUNES):
            parser.error(f"Unknown fine-tunes: {sorted(wanted - set(FINE_TUNES))}")
        selected = [tid for tid in FINE_TUNES if tid in wanted]
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (INPUT_ROOT / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight()
        if args.dry_run:
            print("[overnight] QUEUE", ["and_mask", *selected], flush=True)
            return
        for stage in ("and_mask", *selected):
            status_path = MODEL_ROOT / ("smoke_status.json" if args.smoke else "status.json")
            _atomic_json(status_path, {"state": "running", "stage": stage, "queue": selected})
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", stage]
            if args.smoke:
                command.append("--smoke")
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                _atomic_json(status_path, {"state": "failed", "stage": stage, "returncode": result.returncode})
                raise RuntimeError(f"Stage {stage} failed; dependent stages were not started")
        _atomic_json(status_path, {"state": "complete", "stages": ["and_mask", *selected]})
        print("[overnight] ALL STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    main()