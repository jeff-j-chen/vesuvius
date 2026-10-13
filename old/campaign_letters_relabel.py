"""Versioned fresh-label comparisons and follow-up transfer probes.

Reuses the completed supervised AND-mask model; never starts supervised training.
Each run name owns an immutable label snapshot and separate checkpoints/logs.
The original relabel suite stays unchanged; --suite probes adds early stopping,
single-source training and weak PU. --after-run waits for a successful predecessor.
--suite single_remaining trains each of the other three labeled patches independently.
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

import campaign_letters_overnight as overnight

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = "letters_relabel_v2"
ARMS = ("baseline", *(f"loo_{sid}" for sid in overnight.TRAIN_IDS), "brightness_contrast")
SINGLE_SOURCE = 20261012073246
PROBE_ARMS = ("early_stop_2", f"single_{SINGLE_SOURCE}", "weak_pu")
REMAINING_SINGLE_ARMS = tuple(f"single_{sid}" for sid in overnight.TRAIN_IDS if sid != SINGLE_SOURCE)
REBUILD_ARMS = ("rebuilt_no_offset_weak_pu_photo",)
PAIRED_HOLDOUTS = {
    "A_train3_holdout_offset": (20261012073250,),
    "B_train3_holdout_065806": (20261012065806,),
    "C_train2_holdout_offset_065806": (20261012073250, 20261012065806),
}
PAIRED_ARMS = tuple(PAIRED_HOLDOUTS)
TWO_SOURCE_ARMS = ("train_065806_0009b",)
TWO_SOURCE_IDS = (20261012065806, overnight.letters.PHERC0009B_ID)
HOLDOUT_065806_ARMS = ("train3_holdout_065806_v6",)
LOSS_ABLATIONS = {
    "holdout_065806_no_nnpu": {"tra.pu_lambda": 0.0, "data.far_negative_share": 0.0},
    "holdout_065806_no_bag_rank": {"tra.character_bag_ranking": False},
    "holdout_065806_no_nnpu_no_bag_rank": {
        "tra.pu_lambda": 0.0, "data.far_negative_share": 0.0, "tra.character_bag_ranking": False,
    },
}
LOSS_ABLATION_ARMS = tuple(LOSS_ABLATIONS)
ALL_ARMS = (*ARMS, *PROBE_ARMS, *REMAINING_SINGLE_ARMS, *REBUILD_ARMS, *PAIRED_ARMS, *TWO_SOURCE_ARMS, *HOLDOUT_065806_ARMS, *LOSS_ABLATION_ARMS)
SUITES = {"relabel": ARMS, "probes": PROBE_ARMS, "single_remaining": REMAINING_SINGLE_ARMS,
          "rebuilt_no_offset": REBUILD_ARMS, "paired_holdouts": PAIRED_ARMS,
          "065806_0009b": TWO_SOURCE_ARMS, "holdout_065806": HOLDOUT_065806_ARMS,
          "holdout_loss_ablations": LOSS_ABLATION_ARMS}
NEW_VIS_PATCHES = {
    20260927200859: "auto_grown_20260927200859829_abf",
    20261002075023: "auto_grown_20261002075023617_abf",
}


def suite_arms(name=DEFAULT_RUN, suite=None):
    inputs, _, _ = run_paths(name)
    manifest = inputs / "inputs.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text()).get("suite", "relabel")
        if suite is not None and suite != saved:
            raise ValueError(f"Existing snapshot uses suite {saved}, not {suite}")
        suite = saved
    return SUITES[suite or "relabel"]


def run_paths(name):
    if Path(name).name != name or name in (".", "..", ""):
        raise ValueError("Run name must be a nonempty directory name")
    return ROOT / "_ves_tmp" / name, ROOT / "models" / name, ROOT / f"runs_{name}"


def build_config(tid, name=DEFAULT_RUN):
    if tid not in ALL_ARMS:
        raise ValueError(f"Unknown arm: {tid}")
    inputs, models, logs = run_paths(name)
    settings_path = inputs / "run_settings.json"
    if settings_path.is_file():
        logs = Path(json.loads(settings_path.read_text())["log_dir"])
    source_tid = "loo_20261012065806" if tid in (*HOLDOUT_065806_ARMS, *LOSS_ABLATION_ARMS) else tid
    config = overnight.finetune_config("baseline" if tid in ("brightness_contrast", *PROBE_ARMS, *REMAINING_SINGLE_ARMS, *REBUILD_ARMS, *PAIRED_ARMS, *TWO_SOURCE_ARMS) else source_tid)
    config.model_dir = str(models / tid)
    config.save_final = str(models / tid / "final.pth")
    config.tra.log_dir = str(logs)
    config.exp_name = config.run_name = tid
    config.data.inklabel_dir = str(inputs / "inklabels")
    config.data.train_mask_dir = str(inputs / "train_masks" / tid)
    config.dl.brightness_prob = config.dl.contrast_prob = 0.0
    if name != DEFAULT_RUN:
        config.data.vis_scroll_ids = [*config.data.vis_scroll_ids, *NEW_VIS_PATCHES]
        config.data.label_free_vis_scroll_ids = [*config.data.label_free_vis_scroll_ids, *NEW_VIS_PATCHES]
        config.data.surface_window_offset_by_scroll.update({sid: -1 for sid in NEW_VIS_PATCHES})
        config.tra.eval_int_scrolls = len(config.data.vis_scroll_ids)
    if tid == "early_stop_2":
        config.tra.n_epochs = 2
        config.tra.eval_int = 2
    elif tid.startswith("single_"):
        source_id = int(tid.removeprefix("single_"))
        config.data.scrolls = [scroll for scroll in config.data.scrolls if int(scroll.scroll_id) == source_id]
        config.data.train_scroll_dict = {"pherc0846a": [source_id]}
        config.data.train_scroll_weights = [1]
        config.data.train_only_scroll_ids = [source_id]
        config.data.holdout_domains = []
        config.data.validation_scroll_ids = None
        config.tra.dann_n_domains = 1
    elif tid in REBUILD_ARMS:
        offset_id = overnight.letters.FLAT_IDS[overnight.letters.OFFSET_ID]
        config.data.scrolls = [scroll for scroll in config.data.scrolls if int(scroll.scroll_id) != offset_id]
        config.data.train_scroll_dict["pherc0846a"] = [sid for sid in overnight.TRAIN_IDS if sid != offset_id]
        config.data.train_only_scroll_ids = list(config.data.train_scroll_dict["pherc0846a"])
        config.data.train_scroll_weights = [5, 1]
    elif tid in PAIRED_HOLDOUTS:
        heldout = PAIRED_HOLDOUTS[tid]
        config.data.train_scroll_dict = {
            "pherc0846a": [sid for sid in overnight.TRAIN_IDS if sid not in heldout],
            "pherc0009b": [overnight.letters.PHERC0009B_ID],
            **{f"heldout_{sid}": [sid] for sid in heldout},
        }
        config.data.train_scroll_weights = [5, 1, *([1] * len(heldout))]
        config.data.train_only_scroll_ids = list(config.data.train_scroll_dict["pherc0846a"])
        config.data.holdout_domains = [f"heldout_{sid}" for sid in heldout]
        config.data.validation_scroll_ids = list(heldout)
        config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    elif tid in TWO_SOURCE_ARMS:
        config.data.scrolls = [scroll for scroll in config.data.scrolls if int(scroll.scroll_id) in TWO_SOURCE_IDS]
        config.data.train_scroll_dict = {"pherc0846a": [TWO_SOURCE_IDS[0]], "pherc0009b": [TWO_SOURCE_IDS[1]]}
        config.data.train_scroll_weights = [5, 1]
        config.data.train_only_scroll_ids = [TWO_SOURCE_IDS[0]]
        config.data.holdout_domains = []
        config.data.validation_scroll_ids = None
        config.data.vis_scroll_ids = list(overnight.TRAIN_IDS)
        config.data.label_free_vis_scroll_ids = []
        config.data.surface_window_offset_by_scroll = {sid: -1 for sid in overnight.TRAIN_IDS}
        config.tra.eval_int_scrolls = len(overnight.TRAIN_IDS)
        config.tra.dann_n_domains = 2
    if tid in (*HOLDOUT_065806_ARMS, *LOSS_ABLATION_ARMS):
        config.data.vis_scroll_ids = list(overnight.TRAIN_IDS)
        config.data.label_free_vis_scroll_ids = []
        config.data.surface_window_offset_by_scroll = {sid: -1 for sid in overnight.TRAIN_IDS}
        config.tra.eval_int_scrolls = len(overnight.TRAIN_IDS)
    if tid == "weak_pu" or tid in (*REBUILD_ARMS, *PAIRED_ARMS, *TWO_SOURCE_ARMS, *HOLDOUT_065806_ARMS, *LOSS_ABLATION_ARMS):
        config.tra.pu_lambda = 0.1
        config.tra.pu_prior = 0.2
        config.data.far_negative_share = 0.15
    if tid == "brightness_contrast" or tid in (*REBUILD_ARMS, *PAIRED_ARMS, *TWO_SOURCE_ARMS, *HOLDOUT_065806_ARMS, *LOSS_ABLATION_ARMS):
        config.dl.data_aug = True
        config.dl.brightness_prob = config.dl.contrast_prob = 0.5
        config.dl.brightness_delta = 0.15
        config.dl.contrast_delta = 0.20
    for key, value in LOSS_ABLATIONS.get(tid, {}).items():
        section, field = key.split(".")
        setattr(getattr(config, section), field, value)
    return config


def verify_snapshot(name):
    inputs, _, _ = run_paths(name)
    record = json.loads((inputs / "inputs.json").read_text())
    if record["arms"] != list(suite_arms(name)):
        raise ValueError("Snapshot arm definitions differ from this runner")
    for relative, expected in record["files"].items():
        if hashlib.sha256((inputs / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot changed: {relative}")
    if hashlib.sha256(overnight.SUPERVISED_WEIGHTS.read_bytes()).hexdigest() != record["init_sha256"]:
        raise ValueError("Supervised initialization changed")
    return record


def prepare_inputs(name, suite=None):
    import cv2
    import numpy as np
    from utils.dataloader import split_extra_labels

    inputs, _, _ = run_paths(name)
    arms = suite_arms(name, suite)
    if (inputs / "inputs.json").is_file():
        return verify_snapshot(name)
    manifest = json.loads((overnight.SUPERVISED_WEIGHTS.parent / "complete.json").read_text())
    init_hash = hashlib.sha256(overnight.SUPERVISED_WEIGHTS.read_bytes()).hexdigest()
    if not manifest["complete"] or manifest["checkpoint_sha256"] != init_hash:
        raise ValueError("Supervised checkpoint does not match its completion record")
    ink_dir = inputs / "inklabels"
    ink_dir.mkdir(parents=True, exist_ok=True)
    counts, files = {}, {}
    reuse_labels = arms not in (ARMS, REBUILD_ARMS, PAIRED_ARMS, TWO_SOURCE_ARMS, HOLDOUT_065806_ARMS, LOSS_ABLATION_ARMS)
    source_dir = run_paths(DEFAULT_RUN)[0] / "inklabels" if reuse_labels else ROOT / overnight.letters.INKLABEL_DIR
    if reuse_labels:
        verify_snapshot(DEFAULT_RUN)
    ids = [*overnight.TRAIN_IDS, overnight.letters.PHERC0009B_ID]
    for sid in ids:
        source = source_dir / f"{sid}.png"
        target = ink_dir / source.name
        shutil.copyfile(source, target)
        label = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Unreadable label: {source}")
        split_root = ROOT / "train_masks" if arms in (REBUILD_ARMS, PAIRED_ARMS, TWO_SOURCE_ARMS, HOLDOUT_065806_ARMS, LOSS_ABLATION_ARMS) else overnight.INPUT_ROOT / "train_masks" / "baseline"
        split_source = split_root / f"{sid}.png"
        split = cv2.imread(str(split_source), cv2.IMREAD_GRAYSCALE)
        if split is None or any(actual < expected or actual > expected + 1
                                for actual, expected in zip(label.shape, split.shape)):
            raise ValueError(f"Previous split frame does not match new labels: {split_source}")
        if label.shape != split.shape:
            print(f"[relabel] {sid}: loader common-frame crop {label.shape} -> {split.shape}; source unchanged", flush=True)
        regular, _ = split_extra_labels(label[:split.shape[0], :split.shape[1]])
        _, _, stats, _ = cv2.connectedComponentsWithStats(regular.astype(np.uint8), 8)
        components = int((stats[1:, cv2.CC_STAT_AREA] >= 8).sum())
        if not components:
            raise ValueError(f"No retained ink components: {source}")
        counts[str(sid)] = {"components": components, "ink_pixels": int(regular.sum()),
                            "source_shape": list(label.shape), "effective_shape": list(split.shape)}
        files[str(target.relative_to(inputs))] = hashlib.sha256(target.read_bytes()).hexdigest()
        for tid in arms:
            folder = inputs / "train_masks" / tid
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{sid}.png"
            if (tid == f"loo_{sid}" or (tid.startswith("single_") and sid != int(tid.removeprefix("single_")))
                    or (tid in REBUILD_ARMS and sid == overnight.letters.FLAT_IDS[overnight.letters.OFFSET_ID])
                    or sid in PAIRED_HOLDOUTS.get(tid, ())
                    or (tid in TWO_SOURCE_ARMS and sid not in TWO_SOURCE_IDS)
                    or (tid in (*HOLDOUT_065806_ARMS, *LOSS_ABLATION_ARMS) and sid == 20261012065806)):
                if not cv2.imwrite(str(target), np.zeros_like(split)):
                    raise OSError(target)
            else:
                shutil.copyfile(split_source, target)
            files[str(target.relative_to(inputs))] = hashlib.sha256(target.read_bytes()).hexdigest()
    record = {"arms": list(arms), "suite": suite or "relabel", "files": files, "counts": counts,
              "init_weights": str(overnight.SUPERVISED_WEIGHTS), "init_sha256": init_hash,
              "label_source": str(source_dir),
              "split_source": str(split_root),
              "note": "Updated labels; unchanged split masks; supervised holdouts, not MAE-unseen targets"}
    overnight._atomic_json(inputs / "inputs.json", record)
    return record


def preflight(name, suite=None):
    record = prepare_inputs(name, suite)
    if name != DEFAULT_RUN and tuple(record["arms"]) not in (TWO_SOURCE_ARMS, HOLDOUT_065806_ARMS, LOSS_ABLATION_ARMS):
        for sid, source in NEW_VIS_PATCHES.items():
            overnight.prepare_extra(source, sid)
    overnight.checkpoint_check(overnight.SUPERVISED_WEIGHTS, build_config(record["arms"][0], name))
    for tid in record["arms"]:
        overnight.check_inputs(build_config(tid, name))
    if shutil.disk_usage(ROOT).free < 8 * 1024 ** 3:
        raise RuntimeError("Need at least 8 GiB free for the new outputs")
    print(f"[relabel] {name}: {len(record['arms'])} fine-tunes, no supervised retraining", flush=True)
    print(f"[relabel] label inventory: {record['counts']}", flush=True)


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--suite", choices=tuple(SUITES))
    parser.add_argument("--log-dir", help="existing TensorBoard root; remembered for subsequent restarts")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="archive incomplete attempts and restart them from the supervised initialization")
    parser.add_argument("--after-run", help="wait for another relabel run to complete successfully")
    parser.add_argument("--worker", choices=ALL_ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    inputs, models, _ = run_paths(args.run_name)
    arms = suite_arms(args.run_name, args.suite)
    if args.worker:
        if args.worker not in arms:
            parser.error("Worker is not part of this run's suite")
        verify_snapshot(args.run_name)
        overnight.run_config(build_config(args.worker, args.run_name), args.worker,
                     retry_incomplete=args.retry_incomplete)
        return
    if args.after_run and not args.dry_run:
        if args.after_run == args.run_name:
            parser.error("A run cannot depend on itself")
        previous_inputs, previous_models, _ = run_paths(args.after_run)
        overnight._atomic_json(models / "status.json", {"state": "waiting", "dependency": args.after_run,
                                                        "queue": list(arms)})
        print(f"[relabel] waiting for {args.after_run} to release its runner lock", flush=True)
        with (previous_inputs / "runner.lock").open("a") as previous_lock:
            fcntl.flock(previous_lock, fcntl.LOCK_EX)
            status = json.loads((previous_models / "status.json").read_text())
            if status.get("state") != "complete":
                overnight._atomic_json(models / "status.json", {"state": "blocked", "dependency": args.after_run,
                                                                "dependency_status": status})
                raise RuntimeError(f"Dependency did not complete successfully: {status}")
    inputs.mkdir(parents=True, exist_ok=True)
    with (inputs / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.log_dir:
            overnight._atomic_json(inputs / "run_settings.json", {"log_dir": str(Path(args.log_dir).resolve())})
        preflight(args.run_name, args.suite)
        if args.dry_run:
            print("[relabel] QUEUE", list(arms), flush=True)
            return
        for tid in arms:
            overnight._atomic_json(models / "status.json", {"state": "running", "stage": tid, "queue": list(arms)})
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--run-name", args.run_name, "--worker", tid]
            if args.retry_incomplete:
                command.append("--retry-incomplete")
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                overnight._atomic_json(models / "status.json", {"state": "failed", "stage": tid,
                                                                  "returncode": result.returncode})
                raise RuntimeError(f"Fine-tune failed: {tid} (worker exit {result.returncode}); remaining arms were not started")
        overnight._atomic_json(models / "status.json", {"state": "complete", "stages": list(arms)})
        print("[relabel] ALL FINE-TUNES COMPLETE", flush=True)


if __name__ == "__main__":
    main()