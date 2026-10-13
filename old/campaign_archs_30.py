"""campaign 30: depth-collapse and 2D U-Net architecture study

All standard arms retain Campaign 29 data, 192x192/ds2 input, eight slices,
literal surface input, and the 4x4 multitile head. The researcher-like arm is
the sole ds1 exception.

Usage:
    python3 campaign_archs_30.py --dry-run
    python3 campaign_archs_30.py --only early_gated
    python3 campaign_archs_30.py --from mid_residual2d
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import campaign_archs_29 as campaign29
from utils.config import DEFAULT_SCROLLS, DEFAULT_TEST_SCROLL_IDS


ROOT = Path(__file__).resolve().parent
LOG_DIR = "./runs_archs29"
MODEL_DIR = "models/archs30"
PRETRAIN_STEPS = 2_000
MIN_TRANSFER_COVERAGE = 0.85
_MATCHED_PRETRAIN_OVERRIDES: set[str] = set()
ALL_PRETRAIN_SCROLL_IDS = tuple(dict.fromkeys(
    [int(scroll.scroll_id) for scroll in DEFAULT_SCROLLS]
    + [int(scroll_id) for scroll_id in DEFAULT_TEST_SCROLL_IDS]
))


def _spec(*args, required=(), ctx=192, ds=2, batch_size=32, accum_steps=1):
    return {
        "args": tuple(args),
        "required": tuple(required),
        "ctx": int(ctx),
        "ds": int(ds),
        "batch_size": int(batch_size),
        "accum_steps": int(accum_steps),
    }


PRETRAIN_SPECS = {
    "mid_gated_c29": {
        "reuse_campaign29": "mid_gated_all",
        "required": ("gated_cue_stem.", "mid_depth_attn.", "mid2d_"),
        "ctx": 192,
        "ds": 2,
    },
    "early_gated": _spec(
        "--early-2d-unet", "--gated-stems", "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "early_depth_attn.", "early2d_"),
    ),
    "early_wide15_gated": _spec(
        "--early-2d-unet", "--early-2d-channels-mult", "1.5",
        "--gated-stems", "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "early_depth_attn.", "early2d_"),
        batch_size=24,
    ),
    "mid_wide15_gated": _spec(
        "--mid-2d-unet", "--mid-2d-channels-mult", "1.5",
        "--gated-stems", "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "mid_depth_attn.", "mid2d_"),
        batch_size=24,
    ),
    "early_residual2d": _spec(
        "--early-2d-unet", "--residual-2d-unet", "--gated-stems",
        "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "early2d_enc2.shortcut.", "early2d_"),
    ),
    "mid_residual2d": _spec(
        "--mid-2d-unet", "--residual-2d-unet", "--gated-stems",
        "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "mid2d_enc3.shortcut.", "mid2d_"),
    ),
    "mid_deep2d": _spec(
        "--mid-2d-unet", "--two-d-block-depth", "3",
        "--two-d-extra-channels", "320", "--gated-stems",
        "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "mid2d_extra_encoders.", "mid2d_"),
        batch_size=24,
    ),
    "mid_deep_residual2d": _spec(
        "--mid-2d-unet", "--residual-2d-unet", "--two-d-block-depth", "3",
        "--two-d-extra-channels", "320", "--gated-stems",
        "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "mid2d_extra_encoders.", "mid2d_"),
        batch_size=20,
    ),
    "early_deep_residual2d": _spec(
        "--early-2d-unet", "--residual-2d-unet", "--two-d-block-depth", "3",
        "--two-d-extra-channels", "320", "--gated-stems",
        "--norm-mode", "ibn_full",
        required=("gated_cue_stem.", "early2d_extra_encoders.", "early2d_"),
        batch_size=20,
    ),
    "mid_pure_instance": _spec(
        "--mid-2d-unet", "--gated-stems", "--norm-mode", "instance",
        required=("gated_cue_stem.", "mid_depth_attn.", "mid2d_"),
    ),
    "mid_raw_only": _spec(
        "--mid-2d-unet", "--raw-only-stem", "--norm-mode", "ibn_full",
        required=("enc1.", "mid_depth_attn.", "mid2d_"),
    ),
    "early_raw_instance": _spec(
        "--early-2d-unet", "--raw-only-stem", "--norm-mode", "instance",
        required=("enc1.", "early_depth_attn.", "early2d_"),
    ),
    "researcher_like": _spec(
        "--early-2d-unet", "--raw-only-stem", "--norm-mode", "instance",
        "--channels-mult", "0.5", "--residual-2d-unet",
        "--two-d-extra-channels", "256", "320",
        required=("enc1.", "early_depth_attn.", "early2d_extra_encoders.", "early2d_"),
        ctx=192,
        ds=1,
        batch_size=4,
        accum_steps=8,
    ),
}

for _reuse_key in ("mid_residual2d", "mid_pure_instance", "mid_raw_only"):
    PRETRAIN_SPECS[_reuse_key]["reuse_campaign29"] = "mid_gated_all"


def _test(tid: str, pretrain_key: str, **overrides) -> dict:
    test = {
        "tid": tid,
        "tag": f"30_{tid}",
        "pretrain_key": pretrain_key,
        "early_2d_unet": False,
        "mid_2d_unet": True,
        "gated_stems": True,
        "norm_mode": "ibn_full",
        "context_downsample": 2,
        "batch_size": 96,
        "channels_mult": 1.0,
        "early_2d_channels_mult": 1.0,
        "mid_2d_channels_mult": 1.0,
        "residual_2d_unet": False,
        "two_d_block_depth": 2,
        "two_d_extra_levels": 0,
        "two_d_extra_channels": (),
        "raw_only_stem": False,
        "pcgrad_lite": False,
        "pcgrad_lite_max_domains": 4,
        "pcgrad_lite_scope": "head",
    }
    test.update(overrides)
    return test


TESTS = [
    _test("current_mid_control", "mid_gated_c29"),
    _test("pcgrad_lite_head4", "mid_gated_c29", pcgrad_lite=True),
    _test("early_gated", "early_gated", early_2d_unet=True, mid_2d_unet=False),
    _test(
        "early_wide15_gated",
        "early_wide15_gated",
        early_2d_unet=True,
        mid_2d_unet=False,
        early_2d_channels_mult=1.5,
    ),
    _test("mid_wide15_gated", "mid_wide15_gated", mid_2d_channels_mult=1.5),
    _test(
        "early_residual2d",
        "early_residual2d",
        early_2d_unet=True,
        mid_2d_unet=False,
        residual_2d_unet=True,
    ),
    _test("mid_residual2d", "mid_residual2d", residual_2d_unet=True),
    _test(
        "mid_deep2d",
        "mid_deep2d",
        two_d_block_depth=3,
        two_d_extra_channels=(320,),
    ),
    _test(
        "mid_deep_residual2d",
        "mid_deep_residual2d",
        residual_2d_unet=True,
        two_d_block_depth=3,
        two_d_extra_channels=(320,),
    ),
    _test(
        "early_deep_residual2d",
        "early_deep_residual2d",
        early_2d_unet=True,
        mid_2d_unet=False,
        residual_2d_unet=True,
        two_d_block_depth=3,
        two_d_extra_channels=(320,),
    ),
    _test("mid_pure_instance", "mid_pure_instance", norm_mode="instance"),
    _test("mid_raw_only", "mid_raw_only", gated_stems=False, raw_only_stem=True),
    _test(
        "early_raw_instance",
        "early_raw_instance",
        early_2d_unet=True,
        mid_2d_unet=False,
        gated_stems=False,
        raw_only_stem=True,
        norm_mode="instance",
    ),
    _test(
        "researcher_like",
        "researcher_like",
        early_2d_unet=True,
        mid_2d_unet=False,
        gated_stems=False,
        raw_only_stem=True,
        norm_mode="instance",
        context_downsample=1,
        batch_size=4,
        channels_mult=0.5,
        residual_2d_unet=True,
        two_d_extra_channels=(256, 320),
    ),
]


def _pretrain_name(key: str) -> str:
    return f"mae_nnunet_192_campaign30_{key}_2k"


def _pretrain_path(key: str) -> Path:
    spec = PRETRAIN_SPECS[key]
    reused = spec.get("reuse_campaign29")
    if reused and key not in _MATCHED_PRETRAIN_OVERRIDES:
        return campaign29._pretrain_path(str(reused))
    return ROOT / "models" / f"{_pretrain_name(key)}.pth"


def _pretrain_marker(key: str) -> Path:
    return _pretrain_path(key).with_suffix(".complete.json")


def _pretrain_metadata(key: str) -> dict:
    spec = PRETRAIN_SPECS[key]
    return {
        "campaign": 30,
        "key": key,
        "steps": PRETRAIN_STEPS,
        "scroll_ids": list(ALL_PRETRAIN_SCROLL_IDS),
        "architecture_args": list(spec["args"]),
        "depth": 8,
        "d_start": 10,
        "d_end": 18,
        "ctx": spec["ctx"],
        "ds": spec["ds"],
        "from_scratch": True,
        "sampling": "physical_round_robin",
        "checkpoint": str(_pretrain_path(key).relative_to(ROOT)),
    }


def _pretraining_complete(key: str) -> bool:
    spec = PRETRAIN_SPECS[key]
    reused = spec.get("reuse_campaign29")
    if reused and key not in _MATCHED_PRETRAIN_OVERRIDES:
        return campaign29._pretraining_complete(str(reused))
    checkpoint = _pretrain_path(key)
    marker = _pretrain_marker(key)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    return metadata == _pretrain_metadata(key) and all(
        any(name.startswith(prefix) for name in state)
        for prefix in spec["required"]
    )


def _transfer_coverage(test: dict, checkpoint: Path) -> tuple[float, int, int, int, int]:
    from utils.model import create_model

    config = build_config(test)
    config.device = "cpu"
    config.model.compile_model = False
    model, _ = create_model(config)
    if config.model.early_2d_unet:
        prefixes = ["enc1.", "early_depth_attn.", "early_depth_fuse.", "early2d_"]
    else:
        prefixes = [
            "enc1.", "enc2.", "mid_depth_attn.", "mid_depth_fuse.",
            "mid_skip1_fuse.", "mid2d_",
        ]
    if config.model.gated_stems:
        prefixes.append("gated_cue_stem.")
    excluded = ("early2d_head.", "mid2d_head.", "new_surface_input.")
    active = {
        name: parameter
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
        and not name.startswith(excluded)
    }
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    compatible = {
        name: parameter
        for name, parameter in active.items()
        if name in state and tuple(state[name].shape) == tuple(parameter.shape)
    }
    compatible_parameters = sum(parameter.numel() for parameter in compatible.values())
    active_parameters = sum(parameter.numel() for parameter in active.values())
    del state, model
    gc.collect()
    return (
        compatible_parameters / max(active_parameters, 1),
        compatible_parameters,
        active_parameters,
        len(compatible),
        len(active),
    )


def build_config(test: dict):
    base = {
        "tid": test["tid"],
        "tag": test["tag"],
        "pretrain_key": "mid_gated_all",
    }
    config = campaign29.build_config(base)
    config.exp_name = f"cmp_archs30_{test['tag']}"
    config.tra.log_dir = LOG_DIR
    config.dl.batch_size = int(test["batch_size"])
    config.data.context_size = 192
    config.data.context_downsample = int(test["context_downsample"])
    config.data.depth = 8
    config.model.multitile = True
    config.model.multitile_subtile = 16
    config.model.multitile_grid = 4
    config.model.surface_teacher_input = True
    config.model.early_2d_unet = bool(test["early_2d_unet"])
    config.model.mid_2d_unet = bool(test["mid_2d_unet"])
    config.model.gated_stems = bool(test["gated_stems"])
    config.model.raw_only_stem = bool(test["raw_only_stem"])
    config.model.norm_mode = str(test["norm_mode"])
    config.model.use_ibn = config.model.norm_mode == "ibn"
    config.model.channels_mult = float(test["channels_mult"])
    config.model.early_2d_channels_mult = float(test["early_2d_channels_mult"])
    config.model.mid_2d_channels_mult = float(test["mid_2d_channels_mult"])
    config.model.residual_2d_unet = bool(test["residual_2d_unet"])
    config.model.two_d_block_depth = int(test["two_d_block_depth"])
    config.model.two_d_extra_levels = int(test["two_d_extra_levels"])
    config.model.two_d_extra_channels = tuple(test["two_d_extra_channels"])
    config.model.two_d_bottleneck_channels = 0
    config.model.compile_model = False
    pretrain_key = str(test["pretrain_key"])
    config.model.require_architecture_init = not (
        pretrain_key != "mid_gated_c29"
        and PRETRAIN_SPECS[pretrain_key].get("reuse_campaign29")
        and pretrain_key not in _MATCHED_PRETRAIN_OVERRIDES
    )
    config.tra.pcgrad = False
    config.tra.pcgrad_lite = bool(test["pcgrad_lite"])
    config.tra.pcgrad_lite_max_domains = int(test["pcgrad_lite_max_domains"])
    config.tra.pcgrad_lite_scope = str(test["pcgrad_lite_scope"])
    config.init_weights = str(_pretrain_path(str(test["pretrain_key"])).relative_to(ROOT))

    checkpoint_dir = os.path.join(MODEL_DIR, test["tid"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    config.model_dir = checkpoint_dir
    config.save_final = os.path.join(checkpoint_dir, "final.pth")
    return config


def preflight_pretraining(selected: list[dict], dry_run: bool) -> None:
    missing = [
        scroll_id for scroll_id in ALL_PRETRAIN_SCROLL_IDS
        if not (ROOT / "ves_zarrs2" / f"{scroll_id}.zarr").is_dir()
    ]
    if missing:
        message = f"Campaign 30 requires every training and test zarr; missing={missing}"
        if not dry_run:
            raise FileNotFoundError(message)
        print(f"[campaign30] WARNING {message}", flush=True)

    source_key = "mid_gated_all"
    if not campaign29._pretraining_complete(source_key):
        campaign29.preflight_pretraining(
            [{"pretrain_key": source_key}],
            dry_run,
        )
    source_path = campaign29._pretrain_path(source_key)
    tests_by_key = {
        str(test["pretrain_key"]): test
        for test in selected
    }
    if source_path.is_file():
        for test in selected:
            key = str(test["pretrain_key"])
            fraction, matched, total, tensor_matched, tensor_total = _transfer_coverage(
                test,
                source_path,
            )
            print(
                f"[campaign30] transfer {test['tid']}: "
                f"params={100.0 * fraction:.2f}% ({matched}/{total}) "
                f"tensors={tensor_matched}/{tensor_total}",
                flush=True,
            )
            if fraction < MIN_TRANSFER_COVERAGE:
                _MATCHED_PRETRAIN_OVERRIDES.add(key)

    keys = list(dict.fromkeys(str(test["pretrain_key"]) for test in selected))
    for key in keys:
        if key not in PRETRAIN_SPECS:
            raise ValueError(f"unknown Campaign 30 pretrain key: {key}")
        if _pretraining_complete(key):
            continue
        spec = PRETRAIN_SPECS[key]
        if dry_run:
            print(
                f"[campaign30] would pretrain {key}: {PRETRAIN_STEPS} steps, "
                f"ctx={spec['ctx']} ds={spec['ds']}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(ROOT / "mae_pretrain_nnunet.py"),
            "--name", _pretrain_name(key),
            "--scroll-ids", *(str(value) for value in ALL_PRETRAIN_SCROLL_IDS),
            "--require-all-scrolls",
            "--physical-round-robin",
            "--ctx", str(spec["ctx"]),
            "--ds", str(spec["ds"]),
            "--depth", "8",
            "--d-start", "10",
            "--d-end", "18",
            "--steps", str(PRETRAIN_STEPS),
            "--batch-size", str(spec["batch_size"]),
            "--accum-steps", str(spec["accum_steps"]),
            "--no-figures",
            *spec["args"],
        ]
        print(f"[campaign30] pretraining {key} from scratch", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        _pretrain_marker(key).write_text(
            json.dumps(_pretrain_metadata(key), indent=2) + "\n",
            encoding="utf-8",
        )
        if not _pretraining_complete(key):
            raise RuntimeError(f"Campaign 30 MAE pretraining failed validation: {key}")
        test = tests_by_key[key]
        fraction, matched, total, _, _ = _transfer_coverage(test, _pretrain_path(key))
        if fraction < MIN_TRANSFER_COVERAGE:
            raise RuntimeError(
                f"Campaign 30 MAE transfer below {MIN_TRANSFER_COVERAGE:.0%} for {key}: "
                f"{matched}/{total} ({fraction:.2%})"
            )


def run_test(config, dry_run: bool) -> bool:
    print(f"\n{'=' * 78}\n[campaign30] {config.exp_name}\n{'=' * 78}", flush=True)
    print(
        f"  pretrain={config.init_weights} batch={config.dl.batch_size} "
        f"context={config.data.context_size}/ds{config.data.context_downsample} "
        f"depth={config.data.depth}",
        flush=True,
    )
    print(
        f"  early={config.model.early_2d_unet}:x{config.model.early_2d_channels_mult} "
        f"mid={config.model.mid_2d_unet}:x{config.model.mid_2d_channels_mult} "
        f"residual2d={config.model.residual_2d_unet} "
        f"block_depth={config.model.two_d_block_depth} "
        f"extra={config.model.two_d_extra_channels} raw={config.model.raw_only_stem} "
        f"norm={config.model.norm_mode}",
        flush=True,
    )
    if dry_run:
        print("  [DRY RUN] skipping", flush=True)
        return True

    from train import Trainer
    from utils.dataloader import cleanup_mmap_files

    trainer = None
    try:
        trainer = Trainer(config)
        trainer.vis.writer.add_scalar("Run/Initialized", 1.0, 0)
        trainer.vis.writer.flush()
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return False
    finally:
        if trainer is not None:
            trainer.close()
        del trainer
        gc.collect()
        cleanup_mmap_files()


def _open_fd_count() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def _cgroup_oom_kill_count() -> int:
    try:
        entries = dict(
            line.split(maxsplit=1)
            for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
        )
        return int(entries.get("oom_kill", 0))
    except (OSError, TypeError, ValueError):
        return -1


def run_test_isolated(config) -> bool:
    if not hasattr(os, "fork"):
        return run_test(config, False)
    before = _open_fd_count()
    oom_before = _cgroup_oom_kill_count()
    pid = os.fork()
    if pid == 0:
        try:
            success = run_test(config, False)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if success else 1)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    after = _open_fd_count()
    oom_after = _cgroup_oom_kill_count()
    if os.WIFEXITED(status):
        detail = f"exit={os.WEXITSTATUS(status)}"
    elif os.WIFSIGNALED(status):
        number = os.WTERMSIG(status)
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = str(number)
        detail = f"signal={name}"
        if oom_after > oom_before >= 0:
            detail += f" cgroup_oom_kill={oom_before}->{oom_after}"
    else:
        detail = "unknown"
    print(
        f"[campaign30] isolated arm pid={pid} status={status} {detail} "
        f"controller_fds={before}->{after}",
        flush=True,
    )
    if before >= 0 and after > before + 4:
        raise RuntimeError(f"Campaign 30 controller leaked file descriptors: {before}->{after}")
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="campaign 30: 3D-to-2D architecture study")
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

    campaign29.preflight_train_masks(
        campaign29.CAMPAIGN28_SCROLLS,
        inklabel_dir=ROOT / "inklabels",
        strict=not args.dry_run,
    )
    preflight_pretraining(selected, args.dry_run)
    print(f"[campaign30] {len(selected)} run(s) queued (log -> {LOG_DIR})")

    results = {}
    for test in selected:
        config = build_config(test)
        if args.dry_run:
            success = run_test(config, True)
        else:
            campaign29.prewarm_data_cache(config)
            success = run_test_isolated(config)
        results[test["tid"]] = "OK" if success else "FAIL"
        if not args.dry_run:
            del config
            gc.collect()

    print(f"\n{'=' * 78}\n[campaign30] SUMMARY\n{'=' * 78}")
    for tid, status in results.items():
        print(f"  {tid}: {status}")


if __name__ == "__main__":
    main()
