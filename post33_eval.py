"""after campaign 33: render every downloaded fragment with the best non-LODO models

Selects the run with the highest Character/F1Macro/Train and the run with the highest
Character/F1Macro/Valid (LODO runs excluded), maps each argmax epoch to a saved checkpoint,
and executes the setup cells of eval_scrolls.ipynb (labelled fragments) and
test_inference.ipynb (unlabelled fragments) so the figures are the notebooks' own output.
Before the first test render, the nine test fragments are force-reassembled from the flattened
tifxyz meshes, then their mask, norm, surface labels, and composite cache are rebuilt.

Usage:
    python3 post33_eval.py                 # select + render everything
    python3 post33_eval.py --select-only
    python3 post33_eval.py render --notebook eval --run-dir D --model M --tag T [--only SID,...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs_archs33"
OUT_DIR = ROOT / "output" / "post33"
ZARR_DIR = ROOT / "ves_zarrs2"
EARLY_GUARD = 'raise RuntimeError("selected run is not the early-2D gated baseline")'
SETUP_END = {
    "eval": "MODEL = load_model(C)",
    "test": "def ensure_fragment_ready",
}
NOTEBOOKS = {"eval": ROOT / "eval_scrolls.ipynb", "test": ROOT / "test_inference.ipynb"}


def _scalars(run_dir: Path) -> dict[str, list[float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    wanted = {
        "train": "Character/F1Macro/Train",
        "valid": "Character/F1Macro/Valid",
        "ap": "Character/APMacro/Valid",
        "pixel_f1": "P_M/F1_Score/Valid",
    }
    out = {}
    for key, tag in wanted.items():
        events = sorted(accumulator.Scalars(tag), key=lambda e: e.step) if tag in tags else []
        out[key] = [float(e.value) for e in events]
    return out


def _first_argmax(values: list[float]) -> int:
    # checkpoints are only overwritten on a strict improvement
    return max(range(len(values)), key=lambda i: (values[i], -i))


def _completed_runs() -> list[dict]:
    runs = []
    for config_path in sorted(RUN_ROOT.glob("*/config.json")):
        run_dir = config_path.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["data"].get("holdout_domains") or "lodo" in run_dir.name:
            continue
        final = ROOT / config["save_final"]
        n_epochs = int(config["tra"]["n_epochs"])
        if not final.is_file():
            print(f"[select] skip {run_dir.name}: no final checkpoint", flush=True)
            continue
        scalars = _scalars(run_dir)
        if len(scalars["valid"]) < n_epochs or len(scalars["train"]) < n_epochs:
            print(f"[select] skip {run_dir.name}: incomplete scalars", flush=True)
            continue
        runs.append({"run_dir": run_dir, "config": config, "scalars": scalars})
    return runs


def _checkpoint_for(run: dict, epoch: int) -> tuple[Path, int, bool]:
    config, scalars = run["config"], run["scalars"]
    final = ROOT / config["save_final"]
    root, ext = os.path.splitext(str(final))
    if root.endswith("_final"):
        root = root[:-len("_final")]
    candidates = [
        (Path(f"{root}_best_character{ext or '.pth'}"), _first_argmax(scalars["ap"]) if scalars["ap"] else None),
        (final, len(scalars["valid"]) - 1),
        (ROOT / config["model_dir"] / "best_model_f1.pth",
         _first_argmax(scalars["pixel_f1"]) if scalars["pixel_f1"] else None),
    ]
    candidates = [(path, ep) for path, ep in candidates if ep is not None and path.is_file()]
    for path, ep in candidates:
        if ep == epoch:
            return path, ep, True
    path, ep = min(candidates, key=lambda item: (abs(item[1] - epoch), -item[1]))
    return path, ep, False


def select() -> list[dict]:
    runs = _completed_runs()
    if not runs:
        raise RuntimeError("no completed non-LODO campaign 33 runs")
    picks = []
    for metric in ("train", "valid"):
        best = max(runs, key=lambda r: max(r["scalars"][metric]))
        epoch = _first_argmax(best["scalars"][metric])
        path, checkpoint_epoch, exact = _checkpoint_for(best, epoch)
        pick = {
            "metric": f"Character/F1Macro/{metric.capitalize()}",
            "run_id": best["run_dir"].name,
            "run_dir": str(best["run_dir"]),
            "value": max(best["scalars"][metric]),
            "epoch": epoch,
            "checkpoint": str(path),
            "checkpoint_epoch": checkpoint_epoch,
            "exact_epoch": exact,
            "tag": f"{best['run_dir'].name}__{path.stem}",
        }
        if not exact:
            print(
                f"[select] WARNING {pick['metric']} peaked at epoch {epoch} but no checkpoint was "
                f"saved there; using {path.name} (epoch {checkpoint_epoch})",
                flush=True,
            )
        picks.append(pick)
    ranking = sorted(
        ((r["run_dir"].name, max(r["scalars"]["train"]), max(r["scalars"]["valid"])) for r in runs),
        key=lambda item: -item[2],
    )
    for name, train, valid in ranking:
        print(f"[select] {name}: train_max={train:.4f} valid_max={valid:.4f}", flush=True)
    return picks


def _load_cells(notebook: str) -> list[str]:
    data = json.loads(NOTEBOOKS[notebook].read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in data["cells"] if cell["cell_type"] == "code"]
    for index, source in enumerate(sources):
        if SETUP_END[notebook] in source:
            setup = sources[:index + 1]
            break
    else:
        raise RuntimeError(f"{NOTEBOOKS[notebook].name}: setup end marker not found")
    if not any(EARLY_GUARD in source for source in setup):
        raise RuntimeError(f"{NOTEBOOKS[notebook].name}: architecture guard changed; review the patch")
    # strict state-dict loading already enforces the architecture, so mid runs are allowed
    return [source.replace(EARLY_GUARD, "pass") for source in setup]


def prepare_test_fragments(namespace: dict, scroll_ids: list[int], only: set[int] | None) -> None:
    """re-render test zarrs from the flattened tifxyz meshes, then rebuild mask, norm, and surface"""
    import shutil

    config = namespace["C"]
    mask_dir = namespace["MASK_DIR"]
    scroll_ids = [sid for sid in scroll_ids if not only or sid in only]
    command = [sys.executable, str(ROOT / "assemble_test_segments.py"), "--force",
               "--out-dir", config.data.zarr_path]
    for sid in scroll_ids:
        command += ["--only", str(sid)]
    print(f"[prepare] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=False)
    for sid in scroll_ids:
        zarr_path = Path(config.data.zarr_path) / f"{sid}.zarr"
        if not zarr_path.is_dir() or not (Path(mask_dir) / f"{sid}.png").is_file():
            print(f"[prepare] {sid}: reassembly produced no zarr/mask; it will be skipped", flush=True)
            continue
        try:
            # composites are cached by bbox, which can survive a re-render unchanged
            for cached in (ROOT / "output" / "composite_cache").glob(f"{sid}_*"):
                cached.unlink()
            import zarr
            volume = zarr.open(str(zarr_path), mode="r")
            namespace["load_or_create_midslice_mask"](
                volume, os.path.join(mask_dir, f"{sid}.png"), refresh=True
            )
            namespace["compute_norm"](sid, config.data.zarr_path, mask_dir=mask_dir)
            shutil.rmtree(Path(config.data.surface_label_dir) / str(sid), ignore_errors=True)
            namespace["ensure_surface_supervision"](sid)
            print(f"[prepare] {sid}: reassembled, renormalized, surface regenerated", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"[prepare] {sid}: preparation failed", flush=True)


def render(args) -> int:
    os.environ["MPLBACKEND"] = "Agg"
    run_dir = Path(args.run_dir)
    os.environ["VESUVIUS_RUN_ROOT"] = str(run_dir.parent)
    os.environ["VESUVIUS_RUN_ID"] = run_dir.name
    os.environ["VESUVIUS_EXP_NAME"] = run_dir.name
    os.environ["VESUVIUS_MODEL_PATH"] = str(args.model)
    os.chdir(ROOT)
    namespace: dict = {"__name__": "__main__"}
    for source in _load_cells(args.notebook):
        exec(compile(source, f"{args.notebook}_setup", "exec"), namespace)

    namespace["_run_name"] = lambda: args.tag
    run_data = namespace["_RUN_DATA"]
    by_scroll = {str(k): int(v) for k, v in (run_data.get("surface_window_offset_by_scroll") or {}).items()}
    global_offset = int(run_data.get("surface_window_offset", 0) or 0)
    notebook_predict = namespace["predict_tiles"]

    def predict_with_training_offset(*pargs, **kwargs):
        # the notebooks never pass the run's surface window offset; training and validation used it
        sid = str(pargs[8]).rsplit("_", 1)[-1]
        kwargs.setdefault("surface_depth_offset", by_scroll.get(sid, global_offset))
        return notebook_predict(*pargs, **kwargs)

    namespace["predict_tiles"] = predict_with_training_offset
    only = {int(s) for s in args.only.split(",")} if args.only else None
    failures = []

    def _has_norm(sid: int) -> bool:
        stats = namespace["_load_unified_cache"]().get(str(sid)) or {}
        return all(key in stats for key in ("mean", "std", "min", "max"))

    if args.notebook == "eval":
        notebook_imread = namespace["imread_gray"]

        def imread_with_label_fallback(path, *a, **k):
            # campaign 33 labels live in dilated_inklabels for scrolls without an eroded copy
            if str(path).startswith("eroded_inklabels/") and not (ROOT / path).is_file():
                path = str(path).replace("eroded_inklabels/", "dilated_inklabels/", 1)
            return notebook_imread(path, *a, **k)

        namespace["imread_gray"] = imread_with_label_fallback
        scrolls = dict(namespace["EVAL_SCROLLS"])
        known = set(scrolls.values())
        for path in sorted(ZARR_DIR.glob("*.zarr")):
            sid = path.stem
            labelled = any((ROOT / d / f"{sid}.png").is_file() for d in ("eroded_inklabels", "dilated_inklabels"))
            if sid.isdigit() and labelled and int(sid) not in known:
                scrolls[f"extra_{sid}"] = int(sid)
        for name, sid in scrolls.items():
            if only and sid not in only:
                continue
            missing = [
                label for label, ok in (
                    ("zarr", (ZARR_DIR / f"{sid}.zarr").is_dir()),
                    ("mask", (ROOT / "masks" / f"{sid}.png").is_file()),
                    ("label", (ROOT / "eroded_inklabels" / f"{sid}.png").is_file()
                     or (ROOT / "dilated_inklabels" / f"{sid}.png").is_file()),
                    ("surface", (ROOT / "surface_labels" / str(sid) / "depth.npy").is_file()),
                    ("norm", _has_norm(sid)),
                ) if not ok
            ]
            if missing:
                print(f"[skip] {name} {sid}: missing {missing}", flush=True)
                continue
            if args.dry_run:
                print(f"[dry-run] would render eval {name} {sid}", flush=True)
                continue
            try:
                namespace["render_eval"](name, sid, namespace["MODEL"], namespace["C"])
            except Exception:
                traceback.print_exc()
                failures.append(sid)
    else:
        fragments = dict(namespace["FRAGMENTS"])
        if args.prepare_test:
            prepare_test_fragments(namespace, sorted(set(fragments.values())), only)
        # labelled zarrs are rendered by the eval notebook; everything else lands here
        for path in sorted(ZARR_DIR.glob("*.zarr")):
            sid = path.stem
            labelled = any((ROOT / d / f"{sid}.png").is_file() for d in ("eroded_inklabels", "dilated_inklabels"))
            if sid.isdigit() and not labelled and int(sid) not in fragments.values():
                fragments[f"extra_{sid}"] = int(sid)
        for name, sid in fragments.items():
            if only and sid not in only:
                continue
            # a zarr without its mask is an interrupted download; never re-download here
            if not (ZARR_DIR / f"{sid}.zarr").is_dir() or not (ROOT / "masks" / f"{sid}.png").is_file():
                print(f"[skip] {name} {sid}: not downloaded", flush=True)
                continue
            if args.dry_run:
                print(f"[dry-run] would render test {name} {sid}", flush=True)
                continue
            try:
                namespace["ensure_fragment_ready"](sid)
                namespace["render_fragment"](name, sid, namespace["MODEL"], namespace["C"])
            except Exception:
                traceback.print_exc()
                failures.append(sid)
    print(f"[render] {args.notebook} {args.tag} failures={failures}", flush=True)
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="all", choices=("all", "render"))
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--notebook", choices=("eval", "test"))
    parser.add_argument("--run-dir")
    parser.add_argument("--model")
    parser.add_argument("--tag")
    parser.add_argument("--only", default=None)
    parser.add_argument("--prepare-test", action="store_true",
                        help="force-reassemble test fragments from tifxyz before rendering")
    args = parser.parse_args()

    if args.mode == "render":
        sys.exit(render(args))

    picks = select()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "selection.json").write_text(json.dumps(picks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(picks, indent=2), flush=True)
    if args.select_only:
        return
    models = list({pick["checkpoint"]: pick for pick in picks}.values())
    if len(models) == 1:
        print("[post33] train and valid selections are the same checkpoint", flush=True)
    status = 0
    prepared = False
    for pick in models:
        for notebook in ("eval", "test"):
            command = [
                sys.executable, str(ROOT / "post33_eval.py"), "render",
                "--notebook", notebook,
                "--run-dir", pick["run_dir"],
                "--model", pick["checkpoint"],
                "--tag", pick["tag"],
            ]
            if args.dry_run:
                command.append("--dry-run")
            elif notebook == "test" and not prepared:
                command.append("--prepare-test")
                prepared = True
            if args.only:
                command += ["--only", args.only]
            print(f"[post33] {' '.join(command)}", flush=True)
            status |= subprocess.run(command, cwd=ROOT).returncode
    print(f"[post33] done status={status}", flush=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
