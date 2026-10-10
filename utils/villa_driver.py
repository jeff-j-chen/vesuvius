#!/usr/bin/env python3
"""villa_driver.py -- run the four PHerc0343 "First Letters" ink models on one surface volume.

Runs inside the isolated villa environment (/opt/villa-venv, villa @ 0fb38c45, python 3.14), not the project
python. Reproduces Nieuwlaar/pherc343-first-letters ink/scripts/run_ink.sh exactly, but loads each model once
per fragment and prints machine-readable progress lines for the notebook:

    PROGRESS <task> <done> <total>        patches finished in the current map
    TASK <task> <seconds>                 one map written

  villa models (dnative, hybrid_3d2d-seed42/43): vesuvius.ink_detection.inference.infer, overlap 0.5, hann
      blend, 17-layer window s..s+16 with s = N//2 + d - 8; outputs <name>__<model>__L<N>s<s>[_reverse].tif
  hecate: hecate.py on a 9.6 um isotropic resample (hecate_prep.py algorithm), 16 planes centred on native layer
      (N-1)/2 + d; outputs <name>__hecate__L<N>zc<zc>[_reverse].png on the 9.6 um grid

usage: villa_driver.py --surface <id>.zarr --out <maps_dir> --ckpt-dir <dir> --native-um 9.362
                       [--models ...] [--depths 0 1] [--directions forward reverse]
"""
import argparse
import gc
import importlib.util
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

VILLA_CKPTS = {
    "dnative": "dense_native-016000.rebuilt.pth",
    "hybrid_3d2d-seed42": "hybrid_3d2d-seed42/step-075000.pth",
    "hybrid_3d2d-seed43": "hybrid_3d2d-seed43/step-075000.pth",
}
HECATE_CKPT = "hecate_9.6um.pth"
V8IN_DIR = "v8in_loo_w062"
ALL_MODELS = ("dnative", "hybrid_3d2d-seed42", "hybrid_3d2d-seed43", "hecate", "v8in-loo-w062")


def emit(*parts):
    print(" ".join(str(p) for p in parts), flush=True)


def villa_name(name, model, n, d, direction):
    s = n // 2 + d - 8
    return f"{name}__{model}__L{n}s{s}{'_reverse' if direction == 'reverse' else ''}.tif"


def v8in_name(name, n, d, direction, stride=None):
    tag = f"st{stride}" if stride else ""
    return f"{name}__v8in-loo-w062__L{n}s{(n - 24) // 2 + d}{tag}{'_reverse' if direction == 'reverse' else ''}.tif"


def run_v8in(surface, out, ckpt_dir, name, n, depths, directions, batch, stride=None):
    """YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062 via its own ink8um.predict_stack (24 centred layers + d).

    Its training renders (PHerc1447 HF) are stored opposite to our orientation and read reversed, which equals
    our zarr order, so direction 'forward' = reverse=False here."""
    import tifffile
    import zarr

    todo = [(d, r) for d in depths for r in directions if not (out / v8in_name(name, n, d, r, stride)).exists()]
    if not todo:
        return
    sys.path.insert(0, str(ckpt_dir / V8IN_DIR))
    from ink8um import InkDetector, predict_stack

    model = InkDetector.from_pretrained(str(ckpt_dir / V8IN_DIR))
    root = zarr.open(str(surface), mode="r")
    vol = np.asarray((root if hasattr(root, "shape") else root["0"])[:])
    for d in sorted({d for d, _ in todo}):
        start = (n - model.in_depth) // 2 + d
        if start < 0 or start + model.in_depth > n:
            print(f"v8in d={d} needs layers {start}..{start + model.in_depth - 1}, outside 0..{n - 1}; skipped",
                  flush=True)
            continue
        stack = np.ascontiguousarray(np.clip(vol[start:start + model.in_depth], 0, model.clip_max).transpose(1, 2, 0))
        for direction in [r for dd, r in todo if dd == d]:
            target = out / v8in_name(name, n, d, direction, stride)
            t0 = time.time()
            prob = predict_stack(model, stack, reverse=direction == "reverse", batch_size=batch, stride=stride,
                                 progress=lambda done, total: emit("PROGRESS", target.name, done, total))
            partial = target.with_name(target.stem + ".partial.tif")
            tifffile.imwrite(partial, (prob * 65535).round().astype(np.uint16), compression="zlib")
            partial.replace(target)
            emit("TASK", target.name, f"{time.time() - t0:.0f}")


def hecate_name(name, n, d, direction):
    return f"{name}__hecate__L{n}zc{(n - 1) / 2 + d:.2f}{'_reverse' if direction == 'reverse' else ''}.png"


def run_villa(model_key, surface, out, ckpt_dir, name, n, depths, directions, batch, compile_model):
    import torch
    from vesuvius.ink_detection.inference import infer

    todo = [(d, r) for d in depths for r in directions
            if not (out / villa_name(name, model_key, n, d, r)).exists()]
    if not todo:
        return
    args = types.SimpleNamespace(
        checkpoint=ckpt_dir / VILLA_CKPTS[model_key], amp_dtype="auto", overlap=0.5, stride=None,
        blend_mode="hann", batch_size=batch, gpu_ids=(), num_workers=4, prefetch_factor=2, resolution="0",
        mask_path=None, tta_mirror=False, tta_batch_size=None, layer_start=None, layer_end=None)
    torch.set_float32_matmul_precision("high")
    configured = infer.configure_model(args)
    model, device = infer.prepare_model_for_inference(
        configured.model, gpu_ids=(), compile_model=compile_model, compile_mode="reduce-overhead")
    configured = infer.replace(configured, model=model)

    task = {"name": ""}
    # keep the original class: re-wrapping per model would stack progress wrappers
    base_loader = getattr(infer, "_original_DataLoader", infer.DataLoader)
    infer._original_DataLoader = base_loader

    class ProgressLoader(base_loader):
        def __iter__(self):
            total, done = len(self.dataset), 0
            emit("PROGRESS", task["name"], 0, total)
            for batch_ in super().__iter__():
                yield batch_
                done += len(batch_[1])
                emit("PROGRESS", task["name"], done, total)

    infer.DataLoader = ProgressLoader
    for d, direction in todo:
        s = n // 2 + d - 8
        if s < 0 or s + 17 > n:
            raise SystemExit(f"d={d} needs layers {s}..{s + 16}, outside 0..{n - 1}")
        args.layer_start, args.layer_end = s, s + 17
        target = out / villa_name(name, model_key, n, d, direction)
        task["name"] = target.name
        t0 = time.time()
        partial = target.with_name(target.stem + ".partial.tif")
        infer.infer_single_zarr(args=args, input_zarr=surface, configured_model=configured, device=device,
                                output_tiff=partial, layer_direction=direction)
        partial.replace(target)
        emit("TASK", target.name, f"{time.time() - t0:.0f}")
    del configured, model
    torch.cuda.empty_cache()


def hecate_prep(vol, native_um, z_center, nz=16, target_um=9.6):
    """hecate_prep.py: linear z resample to target_um around z_center, then in-plane resize to target_um."""
    import cv2

    depth, height, width = vol.shape
    pos = z_center + (np.arange(nz) - (nz - 1) / 2.0) * (target_um / native_um)
    if pos.min() < 0 or pos.max() > depth - 1:
        raise SystemExit(f"hecate z window {pos.min():.2f}..{pos.max():.2f} lies outside 0..{depth - 1}")
    lo = np.floor(pos).astype(int)
    frac = (pos - lo).astype(np.float32)
    hi = np.minimum(lo + 1, depth - 1)
    s = native_um / target_um
    nh, nw = max(1, int(round(height * s))), max(1, int(round(width * s)))
    interp = cv2.INTER_AREA if s < 0.95 else cv2.INTER_LINEAR
    out = np.empty((nz, nh, nw), np.uint8)
    # one plane at a time: the full float32 stack is ~13 GB on the largest segments
    for i in range(nz):
        plane = vol[lo[i]].astype(np.float32) * (1 - frac[i]) + vol[hi[i]].astype(np.float32) * frac[i]
        out[i] = np.clip(cv2.resize(plane, (nw, nh), interpolation=interp), 0, 255).astype(np.uint8)
    return out


def run_hecate(surface, out, ckpt_dir, name, n, depths, directions, batch, native_um, precision="fp32"):
    import torch
    import zarr

    todo = [(d, r) for d in depths for r in directions if not (out / hecate_name(name, n, d, r)).exists()]
    if not todo:
        return
    spec = importlib.util.spec_from_file_location("hecate", ckpt_dir / "hecate.py")
    hecate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hecate)
    model = hecate.load_model(ckpt_dir / HECATE_CKPT, "cuda" if torch.cuda.is_available() else "cpu")
    task = {"name": ""}

    def progress_print(message, **_kwargs):
        done, total = str(message).split()[0].split("/")
        emit("PROGRESS", task["name"], done, total)

    hecate.print = progress_print
    root = zarr.open(str(surface), mode="r")
    vol = np.asarray((root if hasattr(root, "shape") else root["0"])[:])
    if vol.dtype != np.uint8:
        raise SystemExit(f"hecate needs a uint8 surface volume, got {vol.dtype}")
    for d in sorted({d for d, _ in todo}):
        prepped = hecate_prep(vol, native_um, (n - 1) / 2 + d)
        for direction in [r for dd, r in todo if dd == d]:
            target = out / hecate_name(name, n, d, direction)
            task["name"] = target.name
            t0 = time.time()
            with tempfile.TemporaryDirectory(prefix="hecate-") as tmp:
                result = np.memmap(Path(tmp) / "prediction", mode="w+", dtype="uint8", shape=prepped.shape[-2:])
                hecate.predict(model, prepped, result, direction == "reverse", None, batch, precision)
                partial = target.with_name(target.stem + ".partial.png")
                hecate.write_png(result, partial)
                partial.replace(target)
                del result
            emit("TASK", target.name, f"{time.time() - t0:.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--native-um", required=True, type=float, help="voxel size of the volume the mesh was rendered from")
    ap.add_argument("--name", help="map name prefix (default: surface folder name without .zarr)")
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    ap.add_argument("--depths", nargs="+", type=int, default=[0])
    ap.add_argument("--directions", nargs="+", default=["forward", "reverse"], choices=("forward", "reverse"))
    ap.add_argument("--villa-batch", type=int, default=4)
    ap.add_argument("--hecate-batch", type=int, default=4)
    ap.add_argument("--hecate-precision", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--v8in-stride", type=int, default=None,
                    help="v8in tile stride (default: model config, 21); maps at other strides are named ...st<stride>")
    ap.add_argument("--no-compile", action="store_true", help="skip torch.compile for the villa models")
    a = ap.parse_args()
    import torch
    import zarr

    root = zarr.open(str(a.surface), mode="r")
    n = int((root if hasattr(root, "shape") else root["0"]).shape[0])
    name = a.name or a.surface.name.removesuffix(".zarr")
    a.out.mkdir(parents=True, exist_ok=True)
    for model_key in a.models:
        emit("MODEL", model_key)
        if model_key == "hecate":
            run_hecate(a.surface, a.out, a.ckpt_dir, name, n, a.depths, a.directions, a.hecate_batch, a.native_um, a.hecate_precision)
        elif model_key == "v8in-loo-w062":
            run_v8in(a.surface, a.out, a.ckpt_dir, name, n, a.depths, a.directions, 16, a.v8in_stride)
        else:
            run_villa(model_key, a.surface, a.out, a.ckpt_dir, name, n, a.depths, a.directions, a.villa_batch,
                      not a.no_compile)
        gc.collect()
        torch.cuda.empty_cache()
    emit("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
