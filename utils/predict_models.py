"""inference of the three mix6 models on one sheet, each the way it reads a scroll natively.

  cue_dropout  the sequence model (train.py): flat depth-13 surface (window offset -1 for 0846A), one probability per 16 px cell;
               run in the caller's process (ensemble_eval.ipynb / 0846A_eval.ipynb replay the checkpoint's run config)
  reader_v3    koine_machines.inference.infer in reader-v3/.venv: the checkpoint's 17 central layers, 384 px tiles, hann blend,
               overlap 0.5, one probability per pixel
  v8in         ink8um in nader-fork/.venv: the central 24 layers, one probability per pixel (stride 21)

`detect_kind(checkpoint)` tells them apart; `mix6.MODEL_PATHS` is where the trainings leave them.
"""
from __future__ import annotations

import glob
import itertools
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .mix6 import zarr_mtime

REPO = Path(__file__).resolve().parent.parent
READER_DIR = Path(os.environ.get("READER_V3_DIR", REPO / "reader-v3"))
FORK_DIR = Path(os.environ.get("NADER_FORK_DIR", REPO / "nader-fork"))
KINDS = ("cue_dropout", "reader_v3", "v8in")
V8IN_DEPTH = 24
# renders are thousands of 64 KB chunk files on the network drive: read serially, one sheet takes ~90 s
LOCAL_ZARR_DIR = Path(os.environ.get("VES_LOCAL_ZARR_DIR", "/dev/shm/ves_zarr_cache"))
READ_WORKERS = 32


def read_volume(zarr_path) -> np.ndarray:
    """the whole zarr array in memory, one thread per chunk block."""
    import zarr

    array = zarr.open(str(zarr_path), mode="r")
    out = np.empty(array.shape, array.dtype)
    origins = itertools.product(*[range(0, size, step) for size, step in zip(array.shape, array.chunks)])
    blocks = [tuple(slice(start, start + step) for start, step in zip(origin, array.chunks)) for origin in origins]

    def fill(block):
        out[block] = array[block]

    with ThreadPoolExecutor(READ_WORKERS) as pool:
        list(pool.map(fill, blocks))
    return out


def local_zarr(zarr_path) -> Path:
    """a copy of the render in RAM (/dev/shm) with 512 px chunks, rebuilt when the render is regenerated."""
    import zarr

    source = Path(zarr_path)
    target = LOCAL_ZARR_DIR / source.name
    if target.is_dir() and zarr_mtime(target) >= zarr_mtime(source):
        return target
    volume = read_volume(source)
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.rmtree(target, ignore_errors=True)
    zarr.save_array(str(partial), volume, chunks=(volume.shape[0], 512, 512), compressor=None)
    partial.rename(target)
    return target


# importing torch & co. from a venv on the network drive opens ~30k small files at ~1 ms each (~1 min per subprocess start)
# local disk, not /dev/shm: that is mounted noexec, so shared libraries could not be loaded from it
LOCAL_VENV_DIR = Path(os.environ.get("VES_LOCAL_VENV_DIR", "/root/.cache/ves_venvs"))
BIG_FILE = 1 << 20


def venv_python(venv) -> str:
    """python of a RAM mirror of the venv: small files copied, big ones (shared libraries, cached fine) symlinked back.
    Rebuilt when packages are installed or removed (site-packages changes)."""
    venv = Path(venv)
    site = next((venv / "lib").glob("python3*/site-packages"))
    stamp = f"{venv.resolve()} {site.stat().st_mtime} {(venv / 'pyvenv.cfg').stat().st_mtime}"
    target = LOCAL_VENV_DIR / f"{venv.parent.name}{venv.name}"
    if (target / ".mirror_stamp").is_file() and (target / ".mirror_stamp").read_text() == stamp:
        return str(target / "bin" / "python")
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    print(f"[predict_models] mirroring {venv} to {target} (once per venv change / pod restart)", flush=True)
    jobs = []
    for root, dirs, files in os.walk(venv):
        local = partial / Path(root).relative_to(venv)
        local.mkdir(parents=True, exist_ok=True)
        for name in [*dirs, *files]:
            source = Path(root) / name
            if source.is_symlink():
                (local / name).symlink_to(os.readlink(source))
                if name in dirs:
                    dirs.remove(name)
            elif name in files:
                jobs.append((source, local / name))

    def place(job):
        source, destination = job
        if source.stat().st_size >= BIG_FILE:
            destination.symlink_to(source)
        else:
            shutil.copy2(source, destination)

    with ThreadPoolExecutor(READ_WORKERS) as pool:
        list(pool.map(place, jobs))
    (partial / ".mirror_stamp").write_text(stamp)
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    return str(target / "bin" / "python")

V8IN_PREDICT = '''
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from ink8um import InkDetector, predict_surface
model = InkDetector.from_pretrained(sys.argv[2])
prediction = predict_surface(model, sys.argv[3], reverse=model.reverse_layers, batch_size=int(sys.argv[5]))
np.save(sys.argv[4], prediction.astype(np.float16))
'''


def detect_kind(checkpoint) -> str:
    """cue_dropout (a flat state dict of the sequence model), reader_v3 (a koine checkpoint with 'model' and 'config') or v8in
    (an exported model folder, model.safetensors, or a Lightning .ckpt)."""
    path = Path(str(checkpoint))
    if path.is_dir():
        if (path / "model.safetensors").is_file():
            return "v8in"
        raise ValueError(f"{path} is a folder without model.safetensors")
    if path.suffix in (".ckpt", ".safetensors"):
        return "v8in"
    if path.suffix == ".pth":
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        if isinstance(state, dict) and "model" in state and "config" in state:
            return "reader_v3"
        if isinstance(state, dict) and ("enc1.net.0.weight" in state or "model_state_dict" in state):
            return "cue_dropout"
    raise ValueError(f"cannot tell which of {KINDS} the checkpoint {path} belongs to")


def reader_depth(checkpoint) -> int:
    import torch

    return int(torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)["config"]["patch_size"][0])


def native_window(kind: str, depth_count: int, checkpoint=None) -> tuple[int, int]:
    """slice range [start, end) of the render a model reads: v8-in the central 24, reader-v3 the central patch depth (17)."""
    if kind == "v8in":
        start = (depth_count - V8IN_DEPTH) // 2
        return start, start + V8IN_DEPTH
    if kind == "reader_v3":
        depth = reader_depth(checkpoint) if checkpoint is not None else 17
        start = depth_count // 2 - depth // 2  # center_crop_layer_indices of koine_machines.inference.infer
        return start, start + depth
    raise ValueError("the window of the sequence model depends on the surface (flat 13 + offset)")


def checkpoint_mtime(checkpoint) -> float:
    path = Path(str(checkpoint))
    path = path / "model.safetensors" if path.is_dir() else path
    matches = [path] if path.exists() else [Path(match) for match in glob.glob(str(checkpoint))]
    return max((match.stat().st_mtime for match in matches), default=0.0)


def drop_stale(cached, zarr_path, checkpoint) -> None:
    """remove cached predictions older than their render or checkpoint, so a regenerated sheet or a retrained model is predicted again."""
    newest = max(zarr_mtime(zarr_path), checkpoint_mtime(checkpoint))
    for path in map(Path, cached):
        if path.is_file() and path.stat().st_mtime < newest:
            path.unlink()


def predict_reader(zarr_path, checkpoint, tif_path, batch_size: int = 4) -> np.ndarray:
    """reader-v3 map in [0, 1] at the render's pixel grid (the CLI's uint8 tiff is kept next to the cached .npy)."""
    tif, python = Path(tif_path), venv_python(READER_DIR / ".venv")
    drop_stale([tif, tif.with_suffix(".npy")], zarr_path, checkpoint)
    if not tif.is_file():
        tif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([python, "-m", "koine_machines.inference.infer", str(local_zarr(zarr_path)), str(checkpoint), str(tif), "--overlap", "0.5",
                        "--blend-mode", "hann", "--direction", "forward", "--no-compile", "--batch-size", str(batch_size)],
                       cwd=READER_DIR, check=True)
    npy = tif.with_suffix(".npy")
    if not npy.is_file():  # imagecodecs (LZW) lives in the reader environment
        subprocess.run([python, "-c", "import sys, numpy as np, tifffile; np.save(sys.argv[2], tifffile.imread(sys.argv[1]))",
                        str(tif), str(npy)], check=True)
    return np.load(npy).astype(np.float32) / 255.0


def v8in_model_dir(checkpoint, export_dir) -> Path:
    """the folder InkDetector.from_pretrained reads: the given exported folder, or a Lightning .ckpt (glob allowed) exported once."""
    path = Path(str(checkpoint))
    if path.is_dir():
        return path
    matches = sorted(glob.glob(str(checkpoint)))
    if not matches:
        raise FileNotFoundError(f"no checkpoint matches {checkpoint}")
    export_dir = Path(export_dir)
    exported = export_dir / "model.safetensors"
    if exported.is_file() and exported.stat().st_mtime < Path(matches[-1]).stat().st_mtime:
        exported.unlink()  # the checkpoint was retrained since it was exported
    if not exported.is_file():
        subprocess.run([venv_python(FORK_DIR / ".venv"), str(FORK_DIR / "training" / "export_checkpoint.py"), matches[-1],
                        str(export_dir)], check=True)
    return export_dir


def v8in_layers(zarr_path, folder) -> Path:
    """the central 24 layers of the render as 00.tif..23.tif (uint8, forward order)."""
    import tifffile
    import zarr

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    volume = zarr.open(str(local_zarr(zarr_path)), mode="r")
    start = (volume.shape[0] - V8IN_DEPTH) // 2
    slab = np.asarray(volume[start:start + V8IN_DEPTH])
    for index in range(V8IN_DEPTH):
        path = folder / f"{index:02d}.tif"
        if not path.is_file():
            tifffile.imwrite(path, np.clip(slab[index], 0, 255).astype(np.uint8))
    return folder


def predict_v8in(zarr_path, checkpoint, cache_dir, batch_size: int = 16, layers=None) -> np.ndarray:
    """v8-in map in [0, 1] at the render's pixel grid, cached in cache_dir. `layers` = an already staged folder of the 24 tiffs,
    otherwise they are staged in a scratch folder that is removed afterwards."""
    import shutil
    import tempfile

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = cache_dir / "v8in_prediction.npy"
    drop_stale([prediction_path], zarr_path, checkpoint)
    if not prediction_path.is_file():
        model_dir = v8in_model_dir(checkpoint, cache_dir / "v8in_model")
        LOCAL_ZARR_DIR.mkdir(parents=True, exist_ok=True)
        scratch = None if layers is not None else tempfile.mkdtemp(dir=LOCAL_ZARR_DIR)
        try:
            layers = Path(layers) if layers is not None else v8in_layers(zarr_path, Path(scratch) / "layers")
            subprocess.run([venv_python(FORK_DIR / ".venv"), "-c", V8IN_PREDICT, str(FORK_DIR), str(model_dir), str(layers),
                            str(prediction_path), str(batch_size)], check=True)
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
    return np.load(prediction_path).astype(np.float32)
