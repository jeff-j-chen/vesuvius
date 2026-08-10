"""platform.py -- detect which machine we're running on and provide platform-specific defaults.

three platforms:
  1. windows (jeff's laptop): C:\\Users\\ChenJeff\\Documents\\ves_zarrs2
  2. linux-desktop (jeff's pop-os): /media/jeff/Seagate/vesuvius (this machine!)
  3. linux-runpod (remote server): /vesuvius

detection logic: check for /media/jeff/Seagate/ existence (desktop-specific path).
if exists -> desktop; if posix but not exists -> runpod; otherwise windows.
"""
import os
from typing import Literal

PlatformType = Literal["windows", "linux-desktop", "linux-runpod"]


def detect_platform() -> PlatformType:
    """detect which machine we're on based on filesystem markers"""
    if os.name != "posix":
        return "windows"
    # both linux machines are posix, differentiate by desktop-specific marker
    if os.path.exists("/media/jeff/Seagate/"):
        return "linux-desktop"
    return "linux-runpod"


def get_zarr_dir() -> str:
    """platform-aware default zarr directory (respects VESUVIUS_ZARR_PATH override)"""
    override = os.getenv("VESUVIUS_ZARR_PATH")
    if override:
        return override
    
    platform = detect_platform()
    if platform == "windows":
        return r"C:\Users\ChenJeff\Documents\ves_zarrs2"
    elif platform == "linux-desktop":
        return "/media/jeff/Seagate/ves_zarrs2"
    else:  # linux-runpod
        return "/vesuvius/ves_zarrs2"


def get_workspace_dir() -> str:
    """platform-aware workspace root (where train.py / utils/ etc live)"""
    platform = detect_platform()
    if platform == "windows":
        return r"C:\Users\ChenJeff\Documents\vesuvius"
    elif platform == "linux-desktop":
        return "/media/jeff/Seagate/vesuvius"
    else:  # linux-runpod
        return "/vesuvius"


def is_high_perf() -> bool:
    """true if running on a high-performance linux box (runpod), false for laptop/windows"""
    return detect_platform() == "linux-runpod"


def get_default_workers() -> int:
    """default dataloader workers for this platform"""
    platform = detect_platform()
    if platform == "windows":
        return 0  # windows multiprocessing issues
    elif platform == "linux-desktop":
        return 0  # external HDD - single-threaded is faster than multi-worker I/O contention
    else:  # linux-runpod
        return 8  # fast SSD can handle multiple workers


def get_default_batch_size() -> int:
    """default batch size for this platform"""
    platform = detect_platform()
    if platform == "windows":
        return 32  # rtx 4090 (24gb vram) but conservative
    elif platform == "linux-desktop":
        return 16  # rtx 3060 mobile (6gb vram)
    else:  # linux-runpod
        return 96  # rtx 4080 (16gb vram)


def get_default_eval_bs() -> int:
    """default eval/inference batch size"""
    platform = detect_platform()
    if platform == "windows":
        return 96  # rtx 4090 (24gb vram) can handle larger inference batches
    elif platform == "linux-desktop":
        return 48  # rtx 3060 mobile (6gb vram) conservative
    else:  # linux-runpod
        return 256  # rtx 4080 (16gb vram)


def get_default_lr() -> float:
    """default learning rate scaled for batch size"""
    batch_size = get_default_batch_size()
    # empirically tested: bs=32->1e-4, bs=96->1.5e-4
    # use linear interpolation for other batch sizes
    if batch_size >= 96:
        return 1.5e-4
    elif batch_size >= 32:
        return 1.0e-4
    elif batch_size >= 16:
        return 7e-5  # conservative for smaller batches (noisier gradients)
    else:
        return 5e-5  # very small batches
