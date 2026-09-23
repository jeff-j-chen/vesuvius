import numpy as np
import torch
import torch.multiprocessing as torch_multiprocessing
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from collections import Counter
import zarr
import cv2
import math
import random
import os
import uuid
import atexit
import tempfile
import threading
from functools import partial
from typing import Iterator
from .config import Config
import json
from tqdm import tqdm

UNIFIED_CACHE_PATH = "./norm_cache.json"

# DataLoader's default file-descriptor tensor sharing can exhaust RLIMIT_NOFILE
# after many persistent-worker lifecycles. File-system sharing preserves worker
# throughput and campaign-level RAM caches without retaining one FD per transfer.
torch_multiprocessing.set_sharing_strategy("file_system")


_PREPARED_DATASET_CACHE: dict[tuple, dict] = {}
_PREPARED_DATASET_CACHE_LOCK = threading.Lock()


def _freeze_cache_value(value):
    """convert configuration values into stable, hashable cache-key parts."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_cache_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_cache_value(item) for item in value))
    return value


def _selected_config_key(section, names) -> tuple:
    """snapshot selected config values without retaining the config object."""
    return tuple(
        (name, _freeze_cache_value(getattr(section, name, None)))
        for name in names
    )


def _path_fingerprint(path: str) -> tuple:
    """identify an input path and invalidate cached preparation after source changes."""
    absolute = os.path.abspath(path)
    try:
        stat = os.stat(absolute)
    except OSError:
        return absolute, None, None
    return absolute, int(stat.st_size), int(stat.st_mtime_ns)


def needs_domain_ids(config: Config) -> bool:
    """whether dataset samples must carry their physical-domain identifier."""
    return any([
        bool(getattr(config.tra, "dann", False)),
        bool(getattr(config.tra, "supcon_cross_frag", False)),
        bool(getattr(config.tra, "per_scroll_metrics", False)),
        bool(getattr(config.tra, "prototype_align", False)),
        bool(getattr(config.tra, "coral_align", False)),
        bool(getattr(config.tra, "cdan", False)),
        bool(getattr(config.tra, "mldg", False)),
        bool(getattr(config.tra, "physical_domain_groupdro", False)),
        bool(getattr(config.tra, "domain_vrex", False)),
        bool(getattr(config.tra, "domain_cvar", False)),
        bool(getattr(config.tra, "pcgrad", False)),
        bool(getattr(config.tra, "pcgrad_lite", False)),
        bool(getattr(config.tra, "pcgrad_gram", False)),
        bool(getattr(config.tra, "domain_gradient_mode", "")),
        bool(getattr(config.model, "mixstyle", False)),
        bool(getattr(config.model, "sagnet", False)),
    ])


def needs_patch_ids(config: Config) -> bool:
    """whether dataset samples must carry their individual patch identifier."""
    return any([
        bool(getattr(config.tra, "per_scroll_metrics", False)),
        bool(getattr(config.tra, "physical_patch_groupdro", False)),
    ])


def imread_gray(path):
    """grayscale PNG reader that survives huge (>1 Gpx) images. cv2.imread enforces a
    ~1.07 Gpx cap (and this build ignores CV_IO_MAX_IMAGE_PIXELS), raising on native 2.4um
    masks/labels (~1.3 Gpx). fall back to PIL, which we uncap. returns uint8 ndarray or None."""
    if not os.path.exists(path):
        return None
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except cv2.error:
        pass
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(path).convert("L"))


# ---- memmap scratch backing for mask/labels --------------------------------
# at the 5-10 fragment scale the per-scroll uint8 mask/labels (hundreds of MB
# each for the big scroll) get pickled to every spawned DataLoader worker on
# windows, multiplying RAM by (1 + num_workers) and risking the spawn pickle
# crash. backing them with an on-disk memmap fixes this: a memmap-backed dataset
# pickles only the FILE PATH (a few bytes) instead of the array, and every
# process mmaps the same read-only file so the OS shares one set of pages.
#
# NB: a numpy memmap pickled directly would MATERIALIZE its data (defeating the
# purpose), so the dataset must store the path and exclude the open memmap from
# its pickled state (see InkVolumeDataset.__getstate__), reopening lazily.

# files created by THIS process, cleaned up at its exit. spawned workers reimport
# this module fresh (empty list, own pid), so they never delete the creator's files.
_MMAP_FILES = []
_MMAP_OWNER_PID = os.getpid()
_MMAP_STALE_CLEANED = False


def _process_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_stale_mmap_files(directory):
    global _MMAP_STALE_CLEANED
    if _MMAP_STALE_CLEANED:
        return
    _MMAP_STALE_CLEANED = True
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if not name.startswith("mm_"):
            continue
        try:
            owner_pid = int(name.split("_", 2)[1])
        except (IndexError, ValueError):
            continue
        if owner_pid == _MMAP_OWNER_PID or _process_is_alive(owner_pid):
            continue
        try:
            os.remove(os.path.join(directory, name))
        except OSError:
            pass


def _mmap_scratch_dir():
    """scratch directory for memmap backing files (override via VESUVIUS_MMAP_DIR)"""
    d = os.environ.get("VESUVIUS_MMAP_DIR")
    if not d:
        preferred_root = "/data/extra/tmp"
        scratch_root = preferred_root if os.path.isdir(preferred_root) else tempfile.gettempdir()
        d = os.path.join(scratch_root, "vesuvius_mmap")
    os.makedirs(d, exist_ok=True)
    _cleanup_stale_mmap_files(d)
    return d


def _write_memmap(arr, pack_bits=False, original_shape=None):
    """persist a (binary uint8) array to a unique .npy and return its path.
    if pack_bits=True, packs to 1 bit/pixel (8x smaller) and saves shape separately."""
    path = os.path.join(_mmap_scratch_dir(), f"mm_{os.getpid()}_{uuid.uuid4().hex}.npy")
    shape_path = path.replace('.npy', '_shape.npy') if pack_bits else None
    pending = [path, *([shape_path] if shape_path is not None else [])]
    temporary = [f"{output}.partial" for output in pending]
    try:
        value = np.packbits(arr.ravel()) if pack_bits else np.ascontiguousarray(arr)
        with open(temporary[0], "wb") as handle:
            np.save(handle, value)
        if shape_path is not None:
            with open(temporary[1], "wb") as handle:
                np.save(handle, np.array(original_shape or arr.shape, dtype=np.int32))
        for source, destination in zip(temporary, pending):
            os.replace(source, destination)
    except Exception:
        for output in [*temporary, *pending]:
            try:
                os.remove(output)
            except OSError:
                pass
        raise
    _MMAP_FILES.extend(pending)
    return path


def cleanup_mmap_files():
    """remove memmap files at interpreter exit, but only in the creating process.
    on windows a still-mapped file can refuse deletion; that is non-fatal (the
    files live in temp), so failures are swallowed."""
    if os.getpid() != _MMAP_OWNER_PID:
        return
    remaining = []
    for p in _MMAP_FILES:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except OSError:
            remaining.append(p)
    _MMAP_FILES[:] = remaining


atexit.register(cleanup_mmap_files)


def _is_norm_stats(entry):
    return isinstance(entry, dict) and all(k in entry for k in ("mean", "std", "min", "max"))


def _load_unified_cache(cache_path=UNIFIED_CACHE_PATH):
    """loads cache in legacy top-level-by-scroll layout"""
    try:
        with open(cache_path, "r") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        raw = {}

    if not isinstance(raw, dict):
        return {}

    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def _save_unified_cache(cache, cache_path=UNIFIED_CACHE_PATH):
    """saves legacy top-level-by-scroll cache to disk"""
    payload = cache if isinstance(cache, dict) else {}
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=4)

class Transform:
    """handles data augmentation transforms"""
    def __init__(self, config: Config):
        self.rotation_prob = float(getattr(config.dl, "rotation_prob", 0.25))
        self.flip_prob = float(getattr(config.dl, "flip_prob", 0.25))
        self.noise_prob = float(getattr(config.dl, "noise_prob", 0.30))
        self.brightness_prob = float(getattr(config.dl, "brightness_prob", 0.50))
        self.contrast_prob = float(getattr(config.dl, "contrast_prob", 0.50))
        # augmentation magnitudes (config-tracked; see DataloaderConfig)
        self.brightness_delta = float(getattr(config.dl, "brightness_delta", 0.15))
        self.contrast_delta   = float(getattr(config.dl, "contrast_delta", 0.15))
        self.noise_std_min    = float(getattr(config.dl, "noise_std_min", 0.001))
        self.noise_std_max    = float(getattr(config.dl, "noise_std_max", 0.005))
        # specaugment-style masking
        self.cutout_prob      = float(getattr(config.dl, "cutout_prob", 0.0))
        self.cutout_max_frac  = float(getattr(config.dl, "cutout_max_frac", 0.35))
        self.cutout_n_patches = int(getattr(config.dl, "cutout_n_patches", 1))
        self.cutout_protect_center = bool(getattr(config.dl, "cutout_protect_center", False))
        self.depth_mask_prob  = float(getattr(config.dl, "depth_mask_prob", 0.0))
        self.depth_mask_mode  = str(getattr(config.dl, "depth_mask_mode", "zero"))
        if self.depth_mask_mode not in ("zero", "interp"):
            raise ValueError("depth_mask_mode must be 'zero' or 'interp'")
        self.elastic_prob     = float(getattr(config.dl, "elastic_prob", 0.0))
        self.elastic_alpha    = float(getattr(config.dl, "elastic_alpha", 15.0))
        self.elastic_sigma    = float(getattr(config.dl, "elastic_sigma", 5.0))
        self.depth_warp_prob = float(getattr(config.dl, "depth_warp_prob", 0.0))
        self.depth_warp_max = float(getattr(config.dl, "depth_warp_max", 2.0))
        self.depth_warp_sigma = float(getattr(config.dl, "depth_warp_sigma", 24.0))
        self.surface_atten_prob = float(getattr(config.dl, "surface_atten_prob", 0.0))
        self.surface_atten_min = float(getattr(config.dl, "surface_atten_min", 0.1))
        self.surface_atten_max = float(getattr(config.dl, "surface_atten_max", 0.35))
        self.surface_atten_sigma = float(getattr(config.dl, "surface_atten_sigma", 2.0))
        self.acquisition_blur_prob = float(getattr(config.dl, "acquisition_blur_prob", 0.0))
        self.acquisition_blur_min = float(getattr(config.dl, "acquisition_blur_min", 0.4))
        self.acquisition_blur_max = float(getattr(config.dl, "acquisition_blur_max", 0.9))
        self.correlated_noise_prob = float(getattr(config.dl, "correlated_noise_prob", 0.0))
        self.correlated_noise_min = float(getattr(config.dl, "correlated_noise_min", 0.003))
        self.correlated_noise_max = float(getattr(config.dl, "correlated_noise_max", 0.015))
        self.correlated_noise_sigma = float(getattr(config.dl, "correlated_noise_sigma", 6.0))
        self.context_replace_keep_size = int(getattr(config.dl, "context_replace_keep_size", 0))
        self.context_replace_margin = int(getattr(config.dl, "context_replace_margin", 16))
        self.context_replace_feather = int(getattr(config.dl, "context_replace_feather", 16))
        self.context_replace_surface_align = bool(
            getattr(config.dl, "context_replace_surface_align", True)
        )
        self.tile_size = int(getattr(config.data, "tile_size", 16))
        self.multitile        = bool(getattr(config.model, "multitile", False))
        self.multitile_grid   = max(1, int(getattr(config.model, "multitile_grid", 1)))
        self.multitile_subtile = max(1, int(getattr(config.model, "multitile_subtile", 1)))
        self._warned_elastic_multitile = False

    def __call__(self, block, label=None, mask=None, component_ids=None, target_offset=None):
        """apply transforms, synchronizing discrete geometry with multitile targets."""
        self._last_geometry = []
        if random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])
            block = np.rot90(block, k=k, axes=(1, 2)).copy()
            self._last_geometry.append(("rotate", k))
            label = self._rotate_target(label, k)
            mask = self._rotate_target(mask, k)
            component_ids = self._rotate_target(component_ids, k)
            target_offset = self._rotate_offset(target_offset, k)
        if random.random() < self.flip_prob:
            axis = random.choice([1, 2])
            block = np.flip(block, axis=axis).copy()
            self._last_geometry.append(("flip", axis))
            target_axis = 0 if axis == 1 else 1
            label = self._flip_target(label, target_axis)
            mask = self._flip_target(mask, target_axis)
            component_ids = self._flip_target(component_ids, target_axis)
            target_offset = self._flip_offset(target_offset, axis)
        if random.random() < self.noise_prob:
            block = self._apply_gaussian_noise(block)
        if random.random() < self.brightness_prob:
            block = self._apply_brightness_adjustment(block)
        if random.random() < self.contrast_prob:
            block = self._apply_contrast_adjustment(block)
        if random.random() < self.cutout_prob:
            block = self._apply_cutout(block, target_offset)
        if self.depth_mask_prob > 0:
            block = self._apply_depth_mask(block)
        if self.depth_warp_prob > 0 and random.random() < self.depth_warp_prob:
            block = self._apply_smooth_depth_warp(block)
            self._last_geometry.append(("depth_warp", self._last_depth_warp_field))
        if self.surface_atten_prob > 0 and random.random() < self.surface_atten_prob:
            block = self._apply_surface_attenuation(block)
        if self.acquisition_blur_prob > 0 and random.random() < self.acquisition_blur_prob:
            block = self._apply_acquisition_blur(block)
        if self.correlated_noise_prob > 0 and random.random() < self.correlated_noise_prob:
            block = self._apply_correlated_noise(block)
        if self.elastic_prob > 0 and random.random() < self.elastic_prob:
            if self.multitile and label is not None:
                if not self._warned_elastic_multitile:
                    print("[augment] elastic disabled for multitile: dense target warp is not implemented")
                    self._warned_elastic_multitile = True
            else:
                block = self._apply_elastic_deformation(block)
                self._last_geometry.append(("elastic", self._last_elastic_coords))
        # ensure the final result is contiguous to avoid negative strides
        block = np.ascontiguousarray(block)
        if label is None and mask is None and component_ids is None and target_offset is None:
            return block
        if component_ids is None:
            if target_offset is None:
                return block, label.contiguous(), mask.contiguous()
            return block, label.contiguous(), mask.contiguous(), target_offset.contiguous()
        if target_offset is None:
            return block, label.contiguous(), mask.contiguous(), component_ids.contiguous()
        return (
            block,
            label.contiguous(),
            mask.contiguous(),
            component_ids.contiguous(),
            target_offset.contiguous(),
        )

    def transform_surface_teacher(self, depth, confidence):
        """apply the most recent discrete image geometry to dense teacher maps."""
        for operation, value in getattr(self, "_last_geometry", []):
            if operation == "rotate":
                depth = np.rot90(depth, k=value, axes=(0, 1)).copy()
                confidence = np.rot90(confidence, k=value, axes=(0, 1)).copy()
            elif operation == "flip":
                axis = 0 if value == 1 else 1
                depth = np.flip(depth, axis=axis).copy()
                confidence = np.flip(confidence, axis=axis).copy()
            elif operation == "depth_warp":
                valid = confidence > 0
                depth = np.where(valid, depth - value, depth)
            elif operation == "elastic":
                from scipy.ndimage import map_coordinates

                shape = depth.shape
                depth = map_coordinates(
                    depth,
                    value,
                    order=1,
                    mode="constant",
                    cval=-1.0,
                ).reshape(shape)
                confidence = map_coordinates(
                    confidence,
                    value,
                    order=1,
                    mode="constant",
                    cval=0.0,
                ).reshape(shape)
                confidence[depth < 0] = 0.0
        return depth, confidence

    def paired(self, block, paired_block, label, mask, component_ids=None, target_offset=None):
        """apply identical random transforms to an original/context-intervened pair."""
        py_state = random.getstate()
        np_state = np.random.get_state()
        primary = self(block, label, mask, component_ids, target_offset)
        random.setstate(py_state)
        np.random.set_state(np_state)
        paired = self(
            paired_block,
            label.clone(),
            mask.clone(),
            component_ids.clone() if component_ids is not None else None,
            target_offset.clone() if target_offset is not None else None,
        )
        paired_image = paired[0] if isinstance(paired, tuple) else paired
        return primary, paired_image

    def _target_grid(self, target):
        if target is None or not self.multitile:
            return None
        if target.numel() != self.multitile_grid * self.multitile_grid:
            return None
        return target.view(self.multitile_grid, self.multitile_grid)

    def _rotate_target(self, target, k):
        grid = self._target_grid(target)
        return torch.rot90(grid, k=k, dims=(0, 1)).reshape(-1) if grid is not None else target

    def _flip_target(self, target, axis):
        grid = self._target_grid(target)
        return torch.flip(grid, dims=(axis,)).reshape(-1) if grid is not None else target

    @staticmethod
    def _rotate_offset(target_offset, k):
        if target_offset is None:
            return None
        dy, dx = target_offset.unbind()
        for _ in range(k % 4):
            dy, dx = -dx, dy
        return torch.stack((dy, dx))

    @staticmethod
    def _flip_offset(target_offset, block_axis):
        if target_offset is None:
            return None
        out = target_offset.clone()
        out[0 if block_axis == 1 else 1] *= -1
        return out

    def _apply_elastic_deformation(self, block):
        """smooth elastic deformation on the XY plane, shared across all depth slices.
        uses the same displacement field for every depth slice so the through-depth
        intensity profile (and dz signal) is preserved -- only spatial shape is warped."""
        from scipy.ndimage import gaussian_filter, map_coordinates
        _, H, W = block.shape
        rng = np.random.default_rng()
        dy = gaussian_filter(rng.standard_normal((H, W)), sigma=self.elastic_sigma) * self.elastic_alpha
        dx = gaussian_filter(rng.standard_normal((H, W)), sigma=self.elastic_sigma) * self.elastic_alpha
        y_grid, x_grid = np.mgrid[0:H, 0:W]
        coords_y = (y_grid + dy).clip(0, H - 1)
        coords_x = (x_grid + dx).clip(0, W - 1)
        coords_flat = [coords_y.ravel(), coords_x.ravel()]
        self._last_elastic_coords = coords_flat
        out = np.empty_like(block)
        for d in range(block.shape[0]):
            out[d] = map_coordinates(block[d], coords_flat, order=1, mode='reflect').reshape(H, W)
        return out

    def _apply_cutout(self, block, target_offset=None):
        """zero out random XY patches across all depth slices (specaugment-style).
        forces the model to use distributed spatial evidence rather than
        memorizing specific locations."""
        out = block.copy()
        _, H, W = out.shape
        for _ in range(self.cutout_n_patches):
            ph = random.randint(1, max(1, int(H * self.cutout_max_frac)))
            pw = random.randint(1, max(1, int(W * self.cutout_max_frac)))
            y0 = x0 = 0
            for _attempt in range(20):
                y0 = random.randint(0, H - ph)
                x0 = random.randint(0, W - pw)
                if not self.cutout_protect_center:
                    break
                protected = (
                    self.multitile_grid * self.multitile_subtile
                    if self.multitile else self.tile_size
                )
                if target_offset is None:
                    dy = dx = 0
                else:
                    dy, dx = (int(value) for value in target_offset.tolist())
                cy0 = (H - protected) // 2 + dy
                cx0 = (W - protected) // 2 + dx
                if y0 + ph <= cy0 or y0 >= cy0 + protected \
                    or x0 + pw <= cx0 or x0 >= cx0 + protected:
                    break
            else:
                continue
            out[:, y0:y0 + ph, x0:x0 + pw] = 0.0
        return out

    def _apply_smooth_depth_warp(self, block):
        """spatially vary depth position to simulate residual sheet undulation."""
        from scipy.ndimage import gaussian_filter, map_coordinates

        depth, height, width = block.shape
        field = gaussian_filter(
            np.random.standard_normal((height, width)).astype(np.float32),
            sigma=max(self.depth_warp_sigma, 1.0),
        )
        field = field / max(float(field.std()), 1e-6)
        amplitude = random.uniform(0.5 * self.depth_warp_max, self.depth_warp_max)
        field = np.clip(field * amplitude, -self.depth_warp_max, self.depth_warp_max)
        self._last_depth_warp_field = field
        yy, xx = np.mgrid[0:height, 0:width]
        out = np.empty_like(block)
        for depth_index in range(depth):
            zz = np.clip(depth_index + field, 0, depth - 1)
            out[depth_index] = map_coordinates(
                block,
                [zz, yy, xx],
                order=1,
                mode="nearest",
            )
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _estimate_surface_depth(block):
        """estimate the strongest smoothed papyrus-to-air transition per column."""
        from scipy.ndimage import gaussian_filter

        smooth = gaussian_filter(block, sigma=(0.75, 2.0, 2.0))
        return np.maximum(smooth[:-1] - smooth[1:], 0.0).argmax(axis=0)

    def _apply_surface_attenuation(self, block):
        """reduce local spatial contrast only near the estimated surface band."""
        from scipy.ndimage import gaussian_filter

        surface = self._estimate_surface_depth(block)
        depth_axis = np.arange(block.shape[0], dtype=np.float32)[:, None, None]
        sigma = max(self.surface_atten_sigma, 0.25)
        band = np.exp(-0.5 * ((depth_axis - surface[None]) / sigma) ** 2)
        local_mean = gaussian_filter(block, sigma=(0.0, 3.0, 3.0))
        strength = random.uniform(self.surface_atten_min, self.surface_atten_max)
        return np.clip(block - strength * band * (block - local_mean), 0.0, 1.0)

    def _apply_acquisition_blur(self, block):
        """apply a mild in-plane point-spread blur while preserving depth resolution."""
        from scipy.ndimage import gaussian_filter

        sigma = random.uniform(self.acquisition_blur_min, self.acquisition_blur_max)
        return gaussian_filter(block, sigma=(0.0, sigma, sigma)).astype(block.dtype, copy=False)

    def _apply_correlated_noise(self, block):
        """add low-frequency reconstruction-like noise correlated across space and depth."""
        from scipy.ndimage import gaussian_filter

        noise = gaussian_filter(
            np.random.standard_normal(block.shape).astype(np.float32),
            sigma=(1.0, self.correlated_noise_sigma, self.correlated_noise_sigma),
        )
        noise = noise / max(float(noise.std()), 1e-6)
        strength = random.uniform(self.correlated_noise_min, self.correlated_noise_max)
        return np.clip(block + strength * noise, 0.0, 1.0)

    def apply_context_replacement(self, block, donor, target_offset=None):
        """replace outer context with surface-aligned real papyrus from another window."""
        if block.shape != donor.shape:
            return block
        donor_aligned = self._surface_align_context(donor, block) \
            if self.context_replace_surface_align else donor
        _, height, width = block.shape
        if target_offset is None:
            dy = dx = 0
        else:
            dy, dx = (int(value) for value in target_offset.tolist())
        center_y = (height - 1) / 2.0 + dy
        center_x = (width - 1) / 2.0 + dx
        prediction_center = (
            self.multitile_grid * self.multitile_subtile
            if self.multitile else self.tile_size
        )
        protected = self.context_replace_keep_size
        if protected <= 0:
            protected = prediction_center + 2 * max(0, self.context_replace_margin)
        protected = min(max(protected, prediction_center), height, width)
        half = protected / 2.0
        feather = max(float(self.context_replace_feather), 1.0)
        yy, xx = np.mgrid[0:height, 0:width]
        outside_y = np.maximum(np.abs(yy - center_y) - half, 0.0)
        outside_x = np.maximum(np.abs(xx - center_x) - half, 0.0)
        distance = np.maximum(outside_y, outside_x)
        keep = np.clip(1.0 - distance / feather, 0.0, 1.0).astype(np.float32)
        mixed = keep[None] * block + (1.0 - keep[None]) * donor_aligned
        return np.ascontiguousarray(np.clip(mixed, 0.0, 1.0).astype(np.float32))

    def _surface_align_context(self, donor, recipient):
        """warp donor depth columns so their estimated surfaces match the recipient."""
        from scipy.ndimage import map_coordinates

        donor_surface = self._estimate_surface_depth(donor).astype(np.float32)
        recipient_surface = self._estimate_surface_depth(recipient).astype(np.float32)
        depth, height, width = donor.shape
        yy, xx = np.mgrid[0:height, 0:width]
        shift = donor_surface - recipient_surface
        out = np.empty_like(donor)
        for depth_index in range(depth):
            zz = np.clip(depth_index + shift, 0, depth - 1)
            out[depth_index] = map_coordinates(
                donor,
                [zz, yy, xx],
                order=1,
                mode="nearest",
            )
        return out

    def _apply_depth_mask(self, block):
        """remove depth information from slices; interp avoids zeros that fake an air edge."""
        out = block.copy()
        if self.depth_mask_mode == "interp":
            if random.random() < self.depth_mask_prob:
                index = random.randrange(out.shape[0])
                neighbours = [k for k in (index - 1, index + 1) if 0 <= k < out.shape[0]]
                out[index] = block[neighbours].mean(axis=0)
            return out
        for d in range(out.shape[0]):
            if random.random() < self.depth_mask_prob:
                out[d] = 0.0
        return out

    def _apply_brightness_adjustment(self, block):
        """applies ONE brightness factor to the whole block (shared across depth).
        per-depth factors distort the through-depth intensity profile the model keys on."""
        factor = random.uniform(1.0 - self.brightness_delta, 1.0 + self.brightness_delta)
        return np.clip(block * factor, 0, 1)
    
    def _apply_contrast_adjustment(self, block):
        """applies ONE contrast factor across all depth slices (shared factor; per-slice
        mean preserved) so the depth profile is scaled uniformly, not warped per slice."""
        factor = random.uniform(1.0 - self.contrast_delta, 1.0 + self.contrast_delta)
        adj_block = block.copy()
        for i in range(block.shape[0]):
            channel = block[i]
            mean = np.mean(channel)
            adj_block[i] = np.clip((channel - mean) * factor + mean, 0, 1)
        return adj_block
    
    def _apply_gaussian_noise(self, block):
        """applies gaussian noise to each channel independently"""
        std = random.uniform(self.noise_std_min, self.noise_std_max)
        noise = np.random.normal(0, std, block.shape)
        return np.clip(block + noise, 0, 1)
    
    def _apply_rotation(self, block):
        """applies 90 180 270 degree rotations to all channels"""
        k = random.choice([1, 2, 3])
        # np.rot90 can produce negative strides so force a copy
        return np.rot90(block, k=k, axes=(1, 2)).copy()
    
    def _apply_flip(self, block):
        """applies horizontal or vertical flip to all channels"""
        axis = random.choice([1, 2])
        # np.flip returns a view with negative strides so force a copy
        return np.flip(block, axis=axis).copy()

class InkVolumeDataset(IterableDataset):
    """iterable dataset for ink volume data"""
    def __init__(self, volume, mask, labels, config, x_range, y_range, norm_stats, shuffle=True, soft_labels=None, scroll_id=None, domain_id=None, character_namespace=None, scroll_mask=None, split_mask=None, character_grid=None, explicit_negative_mask=None, explicit_positive_mask=None, prepared_state=None):
        """initializes the dataset.
        scroll_mask: optional papyrus mask distinct from `mask` (which may be ring-restricted);
        multitile uses it to drop sub-tiles straddling the scroll boundary. defaults to `mask`.
        split_mask: optional manual train/validation assignment for multitile targets. `mask`
        still gates the same ring windows as the legacy path; this mask partitions their targets.
        character_grid: optional subtile-resolution connected-component ids for character metrics.
        soft_labels: optional full-res float [0,1] ink-probability map (expanded+blurred
        eroded labels). when given AND config.data.dense_soft_labels is set, the dense
        per-pixel target uses these CONTINUOUS values instead of the hard binary label —
        calibrated soft edges (see _fetch/__next__ dense path). stored as uint8 0-255.
        scroll_id: integer scroll id for bookkeeping.
        domain_id: compact physical-scroll id used by DANN/SupCon when enabled.
        character_namespace: compact segment id that prevents component collisions when
        several segments intentionally share one physical domain id."""
        self.scroll_id = int(scroll_id) if scroll_id is not None else 0
        self.domain_id = int(domain_id) if domain_id is not None else 0
        self.character_namespace = int(
            character_namespace if character_namespace is not None else self.domain_id
        )
        self._volume_shape = tuple(int(value) for value in volume.shape)
        self._volume_is_normalized = bool(getattr(volume, "normalized", False))
        # store zarr path + segment id instead of the open zarr object so that
        # the dataset can be safely pickled for multiprocessing workers on Windows;
        # each worker opens its own zarr handle lazily on first access
        if hasattr(volume, 'store') and hasattr(volume.store, 'path'):
            self._zarr_path = str(volume.store.path)
            # CRITICAL: do NOT store the zarr object — it is not picklable on Windows
            # and will crash workers with OSError [Errno 22]. workers reopen via _zarr_path
            self._vol_obj = None
        else:
            # numpy array (preloaded) or other picklable type — store directly
            self._zarr_path = None
            self._vol_obj = volume
        self._worker_vol = None         # populated lazily inside worker process

        self._surface_depth_path = None
        self._surface_confidence_path = None
        self._surface_depth_arr = None
        self._surface_confidence_arr = None
        self._use_surface_teacher = any([
            bool(getattr(config.model, "new_learned_surface", False)),
            bool(getattr(config.model, "better_surface", False)),
            bool(getattr(config.model, "surface_teacher_input", False)),
        ])
        self._surface_relative_depth_window = bool(
            getattr(config.data, "surface_relative_depth_window", False)
        )
        self._surface_window_offset = int(
            (getattr(config.data, "surface_window_offset_by_scroll", None) or {}).get(
                self.scroll_id,
                getattr(config.data, "surface_window_offset", 0),
            )
        )
        if self._surface_window_offset and not self._surface_relative_depth_window:
            raise ValueError("surface_window_offset requires surface_relative_depth_window")
        if self._surface_relative_depth_window and not self._use_surface_teacher:
            raise ValueError("surface_relative_depth_window requires pre-generated surface maps")
        if self._use_surface_teacher:
            surface_dir = os.path.abspath(getattr(config.data, "surface_label_dir", "./surface_labels"))
            self._surface_depth_path = os.path.join(
                surface_dir,
                str(self.scroll_id),
                "depth.npy",
            )
            self._surface_confidence_path = os.path.join(
                surface_dir,
                str(self.scroll_id),
                "confidence.npy",
            )
            missing = [
                path for path in (self._surface_depth_path, self._surface_confidence_path)
                if not os.path.isfile(path)
            ]
            if missing:
                raise FileNotFoundError(
                    f"pre-generated surface supervision is required for scroll {self.scroll_id}; "
                    f"missing: {', '.join(missing)}"
                )
            expected_shape = tuple(int(value) for value in volume.shape[-2:])
            depth_shape = np.load(self._surface_depth_path, mmap_mode="r").shape
            confidence_shape = np.load(self._surface_confidence_path, mmap_mode="r").shape
            if depth_shape != expected_shape or confidence_shape != expected_shape:
                raise ValueError(
                    f"surface supervision shape mismatch for {self.scroll_id}: "
                    f"expected {expected_shape}, got depth={depth_shape}, "
                    f"confidence={confidence_shape}"
                )
            if bool(getattr(config.data, "preload_volumes", False)):
                self._surface_depth_arr = np.load(self._surface_depth_path)
                self._surface_confidence_arr = np.load(self._surface_confidence_path)
                self._surface_depth_arr.setflags(write=False)
                self._surface_confidence_arr.setflags(write=False)

        # store mask/labels as uint8 (binary), not float64. the source arrays are
        # mask/255.0 and labels/255.0 (float64): for the big scroll (13513x17381)
        # that is ~1.88 GB EACH. when DataLoader spawns workers on Windows, the whole
        # dataset is pickled to each worker; two float64 full-res arrays (plus a ring
        # mask) exceed the spawn pickle limit -> OSError [Errno 22] / "pickle data was
        # truncated". these arrays are only ever used as binary tests (>0.5, sum>0), so
        # uint8 is exact and 8x smaller, which keeps multiscroll+ring picklable at nw>0.
        mask_array = np.asarray(mask)
        labels_array = np.asarray(labels)
        mask_u8 = mask_array if mask_array.dtype == np.uint8 else (mask_array > 0.5).astype(np.uint8)
        labels_u8 = (
            labels_array if labels_array.dtype == np.uint8
            else (labels_array > 0.5).astype(np.uint8)
        )

        # optionally back the (already tiny, but still N x hundreds-of-MB at the 5-10
        # fragment scale) binary arrays with an on-disk memmap so they pickle as a path
        # rather than data. _mask_path/_labels_path is the on-disk source of truth;
        # _mask_arr/_labels_arr is the per-process handle (real array when not memmapped,
        # a lazily-opened read-only memmap when memmapped). see the mask/labels properties.
        # CRITICAL: use bit-packing (1 bit/pixel) to save 8x RAM -> 6GB saved for 15 scrolls
        use_bitpack = getattr(config.data, "mask_bitpack", True)  # default ON
        if getattr(config.data, "mask_memmap", False):
            self._mask_path = _write_memmap(mask_u8, pack_bits=use_bitpack, original_shape=mask_u8.shape)
            self._labels_path = _write_memmap(labels_u8, pack_bits=use_bitpack, original_shape=labels_u8.shape)
            self._mask_arr = None
            self._labels_arr = None
            self._mask_shape = mask_u8.shape
            self._labels_shape = labels_u8.shape
            self._use_bitpack = use_bitpack
        else:
            self._mask_path = None
            self._labels_path = None
            self._mask_arr = mask_u8
            self._labels_arr = labels_u8
            self._mask_shape = None
            self._labels_shape = None
            self._use_bitpack = False
        # scroll (papyrus) mask, kept separate from the training `mask` (a ring-restricted
        # subset when ring_negatives is on). multitile uses it to mask out sub-tiles that
        # straddle the papyrus boundary (req c). absent -> equals the training mask.
        self._has_scroll_mask = scroll_mask is not None
        if self._has_scroll_mask:
            scroll_array = np.asarray(scroll_mask)
            sm_u8 = (
                scroll_array if scroll_array.dtype == np.uint8
                else (scroll_array > 0.5).astype(np.uint8)
            )
            if getattr(config.data, "mask_memmap", False):
                self._scroll_mask_path = _write_memmap(sm_u8, pack_bits=use_bitpack, original_shape=sm_u8.shape)
                self._scroll_mask_arr = None
                self._scroll_mask_shape = sm_u8.shape
            else:
                self._scroll_mask_path = None
                self._scroll_mask_arr = sm_u8
                self._scroll_mask_shape = None
        else:
            self._scroll_mask_path = None
            self._scroll_mask_arr = None
            self._scroll_mask_shape = None
        self._has_split_mask = split_mask is not None
        if self._has_split_mask:
            split_array = np.asarray(split_mask)
            split_u8 = (
                split_array if split_array.dtype == np.uint8
                else (split_array > 0.5).astype(np.uint8)
            )
            if getattr(config.data, "mask_memmap", False):
                self._split_mask_path = _write_memmap(
                    split_u8, pack_bits=use_bitpack, original_shape=split_u8.shape
                )
                self._split_mask_arr = None
                self._split_mask_shape = split_u8.shape
            else:
                self._split_mask_path = None
                self._split_mask_arr = split_u8
                self._split_mask_shape = None
        else:
            self._split_mask_path = None
            self._split_mask_arr = None
            self._split_mask_shape = None
        self._has_explicit_negative_mask = explicit_negative_mask is not None
        if self._has_explicit_negative_mask:
            explicit_array = np.asarray(explicit_negative_mask)
            explicit_u8 = (
                explicit_array if explicit_array.dtype == np.uint8
                else (explicit_array > 0.5).astype(np.uint8)
            )
            if getattr(config.data, "mask_memmap", False):
                self._explicit_negative_path = _write_memmap(
                    explicit_u8,
                    pack_bits=use_bitpack,
                    original_shape=explicit_u8.shape,
                )
                self._explicit_negative_arr = None
                self._explicit_negative_shape = explicit_u8.shape
            else:
                self._explicit_negative_path = None
                self._explicit_negative_arr = explicit_u8
                self._explicit_negative_shape = None
        else:
            self._explicit_negative_path = None
            self._explicit_negative_arr = None
            self._explicit_negative_shape = None
        self._has_explicit_positive_mask = explicit_positive_mask is not None
        if self._has_explicit_positive_mask:
            explicit_positive_array = np.asarray(explicit_positive_mask)
            explicit_positive_u8 = (
                explicit_positive_array if explicit_positive_array.dtype == np.uint8
                else (explicit_positive_array > 0.5).astype(np.uint8)
            )
            if getattr(config.data, "mask_memmap", False):
                self._explicit_positive_path = _write_memmap(
                    explicit_positive_u8,
                    pack_bits=use_bitpack,
                    original_shape=explicit_positive_u8.shape,
                )
                self._explicit_positive_arr = None
                self._explicit_positive_shape = explicit_positive_u8.shape
            else:
                self._explicit_positive_path = None
                self._explicit_positive_arr = explicit_positive_u8
                self._explicit_positive_shape = None
        else:
            self._explicit_positive_path = None
            self._explicit_positive_arr = None
            self._explicit_positive_shape = None
        # optional soft labels (continuous ink probability, 0-255 uint8). stored parallel
        # to the hard labels; used only by the dense target path when dense_soft_labels is on.
        self._soft_path = None
        self._soft_arr = None
        self._soft_shape = None
        if soft_labels is not None:
            soft_u8 = np.clip(np.asarray(soft_labels) * 255.0, 0, 255).astype(np.uint8)
            if getattr(config.data, "mask_memmap", False):
                # soft labels are uint8 0-255, not binary, so don't bitpack
                self._soft_path = _write_memmap(soft_u8, pack_bits=False)
                self._soft_shape = soft_u8.shape
            else:
                self._soft_arr = soft_u8
        self.c = config
        self.tile_size = config.data.tile_size
        self.depth = config.data.depth
        self.apply_transforms = False # controlled by trainer
        self.shuffle = shuffle
        self.norm_stats = norm_stats
        self.transform = Transform(config)
        # multitile: emit a grid x grid map of per-sub-tile labels (papyrus unless .any() ink)
        self._mt = bool(getattr(config.model, "multitile", False))
        self._mt_grid = max(1, int(getattr(config.model, "multitile_grid", 4)))
        self._mt_sub = max(1, int(getattr(config.model, "multitile_subtile", 8)))
        # pos-only: in ink windows, supervise only ink sub-tiles (mask out non-ink ones)
        self._mt_pos_only = bool(getattr(config.data, "multitile_pos_only", False))
        self._mt_ring_gate = self._mt_pos_only or bool(
            getattr(config.data, "multitile_ring_gate", False)
        )
        self._manual_split = not bool(getattr(config.data, "simple_split", True))
        self._character_metrics = bool(getattr(config.tra, "character_macro_metrics", False))
        self._character_balanced = bool(
            getattr(config.data, "character_balanced_sampling", False)
        ) and self.shuffle
        self._character_grid = character_grid
        self._character_nearest = None
        self._character_pos_coords = {}
        self._character_neg_coords = {}
        self._explicit_neg_coords = []
        self._explicit_negative_share = float(
            getattr(config.data, "explicit_negative_share", 0.0)
        )
        self._context_replace_prob = float(getattr(config.dl, "context_replace_prob", 0.0))
        self._context_consistency = bool(
            getattr(config.tra, "context_consistency", False)
        ) and self.shuffle
        self._context_consistency_prob = float(
            getattr(config.tra, "context_consistency_prob", 0.25)
        )
        self._depth_view_consistency = bool(
            getattr(config.tra, "depth_view_consistency", False)
        ) and self.shuffle
        self._depth_view_consistency_prob = float(
            getattr(config.tra, "depth_view_consistency_prob", 0.5)
        )
        self._depth_view_consistency_offset = int(
            getattr(config.tra, "depth_view_consistency_offset", 2)
        )
        if self._depth_view_consistency and self._context_consistency:
            raise ValueError("depth-view and context consistency cannot share one paired sample")
        if self._depth_view_consistency and not self._surface_relative_depth_window:
            raise ValueError("depth-view consistency requires surface-relative depth windows")
        if self._depth_view_consistency_offset < 1:
            raise ValueError("depth_view_consistency_offset must be positive")
        self._context_donor_coords = []

        self.z_start = getattr(self.c.data, "train_d_start", self.c.data.d_start)
        self.z_end   = getattr(self.c.data, "train_d_end",   self.c.data.d_end)
        self.y_start, self.y_end = y_range
        self.x_start, self.x_end = x_range
        
        # pre-calculate all valid block coordinates unless immutable preparation was reused
        if prepared_state is None:
            self.block_coords = self._gen_tile_coords()
        else:
            self.block_coords = prepared_state["block_coords"]
        # optional per-epoch tile cap: on very large volumes (native 2.4um) a full pass
        # is prohibitively slow. when set (and this is the shuffled TRAIN set), each epoch
        # draws a fresh random subset of this many coords, bounding epoch wall-time without
        # changing per-step behavior or the depth window. validation stays full.
        self._max_samples = getattr(self.c.data, "max_samples_per_epoch", None)
        if self._max_samples is None:
            _env_cap = os.getenv("VESUVIUS_MAX_SAMPLES_PER_EPOCH")
            if _env_cap:
                self._max_samples = int(_env_cap)
        if self.shuffle and self._max_samples is not None:
            self.samples_per_epoch = min(len(self.block_coords), int(self._max_samples))
        else:
            self.samples_per_epoch = len(self.block_coords)
        if prepared_state is not None:
            self._character_nearest = prepared_state["character_nearest"]
            self._character_pos_coords = prepared_state["character_pos_coords"]
            self._character_neg_coords = prepared_state["character_neg_coords"]
            self._character_ids = prepared_state["character_ids"]
            self._explicit_neg_coords = prepared_state.get("explicit_neg_coords", [])
            self._context_donor_coords = prepared_state["context_donor_coords"]
        elif self._character_metrics or self._character_balanced:
            if not self._mt or self._character_grid is None:
                raise ValueError("character-aware mode requires multitile labels and a character grid")
            self._prepare_character_targets()
        if prepared_state is None and (self._context_replace_prob > 0 or self._context_consistency) and self.shuffle:
            self._prepare_context_donors()

    def prepared_state(self) -> dict:
        """return immutable expensive-to-build state suitable for fresh wrappers."""
        return {
            "block_coords": self.block_coords,
            "character_nearest": self._character_nearest,
            "character_pos_coords": self._character_pos_coords,
            "character_neg_coords": self._character_neg_coords,
            "character_ids": getattr(self, "_character_ids", ()),
            "explicit_neg_coords": self._explicit_neg_coords,
            "context_donor_coords": self._context_donor_coords,
        }

    def _mt_center_bounds(self, y_off, x_off):
        """absolute y/x bounds of the multitile center window for this sample.
        for tile16 + grid4*sub8 this is a 32x32 window centered on the 16x16 tile."""
        n, sub = self._mt_grid, self._mt_sub
        center = n * sub
        y0 = self.y_start + y_off + (self.tile_size - center) // 2
        x0 = self.x_start + x_off + (self.tile_size - center) // 2
        return y0, y0 + center, x0, x0 + center

    def _mt_window_touches_ring(self, y_off, x_off):
        """true when the 32x32 multitile center overlaps any training (ring) mask pixel."""
        y0, y1, x0, x1 = self._mt_center_bounds(y_off, x_off)
        lbl = self.mask
        H, W = int(lbl.shape[0]), int(lbl.shape[1])
        ys, ye = max(0, y0), min(H, y1)
        xs, xe = max(0, x0), min(W, x1)
        if ys >= ye or xs >= xe:
            return False
        return bool(np.any(lbl[ys:ye, xs:xe] > 0.5))

    def _mt_window_touches_split(self, y_off, x_off):
        """true when the multitile center contains a target assigned to this split."""
        if not self._has_split_mask:
            return True
        y0, y1, x0, x1 = self._mt_center_bounds(y_off, x_off)
        split = self.split_mask
        h, w = int(split.shape[0]), int(split.shape[1])
        ys, ye = max(0, y0), min(h, y1)
        xs, xe = max(0, x0), min(w, x1)
        return ys < ye and xs < xe and bool(np.any(split[ys:ye, xs:xe] > 0))

    def _prepare_character_targets(self):
        """associate each valid target cell and ring negative with one character."""
        from scipy.ndimage import distance_transform_edt

        chars = np.asarray(self._character_grid, dtype=np.int32).copy()
        if self._has_split_mask:
            split = self.split_mask
            sub = self._mt_sub
            gh, gw = chars.shape
            split_cells = split[:gh * sub, :gw * sub].reshape(
                gh, sub, gw, sub
            ).all(axis=(1, 3))
            chars[~split_cells] = 0
        if not np.any(chars > 0):
            raise ValueError(f"character-aware split has no positive characters for {self.scroll_id}")

        _, nearest_indices = distance_transform_edt(chars == 0, return_indices=True)
        self._character_nearest = chars[nearest_indices[0], nearest_indices[1]]

        if not self._character_balanced:
            return
        for coord in self.block_coords:
            _, y_off, x_off = coord
            labels = self._fetch_label_mt(y_off, x_off).numpy()
            valid = self._fetch_mask_mt(y_off, x_off).numpy() > 0
            component_ids = self._fetch_character_ids(y_off, x_off, labels, valid)
            positive_ids = np.unique(component_ids[(labels > 0) & valid])
            if positive_ids.size:
                positive_ids = positive_ids[positive_ids > 0]
                if positive_ids.size == 1:
                    component_id = int(positive_ids[0])
                    positive_count = int((
                        (component_ids == component_id) & (labels > 0) & valid
                    ).sum())
                    self._character_pos_coords.setdefault(component_id, []).extend(
                        [coord] * positive_count
                    )
                continue
            negative_ids = np.unique(component_ids[(labels <= 0) & valid])
            negative_ids = negative_ids[negative_ids > 0]
            if np.any((labels < 0) & valid):
                self._explicit_neg_coords.append(coord)
            if negative_ids.size == 1:
                self._character_neg_coords.setdefault(int(negative_ids[0]), []).append(coord)

        valid_chars = sorted(set(self._character_pos_coords) & set(self._character_neg_coords))
        self._character_ids = valid_chars
        if not valid_chars:
            raise ValueError(f"no characters have both positive and ring-negative windows for {self.scroll_id}")
        print(
            f"[character-sampling] scroll {self.scroll_id}: {len(valid_chars)} characters "
            f"with positive and ring-negative windows"
        )

    def _fetch_character_ids(self, y_off, x_off, labels, valid):
        """return one associated component id per multitile target."""
        out = np.zeros(self._mt_grid * self._mt_grid, dtype=np.int64)
        if self._character_grid is None or self._character_nearest is None:
            return out
        y0, _, x0, _ = self._mt_center_bounds(y_off, x_off)
        gh, gw = self._character_grid.shape
        for iy in range(self._mt_grid):
            gy = (y0 + iy * self._mt_sub) // self._mt_sub
            if gy < 0 or gy >= gh:
                continue
            for ix in range(self._mt_grid):
                index = iy * self._mt_grid + ix
                if not valid[index]:
                    continue
                gx = (x0 + ix * self._mt_sub) // self._mt_sub
                if gx < 0 or gx >= gw:
                    continue
                if labels[index] > 0:
                    out[index] = int(self._character_grid[gy, gx])
                else:
                    out[index] = int(self._character_nearest[gy, gx])
                if out[index] > 0:
                    out[index] += int(self.character_namespace) * 1_000_000
        return out

    def _character_sampling_weights(self, character_ids):
        """per-character draw weights written by the trainer after each epoch."""
        path = str(getattr(self.c.tra, "character_forgetting_path", "") or "")
        if not bool(getattr(self.c.tra, "character_forgetting", False)) or not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, ValueError):
            return None
        weights = np.array(
            [float(stored.get(str(component_id), 1.0)) for component_id in character_ids],
            dtype=np.float64,
        )
        return weights / weights.sum()

    def _character_balanced_coords(self):
        """pair one positive and one local-ring window per sampled character."""
        target = self.samples_per_epoch
        coords = []
        character_ids = list(self._character_ids)
        weights = self._character_sampling_weights(character_ids)
        explicit_share = self._explicit_negative_share if self._explicit_neg_coords else 0.0
        while len(coords) < target:
            if weights is None:
                cycle = list(character_ids)
                np.random.shuffle(cycle)
            else:
                cycle = list(np.random.choice(character_ids, size=len(character_ids), p=weights))
            for component_id in cycle:
                positive = self._character_pos_coords[component_id]
                negative = self._character_neg_coords[component_id]
                coords.append(positive[np.random.randint(len(positive))])
                if len(coords) >= target:
                    break
                if explicit_share > 0 and np.random.random() < explicit_share:
                    explicit = self._explicit_neg_coords
                    coords.append(explicit[np.random.randint(len(explicit))])
                else:
                    coords.append(negative[np.random.randint(len(negative))])
                if len(coords) >= target:
                    break
        return coords

    def _prepare_context_donors(self):
        """collect same-split contexts with valid papyrus and no known ink anywhere."""
        ctx = int(getattr(self.c.data, "context_size", 0) or self.tile_size)
        pad = max(0, (ctx - self.tile_size) // 2)
        scroll = np.asarray(self.scroll_mask, dtype=np.uint8)
        labels = np.asarray(self.labels, dtype=np.uint8)
        stride = math.gcd(max(1, ctx), math.gcd(max(1, pad), max(1, self._mt_sub)))
        stride = max(1, stride)
        height_full = (scroll.shape[0] // stride) * stride
        width_full = (scroll.shape[1] // stride) * stride
        coarse_mask = scroll[:height_full, :width_full].reshape(
            height_full // stride,
            stride,
            width_full // stride,
            stride,
        ).mean(axis=(1, 3)).astype(np.float32)
        coarse_ink = labels[:height_full, :width_full].reshape(
            height_full // stride,
            stride,
            width_full // stride,
            stride,
        ).max(axis=(1, 3)).astype(np.uint8)
        mask_integral = cv2.integral(coarse_mask)
        ink_integral = cv2.integral(coarse_ink)
        min_fraction = float(getattr(self.c.dl, "context_replace_min_mask_frac", 0.8))
        height, width = scroll.shape
        donors = []
        step = max(1, int(getattr(self.c.data, "multitile_train_step", self.tile_size)))
        y_span = max(0, self.y_end - self.y_start - self.tile_size + 1)
        x_span = max(0, self.x_end - self.x_start - self.tile_size + 1)
        for y_off in range(0, y_span, step):
            for x_off in range(0, x_span, step):
                y = self.y_start + y_off
                x = self.x_start + x_off
                if self._has_split_mask and not np.all(
                    self.split_mask[y:y + self.tile_size, x:x + self.tile_size] > 0
                ):
                    continue
                y0 = y - pad
                x0 = x - pad
                if y0 < 0 or x0 < 0 or y0 + ctx > height or x0 + ctx > width:
                    continue
                cy0, cx0 = y0 // stride, x0 // stride
                cy1, cx1 = (y0 + ctx) // stride, (x0 + ctx) // stride
                total = (cy1 - cy0) * (cx1 - cx0)
                valid_count = (
                    mask_integral[cy1, cx1] - mask_integral[cy0, cx1]
                    - mask_integral[cy1, cx0] + mask_integral[cy0, cx0]
                )
                ink_count = (
                    ink_integral[cy1, cx1] - ink_integral[cy0, cx1]
                    - ink_integral[cy1, cx0] + ink_integral[cy0, cx0]
                )
                if (
                    total > 0
                    and float(valid_count) / total >= min_fraction
                    and int(ink_count) == 0
                ):
                    donors.append((y_off, x_off))
        self._context_donor_coords = donors
        if not donors:
            raise ValueError(f"no valid context-replacement donors for scroll {self.scroll_id}")
        print(
            f"[context-replace] scroll {self.scroll_id}: {len(donors)} "
            f"fully ink-free donors with mask>={min_fraction:.2f}"
        )

    @property
    def mask(self):
        """binary uint8 mask; a real array unless memmapped, in which case the
        read-only memmap is opened lazily per process (main or worker)."""
        if self._mask_arr is None and self._mask_path is not None:
            packed = np.load(self._mask_path, mmap_mode='r')
            if self._use_bitpack:
                # unpack bits and reshape to original dimensions
                unpacked = np.unpackbits(packed)
                # trim to exact size (packbits pads to byte boundary)
                total_pixels = int(np.prod(self._mask_shape))
                self._mask_arr = unpacked[:total_pixels].reshape(self._mask_shape)
            else:
                self._mask_arr = packed
        return self._mask_arr

    @property
    def labels(self):
        """binary uint8 labels; lazily memmapped per process when memmap is enabled."""
        if self._labels_arr is None and self._labels_path is not None:
            packed = np.load(self._labels_path, mmap_mode='r')
            if self._use_bitpack:
                # unpack bits and reshape to original dimensions
                unpacked = np.unpackbits(packed)
                total_pixels = int(np.prod(self._labels_shape))
                self._labels_arr = unpacked[:total_pixels].reshape(self._labels_shape)
            else:
                self._labels_arr = packed
        return self._labels_arr

    @property
    def soft_labels(self):
        """continuous ink-probability map in [0,1] (from uint8 0-255), or None if unset.
        lazily memmapped per process, same as labels/mask."""
        if self._soft_arr is None and self._soft_path is not None:
            self._soft_arr = np.load(self._soft_path, mmap_mode='r')
        if self._soft_arr is None:
            return None
        return self._soft_arr

    @property
    def scroll_mask(self):
        """binary papyrus mask; falls back to the training mask when none was provided
        (ring_negatives off). lazily memmapped per process, mirroring `mask`."""
        if not self._has_scroll_mask:
            return self.mask
        if self._scroll_mask_arr is None and self._scroll_mask_path is not None:
            packed = np.load(self._scroll_mask_path, mmap_mode='r')
            if self._use_bitpack:
                unpacked = np.unpackbits(packed)
                total_pixels = int(np.prod(self._scroll_mask_shape))
                self._scroll_mask_arr = unpacked[:total_pixels].reshape(self._scroll_mask_shape)
            else:
                self._scroll_mask_arr = packed
        return self._scroll_mask_arr

    @property
    def split_mask(self):
        """binary manual assignment mask, lazily reopened in each worker."""
        if not self._has_split_mask:
            return None
        if self._split_mask_arr is None and self._split_mask_path is not None:
            packed = np.load(self._split_mask_path, mmap_mode='r')
            if self._use_bitpack:
                unpacked = np.unpackbits(packed)
                total_pixels = int(np.prod(self._split_mask_shape))
                self._split_mask_arr = unpacked[:total_pixels].reshape(self._split_mask_shape)
            else:
                self._split_mask_arr = packed
        return self._split_mask_arr

    @property
    def explicit_negative_mask(self):
        """binary guaranteed-negative mask, lazily reopened in each worker."""
        if not self._has_explicit_negative_mask:
            return None
        if self._explicit_negative_arr is None and self._explicit_negative_path is not None:
            packed = np.load(self._explicit_negative_path, mmap_mode="r")
            if self._use_bitpack:
                unpacked = np.unpackbits(packed)
                total_pixels = int(np.prod(self._explicit_negative_shape))
                self._explicit_negative_arr = unpacked[:total_pixels].reshape(
                    self._explicit_negative_shape
                )
            else:
                self._explicit_negative_arr = packed
        return self._explicit_negative_arr

    @property
    def explicit_positive_mask(self):
        """binary guaranteed-positive mask, lazily reopened in each worker."""
        if not self._has_explicit_positive_mask:
            return None
        if self._explicit_positive_arr is None and self._explicit_positive_path is not None:
            packed = np.load(self._explicit_positive_path, mmap_mode="r")
            if self._use_bitpack:
                unpacked = np.unpackbits(packed)
                total_pixels = int(np.prod(self._explicit_positive_shape))
                self._explicit_positive_arr = unpacked[:total_pixels].reshape(
                    self._explicit_positive_shape
                )
            else:
                self._explicit_positive_arr = packed
        return self._explicit_positive_arr

    def __getstate__(self):
        """pickle only the memmap PATHS, never the open memmap. pickling a numpy
        memmap would copy its full contents into the pickle stream — exactly the
        windows spawn pickle-size blowup memmap exists to avoid. workers reopen the
        memmap lazily via the property. (when not memmapped, _mask_arr is a small
        uint8 array and is pickled normally, preserving prior behavior.)"""
        state = self.__dict__.copy()
        if state.get("_mask_path") is not None:
            state["_mask_arr"] = None
        if state.get("_labels_path") is not None:
            state["_labels_arr"] = None
        if state.get("_soft_path") is not None:
            state["_soft_arr"] = None
        if state.get("_scroll_mask_path") is not None:
            state["_scroll_mask_arr"] = None
        if state.get("_split_mask_path") is not None:
            state["_split_mask_arr"] = None
        if state.get("_explicit_negative_path") is not None:
            state["_explicit_negative_arr"] = None
        if state.get("_explicit_positive_path") is not None:
            state["_explicit_positive_arr"] = None
        state["_surface_depth_arr"] = None
        state["_surface_confidence_arr"] = None
        # never pickle an open zarr handle to a spawned worker (unpicklable on Windows,
        # OSError [Errno 22]). the main process may now hold one (vol opens lazily in the
        # main process too, for num_workers=0 validation); drop it so workers reopen via
        # _zarr_path. harmless when it was already None.
        if state.get("_zarr_path") is not None:
            state["_worker_vol"] = None
        return state

    @property
    def surface_depth(self):
        """memory-mapped absolute surface depth teacher."""
        if self._surface_depth_arr is None and self._surface_depth_path is not None:
            self._surface_depth_arr = np.load(self._surface_depth_path, mmap_mode="r")
        return self._surface_depth_arr

    @property
    def surface_confidence(self):
        """memory-mapped surface confidence teacher."""
        if self._surface_confidence_arr is None and self._surface_confidence_path is not None:
            self._surface_confidence_arr = np.load(self._surface_confidence_path, mmap_mode="r")
        return self._surface_confidence_arr


    @property
    def vol(self):
        """return volume; numpy arrays are returned directly (preloaded path);
        zarr objects are opened lazily per process to avoid pickle errors on Windows"""
        # fast path: volume already in RAM as numpy array
        if isinstance(self._vol_obj, np.ndarray):
            return self._vol_obj
        if self._worker_vol is not None:
            return self._worker_vol
        if self._zarr_path is not None:
            # open a fresh zarr handle lazily — in a DataLoader worker OR the main
            # process. BUGFIX: this previously opened ONLY inside workers (guarded by
            # `if get_worker_info() is not None`). validation runs with num_workers=0,
            # i.e. in the MAIN process, so vol fell through to `return self._vol_obj`
            # (None for a non-preloaded zarr). _fetch_block then read None[...], hit its
            # bare except, and returned ALL-ZERO tiles -> constant score -> every VALID
            # metric frozen (roc_auc=0.5000, pr_auc=prevalence, f1=0) identically across
            # epochs AND architectures. only surfaced on scroll4 because scroll1 was small
            # enough to preload_to_ram (_vol_obj = ndarray, so the main process had data).
            import zarr as _zarr
            self._worker_vol = _zarr.open(self._zarr_path, mode='r')
            return self._worker_vol
        return self._vol_obj

    def set_volume(self, volume) -> None:
        """replace the read backend before worker processes are created."""
        if tuple(int(value) for value in volume.shape) != self._volume_shape:
            raise ValueError(
                f"replacement volume shape {tuple(volume.shape)} != {self._volume_shape}"
            )
        self._zarr_path = None
        self._vol_obj = volume
        self._worker_vol = None
        self._volume_is_normalized = bool(getattr(volume, "normalized", False))

    def required_volume_spatial_chunks(self, chunk_shape) -> set[tuple[int, int]]:
        """enumerate every spatial chunk reachable by candidates, jitter, or donors."""
        chunk_y, chunk_x = int(chunk_shape[1]), int(chunk_shape[2])
        height, width = self._volume_shape[1:]
        required: set[tuple[int, int]] = set()

        def add_window(y0: int, x0: int, size: int) -> None:
            ys, ye = max(0, int(y0)), min(height, int(y0 + size))
            xs, xe = max(0, int(x0)), min(width, int(x0 + size))
            if ys >= ye or xs >= xe:
                return
            for cy in range(ys // chunk_y, (ye - 1) // chunk_y + 1):
                for cx in range(xs // chunk_x, (xe - 1) // chunk_x + 1):
                    required.add((cy, cx))

        context = int(getattr(self.c.data, "context_size", 0) or self.tile_size)
        has_context = context > self.tile_size
        pad = (context - self.tile_size) // 2 if has_context else 0
        jitter = int(getattr(self.c.data, "ctx_jitter", 0)) if self.shuffle else 0
        for _, y_offset, x_offset in self.block_coords:
            y = self.y_start + int(y_offset)
            x = self.x_start + int(x_offset)
            if has_context:
                add_window(y - pad - jitter, x - pad - jitter, context + 2 * jitter)
            else:
                add_window(y, x, self.tile_size)

        if has_context:
            for donor_y, donor_x in self._context_donor_coords:
                add_window(
                    self.y_start + int(donor_y) - pad,
                    self.x_start + int(donor_x) - pad,
                    context,
                )
        return required

    def _gen_tile_coords(self):
        """generates all valid (z, y, x) block start coordinates"""
        z_range_size = max(0, self.z_end - self.z_start - self.depth + 1)
        y_range_size = max(0, self.y_end - self.y_start - self.tile_size + 1)
        x_range_size = max(0, self.x_end - self.x_start - self.tile_size + 1)
        
        coords = []
        z_step = max(1, int(self.depth // 2))
        # multitile steps the window by a larger stride (each window supervises grid^2 sub-tiles);
        # single-tile steps by tile_size as before.
        xy_step = int(getattr(self.c.data, "multitile_train_step", self.tile_size)) if getattr(self, "_mt", False) else self.tile_size
        xy_step = max(1, xy_step)

        # iterate over the volume with specified step sizes to generate coordinates
        depth_offsets = [0] if self._surface_relative_depth_window else range(0, z_range_size, z_step)
        for d in depth_offsets:
            if self.z_start + d + self.depth > self.z_end: continue
            for y in range(0, y_range_size, xy_step):
                for x in range(0, x_range_size, xy_step):
                    # multitile training windows: keep ONLY windows whose 32x32 center overlaps
                    # the ring (training) mask. pos_only later keeps actual ink cells and true
                    # ring-negative cells while leaving the exclusion gap unsupervised.
                    if self._mt and (self.shuffle or self._manual_split):
                        if (self._mt_window_touches_ring(y, x)
                                and self._mt_window_touches_split(y, x)
                                and self._fetch_mask_mt(y, x).sum().item() > 0):
                            coords.append((d, y, x))
                        continue
                    # check if the corresponding mask area has any valid pixels
                    mask_block = self.mask[
                        self.y_start + y : self.y_start + y + self.tile_size,
                        self.x_start + x : self.x_start + x + self.tile_size
                    ]
                    if np.sum(mask_block) > 0:
                        coords.append((d, y, x))
        return coords

    def __len__(self):
        """returns the number of samples per epoch"""
        return self.samples_per_epoch

    def _normalize_block(self, block):
        """normalizes a block using cached global z-score stats"""
        if self._volume_is_normalized:
            return np.ascontiguousarray(block, dtype=np.float32)
        mean, std, g_min, g_max = self.norm_stats
        if std == 0:
            return block.astype(np.float32, copy=False)
        
        # z score normalization followed by scaling to [0, 1]
        norm_block = (block.astype(np.float32, copy=False) - mean) / std
        norm_block = (norm_block - g_min) / (g_max - g_min)
        # ensure dtype and contiguity
        return np.ascontiguousarray(np.clip(norm_block, 0, 1).astype(np.float32, copy=False))

    def _fetch_block(
        self,
        z_off,
        y_off,
        x_off,
        allow_jitter=True,
        depth_shift_override=None,
    ):
        """fetches and normalizes a block from zarr volume"""
        z = self.z_start + z_off
        y = self.y_start + y_off
        x = self.x_start + x_off
        tile = self.tile_size

        # context window: for single mode, read a larger crop centered on the tile so the
        # model sees the surround. the LABEL/mask stay the center tile (unchanged), so ring
        # supervision is respected -- context enters only via the conv receptive field.
        ctx = int(getattr(self.c.data, "context_size", 0) or 0)
        use_ctx = ctx > tile
        sp = ctx if use_ctx else tile
        target_offset = (0, 0)
        dj = 0
        augmentation_depth_shift = 0

        try:
            if use_ctx:
                pad = (ctx - tile) // 2
                target_aware = bool(getattr(self.c.data, "target_aware_ctx_jitter", False))
                max_j = int(getattr(self.c.data, "ctx_jitter", 0))
                if self._mt and not target_aware:
                    max_j = 0
                if self._mt and target_aware:
                    center = self._mt_grid * self._mt_sub
                    max_j = min(max_j, max(0, (ctx - center) // 2))
                if max_j > 0 and self.shuffle and allow_jitter:
                    # shift the context window; labeled tile moves to (pad+jy, pad+jx) in the block
                    # target-aware mode passes this offset to every model-side prediction crop
                    step = max(1, int(getattr(self.c.data, "context_downsample", 1)))
                    max_step = max_j // step
                    jy = random.randint(-max_step, max_step) * step
                    jx = random.randint(-max_step, max_step) * step
                    target_offset = (jy, jx)
                else:
                    jy = jx = 0
                # depth window jitter: shift which slices we read to attack depth-profile position memorization
                max_dj = int(getattr(self.c.data, "depth_jitter", 0))
                volume_depth = int(self.vol.shape[0])
                if self._surface_relative_depth_window:
                    surface_start = self._surface_centered_start(
                        y - pad - jy,
                        x - pad - jx,
                        ctx,
                        volume_depth,
                        target_y=y,
                        target_x=x,
                    )
                    if depth_shift_override is not None:
                        jitter = int(depth_shift_override)
                    else:
                        jitter = random.randint(-max_dj, max_dj) \
                            if max_dj > 0 and self.shuffle and allow_jitter else 0
                    augmentation_depth_shift = jitter
                    selected_start = min(
                        max(surface_start + jitter, 0),
                        max(volume_depth - self.depth, 0),
                    )
                    dj = selected_start - z
                elif max_dj > 0 and self.shuffle and allow_jitter:
                    min_dj = max(-max_dj, -z)
                    max_valid_dj = min(max_dj, volume_depth - (z + self.depth))
                    dj = random.randint(min_dj, max_valid_dj)
                    augmentation_depth_shift = dj
                block = self._read_ctx_block(z + dj, self.depth, y - pad - jy, x - pad - jx, ctx)
            else:
                if self._surface_relative_depth_window:
                    selected_start = self._surface_centered_start(
                        y,
                        x,
                        tile,
                        int(self.vol.shape[0]),
                        target_y=y,
                        target_x=x,
                    )
                    max_dj = int(getattr(self.c.data, "depth_jitter", 0))
                    if depth_shift_override is not None:
                        jitter = int(depth_shift_override)
                    else:
                        jitter = random.randint(-max_dj, max_dj) \
                            if max_dj > 0 and self.shuffle and allow_jitter else 0
                    augmentation_depth_shift = jitter
                    selected_start = min(
                        max(selected_start + jitter, 0),
                        max(int(self.vol.shape[0]) - self.depth, 0),
                    )
                    dj = selected_start - z
                block = np.asarray(
                    self.vol[z+dj:z+dj+self.depth, y:y+tile, x:x+tile],
                    dtype=np.float32,
                )
        except Exception:
            # any read error (OSError, corrupt chunk, zarr internal error) — return zeros
            block = np.zeros((self.depth, sp, sp), dtype=np.float32)

        # guard: zarr can silently return wrong shape on Windows under load
        if block.shape != (self.depth, sp, sp):
            block = np.zeros((self.depth, sp, sp), dtype=np.float32)

        return self._normalize_block(block), target_offset, dj, augmentation_depth_shift

    def _surface_centered_start(
        self,
        y0,
        x0,
        size,
        volume_depth,
        target_y=None,
        target_x=None,
    ):
        """choose a contiguous source window centered on the patch's literal surface."""
        height, width = self.surface_depth.shape
        target_size = (
            self._mt_grid * self._mt_sub if self._mt else self.tile_size
        )
        target_size = min(target_size, size)
        if target_y is None or target_x is None:
            target_y = y0 + (size - target_size) // 2
            target_x = x0 + (size - target_size) // 2
        elif self._mt:
            target_y += (self.tile_size - target_size) // 2
            target_x += (self.tile_size - target_size) // 2
        ys, ye = max(0, target_y), min(height, target_y + target_size)
        xs, xe = max(0, target_x), min(width, target_x + target_size)
        center_y = min(max(int(target_y + target_size // 2), 0), height - 1)
        center_x = min(max(int(target_x + target_size // 2), 0), width - 1)
        center_depth = int(self.surface_depth[center_y, center_x])
        center_confidence = int(self.surface_confidence[center_y, center_x])
        offset = self._surface_window_offset
        if center_depth != 255 and center_confidence > 0:
            return min(
                max(center_depth - (self.depth - 1) // 2 + offset, 0),
                max(int(volume_depth) - self.depth, 0),
            )
        if ys < ye and xs < xe:
            depth = np.asarray(self.surface_depth[ys:ye, xs:xe], dtype=np.uint8)
            confidence = np.asarray(self.surface_confidence[ys:ye, xs:xe], dtype=np.uint8)
            valid = (depth != 255) & (confidence > 0)
            if valid.any():
                center = int(np.rint(np.median(depth[valid].astype(np.float32))))
                return min(
                    max(center - (self.depth - 1) // 2 + offset, 0),
                    max(int(volume_depth) - self.depth, 0),
                )
        return min(max(int(self.z_start), 0), max(int(volume_depth) - self.depth, 0))

    def _fetch_surface_teacher(self, z_off, y_off, x_off, target_offset, depth_shift):
        """crop the offline teacher and convert absolute depths to input-local depths."""
        z = self.z_start + z_off + depth_shift
        y = self.y_start + y_off
        x = self.x_start + x_off
        tile = self.tile_size
        ctx = int(getattr(self.c.data, "context_size", 0) or 0)
        use_ctx = ctx > tile
        size = ctx if use_ctx else tile
        if use_ctx:
            pad = (ctx - tile) // 2
            jy, jx = target_offset
            y -= pad + jy
            x -= pad + jx

        depth = np.full((size, size), -1.0, dtype=np.float32)
        confidence = np.zeros((size, size), dtype=np.float32)
        height, width = self.surface_depth.shape
        ys, ye = max(0, y), min(height, y + size)
        xs, xe = max(0, x), min(width, x + size)
        if ys < ye and xs < xe:
            dst_y = slice(ys - y, ye - y)
            dst_x = slice(xs - x, xe - x)
            absolute = np.asarray(self.surface_depth[ys:ye, xs:xe], dtype=np.float32)
            local = absolute - float(z)
            local_confidence = np.asarray(
                self.surface_confidence[ys:ye, xs:xe],
                dtype=np.float32,
            ) / 255.0
            valid = (absolute != 255) & (local >= 0) & (local <= self.depth - 1)
            depth[dst_y, dst_x] = np.where(valid, local, -1.0)
            confidence[dst_y, dst_x] = np.where(valid, local_confidence, 0.0)
        return depth, confidence

    def _read_ctx_block(self, z, ndepth, y0, x0, ctx):
        """read a ctx x ctx spatial crop starting at absolute (y0,x0), zero-padding any region
        outside the volume/frame. used for the context-window input (centered on a tile)."""
        vol = self.vol
        D, H, W = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
        out = np.zeros((ndepth, ctx, ctx), dtype=np.float32)
        if z + ndepth > D:
            return out
        ys, ye = max(0, y0), min(H, y0 + ctx)
        xs, xe = max(0, x0), min(W, x0 + ctx)
        if ys < ye and xs < xe:
            try:
                src = np.asarray(vol[z:z+ndepth, ys:ye, xs:xe], dtype=np.float32)
                out[:, ys - y0:ye - y0, xs - x0:xe - x0] = src
            except Exception:
                pass
        return out

    def _fetch_label(self, y_off, x_off):
        """fetches a binary label tile"""
        if self._mt:
            return self._fetch_label_mt(y_off, x_off)
        y = self.y_start + y_off
        x = self.x_start + x_off
        label_tile = self.labels[y:y+self.tile_size, x:x+self.tile_size]
        if self._has_explicit_negative_mask and np.any(
            self.explicit_negative_mask[y:y+self.tile_size, x:x+self.tile_size] > 0
        ):
            return torch.tensor([-1.0], dtype=torch.float32)
        if self._has_explicit_positive_mask and np.any(
            self.explicit_positive_mask[y:y+self.tile_size, x:x+self.tile_size] > 0
        ):
            return torch.tensor([1.0], dtype=torch.float32)
        has_ink = bool(np.any(label_tile > 0.5))
        return torch.tensor([float(has_ink)], dtype=torch.float32)

    def _fetch_label_mt(self, y_off, x_off):
        """per-sub-tile labels over the grid*sub px center: 1 if .any() eroded ink else 0.
        every sub-tile is a target (papyrus unless ink); OOB reads clamp to papyrus (0)."""
        n, sub = self._mt_grid, self._mt_sub
        y0, _, x0, _ = self._mt_center_bounds(y_off, x_off)
        lbl = self.labels
        Hl, Wl = int(lbl.shape[0]), int(lbl.shape[1])
        out = np.zeros(n * n, dtype=np.float32)
        for iy in range(n):
            ys, ye = y0 + iy * sub, y0 + (iy + 1) * sub
            if ye <= 0 or ys >= Hl:
                continue
            ysc, yec = max(0, ys), min(Hl, ye)
            for ix in range(n):
                xs, xe = x0 + ix * sub, x0 + (ix + 1) * sub
                if xe <= 0 or xs >= Wl:
                    continue
                xsc, xec = max(0, xs), min(Wl, xe)
                index = iy * n + ix
                if self._has_explicit_negative_mask and np.any(
                    self.explicit_negative_mask[ysc:yec, xsc:xec] > 0
                ):
                    out[index] = -1.0
                elif self._has_explicit_positive_mask and np.any(
                    self.explicit_positive_mask[ysc:yec, xsc:xec] > 0
                ):
                    out[index] = 1.0
                elif np.any(lbl[ysc:yec, xsc:xec] > 0.5):
                    out[index] = 1.0
        return torch.from_numpy(out)

    def _fetch_mask_mt(self, y_off, x_off):
        """per-sub-tile validity over the grid in row-major order.

        pos_only keeps actual ink sub-tiles plus true ring-negative sub-tiles. non-ink cells
        inside a positive base tile stay unlabeled, and the closed-ring exclusion gap remains
        unlabeled. without pos_only, retain the legacy all-in-scroll center behavior.
        """
        n, sub = self._mt_grid, self._mt_sub
        y0, _, x0, _ = self._mt_center_bounds(y_off, x_off)
        m = self.scroll_mask
        target_mask = self.split_mask
        supervision_mask = self.mask if self._mt_ring_gate else None
        explicit_mask = self.explicit_negative_mask
        explicit_positive_mask = self.explicit_positive_mask
        lbl = self._fetch_label_mt(y_off, x_off).numpy()
        Hm, Wm = int(m.shape[0]), int(m.shape[1])
        out = np.zeros(n * n, dtype=np.float32)
        for iy in range(n):
            ys, ye = y0 + iy * sub, y0 + (iy + 1) * sub
            if ys < 0 or ye > Hm:
                continue
            for ix in range(n):
                xs, xe = x0 + ix * sub, x0 + (ix + 1) * sub
                if xs < 0 or xe > Wm:
                    continue
                explicit = explicit_mask is not None and np.all(
                    explicit_mask[ys:ye, xs:xe] > 0
                )
                explicit_positive = explicit_positive_mask is not None and np.any(
                    explicit_positive_mask[ys:ye, xs:xe] > 0
                )
                if (np.all(m[ys:ye, xs:xe] > 0)
                        and (target_mask is None or np.all(target_mask[ys:ye, xs:xe] > 0))
                        and (supervision_mask is None
                             or np.all(supervision_mask[ys:ye, xs:xe] > 0)
                             or explicit
                             or explicit_positive)):
                    idx = iy * n + ix
                    if self._mt_pos_only and lbl[idx] == 0:
                        # the combined supervision mask also covers the full positive 16px
                        # base tile. reject its non-ink 8px cells; only true ring tiles may
                        # provide negatives.
                        py0 = (ys // self.tile_size) * self.tile_size
                        px0 = (xs // self.tile_size) * self.tile_size
                        parent = self.labels[
                            py0:py0 + self.tile_size,
                            px0:px0 + self.tile_size,
                        ]
                        if np.any(parent > 0.5):
                            continue
                    out[idx] = 1.0
        # a mixed window contributes positive targets only. even true ring negatives are
        # discarded here because adjacency to uncertain ink boundaries is empirically harmful.
        # ink-free ring windows still provide negative supervision.
        if self._mt_pos_only and np.any((lbl > 0) & (out > 0)):
            out = out * (lbl != 0)
        return torch.from_numpy(out)

    def _fetch_mask(self, y_off, x_off):
        """fetches a mask tile"""
        if self._mt:
            return self._fetch_mask_mt(y_off, x_off)
        y = self.y_start + y_off
        x = self.x_start + x_off
        
        # slice the mask tile
        mask_tile = self.mask[y:y+self.tile_size, x:x+self.tile_size]
        return torch.from_numpy(np.asarray(mask_tile, dtype=np.float32))

    def __iter__(self) -> Iterator:
        """sets up the iterator for an epoch"""
        if self._character_balanced:
            shuffled_coords = self._character_balanced_coords()
        else:
            shuffled_coords = self.block_coords.copy()
        if self.shuffle:
            if not self._character_balanced:
                np.random.shuffle(shuffled_coords)
            # cap tiles per epoch (fresh random subset each epoch) to bound wall-time
            if self._max_samples is not None and len(shuffled_coords) > int(self._max_samples):
                shuffled_coords = shuffled_coords[:int(self._max_samples)]
            
        # handle multi-worker data loading
        worker_info = get_worker_info()
        if worker_info is None:
            # single-process loading
            self.worker_indices = shuffled_coords
        else:
            # split workload among workers
            per_worker = int(np.ceil(len(shuffled_coords) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(shuffled_coords))
            self.worker_indices = shuffled_coords[start:end]
            
        self.current_idx = 0
        return self

    def __next__(self):
        """returns the next item in the dataset"""
        if self.current_idx >= len(self.worker_indices):
            raise StopIteration
            
        # get coordinates for the next item
        z_off, y_off, x_off = self.worker_indices[self.current_idx]
        
        # fetch data components
        mask = self._fetch_mask(y_off, x_off)
        block, target_offset, depth_shift, augmentation_depth_shift = self._fetch_block(
            z_off,
            y_off,
            x_off,
        )
        if self._use_surface_teacher:
            surface_depth, surface_confidence = self._fetch_surface_teacher(
                z_off,
                y_off,
                x_off,
                target_offset,
                depth_shift,
            )
        label = self._fetch_label(y_off, x_off)
        component_ids = None
        if self._character_metrics or self._character_balanced:
            component_ids = torch.from_numpy(self._fetch_character_ids(
                y_off,
                x_off,
                label.numpy(),
                mask.numpy() > 0,
            ))
        target_offset_tensor = (
            torch.tensor(target_offset, dtype=torch.long)
            if bool(getattr(self.c.data, "target_aware_ctx_jitter", False)) else None
        )
        paired_block = block
        paired_active = 0.0
        if (
            self._context_consistency
            and self._context_donor_coords
            and random.random() < self._context_consistency_prob
        ):
            donor_y, donor_x = self._context_donor_coords[
                random.randrange(len(self._context_donor_coords))
            ]
            donor, _, _, _ = self._fetch_block(
                z_off,
                donor_y,
                donor_x,
                allow_jitter=False,
            )
            paired_block = self.transform.apply_context_replacement(
                block,
                donor,
                target_offset_tensor,
            )
            paired_active = 1.0
        paired_surface_depth = surface_depth if self._use_surface_teacher else None
        paired_surface_confidence = surface_confidence if self._use_surface_teacher else None
        if (
            self._depth_view_consistency
            and random.random() < self._depth_view_consistency_prob
        ):
            pair_offset = random.choice((
                -self._depth_view_consistency_offset,
                self._depth_view_consistency_offset,
            ))
            paired_block, _, paired_depth_shift, _ = self._fetch_block(
                z_off,
                y_off,
                x_off,
                allow_jitter=False,
                depth_shift_override=pair_offset,
            )
            paired_surface_depth, paired_surface_confidence = self._fetch_surface_teacher(
                z_off,
                y_off,
                x_off,
                target_offset,
                paired_depth_shift,
            )
            paired_active = float(paired_depth_shift != depth_shift)
        if (
            self.apply_transforms
            and self._context_donor_coords
            and random.random() < self._context_replace_prob
        ):
            donor_y, donor_x = self._context_donor_coords[
                random.randrange(len(self._context_donor_coords))
            ]
            donor, _, _, _ = self._fetch_block(
                z_off,
                donor_y,
                donor_x,
                allow_jitter=False,
            )
            block = self.transform.apply_context_replacement(
                block,
                donor,
                target_offset_tensor,
            )
        
        # apply transforms if enabled
        if self.apply_transforms:
            if self._mt:
                if self._context_consistency or self._depth_view_consistency:
                    transformed, paired_block = self.transform.paired(
                        block,
                        paired_block,
                        label,
                        mask,
                        component_ids,
                        target_offset_tensor,
                    )
                else:
                    transformed = self.transform(
                        block,
                        label,
                        mask,
                        component_ids,
                        target_offset_tensor,
                    )
                if component_ids is None and target_offset_tensor is None:
                    block, label, mask = transformed
                elif component_ids is None:
                    block, label, mask, target_offset_tensor = transformed
                elif target_offset_tensor is None:
                    block, label, mask, component_ids = transformed
                else:
                    block, label, mask, component_ids, target_offset_tensor = transformed
            else:
                block = self.transform(block)
            if self._use_surface_teacher:
                surface_depth, surface_confidence = self.transform.transform_surface_teacher(
                    surface_depth,
                    surface_confidence,
                )
                if self._depth_view_consistency:
                    paired_surface_depth, paired_surface_confidence = (
                        self.transform.transform_surface_teacher(
                            paired_surface_depth,
                            paired_surface_confidence,
                        )
                    )
        if self._use_surface_teacher:
            surface_downsample = max(1, int(getattr(self.c.data, "context_downsample", 1)))
            if surface_downsample > 1:
                surface_depth = surface_depth[::surface_downsample, ::surface_downsample]
                surface_confidence = surface_confidence[
                    ::surface_downsample,
                    ::surface_downsample,
                ]
                if self._depth_view_consistency:
                    paired_surface_depth = paired_surface_depth[
                        ::surface_downsample,
                        ::surface_downsample,
                    ]
                    paired_surface_confidence = paired_surface_confidence[
                        ::surface_downsample,
                        ::surface_downsample,
                    ]
        
        # enforce contiguity and dtype before converting to torch to avoid negative strides
        block = np.ascontiguousarray(block, dtype=np.float32)
        paired_block = np.ascontiguousarray(paired_block, dtype=np.float32)
            
        # convert to tensor for the model
        block_tensor = torch.from_numpy(block).unsqueeze(0)
        paired_block_tensor = torch.from_numpy(paired_block).unsqueeze(0)
        
        self.current_idx += 1
        with_domain = needs_domain_ids(self.c)
        result = [block_tensor, label, mask]
        if with_domain:
            result.append(torch.tensor(self.domain_id, dtype=torch.long))
        if needs_patch_ids(self.c):
            result.append(torch.tensor(self.scroll_id, dtype=torch.long))
        if component_ids is not None:
            result.append(component_ids)
        if target_offset_tensor is not None:
            result.append(target_offset_tensor)
        if self._use_surface_teacher:
            result.extend([
                torch.from_numpy(np.ascontiguousarray(surface_depth, dtype=np.float32)).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(surface_confidence, dtype=np.float32)).unsqueeze(0),
            ])
        if bool(getattr(self.c.tra, "depth_shift_aux", False)):
            result.append(torch.tensor(augmentation_depth_shift, dtype=torch.long))
        if self._context_consistency:
            result.extend([
                paired_block_tensor,
                torch.tensor(paired_active, dtype=torch.float32),
            ])
        if self._depth_view_consistency:
            result.extend([
                paired_block_tensor,
                torch.from_numpy(
                    np.ascontiguousarray(paired_surface_depth, dtype=np.float32)
                ).unsqueeze(0),
                torch.from_numpy(
                    np.ascontiguousarray(paired_surface_confidence, dtype=np.float32)
                ).unsqueeze(0),
                torch.tensor(paired_active, dtype=torch.float32),
            ])
        return tuple(result)


class MultiScrollIterableDataset(IterableDataset):
    """merges several InkVolumeDatasets into one stream so a single epoch sees
    tiles from every scroll fragment interleaved (batches are integrated, not
    alternated). each child handles its own per-worker sharding, so worker N
    receives shard N of every scroll."""
    def __init__(
        self,
        datasets,
        balance_scrolls=False,
        sampling_groups=None,
        sampling_weights=None,
    ):
        super().__init__()
        self.datasets = list(datasets)
        self.balance_scrolls = bool(balance_scrolls)
        self.sampling_groups = (
            [list(map(int, group)) for group in sampling_groups]
            if sampling_groups is not None else None
        )
        self.sampling_weights = (
            [int(weight) for weight in sampling_weights]
            if sampling_weights is not None else None
        )
        if self.sampling_groups is not None:
            if len(self.sampling_groups) == 0 or any(not group for group in self.sampling_groups):
                raise ValueError("sampling_groups must contain non-empty groups")
            flattened = sorted(index for group in self.sampling_groups for index in group)
            if flattened != list(range(len(self.datasets))):
                raise ValueError("sampling_groups must partition every dataset exactly once")
            if self.sampling_weights is None:
                self.sampling_weights = [1] * len(self.sampling_groups)
            if len(self.sampling_weights) != len(self.sampling_groups):
                raise ValueError("sampling_weights must match sampling_groups")
            if any(weight <= 0 for weight in self.sampling_weights):
                raise ValueError("sampling_weights must be positive integers")
        self._apply_transforms = False

    @property
    def apply_transforms(self):
        return self._apply_transforms

    @apply_transforms.setter
    def apply_transforms(self, value):
        # propagate to all children so augmentation toggles uniformly
        self._apply_transforms = value
        for d in self.datasets:
            d.apply_transforms = value

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __iter__(self) -> Iterator:
        if self.balance_scrolls and len(self.datasets) > 1:
            worker_info = get_worker_info()
            total = len(self)
            if worker_info is not None:
                total = int(np.ceil(total / float(worker_info.num_workers)))
            yielded = 0
            iterators = [iter(dataset) for dataset in self.datasets]
            if self.sampling_groups is not None:
                group_orders = [list(np.random.permutation(group)) for group in self.sampling_groups]
                group_positions = [0] * len(self.sampling_groups)
                schedule = [
                    group_index
                    for group_index, weight in enumerate(self.sampling_weights)
                    for _ in range(weight)
                ]
                while yielded < total:
                    for group_index in np.random.permutation(schedule):
                        if yielded >= total:
                            break
                        position = group_positions[group_index]
                        order = group_orders[group_index]
                        if position >= len(order):
                            order = list(np.random.permutation(self.sampling_groups[group_index]))
                            group_orders[group_index] = order
                            position = 0
                        dataset_index = order[position]
                        group_positions[group_index] = position + 1
                        try:
                            sample = next(iterators[dataset_index])
                        except StopIteration:
                            iterators[dataset_index] = iter(self.datasets[dataset_index])
                            sample = next(iterators[dataset_index])
                        yielded += 1
                        yield sample
                return
            while yielded < total:
                order = np.random.permutation(len(iterators))
                for index in order:
                    if yielded >= total:
                        break
                    try:
                        sample = next(iterators[index])
                    except StopIteration:
                        iterators[index] = iter(self.datasets[index])
                        sample = next(iterators[index])
                    yielded += 1
                    yield sample
            return
        # build child iterators (each shards itself by worker), then randomly
        # interleave samples until every child is exhausted
        iters = [iter(d) for d in self.datasets]
        active = list(range(len(iters)))
        while active:
            i = random.choice(active)
            try:
                yield next(iters[i])   # passes through the optional 4th (scroll_id) element
            except StopIteration:
                active.remove(i)


class DotPositiveDataset(IterableDataset):
    """yields only positive tiles from a binary dot-label image, no negatives or ring.
    used to inject sparse location-prior positives from ./dots/ alongside main training."""

    def __init__(self, data_manager: "DataManager", dot_label: np.ndarray):
        super().__init__()
        self.c = data_manager.c
        self._dm = data_manager
        T = self.c.data.tile_size
        mask = np.asarray(data_manager.mask)
        H = min(dot_label.shape[0], mask.shape[0])
        W = min(dot_label.shape[1], mask.shape[1])
        dot_b = dot_label[:H, :W] > 127
        mask_b = mask[:H, :W] > 0
        self._coords: list[tuple[int, int]] = []
        for ty in range(H // T):
            for tx in range(W // T):
                sl = (slice(ty * T, (ty + 1) * T), slice(tx * T, (tx + 1) * T))
                if dot_b[sl].any() and mask_b[sl].any():
                    self._coords.append((ty * T, tx * T))
        print(f"[dot-pos] scroll {data_manager.scroll_id}: {len(self._coords)} positive tiles from dots")

    def __len__(self):
        return len(self._coords)

    # flag required by MultiScrollIterableDataset.apply_transforms setter
    @property
    def apply_transforms(self):
        return False

    @apply_transforms.setter
    def apply_transforms(self, _value):
        pass

    def __iter__(self):
        coords = list(self._coords)
        random.shuffle(coords)
        worker_info = get_worker_info()
        if worker_info is not None:
            per = int(np.ceil(len(coords) / float(worker_info.num_workers)))
            coords = coords[worker_info.id * per: (worker_info.id + 1) * per]

        c = self.c
        T = c.data.tile_size
        D = c.data.depth
        z0 = c.data.d_start
        ctx = int(getattr(c.data, "context_size", 0) or 0)
        use_ctx = ctx > T
        sp = ctx if use_ctx else T
        pad = (ctx - T) // 2 if use_ctx else 0
        mean, std, g_min, g_max = self._dm.norm_stats
        domain_id = self._dm.domain_id
        with_domain = needs_domain_ids(c)
        mt_grid = max(1, int(getattr(c.model, "multitile_grid", 4)))

        for y0, x0 in coords:
            vol = self._dm.vol
            if z0 + D > int(vol.shape[0]):
                continue
            try:
                if use_ctx:
                    ys0, xs0 = y0 - pad, x0 - pad
                    Hv, Wv = int(vol.shape[1]), int(vol.shape[2])
                    out = np.zeros((D, ctx, ctx), dtype=np.float32)
                    ys, ye = max(0, ys0), min(Hv, ys0 + ctx)
                    xs, xe = max(0, xs0), min(Wv, xs0 + ctx)
                    if ys < ye and xs < xe:
                        src = np.array(vol[z0:z0 + D, ys:ye, xs:xe], dtype=np.float32)
                        out[:, ys - ys0:ye - ys0, xs - xs0:xe - xs0] = src
                    block = out
                else:
                    block = np.array(vol[z0:z0 + D, y0:y0 + T, x0:x0 + T], dtype=np.float32)
            except Exception:
                continue
            if block.shape != (D, sp, sp):
                continue
            block = (block - mean) / max(std, 1e-8)
            block = np.clip((block - g_min) / max(g_max - g_min, 1e-8), 0.0, 1.0)
            block_t = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).unsqueeze(0)
            # dots are confirmed ink, so every multitile target is positive and valid
            n2 = mt_grid * mt_grid
            label_t = torch.ones(n2, dtype=torch.float32)
            mask_t = torch.ones(n2, dtype=torch.float32)
            with_characters = bool(getattr(c.tra, "character_macro_metrics", False))
            with_offset = bool(getattr(c.data, "target_aware_ctx_jitter", False))
            extras = []
            if with_domain:
                extras.append(torch.tensor(domain_id, dtype=torch.long))
            if needs_patch_ids(c):
                extras.append(torch.tensor(self._dm.scroll_id, dtype=torch.long))
            if with_characters:
                extras.append(torch.zeros_like(label_t, dtype=torch.long))
            if with_offset:
                extras.append(torch.zeros(2, dtype=torch.long))
            yield (block_t, label_t, mask_t, *extras)


class DataManager:
    """manages data loading, splitting, and normalization"""
    def __init__(
        self,
        config: Config,
        scroll_id=None,
        domain_id: int = 0,
        character_namespace: int | None = None,
    ):
        """initializes the data manager.
        scroll_id: which scroll fragment to load; defaults to the first configured scroll.
        passing it explicitly lets the trainer build one manager per fragment."""
        self.c = config
        if scroll_id is None:
            scroll_id = config.data.scrolls[0].scroll_id
        self.scroll_id = int(scroll_id)
        self.domain_id = int(domain_id)
        self.character_namespace = int(
            character_namespace if character_namespace is not None else self.domain_id
        )

        self._prepared_cache_key = self._make_prepared_cache_key()
        with _PREPARED_DATASET_CACHE_LOCK:
            prepared = _PREPARED_DATASET_CACHE.get(self._prepared_cache_key)
        self._prepared_cache_entry = prepared

        if prepared is not None:
            self._restore_prepared_manager(prepared)
            print(f"[dataset-cache] scroll {self.scroll_id}: reusing prepared RAM assets")
            return

        # load raw data and define splits
        self.vol, self.mask, self.labels, self.train_x, self.valid_x, self.y_range = self._load_raw_data()

        # get or compute normalization statistics
        self.norm_stats = self._get_or_compute_norm()

    def _make_prepared_cache_key(self) -> tuple:
        """key all inputs that can affect manager or dataset preparation."""
        scroll = next(
            (item for item in getattr(self.c.data, "scrolls", ())
             if int(item.scroll_id) == self.scroll_id),
            None,
        )
        label_dir = str(getattr(self.c.data, "inklabel_dir", "./eroded_inklabels"))
        train_mask_dir = str(getattr(self.c.data, "train_mask_dir", "./train_masks"))
        paths = [
            os.path.join(str(getattr(self.c.data, "zarr_path", "./ves_zarrs2")), f"{self.scroll_id}.zarr"),
            os.path.join(label_dir, f"{self.scroll_id}.png"),
            os.path.join("./masks", f"{self.scroll_id}.png"),
        ]
        if str(getattr(self.c.data, "ring_label_source", "original")) == "original":
            paths.append(os.path.join("./inklabels", f"{self.scroll_id}.png"))
        if not bool(getattr(self.c.data, "simple_split", True)):
            paths.append(os.path.join(train_mask_dir, f"{self.scroll_id}.png"))
        data_fields = (
            "zarr_path", "tile_size", "depth", "d_start", "d_end", "train_d_start",
            "train_d_end", "mask_memmap", "mask_bitpack", "preload_volumes",
            "ring_negatives", "ring_label_source", "ring_from_inklabel_dir", "ring_close_r", "ring_gap_r",
            "ring_shell_r", "simple_split", "coordinate_hash_split",
            "coordinate_hash_block_size", "coordinate_hash_valid_fraction",
            "coordinate_hash_seed", "train_mask_dir", "surface_label_dir",
            "inklabel_dir", "label_dilate_r", "context_size", "context_downsample", "ctx_jitter",
            "target_aware_ctx_jitter", "depth_jitter", "surface_relative_depth_window",
            "multitile_train_step", "multitile_pos_only", "multitile_ring_gate",
            "character_balanced_sampling",
            "character_min_pixels", "max_samples_per_epoch",
        )
        dataloader_fields = (
            "context_replace_prob", "context_replace_min_mask_frac",
        )
        model_fields = (
            "multitile", "multitile_grid", "multitile_subtile", "new_learned_surface",
            "better_surface", "surface_teacher_input",
        )
        training_fields = (
            "character_macro_metrics",
            "context_consistency", "context_consistency_prob", "depth_view_consistency",
            "depth_view_consistency_prob", "depth_view_consistency_offset",
        )
        return (
            "prepared-dataset-v1",
            self.scroll_id,
            self.domain_id,
            self.character_namespace,
            _freeze_cache_value(vars(scroll) if scroll is not None else None),
            _selected_config_key(self.c.data, data_fields),
            _selected_config_key(self.c.dl, dataloader_fields),
            _selected_config_key(self.c.model, model_fields),
            _selected_config_key(self.c.tra, training_fields),
            needs_domain_ids(self.c),
            tuple(_path_fingerprint(path) for path in paths),
        )

    def _restore_prepared_manager(self, prepared: dict) -> None:
        """bind cached immutable arrays and metadata to a fresh manager."""
        for name, value in prepared["manager"].items():
            setattr(self, name, value)

    def _store_prepared_datasets(
        self,
        train_set: InkVolumeDataset,
        valid_set: InkVolumeDataset,
    ) -> None:
        manager_names = (
            "vol", "mask", "labels", "train_x", "valid_x", "y_range", "norm_stats",
            "full_x_range", "full_y_range", "manual_train_mask", "explicit_negative_mask",
            "explicit_positive_mask",
            "train_range", "valid_range", "shared_range", "split_axis",
        )
        entry = {
            "manager": {name: getattr(self, name) for name in manager_names},
            "train": {
                "mask": train_set.mask,
                "scroll_mask": train_set.scroll_mask if train_set._has_scroll_mask else None,
                "split_mask": train_set.split_mask,
                "character_grid": train_set._character_grid,
                "explicit_negative_mask": train_set.explicit_negative_mask,
                "explicit_positive_mask": train_set.explicit_positive_mask,
                "x_range": (train_set.x_start, train_set.x_end),
                "y_range": (train_set.y_start, train_set.y_end),
                "state": train_set.prepared_state(),
            },
            "valid": {
                "mask": valid_set.mask,
                "scroll_mask": valid_set.scroll_mask if valid_set._has_scroll_mask else None,
                "split_mask": valid_set.split_mask,
                "character_grid": valid_set._character_grid,
                "explicit_negative_mask": valid_set.explicit_negative_mask,
                "explicit_positive_mask": valid_set.explicit_positive_mask,
                "x_range": (valid_set.x_start, valid_set.x_end),
                "y_range": (valid_set.y_start, valid_set.y_end),
                "state": valid_set.prepared_state(),
            },
        }
        with _PREPARED_DATASET_CACHE_LOCK:
            _PREPARED_DATASET_CACHE.setdefault(self._prepared_cache_key, entry)
            self._prepared_cache_entry = _PREPARED_DATASET_CACHE[self._prepared_cache_key]

    def _load_raw_data(self):
        """loads raw zarr data and metadata"""
        # open the zarr volume in read-only mode
        zarr_dir = os.path.join(self.c.data.zarr_path, f"{self.scroll_id}.zarr")
        vol = zarr.open(zarr_dir, mode='r')
        if bool(getattr(self.c.data, "preload_volumes", False)):
            source_shape = tuple(vol.shape)
            source_dtype = np.dtype(vol.dtype)
            print(
                f"[preload] scroll {self.scroll_id}: loading "
                f"{vol.nbytes / 1024**3:.2f} GiB into RAM"
            )
            vol = np.ascontiguousarray(vol[:])
            if tuple(vol.shape) != source_shape or vol.dtype != source_dtype:
                raise RuntimeError(
                    f"preloaded volume integrity check failed for scroll {self.scroll_id}: "
                    f"expected shape={source_shape} dtype={source_dtype}, "
                    f"got shape={vol.shape} dtype={vol.dtype}"
                )
            vol.setflags(write=False)
            print(
                f"[preload] scroll {self.scroll_id}: ready "
                f"shape={vol.shape} dtype={vol.dtype} contiguous={vol.flags.c_contiguous}"
            )

        # load labels and mask, and normalize to [0, 1]
        lbl_dir = getattr(self.c.data, 'inklabel_dir', './eroded_inklabels')
        labels = imread_gray(f"{lbl_dir}/{self.scroll_id}.png")

        mask = imread_gray(f"./masks/{self.scroll_id}.png")

        if labels is None:
            raise FileNotFoundError(f"labels not found for scroll {self.scroll_id}")
        if mask is None:
            raise FileNotFoundError(f"mask not found for scroll {self.scroll_id}")

        label_dilate_r = int(getattr(self.c.data, "label_dilate_r", 0))
        if label_dilate_r < 0:
            raise ValueError("label_dilate_r must be non-negative")
        if label_dilate_r > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * label_dilate_r + 1, 2 * label_dilate_r + 1),
            )
            labels = cv2.dilate(labels, kernel)
            print(
                f"[labels] scroll {self.scroll_id}: dilated authoritative label "
                f"by {label_dilate_r}px"
            )

        labels = (labels.astype(np.float32) / 255.0)  # force float32 to avoid float64 OOM
        mask = mask / 255.0

        manual_mask = None
        train_mask_path = None
        if not bool(getattr(self.c.data, "simple_split", True)):
            train_mask_dir = str(getattr(self.c.data, "train_mask_dir", "./train_masks"))
            train_mask_path = os.path.join(train_mask_dir, f"{self.scroll_id}.png")
            manual_mask = imread_gray(train_mask_path)
            if manual_mask is None:
                raise FileNotFoundError(
                    f"manual train mask not found for scroll {self.scroll_id}: {train_mask_path}"
                )

        volume_h, volume_w = int(vol.shape[1]), int(vol.shape[2])
        if manual_mask is not None and any(
            abs(mask_size - volume_size) > 32
            for mask_size, volume_size in zip(
                manual_mask.shape,
                (volume_h, volume_w),
            )
        ):
            raise ValueError(
                f"manual train mask shape {manual_mask.shape} does not match volume "
                f"{(volume_h, volume_w)} for scroll {self.scroll_id}; resample the mask "
                "to the assembled volume grid"
            )
        common_h = min(volume_h, int(mask.shape[0]), int(labels.shape[0]))
        common_w = min(volume_w, int(mask.shape[1]), int(labels.shape[1]))
        if manual_mask is not None:
            common_h = min(common_h, int(manual_mask.shape[0]))
            common_w = min(common_w, int(manual_mask.shape[1]))
        if (volume_h, volume_w) != (common_h, common_w) \
                or mask.shape != (common_h, common_w) \
                or labels.shape != (common_h, common_w):
            print(
                f"[align] scroll {self.scroll_id}: volume={(volume_h, volume_w)} "
                f"mask={mask.shape} labels={labels.shape} -> common={(common_h, common_w)}"
            )
        mask = np.ascontiguousarray(mask[:common_h, :common_w])
        labels = np.ascontiguousarray(labels[:common_h, :common_w])
        if manual_mask is not None:
            manual_mask = np.ascontiguousarray(manual_mask[:common_h, :common_w])

        # define the working area and split for train/validation.
        # optional region crop (fractions of the full frame) trims the usable area so a run
        # can train on only a sub-region. then the train/valid split is applied along the
        # configured axis: 'x' = legacy vertical (left train / right valid), 'y' = horizontal
        # (top train / bottom valid). all boundaries are tile-aligned so the eval pred-map and
        # label-map shapes stay consistent.
        T = int(self.c.data.tile_size)
        H, W = common_h, common_w

        # per-scroll crop and split: first look up this scroll's ScrollConfig if it exists,
        # then fall back to global config fields for backward compatibility.
        _sc = None
        if hasattr(self.c.data, "scrolls"):
            for s in self.c.data.scrolls:
                if s.scroll_id == self.scroll_id:
                    _sc = s; break
        cxf = _sc.crop_x_frac if _sc else getattr(self.c.data, "crop_x_frac", (0.0, 1.0))
        cyf = _sc.crop_y_frac if _sc else getattr(self.c.data, "crop_y_frac", (0.0, 1.0))

        x0 = (int(W * float(cxf[0])) // T) * T
        x1 = (int(W * float(cxf[1])) // T) * T
        y0 = (int(H * float(cyf[0])) // T) * T
        y1 = (int(H * float(cyf[1])) // T) * T
        x1 = max(x1, x0 + T); y1 = max(y1, y0 + T)
        self.full_x_range = (x0, x1)
        self.full_y_range = (y0, y1)

        # manual split mode assigns the existing positive/ring supervision units by a
        # per-scroll binary image. fail loudly rather than silently reverting to the old
        # axis split: a missing mask would invalidate the experiment's train/valid meaning.
        self.manual_train_mask = None
        self.explicit_negative_mask = None
        self.explicit_positive_mask = None
        if not bool(getattr(self.c.data, "simple_split", True)):
            if manual_mask.shape != mask.shape or manual_mask.shape != labels.shape:
                raise ValueError(
                    f"manual train mask shape {manual_mask.shape} does not match scroll mask "
                    f"{mask.shape} and labels {labels.shape} for scroll {self.scroll_id}"
                )
            normal_train = manual_mask >= 240
            # current masks use exact paint-tool palette values: 119, 185, and 255
            explicit_negative = (manual_mask >= 112) & (manual_mask <= 143)
            explicit_positive = manual_mask == 185
            explicit_unit = (
                int(getattr(self.c.model, "multitile_subtile", T))
                if getattr(self.c.model, "multitile", False)
                else T
            )
            explicit_negative = self._align_manual_mask(
                explicit_negative.astype(np.uint8),
                explicit_unit,
            ) > 0
            explicit_negative &= mask > 0
            explicit_positive &= mask > 0
            overlap = explicit_negative & (labels > 0.5)
            if overlap.any():
                print(
                    f"[explicit-negative] scroll {self.scroll_id}: overriding "
                    f"{int(overlap.sum()):,} ink-label pixels"
                )
            conflict = explicit_negative & explicit_positive
            if conflict.any():
                raise ValueError(
                    f"manual mask has {int(conflict.sum()):,} pixels marked both positive and negative"
                )
            labels[explicit_positive] = 1.0
            if bool(getattr(self.c.data, "coordinate_hash_split", False)):
                self.manual_train_mask = self._coordinate_hash_assignment(
                    labels.shape,
                    explicit_unit,
                )
            else:
                self.manual_train_mask = (
                    normal_train | explicit_negative | explicit_positive
                ).astype(np.uint8)
            self.explicit_negative_mask = explicit_negative.astype(np.uint8)
            self.explicit_positive_mask = explicit_positive.astype(np.uint8)
            frac_train = float(self.manual_train_mask.mean())
            print(
                f"[split] scroll {self.scroll_id}: manual mask={train_mask_path} "
                f"train_pixels={100.0 * frac_train:.1f}% "
                f"explicit_negative_pixels={int(explicit_negative.sum()):,} "
                f"explicit_positive_pixels={int(explicit_positive.sum()):,}"
            )

        # resolve split axis and fraction: ScrollConfig takes priority, then split_overrides
        # dict (backward compat with campaign runners), then global config defaults.
        axis = getattr(self.c.data, "split_axis", "x")
        frac = getattr(self.c.data, "train_split_frac", 0.75)
        if _sc:
            axis = _sc.split_axis
            frac = _sc.train_split_frac
            print(f"[split] scroll {self.scroll_id}: axis={axis} train_frac={frac}")
        else:
            # legacy: check split_overrides dict produced by Config.split_overrides() or
            # passed explicitly by campaign runners using the old API
            _ov = {}
            if callable(getattr(self.c, "split_overrides", None)):
                _ov = self.c.split_overrides()
            elif isinstance(getattr(self.c.data, "split_overrides", None), dict):
                _ov = self.c.data.split_overrides
            ov = _ov.get(self.scroll_id, _ov.get(str(self.scroll_id)))
            if ov:
                axis = ov.get("axis", axis)
                frac = ov.get("frac", frac)
                print(f"[split-override] scroll {self.scroll_id}: axis={axis} train_frac={frac}")
        axis = str(axis).lower()

        if axis == "y":
            # horizontal split: train = top, valid = bottom; x fully shared (cropped)
            span = y1 - y0
            split = (int(span * frac) // T) * T
            self.train_range = (y0, y0 + split)       # y-range for TRAIN
            self.valid_range = (y0 + split, y1)       # y-range for VALID
            self.shared_range = (x0, x1)              # x-range shared by both
            # legacy attrs kept defined (unused on the y path)
            train_x_range = (x0, x1)
            valid_x_range = (x0, x1)
            y_range = (y0, y1)
        else:
            # legacy vertical split: train = left, valid = right; y fully shared (cropped)
            span = x1 - x0
            split = (int(span * frac) // T) * T
            train_x_range = (x0, x0 + split)
            valid_x_range = (x0 + split, x1)
            y_range = (y0, y1)
            self.train_range = train_x_range
            self.valid_range = valid_x_range
            self.shared_range = y_range
        self.split_axis = axis

        return vol, mask, labels, train_x_range, valid_x_range, y_range

    def _get_or_compute_norm(self):
        """retrieve cached norm stats; if absent, compute with the fast chunk-aligned method."""
        from .norm import compute_norm, load_cached_norm, UNIFIED_CACHE_PATH
        seg_id = str(self.scroll_id)
        cached = load_cached_norm(seg_id, UNIFIED_CACHE_PATH)
        if cached is not None:
            return cached
        print(f"[info] computing normalization for segment {seg_id} (chunk-aligned pass)")
        zarr_path = getattr(self.c.data, "zarr_path", "./ves_zarrs2")
        return compute_norm(seg_id, zarr_path, UNIFIED_CACHE_PATH)

    def enable_selective_chunk_preload(self, *datasets: InkVolumeDataset) -> None:
        """eagerly retain the union of future dataset reads in campaign RAM."""
        if isinstance(self.vol, np.ndarray):
            raise ValueError("selective chunk preload cannot follow full volume preload")
        from .chunk_cache import SelectiveChunkVolume, get_selective_chunk_volume

        cached_volume = (
            self.vol if isinstance(self.vol, SelectiveChunkVolume)
            else get_selective_chunk_volume(self.vol, norm_stats=self.norm_stats)
        )
        if all(dataset._vol_obj is cached_volume and dataset._volume_is_normalized for dataset in datasets):
            self.vol = cached_volume
            print(
                f"[selective-cache] scroll {self.scroll_id}: prepared manifest already bound",
                flush=True,
            )
            return
        required: set[tuple[int, int]] = set()
        for dataset in datasets:
            required.update(dataset.required_volume_spatial_chunks(cached_volume.chunks))
        total_spatial = math.ceil(cached_volume.shape[1] / cached_volume.chunks[1]) \
            * math.ceil(cached_volume.shape[2] / cached_volume.chunks[2])
        print(
            f"[selective-cache] scroll {self.scroll_id}: manifest={len(required):,}/"
            f"{total_spatial:,} spatial chunks "
            f"({100.0 * len(required) / max(total_spatial, 1):.1f}%)",
            flush=True,
        )
        cached_volume.preload(
            required,
            self.vol,
            workers=int(getattr(self.c.data, "selective_chunk_workers", 8)),
        )
        self.vol = cached_volume
        for dataset in datasets:
            dataset.set_volume(cached_volume)
        if self._prepared_cache_entry is not None:
            self._prepared_cache_entry["manager"]["vol"] = cached_volume

    def get_datasets(self):
        """creates train and validation datasets.
        for split_axis='y' (horizontal): train=top rows, valid=bottom rows, x fully shared.
        for split_axis='x' (legacy vertical): train=left cols, valid=right cols, y fully shared.
        when simple_split=False, both datasets span the full cropped frame and the binary
        train_masks/<scroll_id>.png partitions the existing ring supervision into disjoint
        train and validation masks.
        InkVolumeDataset takes (x_range, y_range); we feed the split range on the split axis and
        the shared range on the other axis."""
        prepared_train = (
            self._prepared_cache_entry["train"]
            if self._prepared_cache_entry is not None else None
        )
        prepared_valid = (
            self._prepared_cache_entry["valid"]
            if self._prepared_cache_entry is not None else None
        )
        if prepared_train is not None and prepared_valid is not None:
            train_set = InkVolumeDataset(
                self.vol,
                prepared_train["mask"],
                self.labels,
                self.c,
                prepared_train["x_range"],
                prepared_train["y_range"],
                self.norm_stats,
                shuffle=True,
                scroll_id=self.scroll_id,
                domain_id=self.domain_id,
                character_namespace=self.character_namespace,
                scroll_mask=prepared_train["scroll_mask"],
                split_mask=prepared_train["split_mask"],
                character_grid=prepared_train["character_grid"],
                explicit_negative_mask=prepared_train["explicit_negative_mask"],
                explicit_positive_mask=prepared_train["explicit_positive_mask"],
                prepared_state=prepared_train["state"],
            )
            valid_set = InkVolumeDataset(
                self.vol,
                prepared_valid["mask"],
                self.labels,
                self.c,
                prepared_valid["x_range"],
                prepared_valid["y_range"],
                self.norm_stats,
                shuffle=False,
                scroll_id=self.scroll_id,
                domain_id=self.domain_id,
                character_namespace=self.character_namespace,
                scroll_mask=prepared_valid["scroll_mask"],
                split_mask=prepared_valid["split_mask"],
                character_grid=prepared_valid["character_grid"],
                explicit_negative_mask=prepared_valid["explicit_negative_mask"],
                explicit_positive_mask=prepared_valid["explicit_positive_mask"],
                prepared_state=prepared_valid["state"],
            )
            return train_set, valid_set
        supervision_mask = self._make_ring_mask() if getattr(self.c.data, 'ring_negatives', False) else self.mask
        manual_split = not bool(getattr(self.c.data, "simple_split", True))
        coordinate_hash_split = bool(
            getattr(self.c.data, "coordinate_hash_split", False)
        )
        character_aware = (
            bool(getattr(self.c.tra, "character_macro_metrics", False))
            or bool(getattr(self.c.data, "character_balanced_sampling", False))
        )
        split_unit = (
            int(getattr(self.c.model, "multitile_subtile", self.c.data.tile_size))
            if getattr(self.c.model, "multitile", False)
            else int(self.c.data.tile_size)
        )
        character_grid = None
        if character_aware:
            if not bool(getattr(self.c.model, "multitile", False)):
                raise ValueError("character-aware mode currently requires multitile=True")
            character_grid = self._build_character_grid(
                self.labels,
                split_unit,
                int(getattr(self.c.data, "character_min_pixels", 8)),
            )
        if manual_split:
            if self.manual_train_mask is None:
                raise RuntimeError(f"manual split mask was not loaded for scroll {self.scroll_id}")

            # align the hand mask to the model's actual target grid. a target unit is assigned
            # to train if any hand-mask pixel touches it; the expanded unit is then wholly train
            # or wholly valid, so no multitile target can leak into both datasets.
            assignment = self._align_manual_mask(self.manual_train_mask, split_unit)
            explicit_negative = self._align_manual_mask(
                self.explicit_negative_mask,
                split_unit,
            )
            explicit_negative = (
                (explicit_negative > 0) & (np.asarray(self.mask) > 0.5)
            ).astype(np.uint8)
            explicit_positive = self._align_manual_mask(
                self.explicit_positive_mask,
                split_unit,
            )
            explicit_positive = (
                (explicit_positive > 0) & (np.asarray(self.mask) > 0.5)
            ).astype(np.uint8)
            if character_grid is not None:
                character_grid = self._exclude_characters_crossing_split(
                    character_grid,
                    assignment,
                    split_unit,
                )
            eligible = np.asarray(supervision_mask) > 0.5
            if getattr(self.c.model, "multitile", False):
                # preserve the legacy ring-window gate and partition only the emitted targets
                combined_mask = np.maximum.reduce(
                    (supervision_mask, explicit_negative, explicit_positive)
                )
                train_mask = combined_mask
                valid_mask = combined_mask if coordinate_hash_split else supervision_mask
                train_split_mask = assignment
                valid_split_mask = (assignment == 0).astype(np.uint8)
            else:
                train_mask = (
                    (eligible & (assignment > 0))
                    | (explicit_negative > 0)
                    | (explicit_positive > 0)
                ).astype(np.uint8)
                valid_mask = (eligible & (assignment == 0)).astype(np.uint8)
                train_split_mask = valid_split_mask = None
            print(
                f"[manual-split] scroll {self.scroll_id}: unit={split_unit}px "
                f"train_ring={int((eligible & (assignment > 0)).sum())}px "
                f"valid_ring={int((eligible & (assignment == 0)).sum())}px"
            )
        else:
            train_mask = supervision_mask
            explicit_negative = None
            explicit_positive = None
        # when ring_negatives is on, restrict validation to ring tiles too so validation
        # throughput and signal quality match the training distribution. without this,
        # the full valid region (tens of thousands of easy tiles) swamps the validation
        # loop and makes it take 5-10× longer than necessary.
            valid_mask = train_mask if getattr(self.c.data, 'ring_negatives', False) else self.mask
            train_split_mask = valid_split_mask = None
        # multitile needs the true scroll mask (papyrus bounds) separate from the ring-
        # restricted training mask. only allocate it for multitile: single-tile never reads
        # scroll_mask, so passing None keeps the control byte-identical (no extra array).
        scroll_mask_arg = self.mask if (
            getattr(self.c.data, 'ring_negatives', False)
            and getattr(self.c.model, 'multitile', False)
        ) else None
        if manual_split:
            train_x = valid_x = self.full_x_range
            train_y = valid_y = self.full_y_range
        elif getattr(self, "split_axis", "x") == "y":
            train_x, train_y = self.shared_range, self.train_range
            valid_x, valid_y = self.shared_range, self.valid_range
        else:
            train_x, train_y = self.train_range, self.shared_range
            valid_x, valid_y = self.valid_range, self.shared_range
        if character_grid is not None and not manual_split:
            character_grid = self._exclude_characters_crossing_ranges(
                character_grid,
                split_unit,
                train_x,
                train_y,
                valid_x,
                valid_y,
            )
        train_set = InkVolumeDataset(
            self.vol,
            train_mask,
            self.labels,
            self.c,
            train_x,
            train_y,
            self.norm_stats,
            shuffle=True,
            scroll_id=self.scroll_id,
            domain_id=self.domain_id,
            character_namespace=self.character_namespace,
            scroll_mask=scroll_mask_arg,
            split_mask=train_split_mask,
            character_grid=character_grid,
            explicit_negative_mask=explicit_negative,
            explicit_positive_mask=explicit_positive,
        )
        valid_set = InkVolumeDataset(
            self.vol,
            valid_mask,
            self.labels,
            self.c,
            valid_x,
            valid_y,
            self.norm_stats,
            shuffle=False,
            scroll_id=self.scroll_id,
            domain_id=self.domain_id,
            character_namespace=self.character_namespace,
            scroll_mask=scroll_mask_arg,
            split_mask=valid_split_mask,
            character_grid=character_grid,
        )
        # the datasets have already copied what they need as uint8. the manager's own
        # float64 mask/labels (mask/255.0) are not used on the training side afterward
        # — for the big scroll they are ~1.9 GB EACH, so a many-scroll run would carry
        # gigabytes of dead float arrays in the main process. downcast to binary uint8
        # (8x smaller; only ever tested as >0.5 / >0, so exact). idempotent, so the
        # alternating-ring path's second get_datasets() call is safe. NB: the figure
        # visualizer keeps its OWN separate float copies and never calls this method.
        self.mask = (np.asarray(self.mask) > 0.5).astype(np.uint8)
        self.labels = (np.asarray(self.labels) > 0.5).astype(np.uint8)
        train_set._labels_arr = self.labels
        valid_set._labels_arr = self.labels
        if self._prepared_cache_entry is None:
            self._store_prepared_datasets(train_set, valid_set)
        return train_set, valid_set

    @staticmethod
    def _build_character_grid(labels, unit, min_pixels=8):
        """map full-resolution connected ink components onto the multitile target grid."""
        binary = (np.asarray(labels) > 0.5).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        keep = stats[:, cv2.CC_STAT_AREA] >= max(1, int(min_pixels))
        keep[0] = False
        remap = np.zeros(count, dtype=np.int32)
        kept_ids = np.flatnonzero(keep)
        remap[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
        components = remap[components]

        unit = max(1, int(unit))
        height = (components.shape[0] // unit) * unit
        width = (components.shape[1] // unit) * unit
        grid_h, grid_w = height // unit, width // unit
        character_grid = np.zeros((grid_h, grid_w), dtype=np.int32)
        ys, xs = np.nonzero(components[:height, :width])
        if ys.size:
            component_ids = components[ys, xs].astype(np.int64)
            cell_ids = (ys // unit).astype(np.int64) * grid_w + (xs // unit)
            keys = cell_ids * (len(kept_ids) + 1) + component_ids
            unique_keys, pixel_counts = np.unique(keys, return_counts=True)
            unique_cells = unique_keys // (len(kept_ids) + 1)
            unique_components = unique_keys % (len(kept_ids) + 1)
            order = np.lexsort((-pixel_counts, unique_cells))
            sorted_cells = unique_cells[order]
            first = np.concatenate(([True], sorted_cells[1:] != sorted_cells[:-1]))
            chosen_cells = sorted_cells[first]
            chosen_components = unique_components[order][first]
            character_grid.flat[chosen_cells] = chosen_components.astype(np.int32)
        print(
            f"[character-components] {len(kept_ids)} components at {unit}px target resolution"
        )
        return character_grid

    @staticmethod
    def _exclude_characters_crossing_split(character_grid, assignment, unit, max_minority=0.05):
        """assign split-crossing components to their majority side; exclude large crossings."""
        unit = max(1, int(unit))
        grid_h, grid_w = character_grid.shape
        cells = assignment[:grid_h * unit, :grid_w * unit].reshape(
            grid_h, unit, grid_w, unit
        ).all(axis=(1, 3))
        crossing = []
        trimmed = 0
        out = character_grid.copy()
        for component_id in np.unique(character_grid):
            if component_id <= 0:
                continue
            member = character_grid == component_id
            values = cells[member]
            if values.any() and not values.all():
                train_fraction = float(values.mean())
                if min(train_fraction, 1.0 - train_fraction) > max_minority:
                    crossing.append(int(component_id))
                else:
                    # minority cells lose their character id so no character spans both splits
                    out[member & (cells != (train_fraction > 0.5))] = 0
                    trimmed += 1
        if crossing:
            out[np.isin(out, crossing)] = 0
        if crossing or trimmed:
            print(
                f"[character-split] excluded {len(crossing)} character(s) crossing the fixed split; "
                f"assigned {trimmed} to their majority side (minority <= {max_minority:.0%})"
            )
        return out

    @staticmethod
    def _exclude_characters_crossing_ranges(
        character_grid,
        unit,
        train_x,
        train_y,
        valid_x,
        valid_y,
    ):
        """exclude connected characters spanning both axis-based dataset ranges."""
        unit = max(1, int(unit))
        height, width = character_grid.shape

        def ids_in(x_range, y_range):
            y0 = max(0, int(y_range[0]) // unit)
            y1 = min(height, int(y_range[1]) // unit)
            x0 = max(0, int(x_range[0]) // unit)
            x1 = min(width, int(x_range[1]) // unit)
            return set(np.unique(character_grid[y0:y1, x0:x1])) - {0}

        crossing = ids_in(train_x, train_y) & ids_in(valid_x, valid_y)
        out = character_grid.copy()
        if crossing:
            out[np.isin(out, list(crossing))] = 0
            print(
                f"[character-split] excluded {len(crossing)} character(s) crossing axis split"
            )
        return out

    @staticmethod
    def _align_manual_mask(mask, unit):
        """expand a binary hand mask to disjoint, origin-anchored model target units."""
        mask = np.asarray(mask) > 0
        unit = max(1, int(unit))
        h, w = mask.shape
        h_full = (h // unit) * unit
        w_full = (w // unit) * unit
        aligned = np.zeros((h, w), dtype=np.uint8)
        if h_full > 0 and w_full > 0:
            cells = mask[:h_full, :w_full].reshape(
                h_full // unit, unit, w_full // unit, unit
            ).any(axis=(1, 3))
            aligned[:h_full, :w_full] = np.repeat(
                np.repeat(cells, unit, axis=0), unit, axis=1
            )
        return aligned

    def _coordinate_hash_assignment(self, shape, unit):
        """assign stable spatial blocks to train or validation without using labels."""
        height, width = map(int, shape)
        unit = max(1, int(unit))
        requested_block = int(getattr(self.c.data, "coordinate_hash_block_size", 256))
        block = max(unit, ((requested_block + unit - 1) // unit) * unit)
        valid_fraction = float(
            getattr(self.c.data, "coordinate_hash_valid_fraction", 0.25)
        )
        if not 0.0 < valid_fraction < 1.0:
            raise ValueError("coordinate_hash_valid_fraction must be in (0, 1)")
        seed = int(getattr(self.c.data, "coordinate_hash_seed", 29))
        block_rows = (height + block - 1) // block
        block_cols = (width + block - 1) // block
        valid_blocks = np.zeros((block_rows, block_cols), dtype=bool)
        mask64 = (1 << 64) - 1
        threshold = int(valid_fraction * (1 << 64))

        for block_y in range(block_rows):
            for block_x in range(block_cols):
                value = (
                    seed
                    ^ (int(self.scroll_id) * 0x9E3779B97F4A7C15)
                    ^ (block_y * 0xBF58476D1CE4E5B9)
                    ^ (block_x * 0x94D049BB133111EB)
                ) & mask64
                value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask64
                value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask64
                value = (value ^ (value >> 31)) & mask64
                valid_blocks[block_y, block_x] = value < threshold

        if valid_blocks.size > 1:
            if valid_blocks.all():
                valid_blocks[0, 0] = False
            elif not valid_blocks.any():
                valid_blocks[0, 0] = True
        assignment = np.repeat(
            np.repeat(~valid_blocks, block, axis=0),
            block,
            axis=1,
        )[:height, :width].astype(np.uint8)
        print(
            f"[coordinate-hash-split] scroll {self.scroll_id}: block={block}px "
            f"seed={seed} valid_blocks={int(valid_blocks.sum())}/{valid_blocks.size}"
        )
        return assignment

    def _make_ring_mask(self):
        """build training mask from ring around ink labels, computed at TILE level.

        uses ORIGINAL inklabels (not eroded) to determine which tiles contain ink
        for the ring boundary. this prevents original-ink boundary tiles from
        becoming false-negative ring tiles (which was causing 20.9% contamination).

        training POSITIVE labels still come from eroded_inklabels (conservative).
        ring NEGATIVES are tiles adjacent to original-ink tiles with zero original ink.
        """
        h = min(self.labels.shape[0], self.mask.shape[0])
        w = min(self.labels.shape[1], self.mask.shape[1])
        labels_crop = self.labels[:h, :w]   # eroded — used for positive tile detection
        mask_crop   = self.mask[:h, :w]
        T = self.c.data.tile_size

        # determine which labels to use for ring boundary computation
        ring_source = getattr(self.c.data, 'ring_label_source', 'original')
        if ring_source == 'closed' and bool(getattr(self.c.data, 'ring_from_inklabel_dir', False)):
            # forced positives/negatives from train_masks never seed a ring
            ring_labels = np.array(labels_crop, dtype=np.float32)
            for forced in (self.explicit_positive_mask, self.explicit_negative_mask):
                if forced is not None:
                    ring_labels[np.asarray(forced)[:h, :w] > 0] = 0.0
        elif ring_source in {'original', 'closed'}:
            orig_path = f"./inklabels/{self.scroll_id}.png"
            orig_img = imread_gray(orig_path)
            if orig_img is not None:
                orig_img = (orig_img / 255.0)[:h, :w]
                ring_labels = orig_img
            else:
                print(f"[ring] original inklabels not found at {orig_path}, falling back to eroded")
                ring_labels = labels_crop
        else:
            # 'eroded' builds the ring from the configured authoritative label map
            ring_labels = labels_crop

        # build tile-level maps using ring_labels for boundary, eroded for positives
        n_ty = h // T
        n_tx = w // T
        # ink_tile: positive training tiles (eroded)
        ink_tile_eroded = np.zeros((n_ty, n_tx), dtype=np.uint8)
        # ink_tile for ring boundary (original or eroded depending on ring_source)
        ink_tile_ring   = np.zeros((n_ty, n_tx), dtype=np.uint8)
        mask_tile = np.zeros((n_ty, n_tx), dtype=np.uint8)
        for ty in range(n_ty):
            for tx in range(n_tx):
                tile_lbl_ero  = labels_crop[ty*T:(ty+1)*T, tx*T:(tx+1)*T]
                tile_lbl_ring = ring_labels[ty*T:(ty+1)*T, tx*T:(tx+1)*T]
                tile_mask     = mask_crop[ty*T:(ty+1)*T, tx*T:(tx+1)*T]
                if np.any(tile_lbl_ero  > 0.5): ink_tile_eroded[ty, tx] = 1
                if np.any(tile_lbl_ring > 0.5): ink_tile_ring[ty, tx]   = 1
                if np.any(tile_mask     > 0.5): mask_tile[ty, tx]        = 1

        # for 'closed': close letter holes then add explicit air gap before ring.
        # base map is the (hand-cleaned) eroded ink (see ring_labels above). radii are in
        # TILE units, config-driven; physical distance = radius * tile_size.
        if ring_source == 'closed':
            CLOSE_R = int(getattr(self.c.data, 'ring_close_r', 3))
            GAP_R   = int(getattr(self.c.data, 'ring_gap_r', 3))
            # stage 1: close interior holes in letters (mild closing)
            if CLOSE_R > 0:
                k_close = 2 * CLOSE_R + 1
                kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
                ink_tile_ring = cv2.erode(cv2.dilate(ink_tile_ring, kern_close), kern_close) & mask_tile
            # stage 2: dilate closed region by GAP_R -> exclusion zone; ring starts outside this.
            # ink_tile_ring now holds the exclusion zone used by the ring computation below.
            if GAP_R > 0:
                k_gap = 2 * GAP_R + 1
                kern_gap = cv2.getStructuringElement(cv2.MORPH_RECT, (k_gap, k_gap))
                ink_tile_ring = cv2.dilate(ink_tile_ring, kern_gap) & mask_tile
            print(f"[ring] closed(base=regular): CLOSE_R={CLOSE_R} GAP_R={GAP_R} exclusion_tiles={ink_tile_ring.sum()}")

        ink_count = int(ink_tile_eroded.sum())
        if ink_count == 0:
            return self.mask

        shell_r = int(getattr(self.c.data, 'ring_shell_r', 0))
        if shell_r > 0:
            # fixed ring shell width (tiles): a shell_r-thick band just outside the ring
            # boundary/exclusion zone. count is whatever the geometry yields (NOT balanced).
            best_r  = shell_r
            k       = 2 * best_r + 1
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            dilated = cv2.dilate(ink_tile_ring, kernel)
            ring    = ((dilated - ink_tile_ring) > 0) & (mask_tile > 0)
        else:
            # dilate ring_labels tile map until ring count >= eroded ink count (balanced)
            lo, hi = 1, 50
            best_r = hi
            while lo <= hi:
                mid = (lo + hi) // 2
                k = 2 * mid + 1
                kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                dilated = cv2.dilate(ink_tile_ring, kernel)
                # ring: adjacent to ring-source ink, but contains NO ring-source ink
                ring    = ((dilated - ink_tile_ring) > 0) & (mask_tile > 0)
                if int(ring.sum()) >= ink_count:
                    best_r = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            k       = 2 * best_r + 1
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            dilated = cv2.dilate(ink_tile_ring, kernel)
            ring    = ((dilated - ink_tile_ring) > 0) & (mask_tile > 0)

        ring_count = int(ring.sum())
        print(f"[ring_negatives] source='{ring_source}' tile_radius={best_r}  "
              f"ink_tiles={ink_count}  ring_tiles={ring_count}  "
              f"ratio={ring_count/max(ink_count,1):.2f}")

        # expand back to pixel level: positive = eroded ink, negative = ring
        train_mask = np.zeros_like(self.mask, dtype=np.float32)
        for ty in range(n_ty):
            for tx in range(n_tx):
                if ink_tile_eroded[ty, tx] or ring[ty, tx]:
                    y0, x0 = ty*T, tx*T
                    train_mask[y0:y0+T, x0:x0+T] = 1.0

        return train_mask

def _worker_init(worker_id, base_seed):
    """deterministic per-worker seeding: each spawned worker reseeds numpy+random from a
    base seed + worker id, so the augmentations (which draw from the GLOBAL np.random /
    random state) are reproducible across runs even with num_workers>0. MUST be module
    level (not a closure) so it can be pickled for the Windows 'spawn' start method."""
    s = base_seed + worker_id
    np.random.seed(s)
    random.seed(s)

def get_dataloaders(train_dataset, valid_dataset, config: Config):
    """creates dataloader objects from datasets"""
    # with num_workers=0 the seeding is unused and the main-process set_seed already covers it.
    _base_seed = int(getattr(config.tra, "seed", 41))

    # build dataloader kwargs conditionally to avoid ValueError when num_workers=0
    train_loader_kwargs = {
        "batch_size": config.dl.batch_size,
        "num_workers": config.dl.num_workers,
        "pin_memory": True,
        "drop_last": True,   # prevents trailing batch of 1 from crashing BatchNorm
    }
    
    # only add these params when using multiprocessing (num_workers > 0)
    if config.dl.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True  # avoids worker respawn per epoch
        train_loader_kwargs["prefetch_factor"] = max(
            1, int(getattr(config.dl, "prefetch_factor", 3))
        )
        train_loader_kwargs["worker_init_fn"] = partial(_worker_init, base_seed=_base_seed)
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    # validation uses platform-default workers (0 on Windows/desktop, 4 on runpod)
    from .platform import get_default_val_workers
    val_workers = get_default_val_workers()
    valid_loader_kwargs: dict = {
        "batch_size": config.dl.batch_size,
        "num_workers": val_workers,
        "pin_memory": True,
    }
    if val_workers > 0:
        valid_loader_kwargs["prefetch_factor"] = max(
            1, int(getattr(config.dl, "prefetch_factor", 3))
        )
        valid_loader_kwargs["persistent_workers"] = True  # prevents zombie accumulation when iter() is cycled (entropy_min)
        valid_loader_kwargs["worker_init_fn"] = partial(_worker_init, base_seed=_base_seed + 9999)
    valid_loader = DataLoader(valid_dataset, **valid_loader_kwargs)

    return train_loader, valid_loader

def _sample_labels(dataset, sample_size):
    """helper function to sample labels from a dataset"""
    labels = []
    dataset_iter = iter(dataset)
    for _ in range(sample_size):
        try:
            # get next item; when dann=True the dataset yields a 4-tuple (block, label, mask, sid)
            batch = next(dataset_iter)
            _, label, mask = batch[0], batch[1], batch[2]
            if mask.sum() > 0:
                labels.append(int(label.item()))
        except StopIteration:
            break
    return labels


def calc_dense_pos_weight(dataset, n_samples=200, clamp=(1.0, 20.0)):
    """pos_weight for dense per-pixel BCE = (neg_px / pos_px) over sampled valid pixels.
    the dataset yields (block, label_map (1,T,T), mask (T,T)); we count ink vs non-ink
    pixels inside the mask. returns a (1,) tensor, clamped to a sane range."""
    pos, tot = 0, 0
    it = iter(dataset)
    for _ in range(n_samples):
        try:
            batch = next(it)
            _, label_map, mask = batch[0], batch[1], batch[2]
        except StopIteration:
            break
        m = (mask > 0)
        if m.sum() <= 0:
            continue
        lm = label_map.squeeze(0) if label_map.dim() == 3 else label_map
        pos += int(((lm > 0.5) & m).sum().item())
        tot += int(m.sum().item())
    if tot == 0 or pos == 0:
        print("[dense] pos_weight fallback -> 1.0 (no ink pixels sampled)")
        return torch.tensor([1.0], dtype=torch.float32)
    p = pos / tot
    pw = float(np.clip((1 - p) / p, clamp[0], clamp[1]))
    print(f"[dense] sampled ink pixel fraction={p:.3f}  pos_weight={pw:.2f}")
    return torch.tensor([pw], dtype=torch.float32)


def calc_class_wgts(train_set, valid_set, scroll_id=None, cache_path=UNIFIED_CACHE_PATH):
    """calculates class weights from dataset samples"""
    cache_key = str(scroll_id) if scroll_id is not None else None
    if cache_key is not None:
        cache = _load_unified_cache(cache_path)
        cached_entry = cache.get(cache_key, {})
        cached = cached_entry.get("class_weight") if isinstance(cached_entry, dict) else None
        if isinstance(cached, dict) and "pos_weight" in cached:
            if cached["pos_weight"] is None:
                print(f"using cached class weight result for scroll {cache_key}: no pos_weight")
                return None
            cached_w = float(cached["pos_weight"])
            print(f"using cached pos_weight for scroll {cache_key}: {cached_w:.2f}")
            return torch.tensor([cached_w], dtype=torch.float32)

    print("sampling datasets to calculate average class weights")
    sample_size = 2500
    
    # sample from both training and validation sets for a representative distribution
    labels_a = _sample_labels(train_set, sample_size * 2)
    labels_b = _sample_labels(valid_set, sample_size)
    all_labels = labels_a + labels_b

    if not all_labels:
        print("warning: no samples found for class weight calculation")
        return None

    # count positive and negative samples
    counts = Counter(all_labels)
    print(f"label distribution (from {len(all_labels)} samples): {counts}")

    # calculate weight for the positive class
    if counts.get(0, 0) > 0 and counts.get(1, 0) > 0:
        pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32)
        print(f"using average pos_weight: {pos_weight.item():.2f}")

        if cache_key is not None:
            cache = _load_unified_cache(cache_path)
            entry = cache.get(cache_key, {})
            if not isinstance(entry, dict):
                entry = {}
            entry["class_weight"] = {
                "pos_weight": float(pos_weight.item()),
                "counts": {"0": int(counts[0]), "1": int(counts[1])},
                "samples": int(len(all_labels)),
            }
            cache[cache_key] = entry
            _save_unified_cache(cache, cache_path)
            print(f"saved pos_weight cache for scroll {cache_key} to {cache_path}")

        return pos_weight
    
    print("warning: only one class present in sampled data")

    if cache_key is not None:
        cache = _load_unified_cache(cache_path)
        entry = cache.get(cache_key, {})
        if not isinstance(entry, dict):
            entry = {}
        entry["class_weight"] = {
            "pos_weight": None,
            "counts": {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))},
            "samples": int(len(all_labels)),
        }
        cache[cache_key] = entry
        _save_unified_cache(cache, cache_path)

    return None


def _count_supervised_units(dataset, n_samples=3000):
    """(pos, neg) over the SUPERVISED units of a dataset, mode-aware:
    single-tile -> one tile label per window (gated by the window mask);
    multitile   -> per-sub-tile labels, counting only sub-tiles the mask keeps.
    reads labels/mask (memmapped numpy) only -- no zarr/image reads."""
    coords = list(getattr(dataset, "block_coords", []))
    if not coords:
        return 0, 0
    if len(coords) > n_samples:
        sel = np.random.choice(len(coords), n_samples, replace=False)
        coords = [coords[i] for i in sel]
    pos = neg = 0
    for (_d, y, x) in coords:
        lbl = np.asarray(dataset._fetch_label(y, x)).reshape(-1)
        msk = np.asarray(dataset._fetch_mask(y, x)).reshape(-1)
        if msk.shape[0] == lbl.shape[0]:      # multitile: per-unit validity mask
            keep = msk > 0
            units = lbl[keep]
        else:                                  # single-tile: window-level gate
            if msk.sum() <= 0:
                continue
            units = lbl
        pos += int((units > 0.5).sum())
        neg += int((units <= 0.5).sum())
    return pos, neg


def get_tile_pos_weight(train_children, config, cache_path=UNIFIED_CACHE_PATH, clamp=(1.0, 20.0)):
    """mode-aware pos_weight (neg/pos over supervised units), cached per scroll + mode signature.
    the signature captures mode (single/multitile), sub-tile grid, pos_only, and the inklabel dir,
    so switching any of them recomputes instead of reusing a stale value. aggregates the per-scroll
    counts into ONE global pos_weight for the loss. returns a (1,) tensor or None."""
    tot_pos = tot_neg = 0
    for ds in train_children:
        mt = bool(getattr(ds, "_mt", False))
        ink = os.path.basename(str(getattr(config.data, "inklabel_dir", "")).rstrip("/"))
        if mt:
            gate = "_ringgate" if ds._mt_ring_gate and not ds._mt_pos_only else ""
            if bool(getattr(config.data, "ring_from_inklabel_dir", False)):
                gate += "_ringsrc"
            sig = f"mt_ringtargets_v2_s{ds._mt_sub}_g{ds._mt_grid}_pos{int(ds._mt_pos_only)}{gate}_{ink}"
            key = f"class_weight_multitile_s{ds._mt_sub}_g{ds._mt_grid}_pos{int(ds._mt_pos_only)}{gate}"
        else:
            sig = f"single_{ink}"
            key = "class_weight"
        if not bool(getattr(config.data, "simple_split", True)):
            sig += "_manual_split"
        sid = str(getattr(ds, "scroll_id", 0))
        cache = _load_unified_cache(cache_path)
        entry = cache.get(sid, {})
        if not isinstance(entry, dict):
            entry = {}
        cached = entry.get(key)
        if isinstance(cached, dict) and cached.get("sig") == sig and "counts" in cached:
            p = int(cached["counts"].get("1", 0)); n = int(cached["counts"].get("0", 0))
        else:
            p, n = _count_supervised_units(ds)
            pw = float(np.clip(n / max(p, 1), *clamp)) if p > 0 else None
            entry[key] = {"pos_weight": pw, "counts": {"0": int(n), "1": int(p)}, "sig": sig}
            cache[sid] = entry
            _save_unified_cache(cache, cache_path)
            print(f"[pos_weight] {key} scroll {sid} ({sig}): pos={p} neg={n} pw={pw}")
        tot_pos += p; tot_neg += n
    if tot_pos <= 0:
        print("[pos_weight] no positives sampled -> pos_weight=None")
        return None
    pw = float(np.clip(tot_neg / tot_pos, *clamp))
    print(f"[pos_weight] aggregate over {len(train_children)} scroll(s): pos={tot_pos} neg={tot_neg} -> pos_weight={pw:.2f}")
    return torch.tensor([pw], dtype=torch.float32)