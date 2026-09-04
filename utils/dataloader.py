import numpy as np
import torch
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
from functools import partial
from typing import Iterator
from .config import Config
import json
from tqdm import tqdm

UNIFIED_CACHE_PATH = "./norm_cache.json"


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


def _mmap_scratch_dir():
    """scratch directory for memmap backing files (override via VESUVIUS_MMAP_DIR)"""
    d = os.environ.get("VESUVIUS_MMAP_DIR") or os.path.join(tempfile.gettempdir(), "vesuvius_mmap")
    os.makedirs(d, exist_ok=True)
    return d


def _write_memmap(arr, pack_bits=False, original_shape=None):
    """persist a (binary uint8) array to a unique .npy and return its path.
    if pack_bits=True, packs to 1 bit/pixel (8x smaller) and saves shape separately."""
    path = os.path.join(_mmap_scratch_dir(), f"mm_{os.getpid()}_{uuid.uuid4().hex}.npy")
    if pack_bits:
        # pack to 1 bit per pixel (8x compression)
        packed = np.packbits(arr.ravel())
        np.save(path, packed)
        # save shape separately so we can unpack correctly
        shape_path = path.replace('.npy', '_shape.npy')
        np.save(shape_path, np.array(original_shape or arr.shape, dtype=np.int32))
        _MMAP_FILES.append(shape_path)
    else:
        np.save(path, np.ascontiguousarray(arr))
    _MMAP_FILES.append(path)
    return path


def _cleanup_mmap_files():
    """remove memmap files at interpreter exit, but only in the creating process.
    on windows a still-mapped file can refuse deletion; that is non-fatal (the
    files live in temp), so failures are swallowed."""
    if os.getpid() != _MMAP_OWNER_PID:
        return
    for p in _MMAP_FILES:
        try:
            os.remove(p)
        except OSError:
            pass


atexit.register(_cleanup_mmap_files)


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
        if random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])
            block = np.rot90(block, k=k, axes=(1, 2)).copy()
            label = self._rotate_target(label, k)
            mask = self._rotate_target(mask, k)
            component_ids = self._rotate_target(component_ids, k)
            target_offset = self._rotate_offset(target_offset, k)
        if random.random() < self.flip_prob:
            axis = random.choice([1, 2])
            block = np.flip(block, axis=axis).copy()
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
            block = self._apply_cutout(block)
        if self.depth_mask_prob > 0:
            block = self._apply_depth_mask(block)
        if self.depth_warp_prob > 0 and random.random() < self.depth_warp_prob:
            block = self._apply_smooth_depth_warp(block)
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
        out = np.empty_like(block)
        for d in range(block.shape[0]):
            out[d] = map_coordinates(block[d], coords_flat, order=1, mode='reflect').reshape(H, W)
        return out

    def _apply_cutout(self, block):
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
                cy0 = (H - protected) // 2
                cx0 = (W - protected) // 2
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
        """independently zero out depth slices with depth_mask_prob each.
        forces robustness to missing depth planes."""
        out = block.copy()
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
    def __init__(self, volume, mask, labels, config, x_range, y_range, norm_stats, shuffle=True, soft_labels=None, scroll_id=None, domain_id=None, scroll_mask=None, split_mask=None, character_grid=None):
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
        domain_id: compact 0..N-1 fragment id used by DANN when enabled."""
        self.scroll_id = int(scroll_id) if scroll_id is not None else 0
        self.domain_id = int(domain_id) if domain_id is not None else 0
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

        # store mask/labels as uint8 (binary), not float64. the source arrays are
        # mask/255.0 and labels/255.0 (float64): for the big scroll (13513x17381)
        # that is ~1.88 GB EACH. when DataLoader spawns workers on Windows, the whole
        # dataset is pickled to each worker; two float64 full-res arrays (plus a ring
        # mask) exceed the spawn pickle limit -> OSError [Errno 22] / "pickle data was
        # truncated". these arrays are only ever used as binary tests (>0.5, sum>0), so
        # uint8 is exact and 8x smaller, which keeps multiscroll+ring picklable at nw>0.
        mask_u8 = (np.asarray(mask) > 0.5).astype(np.uint8)
        labels_u8 = (np.asarray(labels) > 0.5).astype(np.uint8)

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
            sm_u8 = (np.asarray(scroll_mask) > 0.5).astype(np.uint8)
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
            split_u8 = (np.asarray(split_mask) > 0.5).astype(np.uint8)
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
        self._manual_split = not bool(getattr(config.data, "simple_split", True))
        self._character_metrics = bool(getattr(config.tra, "character_macro_metrics", False))
        self._character_balanced = bool(
            getattr(config.data, "character_balanced_sampling", False)
        ) and self.shuffle
        self._character_grid = character_grid
        self._character_nearest = None
        self._character_pos_coords = {}
        self._character_neg_coords = {}
        self._context_replace_prob = float(getattr(config.dl, "context_replace_prob", 0.0))
        self._context_donor_coords = []

        self.z_start = getattr(self.c.data, "train_d_start", self.c.data.d_start)
        self.z_end   = getattr(self.c.data, "train_d_end",   self.c.data.d_end)
        self.y_start, self.y_end = y_range
        self.x_start, self.x_end = x_range
        
        # pre-calculate all valid block coordinates
        self.block_coords = self._gen_tile_coords()
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
        if self._character_metrics or self._character_balanced:
            if not self._mt or self._character_grid is None:
                raise ValueError("character-aware mode requires multitile labels and a character grid")
            self._prepare_character_targets()
        if self._context_replace_prob > 0 and self.shuffle:
            self._prepare_context_donors()

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
                    out[index] += int(self.domain_id) * 1_000_000
        return out

    def _character_balanced_coords(self):
        """pair one positive and one local-ring window per uniformly sampled character."""
        target = self.samples_per_epoch
        coords = []
        while len(coords) < target:
            character_ids = list(self._character_ids)
            np.random.shuffle(character_ids)
            for component_id in character_ids:
                positive = self._character_pos_coords[component_id]
                negative = self._character_neg_coords[component_id]
                coords.append(positive[np.random.randint(len(positive))])
                if len(coords) >= target:
                    break
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
        # never pickle an open zarr handle to a spawned worker (unpicklable on Windows,
        # OSError [Errno 22]). the main process may now hold one (vol opens lazily in the
        # main process too, for num_workers=0 validation); drop it so workers reopen via
        # _zarr_path. harmless when it was already None.
        if state.get("_zarr_path") is not None:
            state["_worker_vol"] = None
        return state


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
        for d in range(0, z_range_size, z_step):
            if self.z_start + d + self.depth > self.z_end: continue
            for y in range(0, y_range_size, xy_step):
                for x in range(0, x_range_size, xy_step):
                    # multitile training windows: keep ONLY windows whose 32x32 center overlaps
                    # the ring (training) mask. pos_only later keeps actual ink cells and true
                    # ring-negative cells while leaving the exclusion gap unsupervised.
                    if self._mt and (self.shuffle or self._manual_split):
                        if (self._mt_window_touches_ring(y, x)
                                and self._mt_window_touches_split(y, x)):
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
        mean, std, g_min, g_max = self.norm_stats
        if std == 0:
            return block.astype(np.float32, copy=False)
        
        # z score normalization followed by scaling to [0, 1]
        norm_block = (block.astype(np.float32, copy=False) - mean) / std
        norm_block = (norm_block - g_min) / (g_max - g_min)
        # ensure dtype and contiguity
        return np.ascontiguousarray(np.clip(norm_block, 0, 1).astype(np.float32, copy=False))

    def _fetch_block(self, z_off, y_off, x_off, allow_jitter=True):
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
                dj = random.randint(-max_dj, max_dj) \
                    if max_dj > 0 and self.shuffle and allow_jitter else 0
                block = self._read_ctx_block(z + dj, self.depth, y - pad - jy, x - pad - jx, ctx)
            else:
                block = np.array(self.vol[z:z+self.depth, y:y+tile, x:x+tile]).astype(np.float32)
        except Exception:
            # any read error (OSError, corrupt chunk, zarr internal error) — return zeros
            block = np.zeros((self.depth, sp, sp), dtype=np.float32)

        # guard: zarr can silently return wrong shape on Windows under load
        if block.shape != (self.depth, sp, sp):
            block = np.zeros((self.depth, sp, sp), dtype=np.float32)

        return self._normalize_block(block), target_offset

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
                src = np.array(vol[z:z+ndepth, ys:ye, xs:xe]).astype(np.float32)
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
                if np.any(lbl[ysc:yec, xsc:xec] > 0.5):
                    out[iy * n + ix] = 1.0
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
        supervision_mask = self.mask if self._mt_pos_only else None
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
                if (np.all(m[ys:ye, xs:xe] > 0)
                        and (target_mask is None or np.all(target_mask[ys:ye, xs:xe] > 0))
                        and (supervision_mask is None
                             or np.all(supervision_mask[ys:ye, xs:xe] > 0))):
                    idx = iy * n + ix
                    if self._mt_pos_only and lbl[idx] <= 0:
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
        if self._mt_pos_only and (lbl * out).sum() > 0:
            out = out * lbl
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
        block, target_offset = self._fetch_block(z_off, y_off, x_off)
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
        if (
            self.apply_transforms
            and self._context_donor_coords
            and random.random() < self._context_replace_prob
        ):
            donor_y, donor_x = self._context_donor_coords[
                random.randrange(len(self._context_donor_coords))
            ]
            donor, _ = self._fetch_block(
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
        
        # enforce contiguity and dtype before converting to torch to avoid negative strides
        block = np.ascontiguousarray(block, dtype=np.float32)
            
        # convert to tensor for the model
        block_tensor = torch.from_numpy(block).unsqueeze(0)
        
        self.current_idx += 1
        with_domain = bool(getattr(self.c.tra, "dann", False)) or bool(
            getattr(self.c.tra, "supcon_cross_frag", False)
        )
        if with_domain and component_ids is not None and target_offset_tensor is not None:
            return (
                block_tensor,
                label,
                mask,
                torch.tensor(self.domain_id, dtype=torch.long),
                component_ids,
                target_offset_tensor,
            )
        if with_domain and component_ids is not None:
            return block_tensor, label, mask, torch.tensor(self.domain_id, dtype=torch.long), component_ids
        if with_domain and target_offset_tensor is not None:
            return block_tensor, label, mask, torch.tensor(self.domain_id, dtype=torch.long), target_offset_tensor
        if with_domain:
            return block_tensor, label, mask, torch.tensor(self.domain_id, dtype=torch.long)
        if component_ids is not None and target_offset_tensor is not None:
            return block_tensor, label, mask, component_ids, target_offset_tensor
        if component_ids is not None:
            return block_tensor, label, mask, component_ids
        if target_offset_tensor is not None:
            return block_tensor, label, mask, target_offset_tensor
        return block_tensor, label, mask


class MultiScrollIterableDataset(IterableDataset):
    """merges several InkVolumeDatasets into one stream so a single epoch sees
    tiles from every scroll fragment interleaved (batches are integrated, not
    alternated). each child handles its own per-worker sharding, so worker N
    receives shard N of every scroll."""
    def __init__(self, datasets, balance_scrolls=False):
        super().__init__()
        self.datasets = list(datasets)
        self.balance_scrolls = bool(balance_scrolls)
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
        mask_arr = np.asarray(self._dm.mask)
        with_dann = bool(getattr(c.tra, "dann", False)) or bool(getattr(c.tra, "supcon_cross_frag", False))
        # multitile mode: dot labels/masks must match InkVolumeDataset's [grid²] shape
        mt = bool(getattr(c.model, "multitile", False))
        mt_grid = max(1, int(getattr(c.model, "multitile_grid", 4))) if mt else 1
        mt_sub  = max(1, int(getattr(c.model, "multitile_subtile", 8))) if mt else T

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
            if mt:
                # multitile: label = all-1 [grid²] (dot = confirmed ink); mask = all-1 [grid²]
                # (all sub-tiles are fully positive; no boundary ambiguity at a dot location)
                n2 = mt_grid * mt_grid
                label_t = torch.ones(n2, dtype=torch.float32)
                mask_t  = torch.ones(n2, dtype=torch.float32)
            else:
                mask_tile = mask_arr[y0:y0 + T, x0:x0 + T].astype(np.float32)
                label_t = torch.tensor([1.0])
                mask_t  = torch.from_numpy(mask_tile)
            with_characters = bool(getattr(c.tra, "character_macro_metrics", False))
            with_offset = bool(getattr(c.data, "target_aware_ctx_jitter", False))
            extras = []
            if with_dann:
                extras.append(torch.tensor(domain_id, dtype=torch.long))
            if with_characters:
                extras.append(torch.zeros_like(label_t, dtype=torch.long))
            if with_offset:
                extras.append(torch.zeros(2, dtype=torch.long))
            yield (block_t, label_t, mask_t, *extras)


class DataManager:
    """manages data loading, splitting, and normalization"""
    def __init__(self, config: Config, scroll_id=None, domain_id: int = 0):
        """initializes the data manager.
        scroll_id: which scroll fragment to load; defaults to the first configured scroll.
        passing it explicitly lets the trainer build one manager per fragment."""
        self.c = config
        if scroll_id is None:
            scroll_id = config.data.scrolls[0].scroll_id
        self.scroll_id = int(scroll_id)
        self.domain_id = int(domain_id)

        # load raw data and define splits
        self.vol, self.mask, self.labels, self.train_x, self.valid_x, self.y_range = self._load_raw_data()

        # get or compute normalization statistics
        self.norm_stats = self._get_or_compute_norm()

    def _load_raw_data(self):
        """loads raw zarr data and metadata"""
        # open the zarr volume in read-only mode
        zarr_dir = os.path.join(self.c.data.zarr_path, f"{self.scroll_id}.zarr")
        vol = zarr.open(zarr_dir, mode='r')

        # load labels and mask, and normalize to [0, 1]
        lbl_dir = getattr(self.c.data, 'inklabel_dir', './eroded_inklabels')
        labels = imread_gray(f"{lbl_dir}/{self.scroll_id}.png")

        mask = imread_gray(f"./masks/{self.scroll_id}.png")

        if labels is None:
            raise FileNotFoundError(f"labels not found for scroll {self.scroll_id}")
        if mask is None:
            raise FileNotFoundError(f"mask not found for scroll {self.scroll_id}")

        labels = (labels.astype(np.float32) / 255.0)  # force float32 to avoid float64 OOM
        mask = mask / 255.0

        # define the working area and split for train/validation.
        # optional region crop (fractions of the full frame) trims the usable area so a run
        # can train on only a sub-region. then the train/valid split is applied along the
        # configured axis: 'x' = legacy vertical (left train / right valid), 'y' = horizontal
        # (top train / bottom valid). all boundaries are tile-aligned so the eval pred-map and
        # label-map shapes stay consistent.
        T = int(self.c.data.tile_size)
        H, W = int(vol.shape[1]), int(vol.shape[2])

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
        if not bool(getattr(self.c.data, "simple_split", True)):
            train_mask_dir = str(getattr(self.c.data, "train_mask_dir", "./train_masks"))
            train_mask_path = os.path.join(train_mask_dir, f"{self.scroll_id}.png")
            manual_mask = imread_gray(train_mask_path)
            if manual_mask is None:
                raise FileNotFoundError(
                    f"manual train mask not found for scroll {self.scroll_id}: {train_mask_path}"
                )
            if manual_mask.shape != mask.shape or manual_mask.shape != labels.shape:
                raise ValueError(
                    f"manual train mask shape {manual_mask.shape} does not match scroll mask "
                    f"{mask.shape} and labels {labels.shape} for scroll {self.scroll_id}"
                )
            self.manual_train_mask = (manual_mask > 0).astype(np.uint8)
            frac_train = float(self.manual_train_mask.mean())
            print(
                f"[split] scroll {self.scroll_id}: manual mask={train_mask_path} "
                f"train_pixels={100.0 * frac_train:.1f}%"
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

    def get_datasets(self):
        """creates train and validation datasets.
        for split_axis='y' (horizontal): train=top rows, valid=bottom rows, x fully shared.
        for split_axis='x' (legacy vertical): train=left cols, valid=right cols, y fully shared.
        when simple_split=False, both datasets span the full cropped frame and the binary
        train_masks/<scroll_id>.png partitions the existing ring supervision into disjoint
        train and validation masks.
        InkVolumeDataset takes (x_range, y_range); we feed the split range on the split axis and
        the shared range on the other axis."""
        supervision_mask = self._make_ring_mask() if getattr(self.c.data, 'ring_negatives', False) else self.mask
        manual_split = not bool(getattr(self.c.data, "simple_split", True))
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
            if character_grid is not None:
                character_grid = self._exclude_characters_crossing_split(
                    character_grid,
                    assignment,
                    split_unit,
                )
            eligible = np.asarray(supervision_mask) > 0.5
            if getattr(self.c.model, "multitile", False):
                # preserve the legacy ring-window gate and partition only the emitted targets
                train_mask = valid_mask = supervision_mask
                train_split_mask = assignment
                valid_split_mask = (assignment == 0).astype(np.uint8)
            else:
                train_mask = (eligible & (assignment > 0)).astype(np.uint8)
                valid_mask = (eligible & (assignment == 0)).astype(np.uint8)
                train_split_mask = valid_split_mask = None
            print(
                f"[manual-split] scroll {self.scroll_id}: unit={split_unit}px "
                f"train_ring={int((eligible & (assignment > 0)).sum())}px "
                f"valid_ring={int((eligible & (assignment == 0)).sum())}px"
            )
        else:
            train_mask = supervision_mask
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
            scroll_mask=scroll_mask_arg,
            split_mask=train_split_mask,
            character_grid=character_grid,
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
    def _exclude_characters_crossing_split(character_grid, assignment, unit):
        """exclude components crossing the fixed manual split from character-aware analysis."""
        unit = max(1, int(unit))
        grid_h, grid_w = character_grid.shape
        cells = assignment[:grid_h * unit, :grid_w * unit].reshape(
            grid_h, unit, grid_w, unit
        ).all(axis=(1, 3))
        crossing = []
        for component_id in np.unique(character_grid):
            if component_id <= 0:
                continue
            member = character_grid == component_id
            values = cells[member]
            if values.any() and not values.all():
                crossing.append(int(component_id))
        out = character_grid.copy()
        if crossing:
            out[np.isin(out, crossing)] = 0
            print(
                f"[character-split] excluded {len(crossing)} character(s) crossing the fixed split"
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
        if ring_source == 'original':
            orig_path = f"./inklabels/{self.scroll_id}.png"
            orig_img = imread_gray(orig_path)
            if orig_img is not None:
                orig_img = (orig_img / 255.0)[:h, :w]
                ring_labels = orig_img
            else:
                print(f"[ring] original inklabels not found at {orig_path}, falling back to eroded")
                ring_labels = labels_crop
        else:
            # 'eroded' and 'closed' both build the ring off the (hand-cleaned) eroded map
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
            print(f"[ring] closed(base=eroded): CLOSE_R={CLOSE_R} GAP_R={GAP_R} exclusion_tiles={ink_tile_ring.sum()}")

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
        train_loader_kwargs["prefetch_factor"] = 3
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
        valid_loader_kwargs["prefetch_factor"] = 2
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
            sig = f"mt_ringtargets_v2_s{ds._mt_sub}_g{ds._mt_grid}_pos{int(ds._mt_pos_only)}_{ink}"
            key = f"class_weight_multitile_s{ds._mt_sub}_g{ds._mt_grid}_pos{int(ds._mt_pos_only)}"
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