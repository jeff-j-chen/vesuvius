import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from collections import Counter
import zarr
import cv2
import random
import os
import uuid
import atexit
import tempfile
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


def _write_memmap(arr):
    """persist a (binary uint8) array to a unique .npy and return its path"""
    path = os.path.join(_mmap_scratch_dir(), f"mm_{os.getpid()}_{uuid.uuid4().hex}.npy")
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
        self.channel_mixing_prob = float(getattr(config.dl, "channel_mixing_prob", 0.25))
        self.rotation_prob = float(getattr(config.dl, "rotation_prob", 0.25))
        self.flip_prob = float(getattr(config.dl, "flip_prob", 0.25))
        self.noise_prob = float(getattr(config.dl, "noise_prob", 0.30))
        self.brightness_prob = float(getattr(config.dl, "brightness_prob", 0.50))
        self.contrast_prob = float(getattr(config.dl, "contrast_prob", 0.50))

    def __call__(self, block):
        """applies a random sequence of transforms to a block"""
        # each transform is applied with a certain probability
        if random.random() < self.channel_mixing_prob:
            block = self._apply_channel_mixing(block)
        if random.random() < self.rotation_prob:
            block = self._apply_rotation(block)
        if random.random() < self.flip_prob:
            block = self._apply_flip(block)
        if random.random() < self.noise_prob:
            block = self._apply_gaussian_noise(block)
        if random.random() < self.brightness_prob:
            block = self._apply_brightness_adjustment(block)
        if random.random() < self.contrast_prob:
            block = self._apply_contrast_adjustment(block)
        # ensure the final result is contiguous to avoid negative strides
        return np.ascontiguousarray(block)

    def _apply_channel_mixing(self, block):
        """mixes the order of the depth channels"""
        indices = np.random.permutation(block.shape[0])
        mixed = block[indices]
        # guard against any non contiguous result from advanced indexing
        return np.ascontiguousarray(mixed)
    
    def _apply_brightness_adjustment(self, block):
        """applies brightness adjustment to each channel independently"""
        factors = np.random.uniform(0.85, 1.15, size=(block.shape[0], 1, 1))
        return np.clip(block * factors, 0, 1)
    
    def _apply_contrast_adjustment(self, block):
        """applies contrast adjustment to each channel independently"""
        adj_block = block.copy()
        for i in range(block.shape[0]):
            channel = block[i]
            factor = random.uniform(0.85, 1.15)
            mean = np.mean(channel)
            adj_block[i] = np.clip((channel - mean) * factor + mean, 0, 1)
        return adj_block
    
    def _apply_gaussian_noise(self, block):
        """applies gaussian noise to each channel independently"""
        std = random.uniform(0.005, 0.015)
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
    def __init__(self, volume, mask, labels, config, x_range, y_range, norm_stats, shuffle=True, soft_labels=None):
        """initializes the dataset.
        soft_labels: optional full-res float [0,1] ink-probability map (expanded+blurred
        eroded labels). when given AND config.data.dense_soft_labels is set, the dense
        per-pixel target uses these CONTINUOUS values instead of the hard binary label —
        calibrated soft edges (see _fetch/__next__ dense path). stored as uint8 0-255."""
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
        if getattr(config.data, "mask_memmap", False):
            self._mask_path = _write_memmap(mask_u8)
            self._labels_path = _write_memmap(labels_u8)
            self._mask_arr = None
            self._labels_arr = None
        else:
            self._mask_path = None
            self._labels_path = None
            self._mask_arr = mask_u8
            self._labels_arr = labels_u8
        # optional soft labels (continuous ink probability, 0-255 uint8). stored parallel
        # to the hard labels; used only by the dense target path when dense_soft_labels is on.
        self._soft_path = None
        self._soft_arr = None
        if soft_labels is not None:
            soft_u8 = np.clip(np.asarray(soft_labels) * 255.0, 0, 255).astype(np.uint8)
            if getattr(config.data, "mask_memmap", False):
                self._soft_path = _write_memmap(soft_u8)
            else:
                self._soft_arr = soft_u8
        self.c = config
        self.tile_size = config.data.tile_size
        self.depth = config.data.depth
        self.apply_transforms = False # controlled by trainer
        self.shuffle = shuffle
        self.norm_stats = norm_stats
        self.transform = Transform(config)

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

    @property
    def mask(self):
        """binary uint8 mask; a real array unless memmapped, in which case the
        read-only memmap is opened lazily per process (main or worker)."""
        if self._mask_arr is None and self._mask_path is not None:
            self._mask_arr = np.load(self._mask_path, mmap_mode='r')
        return self._mask_arr

    @property
    def labels(self):
        """binary uint8 labels; lazily memmapped per process when memmap is enabled."""
        if self._labels_arr is None and self._labels_path is not None:
            self._labels_arr = np.load(self._labels_path, mmap_mode='r')
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

        # iterate over the volume with specified step sizes to generate coordinates
        for d in range(0, z_range_size, z_step):
            if self.z_start + d + self.depth > self.z_end: continue
            for y in range(0, y_range_size, self.tile_size):
                for x in range(0, x_range_size, self.tile_size):
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
        """normalizes a block using pre computed global stats"""
        mean, std, g_min, g_max = self.norm_stats
        if std == 0:
            return block.astype(np.float32, copy=False)
        
        # z score normalization followed by scaling to [0, 1]
        norm_block = (block.astype(np.float32, copy=False) - mean) / std
        norm_block = (norm_block - g_min) / (g_max - g_min)
        # ensure dtype and contiguity
        return np.ascontiguousarray(np.clip(norm_block, 0, 1).astype(np.float32, copy=False))

    def _fetch_block_at_z(self, z_abs, y_off, x_off):
        """fetch block starting at absolute z position (used for soft label flanking bands)"""
        y = self.y_start + y_off
        x = self.x_start + x_off
        tile = self.tile_size
        mode = getattr(self.c.data, "input_mode", "single")
        # flanking bands always return a single-band block of self.depth slices
        try:
            block = np.array(self.vol[z_abs:z_abs+self.depth, y:y+tile, x:x+tile]).astype(np.float32)
        except Exception:
            block = np.zeros((self.depth, tile, tile), dtype=np.float32)
        if block.shape != (self.depth, tile, tile):
            block = np.zeros((self.depth, tile, tile), dtype=np.float32)
        # for diff mode, compute flanking - pre (flanking IS the reference, so use zeros)
        if mode == "diff":
            block = np.zeros_like(block)  # no ink expected in flanking band, diff ≈ 0
        return self._normalize_block(block)

    def _fetch_block(self, z_off, y_off, x_off):
        """fetches and normalizes a block from zarr volume.

        input_mode controls the returned tensor shape:
          single: (8, 32, 32)  — current behavior
          diff:   (8, 32, 32)  — ink_band - pre_band (differential absorption)
          triple: (24, 32, 32) — concat(pre_band, ink_band, post_band)
        """
        z = self.z_start + z_off
        y = self.y_start + y_off
        x = self.x_start + x_off
        tile = self.tile_size
        mode = getattr(self.c.data, "input_mode", "single")

        expected_d = {"triple": self.depth * 3, "double": self.depth * 2,
                      "fulldepth": int(getattr(self.vol, 'shape', [64])[0])}.get(mode, self.depth)
        try:
            if mode == "diff":
                ink  = np.array(self.vol[z:z+self.depth, y:y+tile, x:x+tile]).astype(np.float32)
                pre_z = getattr(self.c.data, "pre_band_start", 20)
                pre  = np.array(self.vol[pre_z:pre_z+self.depth, y:y+tile, x:x+tile]).astype(np.float32)
                block = np.clip(ink - pre, 0, None)  # take positive part of the delta
            elif mode == "triple":
                pre_z  = getattr(self.c.data, "pre_band_start", 20)
                post_z = getattr(self.c.data, "post_band_start", 40)
                pre    = np.array(self.vol[pre_z:pre_z+self.depth,  y:y+tile, x:x+tile]).astype(np.float32)
                ink    = np.array(self.vol[z:z+self.depth,           y:y+tile, x:x+tile]).astype(np.float32)
                post   = np.array(self.vol[post_z:post_z+self.depth, y:y+tile, x:x+tile]).astype(np.float32)
                block  = np.concatenate([pre, ink, post], axis=0)
            elif mode == "double":
                pre_z = getattr(self.c.data, "pre_band_start", 20)
                ink   = np.array(self.vol[z:z+self.depth,            y:y+tile, x:x+tile]).astype(np.float32)
                pre   = np.array(self.vol[pre_z:pre_z+self.depth,   y:y+tile, x:x+tile]).astype(np.float32)
                block = np.concatenate([ink, pre], axis=0)  # (16, H, W) for siamese
            elif mode == "fulldepth":
                full_d = int(self.vol.shape[0])
                block = np.array(self.vol[0:full_d, y:y+tile, x:x+tile]).astype(np.float32)
            else:
                block = np.array(self.vol[z:z+self.depth, y:y+tile, x:x+tile]).astype(np.float32)
        except Exception:
            # any read error (OSError, corrupt chunk, zarr internal error) — return zeros
            block = np.zeros((expected_d, tile, tile), dtype=np.float32)

        # guard: zarr can silently return wrong shape on Windows under load
        if block.shape != (expected_d, tile, tile):
            block = np.zeros((expected_d, tile, tile), dtype=np.float32)

        return self._normalize_block(block)

    def _fetch_label(self, y_off, x_off, soft_override: float = -1.0):
        """fetches a label tile; soft_override replaces ink label if >= 0"""
        y = self.y_start + y_off
        x = self.x_start + x_off
        label_tile = self.labels[y:y+self.tile_size, x:x+self.tile_size]
        has_ink = bool(np.any(label_tile > 0.5))
        if has_ink and soft_override >= 0:
            return torch.tensor([soft_override], dtype=torch.float32)
        return torch.tensor([float(has_ink)], dtype=torch.float32)

    def _fetch_mask(self, y_off, x_off):
        """fetches a mask tile"""
        y = self.y_start + y_off
        x = self.x_start + x_off
        
        # slice the mask tile
        mask_tile = self.mask[y:y+self.tile_size, x:x+self.tile_size]
        return torch.tensor(mask_tile, dtype=torch.float32)

    def __iter__(self) -> Iterator:
        """sets up the iterator for an epoch"""
        shuffled_coords = self.block_coords.copy()
        if self.shuffle:
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

        # dense per-pixel supervision: emit the full (1,T,T) ink-label MAP instead of a
        # single scalar tile label. this is the switch away from binary tile labels — the
        # trainer applies per-pixel masked BCE against this map. soft/flanking label logic
        # is bypassed (it is a tile-scalar concept). no spatial aug on the label (keep off).
        if getattr(self.c.data, "dense_labels", False):
            block = self._fetch_block(z_off, y_off, x_off)
            block = np.ascontiguousarray(block, dtype=np.float32)
            block_tensor = torch.tensor(block, dtype=torch.float32).unsqueeze(0)
            y = self.y_start + y_off
            x = self.x_start + x_off
            soft = self.soft_labels
            if getattr(self.c.data, "dense_soft_labels", False) and soft is not None:
                # continuous target: expanded+blurred ink probability in [0,1]
                lbl = np.asarray(soft[y:y+self.tile_size, x:x+self.tile_size]).astype(np.float32) / 255.0
            else:
                lbl = (np.asarray(self.labels[y:y+self.tile_size, x:x+self.tile_size]) > 0.5).astype(np.float32)
            label_map = torch.tensor(lbl, dtype=torch.float32).unsqueeze(0)
            self.current_idx += 1
            return block_tensor, label_map, mask

        # soft depth label: randomly replace ink-band block with flanking band + soft label
        soft_label_prob  = float(getattr(self.c.data, "soft_label_prob", 0.0))
        soft_label_value = float(getattr(self.c.data, "soft_label_value", 0.3))
        use_soft = (soft_label_prob > 0 and random.random() < soft_label_prob)

        if use_soft:
            # pick pre or post band randomly, fetch from there
            if random.random() < 0.5:
                flanking_z = getattr(self.c.data, "pre_band_start", 20)
            else:
                flanking_z = getattr(self.c.data, "post_band_start", 40)
            flanking_off = flanking_z - self.z_start  # adjust to relative offset
            # clamp in case of underflow; use absolute z in _fetch_block via override
            block = self._fetch_block_at_z(flanking_z, y_off, x_off)
            label = self._fetch_label(y_off, x_off, soft_override=soft_label_value)
        else:
            block = self._fetch_block(z_off, y_off, x_off)
            label = self._fetch_label(y_off, x_off)
        
        # apply transforms if enabled
        if self.apply_transforms:
            block = self.transform(block)
        
        # enforce contiguity and dtype before converting to torch to avoid negative strides
        block = np.ascontiguousarray(block, dtype=np.float32)
            
        # convert to tensor for the model
        block_tensor = torch.tensor(block, dtype=torch.float32).unsqueeze(0)
        
        self.current_idx += 1
        return block_tensor, label, mask


class MultiScrollIterableDataset(IterableDataset):
    """merges several InkVolumeDatasets into one stream so a single epoch sees
    tiles from every scroll fragment interleaved (batches are integrated, not
    alternated). each child handles its own per-worker sharding, so worker N
    receives shard N of every scroll."""
    def __init__(self, datasets):
        super().__init__()
        self.datasets = list(datasets)
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
        # build child iterators (each shards itself by worker), then randomly
        # interleave samples until every child is exhausted
        iters = [iter(d) for d in self.datasets]
        active = list(range(len(iters)))
        while active:
            i = random.choice(active)
            try:
                yield next(iters[i])
            except StopIteration:
                active.remove(i)


class DataManager:
    """manages data loading, splitting, and normalization"""
    def __init__(self, config: Config, scroll_id=None):
        """initializes the data manager.
        scroll_id: which scroll fragment to load; defaults to config.data.tra_scroll_id.
        passing it explicitly lets the trainer build one manager per fragment."""
        self.c = config
        self.scroll_id = int(scroll_id) if scroll_id is not None else int(config.data.tra_scroll_id)

        # load raw data and define splits
        self.vol, self.mask, self.labels, self.train_x, self.valid_x, self.y_range = self._load_raw_data()

        # get or compute normalization statistics
        self.norm_stats = self._get_or_compute_norm()

    def _load_raw_data(self):
        """loads raw zarr data and metadata"""
        # open the zarr volume in read-only mode
        zarr_dir = os.path.join(self.c.data.zarr_path, f"{self.scroll_id}.zarr")
        vol = zarr.open(zarr_dir, mode='r')

        # optionally preload the entire volume into RAM so all reads are RAM-speed
        # keep the zarr object as fallback for large-scale training
        if getattr(self.c.data, 'preload_to_ram', False):
            est_gb = (vol.shape[0] * vol.shape[1] * vol.shape[2] * 2) / 1e9
            # gate on available RAM: need est_gb + ~2GB headroom
            try:
                import psutil
                free_gb = psutil.virtual_memory().available / 1e9
            except ImportError:
                free_gb = float('inf')  # can't check; proceed and hope for the best
            if free_gb < est_gb + 2.0:
                print(f"[preload] skipping: need {est_gb:.1f} GB but only {free_gb:.1f} GB available")
            else:
                print(f"[preload] loading {est_gb:.2f} GB into RAM ({free_gb:.1f} GB available)...")
                try:
                    vol = vol[:]   # loads full zarr into a numpy array
                    print(f"[preload] done — {vol.nbytes / 1e9:.2f} GB in RAM")
                except Exception as e:
                    print(f"[preload] FAILED ({e}); falling back to streaming zarr reads")
                    # vol stays as the zarr object; workers will lazy-open their own handles
        
        # load labels and mask, and normalize to [0, 1]
        labels = imread_gray(f"./eroded_inklabels/{self.scroll_id}.png")

        mask = imread_gray(f"./masks/{self.scroll_id}.png")

        if labels is None:
            raise FileNotFoundError(f"labels not found for scroll {self.scroll_id}")
        if mask is None:
            raise FileNotFoundError(f"mask not found for scroll {self.scroll_id}")

        labels = labels / 255.0
        mask = mask / 255.0

        # optional soft labels (continuous ink probability) for dense soft-label training.
        # loaded here so the hard `labels` (used for ring + tile detection) stay unchanged;
        # only the dense per-pixel TARGET uses the soft map. None if the file is absent.
        self.soft_labels = None
        if getattr(self.c.data, "dense_soft_labels", False):
            soft = imread_gray(f"./soft_inklabels/{self.scroll_id}.png")
            if soft is None:
                print(f"[soft_labels] soft_inklabels/{self.scroll_id}.png not found — "
                      f"dense_soft_labels requested but falling back to hard labels")
            else:
                self.soft_labels = (soft / 255.0).astype(np.float32)
                print(f"[soft_labels] loaded soft_inklabels/{self.scroll_id}.png "
                      f"(mean={self.soft_labels.mean():.4f})")

        # define the working area and split for train/validation.
        # optional region crop (fractions of the full frame) trims the usable area so a run
        # can train on only a sub-region. then the train/valid split is applied along the
        # configured axis: 'x' = legacy vertical (left train / right valid), 'y' = horizontal
        # (top train / bottom valid). all boundaries are tile-aligned so the eval pred-map and
        # label-map shapes stay consistent.
        T = int(self.c.data.tile_size)
        H, W = int(vol.shape[1]), int(vol.shape[2])
        cxf = getattr(self.c.data, "crop_x_frac", (0.0, 1.0))
        cyf = getattr(self.c.data, "crop_y_frac", (0.0, 1.0))
        x0 = (int(W * float(cxf[0])) // T) * T
        x1 = (int(W * float(cxf[1])) // T) * T
        y0 = (int(H * float(cyf[0])) // T) * T
        y1 = (int(H * float(cyf[1])) // T) * T
        x1 = max(x1, x0 + T); y1 = max(y1, y0 + T)

        axis = str(getattr(self.c.data, "split_axis", "x")).lower()
        frac = float(getattr(self.c.data, "train_split_frac", 0.75))

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
        """retrieves or computes normalization statistics"""
        seg_id = str(self.scroll_id)

        # first, try to load from cache
        cache = _load_unified_cache()
        stats = cache.get(seg_id)
        if isinstance(stats, dict) and _is_norm_stats(stats):
            print(f"[info] using cached normalization for segment {seg_id}")
            return stats["mean"], stats["std"], stats["min"], stats["max"]

        # if not in cache, compute the statistics
        print(f"[info] computing normalization for segment {seg_id}")
        total_sum, total_sq_sum, total_count = 0.0, 0.0, 0
        
        # first pass: calculate mean and standard deviation
        for z in tqdm(range(self.vol.shape[0])): # type: ignore
            chunk = self.vol[z, :, :]
            mask_chunk = self.mask[:, :]
            valid_pixels = chunk[mask_chunk > 0]
            if valid_pixels.size == 0: continue
            
            total_sum += np.sum(valid_pixels, dtype=np.float64) # type: ignore
            total_sq_sum += np.sum(np.square(valid_pixels, dtype=np.float64), dtype=np.float64) # type: ignore
            total_count += valid_pixels.size # type: ignore

        if total_count == 0: raise ValueError("no valid pixels found")

        mean = total_sum / total_count
        std = np.sqrt((total_sq_sum / total_count) - np.square(mean))
        
        # second pass: calculate min and max of normalized values
        g_min, g_max = float('inf'), float('-inf')
        for z in tqdm(range(self.vol.shape[0])): # type: ignore
            chunk = self.vol[z, :, :]
            mask_chunk = self.mask[:, :]
            valid_pixels = chunk[mask_chunk > 0]
            if valid_pixels.size == 0: continue
            
            norm_pixels = (valid_pixels.astype(np.float64) - mean) / std # type: ignore
            g_min = min(g_min, norm_pixels.min())
            g_max = max(g_max, norm_pixels.max())

        stats = {"mean": mean, "std": std, "min": g_min, "max": g_max}

        # update unified cache file
        cache = _load_unified_cache()
        entry = cache.get(seg_id, {})
        if not isinstance(entry, dict):
            entry = {}
        entry["mean"] = mean
        entry["std"] = std
        entry["min"] = g_min
        entry["max"] = g_max
        cache[seg_id] = entry
        _save_unified_cache(cache)
            
        return mean, std, g_min, g_max

    def get_datasets(self):
        """creates train and validation datasets.
        for split_axis='y' (horizontal): train=top rows, valid=bottom rows, x fully shared.
        for split_axis='x' (legacy vertical): train=left cols, valid=right cols, y fully shared.
        InkVolumeDataset takes (x_range, y_range); we feed the split range on the split axis and
        the shared range on the other axis."""
        train_mask = self._make_ring_mask() if getattr(self.c.data, 'ring_negatives', False) else self.mask
        # when ring_negatives is on, restrict validation to ring tiles too so validation
        # throughput and signal quality match the training distribution. without this,
        # the full valid region (tens of thousands of easy tiles) swamps the validation
        # loop and makes it take 5-10× longer than necessary.
        valid_mask = train_mask if getattr(self.c.data, 'ring_negatives', False) else self.mask
        if getattr(self, "split_axis", "x") == "y":
            train_x, train_y = self.shared_range, self.train_range
            valid_x, valid_y = self.shared_range, self.valid_range
        else:
            train_x, train_y = self.train_range, self.shared_range
            valid_x, valid_y = self.valid_range, self.shared_range
        train_set = InkVolumeDataset(self.vol, train_mask, self.labels, self.c, train_x, train_y, self.norm_stats, shuffle=True, soft_labels=getattr(self, "soft_labels", None))
        valid_set = InkVolumeDataset(self.vol, valid_mask, self.labels, self.c, valid_x, valid_y, self.norm_stats, shuffle=False, soft_labels=getattr(self, "soft_labels", None))
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
        if ring_source in ('original', 'closed'):
            orig_path = f"./inklabels/{self.scroll_id}.png"
            orig_img = imread_gray(orig_path)
            if orig_img is not None:
                orig_img = (orig_img / 255.0)[:h, :w]
                ring_labels = orig_img
            else:
                print(f"[ring] original inklabels not found at {orig_path}, falling back to eroded")
                ring_labels = labels_crop
        else:
            ring_labels = labels_crop  # 'eroded' — old behavior

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

        # for 'closed': close letter holes then add explicit air gap before ring
        if ring_source == 'closed':
            # stage 1: close interior holes in letters (mild closing)
            CLOSE_R = 3  # tiles
            k_close = 2 * CLOSE_R + 1
            kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
            ink_tile_ring = cv2.erode(cv2.dilate(ink_tile_ring, kern_close), kern_close) & mask_tile
            # stage 2: dilate closed region by GAP_R → exclusion zone; ring starts outside this
            # GAP_R is searched dynamically like other ring sources
            # we store it in ink_tile_ring as the "exclusion zone" for the binary search below
            GAP_R = 3  # tiles of air gap between ink edge and ring start
            k_gap = 2 * GAP_R + 1
            kern_gap = cv2.getStructuringElement(cv2.MORPH_RECT, (k_gap, k_gap))
            ink_tile_ring = cv2.dilate(ink_tile_ring, kern_gap) & mask_tile
            print(f"[ring] closed: CLOSE_R={CLOSE_R} GAP_R={GAP_R} exclusion_tiles={ink_tile_ring.sum()}")

        ink_count = int(ink_tile_eroded.sum())
        if ink_count == 0:
            return self.mask

        # dilate ring_labels tile map until ring count >= eroded ink count
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

def get_dataloaders(train_dataset, valid_dataset, config: Config):
    """creates dataloader objects from datasets"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dl.batch_size,
        num_workers=config.dl.num_workers,
        pin_memory=True,
        persistent_workers=config.dl.num_workers > 0,
        prefetch_factor=2 if config.dl.num_workers > 0 else None,
        drop_last=True,   # prevents trailing batch of 1 from crashing BatchNorm
    )

    # validation always uses 0 workers (main thread only) — on Windows, spawned
    # worker subprocesses receive CTRL_CLOSE_EVENT from the OS console job object
    # which kills them unpredictably, causing RuntimeError mid-validation
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dl.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, valid_loader

def _sample_labels(dataset, sample_size):
    """helper function to sample labels from a dataset"""
    labels = []
    dataset_iter = iter(dataset)
    for _ in range(sample_size):
        try:
            # get next item and only consider it if the mask is valid
            _, label, mask = next(dataset_iter)
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
            _, label_map, mask = next(it)
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