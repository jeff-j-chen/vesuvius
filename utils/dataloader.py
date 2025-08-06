import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from collections import Counter
import zarr
import cv2
import random
import os
from typing import Iterator, List, Optional, Union
from .config import Config
import json
from tqdm import tqdm


def generate_tile_coords(z_range, y_range, x_range, config, volume):
    """
    Generate all valid (z, y, x) block start coordinates for a given region.
    Returns a list of (z_offset, y_offset, x_offset) tuples.
    
    Args:
        z_range: (start, end) for z dimension
        y_range: (start, end) for y dimension  
        x_range: (start, end) for x dimension
        depth: depth of the 3D block
        tile_size: size of the 2D tile
        volume: zarr volume to check for empty regions (optional)
        empty_threshold: minimum mean value to consider a block non-empty
    """
    z_start, z_end = z_range
    y_start, y_end = y_range
    x_start, x_end = x_range
    z_range_size = max(0, z_end - z_start - config.data.depth + 1)
    y_range_size = max(0, y_end - y_start - config.data.tile_size + 1)
    x_range_size = max(0, x_end - x_start - config.data.tile_size + 1)
    
    block_coords = []
    
    for d in range(0, z_range_size, max(1, int(config.data.depth//2))):
        if z_start + d + config.data.depth > z_end:
            continue
        for y in range(0, y_range_size, config.data.tile_size):
            for x in range(0, x_range_size, config.data.tile_size):
                block_coords.append((d, y, x))
    
    return block_coords

class InkVolumeDataset(IterableDataset):
    def __init__(self, volume, mask: np.ndarray, labels: np.ndarray, config: Config, 
                 x_range: tuple, y_range: tuple,
                 global_mean: float,
                 global_std: float,
                 global_min: float,
                 global_max: float,
                 apply_transforms: bool = False,
                 shuffle: bool = True):
        """
        volume: zarr.Array
        labels: [H, W] - 2D binary mask (full mask, not cropped)
        config: Configuration object
        x_range: (start, end) for x dimension (global)
        y_range: (start, end) for y dimension (global)
        apply_transforms: Whether to apply data augmentation
        shuffle: Whether to shuffle the order of tiles (for training) or not (for evaluation)
        """
        self.volume = volume
        self.mask = mask
        self.labels = labels
        self.config = config
        self.tile_size = config.data.tile_size
        self.depth = config.data.depth
        self.apply_transforms = apply_transforms
        self.shuffle = shuffle
        self.global_mean = global_mean
        self.global_std = global_std
        self.global_min = global_min
        self.global_max = global_max

        # Store coordinate ranges (global)
        self.z_start, self.z_end = self.config.data.start_level, self.config.data.end_level
        self.y_start, self.y_end = y_range
        self.x_start, self.x_end = x_range
        
        # Pre-calculate all valid block coordinates with overlapping sampling (global coordinates)
        # Pass volume for empty region filtering if requested
        
        self.block_coords = generate_tile_coords(
            (self.z_start, self.z_end),
            (self.y_start, self.y_end),
            (self.x_start, self.x_end),
            self.config,
            volume  # Adjust this threshold as needed
        )
        
        self.samples_per_epoch = len(self.block_coords)

    def __len__(self):
        """Return the number of samples per epoch for progress bars and DataLoader."""
        return self.samples_per_epoch

    def _apply_channel_mixing(self, block):
        """Mix the order of the depth channels."""
        indices = np.random.permutation(block.shape[0])
        return block[indices]
    
    def _apply_brightness_adjustment(self, block):
        """Apply brightness adjustment to each channel independently."""

        brightness_factors = np.random.uniform(0.85, 1.15, size=(block.shape[0], 1, 1))
        return np.clip(block * brightness_factors, 0, 1)
    
    def _apply_contrast_adjustment(self, block):
        """Apply contrast adjustment to each channel independently."""

        adjusted_block = block.copy()
        for i in range(block.shape[0]):
            channel = block[i]
            contrast_factor = random.uniform(0.85, 1.15)
            mean_val = np.mean(channel)
            adjusted_block[i] = np.clip((channel - mean_val) * contrast_factor + mean_val, 0, 1)
        return adjusted_block
    
    def _apply_gaussian_noise(self, block):
        """Apply Gaussian noise to each channel independently."""

        noise_std = random.uniform(0.005, 0.015)
        noise = np.random.normal(0, noise_std, block.shape)
        return np.clip(block + noise, 0, 1)
    
    def _apply_rotation(self, block):
        """Apply 90/180/270 degree rotations to all channels."""

        rotation = random.choice([1, 2, 3])  # Rotate by 90, 180, or 270 degrees
        rotated_block = np.zeros_like(block)
        for i in range(block.shape[0]):  # Rotate each channel independently
            rotated_block[i] = np.rot90(block[i], k=rotation)
        return rotated_block
    
    def _apply_flip(self, block):
        """Apply horizontal or vertical flip to all channels."""

        flip_type = random.choice([0, 1])
        flipped_block = np.zeros_like(block)
        for i in range(block.shape[0]):
            if flip_type == 0:  # Horizontal flip
                flipped_block[i] = np.flip(block[i], axis=1)
            elif flip_type == 1:  # Vertical flip
                flipped_block[i] = np.flip(block[i], axis=0)
        return flipped_block

    def _normalize_block(self, block, mask):
        """Normalize a block using global mean, std, min, and max."""
        if self.global_std == 0:
            print("[WARNING] Standard deviation is 0, skipping normalization.")
            return block.astype(np.float32)

        # Convert mask to PyTorch tensor if it's not already
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)

        # Subtract mean and divide by std
        normalized_block = (block.astype(np.float32) - self.global_mean) / self.global_std

        # Apply mask to exclude invalid regions
        mask_exp = mask.unsqueeze(0).expand_as(torch.tensor(normalized_block))
        normalized_block[mask_exp == 0] = 0  # Zero out masked regions

        # Scale to [0, 1] using global min and max
        normalized_block = (normalized_block - self.global_min) / (self.global_max - self.global_min)
        normalized_block = np.clip(normalized_block, 0, 1)  # Ensure values are within [0, 1]

        return normalized_block

    def _fetch_block(self, z_offset, y_offset, x_offset):
        """Fetch and normalize a block from zarr volume."""
        z_start = self.z_start + z_offset
        y_start = self.y_start + y_offset
        x_start = self.x_start + x_offset
        block = self.volume[
            z_start:z_start + self.config.data.depth, 
            y_start:y_start + self.config.data.tile_size, 
            x_start:x_start + self.config.data.tile_size
        ]
        return block

    def _fetch_label(self, y_offset, x_offset):
        """Fetch a label tile from the full mask."""
        y_start = self.y_start + y_offset
        x_start = self.x_start + x_offset
        label_tile = self.labels[
            y_start:y_start + self.config.data.tile_size, 
            x_start:x_start + self.config.data.tile_size
        ]
        # Determine if the region contains ink (binary label)
        has_ink = np.any(label_tile > 0.5)  # True if any pixel in the region has ink
        return torch.tensor([float(has_ink)], dtype=torch.float32)

    def _fetch_mask(self, y_offset, x_offset):
        """Fetch a mask tile from the full mask"""
        y_start = self.y_start + y_offset
        x_start = self.x_start + x_offset
        mask_tile = self.mask[
            y_start:y_start + self.config.data.tile_size, 
            x_start:x_start + self.config.data.tile_size, 
        ]
        return torch.tensor(mask_tile, dtype=torch.float32)

    def __iter__(self) -> Iterator:
        self._shuffled_blocks = self.block_coords.copy()
        if self.shuffle:
            np.random.shuffle(self._shuffled_blocks)
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading, return the full iterator
            self._worker_indices = self._shuffled_blocks
        else:
            # In a worker process, split the workload
            per_worker = int(np.ceil(len(self._shuffled_blocks) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self._shuffled_blocks))
            self._worker_indices = self._shuffled_blocks[start:end]
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= len(self._worker_indices):
            raise StopIteration
        z_offset, y_offset, x_offset = self._worker_indices[self._current_idx]
        mask = self._fetch_mask(y_offset, x_offset)
        block = self._fetch_block(z_offset, y_offset, x_offset)
        block_normalized = self._normalize_block(block, mask)
        if isinstance(block_normalized, torch.Tensor):
            block_normalized = block_normalized.numpy()
        label = self._fetch_label(y_offset, x_offset)
        if self.apply_transforms:
            if random.random() < 0.25: block_normalized = self._apply_channel_mixing(block_normalized)
            if random.random() < 0.25: block_normalized = self._apply_rotation(block_normalized)
            if random.random() < 0.25: block_normalized = self._apply_flip(block_normalized)
            if random.random() < 0.30: block_normalized = self._apply_gaussian_noise(block_normalized)
            if random.random() < 0.50: block_normalized = self._apply_brightness_adjustment(block_normalized)
            if random.random() < 0.50: block_normalized = self._apply_contrast_adjustment(block_normalized)
        # Convert block to PyTorch tensor at the very end
        block_normalized = torch.tensor(block_normalized, dtype=torch.float32).unsqueeze(0)
        self._current_idx += 1
        return block_normalized, label, mask

def get_or_compute_normalization(config, volume, mask):
    """Retrieve or compute global mean, std, min, and max for normalization."""
    segment_id = config.data.segment_id
    cache = _load_normalization_cache()

    if str(segment_id) in cache:
        print(f"[INFO] Using cached normalization for segment {segment_id}")
        return cache[str(segment_id)]["mean"], cache[str(segment_id)]["std"], cache[str(segment_id)]["min"], cache[str(segment_id)]["max"]

    print(f"[INFO] Computing normalization for segment {segment_id}")
    total_sum, total_squared_sum, total_count = 0.0, 0.0, 0
    global_min, global_max = float('inf'), float('-inf')

    for z in tqdm(range(volume.shape[0])):
        for y in range(0, volume.shape[1], 1024):
            for x in range(0, volume.shape[2], 1024):
                chunk = volume[z, y:y+1024, x:x+1024]
                mask_chunk = mask[y:y+1024, x:x+1024]
                valid_pixels = chunk[mask_chunk > 0]
                if valid_pixels.size == 0:
                    continue

                total_sum += np.sum(valid_pixels, dtype=np.float64)
                total_squared_sum += np.sum(valid_pixels.astype(np.float64) ** 2, dtype=np.float64)
                total_count += valid_pixels.size

    if total_count == 0:
        raise ValueError("No valid pixels found in the dataset.")

    # Compute global mean and std
    global_mean = total_sum / total_count
    mean_of_squares = total_squared_sum / total_count
    square_of_mean = global_mean ** 2
    variance = max(mean_of_squares - square_of_mean, 0)
    global_std = np.sqrt(variance)

    # Compute global min and max after normalization
    for z in tqdm(range(volume.shape[0])):
        for y in range(0, volume.shape[1], 1024):
            for x in range(0, volume.shape[2], 1024):
                chunk = volume[z, y:y+1024, x:x+1024]
                mask_chunk = mask[y:y+1024, x:x+1024]
                valid_pixels = chunk[mask_chunk > 0]
                if valid_pixels.size == 0:
                    continue

                # Normalize the valid pixels
                normalized_pixels = (valid_pixels.astype(np.float64) - global_mean) / global_std

                # Update global min and max
                global_min = min(global_min, normalized_pixels.min())
                global_max = max(global_max, normalized_pixels.max())

    print(f"Final Statistics:")
    print(f"  Global Mean: {global_mean:.6f}")
    print(f"  Global Std: {global_std:.6f}")
    print(f"  Global Min (after normalization): {global_min:.6f}")
    print(f"  Global Max (after normalization): {global_max:.6f}")

    _update_normalization_cache(segment_id, global_mean, global_std, global_min, global_max)
    return global_mean, global_std, global_min, global_max


def _load_normalization_cache():
    """Load normalization cache from file."""
    if not os.path.exists("./normalization_cache.json"):
        print(f"[INFO] No cache found.")
        return {}
    with open("./normalization_cache.json", "r") as f:
        print(f"[INFO] Cache found, loading...")
        return json.load(f)

def _update_normalization_cache(segment_id, mean, std, min_val, max_val):
    """Update normalization cache with new values."""
    cache = _load_normalization_cache()
    cache[str(segment_id)] = {
        "mean": float(mean),  # Convert to float
        "std": float(std),    # Convert to float
        "min": float(min_val),  # Convert to float
        "max": float(max_val)   # Convert to float
    }

    try:
        with open("./normalization_cache.json", "w") as f:
            json.dump(cache, f, indent=4)  # Use indent for readability
            f.flush()  # Ensure data is written to disk
        print(f"[INFO] Updated normalization cache for segment {segment_id} with mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}")
    except Exception as e:
        print(f"[ERROR] Failed to update normalization cache: {e}")


def load_tv_data(config: Config):
    """Load labels and determine coordinate ranges for train/validation split, streaming Zarr data chunk by chunk only."""
    # Construct zarr path for the segment
    zarr_path = os.path.join(config.data.zarr_path, f"{config.data.segment_id}_fixed.zarr")
    # Open zarr just to get dimensions (do not read the full array)
    volume = zarr.open(zarr_path, mode='r')
    D, H, W = map(int, volume.shape)
    # Load labels (these are small, so OK to load fully)
    labels_path = f"./eroded_inklabels/{config.data.segment_id}.png"
    labels = cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE) / 255.0

    mask_path = f"./masks/{config.data.segment_id}.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
    # Apply segment-specific processing
    if config.data.segment_id == 20230827161847:
        z_start, z_end = config.data.start_level, config.data.end_level
        y_start, y_end = 200, 5600
        x_start, x_end = 1000, 4600
    elif config.data.segment_id == 20231106155351:
        z_start, z_end = 0, D
        y_start, y_end = 0, H
        x_start, x_end = 4500, W
    else:
        z_start, z_end = 0, D
        y_start, y_end = 0, H
        x_start, x_end = 0, W
    # Calculate train/validation split along x-axis
    working_width = int(x_end) - int(x_start)
    split_x = int(working_width * 0.75)
    # Define ranges for train and validation
    train_x_range = (x_start, x_start+split_x)
    valid_x_range = (x_start+split_x, x_end)
    y_range = (y_start, y_end)
    # Split labels accordingly
    # print(f"[DEBUG] Zarr shape: (D={D}, H={H}, W={W})")
    # print(f"[DEBUG] Train x_range: {train_x_range}, y_range: {train_y_range}, z_range: {train_z_range}, shape: {train_labels.shape}")
    # print(f"[DEBUG] Valid x_range: {valid_x_range}, y_range: {valid_y_range}, z_range: {valid_z_range}, shape: {valid_labels.shape}")
    return (volume, mask, labels, train_x_range, valid_x_range, y_range)


def get_test_dataset(config: Config):
    """Load test data from zarr - returns a view for on-demand access (never loads full array)."""
    zarr_path = os.path.join(config.data.zarr_path, f"{config.data.segment_id}.zarr")
    volume = zarr.open(zarr_path, mode='r')
    class TestVolumeView:
        def __init__(self, zarr_volume):
            self.zarr_volume = zarr_volume
            self.shape = (zarr_volume.shape[0], zarr_volume.shape[1] - 4000, zarr_volume.shape[2])
        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 3:
                z_slice, y_slice, x_slice = key
                y_adj = slice(y_slice.start + 4000, y_slice.stop + 4000)
                # Only loads the requested chunk/slice
                return self.zarr_volume[z_slice, y_adj, x_slice] / 65535.0
            else:
                return self.zarr_volume[key] / 65535.0
    test_volume = TestVolumeView(volume)
    print("Test volume shape:", test_volume.shape)
    print(test_volume[14, 3000:3005, 3000:3005])
    return test_volume


def load_scroll4_data(config: Config):
    """Load scroll4 data - keeping original implementation unchanged"""
    data = np.load(config.data.scroll4_path)
    volume = data['stack']
    print("Scroll4 volume shape:", volume.shape)
    print(volume[14, 1000:1005, 1000:1005])
    return volume


def get_tv_datasets(config: Config):
    (volume, mask, labels, train_x_range, valid_x_range, y_range) = load_tv_data(config)
    global_mean, global_std, global_min, global_max = get_or_compute_normalization(config, volume, mask)
    train_dataset = InkVolumeDataset(volume, mask, labels, config, train_x_range, y_range, global_mean, global_std, global_min, global_max, shuffle=True, apply_transforms=True)
    valid_dataset = InkVolumeDataset(volume, mask, labels, config, valid_x_range, y_range, global_mean, global_std, global_min, global_max, shuffle=False, apply_transforms=False)
    return train_dataset, valid_dataset


def get_dataloaders(train_dataset, valid_dataset, config: Config):
    """Create DataLoader objects from datasets"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=True,
    )
    
    return train_loader, valid_loader

def _sample_labels(dataset, sample_size):
    """Helper function to sample labels from a dataset, respecting the mask."""
    all_labels = []
    dataset_iter = iter(dataset)
    for _ in range(sample_size):
        try:
            _, label, mask = next(dataset_iter)  # Unpack block, label, and mask
            if mask.sum() > 0:  # Only include labels where the mask is non-zero
                all_labels.append(int(label.item()))
        except StopIteration:
            break
    return all_labels

def calculate_class_weights(train_set, valid_set):
    """Calculate class weights, ensuring sampling respects the mask."""
    print("Sampling datasets to calculate average class weights...")
    sample_size = 100
    # Sample labels from both datasets
    labels_a = _sample_labels(train_set, sample_size * 2)
    print('got a')
    labels_b = _sample_labels(valid_set, sample_size)

    # Combine labels from both datasets
    all_labels = labels_a + labels_b

    if len(all_labels) == 0:
        print("Warning: No samples found for class weight calculation!")
        return None

    # Calculate label counts
    label_counts = Counter(all_labels)
    print(f"Label distribution (from {len(all_labels)} samples): {label_counts}")

    pos_weight = None
    if label_counts.get(0, 0) > 0 and label_counts.get(1, 0) > 0:
        pos_weight = torch.tensor([label_counts[0] / label_counts[1]])
        print(f"Using average pos_weight: {pos_weight.item():.2f}")
    else:
        print("Warning: Only one class present in sampled data!")

    return pos_weight


def get_tile_coords_for_split(config: Config, split: str):
    """
    Returns tile coordinates for a given split ('train' or 'valid') using the same logic as InkVolumeDataset.
    """
    volume, mask, labels, train_x_range, valid_x_range, y_range = load_tv_data(config)
    z_range = (config.data.start_level, config.data.end_level)
    if split == 'train':
        x_range = train_x_range
    elif split == 'valid':
        x_range = valid_x_range
    else:
        raise ValueError(f"Unknown split: {split}")
    coords = generate_tile_coords(z_range, y_range, x_range, config, volume)
    return coords, y_range, x_range, z_range