import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import vesuvius
from vesuvius import Volume
from .config import Config
import cv2
import random
import os


class InkVolumeDataset(Dataset):
    def __init__(self, volume_params, label_path, config, block_coords, apply_transforms=False):
        """
        RAM-efficient dataset for lazy loading of 3D volume blocks using Zarr-backed Volume.
        volume_params: dict of args for Volume (e.g., segment_id, normalization, etc.)
        label_path: path to the label PNG
        config: config object
        block_coords: list of (d, y, x) tuples for valid blocks
        apply_transforms: Whether to apply data augmentation
        """
        self.volume_params = volume_params
        self.label_path = label_path
        self.config = config
        self.block_coords = block_coords
        self.apply_transforms = apply_transforms
        self.tile_size = config.data.tile_size
        self.depth = config.data.depth
        self._volume = None
        self._labels = None

    @property
    def volume(self):
        # Lazily instantiate Volume per worker
        if self._volume is None:
            self._volume = Volume(**self.volume_params)
        return self._volume

    @property
    def labels(self):
        # Lazily load labels per worker
        if self._labels is None:
            labels = cv2.imread(self.label_path, cv2.IMREAD_GRAYSCALE)
            labels = labels / 255.0
            self._labels = labels
        return self._labels

    def _apply_channel_mixing(self, block):
        """Mix the order of the 8 depth channels"""
        # block shape: [D, H, W] where D=8
        indices = torch.randperm(block.shape[0])
        return block[indices]
    
    def _apply_brightness_adjustment(self, block):
        """Apply brightness adjustment to each channel independently"""
        brightness_factors = torch.rand(block.shape[0], 1, 1) * 0.3 + 0.85
        return torch.clamp(block * brightness_factors, 0, 1)
    
    def _apply_contrast_adjustment(self, block):
        """Apply contrast adjustment to each channel independently"""
        adjusted_block = block.clone()
        for i in range(block.shape[0]):
            channel = block[i]
            # Random contrast factor (0.8 to 1.2)
            contrast_factor = random.uniform(0.85, 1.15)
            # Apply contrast: new_val = (old_val - mean) * contrast + mean
            mean_val = torch.mean(channel)
            adjusted_block[i] = torch.clamp(
                (channel - mean_val) * contrast_factor + mean_val, 0, 1
            )
        return adjusted_block
    
    def _apply_gaussian_noise(self, block):
        """Apply Gaussian noise to each channel independently"""
        # Small noise to avoid destroying signal (std=0.01 to 0.03)
        noise_std = random.uniform(0.005, 0.015)
        noise = torch.randn_like(block) * noise_std
        return torch.clamp(block + noise, 0, 1)
    
    def _apply_rotation(self, block):
        """Apply 90/180/270 degree rotations to all channels and label"""
        # Choose rotation: 1 (90°), 2 (180°), 3 (270°)
        rotation = random.choice([1, 2, 3])
        
        # Apply rotation to each channel
        rotated_block = torch.zeros_like(block)
        for i in range(block.shape[0]):
            rotated_block[i] = torch.rot90(block[i], k=rotation, dims=[0, 1])
        
        return rotated_block
    
    def _apply_flip(self, block):
        """Apply horizontal or vertical flip to all channels and label"""
        # Choose flip: 0 (no flip), 1 (horizontal), 2 (vertical)
        flip_type = random.choice([0, 1])
        
        # Apply flip to each channel
        flipped_block = torch.zeros_like(block)
        for i in range(block.shape[0]):
            if flip_type == 0:  # Horizontal flip
                flipped_block[i] = torch.flip(block[i], dims=[1])  # flip along width
            elif flip_type == 1:  # Vertical flip
                flipped_block[i] = torch.flip(block[i], dims=[0])  # flip along height
        
        return flipped_block

    def __len__(self):
        return len(self.block_coords)

    def __getitem__(self, idx):
        d, y, x = self.block_coords[idx]
        # Only load the required block from disk (lazy, RAM-efficient)
        block = self.volume[d:d+self.depth, y:y+self.tile_size, x:x+self.tile_size]
        label_tile = self.labels[y:y+self.tile_size, x:x+self.tile_size]
        block = torch.tensor(block, dtype=torch.float32)
        # Apply transforms if enabled
        if self.apply_transforms:
            if random.random() < 0.25:
                block = self._apply_channel_mixing(block)
            if random.random() < 0.25:
                block = self._apply_rotation(block)
            if random.random() < 0.25:
                block = self._apply_flip(block)
            if random.random() < 0.3:
                block = self._apply_gaussian_noise(block)
            if random.random() < 0.5:
                block = self._apply_brightness_adjustment(block)
            if random.random() < 0.5:
                block = self._apply_contrast_adjustment(block)
        block = block.unsqueeze(0)
        has_ink = np.mean(label_tile) > 0.5
        label = torch.tensor([float(has_ink)], dtype=torch.float32)
        return block, label

# Helper to compute valid block coordinates without loading the full volume

def compute_block_coords(volume_shape, tile_size, depth):
    """
    Compute valid (d, y, x) block coordinates for a given volume shape.
    volume_shape: (D, H, W)
    tile_size: int
    depth: int
    Returns: list of (d, y, x)
    """
    D, H, W = volume_shape
    coords = []
    for d in range(0, D - depth + 1, max(1, depth // 2)):
        for y in range(0, H - tile_size + 1, tile_size):
            for x in range(0, W - tile_size + 1, tile_size):
                coords.append((d, y, x))
    return coords

# Updated _load_tv_data for lazy loading

def _load_tv_data(config: Config):
    """Prepare volume parameters and label path for lazy loading."""
    volume_params = dict(type='segment', segment_id=config.data.segment_id, normalization_scheme='none')
    segment = Volume(**volume_params)
    # Determine cropping based on segment_id
    if config.data.segment_id == 20230827161847:
        y0, y1, x0, x1 = 200, 5600, 1000, 4600
        volume_shape = (segment.shape()[0], y1 - y0, x1 - x0)
        label_path = f"./inklabels/{config.data.segment_id}.png"
        # Labels will be cropped in the Dataset
    elif config.data.segment_id == 20231106155351:
        y0, y1, x0, x1 = 0, segment.shape()[1], 4500, segment.shape()[2]
        volume_shape = (segment.shape()[0], y1 - y0, segment.shape()[2] - x0)
        label_path = f"./inklabels/{config.data.segment_id}.png"
    else:
        y0, y1, x0, x1 = 0, segment.shape()[1], 0, segment.shape()[2]
        volume_shape = (segment.shape()[0], y1 - y0, x1 - x0)
        label_path = f"./inklabels/{config.data.segment_id}.png"
    # Compute block coordinates
    block_coords = compute_block_coords(volume_shape, config.data.tile_size, config.data.depth)
    # For splitting, use 75%/25% along width
    split_x = int(volume_shape[2] * 0.75)
    train_block_coords = [c for c in block_coords if c[2] < split_x]
    valid_block_coords = [c for c in block_coords if c[2] >= split_x]
    return volume_params, label_path, y0, x0, train_block_coords, valid_block_coords

def create_datasets(config: Config):
    """Create RAM-efficient, lazy datasets for train/validation."""
    volume_params, label_path, y0, x0, train_block_coords, valid_block_coords = _load_tv_data(config)
    # Pass cropping info to the dataset via config (or as extra args if needed)
    train_dataset = InkVolumeDataset(volume_params, label_path, config, train_block_coords, apply_transforms=False)
    valid_dataset = InkVolumeDataset(volume_params, label_path, config, valid_block_coords, apply_transforms=False)
    return train_dataset, valid_dataset, None, None, None  # No full volumes/labels loaded

def create_dataloaders(train_dataset, valid_dataset, config: Config):
    """Create DataLoader objects from datasets (RAM-efficient)."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.train_batch_size,
        num_workers=config.dataloader.train_num_workers,
        shuffle=config.dataloader.train_shuffle,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dataloader.valid_batch_size,
        num_workers=config.dataloader.valid_num_workers,
        shuffle=config.dataloader.valid_shuffle,
        pin_memory=True,
    )
    return train_loader, valid_loader

def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced data"""
    all_labels = [int(label.item()) for _, label in dataset]
    label_counts = Counter(all_labels)
    print(f"Label distribution: {label_counts}")
    
    pos_weight = None
    if label_counts[0] > 0 and label_counts[1] > 0:
        pos_weight = torch.tensor([label_counts[0] / label_counts[1]])
        print(f"Using pos_weight: {pos_weight.item():.2f}")
    else:
        print("Warning: Only one class present in training data!")
    
    return pos_weight

def load_test_data(config: Config):
    segment = Volume(config.data.segment_id, normalize=config.data.normalize)
    volume = segment[:, 4000:, :] # type: ignore
    # print(f"test data loaded with shape: {volume.shape}, dtype: {volume.dtype}, std {volume.std():.4f}, mean {volume.mean():.4f}")
    print(volume[14, 1000:1005, 1000:1005])
    return volume

def load_scroll4_data(config: Config):
    data = np.load(config.data.scroll4_path)
    volume = data['stack']
    # print(f"scroll4 data loaded with shape: {volume.shape}, dtype: {volume.dtype}, std {volume.std():.4f}, mean {volume.mean():.4f}")
    print(volume[14, 1000:1005, 1000:1005])
    return volume
