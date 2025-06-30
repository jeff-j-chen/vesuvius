import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import vesuvius
from vesuvius import Volume
from .config import Config
import cv2
import random


class InkVolumeDataset(Dataset):
    def __init__(self, volume, labels, config, apply_transforms=False):
        """
        volume: [D, H, W] - 3D volume of grayscale slices
        labels: [H, W] - 2D binary mask shared across depth
        config: Configuration object containing tile_size and depth
        apply_transforms: Whether to apply data augmentation
        """
        self.volume = volume
        self.labels = labels
        self.tile_size = config.data.tile_size
        self.depth = config.data.depth
        self.D, self.H, self.W = volume.shape
        self.apply_transforms = apply_transforms

        self.blocks = []
        for d in range(0, self.D - self.depth + 1, int(self.depth//2)):
            for y in range(0, self.H - self.tile_size + 1, self.tile_size):
                for x in range(0, self.W - self.tile_size + 1, self.tile_size):
                    label_tile = labels[y:y+self.tile_size, x:x+self.tile_size]
                    if label_tile.shape == (self.tile_size, self.tile_size):
                        self.blocks.append((d, y, x))

    def _apply_channel_mixing(self, block):
        """Mix the order of the 8 depth channels"""
        # block shape: [D, H, W] where D=8
        indices = torch.randperm(block.shape[0])
        return block[indices]
    
    def _apply_brightness_adjustment(self, block):
        """Apply brightness adjustment to each channel independently"""
        brightness_factors = torch.rand(block.shape[0], 1, 1) * 0.2 + 0.9
        return torch.clamp(block * brightness_factors, 0, 1)
    
    def _apply_contrast_adjustment(self, block):
        """Apply contrast adjustment to each channel independently"""
        adjusted_block = block.clone()
        for i in range(block.shape[0]):
            channel = block[i]
            # Random contrast factor (0.8 to 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            # Apply contrast: new_val = (old_val - mean) * contrast + mean
            mean_val = torch.mean(channel)
            adjusted_block[i] = torch.clamp(
                (channel - mean_val) * contrast_factor + mean_val, 0, 1
            )
        return adjusted_block
    
    def _apply_gaussian_noise(self, block):
        """Apply Gaussian noise to each channel independently"""
        # Small noise to avoid destroying signal (std=0.01 to 0.05)
        noise_std = random.uniform(0.01, 0.05)
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
        
        if flip_type == 0:
            return block
        
        # Apply flip to each channel
        flipped_block = torch.zeros_like(block)
        for i in range(block.shape[0]):
            if flip_type == 0:  # Horizontal flip
                flipped_block[i] = torch.flip(block[i], dims=[1])  # flip along width
            elif flip_type == 1:  # Vertical flip
                flipped_block[i] = torch.flip(block[i], dims=[0])  # flip along height
        
        return flipped_block

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        d, y, x = self.blocks[idx]
        
        block = self.volume[d:d+self.depth, y:y+self.tile_size, x:x+self.tile_size]
        label_tile = self.labels[y:y+self.tile_size, x:x+self.tile_size]

        # Convert to tensor and ensure proper normalization
        block = torch.tensor(block, dtype=torch.float32)
        
        # Apply transforms if enabled (before adding channel dimension)
        if self.apply_transforms:
            if random.random() < 0.5:
                block = self._apply_channel_mixing(block)
            transform_type = random.choice(["brightness", "contrast", "noise", "rotate", "flip", None])
            if transform_type == "brightness":
                block = self._apply_brightness_adjustment(block)
            elif transform_type == "contrast":
                block = self._apply_contrast_adjustment(block)
            elif transform_type == "noise":
                block = self._apply_gaussian_noise(block)
            elif transform_type == "rotate":
                block = self._apply_rotation(block)
            elif transform_type == "flip":
                block = self._apply_flip(block)


        # Add channel dimension: [D, H, W] -> [1, D, H, W]
        block = block.unsqueeze(0)

        # # Binary label: 1 if any ink present (more robust checking)
        # if d <= 6 or d > 24:
        #     # For the first and last few slices, assume no ink
        #     has_ink = False
        # else:
        #     has_ink = np.any(label_tile > 0.5)
        # has_ink = np.any(label_tile > 0.5)
        # instead of any, check if the average is above a threshold
        has_ink = np.mean(label_tile) > 0.5  # Adjust
        label = torch.tensor([float(has_ink)], dtype=torch.float32)

        return block, label

def _load_tv_data(config: Config):
    """Load and prepare the volume data according to configuration"""
    segment = Volume(config.data.segment_id, normalize=config.data.normalize)
    
    # Extract volume and labels according to config
    if config.data.segment_id == 20230827161847:
        volume = segment[config.data.start_level:config.data.end_level, 200:5600, 1000:4600] # type: ignore
    elif config.data.segment_id == 20231106155351:
        volume = segment[:, :, 4500:] # type: ignore
    else:
        volume = segment[:, :, :] # type: ignore
    labels_path = f"./inklabels/{config.data.segment_id}.png"
    labels = cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE)

    # Normalize labels to range [0, 1]
    labels = labels[200:5600, 1000:4600] / 255.0
    split_x = int(volume.shape[2] * 0.75)
    
    train_volume = volume[:, :, :split_x]
    train_labels = labels[:, :split_x]
    valid_volume = volume[:, :, split_x:]
    valid_labels = labels[:, split_x:]
    
    return train_volume, train_labels, valid_volume, valid_labels, labels

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

def create_datasets(config: Config):
    """Split data and create train/validation datasets"""
    train_volume, train_labels, valid_volume, valid_labels, labels = _load_tv_data(config)
    
    train_dataset = InkVolumeDataset(train_volume, train_labels, config, False)
    valid_dataset = InkVolumeDataset(valid_volume, valid_labels, config, False)
    
    return train_dataset, valid_dataset, train_volume, valid_volume, labels

def create_dataloaders(train_dataset, valid_dataset, config: Config):
    """Create DataLoader objects from datasets"""
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
