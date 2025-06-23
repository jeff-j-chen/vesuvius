import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import vesuvius
from vesuvius import Volume
from .config import Config
import cv2

class InkVolumeDataset(Dataset):
    def __init__(self, volume, labels, config: Config):
        """
        volume: [D, H, W] - 3D volume of grayscale slices
        labels: [H, W] - 2D binary mask shared across depth
        config: Configuration object containing tile_size and depth
        """
        self.volume = volume
        self.labels = labels
        self.tile_size = config.data.tile_size
        self.depth = config.data.depth
        self.D, self.H, self.W = volume.shape

        self.blocks = []
        for d in range(0, self.D - self.depth + 1, int(self.depth//2)):
            for y in range(0, self.H - self.tile_size + 1, self.tile_size):
                for x in range(0, self.W - self.tile_size + 1, self.tile_size):
                    label_tile = labels[y:y+self.tile_size, x:x+self.tile_size]
                    if label_tile.shape == (self.tile_size, self.tile_size):
                        self.blocks.append((d, y, x))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        d, y, x = self.blocks[idx]
        
        block = self.volume[d:d+self.depth, y:y+self.tile_size, x:x+self.tile_size]
        label_tile = self.labels[y:y+self.tile_size, x:x+self.tile_size]

        # Convert to tensor and ensure proper normalization
        block = torch.tensor(block, dtype=torch.float32)
        
        # Add channel dimension: [D, H, W] -> [1, D, H, W]
        block = block.unsqueeze(0)

        # Binary label: 1 if any ink present (more robust checking)
        has_ink = np.any(label_tile > 0.5)
        label = torch.tensor([float(has_ink)], dtype=torch.float32)

        return block, label

def load_data(config: Config):
    """Load and prepare the volume data according to configuration"""
    segment = Volume(config.data.segment_id, normalize=config.data.normalize)
    
    # Extract volume and labels according to config
    volume = segment[27:44, 200:5600, 1000:4600] # type: ignore
    # labels = segment.inklabel[200:5600, 1000:4600] / 255.0
    # instead of base labels, define as those taken from file /media/jeff/Seagate/vesuvius/fixed_inklabels.png
    labels_path = "/media/jeff/Seagate/vesuvius/fixed_inklabels.png"
    labels = cv2.imread(labels_path, cv2.IMREAD_GRAYSCALE)

    # Normalize labels to range [0, 1]
    labels = labels[200:5600, 1000:4600] / 255.0
    
    return volume, labels

def create_datasets(volume, labels, config: Config):
    """Split data and create train/validation datasets"""
    split_x = int(volume.shape[2] * 0.75)
    
    train_volume = volume[:, :, :split_x]
    train_labels = labels[:, :split_x]
    valid_volume = volume[:, :, split_x:]
    valid_labels = labels[:, split_x:]
    
    train_dataset = InkVolumeDataset(train_volume, train_labels, config)
    valid_dataset = InkVolumeDataset(valid_volume, valid_labels, config)
    
    return train_dataset, valid_dataset, train_volume, train_labels, valid_volume, valid_labels

def create_dataloaders(train_dataset, valid_dataset, config: Config):
    """Create DataLoader objects from datasets"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=config.dataloader.shuffle_train
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=config.dataloader.shuffle_valid
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
