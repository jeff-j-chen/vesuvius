import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .config import Config
import random

class FinetuneDataset(Dataset):
    def __init__(self, volume, labels, mask, locs, config: Config, 
                 global_mean, global_std, global_min, global_max,
                 shuffle=True):
        self.volume = volume
        self.labels = labels
        self.mask = mask
        self.config = config
        self.tile_size = config.data.tile_size
        self.depth = config.data.depth
        self.global_mean = global_mean
        self.global_std = global_std
        self.global_min = global_min
        self.global_max = global_max
        self.shuffle = shuffle
        
        self.tiles = []
        self._prepare_tiles(locs)
        
        if self.shuffle:
            random.shuffle(self.tiles)

    def _prepare_tiles(self, locs):
        """Pre-calculate all tile coordinates and labels from the locs."""
        for x_start, y_start, width, height in locs:
            for y in range(y_start, y_start + height, self.tile_size):
                for x in range(x_start, x_start + width, self.tile_size):
                    # Check if the tile is within the mask
                    mask_tile = self.mask[y:y + self.tile_size, x:x + self.tile_size]
                    if np.sum(mask_tile) == 0:
                        continue

                    # Pre-calculate label
                    label_tile = self.labels[y:y + self.tile_size, x:x + self.tile_size]
                    has_ink = np.any(label_tile > 0.5)
                    label = torch.tensor([float(has_ink)], dtype=torch.float32)
                    
                    self.tiles.append({'x': x, 'y': y, 'label': label})

    def __len__(self):
        return len(self.tiles)

    def _normalize_block(self, block, mask_tile):
        """Normalize a block using global stats."""
        block = (block.astype(np.float32) - self.global_mean) / self.global_std
        
        mask_exp = np.expand_dims(mask_tile, axis=0)
        mask_exp = np.broadcast_to(mask_exp, block.shape)
        block[mask_exp == 0] = 0
        
        block = (block - self.global_min) / (self.global_max - self.global_min)
        block = np.clip(block, 0, 1)
        return block

    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        x, y, label = tile_info['x'], tile_info['y'], tile_info['label']
        
        # Fetch 3D block from zarr volume for a random depth slice
        z_start = random.randint(self.config.data.d_start, self.config.data.d_end - self.depth)
        
        block = np.array(self.volume[
            z_start : z_start + self.depth,
            y : y + self.tile_size,
            x : x + self.tile_size
        ])
        
        mask_tile = self.mask[y:y + self.tile_size, x:x + self.tile_size]
        
        block_normalized = self._normalize_block(block, mask_tile)
        
        block_tensor = torch.tensor(block_normalized, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_tile, dtype=torch.float32)
        
        return block_tensor, label, mask_tensor

def get_finetune_dataloaders(config, volume, labels, mask, norm_stats, locs):
    """Create train and validation dataloaders for fine-tuning."""
    train_locs = locs[:10]
    valid_locs = locs[10:]

    mean, std, min_val, max_val = norm_stats

    train_dataset = FinetuneDataset(volume, labels, mask, train_locs, config, mean, std, min_val, max_val, shuffle=True)
    valid_dataset = FinetuneDataset(volume, labels, mask, valid_locs, config, mean, std, min_val, max_val, shuffle=False)

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
