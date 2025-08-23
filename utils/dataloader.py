import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from collections import Counter
import zarr
import cv2
import random
import os
from typing import Iterator
from .config import Config
import json
from tqdm import tqdm

class Transform:
    """handles data augmentation transforms"""
    def __call__(self, block):
        """applies a random sequence of transforms to a block"""
        # each transform is applied with a certain probability
        if random.random() < 0.25: 
            block = self._apply_channel_mixing(block)
        if random.random() < 0.25: 
            block = self._apply_rotation(block)
        if random.random() < 0.25: 
            block = self._apply_flip(block)
        if random.random() < 0.30: 
            block = self._apply_gaussian_noise(block)
        if random.random() < 0.50: 
            block = self._apply_brightness_adjustment(block)
        if random.random() < 0.50: 
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
    def __init__(self, volume, mask, labels, config, x_range, y_range, norm_stats, shuffle=True):
        """initializes the dataset"""
        self.vol = volume
        self.mask = mask
        self.labels = labels
        self.c = config
        self.tile_size = config.data.tile_size
        self.depth = config.data.depth
        self.apply_transforms = False # controlled by trainer
        self.shuffle = shuffle
        self.norm_stats = norm_stats
        self.transform = Transform()

        self.z_start, self.z_end = self.c.data.d_start, self.c.data.d_end
        self.y_start, self.y_end = y_range
        self.x_start, self.x_end = x_range
        
        # pre-calculate all valid block coordinates
        self.block_coords = self._gen_tile_coords()
        self.samples_per_epoch = len(self.block_coords)

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

    def _fetch_block(self, z_off, y_off, x_off):
        """fetches and normalizes a block from zarr volume"""
        z = self.z_start + z_off
        y = self.y_start + y_off
        x = self.x_start + x_off
        
        # slice the block from the zarr volume
        block = self.vol[z:z+self.depth, y:y+self.tile_size, x:x+self.tile_size]
        return self._normalize_block(block)

    def _fetch_label(self, y_off, x_off):
        """fetches a label tile"""
        y = self.y_start + y_off
        x = self.x_start + x_off
        
        # slice the label tile and check for ink presence
        label_tile = self.labels[y:y+self.tile_size, x:x+self.tile_size]
        has_ink = np.any(label_tile > 0.5)
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

class DataManager:
    """manages data loading, splitting, and normalization"""
    def __init__(self, config: Config):
        """initializes the data manager"""
        self.c = config
        
        # load raw data and define splits
        self.vol, self.mask, self.labels, self.train_x, self.valid_x, self.y_range = self._load_raw_data()
        
        # get or compute normalization statistics
        self.norm_stats = self._get_or_compute_norm()

    def _load_raw_data(self):
        """loads raw zarr data and metadata"""
        # open the zarr volume in read-only mode
        vol = zarr.open(
            os.path.join(
                self.c.data.zarr_path, f"{self.c.data.scroll1_id}.zarr"
            ),
            mode='r'
        )
        
        # load labels and mask, and normalize to [0, 1]
        labels = cv2.imread(
            f"./eroded_inklabels/{self.c.data.scroll1_id}.png",
            cv2.IMREAD_GRAYSCALE
        ) / 255.0

        mask = cv2.imread(
            f"./masks/{self.c.data.scroll1_id}.png", 
            cv2.IMREAD_GRAYSCALE
        ) / 255.0
        
        # define the working area and split for train/validation
        x_start, x_end = 0, vol.shape[2]
        y_start, y_end = 0, vol.shape[1]
        
        split_x = int((x_end - x_start) * 0.75) # type: ignore
        train_x_range = (x_start, x_start + split_x)
        valid_x_range = (x_start + split_x, x_end)
        y_range = (y_start, y_end)
        
        return vol, mask, labels, train_x_range, valid_x_range, y_range

    def _get_or_compute_norm(self):
        """retrieves or computes normalization statistics"""
        cache_path = "./norm_cache.json"
        seg_id = str(self.c.data.scroll1_id)
        
        # first, try to load from cache
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cache = json.load(f)
            if seg_id in cache:
                print(f"[info] using cached normalization for segment {seg_id}")
                stats = cache[seg_id]
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
        
        # update the cache file
        try:
            with open(cache_path, "r") as f: 
                cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}
        cache[seg_id] = stats
        with open(cache_path, "w") as f: 
            json.dump(cache, f, indent=4)
            
        return mean, std, g_min, g_max

    def get_datasets(self):
        """creates train and validation datasets"""
        train_set = InkVolumeDataset(self.vol, self.mask, self.labels, self.c, self.train_x, self.y_range, self.norm_stats, shuffle=True)
        valid_set = InkVolumeDataset(self.vol, self.mask, self.labels, self.c, self.valid_x, self.y_range, self.norm_stats, shuffle=False)
        return train_set, valid_set

def get_dataloaders(train_dataset, valid_dataset, config: Config):
    """creates dataloader objects from datasets"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dl.batch_size,
        num_workers=config.dl.num_workers,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dl.batch_size,
        num_workers=config.dl.num_workers,
        pin_memory=True,
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

def calc_class_wgts(train_set, valid_set):
    """calculates class weights from dataset samples"""
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
        pos_weight = torch.tensor([counts[0] / counts[1]])
        print(f"using average pos_weight: {pos_weight.item():.2f}")
        return pos_weight
    
    print("warning: only one class present in sampled data")
    return None