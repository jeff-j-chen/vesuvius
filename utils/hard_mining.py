import json, os, math, random
import torch
import numpy as np

class HardMiningInjector:
    """Iterator providing hard examples to inject into batches."""
    def __init__(self, samples, dataset):
        random.shuffle(samples)
        self.samples = samples
        self.idx = 0
        self.dataset = dataset  # for access to volume, mask, labels, normalization & config
    
    def remaining(self):
        return len(self.samples) - self.idx
    
    def has_next(self):
        return self.idx < len(self.samples)
    
    def next_sample(self):
        if not self.has_next():
            return None
        rec = self.samples[self.idx]
        self.idx += 1
        return self._load(rec)
    
    def _load(self, rec):
        # rec: {"z":..., "y":..., "x":..., "label":0/1}
        cfg = self.dataset.config
        tile = cfg.data.tile_size
        depth = cfg.data.depth
        z = rec["z"]
        y = rec["y"]
        x = rec["x"]
        # Safety bounds
        if z + depth > self.dataset.volume.shape[0]:
            return None
        block = np.array(self.dataset.volume[
            z:z+depth,
            y:y+tile,
            x:x+tile
        ]).astype(np.float32)
        mask_tile = self.dataset.mask[y:y+tile, x:x+tile]
        # Normalize (reuse dataset method)
        norm_block = (block - self.dataset.global_mean) / self.dataset.global_std
        mask_exp = np.expand_dims(mask_tile, 0)
        mask_exp = np.broadcast_to(mask_exp, norm_block.shape)
        norm_block[mask_exp == 0] = 0
        norm_block = (norm_block - self.dataset.global_min) / (self.dataset.global_max - self.dataset.global_min)
        norm_block = np.clip(norm_block, 0, 1)
        block_tensor = torch.tensor(norm_block, dtype=torch.float32).unsqueeze(0)  # (1,D,H,W)
        label_tensor = torch.tensor([float(rec["label"])], dtype=torch.float32)
        mask_tensor = torch.tensor(mask_tile, dtype=torch.float32)
        return block_tensor, label_tensor, mask_tensor

class HardMiningManager:
    """Handles reading mined examples via reservoir sampling."""
    def __init__(self, log_dir):
        self.log_dir = log_dir
    
    def _epoch_file(self, epoch):
        return os.path.join(self.log_dir, f"hard_mining_epoch_{epoch}.jsonl")
    
    def mined_file_exists(self, epoch):
        return os.path.exists(self._epoch_file(epoch))
    
    def sample_for_epoch(self, prev_epoch, target_count):
        """Reservoir sample from previous epoch mining file."""
        path = self._epoch_file(prev_epoch)
        if target_count <= 0 or not os.path.exists(path):
            return []
        reservoir = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                except:
                    continue
                if item.get("_type") == "meta":  # skip meta lines
                    continue
                if len(reservoir) < target_count:
                    reservoir.append(item)
                else:
                    j = random.randint(0, i)
                    if j < target_count:
                        reservoir[j] = item
        return reservoir
