import json, os, math, random
import torch
import numpy as np

class HardMiningInjector:
    """Iterator providing hard examples to inject into batches."""
    def __init__(self, samples, dataset):
        random.shuffle(samples)
        self.samples = samples
        self.idx = 0
        self.dataset = dataset
        # --- diagnostics ---
        self.used = 0
        self.skipped = 0
        self.skip_reasons = {
            'z_overflow': 0,
            'y_oob': 0,
            'x_oob': 0,
            'mask_empty': 0,
            'other': 0
        }

    def remaining(self):
        return len(self.samples) - self.idx
    
    def has_next(self):
        return self.idx < len(self.samples)
    
    def next_sample(self):
        if not self.has_next():
            return None
        rec = self.samples[self.idx]
        self.idx += 1
        loaded = self._load(rec)
        if loaded is None:
            self.skipped += 1
        else:
            self.used += 1
        return loaded

    def peek(self, n=5):
        """Return first n raw sample records (without advancing)."""
        return self.samples[:min(n, len(self.samples))]

    def stats(self):
        return {
            'requested': len(self.samples),
            'consumed': self.idx,
            'used': self.used,
            'skipped': self.skipped,
            'skip_reasons': dict(self.skip_reasons),
            'remaining': self.remaining()
        }
    
    def _load(self, rec):
        # rec: {"z":..., "y":..., "x":..., "label":0/1}
        cfg = self.dataset.config
        tile = cfg.data.tile_size
        depth = cfg.data.depth
        z = rec["z"]; y = rec["y"]; x = rec["x"]
        # Bounds for the TRAIN dataset (global coords)
        y_min, y_max = self.dataset.y_start, self.dataset.y_end
        x_min, x_max = self.dataset.x_start, self.dataset.x_end
        # Validate spatial bounds (skip but record reason if out-of-range)
        if not (y_min <= y <= y_max - tile):
            self.skip_reasons['y_oob'] += 1
            return None
        if not (x_min <= x <= x_max - tile):
            self.skip_reasons['x_oob'] += 1
            return None
        # Depth check
        if z + depth > self.dataset.volume.shape[0]:
            self.skip_reasons['z_overflow'] += 1
            return None
        try:
            block = np.array(self.dataset.volume[
                z:z+depth,
                y:y+tile,
                x:x+tile
            ]).astype(np.float32)
        except Exception:
            self.skip_reasons['other'] += 1
            return None
        mask_tile = self.dataset.mask[y:y+tile, x:x+tile]
        if np.sum(mask_tile) == 0:
            self.skip_reasons['mask_empty'] += 1
            return None
        # Normalize (reuse dataset stats)
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
    
    def _epoch_file(self, epoch):
        return f"hard_mining_epoch_{epoch}.jsonl"
    
    def sample_for_epoch(self, prev_epoch, target_count):
        """Reservoir sample from previous epoch mining file (supports legacy CWD location)."""
        if target_count <= 0:
            return []
        
        filename = self._epoch_file(prev_epoch)
        path = os.path.join("./hard_negs", filename)

        if not os.path.exists(path):
            print(f"[ERROR] Mining file not found at '{path}'")
            return []

        reservoir = []
        scanned = 0
        with open(path, "r") as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                except:
                    continue
                if item.get("_type") == "meta":
                    continue
                scanned += 1
                if len(reservoir) < target_count:
                    reservoir.append(item)
                else:
                    j = random.randint(0, i)
                    if j < target_count:
                        reservoir[j] = item
        print(f"Loaded mining file '{path}' scanned={scanned} target={target_count} sampled={len(reservoir)}")
        return reservoir
