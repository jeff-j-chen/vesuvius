import json, os, math, random
import torch
import numpy as np

class HardMiningInjector:
    """iterator providing hard examples to inject into batches"""
    def __init__(self, samples, dataset):
        """initializes the hard mining injector"""
        random.shuffle(samples)
        self.samples = samples
        self.dataset = dataset
        self.idx = 0
        
        # diagnostics
        self.used = 0
        self.skipped = 0

    def remaining(self):
        """returns the number of samples not yet consumed"""
        return len(self.samples) - self.idx
    
    def has_next(self):
        """checks if there are more samples to consume"""
        return self.idx < len(self.samples)
    
    def next_sample(self):
        """loads and returns the next valid sample"""
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

    def stats(self):
        """returns consumption statistics"""
        return {
            'requested': len(self.samples),
            'consumed': self.idx,
            'used': self.used,
            'skipped': self.skipped,
            'remaining': self.remaining()
        }
    
    def _load(self, rec):
        """loads a single hard-mined record"""
        # unpack record and config
        cfg = self.dataset.c
        tile = cfg.data.tile_size
        depth = cfg.data.depth
        z, y, x = rec["z"], rec["y"], rec["x"]
        
        # validate spatial bounds against the training dataset's range
        y_min, y_max = self.dataset.y_start, self.dataset.y_end
        x_min, x_max = self.dataset.x_start, self.dataset.x_end
        
        if not (y_min <= y < y_max - tile and x_min <= x < x_max - tile):
            return None
        if z + depth > self.dataset.vol.shape[0]:
            return None
            
        # fetch block and mask
        try:
            block = self.dataset.vol[z:z+depth, y:y+tile, x:x+tile]
            mask_tile = self.dataset.mask[y:y+tile, x:x+tile]
        except Exception:
            return None
            
        if np.sum(mask_tile) == 0:
            return None
            
        # normalize block using dataset's pre-computed stats
        norm_block = self.dataset._normalize_block(block)
        
        # create tensors
        block_tensor = torch.tensor(norm_block, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([float(rec["label"])], dtype=torch.float32)
        mask_tensor = torch.tensor(mask_tile, dtype=torch.float32)
        
        return block_tensor, label_tensor, mask_tensor

class HardMiningManager:
    """handles reading mined examples via reservoir sampling"""
    
    def _epoch_file(self, epoch):
        """returns the filename for a given epoch's hard mining data"""
        return f"hard_mining_epoch_{epoch}.jsonl"
    
    def sample_for_epoch(self, prev_epoch, target_count):
        """reservoir samples from previous epoch mining file"""
        if target_count <= 0:
            return []
        
        filename = self._epoch_file(prev_epoch)
        path = os.path.join("./hard_negs", filename)

        if not os.path.exists(path):
            print(f"[error] mining file not found at '{path}'")
            return []

        reservoir = []
        scanned = 0
        # track unique keys within this file to prevent duplicates
        seen = set()
        unique_idx = 0
        
        with open(path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                if item.get("_type") == "meta":
                    continue

                # dedup by spatial key and label
                try:
                    k = (int(item["z"]), int(item["y"]), int(item["x"]), int(item["label"]))
                except Exception:
                    continue
                if k in seen:
                    continue
                seen.add(k)

                scanned += 1

                # fill the reservoir
                if len(reservoir) < target_count:
                    reservoir.append(item)
                else:
                    # replace elements with decreasing probability using unique index
                    j = random.randint(0, unique_idx)
                    if j < target_count:
                        reservoir[j] = item

                unique_idx += 1
                        
        print(f"loaded mining file '{path}' unique_scanned={unique_idx} target={target_count} sampled={len(reservoir)}")
        return reservoir
