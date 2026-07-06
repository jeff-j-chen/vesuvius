import json, os, math, random
import torch
import numpy as np

class HardMiningInjector:
    """iterator providing hard examples to inject into batches.

    accepts either a single InkVolumeDataset (single-scroll, back-compat) or a
    dict {scroll_id: InkVolumeDataset} (multi-scroll). each mined record carries a
    scroll_id, which is used to route it back to the matching scroll's volume,
    mask, normalization stats and spatial bounds. records without a scroll_id (old
    files) fall back to the default dataset."""
    def __init__(self, samples, datasets):
        """initializes the hard mining injector"""
        random.shuffle(samples)
        self.samples = samples
        self.idx = 0

        # normalize the datasets argument into a {scroll_id: dataset} map plus a
        # default fallback (used for single-scroll or untagged records)
        if isinstance(datasets, dict):
            self._ds_map = {int(k): v for k, v in datasets.items()}
            self._default_ds = next(iter(self._ds_map.values()), None)
        else:
            self._ds_map = {}
            self._default_ds = datasets

        # the dataset.vol property only returns a real handle inside dataloader
        # workers; in the MAIN process (where injection happens) it returns None for
        # zarr-backed volumes. so the injector opens and caches its own handle here.
        self._vol_cache = {}

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

    def _resolve_ds(self, rec):
        """pick the dataset matching this record's scroll_id, else the default"""
        sid = rec.get("scroll_id")
        if sid is not None:
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                sid = None
        if sid is not None and sid in self._ds_map:
            return self._ds_map[sid]
        return self._default_ds

    def _volume(self, ds):
        """return a main-process-safe volume handle for a dataset (cached).

        preloaded numpy volumes are used directly; zarr-backed datasets are opened
        once from their stored path (the lazy .vol property only opens inside workers)."""
        key = id(ds)
        if key in self._vol_cache:
            return self._vol_cache[key]
        vol = getattr(ds, "_vol_obj", None)
        if not isinstance(vol, np.ndarray):
            zp = getattr(ds, "_zarr_path", None)
            if zp:
                import zarr as _zarr
                vol = _zarr.open(zp, mode='r')
        self._vol_cache[key] = vol
        return vol

    def _load(self, rec):
        """loads a single hard-mined record from its scroll's volume"""
        ds = self._resolve_ds(rec)
        if ds is None:
            return None
        cfg = ds.c
        tile = cfg.data.tile_size
        depth = cfg.data.depth
        z, y, x = rec["z"], rec["y"], rec["x"]

        # validate spatial bounds against the resolved dataset's range
        y_min, y_max = ds.y_start, ds.y_end
        x_min, x_max = ds.x_start, ds.x_end

        if not (y_min <= y < y_max - tile and x_min <= x < x_max - tile):
            return None

        vol = self._volume(ds)
        if vol is None or z + depth > vol.shape[0]:
            return None

        # fetch block and mask
        try:
            block = np.asarray(vol[z:z+depth, y:y+tile, x:x+tile])
            mask_tile = ds.mask[y:y+tile, x:x+tile]
        except Exception:
            return None

        if np.sum(mask_tile) == 0:
            return None

        # normalize block using the resolved dataset's pre-computed stats
        norm_block = ds._normalize_block(block)
        
        # create tensors
        block_tensor = torch.tensor(norm_block, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([float(rec["label"])], dtype=torch.float32)
        mask_tensor = torch.tensor(mask_tile, dtype=torch.float32)
        
        return block_tensor, label_tensor, mask_tensor

class HardMiningManager:
    """handles reading mined examples via reservoir sampling"""
    def __init__(self, hm_dir="./hard_negs"):
        self.hm_dir = hm_dir
    
    def _epoch_file(self, epoch):
        """returns the filename for a given epoch's hard mining data"""
        return f"hard_mining_epoch_{epoch}.jsonl"
    
    def sample_for_epoch(self, prev_epoch, target_count):
        """reservoir samples from previous epoch mining file"""
        if target_count <= 0:
            return []
        
        filename = self._epoch_file(prev_epoch)
        path = os.path.join(self.hm_dir, filename)

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

    def sample_for_epoch_scrolls(self, prev_epoch, target_count, scroll_ids):
        """reservoir-sample target_count records pooled across per-scroll mining dirs.

        mining files are written per fragment at <base>/scroll_<sid>/hard_mining_epoch_N.jsonl
        (see visualizer._hard_mining_dir). this pools every scroll's file for the
        epoch, tags each record with its scroll_id so the injector can route it, and
        dedups on (scroll_id, z, y, x, label) so identical coordinates on different
        scrolls stay distinct. sampling proportionally reflects each scroll's supply
        of hard examples. also works for a single scroll (list of length 1)."""
        if target_count <= 0:
            return []

        reservoir = []
        seen = set()
        unique_idx = 0
        per_scroll_counts = {}

        for sid in scroll_ids:
            sid = int(sid)
            path = os.path.join(self.hm_dir, f"scroll_{sid}", self._epoch_file(prev_epoch))
            if not os.path.exists(path):
                print(f"[HARD] no mining file for scroll {sid} at '{path}'")
                continue

            with open(path, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("_type") == "meta":
                        continue

                    # dedup includes scroll_id so cross-scroll coord collisions are kept
                    try:
                        k = (sid, int(item["z"]), int(item["y"]), int(item["x"]), int(item["label"]))
                    except Exception:
                        continue
                    if k in seen:
                        continue
                    seen.add(k)

                    # ensure the record is tagged so the injector can route it
                    item["scroll_id"] = sid
                    per_scroll_counts[sid] = per_scroll_counts.get(sid, 0) + 1

                    if len(reservoir) < target_count:
                        reservoir.append(item)
                    else:
                        j = random.randint(0, unique_idx)
                        if j < target_count:
                            reservoir[j] = item
                    unique_idx += 1

        print(f"[HARD] pooled mining epoch={prev_epoch} per_scroll={per_scroll_counts} "
              f"unique={unique_idx} target={target_count} sampled={len(reservoir)}")
        return reservoir
