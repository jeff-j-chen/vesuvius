import os, json, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter

HARD_NEGS_DIR = "hard_negs"

def ensure_hard_negs_dir():
    os.makedirs(HARD_NEGS_DIR, exist_ok=True)
    return HARD_NEGS_DIR

def mining_filename(epoch: int):
    return os.path.join(HARD_NEGS_DIR, f"hard_mining_epoch_{epoch}.jsonl")

def list_mining_files(upto_epoch):
    """List mining files (sorted by epoch)."""
    if not os.path.isdir(HARD_NEGS_DIR):
        return []
    files = []
    for fn in os.listdir(HARD_NEGS_DIR):
        if fn.startswith("hard_mining_epoch_") and fn.endswith(".jsonl"):
            try:
                ep = int(fn[len("hard_mining_epoch_"):-len(".jsonl")])
                if upto_epoch is None or ep <= upto_epoch:
                    files.append((ep, os.path.join(HARD_NEGS_DIR, fn)))
            except ValueError:
                continue
    return [p for _, p in sorted(files)]

def load_mining_records(json_path):
    """Load records (exclude meta)."""
    records = []
    with open(json_path, "r") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except:
                continue
            if obj.get("_type") == "meta":
                continue
            # required keys: z,y,x,label
            if not all(k in obj for k in ("z","y","x","label")):
                continue
            records.append(obj)
    return records

class HardMinedDataset(Dataset):
    """
    Dataset over mined hard examples. Each item:
      block_tensor (1,D,H,W), label (1,), mask (1,)
    Changes:
      - Use stored record 'label' if present; fallback to recomputed.
      - Single scalar mask (aligns with main training loop expectation).
      - Remove second min-max scaling & pixel-level masking to match base pipeline.
    """
    def __init__(self, records, volume, mask, labels,
                 global_mean, global_std, global_min, global_max,
                 config):
        self.records = records
        self.volume = volume
        self.mask = mask
        self.labels = labels
        self.gmean = global_mean
        self.gstd = global_std if global_std != 0 else 1.0
        # Keep gmin/gmax for potential future diagnostics but not applied.
        self.cfg = config
        self.tile = config.data.tile_size
        self.depth = config.data.depth
        self._did_debug = 0
        self.labels_arr = labels  # 2D ground truth map (H,W) assumed
        self.verify = True
        self.override_on_mismatch = True
        self.mismatch_log_limit = 15
        self._mismatch_count = 0
        self._skipped_edge = 0
        self._skipped_shape = 0
        self._skipped_oob = 0
        self._reported_final = False
        # Optional class balance (downsample majority)
        balance_cfg = getattr(config, "hard_mining_balance", None)
        if balance_cfg:
            self.records = self._apply_class_balance(self.records, balance_cfg)

    def _apply_class_balance(self, recs, balance_cfg):
        # balance_cfg: dict like {"max_ratio": 2.0, "min_per_class": 50}
        max_ratio = balance_cfg.get("max_ratio", 2.0)
        min_per = balance_cfg.get("min_per_class", 0)
        by_class = {0: [], 1: []}
        for r in recs:
            rl = 1 if r.get("label", 0) >= 0.5 else 0
            by_class[rl].append(r)
        c0, c1 = len(by_class[0]), len(by_class[1])
        if c0 == 0 or c1 == 0:
            return recs  # cannot balance
        maj_class = 0 if c0 > c1 else 1
        min_class = 1 - maj_class
        major_list = by_class[maj_class]
        minor_list = by_class[min_class]
        desired_major = int(min(len(major_list), max_ratio * len(minor_list)))
        # Keep at least min_per each
        if len(minor_list) < min_per:
            return recs
        if len(major_list) < min_per:
            return recs
        if desired_major < len(major_list):
            rng = np.random.default_rng(seed=42)
            keep_idx = rng.choice(len(major_list), size=desired_major, replace=False)
            major_list = [major_list[i] for i in keep_idx]
        balanced = major_list + minor_list
        np.random.default_rng(seed=123).shuffle(balanced)
        print(f"[HardMinedDataset][balance] c0={c0} c1={c1} -> balanced {Counter(1 if r.get('label',0)>=0.5 else 0 for r in balanced)}")
        return balanced

    def _final_report(self):
        if self._reported_final:
            return
        self._reported_final = True
        print(f"[HardMinedDataset][summary] mismatches={self._mismatch_count} skipped_edge={self._skipped_edge} skipped_shape={self._skipped_shape} skipped_oob={self._skipped_oob}")

    def __len__(self):
        return len(self.records)

    def _recompute_label(self, y, x):
        tile_slice = self.labels_arr[y:y+self.tile, x:x+self.tile]
        if tile_slice.shape != (self.tile, self.tile):
            return None
        return 1.0 if np.any(tile_slice > 0.5) else 0.0

    def __getitem__(self, idx):
        rec = self.records[idx]
        z = rec["z"]; y = rec["y"]; x = rec["x"]

        # Bounds checks
        if z < 0 or y < 0 or x < 0 or \
           z + self.depth > self.volume.shape[0] or \
           y + self.tile > self.volume.shape[1] or \
           x + self.tile > self.volume.shape[2]:
            self._skipped_oob += 1
            # Fallback: return a zero block (still consistent shape) with mask=0 so loss ignores it
            zero_block = torch.zeros(1, self.depth, self.tile, self.tile, dtype=torch.float32)
            return zero_block, torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

        block_np = self.volume[z:z+self.depth, y:y+self.tile, x:x+self.tile]
        if block_np.shape != (self.depth, self.tile, self.tile):
            self._skipped_shape += 1
            zero_block = torch.zeros(1, self.depth, self.tile, self.tile, dtype=torch.float32)
            return zero_block, torch.zeros(1), torch.zeros(1)

        block = np.asarray(block_np, dtype=np.float32)
        norm = (block - self.gmean) / self.gstd

        stored = rec.get("label", None)
        if stored is None:
            print("failed to find label in record, recomputing...")
            rec_label = self._recompute_label(y, x)
        else:
            try:
                rec_label = float(stored)
            except (ValueError, TypeError):
                rec_label = None

        if self.verify:
            gt = self._recompute_label(y, x)
            if gt is None:
                self._skipped_edge += 1
                zero_block = torch.zeros(1, self.depth, self.tile, self.tile, dtype=torch.float32)
                return zero_block, torch.zeros(1), torch.zeros(1)
            # Decide final label
            if rec_label is None:
                rec_label = gt
            else:
                # Binarize stored value
                bin_stored = 1.0 if rec_label >= 0.5 else 0.0
                if bin_stored != gt:
                    if self.override_on_mismatch:
                        if self._mismatch_count < self.mismatch_log_limit:
                            print(f"[HardMinedDataset][mismatch] idx={idx} stored={rec_label:.3f} gt={gt} -> override")
                        self._mismatch_count += 1
                        rec_label = gt
                    else:
                        rec_label = bin_stored
                else:
                    rec_label = gt  # ensure exact 0/1
        else:
            # If not verifying, still clamp to {0,1}
            if rec_label is None:
                rec_label = 0.0
            rec_label = 1.0 if rec_label >= 0.5 else 0.0

        if not np.isfinite(rec_label):
            rec_label = 0.0

        block_tensor = torch.from_numpy(norm).unsqueeze(0)  # (1,D,H,W)
        label_tensor = torch.tensor([rec_label], dtype=torch.float32)
        mask_tensor = torch.tensor([1.0], dtype=torch.float32)

        if self._did_debug < 3:
            self._did_debug += 1
            with torch.no_grad():
                mn = float(block_tensor.mean()); sd = float(block_tensor.std())
                print(f"[HardMinedDataset][dbg] idx={idx} mean={mn:.4f} std={sd:.4f} label={rec_label}")

        if idx == len(self.records) - 1:
            self._final_report()

        return block_tensor, label_tensor, mask_tensor

def create_hard_mined_loader(records, volume, mask, labels, stats, config):
    ds = HardMinedDataset(records, volume, mask, labels, stats["mean"], stats["std"], stats["min"], stats["max"], config)
    loader = DataLoader(
        ds,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=getattr(config.dataloader, "drop_last", False)
    )
    return ds, loader

# === New debugging utilities ===

def _recompute_tile_label(labels_arr, y, x, tile, threshold=0.5):
    tile_slice = labels_arr[y:y+tile, x:x+tile]
    if tile_slice.shape[0] != tile or tile_slice.shape[1] != tile:
        # Edge truncation -> ambiguous; mark as None
        return None
    return 1.0 if np.any(tile_slice > threshold) else 0.0

def analyze_mining_files(mining_files, labels_arr, volume_depth, tile_size, depth, sample_cap=5000):
    """
    Diagnose potential label inversions or coordinate issues.
    Returns dict with stats; prints human-readable summary.
    """
    checked = 0
    mismatch = 0
    inverted_like = False
    edge_trunc = 0
    coord_oob = 0
    depth_oob = 0
    label_counts = {0:0, 1:0}
    rec_vs_actual = { (0,0):0, (0,1):0, (1,0):0, (1,1):0 }  # (record_label, actual_label)
    for mf in mining_files:
        try:
            with open(mf, "r") as f:
                for line in f:
                    if checked >= sample_cap:
                        break
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        obj=json.loads(line)
                    except:
                        continue
                    if obj.get("_type")=="meta":
                        continue
                    if not all(k in obj for k in ("z","y","x","label")):
                        continue
                    z,y,x,obj_label = obj["z"], obj["y"], obj["x"], obj["label"]
                    # Coordinate checks
                    if z < 0 or y < 0 or x < 0:
                        coord_oob += 1
                        continue
                    if z + depth > volume_depth:
                        depth_oob += 1
                        continue
                    actual = _recompute_tile_label(labels_arr, y, x, tile_size)
                    if actual is None:
                        edge_trunc += 1
                        continue
                    rl = 1 if obj_label >= 0.5 else 0
                    label_counts[rl] += 1
                    rec_vs_actual[(rl,int(actual))] += 1
                    if rl != actual:
                        mismatch += 1
                    checked += 1
                if checked >= sample_cap:
                    break
        except Exception as e:
            print(f"[MiningDebug] Failed reading {mf}: {e}")
    mismatch_rate = mismatch/checked if checked else 0
    if mismatch_rate > 0.8 and rec_vs_actual[(0,1)] + rec_vs_actual[(1,0)] == mismatch:
        inverted_like = True
    print("[MiningDebug] ===== Mining Label Integrity Report =====")
    print(f"[MiningDebug] Files inspected: {len(mining_files)}  Samples checked: {checked}  Sample cap: {sample_cap}")
    print(f"[MiningDebug] Record label distribution: {label_counts}")
    print(f"[MiningDebug] Pair counts (record -> actual): {rec_vs_actual}")
    print(f"[MiningDebug] Mismatch rate: {mismatch_rate:.3f}")
    if inverted_like:
        print("[MiningDebug][ALERT] High mismatch rate suggests possible global inversion or systematic mislabeling.")
    print(f"[MiningDebug] Edge-truncated tiles skipped: {edge_trunc}")
    print(f"[MiningDebug] Coord OOB skipped: {coord_oob}  Depth OOB skipped: {depth_oob}")
    print("[MiningDebug] =========================================")
    return {
        "checked": checked,
        "mismatch_rate": mismatch_rate,
        "label_counts": label_counts,
        "pairs": rec_vs_actual,
        "inversion_suspected": inverted_like
    }
