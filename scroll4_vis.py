import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
import cv2
import zarr
import json

from utils.config import Config
from utils.model import create_model
from utils.visualizer import group_by_depth, predict_tiles

def load_scroll4_region(config: Config):
    """open scroll4 zarr volume and mask and return region ranges"""
    sid = config.data.scroll4_id
    zarr_path = os.path.join(config.data.zarr_path, f"{sid}.zarr")
    vol = zarr.open(zarr_path, mode='r')
    _, H, W = map(int, vol.shape)
    mask_path = f"./masks/{sid}.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
    y_range = (6500 if H > 6500 else 0, H)
    x_range = (0, min(5000, W))
    return vol, mask, y_range, x_range

def get_or_compute_norm(vol, mask, seg_id: str):
    """compute or load cached normalization stats consistent with visualizer"""
    cache_path = "./norm_cache.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            if seg_id in cache:
                s = cache[seg_id]
                return s["mean"], s["std"], s["min"], s["max"]
        except Exception:
            pass

    total_sum, total_sq_sum, total_count = 0.0, 0.0, 0
    for z in tqdm(range(vol.shape[0]), desc="norm pass1", leave=False):
        chunk = vol[z, :, :]
        valid = chunk[mask[:, :] > 0]
        if valid.size == 0:
            continue
        total_sum += float(np.sum(valid, dtype=np.float64))
        total_sq_sum += float(np.sum(np.square(valid, dtype=np.float64), dtype=np.float64))
        total_count += int(valid.size)
    if total_count == 0:
        raise ValueError("no valid pixels found for normalization")
    mean = total_sum / total_count
    std = float(np.sqrt(max((total_sq_sum / total_count) - (mean * mean), 1e-12)))

    g_min, g_max = float('inf'), float('-inf')
    for z in tqdm(range(vol.shape[0]), desc="norm pass2", leave=False):
        chunk = vol[z, :, :]
        valid = chunk[mask[:, :] > 0]
        if valid.size == 0:
            continue
        norm = (valid.astype(np.float64) - mean) / std
        g_min = min(g_min, float(norm.min()))
        g_max = max(g_max, float(norm.max()))

    try:
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
        cache[seg_id] = {"mean": mean, "std": std, "min": g_min, "max": g_max}
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=4)
    except Exception:
        pass
    return mean, std, g_min, g_max

def generate_tile_coords(z_range, y_range, x_range, config: Config, mask):
    """generate valid tile coords filtered by mask like in visualizer"""
    z0, z1 = z_range
    y0, y1 = y_range
    x0, x1 = x_range
    depth = config.data.depth
    tile = config.data.tile_size
    z_span = max(0, z1 - z0 - depth + 1)
    y_span = max(0, y1 - y0 - tile + 1)
    x_span = max(0, x1 - x0 - tile + 1)
    coords = []
    z_step = max(1, depth // 2)
    for d in range(0, z_span, z_step):
        if z0 + d + depth > z1:
            continue
        for y in range(0, y_span, tile):
            for x in range(0, x_span, tile):
                m_tile = mask[y0 + y:y0 + y + tile, x0 + x:x0 + x + tile]
                if np.sum(m_tile) > 0:
                    coords.append((d, y, x))
    return coords

def main(config: Config, model_path: str):
    """
    Main function to load data, run inference on Scroll 4, and visualize the result.
    """
    # --- 1. Setup ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print("Loading model...")
    model, _ = create_model(config)
    model.load_state_dict(torch.load(model_path))
    model.to(config.device)
    model.eval()

    # --- 2. Load Data and Normalization ---
    print("Loading Scroll 4 data...")
    volume, mask, y_range, x_range = load_scroll4_region(config)

    print("Calculating normalization stats for Scroll 4...")
    mean, std, min_val, max_val = get_or_compute_norm(volume, mask, str(config.data.scroll4_id))

    # --- 3. Generate Coordinates and Predict ---
    print("Generating tile coordinates for the entire scroll...")
    # scan full depth with half-depth step (0..D with step depth//2)
    z_range = (0, volume.shape[0])
    all_coords = generate_tile_coords(z_range, y_range, x_range, config, mask)
    grouped_coords = group_by_depth(all_coords)
    
    all_predictions = []
    for depth_start in sorted(grouped_coords.keys()):
        block_coords = grouped_coords[depth_start]
        print(f"\nPredicting for depth block starting at {depth_start}...")

        prediction_map = predict_tiles(
            config, model, volume, mask, block_coords, y_range, x_range,
            depth_start, "scroll4", mean, std, min_val, max_val
        )
        all_predictions.append(prediction_map)

    # --- 4. Combine and Visualize Predictions ---
    if not all_predictions:
        print("No predictions were generated.")
        return

    print("Combining prediction maps from all depth blocks...")
    # Combine predictions from all depth blocks by taking the maximum value for each pixel
    final_prediction = np.max(np.stack(all_predictions, axis=0), axis=0)

    print("Displaying final prediction map...")
    plt.figure(figsize=(20, 12))
    plt.imshow(final_prediction, cmap='inferno')
    plt.title(f"Full Prediction Map for Scroll 4\n(Model: {os.path.basename(model_path)})")
    plt.colorbar(label="Ink Prediction Score")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model predictions on the full Scroll 4.")
    parser.add_argument(
        "-m", "--model_path", 
        type=str, 
        default="models/best_model_f1.pth", 
        help="Path to the trained model file."
    )
    args = parser.parse_args()
    
    config = Config()
    main(config, args.model_path)
