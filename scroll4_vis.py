import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import argparse
import os

from utils.config import Config
from utils.dataloader import load_scroll4_data, get_or_compute_normalization, generate_tile_coords
from utils.model import create_model
from utils.visualizer import group_by_depth, predict_tiles

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
    volume, mask, y_range, x_range = load_scroll4_data(config)
    
    print("Calculating normalization stats for Scroll 4...")
    mean, std, min_val, max_val = get_or_compute_normalization(
        config.data.scroll4_segment_id, volume, mask
    )

    # --- 3. Generate Coordinates and Predict ---
    print("Generating tile coordinates for the entire scroll...")
    z_range = (config.data.start_level, config.data.end_level)
    all_coords = generate_tile_coords(z_range, y_range, x_range, config, volume, mask)
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
    args = parser.parse_args()
    
    config = Config()
    main(config, args.model_path)
