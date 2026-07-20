import cv2
import numpy as np
import os
from pathlib import Path

def erode_ink_labels(input_folder="inklabels", output_folder="eroded_inklabels", erosion_size=3, iterations=1):
    """
    Apply erosion to grayscale ink label images to shrink white regions (1s).
    
    Parameters:
    input_folder (str): Path to folder containing input PNG files
    output_folder (str): Path to folder where eroded images will be saved
    erosion_size (int): Size of the erosion kernel (odd numbers work best)
    iterations (int): Number of times to apply erosion
    """
    
    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)
    
    # Get all PNG files from input folder
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return
    
    png_files = list(input_path.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in '{input_folder}' folder!")
        return
    
    # Create erosion kernel
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    
    print(f"Processing {len(png_files)} PNG files...")
    print(f"Erosion kernel size: {erosion_size}x{erosion_size}")
    print(f"Iterations: {iterations}")
    
    for i, file_path in enumerate(png_files):
        try:
            # Read the image in grayscale
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not read {file_path.name}")
                continue
            
            # Apply erosion
            eroded_img = cv2.erode(img, kernel, iterations=iterations)
            
            # Save the eroded image with the same filename
            output_path = Path(output_folder) / file_path.name
            success = cv2.imwrite(str(output_path), eroded_img)
            
            if success:
                print(f"Processed ({i+1}/{len(png_files)}): {file_path.name}")
            else:
                print(f"Error saving: {file_path.name}")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    print(f"\nCompleted! Eroded images saved in '{output_folder}' folder.")

if __name__ == "__main__":
    # Configuration - modify these values as needed
    INPUT_FOLDER = "inklabels"
    OUTPUT_FOLDER = "eroded_inklabels"
    EROSION_SIZE = 3  # Size of erosion kernel (3x3, 5x5, etc.)
    ITERATIONS = 12    # Number of erosion iterations
    
    # Run the erosion process
    erode_ink_labels(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        erosion_size=EROSION_SIZE,
        iterations=ITERATIONS
    )