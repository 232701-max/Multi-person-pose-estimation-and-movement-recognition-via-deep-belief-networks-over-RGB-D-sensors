import cv2
import numpy as np
import os
from rembg import remove
from PIL import Image
import io


def preprocess_rgb_image(rgb_image_path):
    """
    Removes the background from an RGB image using the rembg library.
    """
    with open(rgb_image_path, "rb") as f:
        rgb_data = f.read()
    background_removed = remove(rgb_data)
    background_removed_image = Image.open(io.BytesIO(background_removed)).convert("RGB")
    return np.array(background_removed_image)


def refine_depth_map_16bit(rgb_image, depth_image_path, output_size):
    """
    Processes a 16-bit depth map with a background-removed RGB image.
    """
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Unable to load depth image: {depth_image_path}")

    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_resized = cv2.resize(depth_normalized, output_size)
    rgb_resized = cv2.resize(rgb_image, output_size)

    gray_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_rgb, 1, 255, cv2.THRESH_BINARY)

    segmented_depth = cv2.bitwise_and(depth_resized, depth_resized, mask=binary_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_segment = cv2.morphologyEx(segmented_depth, cv2.MORPH_CLOSE, kernel)

    return cleaned_segment


def process_all_folders(parent_rgb_dir, parent_depth_dir, parent_output_dir, output_size):
    """
    Processes all subfolders within the given parent directories.
    Each subfolder inside the parent directory will be processed automatically.
    """
    # Ensure the parent output directory exists
    os.makedirs(parent_output_dir, exist_ok=True)

    # List all activity subfolders in RGB and Depth parent directories
    rgb_subfolders = sorted([f for f in os.listdir(parent_rgb_dir) if os.path.isdir(os.path.join(parent_rgb_dir, f))])
    depth_subfolders = sorted(
        [f for f in os.listdir(parent_depth_dir) if os.path.isdir(os.path.join(parent_depth_dir, f))])

    for activity in rgb_subfolders:
        if activity not in depth_subfolders:
            print(f"Warning: No matching depth folder found for {activity}, skipping.")
            continue

        # Define paths for RGB, Depth, and Output directories
        rgb_dir = os.path.join(parent_rgb_dir, activity)
        depth_dir = os.path.join(parent_depth_dir, activity)
        output_dir = os.path.join(parent_output_dir, activity)

        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        print(f"Processing activity: {activity}")

        # List all image files
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith('.png')])

        if len(rgb_files) != len(depth_files):
            print(
                f"Warning: Mismatch between RGB and depth image counts in {activity} ({len(rgb_files)} vs {len(depth_files)})")

        for rgb_file, depth_file in zip(rgb_files, depth_files):
            print(f"Processing: {rgb_file} & {depth_file}")

            rgb_path = os.path.join(rgb_dir, rgb_file)
            depth_path = os.path.join(depth_dir, depth_file)
            output_path = os.path.join(output_dir, f"segmented_{depth_file}")

            try:
                background_removed_rgb = preprocess_rgb_image(rgb_path)
                segmented_depth = refine_depth_map_16bit(background_removed_rgb, depth_path, output_size)
                cv2.imwrite(output_path, segmented_depth)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {rgb_file} and {depth_file}: {e}")


# Example usage
parent_rgb_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\1- Image Frames\5- Fight"
parent_depth_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\1- Image Frames\5- Fight"
parent_output_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\6- Body Segmentation\5- Fight"

output_dimensions = (640, 480)  # Resize dimensions (width, height)

process_all_folders(parent_rgb_directory, parent_depth_directory, parent_output_directory, output_dimensions)
