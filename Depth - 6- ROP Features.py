import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Input and output directories
segmented_input_dir = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\6- Body Segmentation"
feature_output_dir = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\8- ROP Features\Excel"
visual_output_dir = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\8- ROP Features\Visuals"

# Ensure output directories exist
os.makedirs(feature_output_dir, exist_ok=True)
os.makedirs(visual_output_dir, exist_ok=True)


def replace_black_with_white(image):
    """Replaces black background (zero pixels) with white in a grayscale image."""
    image_with_white_bg = image.copy()
    image_with_white_bg[image_with_white_bg == 0] = 255  # Convert black (0) to white (255)
    return image_with_white_bg


def random_occupancy_pattern(segmented_image, grid_size=(10, 10), num_blocks=50, visualize=False,
                             visual_output_path=None):
    """Extracts Random Occupancy Pattern (ROP) features and visualizes grid selection."""
    height, width = segmented_image.shape
    block_height = height // grid_size[0]
    block_width = width // grid_size[1]

    valid_blocks = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])
                    if np.count_nonzero(segmented_image[i * block_height:(i + 1) * block_height,
                                        j * block_width:(j + 1) * block_width]) > 0]

    if len(valid_blocks) < num_blocks:
        num_blocks = len(valid_blocks)

    selected_blocks = random.sample(valid_blocks, num_blocks)

    if visualize and visual_output_path:
        fig, ax = plt.subplots()
        ax.imshow(segmented_image, cmap='gray')

        for i in range(1, grid_size[0]):
            ax.axhline(i * block_height, color='red', linestyle='--')
        for j in range(1, grid_size[1]):
            ax.axvline(j * block_width, color='red', linestyle='--')

        for block in selected_blocks:
            i, j = block
            rect = plt.Rectangle((j * block_width, i * block_height), block_width, block_height,
                                 linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        plt.savefig(visual_output_path, bbox_inches='tight')
        plt.close()

    feature_vector = []
    for i, j in selected_blocks:
        block_region = segmented_image[i * block_height: (i + 1) * block_height, j * block_width: (j + 1) * block_width]
        feature_vector.append(np.count_nonzero(block_region))

    return feature_vector


def process_segmented_images_for_rop(segmented_input_dir, feature_output_dir, visual_output_dir, visualize=False):
    """Processes segmented images, applies ROP feature extraction, and saves results on a white background."""
    for root, _, files in os.walk(segmented_input_dir):
        for file_name in tqdm(files, desc="Processing segmented images for ROP feature extraction"):
            if file_name.endswith('.png'):
                image_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(root, segmented_input_dir)
                feature_output_path = os.path.join(feature_output_dir, relative_path, file_name.replace('.png', '.npy'))
                visual_output_path = os.path.join(visual_output_dir, relative_path,
                                                  file_name.replace('.png', '_rop_visual.png'))

                os.makedirs(os.path.dirname(feature_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(visual_output_path), exist_ok=True)

                segmented_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if segmented_image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                segmented_image = replace_black_with_white(segmented_image)

                rop_features = random_occupancy_pattern(segmented_image, grid_size=(10, 10), num_blocks=50,
                                                        visualize=visualize, visual_output_path=visual_output_path)
                np.save(feature_output_path, np.array(rop_features))


process_segmented_images_for_rop(segmented_input_dir, feature_output_dir, visual_output_dir, visualize=True)
