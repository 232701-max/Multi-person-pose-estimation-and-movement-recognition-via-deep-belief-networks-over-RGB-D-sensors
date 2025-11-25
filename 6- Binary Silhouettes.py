import cv2
import numpy as np
import os

def apply_grabcut_with_refinement(image_path, mask_dir, rect):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Initialize mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Create temporary arrays for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut with initial rectangle (cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Refine mask (cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask_refined = mask.copy()
    cv2.grabCut(image, mask_refined, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Create the final refined mask
    mask_final = np.where((mask_refined == 2) | (mask_refined == 0), 0, 1).astype('uint8')

    # Get the base name of the image (e.g., frame_0001.jpg)
    base_name = os.path.basename(image_path)

    # Ensure output mask is saved with the correct name in the correct structure
    mask_output_path = os.path.join(mask_dir, base_name)

    # Save the mask as a PNG file
    cv2.imwrite(mask_output_path, mask_final * 255)  # Multiply by 255 to scale mask to [0, 255]

    print(f"Mask saved to {mask_output_path}")

def process_videos_in_directory(input_directory, output_directory, rect):
    # Loop through all classes in the input directory
    for class_name in os.listdir(input_directory):
        class_path = os.path.join(input_directory, class_name)
        output_class_path = os.path.join(output_directory, class_name)

        # Ensure the output class directory exists
        os.makedirs(output_class_path, exist_ok=True)

        # Loop through all videos in the class directory
        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)

            # Ensure the output video directory exists
            output_video_path = os.path.join(output_class_path, video_name)
            os.makedirs(output_video_path, exist_ok=True)

            # Loop through all images in the video directory
            for image_name in os.listdir(video_path):
                image_path = os.path.join(video_path, image_name)
                # Apply GrabCut and save the corresponding mask
                apply_grabcut_with_refinement(image_path, output_video_path, rect)

# Example usage
input_directory = r"D:\Research\1- Research Papers\7- MDPI - Electronics\Implementation\1- RGB\1- Image Frames\6- Body Segmentation"  # Your input folder
output_directory = r"D:\Research\1- Research Papers\7- MDPI - Electronics\Implementation\1- RGB\6- Binary Silhouette"  # Your output folder
rect = (1, 1, 900, 900)  # Example rectangle (x, y, w, h)

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process the directory
process_videos_in_directory(input_directory, output_directory, rect)

print("Processing completed. Masks saved in the output directory.")
