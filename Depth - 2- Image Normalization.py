import cv2
import os
import numpy as np

# Function to normalize a 16-bit depth image to 8-bit for visualization
def normalize_depth_image(image):
    """
    Normalizes a 16-bit depth image to 8-bit for better visualization.
    - image: 16-bit single-channel depth image.
    Returns a normalized 8-bit image.
    """
    # Normalize to range 0-255
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(normalized)  # Convert to 8-bit

# Define input and output base folder paths
input_base_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\2- Keyframes"  # Main input folder
output_base_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\3- Normalized Images"  # Folder for normalized images

# Loop through all class folders in the input base folder
for class_folder in os.listdir(input_base_folder):
    class_path = os.path.join(input_base_folder, class_folder)

    if os.path.isdir(class_path):  # Ensure it's a folder
        output_class_path = os.path.join(output_base_folder, class_folder)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        # Loop through all subfolders (video folders)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)

            if os.path.isdir(video_path):  # Ensure it's a folder
                output_video_path = os.path.join(output_class_path, video_folder)
                if not os.path.exists(output_video_path):
                    os.makedirs(output_video_path)

                # Process all image files in the video folder
                for filename in os.listdir(video_path):
                    if filename.endswith('.png'):  # Depth images are usually PNG
                        img_path = os.path.join(video_path, filename)

                        # Load 16-bit depth image
                        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                        if img is not None and len(img.shape) == 2 and img.dtype == np.uint16:  # Ensure it's 16-bit
                            # Normalize depth image
                            normalized_img = normalize_depth_image(img)

                            # Save the normalized image as 8-bit PNG for visualization
                            output_img_path = os.path.join(output_video_path, filename)
                            cv2.imwrite(output_img_path, normalized_img)

print("✅ Normalized 16-bit depth images to 8-bit and saved for visualization!")
