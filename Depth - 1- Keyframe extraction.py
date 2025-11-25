import os
import cv2
import numpy as np

# Define input and output base folders
input_base_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\1- Image Frames"  # Change this to your input folder
output_base_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\2- Keyframes"  # Where keyframes will be stored

# Define the number of keyframes to extract per folder
num_keyframes = 20

# Function to select keyframes
def select_keyframes(image_files, num_keyframes):
    """
    Selects `num_keyframes` evenly distributed frames.
    If there are fewer than `num_keyframes`, return all.
    """
    if len(image_files) <= num_keyframes:
        return image_files  # Keep all if not enough frames

    # Select 20 evenly spaced indices
    indices = np.linspace(0, len(image_files) - 1, num_keyframes, dtype=int)
    return [image_files[i] for i in indices]

# Loop through each folder in the input directory
for class_folder in os.listdir(input_base_folder):
    class_path = os.path.join(input_base_folder, class_folder)

    if os.path.isdir(class_path):  # Ensure it's a folder
        # Create corresponding class folder in output directory
        output_class_path = os.path.join(output_base_folder, class_folder)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        # Loop through each subfolder (video folder)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)

            if os.path.isdir(video_path):  # Ensure it's a folder
                # Create corresponding video folder in output directory
                output_video_path = os.path.join(output_class_path, video_folder)
                if not os.path.exists(output_video_path):
                    os.makedirs(output_video_path)

                # Get list of all depth images in the folder (sorted)
                image_files = sorted([f for f in os.listdir(video_path) if f.endswith('.png')])

                # Select keyframes
                selected_frames = select_keyframes(image_files, num_keyframes)

                # Copy selected keyframes to the output folder
                for filename in selected_frames:
                    input_img_path = os.path.join(video_path, filename)
                    output_img_path = os.path.join(output_video_path, filename)

                    # Read and save the image
                    img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        cv2.imwrite(output_img_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print("✅ Keyframe extraction completed! 20 keyframes per folder saved successfully.")
