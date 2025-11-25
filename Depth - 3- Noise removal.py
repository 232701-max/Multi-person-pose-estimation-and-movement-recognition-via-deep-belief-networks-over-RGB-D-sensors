import cv2
import os

# Function to apply Non-Local Means Denoising for 16-bit single-channel images
def apply_nl_means_denoising(image, h=15):
    """
    Parameters:
    - h: Strength of the filter (higher values remove more noise but may blur details)
    """
    return cv2.fastNlMeansDenoising(image, None, h, 7, 21)  # Works for single-channel grayscale images

# Define input and output base folder paths
input_base_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\3- Normalized Images"  # Main input folder
output_base_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\3- Noise Removal"  # Main output folder

# Loop through all class folders in the input base folder
for class_folder in os.listdir(input_base_folder):
    class_path = os.path.join(input_base_folder, class_folder)

    if os.path.isdir(class_path):  # Ensure it's a folder
        # Create corresponding class folder in output directory
        output_class_path = os.path.join(output_base_folder, class_folder)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        # Loop through all subfolders (video folders)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)

            if os.path.isdir(video_path):  # Ensure it's a folder
                # Create corresponding video folder in output directory
                output_video_path = os.path.join(output_class_path, video_folder)
                if not os.path.exists(output_video_path):
                    os.makedirs(output_video_path)

                # Process all image files in the video folder
                for filename in os.listdir(video_path):
                    if filename.endswith('.png'):  # Depth images are usually PNG
                        img_path = os.path.join(video_path, filename)

                        # Load 16-bit depth image
                        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                        if img is not None and len(img.shape) == 2:  # Ensure it's single-channel
                            # Apply Non-Local Means Denoising
                            denoised_img = apply_nl_means_denoising(img, h=15)

                            # Save the processed image in the corresponding output folder as 16-bit PNG
                            output_img_path = os.path.join(output_video_path, filename)
                            cv2.imwrite(output_img_path, denoised_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print("✅ Non-Local Means Denoising applied to all 16-bit depth images and saved successfully while maintaining the directory structure!")
