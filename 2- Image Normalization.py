import os
import cv2


def min_max_normalization(image):
    # Normalize the image to range [0, 255]
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_image


def normalize_images_in_folder(input_folder, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through each file in the input directory
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Full path for input and output files
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir_path = os.path.join(output_folder, relative_path)

                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)

                output_file_path = os.path.join(output_dir_path, file)

                # Read the image
                image = cv2.imread(input_file_path)

                # Apply Min-Max normalization
                normalized_image = min_max_normalization(image)

                # Save the normalized image to the output directory
                cv2.imwrite(output_file_path, normalized_image)

                print(f"Normalized {input_file_path} -> {output_file_path}")


def process_normalization_for_all_folders(input_root_folder, output_root_folder):
    for folder in os.listdir(input_root_folder):
        input_folder_path = os.path.join(input_root_folder, folder)
        output_folder_path = os.path.join(output_root_folder, folder)
        if os.path.isdir(input_folder_path):
            normalize_images_in_folder(input_folder_path, output_folder_path)


# Example usage
input_root_folder = r"D:\Research\1- Research Papers\7- MDPI - Electronics\Implementation\1- RGB\3- Noise Removal"
output_root_folder = r"D:\Research\1- Research Papers\7- MDPI - Electronics\Implementation\1- RGB\4- Image Normalization"
process_normalization_for_all_folders(input_root_folder, output_root_folder)
