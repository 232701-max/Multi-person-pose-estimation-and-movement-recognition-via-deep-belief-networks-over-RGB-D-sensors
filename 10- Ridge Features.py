import cv2
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import natsort


def compute_hessian_matrix(image):
    ddepth = cv2.CV_64F
    Ixx = cv2.Sobel(image, ddepth, 2, 0, ksize=3)
    Iyy = cv2.Sobel(image, ddepth, 0, 2, ksize=3)
    Ixy = cv2.Sobel(image, ddepth, 1, 1, ksize=3)
    return Ixx, Iyy, Ixy


def detect_ridges(Ixx, Iyy, Ixy):
    lambda1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2))
    lambda2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2))
    ridge_response = (lambda1 < 0) & (lambda2 < 0)
    return ridge_response.astype(np.uint8) * 255


def visualize_ridges_on_white(ridge_coords, output_path, image_shape):
    # Create a white background in BGR format
    white_background = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255

    # Choose a bright ridge color (e.g., cyan: (255, 0, 255), red: (0, 0, 255))
    ridge_color = (0, 100, 255)  # Cyan

    for coord in ridge_coords:
        white_background[coord[0], coord[1]] = ridge_color  # Apply bright color to ridges

    cv2.imwrite(output_path, white_background)


def process_image(image_path, binary_path, ridge_output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    if image is None or binary_mask is None:
        return False, [], None

    silhouette = cv2.bitwise_and(image, image, mask=binary_mask)
    Ixx, Iyy, Ixy = compute_hessian_matrix(silhouette)
    ridges = detect_ridges(Ixx, Iyy, Ixy)
    ridge_coords = np.column_stack(np.where(ridges == 255))

    if len(ridge_coords) > 0:
        visualize_ridges_on_white(ridge_coords, ridge_output_path, image.shape)

    return True, ridge_coords, image.shape


# Define base directories
base_input_folder = r'D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\8- Greyscale'
base_binary_folder = r'D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\6- Binary Silhouette'
base_ridge_folder = r'D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\99- Full Ridge\FBR'
base_csv_folder = r'D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\99- Full Ridge\excel'

os.makedirs(base_ridge_folder, exist_ok=True)
os.makedirs(base_csv_folder, exist_ok=True)

all_summary_results = []
for class_folder in sorted(os.listdir(base_input_folder)):
    input_class_path = os.path.join(base_input_folder, class_folder)
    binary_class_path = os.path.join(base_binary_folder, class_folder)
    ridge_class_path = os.path.join(base_ridge_folder, class_folder)
    csv_class_path = os.path.join(base_csv_folder, 'ridge_features.csv')

    if os.path.isdir(input_class_path):
        os.makedirs(ridge_class_path, exist_ok=True)

        for video_folder in sorted(os.listdir(input_class_path)):
            input_video_path = os.path.join(input_class_path, video_folder)
            binary_video_path = os.path.join(binary_class_path, video_folder)
            ridge_video_path = os.path.join(ridge_class_path, video_folder)

            if os.path.isdir(input_video_path):
                os.makedirs(ridge_video_path, exist_ok=True)

                image_paths = natsort.natsorted(glob.glob(os.path.join(input_video_path, '*.png')))
                binary_paths = natsort.natsorted(glob.glob(os.path.join(binary_video_path, '*.jpg')))

                if not image_paths or not binary_paths:
                    print(f"No images or binary masks found in {video_folder}.")
                    continue

                for img_path, bin_path in zip(image_paths, binary_paths):
                    ridge_output_path = os.path.join(ridge_video_path, os.path.basename(img_path))
                    success, ridge_coords, image_shape = process_image(img_path, bin_path, ridge_output_path)

                    if success:
                        filename = os.path.basename(img_path)
                        total_ridges = len(ridge_coords)
                        if total_ridges > 0 and image_shape is not None:
                            if len(ridge_coords) > 1:
                                mean_x, mean_y = np.mean(ridge_coords, axis=0)
                                var_x, var_y = np.var(ridge_coords, axis=0)
                            else:
                                mean_x, mean_y = ridge_coords[0] if len(ridge_coords) == 1 else (0, 0)
                                var_x, var_y = 0, 0

                            density = total_ridges / (image_shape[0] * image_shape[1])
                            all_summary_results.append(
                                [class_folder, filename, total_ridges, mean_x, mean_y, var_x, var_y, density])

if all_summary_results:
    df_summary = pd.DataFrame(all_summary_results,
                              columns=['Class', 'Filename', 'Total Ridges', 'Mean X', 'Mean Y', 'Variance X',
                                       'Variance Y', 'Density'])
    df_summary.to_csv(csv_class_path, index=False)
    print(f"Ridge extraction completed. CSV saved to {csv_class_path}.")

print("All ridge images and CSV files saved successfully.")
