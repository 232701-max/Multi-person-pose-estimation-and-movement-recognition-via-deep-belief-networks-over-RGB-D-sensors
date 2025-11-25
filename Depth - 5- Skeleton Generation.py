import os
import numpy as np
import cv2
import open3d as o3d


def depth_to_point_cloud_with_grayscale(depth_image, intrinsic_matrix=None):
    if intrinsic_matrix is None:
        intrinsic_matrix = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]])

    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    z = depth_image.flatten().astype(np.float32) / 1000.0
    valid_indices = z > 0
    u = u[valid_indices]
    v = v[valid_indices]
    z = z[valid_indices]

    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)
    normalized_depth = (z - np.min(z)) / (np.max(z) - np.min(z))
    colors = np.stack((normalized_depth, normalized_depth, normalized_depth), axis=-1)

    return points, colors


def save_grayscale_point_cloud(points, colors, filename="point_cloud.ply"):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"Point cloud saved to {filename}")


def save_point_cloud_visualization(points, colors, visualization_path):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(visualization_path)
    print(f"Visualization saved to {visualization_path}")
    vis.destroy_window()


def process_directory_with_grayscale(input_dir, output_dir, visualization_dir, intrinsic_matrix=None):
    """
    Processes all depth images in the input directory, saves point clouds with grayscale shading
    to the output directory, and saves visualizations as images to the visualization directory.

    Parameters:
    - input_dir: str, path to the input directory containing depth images.
    - output_dir: str, path to the output directory to save point clouds.
    - visualization_dir: str, path to the visualization output directory.
    - intrinsic_matrix: np.array, camera intrinsic matrix (default uses typical parameters).
    """
    for class_folder in os.listdir(input_dir):
        class_folder_path = os.path.join(input_dir, class_folder)
        if not os.path.isdir(class_folder_path):
            continue

        # Create corresponding class folder in the output directories
        class_output_dir = os.path.join(output_dir, f"class {class_folder}")
        class_visualization_dir = os.path.join(visualization_dir, f"class {class_folder}")
        os.makedirs(class_output_dir, exist_ok=True)
        os.makedirs(class_visualization_dir, exist_ok=True)

        for folder in os.listdir(class_folder_path):
            folder_path = os.path.join(class_folder_path, folder)
            if not os.path.isdir(folder_path):
                continue

            # Create corresponding subfolders in output directories
            folder_output_dir = os.path.join(class_output_dir, folder)
            folder_visualization_dir = os.path.join(class_visualization_dir, folder)
            os.makedirs(folder_output_dir, exist_ok=True)
            os.makedirs(folder_visualization_dir, exist_ok=True)

            depth_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.tif', '.tiff'))]

            for depth_file in depth_files:
                print(f"Processing {depth_file}...")

                depth_path = os.path.join(folder_path, depth_file)
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    print(f"Skipping {depth_file}: Unable to load depth image.")
                    continue

                points, colors = depth_to_point_cloud_with_grayscale(depth_image, intrinsic_matrix)

                output_file = os.path.join(folder_output_dir, f"{os.path.splitext(depth_file)[0]}_point_cloud.ply")
                save_grayscale_point_cloud(points, colors, output_file)

                visualization_path = os.path.join(folder_visualization_dir,
                                                  f"{os.path.splitext(depth_file)[0]}_visualization.png")
                save_point_cloud_visualization(points, colors, visualization_path)


# Example usage
input_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\6- Body Segmentation"  # Replace with your input directory path
output_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\SKELETON 3-D Point Cloud\ply_files"  # Replace with your output directory path
visualization_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\SKELETON 3-D Point Cloud\visualization"  # Replace with your visualization directory path

camera_intrinsics = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]])

process_directory_with_grayscale(input_directory, output_directory, visualization_directory, camera_intrinsics)
