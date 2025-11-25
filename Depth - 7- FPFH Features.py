import numpy as np
import open3d as o3d
import os
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def depth_to_point_cloud(depth_image, fx=525.0, fy=525.0, cx=319.5, cy=239.5):
    """
    Convert depth image to point cloud.
    """
    height, width = depth_image.shape
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    depth_meters = depth_image.astype(float) / 1000.0
    X = (x_grid - cx) * depth_meters / fx
    Y = (y_grid - cy) * depth_meters / fy
    Z = depth_meters
    points = np.stack([X, Y, Z], axis=-1)
    valid_points = points[depth_meters > 0]
    valid_points = valid_points[valid_points[:, 2] < 10]
    return valid_points


def load_data(file_path):
    """
    Load data from either depth image or PLY file.
    """
    if file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        depth_img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        points = depth_to_point_cloud(depth_img)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return points


def compute_fpfh_features(point_cloud_data, normal_radius=0.05, fpfh_radius=0.1):
    """
    Compute FPFH features for a point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))

    pcd.orient_normals_consistent_tangent_plane(k=15)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100))

    return np.array(fpfh.data).T, np.asarray(pcd.points)


def visualize_and_save_results(point_cloud, fpfh_features, output_path, filename):
    """
    Visualize and save FPFH features and point cloud visualization.
    """
    vis_dir = os.path.join(output_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Point Cloud Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               c='b', marker='o', s=1)
    ax.set_title('3D Point Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(os.path.join(vis_dir, f'{filename}_point_cloud.png'))
    plt.close()

    # 2. FPFH Feature Visualization
    avg_feature = np.mean(fpfh_features, axis=0)
    plt.figure(figsize=(15, 5))
    plt.bar(range(33), avg_feature)
    plt.title('Average FPFH Feature Histogram')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Feature Value')
    plt.savefig(os.path.join(vis_dir, f'{filename}_avg_fpfh.png'))
    plt.close()

    # 3. Feature Distribution Plot
    plt.figure(figsize=(15, 5))
    plt.boxplot(fpfh_features)
    plt.title('FPFH Feature Distribution')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Feature Value')
    plt.savefig(os.path.join(vis_dir, f'{filename}_fpfh_distribution.png'))
    plt.close()

    # 4. Feature Heatmap
    plt.figure(figsize=(15, 5))
    plt.imshow(fpfh_features[:100].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Value')
    plt.title('FPFH Feature Heatmap (First 100 Points)')
    plt.xlabel('Point Index')
    plt.ylabel('Feature Dimension')
    plt.savefig(os.path.join(vis_dir, f'{filename}_fpfh_heatmap.png'))
    plt.close()


def process_all_files(input_directory, output_directory):
    """
    Process all depth images and PLY files in a directory and save FPFH features.
    """
    os.makedirs(output_directory, exist_ok=True)

    depth_files = glob.glob(os.path.join(input_directory, "*.png"))
    ply_files = glob.glob(os.path.join(input_directory, "*.ply"))
    all_files = depth_files + ply_files

    for file_path in tqdm(all_files, desc="Processing files"):
        try:
            # Load point cloud data
            point_cloud = load_data(file_path)

            # Compute FPFH features
            fpfh_features, downsampled_points = compute_fpfh_features(
                point_cloud,
                normal_radius=0.05,
                fpfh_radius=0.1
            )

            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            features_path = os.path.join(output_directory, f"{base_name}_fpfh.npy")
            points_path = os.path.join(output_directory, f"{base_name}_points.npy")

            # Save features and corresponding points
            np.save(features_path, fpfh_features)
            np.save(points_path, downsampled_points)

            # Visualize and save results
            visualize_and_save_results(downsampled_points, fpfh_features,
                                       output_directory, base_name)

            print(f"Processed {file_path}")
            print(f"Number of points: {len(downsampled_points)}")
            print(f"Feature shape: {fpfh_features.shape}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    input_dir = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\SKELETON 3-D Point Cloud\ply_files\class 1- Handshake\1"
    output_dir = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\9- FPFH\1- Handshake\1"

    process_all_files(input_dir, output_dir)