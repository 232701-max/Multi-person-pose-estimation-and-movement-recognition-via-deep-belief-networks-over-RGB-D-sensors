import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import cv2


# Function to create a point cloud and visualize it for human silhouettes
def create_point_cloud_for_human_silhouette(depth_map, point_cloud_output_path, visualization_output_path):
    height, width = depth_map.shape
    point_cloud_data = []  # To store point cloud data
    x_coords, y_coords, z_coords = [], [], []  # To store coordinates for visualization
    timestamp = 0  # Initialize timestamp counter

    # Generate point cloud data
    for v in range(height):
        for u in range(width):
            Z = depth_map[v, u]
            if Z > 0:  # Only process points with non-zero depth
                timestamp += 1
                point_cloud_data.append([timestamp, u, v, Z])  # Store point cloud data
                x_coords.append(u)
                y_coords.append(height - v)  # Invert `v` to align with a natural human view
                z_coords.append(Z)

    # Save point cloud data to Excel
    df = pd.DataFrame(point_cloud_data, columns=['Timestamp', 'x', 'y', 'Z'])
    df.to_excel(point_cloud_output_path, index=False)
    print(f"Saved point cloud data: {point_cloud_output_path}")

    # Visualization of 2.5D point cloud for human silhouette
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=z_coords, cmap='plasma', s=2)

    # Adjust axes to match human proportions
    ax.set_box_aspect([width, height, np.max(z_coords) * 0.5])  # Proportional aspect ratio
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, np.max(z_coords))

    ax.set_xlabel("Width (X)")
    ax.set_ylabel("Height (Y)")
    ax.set_zlabel("Depth (Z)")
    ax.set_title("2.5D Point Cloud of Human Silhouette")
    plt.colorbar(scatter, ax=ax, label="Depth (Z)")
    ax.view_init(elev=45, azim=-90)  # Adjust elevation and azimuth for better visibility
    plt.savefig(visualization_output_path)
    plt.close()
    print(f"Saved visualization: {visualization_output_path}")

# Main script to process all depth maps in a folder
def process_depth_maps(depth_map_folder, output_pointcloud_folder, output_visualization_folder):
    # Create output directories if they don't exist
    os.makedirs(output_pointcloud_folder, exist_ok=True)
    os.makedirs(output_visualization_folder, exist_ok=True)

    for root, _, files in os.walk(depth_map_folder):
        for file in files:
            if file.endswith(".png"):
                # Input depth map path
                depth_map_path = os.path.join(root, file)

                # Output paths
                relative_path = os.path.relpath(root, depth_map_folder)
                point_cloud_output_path = os.path.join(output_pointcloud_folder, relative_path, file.replace('.png', '_pointcloud.xlsx'))
                visualization_output_path = os.path.join(output_visualization_folder, relative_path, file.replace('.png', '_visualization.png'))

                # Ensure subdirectories exist for outputs
                os.makedirs(os.path.dirname(point_cloud_output_path), exist_ok=True)
                os.makedirs(os.path.dirname(visualization_output_path), exist_ok=True)

                print(f"Processing depth map: {depth_map_path}")

                # Load depth map
                depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
                if depth_map is None:
                    print(f"Failed to load depth map: {depth_map_path}")
                    continue

                # Call the function to generate point cloud and visualization
                create_point_cloud_for_human_silhouette(depth_map, point_cloud_output_path, visualization_output_path)

# Input and output directories
depth_map_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\6- Body Segmentation"
output_pointcloud_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\7- PointCloud"
visualization_folder = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\2- Depth\7- PointCloud\Visualizations"

process_depth_maps(depth_map_folder, output_pointcloud_folder, visualization_folder)

