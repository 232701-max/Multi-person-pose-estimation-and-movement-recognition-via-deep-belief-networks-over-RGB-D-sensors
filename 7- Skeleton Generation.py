import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Define the indices of landmarks to extract
selected_landmarks_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # 29 will be added as neck

# Define skeletal connections
connections = [
    (0, 29),  # Nose (head) to neck
    (29, 11), (29, 12),  # Neck to shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (23, 24)  # Connect hips
]

# Function to calculate the midpoint between two points (for neck calculation)
def calculate_midpoint(p1, p2):
    return (p1.x + p2.x) / 2, (p1.y + p2.y) / 2

# Function to process both silhouettes in each image
def process_images_in_directory(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for root, _, files in os.walk(input_directory):
        relative_path = os.path.relpath(root, input_directory)
        output_folder = os.path.join(output_directory, relative_path)
        os.makedirs(output_folder, exist_ok=True)  # Create subfolders in the output directory

        for filename in files:
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue

                # Split the image into two halves
                height, width, _ = image.shape
                left_half = image[:, :width // 2]  # Left silhouette
                right_half = image[:, width // 2:]  # Right silhouette

                # Create a white canvas for each half
                skeleton_left = np.ones_like(left_half) * 255
                skeleton_right = np.ones_like(right_half) * 255

                # Process both halves separately
                processed_left = process_silhouette(left_half, skeleton_left)
                processed_right = process_silhouette(right_half, skeleton_right)

                # Combine processed images (side by side)
                combined_skeleton = cv2.hconcat([processed_left, processed_right])

                # Save the output
                output_path = os.path.join(output_folder, f"skeleton_{filename}")
                cv2.imwrite(output_path, combined_skeleton)
                print(f"✅ Processed: {image_path} -> {output_path}")

# Function to process a single silhouette image
def process_silhouette(image, blank_canvas):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Calculate neck landmark (index 29)
        landmark_11 = results.pose_landmarks.landmark[11]
        landmark_12 = results.pose_landmarks.landmark[12]
        neck_x, neck_y = calculate_midpoint(landmark_11, landmark_12)

        # Draw landmarks on the blank canvas
        for idx in selected_landmarks_indices:
            landmark = results.pose_landmarks.landmark[idx]
            cx, cy = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(blank_canvas, (cx, cy), 5, (0, 255, 0), -1)  # Green circle

        # Draw neck keypoint (29)
        neck_cx, neck_cy = int(neck_x * image.shape[1]), int(neck_y * image.shape[0])
        cv2.circle(blank_canvas, (neck_cx, neck_cy), 5, (255, 0, 0), -1)  # Blue circle

        # Draw skeletal connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx == 29:
                start_point = (neck_cx, neck_cy)
            else:
                start_point = (int(results.pose_landmarks.landmark[start_idx].x * image.shape[1]),
                               int(results.pose_landmarks.landmark[start_idx].y * image.shape[0]))

            if end_idx == 29:
                end_point = (neck_cx, neck_cy)
            else:
                end_point = (int(results.pose_landmarks.landmark[end_idx].x * image.shape[1]),
                             int(results.pose_landmarks.landmark[end_idx].y * image.shape[0]))

            cv2.line(blank_canvas, start_point, end_point, (0, 0, 255), 2)

        return blank_canvas

    return blank_canvas  # Return blank canvas if no pose is detected

# Example usage
input_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\5- Body Segmentation"
output_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\7- Skeleton and Keypoint Generation"

process_images_in_directory(input_directory, output_directory)

# Close the MediaPipe Pose model
pose.close()

print("🎉 Processing completed! Skeletons are now saved on a white background.")
