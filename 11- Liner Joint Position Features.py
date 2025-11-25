import cv2
import mediapipe as mp
import os
import numpy as np
import csv
from natsort import natsorted

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Define selected joint indices (adding neck at index 29)
selected_landmarks_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # 29 is for neck

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

# Function to calculate the midpoint (for neck estimation)
def calculate_midpoint(p1, p2):
    return (p1.x + p2.x) / 2, (p1.y + p2.y) / 2

# Function to process images and save results
def process_images_in_directory(input_directory, output_directory, csv_path):
    os.makedirs(output_directory, exist_ok=True)

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Create CSV header
        header = ["Class", "Filename"]
        for idx in selected_landmarks_indices:
            header.append(f"Silhouette_1_Joint_{idx}")
        for idx in selected_landmarks_indices:
            header.append(f"Silhouette_2_Joint_{idx}")
        writer.writerow(header)

        # Traverse directory
        for class_folder in natsorted(os.listdir(input_directory)):
            class_path = os.path.join(input_directory, class_folder)
            if not os.path.isdir(class_path):
                continue

            for subfolder in natsorted(os.listdir(class_path)):
                subfolder_path = os.path.join(class_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                output_subfolder = os.path.join(output_directory, class_folder, subfolder)
                os.makedirs(output_subfolder, exist_ok=True)

                image_files = natsorted([f for f in os.listdir(subfolder_path) if f.endswith((".jpg", ".jpeg", ".png"))])

                for filename in image_files:
                    image_path = os.path.join(subfolder_path, filename)
                    image = cv2.imread(image_path)

                    if image is None:
                        print(f"Error loading image: {image_path}")
                        continue

                    # Split image into two halves
                    height, width, _ = image.shape
                    left_half = image[:, :width // 2]  # Left silhouette
                    right_half = image[:, width // 2:]  # Right silhouette

                    # Create white background for visualization
                    skeleton_left = np.ones_like(left_half) * 255
                    skeleton_right = np.ones_like(right_half) * 255

                    # Process silhouettes and write joint values
                    ljpf_values = []
                    processed_left, ljpf_left = process_silhouette(left_half, skeleton_left, "L1")
                    processed_right, ljpf_right = process_silhouette(right_half, skeleton_right, "L2")
                    ljpf_values.extend(ljpf_left)  # Silhouette 1 values
                    ljpf_values.extend(ljpf_right)  # Silhouette 2 values

                    # Combine images side by side
                    combined_skeleton = cv2.hconcat([processed_left, processed_right])

                    # Save skeleton image
                    output_path = os.path.join(output_subfolder, f"skeleton_{filename}")
                    cv2.imwrite(output_path, combined_skeleton)
                    print(f"✅ Processed: {image_path} -> {output_path}")

                    # Save LJPF values to CSV
                    writer.writerow([class_folder, f"{subfolder}/{filename}"] + ljpf_values)

# Function to process a silhouette and write joint values on image
def process_silhouette(image, blank_canvas, label_prefix):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    ljpf_values = []

    if results.pose_landmarks:
        # Calculate neck landmark (index 29)
        landmark_11 = results.pose_landmarks.landmark[11]
        landmark_12 = results.pose_landmarks.landmark[12]
        neck_x, neck_y = calculate_midpoint(landmark_11, landmark_12)

        # Draw landmarks, values, and connections
        for idx in selected_landmarks_indices:
            landmark = results.pose_landmarks.landmark[idx]
            cx, cy = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])

            # Draw keypoint
            cv2.circle(blank_canvas, (cx, cy), 5, (0, 0, 0), -1)

            # Write joint value
            normalized_distance = round(abs(landmark.y - results.pose_landmarks.landmark[0].y), 2)
            cv2.putText(blank_canvas, f"{normalized_distance}", (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            ljpf_values.append(normalized_distance)

        # Draw neck keypoint (29)
        neck_cx, neck_cy = int(neck_x * image.shape[1]), int(neck_y * image.shape[0])
        cv2.circle(blank_canvas, (neck_cx, neck_cy), 5, (0, 0, 255), -1)

        # Draw connections
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

            cv2.line(blank_canvas, start_point, end_point, (0, 0, 0), 2)

    return blank_canvas, ljpf_values


# Example usage
input_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\5- Body Segmentation"
output_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\11- LPJffF\LPJF"
csv_path = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\11- LPJffF\ljpf_values.csv"


process_images_in_directory(input_directory, output_directory, csv_path)

# Close the MediaPipe Pose model
pose.close()

print("🎉 Processing completed! Skeletons and LJPF CSV saved.")
