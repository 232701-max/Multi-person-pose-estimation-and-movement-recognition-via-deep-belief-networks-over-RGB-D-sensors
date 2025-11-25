import cv2
import mediapipe as mp
import os
import numpy as np
import csv
from natsort import natsorted

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Define the indices of landmarks to extract (from your original code)
selected_landmarks_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Define skeletal connections (from your original code)
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

def calculate_midpoint(p1, p2):
    return (p1.x + p2.x) / 2, (p1.y + p2.y) / 2


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def calculate_text_position(p1, p2, p3, canvas_shape):
    """Calculate optimal text position to avoid overlap"""
    # Get the midpoint between vertex and the middle of other two points
    mid_x = (p1[0] + p3[0]) // 2
    mid_y = (p1[1] + p3[1]) // 2

    # Calculate vector from vertex to midpoint
    dx = mid_x - p2[0]
    dy = mid_y - p2[1]

    # Normalize the vector and scale it
    length = np.sqrt(dx ** 2 + dy ** 2)
    if length > 0:
        dx = dx / length * 40  # Scale factor of 40 pixels
        dy = dy / length * 40

    # Calculate text position
    text_x = int(p2[0] + dx)
    text_y = int(p2[1] + dy)

    # Ensure text stays within image bounds
    margin = 10
    text_x = max(margin, min(text_x, canvas_shape[1] - margin))
    text_y = max(margin, min(text_y, canvas_shape[0] - margin))

    return (text_x, text_y)


def process_silhouette(image, blank_canvas):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    angles = []

    if results.pose_landmarks:
        # Calculate neck landmark (index 29)
        landmark_11 = results.pose_landmarks.landmark[11]
        landmark_12 = results.pose_landmarks.landmark[12]
        neck_x, neck_y = calculate_midpoint(landmark_11, landmark_12)

        # Draw the original skeleton
        for idx in selected_landmarks_indices:
            landmark = results.pose_landmarks.landmark[idx]
            cx, cy = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(blank_canvas, (cx, cy), 5, (0, 0, 0), -1)

        # Draw neck keypoint
        neck_cx, neck_cy = int(neck_x * image.shape[1]), int(neck_y * image.shape[0])
        cv2.circle(blank_canvas, (neck_cx, neck_cy), 5, (0, 0, 0), -1)

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

            cv2.line(blank_canvas, start_point, end_point, (0, 0, 0), 2)

        # Calculate angles for specified joints
        angle_configs = [
            (11, 13, 15, "Left Arm"),
            (12, 14, 16, "Right Arm"),
            (23, 25, 27, "Left Leg"),
            (24, 26, 28, "Right Leg")
        ]

        # First calculate all angles
        for start_idx, mid_idx, end_idx, joint_name in angle_configs:
            p1 = (int(results.pose_landmarks.landmark[start_idx].x * image.shape[1]),
                  int(results.pose_landmarks.landmark[start_idx].y * image.shape[0]))
            p2 = (int(results.pose_landmarks.landmark[mid_idx].x * image.shape[1]),
                  int(results.pose_landmarks.landmark[mid_idx].y * image.shape[0]))
            p3 = (int(results.pose_landmarks.landmark[end_idx].x * image.shape[1]),
                  int(results.pose_landmarks.landmark[end_idx].y * image.shape[0]))

            angle = calculate_angle(p1, p2, p3)
            angles.append(round(angle, 2))

            # Calculate optimal text position
            text_pos = calculate_text_position(p1, p2, p3, blank_canvas.shape)

            # Draw white background for text
            text = f"{angle:.1f}"  # Removed the degree symbol for cleaner look
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(blank_canvas,
                          (text_pos[0] - 2, text_pos[1] - text_height - 2),
                          (text_pos[0] + text_width + 2, text_pos[1] + 2),
                          (255, 255, 255), -1)

            # Draw text
            cv2.putText(blank_canvas, text, text_pos,
                        font, font_scale, (0, 0, 0), thickness)

    return blank_canvas, angles

def process_images_in_directory(input_directory, output_directory, csv_path):
    os.makedirs(output_directory, exist_ok=True)

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Class", "Filename",
                  "S1_Left_Arm", "S1_Right_Arm", "S1_Left_Leg", "S1_Right_Leg",
                  "S2_Left_Arm", "S2_Right_Arm", "S2_Left_Leg", "S2_Right_Leg"]
        writer.writerow(header)

        for root, _, files in os.walk(input_directory):
            relative_path = os.path.relpath(root, input_directory)
            output_folder = os.path.join(output_directory, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            for filename in files:
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, filename)
                    image = cv2.imread(image_path)

                    if image is None:
                        print(f"Error loading image: {image_path}")
                        continue

                    height, width, _ = image.shape
                    left_half = image[:, :width // 2]
                    right_half = image[:, width // 2:]

                    skeleton_left = np.ones_like(left_half) * 255
                    skeleton_right = np.ones_like(right_half) * 255

                    processed_left, angles_left = process_silhouette(left_half, skeleton_left)
                    processed_right, angles_right = process_silhouette(right_half, skeleton_right)

                    combined_skeleton = cv2.hconcat([processed_left, processed_right])

                    output_path = os.path.join(output_folder, f"ajpf_{filename}")
                    cv2.imwrite(output_path, combined_skeleton)

                    writer.writerow([os.path.basename(root), filename] + angles_left + angles_right)
                    print(f"✅ Processed: {image_path} -> {output_path}")


# Example usage
input_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\5- Body Segmentation"
output_directory = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\12- AJPF"
csv_path = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\12- AJPF\ajpf_values.csv"

process_images_in_directory(input_directory, output_directory, csv_path)
pose.close()

print("🎉 Processing completed! AJPF values and visualizations saved.")