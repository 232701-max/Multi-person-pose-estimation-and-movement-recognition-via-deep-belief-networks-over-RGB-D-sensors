import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    """Calculate and normalize a color histogram."""
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten()

def compute_histogram_differences(frame_folder):
    """Compute histogram differences between consecutive frames."""
    frames = sorted(os.listdir(frame_folder))
    histograms = []
    differences = []

    for i in range(len(frames)):
        frame_path = os.path.join(frame_folder, frames[i])
        image = cv2.imread(frame_path)

        if image is None:
            continue

        histogram = calculate_histogram(image)
        histograms.append((frames[i], histogram))

        if i > 0:
            diff = cv2.compareHist(histograms[i - 1][1], histogram, cv2.HISTCMP_CHISQR)
            differences.append((frames[i], diff))

    return histograms, differences

def select_top_keyframes(differences, num_keyframes=20):
    """Select top keyframes based on histogram differences."""
    differences.sort(key=lambda x: x[1], reverse=True)
    return differences[:num_keyframes]

def compute_variation_histogram(keyframes, histograms):
    """Compute a single histogram showing variation across selected keyframes."""
    total_histogram = None
    num_keyframes = len(keyframes)

    for keyframe, _ in keyframes:
        for frame, histogram in histograms:
            if frame == keyframe:
                if total_histogram is None:
                    total_histogram = histogram
                else:
                    total_histogram += histogram  # Accumulate histograms
                break

    if num_keyframes > 0:
        avg_histogram = total_histogram / num_keyframes  # Normalize by the number of keyframes
        return avg_histogram
    else:
        return None

def save_keyframes_and_variation_histogram(frame_folder, keyframes, histograms, keyframes_root_folder, histograms_root_folder, class_folder_name, video_folder_name):
    """Save keyframes and the histogram showing variation across keyframes."""
    keyframe_output_folder = os.path.join(keyframes_root_folder, class_folder_name, video_folder_name)
    histogram_output_folder = os.path.join(histograms_root_folder, class_folder_name, video_folder_name)

    os.makedirs(keyframe_output_folder, exist_ok=True)
    os.makedirs(histogram_output_folder, exist_ok=True)

    # Save keyframes
    for keyframe, _ in keyframes:
        frame_path = os.path.join(frame_folder, keyframe)
        image = cv2.imread(frame_path)

        if image is not None:
            keyframe_output_path = os.path.join(keyframe_output_folder, keyframe)
            cv2.imwrite(keyframe_output_path, image)

    # Compute and save variation histogram
    variation_histogram = compute_variation_histogram(keyframes, histograms)
    if variation_histogram is not None:
        histogram_output_path = os.path.join(histogram_output_folder, "variation_histogram.png")
        plt.figure()
        plt.plot(variation_histogram)
        plt.title(f"Variation Histogram for {video_folder_name}")
        plt.xlabel("Bins")
        plt.ylabel("Frequency")
        plt.savefig(histogram_output_path)
        plt.close()

def process_keyframes_for_all_videos(input_directory, output_directory, num_keyframes=20):
    """Process all videos and save keyframes + the histogram showing keyframe variations."""
    keyframes_root_folder = os.path.join(output_directory, "keyframes")
    histograms_root_folder = os.path.join(output_directory, "histograms")

    os.makedirs(keyframes_root_folder, exist_ok=True)
    os.makedirs(histograms_root_folder, exist_ok=True)

    # Iterate over each class folder
    for class_folder in os.listdir(input_directory):
        class_folder_path = os.path.join(input_directory, class_folder)

        if os.path.isdir(class_folder_path):
            for video_folder_name in os.listdir(class_folder_path):
                frame_folder = os.path.join(class_folder_path, video_folder_name)

                if os.path.isdir(frame_folder):
                    histograms, differences = compute_histogram_differences(frame_folder)
                    keyframes = select_top_keyframes(differences, num_keyframes)

                    save_keyframes_and_variation_histogram(frame_folder, keyframes, histograms, keyframes_root_folder, histograms_root_folder, class_folder, video_folder_name)

if __name__ == "__main__":
    input_directory = r"D:\Research\1- Research Papers\7- MDPI - Electronics\Implementation\1- RGB\1- Image Frames"
    output_directory = r"D:\Research\1- Research Papers\7- MDPI - Electronics\Implementation\1- RGB\2- Keyframes"

    process_keyframes_for_all_videos(input_directory, output_directory, num_keyframes=20)
