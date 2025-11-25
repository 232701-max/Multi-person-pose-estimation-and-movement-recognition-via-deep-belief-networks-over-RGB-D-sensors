import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import natsort

def load_images_from_directory(input_dir):
    images = []
    filenames = []
    for index, file in enumerate(natsort.natsorted(os.listdir(input_dir)), start=1):
        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            filename = f"{index}.png"
            filenames.append(filename)
    return images, filenames

def compute_kinetic_energy(frames):
    kinetic_energy_values = []
    diff_images = []
    for t in range(len(frames) - 1):
        I_t = frames[t]
        I_t_plus_1 = frames[t + 1]
        diff = I_t_plus_1 - I_t
        kinetic_energy = np.sum(np.square(diff))
        kinetic_energy_values.append(kinetic_energy)
        diff_norm = cv2.normalize(np.abs(diff), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_colormap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_HOT)
        diff_images.append(diff_colormap)
    return kinetic_energy_values, diff_images

def plot_kinetic_energy(kinetic_energy_values, output_plot_path):
    plt.figure()
    plt.plot(kinetic_energy_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Frame Index')
    plt.ylabel('Kinetic Energy')
    plt.title('Kinetic Energy Over Frames')
    plt.grid()
    plt.savefig(output_plot_path)
    plt.close()

def save_energy_visualization(diff_images, output_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    for i, diff_img in enumerate(diff_images):
        output_path = os.path.join(output_img_dir, f"energy_{i+1}.png")
        cv2.imwrite(output_path, diff_img)

def process_class_folders(class_name, class_dir, output_dir, csv_data):
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    for subdir_name in natsort.natsorted(os.listdir(class_dir)):
        subdir_path = os.path.join(class_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue
        output_subdir = os.path.join(class_output_dir, subdir_name)
        os.makedirs(output_subdir, exist_ok=True)
        frames, filenames = load_images_from_directory(subdir_path)
        if len(frames) < 2:
            continue
        kinetic_energy_values, diff_images = compute_kinetic_energy(frames)
        plot_kinetic_energy(kinetic_energy_values, os.path.join(output_subdir, "kinetic_energy_plot.png"))
        save_energy_visualization(diff_images, output_subdir)
        mean_ke = np.mean(kinetic_energy_values)
        var_ke = np.var(kinetic_energy_values)
        max_ke = np.max(kinetic_energy_values)
        min_ke = np.min(kinetic_energy_values)
        for i, ke in enumerate(kinetic_energy_values):
            csv_data.append({
                'Class': class_name,
                'Filename': filenames[i],
                'Kinetic_Energy': ke,
                'Mean Kinetic Energy': mean_ke,
                'Variance of Kinetic Energy': var_ke,
                'Max Kinetic Energy': max_ke,
                'Min Kinetic Energy': min_ke
            })

def process_all_classes(root_dir, output_dir, csv_output_path):
    csv_data = []
    for class_name in natsort.natsorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"Processing class: {class_name}")
            process_class_folders(class_name, class_dir, output_dir, csv_data)
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_output_path, index=False)

root_dir = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\6- Binary Silhouetteee"
output_dir = r"D:\Research\1- Research Papers\7- MDPI - Sensors Journal 3\Implementation\1- RGB\99- Kinetic Energyyyy"
csv_output_path = os.path.join(output_dir, "kinetic_energy_results.csv")
os.makedirs(output_dir, exist_ok=True)
process_all_classes(root_dir, output_dir, csv_output_path)
