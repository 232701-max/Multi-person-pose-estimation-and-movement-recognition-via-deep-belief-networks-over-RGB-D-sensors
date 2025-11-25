from PIL import Image
import os

def convert_to_grayscale(input_image, output_path):
    image = Image.open(input_image)
    grayscale_image = image.convert("L")  # Convert to grayscale
    grayscale_image.save(output_path)

# Define input and output base folder paths
input_base_folder = r"D:\Research\1- Research Papers\7- MDPI - Electronics\Implementation\1- RGB\3- Noise Removal"
output_base_folder = r"D:\Research\1- Research Papers\7- MDPI - Electronics\Implementation\1- RGB\8- Greyscale"

# Loop through all class folders in the input base folder
for class_folder in os.listdir(input_base_folder):
    class_path = os.path.join(input_base_folder, class_folder)

    if os.path.isdir(class_path):  # Ensure it's a folder
        # Create corresponding class folder in output directory
        output_class_path = os.path.join(output_base_folder, class_folder)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        # Loop through all subfolders (video folders)
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)

            if os.path.isdir(video_path):  # Ensure it's a folder
                # Create corresponding video folder in output directory
                output_video_path = os.path.join(output_class_path, video_folder)
                if not os.path.exists(output_video_path):
                    os.makedirs(output_video_path)

                # Process all image files in the video folder
                for filename in os.listdir(video_path):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
                        input_image_path = os.path.join(video_path, filename)
                        output_image_path = os.path.join(output_video_path, f"grayscale_{filename}")

                        convert_to_grayscale(input_image_path, output_image_path)
                        print(f"Converted {filename} to grayscale and saved as grayscale_{filename}")

print("✅ Grayscale conversion applied to all images while maintaining the directory structure!")
