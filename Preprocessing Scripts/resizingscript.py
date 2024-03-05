import os
from PIL import Image

def resize_images(input_path, output_folder, size=(256, 256)):#set required size
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # If input_path is a file, process it directly
    if os.path.isfile(input_path):
        resize_image(input_path, output_folder, size)
    # If input_path is a directory, process all files and subdirectories recursively
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, filename)
                    try:
                        resize_image(image_path, output_folder, size)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

def resize_image(image_path, output_folder, size):
    # Open image
    with Image.open(image_path) as img:
        # Convert RGBA images to RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        if img.mode == 'P':
            img = img.convert('RGB')
        # Resize image
        resized_img = img.resize(size, Image.LANCZOS)
        # Save resized image in output folder
        relative_path = os.path.relpath(os.path.dirname(image_path), input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        output_path = os.path.join(output_subfolder, os.path.splitext(os.path.basename(image_path))[0] + '.jpg')
        resized_img.save(output_path, 'JPEG')
        print(f"Resized and saved: {output_path}")

input_folder = "./Dataset"#path to raw images
output_folder = "./resized_Dataset"#destination path

resize_images(input_folder, output_folder)