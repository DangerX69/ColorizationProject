import os
from PIL import Image

def is_black_and_white(image_path, threshold=50):
    with Image.open(image_path) as img:
        grayscale_img = img.convert("L")  # Convert to grayscale
        average_intensity = sum(grayscale_img.getdata()) / len(grayscale_img.getdata())
        return average_intensity < threshold

def delete_black_and_white_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            if is_black_and_white(image_path):
                os.remove(image_path)
                print(f"Deleted: {image_path}")

folder_path = r"E:\Final\datasets\testing"  # Use a raw string to avoid escape characters

delete_black_and_white_images(folder_path)

