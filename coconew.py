import json
import os
import shutil

# Define paths
input_path = r'C:\Users\BIT\Desktop\try segment\annotations'
output_path = r'C:\Users\BIT\Desktop\try segment\labels'
train_images_path = r'C:\Users\BIT\Desktop\try segment\mask'
annotations_file = r'C:\Users\BIT\Desktop\try segment\train.json'

# Ensure output directories exist
os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)

# Load the JSON data
with open(annotations_file) as f:
    data = json.load(f)

file_names = []

def load_images_from_folder(folder):
    count = 0
    for filename in os.listdir(folder):
        source = os.path.join(folder, filename)
        destination = os.path.join(output_path, f"images/img{count}.jpg")

        try:
            shutil.copy(source, destination)
            print(f"File {filename} copied successfully to {destination}.")
        except shutil.SameFileError:
            print("Source and destination represent the same file.")

        file_names.append(filename)
        count += 1

def get_img_ann(image_id):
    img_ann = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    return img_ann if img_ann else None

def get_img(filename):
    for img in data['images']:
        if img['file_name'] == filename:
            return img
    return None

# Load images and copy them to the destination folder
load_images_from_folder(train_images_path)

count = 0

for filename in file_names:
    # Extracting image metadata
    img = get_img(filename)
    if not img:
        print(f"Metadata for image {filename} not found.")
        continue

    img_id = img['id']
    img_w = img['width']
    img_h = img['height']

    # Get Annotations for this image
    img_ann = get_img_ann(img_id)

    if img_ann:
        # Open file for current image annotations
        label_file_path = os.path.join(output_path, f"labels/img{count}.txt")
        with open(label_file_path, "w") as file_object:
            for ann in img_ann:
                current_category = ann['category_id'] - 1  # YOLO format labels start from 0
                current_bbox = ann['bbox']
                x, y, w, h = current_bbox

                # Finding midpoints
                x_centre = (x + (x + w)) / 2
                y_centre = (y + (y + h)) / 2

                # Normalization
                x_centre /= img_w
                y_centre /= img_h
                w /= img_w
                h /= img_h

                # Limiting to fixed number of decimal places
                x_centre = format(x_centre, '.6f')
                y_centre = format(y_centre, '.6f')
                w = format(w, '.6f')
                h = format(h, '.6f')

                # Writing current object
                file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

        print(f"Annotation for image {filename} saved to {label_file_path}.")
    count += 1
