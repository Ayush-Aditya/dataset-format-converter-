import os
import cv2
import numpy as np

input_dir = r'C:\Users\BIT\Desktop\New folder (2)\SegmentationClass'
output_dir = r'C:\Users\BIT\Desktop\exp\data\labels\train'

# Mapping BGR values to class numbers (reversed RGB to BGR)
class_mapping = {
    (48, 115, 230): 0,  # Heart
    (151, 67, 198): 1,  # Liver
    (124, 166, 247): 2, # Intestine
    (103, 236, 21): 3,  # Fat
    (104, 95, 255): 4,  # Pancreas
    (173, 19, 141): 5,  # Bone
    (177, 165, 155): 6, # Muscle
    (161, 134, 77): 7,  # Connective Tissue
    (130, 201, 155): 8  # Tissue
}

def get_unique_colors(image):
    # Get unique colors in the image
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    return unique_colors

def print_unique_colors(image_path):
    mask = cv2.imread(image_path)
    unique_colors = get_unique_colors(mask)
    print(f"Unique colors in {image_path}: {unique_colors}")

def find_contours_with_tolerance(image, bgr, tolerance=10):
    lower = np.array([max(0, c - tolerance) for c in bgr])
    upper = np.array([min(255, c + tolerance) for c in bgr])
    mask = cv2.inRange(image, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # Print unique colors in the image
    print_unique_colors(image_path)
    
    # Load the non-binary mask
    mask = cv2.imread(image_path)
    H, W, _ = mask.shape
    print(f"Processing {image_path} with dimensions (H: {H}, W: {W})")

    polygons_dict = {i: [] for i in range(9)}  # Dictionary to store polygons for each class

    for bgr, class_num in class_mapping.items():
        contours = find_contours_with_tolerance(mask, bgr)
        print(f"Class {class_num} ({bgr}): Found {len(contours)} contours")

        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons_dict[class_num].append(polygon)
                print(f"Class {class_num}: Added polygon with {len(polygon)//2} points")

    output_file = os.path.join(output_dir, '{}.txt'.format(j[:-4]))
    with open(output_file, 'w') as f:
        for class_num, polygons in polygons_dict.items():
            for polygon in polygons:
                f.write('{} '.format(class_num))
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    else:
                        f.write('{} '.format(p))
        print(f"Wrote {output_file} with {sum(len(p) for p in polygons_dict.values())} polygons")
