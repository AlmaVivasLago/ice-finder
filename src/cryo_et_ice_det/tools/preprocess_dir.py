import argparse
import json
import os

import cv2
from PIL import Image
import mrcfile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def calculate_dimensions(image_shape, target_size):
    old_size = image_shape  # old_size is (height, width)
    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    padding = (delta_h // 2, delta_h - (delta_h // 2), delta_w // 2, delta_w - (delta_w // 2))

    return new_size, padding

def process_image(image, new_size, padding, pad=False):
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))

    if pad:
        padded_image = cv2.copyMakeBorder(resized_image, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        norm_image = cv2.normalize(padded_image, None, 0, 255, cv2.NORM_MINMAX)
    else:
        norm_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX)

    return Image.fromarray(norm_image.astype(np.uint8))


def generate_identifier(counter):
    """ Generate a unique identifier based on a counter. """
    return f"{counter:03d}"

def process_stack(stack_path, output_dir, stack_id, target_size=(1024, 1024), pad=False):
    try:
        with mrcfile.open(stack_path, mode='r') as mrc:
            if pad:
                new_size, padding = calculate_dimensions((mrc.data).shape[1:], target_size)
            else:
                new_size, padding = target_size, None
                
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_image, mrc.data[i, :, :], new_size, padding, pad=pad): i for i in range(mrc.data.shape[0])}
                for future in as_completed(futures):
                    img = future.result()
                    filename = f"lamella1_ts_{stack_id}_{futures[future]}.tiff"
                    save_image(img, filename, output_dir)
    except Exception as e:
        print(f"Failed to process stack {stack_path}: {e}")

def save_image(image, filename, directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    image.save(os.path.join(directory, filename), format='TIFF')

def preprocess_dir(input_dir, output_dir, size=(1024, 1024), pad=False):
    file_mapping = {}
    files = [f for f in os.listdir(input_dir) if f.endswith('.mrc') or f.endswith('.st')]
    counter = 0
    for file in tqdm(files):
        stack_id = generate_identifier(counter)
        process_stack(os.path.join(input_dir, file), output_dir, stack_id, target_size=size, pad=pad)
        file_mapping[stack_id] = file
        counter += 1
    with open(os.path.join(output_dir, 'file_mapping.json'), 'w') as f:
        json.dump(file_mapping, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process and normalize image stacks.")
    parser.add_argument("input_dir", default="data/selected", nargs='?', help="Directory containing the image stacks.")
    parser.add_argument("--size", type=int, nargs=2, default=(1024, 1024), help="Target size for the images as two integers (height width).")
    parser.add_argument("--pad", action="store_true", help="Flag to pad the images instead of resizing.")
    args = parser.parse_args()
    output_dir = os.path.join(args.input_dir, "preprocessed")

    preprocess_dir(args.input_dir, output_dir, size=tuple(args.size), pad=args.pad)

if __name__ == "__main__":
    main()
