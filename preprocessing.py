import os
import glob
import numpy as np
import argparse
import cv2
import sys

def preprocess_images(input_folder, out_folder , target_size=(640, 640), do_normalize=False):
    os.makedirs(out_folder, exist_ok=True)

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')

    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, "**", ext), recursive=True))    

    print(f"Found {len(image_paths)} images in {input_folder}")

    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            continue

        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        if do_normalize:
            img = img.astype(np.float32)/255.0

        rel_path = os.path.relpath(image_path, input_folder)
        out_path = os.path.join(out_folder, rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)


        if do_normalize:
            img_to_save = (img*255).clip(0, 255).astype(np.uint8)
        else:
            img_to_save = img

        cv2.imwrite(out_path, img_to_save)

        if (1+1) % 100 == 0:
            print(f"Processed {i+1}/{len(image_paths)} images")

    print(f"Processed {len(image_paths)} images")


def main():
    parser = argparse.ArgumentParser(description="Preprocess images in train/valid/test")
    parser.add_argument("--input", type=str, help="Path to input folder")
    parser.add_argument("--output", type=str, help="Path to output folder")
    parser.add_argument("--width",  type=int, default=640, help="Width of the output image")
    parser.add_argument("--height", type=int, default=640, help="Height of the output image")
    parser.add_argument("--normalize", action="store_true", help="Normalize pixel values to [0,1]")
    args = parser.parse_args()

    target_size = (args.width, args.height)

    subsets = ["train", "valid", "test"]
    for subset in subsets:
        in_dir = os.path.join(args.input, subset , "images")
        out_dir = os.path.join(args.output, subset , "images")

        if not os.path.exists(in_dir):
            print(f"Error: {in_dir} does not exist")
            continue

            

        preprocess_images(
            input_folder=args.input,
            out_folder=args.output,
            target_size=target_size,
            do_normalize=args.normalize
        )

if __name__ == '__main__':
    main()