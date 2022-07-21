import argparse
import os
import numpy as np
from PIL import Image
import re

import mediapy


def read_images_from_folder(folder_path:str):
    images = []
    # A regex to extract the integer from a string.
    regex = re.compile(r'\d+')
    lsorted = sorted(os.listdir(folder_path), key=lambda x:int(os.path.splitext(x)[0]))
    for image_name in lsorted:
        image = np.array(Image.open(os.path.join(folder_path, image_name)))
        images.append(image)
    return images


def read_images_from_folder_2(folder_path:str):
    images = []
    # A regex to extract the integer from a string.
    regex = re.compile(r'\d+')
    import IPython; IPython.embed()
    lsorted = sorted(os.listdir(folder_path), key=lambda x:int(regex.findall(os.path.splitext(x)[0])[0]))
    for image_name in lsorted:
        image = np.array(Image.open(os.path.join(folder_path, image_name)))
        images.append(image)
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge the nerf images')
    parser.add_argument('input_folder_path', type=str,
                        help='first nerf path')
    parser.add_argument('output_folder_path', type=str,
                        help='first nerf path')

    args = parser.parse_args()
    images1 = read_images_from_folder_2(args.input_folder_path)
    mediapy.write_video(os.path.join(args.output_folder_path, "overlay.mp4"), images1, fps=240)

