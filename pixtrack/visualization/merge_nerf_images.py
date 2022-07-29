import argparse
import os
import numpy as np
from PIL import Image
import re


def read_images_from_folder(folder_path:str):
    images = []
    # A regex to extract the integer from a string.
    regex = re.compile(r'\d+') 
    lsorted = sorted(os.listdir(folder_path), key=lambda x:int(regex.findall(os.path.splitext(x)[0])[0]))
    for image_name in lsorted:
        image = np.array(Image.open(os.path.join(folder_path, image_name)))
        images.append(image)
    return images


def merge_images(images1, images2):
    merged_images = []
    for image_number in range(len(images1)):
        # Assumuing black background for the nerf images.
<<<<<<< HEAD
        merged_images.append(np.minimum(images1[image_number], images2[image_number]).astype(np.uint8))
=======
        merged_images.append(np.maximum(images1[image_number], images2[image_number]).astype(np.uint8))
>>>>>>> origin/main
    return merged_images


def save_images(merged_images, save_path):
    for number, image in enumerate(merged_images):
        Image.fromarray(image.astype(np.uint8)).save(os.path.join(save_path, f"{number}.jpg"))


def blend_images(image1, image2, alpha=0.5):
    blend_img = image1 * (alpha) + image2 * (1 - alpha)
    return blend_img.astype(np.uint8)


def blend_all_images(images1:list, images2:list, alpha:float=0.5):
    """
    """
    assert len(images1) == len(images2), "Number of images to blend have to be equal"
    blended_images = []
    for image_id in range(len(images1)):
        blended_image = blend_images(images1[image_id], images2[image_id])
        blended_images.append(blended_image)
    return blended_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge the nerf images')
    parser.add_argument('one', type=str, 
                        help='first nerf path')
    parser.add_argument('two', type=str, 
                        help='second nerf path')
    parser.add_argument('three', type=str, default=None,
                        help='third nerf path')
    parser.add_argument('save_path', type=str, default=None,
                        help='third nerf path')

    args = parser.parse_args()
    images1 = read_images_from_folder(args.one)
    images2 = read_images_from_folder(args.two)

    assert len(images1) == len(images2), f"{len(images1)}, {len(images2)}"
    merged_images = merge_images(images1, images2)
    if args.three is not None:
        images3 = read_images_from_folder(args.three)
        assert len(images2) == len(images3)
        merged_images = merge_images(merged_images, images3)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_images(merged_images, args.save_path)

