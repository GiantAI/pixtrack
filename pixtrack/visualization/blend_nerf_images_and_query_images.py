import argparse
import os
import numpy as np
from PIL import Image
import re


def read_images_from_folder(folder_path: str):
    images = []
    # A regex to extract the integer from a string.
    regex = re.compile(r"\d+")
    lsorted = sorted(
        os.listdir(folder_path),
        key=lambda x: int(regex.findall(os.path.splitext(x)[0])[0]),
    )
    for image_name in lsorted:
        image = np.array(Image.open(os.path.join(folder_path, image_name)))
        images.append(image)
    return images


def save_images(merged_images, save_path):
    for number, image in enumerate(merged_images):
        Image.fromarray(image.astype(np.uint8)).save(
            os.path.join(save_path, f"{number}.jpg")
        )


def blend_images(image1, image2, alpha=0.5):
    blend_img = image1 * (alpha) + image2 * (1 - alpha)
    return blend_img.astype(np.uint8)


def blend_all_images(images1: list, images2: list, alpha: float = 0.5):
    """ """
    assert len(images1) == len(images2), "Number of images to blend have to be equal"
    blended_images = []
    for image_id in range(len(images1)):
        blended_image = blend_images(images1[image_id], images2[image_id])
        blended_images.append(blended_image)
    return blended_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="merge the nerf images")
    parser.add_argument(
        "nerf_images",
        type=str,
        help="images with the nerf renders and white background",
    )
    parser.add_argument("query", type=str, help="query_images")
    parser.add_argument("save_path", type=str, default=None, help="third nerf path")

    args = parser.parse_args()
    images1 = read_images_from_folder(args.nerf_images)
    images2 = read_images_from_folder(args.query)

    assert len(images1) == len(images2), f"{len(images1)}, {len(images2)}"
    blend_images = blend_all_images(images1, images2)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_images(blend_images, args.save_path)
