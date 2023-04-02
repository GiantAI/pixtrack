import argparse
import ast
import re
import os

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tqdm
import pickle as pkl
import pycolmap

from pixtrack.utils.pose_utils import (
    get_world_in_camera_from_pixpose,
    get_camera_in_world_from_pixpose,
    rotate_image,
)


def project_3d_to_2d(pts_3d, K=np.eye(3)):
    pts_2d = K @ pts_3d.T
    pts_2d = pts_2d / pts_2d[2, :]
    pts_2d = pts_2d[:2, :].T
    return pts_2d


def draw_axes(image, pts_3d, K=np.eye(3), t=10):
    pts_2d = project_3d_to_2d(pts_3d, K).astype(np.int16)
    image = cv2.line(image, pts_2d[0], pts_2d[1], (255, 0, 0), int(t))
    image = cv2.line(image, pts_2d[2], pts_2d[3], (0, 255, 0), int(t))
    image = cv2.line(image, pts_2d[4], pts_2d[5], (0, 0, 255), int(t))
    return image


def read_images_from_folder(folder_path: str):
    images = {}
    # A regex to extract the integer from a string.
    regex = re.compile(r"\d+")
    lsorted = sorted(
        os.listdir(folder_path),
        key=lambda x: int(regex.findall(os.path.splitext(x)[0])[0]),
    )
    for image_name in lsorted:
        image = np.array(Image.open(os.path.join(folder_path, image_name)))
        images[image_name] = image
    return images


def add_pose_axes(
    image,
    camera,
    pose,
    axes_center=[0.1179, 1.1538, 1.3870, 0.0],
):
    width, height = camera.size
    focal = float(camera.f[0])

    u = float(width / 2)
    v = float(height / 2)
    K = [[focal, 0.0, u], [0.0, focal, v], [0.0, 0.0, 1.0]]
    K = np.array(K)
    x, y, z = 0.0, 0.0, 0.0
    s = 0.25
    t = 5.0
    axes = [
        [x, y, z],
        [x + s, y, z],
        [x, y, z],
        [x, y - s, z],
        [x, y, z],
        [x, y, z - s],
    ]
    axes = np.array(axes)
    axes = np.hstack((axes, np.ones((axes.shape[0], 1))))
    axes += np.array(axes_center)
    pts_3d = axes @ np.linalg.inv(pose).T[:, :3]
    result_img = draw_axes(image.copy(), pts_3d, K)
    return result_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder where the images get stored",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder with the input images on which the pose axes are added.",
    )
    parser.add_argument(
        "--pixtrack_output",
        type=str,
        required=True,
        help="Pixtrack output for an object",
    )
    args = parser.parse_args()
    obj = os.environ["OBJECT"]

    sfm_dir = (
        Path(os.environ["PIXTRACK_OUTPUTS"]) / "nerf_sfm" / ("aug_%s" % obj) / "aug_sfm"
    )
    recon = pycolmap.Reconstruction(sfm_dir)
    object_center = ast.literal_eval(os.environ["OBJ_CENTER"]) + [0]
    poses_path = Path(args.pixtrack_output) / "poses.pkl"
    pose_stream = pkl.load(open(poses_path, "rb"))
    input_images = read_images_from_folder(args.input_folder)
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    for number, name_q in enumerate(tqdm.tqdm(pose_stream)):
        path_q = pose_stream[name_q]["query_path"]
        ref_ids = pose_stream[name_q]["reference_ids"]
        camera = pose_stream[name_q]["camera"]

        if "T_refined" not in pose_stream[name_q]:
            continue
        wIc_pix = pose_stream[name_q]["T_refined"]
        cIw_sfm = get_camera_in_world_from_pixpose(wIc_pix)
        if (
            not name_q in list(input_images.keys())[number]
            and list(input_images.keys())[number] not in name_q
        ):
            print(name_q, list(input_images.keys())[number])
            assert False, "something went wront in the image ordering"
        result_img = add_pose_axes(
            input_images[list(input_images.keys())[number]],
            camera,
            cIw_sfm,
            object_center,
        )
        Image.fromarray(result_img).save(os.path.join(args.output_folder, name_q))
