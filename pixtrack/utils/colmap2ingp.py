#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
import pickle as pkl
import colmap_read_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert a bin colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place"
    )

    parser.add_argument(
        "--video_in",
        default="",
        help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also",
    )
    parser.add_argument("--video_fps", default=2)
    parser.add_argument(
        "--run_colmap", action="store_true", help="run colmap first on the image folder"
    )
    parser.add_argument(
        "--colmap_matcher",
        default="sequential",
        choices=["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"],
        help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images",
    )
    parser.add_argument(
        "--colmap_db", default="colmap.db", help="colmap database filename"
    )
    parser.add_argument("--images", default="images", help="input path to the images")
    parser.add_argument(
        "--bin",
        default="bin",
        help="input path to the colmap bin files (set automatically if run_colmap is used)",
    )
    parser.add_argument(
        "--aabb_scale",
        default=16,
        choices=["1", "2", "4", "8", "16"],
        help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16",
    )
    parser.add_argument(
        "--skip_early", default=0, help="skip this many images from the start"
    )
    parser.add_argument("--out", default="transforms.json", help="output path")
    args = parser.parse_args()
    return args


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def run_ffmpeg(args):
    if not os.path.isabs(args.images):
        args.images = os.path.join(os.path.dirname(args.video_in), args.images)
    images = args.images
    video = args.video_in
    fps = float(args.video_fps) or 1.0
    print(
        f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}."
    )
    if (
        input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)")
        .lower()
        .strip()
        + "y"
    )[:1] != "y":
        sys.exit(1)
    try:
        shutil.rmtree(images)
    except:
        pass
    do_system(f"mkdir {images}")
    do_system(
        f'ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf "fps={fps}" {images}/%04d.jpg'
    )


def run_colmap(args):
    db = args.colmap_db
    images = args.images
    db_noext = str(Path(db).with_suffix(""))

    if args.bin == "bin":
        args.bim = db_noext + "_bin"
    bin = args.bin
    sparse = db_noext + "_sparse"
    print(
        f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\tbin={bin}"
    )
    if (
        input(
            f"warning! folders '{sparse}' and '{bin}' will be deleted/replaced. continue? (Y/n)"
        )
        .lower()
        .strip()
        + "y"
    )[:1] != "y":
        sys.exit(1)
    if os.path.exists(db):
        os.remove(db)
    do_system(
        f"colmap feature_extractor --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 --database_path {db} --image_path {images}"
    )
    do_system(f"colmap {args.colmap_matcher}_matcher --database_path {db}")
    try:
        shutil.rmtree(sparse)
    except:
        pass
    do_system(f"mkdir {sparse}")
    do_system(
        f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse}"
    )
    do_system(
        f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1"
    )
    try:
        shutil.rmtree(bin)
    except:
        pass
    do_system(f"mkdir {bin}")
    do_system(
        f"colmap model_converter --input_path {sparse}/0 --output_path {bin} --output_type TXT"
    )


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(
    oa, da, ob, db
):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta < 0:
        ta = 0
    if tb < 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


if __name__ == "__main__":
    args = parse_args()
    if args.video_in != "":
        run_ffmpeg(args)
    if args.run_colmap:
        run_colmap(args)
    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)
    IMAGE_FOLDER = args.images
    BIN_FOLDER = args.bin
    OUT_PATH = args.out
    print(f"outputting to {OUT_PATH}...")
    cameras, images, points3D = colmap_read_model.read_model(
        path=BIN_FOLDER, ext=".bin"
    )
    angle_x = math.pi / 2
    # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
    # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
    # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
    camera = cameras[1]
    w = float(camera.width)
    h = float(camera.height)
    fl_x = float(camera.params[0])
    fl_y = float(camera.params[0])
    k1 = 0
    k2 = 0
    p1 = 0
    p2 = 0
    cx = w / 2
    cy = h / 2
    if camera.model == "SIMPLE_RADIAL":
        w = float(camera.width)
        h = float(camera.height)
        fl_x = float(camera.params[0])
        fl_y = float(camera.params[0])
        cx = float(camera.params[1])
        cy = float(camera.params[2])
        k1 = float(camera.params[3])
    else:
        raise Exception("Not implemented")
        print("unknown camera model ", els[1])
    # fl = 0.5 * w / tan(0.5 * angle_x);
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    fovx = angle_x * 180 / math.pi
    fovy = angle_y * 180 / math.pi

    print(
        f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} "
    )

    i = 0
    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": AABB_SCALE,
        "frames": [],
    }
    centroid = np.zeros(3)
    up = np.zeros(3)
    # elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9 is filename
    # name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
    # why is this requireing a relitive path while using ^
    for key in images:
        image_rel = os.path.relpath(IMAGE_FOLDER)
        image = images[key]
        name = image.name
        name = name.split("/")[-1]
        name = os.path.join(image_rel, name)
        b = sharpness(name)
        print(name, "sharpness=", b)
        image_id = key
        qvec = image.qvec
        tvec = image.tvec
        # qvec = np.array(tuple(map(float, elems[1:5])))
        # tvec = np.array(tuple(map(float, elems[5:8])))
        R = qvec2rotmat(-qvec)
        t = tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(m)
        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
        c2w[2, :] *= -1  # flip whole world upside down

        centroid += c2w[0:3, 3]
        up += c2w[0:3, 1]

        relative_path = os.path.join(os.path.basename(image_rel), name.split("/")[-1])
        frame = {"file_path": relative_path, "sharpness": b, "transform_matrix": c2w}
        out["frames"].append(frame)
    nframes = len(out["frames"])
    centroid *= 1 / nframes
    up = up / np.linalg.norm(up)
    print("up vector was ", up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    avglen = 0.0
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= centroid
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes
    print("avg camera distance from origin ", avglen)

    for f in out["frames"]:
        # print(f["transform_matrix"])
        f["transform_matrix"][0:3, 3] *= 3.0 / avglen  # scale to "nerf sized"
        f["transform_matrix"] = np.matmul(
            R, f["transform_matrix"]
        )  # rotate up to be the z axis
        # f["transform_matrix"][2,3]+=0.5 # shift up a bit as cameras under ground are rare
        # print(f["transform_matrix"])

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0
    totp = [0, 0, 0]
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(totp)  # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    print(f"writing {OUT_PATH}")
    nerf2sfm_dict = {
        "up": up,
        "centroid": centroid,
        "avglen": avglen,
        "totp": totp,
        "R": R,
    }
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    n2s_path = os.path.join(OUT_PATH, "nerf2sfm.pkl")

    with open(n2s_path, "wb") as f:
        pkl.dump(nerf2sfm_dict, f)
    t_path = os.path.join(OUT_PATH, "transforms.json")
    with open(t_path, "w") as outfile:
        json.dump(out, outfile, indent=2)
