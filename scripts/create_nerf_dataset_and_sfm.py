import argparse
import ast
import json
import os
from pathlib import Path
import sys

import numpy as np
import tqdm
import cv2

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
    triangulation,
)

from pixtrack.utils.ingp_utils import (
    load_nerf2sfm,
    initialize_ingp,
    sfm_to_nerf_pose,
    get_nerf_aabb_from_sfm,
)


def create_features_matches(images, outputs):
    images = Path(images)
    outputs = Path(outputs)
    sfm_pairs = outputs / "pairs-sfm.txt"
    loc_pairs = outputs / "pairs-loc.txt"
    sfm_dir = outputs / "ref"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]
    matcher_conf["model"]["weights"] = "indoor"

    references = [
        p.relative_to(images).as_posix() for p in (images / "mapping/").iterdir()
    ]
    print(len(references), "mapping images")
    extract_features.main(
        feature_conf, images, image_list=references, feature_path=features
    )
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    return features, matches, sfm_pairs


def render_nerf_views(nerf_weights, transforms_file, out_dir, aabb):
    spp = 8
    with open(transforms_file) as f:
        transforms = json.load(f)
    # The default is the gimbal.
    testbed = initialize_ingp(
        snapshot_path=nerf_weights,
        aabb=aabb,
    )
    testbed.fov = transforms["camera_angle_x"] * 180 / np.pi
    height = int(transforms["h"])
    width = int(transforms["w"])
    transform_list = list(enumerate(transforms["frames"]))
    for i in tqdm.tqdm(range(len(transform_list))):
        frame = transform_list[i][1]
        transform_matrix = np.matrix(frame["transform_matrix"])[:-1, :]
        testbed.set_nerf_camera_matrix(transform_matrix)
        nerf_img = testbed.render(width, height, spp, True)
        nerf_img = nerf_img[:, :, :3] * 255.0
        nerf_img = nerf_img.astype(np.uint8)
        nerf_img = cv2.cvtColor(nerf_img, cv2.COLOR_BGR2RGB)
        name = frame["file_path"].split("/")[-1]
        path = os.path.join(out_dir, name)
        cv2.imwrite(path, nerf_img)


def triangulate_nerf_views(ref_sfm_path, out_dir):
    images = Path(out_dir)
    outputs = Path(out_dir)
    sfm_pairs = outputs / "pairs-sfm.txt"
    loc_pairs = outputs / "pairs-loc.txt"
    sfm_dir = outputs / "ref"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    features, matches, sfm_pairs = create_features_matches(
        images=images, outputs=outputs
    )

    reference_model = Path(ref_sfm_path)
    print("Running triangulation")
    model = triangulation.main(
        sfm_dir,
        reference_model,
        images,
        sfm_pairs,
        features,
        matches,
        skip_geometric_verification=False,
        min_match_score=None,
        verbose=True,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=Path)
    parser.add_argument("--object_aabb", type=str, default="")
    args = parser.parse_args()

    obj_path = args.object_path
    out_dir = obj_path / "pixtrack/nerf_sfm"
    ref_sfm = obj_path / "pixtrack/pixsfm/outputs/ref"
    nerf_weights = obj_path / "pixtrack/instant-ngp/snapshots/weights.msgpack"
    nerf_transforms = obj_path / "pixtrack/pixsfm/dataset/transforms.json"
    nerf2sfm_path = obj_path / "pixtrack/pixsfm/dataset/nerf2sfm.pkl"

    nerf_im_dir = out_dir / "mapping"
    if not os.path.isdir(nerf_im_dir):
        os.makedirs(nerf_im_dir)

    if args.object_aabb != "":
        obj_aabb = np.array(ast.literal_eval(args.object_aabb)).copy()
    else:
        obj_aabb = get_nerf_aabb_from_sfm(ref_sfm, nerf2sfm_path)

    render_nerf_views(
        str(nerf_weights),
        str(nerf_transforms),
        str(nerf_im_dir),
        obj_aabb,
    )
    triangulate_nerf_views(str(ref_sfm), str(out_dir))
