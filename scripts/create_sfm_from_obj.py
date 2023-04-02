import os
import argparse
from pathlib import Path
import torch
import numpy as np
import pycolmap
import cv2
from tqdm import tqdm
from pixtrack.utils.pytorch3d_render_utils import (
    create_look_at_poses_for_mesh,
    render_image,
    create_colmap_camera,
    create_colmap_image_from_pytorch3d_RT,
)
from hloc import extract_features, match_features, pairs_from_exhaustive
import hloc.triangulation as triangulation
from hloc.triangulation import (
    create_db_from_model,
    import_features,
    import_matches,
    geometric_verification,
)
from hloc.reconstruction import create_empty_db, import_images, get_image_ids
from hloc.utils.read_write_model import (
    write_images_binary,
    write_points3D_binary,
    write_cameras_binary,
)


def render_dataset_and_get_colmap_images_and_cameras(
    mesh_path, dataset_path, fx, fy, cx, cy, W, H, k1, device, subdivisions=2
):
    Rs, Ts, mesh = create_look_at_poses_for_mesh(
        fx, fy, W, H, mesh_path, subdivisions=subdivisions, device=device
    )
    dataset_path.mkdir(parents=True, exist_ok=True)
    mapping_path = dataset_path / "mapping"
    mapping_path.mkdir(parents=True, exist_ok=True)
    colmap_images = {}
    colmap_camera_id = 1
    colmap_camera = create_colmap_camera(W, H, fx, cx, cy, k1)
    colmap_cameras = {colmap_camera_id: colmap_camera}
    for i in tqdm(range(Rs.shape[0])):
        R = Rs[i]
        T = Ts[i]
        image = render_image(mesh, fx, fy, cx, cy, W, H, R.unsqueeze(0), T.unsqueeze(0))
        image = (image[..., :3] * 255.0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_name = ("%d.png" % (i + 1)).zfill(8)
        cv2.imwrite(str(mapping_path / img_name), image)

        colmap_image_name = "mapping/" + img_name
        colmap_image_id = i + 1
        colmap_image = create_colmap_image_from_pytorch3d_RT(
            R, T, colmap_image_name, colmap_image_id, colmap_camera_id
        )
        colmap_images[colmap_image_id] = colmap_image
    return colmap_images, colmap_cameras


def create_sfm_from_colmap_images_and_cameras(
    dataset_path, output_path, colmap_images, colmap_cameras
):
    # Configure SFM
    images = Path(dataset_path)
    outputs = Path(output_path)
    sfm_pairs = outputs / "pairs-sfm.txt"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"
    raw_dir = outputs / "raw"
    sfm_dir = outputs / "ref"
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superglue"]
    matcher_conf["model"]["weights"] = "indoor"
    references = [str(p.relative_to(images)) for p in (images / "mapping/").iterdir()]
    print(len(references), "mapping images")

    # Extract features and matches
    extract_features.main(
        feature_conf, images, image_list=references, feature_path=features
    )
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    pairs = sfm_pairs
    assert features.exists(), f"Failed to find features. Found {features} instead"
    assert pairs.exists(), f"Failed to find pairs. Found {pairs} instead"
    assert matches.exists(), f"Failed to find matches. Found {matches} instead"

    # Create a database and import images, features and matches
    raw_dir.mkdir(parents=True, exist_ok=True)
    database = raw_dir / "database.db"
    create_empty_db(database)
    camera_mode = pycolmap.CameraMode.AUTO.name
    import_images(
        image_dir=images,
        database_path=database,
        camera_mode=camera_mode,
        image_list=None,
    )
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score=None,
        skip_geometric_verification=False,
    )
    geometric_verification(database, pairs, verbose=True)

    # Create images, cameras and points3D binaries
    write_images_binary(colmap_images, raw_dir / "images.bin")
    write_cameras_binary(colmap_cameras, raw_dir / "cameras.bin")
    write_points3D_binary({}, raw_dir / "points3D.bin")  # Should be empty

    # Triangulate for 3D points
    reference = pycolmap.Reconstruction(raw_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    create_db_from_model(reference, database)
    triangulation.main(
        sfm_dir,
        raw_dir,
        images,
        pairs,
        features,
        matches,
        skip_geometric_verification=False,
        min_match_score=None,
        verbose=True,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True, type=Path)
    args = parser.parse_args()

    object_name = args.mesh_path.parent.name
    dir_path = args.mesh_path.parent
    default_output_path = dir_path / "pixtrack/pixsfm/outputs"
    default_dataset_path = dir_path / "pixtrack/pixsfm/dataset"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # TODO @prajwal: Choose these parameters automatically instead on hardcoding them
    fx = 900.0 * 3
    fy = 900.0 * 3
    cx = 512.0 * 3
    cy = 512.0 * 3
    W = 1024 * 3
    H = 1024 * 3
    k1 = 0.0

    # fx = 1.066778e+03 * 1
    # fy = 1.066778e+03 * 1
    # cx = 3.129869e+02 * 1
    # cy = 2.413109e+02 * 1
    # W = 640 * 1
    # H = 480 * 1
    # k1 = 0.0
    dataset_path = default_dataset_path
    output_path = default_output_path
    mesh_path = args.mesh_path

    # Render images from obj, get colmap images and cameras
    colmap_images, colmap_cameras = render_dataset_and_get_colmap_images_and_cameras(
        mesh_path, dataset_path, fx, fy, cx, cy, W, H, k1, device
    )

    # Create sfm from colmap images and cameras
    create_sfm_from_colmap_images_and_cameras(
        dataset_path, output_path, colmap_images, colmap_cameras
    )
