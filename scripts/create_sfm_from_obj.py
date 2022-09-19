import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pycolmap
import cv2
from tqdm import tqdm
from pixtrack.utils.pytorch3d_render_utils import (create_look_at_poses_for_mesh, 
                                                   render_image,
                                                   create_colmap_camera,
                                                   create_colmap_image_from_pytorch3d_RT)
from hloc import (extract_features, 
                  match_features, 
                  reconstruction, 
                  pairs_from_exhaustive)
import hloc.triangulation as triangulation
from hloc.triangulation import (run_triangulation, 
                                create_db_from_model,
                                import_features, 
                                import_matches, 
                                geometric_verification)
from hloc.reconstruction import (create_empty_db, 
                                 import_images, 
                                 get_image_ids)
from hloc.utils.read_write_model import (write_images_binary, 
                                         write_points3D_binary,
                                         write_cameras_binary)


def render_dataset_and_get_colmap_images_and_cameras(mesh_path, dataset_path, 
                                                     fx, fy, cx, cy, W, H):
    mesh_path = args.mesh_path
    Rs, Ts, mesh = create_look_at_poses_for_mesh(mesh_path, subdivisions=2)
    dataset_path.mkdir(parents=True, exist_ok=True)
    mapping_path = dataset_path / 'mapping'
    mapping_path.mkdir(parents=True, exist_ok=True)
    colmap_images = {}
    colmap_camera_id = 1
    colmap_camera = create_colmap_camera(W, H, fx, cx, cy, k1=0.)
    colmap_cameras = {colmap_camera_id: colmap_camera}
    for i in tqdm(range(Rs.shape[0])):
        R = Rs[i].unsqueeze(0)
        T = Ts[i].unsqueeze(0)
        image = render_image(mesh, fx, fy, cx, cy, W, H, R, T)
        image = (image[..., :3] * 255.).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_name = ('%d.png' % (i + 1)).zfill(8)
        cv2.imwrite(str(mapping_path / img_name), image)

        colmap_image_name = 'mapping/' + img_name
        colmap_image_id = i + 1
        colmap_image = create_colmap_image_from_pytorch3d_RT(R[0], T[0], 
                                                          colmap_image_name,
                                                          colmap_image_id,
                                                          colmap_camera_id)
        colmap_images[colmap_image_id] = colmap_image
    return colmap_images, colmap_cameras

def create_sfm_from_colmap_images_and_cameras(dataset_path, output_path,
                                              colmap_images, colmap_cameras): 
    # Configure SFM
    images = Path(dataset_path)
    outputs = Path(output_path)
    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'
    raw_dir = outputs / "raw"
    sfm_dir = outputs / "sfm"
    feature_conf = extract_features.confs['superpoint_max']
    matcher_conf = match_features.confs['superglue']
    matcher_conf['model']['weights'] = 'indoor'
    references = [str(p.relative_to(images)) for p in (images / 'mapping/').iterdir()]
    print(len(references), "mapping images")
    
    # Extract features and matches
    extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    pairs = sfm_pairs
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    # Create a database and import images, features and matches
    raw_dir.mkdir(parents=True, exist_ok=True)
    database = raw_dir / 'database.db'
    create_empty_db(database)
    camera_mode = 'AUTO'
    import_images(image_dir=images, 
                  database_path=database, 
                  camera_mode=camera_mode, 
                  image_list=None)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score=None, 
                   skip_geometric_verification=False)
    geometric_verification(database, pairs, verbose=True)

    # Create images, cameras and points3D binaries
    write_images_binary(colmap_images, raw_dir / 'images.bin')
    write_cameras_binary(colmap_cameras, raw_dir / 'cameras.bin')
    write_points3D_binary({}, raw_dir / 'points3D.bin') #Should be empty

    # Triangulate for 3D points
    reference = pycolmap.Reconstruction(raw_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'
    image_ids2 = create_db_from_model(reference, database)
    triangulation.main(sfm_dir, raw_dir, images, 
                       pairs, features, matches,
                       skip_geometric_verification=False, 
                       min_match_score=None, verbose=True)
    return

if __name__ == '__main__':
    object_name = os.environ['OBJECT']
    default_output_path = Path(os.environ['PIXTRACK_OUTPUTS']) / 'nerf_sfm' / object_name
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', required=True, type=Path)
    parser.add_argument('--output_path', default=default_output_path, type=Path)
    parser.add_argument('--dataset_path', default=default_output_path, type=Path)
    args = parser.parse_args()

    #TODO @prajwal: Choose these parameters automatically instead on hardcoding them
    fx = 900. * 3
    fy = 900. * 3
    cx = 512. * 3
    cy = 512. * 3
    W = 1024 * 3
    H = 1024 * 3
    dataset_path = args.dataset_path
    output_path = args.output_path
    mesh_path = args.mesh_path

    # Render images from obj, get colmap images and cameras
    render_dataset_and_get_colmap_images_and_cameras(mesh_path, dataset_path, 
                                                     fx, fy, cx, cy, W, H)

    # Create sfm from colmap images and cameras
    create_sfm_from_colmap_images_and_cameras(dataset_path, output_path,
                                              colmap_images, colmap_cameras)

