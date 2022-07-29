import argparse
import ast
import json
import os
from pathlib import Path
import sys

import numpy as np
import tqdm
import cv2

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, triangulation

from pixtrack.utils.ingp_utils import load_nerf2sfm, initialize_ingp, sfm_to_nerf_pose


def create_features_matches(images, outputs):
    images = Path(images)
    outputs = Path(outputs)
    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    matcher_conf['model']['weights'] = 'indoor'
    
    references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
    print(len(references), "mapping images")
    extract_features.main(feature_conf, images, image_list=references, feature_path=features)
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
    testbed.fov = transforms['camera_angle_x'] * 180 / np.pi
    height = int(transforms['h'])
    width = int(transforms['w'])
    transform_list = list(enumerate(transforms['frames']))
    for i in tqdm.tqdm(range(len(transform_list))):
        frame = transform_list[i][1]
        transform_matrix = np.matrix(frame['transform_matrix'])[:-1, :]
        testbed.set_nerf_camera_matrix(transform_matrix)
        nerf_img = testbed.render(width, height, spp, True)
        nerf_img = nerf_img[:, :, :3] * 255.
        nerf_img = nerf_img.astype(np.uint8)
        nerf_img = cv2.cvtColor(nerf_img, cv2.COLOR_BGR2RGB)
        name = frame['file_path'].split('/')[-1]
        path = os.path.join(out_dir, name)
        cv2.imwrite(path, nerf_img)

def triangulate_nerf_views(ref_sfm_path, out_dir):
    images = Path(out_dir)
    outputs = Path(out_dir)
    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    features, matches, sfm_pairs = create_features_matches(images=images, outputs=outputs)
    
    reference_model = Path(ref_sfm_path)
    print('Running triangulation')
    model = triangulation.main(sfm_dir, reference_model, images,
                              sfm_pairs, features, matches,
                              skip_geometric_verification=False,
                              min_match_score=None, verbose=True)
    return model


if __name__ == '__main__':
    obj = Path(os.environ['OBJECT'])
    out_dir = Path(os.environ['PIXTRACK_OUTPUTS']) / 'nerf_sfm' / obj
    ref_sfm = Path(os.environ['PIXSFM_OUTPUTS']) / obj / 'ref'
    nerf_weights = Path(os.environ['SNAPSHOT_PATH']) / 'weights.msgpack'
    nerf_transforms = Path(os.environ['PIXSFM_DATASETS']) / obj / 'transforms.json'
    obj_aabb = os.environ['OBJ_AABB']
    obj_aabb = np.array(ast.literal_eval(obj_aabb)).copy()

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=out_dir)
    parser.add_argument('--ref_sfm', default=ref_sfm)
    parser.add_argument('--nerf_weights', default=nerf_weights)
    parser.add_argument('--nerf_transforms', default=nerf_transforms)
    args = parser.parse_args()

    nerf_im_dir = args.out_dir / 'mapping'
    if not os.path.isdir(nerf_im_dir):
        os.makedirs(nerf_im_dir)
    render_nerf_views(str(args.nerf_weights), 
                      str(args.nerf_transforms), 
                      str(nerf_im_dir),
                      obj_aabb,)
    triangulate_nerf_views(str(args.ref_sfm), 
                           str(args.out_dir))

