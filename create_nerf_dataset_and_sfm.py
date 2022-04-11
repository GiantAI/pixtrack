import os
import sys
import json
import numpy as np
import tqdm
import cv2
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, triangulation
import argparse
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

def render_nerf_views(nerf_weights, transforms_file, out_dir):
    spp = 8
    with open(transforms_file) as f:
        transforms = json.load(f)
    testbed = initialize_ingp(nerf_weights)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='outputs/nerf_sfm/gimble_04MAR2022')
    parser.add_argument('--ref_sfm', default='pixel-perfect-sfm/outputs/gimble_04MAR2022/ref')
    parser.add_argument('--nerf_weights', default='instant-ngp/snapshots/gimble_04MAR2022/weights.msgpack')
    parser.add_argument('--nerf_transforms', default='instant-ngp/data/nerf/gimble_04MAR2022/transforms.json')
    args = parser.parse_args()

    nerf_im_dir = os.path.join(args.out_dir, 'mapping')
    if not os.path.isdir(nerf_im_dir):
        os.makedirs(nerf_im_dir)
    render_nerf_views(args.nerf_weights, args.nerf_transforms, nerf_im_dir)
    triangulate_nerf_views(args.ref_sfm, args.out_dir)




