import tqdm
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive, visualization
from hloc.visualization import plot_images, read_image
from hloc.utils.viz_3d import init_figure, plot_points, plot_reconstruction, plot_camera_colmap

from pixsfm.util.visualize import init_image, plot_points2D
from pixsfm.refine_hloc import PixSfM
from pixsfm import ostream_redirect

import pycolmap

# redirect the C++ outputs to notebook cells
cpp_out = ostream_redirect(stderr=True, stdout=True)
cpp_out.__enter__()

images = Path('pixel-perfect-sfm/datasets/gimble_04MAR2022')
outputs = Path('pixel-perfect-sfm/outputs/gimble_04MAR2022')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'
raw_dir = outputs / "raw"
ref_dir = outputs / "ref"

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']
matcher_conf['model']['weights'] = 'indoor'

references = [str(p.relative_to(images)) for p in (images / 'mapping/').iterdir()]
print(len(references), "mapping images")

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

sfm = PixSfM({"dense_features": {"max_edge": 1024}})
refined, sfm_outputs = sfm.reconstruction(ref_dir, images, sfm_pairs, features, matches, 
                                          image_list=references, camera_mode=pycolmap.CameraMode.SINGLE)

