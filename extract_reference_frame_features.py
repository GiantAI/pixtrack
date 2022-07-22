import os
import sys
import h5py
import tqdm
import numpy as np
from pixloc.utils.data import Paths
from pixtrack.localization.pixloc_pose_refiners import PoseTrackerLocalizer
from pathlib import Path

class FeatureExtractor:
    def __init__(self, data_path, loc_path):
        default_paths = Paths(
                            query_images='query/',
                            reference_images=loc_path,
                            reference_sfm='aug_sfm',
                            query_list='*_with_intrinsics.txt',
                            global_descriptors='features.h5',
                            retrieval_pairs='pairs_query.txt',
                            results='pixloc_gimble.txt',)
        conf = {
                'experiment': 'pixloc_megadepth',
                'features': {},
                'optimizer': {
                              'num_iters': 150,
                              'pad': 1,
                             },
                'refinement': {
                               'num_dbs': 1,
                               'multiscale': [4, 1],
                               'point_selection': 'all',
                               'normalize_descriptors': True,
                               'average_observations': False,
                               'do_pose_approximation': False,
                              },
                }
        eval_path = ''
        paths = default_paths.add_prefixes(Path(data_path), 
                                           Path(loc_path), 
                                           Path(eval_path))
        self.localizer = PoseTrackerLocalizer(paths, conf)
        self.features_path = Path(loc_path) / 'reference_features.h5'
        self.reference_scale = 0.25
        self.localizer.refiner.reference_scale = self.reference_scale

    def write_features(self, ref_id, features):
        with h5py.File(str(self.features_path), 'a') as fd:
            for scale in features:
                p3dids = np.array(features[scale]['p3dids'])
                id_grp_name = '%s/%s' % (ref_id, scale)
                if id_grp_name in fd:
                    del fd[id_grp_name]
                id_grp = fd.create_group(id_grp_name)
                id_grp.create_dataset('p3dids', data=p3dids)

                for level in range(len(features[scale]['p3did_to_feat'][0])):
                    feature_list = [feats[level].cpu().numpy() for feats in features[scale]['p3did_to_feat']]
                    feature_array = np.array(feature_list)
                    feat_grp_name = '%s/%s/%s' % (ref_id, scale, level)
                    if feat_grp_name in fd:
                        del fd[feat_grp_name]
                    feat_grp = fd.create_group(feat_grp_name)
                    feat_grp.create_dataset('p3did_to_feat', data=feature_array)
                    
        return

    def run(self):
        reference_ids = list(self.localizer.model3d.dbs.keys())
        for ref_id in tqdm.tqdm(reference_ids):
            features = self.localizer.refiner.extract_reference_features([ref_id])
            self.write_features(ref_id, features)

if __name__ == '__main__':
    obj = os.environ['OBJECT']
    data_path = Path(os.environ['PIXSFM_DATASETS']) / obj
    loc_path = Path(os.environ['PIXTRACK_OUTPUTS']) / 'nerf_sfm' / ('aug_%s' % obj)
    extractor = FeatureExtractor(data_path=str(data_path),
                                 loc_path=str(loc_path))
    extractor.run()

