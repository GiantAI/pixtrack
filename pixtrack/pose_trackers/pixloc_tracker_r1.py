import os
import pickle as pkl
from pathlib import Path

from pixtrack.pose_trackers.base_pose_tracker import PoseTracker
from pixtrack.localization.pixloc_pose_refiners import PoseTrackerLocalizer
from pixtrack.utils.io import ImagePathIterator
from pixtrack.utils.hloc_utils import extract_covisibility

import pycolmap
from pycolmap import infer_camera_from_image

from pixloc.utils.data import Paths
from pixloc.utils.colmap import Camera as ColCamera
from pixloc.pixlib.geometry import Camera as PixCamera, Pose


class PixLocPoseTrackerR1(PoseTracker):
    def __init__(self, data_path, loc_path, eval_path):
        default_paths = Paths(
                            query_images='query/',
                            reference_images='',
                            reference_sfm='sfm',
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
                               'num_dbs': 3,
                               'multiscale': [4, 1],
                               'point_selection': 'all',
                               'normalize_descriptors': True,
                               'average_observations': False,
                               'do_pose_approximation': False,
                              },
                }
        paths = default_paths.add_prefixes(Path(data_path), 
                                           Path(loc_path), 
                                           Path(eval_path))
        self.localizer = PoseTrackerLocalizer(paths, conf)
        self.eval_path = eval_path
        self.covis = extract_covisibility(paths.reference_sfm)
        self.pose_history = {}
        self.cold_start = True
        self.pose = None
        self.reference_ids = [self.localizer.model3d.name2id['mapping/IMG_9531.png']]

    def relocalize(self, query):
        if self.cold_start:
            self.camera = self.get_query_camera(query)
            self.cold_start = False
        ref_img = self.localizer.model3d.dbs[self.reference_ids[0]]
        pose_init = Pose.from_Rt(ref_img.qvec2rotmat(),
                                 ref_img.tvec)
        self.pose = pose_init
        return 

    def update_reference_ids(self):
        reference_ids = self.reference_ids
        return reference_ids

    def get_query_camera(self, query):
        camera = infer_camera_from_image(query)
        camera = ColCamera(None, 
                        camera.model_name,
                        int(camera.width),
                        int(camera.height),
                        camera.params)
        camera = PixCamera.from_colmap(camera)
        return camera

    def refine(self, query):
        if self.cold_start:
            self.relocalize(query)
            return True
        reference_ids = self.update_reference_ids()
        ref_img = self.localizer.model3d.dbs[reference_ids[0]]
        translation = self.pose.numpy()[1]
        pose_init = Pose.from_Rt(ref_img.qvec2rotmat(), translation)
        ret = self.localizer.run_query(query,
                                self.camera,
                                pose_init,
                                reference_ids)
        success = ret['success']
        if success:
            ret['camera'] = self.camera
            ret['reference_ids'] = reference_ids
            ret['query_path'] = query
            self.pose = ret['T_refined']
            img_name = os.path.basename(query)
            self.pose_history[img_name] = ret
        return success

    def get_query_frame_iterator(self, image_folder):
        iterator = ImagePathIterator(image_folder)
        return iterator

    def save_poses(self):
        path = os.path.join(self.eval_path, 'poses.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self.pose_history, f)

if __name__ == '__main__':
    exp_name = 'IMG_3813'
    data_path = '/home/prajwal.chidananda/code/pixtrack/outputs/nerf_sfm/gimble_04MAR2022'
    eval_path = '/home/prajwal.chidananda/code/pixtrack/outputs/%s' % exp_name
    loc_path =  '/home/prajwal.chidananda/code/pixtrack/outputs/nerf_sfm/gimble_04MAR2022'
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)
    tracker = PixLocPoseTrackerR1(data_path=data_path,
                                  eval_path=eval_path,
                                  loc_path=loc_path)
    query_path = os.path.join(data_path, 'query', exp_name)
    tracker.run(query_path)
    tracker.save_poses()
    print('Done')

