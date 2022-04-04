import os
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
        self.covis = extract_covisibility(paths.reference_sfm)
        self.pose_history = []
        self.cold_start = True
        self.pose = None
        self.reference_images = [self.localizer.model3d.name2id['mapping/IMG_9531.png']]

    def relocalize(self, query):
        if self.cold_start:
            self.camera = self.get_query_camera(query)
            self.cold_start = False
        ref_img = self.localizer.model3d.dbs[self.reference_images[0]]
        pose_init = Pose.from_Rt(ref_img.qvec2rotmat(),
                                 ref_img.tvec)
        self.pose = pose_init
        return 

    def update_reference_images(self):
        reference_images = self.reference_images
        return reference_images

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
        reference_images = self.update_reference_images()
        ref_img = self.localizer.model3d.dbs[reference_images[0]]
        translation = self.pose.numpy()[1]
        pose_init = Pose.from_Rt(ref_img.qvec2rotmat(), translation)
        ret = self.localizer.run_query(query,
                                self.camera,
                                pose_init,
                                reference_images)
        success = ret['success']
        return success

    def get_query_frame_iterator(self, image_folder):
        iterator = ImagePathIterator(image_folder)
        return iterator

if __name__ == '__main__':
    data_path = '/home/prajwal.chidananda/code/pixloc/datasets/Gimble'
    eval_path = '/home/prajwal.chidananda/code/pixloc/datasets/Gimble'
    loc_path =  '/home/prajwal.chidananda/code/pixloc/outputs/hloc/Gimble'
    tracker = PixLocPoseTrackerR1(data_path=data_path,
                                  eval_path=eval_path,
                                  loc_path=loc_path)
    query_path = os.path.join(data_path, 'query', 'IMG_3813')
    tracker.run(query_path)
    print('Done')

