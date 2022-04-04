from pixtrack.pose_trackers.base_pose_tracker import PoseTracker
from pixtrack.localizer import PoseTrackerLocalizer
from pixtrack.io import ImagePathStreamer

from hloc import extract_covisibility

import pycolmap
from pycolmap import infer_camera_from_image

from pixloc.utils.colmap import Camera


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
                'experiment': 'R1',
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
        paths = default_paths.add_prefixes(data_path, 
                                           loc_path, 
                                           eval_path)
        self.localizer = PoseTrackerLocalizer(paths, conf)
        self.covis = extract_covisibility.main(paths.reference_sfm)
        self.pose_history = []
        self.cold_start = True
        self.pose = None

    def relocalize(self):
        if self.cold_start:
            self.camera = self.get_query_camera(query)
            self.cold_start = False
        return 

    def update_reference_frames(self):
        reference_frames = []
        return reference_frames

    def get_query_camera(self, query):
        camera = infer_camera_from_image(query)
        camera = Camera(None, 
                        camera.model_name,
                        int(camera.width),
                        int(camera.height),
                        camera.params)
        return camera

    def refine(self, query):
        if self.cold_start:
            self.camera = self.get_query_camera(query)
            self.cold_start = False
        reference_images = self.update_reference_frames()
        ret = self.localizer.run_query(query, 
                                self.camera,
                                reference_images[0])
        success = ret['success']
        return success

    def get_query_frame_iterator(self, image_folder):
        iterator = ImagePathStreamer(image_folder)
        return iterator

