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
        import pdb; pdb.set_trace()
        paths = default_paths.add_prefixes(data_path, 
                                           loc_path, 
                                           eval_path)
        self.localizer = PoseTrackerLocalizer(paths, conf)
        self.covis = extract_covisibility.main(paths.reference_sfm)
        self.pose_history = []
        self.cold_start = True
        self.pose = None
        self.reference_frames = [self.localizer.model3d.name2id['mapping/IMG_9531.png']]

    def relocalize(self):
        if self.cold_start:
            self.camera = self.get_query_camera(query)
            self.cold_start = False
        ref_img = self.localizer.model3d.dbs[reference_images[0]]
        pose_init = Pose.from_Rt(ref_img.qvec2rotmat(),
                                 ref_img.tvec)
        self.pose = pose_init
        return 

    def update_reference_frames(self):
        reference_frames = self.reference_frames
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
            self.relocalize()
            return True
        reference_images = self.update_reference_frames()
        ref_img = self.localizer.model3d.dbs[reference_images[0]]
        translation = self.pose.numpy()[1]
        pose_init = Pose.from_Rt(ref_img.qvec2rotmat(), translation)
        ret = self.localizer.run_query(query,
                                self.camera,
                                pose_init,
                                reference_images[0])
        success = ret['success']
        return success

    def get_query_frame_iterator(self, image_folder):
        iterator = ImagePathStreamer(image_folder)
        return iterator

if __name__ == '__main__':
    data_path = '~/code/pixloc/datasets/Gimble'
    eval_path = '~/code/pixloc/datasets/Gimble'
    loc_path = '~/code/pixloc/outputs/hloc/Gimble'
    tracker = PixLocPoseTrackerR1(data_path=data_path,
                                  eval_path=eval_path,
                                  loc_path=loc_path)
    import pdb; pdb.set_trace()
    query_path = os.path.join(data_path, 'query', 'IMG_3813')
    tracker.run(query_path)
    print('Done')

