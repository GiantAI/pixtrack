import os
import numpy as np
from pathlib import Path

from pixtrack.pose_trackers.pixloc_tracker_r1 import PixLocPoseTrackerR1
from pixtrack.utils.pose_utils import geodesic_distance_for_rotations, get_camera_in_world_from_pixpose
from pixloc.localization import SimpleTracker
from pixloc.pixlib.geometry import Pose

from pixtrack.pose_trackers.base_pose_tracker import PoseTracker
from pixtrack.localization.pixloc_pose_refiners import PoseTrackerLocalizer
from pixtrack.utils.io import ImagePathIterator
from pixtrack.utils.hloc_utils import extract_covisibility
from pixtrack.utils.ingp_utils import load_nerf2sfm, initialize_ingp, sfm_to_nerf_pose
from pixtrack.visualization.run_vis_on_poses import get_nerf_image
from pixloc.pixlib.geometry import Camera as PixCamera

from pixloc.utils.data import Paths
import cv2

class PixLocPoseTrackerR6(PixLocPoseTrackerR1):
    def __init__(self, data_path, loc_path, eval_path):
        default_paths = Paths(
                            query_images='query/',
                            reference_images='',
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
        nerf_path = '/home/prajwal.chidananda/code/pixtrack/instant-ngp/snapshots/gimble_04MAR2022/weights.msgpack'
        nerf2sfm_path = '/home/prajwal.chidananda/code/pixtrack/instant-ngp/data/nerf/gimble_04MAR2022/nerf2sfm.pkl'
        self.nerf2sfm = load_nerf2sfm(nerf2sfm_path)
        self.testbed = initialize_ingp(nerf_path)

    def update_reference_ids(self):
        curr_refs = self.reference_ids
        curr_pose = self.pose
        R_qry = curr_pose.numpy()[0]
        cimg = self.localizer.model3d.dbs[curr_refs[0]]
        R_ref = cimg.qvec2rotmat()
        curr_gdist = geodesic_distance_for_rotations(R_qry, R_ref)

        covis = self.covis[curr_refs[0]]
        N = 50
        covis = {k: covis[k] for k in covis if covis[k] > N}
        gdists = {curr_refs[0]: curr_gdist}
        for ref in covis:
            cimg = self.localizer.model3d.dbs[ref]
            R_ref = cimg.qvec2rotmat()
            gdist = geodesic_distance_for_rotations(R_qry, R_ref)
            gdists[ref] = gdist

        reference_ids = sorted(gdists, key=lambda x: gdists[x])
        K = 2
        self.reference_ids = reference_ids[:K]
        return self.reference_ids

    def get_reference_image(self, pose):
        cIw = get_camera_in_world_from_pixpose(pose)
        nerf_pose = sfm_to_nerf_pose(self.nerf2sfm, cIw)
        ref_camera = self.localizer.model3d.cameras[1]
        ref_camera = PixCamera.from_colmap(ref_camera)
        nerf_img = get_nerf_image(self.testbed, nerf_pose, ref_camera)
        #nerf_img = cv2.cvtColor(nerf_img, cv2.COLOR_BGR2RGB)
        return nerf_img

    def refine(self, query):
        if self.cold_start:
            self.relocalize(query)
            self.cold_start = False
        
        reference_ids = self.update_reference_ids()
        reference_image = self.get_reference_image(self.pose)
        translation = self.pose.numpy()[1]
        trackers = {}
        rets = {}
        costs = {}
        for ref_id in reference_ids:
            ref_img = self.localizer.model3d.dbs[ref_id]
            pose_init = Pose.from_Rt(ref_img.qvec2rotmat(), translation)
            tracker = SimpleTracker(self.localizer.refiner)
            ret = self.localizer.run_query(query,
                                self.camera,
                                pose_init,
                                [ref_id],
                                pose=self.pose,
                                reference_images_raw=[reference_image])
            rets[ref_id] = ret
            trackers[ref_id] = tracker
            avg_cost = np.mean([x[-1] for x in tracker.costs])
            costs[ref_id] = avg_cost
        best_ref_id = min(costs, key=costs.get)
        ret = rets[best_ref_id]
        success = ret['success']
        if success:
            self.pose = ret['T_refined']
        ret['camera'] = self.camera
        ret['reference_ids'] = reference_ids
        ret['query_path'] = query
        img_name = os.path.basename(query)
        self.pose_history[img_name] = ret
        return success
            

if __name__ == '__main__':
    exp_name = 'IMG_4117'
    data_path = '/home/prajwal.chidananda/code/pixtrack/outputs/nerf_sfm/aug_gimble_04MAR2022'
    eval_path = '/home/prajwal.chidananda/code/pixtrack/outputs/%s' % exp_name
    loc_path =  '/home/prajwal.chidananda/code/pixtrack/outputs/nerf_sfm/aug_gimble_04MAR2022'
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)
    tracker = PixLocPoseTrackerR6(data_path=data_path,
                                  eval_path=eval_path,
                                  loc_path=loc_path)
    query_path = os.path.join(data_path, 'query', exp_name)
    tracker.run(query_path, max_frames=np.inf)
    #tracker.run(query_path, max_frames=9)
    tracker.save_poses()
    print('Done')
