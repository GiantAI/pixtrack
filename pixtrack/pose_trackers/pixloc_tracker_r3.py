import os
import numpy as np
import copy
import cv2
from pixtrack.pose_trackers.pixloc_tracker_r2 import PixLocPoseTrackerR2, PixLocPoseTrackerR1
from pixtrack.utils.pose_utils import rotate_image, rotate_pixpose
from pixloc.localization import SimpleTracker
from pixloc.pixlib.geometry import Pose
from pixloc.pixlib.datasets.view import read_image
from scipy.spatial.transform import Rotation as R

class PixLocPoseTrackerR3(PixLocPoseTrackerR1):
    def refine(self, query):
        if self.cold_start:
            self.relocalize(query)
            self.tracked_roll = 0.
            return True
        
        image_query = self.pre_opt_rotation(query)

        reference_ids = self.update_reference_ids()
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
                                image_query)
            rets[ref_id] = ret
            trackers[ref_id] = tracker
            avg_cost = np.mean([x[-1] for x in tracker.costs])
            costs[ref_id] = avg_cost
        best_ref_id = min(costs, key=costs.get)
        ret = rets[best_ref_id]
        success = ret['success']
        if success:
            self.pose = ret['T_refined']
            ret = self.post_opt_rotation(ret, best_ref_id)
            ret['camera'] = self.camera
            ret['reference_ids'] = reference_ids
            ret['query_path'] = query
        img_name = os.path.basename(query)
        self.pose_history[img_name] = ret

        return success

    def pre_opt_rotation(self, query):
        image_query = read_image(query)
        image_query = rotate_image(image_query, -self.tracked_roll)
        return image_query

    def post_opt_rotation(self, ret, ref_id):
        pose_pre = copy.deepcopy(ret['T_refined'].detach())
        pose_post = rotate_pixpose(ret['T_refined'].detach(), 
                                   rz=self.tracked_roll)
        ret['T_refined'] = pose_post
        ret['tracked_roll'] = copy.deepcopy(self.tracked_roll)

        rot_pre = pose_pre.numpy()[0].T
        roll_pre, _, _ = R.from_matrix(rot_pre).as_euler('zyx', degrees=True)

        rot_post = pose_post.numpy()[0].T
        roll_post, _, _ = R.from_matrix(rot_post).as_euler('zyx', degrees=True)

        ref_img = self.localizer.model3d.dbs[ref_id]
        rot_ref = ref_img.qvec2rotmat().T
        roll_ref, _, _ = R.from_matrix(rot_ref).as_euler('zyx', degrees=True)

        THRESH = 0.
        if abs(roll_pre - roll_ref) > THRESH:
            self.tracked_roll = roll_post - roll_ref
        return ret

if __name__ == '__main__':
    exp_name = 'IMG_4071'
    data_path = '/home/prajwal.chidananda/code/pixtrack/outputs/nerf_sfm/gimble_04MAR2022'
    eval_path = '/home/prajwal.chidananda/code/pixtrack/outputs/%s' % exp_name
    loc_path =  '/home/prajwal.chidananda/code/pixtrack/outputs/nerf_sfm/gimble_04MAR2022'
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)
    tracker = PixLocPoseTrackerR3(data_path=data_path,
                                  eval_path=eval_path,
                                  loc_path=loc_path)
    query_path = os.path.join(data_path, 'query', exp_name)
    #tracker.run(query_path, max_frames=200)
    tracker.run(query_path, max_frames=np.inf)
    tracker.save_poses()
    print('Done')
