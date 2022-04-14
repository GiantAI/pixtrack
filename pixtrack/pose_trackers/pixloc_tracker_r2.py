import os
import numpy as np
from pixtrack.pose_trackers.pixloc_tracker_r1 import PixLocPoseTrackerR1
from pixtrack.utils.pose_utils import geodesic_distance_for_rotations
from pixloc.localization import SimpleTracker
from pixloc.pixlib.geometry import Pose

class PixLocPoseTrackerR2(PixLocPoseTrackerR1):
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

    def refine(self, query):
        if self.cold_start:
            self.relocalize(query)
            return True
        
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
                                [ref_id])
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
    exp_name = 'IMG_4065'
    data_path = '/home/prajwal.chidananda/code/pixtrack/outputs/nerf_sfm/gimble_04MAR2022'
    eval_path = '/home/prajwal.chidananda/code/pixtrack/outputs/%s' % exp_name
    loc_path =  '/home/prajwal.chidananda/code/pixtrack/outputs/nerf_sfm/gimble_04MAR2022'
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)
    tracker = PixLocPoseTrackerR2(data_path=data_path,
                                  eval_path=eval_path,
                                  loc_path=loc_path)
    query_path = os.path.join(data_path, 'query', exp_name)
    tracker.run(query_path, max_frames=np.inf)
    tracker.save_poses()
    print('Done')
