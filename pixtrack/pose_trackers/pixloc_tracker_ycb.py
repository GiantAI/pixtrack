import os
import numpy as np
import torch
from pathlib import Path

from pixloc.utils.colmap import Camera as ColCamera
from pixloc.pixlib.geometry import Camera as PixCamera, Pose
from pixloc.utils.data import Paths

from pixtrack.utils.pose_utils import (
    geodesic_distance_for_rotations,
    get_camera_in_world_from_pixpose,
)
from pixtrack.pose_trackers.base_pose_tracker import PoseTracker
from pixtrack.localization.pixloc_pose_refiners import PoseTrackerLocalizer
from pixtrack.localization.tracker import DebugTracker
from pixtrack.utils.io import ImageIterator, YCBVideoIterator
from pixtrack.utils.hloc_utils import extract_covisibility
from pixtrack.utils.ingp_utils import (
    load_nerf2sfm,
    initialize_ingp,
    sfm_to_nerf_pose,
    get_nerf_aabb_from_sfm,
)
from pixtrack.visualization.run_vis_on_poses import get_nerf_image
from scipy.spatial.transform import Rotation as R

import pycolmap
from pycolmap import infer_camera_from_image

import cv2
import pickle as pkl
import argparse
import tqdm


class PixLocPoseTrackerYCB(PoseTracker):
    def __init__(self, data_path, loc_path, eval_path, object_path, debug=False):
        default_paths = Paths(
            query_images="query/",
            reference_images=loc_path,
            reference_sfm="aug_sfm",
            query_list="*_with_intrinsics.txt",
            global_descriptors="features.h5",
            retrieval_pairs="pairs_query.txt",
            results="pixloc_object.txt",
        )
        pixloc_conf = {
            "experiment": "pixloc_megadepth",
            "features": {},
            "optimizer": {
                "num_iters": 150,
                "pad": 1,
            },
            "refinement": {
                "num_dbs": 1,
                "multiscale": [
                    1,
                ],
                "point_selection": "all",
                "normalize_descriptors": True,
                "average_observations": False,
                "do_pose_approximation": False,
            },
        }

        self.object_path = object_path
        self.debug = debug
        self.ycb_root = Path("/data/ycb/")
        paths = default_paths.add_prefixes(
            Path(data_path), Path(loc_path), Path(eval_path)
        )
        self.localizer = PoseTrackerLocalizer(paths, pixloc_conf)
        self.eval_path = eval_path
        covis_path = Path(paths.reference_sfm) / "covis.pkl"
        if os.path.isfile(covis_path):
            self.covis = pkl.load(open(covis_path, "rb"))
        else:
            self.covis = extract_covisibility(paths.reference_sfm)
            with open(str(covis_path), "wb") as f:
                pkl.dump(self.covis, f)
        self.pose_history = {}
        self.pose_tracker_history = {}
        self.cold_start = True
        self.pose = None
        self.reference_ids = None
        nerf_path = Path(object_path) / "pixtrack/instant-ngp/snapshots/weights.msgpack"
        nerf2sfm_path = Path(data_path) / "nerf2sfm.pkl"
        self.reference_scale = 0.3
        self.localizer.refiner.reference_scale = self.reference_scale
        self.nerf2sfm = load_nerf2sfm(str(nerf2sfm_path))
        aabb = get_nerf_aabb_from_sfm(os.path.join(loc_path, "aug_sfm"), nerf2sfm_path)
        self.testbed = initialize_ingp(str(nerf_path), aabb)
        self.testbed.compute_and_save_marching_cubes_mesh("test.obj", [128, 128, 128])
        self.dynamic_id = None
        self.hits = 0
        self.misses = 0
        self.cache_hit = False
        self.relocalization_count = 0

    def relocalize(self, query_path):
        gt_pose = self.gt_pose
        gt_camera = self.gt_camera
        if self.cold_start:
            self.camera = gt_camera  # self.get_query_camera(query_path)
            # self.camera = self.get_query_camera(query_path)
            self.cold_start = False
        # ref_img = self.localizer.model3d.dbs[self.reference_ids[0]]
        # rotation = ref_img.qvec2rotmat()
        # translation = ref_img.tvec
        # pose_init = Pose.from_Rt(rotation, translation)
        self.pose = gt_pose
        self.set_reference_ids()
        self.relocalization_count += 1
        return

    def set_reference_ids(self):
        curr_pose = self.pose
        R_qry = curr_pose.numpy()[0]
        gdists = {}
        for ref in tqdm.tqdm(self.covis):
            cimg = self.localizer.model3d.dbs[ref]
            R_ref = cimg.qvec2rotmat()
            gdist = geodesic_distance_for_rotations(R_qry, R_ref)
            gdists[ref] = gdist

        reference_ids = sorted(gdists, key=lambda x: gdists[x])
        K = 1
        self.reference_ids = reference_ids[:K]
        return self.reference_ids

    def get_query_camera(self, query):
        camera = infer_camera_from_image(query)
        camera = ColCamera(
            None,
            camera.model_name,
            int(camera.width),
            int(camera.height),
            camera.params,
        )
        camera = PixCamera.from_colmap(camera)
        return camera

    def update_reference_ids(self):
        if self.cache_hit == True:
            return self.reference_ids
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
        K = 1
        self.reference_ids = reference_ids[:K]
        return self.reference_ids

    def get_reference_image(self, pose):
        cIw = get_camera_in_world_from_pixpose(pose)
        nerf_pose = sfm_to_nerf_pose(self.nerf2sfm, cIw)
        ref_camera = self.localizer.model3d.cameras[1]
        ref_camera = PixCamera.from_colmap(ref_camera)
        # ref_camera = self.ref_camera
        ref_camera = ref_camera.scale(self.reference_scale)
        nerf_img = get_nerf_image(self.testbed, nerf_pose, ref_camera)
        return nerf_img

    def get_mask(self, pose):
        cIw = get_camera_in_world_from_pixpose(pose)
        nerf_pose = sfm_to_nerf_pose(self.nerf2sfm, cIw)
        depth = get_nerf_image(self.testbed, nerf_pose, self.camera, depth=True)
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode((depth != 0).astype(np.uint8), kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=5)
        return img_dilation

    def create_dynamic_reference_image(self, pose):
        nerf_img = self.get_reference_image(pose)
        dynamic_id = hash(str(pose.numpy()[0]))
        features = self.localizer.refiner.extract_reference_features(
            self.reference_ids, pose, nerf_img
        )
        return dynamic_id, features

    def get_dynamic_id(self, pose):
        features_dicts = self.localizer.refiner.features_dicts
        if self.dynamic_id is None:
            self.dynamic_id, features = self.create_dynamic_reference_image(self.pose)
            features_dicts[self.dynamic_id] = {}
            features_dicts[self.dynamic_id]["pose"] = self.pose
            features_dicts[self.dynamic_id]["features"] = features
            return self.dynamic_id

        self.THRESH = 0  # 0.1
        curr_pose = self.pose
        R_qry = curr_pose.numpy()[0]
        dynamic_pose = features_dicts[self.dynamic_id]["pose"]
        R_drf = dynamic_pose.numpy()[0]
        curr_gdist = geodesic_distance_for_rotations(R_qry, R_drf)
        gdists = {self.dynamic_id: curr_gdist}
        if curr_gdist > self.THRESH:
            for did in features_dicts:
                dpose = features_dicts[did]["pose"]
                R_drf = dpose.numpy()[0]
                gdist = geodesic_distance_for_rotations(R_qry, R_drf)
                gdists[did] = gdist

        dids = sorted(gdists, key=lambda x: gdists[x])
        gdist_min = gdists[dids[0]]
        if gdist_min < self.THRESH:
            self.dynamic_id = dids[0]
            self.reference_ids = features_dicts[self.dynamic_id]["ref_ids"]
            self.hits += 1
            self.cache_hit = True
        else:
            # print('New reference frame! Distance: %f, Threshold: %f' % (gdists[dids[0]], self.THRESH))
            # print(gdists)
            # print(self.localizer.refiner.features_dicts.keys())
            self.cache_hit = False
            self.dynamic_id, features = self.create_dynamic_reference_image(self.pose)
            features_dicts[self.dynamic_id] = {}
            features_dicts[self.dynamic_id]["pose"] = self.pose
            features_dicts[self.dynamic_id]["features"] = features
            features_dicts[self.dynamic_id]["ref_ids"] = self.update_reference_ids()
            self.misses += 1
            self.cache_hit = True

        return self.dynamic_id

    def refine(self, query):
        query_path, query_image, gt_pose, gt_camera = query
        self.gt_pose = gt_pose
        self.gt_camera = gt_camera
        if self.cold_start:
            self.relocalize(query_path)
            self.cold_start = False

        mask = self.get_mask(self.pose)
        query_image = query_image * mask

        self.dynamic_id = self.get_dynamic_id(self.pose)
        translation = self.pose.numpy()[1]
        rotation = self.pose.numpy()[0]
        rot = R.from_matrix(rotation)
        rotation = rot.as_matrix()
        trackers = {}
        rets = {}
        costs = {}
        for ref_id in self.reference_ids:
            ref_img = self.localizer.model3d.dbs[ref_id]
            pose_init = Pose.from_Rt(rotation, translation)
            tracker = DebugTracker(self.localizer.refiner, self.debug)
            ret = self.localizer.run_query(
                query_path,
                self.camera,
                pose_init,
                [ref_id],
                image_query=query_image,
                pose=self.pose,
                reference_images_raw=None,
                dynamic_id=self.dynamic_id,
            )
            rets[ref_id] = ret
            trackers[ref_id] = tracker
            avg_cost = np.mean([x[-1] for x in tracker.costs])
            costs[ref_id] = avg_cost
        best_ref_id = min(costs, key=costs.get)
        ret = rets[best_ref_id]
        self.calculate_error()
        ret["camera"] = self.camera
        ret["reference_ids"] = self.reference_ids
        ret["query_path"] = query_path
        ret["gt_pose"] = gt_pose
        success = ret["success"] and self.t_err < 10 and self.r_err < 10
        if success:
            self.pose = ret["T_refined"]
            ret["success"] = True
        else:
            ret["success"] = False

        img_name = os.path.basename(query_path)
        self.pose_history[img_name] = ret
        self.pose_tracker_history[img_name] = trackers[best_ref_id]
        return success

    def calculate_error(self):
        gt_R, gt_T = self.gt_pose.numpy()
        pr_R, pr_T = self.pose.numpy()
        self.t_err = np.linalg.norm(gt_T - pr_T) * 100.0
        self.r_err = geodesic_distance_for_rotations(gt_R, pr_R) * 180 / np.pi
        message = f"Translation error: {self.t_err:.2f}cm, Rotation error: {self.r_err:.2f} degrees, relocalizations: {self.relocalization_count}"
        self.pbar.set_description(message)

    def get_query_frame_iterator(self, path, max_frames):
        iterator = YCBVideoIterator(object_path=self.object_path, expression=path)
        return iterator

    def save_poses(self):
        path = os.path.join(self.eval_path, "poses.pkl")
        with open(path, "wb") as f:
            pkl.dump(self.pose_history, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path", type=Path)
    parser.add_argument("--query", default="7:10")
    parser.add_argument("--out_dir", default="ycb_7")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    # obj_path = Path(os.environ["OBJECT_PATH"])
    obj_path = args.object_path
    data_path = obj_path / "pixtrack/pixsfm/dataset"
    eval_path = Path(args.out_dir)
    loc_path = obj_path / "pixtrack/aug_nerf_sfm"
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)
    tracker = PixLocPoseTrackerYCB(
        data_path=str(data_path),
        eval_path=str(eval_path),
        loc_path=str(loc_path),
        object_path=obj_path,
        debug=args.debug,
    )
    tracker.run(args.query, max_frames=args.frames)
    print("Relocalization count: ", tracker.relocalization_count)
    tracker.save_poses()
    print("Cache hits: %d, misses: %d" % (tracker.hits, tracker.misses))
    tracker_path = os.path.join(tracker.eval_path, "trackers.pkl")
    with open(tracker_path, "wb") as f:
        pkl.dump(tracker.pose_tracker_history, f)

    print("Done")
