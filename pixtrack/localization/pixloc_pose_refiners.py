import torch
from typing import Optional, Dict, Tuple, Union, List
from omegaconf import DictConfig, OmegaConf as oc
from pathlib import Path
import pixloc
from pixloc.utils.data import Paths
from pixloc.localization.base_refiner import BaseRefiner
from pixloc.localization.localizer import Localizer
from pixloc.utils.io import parse_image_lists, parse_retrieval, load_hdf5
from pixloc.pixlib.geometry import Camera, Pose
from pixloc.pixlib.datasets.view import read_image
import logging
import numpy as np
import copy
import cv2
logger = logging.getLogger(__name__)

class PoseTrackerLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
            device: Optional[torch.device] = None):
        super().__init__(paths, conf, device)

        if paths.global_descriptors is not None:
            global_descriptors = load_hdf5(paths.global_descriptors)
        else:
            global_descriptors = None

        self.refiner = PoseTrackerRefiner(
            self.device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement, global_descriptors=global_descriptors)
        self.logs = None

    def run_query(self, name: str, camera: Camera, pose_init: Pose, reference_images: int, image_query: np.ndarray = None, pose: Pose = None, reference_images_raw: List[np.ndarray] = None):
        loc = None if self.logs is None else self.logs[name]
        ret = self.refiner.refine(name, camera, pose_init, reference_images, loc=loc, image_query=image_query, pose=pose, reference_images=reference_images_raw)
        return ret

class PoseTrackerRefiner(BaseRefiner):
    default_config = dict(
        multiscale=None,
        filter_covisibility=False,
        do_pose_approximation=False,
        do_inlier_ranking=False,
    )

    def __init__(self, *args, **kwargs):
        self.global_descriptors = kwargs.pop('global_descriptors', None)
        self.reference_scale = 1.
        super().__init__(*args, **kwargs)

    def refine(self, qname: str, qcamera: Camera, pose_init: Pose, 
               dbids: List[int], loc: Optional[Dict] = None, 
               image_query: np.ndarray = None, pose: Pose = None, 
               reference_images: List[np.ndarray] = None) -> Dict:

        fail = {'success': False, 'T_init': pose_init, 'dbids': dbids}
        inliers = None

        p3did_to_dbids = self.model3d.get_p3did_to_dbids(
                dbids, loc, inliers, self.conf.point_selection,
                self.conf.min_track_length)

        # Abort if there are not enough 3D points after filtering
        if len(p3did_to_dbids) < self.conf.min_points_opt:
            logger.debug("Not enough valid 3D points to optimize")
            return fail

        ret = self.refine_query_pose(qname, qcamera, pose_init, p3did_to_dbids,
                                     self.conf.multiscale, image_query, pose, reference_images)
     
        ret = {**ret, 'dbids': dbids}
        return ret

    def refine_query_pose(self, qname: str, qcamera: Camera, T_init: Pose,
                          p3did_to_dbids: Dict[int, List],
                          multiscales: Optional[List[int]] = None, 
                          image_query: np.ndarray = None, 
                          pose: Pose = None, 
                          reference_images: List[np.ndarray] = None) -> Dict:

        dbid_to_p3dids = self.model3d.get_dbid_to_p3dids(p3did_to_dbids)
        if multiscales is None:
            multiscales = [1]

        rnames = [self.model3d.dbs[i].name for i in dbid_to_p3dids.keys()]

        if reference_images is not None:
            images_ref = reference_images
        else:
            images_ref = [read_image(self.paths.reference_images / n)
                  for n in rnames] 
            scale = self.reference_scale
            ref_dims = [(int(rimg.shape[1] * scale), 
                         int(rimg.shape[0] * scale)) for rimg in images_ref]
            images_ref = [cv2.resize(images_ref[x], ref_dims[x], interpolation=cv2.INTER_AREA) \
                          for x in range(len(images_ref))]
            print('Reading reference images from disk!!')

        image_orig = image_query
        for image_scale in multiscales:
            # Compute the reference observations
            # TODO: can we compute this offline before hand?
            dbid_p3did_to_feats = dict()
            for idx, dbid in enumerate(dbid_to_p3dids):
                p3dids = dbid_to_p3dids[dbid]

                features_ref_dense, scales_ref = self.dense_feature_extraction(
                        images_ref[idx], rnames[idx], image_scale)
                dbid_p3did_to_feats[dbid] = self.interp_sparse_observations(
                        features_ref_dense, scales_ref, dbid, p3dids, pose)
                del features_ref_dense

            p3did_to_feat = self.aggregate_features(
                    p3did_to_dbids, dbid_p3did_to_feats)
            if self.conf.average_observations:
                p3dids = list(p3did_to_feat.keys())
                p3did_to_feat = [p3did_to_feat[p3did] for p3did in p3dids]
            else:  # duplicate the observations
                p3dids, p3did_to_feat = list(zip(*[
                    (p3did, feat) for p3did, feats in p3did_to_feat.items()
                    for feat in zip(*feats)]))

            # Compute dense query feature maps
            if image_query is None:
                image_query = read_image(self.paths.query_images / qname)
                image_orig = copy.deepcopy(image_query)
            else:
                image_query = copy.deepcopy(image_orig)
            features_query, scales_query = self.dense_feature_extraction(
                        image_query, qname, image_scale)

            ret = self.refine_pose_using_features(features_query, scales_query,
                                                  qcamera, T_init,
                                                  p3did_to_feat, p3dids)
            if not ret['success']:
                logger.info(f"Optimization failed for query {qname}")
                break
            else:
                T_init = ret['T_refined']
        return ret

    def interp_sparse_observations(self,
                                   feature_maps: List[torch.Tensor],
                                   feature_scales: List[float],
                                   image_id: float,
                                   p3dids: List[int],
                                   pose: Pose = None) -> Dict[int, torch.Tensor]:
        image = self.model3d.dbs[image_id]
        camera = Camera.from_colmap(self.model3d.cameras[image.camera_id])
        camera = camera.scale(self.reference_scale)
        T_w2cam = Pose.from_colmap(image)
        if pose is not None:
            T_w2cam = copy.deepcopy(pose)
        p3d = np.array([self.model3d.points3D[p3did].xyz for p3did in p3dids])
        p3d_cam = T_w2cam * p3d

        # interpolate sparse descriptors and store
        feature_obs = []
        masks = []
        for i, (feats, sc) in enumerate(zip(feature_maps, feature_scales)):
            p2d_feat, valid = camera.scale(sc).world2image(p3d_cam)
            opt = self.optimizer
            opt = opt[len(opt)-i-1] if isinstance(opt, (tuple, list)) else opt
            obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
            assert not obs.requires_grad
            feature_obs.append(obs)
            masks.append(mask & valid.to(mask))

        mask = torch.all(torch.stack(masks, dim=0), dim=0)

        # We can't stack features because they have different # of channels
        feature_obs = [[feature_obs[i][j] for i in range(len(feature_maps))]
                       for j in range(len(p3dids))]  # N x K x D

        feature_dict = {p3id: feature_obs[i]
                        for i, p3id in enumerate(p3dids) if mask[i]}

        return feature_dict
