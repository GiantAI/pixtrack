import torch
import torch.nn.functional as F
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
from pixloc.localization.model3d import Model3D
from pixloc.pixlib.utils.experiments import load_experiment
from pixloc.pixlib.models import get_model

from pixtrack.localization.feature_extractor import PixTrackFeatureExtractor
from pixtrack.optimizers.pixtrack_optimizer import PixTrackOptimizer
import logging
import numpy as np
import copy
import cv2
from collections import defaultdict
import h5py

logger = logging.getLogger(__name__)


class PoseTrackerLocalizer(Localizer):
    def __init__(
        self,
        paths: Paths,
        conf: Union[DictConfig, Dict],
        device: Optional[torch.device] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

        self.model3d = Model3D(paths.reference_sfm)
        cameras = parse_image_lists(paths.query_list, with_intrinsics=True)
        self.queries = {n: c for n, c in cameras}

        # Loading feature extractor and optimizer from experiment or scratch
        conf = oc.create(conf)
        conf_features = conf.features.get("conf", {})
        conf_optim = conf.get("optimizer", {})
        if conf.get("experiment"):
            pipeline = load_experiment(
                conf.experiment, {"extractor": conf_features, "optimizer": conf_optim}
            )
            pipeline = pipeline.to(device)
            logger.debug(
                "Use full pipeline from experiment %s with config:\n%s",
                conf.experiment,
                oc.to_yaml(pipeline.conf),
            )
            extractor = pipeline.extractor
            optimizer = pipeline.optimizer
            if isinstance(optimizer, torch.nn.ModuleList):
                optimizer = list(optimizer)
        else:
            assert "name" in conf.features
            extractor = get_model(conf.features.name)(conf_features)
            optimizer = get_model(conf.optimizer.name)(conf_optim)

        self.paths = paths
        self.conf = conf
        self.device = device
        for opt in optimizer:
            opt.__class__ = PixTrackOptimizer
        self.optimizer = optimizer
        self.optimizer
        self.extractor = PixTrackFeatureExtractor(
            extractor, device, conf.features.get("preprocessing", {})
        )

        if paths.global_descriptors is not None:
            global_descriptors = load_hdf5(paths.global_descriptors)
        else:
            global_descriptors = None

        self.refiner = PoseTrackerRefiner(
            self.device,
            self.optimizer,
            self.model3d,
            self.extractor,
            paths,
            self.conf.refinement,
            global_descriptors=global_descriptors,
        )
        self.logs = None

    def run_query(
        self,
        name: str,
        camera: Camera,
        pose_init: Pose,
        reference_images: int,
        image_query: np.ndarray = None,
        depth_query: np.ndarray = None,
        pose: Pose = None,
        reference_images_raw: List[np.ndarray] = None,
        dynamic_id: int = None,
        occlusion_mask: np.ndarray = None,
    ):
        loc = None if self.logs is None else self.logs[name]
        ret = self.refiner.refine(
            name,
            camera,
            pose_init,
            reference_images,
            loc=loc,
            image_query=image_query,
            depth_query=depth_query,
            pose=pose,
            reference_images=reference_images_raw,
            dynamic_id=dynamic_id,
            occlusion_mask=occlusion_mask
        )
        return ret


class PoseTrackerRefiner(BaseRefiner):
    default_config = dict(
        multiscale=None,
        filter_covisibility=False,
        do_pose_approximation=False,
        do_inlier_ranking=False,
    )

    def __init__(self, *args, **kwargs):
        self.global_descriptors = kwargs.pop("global_descriptors", None)
        self.reference_scale = 1.0
        self.choices = {}
        self.features_dicts = {}
        super().__init__(*args, **kwargs)

    def refine(
        self,
        qname: str,
        qcamera: Camera,
        pose_init: Pose,
        dbids: List[int],
        loc: Optional[Dict] = None,
        image_query: np.ndarray = None,
        depth_query: np.ndarray = None,
        pose: Pose = None,
        reference_images: List[np.ndarray] = None,
        dynamic_id: int = None,
        occlusion_mask: np.ndarray = None
    ) -> Dict:
        fail = {"success": False, "T_init": pose_init, "dbids": dbids}
        inliers = None

        p3did_to_dbids = self.model3d.get_p3did_to_dbids(
            dbids, loc, inliers, self.conf.point_selection, self.conf.min_track_length
        )

        # Abort if there are not enough 3D points after filtering
        if len(p3did_to_dbids) < self.conf.min_points_opt:
            logger.debug("Not enough valid 3D points to optimize")
            return fail

        ret = self.refine_query_pose(
            qname,
            qcamera,
            pose_init,
            p3did_to_dbids,
            self.conf.multiscale,
            image_query,
            depth_query,
            pose,
            reference_images,
            dynamic_id,
            occlusion_mask=occlusion_mask
        )

        ret = {**ret, "dbids": dbids}
        return ret

    def read_features(self, ref_id):
        ref_id = str(ref_id)
        features_path = self.paths.dumps / "reference_features.h5"
        features = {}
        with h5py.File(str(features_path), "r") as f:
            for scale in self.conf.multiscale:
                scale = str(scale)
                features[scale] = {}
                p3dids = f[ref_id][scale]["p3dids"]
                features[scale]["p3dids"] = np.array(p3dids).tolist()
                p3did_to_feats = []
                levels = f[ref_id][scale]
                levels = list(levels.keys())
                levels.remove("p3dids")
                for level in levels:
                    p3did_to_feat = f[ref_id][scale][level]["p3did_to_feat"]
                    p3did_to_feat = torch.tensor(p3did_to_feat).to(self.device)
                    p3did_to_feats.append(p3did_to_feat)
                p3did_to_feat = [
                    tuple([p3did_to_feats[int(level)][p3did] for level in levels])
                    for p3did in range(len(p3dids))
                ]
                features[scale]["p3did_to_feat"] = p3did_to_feat
        return features

    def refine_query_pose(
        self,
        qname: str,
        qcamera: Camera,
        T_init: Pose,
        p3did_to_dbids: Dict[int, List],
        multiscales: Optional[List[int]] = None,
        image_query: np.ndarray = None,
        depth_query: np.ndarray = None,
        pose: Pose = None,
        reference_images: List[np.ndarray] = None,
        dynamic_id: int = None,
        occlusion_mask: np.ndarray= None,
    ) -> Dict:
        dbid_to_p3dids = self.model3d.get_dbid_to_p3dids(p3did_to_dbids)
        ref_ids = list(dbid_to_p3dids.keys())
        assert len(ref_ids) == 1
        ref_id = ref_ids[0]
        if multiscales is None:
            multiscales = [1]

        rnames = [self.model3d.dbs[i].name for i in dbid_to_p3dids.keys()]
        images_ref = reference_images

        for image_scale in multiscales:
            # Compute the reference observations
            # TODO: can we compute this offline before hand?
            if images_ref is not None:
                dbid_p3did_to_feats = dict()
                for idx, dbid in enumerate(dbid_to_p3dids):
                    p3dids = dbid_to_p3dids[dbid]

                    features_ref_dense, scales_ref = self.dense_feature_extraction(
                        images_ref[idx], rnames[idx], image_scale
                    )
                    dbid_p3did_to_feats[dbid] = self.interp_sparse_observations(
                        features_ref_dense, scales_ref, dbid, p3dids, pose
                    )
                    del features_ref_dense

                p3dids = list(dbid_p3did_to_feats[ref_id].keys())
                p3did_to_feat = [
                    tuple(dbid_p3did_to_feats[ref_id][p3did]) for p3did in p3dids
                ]
            else:
                if dynamic_id is None:
                    if ref_id in self.features_dicts:
                        features_dict = self.features_dicts[ref_id]
                    else:
                        features_dict = self.read_features(ref_id)
                    self.features_dicts[ref_id] = features_dict
                else:
                    features_dict = self.features_dicts[dynamic_id]["features"]

                p3dids = features_dict[str(image_scale)]["p3dids"]
                p3did_to_feat = features_dict[str(image_scale)]["p3did_to_feat"]

            features_query, scales_query = self.dense_feature_extraction(
                image_query, qname, image_scale
            )

            #if depth_query is not None:
            #    features_query = self.augment_depth(features_query, depth_query, scales_query)

            ret = self.refine_pose_using_features(
                features_query, scales_query, qcamera, T_init, p3did_to_feat, p3dids, depth_query, occlusion_mask=occlusion_mask
            )
            if not ret["success"]:
                logger.info(f"Optimization failed for query {qname}")
                break
            else:
                T_init = ret["T_refined"]
        return ret

    def refine_pose_using_features(self,
                                   features_query: List[torch.tensor],
                                   scales_query: List[float],
                                   qcamera: Camera,
                                   T_init: Pose,
                                   features_p3d: List[List[torch.Tensor]],
                                   p3dids: List[int],
				   depth_query: torch.tensor, 
                                   occlusion_mask: np.ndarray = None) -> Dict:
        """Perform the pose refinement using given dense query feature-map.
        """
        # decompose descriptors and uncertainities, normalize descriptors
        weights_ref = []
        features_ref = []
        for level in range(len(features_p3d[0])):
            feats = torch.stack([feat[level] for feat in features_p3d], dim=0)
            feats = feats.to(self.device)
            if self.conf.compute_uncertainty:
                feats, weight = feats[:, :-1], feats[:, -1:]
                weights_ref.append(weight)
            if self.conf.normalize_descriptors:
                feats = torch.nn.functional.normalize(feats, dim=1)
            assert not feats.requires_grad
            features_ref.append(feats)

        # query dense features decomposition and normalization
        features_query = [feat.to(self.device) for feat in features_query]
        if self.conf.compute_uncertainty:
            weights_query = [feat[-1:] for feat in features_query]
            features_query = [feat[:-1] for feat in features_query]
        if self.conf.normalize_descriptors:
            features_query = [torch.nn.functional.normalize(feat, dim=0)
                              for feat in features_query]

        p3d = np.stack([self.model3d.points3D[p3did].xyz for p3did in p3dids])

        T_i = T_init
        ret = {'T_init': T_init}
        # We will start with the low res feature map first
        for idx, level in enumerate(reversed(range(len(features_query)))):
            F_q, F_ref = features_query[level], features_ref[level]
            qcamera_feat = qcamera.scale(scales_query[level])

            if self.conf.compute_uncertainty:
                W_ref_query = (weights_ref[level], weights_query[level])
            else:
                W_ref_query = None

            logger.debug(f'Optimizing at level {level}.')
            opt = self.optimizer
            if isinstance(opt, (tuple, list)):
                if self.conf.layer_indices:
                    opt = opt[self.conf.layer_indices[level]]
                else:
                    opt = opt[level]

            depth_query_scaled = depth_query
            _, H, W = F_q.shape
            depth_query_scaled = F.interpolate(depth_query.unsqueeze(0).unsqueeze(0), (H, W), mode='bilinear').squeeze(0).float()
            T_opt, fail = opt.run(p3d, F_ref, F_q, T_i.to(F_q),
                                  qcamera_feat.to(F_q),
                                  W_ref_query=W_ref_query,
                                  D_query=depth_query_scaled.to(F_q))

            self.log_optim(i=idx, T_opt=T_opt, fail=fail, level=level,
                           p3d=p3d, p3d_ids=p3dids,
                           T_init=T_init, camera=qcamera_feat)
            if fail:
                return {**ret, 'success': False}
            T_i = T_opt

        # Compute relative pose w.r.t. initilization
        T_opt = T_opt.cpu().double()
        dR, dt = (T_init.inv() @ T_opt).magnitude()
        return {
            **ret,
            'success': True,
            'T_refined': T_opt,
            'diff_R': dR.item(),
            'diff_t': dt.item(),
        }


    def augment_depth(self, features, depth, scale):
        for level in range(len(features)):
            _, H, W = features[level].shape
            depth_scaled = F.interpolate(depth.unsqueeze(0).unsqueeze(0), (H, W), mode='nearest').squeeze(0)
            features[level] = torch.cat((features[level], depth_scaled), dim=0)
        return features

    def extract_reference_features(self, dbids, pose=None, reference_image=None):
        loc = None
        inliers = None
        p3did_to_dbids = self.model3d.get_p3did_to_dbids(
            dbids, loc, inliers, self.conf.point_selection, self.conf.min_track_length
        )
        dbid_to_p3dids = self.model3d.get_dbid_to_p3dids(p3did_to_dbids)
        multiscales = self.conf.multiscale
        if multiscales is None:
            multiscales = [1]
        rnames = [self.model3d.dbs[i].name for i in dbid_to_p3dids.keys()]
        if reference_image is None:
            images_ref = [
                read_image(self.paths.reference_images / n).astype(np.float32)
                for n in rnames
            ]
            scale = self.reference_scale
            ref_dims = [
                (int(rimg.shape[1] * scale), int(rimg.shape[0] * scale))
                for rimg in images_ref
            ]
            images_ref = [
                cv2.resize(images_ref[x], ref_dims[x], interpolation=cv2.INTER_AREA)
                for x in range(len(images_ref))
            ]
        else:
            images_ref = [reference_image.astype(np.float32)]
        features = {}
        ref_img = self.model3d.dbs[dbids[0]]
        if pose is None:
            pose = Pose.from_Rt(ref_img.qvec2rotmat(), ref_img.tvec)

        for image_scale in multiscales:
            dbid_p3did_to_feats = dict()
            for idx, dbid in enumerate(dbid_to_p3dids):
                p3dids = dbid_to_p3dids[dbid]

                features_ref_dense, scales_ref = self.dense_feature_extraction(
                    images_ref[idx], rnames[idx], image_scale
                )
                dbid_p3did_to_feats[dbid] = self.interp_sparse_observations(
                    features_ref_dense, scales_ref, dbid, p3dids, pose
                )
                del features_ref_dense

            p3dids = list(dbid_p3did_to_feats[dbids[0]].keys())
            p3did_to_feat = [
                tuple(dbid_p3did_to_feats[dbids[0]][p3did]) for p3did in p3dids
            ]
            features[str(image_scale)] = {}
            features[str(image_scale)]["p3dids"] = p3dids
            features[str(image_scale)]["p3did_to_feat"] = p3did_to_feat
        return features

    def interp_sparse_observations(
        self,
        feature_maps: List[torch.Tensor],
        feature_scales: List[float],
        image_id: float,
        p3dids: List[int],
        pose: Pose = None,
    ) -> Dict[int, torch.Tensor]:
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
            opt = opt[len(opt) - i - 1] if isinstance(opt, (tuple, list)) else opt
            obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
            assert not obs.requires_grad
            feature_obs.append(obs)
            masks.append(mask & valid.to(mask))

        mask = torch.all(torch.stack(masks, dim=0), dim=0)

        # We can't stack features because they have different # of channels
        feature_obs = [
            [feature_obs[i][j] for i in range(len(feature_maps))]
            for j in range(len(p3dids))
        ]  # N x K x D

        feature_dict = {
            p3id: feature_obs[i] for i, p3id in enumerate(p3dids) if mask[i]
        }

        return feature_dict

    def aggregate_features(
        self,
        p3did_to_dbids: Dict,
        dbid_p3did_to_feats: Dict,
    ) -> Dict[int, List[torch.Tensor]]:
        """Aggregate descriptors from covisible images through averaging."""
        p3did_to_feat = defaultdict(list)
        for p3id, obs_dbids in p3did_to_dbids.items():
            features = []
            for obs_imgid in obs_dbids:
                if p3id not in dbid_p3did_to_feats[obs_imgid]:
                    continue
                features.append(dbid_p3did_to_feats[obs_imgid][p3id])
            if len(features) > 0:
                # list with one entry per layer, grouping all 3D observations
                for level in range(len(features[0])):
                    observation = [f[level] for f in features]
                    if self.conf.average_observations:
                        observation = torch.stack(observation, 0)
                        if self.conf.compute_uncertainty:
                            feat, w = observation[:, :-1], observation[:, -1:]
                            feat = (feat * w).sum(0) / w.sum(0)
                            observation = torch.cat([feat, w.mean(0)], -1)
                        else:
                            observation = observation.mean(0)
                    p3did_to_feat[p3id].append(observation)
        return dict(p3did_to_feat)
