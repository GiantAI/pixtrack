from pixloc.localization.base_refiner import BaseRefiner
from pixloc.localization.localizer import Localizer
from pixloc.utils.io import parse_image_lists, parse_retrieval, load_hdf5
import logging
logger = logging.getLogger(__name__)

class PoseTrackerLocalizer(Localizer):
    def __init__(self, paths: Paths, conf: Union[DictConfig, Dict],
                 device: Optional[torch.device] = None,
                 T_init: Pose = None, reference_image: str = ''):
        super().__init__(paths, conf, device)

        if paths.global_descriptors is not None:
            global_descriptors = load_hdf5(paths.global_descriptors)
        else:
            global_descriptors = None

        self.refiner = PoseTrackerRefiner(
            self.device, self.optimizer, self.model3d, self.extractor, paths,
            self.conf.refinement, global_descriptors=global_descriptors,
            T_init=T_init)
        self.logs = None
        self.refiner.dbids = [self.model3d.name2id[reference_image]]

    def run_query(self, name: str, camera: Camera):
        dbs = self.refiner.dbids
        loc = None if self.logs is None else self.logs[name]
        ret = self.refiner.refine(name, camera, dbs, loc=loc)
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
        super().__init__(*args, **kwargs)

    def refine(self, qname: str, qcamera: Camera, dbids: List[int],
               loc: Optional[Dict] = None, T_init = None) -> Dict:

        fail = {'success': False, 'T_init': T_init, 'dbids': dbids}
        inliers = None

        p3did_to_dbids = self.model3d.get_p3did_to_dbids(
                dbids, loc, inliers, self.conf.point_selection,
                self.conf.min_track_length)

        # Abort if there are not enough 3D points after filtering
        if len(p3did_to_dbids) < self.conf.min_points_opt:
            logger.debug("Not enough valid 3D points to optimize")
            return fail

        ret = self.refine_query_pose(qname, qcamera, T_init, p3did_to_dbids,
                                     self.conf.multiscale)
     
        ret = {**ret, 'dbids': dbids}
        return ret
