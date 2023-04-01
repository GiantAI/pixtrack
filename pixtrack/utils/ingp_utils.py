import sys
import os
sys.path.append(os.path.join(os.environ['PROJECT_ROOT'], 'instant-ngp/scripts'))
import commentjson as json
from common import *
from scenes import scenes_nerf
import pyngp as ngp
import pickle as pkl
import ast
import pycolmap
from pixtrack.utils.pose_utils import get_camera_in_world_from_colmap_image
from sklearn.cluster import DBSCAN

def load_nerf2sfm(path):
    with open(path, 'rb') as f:
        nerf2sfm = pkl.load(f)
    return nerf2sfm


def initialize_ingp(snapshot_path, 
        aabb, background=None):
    if background is None:
        background = [255, 255, 255, 0.]
    mode = ngp.TestbedMode.Nerf
    configs_dir = os.path.join(ROOT_DIR, 'configs', 'nerf')
    scenes = scenes_nerf
    base_network = os.path.join(configs_dir, 'base.json')
    network = base_network
    network_stem = os.path.splitext(os.path.basename(network))[0]
    testbed = ngp.Testbed(mode)
    testbed.nerf.sharpen = 0.
    testbed.load_snapshot(snapshot_path)
    testbed.nerf.render_with_camera_distortion = True
    testbed.background_color = background
    testbed.snap_to_pixel_centers = True
    testbed.nerf.rendering_min_transmittance = 1e-7
    #testbed.nerf.rendering_min_alpha = 1e-4 * 10
    testbed.fov_axis = 0
    testbed.shall_train = False
    testbed.render_aabb.min = aabb[0]
    testbed.render_aabb.max = aabb[1]
    testbed.exposure = 0.
    return testbed

def sfm_to_nerf_pose(nerf2sfm, sfm_pose):
    rotate_over_x = np.array([
            [1.,  0.,  0., 0.],
            [0., -1.,  0., 0.],
            [0.,  0., -1., 0.],
            [0.,  0.,  0., 1.],
        ])
    p1 = sfm_pose @ rotate_over_x
    p1 = p1[[1,0,2,3],:]
    p1[2, :] *= -1
    p1[0:3, 3] -= nerf2sfm['centroid']
    p1[0:3, 3] *= 3. / nerf2sfm['avglen']
    p1 = nerf2sfm['R'] @ p1
    p1[0:3, 3] -= nerf2sfm['totp']
    return p1

def nerf_to_sfm_pose(nerf2sfm, nerf_pose):
    rotate_over_x = np.array([
            [1.,  0.,  0., 0.],
            [0., -1.,  0., 0.],
            [0.,  0., -1., 0.],
            [0.,  0.,  0., 1.],
        ])
    p2 = nerf_pose.copy()
    p2[0:3, 3] += nerf2sfm['totp']
    p2 = np.linalg.inv(nerf2sfm['R']) @ p2
    p2[0:3, 3] /= 3. / nerf2sfm['avglen']
    p2[0:3, 3] += nerf2sfm['centroid']
    p2[2, :] *= -1
    p2 = p2[[1,0,2,3],:]
    p2 = p2 @ rotate_over_x
    return p2


def get_nerf_aabb_from_sfm(sfm_path, nerf2sfm_path):
    nerf2sfm = load_nerf2sfm(nerf2sfm_path)
    model = pycolmap.Reconstruction(sfm_path)
    points3d_in_nerf = []
    for point in model.points3D:
        pose_placeholder = np.eye(4)
        pose_placeholder[:3, -1] = model.points3D[point].xyz
        pose_placeholder_nerf = sfm_to_nerf_pose(nerf2sfm=nerf2sfm, sfm_pose=pose_placeholder)[:3, -1]
        points3d_in_nerf.append(pose_placeholder_nerf)
    points3d_in_nerf = np.array(points3d_in_nerf)
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(points3d_in_nerf)
    min_pts = np.min(points3d_in_nerf[clustering.core_sample_indices_], axis=0)
    max_pts = np.max(points3d_in_nerf[clustering.core_sample_indices_], axis=0)

    min_xyz = points3d_in_nerf.min(axis=0)
    max_xyz = points3d_in_nerf.max(axis=0)
    min_bounds = min_xyz / 3. + np.array([0.5, 0.5, 0.5])
    max_bounds = max_xyz / 3. + np.array([0.5, 0.5, 0.5])
    min_bounds = [min_bounds[1], min_bounds[2], min_bounds[0]]
    max_bounds = [max_bounds[1], max_bounds[2], max_bounds[0]]
    aabb_bounds = [min_bounds, max_bounds]
    return aabb_bounds

def get_object_center_from_sfm(sfm_path):
    model = pycolmap.Reconstruction(sfm_path)
    xyzs = np.array([p3D.xyz for _, p3D in model.points3D.items()])
    center = np.mean(xyzs, axis=0)
    return center
