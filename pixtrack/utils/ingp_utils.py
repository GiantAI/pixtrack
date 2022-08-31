import sys
import os
sys.path.append(os.path.join(os.environ['PROJECT_ROOT'], 'instant-ngp/scripts'))
import commentjson as json
from common import *
from scenes import scenes_nerf
import pyngp as ngp
import pickle as pkl
import ast


def load_nerf2sfm(path):
    with open(path, 'rb') as f:
        nerf2sfm = pkl.load(f)
    return nerf2sfm


def initialize_ingp(snapshot_path, 
        aabb=ast.literal_eval(os.environ['OBJ_AABB']), background=None):
    if background is None:
        background = [0., 0., 0., 0.]
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


