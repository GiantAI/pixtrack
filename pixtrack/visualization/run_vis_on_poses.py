import os
import pickle as pkl
import numpy as np
import argparse
from pixtrack.utils.pose_utils import get_world_in_camera_from_pixpose, get_camera_in_world_from_pixpose
from pixtrack.utils.ingp_utils import load_nerf2sfm, initialize_ingp, sfm_to_nerf_pose
import pycolmap
import cv2
import tqdm
import math

def get_nerf_image(testbed, nerf_pose, camera):
    spp = 8
    width, height = camera.size
    width = int(width)
    height = int(height)
    fl_x = float(camera.f[0])
    angle_x = math.atan(width / (fl_x * 2)) * 2

    testbed.fov = angle_x * 180 / np.pi
    testbed.set_nerf_camera_matrix(nerf_pose[:3, :])

    nerf_img = testbed.render(width, height, spp, True)
    nerf_img = nerf_img[:, :, :3] * 255.
    nerf_img = nerf_img.astype(np.uint8)
    return nerf_img

def get_query_image(path):
    assert os.path.isfile(path)
    img = cv2.imread(path, -1)
    return img

def project_3d_to_2d(pts_3d, K=np.eye(3)):
    pts_2d = K @ pts_3d.T
    pts_2d = pts_2d / pts_2d[2, :]
    pts_2d = pts_2d[:2, :].T
    return pts_2d

def draw_axes(image, pts_3d, K=np.eye(3), t=10):
    pts_2d = project_3d_to_2d(pts_3d, K).astype(np.int16)
    image = cv2.line(image, pts_2d[0], pts_2d[1], (255, 0, 0), int(t))
    image = cv2.line(image, pts_2d[2], pts_2d[3], (0, 255, 0), int(t))
    image = cv2.line(image, pts_2d[4], pts_2d[5], (0, 0, 255), int(t))
    return image

def add_pose_axes(image, camera, pose, axes_center=[0.1179, 1.1538, 1.3870, 0.]):
    width, height = camera.size
    focal = float(camera.f[0])

    u = float(width / 2)
    v = float(height / 2)
    K = [[focal, 0., u], 
         [0., focal, v], 
         [0., 0., 1.]]
    K = np.array(K)
    x, y, z = 0., 0., 0.
    s = 0.25
    t = 5.
    axes = [[x, y, z], 
            [x + s, y, z],
            [x, y, z], 
            [x, y - s, z],
            [x, y, z], 
            [x, y, z - s]]
    axes = np.array(axes)
    axes = np.hstack((axes, np.ones((axes.shape[0],1))))
    axes += np.array(axes_center)
    pts_3d = axes @ np.linalg.inv(pose).T[:, :3]
    result_img = draw_axes(image, pts_3d, K)
    return result_img
    
def blend_images(query_image, nerf_image):
    #query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    blend_img = query_image * 0.5 + nerf_image * 0.5
    blend_img = blend_img.astype(np.uint8)
    return blend_img

def add_reference_images(base_image, recon, ref_ids, sfm_images_dir, s=0.25):
    names = [recon.images[x].name for x in ref_ids]
    path = os.path.join(sfm_images_dir, names[0])
    ref_img = cv2.imread(path, cv2.IMREAD_COLOR)
    base_shape = base_image.shape
    scale = base_shape[1] * s / ref_img.shape[1]
    ref_dim = (int(ref_img.shape[1] * scale), 
               int(ref_img.shape[0] * scale))
    ref_img = cv2.resize(ref_img, ref_dim, interpolation=cv2.INTER_AREA)
    base_image[:ref_dim[1], :ref_dim[0]] = ref_img
    return base_image

if __name__ == '__main__':
    poses_path = 'outputs/IMG_3813/poses.pkl'
    sfm_dir = '/home/prajwal.chidananda/code/pixloc/outputs/hloc/Gimble/sfm'
    nerf_path = '/home/prajwal.chidananda/code/pixtrack/instant-ngp/snapshots/gimble_04MAR2022/weights.msgpack'
    nerf2sfm_path = '/home/prajwal.chidananda/code/instant-ngp/data/nerf/gimble_04MAR2022/nerf2sfm.pkl'
    sfm_images_dir = '/home/prajwal.chidananda/code/pixloc/datasets/Gimble'
    out_dir = 'outputs/IMG_3813'
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses', default=poses_path)
    parser.add_argument('--nerf_path', default=nerf_path)
    parser.add_argument('--nerf2sfm_path', default=nerf2sfm_path)
    parser.add_argument('--sfm_dir', default=sfm_dir)
    parser.add_argument('--sfm_images_dir', default=sfm_images_dir)
    parser.add_argument('--out_dir', default=out_dir)
    args = parser.parse_args()

    PROJECT_ROOT = os.environ['PROJECT_ROOT']

    poses_path = os.path.join(PROJECT_ROOT, args.poses)
    pose_stream = pkl.load(open(poses_path, 'rb'))
    recon = pycolmap.Reconstruction(args.sfm_dir)
    nerf2sfm = load_nerf2sfm(args.nerf2sfm_path)
    testbed = initialize_ingp(args.nerf_path)

    for name_q in tqdm.tqdm(pose_stream):
        path_q = pose_stream[name_q]['query_path']
        ref_ids = pose_stream[name_q]['reference_ids']
        camera = pose_stream[name_q]['camera']
        wIc_pix = pose_stream[name_q]['T_refined']
        cIw_sfm = get_camera_in_world_from_pixpose(wIc_pix)
        nerf_pose = sfm_to_nerf_pose(nerf2sfm, cIw_sfm)
        
        nerf_img = get_nerf_image(testbed, nerf_pose, camera)
        query_img = get_query_image(path_q)

        result_img = blend_images(query_img, nerf_img)
        result_img = add_reference_images(result_img, recon, 
                                        ref_ids, args.sfm_images_dir)
        result_img = add_pose_axes(result_img, camera, cIw_sfm)

        result_name = 'result_%s' % os.path.basename(path_q)
        result_path = os.path.join(args.out_dir, result_name)
        cv2.imwrite(result_path, result_img)

