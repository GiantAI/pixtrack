import ast
import os
import pickle as pkl
import numpy as np
import argparse
from pixtrack.utils.pose_utils import get_world_in_camera_from_pixpose, get_camera_in_world_from_pixpose, rotate_image, geodesic_distance_for_rotations
from pixtrack.utils.ingp_utils import load_nerf2sfm, initialize_ingp, sfm_to_nerf_pose, nerf_to_sfm_pose

import pycolmap
import cv2
import tqdm
import math
from pathlib import Path


def get_nerf_image(testbed, nerf_pose, camera, depth=False):
    spp = 8
    width, height = camera.size
    width = int(width)
    height = int(height)
    fl_x = float(camera.f[0])
    fl_y = float(camera.f[1])
    angle_x = math.atan(width / (fl_x * 2)) * 2
    angle_y = math.atan(height / (fl_y * 2)) * 2

    testbed.fov = angle_x * 180 / np.pi
    #testbed.fovx = angle_x * 180 / np.pi
    #testbed.fovy = angle_y * 180 / np.pi
    #testbed.fov_xy = np.array((angle_y * 180 / np.pi, angle_x * 180 / np.pi))


    testbed.set_nerf_camera_matrix(nerf_pose[:3, :])

    #ncx = camera.c[0] / float(width)
    #ncy = camera.c[1] / float(height)
    #testbed.screen_center = np.array([ncy, ncx])

    if depth:
        testbed.render_mode = testbed.render_mode.Depth
    nerf_img = testbed.render(width, height, spp, True)
    nerf_img = nerf_img[:, :, :3] * 255.
    nerf_img = nerf_img.astype(np.uint8)
    if depth:
        testbed.render_mode = testbed.render_mode.Shade
    return nerf_img


def get_query_image(path):
    assert os.path.isfile(path)
    img = cv2.imread(str(path), -1)
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


def add_pose_axes(
    image, camera, pose, axes_center=[0.1179, 1.1538, 1.3870, 0.],
):
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

def draw_points(image, pts_3d, K=np.eye(3), t=15, color=(255, 255, 255)):
    pts_2d = project_3d_to_2d(pts_3d, K).astype(np.int16)
    for pt_num in range(len(pts_2d)):
        image = cv2.circle(image, pts_2d[pt_num], radius=0, color=color, thickness=t)
    return image

def draw_points_with_lines(image, pts_3d, K=np.eye(3), t=15, color=(0, 50, 200)):
    pts_2d = project_3d_to_2d(pts_3d, K).astype(np.int16)
    pt_num = 0
    while(pt_num < len(pts_2d)):
        #image = cv2.circle(image, pts_2d[pt_num], radius=3, color=(255, 0, 0), thickness=t)
        image = cv2.line(image, pts_2d[pt_num], pts_2d[pt_num + 1], color=color, thickness=t)
        pt_num+=2
    return image


def add_object_center(image, camera, pose, object_center=[0.33024578, 1.79926808, 1.71986272]):
    width, height = camera.size
    focal = float(camera.f[0])

    u = float(width / 2)
    v = float(height / 2)
    K = [[focal, 0., u],
         [0., focal, v],
         [0., 0., 1.]]
    K = np.array(K)
    object_center = np.array(object_center)
    object_center0 = object_center[np.newaxis, :]
    object_center0 = np.hstack((object_center0, 
                      np.ones((object_center0.shape[0],1))))
    pts_3d = object_center0 @ np.linalg.inv(pose).T[:, :3]
    result_img = draw_points(image, pts_3d, K)
    return result_img


def add_object_bounding_box(image, camera, pose, obj_aabb, nerf2sfm):
    width, height = camera.size
    focal = float(camera.f[0])
    
    u = float(width / 2)
    v = float(height / 2)
    K = [[focal, 0., u],
         [0., focal, v],
         [0., 0., 1.]]
    K = np.array(K)
    MIN = 0
    MAX = 1
    nerf_aabb = np.array(obj_aabb).copy()
    #print(obj_aabb, "tjos")
    scale = 4.5
    bound_diffs = scale*(nerf_aabb[1] - nerf_aabb[0])
    bound_diffs[0], bound_diffs[2] = bound_diffs[2], bound_diffs[0]

    obj_aabb[0] = np.array([0.33024578, 1.79926808, 1.71986272]) - bound_diffs/2
    obj_aabb[1] = np.array([0.33024578, 1.79926808, 1.71986272]) + bound_diffs/2
    #print(obj_aabb)
    #delta = 0.5
    #obj_aabb[0] = np.array([0.33024578, 1.79926808, 1.71986272]) - delta
    #obj_aabb[1] = np.array([0.33024578, 1.79926808, 1.71986272]) + delta
    aabb_pts = [
        [obj_aabb[MIN][0], obj_aabb[MIN][1], obj_aabb[MIN][2]], # Xmin, ymin, zmin
        [obj_aabb[MAX][0], obj_aabb[MAX][1], obj_aabb[MAX][2]], # Xmax, ymin, zmin
    ]
    aabb_pts = [
        # Front square
        [obj_aabb[MIN][0], obj_aabb[MIN][1], obj_aabb[MIN][2]], # Xmin, ymin, zmin
        [obj_aabb[MAX][0], obj_aabb[MIN][1], obj_aabb[MIN][2]], # Xmax, ymin, zmin
        [obj_aabb[MAX][0], obj_aabb[MAX][1], obj_aabb[MIN][2]],
        [obj_aabb[MAX][0], obj_aabb[MIN][1], obj_aabb[MIN][2]],
        [obj_aabb[MIN][0], obj_aabb[MAX][1], obj_aabb[MIN][2]],
        [obj_aabb[MIN][0], obj_aabb[MIN][1], obj_aabb[MIN][2]],
        [obj_aabb[MIN][0], obj_aabb[MAX][1], obj_aabb[MIN][2]],
        [obj_aabb[MAX][0], obj_aabb[MAX][1], obj_aabb[MIN][2]],
        # Back square
        [obj_aabb[MIN][0], obj_aabb[MIN][1], obj_aabb[MAX][2]], # Xmin, ymin, zmin
        [obj_aabb[MAX][0], obj_aabb[MIN][1], obj_aabb[MAX][2]], # Xmax, ymin, zmin
        [obj_aabb[MAX][0], obj_aabb[MAX][1], obj_aabb[MAX][2]],
        [obj_aabb[MAX][0], obj_aabb[MIN][1], obj_aabb[MAX][2]],
        [obj_aabb[MIN][0], obj_aabb[MAX][1], obj_aabb[MAX][2]],
        [obj_aabb[MIN][0], obj_aabb[MIN][1], obj_aabb[MAX][2]],
        [obj_aabb[MIN][0], obj_aabb[MAX][1], obj_aabb[MAX][2]],
        [obj_aabb[MAX][0], obj_aabb[MAX][1], obj_aabb[MAX][2]],

        [obj_aabb[MIN][0], obj_aabb[MIN][1], obj_aabb[MIN][2]],
        [obj_aabb[MIN][0], obj_aabb[MIN][1], obj_aabb[MAX][2]],
        [obj_aabb[MIN][0], obj_aabb[MAX][1], obj_aabb[MIN][2]],
        [obj_aabb[MIN][0], obj_aabb[MAX][1], obj_aabb[MAX][2]],
        
        [obj_aabb[MAX][0], obj_aabb[MIN][1], obj_aabb[MIN][2]],
        [obj_aabb[MAX][0], obj_aabb[MIN][1], obj_aabb[MAX][2]],
        [obj_aabb[MAX][0], obj_aabb[MAX][1], obj_aabb[MIN][2]],
        [obj_aabb[MAX][0], obj_aabb[MAX][1], obj_aabb[MAX][2]],
    ]

    transformed_pts = []
    for point in aabb_pts:
        #pt_in_nerf = np.eye(4)
        #pt_in_nerf[:3, -1] = np.array(point)
        #pt_in_sfm = nerf_to_sfm_pose(nerf2sfm, pt_in_nerf)[:3, -1]
        pt_3d = (np.linalg.inv(pose) @ np.append(point, [1.0]))[:3]
        transformed_pts.append(pt_3d)

    result_img = draw_points_with_lines(image, np.array(transformed_pts), K)
    return result_img

    
def blend_images(query_image, nerf_image, alpha=0.5):
    nerf_image = cv2.cvtColor(nerf_image, cv2.COLOR_BGR2RGB)
    blend_img = query_image * (alpha) + nerf_image * (1 - alpha)
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

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255, 0, 0)
    thickness = 2
    org = (5, ref_dim[1] -10)
    name_t = names[0].split('/')[1].split('.')[0]
    name_t = 'Reference image: %s' % name_t
    overlay_img = cv2.putText(base_image, name_t, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    return base_image


def add_normalized_query_image(base_image, path, angle, center=None, s=0.25):
    q_img = cv2.imread(path, -1)
    q_img = rotate_image(q_img, -angle, center)
    base_shape = base_image.shape
    scale = base_shape[1] * s / q_img.shape[1]
    q_dim = (int(q_img.shape[1] * scale),
              int(q_img.shape[0] * scale))
    q_img = cv2.resize(q_img, q_dim, interpolation=cv2.INTER_AREA)
    base_image[-q_dim[1]:, :q_dim[0]] = q_img
    return base_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=Path(os.environ['PIXTRACK_OUTPUTS']) / 'IMG_4117')
    parser.add_argument('--reference_image', default=False) 
    parser.add_argument('--no_axes', action='store_true', default=False) 
    parser.add_argument('--obj_center', action='store_true', default=False) 
    parser.add_argument('--pose_error', action='store_true', default=False)
    args = parser.parse_args()

    PROJECT_ROOT = os.environ['PROJECT_ROOT']
    obj = os.environ['OBJECT']
    obj_path = Path(os.environ['OBJECT_PATH'])
    obj_aabb = os.environ['OBJ_AABB']
    obj_aabb = np.array(ast.literal_eval(obj_aabb)).copy()

    poses_path = Path(args.out_dir) / 'poses.pkl'
    sfm_dir = obj_path / 'pixtrack/aug_nerf_sfm/aug_sfm'
    nerf_path = obj_path / 'pixtrack/instant-ngp/snapshots/weights.msgpack'
    nerf2sfm_path = obj_path / 'pixtrack/pixsfm/dataset/nerf2sfm.pkl'
    sfm_images_dir = obj_path / 'pixtrack/aug_nerf_sfm'

    pose_stream = pkl.load(open(poses_path, 'rb'))
    recon = pycolmap.Reconstruction(sfm_dir)
    nerf2sfm = load_nerf2sfm(nerf2sfm_path)
    testbed = initialize_ingp(str(nerf_path))

    for name_q in tqdm.tqdm(pose_stream):
        path_q = pose_stream[name_q]['query_path']
        ref_ids = pose_stream[name_q]['reference_ids']
        camera = pose_stream[name_q]['camera']

        query_img = get_query_image(path_q)
        if 'T_refined' in pose_stream[name_q]:
            wIc_pix = pose_stream[name_q]['T_refined']
            cIw_sfm = get_camera_in_world_from_pixpose(wIc_pix)
            nerf_pose = sfm_to_nerf_pose(nerf2sfm, cIw_sfm)
            nerf_img = get_nerf_image(testbed, nerf_pose, camera)
        else:
            nerf_img = (np.ones(query_img.shape) * 255).astype(np.uint8)
        p = cIw_sfm.copy()
        result_img = blend_images(query_img, nerf_img)
       
        #result_img = add_object_bounding_box(image=query_img, camera=camera, pose=p, obj_aabb=obj_aabb.copy(), nerf2sfm=nerf2sfm)
        if args.reference_image:
            result_img = add_reference_images(result_img, recon, 
                                            ref_ids, sfm_images_dir)
        if 'tracked_roll' in pose_stream[name_q]:
            tracked_roll = pose_stream[name_q]['tracked_roll']
            tracked_center = pose_stream[name_q]['tracked_center']
            result_img = add_normalized_query_image(result_img, path_q, tracked_roll, tracked_center)
        object_center = ast.literal_eval(os.environ['OBJ_CENTER']) + [0]
        base_result_image = result_img.copy()
        if not args.no_axes:
            result_img = add_pose_axes(result_img, camera, cIw_sfm, object_center)
        if not args.obj_center:
            result_img = add_object_center(result_img, camera, cIw_sfm)

        result_name = 'result_%s' % os.path.basename(path_q)
        pose_axis_dir = os.path.join(args.out_dir, "results")
        if(not os.path.exists(pose_axis_dir)):
            os.mkdir(pose_axis_dir)
        result_path = os.path.join(pose_axis_dir, result_name)
        if not args.no_axes:
            result_img = add_pose_axes(result_img, camera, cIw_sfm, object_center)

        if args.pose_error and 'T_refined' in pose_stream[name_q]:
            pr_R, pr_T = pose_stream[name_q]['T_refined'].numpy()
            gt_R, gt_T = pose_stream[name_q]['gt_pose'].numpy()
            rotation_error = geodesic_distance_for_rotations(gt_R, pr_R) * 180. / np.pi
            translation_error = np.linalg.norm(gt_T - pr_T) * 100.

            rerror_text = f'Rotation error: {rotation_error:.4f} degrees'
            terror_text = f'Translation error: {translation_error:.4f} cm'
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.8
            color = (255, 0, 0)
            thickness = 2
            text_anchor = (20, 30)
            result_img = cv2.putText(result_img, rerror_text, text_anchor, font,
                                     fontScale, color, thickness, cv2.LINE_AA)
            text_anchor = (20, 60)
            result_img = cv2.putText(result_img, terror_text, text_anchor, font,
                                     fontScale, color, thickness, cv2.LINE_AA)

        cv2.imwrite(result_path, result_img)


        '''
        # To get the contour!
        gray = cv2.cvtColor(nerf_img, cv2.COLOR_RGB2GRAY) # convert to grayscale
        blur = cv2.blur(gray, (3, 3)) # blur the image
        ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img = cv2.fillPoly(nerf_img, pts = contours, color =(0, 255, 75))
        mask = np.array((img != np.array([0, 255, 75]))*np.array([0, 255, 75]), dtype=np.uint8)
        
        # Plotting the contours images
        result_img = cv2.drawContours(base_result_image.copy(), contours, -1, (0,255,75), 2)
        result_name = 'result_%s' % os.path.basename(path_q)
        contour_dir = os.path.join(args.out_dir, "contour")
        if(not os.path.exists(contour_dir)):
            os.mkdir(contour_dir)
        result_path = os.path.join(contour_dir, result_name)
        if args.reference_image:
            result_img = add_reference_images(result_img, recon, ref_ids, sfm_images_dir)
        if not args.no_axes:
            result_img = add_pose_axes(result_img, camera, cIw_sfm, object_center)
        if not args.obj_center:
            result_img = add_object_center(result_img, camera, cIw_sfm)
        cv2.imwrite(result_path, result_img)
         
        # Segmentation mask images
        result_name = 'result_%s' % os.path.basename(path_q)
        result_path_segmask = os.path.join(os.path.join(args.out_dir, "segmask"), result_name)
        segmask_dir = os.path.join(args.out_dir, "segmask")
        if(not os.path.exists(segmask_dir)):
            os.mkdir(segmask_dir)
        result_img = blend_images(base_result_image.copy(), mask)
        if args.reference_image:
            result_img = add_reference_images(result_img, recon, ref_ids, sfm_images_dir)
        if not args.no_axes:
            result_img = add_pose_axes(result_img, camera, cIw_sfm)
        if not args.obj_center:
            result_img = add_object_center(result_img, camera, cIw_sfm)
        cv2.imwrite(result_path_segmask, result_img)
        '''
