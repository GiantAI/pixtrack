import numpy as np
import pycolmap
from pixloc.pixlib.geometry import Pose
from scipy.spatial.transform import Rotation as R
import cv2


def geodesic_distance_for_rotations(R1, R2):
    Rd = R1 @ R2.T
    rot = R.from_matrix(Rd)
    rot_aa = rot.as_rotvec()
    angle = np.linalg.norm(rot_aa)
    return angle


def get_world_in_camera_from_pixpose(pixpose):
    pixpose = pixpose.cpu().numpy()
    wIc = np.eye(4)
    wIc[:3, :3] = pixpose[0]
    wIc[:3, 3] = pixpose[1]
    return wIc


def get_camera_in_world_from_pixpose(pixpose):
    wIc = get_world_in_camera_from_pixpose(pixpose)
    cIw = np.linalg.inv(wIc)
    return cIw


def get_pixpose_from_world_in_camera(wIc):
    rotation = wIc[:3, :3]
    tvec = wIc[:3, 3]
    pixpose = Pose.from_Rt(rotation, tvec)
    return pixpose


def get_pixpose_from_camera_in_world(cIw):
    wIc = np.linalg.inv(cIw)
    pixpose = get_pixpose_from_world_in_camera(wIc)
    return pixpose


def get_world_in_camera_from_colmap_image(colmap_image):
    wIc = np.eye(4)
    R = colmap_image.rotation_matrix()
    t = colmap_image.tvec
    wIc[:3, :3] = R
    wIc[:3, 3] = t
    return wIc


def get_camera_in_world_from_colmap_image(colmap_image):
    cIw = np.eye(4)
    R = colmap_image.rotation_matrix().T
    t = colmap_image.projection_center()
    cIw[:3, :3] = R
    cIw[:3, 3] = t
    return cIw


def get_colmap_image_from_cIw(cIw):
    wIc = np.linalg.inv(cIw)
    return get_colmap_image_from_wIc(wIc)


def get_colmap_image_from_wIc(wIc):
    qvec = pycolmap.rotmat_to_qvec(wIc[:3, :3])
    tvec = wIc[:3, 3]
    return pycolmap.Image(tvec=tvec, qvec=qvec)


def post_rotate_cIw(cIw, rz=0, rx=0, ry=0):
    rz = rz * np.pi / 180.0
    ry = ry * np.pi / 180.0
    rx = rx * np.pi / 180.0
    ro_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )
    ro_y = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    ro_x = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )
    R = np.eye(4)
    R[:3, :3] = ro_z @ ro_y @ ro_x
    cIw_r = cIw @ R
    return cIw_r


def post_transform_cIw(cIw, rz=0, rx=0, ry=0, trans=None):
    rz = rz * np.pi / 180.0
    ry = ry * np.pi / 180.0
    rx = rx * np.pi / 180.0
    ro_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )
    ro_y = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    ro_x = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )
    R = np.eye(4)
    R[:3, :3] = ro_z @ ro_y @ ro_x

    T = np.eye(4)
    if trans is not None:
        T[:3, 3] = [trans[0], trans[1], 0]

    cIw_r = cIw @ T @ R @ np.linalg.inv(T)
    return cIw_r


def transform_pixpose(pixpose, rz=0, rx=0, ry=0, trans=None):
    cIw = get_camera_in_world_from_pixpose(pixpose)
    cIw_r = post_transform_cIw(cIw, rz, rx, ry, trans=trans)
    rpixpose = get_pixpose_from_camera_in_world(cIw_r)
    return rpixpose


def rotate_pycolmap_image(cimg, rz=0, rx=0, ry=0):
    cIw = get_camera_in_world_from_colmap_image(cimg)
    cIw_r = post_rotate_cIw(cIw, rz, rx, ry)
    rcimg = get_colmap_image_from_cIw(cIw_r)
    return rcimg


def rotate_pixpose(pixpose, rz=0, rx=0, ry=0):
    cIw = get_camera_in_world_from_pixpose(pixpose)
    cIw_r = post_rotate_cIw(cIw, rz, rx, ry)
    rpixpose = get_pixpose_from_camera_in_world(cIw_r)
    return rpixpose


def rotate_image(img, angle, center=None):
    height, width = img.shape[:2]
    if center is None:
        center = (width / 2.0, height / 2.0)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height), borderValue=(255, 255, 255)
    )
    return rotated_image
