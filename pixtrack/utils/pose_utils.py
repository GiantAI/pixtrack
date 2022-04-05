import numpy as np
import pycolmap


def get_world_in_camera_from_pixpose(pixpose):
    pixpose = pixpose.numpy()
    wIc = np.eye(4)
    wIc[:3, :3] = pixpose[0]
    wIc[:3, 3] = pixpose[1]
    return wIc

def get_camera_in_world_from_pixpose(pixpose):
    wIc = get_world_in_camera_from_pixpose(pixpose)
    cIw = np.linalg.inv(wIc)
    return cIw

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

def rotate_pycolmap_image(cimg, rz=0, rx=0, ry=0):
    cIw = get_camera_in_world(cimg)
    
    rz = rz * np.pi / 180.
    ry = ry * np.pi / 180.
    rx = rx * np.pi / 180.
    ro_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz),  np.cos(rz), 0],
                     [        0,            0, 1]])
    ro_y = np.array([[ np.cos(ry),  0, np.sin(ry)],
                     [          0,  1,          0],
                     [-np.sin(ry),  0, np.cos(ry)]])
    ro_x = np.array([[1,          0,           0],
                     [0, np.cos(rx), -np.sin(rx)],
                     [0, np.sin(rx),  np.cos(rx)]])
    R = np.eye(4)
    R[:3, :3] = ro_z @ ro_y @ ro_x
    cIw_r = cIw @ R   
    rcimg = get_colmap_image_from_cIw(cIw_r)
    return rcimg
