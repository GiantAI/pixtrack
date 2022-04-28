import numpy as np
import tqdm
from collections import defaultdict
from hloc.utils.read_write_model import read_model
from hloc.visualization import read_image
from hloc.utils.io import get_keypoints, get_matches
from hloc.utils.parsers import names_to_pair
import cv2
import h5py
import copy
from pathlib import Path
import pycolmap
from hloc.utils.read_write_model import read_images_binary, \
     read_cameras_binary, read_points3D_binary, Image, \
     write_points3D_binary, write_images_binary, Point3D, \
     write_cameras_binary
from hloc.utils.database import COLMAPDatabase
from hloc import logger
from pixtrack.utils.pose_utils import rotate_pycolmap_image

def extract_covisibility(model_path):
    cameras, images, points3D = read_model(model_path)
    pairs = []
    covis_all = {}
    for image_id, image in tqdm(images.items()):
        matched = image.point3D_ids != -1
        points3D_covis = image.point3D_ids[matched]

        covis = defaultdict(int)
        for point_id in points3D_covis:
            for image_covis_id in points3D[point_id].image_ids:
                if image_covis_id != image_id:
                    covis[image_covis_id] += 1

        if len(covis) == 0:
            continue

        covis_ids = np.array(list(covis.keys()))
        covis_all[image_id] = covis
    return covis_all

# Read features from features.h5
def read_features(features_path, image_name):
    features_dict = {}
    with h5py.File(str(features_path), 'r') as f:
        feat_o = f[image_name]
        features_dict['keypoints'] = np.array(feat_o['keypoints'])
        features_dict['descriptors'] = np.array(feat_o['descriptors'])
        features_dict['scores'] = np.array(feat_o['scores'])
        features_dict['image_size'] = np.array(feat_o['image_size'])
    return features_dict

# Write features to features.h5
def write_features(feature_path, image_name, features_dict):
    pred = features_dict
    name = image_name
    with h5py.File(str(feature_path), 'a') as fd:
        try:
            if name in fd:
                del fd[name]
            grp = fd.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
            #if 'keypoints' in pred:
            #    grp['keypoints'].attrs['uncertainty'] = uncertainty
        except OSError as error:
            if 'No space left on device' in error.args[0]:
                logger.error(
                    'Out of disk space: storing features on disk can take '
                    'significant space, did you enable the as_half flag?')
                del grp, fd[name]
            raise error
    return

# Write matches to matches.h5
def write_matches(match_path, name0, name1, matches, scores=None):
    pred = matches
    pair = names_to_pair(name0, name1)
    with h5py.File(str(match_path), 'a') as fd:
        if pair in fd:
            #raise Exception('Matches already in the database')
            del fd[pair]
        grp = fd.create_group(pair)
        #matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)

        if scores is not None:
            #scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)
    return

def add_rotation_augmentation_to_features_and_matches(image_list, images, features_path, matches_path, angle_step=30):
    completed_images = copy.deepcopy(image_list)
    image_dict = defaultdict(dict)
    for image_name in tqdm.tqdm(image_list):
        image_path = images / image_name
        features_dict_orig = read_features(str(features_path), image_name)
        img = read_image(image_path)
        height, width = img.shape[:2]
        center = (width / 2., height / 2.)
        for angle in tqdm.tqdm(range(angle_step, 360, angle_step)):
            # Rotate image
            rotate_matrix = cv2.getRotationMatrix2D(center=center, 
                                            angle=angle, 
                                            scale=1)
            rotated_image = cv2.warpAffine(src=img, 
                                       M=rotate_matrix, 
                                       dsize=(width, height),
                                       borderValue=(255, 255, 255))
            # Save image
            aug_name = 'mapping/%d_%s' % (angle, image_name.split('/')[-1])
            out_path = str(images / aug_name)
            rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, rotated_image)

            # Rotate keypoints
            kps_src = features_dict_orig['keypoints']
            kps_aug = kps_src @ rotate_matrix[:2, :2].T
            kps_aug += rotate_matrix[:, 2]
            features_dict = copy.deepcopy(features_dict_orig)
            features_dict['keypoints'] = kps_aug

            # Save features
            write_features(features_path, aug_name, features_dict)
            
            # Write matches
            for mimage in completed_images:
                try:
                    matches, scores = get_matches(matches_path, image_name, mimage)
                except Exception as e:
                    continue
                matches_arr = np.ones(kps_aug.shape[0]) * -1
                matches_arr[matches[:, 0]] = matches[:, 1]
                matches_arr = matches_arr.astype(np.int16)
                scores_arr = np.zeros(kps_aug.shape[0])
                scores_arr[matches[:, 0]] = 0.999
                write_matches(matches_path, aug_name, mimage, matches_arr, scores_arr)
                
            matches_arr = np.array(range(kps_aug.shape[0]))
            matches_arr = matches_arr.astype(np.int16)
            scores_arr = np.repeat(0.999, kps_aug.shape[0])
            write_matches(matches_path, image_name, aug_name, matches_arr, scores_arr)
            
            completed_images.append(aug_name)
            image_dict[image_name][angle] = aug_name
            #break
        #break
    return image_dict

def create_db_from_model(reconstruction, database_path, augmented_images=None):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in reconstruction.cameras.items():
        db.add_camera(
            camera.model_id, camera.width, camera.height, camera.params,
            camera_id=i, prior_focal_length=True)

    for i, image in reconstruction.images.items():
        db.add_image(image.name, image.camera_id, image_id=i)        
    dref = {image.name: i for i, image in reconstruction.images.items()}
    
    if augmented_images is not None:
        for i, image in augmented_images.items():
            db.add_image(image.name, image.camera_id, image_id=i)
        daug = {image.name: i for i, image in augmented_images.items()}
        dref.update(daug)

    db.commit()
    db.close()
    return dref

def get_image_from_name(imname, images):
    for img_id in images:
        if images[img_id].name == imname:
            return images[img_id]
    return None

def augment_rotation(model, image, angle, cameras, new_name=None, new_idx=None):
    image_id = image.id
    pyimg = model.images[image_id]
    rot_img = rotate_pycolmap_image(pyimg, rz=angle)
    qvec = rot_img.qvec
    tvec = rot_img.tvec
    camera_id = image.camera_id
    point3D_ids = image.point3D_ids
    xys = image.xys
    
    camera = cameras[camera_id]
    width = camera.width
    height = camera.height
    center = (width / 2., height / 2.)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, 
                        angle=angle, 
                        scale=1)
    # Rotate keypoints
    kps_src = copy.deepcopy(xys)
    kps_aug = kps_src @ rotate_matrix[:2, :2].T
    kps_aug += rotate_matrix[:, 2]
    
    aug_img = Image(id=new_idx, qvec=qvec, tvec=tvec,
          camera_id=camera_id, name=new_name,
          xys=xys, point3D_ids=point3D_ids)
    
    return aug_img

def augment_images_and_points3D(outputs, image_dict):
    sfm_dir = outputs / 'sfm'
    aug_sfm_dir = outputs / 'aug_sfm'
    aug_imgs = aug_sfm_dir / 'images.bin'
    aug_pts3d = aug_sfm_dir / 'points3D.bin'
    aug_cams = aug_sfm_dir / 'cameras.bin'

    augmented_images = {}
    points_3d = defaultdict(lambda: defaultdict(list))
    model = pycolmap.Reconstruction(sfm_dir)
    idx = max(model.images.keys())

    images_bin = read_images_binary(sfm_dir / 'images.bin')
    cameras_bin = read_cameras_binary(sfm_dir / 'cameras.bin')
    points3d_bin = read_points3D_binary(sfm_dir / 'points3D.bin')

    for img_name in tqdm.tqdm(image_dict):
        image = get_image_from_name(img_name, images_bin)
        camera = cameras_bin[image.camera_id]
        point3D_ids = image.point3D_ids

        for angle in tqdm.tqdm(image_dict[img_name]):
            idx += 1
            aug_img = augment_rotation(model, image, angle, cameras_bin, 
                                       new_name=image_dict[img_name][angle], 
                                       new_idx=idx)
            augmented_images[idx] = aug_img
            for i in range(len(point3D_ids)):
                point3d_id = point3D_ids[i]
                if point3d_id == -1:
                    continue
                points_3d[point3d_id]['image_ids'].append(idx)
                points_3d[point3d_id]['point2D_idxs'].append(i)

    aug_imgs_bin = copy.deepcopy(images_bin)
    aug_imgs_bin.update(augmented_images)
    aug_points3d_bin = {}
    for point3D_id in points3d_bin:
        xyz = points3d_bin[point3D_id].xyz
        rgb = points3d_bin[point3D_id].rgb
        error = points3d_bin[point3D_id].error
        image_ids = points3d_bin[point3D_id].image_ids
        point2D_idxs = points3d_bin[point3D_id].point2D_idxs
        if point3D_id in points_3d:
            image_ids = np.hstack((image_ids, points_3d[point3D_id]['image_ids']))
            point2D_idxs = np.hstack((point2D_idxs, points_3d[point3D_id]['point2D_idxs']))
        aug_pt = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                   error=error, image_ids=image_ids,
                   point2D_idxs=point2D_idxs)
        aug_points3d_bin[point3D_id] = aug_pt

    write_points3D_binary(aug_points3d_bin, aug_pts3d)
    write_images_binary(aug_imgs_bin, aug_imgs)
    write_cameras_binary(cameras_bin, aug_cams)
    return augmented_images
