import math 

import torch
import os
import pickle as pkl
import trimesh
import numpy as np
from tqdm import tqdm
import argparse


from pixtrack.utils.pose_utils import geodesic_distance_for_rotations
from pykdtree.kdtree import KDTree


def get_vertices(mesh_path):
    object_mesh = trimesh.load(mesh_path)
    vertices = np.array(object_mesh.vertices)
    #vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    return vertices


def get_results_info(results_pickle_path): 
    with open(results_pickle_path, "rb") as f:
        poses_file = pkl.load(f)
    return poses_file


def get_pose_mat_from_tensor(pose_tensor):
    translation = pose_tensor.t.cpu().numpy()
    rotation = pose_tensor.R.cpu().numpy()
    mesh_pose_in_cam = np.eye(4)
    mesh_pose_in_cam[:3, :3] = rotation
    mesh_pose_in_cam[:3, -1] = translation
    return mesh_pose_in_cam


def similarity_transform(from_points, to_points):
    
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"
    
    N, m = from_points.shape
    
    mean_from = from_points.mean(axis = 0)
    mean_to = to_points.mean(axis = 0)
    
    delta_from = from_points - mean_from # N x m
    delta_to = to_points - mean_to       # N x m
    
    sigma_from = (delta_from * delta_from).sum(axis = 1).mean()
    sigma_to = (delta_to * delta_to).sum(axis = 1).mean()
    
    cov_matrix = delta_to.T.dot(delta_from) / N
    
    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)
    
    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m-1, m-1] = -1
    elif cov_rank < m-1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
    
    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c*R.dot(mean_from)
    
    return R, c, t


def lstsq_sphere_fitting(points):
    # add column of ones to pos_xyz to construct matrix A
    num_pts = points.shape[0]
    A = np.ones((num_pts, 4))
    A[:,0:3] = points

    # construct vector f
    f = np.sum(np.multiply(points, points), axis=1)
    sol, residules, rank, singval = np.linalg.lstsq(A,f)

    # solve the radius
    radius = math.sqrt((sol[0]*sol[0]/4.0)+(sol[1]*sol[1]/4.0)+(sol[2]*sol[2]/4.0)+sol[3])

    return radius, sol[0]/2.0, sol[1]/2.0, sol[2]/2.0


def get_pose_offset(poses_file):
    from_trs = []
    to_trs = []
    for image_key in poses_file:
        if (not poses_file[image_key]["success"]):
            continue
        to_trs.append(poses_file[image_key]["T_refined"].t.cpu().numpy())
        from_trs.append(poses_file[image_key]["gt_pose"].t.cpu().numpy())
    R, c, t = similarity_transform(np.array(from_trs), np.array(to_trs))
    pose_from_res_to_gt = np.eye(4)
    pose_from_res_to_gt[:3, :3] = R
    pose_from_res_to_gt[:3, -1] = t
    return pose_from_res_to_gt


def get_add_metric(gt_pts, predicted_pts):
    distance = np.linalg.norm(gt_pts - predicted_pts, axis=1)
    return np.mean(distance)



def get_adds_metric(gt_pts, predicted_pts):
    kdt = KDTree(gt_pts)
    distance, _ = kdt.query(predicted_pts, k=1)
    return np.mean(distance)


def get_metrics(results):
    object_path = results[next(iter(results))]["object_path"]
    video_name = str(results[next(iter(results))]["query_path"]).split('/')[-2]
    object_name = str(object_path).split('/')[-1]
    vertices = get_vertices(object_path/"textured.obj")
    radius, _, _, _ = lstsq_sphere_fitting(vertices)
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    diameter = radius * 2 * 100
    distances = []
    add_ss = []
    pose_dists = []

    pose_from_res_to_gt = get_pose_offset(results)
    bad_count = 0

    relocalizations =0
    add_vals = []
    adds_vals = []
    poses_results = results # results["estimation_results"]
    count = 1
    for image_key in tqdm(poses_results):
        if (not poses_results[image_key]["success"]):
            relocalizations += 1
            #print(f"skipped {image_key}")
            continue
        res_pose_mat = get_pose_mat_from_tensor(poses_results[image_key]["T_refined"])
        gt_pose_mat = get_pose_mat_from_tensor(poses_results[image_key]["gt_pose"])
        aligned_res_pose = np.dot(pose_from_res_to_gt, res_pose_mat)
        tr_dist = np.linalg.norm(gt_pose_mat[:3, -1] - aligned_res_pose[:3, -1]) * 100
        rot_dist = geodesic_distance_for_rotations(gt_pose_mat[:3, :3], aligned_res_pose[:3, :3]) * 180 / np.pi
        res_vertices = np.dot(pose_from_res_to_gt, np.dot(res_pose_mat, vertices.T)).T[:, :3] * 100
        gt_vertices = np.dot(gt_pose_mat, vertices.T).T[:, :3] * 100
        add_val = get_add_metric(res_vertices, gt_vertices)
        adds_val = get_adds_metric(gt_vertices, res_vertices)
        l2_dist = np.linalg.norm(res_vertices - gt_vertices, axis=1).mean()
        pose_dists.append(tr_dist)
        add_vals.append(add_val)
        adds_vals.append(adds_val)
        distances.append(l2_dist)
        count += 1
        
    evaluation_results = {}
    evaluation_results["average_error_vertices"] = np.mean(distances)
    evaluation_results["add_vals"] = np.array(add_vals)
    evaluation_results["adds_vals"] = np.array(adds_vals)
    evaluation_results["diameter"] = diameter
    evaluation_results["total_count"] = count
    evaluation_results["max_error"] = np.max(distances)
    evaluation_results["max_translation_error"] = np.max(pose_dists)
    evaluation_results["average_translation_error_pose"] = np.mean(pose_dists)
    evaluation_results["total_frames"] = len(results)
    evaluation_results["relocalizations"] = relocalizations
    evaluation_results["object_name"] = object_name
    evaluation_results["video_name"] = video_name
    return evaluation_results


def save_metrics(metrics, path):
    with open(os.path.join(path, 'metrics.pkl'), 'wb') as handle:
        pkl.dump(metrics, handle, protocol=pkl.HIGHEST_PROTOCOL)


def compute_metrics_for_multiple_thresholds(result_information):
    thresholds = [0.1, 0.15, 0.2]
    all_metrics = {}
    metrics = get_metrics(result_information)
    all_metrics["metrics"] = metrics
    for threshold_num in range(len(thresholds)):
        add_vals = metrics["add_vals"]
        adds_vals = metrics["adds_vals"]
        diameter = metrics["diameter"]
        count = metrics["total_count"]
        add_score = (add_vals < (thresholds[threshold_num] * diameter)).sum() / count
        adds_score = (adds_vals < (thresholds[threshold_num] * diameter)).sum() / count
        print(add_score, adds_score, thresholds[threshold_num])
        all_metrics[f"add_{thresholds[threshold_num]}"] = add_score
        all_metrics[f"adds_{thresholds[threshold_num]}"] = adds_score
    return all_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_file_path',
    )
    args = parser.parse_args()
    result_information = get_results_info(args.results_file_path)
    metrics = compute_metrics_for_multiple_thresholds(result_information)
    object_name = metrics["metrics"]["object_name"]
    video_name = metrics["metrics"]["video_name"]
    base_path = "/home/wayve/saurabh/test_metric"
    output_path = os.path.join(base_path, object_name, video_name)
    os.makedirs(output_path, exist_ok=True)
    save_metrics(metrics=metrics, path=output_path)
