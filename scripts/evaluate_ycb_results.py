import torch
import os
import pickle as pkl
import trimesh
import numpy as np

from pixtrack.utils.pose_utils import geodesic_distance_for_rotations
from sklearn.neighbors import KDTree


def get_vertices(mesh_path):
    object_mesh = trimesh.load(mesh_path)
    vertices = np.array(object_mesh.vertices)
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
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


def get_metrics(results, tr_threshold, rot_threshold, add_tr_threshold):
    vertices = get_vertices(results.get("mesh_path", "/mnt/remote/data/prajwal/YCB_Video_Dataset/models/035_power_drill/textured.obj"))
    distances = []
    add_ss = []
    pose_dists = []

    pose_from_res_to_gt = get_pose_offset(results)
    bad_count = 0

    relocalizations =0
    add_vals = []
    adds_vals = []
    poses_results = results # results["estimation_results"]
    for image_key in poses_results:
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
        add_val = get_add_metric(res_vertices, gt_vertices, add_threshold)
        adds_val = get_adds_metric(res_vertices, gt_vertices, add_threshold)

        l2_distances = np.linalg.norm(gt_vertices - res_vertices, axis=1)
        pose_dists.append(tr_dist)
        add_vals.append(add_val)
        adds_vals.append(adds_val)
        l2_dist = np.mean(l2_distances)
        if tr_dist > tr_threshold or rot_dist > rot_threshold:
            bad_count += 1
            
        distances.append(l2_dist)
    add_score = (add_vals < add_tr_threshold).sum() / count
    adds_score = (adds_vals < add_tr_threshold).sum() / count
        
    evaluation_results = {}
    evaluation_results["average_error_vertices"] = np.mean(distances)
    evaluation_results["max_error"] = np.max(distances)
    evaluation_results["max_translation_error"] = np.max(pose_dists)
    evaluation_results["average_translation_error_pose"] = np.mean(pose_dists)
    evaluation_results["bad_count"] = bad_count
    evaluation_results["total_frames"] = len(results)
    evaluation_results["accuracy"] = (1.0*(len(results) - bad_count))/(1.0*len(results) )
    evaluation_results["relocalizations"] = relocalizations
    return evaluation_results


def save_metrics(metrics, path):
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_metrics_for_multiple_thresholds(result_information):
    thresholds = {"tr_threshold":[0.03, 0.05, 0.08], "rot_threshold":[3, 5, 8]}
    all_metrics = {}
    for threshold_num in range(len(thresholds["tr_threshold"])):
        tr_threshold = thresholds["tr_threshold"][threshold_num]
        rot_threshold = thresholds["tr_threshold"][threshold_num]
        metrics = get_metrics(result_information, tr_threshold, rot_threshold, tr_threshold)
        all_metrics[f"{tr_threshold}"] = metrics
    return all_metrics


if __name__ == "__main__":
    results_file_path = "/mnt/remote/data/prajwal/pixtrack/results/003_cracker_box/ycb_16/poses.pkl"
    result_information = get_results_info(results_file_path)
    metrics = compute_metrics_for_multiple_thresholds(result_information)
    save_metrics(metrics=metrics, path=os.path.join(outputs_path, result_information["object_name"], result_information["video_path"]))
