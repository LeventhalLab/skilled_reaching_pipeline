import cv2
import numpy as np
import navigation_utilities
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten
from collections import defaultdict, Counter
import queue
import pandas as pd
import os
import glob
import copy
from utils import load_pose2d_fnames
import computer_vision_basics as cvb

def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    return out

def get_rtvec(M):
    rvec = cv2.Rodrigues(M[:3, :3])[0].flatten()
    tvec = M[:3, 3].flatten()
    return rvec, tvec

def get_most_common(vals):
    Z = linkage(whiten(vals), 'ward')
    n_clust = max(len(vals)/10, 3)
    clusts = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusts[clusts >= 0])
    most = cc.most_common(n=1)
    top = most[0][0]
    good = clusts == top
    return good

def select_matrices(Ms):
    Ms = np.array(Ms)
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in Ms]
    tvecs = np.array([M[:3, 3] for M in Ms])
    best = get_most_common(np.hstack([rvecs, tvecs]))
    Ms_best = Ms[best]
    return Ms_best


def mean_transform(M_list):
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in M_list]
    tvecs = [M[:3, 3] for M in M_list]

    rvec = np.mean(rvecs, axis=0)
    tvec = np.mean(tvecs, axis=0)

    return make_M(rvec, tvec)

def mean_transform_robust(M_list, approx=None, error=0.3):
    if approx is None:
        M_list_robust = M_list
    else:
        M_list_robust = []
        for M in M_list:
            rot_error = (M - approx)[:3,:3]
            m = np.max(np.abs(rot_error))
            if m < error:
                M_list_robust.append(M)
    return mean_transform(M_list_robust)


def get_transform(rtvecs, left, right):
    L = []
    for dix in range(rtvecs.shape[1]):
        d = rtvecs[:, dix]
        good = ~np.isnan(d[:, 0])

        if good[left] and good[right]:
            M_left = make_M(d[left, 0:3], d[left, 3:6])
            M_right = make_M(d[right, 0:3], d[right, 3:6])
            M = np.matmul(M_left, np.linalg.inv(M_right))
            L.append(M)
    L_best = select_matrices(L)
    M_mean = mean_transform(L_best)
    # M_mean = mean_transform_robust(L, M_mean, error=0.5)
    # M_mean = mean_transform_robust(L, M_mean, error=0.2)
    M_mean = mean_transform_robust(L, M_mean, error=0.1)
    return M_mean


def get_connections(xs, cam_names=None, both=True):
    n_cams = xs.shape[0]
    n_points = xs.shape[1]

    if cam_names is None:
        cam_names = np.arange(n_cams)

    connections = defaultdict(int)

    for rnum in range(n_points):
        ixs = np.where(~np.isnan(xs[:, rnum, 0]))[0]
        keys = [cam_names[ix] for ix in ixs]
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a = keys[i]
                b = keys[j]
                connections[(a,b)] += 1
                if both:
                    connections[(b,a)] += 1

    return connections


def get_calibration_graph(rtvecs, cam_names=None):
    n_cams = rtvecs.shape[0]
    n_points = rtvecs.shape[1]

    if cam_names is None:
        cam_names = np.arange(n_cams)

    connections = get_connections(rtvecs, np.arange(n_cams))

    components = dict(zip(np.arange(n_cams), range(n_cams)))
    edges = set(connections.items())

    graph = defaultdict(list)

    for edgenum in range(n_cams-1):
        if len(edges) == 0:
            component_names = dict()
            for k,v in list(components.items()):
                component_names[cam_names[k]] = v
            raise ValueError("""
Could not build calibration graph.
Some group of cameras could not be paired by simultaneous calibration board detections.
Check which cameras have different group numbers below to see the missing edges.
{}""".format(component_names))

        (a, b), weight = max(edges, key=lambda x: x[1])
        graph[a].append(b)
        graph[b].append(a)

        match = components[a]
        replace = components[b]
        for k, v in components.items():
            if match == v:
                components[k] = replace

        for e in edges.copy():
            (a,b), w = e
            if components[a] == components[b]:
                edges.remove(e)

    return graph

def find_calibration_pairs(graph, source=None):
    pairs = []
    explored = set()

    if source is None:
        source = sorted(graph.keys())[0]

    q = queue.deque()
    q.append(source)

    while len(q) > 0:
        item = q.pop()
        explored.add(item)

        for new in graph[item]:
            if new not in explored:
                q.append(new)
                pairs.append( (item, new) )
    return pairs

def compute_camera_matrices(rtvecs, pairs):
    extrinsics = dict()
    source = pairs[0][0]
    extrinsics[source] = np.identity(4)
    for (a,b) in pairs:
        ext = get_transform(rtvecs, b, a)
        extrinsics[b] = np.matmul(ext, extrinsics[a])
    return extrinsics

def get_initial_extrinsics(rtvecs, cam_names=None):
    graph = get_calibration_graph(rtvecs, cam_names)
    pairs = find_calibration_pairs(graph, source=0)
    extrinsics = compute_camera_matrices(rtvecs, pairs)

    n_cams = rtvecs.shape[0]
    rvecs = []
    tvecs = []
    for cnum in range(n_cams):
        rvec, tvec = get_rtvec(extrinsics[cnum])
        rvecs.append(rvec)
        tvecs.append(tvec)
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    return rvecs, tvecs


def crop_points_2_full_frame(pose_data, h5_group, cam_intrinsics):
    '''

    :param pose_data: dictionary containing: cam_names, points, scores, bodyparts
        cam_names = name of each camera
        points = num_cams x num_frames x num_joints x 2 array containing points as identified in the cropped views
        scores = num_cams x num_frames x num_joints array containing the DLC score for each point
        bodyparts = list of joints
    :param h5_group:
    :return:
    '''

    num_frames = np.shape(pose_data['points'])[1]
    for i_file, h5_file in enumerate(h5_group):
        h5_metadata = navigation_utilities.parse_dlc_output_h5_name(h5_file)
        dx = h5_metadata['crop_window'][0]
        dy = h5_metadata['crop_window'][2]
        crop_w = h5_metadata['crop_window'][1] - h5_metadata['crop_window'][0] + 1
        for i_frame in range(num_frames):
            # translate points from the cropped video to the full frame
            if 'fliplr' in h5_file:
                # video was flipped left-right
                pose_data['points'][i_file, i_frame, :, 0] = crop_w - pose_data['points'][i_file, i_frame, :, 0]
            pose_data['points'][i_file, i_frame, :, 0] += dx
            pose_data['points'][i_file, i_frame, :, 1] += dy

            # now undistort the full frame points
            pts = pose_data['points'][i_file, i_frame, :, :].astype(np.float64)
            pts_ud_norm = cv2.undistortPoints(pts, cam_intrinsics['mtx'], cam_intrinsics['dist'])
            pts_ud = cvb.unnormalize_points(pts_ud_norm, cam_intrinsics['mtx'])

            pose_data['points'][i_file, i_frame, :, :] = pts_ud

    return pose_data


def crop_all_points_2_full_frame(pose_data, h5_group, cam_intrinsics):
    '''

    :param pose_data: dictionary containing: cam_names, points, scores, bodyparts
        cam_names = name of each camera
        all_points = num_cams x num_frames x num_joints x num_outputs x 3 array containing points as identified in the cropped views
        bodyparts = list of joints
    :param h5_group:
    :return:
    '''

    n_bodyparts = np.shape(pose_data['all_points'])[2]
    n_frames = np.shape(pose_data['all_points'])[1]
    n_possible = np.shape(pose_data['all_points'])[3]
    for i_file, h5_file in enumerate(h5_group):
        h5_metadata = navigation_utilities.parse_dlc_output_h5_name(h5_file)
        dx = h5_metadata['crop_window'][0]
        dy = h5_metadata['crop_window'][2]
        crop_w = h5_metadata['crop_window'][1] - h5_metadata['crop_window'][0] + 1
        for i_frame in range(n_frames):
            # translate points from the cropped video to the full frame
            if 'fliplr' in h5_file:
                # video was flipped left-right
                try:
                    pose_data['all_points'][i_file, i_frame, :, :, 0] = crop_w - pose_data['all_points'][i_file, i_frame, :, :, 0]
                except:
                    pass
            pose_data['all_points'][i_file, i_frame, :, :, 0] += dx
            pose_data['all_points'][i_file, i_frame, :, :, 1] += dy

            for i_bp in range(n_bodyparts):
                # now undistort the full frame points
                pts_ud = np.full((n_possible, 2), fill_value=np.nan)
                for i_out in range(n_possible):
                    pts_ud_norm = cv2.undistortPoints(pose_data['all_points'][i_file, i_frame, i_bp, i_out, :2], cam_intrinsics['mtx'], cam_intrinsics['dist'])
                    pts_ud[i_out, :] = cvb.unnormalize_points(pts_ud_norm, cam_intrinsics['mtx'])

                pose_data['all_points'][i_file, i_frame, i_bp, :, :2] = pts_ud

    return pose_data



def match_dlc_points_from_all_views(h5_list, cam_names, calibration_data, parent_directories, min_valid_score=0.99, filtered=False, pctile_to_keep=10):
    # in general, use a very restrictive score cutoff since this is just to optimize the calibration (don't need full paw tracking)

    fname_dict = dict.fromkeys(cam_names)
    # find matching files from each view
    num_matched_pts = 0
    for h5_file in h5_list[0]:
        h5_vid_metadata = navigation_utilities.parse_dlc_output_h5_name(h5_file)
        cropped_session_folder = navigation_utilities.find_rat_cropped_session_folder(h5_vid_metadata, parent_directories)
        _, cropped_session_folder_name = os.path.split(cropped_session_folder)
        h5_group = [h5_file]
        for cam_name in cam_names[1:]:
            cam_folder_name = os.path.join(cropped_session_folder, '_'.join((cropped_session_folder_name, cam_name)))
            test_name = navigation_utilities.test_dlc_h5_name_from_h5_metadata(h5_vid_metadata)
            full_test_name = os.path.join(cam_folder_name, test_name)

            view_h5_list = glob.glob(full_test_name)

            # eliminate the processed versions of the .h5 files from dlc
            view_h5_list = [h5_file for h5_file in view_h5_list if '_el' not in h5_file]
            if len(view_h5_list) == 1:
                # found exactly one .h5 file to match the one from the direct view
                h5_group.append(view_h5_list[0])

        # now have a group of .h5 files for a single trial
        if len(h5_group) != 3:
            continue

        for i_cam, cam_name in enumerate(cam_names):
            fname_dict[cam_name] = h5_group[i_cam]

        d = load_pose2d_fnames(fname_dict, cam_names=cam_names)

        # this version of d has an "all_points" array which is a num_cams x num_frames x num_bodyparts x num_possibilities x 3 array
        # the last 3 values are the x value, y value, and score. Need to rearrange into points and scores arrays
        ind_points_scores = np.squeeze(d['all_points'][:, :, :, 0, :])   # eliminates other "possible" points, which shouldn't matter for purposes of recalibrating
        d['points'] = ind_points_scores[:, :, :, :2]
        d['scores'] = ind_points_scores[:, :, :, 2]
        d = crop_points_2_full_frame(d, h5_group, calibration_data['cam_intrinsics'])

        # test_pose_data(h5_metadata, session_metadata, d, calibration_data['cam_intrinsics'], parent_directories)

        n_cams, n_points, n_joints, _ = d['points'].shape

        # need to copy so full dlc_output gets written to r3d_data
        points = copy.deepcopy(d['points'])
        scores = d['scores']

        # remove points that are below threshold
        points[scores < min_valid_score] = np.nan
        imgp = match_camera_view_pts(points)

        # todo: calculate reconstruction errors with current stereo parameters, maybe eliminate points with errors that are too large (presumably at least one is a mistake)
        # p3ds_flat = cgroup.triangulate(imgp, progress=True, undistort=False)
        # reprojerr_flat = cgroup.reprojection_error(p3ds_flat, imgp, mean=True)

        if num_matched_pts == 0:
            all_imgp = imgp
            # all_err = reprojerr_flat
        else:
            all_imgp = [np.vstack((prev_pts, vid_imgp)) for (prev_pts, vid_imgp) in zip(all_imgp, imgp)]
            # all_err = [np.vstack((prev_err, vid_imgp)) for (prev_err, vid_imgp) in zip(all_err, reprojerr_flat)]
        num_matched_pts += np.shape(imgp)[1]

    cgroup = calibration_data['cgroup']
    all_imgp = np.array(all_imgp)
    p3ds_flat = cgroup.triangulate(all_imgp, progress=True, undistort=False)
    reprojerr_flat = cgroup.reprojection_error(p3ds_flat, all_imgp, mean=True)

    # get rid of bad reprojections - assume those are mislabeled points despite the high confidence level
    pctile_cutoff = np.percentile(reprojerr_flat, pctile_to_keep)
    valid_imgp = all_imgp[:, reprojerr_flat < pctile_cutoff, :]

    return valid_imgp


def match_camera_view_pts(points):

    num_cams = np.shape(points)[0]
    num_frames = np.shape(points)[1]
    num_bodyparts = np.shape(points)[2]

    # imgp = np.array((num_cams, num_frames, ))
    num_valid_pts = 0
    for i_frame in range(num_frames):
        valid_bool = np.ones((num_bodyparts), dtype=bool)
        for i_cam in range(num_cams):
            valid_bool = np.logical_and(valid_bool, np.logical_not(np.isnan(points[i_cam, i_frame, :, 0])))
        if num_valid_pts == 0:
            imgp = [cam_pts[i_frame, valid_bool, :] for cam_pts in points]
        else:
            imgp = [np.vstack((prev_cam_pts, frame_cam_pts[i_frame, valid_bool, :])) for (prev_cam_pts, frame_cam_pts) in zip(imgp, points)]
        num_valid_pts += np.shape(imgp)[1]

    return imgp