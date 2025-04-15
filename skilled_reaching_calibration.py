import navigation_utilities
import skilled_reaching_calibration
import skilled_reaching_io
import reconstruct_3d_optitrack
import sr_visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import plot_utilities
import crop_videos
import os
import csv
import numpy as np
import copy
import random
import cv2
from cv2 import aruco
import glob
from tqdm import trange
import computer_vision_basics as cvb
import skilled_reaching_io
from boards import CharucoBoard, Checkerboard, merge_rows, extract_points, extract_rtvecs
from cameras import Camera, CameraGroup
from utils import load_pose2d_fnames, get_initial_extrinsics, make_M, get_rtvec, get_connections
from anipose_utils import match_dlc_points_from_all_views
from random import randint
from pprint import pprint
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')


def refine_calibration(calibration_data, h5_list, parent_directories, min_conf=0.99, verbose=False):
    '''

    :param calibration_data:
    :param h5_list: list of lists of dlc output (.h5) files containing dlc output
    :return:
    '''
    h5_metadata = navigation_utilities.parse_dlc_output_h5_name(h5_list[0][0])
    print('refining calibration for {}, {}, session {:d}'.format(h5_metadata['ratID'], h5_metadata['triggertime'].strftime('%m/%d/%Y'), h5_metadata['session_num']))

    cgroup = calibration_data['cgroup']
    cam_names = cgroup.get_names()
    calibration_data['original_cgroup'] = copy.deepcopy(cgroup)

    imgp = match_dlc_points_from_all_views(h5_list, cam_names, calibration_data, parent_directories)

    # do the fundamental and essential matrices need to be recalculated?
    # imgp_dict ={cam_name: imgp[i_cam] for i_cam, cam_name in enumerate(cam_names)}
    # cam_intrinsics = calibration_data['cam_intrinsics']
    # E, F, rot, t = mirror_stereo_cal(imgp_dict, cam_intrinsics, view_names=cam_names)
    # error = cgroup.bundle_adjust_iter_fixed_dist(imgp, extra=None, verbose=verbose)
    error = cgroup.bundle_adjust_iter_fixed_intrinsics(imgp, undistort=False, extra=None, verbose=verbose)


    # cgroup was modified by the bundle_adjust_iter_fixed_intrinsics function
    cgroup_name = '_'.join((h5_metadata['ratID'],
                            h5_metadata['triggertime'].strftime('%Y%m%d'),
                            'ses{:02d}'.format(h5_metadata['session_num']),
                            'cgroup'))
    calibration_data[cgroup_name] = cgroup
    cgroup_error_name = '_'.join((h5_metadata['ratID'],
                            h5_metadata['triggertime'].strftime('%Y%m%d'),
                            'ses{:02d}'.format(h5_metadata['session_num']),
                            'cgroup_error'))
    calibration_data[cgroup_error_name] = error

    return calibration_data


def calibration_metadata_from_df(session_metadata, calibration_metadata_df):
    '''

    :param session_metadata: dictionary with keys 'time' and 'boxnum'
    :param calibration_metadata_df: dataframe with columns 'nrows', 'ncols', 'date', 'box_num', 'calib_type'
        calib_type can be 'chessboard'
    :return:
    '''

    calibration_metadata = {'nrows': 0,
                            'ncols': 0,
                            'square_length': 0,
                            'marker_length': 0,
                            'board_type': 'none'}

    session_row = calibration_metadata_df[(calibration_metadata_df['box_num'] == session_metadata['boxnum']) &
                                          (calibration_metadata_df['date'] == session_metadata['time'].date())]

    if len(session_row) == 1:
        try:
            calibration_metadata['nrows'] = int(session_row['nrows'].values[0])
        except:
            pass
        calibration_metadata['ncols'] = int(session_row['ncols'].values[0])
        calibration_metadata['square_length'] = session_row['square_length'].values[0]
        calibration_metadata['marker_length'] = session_row['marker_length'].values[0]
        calibration_metadata['board_type'] = session_row['calib_type'].values[0]
    elif len(session_num) > 1:
        print('ambiguous session metadata')
    elif len(session_num) == 0:
        print('no calibration metadata for this session')

    return calibration_metadata


def anipose_calibrate_3d(vid_list, cam_names, calibration_metadata, manual_verify=False):
    '''

    :param vid_list:
    :param calibration_metadata: dictionary with the following keys:
        'nrows' - number of rows in the chessboard/charuco board
        'ncols' - number of columns in the chessboard/charuco board
        'square_length' - length of each chessboard square (usually in mm)
        'marker_length' - length of each Aruco marker (usually in mm - should be same units as square_length for sure)
        'calib_type' - 'chessboard' or 'charuco'
    :return:
    '''

    calibration_path, _ = os.path.split(vid_list[0])


    cal_name = '_'.join(['calibration',
                         'box{:02d}'.format(calibration_metadata['boxnum']),
                         '{}.toml'.format(calibration_metdata['time'].strftime('%Y%m%d-%H-%M-%S'))])
    cal_name = os.path.join(calibration_path, cal_name)

    if calibration_metadata['board_type'].lower() in ['chessboard', 'checkerboard']:
        board = Checkerboard(calibration_metadata['nrows'], calibration_metadata['ncols'], square_length=calibration_metadata['square_length'], manually_verify=manual_verify)
    elif calibration_metadata['board_type'].lower() in ['charuco']:
        board = CharucoBoard(calibration_metadata['nrows'], calibration_metadata['ncols'],
                             square_length=calibration_metadata['square_length'], marker_length=calibration_metadata['marker_length'], manually_verify=manual_verify)

    ncams = len(vid_list)
    cgroup = CameraGroup.from_names(cam_names)

    # weird thing about anipose - they want a list of lists of videos (i.e., vid_list = [[vid1], [vid2], vid3], ...],
    # NOT vid_list = [vid1, vid2, vid3,...]
    if isinstance(vid_list[0], str):
        vid_list = [[vid_name] for vid_name in vid_list]
    cgroup.calibrate_videos(vid_list, board)

    cgroup.dump(cal_name)

    pass



def refine_optitrack_calibration_from_dlc(session_metadata, parent_directories, min_conf2match=0.98):

    # find the folder containing cropped videos and dlc pickle files for this session
    session_folder, cam_folders = navigation_utilities.find_cropped_session_folder(session_metadata, parent_directories)
    cal_data_parent = parent_directories['cal_data_parent']

    # find the nearest calibration file based on chessboard calibration
    # find the closest calibration data file
    metadata_keys = session_metadata.keys()
    test_datetime_keys = ['trialtime', 'date', 'time']
    for test_key in test_datetime_keys:
        if test_key in metadata_keys:
            session_datetime = session_metadata[test_key]
    cal_file = navigation_utilities.find_optitrack_calibration_data_name(cal_data_parent, session_datetime, max_days_to_look_back=5, basename='calibrationdata')
    cal_data = skilled_reaching_io.read_pickle(cal_file)

    # collect matched dlc points
    _, session_foldername = os.path.split(session_folder)
    test_name = session_foldername + '_*_full.pickle'
    cam_pickles = [glob.glob(os.path.join(cf, test_name)) for cf in cam_folders]

    matched_points, matched_conf, cam01_pickle_files = collect_matched_dlc_points(cam_pickles, parent_directories)

    # matched_points should be a list of pairs of arrays of matched points

    all_pts, all_conf, valid_pts_bool = trialpts2allpts(matched_points, matched_conf, min_conf2match)
    valid_pts = [cam_pts[valid_pts_bool, :] for cam_pts in all_pts]

    # collect_cam_undistorted_points(valid_pts, cal_data)
    stereo_ud, stereo_ud_norm = collect_cam_undistorted_points(valid_pts, cal_data)

    # matched_points is a num_trials x 2 list of lists. matched_points[i_cam]
    # recal_E, E_mask = cv2.findEssentialMat(stereo_ud[0], stereo_ud[1], cal_data['mtx'][0], None,
    #                                  cal_data['mtx'][1], None, cv2.FM_RANSAC, 0.999, 0.1)
    #
    # # convert E_mask into a boolean vector for indexing
    # inlier_idx = np.squeeze(E_mask) != 0
    #
    # inliers = [np.squeeze(cam_val_pts[inlier_idx, :]) for cam_val_pts in stereo_ud]

    # calculate R and T based on the recalibrated essential matrix
    # select 2000 points at random for chirality check (using all the points takes a really long time)
    # num_pts = np.shape(stereo_ud[0])[0]
    # pt_idx = np.random.randint(0, num_pts, 2000)
    # _, R_from_E, T_Eunit, ffm_msk = cv2.recoverPose(E, stereo_ud[0][pt_idx, :], stereo_ud[1][pt_idx, :],
    #                                                 cal_data['mtx'][0])
    # _, R_from_E_recal, T_Eunit_recal, rp_msk = cv2.recoverPose(recal_E, inliers[0], inliers[1],
    #                                                 cal_data['mtx'][0])
    #
    # R1, R2, T = cv2.decomposeEssentialMat(recal_E)
    #
    dist = np.zeros((1,5))
    # _, E_in, R_in, T_in, msk = cv2.recoverPose(inliers[0], inliers[1], cal_data['mtx'][0], dist, cal_data['mtx'][1], dist,
    #                                            method=cv2.FM_RANSAC, prob=0.999, threshold=0.1)
    _, E_in, R_in, T_in, msk = cv2.recoverPose(stereo_ud[0], stereo_ud[1], cal_data['mtx'][0], dist, cal_data['mtx'][1], dist,
                                               method=cv2.FM_RANSAC, prob=0.999, threshold=0.1)

    # matched_unnorm = []
    # for i_cam in range(2):
    #     pts_ud = cv2.undistortPoints(valid_pts[i_cam], cal_data['mtx'][i_cam], cal_data['dist'][i_cam])
    #     pts_un = cvb.unnormalize_points(pts_ud, cal_data['mtx'][i_cam])
    #     matched_unnorm.append(pts_un)
    #
    # F, F_mask = cv2.findFundamentalMat(matched_unnorm[0], matched_unnorm[1], cv2.FM_RANSAC, 0.1, 0.999)

    # F_from_E = np.linalg.inv(cal_data['mtx'][1].T) @ E @ np.linalg.inv(cal_data['mtx'][0])
    # E_from_F = cal_data['mtx'][1].T @ F @ cal_data['mtx'][0]

    # test that matched points are in the right place, and the F gives epipolar lines that look good
    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # trial_idx = 0
    # i_frame = 10
    # cam_pickle_files = navigation_utilities.find_other_optitrack_pickles(cam01_pickle_files[trial_idx], parent_directories)
    # cam_meta_files = [pickle_file.replace('full.pickle', 'meta.pickle') for pickle_file in cam_pickle_files]
    # dlc_metadata = [skilled_reaching_io.read_pickle(cam_meta_file) for cam_meta_file in cam_meta_files]
    # im_size = []
    # for i_cam in range(2):
    #     pickle_metadata = navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam_pickle_files[i_cam])
    #
    #     im_size.append(sr_visualization.overlay_pts_on_original_frame(matched_points[trial_idx][i_cam][i_frame], matched_conf[trial_idx][i_cam][i_frame, :], pickle_metadata, dlc_metadata[i_cam], i_frame, cal_data, parent_directories,
    #                                   axs[i_cam], plot_undistorted=True, frame_pts_already_undistorted=False, min_conf=min_conf2match))
    # for i_cam in range(2):
    #     plot_point_bool = matched_conf[trial_idx][i_cam][i_frame, :] > min_conf2match
    #     bodyparts = dlc_metadata[i_cam]['data']['DLC-model-config file']['all_joints_names']
    #     sr_visualization.F(matched_points[trial_idx][i_cam][i_frame], 1 + i_cam, F_from_E, im_size[i_cam], bodyparts, plot_point_bool, axs[1-i_cam], lwidth=0.5,
    #                                linestyle='-')
    #
    # plt.show()

    return E_in, R_in, T_in


def collect_cam_undistorted_points(distorted_pts, cal_data):
    # todo: WORKING HERE... return undistorted points given distorted points
    # make this work whether distorted_pts is a list of single frame distorted points or one large array of distorted points

    # convert to normalized coordinates for pose recovery
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    pts_ud_norm = []
    pts_ud = []
    stereo_ud_norm_for_T = []
    stereo_ud_for_T = []
    for i_cam in range(2):
        pts = np.array(distorted_pts[i_cam])

        if np.ndim(pts) == 2:
            # pts is an n x 2 array, convert directly to undistorted points
            pts_ud_norm.append(cv2.undistortPoints(pts, mtx[i_cam], dist[i_cam]))
            pts_ud.append(cvb.unnormalize_points(pts_ud_norm[i_cam], mtx[i_cam]))
        else:
            # need to clean this up at some points to undistort points on a per-frame basis
            framepts_ud_norm = [cv2.undistortPoints(frame_pts, mtx[i_cam], dist[i_cam]) for frame_pts in pts]
            framepts_ud = [cvb.unnormalize_points(fpts_ud_norm, mtx[i_cam]) for fpts_ud_norm in framepts_ud_norm]
            num_frames = np.shape(framepts_ud)[0]
            pts_per_frame = np.shape(framepts_ud)[1]
            pts_r = np.reshape(pts, (-1, 2))
            pts_ud_norm = cv2.undistortPoints(pts_r, mtx[i_cam], cal_data['dist'][i_cam])
            pts_ud = cvb.unnormalize_points(pts_ud_norm, mtx[i_cam])
            stereo_ud_norm.append(pts_ud_norm)
            stereo_ud.append(pts_ud)
            stereo_ud_norm_for_T.append(framepts_ud_norm)

        # for i_frame, fpts in enumerate(framepts_ud):
        #     framepts_ud[i_frame] = np.reshape(fpts, (pts_per_frame, 1, 2))
        # stereo_ud_for_T.append(framepts_ud)

    return pts_ud, pts_ud_norm
    # select 2000 points at random for chirality check (using all the points takes a really long time)
    # num_pts = np.shape(pts_ud)[0]
    # pt_idx = np.random.randint(0, num_pts, 2000)
    # _, R_from_E, T_Eunit, ffm_msk = cv2.recoverPose(E, stereo_ud[0][pt_idx, :], stereo_ud[1][pt_idx, :],
    #                                                 cal_data['mtx'][0])
    # _, R_from_F, T_Funit, ffm_msk = cv2.recoverPose(E_from_F, stereo_ud[0][pt_idx, :], stereo_ud[1][pt_idx, :],
    #                                                 cal_data['mtx'][0])
    # _, R_norm, T_norm_unit, ffm_msk = cv2.recoverPose(E_norm, stereo_ud_norm[0][pt_idx, :],
    #                                                   stereo_ud_norm[1][pt_idx, :],
    #                                                   np.identity(3))
    # norm_mtx = [np.identity(3) for i_cam in range(2)]
    # T_norm = estimate_T_from_ffm(stereo_objpoints, stereo_ud_norm_for_T, im_size, norm_mtx, R_norm)
    # T_from_E = estimate_T_from_ffm(stereo_objpoints, stereo_ud_for_T, im_size, mtx, R_from_E)
    # T_from_F = estimate_T_from_ffm(stereo_objpoints, stereo_ud_for_T, im_size, mtx, R_from_F)

def collect_matched_dlc_points(cam_pickles, parent_directories, num_trials_to_match=5):

    num_cams = len(cam_pickles)
    num_trials = len(cam_pickles[0])
    trial_idx_to_match = random.sample(range(num_trials), num_trials_to_match)

    #todo: pick num_trials_to_match trials at random for point-matching
    matched_dlc_points = []
    matched_dlc_conf = []
    cam01_names = []
    for trial_idx in trial_idx_to_match:
        cam01_pickle = cam_pickles[0][trial_idx]
        cam01_folder, cam01_pickle_name = os.path.split(cam01_pickle)
        print('matching points for {}'.format(cam01_pickle_name))

        cam_pickle_files, both_files_exist = navigation_utilities.find_other_optitrack_pickles(cam01_pickle, parent_directories)

        if not both_files_exist:
            continue
        # cam_pickle_files = []
        # # find corresponding pickle file for camera 2
        # session_folder, _ = os.path.split(cam01_folder)
        # cam01_pickle_stem = cam01_pickle_name[:cam01_pickle_name.find('cam01') + 5]
        # cam02_pickle_stem = cam01_pickle_stem.replace('cam01', 'cam02')
        #
        # cam02_pickle = [c02_pickle for c02_pickle in cam_pickles[1] if cam02_pickle_stem in c02_pickle]
        #
        # # cam02_file_list = glob.glob(os.path.join(view_directories[1], cam02_pickle_stem + '*full.pickle'))
        # if len(cam02_pickle) == 1:
        #     cam02_pickle = cam02_pickle[0]
        #     cam_pickle_files.append(cam01_pickle)
        #     cam_pickle_files.append(cam02_pickle)
        # else:
        #     print('no matching camera 2 file for {}'.format(cam01_file))
        #     continue

        cam01_names.append(cam_pickle_files[0])

        pickle_metadata = [navigation_utilities.parse_dlc_output_pickle_name_optitrack(pickle_name) for pickle_name in cam_pickle_files]

        # read in pickle data from both files
        single_trial_dlc_output = [skilled_reaching_io.read_pickle(pickle_name) for pickle_name in cam_pickle_files]
        cam_meta_files = [pickle_file.replace('full.pickle', 'meta.pickle') for pickle_file in cam_pickle_files]
        dlc_metadata = [skilled_reaching_io.read_pickle(cam_meta_file) for cam_meta_file in cam_meta_files]

        matched_trial_pts, matched_pts_conf = match_trial_points(single_trial_dlc_output, pickle_metadata, dlc_metadata)

        matched_dlc_points.append(matched_trial_pts)
        matched_dlc_conf.append(matched_pts_conf)

        # if 'matched_dlc_points' in locals():
        #     matched_dlc_points = [np.vstack(matched_dlc_points[i_cam], matched_trial_pts[i_cam]) for i_cam in
        #                           range(num_cams)]
        # else:
        #     matched_dlc_points = matched_trial_pts

    # matched_dlc_points is a list containing num_trials_to_match 2-element lists (one for each camera) of
    # num_frames x num_joints x 2 arrays containing matched points from the two videos
    # matched_dlc_conf is a list containing num_trials_to_match lists of 2-element lists (one for each camera) of
    # num_frames x num_joints arrays containing confidence levels
    # cam01_names is a num_trials_to_match element list of pickle file names from which dlc points were extracted
    return matched_dlc_points, matched_dlc_conf, cam01_names
    '''
    findEssentialMat or findFundamentalMat need matched points in the two images; arrays are points 
    '''
    # for i_cam, pickle_name in enumerate(cam_pickle_files):
    #     single_trial_dlc.append(skilled_reaching_io.read_pickle(pickle_name))

    # pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam02_file))


def trialpts2allpts(trials_pts, trials_conf, min_conf):
    '''
    take points from all frames for a trial and concatenate them into a single array for stereo calibration
    :param trials_pts:
    :param trials_conf:
    :param min_conf:
    :return:
    '''

    num_trials = len(trials_pts)
    num_cams = np.shape(trials_pts[0])[0]
    frames_per_trial = [[] for ii in range(num_cams)]
    for i_cam in range(num_cams):
        frames_per_trial[i_cam].append([np.shape(trials_pts[i_trial][i_cam])[0] for i_trial in range(num_trials)])
    frame_per_trial = np.squeeze(np.array(frames_per_trial))
    frame_per_trial = frame_per_trial.T
    # num_joints = np.shape(trials_pts[0][0])[1]

    all_pts = [[] for ii in range(num_cams)]
    all_conf = [[] for ii in range(num_cams)]
    for i_trial in range(num_trials):
        if num_trials == 1:
            current_frames_to_match = min(frame_per_trial)
        else:
            current_frames_to_match = min(frame_per_trial[i_trial, :])
        for i_cam in range(num_cams):
            if i_trial == 0:
                all_pts[i_cam] = np.reshape(trials_pts[i_trial][i_cam][:current_frames_to_match, :, :], (-1, 2))
                all_conf[i_cam] = np.reshape(trials_conf[i_trial][i_cam][:current_frames_to_match], (-1))
            else:
                all_pts[i_cam] = np.vstack((all_pts[i_cam], np.reshape(trials_pts[i_trial][i_cam][:current_frames_to_match, :], (-1, 2))))
                all_conf[i_cam] = np.hstack((all_conf[i_cam], np.reshape(trials_conf[i_trial][i_cam][:current_frames_to_match], (-1))))

    all_conf = np.array(all_conf).T   #np.hstack((all_conf[0].T, all_conf[0].T))


    # only consider points where confidence > threshold for both cameras
    valid_pts_bool = (all_conf > min_conf).all(axis=1)

    return all_pts, all_conf, valid_pts_bool


def match_trial_points(dlc_output, pickle_metadata, dlc_metadata, min_conf=0.98):

    if len(dlc_output) != len(pickle_metadata):
        print('each camera view does not have a .pickle file')
        return

    num_cams = len(dlc_output)

    pts_wrt_orig_img, dlc_conf = reconstruct_3d_optitrack.rotate_translate_optitrack_points(dlc_output, pickle_metadata, dlc_metadata)

    # now have all the identified points moved back into the original coordinate systems that the checkerboards were
    # identified in, and confidence levels. pts_wrt_orig_img is an array (num_frames x num_joints x 2) and dlc_conf
    # is an array (num_frames x num_joints). Zeros are stored where dlc was uncertain (no result for that joint on
    # that frame)

    # reshape arrays so that they are a long list of individual points
    num_frames = np.shape(pts_wrt_orig_img[0])[0]
    num_joints = np.shape(pts_wrt_orig_img[0])[1]
    total_pts = num_frames * num_joints

    # all_pts = [np.reshape(img_pts, (total_pts, 2)) for img_pts in pts_wrt_orig_img]
    # all_conf = [np.reshape(cam_conf, (total_pts, 1)) for cam_conf in dlc_conf]
    # all_conf = np.hstack(all_conf)
    #
    # # only consider points where confidence > threshold for both cameras
    # valid_pts_bool = (all_conf > min_conf).all(axis=1)
    #
    # valid_pts = [cam_pts[valid_pts_bool] for cam_pts in all_pts]

    return pts_wrt_orig_img, dlc_conf


def estimate_E_from_dlc(single_trial_dlc_output, cal_data):
    pass

def refine_calibrations_from_orig_vids(vid_folder_list, parent_directories):
    # this doesn't seem to be working so well. Perhaps better to refine from matched points identified by DLC
    cal_data_parent = parent_directories['cal_data_parent']

    for vf in vid_folder_list:

        mouseID, session_date_str = navigation_utilities.parse_session_dir_name(vf)
        session_date = navigation_utilities.fname_string_to_date(session_date_str)
        calibration_file = navigation_utilities.find_optitrack_calibration_data_name(cal_data_parent, session_date)

        if calibration_file is None or not os.path.exists(calibration_file):
            # if there is no calibration file for this session, skip
            continue
        cal_data = skilled_reaching_io.read_pickle(calibration_file)

        session_nums = navigation_utilities.sessions_in_optitrack_folder(vf)
        # find a pair of videos from each session
        for sn in session_nums:
            vid_pair = navigation_utilities.find_vid_pair_from_session(vf, sn)
            matched_frames = load_vidpair_frames(vid_pair)
            match_points(matched_frames, cal_data)
            pass

        pass
    pass


def match_points(frame_pair, cal_data):
    '''

    :param frame_pair: note that frame_pair should be such that the first image in the list is from camera 1, the second
        image in the list is from camera 2
    :param cal_data:
    :return:
    '''

    if cal_data['calvid_metadata'][0]['cam_num'] == 1:
        # the first elements of lists in the cal_data dictionary is for camera 1
        cam_list = [0, 1]
    else:
        # the first elements of lists in the cal_data dictionary is for camera 2
        cam_list = [1, 0]
    mtx = []
    dist = []
    for i_cam in cam_list:
        mtx.append(cal_data['mtx'][i_cam])
        dist.append(cal_data['dist'][i_cam])

    # first, undistort the images, find the keypoints
    im_ud = []
    sift = cv2.SIFT_create()
    kp = []
    des = []
    for i_cam, mf in enumerate(frame_pair):
        im_ud.append(cv2.undistort(mf, mtx[i_cam], dist[i_cam]))
        a, b = sift.detectAndCompute(im_ud[i_cam], None)
        kp.append(a)
        des.append(b)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des[0], des[1], k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des[0], des[1], k=2)

    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append([m])

    # for BF matching
    # img3 = cv2.drawMatchesKnn(im_ud[0], kp[0], im_ud[1], kp[1], good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # for FLANN matching
    img3 = cv2.drawMatchesKnn(im_ud[0], kp[0], im_ud[1], kp[1], matches, None, **draw_params)

    plt.figure()
    plt.imshow(img3)
    plt.show()


    pass


def load_vidpair_frames(vid_pair):

    # assume camera 1 should be rotated 180 degrees
    img = []
    for vid in vid_pair:
        video_object = cv2.VideoCapture(vid)
        ret, cur_img = video_object.read()

        if ret:
            vid_metadata = navigation_utilities.parse_Burgess_vid_name(vid)
            if vid_metadata['cam_num'] == 1:
                cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)

            img.append(cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY))

    return img

def import_fiji_csv(fname):
    """
    read csv file with points marked in fiji
    :param fname:
    :return:
    """
    # for checkerboards marked manually in fiji PRIOR TO undistorting

    # if file is empty, return empty matrix
    if os.path.getsize(fname) == 0:
        return np.empty(0)

    # determine how many lines are in the .csv file
    with open(fname, newline='\n') as csv_file:
        num_lines = sum(1 for row in csv_file)

    with open(fname, newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        cb_points = np.empty((num_lines-1, 2))
        for i_row, row in enumerate(csv_reader):
            if i_row == 0:
                # check to make sure the header was read in properly
                print(f'Column names are {", ".join(row)}')
            else:
                # read in the rows of checkerboard points
                cb_points[i_row-1, :] = row[-2:]

    return cb_points


def read_cube_calibration_points(calibration_folder, pts_per_board=12):
    """

    :param calibration_folder:
    :return:
    """

    calibration_files = glob.glob(os.path.join(calibration_folder, 'GridCalibration_*.csv'))
    # for each file, there should be 72 total points; 12 per checkerboard
    num_files = len(calibration_files)
    direct_points = np.empty(pts_per_board, 2, 2, num_files)  # num_pts x 2 dims (x,y) x 2 boards (left,right) x num_files
    mirror_points = np.empty(pts_per_board, 2, 2, num_files)  # num_pts x 2 dims (x,y) x 2 boards (left,right) x num_files
    for calib_file in calibration_files:
        cb_pts = import_fiji_csv(calib_file)

        # still working on this, but probably will never actually need it


def sort_points_to_boards(cb_pts):
    """

    :param cb_pts: n x 2 numpy array containing (distorted) checkerboard points from the calibration cubes
    :return:
    """
    #todo: can probably eliminate this - relic from when we were calibrating with cubes instead of videos of checkerboards in a single plane


def fundamental_matrix_from_mirrors(x1, x2):
    """
    function to compute the fundamental matrix for direct camera and mirror image views, taking advantage of the fact
    that F is skew-symmetric in this case. Note x1 and x2 should be undistorted by this point

    :param x1: n x 2 numpy array containing matched points in order from view 1
    :param x2: n x 2 numpy array containing matched points in order from view 2
    :return:
    """
    n1, numcols = x1.shape()
    if numcols != 2:
        print('x1 must have 2 columns')
        return
        #todo: error handling here
    n2, numcols = x2.shape()
    if numcols != 2:
        print('x2 must have 2 columns')
        return
    if n1 != n2:
        print('x1 and x2 must have same number of rows')
        return

    A = np.zeros((n1, 3))
    A[:, 0] = np.multiply(x2[:, 0], x1[:, 1]) - np.multiply(x1[:, 0], x2[:, 1])
    A[:, 1] = x2[:, 0] - x1[:, 0]
    A[:, 2] = x2[:,1] - x1[:, 1]

    # solve the linear system of equations A * [f12,f13,f23]' = 0
    # need to figure out if the matrix needs to be changed using opencv conventions instead of matlab
    _, _, vA = np.linalg.svd(A)
    F = np.zeros((3, 3))
    fvec = vA[:, -1]

    F[0, 1] = fvec[0]
    F[0, 2] = fvec[1]
    F[1, 2] = fvec[2]
    F[1, 0] = -F[0, 1]
    F[2, 0] = -F[0, 2]
    F[2, 1] = -F[1, 2]

    return F


def select_correct_essential_matrix():

    pass


def calibrate_camera_from_video(camera_calibration_vid_name, calibration_parent, cb_size=(6, 9)):
    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_ASPECT_RATIO

    video_object = cv2.VideoCapture(camera_calibration_vid_name)

    im_size = (int(video_object.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbrow = cb_size[0]
    cbcol = cb_size[1]
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    frame_counter = 0
    while True:
        video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        ret, cur_img = video_object.read()

        if ret:
            frame_counter += 1

            cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            found_valid_chessboard, corners = cv2.findChessboardCorners(cur_img_gray, cb_size)

            if found_valid_chessboard:
                # refine the points, then save the checkerboard image and a metadata file to disk
                corners2 = cv2.cornerSubPix(cur_img_gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                objpoints.append(objp)
            # else:
            # this is for showing the corners that weren't properly identified if the plotting lines below are commented in
            #     corners2 = corners

            # below is for checking if corners were correctly identified
            # corners_img = cv2.drawChessboardCorners(cur_img, cb_size, corners2, found_valid_chessboard)
            # # cv2.imwrite(corners_img_name, corners_img)
            # cv2.imshow('image', corners_img)
            # cv2.waitKey(0)

        else:
            # finished with the last frame
            break

    video_object.release()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, im_size, None, None)

    stereo_params = {
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'im_size': im_size
    }

    calibration_metadata = navigation_utilities.parse_camera_calibration_video_name(camera_calibration_vid_name)

    calibration_name = navigation_utilities.create_calibration_filename(calibration_metadata)
    calibration_folder = create_calibration_file_folder_name(calibration_metadata, calibration_parent)
    if not os.path.exists(calibration_folder):
        os.makedirs(calibration_folder)
    full_calibration_name = os.path.join(calibration_folder, calibration_name)

    skilled_reaching_io.write_pickle(full_calibration_name, stereo_params)


def multi_view_calibration(calibration_vids, cal_data_parent, cb_size=(10, 7)):

    cam_cal_flags = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST
    stereo_cal_flags = cv2.CALIB_FIX_INTRINSIC

    vid_name_parts = navigation_utilities.parse_Burgess_calibration_vid_name(calibration_vids[0])
    cal_data_name = navigation_utilities.create_multiview_calibration_data_name(cal_data_parent,
                                                                                vid_name_parts['session_datetime'])

    if not os.path.exists(cal_data_name):
        calibration_data = collect_cb_corners(calibration_vids, cb_size=cb_size)
        # save the imgpoints and objpoints along with metadata
        skilled_reaching_io.write_pickle(cal_data_name, calibration_data)
    else:
        calibration_data = skilled_reaching_io.read_pickle(cal_data_name)


    # if calibration already performed for individual cameras (at least one), skip initializing variables
    # or better yet, let's calibrate the two cameras once well, then just to the stereo calibration on subsequent days

    if 'mtx' not in calibration_data.keys():
        # first, calibrate each camera individually
        # initialize arrays to hold camera intrinsics, distortion coefficients, rotation, translation vectors
        mtx = [[] for ii in calibration_data['cam_objpoints']]
        dist = [[] for ii in calibration_data['cam_objpoints']]
        rvecs = [[] for ii in calibration_data['cam_objpoints']]
        tvecs = [[] for ii in calibration_data['cam_objpoints']]

        for i_cam, objpoints in enumerate(calibration_data['cam_objpoints']):
            # intrinsics haven't been calculated yet for either camera
            imgpoints = calibration_data['cam_imgpoints'][i_cam]
            im_size = calibration_data['im_size'][i_cam]

            print('calibrating camera {:02d}'.format(i_cam + 1))
            ret, mtx[i_cam], dist[i_cam], rvecs[i_cam], tvecs[i_cam] = cv2.calibrateCamera(objpoints[:10], imgpoints[:10], im_size, None, None,
                                                                                           flags=cam_cal_flags)
            calibration_data['mtx'] = mtx
            calibration_data['dist'] = dist
            calibration_data['rvecs'] = rvecs
            calibration_data['tvecs'] = tvecs

            skilled_reaching_io.write_pickle(cal_data_name, calibration_data)
    else:
        for i_cam, objpoints in enumerate(calibration_data['cam_objpoints']):

            # has mtx already been calculated for this camera? it's possible that an empty intrinsics was saved on a previous run
            if len(calibration_data['mtx'][i_cam]) == 0:
                # the intrinsic matrix has not yet been calculated for i_cam
                imgpoints = calibration_data['cam_imgpoints'][i_cam]
                im_size = calibration_data['im_size'][i_cam]

                print('calibrating camera {:02d}'.format(i_cam + 1))
                ret, mtx[i_cam], dist[i_cam], rvecs[i_cam], tvecs[i_cam] = cv2.calibrateCamera(objpoints[:10], imgpoints[:10], im_size, None, None,
                                                                                               flags=cam_cal_flags)
                calibration_data['mtx'] = mtx
                calibration_data['dist'] = dist
                calibration_data['rvecs'] = rvecs
                calibration_data['tvecs'] = tvecs

                skilled_reaching_io.write_pickle(cal_data_name, calibration_data)

    objpoints = calibration_data['stereo_objpoints']
    imgpoints = calibration_data['stereo_imgpoints']

    print('performing stereo calibration')
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints[0], imgpoints[1],
                                                                                                     calibration_data['mtx'][0], calibration_data['dist'][0],
                                                                                                     calibration_data['mtx'][1], calibration_data['dist'][1],
                                                                                                     im_size,
                                                                                                     flags=stereo_cal_flags)
    pass


def collect_cb_corners(calibration_vids, cb_size):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    video_object = []
    num_frames = []
    im_size = []
    cal_vid_metadata = []
    for i_vid, cal_vid in enumerate(calibration_vids):
        video_object.append(cv2.VideoCapture(cal_vid))
        num_frames.append(video_object[i_vid].get(cv2.CAP_PROP_FRAME_COUNT))
        im_size.append((int(video_object[i_vid].get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_object[i_vid].get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cal_vid_metadata.append(navigation_utilities.parse_cropped_calibration_video_name(cal_vid))

    num_frames = [int(nf) for nf in num_frames]
    if all(nf == num_frames[0] for nf in num_frames):
        # check that there are the same number of frames in each calibration video

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        cbrow = cb_size[0]
        cbcol = cb_size[1]
        objp = np.zeros((cbrow * cbcol, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

        cam_objpoints = [[] for ii in calibration_vids]
        stereo_objpoints = []
        cam_imgpoints = [[] for ii in calibration_vids]
        stereo_imgpoints = [[] for ii in calibration_vids]

        valid_frames = [[False for frame_num in range(num_frames[0])] for ii in calibration_vids]

        for i_frame in range(num_frames[0]):
            print(i_frame)

            corners2 = [[] for ii in calibration_vids]
            cur_img = [[] for ii in calibration_vids]
            for i_vid, vid_obj in enumerate(video_object):
                vid_obj.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
                ret, cur_img[i_vid] = vid_obj.read()

                if ret:
                    cur_img_gray = cv2.cvtColor(cur_img[i_vid], cv2.COLOR_BGR2GRAY)
                    found_valid_chessboard, corners = cv2.findChessboardCorners(cur_img_gray, cb_size)
                    valid_frames[i_vid][i_frame] = found_valid_chessboard

                    if found_valid_chessboard:
                        corners2[i_vid] = cv2.cornerSubPix(cur_img_gray, corners, (11, 11), (-1, -1), criteria)
                        cam_objpoints[i_vid].append(objp)
                        cam_imgpoints[i_vid].append(corners2[i_vid])

                    # corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners,
                    #                                         found_valid_chessboard)
                    # vid_path, vid_name = os.path.split(calibration_vids[i_vid])
                    # vid_name, _ = os.path.splitext(vid_name)
                    # frame_path = os.path.join(vid_path, vid_name)
                    # if not os.path.isdir(frame_path):
                    #     os.makedirs(frame_path)
                    # frame_name = vid_name + '_frame{:03d}'.format(i_frame) + '.png'
                    # full_frame_name = os.path.join(frame_path, frame_name)
                    # cv2.imwrite(full_frame_name, corners_img)

            # collect all checkerboard points visible in pairs of images
            if valid_frames[0][i_frame] and valid_frames[1][i_frame]:
                # checkerboards were identified in matching frames
                stereo_objpoints.append(objp)
                for i_vid, corner_pts in enumerate(corners2):
                    stereo_imgpoints[i_vid].append(corner_pts)

            # todo: test that the checkerboard points in each imgpoint array are correct
            # below is for checking if corners were correctly identified
            # for i_vid in range(3):
            #     if valid_frames[i_vid][i_frame]:
            #         corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners2[i_vid], found_valid_chessboard)
                    # cv2.imwrite(corners_img_name, corners_img)
                    # plt.imshow(corners_img)
                    # plt.show()

        calibration_data = {
            'cam_objpoints': cam_objpoints,
            'cam_imgpoints': cam_imgpoints,
            'stereo_objpoints': stereo_objpoints,
            'stereo_imgpoints': stereo_imgpoints,
            'valid_frames': valid_frames,
            'im_size': im_size,
            'cb_size': cb_size,
            'cropped_vid_metadata': cal_vid_metadata
        }

        for vid_obj in video_object:
            vid_obj.release()

        return calibration_data

    else:
        # calibration videos have different numbers of frames, so not clear how to line them up

        return none


def verify_checkerboard_points(calibration_vids, calibration_data):

    for i_vid, cal_vid in enumerate(calibration_vids):

        vid_obj = cv2.VideoCapture(cal_vid)
        num_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        valid_frame_counter = 0
        for i_frame in range(num_frames):

            if calibration_data['valid_frames'][i_vid][i_frame]:

                vid_obj.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
                ret, cur_img = vid_obj.read()

                corners = calibration_data['cam_imgpoints'][i_vid][valid_frame_counter]
                valid_frame_counter += 1

                corners_img = cv2.drawChessboardCorners(cur_img, calibration_data['cb_size'], corners, calibration_data['valid_frames'][i_vid][i_frame])
                cv2.imshow('image', corners_img)
                cv2.waitKey(0)

        vid_obj.release()


def crop_params_dict_from_sessionrow(session_row, view_list=['dir', 'lm', 'rm']):
    crop_params_dict = dict.fromkeys(view_list, None)
    for view in view_list:
        left_edge = session_row[view + '_left'].values[0]
        right_edge = session_row[view + '_right'].values[0]
        top_edge = session_row[view + '_top'].values[0]
        bot_edge = session_row[view + '_bottom'].values[0]

        if any([pd.isna(left_edge), pd.isna(right_edge), pd.isna(top_edge), pd.isna(bot_edge)]):
            crop_params_dict = {}
            break
        else:
            crop_params_dict[view] = [left_edge,
                                      right_edge,
                                      top_edge,
                                      bot_edge]

    return crop_params_dict


def crop_params_dict_from_ratcal_metadata(cal_vid_path, ratcal_metadata, view_list=['direct', 'leftmirror', 'rightmirror']):

    # find the calibration video name in the table of sessions
    ratIDs = list(ratcal_metadata.keys())
    _, cal_vid_name = os.path.split(cal_vid_path)

    crop_params_dict = None

    for ratID in ratIDs:
        # loop through each rat database. If the calibration video is used for this rat, use the same cropping coordinates
        # as we will use for the rat videos for this session
        current_df = ratcal_metadata[ratID]

        session_row = current_df[(current_df['cal_vid_name_mirrors'] == cal_vid_name)]

        if session_row.shape[0] == 1:
            crop_params_dict = dict.fromkeys(view_list, None)
            for view in view_list:
                left_edge = session_row[view + '_left'].values[0]
                right_edge = session_row[view + '_right'].values[0]
                top_edge = session_row[view + '_top'].values[0]
                bot_edge = session_row[view + '_bottom'].values[0]

                if any([pd.isna(left_edge), pd.isna(right_edge), pd.isna(top_edge), pd.isna(bot_edge)]):
                    crop_params_dict = {}
                    break
                else:
                    crop_params_dict[view] = [left_edge,
                                              right_edge,
                                              top_edge,
                                              bot_edge]

    return crop_params_dict


def camera_board_from_df(session_row):

    square_length = float(session_row['square_length_camera'].values[0])
    nrows = session_row['nrows_camera'].values[0]
    ncols = session_row['ncols_camera'].values[0]

    board_type = session_row['board_type_camera'].values[0]

    if 'charuco' in board_type.lower():
        marker_length = float(session_row['marker_length_camera'].values[0])
        marker_bits, dict_size = dict_from_boardtype(board_type)
        cam_board = create_charuco(nrows, ncols, square_length, marker_length, marker_bits=marker_bits, dict_size=dict_size)
    elif board_type.lower() in ['chessboard', 'checkerboard']:
        cam_board = create_checkerboard(nrows, ncols, square_length)

    return cam_board


def dict_from_boardtype(board_type):
    '''

    :param board_type: string of form 'charuco_dict' where 'dict' is of the form (for example) '4X4_dictsize' where dictsize is 50, 250, etc.
    :return:
    '''

    btype_parts = board_type.split('_')

    marker_bits = int(btype_parts[1][0])
    dict_size = int(btype_parts[2])

    return marker_bits, dict_size


def mirror_board_from_df(session_row):

    square_length = float(session_row['square_length_mirrors'].values[0])
    nrows = session_row['nrows_mirrors'].values[0]
    ncols = session_row['ncols_mirrors'].values[0]

    board_type = session_row['board_type_mirrors'].values[0]

    if 'charuco' in board_type.lower():
        marker_length = float(session_row['marker_length_mirrors'].values[0])
        marker_bits, dict_size = dict_from_boardtype(board_type)
        mirror_board = create_charuco(nrows, ncols, square_length, marker_length, marker_bits=marker_bits,
                               dict_size=dict_size)

    elif board_type.lower() in ['chessboard', 'checkerboard']:
        mirror_board = create_checkerboard(nrows, ncols, square_length)

    # test to make sure this board matches the calibration board


    return mirror_board



def create_charuco(squaresX, squaresY, square_length, marker_length, marker_bits=4, dict_size=50, aruco_dict=None, manually_verify=False):

    try:
        board = CharucoBoard(int(squaresX), int(squaresY), square_length, marker_length,
                             marker_bits=marker_bits,
                             dict_size=dict_size,
                             aruco_dict=aruco_dict,
                             manually_verify=manually_verify)
    except:
        pass

    # just to test that the board really looks like the board used for calibration
    # img = board.board.generateImage((600, 600))
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.imshow(img)
    # plt.show()



    return board


def create_checkerboard(squaresX, squaresY, square_length, manually_verify=False):

    board = Checkerboard(squaresX, squaresY, square_length, manually_verify=manually_verify)

    return board


def write_board_image(board, dpi, calib_dir, units='mm'):
    if type(board) is Checkerboard:
        # write_checkerboard_image(board, dpi, calib_dir, units=units)
        pass
    elif type(board) is CharucoBoard:
        write_charuco_image(board, dpi, calib_dir, units=units)

def write_checkerboard_image(board, dpi, calib_dir, units='mm'):
    #todo: will have to build this, but for now just skipping it
    x_total = int(board.squaresX * board.square_length)
    y_total = int(board.squaresY * board.square_length)

    fname = '_'.join(['checkerboard',
                      '{:d}x{:d}'.format(y_total, x_total),
                      '{:d}x{:d}'.format(board.squaresY, board.squaresX),
                      '{:d}'.format(int(board.square_length)),
                      '{:d}dpi.tiff'.format(dpi)])
    fname = os.path.join(calib_dir, fname)

    if units.lower() == 'mm':
        cf = 25.4
    elif units.lower() == 'cm':
        cf = 2.54
    elif units.lower() in ['inch', 'inches']:
        cf = 1.

    xpixels = round(dpi * x_total / cf)
    ypixels = round(dpi * y_total / cf)

    img = board.board.generateImage((xpixels,ypixels))

    cv2.imwrite(fname, img)


def write_charuco_image(board, dpi, calib_dir, units='mm'):

    x_total = int(board.squaresX * board.square_length)
    y_total = int(board.squaresY * board.square_length)

    board_dict = board.board.getDictionary()
    dict_size = board_dict.bytesList.shape[0]
    marker_bits = board_dict.markerSize

    fname = '_'.join(['charuco',
                      '{:d}x{:d}'.format(y_total, x_total),
                      '{:d}x{:d}'.format(board.squaresY, board.squaresX),
                      '{:d}'.format(int(board.square_length)),
                      '{:d}'.format(int(board.marker_length)),
                      'DICT',
                      '{:d}x{:d}'.format(marker_bits, marker_bits),
                      '{:d}'.format(dict_size),
                      '{:d}dpi.tiff'.format(dpi)])
    fname = os.path.join(calib_dir, fname)

    if units.lower() == 'mm':
        cf = 25.4
    elif units.lower() == 'cm':
        cf = 2.54
    elif units.lower() in ['inch', 'inches']:
        cf = 1.

    xpixels = round(dpi * x_total / cf)
    ypixels = round(dpi * y_total / cf)

    img = board.board.generateImage((xpixels,ypixels))

    cv2.imwrite(fname, img)


def crop_calibration_video(calib_vid,
                           session_row,
                           calib_crop_top=100,
                           filtertype='',
                           view_list=['dir', 'lm', 'rm']):

    cc_metadata = navigation_utilities.parse_camera_calibration_video_name(calib_vid)

    session_date = cc_metadata['time'].date()

    crop_params_dict = crop_params_dict_from_sessionrow(session_row, view_list=view_list)

    if crop_params_dict is None:
        return None

    cropped_vid_names = []
    if crop_params_dict:
        # make sure there is plenty of height. crop_params_dict should have keys 'direct','leftmirror','rightmirror'
        # top of cropping window is probably too low for the calibration videos if based on the reaching videos

        calibration_crop_params_dict = crop_params_dict

        for key in calibration_crop_params_dict:
            if 'mirror' in key:
                # if this is one of the mirror views, the cropped video should be flipped left to right
                fliplr = False
                # I think we don't want to flip them for now, will flip them left to right after undistortion
            else:
                fliplr = False
            # calibration_crop_params_dict[key][2] = calib_crop_top

            full_cropped_vid_name = navigation_utilities.create_cropped_calib_vid_name(calib_vid, key, calibration_crop_params_dict, fliplr)
            cropped_vid_names.append(full_cropped_vid_name)
            if os.path.isfile(full_cropped_vid_name):
                # skip if already cropped
                continue

            crop_videos.crop_video(calib_vid,
                                   full_cropped_vid_name,
                                   calibration_crop_params_dict[key],
                                   key,
                                   filtertype=filtertype,
                                   fliplr=fliplr)

    return cropped_vid_names


def multi_mirror_calibration(calibration_data, calibration_summary_name):
    '''
    calibrate across multiple views
    :param calibration_data:
    :return:
    '''

    # initialize camera matrices for direct-->leftmirror, direct-->rightmirror, and leftmirror-->rightmirror
    # todo: initialize camera/fundamental/essential matrices for each set of views and prove that we can reconstruct the
    # chessboard images
    keys = calibration_data.keys()
    if 'camera_intrinsics' not in keys:
        calibration_data = camera_calibration_from_mirror_vids(calibration_data, calibration_summary_name)
    pass


def calibrate_all_Burgess_vids(parent_directories, cb_size=(7, 10), checkerboard_square_size=7.):
    '''
    perform calibration for all checkerboard videos stored in appropriate directory structure under cal_vid_parent.
    Write results into directory structure under cal_vid_data
    :param cal_vid_parent: parent directory for calibration videos. Directory structure should be:
        cal_vid_parent-->
    :param cal_data_parent: parent directory for calibration data extracted from calibration videos. Directory structure should be:
        cal_data_parent
    :param cb_size:
    :param checkerboard_square_size:
    :return:
    '''

    cal_vids_parent = parent_directories['cal_vids_parent']
    cal_data_parent = parent_directories['cal_data_parent']

    paired_cal_vids = navigation_utilities.find_Burgess_calibration_vids(cal_vids_parent)
    '''
    note that find_Burgess_calibration_vids relies on glob, which may not return file names in alphabetical order. This 
    is important because the calibration videos for different cameras may show up in different orders (i.e., calibration
    video for camera 2 may appear in the list before camera 1. This needs to be fixed so that the camera matrices,
    including extrinsics for multi-views are in the right order. That is, R and T should always be rotation and
    translation of camera 2 with respect to camera 1, not the other way around in some instances
    '''
    for vid_pair in paired_cal_vids:

        sorted_vid_pair = navigation_utilities.sort_optitrack_calibration_vid_names_by_camera_number(vid_pair)
        calvid_metadata = [navigation_utilities.parse_Burgess_calibration_vid_name(vid) for vid in sorted_vid_pair]
        # sort vid_pair so that file names are in order of camera number

        cal_data_name = navigation_utilities.create_optitrack_calibration_data_name(cal_data_parent,
                                                                                    calvid_metadata[0]['session_datetime'])
        # comment back in when done testing
        if not os.path.isfile(cal_data_name):
            # collect the checkerboard points, write to file
            collect_cbpoints_Burgess(sorted_vid_pair, cal_data_parent, cb_size=cb_size, checkerboard_square_size=checkerboard_square_size)

        calibrate_Burgess_session(cal_data_name, sorted_vid_pair, parent_directories)


def collect_cbpoints_Burgess(vid_pair, cal_data_parent, cb_size=(7, 10), checkerboard_square_size=7.):
    '''

    :param vid_pair:
    :param cal_data_parent: parent directory for folders containing pickle files with calibration results. Has structure:
        cal_data_parent-->calibration_data_YYYY-->calibration_data_YYYYmm
    :param cb_size:
    :param checkerboard_square_size:
    :return:
    '''

    # extract metadata from file names. Note that cam 01 is upside down

    calvid_metadata = [navigation_utilities.parse_Burgess_calibration_vid_name(vid) for vid in vid_pair]
    cal_data_name = navigation_utilities.create_optitrack_calibration_data_name(cal_data_parent,
                                                                                calvid_metadata[0]['session_datetime'])
    # if os.path.isfile(cal_data_name):
    #     # if file already exists, assume cb points have already been collected
    #     return

        # camera calibrations have been performed, now need to do stereo calibration

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cb_search_window = (5, 5)
    # CBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS

    # create video objects for each calibration_video
    vid_obj = []
    num_frames = []
    im_size = []
    vid_root_names = []
    for i_vid, vid_name in enumerate(vid_pair):
        _, cur_root_name = os.path.split(vid_name)
        vid_root_names.append(cur_root_name)
        vid_obj.append(cv2.VideoCapture(vid_name))
        num_frames.append(int(vid_obj[i_vid].get(cv2.CAP_PROP_FRAME_COUNT)))
        im_size.append((int(vid_obj[i_vid].get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_obj[i_vid].get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if all(nf == num_frames[0] for nf in num_frames):
        # check that there are the same number of frames in each calibration video

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        cbrow = cb_size[0]
        cbcol = cb_size[1]
        objp = np.zeros((cbrow * cbcol, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2)

        cam_objpoints = [[] for ii in vid_pair]
        stereo_objpoints = []
        cam_imgpoints = [[] for ii in vid_pair]
        stereo_imgpoints = [[] for ii in vid_pair]

    # create boolean lists to track which frames have valid checkerboard images. Writing so that can expand to more
    # views eventually
    # valid_frames is a list of length num_cameras of lists that each are of length num_frames
    valid_frames = [[False for frame_num in range(num_frames[0])] for ii in vid_pair]
    stereo_frames = []
    for i_frame in range(num_frames[0]):
        print('frame number: {:04d} for {} and {}'.format(i_frame, vid_root_names[0], vid_root_names[1]))

        corners2 = [[] for ii in vid_pair]
        cur_img = [[] for ii in vid_pair]
        for i_vid, vo in enumerate(vid_obj):
            vo.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            ret, cur_img[i_vid] = vo.read()

            # test_img_name = os.path.join(cal_data_parent, 'test_cam{:02d}.jpg'.format(calibration_metadata[i_vid]['cam_num']))
            # cv2.imwrite(test_img_name, cur_img[i_vid])

            if ret:
                if calvid_metadata[i_vid]['cam_num'] == 1:
                    # rotate the image 180 degrees
                    cur_img[i_vid] = cv2.rotate(cur_img[i_vid], cv2.ROTATE_180)

                cur_img_gray = cv2.cvtColor(cur_img[i_vid], cv2.COLOR_BGR2GRAY)
                # NOTE, tried using several flag options for finChessboardCorners, but didn't help with detection
                found_valid_chessboard, corners = cv2.findChessboardCorners(cur_img_gray, cb_size)
                valid_frames[i_vid][i_frame] = found_valid_chessboard

                if found_valid_chessboard:
                    corners2[i_vid] = cv2.cornerSubPix(cur_img_gray, corners, cb_search_window, (-1, -1), criteria)
                    cam_objpoints[i_vid].append(objp)
                    cam_imgpoints[i_vid].append(corners2[i_vid])

                    corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners2[i_vid],
                                                            found_valid_chessboard)

                else:
                    corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners,
                                                            found_valid_chessboard)
                # vid_path, vid_name = os.path.split(calibration_vids[i_vid])
                # vid_name, _ = os.path.splitext(vid_name)
                # frame_path = os.path.join(vid_path, vid_name)
                # if not os.path.isdir(frame_path):
                #     os.makedirs(frame_path)

                session_date_string = navigation_utilities.datetime_to_string_for_fname(calvid_metadata[i_vid]['session_datetime'])
                test_save_dir = os.path.join(cal_data_parent, 'corner_images', session_date_string, 'cam{:02d}'.format(calvid_metadata[i_vid]['cam_num']))
                if not os.path.isdir(test_save_dir):
                    os.makedirs(test_save_dir)

                test_img_name = os.path.join(test_save_dir,
                                             'test_cboard_{}_cam{:02d}_frame{:04d}.jpg'.format(session_date_string, calvid_metadata[i_vid]['cam_num'], i_frame))
                # frame_name = vid_name + '_frame{:03d}'.format(i_frame) + '.png'

                cv2.imwrite(test_img_name, corners_img)

        # collect all checkerboard points visible in pairs of images
        if valid_frames[0][i_frame] and valid_frames[1][i_frame]:
            # checkerboards were identified in matching frames
            stereo_objpoints.append(objp)
            stereo_frames.append(i_frame)

            for i_vid, corner_pts in enumerate(corners2):
                stereo_imgpoints[i_vid].append(corner_pts)

        # todo: test that the checkerboard points in each imgpoint array are correct
        # below is for checking if corners were correctly identified
        # for i_vid in range(3):
        #     if valid_frames[i_vid][i_frame]:
        #         corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners2[i_vid], found_valid_chessboard)
        # cv2.imwrite(corners_img_name, corners_img)
        # plt.imshow(corners_img)
        # plt.show()

    calibration_data = {
        'cam_objpoints': cam_objpoints,
        'cam_imgpoints': cam_imgpoints,
        'stereo_objpoints': stereo_objpoints,
        'stereo_imgpoints': stereo_imgpoints,
        'stereo_frames': stereo_frames,         # frame numbers for which valid checkerboards were found for both cameras
        'valid_frames': valid_frames,
        'im_size': im_size,
        'cb_size': cb_size,
        'checkerboard_square_size': checkerboard_square_size,
        'calvid_metadata': calvid_metadata
    }

    skilled_reaching_io.write_pickle(cal_data_name, calibration_data)

    for vo in vid_obj:
        vo.release()


def compare_calibration_files(calib_folder):

    calib_files = glob.glob(os.path.join(calib_folder, '*.pickle'))
    all_cal_data = []
    for calib_file in calib_files:
        all_cal_data.append(skilled_reaching_io.read_pickle(calib_file))


def show_cal_images_with_epilines(cal_metadata, parent_directories, plot_undistorted=False):

    # find the videos containing the original calibration videos
    cal_vids_parent = parent_directories['cal_vids_parent']
    cal_videos = navigation_utilities.find_calibration_videos_optitrack(cal_metadata, cal_vids_parent)

    cal_vid_folder, _ = os.path.split(cal_videos[0])

    # find the directory containing the calibration data files
    cal_data_parent = parent_directories['cal_data_parent']
    cal_data_file = navigation_utilities.find_optitrack_calibration_data_name(cal_data_parent, cal_metadata['datetime'])

    # read in calibration data
    cal_data = skilled_reaching_io.read_pickle(cal_data_file)

    w = cal_data['im_size'][0][0]
    h = cal_data['im_size'][0][1]

    # loop through stereo pairs
    vid_objects = [cv2.VideoCapture(cal_vid) for cal_vid in cal_videos]
    cal_vid_metadata = [navigation_utilities.parse_Burgess_calibration_vid_name(cal_vid) for cal_vid in cal_videos]
    cam_num = [cal_vid_md['cam_num'] for cal_vid_md in cal_vid_metadata]
    stereo_imgpoints = cal_data['stereo_imgpoints']

    # E_from_norm, msk_norm = skilled_reaching_calibration.recalculate_E_from_stereo_matches(cal_data)

    for i_frame, frame_num in enumerate(cal_data['frames_for_stereo_calibration']):

        fig, axs = create_cal_frame_figure(w, h, ax3d=[(1, 0)], scale=1.0, dpi=200, nrows=2, ncols=2, wspace=0.05, hspace=0.01, lmargin=0.01, rmargin=0.95, botmargin=0.01, topmargin=0.95)
        img = []
        cb_pts = []
        for cal_idx in range(2):
            vid_objects[cal_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            _, new_img = vid_objects[cal_idx].read()
            ax_idx = cam_num[cal_idx] - 1

            mtx = cal_data['mtx'][ax_idx]
            dist = cal_data['dist'][ax_idx]

            other_mtx = cal_data['mtx'][1-ax_idx]
            other_dist = cal_data['dist'][1 - ax_idx]

            if cam_num[cal_idx] == 1:
                new_img = cv2.rotate(new_img, cv2.ROTATE_180)

            current_cbpoints = stereo_imgpoints[ax_idx][i_frame]
            other_cbpoints = stereo_imgpoints[1-ax_idx][i_frame]

            current_cbpoints_udnorm = cv2.undistortPoints(current_cbpoints, mtx, dist)
            other_cbpoints_udnorm = cv2.undistortPoints(other_cbpoints, other_mtx, other_dist)

            current_cbpoints_ud = cvb.unnormalize_points(current_cbpoints_udnorm, mtx)
            current_cbpoints_ud = current_cbpoints_ud.reshape((-1, 1, 2)).astype('float32')
            other_cbpoints_ud = cvb.unnormalize_points(other_cbpoints_udnorm, other_mtx)
            other_cbpoints_ud = other_cbpoints_ud.reshape((-1, 1, 2)).astype('float32')

            cb_pts = [np.zeros(np.shape(current_cbpoints)), np.zeros(np.shape(current_cbpoints))]
            if plot_undistorted:
                cb_pts[ax_idx] = current_cbpoints_ud
                cb_pts[1-ax_idx] = other_cbpoints_ud
            else:
                cb_pts[ax_idx] = current_cbpoints
                cb_pts[1-ax_idx] = other_cbpoints

            # cam_num is the camera number for each calibration video. ax_idx is also the index for mtx and dist in the cal_data dictionary for this camera (I think)

            new_img_ud = cv2.undistort(new_img, mtx, dist)

            if plot_undistorted:
                cb_img = cv2.drawChessboardCorners(new_img_ud, cal_data['cb_size'], current_cbpoints_ud, True)
                other_cbpoints_for_plot = other_cbpoints_ud
            else:
                cb_img = cv2.drawChessboardCorners(new_img, cal_data['cb_size'], current_cbpoints, True)
                other_cbpoints_for_plot = other_cbpoints
            img.append(cb_img)
            ax_idx = cam_num[cal_idx] - 1

            plot_utilities.draw_epipolar_lines(cb_img, cal_data, cam_num[cal_idx], other_cbpoints, [], use_ffm=False, markertype=['o', '+'], ax=axs[0][ax_idx])
            F_array = np.stack((cal_data['F'], cal_data['F_ffm'], cal_data['F_ffm']), axis=0)
            plot_utilities.compare_epipolar_lines(cb_img, cal_data, cam_num[cal_idx], other_cbpoints, [], F_array,
                                               markertype=['o', '+'], ax=axs[0][ax_idx])

        world_points, reprojected_pts = cvb.triangulate_points(cb_pts, cal_data)
        for ax_idx in range(2):
            axs[0][ax_idx].scatter(reprojected_pts[ax_idx][:, 0], reprojected_pts[ax_idx][:, 1], edgecolors='k', s=6, marker='s', facecolor='none')

        pt_colors = [[0., 0., 1.],
                       [0., 128. / 255., 1.],
                       [0., 200. / 255., 200. / 255.],
                       [0., 1., 0],
                       [200. / 255., 200. / 255., 0.],
                       [1., 0., 0.],
                       [1., 0., 1.]]
        num_pts = np.shape(world_points)[0]
        for i_pt in range(num_pts):
            col_idx = int(i_pt / 7.) % 7
            axs[1][0].scatter(world_points[i_pt, 0], world_points[i_pt, 1], world_points[i_pt, 2], c=pt_colors[col_idx])
        axs[1][0].view_init(elev=110., azim=90.)

        datestring = navigation_utilities.datetime_to_string_for_fname(cal_metadata['datetime'])
        if plot_undistorted:
            jpg_name = '_'.join(('stereotest',
                                 datestring,
                                 'frame{:04d}'.format(frame_num),
                                 'undistorted.jpg'))
        else:
            jpg_name = '_'.join(('stereotest',
                                 datestring,
                                 'frame{:04d}.jpg'.format(frame_num)))
        jpg_name = os.path.join(cal_vid_folder, jpg_name)

        plt.show()
        plt.savefig(jpg_name)
        plt.close(fig)


def create_cal_frame_figure(width, height, ax3d=None, scale=1.0, dpi=100, nrows=1, ncols=1, wspace=0.05, hspace=0.05, lmargin=0.01, rmargin=0.95, botmargin=0.01, topmargin=0.95):

    # create a figure with adjacent axes

    fig_width = (width * scale / dpi) * ncols
    fig_height = (height * scale / dpi) * nrows
    fig = plt.figure(
        frameon=False, figsize=(fig_width, fig_height), dpi=dpi
    )
    fig.tight_layout()

    available_panel_w = rmargin - lmargin - ((ncols - 1) * wspace)
    available_panel_h = topmargin - botmargin - ((nrows - 1) * hspace)
    panel_width = available_panel_w / ncols
    panel_height = available_panel_h / nrows
    axs = []
    for i_row in range(nrows):
        ax_row = []
        bottom = topmargin - i_row * (panel_height + hspace) - panel_height
        # top = topmargin - i_row * (panel_width + hspace)
        for i_col in range(ncols):
            idx = (i_row * ncols) + i_col + 1

            # TEST IF THIS AXES SHOULD BE 3D
            if (i_row, i_col) in ax3d:
                ax_row.append(fig.add_subplot(nrows, ncols, idx, projection='3d'))
                ax_row[i_col].set_xlabel('x')
                ax_row[i_col].set_ylabel('y')
                ax_row[i_col].set_zlabel('z')
                ax_row[i_col].invert_zaxis()
            else:
                ax_row.append(fig.add_subplot(nrows, ncols, idx))
                ax_row[i_col].axis("off")
                ax_row[i_col].set_xlim(0, width)
                ax_row[i_col].set_ylim(0, height)
            ax_row[i_col].invert_yaxis()

            left = lmargin + i_col * (panel_width + wspace)
            # right = lmargin + i_col * (panel_width + wspace) + panel_width

            ax_row[i_col].set_position([left, bottom, panel_width, panel_height])

        axs.append(ax_row)

    return fig, axs


def calibrate_single_camera(cal_vid, board, num_frames2use=20, min_pts_per_frame=10):
    '''

    :param cal_vid:
    :param board:
    :param num_frames2use:
    :param min_pts_per_frame: minimum number of identified charuco points needed in each frame for calibration.
        8 or more should be adequate, but I think if they're colinear the algorithm collapses. Made default 10 and that
        seemed to fix the error (DL, 11/14/2024)
    :return:
    '''
    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_ASPECT_RATIO

    rows, size = detect_video_pts(cal_vid, board, skip=1)
    # size is (w, h)
    # rows = board.detect_video(cal_vid, prefix=None, skip=skip, progress=True)

    objp, imgp = board.get_all_calibration_points(rows)

    skip = int(len(objp) / num_frames2use)

    #
    mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= min_pts_per_frame]
    valid_frames = [ii for ii, o in enumerate(objp) if len(o) >= min_pts_per_frame]

    objp, imgp = zip(*mixed)

    # matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
    num_frames = len(objp)
    n_charuco_pts_per_frame = np.array([len(row['ids']) for row in rows])
    frames_to_use = list(range(0, num_frames, skip))
    objp_to_use = [objp[ii] for ii in frames_to_use]
    imgp_to_use = [imgp[ii] for ii in frames_to_use]
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_to_use, imgp_to_use, size, None, None, flags=CALIBRATION_FLAGS)
    except:
        pass

    cam_intrinsic_data = {'ret': ret,
                          'mtx': mtx,
                          'dist': dist,
                          'rvecs': rvecs,
                          'tvecs': tvecs,
                          'obj': objp,
                          'valid_frames': valid_frames,
                          'frames_used': frames_to_use}
    # valid_frames are frames in which >= 7 valid points were detected (frame numbers in the actual movie)
    # frames_used are the indices of the valid_frames used for calibration

    return cam_intrinsic_data


def collect_matched_mirror_points(merged, board):

    # this will contain the number of points from each frame and which points were identified (more relevant for charuco than for checkerboard)
    # first dictionary in this list is for the left mirror, second dictionary is for the right mirror
    matched_points_metadata = [{'framenumbers': [],'ptids': []}, {'framenumbers': [],'ptids': []}]
    if type(board) is Checkerboard:
        # find the first element of merged that has a 'dir' view entry
        for merged_row in merged:
            if 'dir' in list(merged_row.keys()):
                pts_per_frame = np.shape(merged_row['dir']['corners'])[0]
                break
        # count up all merged rows that contain leftmirror points
        leftmirror_rows = [mr for mr in merged if 'lm' in mr.keys() and 'dir' in mr.keys()]
        rightmirror_rows = [mr for mr in merged if 'rm' in mr.keys() and 'dir' in mr.keys()]

        num_leftmirror_rows = len(leftmirror_rows)
        num_leftmirror_pts = num_leftmirror_rows * pts_per_frame

        num_rightmirror_rows = len(rightmirror_rows)
        num_rightmirror_pts = num_rightmirror_rows * pts_per_frame

        # initialize arrays to hold image points and object points for calibration
        leftmirror_imgp = np.empty((num_leftmirror_pts, 1, 2))
        directleft_imgp = np.empty((num_leftmirror_pts, 1, 2))
        rightmirror_imgp = np.empty((num_rightmirror_pts, 1, 2))
        directright_imgp = np.empty((num_rightmirror_pts, 1, 2))

        left_objp = np.empty((num_leftmirror_pts, 3))
        right_objp = np.empty((num_rightmirror_pts, 3))

        current_lm_row = 0
        current_rm_row = 0
        for merged_row in leftmirror_rows:
            imgp_direct = merged_row['dir']['corners']
            imgp_mirror = merged_row['lm']['corners']

            leftmirror_imgp[current_lm_row:current_lm_row+pts_per_frame, :, :] = imgp_mirror
            directleft_imgp[current_lm_row:current_lm_row+pts_per_frame, :, :] = imgp_direct

            left_objp[current_lm_row:current_lm_row+pts_per_frame, :] = board.get_object_points()

            matched_points_metadata[0]['framenumbers'].append(merged_row['dir']['framenum'])
            matched_points_metadata[0]['ptids'].append(merged_row['dir']['ids'])

            current_lm_row += pts_per_frame

        for merged_row in rightmirror_rows:
            imgp_direct = merged_row['dir']['corners']
            imgp_mirror = merged_row['rm']['corners']
            rightmirror_imgp[current_rm_row:current_rm_row + pts_per_frame, :, :] = imgp_mirror
            directright_imgp[current_rm_row:current_rm_row + pts_per_frame, :, :] = imgp_direct

            right_objp[current_rm_row:current_rm_row + pts_per_frame, :] = board.get_object_points()

            matched_points_metadata[1]['framenumbers'].append(merged_row['dir']['framenum'])
            matched_points_metadata[1]['ptids'].append(merged_row['dir']['ids'])

            current_rm_row += pts_per_frame

    elif type(board) is CharucoBoard:
        leftmirror_rows = [mr for mr in merged if 'lm' in mr.keys() and 'dir' in mr.keys()]
        rightmirror_rows = [mr for mr in merged if 'rm' in mr.keys() and 'dir' in mr.keys()]

        objp = board.get_object_points()

        num_left_rows_in_imgp = 0
        for merged_row in leftmirror_rows:
            sorted_direct_imgp, sorted_mirror_imgp, sorted_objp, sorted_corner_idx = match_points_in_charuco_row(merged_row, objp, 'lm')

            if sorted_direct_imgp.any():
                # if matched points were found for this row
                if num_left_rows_in_imgp == 0:
                    leftmirror_imgp = sorted_mirror_imgp
                    directleft_imgp = sorted_direct_imgp
                    left_objp = sorted_objp
                else:
                    leftmirror_imgp = np.vstack((leftmirror_imgp, sorted_mirror_imgp))
                    directleft_imgp = np.vstack((directleft_imgp, sorted_direct_imgp))
                    left_objp = np.vstack((left_objp, sorted_objp))

                matched_points_metadata[0]['framenumbers'].append(merged_row['dir']['framenum'])
                matched_points_metadata[0]['ptids'].append(sorted_corner_idx)
                # matched_points_metadata[0]['ptids'].append(merged_row['direct']['ids'])

                num_left_rows_in_imgp += 1

        num_right_rows_in_imgp = 0
        for merged_row in rightmirror_rows:
            sorted_direct_imgp, sorted_mirror_imgp, sorted_objp, sorted_corner_idx = match_points_in_charuco_row(
                merged_row, objp, 'rm')

            if sorted_direct_imgp.any():
                # if matched points were found for this row
                if num_right_rows_in_imgp == 0:
                    rightmirror_imgp = sorted_mirror_imgp
                    directright_imgp = sorted_direct_imgp
                    right_objp = sorted_objp
                else:
                    rightmirror_imgp = np.vstack((rightmirror_imgp, sorted_mirror_imgp))
                    directright_imgp = np.vstack((directright_imgp, sorted_direct_imgp))
                    right_objp = np.vstack((right_objp, sorted_objp))

                matched_points_metadata[1]['framenumbers'].append(merged_row['dir']['framenum'])
                # matched_points_metadata[1]['ptids'].append(merged_row['direct']['ids'])
                matched_points_metadata[1]['ptids'].append(sorted_corner_idx)

                num_right_rows_in_imgp += 1

        if num_left_rows_in_imgp == 0:
            # there weren't any matched points for the left mirror
            leftmirror_imgp = None
            directleft_imgp = None
            left_objp = None

        if num_right_rows_in_imgp == 0:
            # there weren't any matched points for the left mirror
            rightmirror_imgp = None
            directright_imgp = None
            right_objp = None


    stereo_cal_points = {'leftmirror': leftmirror_imgp,
                         'directleft': directleft_imgp,
                         'left_objp': left_objp,
                         'rightmirror': rightmirror_imgp,
                         'directright': directright_imgp,
                         'right_objp': right_objp}


    return stereo_cal_points, matched_points_metadata


def match_points_in_charuco_row(merged_row, objp, mirror_view):
    imgp_direct = merged_row['dir']['corners']
    imgp_mirror = merged_row[mirror_view]['corners']

    # if there is only one identified point in one (or both) of the views, the array dimensions get messed up.
    # code below insures the output will be n x 1 x 2 where n is the number of points
    imgp_direct = np.reshape(imgp_direct, (-1, 1, 2))
    imgp_mirror = np.reshape(imgp_mirror, (-1, 1, 2))

    if np.shape(merged_row['dir']['ids'])[0] == 1:
        direct_ids = merged_row['dir']['ids'][0]
    else:
        direct_ids = np.squeeze(merged_row['dir']['ids'])
    mirror_ids = np.squeeze(merged_row[mirror_view]['ids'])

    # make sure points with the same id's are matched
    matched_direct_idx = []
    matched_mirror_idx = []

    sorted_mirror_imgp = []
    sorted_direct_imgp = []
    sorted_corner_idx = []
    sorted_objp = []

    for d_id_idx, d_id in enumerate(direct_ids):
        mirror_id_idx = np.where(mirror_ids == d_id)[0]

        if len(mirror_id_idx) == 1:
            # there is a matched point in the left view for this point in the direct view
            matched_direct_idx.append(d_id_idx)
            matched_mirror_idx.append(mirror_id_idx[0])
            sorted_corner_idx.append(d_id)

            sorted_direct_imgp.append(imgp_direct[matched_direct_idx[-1]])
            sorted_mirror_imgp.append(imgp_mirror[matched_mirror_idx[-1]])
            sorted_objp.append(objp[d_id])

    sorted_direct_imgp = np.array(sorted_direct_imgp)
    sorted_mirror_imgp = np.array(sorted_mirror_imgp)
    sorted_objp = np.array(sorted_objp)

    # sorted_corner_idx probably isn't necessary, but may be helpful for troubleshooting
    return sorted_direct_imgp, sorted_mirror_imgp, sorted_objp, sorted_corner_idx



def select_correct_E_mirror(R1, R2, T, pts1, pts2, mtx):

    # construct the four possible solutions
    rot = np.empty((4, 3, 3))
    t = np.empty((4, 3))

    num_pts = np.shape(pts1)[0]
    if np.shape(pts2)[0] != num_pts:
        print('pts1 and pts2 have different numbers of points')
        return None

    rot[0, :, :] = R1
    rot[1, :, :] = R2
    rot[2, :, :] = R1
    rot[3, :, :] = R2

    t[0, :] = np.squeeze(T)
    t[1, :] = np.squeeze(T)
    t[2, :] = np.squeeze(-T)
    t[3, :] = np.squeeze(-T)

    t = np.expand_dims(t, 1)

    # convert pts to homogeneous coordinates
    pts1_norm = cvb.normalize_points(pts1, mtx)
    pts2_norm = cvb.normalize_points(pts2, mtx)

    x3D = np.empty((4, num_pts, 4))
    # all_pts = np.array([np.squeeze(pts1), np.squeeze(pts2)])
    for ii in range(4):
        # create projection matrices
        P1 = np.eye(N=3, M=4)
        P2 = np.hstack((rot[ii, :, :], t[ii, :, :].T))
        # wp, rp = cvb.triangulate_points()

        x3D[ii, :, :] = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm).T


        # points4D = cv2.triangulatePoints(projMatr1, projMatr2, pts_ud[0], pts_ud[1])
        # world_points = np.squeeze(cv2.convertPointsFromHomogeneous(points4D.T))
        #
        #
        # worldpoints = np.squeeze(cv2.convertPointsFromHomogeneous(points4D.T))

    correct = None
    depth = np.empty((4, 2))
    for ii in range(4):
        # compute the depth and sum the signs
        depth[ii, 0] = np.sum(np.sign(cvb.depth_of_points(x3D[ii, :, :], np.eye(3), np.zeros((1, 3)))))
        depth[ii, 1] = np.sum(np.sign(cvb.depth_of_points(x3D[ii, :, :], rot[ii, :, :], t[ii, :, :])))

        if depth[ii, 0] > 0 and depth[ii, 1] < 0:
            # the correct pair of rotation matrix and translation vector should have all points in front of the real
            # camera and behind the virtual (mirror) camera
            correct = ii

    if correct is None:
        print('No projection matrix has all triangulated points in front of both cameras')
        c_rot = None
        c_t = None
    else:
        c_rot = rot[correct, :, :]
        c_t = t[correct, :, :]

    return c_rot, c_t, correct


def mirror_stereo_cal(stereo_cal_points, cam_intrinsics, view_names=[['directleft', 'leftmirror'], ['directright', 'rightmirror']]):

    view_keys = list(stereo_cal_points.keys())
    for view_key in view_keys:
        if view_key in ['leftmirror', 'left_mirror', 'lm']:
            lm_key = view_key
        elif view_key in ['rightmirror', 'right_mirror', 'rm']:
            rm_key = view_key
        elif view_key in ['direct', 'dir']:
            dir_key = view_key

    # calculate fundamental matrices for each view
    F = np.empty((3, 3, 2))
    if stereo_cal_points[lm_key] is None:
        F[:, :, 0].fill(np.nan)
    else:
        F[:, :, 0] = cvb.fund_matrix_mirror(stereo_cal_points['directleft'], stereo_cal_points['leftmirror'])

    if stereo_cal_points[rm_key] is None:
        F[:, :, 1].fill(np.nan)
    else:
        F[:, :, 1] = cvb.fund_matrix_mirror(stereo_cal_points['directright'], stereo_cal_points['rightmirror'])

    # calculate essential matrices
    E = np.empty((3, 3, 2))
    c_rot = np.empty((3, 3, 2))
    c_t = np.empty((3, 2))
    # P2 = np.empty((3, 4, 2))
    for i_view in range(2):
        if np.isnan(F[0, 0, i_view]).any():
            temp = np.empty((3, 3))
            temp.fill(np.nan)

            E[:, :, i_view] = temp
            c_rot[:, :, i_view] = temp

            temp = np.empty(3)
            temp.fill(np.nan)
            c_t[:, i_view] = temp

        else:
            E[:, :, i_view] = np.linalg.multi_dot((cam_intrinsics['mtx'].T, F[:, :, i_view], cam_intrinsics['mtx']))

            # _, R[:, :, i_view], T[:, i_view], _ = cv2.recoverPose(E[:, :, i_view],
            #                                                 stereo_cal_points[view_names[i_view][0]],
            #                                                 stereo_cal_points[view_names[i_view][1]],
            #                                                 cam_intrinsics['mtx'])
            R1, R2, T = cv2.decomposeEssentialMat(E[:, :, i_view])
            c_rot[:, :, i_view], c_t[:, i_view], correct = select_correct_E_mirror(R1, R2, T, stereo_cal_points[view_names[i_view][0]], stereo_cal_points[view_names[i_view][1]], cam_intrinsics['mtx'])

            t_mat = np.expand_dims(c_t[:, i_view], 1)
        # P2[:, :, i_view] = np.hstack((c_rot[:, :, i_view], t_mat))

    return E, F, c_rot, c_t


def calc_3d_gridspacing_checkerboard(pts3d, board_size):

    num_col_spacings = (board_size[0] - 1) * board_size[1]
    num_row_spacings = (board_size[1] - 1) * board_size[0]

    grid_spacings_per_frame = num_row_spacings + num_col_spacings

    # assume pts3d is a m x 3 where m is the number of points
    num_pts = np.shape(pts3d)[0]
    pts_per_img = np.prod(board_size)
    num_img = int(num_pts / pts_per_img)

    all_dist_start_idx = 0
    distances_per_frame = int(pts_per_img * (pts_per_img - 1) / 2)
    total_distances = num_img * distances_per_frame
    total_grid_spacings = num_img * grid_spacings_per_frame
    all_distances = np.empty(total_distances)
    for i_img in range(num_img):

        start_frame_pt = i_img * pts_per_img
        last_frame_pt = start_frame_pt + pts_per_img
        for i_pt in range(start_frame_pt, last_frame_pt):

            axes_diffs = pts3d[i_pt, :] - pts3d[i_pt + 1:last_frame_pt, :]
            new_distances = np.linalg.norm(axes_diffs, ord=None, axis=1)

            all_distances[all_dist_start_idx:all_dist_start_idx + len(new_distances)] = new_distances

            all_dist_start_idx += len(new_distances)

    all_distances = np.sort(all_distances)
    grid_spacing = all_distances[:total_grid_spacings]

    return grid_spacing


def calc_3d_scale_factor(pts1, pts2, mtx, rot, t, matched_points_metadata, board):
    '''
    
    :param pts1:
    :param pts2:
    :param mtx:
    :param rot:
    :param t:
    :param matched_points_metadata:
    :param board:
    :return:
    '''

    if np.isnan(rot).any():
        scale_factor = np.nan
        return scale_factor

    P2 = cvb.P_from_RT(rot, t)
    num_pts = np.shape(pts1)[0]

    pts1_norm = cvb.normalize_points(pts1, mtx).T
    pts2_norm = cvb.normalize_points(pts2, mtx).T

    camera_mats = np.zeros((3, 4, 2))
    camera_mats[:, :, 0] = np.eye(3, 4)
    camera_mats[:, :, 1] = P2

    pts3d = np.zeros((num_pts, 3))

    for i_pt in range(num_pts):
        pts_match = np.vstack((pts1_norm[i_pt, :], pts2_norm[i_pt, :]))

        pts3d[i_pt, :] = cvb.multiview_ls_triangulation(pts_match, camera_mats)


    # calculate grid spacing for 3d reconstructed points from normalized coordinates
    if type(board) is Checkerboard:
        gridspacing = calc_3d_gridspacing_checkerboard(pts3d, board.get_size())
        mean_spacing = np.mean(gridspacing)
        scale_factor = board.get_square_length() / mean_spacing
    elif type(board) is CharucoBoard:
        # need to figure out how many points were matched for each image, and what the spacing should be between the points

        corners_start_idx = 0
        objp = board.get_object_points()
        frame_scale_factors = []
        for frame_ptids in matched_points_metadata['ptids']:
            num_frame_pts = len(frame_ptids)
            if num_frame_pts == 1:
                # can't get scale from a single point
                corners_start_idx += 1
            else:
                # find all pairwise combinations of points in this frame and the real distances between them
                # this requires knowledge of how points are arranged numerically in the grid; but this should be available
                # in the boards object
                frame_pts3d = pts3d[corners_start_idx:corners_start_idx + num_frame_pts, :]
                frame_scale_factors.append(pt3d_dist_ratios(frame_pts3d, frame_ptids, objp))

                corners_start_idx += num_frame_pts

        scale_factor = np.mean(frame_scale_factors)

    return scale_factor


def pt3d_dist_ratios(pts3d, ptids, objp):

    # find all pairwise combinations of points
    num_frame_pts = len(ptids)
    scale_factors = []
    for i_pt_a in range(num_frame_pts - 1):
        pt3d_a = pts3d[i_pt_a, :]
        for i_pt_b in range(i_pt_a + 1, num_frame_pts):
            pt3d_b = pts3d[i_pt_b, :]

            # calculate distance between 3d point a and 3d point b
            dist_3d = np.linalg.norm(pt3d_a - pt3d_b)
            dist_objp = np.linalg.norm(objp[ptids[i_pt_a], :] - objp[ptids[i_pt_b], :])
            scale_factors.append(dist_objp / dist_3d)

    mean_scaling = np.mean(scale_factors)

    return mean_scaling

def test_board_reconstruction(pts1, pts2, mtx, rot, t, board):

    # todo: if rot is a vector, convert to matrix
    if np.ndim(np.squeeze(rot)) == 1:
        # this is a rotation vector, not a matrix
        rot, _ = cv2.Rodrigues(rot)

    P1 = np.eye(N=3, M=4)
    P2 = cvb.P_from_RT(rot, t)
    num_pts = np.shape(pts1)[0]

    pts1_norm = cvb.normalize_points(pts1, mtx).T
    pts2_norm = cvb.normalize_points(pts2, mtx).T
    # wp, rp = cvb.triangulate_points()

    x3D = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T).T
    x3D_nhom = np.zeros((num_pts, 3))
    for ii in range(3):
        x3D_nhom[:, ii] = x3D[:,ii] / x3D[:,3]

    camera_mats = np.zeros((3, 4, 2))
    camera_mats[:, :, 0] = np.eye(3, 4)
    camera_mats[:, :, 1] = cvb.P_from_RT(rot, t)

    pts3d = np.zeros((num_pts, 3))
    # wpts3d_Kinv = np.zeros((num_pts, 3))
    # wpts3d_K = np.zeros((num_pts, 3))

    for i_pt in range(num_pts):
        pts_match = np.vstack((pts1_norm[i_pt,:], pts2_norm[i_pt, :]))

        pts3d[i_pt, :] = cvb.multiview_ls_triangulation(pts_match, camera_mats)

    calc_3d_gridspacing(pts3d, board.get_size())

    Kinv = np.linalg.inv(mtx)
    wpts3d_Kinv = np.matmul(Kinv, pts3d.T).T
    wpts3d_K = np.matmul(mtx, pts3d.T).T

#todo: figure out the right way to pull out real-world coordinates
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x3D_nhom[:63, 0], x3D_nhom[:63, 1], x3D_nhom[:63, 2])
    ax.scatter(pts3d[:63, 0], pts3d[:63, 1], pts3d[:63, 2])
    # ax.scatter(wpts3d_Kinv[:63, 0], wpts3d_Kinv[:63, 1], wpts3d_Kinv[:63, 2])
    ax.invert_yaxis()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('mutliplied by K')

    #
    # fig2 = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(x3D_nhom[:63, 0], x3D_nhom[:63, 1], x3D_nhom[:63, 2])
    # # ax.scatter(pts3d[:63, 0], pts3d[:63, 1], pts3d[:63, 2])
    # # ax.scatter(wpts3d_K[:63, 0], wpts3d_K[:63, 1], wpts3d_K[:63, 2])
    # ax.invert_yaxis()
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_title('mutliplied by Kinv')

    plt.show()
    pass


def overlay_rows_on_calibration_video(calibration_data, full_calib_vid_name):

    vid_folder, vid_name = os.path.split(full_calib_vid_name)
    labeledvids_folder = os.path.join(vid_folder, 'labeled_vids')
    if not os.path.exists(labeledvids_folder):
        os.makedirs(labeledvids_folder)
    labeled_vid_name = vid_name.replace('.avi', '_labeled.avi')
    labeled_vid_name = os.path.join(labeledvids_folder, labeled_vid_name)
    if os.path.exists(labeled_vid_name):
        return

    all_rows = calibration_data['all_rows']
    n_cams = len(all_rows)
    cv_cap = cv2.VideoCapture(full_calib_vid_name)
    n_frames = int(cv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cv_cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cv_out = cv2.VideoWriter(labeled_vid_name, fourcc, fps, (w, h))
    mtx = calibration_data['cam_intrinsics']['mtx']
    dist = calibration_data['cam_intrinsics']['dist']

    for i_frame in range(n_frames):
        # overlay points
        row_idx = [0, 0, 0]
        ret, img = cv_cap.read()
        if not ret:
            break
        img_ud = cv2.undistort(img, mtx, dist)
        fig = plt.figure()
        ax = fig.add_subplot()
        # ax.imshow(img_ud)

        for i_view, rows in enumerate(all_rows):
            frame_nums = np.array([(i_row, row['framenum']) for i_row, row in enumerate(rows)])
            try:
                frame_row_idx = frame_nums[frame_nums[:,1]==i_frame,0][0]
            except:
                # there aren't data for this frame
                frame_row_idx = None
            row_idx[i_view] = frame_row_idx

        for i_view, rows in enumerate(all_rows):
            if not row_idx[i_view] is None:
                frame_row = rows[row_idx[i_view]]
                for ii, id in enumerate(frame_row['ids']):
                    plt.text(frame_row['corners'][ii, 0, 0], frame_row['corners'][ii, 0, 1],
                             '{:d}'.format(id), c='r', fontsize='small')
                    text_loc = (int(frame_row['corners'][ii, 0, 0]), int(frame_row['corners'][ii, 0, 1]))
                    cv2.putText(img_ud, '{:d}'.format(id), text_loc, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, color=(0, 0, 255))

        cv_out.write(img_ud)
        # frame_name = '{:04d}.jpg'.format(i_frame)
        # frame_name = os.path.join(temp_folder, frame_name)

        # plt.savefig(frame_name)
        plt.close(fig)

    # check_detections()
    cv_cap.release()
    cv_out.release()


def calibrate_mirror_views(cropped_vids, cam_intrinsics, board, cam_names, parent_directories, session_row, calibration_pickle_name,
                           full_calib_vid_name=None, view_names=[['directleft', 'leftmirror'], ['directright', 'rightmirror']], init_extrinsics=True, verbose=True):
    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_USE_INTRINSIC_GUESS

    if os.path.exists(calibration_pickle_name):
        calibration_data = skilled_reaching_io.read_pickle(calibration_pickle_name)
        # overlay_rows_on_calibration_video(calibration_data, full_calib_vid_name)
        cgroup = calibration_data['cgroup']
        return cgroup, None
    else:
        cgroup = CameraGroup.from_names(cam_names, fisheye=False)
        for camera in cgroup.cameras:
            camera.set_camera_matrix(cam_intrinsics['mtx'])
            camera.set_distortions([np.zeros(5)])
        calibration_data = {'cam_intrinsics': cam_intrinsics,
                            'cgroup': cgroup,
                            'session_row': session_row,
                            'scale_factors': np.zeros(2),
                            'F': None,
                            'E': None,
                            'intrinsics_initialized': True,
                            'estimated_poses': False,
                            'extrinsics_initialized': False,
                            'bundle_adjust_completed': False}
    mirror_board = skilled_reaching_calibration.mirror_board_from_df(session_row)
    # get_rows_cropped_vids will
    #  1. detect the checkerboard/charuco board points
    #  2. undistort points in the full original reference frame, then move them back into the cropped
    #      video, then flip them left-right if in a mirror view
    if 'all_rows' not in calibration_data.keys():
        all_rows = get_rows_cropped_vids(cropped_vids, cam_intrinsics, mirror_board, parent_directories, cgroup, full_calib_vid_name=full_calib_vid_name)
        for i, (row, cam) in enumerate(zip(all_rows, cgroup.cameras)):
            # need to make sure the cameras are in the right order; this should have been checked in the code above
            all_rows[i] = mirror_board.estimate_pose_rows(cam, row)
        calibration_data['all_rows'] = all_rows
        # skilled_reaching_io.write_pickle(calibration_pickle_name, calibration_data)
    else:
        all_rows = calibration_data['all_rows']
        for i, (row, cam) in enumerate(zip(all_rows, cgroup.cameras)):
            # need to make sure the cameras are in the right order; this should have been checked in the code above
            all_rows[i] = mirror_board.estimate_pose_rows(cam, row)
        calibration_data['all_rows'] = all_rows

    # at this point, should save the camera groups/boards in a .pickle file so don't have to detect the boards each time
    # also, once saved, write something to verify that the points are what I think they are

    # calculate the fundamental matrices for direct-->left mirror and direct-->right mirror
    merged = merge_rows(all_rows, cam_names=cam_names)
    stereo_cal_points, matched_points_metadata = collect_matched_mirror_points(merged, mirror_board)
    # if calibration_data['E'] is None:
    E, F, rot, t = mirror_stereo_cal(stereo_cal_points, cam_intrinsics, view_names=view_names)

    calibration_data['E'] = E
    calibration_data['F'] = F

    # if not calibration_data['extrinsics_initialized']:
    rvecs = [[0., 0., 0.]]
    cam_t = [[0., 0., 0.]]
    scale_factors = np.zeros(2)
    for i_view in range(2):
        cam_rvec, _ = cv2.Rodrigues(rot[:, :, i_view])
        rvecs.append(cam_rvec)

        scale_factors[i_view] = calc_3d_scale_factor(stereo_cal_points[view_names[i_view][0]],
                                                    stereo_cal_points[view_names[i_view][1]], cam_intrinsics['mtx'],
                                                    rot[:, :, i_view], t[:, i_view], matched_points_metadata[i_view], mirror_board)

        cam_t.append(t[:, i_view] * scale_factors[i_view])
    calibration_data['scale_factors'] = scale_factors

    cgroup.set_rotations(rvecs)
    cgroup.set_translations(cam_t)

    calibration_data['extrinsics_initialized'] = True

    cgroup_old = copy.deepcopy(cgroup)
    imgp, extra = extract_points(merged, mirror_board, cam_names=cam_names, min_cameras=2)

    # if not calibration_data['bundle_adjust_completed']:
        # if one of the views couldn't be calibrated, skip bundle adjustment for now
    if not np.isnan(calibration_data['E']).any():
        # error = cgroup.bundle_adjust_iter_fixed_dist(imgp, extra, verbose=verbose)
        # error = cgroup.bundle_adjust_iter_fixed_intrinsics(imgp, extra, verbose=verbose)
        error = cgroup.bundle_adjust_fixed_intrinsics_and_cam0(imgp, extra, verbose=verbose)
        calibration_data['cgroup'] = cgroup
        calibration_data['error'] = error
        calibration_data['bundle_adjust_completed'] = True
    else:
        # todo: manually calibrate if automatic detection didn't work
        error = None

    skilled_reaching_io.write_pickle(calibration_pickle_name, calibration_data)

    # else:
    #     cgroup = calibration_data['cgroup']
    #     error = calibration_data['error']

    return cgroup, error

    # code to test the 3d reconstructions
    # i_view = 0
    # rot = cgroup.cameras[i_view + 1].get_rotation()
    # t = cgroup.cameras[i_view + 1].get_translation()
    # test_board_reconstruction(stereo_cal_points[view_names[i_view][0]], stereo_cal_points[view_names[i_view][1]], cam_intrinsics['mtx'], rot, t, board)

    # imgp, extra = extract_points(merged, board, cam_names=cam_names, min_cameras=2)
    #
    #
    #
    # # initialize the intrinsic matrix and distortion coefficients (should be all zeros) for each view
    # if not calibration_data['intrinsics_initialized']:
    #     print('initializing camera matrices...')
    #     expected_cam_idx = 0
    #     for rows, cropped_vid in zip(all_rows, cropped_vids):
    #         # get intrinsics for each camera
    #         # figure out which camera this is
    #         this_cam = None
    #         for i_cam, cam_name in enumerate(cgroup.get_names()):
    #             if cam_name in cropped_vid:
    #                 this_cam = cam_name
    #                 cam_idx = i_cam
    #                 if cam_idx != expected_cam_idx:
    #                     # error handling here for the cameras not being in the same order as the videos
    #                     error('cropped video names and cameras are not in the same order')
    #                 expected_cam_idx += 1
    #                 break
    #         if this_cam is None:
    #             # error handling here for no camera corresponding to this video
    #             error('no camera corresponding to {}'.format(cropped_vid))
    #
    #         cap = cv2.VideoCapture(cropped_vid)
    #         w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         size = (w, h)
    #         cap.release()
    #
    #         cgroup.cameras[cam_idx].set_size(size)
    #         # initialize the intrinsic matrix for the cropped view
    #         # distortion coefficients should be zero - points should already be undistorted
    #         objp, imgp = board.get_all_calibration_points(rows)
    #         mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 7]
    #         try:
    #             objp, imgp = zip(*mixed)
    #         except:
    #             pass
    #         matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
    #         # mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, size, matrix, np.zeros((1,5)), flags=CALIBRATION_FLAGS)
    #
    #         cgroup.cameras[cam_idx].set_camera_matrix(matrix)
    #         cgroup.cameras[cam_idx].set_distortions(np.zeros((1,5)))   # because all points should already be undistorted
    #
    #     calibration_data['cgroup'] = cgroup
    #     calibration_data['intrinsics_initialized'] = True
    #     skilled_reaching_io.write_pickle(calibration_pickle_name, calibration_data)
    #
    # # test to see if poses have already been estimated for each frame
    # if not calibration_data['estimated_poses']:
    #     print('estimating poses...')
    #     for i, (row, cam) in enumerate(zip(all_rows, cgroup.cameras)):
    #         # need to make sure the cameras are in the right order; this should have been checked in the code above
    #         all_rows[i] = board.estimate_pose_rows(cam, row)
    #
    #     calibration_data['all_rows'] = all_rows
    #     calibration_data['estimated_poses'] = True
    #     skilled_reaching_io.write_pickle(calibration_pickle_name, calibration_data)
    #
    # merged = merge_rows(all_rows, cam_names=cam_names)
    # imgp, extra = extract_points(merged, board, cam_names=cam_names, min_cameras=2)
    # '''
    #     extra = {
    #     'objp': objp,
    #     'ids': board_ids,
    #     'rvecs': rvecs,
    #     'tvecs': tvecs
    # }'''
    #
    # if not calibration_data['extrinsics initialized'] and init_extrinsics:
    #     print('initializing extrinsics...')
    #     rtvecs = extract_rtvecs(merged)
    #     if verbose:
    #         pprint(get_connections(rtvecs, cam_names))
    #     try:
    #         rvecs, tvecs = get_initial_extrinsics(rtvecs, cam_names)
    #     except:
    #         pass
    #     cgroup.set_rotations(rvecs)
    #     cgroup.set_translations(tvecs)
    #
    #     calibration_data['cgroup'] = cgroup
    #     calibration_data['extrinsics initialized'] = True
    #     skilled_reaching_io.write_pickle(calibration_pickle_name, calibration_data)
    #
    #
    # # need to look and decide if default parameters in bundle_ajust_iter work well here
    # # don't undistort the points - already done in get_rows_cropped_vids
    # if not calibration_data['bundle_adjust_completed']:
    #     print('calculating bundle adjustment...')
    #     error = cgroup.bundle_adjust_iter_fixed_dist(imgp, extra, verbose=verbose)
    #
    #     calibration_data['error'] = error
    #     calibration_data['cgroup'] = cgroup
    #     calibration_data['bundle_adjust_completed'] = True
    #     skilled_reaching_io.write_pickle(calibration_pickle_name, calibration_data)
    # else:
    #     error = calibration_data['error']
    #
    # return cgroup, error


def test_anipose_calibration(session_row, parent_directories):

    calibration_vids_parent = parent_directories['calibration_vids_parent']
    calibration_files_parent = parent_directories['calibration_files_parent']

    mirror_calib_vid_name = session_row.iloc[0]['cal_vid_name_mirrors']
    full_calib_vid_name = navigation_utilities.find_mirror_calibration_video(mirror_calib_vid_name,
                                                                             parent_directories)

    calibration_pickle_name = navigation_utilities.create_calibration_summary_name(full_calib_vid_name, calibration_files_parent)
    if not os.path.exists(calibration_pickle_name):
        print('session has not been calibrated yet')
        return

    calibration_data = skilled_reaching_io.read_pickle(calibration_pickle_name)
    # session_metadata = {
    #                     'ratID': ratID,
    #                     'rat_num': rat_num,
    #                     'date': session_date,
    #                     'task': folder_parts[2],
    #                     'session_num': session_num,
    #                     'current': 0.
    # }
    cgroup = calibration_data['cgroup']
    cam_names = cgroup.get_names()
    num_cams = len(cam_names)
    merged = merge_rows(calibration_data['all_rows'], cam_names=cam_names)
    # imgp, extra = extract_points(merged, calibration_data['mirror_board'], min_cameras=2)

    # loop through frames, see if we can reconstruct the checkerboard
    pts_per_cam = calibration_data['mirror_board'].squaresX * calibration_data['mirror_board'].squaresY
    for rix, row in enumerate(merged):
        frame_pts = np.empty((num_cams, pts_per_cam, 2))
        frame_pts.fill(np.NaN)
        for i_cam, cam in enumerate(cam_names):
            if cam in row.keys():
                frame_pts[i_cam, :, :] = np.squeeze(row[cam]['corners'])
        pts3d = cgroup.triangulate(frame_pts, undistort=False, progress=False)
        """Given an CxNx2 array, this returns an Nx3 array of points,
        where N is the number of points and C is the number of cameras"""

        cols = ['b', 'r', 'g']
        for i_cam in range(num_cams):
            plt.scatter(frame_pts[i_cam,:,0],frame_pts[i_cam,:,1],c=cols[i_cam])
            for i_pt in range(pts_per_cam):
                if not np.isnan(frame_pts[i_cam,i_pt,0]):
                    plt.text(frame_pts[i_cam,i_pt,0], frame_pts[i_cam, i_pt, 1], '{:d}'.format(i_pt), c=cols[i_cam])

        plt.gca().invert_yaxis()
        plt.show()

        pass

    # calibration_toml_name = navigation_utilities.create_calibration_toml_name(full_calib_vid_name,
    #                                                                           calibration_files_parent)
    #
    # cgroup = CameraGroup.load(calibration_toml_name)

    pass


def test_fundamental_matrix(full_calib_vid_name, merged, frame_num, calibration_data, F, lwidth=1):

    cap = cv2.VideoCapture(full_calib_vid_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, img = cap.read()

    cap.release()

    cam_intrinsics = calibration_data['cam_intrinsics']

    img_ud = cv2.undistort(img, cam_intrinsics['mtx'], cam_intrinsics['dist'])

    w = np.shape(img_ud)[1]   # verify index
    h = np.shape(img_ud)[0]   # verify index

    found_frame_row = False
    for merged_row in merged:
        if 'direct' not in merged_row.keys():
            continue
        if merged_row['direct']['framenum'] == frame_num:
            found_frame_row = True
            break

    if not found_frame_row:
        print('matched points not found for frame {:d}'.format(frame_num))
        pass

    # find epipolar lines
    pts1 = np.squeeze(merged_row['direct']['corners'])
    if 'leftmirror' in merged_row.keys():
        pts2 = np.squeeze(merged_row['leftmirror']['corners'])
        fund_mat = F[:, :, 0]
    elif 'rightmirror' in merged_row.keys():
        pts2 = np.squeeze(merged_row['rightmirror']['corners'])
        fund_mat = F[:, :, 1]

    epilines = cv2.computeCorrespondEpilines(pts1, 1, fund_mat)
    plt.imshow(img_ud)

    line_colors = [[0., 0., 1.],
                   [0., 128. / 255., 1.],
                   [0., 200. / 255., 200. / 255.],
                   [0., 1., 0],
                   [200. / 255., 200. / 255., 0.],
                   [1., 0., 0.],
                   [1., 0., 1.]]

    for i_line, epiline in enumerate(epilines):

        # todo: figure out how to match colors to checkerboard points
        epiline = np.squeeze(epiline)
        edge_pts = cvb.find_line_edge_coordinates(epiline, (w, h))

        if not np.all(edge_pts == 0):
            col_idx = int(i_line / 7.) % 7

            try:
                plt.axline(edge_pts[0, :], edge_pts[1, :], color=line_colors[col_idx])
            except:
                pass
    plt.scatter(pts1[:, 0], pts1[:, 1])
    plt.scatter(pts2[:, 0], pts2[:, 1])

    plt.show()

    return


def check_detections(board, all_rows, cropped_vids, full_calib_vid_name, cam_intrinsics):

    cv_cap = cv2.VideoCapture(full_calib_vid_name)
    i_frame = 51

    cv_cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    res, img = cv_cap.read()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(img)

    # overlay points
    row_idx = [0, 0, 0]
    for i_view, rows in enumerate(all_rows):
        frame_nums = np.array([(i_row, row['framenum']) for i_row, row in enumerate(rows)])
        try:
            frame_row_idx = frame_nums[frame_nums[:,1]==i_frame,0][0]
        except:
            # there aren't data for this frame
            frame_row_idx = None
        row_idx[i_view] = frame_row_idx

    for i_view, rows in enumerate(all_rows):
        if not row_idx[i_view] is None:
            frame_row = rows[row_idx[i_view]]
            for ii, id in enumerate(frame_row['ids']):
                plt.text(frame_row['corners'][ii, 0, 0], frame_row['corners'][ii, 0, 1],
                         '{:d}'.format(id[0]), c='r')


    plt.show()
    pass


def find_supporting_lines(pts1, pts2):

    rounded_pts1 = pts1.astype(np.int)
    rounded_pts2 = pts2.astype(np.int)
    all_pts = np.vstack((rounded_pts1, rounded_pts2))

    full_cvx_hull = np.squeeze(cv2.convexHull(all_pts))
    full_cvx_hull = np.vstack((full_cvx_hull, full_cvx_hull[0, :]))
    # wrap around so the last point is the same as the first point for processing purposes below

    n_lines_found = 0
    n_hullpts = np.shape(full_cvx_hull)[0]

    cv_pts = np.empty((2, 2))
    cv_pts_round = np.zeros((2, 2), dtype=np.int)
    supporting_lines = np.zeros((2, 2, 2))
    # each(:,:, p) array contains[x1, y1; x2, y2] coordinates that define the endpoints of a supporting line

    for i_pt, hull_pt in enumerate(full_cvx_hull[:-1, :]):

        cv_pts_round[0, :] = hull_pt
        cv_pts_round[1, :] = full_cvx_hull[i_pt+1, :]

        cv_pts = np.empty((2, 2))

        pts_set = np.zeros(2, dtype=np.int) - 1
        # which sets (pts1 or pts2) do adjacent points in the convex hull belong to?
        pt_set_match = np.all(rounded_pts1 == cv_pts_round[0, :], axis=1)
        if np.any(pt_set_match):
            # if there is a match between set 1 and the first test point on the convex hull
            pts_set[0] = 1
            idx_in_set = np.where(pt_set_match)[0][0]
            cv_pts[0, :] = pts1[idx_in_set, :]
        else:
            # there must be a match between set 2 and the first test point on the convex hull
            pt_set_match = np.all(rounded_pts2 == cv_pts_round[0, :], axis=1)
            pts_set[0] = 2
            idx_in_set = np.where(pt_set_match)[0][0]
            cv_pts[0, :] = pts2[idx_in_set, :]   # go back to the original point instead of the rounded point

        pt_set_match = np.all(rounded_pts1 == cv_pts_round[1, :], axis=1)
        if np.any(pt_set_match):
            # if there is a match between set 1 and the second test point on the convex hull
            pts_set[1] = 1
            idx_in_set = np.where(pt_set_match)[0][0]
            cv_pts[1, :] = pts1[idx_in_set, :]   # go back to the original point instead of the rounded point
        else:
            # there must be a match between set 2 and the second test point on the convex hull
            pt_set_match = np.all(rounded_pts2 == cv_pts_round[1, :], axis=1)
            pts_set[1] = 2
            idx_in_set = np.where(pt_set_match)[0][0]
            cv_pts[1, :] = pts2[idx_in_set, :]   # go back to the original point instead of the rounded point

        if pts_set[0] != pts_set[1]:
            supporting_lines[n_lines_found, :, :] = cv_pts   # this needs to be redone so that we go back to the original points (not the integers)
            n_lines_found += 1

    # fig = plt.figure()
    # ax = fig.add_subplot()
    #
    # ax.scatter(pts1[:, 0], pts1[:, 1])
    # ax.scatter(pts2[:, 0], pts2[:, 1])
    #
    # ax.plot(supporting_lines[0, :, 0], supporting_lines[0, :, 1])
    # ax.plot(supporting_lines[1, :, 0], supporting_lines[1, :, 1])
    # ax.invert_yaxis()
    # plt.show()

    return supporting_lines


def rows_from_csvs(csv_list, board, cam_intrinsics, n_views=3):
    board_size = np.array(board.get_size())
    pts_per_view = np.prod(board_size-1)

    jpg_name = csv_list[0].replace('.csv', '.jpg')
    # read in the jpeg to get the image size
    if os.path.exists(jpg_name):
        img = cv2.imread(jpg_name)
        w = np.shape(img)[1]
        h = np.shape(img)[0]
        size = (w, h)
    else:
        size = (2040, 1024)   # hardcode default for now

    all_rows = [[] for i_view in range(n_views)]
    for csv_file in csv_list:
        csv_metadata = navigation_utilities.parse_frame_csv_name(csv_file)
        csv_table = pd.read_csv(csv_file)

        frame_corners = np.array((csv_table['X'], csv_table['Y'])).T
        n_pts = np.shape(frame_corners)[0]

        n_views_with_pts = int(n_pts / pts_per_view)

        # assume first pts_per_view points belong to the direct view
        dir_corners = frame_corners[:pts_per_view, :]
        mirr_corners = frame_corners[pts_per_view:, :]
        # do the next points belong to the left mirror or right mirror view?
        if frame_corners[pts_per_view, 0] < frame_corners[pts_per_view - 1, 0]:
            # must be the left mirror
            mirror_view_idx = 1
        else:
            mirror_view_idx = 2

        dir_corners, mirr_corners, matched_ids = match_mirror_points(dir_corners, mirr_corners, board)
        # dir_ids and mirr_ids would be the same since the points have been matched
        # rearrange the matched points so that they go left to right, top to bottom in the direct view; right to left and top to bottom in the mirror
        # then the ids should just be 0 to pts_per_view-1
        sorted_dir_corners = np.zeros((pts_per_view, 2))
        sorted_mirr_corners = np.zeros((pts_per_view, 2))
        for i_pt in range(pts_per_view):
            try:
                sorted_dir_corners[i_pt, :] = dir_corners[matched_ids == i_pt, :]
            except:
                pass
            sorted_mirr_corners[i_pt, :] = mirr_corners[matched_ids == i_pt, :]

        dir_corners_ud_norm = cv2.undistortPoints(sorted_dir_corners, cam_intrinsics['mtx'], cam_intrinsics['dist'])
        dir_corners_ud = cvb.unnormalize_points(dir_corners_ud_norm, cam_intrinsics['mtx'])
        dir_corners_ud = np.expand_dims(dir_corners_ud, 1)

        mirr_corners_ud_norm = cv2.undistortPoints(sorted_mirr_corners, cam_intrinsics['mtx'], cam_intrinsics['dist'])
        mirr_corners_ud = cvb.unnormalize_points(mirr_corners_ud_norm, cam_intrinsics['mtx'])
        mirr_corners_ud = np.expand_dims(mirr_corners_ud, 1)

        dir_filled_ud = dir_corners_ud    # not sure what "filled" is, but it seems to work with anipose
        mirr_filled_ud = mirr_corners_ud
        ids = np.arange(pts_per_view)
        dir_row = {'framenum': csv_metadata['framenum'],
                   'corners': dir_corners_ud,
                   'corners_distorted': dir_corners,
                   'filled': dir_filled_ud,
                   'ids': ids}
        all_rows[0].append(dir_row)
        mirrr_row = {'framenum': csv_metadata['framenum'],
                     'corners': mirr_corners_ud,
                     'corners_distorted': mirr_corners,
                     'filled': mirr_corners_ud,
                     'ids': ids}
        all_rows[mirror_view_idx].append(mirrr_row)

    return all_rows, size


def get_rows_cropped_vids(cropped_vids, cam_intrinsics, board, parent_directories, cgroup, skip=20, full_calib_vid_name=None):
    all_rows = []
    # check to see if there is a folder with individual images and a .csv file with points marked in fiji
    csv_list = navigation_utilities.check_for_calibration_csvs(cropped_vids[0], parent_directories)
    n_views = len(cropped_vids)
    if len(csv_list) > 0:
        all_rows, size = rows_from_csvs(csv_list, board, cam_intrinsics, n_views=n_views)
    else:
        for i_vid, cropped_vid in enumerate(cropped_vids):
            # rows_cam = []

            # if 'rm' in cropped_vid:
            #     skip = 1
            camera = cgroup.cameras[i_vid]
            rows, size = detect_video_pts(cropped_vid, board, camera, skip=skip)

            cropped_vid_metadata = navigation_utilities.parse_cropped_calibration_video_name(cropped_vid)
            # undistort the points in the rows list
            # translate points back to full frame, then undistort and unnormalize
            for i_row, row in enumerate(rows):
                orig_coord_x = row['corners'][:,:,0] + cropped_vid_metadata['crop_params'][0]
                orig_coord_y = row['corners'][:,:,1] + cropped_vid_metadata['crop_params'][2]
                orig_coord = np.hstack((orig_coord_x, orig_coord_y))

                # not sure what the difference is between 'filled' and 'corners' in each row dictionary, just trying to make
                # this work with anipose
                orig_filled_x = row['filled'][:, :, 0] + cropped_vid_metadata['crop_params'][0]
                orig_filled_y = row['filled'][:, :, 1] + cropped_vid_metadata['crop_params'][2]
                orig_filled = np.hstack((orig_filled_x, orig_filled_y))

                orig_ud_norm = cv2.undistortPoints(orig_coord, cam_intrinsics['mtx'], cam_intrinsics['dist'])
                corners_ud = cvb.unnormalize_points(orig_ud_norm, cam_intrinsics['mtx'])

                filled_ud_norm = cv2.undistortPoints(orig_filled, cam_intrinsics['mtx'], cam_intrinsics['dist'])
                filled_ud = cvb.unnormalize_points(filled_ud_norm, cam_intrinsics['mtx'])

                if isinstance(board, Checkerboard):
                    # make sure the top left corner is always labeled first in the direct view and the labels go left->right across rows
                    # make sure the top right corner is labeled first in the mirror views and labels go right->left across rows

                    fliplr = True
                    if 'dir' in cropped_vid:
                        fliplr = False

                    oc = corners_ud
                    corners_ud = reorder_checkerboard_points(corners_ud, board.get_size(), fliplr)
                    filled_ud = reorder_checkerboard_points(filled_ud, board.get_size(), fliplr)
                corners_ud = np.expand_dims(corners_ud, 1)
                filled_ud = np.expand_dims(filled_ud, 1)

                rows[i_row]['corners_distorted'] = row['corners']   # these are still in the cropped video reference frame
                rows[i_row]['corners'] = corners_ud
                rows[i_row]['filled'] = filled_ud

                # rows_cam.extend(rows)

            all_rows.append(rows)

    # put in a catch here if the number of points detected is too low
    # check_detections(board, all_rows, cropped_vids, full_calib_vid_name, cam_intrinsics)

    return all_rows

    # mirror_calib_vid_name = navigation_utilities.calib_vid_name_from_cropped_calib_vid_name(cropped_vid)
    #
    # # load original video, undistort a frame, and overlay detected points
    # full_calib_vid_name = navigation_utilities.find_mirror_calibration_video(mirror_calib_vid_name,
    #                                                                          parent_directories)
    # frame_num = 318
    # # find this frame for each "camera"
    # row_idx = np.empty(3)
    # frame_pts = []
    # for cam_rows in all_rows:
    #     # find this frame number for this camera
    #     frame_list = [row['framenum'] for row in cam_rows]
    #     try:
    #         row_idx = frame_list.index(frame_num)
    #         frame_pts.append(np.squeeze(cam_rows[row_idx]['corners']))
    #     except:
    #         pass
    #
    # test_point_id(full_calib_vid_name, frame_num, frame_pts, cam_intrinsics)
    #
    #         # if 'mirror' in cropped_vid and i_row==10:
    #         #     cap = cv2.VideoCapture(cropped_vid)
    #         #     cap.set(cv2.CAP_PROP_POS_FRAMES, row['framenum'])
    #         #     res, img = cap.read()
    #         #
    #         #     cap.release()
    #         #
    #         #     plt.figure()
    #         #     plt.imshow(img)
    #         #     for ii, id in enumerate(row['ids']):
    #         #         plt.text(rows[i_row]['corners_distorted'][ii, 0, 0], rows[i_row]['corners_distorted'][ii, 0, 1], '{:d}'.format(id), c='r')
    #         #     # plt.scatter(row['corners'][:,:,0], row['corners'][:,:,1])
    #         #     # plt.scatter(corners_ud[:,:,0], corners_ud[:,:,1])
    #         #     # plt.show()
    #         #
    #         #     plt.figure()
    #         #     img_flip = cv2.flip(img, 1)
    #         #     plt.imshow(img_flip)
    #         #     for ii, id in enumerate(row['ids']):
    #         #         plt.text(rows[i_row]['corners'][ii, 0, 0], rows[i_row]['corners'][ii, 0, 1], '{:d}'.format(id), c='r')
    #         #
    #         #
    #         #     plt.show()
    #         #     pass
    #
    # return all_rows


def test_point_id(vid_name, frame_num, pts, cam_intrinsics):
    cap = cv2.VideoCapture(vid_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    res, img = cap.read()

    cap.release()

    img_ud = cv2.undistort(img, cam_intrinsics['mtx'], cam_intrinsics['dist'])

    plt.imshow(img_ud)

    if np.shape(pts[0]) != np.shape(pts[1]):
        # the same number of points wasn't found in each view
        for view_pts in pts:
            for i_pt in range(np.shape(view_pts)[0]):
                plt.text(view_pts[i_pt,0], view_pts[i_pt, 1], '{:d}'.format(i_pt), c='b')
    else:
        pts = np.squeeze(pts)
        if pts.ndim == 3:
            for pts_array in pts:
                for i_pt, pt in enumerate(pts_array):
                    plt.text(pt[0], pt[1], '{:d}'.format(i_pt), c='r')
                plt.scatter(pts_array[:, 0], pts_array[:, 1])
        else:
            for i_pt, pt in enumerate(pts):
                plt.text(pt[0], pt[1], '{:d}'.format(i_pt), c='r')
            plt.scatter(pts[:, 0], pts[:, 1])

    plt.show()

    pass


def get_rows_cropped_vids_anipose(cropped_vids, cam_intrinsics, board, parent_directories):
    # all_rows = []
    #
    # for cix, (cam, cam_videos) in enumerate(zip(self.cameras, videos)):
    #     rows_cam = []
    #     for vnum, vidname in enumerate(cam_videos):
    #         if verbose: print(vidname)
    #         rows = board.detect_video(vidname, prefix=vnum, progress=verbose)
    #         if verbose: print("{} boards detected".format(len(rows)))
    #         rows_cam.extend(rows)
    #     all_rows.append(rows_cam)
    #
    # return all_rows

    # currently a legacy version where mirror views are cropped out and flipped left-right with the idea that anipose
    # will consider each view as a virtual camera. That didn't work very well, though.

    all_rows = []
    for cropped_vid in cropped_vids:
        rows_cam = []
        rows, size = detect_video_pts(cropped_vid, board)

        cropped_vid_metadata = navigation_utilities.parse_cropped_calibration_video_name(cropped_vid)
        #tooo: undistort the points in the rows list, and flip them left-right if from a mirror view

        # translate points back to full frame, then undistort, unnormalize, translate back into cropped frame, and fliplr if in a mirror view
        for i_row, row in enumerate(rows):
            orig_coord_x = row['corners'][:,:,0] + cropped_vid_metadata['crop_params'][0]
            orig_coord_y = row['corners'][:,:,1] + cropped_vid_metadata['crop_params'][2]
            orig_coord = np.hstack((orig_coord_x, orig_coord_y))

            orig_ud_norm = cv2.undistortPoints(orig_coord, cam_intrinsics['mtx'], cam_intrinsics['dist'])
            orig_ud = cvb.unnormalize_points(orig_ud_norm, cam_intrinsics['mtx'])

            corners_ud_x = orig_ud[:,0] - cropped_vid_metadata['crop_params'][0]
            corners_ud_y = orig_ud[:,1] - cropped_vid_metadata['crop_params'][2]

            fliplr = False
            if 'mirr' in cropped_vid:
                # need to flip the undistorted points left-right for proper camera group calibration
                # w = width of the cropped image
                w = cropped_vid_metadata['crop_params'][1] - cropped_vid_metadata['crop_params'][0] + 1
                corners_ud_x = w - corners_ud_x
                fliplr = True

            corners_ud = np.array([[c_x, x_y] for c_x, x_y in zip(corners_ud_x, corners_ud_y)])
            if isinstance(board, Checkerboard):
                # make sure the top left corner is always labeled first and the labels go across rows
                corners_ud = reorder_checkerboard_points(corners_ud, board.get_size(), fliplr)
            corners_ud = np.expand_dims(corners_ud, 1)

            rows[i_row]['corners_distorted'] = row['corners']
            rows[i_row]['corners'] = corners_ud

            rows_cam.extend(rows)

        all_rows.append(rows_cam)

            # if 'mirror' in cropped_vid and i_row==10:
            #     cap = cv2.VideoCapture(cropped_vid)
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, row['framenum'])
            #     res, img = cap.read()
            #
            #     cap.release()
            #
            #     plt.figure()
            #     plt.imshow(img)
            #     for ii, id in enumerate(row['ids']):
            #         plt.text(rows[i_row]['corners_distorted'][ii, 0, 0], rows[i_row]['corners_distorted'][ii, 0, 1], '{:d}'.format(id), c='r')
            #     # plt.scatter(row['corners'][:,:,0], row['corners'][:,:,1])
            #     # plt.scatter(corners_ud[:,:,0], corners_ud[:,:,1])
            #     # plt.show()
            #
            #     plt.figure()
            #     img_flip = cv2.flip(img, 1)
            #     plt.imshow(img_flip)
            #     for ii, id in enumerate(row['ids']):
            #         plt.text(rows[i_row]['corners'][ii, 0, 0], rows[i_row]['corners'][ii, 0, 1], '{:d}'.format(id), c='r')
            #
            #
            #     plt.show()
            #     pass

    return all_rows

            # do the ID's need to be rearranged? Is that different for charuco vs checkerboard calibration?


            # comment in lines below to display on full original image
            # load original video frame
            # if 'direct' in cropped_vid:
            #     orig_video = navigation_utilities.find_calibration_video(cropped_vid_metadata, parent_directories['calibration_vids_parent'])
            #     cap = cv2.VideoCapture(orig_video)
            #
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, row['framenum'])
            #     res, img = cap.read()
            #
            #     plt.imshow(img)
            #     for ii, id in enumerate(row['ids']):
            # #         plt.text(orig_coord[ii,0], orig_coord[ii,1], '{:d}'.format(id), c='r')
            #         plt.text(reordered_orig_pts[ii,0], reordered_orig_pts[ii,1], '{:d}'.format(id), c='r')
            #     # plt.scatter(orig_coord[:,0], orig_coord[:,1])
            #     # plt.scatter(orig_ud[:,0], orig_ud[:,1])
            #
            #     cap.release()
            #     plt.show()
            #     pass
            # if 'mirror' in cropped_vid:
            #     orig_video = navigation_utilities.find_calibration_video(cropped_vid_metadata, parent_directories['calibration_vids_parent'])
            #     cap = cv2.VideoCapture(orig_video)
            #
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, row['framenum'])
            #     res, img = cap.read()
            #
            #     plt.imshow(img)
            #     for ii, id in enumerate(row['ids']):
            #         # plt.text(orig_coord[ii,0], orig_coord[ii,1], '{:d}'.format(id), c='r')
            #         plt.text(reordered_orig_pts[ii, 0], reordered_orig_pts[ii, 1], '{:d}'.format(id), c='r')
            #     # plt.scatter(orig_coord[:,0], orig_coord[:,1])
            #     # plt.scatter(orig_ud[:,0], orig_ud[:,1])
            #
            #     cap.release()
            #     plt.show()
            #     pass
            # if 'mirror' in cropped_vid:
            #     cap = cv2.VideoCapture(cropped_vid)
            #     cap.set(cv2.CAP_PROP_POS_FRAMES, row['framenum'])
            #     res, img = cap.read()
            #
            #     plt.imshow(img)
            #     plt.scatter(row['corners'][:,:,0], row['corners'][:,:,1])
            #     plt.scatter(corners_ud[:,:,0], corners_ud[:,:,1])
            #
            #     cap.release()
            #     pass




    # all_rows = []
    #
    # for cix, (cam, cam_videos) in enumerate(zip(self.cameras, videos)):
    #     rows_cam = []
    #     for vnum, vidname in enumerate(cam_videos):
    #         if verbose: print(vidname)
    #         rows = board.detect_video(vidname, prefix=vnum, progress=verbose)
    #         if verbose: print("{} boards detected".format(len(rows)))
    #         rows_cam.extend(rows)
    #     all_rows.append(rows_cam)
    #
    # return all_rows


def point_to_line_distance(line_pts, test_pt):
    '''

    :param line_pts: 2 x 2 or 2 x 3 matrix where each row is a point that defines the line
    :param test_pt: the point to which we are trying to find the distance
    :return:
    '''

    if np.shape(line_pts)[1] == 2:
        d = calc2Ddistance(line_pts[0, :], line_pts[1, :], test_pt)
    elif np.shape(line_pts)[1] == 3:
        d = calc3Ddistance(line_pts[0, :], line_pts[1, :], test_pt)

    return d


def calc2Ddistance(Q1, Q2, test_pt):
    d = abs(np.linalg.det(np.vstack((Q2 - Q1, test_pt - Q1)))) / np.linalg.norm(Q2 - Q1)

    return d


def calc3Ddistance(Q1, Q2, test_pt):

    # hasn't been tested
    d = np.linalg.norm(np.cross(Q2 - Q1, test_pt - Q1)) / np.linalg.norm(Q2 - Q1)

    return d


def match_mirror_points(dir_corners, mirr_corners, board, dir_max_dist_from_line=5, mirr_max_dist_from_line=3):
    # build this based on old matlab code
    n_points = np.shape(mirr_corners)[0]
    remaining_dir_corners = np.copy(dir_corners)
    remaining_mirr_corners = np.copy(mirr_corners)
    n_matches = 0
    match_idx = np.zeros((n_points, 2), dtype=np.int)

    while np.shape(remaining_dir_corners)[0] > 0:

        if np.shape(remaining_dir_corners)[0] == 1:
            # only one point left
            dir_row = np.where((dir_corners == remaining_dir_corners).all(axis=1))
            mir_row = np.where((mirr_corners == remaining_mirr_corners).all(axis=1))

            match_idx[n_matches, 0] = dir_row
            match_idx[n_matches, 1] = mir_row

            remaining_dir_corners = np.array([])
            remaining_mirr_corners = np.array([])

        supporting_lines = find_supporting_lines(remaining_dir_corners, remaining_mirr_corners)

        for i_line in range(2):
            test_pt1 = np.squeeze(supporting_lines[i_line, 0, :])
            test_pt2 = np.squeeze(supporting_lines[i_line, 1, :])

            # is test_pt1 in the direct or mirror view?
            test_is_dir_pt = np.any(np.all(dir_corners == test_pt1, axis=1))
            if test_is_dir_pt:
                # test_pt1 is in the direct view, test_pt2 is in the mirror view
                dir_row = np.where(np.all(dir_corners == test_pt1, axis=1))[0][0]
                mir_row = np.where(np.all(mirr_corners == test_pt2, axis=1))[0][0]

                try:
                    remaining_dir_row = np.where(np.all(remaining_dir_corners == test_pt1, axis=1))[0][0]
                except:
                    pass
                remaining_mirr_row = np.where(np.all(remaining_mirr_corners == test_pt2, axis=1))[0][0]
            else:
                # test_pt2 is in the direct view, test_pt1 is in the mirror view
                mir_row = np.where(np.all(mirr_corners == test_pt1, axis=1))[0][0]
                dir_row = np.where(np.all(dir_corners == test_pt2, axis=1))[0][0]

                remaining_dir_row = np.where(np.all(remaining_dir_corners == test_pt2, axis=1))[0][0]
                remaining_mirr_row = np.where(np.all(remaining_mirr_corners == test_pt1, axis=1))[0][0]

            # have potential matches - is it possible that there is a better match that lies along the same line (hidden
            # by noise in how accurately points were marked/identifed)?
            n_remaining_points = np.shape(remaining_dir_corners)[0]
            mirr_dist_from_line = np.empty(n_remaining_points)
            dir_dist_from_line = np.empty(n_remaining_points)
            mirr_dist_from_line[:] = np.nan
            dir_dist_from_line[:] = np.nan
            supporting_line = np.squeeze(supporting_lines[i_line, :, :])
            for i_corner in range(n_remaining_points):
                try:
                    mirr_dist_from_line[i_corner] = point_to_line_distance(supporting_line, remaining_mirr_corners[i_corner, :])
                except:
                    pass
                dir_dist_from_line[i_corner] = point_to_line_distance(supporting_line,
                                                                       remaining_dir_corners[i_corner, :])

            mirr_candidates = np.where(mirr_dist_from_line < mirr_max_dist_from_line)[0]
            dir_candidates = np.where(dir_dist_from_line < dir_max_dist_from_line)[0]

            if len(mirr_candidates) != len(dir_candidates) or len(mirr_candidates) == 1:
                # only one candidate point in each view or there are mutliple candidates in one view but only one in the other view?
                # I don't think this is quite right, but it seems to work well enough
                match_idx[n_matches, 0] = dir_row
                match_idx[n_matches, 1] = mir_row
                n_matches += 1

                # remove the rows that were just matched
                remaining_dir_corners = np.delete(remaining_dir_corners, remaining_dir_row, axis=0)
                remaining_mirr_corners = np.delete(remaining_mirr_corners, remaining_mirr_row, axis=0)
                continue

            # multiple candidate matches along the supporting line
            mirr_dir_distance = np.zeros((len(mirr_candidates), len(mirr_candidates)))
            for i_dirpt in range(len(mirr_candidates)):
                for i_mirrpt in range(len(mirr_candidates)):
                    mirr_dir_distance[i_dirpt, i_mirrpt] = np.linalg.norm(remaining_dir_corners[dir_candidates[i_dirpt], :] -
                                                                          remaining_mirr_corners[mirr_candidates[i_mirrpt], :])

            # which direct/mirror points are closest together? (they're a match due to mirror symmetry)
            m, n = (mirr_dir_distance == np.min(mirr_dir_distance)).nonzero()  # m is the row, n the column where the minimum is
            cur_dirpt = remaining_dir_corners[dir_candidates[m], :]
            cur_mirrpt = remaining_mirr_corners[mirr_candidates[n], :]

            mir_row = np.where(np.all(mirr_corners == cur_mirrpt, axis=1))[0][0]
            dir_row = np.where(np.all(dir_corners == cur_dirpt, axis=1))[0][0]

            match_idx[n_matches, 0] = dir_row
            match_idx[n_matches, 1] = mir_row
            n_matches += 1

            remaining_dir_row = np.where(np.all(remaining_dir_corners == cur_dirpt, axis=1))[0][0]
            remaining_mirr_row = np.where(np.all(remaining_mirr_corners == cur_mirrpt, axis=1))[0][0]

            # remove the rows that were just matched
            remaining_dir_corners = np.delete(remaining_dir_corners, remaining_dir_row, axis=0)
            remaining_mirr_corners = np.delete(remaining_mirr_corners, remaining_mirr_row, axis=0)

    # dir_corners, mirr_corners, dir_ids, mirr_ids = match_mirror_points(mirr_corners, dir_corners)
    matched_dir_corners = dir_corners[match_idx[:, 0], :]
    matched_mirr_corners = mirr_corners[match_idx[:, 1], :]

    dir_ids = find_pt_ids(matched_dir_corners, board)
    # since the direct and mirror points should be matched, the points order will be the same for both

    return matched_dir_corners, matched_mirr_corners, dir_ids


def find_top_left_corner(corners):
    # have to manipulate x-values because sometimes vertical overwhelms the horizontal if the chessboard in angled too
    # much away from the camera
    stretched_corners = np.copy(corners)
    stretched_corners[:, 0] = (corners[:, 0] - np.min(corners[:, 0])) * 1
    top_left_stretched_pt = stretched_corners[0, :]
    top_left_idx = 0
    for i_corner, corner in enumerate(stretched_corners):
        if np.sum(corner) < np.sum(top_left_stretched_pt):
            top_left_stretched_pt = corner
            top_left_idx = i_corner

    top_left_pt = corners[top_left_idx, :]

    return top_left_pt, top_left_idx


def find_bottom_right_corner(corners):
    botom_right_pt = corners[0, :]
    bottom_right_idx = 0
    for i_corner, corner in enumerate(corners):
        if np.sum(corner) > np.sum(botom_right_pt):
            botom_right_pt = corner
            bottom_right_idx = i_corner

    return botom_right_pt, bottom_right_idx


def find_pt_ids(corners, board):
    board_size = np.array(board.get_size()) - 1
    n_rows = board_size[1]
    n_cols = board_size[0]
    # I suppose the above could be wrong if the geometry changes, but I think this is safe for now, at least for the mirror calibration -DL 12/17/2024

    n_pts = np.shape(corners)[0]
    pt_ids = np.zeros(n_pts, dtype=np.int)

    remaining_corners = np.copy(corners)
    pt_id_idx = 0
    for i_row in range(n_rows):
        # find the top left corner
        if i_row < n_rows - 1:
            top_left, top_left_idx = find_top_left_corner(remaining_corners)
        else:
            # if we're on the last row, the algorithm for finding the top left might fail if the row is angled upwards too steeply
            # just take the left-most point
            top_left = remaining_corners[remaining_corners[:, 0] == min(remaining_corners[:, 0]), :]

        cur_pt = top_left
        top_left_idx = np.where(np.all(corners == top_left, axis=1))[0][0]
        pt_ids[top_left_idx] = pt_id_idx
        pt_id_idx += 1
        cur_remaining_corner_idx = np.where(np.all(remaining_corners == cur_pt, axis=1))[0][0]
        remaining_corners = np.delete(remaining_corners, cur_remaining_corner_idx, axis=0)
        for i_col in range(1, n_cols):   # start with 1 because we already captured the first point in the row
            # the next point will be the closest point to the right. Maybe there's some weird geometry where that isn't true, but should be good enough
            pts_to_right = remaining_corners[remaining_corners[:, 0] > cur_pt[0], :]

            # calculate distance to each of the other points to the right of this one
            d_right = np.linalg.norm(pts_to_right - cur_pt, axis=1)
            try:
                cur_pt = np.squeeze(pts_to_right[d_right == np.min(d_right), :])
            except:
                pass
            cur_corner_idx = np.where(np.all(corners == cur_pt, axis=1))[0][0]

            pt_ids[cur_corner_idx] = pt_id_idx
            pt_id_idx += 1

            cur_remaining_corner_idx = np.where(np.all(remaining_corners == cur_pt, axis=1))[0][0]
            remaining_corners = np.delete(remaining_corners, cur_remaining_corner_idx, axis=0)



    # if top_left[1] == min(corners[:, 1]):
    #     # the top left corner is the highest point in the top row
    #     n_remaining_corners = n_pts
    #     cur_pt = top_left
    #     top_left_idx = np.where(np.all(corners == top_left, axis=1))[0][0]
    #     pt_ids[top_left_idx] = 0
    #     pt_id_idx = 1
    #     for i_row in range(board_size[1]):
    #         # the next board_size[0] - 1 points should be the two points
    #         if top_left[1] == min(remaining_corners[:, 1]):
    #             min_remaining_y_idx = remaining_corners[:, 1] == np.min(remaining_corners[:, 1])
    #             cur_pt = remaining_corners[min_remaining_y_idx, :]
    #
    #
    #     while n_remaining_corners > 1:
    #         # remove the point that already was allocated
    #         remaining_cur_pt_idx = np.where(np.all(remaining_corners == cur_pt, axis=1))[0][0]
    #         remaining_corners = np.delete(remaining_corners, remaining_cur_pt_idx, axis=0)
    #         n_remaining_corners -= 1
    #
    #         # for this tilt, each successively lower point should be the next point in the list
    #         min_remaining_y_idx = remaining_corners[:, 1] == np.min(remaining_corners[:, 1])
    #         cur_pt = remaining_corners[min_remaining_y_idx, :]
    #         try:
    #             cur_pt_idx = np.where(np.all(corners == cur_pt, axis=1))[0][0]
    #         except:
    #             pass
    #         pt_ids[cur_pt_idx] = pt_id_idx
    #         pt_id_idx += 1
    #
    # else:
    #     # I'm pretty sure this means rectangle is tilted up to the right
    #     # this means the other points in the top row will all be higher (lower y) than the top left point
    #     n_remaining_corners = n_pts
    #     cur_top_left = top_left
    #     pt_id_idx = 0
    #     while n_remaining_corners > 0:
    #         # assign the top left point index into the pt_ids vector
    #         top_left_idx = np.where(np.all(corners == cur_top_left, axis=1))[0][0]
    #         pt_ids[top_left_idx] = pt_id_idx
    #         pt_id_idx += 1
    #
    #         # remove the top left point from the remaining points
    #         cur_top_left_idx = np.where(np.all(remaining_corners == cur_top_left, axis=1))[0][0]
    #         remaining_corners = np.delete(remaining_corners, cur_top_left_idx, axis=0)
    #
    #         # find points among the ones that are left that are higher than the current leftmost point
    #         row_pts_bool = remaining_corners[:, 1] < cur_top_left[1]
    #         cur_row_pts = remaining_corners[row_pts_bool, :]
    #         cur_row_idx = np.where(row_pts_bool)[0]
    #
    #         # find the indices of these points in the original corners array
    #         corners_row_idxs = [np.where(np.all(corners == cur_row_pt, axis=1))[0][0] for cur_row_pt in cur_row_pts]
    #         cur_pt_order = np.argsort(corners[corners_row_idxs, 0])
    #
    #         for row_pt in cur_pt_order:
    #             pt_ids[corners_row_idxs[row_pt]] = pt_id_idx
    #             pt_id_idx += 1
    #
    #         # remove points we've already sorted into rows
    #         remaining_corners = np.delete(remaining_corners, cur_row_idx, axis=0)
    #         n_remaining_corners = np.shape(remaining_corners)[0]
    #         if n_remaining_corners > 0:
    #             cur_top_left, _ = find_top_left_corner(remaining_corners)

    ## for testing whether the order came out right
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(corners[:, 0], corners[:, 1])
    # for i_corner, corner in enumerate(corners):
    #     ax.text(corner[0], corner[1], '{:d}'.format(pt_ids[i_corner]))
    # ax.invert_yaxis()

    return pt_ids


def detect_video_pts(calibration_video, board, camera, prefix=None, skip=20, progress=True, min_rows_detected=20):
    # adapted from anipose
    cap = cv2.VideoCapture(calibration_video)

    _, cvid_name = os.path.split(calibration_video)

    if not cap.isOpened():
        raise FileNotFoundError(f'missing video file "{calibration_video}"')

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)

    if length < 10:
        length = int(1e9)
        progress = False
    rows = []

    go = int(skip / 2)

    if progress:
        it = trange(length, ncols=70)
    else:
        it = range(length)

    for framenum in it:
        ret, frame = cap.read()
        if not ret:
            break
        if framenum % skip != 0 and go <= 0:
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if isinstance(board, CharucoBoard):
            if 'rm' in cvid_name or 'lm' in cvid_name or 'mirror' in cvid_name:
                ismirrorview = True
            else:
                ismirrorview = False
            if ismirrorview:
                # aruco markers are flipped left-right in the mirror, so need to flip the frame image to detect them
                orig_frame = np.copy(frame)
                frame = cv2.flip(frame, 1)
            charucoCorners, charucoIds, markerCorners, markerIds = detect_markers(frame, board, camera=camera)

            if ismirrorview and charucoCorners is not None and len(charucoCorners) > 0:
                # now need to flip the identified points left-right to match in the original image
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                for i_cc, cc in enumerate(charucoCorners):
                    cc_x = np.squeeze(cc)[0]
                    charucoCorners[i_cc, 0, 0] = w - cc_x

            if charucoCorners is not None and len(charucoCorners) > 0:
                if prefix is None:
                    key = framenum
                else:
                    key = (prefix, framenum)
                go = int(skip / 2)
                row = {'framenum': key, 'corners': charucoCorners, 'ids': charucoIds}
                rows.append(row)

            go = max(0, go - 1)
        elif isinstance(board, Checkerboard):
            corners, ids = board.detect_image(frame, subpix=True)

            # reorder to make sure that the top left is corner 0

            if corners is not None and len(corners) > 0:
                if prefix is None:
                    key = framenum
                else:
                    key = (prefix, framenum)
                go = int(skip / 2)
                row = {'framenum': key, 'corners': corners, 'ids': ids}
                rows.append(row)

            go = max(0, go - 1)

    cap.release()

    rows = board.fill_points_rows(rows)

    # if not enough rows, save frames for manual point extraction
    if len(rows) < min_rows_detected:
        # todo: check to see if the board corners have already been manually identified
        crop_videos.write_video_frames(calibration_video, img_type='.jpg')

    return rows, size


def test_img(frame, corners):
    corners = np.squeeze(corners)
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.imshow(frame)
    n_corners = np.shape(corners)[0]

    for i_pt in range(n_corners):
        ax.text(corners[i_pt, 0], corners[i_pt, 1], '{:d}'.format(i_pt))

    plt.show()


def reorder_checkerboard_points(corners, size, fliplr):

    # reordeer chessboard corners so first point is top left
    if np.ndim(corners) == 3:
        first_corner = corners[0, 0]
        last_corner = corners[-1, 0]
    else:
        first_corner = corners[0,:]
        last_corner = corners[-1,:]
    num_corners = np.prod(size)

    if not fliplr:
        # what are the relative positions of the first and last points?
        if first_corner[0] < last_corner[0] and first_corner[1] < last_corner[1]:
            # first corner is top left; then it will go left->right (I think)
            # so the order should already be correct
            id_order = np.arange(num_corners)
        elif first_corner[0] > last_corner[0] and first_corner[1] < last_corner[1]:
            # first corner is top right; then it will go down the column (I think)
            # this seems to be working correctly
            id_order = []
            # the first index in size is the row because it always starts counting along the first axis in size
            for i_row in range(size[0]):
                for i_col in range(size[1]):
                    id_order.append((size[1]-i_col-1) * size[0] + i_row)
            id_order = np.array(id_order)
        elif first_corner[0] > last_corner[0] and first_corner[1] > last_corner[1]:
            # first corner is bottom right; then it will go left along the bottom row (I think)
            id_order = []
            # this time the rows are the second index in size
            for i_row in range(size[1]):
                for i_col in range(size[0]):
                    id_order.append(((size[1]-i_row) * size[0]) - i_col - 1)
            id_order = np.array(id_order)
        elif first_corner[0] < last_corner[0] and first_corner[1] > last_corner[1]:
            # first corner is bottom left; then it will go up along the left column (I think)
            id_order = []
            # this time the rows are the first index in size because going up the columns
            for i_row in range(size[0]):
                for i_col in range(size[1]):
                    id_order.append((size[0] * (i_col+1)) - i_row - 1)
            id_order = np.array(id_order)

    else:
        # the points have been flipped left-right, meaning that the points order has also flipped
        # what are the relative positions of the first and last points?
        if first_corner[0] < last_corner[0] and first_corner[1] < last_corner[1]:
            # first corner is top left; should be going from left to right and down. Want them to start at top right
            # and go left then down(I think)
            id_order = []
            # since it's going top-->bottom, first index in size is the number of rows
            for i_row in range(size[1]):
                for i_col in range(size[0]):
                    id_order.append((size[0] - i_col - 1) + (i_row * size[0]))
            id_order = np.array(id_order)
        elif first_corner[0] > last_corner[0] and first_corner[1] < last_corner[1]:
            # first corner is top right; then it will go down the column (I think)
            id_order = []
            # since it's going right-->left, first index in size is the number of columns
            for i_row in range(size[0]):
                for i_col in range(size[1]):
                    id_order.append((i_col * size[0]) + i_row)
            id_order = np.array(id_order)
        elif first_corner[0] > last_corner[0] and first_corner[1] > last_corner[1]:
            # first corner is bottom right; then it will go left along the bottom row (I think)
            id_order = []
            # first index in size is the number of columns
            for i_row in range(size[1]):
                for i_col in range(size[0]):
                    id_order.append((size[0] * size[1] - i_row * size[0]) - (size[0] - i_col))
            id_order = np.array(id_order)
        elif first_corner[0] < last_corner[0] and first_corner[1] > last_corner[1]:
            # first corner is bottom left; then it will go up along the left column (I think)
            id_order = []
            # since it's going bottom-->top, first index in size is the number of rows
            for i_row in range(size[0]):
                for i_col in range(size[1]):
                    id_order.append((size[1] * size[0] - i_row - 1) - size[0] * i_col)
            id_order = np.array(id_order)

    reordered_corners = np.array([corners[ii, :] for ii in id_order])

    return reordered_corners


def reorder_checkerboard_points_anipose(corners, size, fliplr):

    # reorder chessboard corners so first point is top left
    if np.ndim(corners) == 3:
        first_corner = corners[0, 0]
        last_corner = corners[-1, 0]
    else:
        first_corner = corners[0,:]
        last_corner = corners[-1,:]
    num_corners = np.prod(size)

    if not fliplr:
        # what are the relative positions of the first and last points?
        if first_corner[0] < last_corner[0] and first_corner[1] < last_corner[1]:
            # first corner is top left; then it will go left->right (I think)
            # so the order should already be correct
            id_order = np.arange(num_corners)
        elif first_corner[0] > last_corner[0] and first_corner[1] < last_corner[1]:
            # first corner is top right; then it will go down the column (I think)
            # this seems to be working correctly
            id_order = []
            # the first index in size is the row because it always starts counting along the first axis in size
            for i_row in range(size[0]):
                for i_col in range(size[1]):
                    id_order.append((size[1]-i_col-1) * size[0] + i_row)
            id_order = np.array(id_order)
        elif first_corner[0] > last_corner[0] and first_corner[1] > last_corner[1]:
            # first corner is bottom right; then it will go left along the bottom row (I think)
            id_order = []
            # this time the rows are the second index in size
            for i_row in range(size[1]):
                for i_col in range(size[0]):
                    id_order.append(((size[1]-i_row) * size[0]) - i_col - 1)
            id_order = np.array(id_order)
        elif first_corner[0] < last_corner[0] and first_corner[1] > last_corner[1]:
            # first corner is bottom left; then it will go up along the left column (I think)
            id_order = []
            # this time the rows are the first index in size because going up the columns
            for i_row in range(size[0]):
                for i_col in range(size[1]):
                    id_order.append((size[0] * (i_col+1)) - i_row - 1)
            id_order = np.array(id_order)

    else:
        # the points have been flipped left-right, meaning that the points order has also flipped
        # what are the relative positions of the first and last points?
        if first_corner[0] < last_corner[0] and first_corner[1] < last_corner[1]:
            # first corner is top left; this means it used to be top right; then it will go top->bottom (I think)
            id_order = []
            # since it's going top-->bottom, first index in size is the number of rows
            for i_row in range(size[0]):
                for i_col in range(size[1]):
                    id_order.append((size[0] * i_col) + i_row)
            id_order = np.array(id_order)
        elif first_corner[0] > last_corner[0] and first_corner[1] < last_corner[1]:
            # first corner is top right; this means it used to be top left; then it will go right to left (I think)
            id_order = []
            # since it's going right-->left, first index in size is the number of columns
            for i_row in range(size[1]):
                for i_col in range(size[0]):
                    id_order.append(i_row * size[0] + (size[0] - i_col - 1))
            id_order = np.array(id_order)
        elif first_corner[0] > last_corner[0] and first_corner[1] > last_corner[1]:
            # first corner is bottom right; this means it used to be bottom left; then it will go up the columns (I think)
            id_order = []
            # since it's going bottom-->top, first index in size is the number of rows
            for i_row in range(size[0]):
                for i_col in range(size[1]):
                    id_order.append(np.prod(size) - i_col * size[0] - i_row - 1)
            id_order = np.array(id_order)
        elif first_corner[0] < last_corner[0] and first_corner[1] > last_corner[1]:
            # first corner is bottom left; this means it used to be bottom right; then it will go left to right (I think)
            id_order = []
            # since it's going left-->right, first index in size is the number of columns
            for i_row in range(size[1]):
                for i_col in range(size[0]):
                    id_order.append((size[1] * (size[0]-i_row-1)) + i_col)
            id_order = np.array(id_order)

    reordered_corners = np.array([corners[ii, :] for ii in id_order])

    return reordered_corners


def detect_image(image, board, subpix=True):
    # adapted from anipose
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    size = (board.squaresX, board.squaresY)

    pattern_was_found, corners = cv2.findChessboardCorners(gray, size, self.DETECT_PARAMS)

    if corners is not None:

        if subpix:
            corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.SUBPIX_CRITERIA)

def CharucoBoardObject_from_AniposeBoard(board):
    # to create a CharucoDetector object need an aruco.CharucoBoard object
    ch_board = aruco.CharucoBoard((board.squaresX, board.squaresY),
                                   board.square_length,
                                   board.marker_length,
                                   board.dictionary)

    return ch_board


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def detect_markers(image, board, camera=None, refine=True):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    params = aruco.DetectorParameters()

    # values taken from anipose detect_markers function in board class
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 700
    params.adaptiveThreshWinSizeStep = 50
    # params.minMarkerPerimeterRate = 0.05
    params.adaptiveThreshConstant = 0

    ch_detector = aruco.CharucoDetector(board.board)
    ar_detector = aruco.ArucoDetector(board.board.getDictionary(), detectorParams=params)

    markerCorners, markerIds, rejectedImgPoints = ar_detector.detectMarkers(gray)
    temp = unsharp_mask(gray, sigma=1., amount=4.)
    # a,b,c = ar_detector.detectMarkers(temp)

    if refine:
        # detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        #     aruco.refineDetectedMarkers(gray, board.board, charucoCorners, charucoIds,
        #                                 rejectedImgPoints,
        #                                 K, D,
        #                                 parameters=params)
        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = ar_detector.refineDetectedMarkers(gray, board.board, markerCorners, markerIds, rejectedImgPoints)
        # d,e,f,g = ar_detector.refineDetectedMarkers(temp, board.board, a, b, c)
    else:
        detectedCorners, detectedIds = markerCorners, markerIds

    charucoCorners, charucoIds, markerCorners, markerIds  = ch_detector.detectBoard(gray, markerCorners=detectedCorners, markerIds=detectedIds)
    # charuco_img = aruco.drawDetectedCornersCharuco(gray, charucoCorners, charucoIds, (255, 0, 0))

    # todo: do we need to refine the detected corners?
    # if np.shape(charucoCorners)[0] < 30:
    #     fig = plt.figure()
    #     ax = fig.add_subplot()
    #
    #     fig2 = plt.figure()
    #     ax2 = fig2.add_subplot()
    #
    #     detect_markers_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     detect_markers_img = aruco.drawDetectedMarkers(detect_markers_img, detectedCorners, detectedIds)
    #     ax2.imshow(detect_markers_img)
    #
    #     detect_charuco_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     detect_charuco_img = aruco.drawDetectedCornersCharuco(detect_charuco_img, charucoCorners, charucoIds)
    #     ax.imshow(detect_charuco_img)

    return charucoCorners, charucoIds, markerCorners, markerIds


def calibrate_Burgess_session(calibration_data_name, vid_pair, parent_directories, num_frames_for_intrinsics=50, min_frames_for_intrinsics=10, num_frames_for_stereo=20, min_frames_for_stereo=5, use_undistorted_pts_for_stereo_cal=True):
    '''

    :param calibration_data_name:
    :param num_frames_for_intrinsics:
    :param min_frames_for_intrinsics: minimum number of frames to use for intrinsics calibration (if number of valid frames
        is less than num_frames_for_intrinsics, it will use all available frames. If min_frames_for_intrinsics is greater
        than the number of available frames, calibration will be skipped).
    :return:
    '''

    # create video objects for each calibration_video
    vid_obj = []
    for i_vid, vid_name in enumerate(vid_pair):
        vid_obj.append(cv2.VideoCapture(vid_name))

    FFM_tolerance = 0.1
    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_ASPECT_RATIO
    STEREO_FLAGS = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_FIX_PRINCIPAL_POINT
    # initialize camera intrinsics to have an aspect ratio of 1 and assume the center of the 1280 x 1024 field is [639.5, 511.5]
    init_mtx = np.array([[1100, 0, 639.5],[0, 1100, 511.5],[0, 0, 1]])
    cal_data = skilled_reaching_io.read_pickle(calibration_data_name)

    num_cams = len(cal_data['cam_objpoints'])
    if 'mtx' not in cal_data.keys():
        # if there isn't already an intrinsic calibration, initialize lists for camera intrinsic matrices (mtx),
        # distortion coefficients, and number of frames used to calculate intrinsics
        cal_data['mtx'] = []
        cal_data['dist'] = []
        cal_data['frame_nums_for_intrinsics'] = []
    for i_cam in range(num_cams):
        current_cam = cal_data['calvid_metadata'][i_cam]['cam_num']

        session_date_string = navigation_utilities.datetime_to_string_for_fname(
            cal_data['calvid_metadata'][i_cam]['session_datetime'])
        # if intrinsics have already been calculated for this camera, skip
        if i_cam < len(cal_data['mtx']):
            # this camera number is smaller than the number of cameras for which intrinsics have been stored
            # since this one has already been calculated, skip
            print('intrinsics already calculated for {}, camera {:02d}'.format(session_date_string, current_cam))
            continue

        print('working on {}, camera {:02d} intrinsics calibration'.format(session_date_string, current_cam))

        # select num_frames_for_intrinsics evenly spaced frames
        cam_objpoints = cal_data['cam_objpoints'][i_cam]
        cam_imgpoints = cal_data['cam_imgpoints'][i_cam]

        total_valid_frames = np.shape(cam_objpoints)[0]
        num_frames_to_use = min(num_frames_for_intrinsics, total_valid_frames)
        if num_frames_to_use < min_frames_for_intrinsics:
            mtx = np.zeros((3, 3))
            dist = np.zeros((1, 5))
            frame_numbers = []
        else:
            objpoints_for_intrinsics, imgpoints_for_intrinsics, frame_numbers = select_cboards_for_calibration(cam_objpoints, cam_imgpoints, num_frames_to_use)

            # todo: consider testing for poorly-identified chessboard points
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_for_intrinsics,
                                                       imgpoints_for_intrinsics,
                                                       cal_data['im_size'][i_cam],
                                                       init_mtx,
                                                       None,
                                                       flags=CALIBRATION_FLAGS)
            intrinsics_frames = np.shape(objpoints_for_intrinsics)[0]
            valid_frames = cal_data['valid_frames'][i_cam]
            for i_frame in range(intrinsics_frames):
                pp = test_reprojection(objpoints_for_intrinsics[i_frame], imgpoints_for_intrinsics[i_frame], mtx, rvecs[i_frame], tvecs[i_frame], dist)
                cur_frame = [ii for ii, vf in enumerate(valid_frames) if vf == True][frame_numbers[i_frame]]
                vid_obj[i_cam].set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
                ret, cur_img = vid_obj[i_cam].read()

                if current_cam == 1:
                    # rotate the image 180 degrees
                    cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)

                # comment in to check that checkerboard points were correctly identified
                # corners_img = cv2.drawChessboardCorners(cur_img, cal_data['cb_size'], imgpoints_for_intrinsics[i_frame], True)
                # reproj_img = cv2.drawChessboardCorners(corners_img, cal_data['cb_size'], pp, False)
                #
                # img_name = '/home/levlab/Public/mouse_SR_videos_to_analyze/mouse_SR_calibration_data/test_frame_{}_cam{:02d}_frame{:03d}.jpg'.format(session_date_string, current_cam, cur_frame)
                # cv2.imwrite(img_name, reproj_img)

        cal_data['mtx'].append(np.copy(mtx))
        cal_data['dist'].append(np.copy(dist))
        cal_data['frame_nums_for_intrinsics'].append(frame_numbers)

        skilled_reaching_io.write_pickle(calibration_data_name, cal_data)

    # now perform stereo calibration if not already done
    # num_frames_for_stereo = 20, min_frames_for_stereo = 5
    if 'E_from_F' in cal_data.keys():
        print('stereo calibration with findFundamentalMat already calculated for {}'.format(session_date_string))
        return
    stereo_objpoints = cal_data['stereo_objpoints']
    stereo_imgpoints_ud, stereo_imgpoints_ud_norm = undistort_stereo_cbcorners(cal_data['stereo_imgpoints'], cal_data)

    cal_data['stereo_imgpoints_ud_norm'] = stereo_imgpoints_ud_norm
    cal_data['stereo_imgpoints_ud'] = stereo_imgpoints_ud
    cal_data['use_undistorted_pts_for_stereo_cal'] = use_undistorted_pts_for_stereo_cal
    if use_undistorted_pts_for_stereo_cal:
        stereo_imgpoints_for_calibration = cal_data['stereo_imgpoints_ud']
        stereo_imgpoints_for_calibration_norm = cal_data['stereo_imgpoints_ud_norm']
        dist = [np.zeros(5) for i_cam in range(num_cams)]   # points have already been undistorted, so don't undistort again during calibration
    else:
        stereo_imgpoints_for_calibration = cal_data['stereo_imgpoints']
        dist = cal_data['dist']
    num_stereo_pairs = np.shape(stereo_objpoints)[0]
    num_frames_to_use = min(num_frames_for_stereo, num_stereo_pairs)
    objpoints, imgpoints_ud, stereo_frame_idx = select_cboards_for_stereo_calibration(stereo_objpoints, stereo_imgpoints_for_calibration, num_frames_to_use)
    objpoints, imgpoints_ud_norm, stereo_frame_idx = select_cboards_for_stereo_calibration(stereo_objpoints,
                                                                                      stereo_imgpoints_for_calibration_norm,
                                                                                      num_frames_to_use)
    frames_for_stereo_calibration = [sf_idx for sf_idx in cal_data['stereo_frames']]

    mtx = cal_data['mtx']

    im_size = cal_data['im_size']
    # im_size must be the same for both cameras
    if all([ims == im_size[0] for ims in im_size]) and num_frames_to_use >= min_frames_for_stereo:
        # all images have the same size
        print('working on stereo calibration for {}'.format(session_date_string))
        ret, mtx1, dist1, mtx2, dist2, R_st, T_st, E_st, F_st = cv2.stereoCalibrate(objpoints, imgpoints_ud[0], imgpoints_ud[1], mtx[0], dist[0], mtx[1], dist[1], im_size[0], flags=STEREO_FLAGS)
        # hold on to above line for comparison, but may be able to eliminate it if findfundamentalmat works better

        # try recalculating using findFundamentalMat
        all_imgpts_reshaped = [np.reshape(im_pts, (-1, 2)) for im_pts in stereo_imgpoints_for_calibration]
        all_imgpts_norm_reshaped = [np.reshape(im_pts, (-1, 2)) for im_pts in stereo_imgpoints_for_calibration_norm]
        # imgpts_reshaped = [np.reshape(im_pts, (-1, 2)) for im_pts in imgpoints]
        # F_ffm, ffm_mask = cv2.findFundamentalMat(imgpts_reshaped[0], imgpts_reshaped[1], cv2.FM_RANSAC, FFM_tolerance,
        #                                          0.999)
        F_ffm, ffm_mask = cv2.findFundamentalMat(all_imgpts_reshaped[0], all_imgpts_reshaped[1], cv2.FM_RANSAC, FFM_tolerance, 0.999)
        F_ffm_norm, ffm_norm_mask = cv2.findFundamentalMat(all_imgpts_norm_reshaped[0], all_imgpts_norm_reshaped[1], cv2.FM_RANSAC,
                                                 FFM_tolerance, 0.999)
        E, E_mask = cv2.findEssentialMat(all_imgpts_reshaped[0], all_imgpts_reshaped[1], cal_data['mtx'][0], None,
                                         cal_data['mtx'][1], None, cv2.FM_RANSAC, 0.999, 0.1)
        E_norm, E_norm_mask = cv2.findEssentialMat(all_imgpts_norm_reshaped[0], all_imgpts_norm_reshaped[1], np.identity(3), None,
                                         np.identity(3), None, cv2.FM_RANSAC, 0.999, 0.1)
        # E_ffm = mtx[1].T @ F_ffm @ mtx[0]
        F_from_E = np.linalg.inv(cal_data['mtx'][1].T) @ E @ np.linalg.inv(cal_data['mtx'][0])
        E_from_F = cal_data['mtx'][1].T @ F_ffm @ cal_data['mtx'][0]

        # convert to normalized coordinates for pose recovery
        stereo_ud_norm = []
        stereo_ud = []
        stereo_ud_norm_for_T = []
        stereo_ud_for_T = []
        for i_cam in range(2):
            pts = np.array(cal_data['stereo_imgpoints'][i_cam])
            framepts_ud_norm = [cv2.undistortPoints(frame_pts, mtx[i_cam], cal_data['dist'][i_cam]) for frame_pts in pts]
            framepts_ud = [cvb.unnormalize_points(fpts_ud_norm, mtx[i_cam]) for fpts_ud_norm in framepts_ud_norm]
            num_frames = np.shape(framepts_ud)[0]
            pts_per_frame = np.shape(framepts_ud)[1]
            pts_r = np.reshape(pts, (-1, 2))
            pts_ud_norm = cv2.undistortPoints(pts_r, mtx[i_cam], cal_data['dist'][i_cam])
            pts_ud = cvb.unnormalize_points(pts_ud_norm, mtx[i_cam])
            stereo_ud_norm.append(pts_ud_norm)
            stereo_ud.append(pts_ud)
            stereo_ud_norm_for_T.append(framepts_ud_norm)

            for i_frame, fpts in enumerate(framepts_ud):
                framepts_ud[i_frame] = np.reshape(fpts, (pts_per_frame, 1, 2))
            stereo_ud_for_T.append(framepts_ud)
        # select 2000 points at random for chirality check (using all the points takes a really long time)
        num_pts = np.shape(pts_ud)[0]
        pt_idx = np.random.randint(0, num_pts, 2000)
        _, R_from_E, T_Eunit, ffm_msk = cv2.recoverPose(E, stereo_ud[0][pt_idx, :], stereo_ud[1][pt_idx, :], cal_data['mtx'][0])
        _, R_from_F, T_Funit, ffm_msk = cv2.recoverPose(E_from_F, stereo_ud[0][pt_idx, :], stereo_ud[1][pt_idx, :],
                                                       cal_data['mtx'][0])
        _, R_norm, T_norm_unit, ffm_msk = cv2.recoverPose(E_norm, stereo_ud_norm[0][pt_idx, :], stereo_ud_norm[1][pt_idx, :],
                                                np.identity(3))
        norm_mtx = [np.identity(3) for i_cam in range(2)]
        T_norm = estimate_T_from_ffm(stereo_objpoints, stereo_ud_norm_for_T, im_size, norm_mtx, R_norm)
        T_from_E = estimate_T_from_ffm(stereo_objpoints, stereo_ud_for_T, im_size, mtx, R_from_E)
        T_from_F = estimate_T_from_ffm(stereo_objpoints, stereo_ud_for_T, im_size, mtx, R_from_F)
        # WORKING HERE - NEED TO GET UNDISTORTED AND NORMALIZED POINTS IN CORRECT FORMAT FOR CALIBRATION IN estimate_T_from_ffm
        # todo: consider using ffm_mask to identify inliers for redoing camera calibration and repeating...
    else:
        ret = False
        # mtx1 = np.zeros((3, 3))
        # mtx2 = np.zeros((3, 3))
        # dist1 = np.zeros((1, 5))
        # dist2 = np.zeros((1, 5))
        R_from_E = np.zeros((3, 3))
        R_from_F = np.zeros((3, 3))
        R_norm = np.zeros((3, 3))
        E_norm = np.zeros((3, 3))
        # T = np.zeros((3, 1))
        T_from_F = np.zeros((3, 1))
        T_from_E = np.zeros((3, 1))
        T_norm = np.zeros((3, 1))
        T_norm_unit = np.zeros((3, 1))
        E = np.zeros((3, 3))
        F = np.zeros((3, 3))
        F_ffm = np.zeros((3, 3))
        F_ffm_norm = np.zeros((3, 3))
        # E_ffm = np.zeros((3, 3))
        # R_ffm = np.zeros((3, 3))
        T_Eunit = np.zeros((3, 1))
        T_Funit = np.zeros((3, 1))
        # ffm_mask = None
        T_st = np.zeros((3, 1))
        R_st = np.zeros((3, 3))
        F_st = np.zeros((3, 3))
        E_st = np.zeros((3, 3))
        F_from_E = np.zeros((3, 3))
        E_from_F = np.zeros((3, 3))

    cal_data['R_from_E'] = R_from_E
    cal_data['T_Eunit'] = T_Eunit
    cal_data['T_Funit'] = T_Funit
    cal_data['T_norm'] = T_norm
    cal_data['T_norm_unit'] = T_norm_unit
    # cal_data['T_unit'] = T_unit
    cal_data['T_st'] = T_st
    cal_data['R_st'] = R_st
    cal_data['F_st'] = F_st
    cal_data['E_st'] = E_st
    cal_data['E'] = E
    cal_data['F_from_E'] = F_from_E
    cal_data['E_norm'] = E_norm
    cal_data['R_norm'] = R_norm
    cal_data['F_ffm'] = F_ffm
    cal_data['F_ffm_norm'] = F_ffm_norm
    cal_data['E_from_F'] = E_from_F
    cal_data['R_from_F'] = R_from_F
    cal_data['T_from_F'] = T_from_F
    cal_data['T_from_E'] = T_from_E
    # cal_data['E_ffm'] = E_ffm
    # cal_data['ffm_mask'] = ffm_mask
    # cal_data['R_ffm'] = R_ffm
    # cal_data['T_ffm'] = T_ffm
    cal_data['frames_for_stereo_calibration'] = frames_for_stereo_calibration   # frame numbers in original calibration video used for the stereo calibration
    # if valid_frames[0][i_frame] and valid_frames[1][i_frame]:
    #     # checkerboards were identified in matching frames
    #     stereo_objpoints.append(objp)
    #     for i_vid, corner_pts in enumerate(corners2):
    #         stereo_imgpoints[i_vid].append(corner_pts)
    # if num_frames_to_use >= min_frames_for_stereo:
    #     check_Rs(cal_data)
    skilled_reaching_io.write_pickle(calibration_data_name, cal_data)
    # if num_frames_to_use >= min_frames_for_stereo:
    #     cal_metadata = navigation_utilities.parse_optitrack_calibration_data_name(calibration_data_name)
    #     show_cal_images_with_epilines(cal_metadata, parent_directories, plot_undistorted=True)


    # check if calibration worked
    # num_valid_stereo_pairs = np.shape(cal_data['stereo_imgpoints'])[1]
    # for stereo_idx in range(num_valid_stereo_pairs):
    #     frame_num = cal_data['stereo_frames'][stereo_idx]
    #     projPoints = []
    #     for i_cam in range(num_cams):
    #         projPoints.append(cal_data['stereo_imgpoints'][i_cam][stereo_idx])
    #         projPoints[i_cam] = np.squeeze(projPoints[i_cam]).T
    #
    #     print('frame number {:d}'.format(frame_num))
    #     worldpoints = triangulate_points(cal_data, projPoints, frame_num)


def undistort_stereo_cbcorners(stereo_imgpoints, cal_data):

    # stereo_imgpoints is a num_cams element list of num_valid_frames element lists containing num_points x 1 x 2 arrays

    num_cams = np.shape(stereo_imgpoints)[0]
    num_valid_frames = np.shape(stereo_imgpoints)[1]

    stereo_imgpoints_ud = [[] for i_cam in range(num_cams)]
    stereo_imgpoints_ud_norm = [[] for i_cam in range(num_cams)]

    np.zeros(np.shape(stereo_imgpoints))   # make sure data type is consistent

    for i_cam in range(num_cams):

        mtx = cal_data['mtx'][i_cam]
        dist = cal_data['dist'][i_cam]

        for i_frame in range(num_valid_frames):

            cur_pts = stereo_imgpoints[i_cam][i_frame]
            cur_pts_udnorm = cv2.undistortPoints(cur_pts, mtx, dist)
            cur_pts_ud = cvb.unnormalize_points(cur_pts_udnorm, mtx)
            cur_pts_ud = cur_pts_ud.reshape((-1, 1, 2)).astype('float32')

            stereo_imgpoints_ud_norm[i_cam].append(cur_pts_udnorm)
            stereo_imgpoints_ud[i_cam].append(cur_pts_ud)

    return stereo_imgpoints_ud, stereo_imgpoints_ud_norm


def test_reprojection(objpoints, imgpoints, mtx, rvec, tvec, dist):

    # rvec = np.array([0., 0., 0.])
    projected_pts, _ = cv2.projectPoints(objpoints, rvec, tvec, mtx, dist)
    return projected_pts


def select_cboards_for_calibration(objpoints, imgpoints, num_frames_to_extract):

    total_frames = np.shape(objpoints)[0]
    frame_spacing = int(total_frames/num_frames_to_extract)

    selected_objpoints = objpoints[::frame_spacing]
    selected_imgpoints = imgpoints[::frame_spacing]

    frame_numbers = range(0, total_frames, frame_spacing)

    return selected_objpoints, selected_imgpoints, frame_numbers


def select_cboards_for_stereo_calibration(objpoints, imgpoints, num_frames_to_extract):

    total_frames = np.shape(objpoints)[0]

    if num_frames_to_extract == 0:
        frame_numbers = []
        selected_objpoints = []
        selected_imgpoints = []
    else:
        frame_spacing = int(total_frames/num_frames_to_extract)
        frame_numbers = range(0, total_frames, frame_spacing)

        selected_objpoints = objpoints[::frame_spacing]
        selected_imgpoints = [ip[::frame_spacing] for ip in imgpoints]

    return selected_objpoints, selected_imgpoints, frame_numbers


def triangulate_points(cal_data, projPoints, frame_num, parent_directories):
    cal_data_parent = parent_directories['cal_data_parent']

    # first, create projMatr1 and projMatr2 (3 x 4 projection matrices for each camera) for each camera
    mtx = cal_data['mtx']
    dist = cal_data['dist']
    num_cams = len(mtx)

    # undistort points, which should give results in normalized coordinates
    ud_pts = [cv2.undistortPoints(projPoints[i_cam], mtx[i_cam], dist[i_cam]) for i_cam in range(num_cams)]

    num_pts = np.shape(projPoints)[2]
    reshaped_pts = [np.zeros((1, num_pts, 2)) for ii in range(2)]
    projPoints_array = []
    newpoints = [[], []]
    for ii in range(2):
        projPoints_array.append(np.squeeze(np.array([projPoints[ii]]).T))
        reshaped_pts[ii][0,:,:] = projPoints_array[ii]
    newpoints[0], newpoints[1] = cv2.correctMatches(cal_data['F_ffm'], reshaped_pts[0], reshaped_pts[1])
    newpoints = [np_array.astype('float32') for np_array in newpoints]

    new_cornerpoints = [np.squeeze(newpoints[ii]) for ii in range(2)]

    # for i_cam in range(2):
    #     session_date_string = navigation_utilities.datetime_to_string_for_fname(
    #         cal_data['calvid_metadata'][i_cam]['session_datetime'])
    #     test_img_dir = os.path.join(cal_data_parent, 'corner_images', session_date_string,
    #                                  'cam{:02d}'.format(cal_data['calvid_metadata'][i_cam]['cam_num']))
    #
    #     test_img_name = os.path.join(test_img_dir,
    #                                  'test_cboard_{}_cam{:02d}_frame{:04d}.jpg'.format(session_date_string,
    #                                                                                cal_data['calvid_metadata'][i_cam]['cam_num'],
    #                                                                                frame_num))

        # cboard_img = cv2.imread(test_img_name)
        # new_name = os.path.join(test_img_dir, 'refined_pts_{}_cam{:02d}_frame{:04d}.jpg'.format(session_date_string,
        #                                                                            cal_data['calvid_metadata'][i_cam]['cam_num'],
        #                                                                            frame_num))
        # new_img = cv2.drawChessboardCorners(cboard_img, cal_data['cb_size'], np.reshape(new_cornerpoints[ii], (70, 1, 2)), False)
        # try:
        #     cv2.imwrite(new_name, new_img)
        # except:
        #     pass

    # nudp1, nudp2 = cv2.correctMatches(cal_data['F'], ud_pts[0], ud_pts[1])

    projMatr1 = np.eye(3, 4)
    # projMatr2 = np.hstack((cal_data['R'], -np.dot(cal_data['R'].T, cal_data['T'])))
    projMatr2 = np.hstack((cal_data['R'], cal_data['T']))

    points4D = cv2.triangulatePoints(projMatr1, projMatr2, ud_pts[0], ud_pts[1])
    # fig=plt.figure()
    # ax0 = fig.add_subplot(1,2,1)
    # ax1 = fig.add_subplot(1,2,2)
    # ud_plot = [np.squeeze(ud_pts[i_cam]) for i_cam in range(2)]
    # ax0.scatter(ud_plot[0][:,0], ud_plot[0][:,1])
    # ax1.scatter(ud_plot[1][:, 0], ud_plot[1][:, 1])
    #
    # ax0.invert_yaxis()
    # ax1.invert_yaxis()
    #
    # plt.show()

    worldpoints = np.squeeze(cv2.convertPointsFromHomogeneous(points4D.T))

    #project worldpoints back to original images
    projected_pts = []
    unnorm_pts = []
    for i_cam in range(2):
        # mtx = cal_data['mtx'][i_cam]
        # dist = cal_data['dist'][i_cam]
        if i_cam == 0:
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
        else:
            rvec, _ = cv2.Rodrigues(cal_data['R'])
            # tvec = -np.dot(cal_data['R'].T, cal_data['T'])
            tvec = cal_data['T']

        proj_pts, _ = cv2.projectPoints(worldpoints, rvec, tvec, mtx[i_cam], dist[i_cam])
        projected_pts.append(np.squeeze(proj_pts))

        ud_pts_array = np.squeeze(ud_pts[i_cam])
        unnorm_pts.append(cvb.unnormalize_points(ud_pts_array, mtx[i_cam]))

    # fig = plt.figure()
    # ax0 = fig.add_subplot(1, 3, 1)
    # ax1 = fig.add_subplot(1, 3, 2)
    # ax3d = fig.add_subplot(1, 3, 3, projection='3d')
    #
    # ax0.scatter(projPoints_array[0][:, 0], projPoints_array[0][:, 1])
    # ax0.scatter(projected_pts[0][:,0], projected_pts[0][:,1],color='r',marker='+')
    # ax0.scatter(unnorm_pts[0][:,0], unnorm_pts[0][:,1],color='g',marker='*')
    # ax0.scatter(projPoints_array[0][:2, 0], projPoints_array[0][:2, 1],color='m')
    # ax0.set_xlim((0, 1280))
    # ax0.set_ylim((0, 1024))
    # ax0.invert_yaxis()
    # ax1.scatter(projPoints_array[1][:, 0], projPoints_array[1][:, 1])
    # ax1.scatter(projected_pts[1][:, 0], projected_pts[1][:, 1], color='r',marker='+')
    # ax1.scatter(unnorm_pts[1][:, 0], unnorm_pts[1][:, 1], color='g', marker='*')
    # ax1.scatter(projPoints_array[1][:2, 0], projPoints_array[1][:2, 1], color='m')
    # ax1.set_xlim((0, 1280))
    # ax1.set_ylim((0, 1024))
    # ax1.invert_yaxis()
    # ax3d.scatter(worldpoints[:, 0], worldpoints[:, 1], worldpoints[:, 2])
    # ax3d.scatter(worldpoints[:2, 0], worldpoints[:2, 1], worldpoints[:2, 2], color='m')
    # ax3d.set_xlabel('x')
    # ax3d.set_ylabel('y')
    # ax3d.set_zlabel('z')
    # plt.show()

    return worldpoints


def camera_calibration_from_mirror_vids(calibration_data, calibration_summary_name):

    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST

    cb_size = calibration_data['cb_size']
    im_size = calibration_data['orig_im_size']
    vf = calibration_data['valid_frames']
    num_frames = np.shape(vf)[1]

    views = [cvmd['view'] for cvmd in calibration_data['cropped_vid_metadata']]
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbrow = cb_size[0]
    cbcol = cb_size[1]
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # loop through all 3 views, find all valid images, translate them into the full image, perform the calibration

    full_view_imgpoints = []
    for i_view, view in enumerate(views):
        imgpoints = np.array(calibration_data['cam_imgpoints'][i_view])
        crop_params = calibration_data['cropped_vid_metadata'][i_view]['crop_params']

        # translate checkerboard points to the original image
        this_view_imgpoints = imgpoints
        this_view_imgpoints[:,:,:,0] = imgpoints[:,:,:,0] + crop_params[0] - 1
        this_view_imgpoints[:,:,:,1] = imgpoints[:,:,:,1] + crop_params[2] - 1

        full_view_imgpoints.append(this_view_imgpoints)

    all_imgpoints = np.concatenate([ip for ip in full_view_imgpoints])
    num_valid_cb = np.shape(all_imgpoints)[0]
    all_objpoints = [objp for ii in range(num_valid_cb)]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_objpoints, all_imgpoints, im_size, None, None, flags=CALIBRATION_FLAGS)
    calibration_data['camera_intrinsics'] = {'mtx': mtx, 'dist': dist}

    skilled_reaching_io.write_pickle(calibration_summary_name, calibration_data)

    return calibration_data


def estimate_T_from_ffm(objpoints, stereo_imgpoints_ud, im_size, mtx, R, max_frames_to_use=200):
    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_USE_INTRINSIC_GUESS
    num_cams = np.shape(stereo_imgpoints_ud)[0]
    # pts_per_frame = np.shape(stereo_imgpoints_ud)[2]
    num_frames = np.shape(stereo_imgpoints_ud)[1]
    # dist = np.zeros((1, 5))

    rv = []
    tv = []
    if max_frames_to_use >= num_frames:
        frames_idx = list(range(num_frames))
    else:
        frames_idx = np.random.randint(0, num_frames, max_frames_to_use)
    for i_cam in range(num_cams):
        stereo_ud_for_calibration = [stereo_imgpoints_ud[i_cam][f_idx].astype('float32') for f_idx in frames_idx]
        objpoints_for_calibration = [objpoints[f_idx] for f_idx in frames_idx]
        ret, new_mtx, new_dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_for_calibration,
                                                                   stereo_ud_for_calibration,
                                                                   im_size[i_cam],
                                                                   mtx[i_cam],
                                                                   None,
                                                                   flags=CALIBRATION_FLAGS)

        rv.append(np.squeeze(np.array(rvecs)))
        tv.append(np.squeeze(np.array(tvecs)))

    # compute T as T2 - R @ T1, where T1 are the translation vectors for each checkerboard for camera 1, and T2 are the
    # translation vectors for camera 2 for each checkerboard. See opencv stereoCalibrate documntation
    T1 = tv[0].T
    T2 = tv[1].T

    T_for_each_frame = T2 - R @ T1

    T = np.median(T_for_each_frame, 1)

    return T


def check_Rs(cal_data):
    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_USE_INTRINSIC_GUESS
    stereo_imgpoints = cal_data['stereo_imgpoints']
    objpoints = cal_data['stereo_objpoints']

    pts_per_frame = np.shape(stereo_imgpoints)[2]
    num_frames = np.shape(stereo_imgpoints[0])[0]

    rv = []
    tv = []
    for i_cam in range(2):

        im_pts_un = []
        for i_frame in range(num_frames):

            im_pts_ud = cv2.undistortPoints(stereo_imgpoints[i_cam][i_frame], cal_data['mtx'][i_cam], cal_data['dist'][i_cam])
            un_pts = np.float32(cvb.unnormalize_points(im_pts_ud, cal_data['mtx'][i_cam]))
            im_pts_un.append(np.reshape(un_pts, (pts_per_frame, 1, 2)))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       im_pts_un,
                                                       cal_data['im_size'][i_cam],
                                                       cal_data['mtx'][i_cam],
                                                       None,
                                                       flags=CALIBRATION_FLAGS)
        rv.append(np.squeeze(np.array(rvecs)))
        tv.append(np.squeeze(np.array(tvecs)))

    # R2est = []
    # R2_ffmest = []
    T_est = np.zeros((num_frames, 3))
    T_ffmest = np.zeros((num_frames, 3))
    for i_frame in range(num_frames):

        R1, _ = cv2.Rodrigues(np.squeeze(rv[0][i_frame]))
        R2, _ = cv2.Rodrigues(np.squeeze(rv[1][i_frame]))
        R2est = cal_data['R'] @ R1
        R2_ffmest = cal_data['R_ffm'] @ R1

        T1 = np.squeeze(tv[0][i_frame])
        T2 = np.squeeze(tv[1][i_frame])
        T2est = cal_data['R'] @ T1
        T2_ffmest = cal_data['R_ffm'] @ T1
        T = cal_data['T']

        T_est_frame = np.squeeze(T2) - cal_data['R'] @ T1
        T_ffmest_frame = np.squeeze(T2) - cal_data['R_ffm'] @ T1

        T_est[i_frame, :] = T_est_frame
        T_ffmest[i_frame, :] = T_ffmest_frame

        pass
    pass


def extract_valid_cbs_by_frame(calibration_data):

    cb_size = calibration_data['cb_size']
    vf = calibration_data['valid_frames']
    _, num_frames = np.shape(vf)

    views = [cvmd['view'] for cvmd in calibration_data['cropped_vid_metadata']]
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbrow = cb_size[0]
    cbcol = cb_size[1]
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    frame_counter = 0

    # loop through all 3 views, find all valid images, translate them into the full image, perform the calibration
    num_validframes = [0, 0, 0]
    for i_frame in range(num_frames):
        for i_view, view in enumerate(views):
            if vf[i_view][i_frame]:
                imgpoints = np.array(calibration_data['cam_imgpoints'][i_view])
                cur_imgpoints = imgpoints[i_frame,:,:,:]
                crop_params = calibration_data['cropped_vid_metadata'][i_view]['crop_params']

                num_validframes[i_view] += 1

        # test to see if points match up on original image
        # vid_name = '/home/levlab/Public/rat_SR_videos_to_analyze/calibration_videos/calibration_videos_2021/calibration_videos_202107/calibration_videos_202107_box02/GridCalibration_box02_20210713_14-30-28.avi'
        # vid_obj = cv2.VideoCapture(vid_name)
        # vid_obj.set(cv2.CAP_PROP_POS_FRAMES, 134)
        # ret, cur_img = vid_obj.read()
        # corners = full_imgpoints[0,:,:,:]
        # corners_img = cv2.drawChessboardCorners(cur_img, cb_size, corners,
        #                                         True)
        # test_img = '/home/levlab/Public/rat_SR_videos_to_analyze/test_img.png'
        # cv2.imwrite(test_img, corners_img)
        # plt.imshow(corners_img)
        # plt.show()
        pass


def recalculate_E_and_F_from_stereo_matches(cal_data):

    mtx = cal_data['mtx']
    dist = cal_data['dist']
    stereo_im_pts = cal_data['stereo_imgpoints']

    cam_mtx = np.identity(3)
    stereo_unnorm = []
    stereo_ud = []
    pts_renormalized = []
    for i_cam in range(2):
        pts = np.array(stereo_im_pts[i_cam])
        pts_r = np.reshape(pts, (-1, 2))
        pts_ud = cv2.undistortPoints(pts_r, mtx[i_cam], dist[i_cam])
        pts_un = cvb.unnormalize_points(pts_ud, mtx[i_cam])
        stereo_unnorm.append(pts_un)
        stereo_ud.append(np.squeeze(pts_ud))
        pts_renormalized.append(cvb.normalize_points(pts_un, mtx[i_cam]))


    F_new, F_msk = cv2.findFundamentalMat(stereo_unnorm[0], stereo_unnorm[1], cv2.FM_RANSAC, 0.1, 0.999)
    E_new, E_msk = cv2.findEssentialMat(stereo_unnorm[0], stereo_unnorm[1], cal_data['mtx'][0], None,
                                         cal_data['mtx'][1], None, cv2.FM_RANSAC, 0.999, 1)

    F_from_E = np.linalg.inv(mtx[1].T) @ E_new @ np.linalg.inv(mtx[0])
    E_from_F = mtx[1].T @ F_new @ mtx[0]

    _, R_E, t_E, E_msk = cv2.recoverPose(E_new, stereo_unnorm[0], stereo_unnorm[1], np.identity(3))
    _, R_F, t_F, F_msk = cv2.recoverPose(E_from_F, stereo_unnorm[0], stereo_unnorm[1], np.identity(3))

    E = {
        'E_from_F': E_from_F,
        'E_new': E_new,
        'E_msk': E_msk,
         }
    F = {
        'F_from_E': F_from_E,
        'F_new': F_new,
        'F_msk': F_msk,
         }
    R = {
        'R_E': R_E,
        'R_F': R_F,
    }
    T = {
        't_E': t_E,
        't_F': t_F,
    }
    return E, F, R, T
