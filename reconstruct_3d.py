import numpy as np
import cv2
import os
import navigation_utilities
import glob
import skilled_reaching_io
import pandas as pd
import scipy.io as sio
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import computer_vision_basics as cvb
import shapely.geometry as sg
import sr_visualization
import dlc_utilities

def test_reconstruction(parent_directories, rat_df):
    trajectories_parent = parent_directories['trajectories_parent']

    traj_directories = navigation_utilities.get_trajectory_folders(trajectories_parent)

    for td in traj_directories:
        test_singlefolder_reconstruction(td, parent_directories)
    pass


def test_singlefolder_reconstruction(traj_directory, parent_directories):

    traj_files = navigation_utilities.find_traj_files(traj_directory)

    for traj_file in traj_files:
        test_single_trajectory(traj_file, parent_directories)

    pass


def test_single_trajectory(traj_fname, parent_directories):

    videos_root_folder = parent_directories['videos_root_folder']
    traj_metadata = navigation_utilities.parse_paw_trajectory_fname(traj_fname)
    traj_data = skilled_reaching_io.read_pickle(traj_fname)

    # todo: create a movie of 3d reconstruction with video of points super-imposed on videos (also show reprojection errors?)
    # also pick out some specific frames

    orig_video = navigation_utilities.find_orig_rat_video(traj_metadata, videos_root_folder)

    # crop regions is a 2-element list of tuples - first tuple is borders for direct view, second set is for mirror view
    # each tuple is four elements: left, right, top, bottom
    direct_crop = (750, 1250, 500, 900)
    leftmirror_crop = (0, 400, 400, 800)
    rightmirror_crop = (1650, 2039, 400, 800)
    if traj_data['paw_pref'].lower() == 'left':
        # crop right mirror view
        crop_regions = [direct_crop, rightmirror_crop]
    else:
        crop_regions = [direct_crop, leftmirror_crop]
    sr_visualization.animate_vids_plus3d(traj_data, crop_regions, orig_video)
    pass


def reconstruct_folders(folders_to_reconstruct, parent_directories,  rat_df):

    marked_videos_parent = parent_directories['marked_videos_parent']
    calibration_files_parent = parent_directories['calibration_files_parent']
    trajectories_parent = parent_directories['trajectories_parent']

    for folder_to_reconstruct in folders_to_reconstruct:

        # first, figure out if we have calibration files for this session
        session_date = folder_to_reconstruct['session_date']
        box_num = folder_to_reconstruct['session_box']
        calibration_folder = navigation_utilities.find_calibration_files_folder(session_date, box_num, calibration_files_parent)

        if os.path.exists(calibration_folder):
            # is there a calibration file for this session?

            cal_data = skilled_reaching_io.get_calibration_data(session_date, box_num, calibration_folder)
            reconstruct_folder(folder_to_reconstruct, cal_data, rat_df, trajectories_parent)


def reconstruct_folders_anipose(folders_to_reconstruct, parent_directories,  expt):

    cropped_videos_parent = parent_directories['cropped_videos_parent']
    calibration_files_parent = parent_directories['calibration_files_parent']
    trajectories_parent = parent_directories['trajectories_parent']

    videos_root_folder = parent_directories['videos_root_folder']
    session_metadata_xlsx_path = os.path.join(videos_root_folder, 'SR_{}_video_session_metadata.xlsx'.format(expt))

    # load the .xlsx file containing all the info about which calibration files to use for each session
    calibration_metadata_df = skilled_reaching_io.read_session_metadata_xlsx(session_metadata_xlsx_path)

    for folder_to_reconstruct in folders_to_reconstruct:

        # first, figure out if we have calibration files for this session
        ratID = folder_to_reconstruct['ratID']
        session_date = folder_to_reconstruct['session_date']
        box_num = folder_to_reconstruct['boxnum']
        session_num = folder_to_reconstruct['session_num']
        task = folder_to_reconstruct['task']

        rat_md_df = calibration_metadata_df[ratID]
        date_df = rat_md_df.loc[rat_md_df['date'] == session_date]
        task_df = date_df.loc[date_df['task'] == task]
        box_df = task_df.loc[task_df['box_num'] == box_num]
        session_row = box_df.loc[box_df['session_num'] == session_num]

        # calibrate the camera for this session
        mirror_calib_vid_name = session_row['cal_vid_name_mirrors'].values[0]
        if mirror_calib_vid_name.lower() == 'none':
            continue
        full_calib_vid_name = navigation_utilities.find_mirror_calibration_video(mirror_calib_vid_name,
                                                                                 parent_directories)
        calibration_pickle_name = navigation_utilities.create_calibration_summary_name(full_calib_vid_name, calibration_files_parent)

        if not os.path.exists(calibration_pickle_name):
            continue

        calibration_data = skilled_reaching_io.read_pickle(calibration_pickle_name)
        reconstruct_folder_anipose(folder_to_reconstruct, calibration_data, parent_directories)
        cgroup = calibration_data['cgroup']

        calibration_folder = navigation_utilities.find_calibration_files_folder(session_date, box_num, calibration_files_parent)

        if os.path.exists(calibration_folder):
            # is there a calibration file for this session?

            cal_data = skilled_reaching_io.get_calibration_data(session_date, box_num, calibration_folder)
            reconstruct_folder(folder_to_reconstruct, cal_data, rat_df, trajectories_parent)


def reconstruct_folder_anipose(folder_to_reconstruct, calibration_data, parent_directories):

    cgroup = calibration_data['cgroup']

    pass

def reconstruct_folder(folder_to_reconstruct, cal_data, rat_df, trajectories_parent, view_list=('direct', 'leftmirror', 'rightmirror'), vidtype='.avi'):

    if vidtype[0] != '.':
        vidtype = '.' + vidtype
    _, session_name = os.path.split(folder_to_reconstruct['session_folder'])
    # assume need at least a direct view for each video
    view_dirs = [os.path.join(folder_to_reconstruct['session_folder'], session_name + '_' + view) for view in view_list]

    direct_view_dir = [view_dir for view_dir in view_dirs if 'direct' in view_dir][0]
    ratID, session_name = navigation_utilities.parse_session_dir_name(folder_to_reconstruct['session_folder'])

    rat_num = int(ratID[1:])
    paw_pref = rat_df[rat_df['ratID'] == rat_num]['pawPref'].values[0]

    if paw_pref == 'left':
        mirror_view = 'rightmirror'
        mirror_view_dir = [view_dir for view_dir in view_dirs if 'right' in view_dir][0]
    elif paw_pref == 'right':
        mirror_view = 'leftmirror'
        mirror_view_dir = [view_dir for view_dir in view_dirs if 'left' in view_dir][0]
    else:
        print('no paw preference found for rat {}'.format(ratID))
        return

    session_date = navigation_utilities.fname_string_to_date(session_name[:-1])
    date_string = navigation_utilities.date_to_string_for_fname(session_date)
    full_pickle_search_string = os.path.join(direct_view_dir, ratID + '_' + date_string + '_*_direct_*_full.pickle')
    direct_full_pickles = glob.glob(full_pickle_search_string)

    for direct_full_pickle in direct_full_pickles:
        direct_pickle_params = navigation_utilities.parse_dlc_output_pickle_name(direct_full_pickle)
        direct_test_names = navigation_utilities.construct_dlc_output_pickle_names(direct_pickle_params, 'direct')
        mirror_test_names = navigation_utilities.construct_dlc_output_pickle_names(direct_pickle_params, mirror_view)
        # returns 2-element tuple containing test string for full.pickle and meta.pickle files, respectively

        full_mirror_pickle = glob.glob(os.path.join(mirror_view_dir, mirror_test_names[0]))[0]
        meta_mirror_pickle = glob.glob(os.path.join(mirror_view_dir, mirror_test_names[1]))[0]
        meta_direct_pickle = glob.glob(os.path.join(direct_view_dir, direct_test_names[1]))[0]

        mirror_pickle_params = navigation_utilities.parse_dlc_output_pickle_name(full_mirror_pickle)

        dlc_output = {'direct': skilled_reaching_io.read_pickle(direct_full_pickle),
                      mirror_view: skilled_reaching_io.read_pickle(full_mirror_pickle)
                      }
        dlc_metadata = {'direct': skilled_reaching_io.read_pickle(meta_direct_pickle),
                        mirror_view: skilled_reaching_io.read_pickle(meta_mirror_pickle)
                        }
        trajectory_filename = navigation_utilities.create_trajectory_filename(direct_pickle_params)
        trajectory_folder = navigation_utilities.trajectory_folder(trajectories_parent, ratID, session_name)
        full_traj_fname = os.path.join(trajectory_folder, trajectory_filename)
        if os.path.exists(full_traj_fname):
            continue

        pickle_params = {'direct': direct_pickle_params,
                         mirror_view: mirror_pickle_params
                         }
        trajectory_metadata = dlc_utilities.extract_trajectory_metadata(dlc_metadata, pickle_params)
        dlc_data = dlc_utilities.extract_data_from_dlc_output(dlc_output, trajectory_metadata)

        # translate and undistort points
        dlc_data = translate_points_to_full_frame(dlc_data, trajectory_metadata)
        # note that for right view, left and right labels are swapped in the translate_points_to_full_frame function
        # this is because DLC labeling is done on a left-right reversed video of the right mirror, so the labels for
        # the left paw are really the right paw (mirror reversal)
        dlc_data = undistort_points(dlc_data, cal_data)

        invalid_points, diff_per_frame = find_invalid_DLC_points(dlc_data, paw_pref)

        # comment back in to show identified points and undistorted points superimposed on frames
        # test_undistortion(dlc_data, invalid_points, cal_data, direct_pickle_params)

        paw_trajectory, is_estimate = calc_3d_dlc_trajectory(dlc_data, invalid_points, cal_data, paw_pref, direct_pickle_params, im_size=(2040, 1024), max_dist_from_neighbor=60)

        reproj_error, high_p_invalid, low_p_valid = assess_reconstruction_quality(paw_trajectory, dlc_data, invalid_points, cal_data, paw_pref, direct_pickle_params, p_cutoff=0.9)
        # todo: check that multiple pellet ID's are handled
        # todo: save these data, and create movies of reconstructions to see how we're doing...

        views = tuple(dlc_data.keys())
        bodyparts = [tuple(dlc_data[view].keys()) for view in views]
        bp_coords_ud = [dlc_utilities.collect_bp_data(dlc_data[view], 'coordinates_ud') for view in views]

        traj_data = package_trajectory_data_for_pickle(paw_trajectory, is_estimate, invalid_points, paw_pref,
                                                       reproj_error, high_p_invalid, low_p_valid, cal_data, bp_coords_ud, bodyparts)
        skilled_reaching_io.write_pickle(full_traj_fname, traj_data)



        # mat_data = package_data_into_mat(dlc_data, video_metadata, trajectory_metadata)
        # mat_name = navigation_utilities.create_mat_fname_dlc_output(video_metadata, dlc_mat_output_parent)
        #
        # video_name = navigation_utilities.build_video_name(video_metadata, videos_parent)
        # # test_pt_alignment(video_name, dlc_data)
        #
        # sio.savemat(mat_name, mat_data)


def package_trajectory_data_for_pickle(paw_trajectory, is_estimate, invalid_points, paw_pref, reproj_error, high_p_invalid, low_p_valid, cal_data, bp_coords_ud, bodyparts):

    traj_data = {'paw_trajectory': paw_trajectory,
                 'is_estimate': is_estimate,
                 'invalid_points': invalid_points,
                 'paw_pref': paw_pref,
                 'reproj_error': reproj_error,
                 'high_p_invalid': high_p_invalid,
                 'low_p_valid': low_p_valid,
                 'cal_data': cal_data,
                 'bp_coords_ud': bp_coords_ud,
                 'bodyparts': bodyparts
                 }

    return traj_data


def assess_reconstruction_quality(paw_trajectory, dlc_data, invalid_points, cal_data, paw_pref, direct_pickle_params, p_cutoff=0.9):

    view_list = tuple(dlc_data.keys())
    bodyparts = [tuple(dlc_data[view].keys()) for view in view_list]
    num_bodyparts = [len(bp) for bp in bodyparts]

    bp_idx = [group_dlc_bodyparts(bp) for bp in bodyparts]
    bp_coords = [dlc_utilities.collect_bp_data(dlc_data[view], 'coordinates_ud') for view in view_list]
    bp_conf = [dlc_utilities.collect_bp_data(dlc_data[view], 'confidence') for view in view_list]

    high_p = [np.squeeze(view_conf > p_cutoff) for view_conf in bp_conf]

    num_frames = np.shape(paw_trajectory)[0]
    num_bp = np.shape(paw_trajectory)[2]

    high_p_invalid = np.zeros((num_bp, num_frames, 2), dtype=bool)
    low_p_valid = np.zeros((num_bp, num_frames, 2), dtype=bool)
    for i_view, view in enumerate(view_list):
        high_p_invalid[:, :, i_view] = np.logical_and(high_p[i_view], invalid_points[i_view])
        low_p_valid[:, :, i_view] = np.logical_and(np.logical_not(high_p[i_view]), np.logical_not(invalid_points[i_view]))

    # calculate distance between reconstructed points and origiinally idenifiedpoints in the direct and mirror views
    mtx = cal_data['mtx']
    dist = cal_data['dist']
    projMatr = [np.eye(3, 4), []]
    if paw_pref.lower() == 'left':
        # use F for right mirror
        projMatr[1] = cal_data['Pn'][:, :, 2]
        sf = np.mean(cal_data['scalefactor'][2, :])
        view_list = ('direct', 'rightmirror')
    elif paw_pref.lower() == 'right':
        # use F for left mirror
        projMatr[1] = cal_data['Pn'][:, :, 1]
        view_list = ('direct', 'leftmirror')
        sf = np.mean(cal_data['scalefactor'][1, :])

    unscaled_trajectory = paw_trajectory / sf
    reproj_error = [np.zeros((num_bp, num_frames)), np.zeros((num_bp, num_frames))]
    proj_points = [np.zeros((num_bp, num_frames, 2)), np.zeros((num_bp, num_frames, 2))]

    for i_bp in range(num_bp):
        # make sure the correct indexes are identified for the same bodypart each view
        bp_idx = [i_bp, bodyparts[1].index(bodyparts[0][i_bp])]

        current_3d = np.squeeze(unscaled_trajectory[:, :, i_bp])

        # proj_points = [[], []]
        for i_view in range(2):
            rvec, _ = cv2.Rodrigues(projMatr[i_view][:, :3])
            tvec = projMatr[i_view][:, -1]
            temp_pts = cvb.project_points(current_3d, projMatr[i_view], mtx)
            # temp_pts, _ = cv2.projectPoints(current_3d, rvec, tvec, mtx, dist)
            # proj_points[i_view][i_bp, :, :] = np.squeeze(temp_pts)
            # note that cv2.projectPoints requires you to change the rotation matrix to a rotation vector. Somehow,
            # this gets all messed up because we're using mirrors and cv2.Rodrigues can't handle a reflection.
            # So I wrote a separate reprojection function, which works fine.
            proj_points[i_view][i_bp, :, :] = temp_pts.T

            orig_pts = bp_coords[i_view][bp_idx[i_view], :, :]
            reproj_error[i_view][i_bp, :] = np.linalg.norm(orig_pts - proj_points[i_view][i_bp, :, :], axis=1)

    # test_reprojection(dlc_data, invalid_points, proj_points, direct_pickle_params, cal_data)
    return reproj_error, high_p_invalid, low_p_valid


def calc_3d_dlc_trajectory(dlc_data, invalid_points, cal_data, paw_pref, direct_pickle_params, im_size=(2040, 1024), max_dist_from_neighbor=60):

    # F[:,:,0] - fundamental matrix between direct and top mirror views
    # F[:,:,1] - fundamental matrix between direct and left mirror views
    # F[:,:,2] - fundamental matrix between direct and right mirror views

    view_list = tuple(dlc_data.keys())
    bodyparts = [tuple(dlc_data[view].keys()) for view in view_list]

    num_frames = np.shape(dlc_data[view_list[0]][bodyparts[0][0]]['coordinates_ud'])[0]
    frames_to_check = range(num_frames)

    bp_coords, is_estimate = estimate_hidden_points(dlc_data, invalid_points, cal_data, im_size, paw_pref, frames_to_check, direct_pickle_params, max_dist_from_neighbor=max_dist_from_neighbor)

    #todo: once points are estimated, should we filter the 2-d trajectories to increase 3-d accuracy before triangulating?
    # should triangulation and reprojection errors be part of the filtering?
    num_frames = np.shape(bp_coords[0])[1]
    num_bp = len(bodyparts[0])
    paw_trajectory = np.zeros((num_frames, 3, num_bp))    # for now, assume bodyparts match up, same number in both views

    projMatr1 = np.eye(3, 4)
    if paw_pref.lower() == 'left':
        # use F for right mirror
        projMatr2 = cal_data['Pn'][:, :, 2]
        sf = np.mean(cal_data['scalefactor'][2, :])
        view_list = ('direct', 'rightmirror')
    elif paw_pref.lower() == 'right':
        # use F for left mirror
        projMatr2 = cal_data['Pn'][:, :, 1]
        view_list = ('direct', 'leftmirror')
        sf = np.mean(cal_data['scalefactor'][1, :])

    mtx = cal_data['mtx']
    dist = cal_data['dist']

    for i_bp, bp in enumerate(bodyparts[0]):

        valid_direct = np.logical_not(invalid_points[0][i_bp, :])
        mirror_bp_idx = bodyparts[1].index(bp)
        valid_mirror = np.logical_not(invalid_points[1][mirror_bp_idx, :])
        estimate_direct = np.squeeze(is_estimate[0][i_bp, :])
        estimate_mirror = np.squeeze(is_estimate[1][mirror_bp_idx, :])

        all_valid_points = np.logical_and(valid_direct, valid_mirror)
        direct_valid_mirror_est = np.logical_and(valid_direct, estimate_mirror)
        direct_est_mirror_valid = np.logical_and(estimate_direct, valid_mirror)

        valid_points = all_valid_points | direct_valid_mirror_est | direct_est_mirror_valid

        if not any(valid_points):
            # don't bother to loop if no valid points for this bodypart
            continue

        cur_direct_pts = np.squeeze(bp_coords[0][i_bp, valid_points, :])
        cur_mirror_pts = np.squeeze(bp_coords[1][mirror_bp_idx, valid_points, :])

        norm_direct_pts = cvb.normalize_points(cur_direct_pts, mtx)
        norm_mirror_pts = cvb.normalize_points(cur_mirror_pts, mtx)

        point4D = cv2.triangulatePoints(projMatr1, projMatr2, norm_direct_pts, norm_mirror_pts)
        pellet_wp = np.squeeze(cv2.convertPointsFromHomogeneous(point4D.T))

        paw_trajectory[valid_points, :, i_bp] = pellet_wp * sf

    return paw_trajectory, is_estimate
    # bp_toplot = 'leftdig2'
    # bp_idx = bodyparts[0].index(bp_toplot)
    # fig_coords = plt.figure()
    # ax_coords = fig_coords.subplots(3, 1)
    #
    # fig_3d = plt.figure()
    # ax_3d = fig_3d.add_subplot(111, projection='3d')
    #
    # fig_conf = plt.figure()
    # ax_conf = fig_conf.subplots(2, 1)
    # for i_ax in range(3):
    #     ax_coords[i_ax].plot(paw_trajectory[:,i_ax,bp_idx])
    #
    # for i_view, view in enumerate(view_list):
    #     ax_conf[i_view].plot(dlc_data[view][bp_toplot]['confidence'])
    #
    # ax_3d.scatter3D(paw_trajectory[:,0,bp_idx], paw_trajectory[:,1,bp_idx], paw_trajectory[:,2,bp_idx])
    # ax_3d.set_xlabel('x')
    # ax_3d.set_ylabel('y')
    # ax_3d.set_zlabel('z')
    # ax_3d.invert_yaxis()
    # ax_3d.set_zlim(150, 200)
    #
    # plt.show()

def estimate_hidden_points(dlc_data, invalid_points, cal_data, im_size, paw_pref, frames_to_check, direct_pickle_params, max_dist_from_neighbor=60):

    # F[:,:,0] - fundamental matrix between direct and top mirror views
    # F[:,:,1] - fundamental matrix between direct and left mirror views
    # F[:,:,2] - fundamental matrix between direct and right mirror views

    view_list = tuple(dlc_data.keys())
    bodyparts = [tuple(dlc_data[view].keys()) for view in view_list]
    num_bodyparts = [len(bp) for bp in bodyparts]

    bp_idx = [group_dlc_bodyparts(bp) for bp in bodyparts]
    bp_coords = [dlc_utilities.collect_bp_data(dlc_data[view], 'coordinates_ud') for view in view_list]
    num_frames = np.shape(bp_coords[0])[1]
    is_estimate = [np.zeros((num_bodyparts[i_view], num_frames), dtype=bool) for i_view, view in enumerate(view_list)]

    if paw_pref.lower() == 'left':
        # use F for right mirror
        F = cal_data['F'][:, :, 2]
        view_list = ('direct', 'rightmirror')
    elif paw_pref.lower() == 'right':
        # use F for left mirror
        F = cal_data['F'][:, :, 1]
        view_list = ('direct', 'leftmirror')

    num_digits = 4
    digitparts = ('mcp', 'pip', 'dig')
    for i_frame in (300, 301): #frames_to_check:

        for digitpart in digitparts:
            for i_paw, paw in enumerate(('left', 'right')):

                all_paw_parts_idx = collect_all_paw_parts_idx(bp_idx, i_paw)

                all_paw_pts = [np.squeeze(bp_coords[i_view][all_paw_parts_idx[i_view], i_frame, :]) for i_view in range(2)]
                valid_paw_pts = [all_paw_pts[i_view][np.logical_not(invalid_points[i_view][all_paw_parts_idx[i_view], i_frame])] for i_view in range(2)]

                for i_digit in range(num_digits):

                    if digitpart == 'mcp':
                        next_knuckle_test_string = paw + 'pip' + '{:d}'.format(i_digit + 1)
                    elif digitpart == 'pip':
                        next_knuckle_test_string = paw + 'dig' + '{:d}'.format(i_digit + 1)
                    elif digitpart == 'dig':
                        next_knuckle_test_string = paw + 'pip' + '{:d}'.format(i_digit + 1)

                    next_knuckle_idx = [bp.index(next_knuckle_test_string) for bp in bodyparts]
                    # direct_next_knuckle_idx = bodyparts[0].index(next_knuckle_test_string)
                    # mirror_next_knuckle_idx = bodyparts[1].index(next_knuckle_test_string)

                    digit_string = paw + digitpart + '{:d}'.format(i_digit + 1)

                    direct_part_idx = bodyparts[0].index(digit_string)
                    mirror_part_idx = bodyparts[1].index(digit_string)

                    if (invalid_points[0][direct_part_idx, i_frame] and invalid_points[1][mirror_part_idx, i_frame]) or (not invalid_points[0][direct_part_idx, i_frame] and not invalid_points[1][mirror_part_idx, i_frame]):
                        # either neither point was found with certainty or both points were found with certainty. Either way, nothing to do
                        continue

                    # figure out whether the mirror or direct view point was identified
                    next_digit_knuckles = np.empty((2, 2))
                    next_digit_knuckles[:] = np.nan
                    if invalid_points[0][direct_part_idx, i_frame]:
                        # the mirror point was identified
                        # all_paw_points = valid_points[0]
                        view_to_reconstruct = 0
                        known_view = 1
                    else:
                        view_to_reconstruct = 1
                        known_view = 0

                    known_pt = dlc_data[view_list[known_view]][digit_string]['coordinates_ud'][i_frame, :]

                    # find the marked points at neighboring knuckles (e.g., if mcp2, find the location of mcp1 and mcp3)
                    # note offset because python indexes starting at 0, so mcp1 is indexed as point 0
                    other_knuckle_pts = find_other_knuckle_pts(bp_coords[view_to_reconstruct], bodyparts[view_to_reconstruct], invalid_points[view_to_reconstruct], paw, digitpart, i_frame)
                    if i_digit == 0:
                        if not invalid_points[view_to_reconstruct][bp_idx[view_to_reconstruct][digitpart][i_paw][1], i_frame]:
                            next_digit_knuckles[1, :] = bp_coords[view_to_reconstruct][bp_idx[view_to_reconstruct][digitpart][i_paw][1], i_frame, :]
                    elif i_digit in (1, 2):
                        if not invalid_points[view_to_reconstruct][bp_idx[view_to_reconstruct][digitpart][i_paw][i_digit - 1], i_frame]:
                            next_digit_knuckles[0, :] = bp_coords[view_to_reconstruct][bp_idx[view_to_reconstruct][digitpart][i_paw][i_digit - 1], i_frame, :]
                        if not invalid_points[view_to_reconstruct][bp_idx[view_to_reconstruct][digitpart][i_paw][i_digit + 1], i_frame]:
                            next_digit_knuckles[1, :] = bp_coords[view_to_reconstruct][bp_idx[view_to_reconstruct][digitpart][i_paw][i_digit + 1], i_frame, :]
                    elif i_digit == 3:
                        if not invalid_points[view_to_reconstruct][bp_idx[view_to_reconstruct][digitpart][i_paw][2], i_frame]:
                            next_digit_knuckles[0, :] = bp_coords[view_to_reconstruct][bp_idx[view_to_reconstruct][digitpart][i_paw][2], i_frame, :]

                    # find the point marked at the next knuckle on the same digit
                    if not invalid_points[view_to_reconstruct][next_knuckle_idx[view_to_reconstruct], i_frame]:
                        next_knuckle_pt = bp_coords[view_to_reconstruct][next_knuckle_idx[view_to_reconstruct], i_frame, :]
                    else:
                        next_knuckle_pt = []
                    # else:
                    #     # the direct point was identified
                    #     known_pt = dlc_data[view_list[0]][digit_string]['coordinates_ud'][i_frame, :]
                    #
                    #     # find the marked points at neighboring knuckles (e.g., if mcp2, find the location of mcp1 and mcp3)
                    #     # note offset because python indexes starting at 0, so mcp1 is indexed as point 0
                    #     other_knuckle_pts = find_other_knuckle_pts(bp_coords[1], bodyparts[1], invalid_points[1], paw, digitpart, i_frame)
                    #
                    #     if i_digit == 0:
                    #         if not invalid_points[1][bp_idx[1][digitpart][i_paw][1], i_frame]:
                    #             next_digit_knuckles[1, :] = bp_coords[1][bp_idx[1][digitpart][i_paw][1], i_frame, :]
                    #     elif i_digit in (1, 2):
                    #         if not invalid_points[1][bp_idx[1][digitpart][i_paw][i_digit - 1], i_frame]:
                    #             next_digit_knuckles[0, :] = bp_coords[1][bp_idx[1][digitpart][i_paw][i_digit - 1], i_frame, :]
                    #         if not invalid_points[1][bp_idx[1][digitpart][i_paw][i_digit + 1], i_frame]:
                    #             next_digit_knuckles[1, :] = bp_coords[1][bp_idx[1][digitpart][i_paw][i_digit + 1], i_frame, :]
                    #     elif i_digit == 3:
                    #         if not invalid_points[1][bp_idx[digitpart][i_paw][2], i_frame]:
                    #             next_digit_knuckles[0, :] = bp_coords[1][bp_idx[digitpart][i_paw][2], i_frame, :]
                    #
                    #     # find the point marked at the next knuckle on the same digit
                    #     if not invalid_points[1][direct_next_knuckle_idx, i_frame]:
                    #         next_knuckle_pt = bp_coords[1][direct_next_knuckle_idx, i_frame, :]
                    #     else:
                    #         next_knuckle_pt = []

                    if len(next_knuckle_pt) == 0 and len(other_knuckle_pts) == 0:
                        # did not find any adjacent knuckles with which to match the point from the other view
                        continue

                    new_pt = estimate_paw_part(known_pt, next_digit_knuckles, other_knuckle_pts, next_knuckle_pt, valid_paw_pts[view_to_reconstruct], F, im_size, max_dist_from_neighbor)

                    if new_pt is None:
                        continue

                    if invalid_points[0][direct_part_idx, i_frame]:
                        # the mirror point was identified
                        bp_coords[0][direct_part_idx, i_frame, :] = new_pt
                        is_estimate[0][direct_part_idx, i_frame] = True    # direct point for this body part in this frame is estimated
                    else:
                        bp_coords[1][mirror_part_idx, i_frame, :] = new_pt
                        is_estimate[1][mirror_part_idx, i_frame] = True

                    bp = digitpart + '{:d}'.format(i_digit + 1)
                    # test_estimated_pts(dlc_data, new_pt, paw, bp, invalid_points, cal_data, direct_pickle_params)

    final_direct_pawdorsum_pts, is_paw_dorsum_estimate = estimate_direct_paw_dorsum(bp_coords, invalid_points, bodyparts, cal_data, im_size, paw_pref, frames_to_check, max_dist_from_neighbor=max_dist_from_neighbor)
    for i_paw, paw in enumerate(('left', 'right')):
        pd_string = paw + 'pawdorsum'
        pd_idx = bodyparts[0].index(pd_string)
        bp_coords[0][pd_idx, :, :] = final_direct_pawdorsum_pts[i_paw]
        is_estimate[0][pd_idx, :] = is_paw_dorsum_estimate[i_paw]

    return bp_coords, is_estimate


def estimate_direct_paw_dorsum(bp_coords, invalid_points, bodyparts, cal_data, im_size, paw_pref, frames_to_check, max_dist_from_neighbor=60):

    if paw_pref.lower() == 'left':
        # use F for right mirror
        F = cal_data['F'][:, :, 2]
        view_list = ('direct', 'rightmirror')
    elif paw_pref.lower() == 'right':
        # use F for left mirror
        F = cal_data['F'][:, :, 1]
        view_list = ('direct', 'leftmirror')

    digitparts = ('mcp', 'pip', 'dig')

    num_frames_total = np.shape(bp_coords[0])[1]
    num_frames_to_check = len(frames_to_check)

    bp_idx = [group_dlc_bodyparts(bp) for bp in bodyparts]
    # bp_idx[0] is a dict with bodyparts grouped for direct view
    # bp_idx[1] is a dict with bodyparts grouped for mirror view

    is_estimate = [[], []]
    final_direct_pawdorsum_pts = [[], []]
    for i_paw, paw in enumerate(('left', 'right')):

        direct_pd_idx = bp_idx[0]['pawdorsum'][i_paw]
        mirror_pd_idx = bp_idx[1]['pawdorsum'][i_paw]

        direct_pawdorsum_pts_ud = np.squeeze(bp_coords[0][direct_pd_idx, :, :])
        mirror_pawdorsum_pts_ud = np.squeeze(bp_coords[1][mirror_pd_idx, :, :])

        final_direct_pawdorsum_pts[i_paw] = np.squeeze(bp_coords[0][direct_pd_idx, :, :])
        is_estimate[i_paw] = np.zeros(num_frames_total, dtype=bool)

        all_direct_digit_idx = []
        for digitpart in digitparts:
            all_direct_digit_idx.extend(bp_idx[0][digitpart][i_paw])

        direct_digit_pts = bp_coords[0][all_direct_digit_idx, :, :]

        invalid_direct_pd = np.squeeze(invalid_points[0][direct_pd_idx, :])
        invalid_mirror_pd = np.squeeze(invalid_points[1][mirror_pd_idx, :])

        for i_ftc in frames_to_check:
            i_frame = frames_to_check[i_ftc]

            if invalid_direct_pd[i_frame]:
                # paw dorsum was not reliably found in the direct view

                valid_direct_idx = np.logical_not(invalid_points[0][all_direct_digit_idx, i_frame])
                valid_direct_points = direct_digit_pts[valid_direct_idx, i_frame, :]

                if not invalid_mirror_pd[i_frame]:
                    # direct view paw dorsum is constrained to be on the epipolar line through the mirror view point
                    mirror_pd_pt = mirror_pawdorsum_pts_ud[i_frame, :]
                    cur_epiline = cv2.computeCorrespondEpilines(mirror_pd_pt.reshape(-1, 1, 2), 1, F)
                    epiline = np.squeeze(cur_epiline)
                    edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

                found_valid_points = False

                valid_mcp = np.logical_not(invalid_points[0][bp_idx[0]['mcp'][i_paw], i_frame])
                valid_pip = np.logical_not(invalid_points[0][bp_idx[0]['pip'][i_paw], i_frame])
                valid_dig = np.logical_not(invalid_points[0][bp_idx[0]['dig'][i_paw], i_frame])
                # first, look for valid MCPs, then PIPs, then digit tips

                if all(valid_mcp[[1, 2]]) or all(valid_mcp[[0, 3]]):
                    digit_pts = bp_coords[0][bp_idx[0]['mcp'][i_paw], i_frame, :]
                    valid_pts = valid_mcp
                    found_valid_points = True
                elif all(valid_pip[[1, 2]]) or all(valid_pip[[0, 3]]):
                    digit_pts = bp_coords[0][bp_idx[0]['pip'][i_paw], i_frame, :]
                    valid_pts = valid_pip
                    found_valid_points = True
                elif all(valid_dig[[1, 2]]) or all(valid_dig[[0, 3]]):
                    digit_pts = bp_coords[0][bp_idx[0]['dig'][i_paw], i_frame, :]
                    valid_pts = valid_dig
                    found_valid_points = True
                elif any(valid_mcp):
                    digit_pts = bp_coords[0][bp_idx[0]['mcp'][i_paw], i_frame, :]
                    valid_pts = valid_mcp
                    found_valid_points = True
                elif any(valid_pip):
                    digit_pts = bp_coords[0][bp_idx[0]['pip'][i_paw], i_frame, :]
                    valid_pts = valid_pip
                    found_valid_points = True
                elif any(valid_dig):
                    digit_pts = bp_coords[0][bp_idx[0]['dig'][i_paw], i_frame, :]
                    valid_pts = valid_dig
                    found_valid_points = True

                if found_valid_points and not invalid_mirror_pd[i_frame]:

                    digits_midpoint = find_digits_midpoint(digit_pts, valid_pts)

                    # does the epipolar line intersect the region bounded by the identified points?
                    if np.sum(valid_direct_idx) == 1:
                        # if only one valid knuckle found, can't be an intersection with the epipolarline
                        intersect_obj = None
                    elif np.sum(valid_direct_idx) == 2:
                        intersect_obj = cvb.find_line_intersection(edge_pts, valid_direct_points)
                    else:
                        intersect_obj = cvb.line_convex_hull_intersect(edge_pts, valid_direct_points)

                    # test if there was an intersection between the epipolar line and convex hull
                    if intersect_obj is None:

                        d, new_pt = cvb.find_nearest_point_on_line(edge_pts, digits_midpoint)

                        if d < max_dist_from_neighbor:
                            final_direct_pawdorsum_pts[i_paw][i_frame, :] = new_pt
                            is_estimate[i_paw][i_frame] = True

                    else:

                        if type(intersect_obj) is sg.LineString:
                            d, new_pt = cvb.find_nearest_point_on_line(intersect_obj, digits_midpoint)
                        elif type(intersect_obj) is sg.Point:
                            new_pt = np.array([intersect_obj.coords.xy[0][0], intersect_obj.coords.xy[1][0]])
                            d = intersect_obj.distance(sg.asPoint(digits_midpoint))
                        if d < max_dist_from_neighbor:
                            final_direct_pawdorsum_pts[i_paw][i_frame, :] = new_pt
                            is_estimate[i_paw][i_frame] = True

    return final_direct_pawdorsum_pts, is_estimate

def find_digits_midpoint(digit_pts, valid_pts):
    '''
    function to find the presumed midpoint of a set of identified digits from deeplabcut. Algorithm is as follows:
        1. If the 2nd and 3rd digits were identified, the output is the average location of those 2 digits
        2. If the 1st and 4th digits were identified, the output is the average location of those 2 digits
        3. If the 2nd OR 3rd digit is identified (but not the 1st and 4th), take that digit
        4. If only the 1st or 4th digit is identified, take that one
    :param digit_pts:
    :param valid_pts:
    :return:
    '''
    if not any(valid_pts):
        digit_midpoint = None
    elif all(valid_pts[[1, 2]]):
        digit_midpoint = np.mean(digit_pts[1:2, :], axis=0)
    elif all(valid_pts[[0, 3]]):
        digit_midpoint = np.mean(digit_pts[[0, 3], :], axis=0)
    elif valid_pts[1]:
        digit_midpoint = digit_pts[1, :]
    elif valid_pts[2]:
        digit_midpoint = digit_pts[2, :]
    elif valid_pts[0]:
        digit_midpoint = digit_pts[0, :]
    elif valid_pts[3]:
        digit_midpoint = digit_pts[3, :]

    return digit_midpoint


def estimate_paw_part(known_pt, next_digit_knuckles, other_knuckle_pts, next_knuckle_pt, valid_paw_pts, F, im_size, max_dist_from_neighbor):


    # doesn't matter if we label it as image 1 or 2 (2nd argument of computeCorrespondEpilines). You get the same edge
    # points either way
    epiline = cv2.computeCorrespondEpilines(known_pt.reshape(-1, 1, 2), 1, F)   # since it's a mirror in the same image, does it matter if we label it image 1 or 2?
    epiline = np.squeeze(epiline)
    edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

    # try finding the convex hull of the paw parts that were identified in the "other" view if there are at least 3
    # valid points identified on the paw. If only 2 valid points, create the line between those two points
    if np.shape(valid_paw_pts)[0] > 2:
        # need at least 3 points to calculate a convex hull

        # find intersection between epipolar line and polygon defined by paw boundary
        paw_hull = ConvexHull(valid_paw_pts)
        epi_paw_intersect = cvb.line_polygon_intersect(edge_pts, valid_paw_pts[paw_hull.vertices, :])
    elif np.shape(valid_paw_pts)[0] == 2:
        epi_paw_intersect = cvb.find_line_intersection(edge_pts, valid_paw_pts)
    elif np.shape(valid_paw_pts)[0] < 2:
        epi_paw_intersect = None

    if epi_paw_intersect is None:
        # no intersection between the epipolar line and polygon bounded by other paw points

        if len(other_knuckle_pts) == 0:
            # didn't find any other of this knuckle on other digits (I think) - that is, if the reference knuckle is
            # mcp1, didn't find mcp2, 3, or 4
            # find the point on the epipolar line closest to the neck knuckle up the same digit
            nndist, nn_pt = cvb.find_nearest_point_on_line(edge_pts, next_knuckle_pt)
            if nndist < max_dist_from_neighbor:
                new_pt = np.array([nn_pt.coords.xy[0][0], nn_pt.coords.xy[1][0]])
            else:
                new_pt = None
        else:
            nndist, nn_pt = cvb.find_nearest_point_on_line(edge_pts, other_knuckle_pts)
            if nndist < max_dist_from_neighbor:
                new_pt = np.array([nn_pt.coords.xy[0][0], nn_pt.coords.xy[1][0]])
            else:
                new_pt = None
    else:
        if len(next_knuckle_pt) == 0:
            # the epipolar line intersects the polygon defined by the points that were found in the other view, but the
            # point for the next knuckle on the same digit wasn't found either

            if np.logical_not(np.isnan(next_digit_knuckles[:])).any():
                # at least one knuckle on a neighboring digit was found.
                # is this digit 2 or 3, and were both neighboring digits found?
                if np.logical_not(np.isnan(next_digit_knuckles[:])).all():
                    # both neighboring digits were found
                    # find the intersection between the epipolar line and the segment connecting the two adjacent
                    # knuckles
                    li = cvb.find_line_intersection(edge_pts, next_digit_knuckles)
                    if type(li) is sg.Point:
                        new_pt = np.array([li.coords.xy[0][0], li.coords.xy[1][0]])
                    elif li is None:
                        new_pt = None
                else:
                    # only one neighboring digit was found
                    # look for the closest point in the intersection of the epipolar line with the same knuckle on the
                    # neighboring digits

                    nndist, nn_pt = cvb.find_nearest_point_on_line(edge_pts, next_digit_knuckles)
                    if nndist < max_dist_from_neighbor:
                        new_pt = np.array([nn_pt.coords.xy[0][0], nn_pt.coords.xy[1][0]])
                    else:
                        new_pt = None
            else:
                # the neighboring digits for the same knuckle weren't found either
                # find index of other_pts that is closest to the epipolar line
                nndist, nn_pt = cvb.find_nearest_point_on_line(edge_pts, other_knuckle_pts)
                if nndist < max_dist_from_neighbor:
                    new_pt = np.array([nn_pt.coords.xy[0][0], nn_pt.coords.xy[1][0]])
                else:
                    new_pt = None
        else:
            # the epipolar line intersects the polygon defined by the points that were found in the other view. look
            # for the intersection point closest to the next knuckle on the same digit
            if type(epi_paw_intersect) is sg.Point:
                nndist, nn_pt = cvb.find_nearest_neighbor(epi_paw_intersect, next_knuckle_pt)
            else:
                nndist, nn_pt = cvb.find_nearest_point_on_line(epi_paw_intersect, next_knuckle_pt)

            if nndist < max_dist_from_neighbor:
                new_pt = np.array([nn_pt.coords.xy[0][0], nn_pt.coords.xy[1][0]])
            else:
                new_pt = None

    return new_pt


def collect_all_paw_parts_idx(bp_idx, i_paw):

    all_parts_idx = [[], []]
    bp_keys = tuple(bp_idx[0].keys())

    for i_view in range(2):

        for bp_group in bp_keys:

            if bp_group in ('mcp', 'pip', 'dig', 'pawdorsum', 'palm'):
                all_parts_idx[i_view].extend(bp_idx[i_view][bp_group][i_paw])

    return all_parts_idx



def find_other_knuckle_pts(view_bp_coords, view_bodyparts, view_invalid_points, paw, digitpart, i_frame):
    # extract the locations of knuckles in this view for the paw ('left' or 'right') at the given knuckle ('mcp', 'pip',
    # or 'dig'). i.e., if 'mcp', left paw, find coordinates of the mcp of all four digits for the left paw

    test_string = paw + digitpart

    knuckle_indices = [i_bp for i_bp, bp in enumerate(view_bodyparts) if test_string in bp]

    other_knuckle_pts = view_bp_coords[knuckle_indices, i_frame, :]
    valid_knuckle_pts = other_knuckle_pts[np.logical_not(view_invalid_points[knuckle_indices, i_frame]), :]

    valid_knuckle_pts = np.squeeze(valid_knuckle_pts)

    return valid_knuckle_pts


def group_dlc_bodyparts(bodyparts):

    mcp_idx = []
    pip_idx = []
    dig_idx = []
    pawdorsum_idx = []
    palm_idx = []
    elbow_idx = []
    ear_idx = []
    eye_idx = []
    for side in ('left', 'right'):
        test_mcp_string = side + 'mcp'
        test_pip_string = side + 'pip'
        test_dig_string = side + 'dig'
        test_pawdorsum_string = side + 'pawdorsum'
        test_palm_string = side + 'palm'
        test_elbow_string = side + 'elbow'
        test_ear_string = side + 'ear'
        test_eye_string = side + 'eye'

        mcp_idx.append([i_bp for i_bp, bp in enumerate(bodyparts) if test_mcp_string in bp])
        pip_idx.append([i_bp for i_bp, bp in enumerate(bodyparts) if test_pip_string in bp])
        dig_idx.append([i_bp for i_bp, bp in enumerate(bodyparts) if test_dig_string in bp])

        pawdorsum_idx.append([i_bp for i_bp, bp in enumerate(bodyparts) if test_pawdorsum_string in bp])
        palm_idx.append([i_bp for i_bp, bp in enumerate(bodyparts) if test_palm_string in bp])
        elbow_idx.append([i_bp for i_bp, bp in enumerate(bodyparts) if test_elbow_string in bp])
        ear_idx.append([i_bp for i_bp, bp in enumerate(bodyparts) if test_ear_string in bp])
        eye_idx.append([i_bp for i_bp, bp in enumerate(bodyparts) if test_eye_string in bp])

    nose_idx = [i_bp for i_bp, bp in enumerate(bodyparts) if 'nose' in bp]

    # for all bodyparts that are bilateral, these are lists where the first element is for left side, second element for the right side
    bp_idx = {
        'mcp': mcp_idx,
        'pip': pip_idx,
        'dig': dig_idx,
        'pawdorsum': pawdorsum_idx,
        'palm': palm_idx,
        'elbow': elbow_idx,
        'ear': ear_idx,
        'eye': eye_idx,
        'nose': nose_idx
    }

    return bp_idx


def find_invalid_DLC_points(dlc_data, paw_pref, maxdisperframe=30, min_valid_p=0.85, min_certain_p=0.97, max_neighbor_dist=70):

    view_list = tuple(dlc_data.keys())

    bodyparts = tuple(dlc_data['direct'].keys())

    num_frames = np.shape(dlc_data['direct'][bodyparts[0]]['coordinates_ud'])[0]

    invalid_points = []
    diff_per_frame = []
    for view in view_list:
        temp_invalid_points, temp_diff_per_frame = find_invalid_DLC_single_view(dlc_data[view], paw_pref)
        invalid_points.append(temp_invalid_points)
        diff_per_frame.append(temp_diff_per_frame)

    # todo: add in ability to manually invalidate points here?
    return invalid_points, diff_per_frame


def find_invalid_DLC_single_view(view_dlc_data, paw_pref, maxdistperframe=30, min_valid_p=0.85, min_certain_p=0.97, max_neighbor_dist=70):

    bodyparts = tuple(view_dlc_data.keys())

    num_bodyparts = len(bodyparts)
    num_frames = np.shape(view_dlc_data[bodyparts[0]]['coordinates_ud'])[0]

    p = np.zeros((num_bodyparts, num_frames))
    for i_bp, bp in enumerate(bodyparts):
        p[i_bp, :] = np.squeeze(view_dlc_data[bp]['confidence'])

    # collect the paw part indices for both paws - index 0 for left paw, index 1 for right paw
    temp_paw_part_idx, temp_pawdorsum_idx, temp_palm_idx = find_reaching_pawparts(bodyparts, 'left')
    paw_part_idx = [temp_paw_part_idx]
    paw_dorsum_idx = [temp_pawdorsum_idx]
    palm_idx = [temp_palm_idx]
    temp_paw_part_idx, temp_pawdorsum_idx, temp_palm_idx = find_reaching_pawparts(bodyparts, 'right')
    paw_part_idx.append(temp_paw_part_idx)
    paw_dorsum_idx.append(temp_pawdorsum_idx)
    palm_idx.append(temp_palm_idx)

    invalid_points = p < min_valid_p
    certain_points = p > min_certain_p

    diff_per_frame = np.zeros((num_bodyparts, num_frames-1))
    poss_too_far = np.zeros((num_bodyparts, num_frames), dtype=bool)

    all_part_coords = np.zeros((num_bodyparts, num_frames, 2))
    for i_bp, bp in enumerate(bodyparts):

        individual_part_coords = view_dlc_data[bp]['coordinates_ud']
        individual_part_coords[invalid_points[i_bp, :], :] = np.nan
        all_part_coords[i_bp, :, :] = individual_part_coords

        coord_diffs = np.diff(individual_part_coords, n=1, axis=0)
        diff_per_frame[i_bp, :] = np.linalg.norm(coord_diffs, axis=1)

        poss_too_far[i_bp, :-1] = diff_per_frame[i_bp, :] > maxdistperframe
        poss_too_far[i_bp, 1:] = np.logical_or(poss_too_far[i_bp, :-1], poss_too_far[i_bp, 1:])
        # logic is that either the point before or point after could be the bad point if there was too big a location jump between frames

        poss_too_far[i_bp, :] = np.logical_or(poss_too_far[i_bp, :], np.isnan(invalid_points[i_bp, :]))
        # any nan's from low probability parts should be included as potentially too big a jump

        poss_too_far[i_bp, certain_points[i_bp, :]] = False
        # keep any points with p > min_certain_p even if it apparently traveled too far in one frame

    invalid_points = np.logical_or(invalid_points, poss_too_far)

    # make sure all the paw parts are close to each other
    for i_paw in range(2):
        # left paw is index 1, right paw is index 2
        for i_frame in range(num_frames):
            cur_valid_idx = np.nonzero(np.logical_not(invalid_points[paw_part_idx[i_paw], i_frame]))[0]
            num_valid_points = len(cur_valid_idx)

            if num_valid_points > 3:

                cur_paw_coords = np.squeeze(all_part_coords[paw_part_idx[i_paw], i_frame, :])
                valid_paw_coords = cur_paw_coords[cur_valid_idx, :]
                for i_pt in range(num_valid_points):
                    test_idx = np.zeros(num_valid_points, dtype=bool)
                    test_idx[i_pt] = True
                    test_point = valid_paw_coords[test_idx, :]
                    other_points = valid_paw_coords[np.logical_not(test_idx)]

                    nn_dist, _ = cvb.find_nearest_neighbor(test_point, other_points)

                    if nn_dist > max_neighbor_dist:
                        # throw out any points on this paw that are too far away from the cluster of other points, except for the paw dorsum or palm
                        invalidate_idx = cur_valid_idx[i_pt]
                        if invalidate_idx != paw_dorsum_idx[i_paw] and invalidate_idx != palm_idx[i_paw]:
                            invalid_points[paw_part_idx[i_paw][cur_valid_idx[i_pt]], i_frame] = True

    return invalid_points, diff_per_frame


def find_reaching_pawparts(bodyparts, paw_pref, mcp_string='mcp', pip_string='pip', dig_string='dig', pawdorsum_string='pawdorsum', palm_string='palm'):

    test_mcp_string = paw_pref.lower() + mcp_string
    test_pip_string = paw_pref.lower() + pip_string
    test_dig_string = paw_pref.lower() + dig_string
    test_pawdorsum_string = paw_pref.lower() + pawdorsum_string
    test_palm_string = paw_pref.lower() + palm_string

    mcp_idx = [i_bp for i_bp, bp in enumerate(bodyparts) if test_mcp_string in bp]
    pip_idx = [i_bp for i_bp, bp in enumerate(bodyparts) if test_pip_string in bp]
    dig_idx = [i_bp for i_bp, bp in enumerate(bodyparts) if test_dig_string in bp]

    pawdorsum_idx = [i_bp for i_bp, bp in enumerate(bodyparts) if test_pawdorsum_string in bp]
    palm_idx = [i_bp for i_bp, bp in enumerate(bodyparts) if test_palm_string in bp]

    # return the full list of paw part indices in the bodyparts list, but also paw dorsum and palm indices separately
    return mcp_idx + pip_idx + dig_idx + pawdorsum_idx + palm_idx, pawdorsum_idx, palm_idx


def test_undistortion(dlc_data, invalid_points, cal_data, direct_pickle_params):

    min_p = 0.9

    videos_parent = '/home/levlab/Public/rat_SR_videos_to_analyze'   # on the lambda machine
    # videos_parent = '/Users/dan/Documents/deeplabcut/videos_to_analyze'  # on home mac
    # videos_parent = '/Volumes/Untitled/videos_to_analyze'
    video_root_folder = os.path.join(videos_parent, 'videos_to_crop')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_videos')

    # find original video
    orig_vid_name = navigation_utilities.find_orig_rat_video(direct_pickle_params, video_root_folder)
    orig_vid_folder, _ = os.path.split(orig_vid_name)

    test_frame = 300
    vo = cv2.VideoCapture(orig_vid_name)
    vo.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, cur_img = vo.read()
    vo.release()

    jpg_name = os.path.join(orig_vid_folder, 'test.jpg')
    markertypes = ['o', '+']
    overlay_pts_on_orig_ratframe(cur_img, cal_data, dlc_data, invalid_points, markertypes, jpg_name, test_frame, min_p)


def test_reprojection(dlc_data, invalid_points, proj_points, direct_pickle_params, cal_data):

    min_p = 0.9

    videos_parent = '/home/levlab/Public/rat_SR_videos_to_analyze'   # on the lambda machine
    # videos_parent = '/Users/dan/Documents/deeplabcut/videos_to_analyze'  # on home mac
    # videos_parent = '/Volumes/Untitled/videos_to_analyze'
    video_root_folder = os.path.join(videos_parent, 'videos_to_crop')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_videos')

    # find original video
    orig_vid_name = navigation_utilities.find_orig_rat_video(direct_pickle_params, video_root_folder)
    orig_vid_folder, _ = os.path.split(orig_vid_name)

    test_frame = 300
    vo = cv2.VideoCapture(orig_vid_name)
    vo.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, cur_img = vo.read()
    vo.release()

    jpg_name = os.path.join(orig_vid_folder, 'test.jpg')
    markertypes = ['o', '+']
    overlay_reproj_pts_on_orig_ratframe(cur_img, cal_data, dlc_data, invalid_points, proj_points, markertypes, jpg_name, test_frame, min_p)


def overlay_reproj_pts_on_orig_ratframe(img, cal_data, dlc_data, invalid_points, proj_points, markertypes, jpg_name, test_frame, min_p=0):

    dotsize = 6
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    bp_c = sr_visualization.bp_colors()

    im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = np.shape(im_bgr)
    fig_ud_direct_epi, ax_ud_direct_epi = prepare_img_axes(w, h)
    # fig_ud_mirror_epi, ax_ud_mirror_epi = prepare_img_axes(w, h)

    # fig, ax = prepare_img_axes(w, h)

    img_ud = cv2.undistort(im_bgr, mtx, dist)

    ax_ud_direct_epi[0][0].imshow(img_ud)
    # ax_ud_mirror_epi[0][0].imshow(img_ud)

    # ax[0][0].imshow(im_bgr)

    for i_view, view in enumerate(dlc_data.keys()):
        view_dlcdata = dlc_data[view]
        view_invalidpoints = invalid_points[i_view]

        bodyparts = view_dlcdata.keys()

        for i_bp, bp in enumerate(bodyparts):
            p = view_dlcdata[bp]['confidence'][test_frame][0]
            isvalid = not view_invalidpoints[i_bp, test_frame]

            # if p > min_p:
            if isvalid:

                cur_pt = view_dlcdata[bp]['coordinates'][test_frame]

                if all(cur_pt == 0):
                    continue

                cur_pt_ud = view_dlcdata[bp]['coordinates_ud'][test_frame]
                cur_proj_pt = proj_points[i_view][i_bp, test_frame, :]

                # ax[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[0])
                # ax[0][0].scatter(cur_proj_pt[0], cur_proj_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[1])

                ax_ud_direct_epi[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
                ax_ud_direct_epi[0][0].scatter(cur_proj_pt[0], cur_proj_pt[1], c=bp_c[bp], marker=markertypes[1])

                # ax_ud_mirror_epi[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
                # ax_ud_mirror_epi[0][0].scatter(cur_proj_pt[0], cur_proj_pt[1], c=bp_c[bp], marker=markertypes[1])

    # overlay epipolar line
    # draw_epipolar_lines(dlc_data, cal_data, test_frame, [ax_ud_direct_epi[0][0], ax_ud_mirror_epi[0][0]], (w, h), invalid_points)
    # draw_epipolar_lines(dlc_data, cal_data, test_frame, ax_ud_mirror_epi[0][0], (w, h), invalid_points)
    plt.show()

    pass


def test_estimated_pts(dlc_data, new_pt, paw, bp, invalid_points, cal_data, direct_pickle_params):

    videos_parent = '/home/levlab/Public/rat_SR_videos_to_analyze'   # on the lambda machine
    # videos_parent = '/Users/dan/Documents/deeplabcut/videos_to_analyze'  # on home mac
    # videos_parent = '/Volumes/Untitled/videos_to_analyze'
    video_root_folder = os.path.join(videos_parent, 'videos_to_crop')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_videos')

    # find original video
    orig_vid_name = navigation_utilities.find_orig_rat_video(direct_pickle_params, video_root_folder)
    orig_vid_folder, _ = os.path.split(orig_vid_name)

    test_frame = 300
    vo = cv2.VideoCapture(orig_vid_name)
    vo.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, cur_img = vo.read()
    vo.release()

    bp_name = paw + bp
    markertypes = ['o', '+']
    overlay_estimated_pts_on_orig_ratframe(cur_img, cal_data, dlc_data, invalid_points, markertypes, bp_name, test_frame, new_pt)


def triangulate_video(video_id, videos_parent, marked_videos_parent, calibration_parent, dlc_mat_output_parent, rat_df,
                      view_list=('direct', 'leftmirror', 'rightmirror'),
                      min_confidence=0.95):

    if isinstance(video_id, str):
        video_metadata = navigation_utilities.parse_video_name(video_id)
    else:
        video_metadata = video_id

    video_metadata['paw_pref'] = rat_df[rat_df['ratID'] == video_metadata['rat_num']]['pawPref'].values[0]
    dlc_output_pickle_names, dlc_metadata_pickle_names = navigation_utilities.find_dlc_output_pickles(video_metadata, marked_videos_parent, view_list=view_list)
    # above line will not complete if all pickle files with DLC output data are not found

    # find the calibration files
    calibration_file = navigation_utilities.find_calibration_file(video_metadata, calibration_parent)
    if calibration_file == '':
        return

    camera_params = skilled_reaching_io.read_matlab_calibration(calibration_file)
    # above lines will not complete if a calibration file is not found

    # read in the pickle files
    dlc_output = {view: None for view in view_list}
    dlc_metadata = {view: None for view in view_list}
    pickle_name_metadata = {view: None for view in view_list}
    for view in view_list:
        if not dlc_output_pickle_names[view] is None:
            dlc_output[view] = skilled_reaching_io.read_pickle(dlc_output_pickle_names[view])
            dlc_metadata[view] = skilled_reaching_io.read_pickle(dlc_metadata_pickle_names[view])
            pickle_name_metadata[view] = navigation_utilities.parse_dlc_output_pickle_name(dlc_output_pickle_names[view])

    trajectory_filename = navigation_utilities.create_trajectory_filename(video_metadata)

    trajectory_metadata = dlc_utilities.extract_trajectory_metadata(dlc_metadata, pickle_name_metadata)

    dlc_data = dlc_utilities.extract_data_from_dlc_output(dlc_output, trajectory_metadata)
    #todo: preprocessing to get rid of "invalid" points

    # translate and undistort points
    dlc_data = translate_points_to_full_frame(dlc_data, trajectory_metadata)
    dlc_data = undistort_points(dlc_data, camera_params)

    mat_data = package_data_into_mat(dlc_data, video_metadata, trajectory_metadata)
    mat_name = navigation_utilities.create_mat_fname_dlc_output(video_metadata, dlc_mat_output_parent)

    video_name = navigation_utilities.build_video_name(video_metadata, videos_parent)
    # test_pt_alignment(video_name, dlc_data)

    sio.savemat(mat_name, mat_data)

    # reconstruct 3D points
    # reconstruct_trajectories(dlc_data, camera_params)
    pass


def reconstruct_trajectories(dlc_data_ud, camera_params):

    view_list = tuple(dlc_data_ud.keys())
    bodyparts = dlc_data_ud[view_list[0]].keys()   # should be bodyparts from the direct view

    for bp in bodyparts:

        pass


def translate_points_to_full_frame(dlc_data, trajectory_metadata):

    view_list = tuple(trajectory_metadata.keys())

    for view in view_list:
        if trajectory_metadata[view] is None:
            continue

        if view == 'rightmirror':
            crop_width = trajectory_metadata[view]['crop_window'][1] - trajectory_metadata[view]['crop_window'][0] + 1
            # images were reversed after cropping, so need to reverse back before undistorting. Also, left and right
            # labels were swapped in the right mirror view
            for bp in trajectory_metadata[view]['bodyparts']:
                for i_frame in range(trajectory_metadata[view]['num_frames']):
                    if not np.all(dlc_data[view][bp]['coordinates'][i_frame] == 0):
                        # a point was found in this frame (coordinate == 0 if no point found)
                        # x-values should be reflected across the midline of the cropped field
                        x = dlc_data[view][bp]['coordinates'][i_frame, 0]
                        dlc_data[view][bp]['coordinates'][i_frame, 0] = (crop_width - x) + 1

            for bp in trajectory_metadata[view]['bodyparts']:
                # if the right mirror view, need to swap left-sided bodyparts for right-sided
                if 'right' in bp:
                    contra_bp = bp.replace('right', 'left')
                    trajectory_placeholder = dlc_data[view][bp]['coordinates']
                    confidence_placeholder = dlc_data[view][bp]['confidence']
                    dlc_data[view][bp]['coordinates'] = dlc_data[view][contra_bp]['coordinates']
                    dlc_data[view][bp]['confidence'] = dlc_data[view][contra_bp]['confidence']
                    dlc_data[view][contra_bp]['coordinates'] = trajectory_placeholder
                    dlc_data[view][contra_bp]['confidence'] = confidence_placeholder
                    # don't also swap left for right or we'll just swap them back to where they started

        for bp in trajectory_metadata[view]['bodyparts']:
            # translate point
            for i_frame in range(trajectory_metadata[view]['num_frames']):
                if not np.all(dlc_data[view][bp]['coordinates'][i_frame] == 0):
                    # a point was found in this frame (coordinate == 0 if no point found)
                    dlc_data[view][bp]['coordinates'][i_frame] += [trajectory_metadata[view]['crop_window'][0], trajectory_metadata[view]['crop_window'][2]]
                    dlc_data[view][bp]['coordinates'][i_frame] -= 1

    return dlc_data


def undistort_points(dlc_data, camera_params):

    view_list = dlc_data.keys()

    for view in view_list:

        if dlc_data[view] is None:
            continue

        bodyparts = dlc_data[view].keys()

        for bp in bodyparts:
            num_rows = np.shape(dlc_data[view][bp]['coordinates'])[0]
            dlc_data[view][bp]['coordinates_ud'] = np.zeros((num_rows, 2))
            for i_row, row in enumerate(dlc_data[view][bp]['coordinates']):
                if not np.all(row == 0):
                    # a point was found in this frame (coordinate == 0 if no point found)
                    norm_pt_ud = cv2.undistortPoints(row, camera_params['mtx'], camera_params['dist'])
                    pt_ud = unnormalize_points(norm_pt_ud, camera_params['mtx'])
                    dlc_data[view][bp]['coordinates_ud'][i_row, :] = np.squeeze(pt_ud)

    return dlc_data


def normalize_points(pts, mtx):

    pass

def unnormalize_points(pts, mtx):

    homogeneous_pts = np.squeeze(cv2.convertPointsToHomogeneous(pts))
    unnormalized_pts = np.matmul(mtx, homogeneous_pts)
    unnormalized_pts = cv2.convertPointsFromHomogeneous(np.array([unnormalized_pts]))    # np.array needed to make dimensions work for the function call

    return unnormalized_pts


def test_pt_alignment(video_name, dlc_data):

    circ_r = 4
    circ_t = 1

    view_list = tuple(dlc_data.keys())
    bodyparts = tuple(dlc_data[view_list[0]].keys())

    video_object = cv2.VideoCapture(video_name)

    frame_counter = 250
    video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
    ret, cur_img = video_object.read()

    bp_to_view = bodyparts
    for view in view_list:
        for bp in bodyparts:
            if bp in bp_to_view:
                if 'left' in bp:
                    circ_color = (0, 0, 255)
                elif 'right' in bp:
                    circ_color = (0, 255, 0)
                else:
                    circ_color = (255, 0, 0)
                x, y = dlc_data[view][bp]['coordinates'][frame_counter]
                try:
                    x = int(round(x))
                    y = int(round(y))
                    cur_img = cv2.circle(cur_img, (x,y), circ_r, circ_color, thickness=circ_t)
                except:
                    pass

    cv2.imshow('image', cur_img)
    cv2.waitKey(0)

    video_object.release()


def prepare_img_axes(width, height, scale=1.0, dpi=100, nrows=1, ncols=1):
    fig_width = (width * scale / dpi) * ncols
    fig_height = (width * scale / dpi) * nrows
    fig = plt.figure(
        frameon=False, figsize=(fig_width, fig_height), dpi=dpi
    )

    axs = []
    for i_row in range(nrows):
        ax_row = []
        for i_col in range(ncols):
            idx = (i_row * ncols) + i_col + 1
            ax_row.append(fig.add_subplot(nrows, ncols, idx))
            ax_row[i_col].axis("off")
            ax_row[i_col].set_xlim(0, width)
            ax_row[i_col].set_ylim(0, height)
            ax_row[i_col].invert_yaxis()
        axs.append(ax_row)

    # plt.subplots_adjust(0., 0., 0., 0., 0.)

    return fig, axs


def test_undistortion(dlc_data, invalid_points, cal_data, direct_pickle_params):

    min_p = 0.9

    videos_parent = '/home/levlab/Public/rat_SR_videos_to_analyze'   # on the lambda machine
    # videos_parent = '/Users/dan/Documents/deeplabcut/videos_to_analyze'  # on home mac
    # videos_parent = '/Volumes/Untitled/videos_to_analyze'
    video_root_folder = os.path.join(videos_parent, 'videos_to_crop')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_videos')

    # find original video
    orig_vid_name = navigation_utilities.find_orig_rat_video(direct_pickle_params, video_root_folder)
    orig_vid_folder, _ = os.path.split(orig_vid_name)

    test_frame = 300
    vo = cv2.VideoCapture(orig_vid_name)
    vo.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, cur_img = vo.read()
    vo.release()

    jpg_name = os.path.join(orig_vid_folder, 'test.jpg')
    markertypes = ['o', '+']
    overlay_pts_on_orig_ratframe(cur_img, cal_data, dlc_data, invalid_points, markertypes, jpg_name, test_frame, min_p)


def overlay_estimated_pts_on_orig_ratframe(img, cal_data, dlc_data, invalid_points, markertypes, bp_name, test_frame, new_pt):

    dotsize = 6
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    bp_c = sr_visualization.bp_colors()

    im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = np.shape(im_bgr)
    fig_ud_direct_epi, ax_ud_direct_epi = prepare_img_axes(w, h)
    fig_ud_direct_epi.suptitle(bp_name)
    # fig_ud_mirror_epi, ax_ud_mirror_epi = prepare_img_axes(w, h)

    # fig, ax = prepare_img_axes(w, h)

    img_ud = cv2.undistort(im_bgr, mtx, dist)

    ax_ud_direct_epi[0][0].imshow(img_ud)
    # ax_ud_mirror_epi[0][0].imshow(img_ud)

    ax_ud_direct_epi[0][0].scatter(new_pt[0], new_pt[1], edgecolors='k', facecolors='none', marker=markertypes[0])

    # ax[0][0].imshow(im_bgr)

    for i_view, view in enumerate(dlc_data.keys()):
        view_dlcdata = dlc_data[view]
        view_invalidpoints = invalid_points[i_view]

        bodyparts = view_dlcdata.keys()

        for i_bp, bp in enumerate(bodyparts):
            p = view_dlcdata[bp]['confidence'][test_frame][0]
            isvalid = not view_invalidpoints[i_bp, test_frame]

            # if p > min_p:
            if isvalid:

                cur_pt = view_dlcdata[bp]['coordinates'][test_frame]

                if all(cur_pt == 0):
                    continue

                cur_pt_ud = view_dlcdata[bp]['coordinates_ud'][test_frame]

                # ax[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
                # ax[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

                ax_ud_direct_epi[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
                ax_ud_direct_epi[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

                # ax_ud_mirror_epi[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
                # ax_ud_mirror_epi[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

    # overlay epipolar line
    draw_epi_line(new_pt, dlc_data, cal_data, test_frame, ax_ud_direct_epi[0][0], (w, h))
    # draw_epi_line(new_pt, cal_data, test_frame, ax_ud_mirror_epi[0][0], (w, h))
    plt.show()

    pass


def overlay_pts_on_orig_ratframe(img, cal_data, dlc_data, invalid_points, markertypes, jpg_name, test_frame, min_p=0):

    dotsize = 6
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    bp_c = sr_visualization.bp_colors()

    im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = np.shape(im_bgr)
    fig_ud_direct_epi, ax_ud_direct_epi = prepare_img_axes(w, h)
    fig_ud_mirror_epi, ax_ud_mirror_epi = prepare_img_axes(w, h)

    fig, ax = prepare_img_axes(w, h)

    img_ud = cv2.undistort(im_bgr, mtx, dist)

    ax_ud_direct_epi[0][0].imshow(img_ud)
    ax_ud_mirror_epi[0][0].imshow(img_ud)

    # ax[0][0].imshow(im_bgr)

    for i_view, view in enumerate(dlc_data.keys()):
        view_dlcdata = dlc_data[view]
        view_invalidpoints = invalid_points[i_view]

        bodyparts = view_dlcdata.keys()

        for i_bp, bp in enumerate(bodyparts):
            p = view_dlcdata[bp]['confidence'][test_frame][0]
            isvalid = not view_invalidpoints[i_bp, test_frame]

            # if p > min_p:
            if isvalid:

                cur_pt = view_dlcdata[bp]['coordinates'][test_frame]

                if all(cur_pt == 0):
                    continue

                cur_pt_ud = view_dlcdata[bp]['coordinates_ud'][test_frame]

                ax[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
                ax[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

                ax_ud_direct_epi[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
                ax_ud_direct_epi[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

                ax_ud_mirror_epi[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
                ax_ud_mirror_epi[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

    # overlay epipolar line
    draw_epipolar_lines(dlc_data, cal_data, test_frame, [ax_ud_direct_epi[0][0], ax_ud_mirror_epi[0][0]], (w, h), invalid_points)
    # draw_epipolar_lines(dlc_data, cal_data, test_frame, ax_ud_mirror_epi[0][0], (w, h), invalid_points)
    plt.show()

    pass


def draw_epi_line(test_pt, dlc_data, cal_data, test_frame, ax, im_size):
    # F[:,:,0] - fundamental matrix between direct and top mirror views
    # F[:,:,1] - fundamental matrix between direct and left mirror views
    # F[:,:,2] - fundamental matrix between direct and right mirror views
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    bp_c = sr_visualization.bp_colors()

    view_list = dlc_data.keys()
    if 'leftmirror' in view_list:
        F = cal_data['F'][:, :, 1]
    elif 'rightmirror' in view_list:
        F = cal_data['F'][:, :, 2]
    else:
        print('"leftmirror" or "rightmirror" must be one of the views for 3D reconstruction')
        return

    cur_epiline = cv2.computeCorrespondEpilines(test_pt.reshape(-1, 1, 2), 1, F)

    epiline = np.squeeze(cur_epiline)
    edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

    if not np.all(edge_pts == 0):
        ax.plot(edge_pts[:, 0], edge_pts[:, 1], color='k', ls='-', marker='.')


def draw_epipolar_lines(dlc_data, cal_data, test_frame, ax, im_size, invalid_points, min_p=0):

    # F[:,:,0] - fundamental matrix between direct and top mirror views
    # F[:,:,1] - fundamental matrix between direct and left mirror views
    # F[:,:,2] - fundamental matrix between direct and right mirror views
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    bp_c = sr_visualization.bp_colors()

    view_list = dlc_data.keys()
    if 'leftmirror' in view_list:
        F = cal_data['F'][:, :, 1]
    elif 'rightmirror' in view_list:
        F = cal_data['F'][:, :, 2]
    else:
        print('"leftmirror" or "rightmirror" must be one of the views for 3D reconstruction')
        return

    for i_view, view in enumerate(view_list):
        view_dlcdata = dlc_data[view]
        view_invalidpoints = invalid_points[i_view]

        bodyparts = view_dlcdata.keys()

        for i_bp, bp in enumerate(bodyparts):
            isvalid = not view_invalidpoints[i_bp, test_frame]
            p = view_dlcdata[bp]['confidence'][test_frame]

            # if p > min_p:
            if isvalid:
                cur_pt = view_dlcdata[bp]['coordinates'][test_frame]

                if all(cur_pt == 0):
                    continue

                cur_pt_ud = view_dlcdata[bp]['coordinates_ud'][test_frame]

                cur_epiline = cv2.computeCorrespondEpilines(cur_pt_ud.reshape(-1, 1, 2), 1, F)

                epiline = np.squeeze(cur_epiline)
                edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

                if not np.all(edge_pts == 0):
                    # ax[i_view].plot(edge_pts[:, 0], edge_pts[:, 1], color=bp_c[bp], ls='-', marker='.')
                    ax[i_view].plot(edge_pts[:, 0], edge_pts[:, 1], color=bp_c[bp], ls='-', marker='.')

    plt.show()

    pass


def package_data_into_mat(dlc_data, video_metadata, trajectory_metadata):

    if video_metadata['paw_pref'] == 'right':
        mirrorview = 'leftmirror'
    else:
        mirrorview = 'rightmirror'

    bodyparts = trajectory_metadata['direct']['bodyparts']
    num_bp = len(bodyparts)
    num_frames = len(dlc_data['direct'][bodyparts[0]]['confidence'])

    direct_pts_ud = np.zeros((num_bp, num_frames, 2))
    mirror_pts_ud = np.zeros((num_bp, num_frames, 2))
    direct_p = np.zeros((num_bp, num_frames))
    mirror_p = np.zeros((num_bp, num_frames))
    for i_bp, bp in enumerate(bodyparts):
        direct_pts_ud[i_bp, :, :] = dlc_data['direct'][bp]['coordinates']
        mirror_pts_ud[i_bp, :, :] = dlc_data[mirrorview][bp]['coordinates']
        direct_p[i_bp, :] = dlc_data['direct'][bp]['confidence'].T
        mirror_p[i_bp, :] = dlc_data[mirrorview][bp]['confidence'].T

    # crop_window is [left, right, top, bottom]
    # ROIs are [left, top, width, height]
    ROI_direct = convert_lrtb_to_ltwh(trajectory_metadata['direct']['crop_window'])
    ROI_mirror = convert_lrtb_to_ltwh(trajectory_metadata[mirrorview]['crop_window'])
    ROIs = np.vstack((ROI_direct, ROI_mirror))

    mat_data = {'direct_pts_ud': direct_pts_ud,
                'mirror_pts_ud': mirror_pts_ud,
                'direct_bp': trajectory_metadata['direct']['bodyparts'],
                'mirror_bp': trajectory_metadata[mirrorview]['bodyparts'],
                'direct_p': direct_p,
                'mirror_p': mirror_p,
                'paw_pref': video_metadata['paw_pref'],
                'im_size': video_metadata['im_size'],
                'video_number': video_metadata['video_number'],
                'ROIs': ROIs
                }

    return mat_data


def convert_lrtb_to_ltwh(crop_window):

    # crop_window is [left, right, top, bottom]
    # ROIs are [left, top, width, height]

    ROI = np.zeros(4)
    ROI[0] = crop_window[0]
    ROI[1] = crop_window[2]
    w = crop_window[1] - crop_window[0] + 1
    h = crop_window[3] - crop_window[2] + 1
    ROI[2] = w
    ROI[3] = h

    return ROI