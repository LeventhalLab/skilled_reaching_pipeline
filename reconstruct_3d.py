import numpy as np
import cv2
import os
import navigation_utilities
import skilled_reaching_io
import pandas as pd
import scipy.io as sio

def triangulate_video(video_id, videos_parent, marked_videos_parent, calibration_parent, dlc_mat_output_parent, rat_df,
                      view_list=None,
                      min_confidence=0.95):

    if view_list is None:
        view_list = ('direct', 'leftmirror', 'rightmirror')

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
        dlc_output[view] = skilled_reaching_io.read_pickle(dlc_output_pickle_names[view])
        dlc_metadata[view] = skilled_reaching_io.read_pickle(dlc_metadata_pickle_names[view])
        pickle_name_metadata[view] = navigation_utilities.parse_dlc_output_pickle_name(dlc_output_pickle_names[view])

    trajectory_filename = navigation_utilities.create_trajectory_filename(video_metadata)

    trajectory_metadata = extract_trajectory_metadata(dlc_metadata, pickle_name_metadata)

    dlc_data = extract_data_from_dlc_output(dlc_output, trajectory_metadata)
    #todo: preprocessing to get rid of "invalid" points

    # translate and undistort points
    dlc_data = translate_points_to_full_frame(dlc_data, trajectory_metadata)
    dlc_data = undistort_points(dlc_data, camera_params)

    mat_data = package_data_into_mat(dlc_data, video_metadata, trajectory_metadata)
    mat_name = navigation_utilities.create_mat_fname_dlc_output(video_metadata, dlc_mat_output_parent)

    # video_name = navigation_utilities.build_video_name(video_metadata, videos_parent)
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


def extract_trajectory_metadata(dlc_metadata, name_metadata):

    view_list = dlc_metadata.keys()
    trajectory_metadata = {view: None for view in view_list}

    for view in view_list:
        trajectory_metadata[view] = {'bodyparts': dlc_metadata[view]['data']['DLC-model-config file']['all_joints_names'],
                                     'num_frames': dlc_metadata[view]['data']['nframes'],
                                     'crop_window': name_metadata[view]['crop_window']
                                     }
    # todo:check that number of frames and bodyparts are the same in each view

    return trajectory_metadata


def translate_points_to_full_frame(dlc_data, trajectory_metadata):

    view_list = tuple(trajectory_metadata.keys())

    for view in view_list:
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


def extract_data_from_dlc_output(dlc_output, trajectory_metadata):

    view_list = dlc_output.keys()

    dlc_data = {view: None for view in view_list}
    for view in view_list:
        # initialize dictionaries for each bodypart
        num_frames = trajectory_metadata[view]['num_frames']
        dlc_data[view] = {bp: None for bp in trajectory_metadata[view]['bodyparts']}
        for i_bp, bp in enumerate(trajectory_metadata[view]['bodyparts']):

            dlc_data[view][bp] = {'coordinates': np.zeros((num_frames, 2)),
                                  'confidence': np.zeros((num_frames, 1)),
                                  }

            for i_frame in range(num_frames):
                frame_key = 'frame{:04d}'.format(i_frame)

                try:
                    dlc_data[view][bp]['coordinates'][i_frame, :] = dlc_output[view][frame_key]['coordinates'][0][i_bp][0]
                    dlc_data[view][bp]['confidence'][i_frame] = dlc_output[view][frame_key]['confidence'][i_bp][0][0]
                except:
                    # 'coordinates' array and 'confidence' array at this frame are empty - must be a peculiarity of deeplabcut
                    # just leave the dlc_data arrays as empty
                    pass

    return dlc_data


def undistort_points(dlc_data, camera_params):

    view_list = dlc_data.keys()

    for view in view_list:
        bodyparts = dlc_data[view].keys()

        for bp in bodyparts:
            for i_row, row in enumerate(dlc_data[view][bp]['coordinates']):
                if not np.all(row == 0):
                    # a point was found in this frame (coordinate == 0 if no point found)
                    norm_pt_ud = cv2.undistortPoints(row, camera_params['mtx'], camera_params['dist'])  # todo: account for distortion coefficients
                    pt_ud = unnormalize_points(norm_pt_ud, camera_params['mtx'])
                    dlc_data[view][bp]['coordinates'][i_row, :] = np.squeeze(pt_ud)

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

    frame_counter = 310
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