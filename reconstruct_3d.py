import numpy as np
import cv2
import os
import navigation_utilities
import glob
import skilled_reaching_io
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt


def reconstruct_folders(folders_to_reconstruct, marked_videos_parent, calibration_files_parent, rat_df):

    for folder_to_reconstruct in folders_to_reconstruct:

        # first, figure out if we have calibration files for this session
        session_date = folder_to_reconstruct['session_date']
        box_num = folder_to_reconstruct['session_box']
        calibration_folder = navigation_utilities.find_calibration_files_folder(session_date, box_num, calibration_files_parent)

        if os.path.exists(calibration_folder):
            # is there a calibration file for this session?

            cal_data = skilled_reaching_io.get_calibration_data(session_date, box_num, calibration_folder)
            reconstruct_folder(folder_to_reconstruct, cal_data, rat_df)
            pass

def reconstruct_folder(folder_to_reconstruct, cal_data, rat_df, view_list=('direct', 'leftmirror', 'rightmirror'), vidtype='.avi'):

    if vidtype[0] is not '.':
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

        pickle_params = {'direct': direct_pickle_params,
                         mirror_view: mirror_pickle_params
                         }
        trajectory_metadata = extract_trajectory_metadata(dlc_metadata, pickle_params)
        dlc_data = extract_data_from_dlc_output(dlc_output, trajectory_metadata)

        # translate and undistort points
        dlc_data = translate_points_to_full_frame(dlc_data, trajectory_metadata)
        dlc_data = undistort_points(dlc_data, cal_data)

        test_undistortion(dlc_data, dlc_metadata, cal_data, direct_pickle_params)


        pass
    pass


def test_undistortion(dlc_data, dlc_metadata, cal_data, direct_pickle_params):
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
    markertype = ['o', '+']
    overlay_pts_on_orig_ratframe(cur_img, cal_data['mtx'], cal_data['dist'], dlc_data, dlc_metadata, markertype, jpg_name)



    pass


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

    trajectory_metadata = extract_trajectory_metadata(dlc_metadata, pickle_name_metadata)

    dlc_data = extract_data_from_dlc_output(dlc_output, trajectory_metadata)
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


def extract_trajectory_metadata(dlc_metadata, name_metadata):

    view_list = dlc_metadata.keys()
    trajectory_metadata = {view: None for view in view_list}

    for view in view_list:
        if name_metadata[view] is None:
            continue
        trajectory_metadata[view] = {'bodyparts': dlc_metadata[view]['data']['DLC-model-config file']['all_joints_names'],
                                     'num_frames': dlc_metadata[view]['data']['nframes'],
                                     'crop_window': name_metadata[view]['crop_window']
                                     }
    # todo:check that number of frames and bodyparts are the same in each view

    return trajectory_metadata


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


def extract_data_from_dlc_output(dlc_output, trajectory_metadata):

    view_list = dlc_output.keys()

    dlc_data = {view: None for view in view_list}
    for view in view_list:
        # initialize dictionaries for each bodypart
        if trajectory_metadata[view] is None:
            continue
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

        if dlc_data[view] is None:
            continue

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


def overlay_pts_on_orig_ratframe(img, mtx, dist, dlc_data, dlc_metadata, markertype, jpg_name):

    dotsize = 6

    h, w, _ = np.shape(img)
    fig, ax = prepare_img_axes(w, h)

    img_ud = cv2.undistort(img, mtx, dist)

    ax[0][0].imshow(img_ud)

    # todo: loop through views to make sure everything is showing up properly

    for i_pt, pt in enumerate(pts):

        if len(pt) > 0:
            try:
                x, y = pt[0]
            except:
                x, y = pt
            # x = int(round(x))
            # y = int(round(y))

            pt_ud_norm = np.squeeze(cv2.undistortPoints(np.array([x, y]), mtx, dist))
            pt_ud = cvb.unnormalize_points(pt_ud_norm, mtx)
            bp_color = color_from_bodypart(bodyparts[i_pt])

            ax[0][0].plot(pt_ud[0], pt_ud[1], marker=markertype[0], ms=dotsize, color=bp_color)
            # ax.plot(x, y, marker=markertype[0], ms=dotsize, color=bp_color)

    # for i_rpt, rpt in enumerate(reprojected_pts):
    #     if len(rpt) > 0:
    #         try:
    #             x, y = rpt[0]
    #         except:
    #             x, y = rpt
    #         # x = int(round(x))
    #         # y = int(round(y))
    #         bp_color = color_from_bodypart(bodyparts[i_rpt])
    #
    #         ax[0][0].plot(x, y, marker=markertype[1], ms=dotsize, color=bp_color)

    # plt.show()
    fig.savefig(jpg_name)

    return fig, ax


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