import numpy as np
import cv2
import os
import navigation_utilities
import glob
import skilled_reaching_io
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import computer_vision_basics as cvb


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

        # test_undistortion(dlc_data, dlc_metadata, cal_data, direct_pickle_params)

        # todo: preprocessing to get rid of "invalid" points

        invalid_points, diff_per_frame = find_invalid_DLC_points(dlc_data, paw_pref)

        calc_3d_dlc_trajectory(dlc_data, invalid_points, cal_data, paw_pref, im_size=(2040, 1024), max_dist_from_neighbor=60)






        mat_data = package_data_into_mat(dlc_data, video_metadata, trajectory_metadata)
        mat_name = navigation_utilities.create_mat_fname_dlc_output(video_metadata, dlc_mat_output_parent)

        video_name = navigation_utilities.build_video_name(video_metadata, videos_parent)
        # test_pt_alignment(video_name, dlc_data)

        sio.savemat(mat_name, mat_data)

        pass
    pass


def calc_3d_dlc_trajectory(dlc_data, invalid_points, cal_data, paw_pref, im_size=(2040, 1024), max_dist_from_neighbor=60):

    # F[:,:,0] - fundamental matrix between direct and top mirror views
    # F[:,:,1] - fundamental matrix between direct and left mirror views
    # F[:,:,2] - fundamental matrix between direct and right mirror views

    K = cal_data['mtx']
    view_list = dlc_data.keys()
    bodyparts = []
    for view in view_list:
        bodyparts.append(dlc_data[view].keys())

    num_frames = np.shape(dlc_data[view_list[0]][bodyparts[0][0]]['coordinates_ud'])[0]
    frames_to_check = range(num_frames)

    estimate_hidden_points(dlc_data, invalid_points, cal_data, im_size, paw_pref, frames_to_check, max_dist_from_neighbor=max_dist_from_neighbor)



    num_frames = dlc_data[view_list[0]]
    pass


def estimate_hidden_points(dlc_data, invalid_points, cal_data, im_size, paw_pref, frames_to_check, max_dist_from_neighbor=60):

    # F[:,:,0] - fundamental matrix between direct and top mirror views
    # F[:,:,1] - fundamental matrix between direct and left mirror views
    # F[:,:,2] - fundamental matrix between direct and right mirror views

    if paw_pref.lower() == 'left':
        # use F for right mirror
        F = cal_data['F'][:, :, 2]
    elif paw_pref.lower() == 'right':
        # use F for left mirror
        F = cal_data['F'][:, :, 1]




    pass
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
                            invalid_points[paw_part_idx[i_paw], i_frame] = True

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
    markertypes = ['o', '+']
    overlay_pts_on_orig_ratframe(cur_img, cal_data, dlc_data, dlc_metadata, markertypes, jpg_name, test_frame)

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


def overlay_pts_on_orig_ratframe(img, cal_data, dlc_data, dlc_metadata, markertypes, jpg_name, test_frame):

    dotsize = 6
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    bp_c = bp_colors()

    im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = np.shape(im_bgr)
    fig_ud_direct_epi, ax_ud_direct_epi = prepare_img_axes(w, h)
    fig_ud_mirror_epi, ax_ud_mirror_epi = prepare_img_axes(w, h)

    # fig, ax = prepare_img_axes(w, h)

    img_ud = cv2.undistort(im_bgr, mtx, dist)

    ax_ud_direct_epi[0][0].imshow(img_ud)
    ax_ud_mirror_epi[0][0].imshow(img_ud)

    # ax[0][0].imshow(im_bgr)

    for view in dlc_data.keys():
        view_dlcdata = dlc_data[view]

        bodyparts = view_dlcdata.keys()

        for bp in bodyparts:
            cur_pt = view_dlcdata[bp]['coordinates'][test_frame]

            if all(cur_pt == 0):
                continue

            cur_pt_ud = view_dlcdata[bp]['coordinates_ud'][test_frame]

            # ax[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
            # ax[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

            ax_ud_direct_epi[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
            ax_ud_direct_epi[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

            ax_ud_mirror_epi[0][0].scatter(cur_pt[0], cur_pt[1], edgecolors=bp_c[bp], facecolors='none', marker=markertypes[0])
            ax_ud_mirror_epi[0][0].scatter(cur_pt_ud[0], cur_pt_ud[1], c=bp_c[bp], marker=markertypes[1])

    # overlay epipolar lines
    draw_epipolar_lines(dlc_data, cal_data, test_frame, [ax_ud_direct_epi[0][0], ax_ud_mirror_epi[0][0]], (w, h))

    plt.show()

    pass


def draw_epipolar_lines(dlc_data, cal_data, test_frame, ax, im_size):

    # F[:,:,0] - fundamental matrix between direct and top mirror views
    # F[:,:,1] - fundamental matrix between direct and left mirror views
    # F[:,:,2] - fundamental matrix between direct and right mirror views
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    bp_c = bp_colors()

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

        bodyparts = view_dlcdata.keys()

        for bp in bodyparts:
            cur_pt = view_dlcdata[bp]['coordinates'][test_frame]

            if all(cur_pt == 0):
                continue

            cur_pt_ud = view_dlcdata[bp]['coordinates_ud'][test_frame]

            cur_epiline = cv2.computeCorrespondEpilines(cur_pt_ud.reshape(-1, 1, 2), 1, F)

            epiline = np.squeeze(cur_epiline)
            edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

            if not np.all(edge_pts == 0):
                ax[i_view].plot(edge_pts[:, 0], edge_pts[:, 1], color=bp_c[bp], ls='-', marker='.')

    plt.show()

    pass



def bp_colors():

    bp_c = {'leftear':(0, 1, 1)}
    bp_c['rightear'] = tuple(np.array(bp_c['leftear']) * 0.5)

    bp_c['lefteye'] = (1, 0, 1)
    bp_c['righteye'] = tuple(np.array(bp_c['lefteye']) * 0.5)

    bp_c['nose'] = (1, 1, 1)

    bp_c['leftelbow'] = (1, 1, 0)
    bp_c['rightelbow'] = tuple(np.array(bp_c['leftelbow']) * 0.5)

    bp_c['rightpawdorsum'] = (0, 0, 1)
    bp_c['rightpalm'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.5)
    bp_c['rightmcp1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightmcp2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightmcp3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightmcp4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['rightpip1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightpip2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightpip3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightpip4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['rightdig1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightdig2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightdig3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightdig4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['leftpawdorsum'] = (1, 0, 0)
    bp_c['leftpalm'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.5)
    bp_c['leftmcp1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftmcp2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftmcp3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftmcp4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['leftpip1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftpip2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftpip3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftpip4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['leftdig1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftdig2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftdig3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftdig4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['pellet1'] = (0, 0, 0)
    bp_c['pellet2'] = (0.1, 0.1, 0.1)
    bp_c['pellet3'] = (0.2, 0.2, 0.2)

    return bp_c


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