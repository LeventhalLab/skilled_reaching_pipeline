import crop_videos
from datetime import datetime
import navigation_utilities
import reconstruct_3d
import skilled_reaching_calibration
import skilled_reaching_io
import glob
import os
import shutil
import pandas as pd
import toml
import cv2
import sys
import deeplabcut
from boards import CharucoBoard, Checkerboard
from cameras import Camera, CameraGroup
import train_autoencoder
import analyze_3d_recons
import sr_photometry_analysis as srphot_anal
import sr_visualization


def analyze_cropped_videos(folders_to_analyze, view_config_paths, expt_parent_dirs, rat_df, cropped_vid_type='.avi', gputouse=0, save_as_csv=True, create_marked_vids=True):
    '''
    :param folders_to_analyze:
    :param view_config_paths:
    :param expt_parent_dirs:
    :param cropped_vid_type:
    :param gputouse:
    :return: scorernames - dictionary with keys 'direct' and 'mirror' containing the scorername strings returned by
        deeplabcut.analyze_videos
    '''

    marked_vids_parent = expt_parent_dirs['marked_videos_parent']
    cropped_vids_parent = expt_parent_dirs['cropped_videos_parent']

    view_list = folders_to_analyze.keys()
    scorernames = dict.fromkeys(view_config_paths.keys())
    for view in view_list:
        # if 'direct' in view:
        #     dlc_network = 'direct'
        # elif 'mirror' in view:
        #     dlc_network = 'mirror'
        # else:
        #     print(view + ' does not contain the keyword "direct" or "mirror"')
        #     continue


        current_view_folders = folders_to_analyze[view]

        for current_folder in current_view_folders:
            ratID, session_name = navigation_utilities.parse_session_dir_name(current_folder)

            # find the row of the rat dataframe for this ratID, extract a paw preference series, take the first element as this rat's paw preference
            paw_pref = rat_df['pawpref'][rat_df['ratid'] == ratID].values[0]
            if 'direct' in view:
                dlc_network = 'direct'
            elif 'leftmirror' in view:
                if paw_pref.lower() in ('r', 'right'):
                    # left mirror view is the near paw view for a right-pawed rat
                    dlc_network = 'nearpaw'
                elif paw_pref.lower() in ('l', 'left'):
                    # left mirror view is the far paw view for a left-pawed rat
                    dlc_network = 'farpaw'
                else:
                    print('paw preference not correctly specified for {}'.format(ratID))
            elif 'rm' in view:
                if paw_pref.lower() in ('r', 'right'):
                    # right mirror view is the far paw view for a right-pawed rat
                    dlc_network = 'farpaw'
                elif paw_pref.lower() in ('l', 'left'):
                    # right mirror view is the near paw view for a left-pawed rat
                    dlc_network = 'nearpaw'
            else:
                print(view + ' does not contain the keyword "direct", "lm", or "rm"')
                continue
            config_path = view_config_paths[dlc_network]

            cropped_video_list = glob.glob(current_folder + '/{}_*'.format(ratID) + cropped_vid_type)
            vids_to_analyze = []
            for cropped_vid in cropped_video_list:
                cv_path, vid_name = os.path.split(cropped_vid)
                vid_name, vid_ext = os.path.splitext(vid_name)

                test_pickle_name = os.path.join(cv_path, vid_name + 'DLC*.pickle')

                # have pickle files already been created for this video?
                test_pickle_list = glob.glob(test_pickle_name)

                if len(test_pickle_list) == 0:
                    # if the pickle files for this video don't already exist, analyze this video
                    vids_to_analyze.append(cropped_vid)

            if len(vids_to_analyze) > 0:
                scorername = deeplabcut.analyze_videos(config_path,
                                          vids_to_analyze,
                                          videotype=cropped_vid_type,
                                          gputouse=gputouse,
                                          save_as_csv=save_as_csv)
                # might want to add additional options to these commands
                deeplabcut.convert_detections2tracklets(config_path, vids_to_analyze, videotype=cropped_vid_type)
                deeplabcut.filterpredictions(config_path, vids_to_analyze, videotype=cropped_vid_type)
            else:
                scorername = navigation_utilities.scorername_from_fname(test_pickle_list[0])

            scorernames[dlc_network] = scorername

            if create_marked_vids:
                marked_vids_dir = navigation_utilities.create_marked_vids_folder(current_folder, cropped_vids_parent,
                                                                         marked_vids_parent)

                # do the pickles need to be copied to the marked vids folder? I don't think so...
                # pickle_list = glob.glob(os.path.join(current_folder, '*.pickle'))
                # for pickle_file in pickle_list:
                #     # if the file already exists in the marked_vid directory, don't move it
                #     _, pickle_name = os.path.split(pickle_file)
                #     if not os.path.isfile(os.path.join(new_dir, pickle_name)):
                #         shutil.copy(pickle_file, new_dir)

                # cropped_video_list = glob.glob(current_folder + '/*' + cropped_vid_type)

                # figure out if there are already marked videos in the new folder
                vids_to_mark = []
                for cropped_vid in cropped_video_list:
                    cv_path, cropped_vid_name = os.path.split(cropped_vid)
                    cropped_vid_name, ext = os.path.splitext(cropped_vid_name)
                    test_marked_name = os.path.join(marked_vids_dir, '{}*_labeled.mp4'.format(cropped_vid_name))
                    marked_vids = glob.glob(test_marked_name)

                    if len(marked_vids) == 0:
                        # the marked video hasn't been made yet
                        vids_to_mark.append(cropped_vid)

                deeplabcut.create_labeled_video(config_path, vids_to_mark, color_by='bodypart', filtered=True, videotype=cropped_vid_type)
                # deeplabcut.create_video_with_all_detections(config_path, vids_to_mark, videotype=cropped_vid_type)

                test_marked_name = os.path.join(current_folder, '{}_*{}_labeled.mp4'.format(ratID, scorername))
                marked_vid_names = glob.glob(test_marked_name)

                for marked_vid in marked_vid_names:
                    mv_path, mv_name = os.path.split(marked_vid)
                    dest_name = os.path.join(marked_vids_dir, mv_name)
                    shutil.move(marked_vid, dest_name)

    return scorernames


def create_labeled_videos(folders_to_analyze, marked_vids_parent, view_config_paths, scorernames,
                          cropped_vid_type='.avi',
                          skipdirect=False,
                          skipmirror=False,
                          view_list=('dir', 'lm', 'rm')
):
    '''
    
    :param folders_to_analyze: 
    :param view_config_paths: 
    :param scorernames: dictionary with keys 'direct' and 'mirror'
    :param cropped_vid_type:
    :param move_to_new_folder: if True, create a new folder in which the marked videos and analysis files are stored
        to make it easier to move them to another computer without taking the original videos with them
    :return: 
    '''
    # in case there are some previously cropped videos that need to be analyzed
    folders_to_analyze = navigation_utilities.find_folders_to_analyze(cropped_videos_parent, view_list=view_list)
    # view_list = folders_to_analyze.keys()

    for view in view_list:
        if 'dir' in view:
            if skipdirect:
                continue
            dlc_network = 'dir'
        elif 'mirr' in view:
            if skipmirror:
                continue
            dlc_network = 'mirr'
        else:
            print(view + ' does not contain the keyword "direct" or "mirror"')
            continue
        config_path = view_config_paths[dlc_network]
        scorername = scorernames[dlc_network]
        current_view_folders = folders_to_analyze[view]

        for current_folder in current_view_folders:
            cropped_video_list = glob.glob(current_folder + '/*' + cropped_vid_type)
            deeplabcut.create_video_with_all_detections(config_path, cropped_video_list, scorername)

            # current_basename = os.path.basename(current_folder)
            new_dir =  navigation_utilities.create_marked_vids_folder(current_folder, cropped_videos_parent, marked_vids_parent)
            #    os.path.join(marked_vids_parent, current_basename + '_marked')

            test_name = os.path.join(current_folder, '*' + scorername + '*.mp4')
            marked_vid_list = glob.glob(test_name)
            pickle_list = glob.glob(os.path.join(current_folder, '*.pickle'))

            for marked_vid in marked_vid_list:
                 # if the file already exists in the marked_vid directory, don't move it
                 _, marked_vid_name = os.path.split(marked_vid)
                 if not os.path.isfile(os.path.join(new_dir, marked_vid_name)):
                      shutil.move(marked_vid, new_dir)
                 for pickle_file in pickle_list:
                      # if the file already exists in the marked_vid directory, don't move it
                      _, pickle_name = os.path.split(pickle_file)
                      if not os.path.isfile(os.path.join(new_dir, pickle_name)):
                           shutil.move(pickle_file, new_dir)


def calibrate_all_sessions(parent_directories,
                           calibration_metadata_df,
                           cam_names,
                           vidtype='.avi',
                           filtertype='h264',
                           rat_nums='all'):
    '''
    loop through all folders containing calibration videos and store calibration parameters
    :param parent_directories:
    :param crop_params_df: dataframe containing cropping parameters for each box-date
    :param cb_size:
    :return:
    '''

    calibration_vids_parent = parent_directories['calibration_vids_parent']
    calibration_files_parent = parent_directories['calibration_files_parent']

    if vidtype[0] != '.':
        vidtype = '.' + vidtype

    calib_vid_folders = navigation_utilities.find_calibration_vid_folders(calibration_vids_parent)

    expt_ratIDs = list(calibration_metadata_df.keys())
    if rat_nums == 'all':
        ratIDs = expt_ratIDs
    else:
        ratIDs = ['R{:04d}'.format(rn) for rn in rat_nums]

    # to skip to where I'm working...
    # ratIDs = ['R0484', 'R0485', 'R0486', 'R0487']
    # make sure all cameras have been calibrated
    for ratID in ratIDs:
        if ratID not in expt_ratIDs:
            continue
        rat_md_df = calibration_metadata_df[ratID]

        num_sessions = len(rat_md_df)

        for i_session in range(num_sessions):
            session_row = rat_md_df.iloc[[i_session]]

            # calibrate the camera for this session
            cam_cal_vid_name = session_row['cal_vid_name_camera'].values[0]

            cam_cal_pickle = navigation_utilities.create_cam_cal_pickle_name(cam_cal_vid_name, parent_directories)
            if os.path.exists(cam_cal_pickle):
                cam_intrinsics = skilled_reaching_io.read_pickle(cam_cal_pickle)
            else:
                cam_cal_pickle_folder, _ = os.path.split(cam_cal_pickle)
                if not os.path.exists(cam_cal_pickle_folder):
                    os.makedirs(cam_cal_pickle_folder)
                full_cam_cal_vid_path = navigation_utilities.find_camera_calibration_video(cam_cal_vid_name,
                                                                                           parent_directories)
                cam_board = skilled_reaching_calibration.camera_board_from_df(session_row)

                cam_intrinsics = skilled_reaching_calibration.calibrate_single_camera(full_cam_cal_vid_path, cam_board)

                # alternatively, undistort after identifying points, which may be faster
                skilled_reaching_io.write_pickle(cam_cal_pickle, cam_intrinsics)

            # now have the camera intrinsics; use these to undistort points to calibrate the mirror views
            # first, crop the calibration video
            mirror_calib_vid_name = session_row['cal_vid_name_mirrors'].values[0]
            if mirror_calib_vid_name.lower() == 'none':
                continue
            full_calib_vid_name = navigation_utilities.find_mirror_calibration_video(mirror_calib_vid_name,
                                                                                     parent_directories)

            # now identify the points, undistort them
            mirror_board = skilled_reaching_calibration.mirror_board_from_df(session_row)
            # skilled_reaching_calibration.write_board_image(mirror_board, 600, parent_directories['calibration_vids_parent'])

            calibration_pickle_name = navigation_utilities.create_calibration_summary_name(full_calib_vid_name, calibration_files_parent)

            # need this here so we have the video names, not because we need videos cropped
            current_cropped_calibration_vids = skilled_reaching_calibration.crop_calibration_video(
                full_calib_vid_name,
                session_row,
                filtertype=filtertype)
            print('calibrating {}'.format(mirror_calib_vid_name))
            cgroup, error = skilled_reaching_calibration.calibrate_mirror_views(current_cropped_calibration_vids, cam_intrinsics, mirror_board, cam_names, parent_directories, session_row, calibration_pickle_name)
            # note that calibrate_mirror_views writes a pickle file with updated calibration parameters including cgroup


            # cgroup.dump(calibration_toml_name)
            # skilled_reaching_calibration.test_anipose_calibration(session_row, parent_directories)



    # for cf in calib_vid_folders:
    #     # crop the calibration videos
    #     calib_vids = glob.glob(os.path.join(cf, 'GridCalibration_*' + vidtype))
    #
    #     orig_im_size = []
    #     cropped_vid_names = []
    #     for calib_vid in calib_vids:
    #         current_cropped_calibration_vids = skilled_reaching_calibration.crop_calibration_video(calib_vid, calibration_metadata_df, filtertype=filtertype)
    #         if current_cropped_calibration_vids is None:
    #             # crop_calibration_video returns None if there isn't an associated rat session for the calibration video
    #             continue
    #
    #         cropped_vid_names.append(current_cropped_calibration_vids)
    #         vid_obj = cv2.VideoCapture(calib_vid)
    #         orig_im_size.append((int(vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH))))
    #         vid_obj.release()
    #     # cropped_vid_names contains a list of lists; each individual list contains the names of 3 files, one each for
    #     # the direct, leftmirror, and rightmirror views
    #
    #     # FOR NOW, JUST CROP THE VIDEOS. COMMENT THE CALIBRATION PART BACK IN LATER (HOPEFULLY CAN JUST RUN ANIPOSE HERE...)
    #     for i_calib_vid, cropped_vids_set in enumerate(cropped_vid_names):
    #     #     # each cropped_vids_set should contain 3 video names for direct, left, right views
    #     #
    #         session_metadata = navigation_utilities.parse_camera_calibration_video_name(cropped_vids_set[0])
    #         calibration_summary_name = navigation_utilities.create_calibration_summary_name(calib_vids[i_calib_vid], calibration_files_parent)
    #     #
    #         if os.path.isfile(calibration_summary_name):
    #             calibration_data = skilled_reaching_io.read_pickle(calibration_summary_name)
    #         else:
    #             calibration_metadata = skilled_reaching_calibration.calibration_metadata_from_df(session_metadata, calibration_metadata_df)
    #             crop_params_dict = crop_videos.crop_params_dict_from_df(calibration_metadata_df, session_metadata['time'].date(),
    #                                                                     session_metadata['boxnum'])
    #             cam_names = list(crop_params_dict.keys())
    #             skilled_reaching_calibration.anipose_calibrate(cropped_vids_set, cam_names, calibration_metadata)
    #             # calibration_data = skilled_reaching_calibration.collect_cb_corners(cropped_vids_set, cb_size)
    #             calibration_data['orig_im_size'] = orig_im_size[i_calib_vid]
    #             skilled_reaching_io.write_pickle(calibration_summary_name, calibration_data)
        #
        #     # now perform the actual calibration
        #     skilled_reaching_calibration.multi_mirror_calibration(calibration_data, calibration_summary_name)

def perform_calibrations(parent_directories, cam_names=('dir', 'lm', 'rm'),
                         vidtype='.avi', filtertype='h264', rat_nums='all'):

    experiment_list = list(parent_directories.keys())
    for expt in experiment_list:
        videos_root_folder = parent_directories[expt]['videos_root_folder']
        session_metadata_xlsx_path = os.path.join(videos_root_folder, 'SR_{}_video_session_metadata.xlsx'.format(expt))

        # load the .xlsx file containing all the info about which calibration files to use for each session
        calibration_metadata_df = skilled_reaching_io.read_session_metadata_xlsx(session_metadata_xlsx_path)

        crop_videos.crop_all_calibration_videos(parent_directories[expt],
                                    calibration_metadata_df,
                                    vidtype=vidtype,
                                    view_list=cam_names,
                                    filtertype=filtertype,
                                    rat_nums=rat_nums)


        calibrate_all_sessions(parent_directories[expt],
                               calibration_metadata_df,
                               cam_names,
                               filtertype=filtertype,
                               rat_nums=rat_nums)

    return calibration_metadata_df


if __name__ == '__main__':


    # experiment_list = ['GRABAch-rDA', 'sr6OHDA', 'dLight']
    # experiment_list = ['dLight', 'sr6OHDA', 'GRABAch-rDA']
    experiment_list = ['sr6OHDA', 'dLight']
    # experiment_list = ['dLight', 'sr6OHDA']
    rat_db_fnames = {expt: 'rat_{}_SRdb.xlsx'.format(expt) for expt in experiment_list}
    session_scores_fnames = {expt: 'rat_{}_SRsessions.xlsx'.format(expt) for expt in experiment_list}
    create_marked_vids = True

    label_videos = True

    rats_to_analyze = [468, 469, 470, 471, 472, 473, 474, 482, 484, 485, 486, 487, 497, 498, 499, 500, 501, 502, 514, 519, 520, 521, 522, 526, 528, 529, 532, 533, 534, 535, 536, 537]

    # if you only want to label the direct or mirror views, set the skip flag for the other view to True
    skipdirectlabel = False
    skipmirrorlabel = False

    gputouse = 3

    cam_names = ('dir', 'lm', 'rm')
    n_cams = len(cam_names)
    # cgroup = CameraGroup.from_names(cam_names, fisheye=False)

    filtertype = 'h264'

    # # for lambda computer
    # view_config_paths = {
    #     'direct': '/home/levlab/deeplabcut_projects/ratdirsr-DL-2023-06-07/config.yaml',
    #     'nearpaw': '/home/levlab/deeplabcut_projects/ratnearpawmirrsr-DL-2023-06-19/config.yaml',
    #     'farpaw': '/home/levlab/deeplabcut_projects/ratfarpawmirrsr-DL-2023-07-03/config.yaml'
    # }

    if sys.platform in ['win32']:
        # assume DKL computer
        DLC_top_folder = r'C:\Users\dleventh\Documents\deeplabcut_projects'
        # data_root_folder = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\sr'
        data_root_folder = r'X:\data\sr'
    elif sys.platform in ['linux']:
        # lambda computer
        DLC_top_folder = '/home/dleventh/Documents/DLC_projects'
        data_root_folder = '/home/dleventh/SharedX/Neuro-Leventhal/data/sr'

    # to find the config files for each DLC network for each view
    view_keys = ('direct', 'nearpaw', 'farpaw')   #list(DLC_folder_names.keys())
    anipose_config_path = os.path.join(DLC_top_folder, 'sr_anipose', 'config.toml')
    anipose_config = toml.load(anipose_config_path)
    DLC_folder_names = {view_key: anipose_config['DLC_folders'][i_view] for i_view, view_key in enumerate(view_keys)}

        # 'direct': DLC_proj_names['direct'] + '-DL-2023-06-07',
        #                 'nearpaw': DLC_proj_names['nearpaw'] + '-DL-2023-06-19',
        #                 'farpaw': DLC_proj_names['farpaw'] + '-DL-2023-07-03'}
    view_config_paths = {view_key: os.path.join(DLC_top_folder, anipose_config['DLC_folders'][i_view], 'config.yaml') for i_view, view_key in enumerate(view_keys)}

    # store directory tree for each experiment
    videos_parents = {expt: os.path.join(data_root_folder, expt) for expt in experiment_list}
    video_root_folders = {expt: os.path.join(videos_parents[expt], 'data') for expt in experiment_list}
    cropped_videos_parents = {expt: os.path.join(videos_parents[expt], 'cropped') for expt in experiment_list}
    marked_videos_parents = {expt: os.path.join(videos_parents[expt], 'marked') for expt in experiment_list}
    calibration_vids_parents = {expt: os.path.join(videos_parents[expt], 'calibration_videos') for expt in experiment_list}
    calibration_files_parents = {expt: os.path.join(videos_parents[expt], 'calibration_files') for expt in experiment_list}
    dlc_mat_output_parents = {expt: os.path.join(videos_parents[expt], 'matlab_readable_dlc') for expt in experiment_list}
    trajectories_parents = {expt: os.path.join(videos_parents[expt], 'traj_files') for expt in experiment_list}
    trajectory_summaries = {expt: os.path.join(videos_parents[expt], 'traj_summaries') for expt in experiment_list}
    analysis_summaries = {expt: os.path.join(videos_parents[expt], 'analysis') for expt in experiment_list}

    parent_directories = {expt: {
                                'videos_parent': videos_parents[expt],
                                'videos_root_folder': video_root_folders[expt],
                                'cropped_videos_parent': cropped_videos_parents[expt],
                                'marked_videos_parent': marked_videos_parents[expt],
                                'calibration_vids_parent': calibration_vids_parents[expt],
                                'calibration_files_parent': calibration_files_parents[expt],
                                'dlc_mat_output_parent': dlc_mat_output_parents[expt],
                                'trajectories_parent': trajectories_parents[expt],
                                'trajectory_summaries': trajectory_summaries[expt],
                                'analysis': analysis_summaries[expt]
                            }
                            for expt in experiment_list}

    # use the code below to write a charuco board to a file
    # ncols = 10
    # nrows = 7
    # square_length = 16
    # marker_length = 12
    # board = skilled_reaching_calibration.create_charuco(nrows, ncols, square_length, marker_length)
    # skilled_reaching_calibration.write_charuco_image(board, 600, calibration_vids_parents['dLight'])

    # excel file containing the cropping parameters, which calibration file to use for each session, etc.
    # session_metadata_xlsx_path = os.path.join(video_root_folders[expt],
    #                                           'SR_{}_video_session_metadata.xlsx'.format(expt))
    # calibration_metadata_df = skilled_reaching_io.read_session_metadata_xlsx(session_metadata_xlsx_path)

    # crop calibration videos and perform the calibrations
    # perform_calibrations(parent_directories, vidtype='.avi', cam_names=cam_names, filtertype=filtertype, rat_nums=rats_to_analyze)

    test_folder = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\sr\dLight\traj_files\R0452\R0452_20230329_sr_ses01'
    # analyze_3d_recons.analyze_trajectories(test_folder, anipose_config)

    traj_path = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\sr\dLight\traj_files'
    analysis_path = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\sr\dLight\analysis\sr'
    traj3d_fname = 'R0487_b01_20230918_12-36-33_011_r3d.pickle'
    session_metadata = navigation_utilities.metadata_from_traj_name(traj3d_fname)
    datestring = navigation_utilities.datetime_to_string_for_fname(session_metadata['date'])[:8]
    ses_name = '_'.join((session_metadata['ratID'], datestring, 'sr', 'ses01'))
    full_trajpath = os.path.join(traj_path, session_metadata['ratID'], ses_name)
    traj3d_fname = os.path.join(full_trajpath, traj3d_fname)

    # agg_pickle = os.path.join(analysis_path, session_metadata['ratID'] + '_sr_aggregated.pickle')
    trials_df_name = os.path.join(analysis_path, session_metadata['ratID'] + '_sr_trialsdb.pickle')
    trials_df = skilled_reaching_io.read_pickle(trials_df_name)
    session_metadata['task'] = 'sr'
    session_metadata['session_num'] = 1
    processed_phot_name = navigation_utilities.processed_data_pickle_name(session_metadata, parent_directories['dLight'])
    if os.path.exists(processed_phot_name):
        # if no processed photometry file, just reconstruct the 3d points
        processed_phot_data = skilled_reaching_io.read_pickle(processed_phot_name)
        # session_summary, trials_df = srphot_anal.aggregate_data_pre_20230904(processed_phot_data, session_metadata,
        #                                                                      trials_df,
        #                                                                      smooth_window=101,
        #                                                                      f0_pctile=10,
        #                                                                      expected_baseline=0.2)
    else:
        analog_bin_file = navigation_utilities.find_analog_bin_file(parent_directories['dLight'], session_metadata)
        digital_bin_file = navigation_utilities.find_digital_bin_file(parent_directories['dLight'], session_metadata)
        metadata_file = navigation_utilities.find_metadata_file(parent_directories['dLight'], session_metadata)

        data_files = {'analog_bin': analog_bin_file,
                      'digital_bin': digital_bin_file,
                      'metadata': metadata_file
                      }
        # session_summary, trials_df = srphot_anal.aggregate_data_post_20230904(data_files, parent_directories['dLight'],
        #                                                                       session_metadata,
        #                                                                       trials_df,
        #                                                                       smooth_window=101,
        #                                                                       f0_pctile=10,
        #                                                                       expected_baseline=0.2)


    rat_df = skilled_reaching_io.read_rat_db(parent_directories['dLight'], rat_db_fnames['dLight'])

    df_row = rat_df[rat_df['ratid'] == session_metadata['ratID']]
    paw_pref = df_row['pawpref'].values[0]
    cw = [[800, 1200, 475, 900], [200, 600, 425, 850], [1425, 1825, 425, 850]]
    lim_3d = [[-20, 30], [0, 70], [280, 340]]
    # sr_visualization.create_presentation_vid(traj3d_fname, session_metadata, parent_directories['dLight'], session_summary, trials_df,
    #                             paw_pref,
    #                             bpts2plot='reachingpaw', phot_ylim=[-2.5, 5],
    #                             cw=cw,
    #                             lim_3d=lim_3d)
    # sr_visualization.create_presentation_vid_1view(traj3d_fname, session_metadata, parent_directories['dLight'], session_summary, trials_df,
    #                             paw_pref,
    #                             bpts2plot='reachingpaw', phot_ylim=[-2.5, 5],
    #                             cw=cw,
    #                             lim_3d=lim_3d,
    #                             frames2mark={'reach_on': 240, 'contact': 306, 'drop': 308})

    traj3d_fname2 = 'R0487_b01_20230918_12-33-32_005_r3d.pickle'
    traj3d_fname2 = os.path.join(full_trajpath, traj3d_fname2)

    # sr_visualization.create_presentation_vid_1view(traj3d_fname2, session_metadata, parent_directories['dLight'], session_summary, trials_df,
    #                             paw_pref,
    #                             bpts2plot='reachingpaw', phot_ylim=[-2.5, 5],
    #                             cw=cw,
    #                             lim_3d=lim_3d,
    #                             frames2mark={'reach_on': 233, 'contact': 306, 'retract': 330})
    # sr_visualization.create_presentation_vid(traj3d_fname, session_metadata, parent_directories['dLight'], session_summary, trials_df,
    #                             paw_pref,
    #                             bpts2plot='reachingpaw', phot_ylim=[-2.5, 5],
    #                             cw=cw,
    #                             lim_3d=lim_3d)

    for expt in experiment_list:

        # calibration_metadata_csv_path = os.path.join(calibration_vids_parents[expt], 'SR_calibration_vid_metadata.csv')
        session_metadata_xlsx_path = os.path.join(video_root_folders[expt], 'SR_{}_video_session_metadata.xlsx'.format(expt))
        # calibration_metadata_df = skilled_reaching_io.read_calibration_metadata_csv(calibration_metadata_csv_path)
        calibration_metadata_df = skilled_reaching_io.read_session_metadata_xlsx(session_metadata_xlsx_path)
    #
        # crop_videos.crop_all_calibration_videos(parent_directories[expt],
        #                             calibration_metadata_df,
        #                             vidtype='.avi',
        #                             view_list=cam_names,
        #                             filtertype=filtertype,
        #                             rat_nums=rats_to_analyze)
    # #
    # #
    #     calibrate_all_sessions(parent_directories[expt],
    #                            calibration_metadata_df,
    #                            cam_names,
    #                            filtertype=filtertype,
    #                            rat_nums=rats_to_analyze)

    for expt in experiment_list:
        # crop_params_csv_path = os.path.join(video_root_folders[expt], 'SR_video_crop_regions.csv')
        # crop_params_df = skilled_reaching_io.read_crop_params_csv(crop_params_csv_path)
        crop_filtertype = 'h264'  # currently choices are 'h264' or 'mjpeg2jpeg'. Python based vid conversion (vs labview) should use h264

        session_metadata_xlsx_path = os.path.join(video_root_folders[expt],
                                                  'SR_{}_video_session_metadata.xlsx'.format(expt))
        calibration_metadata_df = skilled_reaching_io.read_session_metadata_xlsx(session_metadata_xlsx_path)
        video_folder_list = navigation_utilities.get_video_folders_to_crop(video_root_folders[expt], rats_to_analyze=rats_to_analyze)
        cropped_video_directories = crop_videos.preprocess_videos(video_folder_list, cropped_videos_parents[expt], calibration_metadata_df, cam_names, vidtype='avi', filtertype=crop_filtertype)


    # step 2: run the vids through DLC
    # parameters for running DLC
    # need to update these paths when moved to the lambda machine


    # for expt in experiment_list:
    #     rat_db = skilled_reaching_io.read_rat_db(parent_directories[expt], rat_db_fnames[expt])
    #     folders_to_analyze = navigation_utilities.find_folders_to_analyze(cropped_videos_parents[expt], view_list=cam_names)
    #
    #     scorernames = analyze_cropped_videos(folders_to_analyze, view_config_paths, marked_videos_parents[expt], rat_db, cropped_vid_type=cropped_vid_type, gputouse=gputouse, save_as_csv=True)
    #
    #     if label_videos:
    #         create_labeled_videos(cropped_videos_parent,
    #                               marked_videos_parent,
    #                               view_config_paths,
    #                               scorernames,
    #                               cropped_vid_type=cropped_vid_type,
    #                               skipdirect=skipdirectlabel,
    #                               skipmirror=skipmirrorlabel,
    #                               view_list=cam_names)

    # step 3: make sure calibration has been run for these sessions
    # find list of all analyzed videos; extract dates and boxes for each session

    # step 4: reconstruct the 3d trajectories

    for expt in experiment_list:
        rat_df = skilled_reaching_io.read_rat_db(parent_directories[expt], rat_db_fnames[expt])
        # folders_to_reconstruct = navigation_utilities.find_folders_to_reconstruct(parent_directories[expt]['cropped_videos_parent'], cam_names)

        DLC_folder_keys = DLC_folder_names.keys()
        # for DLC_key in DLC_folder_keys:
        #     train_autoencoder.train_autoencoder(anipose_config, DLC_folder_names[DLC_key])
        # ftr = [folder for folder in folders_to_reconstruct if ((folder['ratID'] == 'R0486') and (folder['date'] == datetime(2023, 9, 8)))]
        # ftr = [folder for folder in folders_to_reconstruct if not folder['ratID'] in ['R0452', 'R0453', 'R0468', 'R0469', 'R0472', 'R0473']]
        # ftr = [folder for folder in folders_to_reconstruct if folder['ratID'] in ['R0472']]
        # ftr = folders_to_reconstruct
        for ratID in ['R0487']:
            reconstruct_3d.reconstruct_folders_anipose(ratID, parent_directories[expt], expt, rat_df, anipose_config, cam_names=cam_names, filtered=False)

    # step 5: post-processing including smoothing (should there be smoothing on the 2-D images first?)


