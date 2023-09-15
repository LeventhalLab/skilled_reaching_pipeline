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
import cv2
import sys
import deeplabcut
from boards import CharucoBoard, Checkerboard
from cameras import Camera, CameraGroup


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
            elif 'rightmirror' in view:
                if paw_pref.lower() in ('r', 'right'):
                    # right mirror view is the far paw view for a right-pawed rat
                    dlc_network = 'farpaw'
                elif paw_pref.lower() in ('l', 'left'):
                    # right mirror view is the near paw view for a left-pawed rat
                    dlc_network = 'nearpaw'
            else:
                print(view + ' does not contain the keyword "direct", "leftmirror", or "rightmirror"')
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
                # todo: might want to add additional options to these commands
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
                          view_list=('direct', 'leftmirror', 'rightmirror')
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
        if 'direct' in view:
            if skipdirect:
                continue
            dlc_network = 'direct'
        elif 'mirror' in view:
            if skipmirror:
                continue
            dlc_network = 'mirror'
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
                           cgroup,
                           vidtype='.avi',
                           view_list=['direct', 'leftmirror', 'rightmirror'],
                           filtertype='h264'):
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

    ratIDs = list(calibration_metadata_df.keys())

    # make sure all cameras have been calibrated
    for ratID in ratIDs:
        rat_metadata_df = calibration_metadata_df[ratID]

        num_sessions = len(rat_metadata_df)

        for i_session in range(num_sessions):
            session_row = rat_metadata_df.iloc[[i_session]]

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
            full_calib_vid_name = navigation_utilities.find_mirror_calibration_video(mirror_calib_vid_name,
                                                                                     parent_directories)

            # now identify the points, undistort them
            mirror_board = skilled_reaching_calibration.mirror_board_from_df(session_row)
            # skilled_reaching_calibration.write_board_image(mirror_board, 600, parent_directories['calibration_vids_parent'])

            # todo: test if chessboard detection is sufficient for the old boards
            calibration_toml_name = navigation_utilities.create_calibration_toml_name(full_calib_vid_name, calibration_files_parent)
            if os.path.exists(calibration_toml_name):
                cgroup = CameraGroup.load(calibration_toml_name)
            else:
                # need this here so we have the video names, not because we need videos cropped
                current_cropped_calibration_vids = skilled_reaching_calibration.crop_calibration_video(
                    full_calib_vid_name,
                    session_row,
                    filtertype=filtertype)
                cgroup, error = skilled_reaching_calibration.calibrate_mirror_views(current_cropped_calibration_vids, cam_intrinsics, mirror_board, cgroup, parent_directories)
                cgroup.dump(calibration_toml_name)
                pass
            pass

    #TODO: load the calibration .toml file and verify points


    for cf in calib_vid_folders:
        # crop the calibration videos
        calib_vids = glob.glob(os.path.join(cf, 'GridCalibration_*' + vidtype))

        orig_im_size = []
        cropped_vid_names = []
        for calib_vid in calib_vids:
            current_cropped_calibration_vids = skilled_reaching_calibration.crop_calibration_video(calib_vid, calibration_metadata_df, filtertype=filtertype)
            if current_cropped_calibration_vids is None:
                # crop_calibration_video returns None if there isn't an associated rat session for the calibration video
                continue

            cropped_vid_names.append(current_cropped_calibration_vids)
            vid_obj = cv2.VideoCapture(calib_vid)
            orig_im_size.append((int(vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH))))
            vid_obj.release()
        # cropped_vid_names contains a list of lists; each individual list contains the names of 3 files, one each for
        # the direct, leftmirror, and rightmirror views

        # FOR NOW, JUST CROP THE VIDEOS. COMMENT THE CALIBRATION PART BACK IN LATER (HOPEFULLY CAN JUST RUN ANIPOSE HERE...)
        for i_calib_vid, cropped_vids_set in enumerate(cropped_vid_names):
        #     # each cropped_vids_set should contain 3 video names for direct, left, right views
        #
            session_metadata = navigation_utilities.parse_camera_calibration_video_name(cropped_vids_set[0])
            calibration_summary_name = navigation_utilities.create_calibration_summary_name(calib_vids[i_calib_vid], calibration_files_parent)
        #
            if os.path.isfile(calibration_summary_name):
                calibration_data = skilled_reaching_io.read_pickle(calibration_summary_name)
            else:
                calibration_metadata = skilled_reaching_calibration.calibration_metadata_from_df(session_metadata, calibration_metadata_df)
                crop_params_dict = crop_videos.crop_params_dict_from_df(calibration_metadata_df, session_metadata['time'].date(),
                                                                        session_metadata['boxnum'])
                cam_names = list(crop_params_dict.keys())
                skilled_reaching_calibration.anipose_calibrate(cropped_vids_set, cam_names, calibration_metadata)
                # calibration_data = skilled_reaching_calibration.collect_cb_corners(cropped_vids_set, cb_size)
                calibration_data['orig_im_size'] = orig_im_size[i_calib_vid]
                skilled_reaching_io.write_pickle(calibration_summary_name, calibration_data)
        #
        #     # now perform the actual calibration
        #     skilled_reaching_calibration.multi_mirror_calibration(calibration_data, calibration_summary_name)


if __name__ == '__main__':


    experiment_list = ['dLightPhotometry', 'sr6OHDA']
    rat_db_fnames = {expt: 'rat_{}_SRdb.xlsx'.format(expt) for expt in experiment_list}
    session_scores_fnames = {expt: 'rat_{}_SRsessions.xlsx'.format(expt) for expt in experiment_list}
    create_marked_vids = True

    # videos_parents = [r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\skilled_reaching\dLight_Photometry',
    #                   r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\skilled_reaching\SR_6OHDA']
    # video_root_folder = os.path.join(videos_parent, 'videos_to_crop')
    # video_root_folder = os.path.join(videos_parent, 'data')
    # cropped_videos_parents = os.path.join(videos_parent, 'cropped')
    # marked_videos_parent = os.path.join(videos_parent, 'marked')
    # calibration_vids_parent = os.path.join(videos_parent, 'calibration_videos')
    # calibration_files_parent = os.path.join(videos_parent, 'calibration_files')
    # dlc_mat_output_parent = os.path.join(videos_parent, 'matlab_readable_dlc')
    # trajectories_parent = os.path.join(videos_parent, 'trajectory_files')



    # cb_size = (6, 9)
    # test_calibration_file = '/Volumes/Untitled/DLC_output/calibration_images/2020/202012_calibration/202012_calibration_files/SR_boxCalibration_box04_20201217.mat'
    # test_pickle_file = '/Users/dan/Documents/deeplabcut/cropped_vids/R0382/R0382_20201216c_direct/R0382_20201216_17-23-50_005_direct_700-1350-270-935DLC_resnet50_skilled_reaching_directOct19shuffle1_200000_full.pickle'
    # skilled_reaching_calibration.read_matlab_calibration(test_calibration_file)
    # pickle_metadata = navigation_utilities.parse_dlc_output_pickle_name(test_pickle_file)
    # test_video_file = '/Users/dan/Documents/deeplabcut/videos_to_analyze/videos_to_crop/R0382/R0382_20201216c/R0382_box02_20201216_17-31-47_010.avi'
    # test_calibration_file = '/Users/dan/Documents/deeplabcut/videos_to_analyze/calibration_files/2021/202102_calibration/camera_calibration_videos_202102/CameraCalibration_box02_20210211_14-33-25.avi'
    # rat_database_name = '/Users/dan/Documents/deeplabcut/videos_to_analyze/SR_rat_database.csv'
    # rat_database_name = '/home/levlab/Public/rat_SR_videos_to_analyze/SR_rat_database.csv'
    label_videos = True

    rats_to_analyze = [452, 453, 468, 469, 470, 471, 472, 473, 474, 484, 485, 486, 487, 497, 498, 499, 500, 501, 502]

    # rat_df = skilled_reaching_io.read_rat_csv_database(rat_database_name)

    # if you only want to label the direct or mirror views, set the skip flag for the other view to True
    skipdirectlabel = False
    skipmirrorlabel = False

    gputouse = 3
    # step 1: preprocess videos to extract left mirror, right mirror, and direct views

    cam_names = ('direct', 'leftmirror', 'rightmirror')
    n_cams = len(cam_names)
    cgroup = CameraGroup.from_names(cam_names, fisheye=False)

    filtertype = 'h264'

    # parameters for cropping
    crop_params_dict = {
        cam_names[0]: [700, 1350, 270, 935],
        cam_names[1]: [1, 470, 270, 920],
        cam_names[2]: [1570, 2040, 270, 920]
    }
    cropped_vid_type = '.avi'

    # videos_parent = '/home/levlab/Public/rat_SR_videos_to_analyze'   # on the lambda machine
    # videos_parent = '/Users/dan/Documents/deeplabcut/videos_to_analyze'  # on home mac
    # videos_parent = '/Volumes/Untitled/videos_to_analyze'

    # view_config_paths = {
    #     'direct': '/home/levlab/Public/skilled_reaching_direct-Dan_Leventhal-2020-10-19/config.yaml',
    #     'mirror': '/home/levlab/Public/skilled_reaching_mirror-Dan_Leventhal-2020-10-19/config.yaml'
    # }

    # for lambda computer
    view_config_paths = {
        'direct': '/home/levlab/deeplabcut_projects/ratdirectsr-DanLeventhal-2023-06-07/config.yaml',
        'nearpaw': '/home/levlab/deeplabcut_projects/ratnearpawmirrorsr-DanLeventhal-2023-06-19/config.yaml',
        'farpaw': '/home/levlab/deeplabcut_projects/ratfarpawmirrorsr-DanLeventhal-2023-07-03/config.yaml'
    }

    # for DKL computer
    # view_config_paths = {
    #     'direct': '/home/levlab/Public/skilled_reaching_direct-Dan_Leventhal-2020-10-19/config.yaml',
    #     'mirror': '/home/levlab/Public/skilled_reaching_mirror-Dan_Leventhal-2020-10-19/config.yaml'
    # }
    DLC_folder_names = {'direct': 'ratdirectsr-DanLeventhal-2023-06-07',
                        'nearpaw': 'ratnearpawmirrorsr-DanLeventhal-2023-06-19',
                        'farpaw': 'ratfarpawmirrorsr-DanLeventhal-2023-07-03'}

    if sys.platform in ['win32']:
        # assume DKL computer
        DLC_top_folder = r'C:\Users\dleventh\Documents\deeplabcut_projects'
        data_root_folder = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\skilled_reaching'
    elif sys.platform in ['linux']:
        # lambda computer
        DLC_top_folder = '/home/dleventh/Documents/DLC_projects'
        data_root_folder = '/home/dleventh/SharedX/Neuro-Leventhal/data/skilled_reaching'

    # for dLight experiments
    videos_parents = {expt: os.path.join(data_root_folder, expt) for expt in experiment_list}
    video_root_folders = {expt: os.path.join(videos_parents[expt], 'data') for expt in experiment_list}
    cropped_videos_parents = {expt: os.path.join(videos_parents[expt], 'cropped') for expt in experiment_list}
    marked_videos_parents = {expt: os.path.join(videos_parents[expt], 'marked') for expt in experiment_list}
    calibration_vids_parents = {expt: os.path.join(videos_parents[expt], 'calibration_videos') for expt in experiment_list}
    calibration_files_parents = {expt: os.path.join(videos_parents[expt], 'calibration_files') for expt in experiment_list}
    dlc_mat_output_parents = {expt: os.path.join(videos_parents[expt], 'matlab_readable_dlc') for expt in experiment_list}
    trajectories_parents = {expt: os.path.join(videos_parents[expt], 'trajectory_files') for expt in experiment_list}

    view_keys = list(DLC_folder_names.keys())
    view_config_paths = {view_key: os.path.join(DLC_top_folder, DLC_folder_names[view_key], 'config.yaml') for view_key in view_keys}
    # view_config_paths = {
    #     'direct': r'C:\Users\dleventh\Documents\deeplabcut_projects\ratdirectsr-DanLeventhal-2023-06-07/config.yaml',
    #     'nearpaw': r'C:\Users\dleventh\Documents\deeplabcut_projects\ratnearpawmirrorsr-DanLeventhal-2023-06-19/config.yaml',
    #     'farpaw': r'C:\Users\dleventh\Documents\deeplabcut_projects\ratfarpawmirrorsr-DanLeventhal-2023-07-03/config.yaml'
    # }

    parent_directories = {expt: {
                                'videos_parent': videos_parents[expt],
                                'videos_root_folder': video_root_folders[expt],
                                'cropped_videos_parent': cropped_videos_parents[expt],
                                'marked_videos_parent': marked_videos_parents[expt],
                                'calibration_vids_parent': calibration_vids_parents[expt],
                                'calibration_files_parent': calibration_files_parents[expt],
                                'dlc_mat_output_parent': dlc_mat_output_parents[expt],
                                'trajectories_parent': trajectories_parents[expt]
                            }
                            for expt in experiment_list}

    # use the function below to write a charuco board to a file
    # 12 columns, 6 rows, square_length=20, marker_length=15
    # board = skilled_reaching_calibration.create_charuco(6,12,20,15)
    # skilled_reaching_calibration.write_charuco_image(board, 600, calibration_vids_parents['dLightPhotometry'])

    # test_cal_vid = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\skilled_reaching\test_calibration\GridCalibration_box01_20230823_18-40-45.avi'
    #
    # ret, mtx, dist = skilled_reaching_calibration.calibrate_single_camera(test_cal_vid, board)
    expt = 'dLightPhotometry'
    session_metadata = {
        'ratID': 'R0452',
        'rat_num': 452,
        'date': datetime(2023, 3, 28),
        'task': 'skilledreaching',
        'session_num': 1,
        'current': 0.
    }
    session_metadata_xlsx_path = os.path.join(video_root_folders[expt],
                                              'SR_{}_video_session_metadata.xlsx'.format(expt))
    calibration_metadata_df = skilled_reaching_io.read_session_metadata_xlsx(session_metadata_xlsx_path)
    # skilled_reaching_calibration.test_calibration(session_metadata, calibration_metadata_df, parent_directories[expt])

    for expt in experiment_list:

        # first, calibrate the cameras and write results into a .toml file

        # calibration_metadata_csv_path = os.path.join(calibration_vids_parents[expt], 'SR_calibration_vid_metadata.csv')
        session_metadata_xlsx_path = os.path.join(video_root_folders[expt], 'SR_{}_video_session_metadata.xlsx'.format(expt))
        # calibration_metadata_df = skilled_reaching_io.read_calibration_metadata_csv(calibration_metadata_csv_path)
        calibration_metadata_df = skilled_reaching_io.read_session_metadata_xlsx(session_metadata_xlsx_path)

        crop_videos.crop_all_calibration_videos(parent_directories[expt],
                                    calibration_metadata_df,
                                    vidtype='.avi',
                                    view_list=cam_names,
                                    filtertype=filtertype)

        # current_cropped_calibration_vids = skilled_reaching_calibration.crop_calibration_video(full_calib_vid_name,
        #                                                                                        session_row,
        #                                                                                        filtertype=filtertype)

        calibrate_all_sessions(parent_directories[expt],
                               calibration_metadata_df,
                               cgroup,
                               view_list=cam_names,
                               filtertype=filtertype)

    for expt in experiment_list:
        rat_df = skilled_reaching_io.read_rat_db(parent_directories[expt], rat_db_fnames[expt])
        folders_to_analyze = navigation_utilities.find_folders_to_analyze(cropped_videos_parents[expt], view_list=cam_names)

        scorernames = analyze_cropped_videos(folders_to_analyze, view_config_paths, parent_directories[expt], rat_df,
                                             cropped_vid_type=cropped_vid_type,
                                             gputouse=gputouse,
                                             save_as_csv=True,
                                             create_marked_vids=create_marked_vids)


        crop_params_csv_path = os.path.join(video_root_folders[expt], 'SR_video_crop_regions.csv')
        crop_params_df = skilled_reaching_io.read_crop_params_csv(crop_params_csv_path)
        crop_filtertype = 'h264'  # currently choices are 'h264' or 'mjpeg2jpeg'. Python based vid conversion (vs labview) should use h264

        video_folder_list = navigation_utilities.get_video_folders_to_crop(video_root_folders[expt], rats_to_analyze=rats_to_analyze)
        cropped_video_directories = crop_videos.preprocess_videos(video_folder_list, cropped_videos_parents[expt], crop_params_df, cam_names, vidtype='avi', filtertype=crop_filtertype)

        # calibrate_all_sessions(calibration_vids_parent, calibration_files_parent, crop_params_df, cb_size=cb_size)

    # skilled_reaching_calibration.calibrate_camera_from_video(test_calibration_file, calibration_parent, cb_size=cb_size)

    # video_metadata = navigation_utilities.parse_video_name(test_video_file)


    # metadata_list = navigation_utilities.find_marked_vids_for_3d_reconstruction(marked_videos_parent, dlc_mat_output_parent, rat_df)
    #
    # for md in metadata_list:
    #     reconstruct_3d.triangulate_video(md, videos_parent, marked_videos_parent, calibration_parent, dlc_mat_output_parent, rat_df, view_list=view_list)

    # vid_folder_list = ['/Users/dan/Documents/deeplabcut/R0382_20200909c','/Users/dan/Documents/deeplabcut/R0230_20181114a']

    # folders_to_reconstruct = navigation_utilities.find_folders_to_reconstruct(cropped_videos_parent)
    # reconstruct_3d.reconstruct_folders(folders_to_reconstruct, parent_directories, rat_df)

    # reconstruct_3d.test_reconstruction(parent_directories, rat_df)

    # step 2: run the vids through DLC
    # parameters for running DLC
    # need to update these paths when moved to the lambda machine


    for expt in experiment_list:
        rat_db = skilled_reaching_io.read_rat_db(parent_directories[expt], rat_db_fnames[expt])
        folders_to_analyze = navigation_utilities.find_folders_to_analyze(cropped_videos_parents[expt], view_list=cam_names)

        scorernames = analyze_cropped_videos(folders_to_analyze, view_config_paths, marked_videos_parents[expt], rat_db, cropped_vid_type=cropped_vid_type, gputouse=gputouse, save_as_csv=True)

        if label_videos:
            create_labeled_videos(cropped_videos_parent,
                                  marked_videos_parent,
                                  view_config_paths,
                                  scorernames,
                                  cropped_vid_type=cropped_vid_type,
                                  skipdirect=skipdirectlabel,
                                  skipmirror=skipmirrorlabel,
                                  view_list=cam_names)

    # step 3: make sure calibration has been run for these sessions
    # find list of all analyzed videos; extract dates and boxes for each session

    # step 4: reconstruct the 3d trajectories
    folders_to_reconstruct = navigation_utilities.find_folders_to_reconstruct(cropped_videos_parent)
    reconstruct_3d.reconstruct_folders(folders_to_reconstruct, marked_videos_parent, calibration_files_parent, trajectories_parent)

    # step 5: post-processing including smoothing (should there be smoothing on the 2-D images first?)


