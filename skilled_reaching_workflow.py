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
import deeplabcut


def analyze_cropped_videos(folders_to_analyze, view_config_paths, marked_vids_parent, cropped_vid_type='.avi', gputouse=0, save_as_csv=True):
    '''

    :param folders_to_analyze:
    :param view_config_paths:
    :param cropped_vid_type:
    :param gputouse:
    :return: scorernames - dictionary with keys 'direct' and 'mirror' containing the scorername strings returned by
        deeplabcut.analyze_videos
    '''

    view_list = folders_to_analyze.keys()
    scorernames = {'direct': '', 'mirror': ''}
    for view in view_list:
        if 'direct' in view:
            dlc_network = 'direct'
        elif 'mirror' in view:
            dlc_network = 'mirror'
        else:
            print(view + ' does not contain the keyword "direct" or "mirror"')
            continue

        config_path = view_config_paths[dlc_network]
        current_view_folders = folders_to_analyze[view]

        for current_folder in current_view_folders:
            cropped_video_list = glob.glob(current_folder + '/*' + cropped_vid_type)
            #todo: skip if analysis already done and stored in the _marked folder
            scorername = deeplabcut.analyze_videos(config_path,
                                      cropped_video_list,
                                      videotype=cropped_vid_type,
                                      gputouse=gputouse,
                                      save_as_csv=save_as_csv)
            scorernames[dlc_network] = scorername

            new_dir = navigation_utilities.create_marked_vids_folder(current_folder, cropped_videos_parent,
                                                                     marked_vids_parent)
            pickle_list = glob.glob(os.path.join(current_folder, '*.pickle'))
            for pickle_file in pickle_list:
                # if the file already exists in the marked_vid directory, don't move it
                _, pickle_name = os.path.split(pickle_file)
                if not os.path.isfile(os.path.join(new_dir, pickle_name)):
                    shutil.copy(pickle_file, new_dir)

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


def calibrate_all_sessions(calibration_vids_parent, calibration_files_parent, crop_params_df, cb_size=(6, 9), vidtype='.avi'):
    '''
    loop through all folders containing calibration videos and store calibration parameters
    :param calibration_vids_parent:
    :param calibration_files_parent:
    :param crop_params_df: dataframe containing cropping parameters for each box-date
    :param cb_size:
    :return:
    '''

    if vidtype[0] != '.':
        vidtype = '.' + vidtype

    calib_vid_folders = navigation_utilities.find_calibration_vid_folders_dLight(calibration_vids_parent)

    for cf in calib_vid_folders:
        # crop the calibration videos
        calib_vids = glob.glob(os.path.join(cf, 'GridCalibration_*' + vidtype))

        orig_im_size = []
        cropped_vid_names = []
        for calib_vid in calib_vids:
            cropped_vid_names.append(skilled_reaching_calibration.crop_calibration_video(calib_vid, crop_params_df))
            vid_obj = cv2.VideoCapture(calib_vid)
            orig_im_size.append((int(vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH))))
            vid_obj.release()
        # cropped_vid_names contains a list of lists; each individual list contains the names of 3 files, one each for
        # the direct, leftmirror, and rightmirror views

        # FOR NOW, JUST CROP THE VIDEOS. COMMENT THE CALIBRATION PART BACK IN LATER (HOPEFULLY CAN JUST RUN ANIPOSE HERE...
        # for i_calib_vid, cropped_vids_set in enumerate(cropped_vid_names):
        #     # each cropped_vids_set should contain 3 video names for direct, left, right views
        #
        #     calibration_summary_name = navigation_utilities.create_calibration_summary_name(calib_vids[i_calib_vid], calibration_files_parent)
        #
        #     if os.path.isfile(calibration_summary_name):
        #         calibration_data = skilled_reaching_io.read_pickle(calibration_summary_name)
        #     else:
        #         calibration_data = skilled_reaching_calibration.collect_cb_corners(cropped_vids_set, cb_size)
        #         calibration_data['orig_im_size'] = orig_im_size[i_calib_vid]
        #         skilled_reaching_io.write_pickle(calibration_summary_name, calibration_data)
        #
        #     # now perform the actual calibration
        #     skilled_reaching_calibration.multi_mirror_calibration(calibration_data, calibration_summary_name)


if __name__ == '__main__':

    # for dLight experiments
    videos_parent = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\skilled_reaching\dLight_Photometry'
    # video_root_folder = os.path.join(videos_parent, 'videos_to_crop')
    video_root_folder = os.path.join(videos_parent, 'data')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped')
    marked_videos_parent = os.path.join(videos_parent, 'marked')
    calibration_vids_parent = os.path.join(videos_parent, 'calibration_videos')
    calibration_files_parent = os.path.join(videos_parent, 'calibration_files')
    dlc_mat_output_parent = os.path.join(videos_parent, 'matlab_readable_dlc')
    trajectories_parent = os.path.join(videos_parent, 'trajectory_files')

    parent_directories = {'data': video_root_folder,
                          'cropped': cropped_videos_parent,
                          'marked': marked_videos_parent,
                          'calibration_vids': calibration_vids_parent,
                          'calibration_files': calibration_files_parent
                          }

    rat_db_fname = 'rat_dlight_SRdb.xlsx'
    rat_db = skilled_reaching_io.read_rat_db(parent_directories, rat_db_fname)


    cb_size = (6, 9)
    # test_calibration_file = '/Volumes/Untitled/DLC_output/calibration_images/2020/202012_calibration/202012_calibration_files/SR_boxCalibration_box04_20201217.mat'
    # test_pickle_file = '/Users/dan/Documents/deeplabcut/cropped_vids/R0382/R0382_20201216c_direct/R0382_20201216_17-23-50_005_direct_700-1350-270-935DLC_resnet50_skilled_reaching_directOct19shuffle1_200000_full.pickle'
    # skilled_reaching_calibration.read_matlab_calibration(test_calibration_file)
    # pickle_metadata = navigation_utilities.parse_dlc_output_pickle_name(test_pickle_file)
    # test_video_file = '/Users/dan/Documents/deeplabcut/videos_to_analyze/videos_to_crop/R0382/R0382_20201216c/R0382_box02_20201216_17-31-47_010.avi'
    # test_calibration_file = '/Users/dan/Documents/deeplabcut/videos_to_analyze/calibration_files/2021/202102_calibration/camera_calibration_videos_202102/CameraCalibration_box02_20210211_14-33-25.avi'
    rat_database_name = '/Users/dan/Documents/deeplabcut/videos_to_analyze/SR_rat_database.csv'
    rat_database_name = '/home/levlab/Public/rat_SR_videos_to_analyze/SR_rat_database.csv'
    label_videos = True

    rats_to_analyze = [452, 453, 468, 469, 470, 471, 472, 473, 474, 484]

    # rat_df = skilled_reaching_io.read_rat_csv_database(rat_database_name)

    # if you only want to label the direct or mirror views, set the skip flag for the other view to True
    skipdirectlabel = False
    skipmirrorlabel = False

    gputouse = 2
    # step 1: preprocess videos to extract left mirror, right mirror, and direct views

    view_list = ('direct', 'leftmirror', 'rightmirror')
    # parameters for cropping
    crop_params_dict = {
        view_list[0]: [700, 1350, 270, 935],
        view_list[1]: [1, 470, 270, 920],
        view_list[2]: [1570, 2040, 270, 920]
    }
    cropped_vid_type = '.avi'

    # videos_parent = '/home/levlab/Public/rat_SR_videos_to_analyze'   # on the lambda machine
    # videos_parent = '/Users/dan/Documents/deeplabcut/videos_to_analyze'  # on home mac
    # videos_parent = '/Volumes/Untitled/videos_to_analyze'



    parent_directories = {
        'videos_parent': videos_parent,
        'videos_root_folder': video_root_folder,
        'cropped_videos_parent': cropped_videos_parent,
        'marked_videos_parent': marked_videos_parent,
        'calibration_vids_parent': calibration_vids_parent,
        'calibration_files_parent': calibration_files_parent,
        'dlc_mat_output_parent': dlc_mat_output_parent,
        'trajectories_parent': trajectories_parent
    }
    crop_params_csv_path = os.path.join(video_root_folder, 'SR_video_crop_regions.csv')
    crop_params_df = skilled_reaching_io.read_crop_params_csv(crop_params_csv_path)
    crop_filtertype = 'h264'  # currently choices are 'h264' or 'mjpeg2jpeg'. Python based vid conversion (vs labview) should use h264

    video_folder_list = navigation_utilities.get_video_folders_to_crop(video_root_folder, rats_to_analyze=rats_to_analyze)
    cropped_video_directories = crop_videos.preprocess_videos(video_folder_list, cropped_videos_parent, crop_params_df, view_list, vidtype='avi', filtertype=crop_filtertype)

    calibrate_all_sessions(calibration_vids_parent, calibration_files_parent, crop_params_df, cb_size=cb_size)

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

    #todo: complete the camera calibration algorithms
    video_folder_list = navigation_utilities.get_video_folders_to_crop(video_root_folder)
    cropped_video_directories = crop_videos.preprocess_videos(video_folder_list, cropped_videos_parent, crop_params_df, view_list, vidtype='avi')

    # step 2: run the vids through DLC
    # parameters for running DLC
    # need to update these paths when moved to the lambda machine
    view_config_paths = {
        'direct': '/home/levlab/Public/skilled_reaching_direct-Dan_Leventhal-2020-10-19/config.yaml',
        'mirror': '/home/levlab/Public/skilled_reaching_mirror-Dan_Leventhal-2020-10-19/config.yaml'
    }
    folders_to_analyze = navigation_utilities.find_folders_to_analyze(cropped_videos_parent, view_list=view_list)

    scorernames = analyze_cropped_videos(folders_to_analyze, view_config_paths, marked_videos_parent, cropped_vid_type=cropped_vid_type, gputouse=gputouse, save_as_csv=True)

    if label_videos:
        create_labeled_videos(cropped_videos_parent,
                              marked_videos_parent,
                              view_config_paths,
                              scorernames,
                              cropped_vid_type=cropped_vid_type,
                              skipdirect=skipdirectlabel,
                              skipmirror=skipmirrorlabel,
                              view_list=view_list)

    # step 3: make sure calibration has been run for these sessions
    # find list of all analyzed videos; extract dates and boxes for each session

    # step 4: reconstruct the 3d trajectories
    folders_to_reconstruct = navigation_utilities.find_folders_to_reconstruct(cropped_videos_parent)
    reconstruct_3d.reconstruct_folders(folders_to_reconstruct, marked_videos_parent, calibration_files_parent, trajectories_parent)

    # step 5: post-processing including smoothing (should there be smoothing on the 2-D images first?)


