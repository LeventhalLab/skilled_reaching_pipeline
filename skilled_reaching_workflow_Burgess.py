import glob
import os
import crop_Burgess_videos
import skilled_reaching_calibration
import reconstruct_3d_optitrack
import navigation_utilities
import skilled_reaching_io
import deeplabcut
import shutil
from datetime import datetime


def analyze_cropped_optitrack_videos(folders_to_analyze, config_path, parent_directories, cropped_vid_type='.avi', gputouse=0, save_as_csv=True):
    '''

    :param folders_to_analyze:
    :param view_config_paths:
    :param cropped_vid_type:
    :param gputouse:
    :return: scorernames - dictionary with keys 'direct' and 'mirror' containing the scorername strings returned by
        deeplabcut.analyze_videos
    '''

    mice_to_skip = ['BT32','BT33','dLight15',
                    'dLight26', 'dLight28', 'dLight31', 'dLight36', 'dLight42', 'dLight49', 'dLight50', 'dLight53', 'dLight54',
                    'GFP3', 'GFP4', 'GFP5', 'GFP14']
    if cropped_vid_type[0]=='.':
        pass
    else:
        cropped_vid_type = '.' + cropped_vid_type

    marked_vids_parent = parent_directories['marked_vids_parent']

    cam_list = folders_to_analyze.keys()
    for cam_name in cam_list:
        current_cam_folders = folders_to_analyze[cam_name]

        for current_folder in current_cam_folders:
            cropped_folder_metadata = navigation_utilities.parse_cropped_optitrack_video_folder(current_folder)
            if cropped_folder_metadata['mouseID'] == 'GFP14' and \
                    cropped_folder_metadata['session_date'] in [datetime(2022,3,9,0,0,0)]:
                continue

            if cropped_folder_metadata['mouseID'] in mice_to_skip:
                continue

            cropped_video_list = glob.glob(current_folder + '/*' + cropped_vid_type)
            #todo: skip if analysis already done and stored in the _marked folder
            scorername = deeplabcut.analyze_videos(config_path,
                                      cropped_video_list,
                                      videotype=cropped_vid_type,
                                      gputouse=gputouse,
                                      save_as_csv=save_as_csv)
            # deeplabcut.convert_detections2tracklets(config_path, cropped_video_list, videotype='mp4', shuffle=1, trainingsetindex=0)
            # deeplabcut.stitch_tracklets(config_path, cropped_video_list, videotype='mp4', shuffle=1, trainingsetindex=0)

            new_dir = navigation_utilities.create_optitrack_marked_vids_folder(current_folder, cropped_vids_parent,
                                                                               marked_vids_parent)
            # pickle_list = glob.glob(os.path.join(current_folder, '*.pickle'))
            # for pickle_file in pickle_list:
            #     # if the file already exists in the marked_vid directory, don't move it
            #     _, pickle_name = os.path.split(pickle_file)
            #     if not os.path.isfile(os.path.join(new_dir, pickle_name)):
            #         shutil.move(pickle_file, new_dir)
            #
            # csv_list = glob.glob(os.path.join(current_folder, '*.csv'))
            # for csv_file in csv_list:
            #     # if the file already exists in the marked_vid directory, don't move it
            #     _, csv_name = os.path.split(csv_file)
            #     if not os.path.isfile(os.path.join(new_dir, csv_name)):
            #         shutil.move(csv_file, new_dir)
            #
            # h5_list = glob.glob(os.path.join(current_folder, '*.h5'))
            # for h5_file in h5_list:
            #     # if the file already exists in the marked_vid directory, don't move it
            #     _, h5_name = os.path.split(h5_file)
            #     if not os.path.isfile(os.path.join(new_dir, h5_name)):
            #         shutil.move(h5_file, new_dir)

    return scorername


def create_labeled_optitrack_videos(folders_to_analyze, parent_directories, config_path, scorername,
                                  cropped_vid_type='.avi'):
    '''
    :param folders_to_analyze:
    :param view_config_paths:
    :param scorernames: dictionary with keys 'direct' and 'mirror'
    :param cropped_vid_type:
    :param move_to_new_folder: if True, create a new folder in which the marked videos and analysis files are stored
        to make it easier to move them to another computer without taking the original videos with them
    :return:
    '''

    marked_vids_parent = parent_directories['marked_vids_parent']
    if cropped_vid_type[0]=='.':
        pass
    else:
        cropped_vid_type = '.' + cropped_vid_type

    # in case there are some previously cropped videos that need to be analyzed
    cam_list = folders_to_analyze.keys()
    for cam_name in cam_list:

        # do some cam02 vids
        cam_name = 'cam02'
        current_cam_folders = folders_to_analyze[cam_name]

        for current_folder in current_cam_folders:
            _, folder_name = os.path.split(current_folder)
            if current_folder[:8] == 'dLight36':
                continue
            cropped_video_list = glob.glob(current_folder + '/*' + cropped_vid_type)
            try:
                deeplabcut.create_video_with_all_detections(config_path, cropped_video_list)
                # deeplabcut.create_video_with_all_detections(config_path, cropped_video_list, scorername)
            except:
                pass

            # current_basename = os.path.basename(current_folder)
            new_dir = navigation_utilities.create_marked_vids_folder(current_folder, cropped_vids_parent,
                                                                     marked_vids_parent)
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
                        shutil.copy(pickle_file, new_dir)


def reconstruct_optitrack_3d(parent_directories):
    '''
    perform 3d reconstruction of all videos for which at least two cropped video views are present and those views have
    been calibrated
    :param cropped_vid_parent: parent directory for cropped videos. Has structure:
        cropped_vids_parent-->mouseID-->mouseID_YYYYmm-->mouseID_YYYYmmdd-->mouseID_YYYYmmdd_camXX (XX = 01 or 02)
    :param cal_data_parent: parent directory for folders containing pickle files with calibration results. Has structure:
        cal_data_parent-->calibration_data_YYYY-->calibration_data_YYYYmm
    :return:
    '''

    folders_to_reconstruct = navigation_utilities.find_optitrack_folders_to_analyze(parent_directories, cam_list=(1, 2))
    cam_names = folders_to_reconstruct.keys()

    # start with cam01, find all matching cam02 data
    cam01_folders = folders_to_reconstruct['cam01']
    cam02_folders = folders_to_reconstruct['cam02']

    # mouseID = 'dLight15'
    session_date = datetime(2021, 7, 16)

    for i_dir, cam01_dir in enumerate(cam01_folders):
        cropped_folder_metadata = navigation_utilities.parse_cropped_optitrack_video_folder(cam01_dir)

        # if not(cropped_folder_metadata['mouseID'] == mouseID and session_date.date() == cropped_folder_metadata['session_date'].date()):
        #     continue
        # for debugging, let's look at the second session in the list
        # if i_dir < 38:
        #     continue


        # check to see if calibration has been performed for this date,then find matching cam02_dir
        cam02_dir = cam01_dir.replace('cam01', 'cam02')
        if cam02_dir not in cam02_folders:
            print('no matching cam02 directory for {}'.format(cam01_dir))
            continue
        view_directories = (cam01_dir, cam02_dir)
        reconstruct_3d_optitrack.reconstruct_optitrack_session(view_directories, parent_directories)


if __name__ == '__main__':

    cropped_vid_type = 'avi'
    gputouse = 0
    cam_list = (1, 2)
    label_videos = True
    checkerboard_square_size = 7   # in mm

    # Burgess_DLC_config_path = '/home/dleventh/Public/mouse_skilledreaching/sr_hf-DanL-2021-11-05/config.yaml'
    Burgess_DLC_config_path = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\deeplabcut_projects\mouse_sr_dlc\sr_hf-DanL-2021-11-05\config.yaml'
    # r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\skilled_reaching\dLight_Photometry'
    # mouse_reaching_parent = '/home/dleventh/SharedX/Neuro-Leventhal/data/hf_mouse/'   # on lambda
    mouse_reaching_parent = r'\\corexfs.med.umich.edu\SharedX\Neuro-Leventhal\data\hf_mouse'  # on office desktop
    # mouse_reaching_parent = 'C:\\Users\\dklev\Dropbox (University of Michigan)\\MED-LeventhalLab\\Burgess_data\\hf_mouse'   # dropbox on laptop
    scoring_csv = 'GFP Video Scoring - GFP8.csv'

    video_root_folder = os.path.join(mouse_reaching_parent, 'mouse_SR_videos_tocrop')
    cropped_vids_parent = os.path.join(mouse_reaching_parent, 'cropped')
    marked_vids_parent = os.path.join(mouse_reaching_parent, 'marked')
    cal_vids_parent = os.path.join(mouse_reaching_parent, 'mouse_SR_calibration_videos')
    cal_data_parent = os.path.join(mouse_reaching_parent, 'mouse_SR_calibration_data')
    reconstruct_3d_parent = os.path.join(mouse_reaching_parent, 'recon3d')
    manual_scoring_parent = os.path.join(mouse_reaching_parent, 'mouse_SR_manual_scoring')

    scoring_csv = os.path.join(mouse_reaching_parent, scoring_csv)

    parent_directories = {
        'mouse_reaching_parent': mouse_reaching_parent,
        'video_root_folder': video_root_folder,
        'cropped_vids_parent': cropped_vids_parent,
        'marked_vids_parent': marked_vids_parent,
        'cal_vids_parent': cal_vids_parent,
        'cal_data_parent': cal_data_parent,
        'reconstruct3d_parent': reconstruct_3d_parent,
        'manual_scoring_parent': manual_scoring_parent
    }
    # test_metadata = {'mouseID': 'dLight15',
    #                  'date': datetime.strptime('20210713', '%Y%m%d')}
    # skilled_reaching_calibration.refine_optitrack_calibration_from_dlc(test_metadata, parent_directories)

    crop_params_csv_path = os.path.join(video_root_folder, 'optitrack_SR_video_crop_regions.csv')

    cb_size = (7, 10)

    # step 1 - run all the calibrations
    # UNCOMMENT BELOW
    # calib_folder = os.path.join(cal_data_parent, 'calibration_data_2022', 'calibration_data_202202')
    # skilled_reaching_calibration.compare_calibration_files(calib_folder)

    # test_cal_file = '/home/levlab/Public/mouse_SR_videos_to_analyze/mouse_SR_calibration_data/calibrationfiles_2022/calibrationfiles_202205/calibrationdata_20220510_10-00-00.pickle'
    # cal_metadata = navigation_utilities.parse_optitrack_calibration_data_name(test_cal_file)
    # skilled_reaching_calibration.show_cal_images_with_epilines(cal_metadata, parent_directories)
    #
    # skilled_reaching_calibration.calibrate_all_Burgess_vids(parent_directories, cb_size=cb_size, checkerboard_square_size=checkerboard_square_size)

    # once all initial calibrations are done, refine for each video pair
    # vid_folder_list = navigation_utilities.get_Burgess_video_folders_to_crop(video_root_folder)
    # skilled_reaching_calibration.refine_calibrations_from_orig_vids(vid_folder_list, parent_directories)

    # step 2 - crop all videos of mice reaching
    vid_folder_list = navigation_utilities.get_Burgess_video_folders_to_crop(video_root_folder)
    crop_params_df = skilled_reaching_io.read_crop_params_csv(crop_params_csv_path)
    # UNCOMMENT BELOW
    # cropped_video_directories = crop_Burgess_videos.preprocess_Burgess_videos(vid_folder_list, parent_directories, crop_params_df, cam_list, vidtype='avi')

    # step 3 - run DLC on each cropped video
    # UNCOMMENT BELOW
    folders_to_analyze = navigation_utilities.find_optitrack_folders_to_analyze(parent_directories, cam_list=cam_list)
    # scorername = analyze_cropped_optitrack_videos(folders_to_analyze, Burgess_DLC_config_path, parent_directories, cropped_vid_type=cropped_vid_type, gputouse=gputouse, save_as_csv=True)
    scorername = 'DLC_dlcrnetms5_mouse_headfixed_skilledreachingNov5shuffle1_100000'

    # UNCOMMENT BELOW
    # if label_videos:
    #     #todo: working here - create labeled videos
    #     # try:
    #     create_labeled_optitrack_videos(folders_to_analyze,
    #                               parent_directories,
    #                               Burgess_DLC_config_path,
    #                               scorername,
    #                               cropped_vid_type=cropped_vid_type
    #                               )
        # except:
        #     pass
    # step 4 - reconstruct 3D images
    reconstruct_optitrack_3d(parent_directories)
    #
    reconstruct_3d_optitrack.test_optitrack_reconstruction(parent_directories)


    reconstruct_3d_optitrack.refine_trajectories(parent_directories)