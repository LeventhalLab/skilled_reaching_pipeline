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


def analyze_cropped_optitrack_videos(folders_to_analyze, config_path, marked_vids_parent, cropped_vid_type='.avi', gputouse=0, save_as_csv=True):
    '''

    :param folders_to_analyze:
    :param view_config_paths:
    :param cropped_vid_type:
    :param gputouse:
    :return: scorernames - dictionary with keys 'direct' and 'mirror' containing the scorername strings returned by
        deeplabcut.analyze_videos
    '''

    if cropped_vid_type[0]=='.':
        pass
    else:
        cropped_vid_type = '.' + cropped_vid_type

    cam_list = folders_to_analyze.keys()
    for cam_name in cam_list:

        current_cam_folders = folders_to_analyze[cam_name]

        for current_folder in current_cam_folders:
            cropped_video_list = glob.glob(current_folder + '/*' + cropped_vid_type)
            #todo: skip if analysis already done and stored in the _marked folder
            scorername = deeplabcut.analyze_videos(config_path,
                                      cropped_video_list,
                                      videotype=cropped_vid_type,
                                      gputouse=gputouse,
                                      save_as_csv=save_as_csv)
            # deeplabcut.convert_detections2tracklets(config_path, cropped_video_list, videotype='mp4', shuffle=1, trainingsetindex=0)
            # deeplabcut.stitch_tracklets(config_path, cropped_video_list, videotype='mp4', shuffle=1, trainingsetindex=0)

            new_dir = navigation_utilities.create_optitrack_marked_vids_folder(current_folder, cropped_videos_parent,
                                                                               marked_vids_parent)
            pickle_list = glob.glob(os.path.join(current_folder, '*.pickle'))
            for pickle_file in pickle_list:
                # if the file already exists in the marked_vid directory, don't move it
                _, pickle_name = os.path.split(pickle_file)
                if not os.path.isfile(os.path.join(new_dir, pickle_name)):
                    shutil.copy(pickle_file, new_dir)

    return scorername


def create_labeled_optitrack_videos(folders_to_analyze, marked_vids_parent, config_path, scorername,
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
    if cropped_vid_type[0]=='.':
        pass
    else:
        cropped_vid_type = '.' + cropped_vid_type

    # in case there are some previously cropped videos that need to be analyzed
    cam_list = folders_to_analyze.keys()
    for cam_name in cam_list:

        current_cam_folders = folders_to_analyze[cam_name]

        for current_folder in current_cam_folders:
            _, folder_name = os.path.split(current_folder)
            if current_folder[:8] == 'dLight36':
                continue
            cropped_video_list = glob.glob(current_folder + '/*' + cropped_vid_type)
            try:
                deeplabcut.create_video_with_all_detections(config_path, cropped_video_list, scorername)
            except:
                pass

            # current_basename = os.path.basename(current_folder)
            new_dir = navigation_utilities.create_marked_vids_folder(current_folder, cropped_videos_parent,
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
                        shutil.move(pickle_file, new_dir)


def reconstruct_optitrack_3d(cropped_vid_parent, cal_data_parent, videos_parent):
    '''

    :param cropped_vid_parent:
    :param cal_data_parent:
    :return:
    '''

    folders_to_reconstruct = navigation_utilities.find_optitrack_folders_to_analyze(cropped_vid_parent, cam_list=(1, 2))
    cam_names = folders_to_reconstruct.keys()

    # start with cam01, find all matching cam02 data
    cam01_folders = folders_to_reconstruct['cam01']
    cam02_folders = folders_to_reconstruct['cam02']

    for cam01_dir in cam01_folders:
        #todo: check to see if calibration has been performed for this date,then find matching cam02_dir
        cam02_dir = cam01_dir.replace('cam01', 'cam02')
        if cam02_dir not in cam02_folders:
            print('no matching cam02 directory for {}'.format(cam01_dir))
            continue
        view_directories = (cam01_dir, cam02_dir)
        reconstruct_3d_optitrack.reconstruct_optitrack_session(view_directories, cal_data_parent, videos_parent)



if __name__ == '__main__':

    cropped_vid_type = 'avi'
    gputouse = 2
    cam_list = (1, 2)
    label_videos = True

    Burgess_DLC_config_path = '/home/levlab/Public/mouse_headfixed_skilledreaching-DanL-2021-11-05/config.yaml'

    videos_parent = '/home/levlab/Public/mouse_SR_videos_to_analyze'
    video_root_folder = os.path.join(videos_parent, 'mouse_SR_videos_tocrop')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_mouse_SR_videos')
    marked_videos_parent = os.path.join(videos_parent, 'marked_mouse_SR_videos')

    cal_vid_parent = os.path.join(videos_parent, 'mouse_SR_calibration_videos')
    cal_data_parent = os.path.join(videos_parent, 'mouse_SR_calibration_data')

    crop_params_csv_path = os.path.join(video_root_folder, 'optitrack_SR_video_crop_regions.csv')

    cb_size = (7, 10)

    # step 1 - run all the calibrations
    # UNCOMMENT BELOW
    # skilled_reaching_calibration.calibrate_all_Burgess_vids(cal_vid_parent, cal_data_parent, cb_size=cb_size)

    # step 2 - crop all videos of mice reaching
    vid_folder_list = navigation_utilities.get_Burgess_video_folders_to_crop(video_root_folder)
    crop_params_df = skilled_reaching_io.read_crop_params_csv(crop_params_csv_path)
    # UNCOMMENT BELOW
    # cropped_video_directories = crop_Burgess_videos.preprocess_Burgess_videos(vid_folder_list, cropped_videos_parent, crop_params_df, cam_list, vidtype='avi')

    # step 3 - run DLC on each cropped video
    # folders_to_analyze = navigation_utilities.find_optitrack_folders_to_analyze(cropped_videos_parent, cam_list=cam_list)
    # UNCOMMENT BELOW
    # scorername = analyze_cropped_optitrack_videos(folders_to_analyze, Burgess_DLC_config_path, marked_videos_parent, cropped_vid_type=cropped_vid_type, gputouse=gputouse, save_as_csv=True)

    # UNCOMMENT BELOW
    # if label_videos:
    #     #todo: working here - create labeled videos
    #     try:
    #         create_labeled_optitrack_videos(folders_to_analyze,
    #                                   marked_videos_parent,
    #                                   Burgess_DLC_config_path,
    #                                   scorername,
    #                                   cropped_vid_type=cropped_vid_type
    #                                   )
    #     except:
    #         pass
    # step 4 - reconstruct 3D images
    reconstruct_optitrack_3d(cropped_videos_parent, cal_data_parent, videos_parent)