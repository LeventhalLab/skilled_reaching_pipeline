import glob
import os
import numpy as np
# from moviepy.editor import *
import subprocess
import cv2
import shutil
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import skilled_reaching_calibration
import navigation_utilities
import skilled_reaching_io


def crop_params_dict_from_df(crop_params_df, session_date, box_num, view_list=['dir', 'lm', 'rm']):

    # find the row with the relevant session data and box number
    date_box_row = crop_params_df[(crop_params_df['date'] == session_date) & (crop_params_df['box_num'] == box_num)]

    if date_box_row.empty:
        # crop_params_dict = {
        #     view_list[0]: [700, 1350, 270, 935],
        #     view_list[1]: [1, 470, 270, 920],
        #     view_list[2]: [1570, 2040, 270, 920]
        # }
        crop_params_dict = {}
    elif date_box_row.shape[0] == 1:
        crop_params_dict = dict.fromkeys(view_list, None)
        for view in view_list:
            left_edge = date_box_row[view + '_left'].values[0]
            right_edge = date_box_row[view + '_right'].values[0]
            top_edge = date_box_row[view + '_top'].values[0]
            bot_edge = date_box_row[view + '_bottom'].values[0]

            if any([pd.isna(left_edge), pd.isna(right_edge), pd.isna(top_edge),  pd.isna(bot_edge)]):
                crop_params_dict = {}
                break
            else:
                crop_params_dict[view] = [left_edge,
                                          right_edge,
                                          top_edge,
                                          bot_edge]
    else:
        # there must be more than one valid row in the table, use default
        # crop_params_dict = {
        #     view_list[0]: [700, 1350, 270, 935],
        #     view_list[1]: [1, 470, 270, 920],
        #     view_list[2]: [1570, 2040, 270, 920]
        # }
        crop_params_dict = {}

    return crop_params_dict


def crop_folders(video_folder_list, cropped_vids_parent, crop_params, view_list, vidtype='avi', filtertype='mjpeg2jpeg'):
    """
    :param video_folder_list:
    :param cropped_vids_parent:
    :param crop_params: either a dictionary with keys 'dir', 'lm', 'rm', each with a 4-element list [left, right, top, bottom]
            OR a pandas dataframe with columns 'date', 'box_num', 'direct_left', 'direct_right',...
    :param vidtype:
    :return:
    """


    cropped_video_directories = navigation_utilities.create_cropped_video_destination_list(cropped_vids_parent, video_folder_list, view_list)
    # make sure vidtype starts with a '.'
    if vidtype[0] != '.':
        vidtype = '.' + vidtype

    for i_path, vids_path in enumerate(video_folder_list):
        # find files with extension vidtype
        vids_list = glob.glob(os.path.join(vids_path, '*' + vidtype))
        if not bool(vids_list):
            # vids_list is empty
            continue

        # if crop_params is a DataFrame object, create a crop_params dictionary based on the current folder
        if isinstance(crop_params, pd.DataFrame):
            # pick an .avi file in this folder
            test_vid = vids_list[0]
            vid_metadata = navigation_utilities.parse_video_name(test_vid)
            session_date = vid_metadata['triggertime'].date()
            crop_params_dict = crop_params_dict_from_df(crop_params, session_date, vid_metadata['boxnum'])
        elif isinstance(crop_params, dict):
            crop_params_dict = crop_params

        if not bool(crop_params_dict):
            # the crop parameters dictionary is empty, skip to the next folder
            continue

        for i_view, view_name in enumerate(view_list):
            if 'rm' in view_name:
                fliplr = True
            else:
                fliplr = False

            current_crop_params = crop_params_dict[view_name]
            dest_folder = cropped_video_directories[i_view][i_path]
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)

            for full_vid_path in vids_list:
                # todo: calibrate the camera and undistort the videos prior to cropping, then don't allow calculation of distortion
                # coefficients, etc. during calibration with anipose
                dest_name = cropped_vid_name(full_vid_path, dest_folder, view_name, current_crop_params, fliplr=fliplr)

                # if video was already cropped, skip it
                if os.path.exists(dest_name):
                    _, dest_fname = os.path.split(dest_name)
                    print(dest_fname + ' already exists, skipping')
                    continue
                else:
                    crop_video(full_vid_path, dest_name, current_crop_params, view_name, filtertype=filtertype, fliplr=fliplr)

    return cropped_video_directories


def write_video_frames(vid_name, img_type='.jpg', dest_folder=None):

    if img_type[0] != '.':
        img_type = '.' + img_type

    vid_path, vid_filename = os.path.split(vid_name)
    vid_filename, _ = os.path.splitext(vid_filename)

    if dest_folder is None:
        dest_folder = navigation_utilities.cal_frames_folder_from_cal_vids_name(vid_name)
        # dest_folder = os.path.join(vid_path, vid_filename + '_frames')
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    full_jpg_path = os.path.join(dest_folder, '{}_frame_%d{}'.format(vid_filename, img_type))

    command = (
        f"ffmpeg -i {vid_name} "
        f"-f image2 {full_jpg_path} "
    )
    subprocess.call(command, shell=True)


def cropped_vid_name(full_vid_path, dest_folder, view_name, crop_params, fliplr=False):
    """
    function to return the name to be used for the cropped video
    :param full_vid_path:
    :param dest_folder: path in which to put the new folder with the cropped videos
    :param view_name: "dir", "lm", or "rm"
    :param crop_params: 4-element list [left, right, top, bottom]
    :return: full_dest_name - name of output file. Is name of input file with "_cropped_left-top-width-height" appended
    """
    vid_root, vid_ext = os.path.splitext(full_vid_path)
    vid_path, vid_name = os.path.split(vid_root)
    # _, vid_folder_name = os.path.split(vid_path)
    crop_params = [int(cp) for cp in crop_params]
    crop_params_str = '-'.join(map(str, crop_params))
    # dest_folder_name = vid_folder_name + '_' + view_name

    # vid_folder_name should be of format 'RXXXX_YYYYMMDD...'
    # vid_name_split = vid_folder_name.split('_')
    # ratID = vid_name_split[0]
    # full_dest_path = os.path.join(dest_folder, ratID, vid_folder_name, dest_folder_name)

    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    dest_name = vid_name + '_' + view_name + '_' + crop_params_str
    if fliplr:
        dest_name = dest_name + '_fliplr'
    dest_name = dest_name + vid_ext

    full_dest_name = os.path.join(dest_folder, dest_name)

    return full_dest_name


def crop_all_calibration_videos(parent_directories,
                               calibration_metadata_df,
                               vidtype='.avi',
                               view_list=['dir', 'lm', 'rm'],
                               filtertype='h264',
                               rat_nums='all'):

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

    for ratID in ratIDs:

        if ratID not in expt_ratIDs:
            continue

        rat_metadata_df = calibration_metadata_df[ratID]
        num_sessions = len(rat_metadata_df)

        for i_session in range(num_sessions):
            try:
                session_row = rat_metadata_df.iloc[[i_session]]
            except:
                pass
            # calibrate the camera for this session
            # cam_cal_vid_name = session_row['cal_vid_name_camera'].values[0]
            #
            # cam_cal_pickle = navigation_utilities.create_cam_cal_pickle_name(cam_cal_vid_name, parent_directories)
            # if os.path.exists(cam_cal_pickle):
            #     cam_intrinsics = skilled_reaching_io.read_pickle(cam_cal_pickle)
            # else:
            #     cam_cal_pickle_folder, _ = os.path.split(cam_cal_pickle)
            #     if not os.path.exists(cam_cal_pickle_folder):
            #         os.makedirs(cam_cal_pickle_folder)
            #     full_cam_cal_vid_path = navigation_utilities.find_camera_calibration_video(cam_cal_vid_name,
            #                                                                                parent_directories)
            #     cam_board = skilled_reaching_calibration.camera_board_from_df(session_row)
            #
            #     cam_intrinsics = skilled_reaching_calibration.calibrate_single_camera(full_cam_cal_vid_path, cam_board)

            mirror_calib_vid_name = session_row['cal_vid_name_mirrors'].values[0]

            if mirror_calib_vid_name.lower() == 'none':
                # no calibration video for this session
                session_date = session_row['date'].values[0]
                session_num = session_row['session_num'].values[0]
                print('no calibration video for session {:d} on {}'.format(session_num, np.datetime_as_string(session_date, unit='D')))
                continue

            full_calib_vid_name = navigation_utilities.find_mirror_calibration_video(mirror_calib_vid_name,
                                                                                     parent_directories)
            if full_calib_vid_name is None:
                session_date = session_row['date'].values[0]
                session_num = session_row['session_num'].values[0]
                print('no calibration video for session {:d} on {}'.format(session_num, np.datetime_as_string(session_date, unit='D')))
                continue

            current_cropped_calibration_vids = skilled_reaching_calibration.crop_calibration_video(full_calib_vid_name,
                                                                                                   session_row,
                                                                                                   filtertype=filtertype)


def crop_video(vid_path_in, vid_path_out, crop_params, view_name, filtertype='mjpeg2jpeg', fliplr=False):
    '''

    :param vid_path_in:
    :param vid_path_out:
    :param crop_params:
    :param view_name:
    :param filtertype:
    :return:
    '''
    # crop videos losslessly. Note that the trick of converting the video into a series of jpegs, cropping them, and
    # re-encoding is a trick that only works because our videos are encoded as mjpegs (which apparently is an old format)

    x1, x2, y1, y2 = [int(cp) for cp in crop_params]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    vid_root, vid_name = os.path.split(vid_path_out)

    print('cropping {}'.format(vid_name))

    if filtertype == 'mjpeg2jpeg':
        jpg_temp_folder = os.path.join(vid_root, 'temp')

        # if path already exists, delete the old temp folder. Either way, make a new one.
        if os.path.isdir(jpg_temp_folder):
            shutil.rmtree(jpg_temp_folder)
        os.mkdir(jpg_temp_folder)

        full_jpg_path = os.path.join(jpg_temp_folder, 'frame_%d.jpg')
        # full_jpg_crop_path = os.path.join(jpg_temp_folder, 'frame_crop_%d.jpg')
        command = (
            f"ffmpeg -i {vid_path_in} "
            f"-c:v copy -bsf:v mjpeg2jpeg {full_jpg_path} "
        )
        subprocess.call(command, shell=True)

        # find the list of jpg frames that were just made, crop them, and resave them
        jpg_list = glob.glob(os.path.join(jpg_temp_folder, '*.jpg'))
        for jpg_name in tqdm(jpg_list):
            img = cv2.imread(jpg_name)
            cropped_img = img[y1-1:y2-1, x1-1:x2-1, :]
            # if view_name == 'rm' or fliplr==True:
            if fliplr == True:
                # flip the image left to right so it can be run through a single "side mirror" DLC network
                # or, if this is a calibration video, both mirror views should be flipped
                cropped_img = cv2.flip(cropped_img, 1)   # 2nd argument flipCode > 0 indicates flip horizontally
            cv2.imwrite(jpg_name, cropped_img)

        # turn the cropped jpegs into a new movie
        command = (
            f"ffmpeg -i {full_jpg_path} "
            f"-c:v copy {vid_path_out}"
        )
        subprocess.call(command, shell=True)

        # destroy the temp jpeg folder
        shutil.rmtree(jpg_temp_folder)
    elif filtertype == 'h264':
        # if view_name == 'rm' or fliplr==True:
        if fliplr == True:
            # flip the image left to right so it can be run through a single "side mirror" DLC network
            # or, if this is a calibration video, both mirror views should be flipped
            command = (
                f"ffmpeg -n -i {vid_path_in} "
                f'-filter:v "crop={w}:{h}:{x1}:{y1}, hflip" '
                f"-c:v h264 -c:a copy {vid_path_out}"
            )
            subprocess.call(command, shell=True)
            # command = (
            #     f"ffmpeg -n -i {vid_path_out} "
            #     f"-vf hflip "
            #     f"-c:v h264 -c:a copy {vid_path_out}"
            # )
            # subprocess.call(command, shell=True)
        else:
            command = (
                f"ffmpeg -n -i {vid_path_in} "
                f"-filter:v crop={w}:{h}:{x1}:{y1} "
                f"-c:v h264 -c:a copy {vid_path_out}"
            )
            subprocess.call(command, shell=True)


def preprocess_videos(vid_folder_list, cropped_vids_parent, crop_params, view_list, vidtype='avi', filtertype='mjpeg2jpeg'):

    cropped_video_directories = crop_folders(vid_folder_list, cropped_vids_parent, crop_params, view_list, vidtype='avi', filtertype=filtertype)

    return cropped_video_directories