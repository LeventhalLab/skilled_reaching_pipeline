import glob
# from moviepy.editor import *
import subprocess
import cv2
import shutil
import pandas as pd
from datetime import datetime
import skilled_reaching_calibration
import navigation_utilities

def crop_optitrack_video(vid_path_in, vid_path_out, crop_params, filtertype='mjpeg2jpeg'):

    x1, x2, y1, y2 = [int(cp) for cp in crop_params]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    vid_root, vid_name = os.path.split(vid_path_out)

    vid_metadata = navigation_utilities.parse_Burgess_vid_name(vid_path_in)
    cam_num = vid_metadata['cam_num']
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
        for jpg_name in jpg_list:
            # should we crop and then rotate or rotate and then crop?
            img = cv2.imread(jpg_name)
            cropped_img = img[y1-1:y2-1, x1-1:x2-1, :]
            if cam_num == 1:
                # rotate the image 180 degrees because camera is rotated in real life
                cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_180)
            cv2.imwrite(jpg_name, cropped_img)

        # turn the cropped jpegs into a new movie
        command = (
            f"ffmpeg -i {full_jpg_path} "
            f"-c:v copy {vid_path_out}"
        )
        subprocess.call(command, shell=True)

        # destroy the temp jpeg folder
        shutil.rmtree(jpg_temp_folder)

    elif filtertype == '':
        #todo: need to add contingency for rotating 180 degrees if needed
        command = (
            f"ffmpeg -n -i {vid_path_in} "
            f"-filter:v crop={w}:{h}:{x1}:{y1} "
            f"-c:v h264 -c:a copy {vid_path_out}"
        )

        subprocess.call(command, shell=True)


def crop_Burgess_folders(video_folder_list, cropped_vid_parent, crop_params, cam_list, vidtype='avi', filtertype='mjpeg2jpeg'):
    """
    given the list of folders containing raw videos, loop through each of them, and crop all videos based on crop_params
    store the cropped videos in the cropped_vid_parent directory with appropriate file directory structure. Currently,
    assumes camera 1 is rotated 180 degrees. It performs cropping, then rotates 180 degrees

    :param video_folder_list: list of folders in which uncropped (raw) videos can be found
    :param cropped_vid_parent: parent directory for cropped videos. Has structure:
        cropped_vids_parent-->mouseID-->mouseID_YYYYmm-->mouseID_YYYYmmdd-->mouseID_YYYYmmdd_camXX (XX = 01 or 02)
    :param crop_params: either a dictionary with keys 'direct', 'leftmirror', 'rightmirror', each with a 4-element list [left, right, top, bottom]
            OR a pandas dataframe with columns 'date', 'box_num', 'direct_left', 'direct_right',...
    :param cam_list: tuple/list of integers containing camera numbers
    :param vidtype: string containing video name extension - 'avi', 'mpg', 'mp4', etc
    :param filtertype:
    :return:
    """

    box_num = 1    # for now, only one mouse skilled reaching box
    cropped_video_directories = navigation_utilities.create_Burgess_cropped_video_destination_list(cropped_vid_parent, video_folder_list, cam_list)
    # create_Burgess_cropped_video_destination_list returns a list of num_cam-element lists, where each list contains
    # folders in which to store cropped videos
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
            vid_metadata = navigation_utilities.parse_Burgess_vid_name(test_vid)
            session_date = vid_metadata['time'].date()
            crop_params_dict = crop_params_optitrack_dict_from_df(crop_params, session_date, box_num, cam_list)

            cam_names = crop_params_dict.keys()
        elif isinstance(crop_params, dict):
            crop_params_dict = crop_params

        if not bool(crop_params_dict):
            # the crop parameters dictionary is empty, skip to the next folder
            continue

        for full_vid_path in vids_list:
            vid_metadata = navigation_utilities.parse_Burgess_vid_name(full_vid_path)
            cam_num = vid_metadata['cam_num']
            cam_name = 'cam{:02d}'.format(cam_num)
            current_crop_params = crop_params_dict[cam_name]
            dest_folder = cropped_video_directories[cam_name][i_path]
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)

            dest_name = optirack_cropped_vid_name(full_vid_path, dest_folder, current_crop_params)

            # if video was already cropped, skip it
            if os.path.exists(dest_name):
                print(dest_name + ' already exists, skipping')
                continue
            else:
                crop_optitrack_video(full_vid_path, dest_name, current_crop_params, filtertype=filtertype)

    return cropped_video_directories


# def crop_optitrack_video(full_vid_path, cropped_vid_dir, crop_params, filtertype='mjpeg2jpeg'):
#
#     x1, x2, y1, y2 = [cp for cp in crop_params]
#
#     trial_metadata = parse_optitrack_name(full_vid_path)
#     trial_metadata['crop_params'] = crop_params   # crop_params is [left, right, top, bottom]
#
#     vid_root, vid_name = os.path.split(full_vid_path)
#     vid_root_name, vid_ext = os.path.splitext(vid_name)
#     cropped_vid_name = vid_root_name + '_cropped' + vid_ext
#     full_cropped_vid_name = os.path.join(cropped_vid_dir, cropped_vid_name)
#     if os.path.exists(full_cropped_vid_name):
#         # this video has already been cropped, don't do it again
#         return
#
#     # make a new directory to store the cropped videos in
#     # cropped_vid_dir = create_cropped_vid_directory(full_vid_path)
#     # revised - create the cropped_vid_dir in the calling function so it only has to be called once
#
#     # first, let's see if we can extract individual frames using ffmpeg
#     jpg_temp_folder = os.path.join(cropped_vid_dir, 'temp')
#
#     # if path already exists, delete the old temp folder. Either way, make a new one.
#     if os.path.isdir(jpg_temp_folder):
#         shutil.rmtree(jpg_temp_folder)
#     os.mkdir(jpg_temp_folder)
#
#     full_jpg_path = os.path.join(jpg_temp_folder, 'frame_%04d.jpg')
#     # full_jpg_crop_path = os.path.join(jpg_temp_folder, 'frame_crop_%d.jpg')
#     command = (
#         f"ffmpeg -i {full_vid_path} "
#         f"-c:v copy -bsf:v mjpeg2jpeg {full_jpg_path} "
#     )
#     subprocess.call(command, shell=True)
#
#     # find the list of jpg frames that were just made, crop them, and resave them
#     jpg_list = glob.glob(os.path.join(jpg_temp_folder, '*.jpg'))
#     for jpg_name in jpg_list:
#         img = cv2.imread(jpg_name)
#         # cropped_img = img[y1 - 1:y2 - 1, x1 - 1:x2 - 1, :]
#         if trial_metadata['cam_location'] == 'left':
#             # rotate the image 180 degrees
#             img = cv2.rotate(img, cv2.ROTATE_180)
#
#         cropped_img = img[y1 - 1:y2 - 1, x1 - 1:x2 - 1, :]
#
#         # cv2.imshow('testwin', cropped_img)
#         # cv2.waitKey(0)
#
#         cv2.imwrite(jpg_name, cropped_img)
#
#     # now resave the cropped frames into a full video
#     command = (
#         f"ffmpeg -i {full_jpg_path} "
#         f"-c:v copy {full_cropped_vid_name}"
#     )
#     subprocess.call(command, shell=True)
#
#     # destroy the temp jpeg folder
#     shutil.rmtree(jpg_temp_folder)
#
#     elif filtertype == '':
#         command = (
#             f"ffmpeg -n -i {vid_path_in} "
#             f"-filter:v crop={w}:{h}:{x1}:{y1} "
#             f"-c:v h264 -c:a copy {vid_path_out}"
#         )
#         subprocess.call(command, shell=True)


def crop_params_optitrack_dict_from_df(crop_params_df, session_date, box_num, cam_list):

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
        dict_keys = ['cam{:02d}'.format(i_cam) for i_cam in cam_list]
        crop_params_dict = dict.fromkeys(dict_keys, None)
        for camID in dict_keys:
            left_edge = date_box_row[camID + '_left'].values[0]
            right_edge = date_box_row[camID + '_right'].values[0]
            top_edge = date_box_row[camID + '_top'].values[0]
            bot_edge = date_box_row[camID + '_bottom'].values[0]

            if any([pd.isna(left_edge), pd.isna(right_edge), pd.isna(top_edge),  pd.isna(bot_edge)]):
                crop_params_dict = {}
                break
            else:
                crop_params_dict[camID] = [left_edge,
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


def optirack_cropped_vid_name(full_vid_path, dest_folder, crop_params):
    """
    function to return the name to be used for the cropped video
    :param full_vid_path:
    :param dest_folder: path in which to put the new folder with the cropped videos
    :param crop_params: 4-element list [left, right, top, bottom]
    :return: full_dest_name - name of output file. Is name of input file with "_cropped_left-top-width-height" appended
    """
    vid_metadata = navigation_utilities.parse_Burgess_vid_name(full_vid_path)

    vid_root, vid_ext = os.path.splitext(full_vid_path)
    vid_path, vid_name = os.path.split(vid_root)
    crop_params = [int(cp) for cp in crop_params]
    crop_params_str = '-'.join(map(str, crop_params))


    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    if vid_metadata['cam_num'] == 1:
        # this video will be cropped AND rotated (cropping happens first)
        dest_name = vid_name + '_' + crop_params_str + '_rotated' + vid_ext
    else:
        # this video will just be cropped
        dest_name = vid_name + '_' + crop_params_str + '_cropped' + vid_ext

    full_dest_name = os.path.join(dest_folder, dest_name)

    return full_dest_name


def preprocess_Burgess_videos(vid_folder_list, parent_directories, crop_params, cam_list, vidtype='avi'):
    '''

    :param vid_folder_list:
    :param cropped_vids_parent:
    :param crop_params:
    :param cam_list: tuple/list of integers containing camera numbers
    :param vidtype:
    :return: cropped_video_directories:
    '''
    cropped_vids_parent = parent_directories['cropped_vids_parent']
    cropped_video_directories = crop_Burgess_folders(vid_folder_list, cropped_vids_parent, crop_params, cam_list,
                                                 vidtype='avi')

    return cropped_video_directories