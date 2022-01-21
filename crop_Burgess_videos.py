import glob
from moviepy.editor import *
import subprocess
import cv2
import shutil
import pandas as pd
from datetime import datetime
import skilled_reaching_calibration
import navigation_utilities

def crop_Burgess_video(vid_path_in, vid_path_out, crop_params, filtertype='mjpeg2jpeg'):

    x1, x2, y1, y2 = [int(cp) for cp in crop_params]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    vid_root, vid_name = os.path.split(vid_path_out)

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
                # flip the image left to right so it can be run through a single "side mirror" DLC network
                cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_180)   # 2nd argument flipCode > 0 indicates flip horizontally
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
        command = (
            f"ffmpeg -n -i {vid_path_in} "
            f"-filter:v crop={w}:{h}:{x1}:{y1} "
            f"-c:v h264 -c:a copy {vid_path_out}"
        )
        subprocess.call(command, shell=True)


def crop_Burgess_folders(video_folder_list, cropped_vids_parent, crop_params, vidtype='avi'):
    """
    :param video_folder_list:
    :param cropped_vids_parent:
    :param crop_params: either a dictionary with keys 'direct', 'leftmirror', 'rightmirror', each with a 4-element list [left, right, top, bottom]
            OR a pandas dataframe with columns 'date', 'box_num', 'direct_left', 'direct_right',...
    :param vidtype:
    :return:
    """
    #todo: write the function below
    cropped_video_directories = navigation_utilities.create_Burgess_cropped_video_destination_list(cropped_vids_parent, video_folder_list, view_list)
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
            current_crop_params = crop_params_dict[view_name]
            dest_folder = cropped_video_directories[i_view][i_path]
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)

            for full_vid_path in vids_list:
                dest_name = cropped_vid_name(full_vid_path, dest_folder, view_name, current_crop_params)

                # if video was already cropped, skip it
                if os.path.exists(dest_name):
                    print(dest_name + ' already exists, skipping')
                    continue
                else:
                    crop_video(full_vid_path, dest_name, current_crop_params, view_name, filtertype=filtertype)

    return cropped_video_directories


def preprocess_Burgess_videos(vid_folder_list, cropped_vids_parent, crop_params, view_list, vidtype='avi'):

    cropped_video_directories = crop_Burgess_folders(vid_folder_list, cropped_vids_parent, crop_params, view_list,
                                                 vidtype='avi')

    return cropped_video_directories