import glob
from moviepy.editor import *
import subprocess
import cv2
import shutil
import pandas as pd
from datetime import datetime
import skilled_reaching_calibration
import navigation_utilities


def crop_params_dict_from_df(crop_params_df, session_date, box_num, view_list=['direct', 'leftmirror', 'rightmirror']):


    # find the row with the relevant session data and box number
    date_box_row = crop_params_df[(crop_params_df['date']==session_date) & (crop_params_df['box_num']==box_num)]

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
    :param crop_params: either a dictionary with keys 'direct', 'leftmirror', 'rightmirror', each with a 4-element list [left, right, top, bottom]
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


def cropped_vid_name(full_vid_path, dest_folder, view_name, crop_params):
    """
    function to return the name to be used for the cropped video
    :param full_vid_path:
    :param dest_folder: path in which to put the new folder with the cropped videos
    :param view_name: "direct", "leftmirror", or "rightmirror"
    :param crop_params: 4-element list [left, right, top, bottom]
    :return: full_dest_name - name of output file. Is name of input file with "_cropped_left-top-width-height" appended
    """
    vid_root, vid_ext = os.path.splitext(full_vid_path)
    vid_path, vid_name = os.path.split(vid_root)
    # _, vid_folder_name = os.path.split(vid_path)
    crop_params_str = '-'.join(map(str, crop_params))
    # dest_folder_name = vid_folder_name + '_' + view_name

    # vid_folder_name should be of format 'RXXXX_YYYYMMDD...'
    # vid_name_split = vid_folder_name.split('_')
    # ratID = vid_name_split[0]
    # full_dest_path = os.path.join(dest_folder, ratID, vid_folder_name, dest_folder_name)

    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    dest_name = vid_name + '_' + view_name + '_' + crop_params_str + vid_ext

    full_dest_name = os.path.join(dest_folder, dest_name)

    return full_dest_name


def crop_video(vid_path_in, vid_path_out, crop_params, view_name, filtertype='mjpeg2jpeg'):

    # crop videos losslessly. Note that the trick of converting the video into a series of jpegs, cropping them, and
    # re-encoding is a trick that only works because our videos are encoded as mjpegs (which apparently is an old format)

    x1, x2, y1, y2 = [int(cp) for cp in crop_params]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    vid_root, vid_name = os.path.split(vid_path_out)

    #todo: speed this up by cropping all 3 views from the same jpegs?
    if filtertype == 'mjpeg2jpeg':
        jpg_temp_folder = os.path.join(vid_root, 'temp')

        # if path already exists, delete the old temp folder. Either way, make a new one.
        if os.path.isdir(jpg_temp_folder):
            shutil.rmtree(jpg_temp_folder)
        os.mkdir(jpg_temp_folder)

        full_jpg_path = os.path.join(jpg_temp_folder,'frame_%d.jpg')
        # full_jpg_crop_path = os.path.join(jpg_temp_folder, 'frame_crop_%d.jpg')
        command = (
            f"ffmpeg -i {vid_path_in} "
            f"-c:v copy -bsf:v mjpeg2jpeg {full_jpg_path} "
        )
        subprocess.call(command, shell=True)

        # find the list of jpg frames that were just made, crop them, and resave them
        jpg_list = glob.glob(os.path.join(jpg_temp_folder, '*.jpg'))
        for jpg_name in jpg_list:
            img = cv2.imread(jpg_name)
            cropped_img = img[y1-1:y2-1, x1-1:x2-1, :]
            if view_name == 'rightmirror':
                # flip the image left to right so it can be run through a single "side mirror" DLC network
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
    elif filtertype == '':
        command = (
            f"ffmpeg -n -i {vid_path_in} "
            f"-filter:v crop={w}:{h}:{x1}:{y1} "
            f"-c:v h264 -c:a copy {vid_path_out}"
        )
        subprocess.call(command, shell=True)
        pass


def preprocess_videos(vid_folder_list, cropped_vids_parent, crop_params, view_list, vidtype='avi'):

    cropped_video_directories = crop_folders(vid_folder_list, cropped_vids_parent, crop_params, view_list, vidtype='avi')

    return cropped_video_directories