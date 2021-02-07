import glob
from moviepy.editor import *
import subprocess
import cv2
import shutil
import skilled_reaching_calibration
import navigation_utilities


def crop_folders(video_folder_list, cropped_vids_parent, crop_params_dict, view_list, vidtype='avi', filtertype='mjpeg2jpeg'):
    """
    :param video_folder_list:
    :param cropped_vids_parent:
    :param crop_params_dict: 4-element list [left, right, top, bottom]
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

        for i_view, view_name in enumerate(view_list):
            dest_folder = cropped_video_directories[i_view][i_path]
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)

            for full_vid_path in vids_list:
                crop_params = crop_params_dict[view_name]
                dest_name = cropped_vid_name(full_vid_path, dest_folder, view_name, crop_params)

                # if video was already cropped, skip it
                if os.path.exists(dest_name):
                    print(dest_name + ' already exists, skipping')
                    continue
                else:
                    crop_video(full_vid_path, dest_name, crop_params, view_name, filtertype=filtertype)

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

    x1, x2, y1, y2 = [cp for cp in crop_params]
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


def preprocess_videos(vid_folder_list, cropped_vids_parent, crop_params_dict, view_list, vidtype='avi'):

    cropped_video_directories = crop_folders(vid_folder_list, cropped_vids_parent, crop_params_dict, view_list, vidtype='avi')

    return cropped_video_directories