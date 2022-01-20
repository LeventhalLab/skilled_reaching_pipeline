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
