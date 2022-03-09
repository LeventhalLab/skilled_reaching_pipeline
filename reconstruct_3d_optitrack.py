import numpy as np
import cv2
import os
import glob
import navigation_utilities
import skilled_reaching_io
import pandas as pd
import scipy.io as sio


def reconstruct_optitrack_session(view_directories):

    # find all the files containing labeled points in view_directories
    full_pickles = []
    # metadata_pickles = []
    for view_dir in view_directories:
        full_pickles.append(glob.glob(os.path.join(view_dir, '*full.pickle')))

    for cam01_file in full_pickles[0]:
        pickle_metadata = []
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam01_file))

        # find corresponding pickle file for camera 2
        _, cam01_pickle_name = os.path.split(cam01_file)
        cam01_pickle_stem = cam01_pickle_name[:cam01_pickle_name.find('cam01') + 5]
        cam02_pickle_stem = cam01_pickle_stem.replace('cam01', 'cam02')

        cam02_file_list = glob.glob(os.path.join(view_directories[1], cam02_pickle_stem + '*full.pickle'))
        if len(cam02_file_list) == 1:
            cam02_file = cam02_file_list[0]
        else:
            print('no matching camera 2 file for {}'.format(cam01_file))
            continue
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam02_file))

        # read in the points
        dlc_output = []
        dlc_output.append(skilled_reaching_io.read_pickle(cam01_file))
        dlc_output.append(skilled_reaching_io.read_pickle(cam02_file))

        rotate_translate_optitrack_points(dlc_output, pickle_metadata)
        pass
    pass


def rotate_translate_optitrack_points(dlc_output, pickle_metadata):

    # note that current algorithm for camera 1 crops, then rotates. We want a rotated, but uncropped transformation of coordinates
    # camera 2 is easy - just crops
    for i_cam, cam_output in enumerate(dlc_output):
        # cam_output is a dictionary where each entry is 'frame0000', 'frame0001', etc.
        # each frame has keys: 'coordinates', 'confidence', and 'costs'
        cam_metadata = pickle_metadata[i_cam]

        # loop through the frames
        frame_list = cam_output.keys()
        for frame in frame_list:
            current_coords = cam_output[frame]['coordinates']

            # if this image was rotated 180 degrees, first reflect back across the midpoint of the current image
            if cam_metadata['isrotated'] == True:
                # rotate points around the center of the cropped image, then translate into position in the original
                # image, then rotate around the center of the original image
                pass
            pass




    pass