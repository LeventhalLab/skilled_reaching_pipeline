import numpy as np
import cv2
import navigation_utilities
import skilled_reaching_io

def triangulate_video(video_name, marked_videos_parent, calibration_parent, view_list=('direct', 'leftmirror', 'rightmirror')):

    video_metadata = navigation_utilities.parse_video_name(video_name)
    dlc_output_pickle_names, dlc_metadata_pickle_names = navigation_utilities.find_dlc_output_pickles(video_metadata, marked_videos_parent, view_list=view_list)

    # find the calibration files

    # read in the pickle files
    dlc_output = {view: None for view in view_list}
    dlc_metadata = {view: None for view in view_list}
    for view in view_list:
        dlc_output[view] = skilled_reaching_io.read_pickle(dlc_output_pickle_names[view])
        dlc_metadata[view] = skilled_reaching_io.read_pickle(dlc_metadata_pickle_names[view])

    pass


def translate_points_to_full_frame():

    pass


def bodyparts_from_metadata(dlc_metadata):

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']