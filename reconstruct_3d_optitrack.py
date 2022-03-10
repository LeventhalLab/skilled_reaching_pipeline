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
    # meta_pickles = []
    for view_dir in view_directories:
        full_pickles.append(glob.glob(os.path.join(view_dir, '*full.pickle')))
        # meta_pickles.append(glob.glob(os.path.join(view_dir, '*meta.pickle')))

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
        dlc_metadata = []
        dlc_output.append(skilled_reaching_io.read_pickle(cam01_file))
        dlc_output.append(skilled_reaching_io.read_pickle(cam02_file))

        cam01_meta = cam01_file.replace('full.pickle', 'meta.pickle')
        cam02_meta = cam02_file.replace('full.pickle', 'meta.pickle')

        dlc_metadata.append(skilled_reaching_io.read_pickle(cam01_meta))
        dlc_metadata.append(skilled_reaching_io.read_pickle(cam02_meta))

        bodyparts = []
        for dlc_md in dlc_metadata:
            bodyparts.append(dlc_md['data']['DLC-model-config file']['all_joints_names'])
        rotate_translate_optitrack_points(dlc_output, pickle_metadata)
        pass
    pass


def rotate_pts_180(pts, im_size):
    '''

    :param pts:
    :param im_size: 1 x 2 list (height, width) (or should it be width, height?)
    :return:
    '''

    # reflect points around the center

    if not isinstance(pts, np.array):
        pts = np.array(pts)

    if not isinstance(im_size, np.array):
        im_size = np.array(im_size)

    reflected_pts = im_size - pts

    return reflected_pts


def rotate_translate_optitrack_points(dlc_output, pickle_metadata):

    # note that current algorithm for camera 1 crops, then rotates. We want a rotated, but uncropped transformation of coordinates
    # camera 2 is easy - just crops
    for i_cam, cam_output in enumerate(dlc_output):
        # cam_output is a dictionary where each entry is 'frame0000', 'frame0001', etc.
        # each frame has keys: 'coordinates', 'confidence', and 'costs'
        cam_metadata = pickle_metadata[i_cam]

        # loop through the frames
        frame_list = cam_output.keys()
        for i_frame, frame in enumerate(frame_list):
            current_coords = cam_output[frame]['coordinates'][0]
            # current_coords is a list of arrays containing data points as (x,y) pairs
            overlay_pts(pickle_metadata[i_cam], current_coords, i_frame)
            # if this image was rotated 180 degrees, first reflect back across the midpoint of the current image
            if cam_metadata['isrotated'] == True:
                # rotate points around the center of the cropped image, then translate into position in the original
                # image, then rotate around the center of the original image
                crop_win = cam_metadata['crop_window']
                crop_win_size = np.array([crop_win[1] - crop_win[0], crop_win[3] - crop_win[2]])

                # rotated_win_coords = np.array([])
                # for
                pass
            pass




    pass


def overlay_pts(pickle_metadata, current_coords, i_frame):

    videos_parent = '/home/levlab/Public/mouse_SR_videos_to_analyze'
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_mouse_SR_videos')

    month_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m')
    day_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m%d')
    cam_dir = day_dir  + '_' + 'cam{:02d}'.format(pickle_metadata['cam_num'])
    cropped_vid_folder = os.path.join(cropped_videos_parent, pickle_metadata['mouseID'], month_dir, day_dir, cam_dir)

    cropped_vid_name = '_'.join([pickle_metadata['mouseID'],
                                pickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                                '{:d}'.format(pickle_metadata['session_num']),
                                '{:03d}'.format(pickle_metadata['video_number']),
                                'cam{:02d}'.format(pickle_metadata['cam_num']),
                                '-'.join(str(x) for x in pickle_metadata['crop_window'])])
    if pickle_metadata['isrotated']:
        cropped_vid_name = cropped_vid_name + '_rotated'
    jpg_name = cropped_vid_name + '_{:04d}'.format(i_frame) + '.jpg'
    cropped_vid_name = cropped_vid_name + '.avi'
    cropped_vid_name = os.path.join(cropped_vid_folder, cropped_vid_name)
    jpg_name = os.path.join(cropped_vid_folder, jpg_name)



    video_object = cv2.VideoCapture(cropped_vid_name)

    video_object.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    ret, cur_img = video_object.read()

    # overlay points

    cv2.imwrite(jpg_name, cur_img)
    pass