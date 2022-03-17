import numpy as np
import cv2
import os
import glob
import navigation_utilities
import skilled_reaching_io
import pandas as pd
import scipy.io as sio


def reconstruct_optitrack_session(view_directories, calibration_parent):

    # find all the files containing labeled points in view_directories
    full_pickles = []
    # meta_pickles = []
    for view_dir in view_directories:
        full_pickles.append(glob.glob(os.path.join(view_dir, '*full.pickle')))
        # meta_pickles.append(glob.glob(os.path.join(view_dir, '*meta.pickle')))

    for cam01_file in full_pickles[0]:
        pickle_metadata = []
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam01_file))

        calibration_file = navigation_utilities.create_multiview_calibration_data_name(calibration_parent, pickle_metadata[0]['trialtime'])
        if not os.path.exists(calibration_file):
            # if there is no calibration file for this session, skip
            continue

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
        pts_wrt_orig_img, dlc_conf = rotate_translate_optitrack_points(dlc_output, pickle_metadata, dlc_metadata)

        # now have all the identified points moved back into the original coordinate systems that the checkerboards were
        # identified in, and confidence levels. pts_wrt_orig_img is an array (num_frames x num_joints x 2) and dlc_conf
        # is an array (num_frames x num_joints). Zeros are stored where dlc was uncertain (no result for that joint on
        # that frame)

        reconstruct3d_single_optitrack_video(calibration_file, pts_wrt_orig_img, dlc_conf)
    pass


def reconstruct3d_single_optitrack_video(calibration_file, pts_wrt_orig_img, dlc_conf):

    # read in the calibration file, make sure we have stereo and camera calibrations
    cal_data = skilled_reaching_io.read_pickle(calibration_file)
    '''
    cal_data is a dictionary containing:
        cam_objpoints
        cam_imgpoints
        stereo_objpoints
        stereo_imgpoints
        stereo_frames
        valid_frames
        im_size
        cb_size
        calvid_metadata
        mtx - 2 x 3 x 3 array with camera intrinsic matrices - mtx[0,:,:] is for camera 1, mtx[1,:,:] is for camera 2
        dist - distortion coefficients; 2 x 5 array, first row is for camera 1, 2nd row for camera 2
        frame_nums_for_intrinsics
        R
        T 
        E  - essential matrix
        F - fundamental matrix
        frames_for_stereo_calibration
    '''
    num_frames = np.shape(pts_wrt_orig_img[0])[0]
    pass


def triangulate_single_point(cal_data, cam01_pt, cam02_pt):

    pass

def rotate_pts_180(pts, im_size):
    '''

    :param pts:
    :param im_size: 1 x 2 list (height, width) (or should it be width, height?)
    :return:
    '''

    # reflect points around the center

    # if not isinstance(pts, np.array):
    #     pts = np.array(pts)
    #
    # if not isinstance(im_size, np.array):
    #     im_size = np.array(im_size)

    reflected_pts = []
    for i_pt, pt in enumerate(pts):
        if len(pt) > 0:
            try:
                x, y = pt[0]
            except:
                pass
            # possible that im_size is width x height or height x width
            try:
                new_x = im_size[0] - x
                new_y = im_size[1] - y
            except:
                pass

            reflected_pts.append([np.array([new_x, new_y])])
        else:
            reflected_pts.append(np.array([]))

    return reflected_pts


def rotate_translate_optitrack_points(dlc_output, pickle_metadata, dlc_metadata, orig_im_size=(1280, 1024)):

    # note that current algorithm for camera 1 crops, then rotates. We want a rotated, but uncropped transformation of coordinates
    # camera 2 is easy - just crops
    pts_wrt_orig_img = []
    dlc_conf = []
    for i_cam, cam_output in enumerate(dlc_output):
        # cam_output is a dictionary where each entry is 'frame0000', 'frame0001', etc.
        # each frame has keys: 'coordinates', 'confidence', and 'costs'
        cam_metadata = pickle_metadata[i_cam]

        # loop through the frames
        frame_list = cam_output.keys()
        num_frames = cam_output['metadata']['nframes']
        num_joints = len(cam_output['metadata']['all_joints_names'])
        pts_wrt_orig_img.append(np.zeros((num_frames, num_joints, 2)))
        dlc_conf.append(np.zeros((num_frames, num_joints)))
        #todo: write the new points/confidences into these arrays so they can be returned by this function

        for i_frame, frame in enumerate(frame_list):
            if frame[:5] != 'frame':
                continue
            current_coords = cam_output[frame]['coordinates'][0]

            # current_coords is a list of arrays containing data points as (x,y) pairs
            # overlay_pts(pickle_metadata[i_cam], current_coords, dlc_metadata[i_cam], i_frame)
            # if this image was rotated 180 degrees, first reflect back across the midpoint of the current image
            if cam_metadata['isrotated'] == True:
                # rotate points around the center of the cropped image, then translate into position in the original
                # image, then rotate around the center of the original image
                crop_win = cam_metadata['crop_window']

                crop_win_size = np.array([crop_win[1] - crop_win[0], crop_win[3] - crop_win[2]])
                reflected_pts = rotate_pts_180(current_coords, crop_win_size)

                # now have the points back in the upside-down version. Now need to rotate the points within the full image
                # to get into the same reference frame as the calibration image
                full_im_size = dlc_metadata[i_cam]['data']['frame_dimensions']
                # full_im_size = (full_im_size[1],full_im_size[0])

                pts_translated_to_orig = translate_back_to_orig_img(pickle_metadata[i_cam], reflected_pts)

                # overlay_pts(pickle_metadata[i_cam], reflected_pts, dlc_metadata[i_cam], i_frame, rotate_img=True)
                # overlay_pts_in_orig_image(pickle_metadata[i_cam], pts_translated_to_orig, dlc_metadata[i_cam], i_frame, rotate_img=False)

                pts_in_calibration_coords = rotate_pts_180(pts_translated_to_orig, orig_im_size)
                # overlay_pts_in_orig_image(pickle_metadata[i_cam], pts_in_calibration_coords, dlc_metadata[i_cam], i_frame,
                #                           rotate_img=True)
            else:
                pts_in_calibration_coords = translate_back_to_orig_img(pickle_metadata[i_cam], current_coords)
                overlay_pts_in_orig_image(pickle_metadata[i_cam], pts_in_calibration_coords, dlc_metadata[i_cam], i_frame,
                                          rotate_img=False)
                #todo: align all the points for the two camera views/frames and store them in a way that can be neatly
                # exported to another function for 3D reconstuction. should also write a function to organize pickled data
                # into a more reasonable format so if/when start using .h5 files, can write another function to organize
                # those
            array_pts = convert_pts_to_array(pts_in_calibration_coords)
            pts_wrt_orig_img[i_cam][i_frame] = array_pts

            #todo: store and return the confidence array
            conf = dlc_output[i_cam][frame]['confidence']
            array_conf = convert_pickle_conf_to_array(conf)

            dlc_conf[i_cam][i_frame, :] = array_conf

    return pts_wrt_orig_img, dlc_conf


def convert_pts_to_array(pickle_format_pts):

    num_joints = len(pickle_format_pts)
    array_pts = np.zeros([num_joints, 2])
    for i_pt, cur_pt in enumerate(pickle_format_pts):
        if len(cur_pt) == 0:
            continue
        else:
            array_pts[i_pt, :] = cur_pt[0]

    return array_pts


def convert_pickle_conf_to_array(pickle_confidence):

    num_joints = len(pickle_confidence)
    array_conf = np.zeros(num_joints)
    for i_conf, cur_conf in enumerate(pickle_confidence):
        if len(cur_conf) == 0:
            continue
        else:
            array_conf[i_conf] = cur_conf[0]

    return array_conf

def translate_back_to_orig_img(pickle_metadata, pts):
    crop_win = pickle_metadata['crop_window']

    translated_pts = []
    for i_pt, pt in enumerate(pts):
        if len(pt) > 0:
            try:
                x, y = pt[0]
            except:
                pass
            # possible that im_size is width x height or height x width
            new_x = crop_win[0] + x
            new_y = crop_win[2] + y

            translated_pts.append([np.array([new_x, new_y])])
        else:
            translated_pts.append(np.array([]))

    return translated_pts


def overlay_pts_in_orig_image(pickle_metadata, current_coords, dlc_metadata, i_frame, rotate_img=False):
    videos_parent = '/home/levlab/Public/mouse_SR_videos_to_analyze'
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_mouse_SR_videos')
    video_root_folder = os.path.join(videos_parent, 'mouse_SR_videos_tocrop')

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

    month_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m')
    day_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, pickle_metadata['mouseID'], month_dir, day_dir)

    cam_dir = day_dir + '_' + 'cam{:02d}'.format(pickle_metadata['cam_num'])
    cropped_vid_folder = os.path.join(cropped_videos_parent, pickle_metadata['mouseID'], month_dir, day_dir, cam_dir)

    orig_vid_name_base = '_'.join([pickle_metadata['mouseID'],
                              pickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                              '{:d}'.format(pickle_metadata['session_num']),
                              '{:03d}'.format(pickle_metadata['video_number']),
                              'cam{:02d}'.format(pickle_metadata['cam_num'])
                              ])
    orig_vid_name = os.path.join(orig_vid_folder, orig_vid_name_base + '.avi')

    video_object = cv2.VideoCapture(orig_vid_name)

    video_object.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    ret, cur_img = video_object.read()

    video_object.release()

    jpg_name = orig_vid_name_base + '_{:04d}'.format(i_frame)
    if rotate_img:
        cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)
        jpg_name = jpg_name + '_rotated'
    jpg_name = os.path.join(cropped_vid_folder, jpg_name + '.jpg')

    # overlay points
    new_img = cur_img
    for i_pt, pt in enumerate(current_coords):
        if len(pt) > 0:
            try:
                x, y = pt[0]
            except:
                pass
            x = int(round(x))
            y = int(round(y))
            bp_color = color_from_bodypart(bodyparts[i_pt])
            new_img = cv2.circle(new_img, (x, y), 3, bp_color, -1)

    cv2.imwrite(jpg_name, new_img)
    pass


def overlay_pts(pickle_metadata, current_coords, dlc_metadata, i_frame, rotate_img=False):

    videos_parent = '/home/levlab/Public/mouse_SR_videos_to_analyze'
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_mouse_SR_videos')

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

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
        cropped_vid_name_base = cropped_vid_name + '_rotated'
    else:
        cropped_vid_name_base = cropped_vid_name

    cropped_vid_name = cropped_vid_name_base + '.avi'
    cropped_vid_name = os.path.join(cropped_vid_folder, cropped_vid_name)

    video_object = cv2.VideoCapture(cropped_vid_name)

    video_object.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    ret, cur_img = video_object.read()

    video_object.release()

    jpg_name = cropped_vid_name_base + '_{:04d}'.format(i_frame)
    if rotate_img:
        cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)
        jpg_name = jpg_name + '_rotatedback.jpg'
    else:
        jpg_name = jpg_name + '.jpg'
    jpg_name = os.path.join(cropped_vid_folder, jpg_name)

    # overlay points
    new_img = cur_img
    for i_pt, pt in enumerate(current_coords):
        if len(pt) > 0:
            try:
                x, y = pt[0]
            except:
                pass
            x = int(round(x))
            y = int(round(y))
            bp_color = color_from_bodypart(bodyparts[i_pt])
            new_img = cv2.circle(new_img, (x, y), 3, bp_color, -1)

    cv2.imwrite(jpg_name, new_img)


def color_from_bodypart(bodypart):

    if bodypart == 'leftear':
        bp_color = (127,0,0)
    elif bodypart == 'rightear':
        bp_color = (255,0,0)
    elif bodypart == 'lefteye':
        bp_color = (150,150,150)
    elif bodypart == 'righteye':
        bp_color = (200,200,200)
    elif bodypart == 'nose':
        bp_color = (0,0,0)
    elif bodypart == 'leftpaw':
        bp_color = (0,50,0)
    elif bodypart == 'leftdigit1':
        bp_color = (0, 100, 0)
    elif bodypart == 'leftdigit2':
        bp_color = (0,150,0)
    elif bodypart == 'leftdigit3':
        bp_color = (0, 200, 0)
    elif bodypart == 'leftdigit4':
        bp_color = (0,250,0)
    elif bodypart == 'rightpaw':
        bp_color = (0,0,50)
    elif bodypart == 'rightdigit1':
        bp_color = (0, 0, 100)
    elif bodypart == 'rightdigit2':
        bp_color = (0,0,150)
    elif bodypart == 'rightdigit3':
        bp_color = (0, 0, 200)
    elif bodypart == 'rightdigit4':
        bp_color = (0,0,250)
    elif bodypart == 'pellet1':
        bp_color = (100,0,100)
    elif bodypart == 'pellet2':
        bp_color = (200,0,200)
    else:
        bp_color = (0,0,255)

    return bp_color