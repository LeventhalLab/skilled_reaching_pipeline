import navigation_utilities
import skilled_reaching_io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import crop_videos
import os
import csv
import numpy as np
import cv2
import glob
from random import randint


def import_fiji_csv(fname):
    """
    read csv file with points marked in fiji
    :param fname:
    :return:
    """
    # for checkerboards marked manually in fiji PRIOR TO undistorting

    # if file is empty, return empty matrix
    if os.path.getsize(fname) == 0:
        return np.empty(0)

    # determine how many lines are in the .csv file
    with open(fname, newline='\n') as csv_file:
        num_lines = sum(1 for row in csv_file)

    with open(fname, newline='\n') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        cb_points = np.empty((num_lines-1, 2))
        for i_row, row in enumerate(csv_reader):
            if i_row == 0:
                # check to make sure the header was read in properly
                print(f'Column names are {", ".join(row)}')
            else:
                # read in the rows of checkerboard points
                cb_points[i_row-1, :] = row[-2:]

    return cb_points


def read_cube_calibration_points(calibration_folder, pts_per_board=12):
    """

    :param calibration_folder:
    :return:
    """

    calibration_files = glob.glob(os.path.join(calibration_folder, 'GridCalibration_*.csv'))
    # for each file, there should be 72 total points; 12 per checkerboard
    num_files = len(calibration_files)
    direct_points = np.empty(pts_per_board, 2, 2, num_files)  # num_pts x 2 dims (x,y) x 2 boards (left,right) x num_files
    mirror_points = np.empty(pts_per_board, 2, 2, num_files)  # num_pts x 2 dims (x,y) x 2 boards (left,right) x num_files
    for calib_file in calibration_files:
        cb_pts = import_fiji_csv(calib_file)

        # still working on this, but probably will never actually need it


def sort_points_to_boards(cb_pts):
    """

    :param cb_pts: n x 2 numpy array containing (distorted) checkerboard points from the calibration cubes
    :return:
    """
    #todo: can probably eliminate this - relic from when we were calibrating with cubes instead of videos of checkerboards in a single plane


def fundamental_matrix_from_mirrors(x1, x2):
    """
    function to compute the fundamental matrix for direct camera and mirror image views, taking advantage of the fact
    that F is skew-symmetric in this case. Note x1 and x2 should be undistorted by this point

    :param x1: n x 2 numpy array containing matched points in order from view 1
    :param x2: n x 2 numpy array containing matched points in order from view 2
    :return:
    """
    n1, numcols = x1.shape()
    if numcols != 2:
        print('x1 must have 2 columns')
        return
        #todo: error handling here
    n2, numcols = x2.shape()
    if numcols != 2:
        print('x2 must have 2 columns')
        return
    if n1 != n2:
        print('x1 and x2 must have same number of rows')
        return

    A = np.zeros((n1, 3))
    A[:, 0] = np.multiply(x2[:, 0], x1[:, 1]) - np.multiply(x1[:, 0], x2[:, 1])
    A[:, 1] = x2[:, 0] - x1[:, 0]
    A[:, 2] = x2[:,1] - x1[:, 1]

    # solve the linear system of equations A * [f12,f13,f23]' = 0
    # need to figure out if the matrix needs to be changed using opencv conventions instead of matlab
    _, _, vA = np.linalg.svd(A)
    F = np.zeros((3, 3))
    fvec = vA[:, -1]

    F[0, 1] = fvec[0]
    F[0, 2] = fvec[1]
    F[1, 2] = fvec[2]
    F[1, 0] = -F[0, 1]
    F[2, 0] = -F[0, 2]
    F[2, 1] = -F[1, 2]

    return F


def select_correct_essential_matrix():

    pass


def calibrate_camera_from_video(camera_calibration_vid_name, calibration_parent, cb_size=(6, 9)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    video_object = cv2.VideoCapture(camera_calibration_vid_name)

    im_size = (int(video_object.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbrow = cb_size[0]
    cbcol = cb_size[1]
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    frame_counter = 0
    while True:
        video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        ret, cur_img = video_object.read()

        if ret:
            frame_counter += 1

            cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            found_valid_chessboard, corners = cv2.findChessboardCorners(cur_img_gray, cb_size)

            if found_valid_chessboard:
                # refine the points, then save the checkerboard image and a metadata file to disk
                corners2 = cv2.cornerSubPix(cur_img_gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                objpoints.append(objp)
            # else:
            # this is for showing the corners that weren't properly identified if the plotting lines below are commented in
            #     corners2 = corners

            # below is for checking if corners were correctly identified
            # corners_img = cv2.drawChessboardCorners(cur_img, cb_size, corners2, found_valid_chessboard)
            # # cv2.imwrite(corners_img_name, corners_img)
            # cv2.imshow('image', corners_img)
            # cv2.waitKey(0)

        else:
            # finished with the last frame
            break

    video_object.release()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, im_size, None, None)

    stereo_params = {
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'im_size': im_size
    }

    calibration_metadata = navigation_utilities.parse_camera_calibration_video_name(camera_calibration_vid_name)

    calibration_name = navigation_utilities.create_calibration_filename(calibration_metadata, calibration_parent)

    skilled_reaching_io.write_pickle(calibration_name, stereo_params)


def multi_view_calibration(calibration_vids, cal_data_parent, cb_size=(10, 7)):

    cam_cal_flags = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST
    stereo_cal_flags = cv2.CALIB_FIX_INTRINSIC

    vid_name_parts = navigation_utilities.parse_Burgess_calibration_vid_name(calibration_vids[0])
    cal_data_name = navigation_utilities.create_multiview_calibration_data_name(cal_data_parent,
                                                                                vid_name_parts['session_datetime'])

    if not os.path.exists(cal_data_name):
        calibration_data = collect_cb_corners(calibration_vids, cb_size=cb_size)
        # save the imgpoints and objpoints along with metadata
        skilled_reaching_io.write_pickle(cal_data_name, calibration_data)
    else:
        calibration_data = skilled_reaching_io.read_pickle(cal_data_name)


    # if calibration already performed for individual cameras (at least one), skip initializing variables
    # or better yet, let's calibrate the two cameras once well, then just to the stereo calibration on subsequent days

    if 'mtx' not in calibration_data.keys():
        # first, calibrate each camera individually
        # initialize arrays to hold camera intrinsics, distortion coefficients, rotation, translation vectors
        mtx = [[] for ii in calibration_data['cam_objpoints']]
        dist = [[] for ii in calibration_data['cam_objpoints']]
        rvecs = [[] for ii in calibration_data['cam_objpoints']]
        tvecs = [[] for ii in calibration_data['cam_objpoints']]

        for i_cam, objpoints in enumerate(calibration_data['cam_objpoints']):
            # intrinsics haven't been calculated yet for either camera
            imgpoints = calibration_data['cam_imgpoints'][i_cam]
            im_size = calibration_data['im_size'][i_cam]

            print('calibrating camera {:02d}'.format(i_cam + 1))
            ret, mtx[i_cam], dist[i_cam], rvecs[i_cam], tvecs[i_cam] = cv2.calibrateCamera(objpoints[:10], imgpoints[:10], im_size, None, None,
                                                                                           flags=cam_cal_flags)
            calibration_data['mtx'] = mtx
            calibration_data['dist'] = dist
            calibration_data['rvecs'] = rvecs
            calibration_data['tvecs'] = tvecs

            skilled_reaching_io.write_pickle(cal_data_name, calibration_data)
    else:
        for i_cam, objpoints in enumerate(calibration_data['cam_objpoints']):

            # has mtx already been calculated for this camera? it's possible that an empty intrinsics was saved on a previous run
            if len(calibration_data['mtx'][i_cam]) == 0:
                # the intrinsic matrix has not yet been calculated for i_cam
                imgpoints = calibration_data['cam_imgpoints'][i_cam]
                im_size = calibration_data['im_size'][i_cam]

                print('calibrating camera {:02d}'.format(i_cam + 1))
                ret, mtx[i_cam], dist[i_cam], rvecs[i_cam], tvecs[i_cam] = cv2.calibrateCamera(objpoints[:10], imgpoints[:10], im_size, None, None,
                                                                                               flags=cam_cal_flags)
                calibration_data['mtx'] = mtx
                calibration_data['dist'] = dist
                calibration_data['rvecs'] = rvecs
                calibration_data['tvecs'] = tvecs

                skilled_reaching_io.write_pickle(cal_data_name, calibration_data)

        #todo: figure out the best way to compute the intrinsic matrices - probably constrain fx and fy to be equal, tangential distortion to be zero, constrain principal point to be at the center
        pass


    objpoints = calibration_data['stereo_objpoints']
    imgpoints = calibration_data['stereo_imgpoints']

    print('performing stereo calibration')
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints[0], imgpoints[1],
                                                                                                     calibration_data['mtx'][0], calibration_data['dist'][0],
                                                                                                     calibration_data['mtx'][1], calibration_data['dist'][1],
                                                                                                     im_size,
                                                                                                     flags=stereo_cal_flags)
    pass


def collect_cb_corners(calibration_vids, cb_size):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    video_object = []
    num_frames = []
    im_size = []
    cal_vid_metadata = []
    for i_vid, cal_vid in enumerate(calibration_vids):
        video_object.append(cv2.VideoCapture(cal_vid))
        num_frames.append(video_object[i_vid].get(cv2.CAP_PROP_FRAME_COUNT))
        im_size.append((int(video_object[i_vid].get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_object[i_vid].get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cal_vid_metadata.append(navigation_utilities.parse_cropped_calibration_video_name(cal_vid))

    num_frames = [int(nf) for nf in num_frames]
    if all(nf == num_frames[0] for nf in num_frames):
        # check that there are the same number of frames in each calibration video

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        cbrow = cb_size[0]
        cbcol = cb_size[1]
        objp = np.zeros((cbrow * cbcol, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

        cam_objpoints = [[] for ii in calibration_vids]
        stereo_objpoints = []
        cam_imgpoints = [[] for ii in calibration_vids]
        stereo_imgpoints = [[] for ii in calibration_vids]

        valid_frames = [[False for frame_num in range(num_frames[0])] for ii in calibration_vids]

        for i_frame in range(num_frames[0]):
            print(i_frame)

            corners2 = [[] for ii in calibration_vids]
            cur_img = [[] for ii in calibration_vids]
            for i_vid, vid_obj in enumerate(video_object):
                vid_obj.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
                ret, cur_img[i_vid] = vid_obj.read()

                if ret:
                    cur_img_gray = cv2.cvtColor(cur_img[i_vid], cv2.COLOR_BGR2GRAY)
                    found_valid_chessboard, corners = cv2.findChessboardCorners(cur_img_gray, cb_size)
                    valid_frames[i_vid][i_frame] = found_valid_chessboard

                    if found_valid_chessboard:
                        corners2[i_vid] = cv2.cornerSubPix(cur_img_gray, corners, (11, 11), (-1, -1), criteria)
                        cam_objpoints[i_vid].append(objp)
                        cam_imgpoints[i_vid].append(corners2[i_vid])

                    corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners,
                                                            found_valid_chessboard)
                    vid_path, vid_name = os.path.split(calibration_vids[i_vid])
                    vid_name, _ = os.path.splitext(vid_name)
                    frame_path = os.path.join(vid_path, vid_name)
                    if not os.path.isdir(frame_path):
                        os.makedirs(frame_path)
                    frame_name = vid_name + '_frame{:03d}'.format(i_frame) + '.png'
                    full_frame_name = os.path.join(frame_path, frame_name)
                    cv2.imwrite(full_frame_name, corners_img)

            # collect all checkerboard points visible in pairs of images
            if valid_frames[0][i_frame] and valid_frames[1][i_frame]:
                # checkerboards were identified in matching frames
                stereo_objpoints.append(objp)
                for i_vid, corner_pts in enumerate(corners2):
                    stereo_imgpoints[i_vid].append(corner_pts)

            # todo: test that the checkerboard points in each imgpoint array are correct
            # below is for checking if corners were correctly identified
            # for i_vid in range(3):
            #     if valid_frames[i_vid][i_frame]:
            #         corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners2[i_vid], found_valid_chessboard)
                    # cv2.imwrite(corners_img_name, corners_img)
                    # plt.imshow(corners_img)
                    # plt.show()

        calibration_data = {
            'cam_objpoints': cam_objpoints,
            'cam_imgpoints': cam_imgpoints,
            'stereo_objpoints': stereo_objpoints,
            'stereo_imgpoints': stereo_imgpoints,
            'valid_frames': valid_frames,
            'im_size': im_size,
            'cb_size': cb_size,
            'cropped_vid_metadata': cal_vid_metadata
        }

        for vid_obj in video_object:
            vid_obj.release()

        return calibration_data

    else:
        # calibration videos have different numbers of frames, so not clear how to line them up

        return none


def verify_checkerboard_points(calibration_vids, calibration_data):

    for i_vid, cal_vid in enumerate(calibration_vids):

        vid_obj = cv2.VideoCapture(cal_vid)
        num_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        valid_frame_counter = 0
        for i_frame in range(num_frames):

            if calibration_data['valid_frames'][i_vid][i_frame]:

                vid_obj.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
                ret, cur_img = vid_obj.read()

                corners = calibration_data['cam_imgpoints'][i_vid][valid_frame_counter]
                valid_frame_counter += 1

                corners_img = cv2.drawChessboardCorners(cur_img, calibration_data['cb_size'], corners, calibration_data['valid_frames'][i_vid][i_frame])
                cv2.imshow('image', corners_img)
                cv2.waitKey(0)

        vid_obj.release()


def crop_calibration_video(calib_vid, crop_params_df, calib_crop_top=100, filtertype=''):

    cc_metadata = navigation_utilities.parse_camera_calibration_video_name(calib_vid)

    session_date = cc_metadata['time'].date()
    crop_params_dict = crop_videos.crop_params_dict_from_df(crop_params_df, session_date, cc_metadata['boxnum'])

    cropped_vid_names = []
    if crop_params_dict:
        # make sure there is plenty of height. crop_params_dict should have keys 'direct','leftmirror','rightmirror'
        # top of cropping window is probably too low for the calibration videos if based on the reaching videos

        calibration_crop_params_dict = crop_params_dict

        for key in calibration_crop_params_dict:
            calibration_crop_params_dict[key][2] = calib_crop_top

            full_cropped_vid_name = navigation_utilities.create_cropped_calib_vid_name(calib_vid, key, calibration_crop_params_dict)
            cropped_vid_names.append(full_cropped_vid_name)
            if os.path.isfile(full_cropped_vid_name):
                # skip if already cropped
                continue

            crop_videos.crop_video(calib_vid, full_cropped_vid_name, calibration_crop_params_dict[key], key, filtertype=filtertype)

    return cropped_vid_names


def multi_mirror_calibration(calibration_data, calibration_summary_name):
    '''
    calibrate across multiple views
    :param calibration_data:
    :return:
    '''

    # initialize camera matrices for direct-->leftmirror, direct-->rightmirror, and leftmirror-->rightmirror
    # todo: initialize camera/fundamental/essential matrices for each set of views and prove that we can reconstruct the
    # chessboard images
    keys = calibration_data.keys()
    if 'camera_intrinsics' not in keys:
        calibration_data = camera_calibration_from_mirror_vids(calibration_data, calibration_summary_name)
    pass


def calibrate_all_Burgess_vids(cal_vid_parent, cal_data_parent, cb_size=(7, 10)):
    '''

    :param cal_vid_parent:
    :param cal_data_parent:
    :param cb_size:
    :return:
    '''

    paired_cal_vids = navigation_utilities.find_Burgess_calibration_vids(cal_vid_parent)

    for vid_pair in paired_cal_vids:
        calvid_metadata = [navigation_utilities.parse_Burgess_calibration_vid_name(vid) for vid in vid_pair]
        cal_data_name = navigation_utilities.create_multiview_calibration_data_name(cal_data_parent,
                                                                                    calvid_metadata[0]['session_datetime'])
        if not os.path.isfile(cal_data_name):
            # collect the checkerboard points, write to file
            collect_cbpoints_Burgess(vid_pair, cal_data_parent, cb_size=cb_size)

        calibrate_Burgess_session(cal_data_name, vid_pair)


def collect_cbpoints_Burgess(vid_pair, cal_data_parent, cb_size=(7, 10)):
    '''

    :param cal_vids:
    :param cal_data_parent:
    :param cb_size:
    :return:
    '''

    # extract metadata from file names. Note that cam 01 is upside down

    calvid_metadata = [navigation_utilities.parse_Burgess_calibration_vid_name(vid) for vid in vid_pair]
    cal_data_name = navigation_utilities.create_multiview_calibration_data_name(cal_data_parent,
                                                                                calvid_metadata[0]['session_datetime'])
    if os.path.isfile(cal_data_name):
        # if file already exists, assume cb points have already been collected
        return

        # camera calibrations have been performed, now need to do stereo calibration

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # CBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    # create video objects for each calibration_video
    vid_obj = []
    num_frames = []
    im_size = []
    for i_vid, vid_name in enumerate(vid_pair):
        vid_obj.append(cv2.VideoCapture(vid_name))
        num_frames.append(int(vid_obj[i_vid].get(cv2.CAP_PROP_FRAME_COUNT)))
        im_size.append((int(vid_obj[i_vid].get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_obj[i_vid].get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if all(nf == num_frames[0] for nf in num_frames):
        # check that there are the same number of frames in each calibration video

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        cbrow = cb_size[0]
        cbcol = cb_size[1]
        objp = np.zeros((cbrow * cbcol, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cbrow, 0:cbcol].T.reshape(-1, 2)

        cam_objpoints = [[] for ii in vid_pair]
        stereo_objpoints = []
        cam_imgpoints = [[] for ii in vid_pair]
        stereo_imgpoints = [[] for ii in vid_pair]

    # create boolean lists to track which frames have valid checkerboard images. Writing so that can expand to more
    # views eventually
    # valid_frames is a list of length num_cameras of lists that each are of length num_frames
    valid_frames = [[False for frame_num in range(num_frames[0])] for ii in vid_pair]
    stereo_frames = []
    for i_frame in range(num_frames[0]):
        print(i_frame)

        corners2 = [[] for ii in vid_pair]
        cur_img = [[] for ii in vid_pair]
        for i_vid, vo in enumerate(vid_obj):
            vo.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            ret, cur_img[i_vid] = vo.read()

            # test_img_name = os.path.join(cal_data_parent, 'test_cam{:02d}.jpg'.format(calibration_metadata[i_vid]['cam_num']))
            # cv2.imwrite(test_img_name, cur_img[i_vid])

            if ret:
                if calvid_metadata[i_vid]['cam_num'] == 1:
                    # rotate the image 180 degrees
                    cur_img[i_vid] = cv2.rotate(cur_img[i_vid], cv2.ROTATE_180)

                cur_img_gray = cv2.cvtColor(cur_img[i_vid], cv2.COLOR_BGR2GRAY)
                found_valid_chessboard, corners = cv2.findChessboardCorners(cur_img_gray, cb_size)
                valid_frames[i_vid][i_frame] = found_valid_chessboard

                if found_valid_chessboard:
                    corners2[i_vid] = cv2.cornerSubPix(cur_img_gray, corners, (11, 11), (-1, -1), criteria)
                    cam_objpoints[i_vid].append(objp)
                    cam_imgpoints[i_vid].append(corners2[i_vid])

                    corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners2[i_vid],
                                                            found_valid_chessboard)

                else:
                    corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners,
                                                            found_valid_chessboard)
                # vid_path, vid_name = os.path.split(calibration_vids[i_vid])
                # vid_name, _ = os.path.splitext(vid_name)
                # frame_path = os.path.join(vid_path, vid_name)
                # if not os.path.isdir(frame_path):
                #     os.makedirs(frame_path)

                session_date_string = navigation_utilities.datetime_to_string_for_fname(calvid_metadata[i_vid]['session_datetime'])
                test_save_dir = os.path.join(cal_data_parent, 'corner_images', session_date_string, 'cam{:02d}'.format(calvid_metadata[i_vid]['cam_num']))
                if not os.path.isdir(test_save_dir):
                    os.makedirs(test_save_dir)

                test_img_name = os.path.join(test_save_dir,
                                             'test_cboard_{}_cam{:02d}_frame{:04d}.jpg'.format(session_date_string, calvid_metadata[i_vid]['cam_num'], i_frame))
                # frame_name = vid_name + '_frame{:03d}'.format(i_frame) + '.png'

                cv2.imwrite(test_img_name, corners_img)

        # collect all checkerboard points visible in pairs of images
        if valid_frames[0][i_frame] and valid_frames[1][i_frame]:
            # checkerboards were identified in matching frames
            stereo_objpoints.append(objp)
            stereo_frames.append(i_frame)

            for i_vid, corner_pts in enumerate(corners2):
                stereo_imgpoints[i_vid].append(corner_pts)

        # todo: test that the checkerboard points in each imgpoint array are correct
        # below is for checking if corners were correctly identified
        # for i_vid in range(3):
        #     if valid_frames[i_vid][i_frame]:
        #         corners_img = cv2.drawChessboardCorners(cur_img[i_vid], cb_size, corners2[i_vid], found_valid_chessboard)
        # cv2.imwrite(corners_img_name, corners_img)
        # plt.imshow(corners_img)
        # plt.show()

    calibration_data = {
        'cam_objpoints': cam_objpoints,
        'cam_imgpoints': cam_imgpoints,
        'stereo_objpoints': stereo_objpoints,
        'stereo_imgpoints': stereo_imgpoints,
        'stereo_frames': stereo_frames,         # frame numbers for which valid checkerboards were found for both cameras
        'valid_frames': valid_frames,
        'im_size': im_size,
        'cb_size': cb_size,
        'calvid_metadata': calvid_metadata
    }

    skilled_reaching_io.write_pickle(cal_data_name, calibration_data)

    for vo in vid_obj:
        vo.release()


def calibrate_Burgess_session(calibration_data_name, vid_pair, num_frames_for_intrinsics=50, min_frames_for_intrinsics=10, num_frames_for_stereo=20, min_frames_for_stereo=5):
    '''

    :param calibration_data_name:
    :param num_frames_for_intrinsics:
    :param min_frames_for_intrinsics: minimum number of frames to use for intrinsics calibration (if number of valid frames
        is less than num_frames_for_intrinsics, it will use all available frames. If min_frames_for_intrinsics is greater
        than the number of available frames, calibration will be skipped).
    :return:
    '''

    # create video objects for each calibration_video
    vid_obj = []
    for i_vid, vid_name in enumerate(vid_pair):
        vid_obj.append(cv2.VideoCapture(vid_name))

    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_ASPECT_RATIO
    STEREO_FLAGS = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_FIX_PRINCIPAL_POINT
    # initialize camera intrinsics to have an aspect ratio of 1 and assume the center of the 1280 x 1024 field is [639.5, 511.5]
    init_mtx = np.array([[2000, 0, 639.5],[0, 2000, 511.5],[0, 0, 1]])
    cal_data = skilled_reaching_io.read_pickle(calibration_data_name)

    # get intrinsics and distortion for each camera
    num_cams = len(cal_data['cam_objpoints'])
    cal_data['mtx'] = []
    cal_data['dist'] = []
    cal_data['frame_nums_for_intrinsics'] = []
    for i_cam in range(num_cams):

        current_cam = cal_data['calvid_metadata'][i_cam]['cam_num']

        session_date_string = navigation_utilities.datetime_to_string_for_fname(
            cal_data['calvid_metadata'][i_cam]['session_datetime'])
        # if 'mtx' in cal_data.keys():
        #     # if intrinsics have already been calculated for this camera, skip
        #     if i_cam >= len(cal_data['mtx']):
        #         # this camera number is larger than the number of cameras for which intrinsics have been stored
        #         print('intrinsics already calculated for {}, camera {:02d}'.format(session_date_string, current_cam))
        #         continue

        print('working on {}, camera {:02d} intrinsics calibration'.format(session_date_string, current_cam))

        # select num_frames_for_intrinsics evenly spaced frames
        cam_objpoints = cal_data['cam_objpoints'][i_cam]
        cam_imgpoints = cal_data['cam_imgpoints'][i_cam]

        total_valid_frames = np.shape(cam_objpoints)[0]
        num_frames_to_use = min(num_frames_for_intrinsics, total_valid_frames)
        if num_frames_to_use < min_frames_for_intrinsics:
            mtx = np.zeros((3, 3))
            dist = np.zeros((1, 5))
            frame_numbers = []
        else:
            objpoints_for_intrinsics, imgpoints_for_intrinsics, frame_numbers = select_cboards_for_calibration(cam_objpoints, cam_imgpoints, num_frames_to_use)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_for_intrinsics,
                                                       imgpoints_for_intrinsics,
                                                       cal_data['im_size'][i_cam],
                                                       init_mtx,
                                                       None,
                                                       flags=CALIBRATION_FLAGS)
            intrinsics_frames = np.shape(objpoints_for_intrinsics)[0]
            valid_frames = cal_data['valid_frames'][i_cam]
            for i_frame in range(intrinsics_frames):
                pp = test_reprojection(objpoints_for_intrinsics[i_frame], imgpoints_for_intrinsics[i_frame], mtx, rvecs[i_frame], tvecs[i_frame], dist)
                cur_frame = [ii for ii, vf in enumerate(valid_frames) if vf == True][frame_numbers[i_frame]]
                vid_obj[i_cam].set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
                ret, cur_img = vid_obj[i_cam].read()

                if current_cam == 1:
                    # rotate the image 180 degrees
                    cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)

                corners_img = cv2.drawChessboardCorners(cur_img, cal_data['cb_size'], imgpoints_for_intrinsics[i_frame], True)
                reproj_img = cv2.drawChessboardCorners(corners_img, cal_data['cb_size'], pp, False)

                img_name = '/home/levlab/Public/mouse_SR_videos_to_analyze/mouse_SR_calibration_data/test_frame_{}_cam{:02d}_frame{:03d}.jpg'.format(session_date_string, current_cam, cur_frame)
                cv2.imwrite(img_name, reproj_img)

        cal_data['mtx'].append(np.copy(mtx))
        cal_data['dist'].append(np.copy(dist))
        cal_data['frame_nums_for_intrinsics'].append(frame_numbers)

        skilled_reaching_io.write_pickle(calibration_data_name, cal_data)

    # now perform stereo calibration
    # num_frames_for_stereo = 20, min_frames_for_stereo = 5
    stereo_objpoints = cal_data['stereo_objpoints']
    stereo_imgpoints = cal_data['stereo_imgpoints']
    num_stereo_pairs = np.shape(stereo_objpoints)[0]
    num_frames_to_use = min(num_frames_for_stereo, num_stereo_pairs)
    objpoints, imgpoints, stereo_frame_idx = select_cboards_for_stereo_calibration(stereo_objpoints, stereo_imgpoints, num_frames_to_use)
    frames_for_stereo_calibration = [sf_idx for sf_idx in cal_data['stereo_frames']]

    mtx = cal_data['mtx']
    dist = cal_data['dist']
    im_size = cal_data['im_size']
    # im_size must be the same for both cameras
    if all([ims == im_size[0] for ims in im_size]) and num_frames_to_use >= min_frames_for_stereo:
        # all images have the same size
        print('working on stereo calibration for {}'.format(session_date_string))
        im_size = im_size[0]
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints[0], imgpoints[1], mtx[0], dist[0], mtx[1], dist[1], im_size, flags=STEREO_FLAGS)
    else:
        ret = False
        mtx1 = np.zeros((3, 3))
        mtx2 = np.zeros((3, 3))
        dist1 = np.zeros((1, 5))
        dist2 = np.zeros((1, 5))
        R = np.zeros((3, 3))
        T = np.zeros((3, 1))
        E = np.zeros((3, 3))
        F = np.zeros((3, 3))

    cal_data['R'] = R
    cal_data['T'] = T
    cal_data['E'] = E
    cal_data['F'] = F
    cal_data['frames_for_stereo_calibration'] = frames_for_stereo_calibration   # frame numbers in original calibration video used for the stereo calibration
    # if valid_frames[0][i_frame] and valid_frames[1][i_frame]:
    #     # checkerboards were identified in matching frames
    #     stereo_objpoints.append(objp)
    #     for i_vid, corner_pts in enumerate(corners2):
    #         stereo_imgpoints[i_vid].append(corner_pts)

    skilled_reaching_io.write_pickle(calibration_data_name, cal_data)

    # check if calibration worked
    num_valid_stereo_pairs = np.shape(cal_data['stereo_imgpoints'])[1]
    for stereo_idx in range(num_valid_stereo_pairs):
        frame_num = cal_data['stereo_frames'][stereo_idx]
        projPoints = []
        for i_cam in range(num_cams):
            projPoints.append(cal_data['stereo_imgpoints'][i_cam][stereo_idx])
            projPoints[i_cam] = np.squeeze(projPoints[i_cam]).T

        print('frame number {:d}'.format(frame_num))
        worldpoints = triangulate_points(cal_data, projPoints, frame_num)

    pass



def test_reprojection(objpoints, imgpoints, mtx, rvec, tvec, dist):

    # rvec = np.array([0., 0., 0.])
    projected_pts, _ = cv2.projectPoints(objpoints, rvec, tvec, mtx, dist)
    return projected_pts


def select_cboards_for_calibration(objpoints, imgpoints, num_frames_to_extract):

    total_frames = np.shape(objpoints)[0]
    frame_spacing = int(total_frames/num_frames_to_extract)

    selected_objpoints = objpoints[::frame_spacing]
    selected_imgpoints = imgpoints[::frame_spacing]

    frame_numbers = range(0, total_frames, frame_spacing)

    return selected_objpoints, selected_imgpoints, frame_numbers


def select_cboards_for_stereo_calibration(objpoints, imgpoints, num_frames_to_extract):

    total_frames = np.shape(objpoints)[0]

    if num_frames_to_extract == 0:
        frame_numbers = []
        selected_objpoints = []
        selected_imgpoints = []
    else:
        frame_spacing = int(total_frames/num_frames_to_extract)
        frame_numbers = range(0, total_frames, frame_spacing)

        selected_objpoints = objpoints[::frame_spacing]
        selected_imgpoints = [ip[::frame_spacing] for ip in imgpoints]

    return selected_objpoints, selected_imgpoints, frame_numbers


def triangulate_points(cal_data, projPoints, frame_num):
    cal_data_parent = '/home/levlab/Public/mouse_SR_videos_to_analyze/mouse_SR_calibration_data'

    # first, create projMatr1 and projMatr2 (3 x 4 projection matrices for each camera) for each camera
    mtx = cal_data['mtx']
    dist = cal_data['dist']
    num_cams = len(mtx)

    # undistort points, which should give results in normalized coordinates
    ud_pts = [cv2.undistortPoints(projPoints[i_cam], mtx[i_cam], dist[i_cam]) for i_cam in range(num_cams)]

    num_pts = np.shape(projPoints)[2]
    reshaped_pts = [np.zeros((1, num_pts, 2)) for ii in range(2)]
    projPoints_array = []
    newpoints = [[], []]
    for ii in range(2):
        projPoints_array.append(np.squeeze(np.array([projPoints[ii]]).T))
        reshaped_pts[ii][0,:,:] = projPoints_array[ii]
    newpoints[0], newpoints[1] = cv2.correctMatches(cal_data['F'], reshaped_pts[0], reshaped_pts[1])
    newpoints = [np_array.astype('float32') for np_array in newpoints]

    new_cornerpoints = [np.squeeze(newpoints[ii]) for ii in range(2)]

    # for i_cam in range(2):
    #     session_date_string = navigation_utilities.datetime_to_string_for_fname(
    #         cal_data['calvid_metadata'][i_cam]['session_datetime'])
    #     test_img_dir = os.path.join(cal_data_parent, 'corner_images', session_date_string,
    #                                  'cam{:02d}'.format(cal_data['calvid_metadata'][i_cam]['cam_num']))
    #
    #     test_img_name = os.path.join(test_img_dir,
    #                                  'test_cboard_{}_cam{:02d}_frame{:04d}.jpg'.format(session_date_string,
    #                                                                                cal_data['calvid_metadata'][i_cam]['cam_num'],
    #                                                                                frame_num))

        # cboard_img = cv2.imread(test_img_name)
        # new_name = os.path.join(test_img_dir, 'refined_pts_{}_cam{:02d}_frame{:04d}.jpg'.format(session_date_string,
        #                                                                            cal_data['calvid_metadata'][i_cam]['cam_num'],
        #                                                                            frame_num))
        # new_img = cv2.drawChessboardCorners(cboard_img, cal_data['cb_size'], np.reshape(new_cornerpoints[ii], (70, 1, 2)), False)
        # try:
        #     cv2.imwrite(new_name, new_img)
        # except:
        #     pass

    # nudp1, nudp2 = cv2.correctMatches(cal_data['F'], ud_pts[0], ud_pts[1])

    projMatr1 = np.eye(3, 4)
    projMatr2 = np.hstack((cal_data['R'], cal_data['T']))

    points4D = cv2.triangulatePoints(projMatr1, projMatr2, ud_pts[0], ud_pts[1])
    worldpoints = np.squeeze(cv2.convertPointsFromHomogeneous(points4D.T))

    fig = plt.figure()
    ax0 = fig.add_subplot(3, 1, 1)
    ax1 = fig.add_subplot(3, 1, 2)
    ax3d = fig.add_subplot(3, 1, 3, projection='3d')

    ax0.scatter(projPoints_array[0][:, 0], projPoints_array[0][:, 1])
    ax1.scatter(projPoints_array[1][:, 0], projPoints_array[0][:, 1])
    ax3d.scatter(worldpoints[:, 0], worldpoints[:, 1], worldpoints[:, 2])
    plt.show()

    return worldpoints


def camera_calibration_from_mirror_vids(calibration_data, calibration_summary_name):

    #todo: working here - perform the camera calibration to get the intrinsics
    CALIBRATION_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_ZERO_TANGENT_DIST

    cb_size = calibration_data['cb_size']
    im_size = calibration_data['orig_im_size']
    vf = calibration_data['valid_frames']
    num_frames = np.shape(vf)[1]

    views = [cvmd['view'] for cvmd in calibration_data['cropped_vid_metadata']]
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbrow = cb_size[0]
    cbcol = cb_size[1]
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # loop through all 3 views, find all valid images, translate them into the full image, perform the calibration

    full_view_imgpoints = []
    for i_view, view in enumerate(views):
        imgpoints = np.array(calibration_data['cam_imgpoints'][i_view])
        crop_params = calibration_data['cropped_vid_metadata'][i_view]['crop_params']

        # translate checkerboard points to the original image
        this_view_imgpoints = imgpoints
        this_view_imgpoints[:,:,:,0] = imgpoints[:,:,:,0] + crop_params[0] - 1
        this_view_imgpoints[:,:,:,1] = imgpoints[:,:,:,1] + crop_params[2] - 1

        full_view_imgpoints.append(this_view_imgpoints)

    all_imgpoints = np.concatenate([ip for ip in full_view_imgpoints])
    num_valid_cb = np.shape(all_imgpoints)[0]
    all_objpoints = [objp for ii in range(num_valid_cb)]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_objpoints, all_imgpoints, im_size, None, None, flags=CALIBRATION_FLAGS)
    calibration_data['camera_intrinsics'] = {'mtx': mtx, 'dist': dist}

    skilled_reaching_io.write_pickle(calibration_summary_name, calibration_data)

    return calibration_data


def match_cb_points(cb1, cb2):

    pass


def extract_valid_cbs_by_frame(calibration_data):

    cb_size = calibration_data['cb_size']
    vf = calibration_data['valid_frames']
    _, num_frames = np.shape(vf)

    views = [cvmd['view'] for cvmd in calibration_data['cropped_vid_metadata']]
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    cbrow = cb_size[0]
    cbcol = cb_size[1]
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    frame_counter = 0

    # loop through all 3 views, find all valid images, translate them into the full image, perform the calibration
    num_validframes = [0, 0, 0]
    for i_frame in range(num_frames):
        for i_view, view in enumerate(views):
            if vf[i_view][i_frame]:
                imgpoints = np.array(calibration_data['cam_imgpoints'][i_view])
                cur_imgpoints = imgpoints[i_frame,:,:,:]
                crop_params = calibration_data['cropped_vid_metadata'][i_view]['crop_params']

                num_validframes[i_view] += 1

        # test to see if points match up on original image
        # vid_name = '/home/levlab/Public/rat_SR_videos_to_analyze/calibration_videos/calibration_videos_2021/calibration_videos_202107/calibration_videos_202107_box02/GridCalibration_box02_20210713_14-30-28.avi'
        # vid_obj = cv2.VideoCapture(vid_name)
        # vid_obj.set(cv2.CAP_PROP_POS_FRAMES, 134)
        # ret, cur_img = vid_obj.read()
        # corners = full_imgpoints[0,:,:,:]
        # corners_img = cv2.drawChessboardCorners(cur_img, cb_size, corners,
        #                                         True)
        # test_img = '/home/levlab/Public/rat_SR_videos_to_analyze/test_img.png'
        # cv2.imwrite(test_img, corners_img)
        # plt.imshow(corners_img)
        # plt.show()
        pass