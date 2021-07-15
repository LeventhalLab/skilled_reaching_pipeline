import navigation_utilities
import skilled_reaching_io
import os
import csv
import numpy as np
import cv2
import glob


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


def calibration_multi_camera(calibration_vids, calibration_parent, cb_size=(6,9))