import numpy as np
import cv2
import os
import glob
import navigation_utilities
import skilled_reaching_io
import matplotlib.pyplot as plt
import computer_vision_basics as cvb
import pandas as pd
import scipy.io as sio


def reconstruct_optitrack_session(view_directories, parent_directories):
    '''

    :param view_directories:
    :param cal_data_parent: parent directory for folders containing pickle files with calibration results. Has structure:
        cal_data_parent-->calibration_data_YYYY-->calibration_data_YYYYmm
    :param videos_parent:
    :return:
    '''
    cal_data_parent = parent_directories['cal_data_parent']

    # find all the files containing labeled points in view_directories
    full_pickles = []
    # meta_pickles = []
    for view_dir in view_directories:
        full_pickles.append(glob.glob(os.path.join(view_dir, '*full.pickle')))
        # meta_pickles.append(glob.glob(os.path.join(view_dir, '*meta.pickle')))

    for cam01_file in full_pickles[0]:
        pickle_metadata = []
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam01_file))
        calibration_file = navigation_utilities.find_multiview_calibration_data_name(cal_data_parent, pickle_metadata[0]['trialtime'])
        if not os.path.exists(calibration_file):
            # if there is no calibration file for this session, skip
            continue
        pickle_files = [cam01_file]  # pickle_files[0] is the full pickle file for camera 1 for the current video

        # find corresponding pickle file for camera 2
        _, cam01_pickle_name = os.path.split(cam01_file)
        cam01_pickle_stem = cam01_pickle_name[:cam01_pickle_name.find('cam01') + 5]
        cam02_pickle_stem = cam01_pickle_stem.replace('cam01', 'cam02')

        cam02_file_list = glob.glob(os.path.join(view_directories[1], cam02_pickle_stem + '*full.pickle'))
        if len(cam02_file_list) == 1:
            cam02_file = cam02_file_list[0]
            pickle_files.append(cam02_file)
        else:
            print('no matching camera 2 file for {}'.format(cam01_file))
            continue
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam02_file))

        # read in the points
        dlc_output = [skilled_reaching_io.read_pickle(pickle_file) for pickle_file in pickle_files]

        cam_meta_files = [pickle_file.replace('full.pickle', 'meta.pickle') for pickle_file in pickle_files]
        dlc_metadata = [skilled_reaching_io.read_pickle(cam_meta_file) for cam_meta_file in cam_meta_files]
        # dlc_metadata.append(skilled_reaching_io.read_pickle(cam01_meta))
        # dlc_metadata.append(skilled_reaching_io.read_pickle(cam02_meta))

        bodyparts = []
        for dlc_md in dlc_metadata:
            bodyparts.append(dlc_md['data']['DLC-model-config file']['all_joints_names'])
        pts_wrt_orig_img, dlc_conf = rotate_translate_optitrack_points(dlc_output, pickle_metadata, dlc_metadata)

        # now have all the identified points moved back into the original coordinate systems that the checkerboards were
        # identified in, and confidence levels. pts_wrt_orig_img is an array (num_frames x num_joints x 2) and dlc_conf
        # is an array (num_frames x num_joints). Zeros are stored where dlc was uncertain (no result for that joint on
        # that frame)

        reconstruct3d_single_optitrack_video(calibration_file, pts_wrt_orig_img, dlc_conf, pickle_files, dlc_metadata, parent_directories)


def reconstruct3d_single_optitrack_video(calibration_file, pts_wrt_orig_img, dlc_conf, pickle_files, dlc_metadata, parent_directories):
    '''

    :param calibration_file: file name with full path of .pickle file containing results of camera intrinsics and stereo calibration
    :param pts_wrt_orig_img: 2 - element list containing numpy arrays num_frames x num_joints x 2
    :param dlc_conf:  2-element list containing num_frames x num_joints array with dlc confidence values
    :return:
    '''

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
        checkerboard_square_size - size of individual checkerboard squares in mm
        calvid_metadata
        mtx - 2 x 3 x 3 array with camera intrinsic matrices - mtx[0,:,:] is for camera 1, mtx[1,:,:] is for camera 2
        dist - distortion coefficients; 2 x 5 array, first row is for camera 1, 2nd row for camera 2
        frame_nums_for_intrinsics
        R - rotation matrix of camera 2 w.r.t. camera 1 (currently, camera 1 is rotated 180 degrees, and R is with respect to camera 1 images rotated 180 degrees so they are upright)
        T - translation matrix of camera 2 w.r.t. camera 1
        E - essential matrix
        F - fundamental matrix
        frames_for_stereo_calibration
    '''
    video_root_folder = parent_directories['video_root_folder']
    reconstruct3d_parent = parent_directories['reconstruct3d_parent']

    # check if this video has already been reconstructed
    dlc_output_pickle_metadata = [navigation_utilities.parse_dlc_output_pickle_name_optitrack(pf) for pf in pickle_files]
    reconstruction3d_fname = navigation_utilities.create_3d_reconstruction_pickle_name(dlc_output_pickle_metadata[0], reconstruct3d_parent)

    if os.path.exists(reconstruction3d_fname):
        print('{} already calculated'.format(reconstruction3d_fname))
        return

    mouseID = dlc_output_pickle_metadata[0]['mouseID']
    session_num = dlc_output_pickle_metadata[0]['session_num']
    video_num = dlc_output_pickle_metadata[0]['video_number']
    session_datestring = dlc_output_pickle_metadata[0]['trialtime'].strftime('%m/%d/%Y')

    # read in the calibration file, make sure we have stereo and camera calibrations
    cal_data = skilled_reaching_io.read_pickle(calibration_file)

    num_cams = len(pts_wrt_orig_img)
    num_frames = np.shape(pts_wrt_orig_img[0])[0]
    pts_per_frame = np.shape(dlc_conf[0])[1]

    pickle_metadata = []
    orig_vid_names = []
    for i_cam in range(num_cams):
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(pickle_files[i_cam]))
        orig_vid_names.append(navigation_utilities.find_original_optitrack_video(video_root_folder, pickle_metadata[i_cam]))

    # perform frame-by-frame reconstruction
    # set up numpy arrays to accept world points, measured points, dlc confidence values, reprojected points, and reprojection errors
    reconstructed_data = {
        'frame_points': np.zeros((num_frames, num_cams, pts_per_frame, 2)),
        'worldpoints': np.zeros((num_frames, pts_per_frame, 3)),
        'reprojected_points': np.zeros((num_frames, num_cams, pts_per_frame, 2)),
        'reprojection_errors': np.zeros((num_frames, num_cams, pts_per_frame)),
        'frame_confidence': np.zeros((num_frames, num_cams, pts_per_frame)),
        'cal_data': cal_data
    }
    for i_frame in range(num_frames):
        print('triangulating frame {:04d} for {}, session number {:d} on {}, video {:03d}'.format(i_frame, mouseID, session_num, session_datestring, video_num))
        frame_pts = np.zeros((num_cams, pts_per_frame, 2))
        frame_conf = np.zeros((num_cams, pts_per_frame))
        for i_cam in range(num_cams):
            frame_pts[i_cam, :, :] = pts_wrt_orig_img[i_cam][i_frame, :, :]
            frame_conf[i_cam, :] = dlc_conf[i_cam][i_frame, :]

        frame_worldpoints, frame_reprojected_pts, frame_reproj_errors, valid_frame_points = reconstruct_one_frame(frame_pts, frame_conf, cal_data, dlc_metadata, pickle_metadata, i_frame, parent_directories)
        # at this point, worldpoints is still in units of checkerboards, needs to be scaled by the size of individual checkerboard squares

        reconstructed_data['frame_points'][i_frame, :, :, :] = frame_pts
        reconstructed_data['worldpoints'][i_frame, :, :] = frame_worldpoints
        reconstructed_data['reprojected_points'][i_frame, :, :, :] = frame_reprojected_pts
        reconstructed_data['reprojection_errors'][i_frame, :, :] = frame_reproj_errors.T
        reconstructed_data['frame_confidence'][i_frame, :, :] = frame_conf

    skilled_reaching_io.write_pickle(reconstruction3d_fname, reconstructed_data)


def reconstruct_one_frame(frame_pts, frame_conf, cal_data, dlc_metadata, pickle_metadata, frame_num, parent_directories):
    '''
    perform 3D reconstruction on a single frame
    :param frame_pts: 
    :param frame_conf: num_pts x num_cams numpy array with dlc confidence values for each point in each camera view
    :param cal_data: 
    :return:
    '''
    # verify that coordinates are mapped correctly into full image
    # pickle_metadata = [navigation_utilities.parse_dlc_output_pickle_name_optitrack(pickle_file) for pickle_file in
    #                    pickle_files]
    # overlay_pts_in_orig_image(pickle_meta[0], frame_pts[0], dlc_metadata[0], i_frame,
    #                           rotate_img=True)
    # overlay_pts_in_orig_image(pickle_meta[1], frame_pts[1], dlc_metadata[1], i_frame,
    #                           rotate_img=False)

    num_cams = len(frame_pts)
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    frame_pts_ud = [cv2.undistortPoints(frame_pts[i_cam, :, :], mtx[i_cam], dist[i_cam]) for i_cam in range(num_cams)]
    # frame_pts_ud = [np.squeeze(ppts) for ppts in frame_pts_ud]
    projMatr1 = np.eye(3, 4)
    projMatr2 = np.hstack((cal_data['R'], cal_data['T']))

    # comment back in to test if frame_pts_ud look correct
    # plot_projpoints(frame_pts_ud, dlc_metadata[0])

    points4D = cv2.triangulatePoints(projMatr1, projMatr2, frame_pts_ud[0], frame_pts_ud[1])
    worldpoints = np.squeeze(cv2.convertPointsFromHomogeneous(points4D.T))

    # alternative calculation of worldpoints using non-opencv triangulation algorithm
    # points4D_new, _ = cvb.linear_LS_triangulation(projPoints[0], projMatr1, projPoints[1], projMatr2)
    # points4D = cv2.triangulatePoints(projMatr1, projMatr2, frame_pts[0], frame_pts[1])
    # worldpoints = points4D_new

    #check that there was good reconstruction of individual points (i.e., the matched points were truly well-matched?)
    reprojected_pts, reproj_errors = check_3d_reprojection(worldpoints, frame_pts, cal_data, dlc_metadata, pickle_metadata, frame_num, parent_directories)

    #todo: check reprojected points and reproj_errors, look for mislabeled points
    #also, consider looking across frames for jumps, and checking the dlc confidence values. Finally, need to check if
    #pellet labels swapped between frames

    valid_frame_points = validate_frame_points(reproj_errors, frame_conf, max_reproj_error=20, min_conf=0.9)

    return worldpoints, reprojected_pts, reproj_errors, valid_frame_points


def validate_frame_points(reproj_errors, frame_conf, max_reproj_error=20, min_conf=0.9):
    '''
    function to check for valid points in a frame. Valid points are points that:
        1) reproject close to the original points found in each camera view
        2) have high confidence in DLC
    :param reproj_errors: num_points x num_cams numpy arrays containing the reprojection error in pixels
        of each triangulated world point back to the original video frames
    :param frame_conf: num_cams x num_points numpy array containing the deeplabcut confidence in point
        identification
    :return: valid_frame_points
    '''
    # consider changing max reproj error and minimum dlc confidence to be specific for each body part

    num_pts = np.shape(reproj_errors)[0]
    num_cams = np.shape(reproj_errors)[1]

    valid_reprojections = reproj_errors < max_reproj_error
    valid_dlc_confidence = frame_conf.T > min_conf

    valid_frame_points = np.logical_and(valid_reprojections, valid_dlc_confidence)

    return valid_frame_points


def plot_projpoints(projPoints, dlc_metadata):

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']
    fig, axs = plt.subplots(1, 2)

    dotsize = 3

    for i_cam in range(2):
        for i_pt, pt in enumerate(projPoints[i_cam]):
            if len(pt) > 0:
                try:
                    x, y = pt[0]
                except:
                    x, y = pt
                # x = int(round(x))
                # y = int(round(y))
                bp_color = color_from_bodypart(bodyparts[i_pt])

                axs[i_cam].scatter(x, y, marker='o', s=dotsize, color=bp_color)

                axs[i_cam].set_ylim(-1, 1)
                axs[i_cam].invert_yaxis()
                axs[i_cam].set_xlim(-1, 1)

    # plt.show()

    pass

def plot_worldpoints(worldpoints, dlc_metadata, pickle_metadata, i_frame, videos_parent='/home/levlab/Public/mouse_SR_videos_to_analyze'):

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    dotsize = 6

    video_root_folder = os.path.join(videos_parent, 'mouse_SR_videos_tocrop')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_mouse_SR_videos')

    month_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m')
    day_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, pickle_metadata['mouseID'], month_dir, day_dir)

    cam_dir = day_dir + '_' + 'cam{:02d}'.format(pickle_metadata['cam_num'])
    cropped_vid_folder = os.path.join(cropped_videos_parent, pickle_metadata['mouseID'], month_dir, day_dir)

    orig_vid_name_base = '_'.join([pickle_metadata['mouseID'],
                              pickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                              '{:d}'.format(pickle_metadata['session_num']),
                              '{:03d}'.format(pickle_metadata['video_number']),
                              'cam{:02d}'.format(pickle_metadata['cam_num'])
                              ])
    orig_vid_name = os.path.join(orig_vid_folder, orig_vid_name_base + '.avi')

    for i_pt, pt in enumerate(worldpoints):

        if len(pt) > 0:
            try:
                x, y, z = pt[0]
            except:
                x, y, z = pt
            # x = int(round(x))
            # y = int(round(y))
            bp_color = color_from_bodypart(bodyparts[i_pt])

            ax.scatter(x, y, z, marker='o', s=dotsize, color=bp_color)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax.invert_yaxis()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    jpg_name = orig_vid_name_base + '_{:04d}'.format(i_frame)
    jpg_name = os.path.join(cropped_vid_folder, jpg_name + '.jpg')

    plt.savefig(jpg_name)
    # plt.show()

    pass


def check_3d_reprojection(worldpoints, frame_pts, cal_data, dlc_metadata, pickle_metadata, frame_num, parent_directories):
    '''
    calculate reprojection of worldpoints back into original images, and return the location of those projected points
    (projected onto the distorted image), as well as the euclidean distance in pixels from the originally identified
    points in DLC to the reprojected points (in the distorted image)
    :param worldpoints: num_points x 3 array containing (x,y,z) triangulated points in world coordinates with the
        origin at the camera 1 lens. ADD X,Y,Z POSITIVE DIRECTIONS HERE
    :param frame_pts: num_cams x num_points x 2 numpy array containing (x,y) pairs of deeplabcut output rotated/
        translated into the original video frame so that the images are upright (i.e., if camera 1 is rotated 180
        degrees, the image/coordinates for camera 1 are rotated)
    :param cal_data:

    :return projected_pts:
    :return reproj_errors:
    '''
    num_cams = np.shape(frame_pts)[0]

    #3d plot of worldpoints if needed to check triangulation
    # plot_worldpoints(worldpoints, dlc_metadata[0], pickle_metadata[0], frame_num, parent_directories=parent_directories)
    projected_pts = []
    pts_per_frame = np.shape(frame_pts)[1]
    reproj_errors = np.zeros((pts_per_frame, num_cams))
    for i_cam in range(num_cams):
        mtx = cal_data['mtx'][i_cam]
        dist = cal_data['dist'][i_cam]
        if i_cam == 0:
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
        else:
            rvec, _ = cv2.Rodrigues(cal_data['R'])
            tvec = cal_data['T']

        ppts, _ = cv2.projectPoints(worldpoints, rvec, tvec, mtx, dist)
        ppts = np.squeeze(ppts)
        projected_pts.append(ppts)
        reproj_errors[:, i_cam] = calculate_reprojection_errors(ppts, frame_pts[i_cam, :, :])

    #     overlay_pts_in_orig_image(pickle_metadata[i_cam], frame_pts[i_cam], dlc_metadata[i_cam], frame_num, mtx, dist, parent_directories, reprojected_pts=projected_pts,
    #                               rotate_img=pickle_metadata[i_cam]['isrotated'])
    # draw_epipolar_lines(cal_data, frame_pts, projected_pts, dlc_metadata, pickle_metadata, frame_num, parent_directories)
    #
    # plt.show()

    return projected_pts, reproj_errors


def calculate_reprojection_errors(reprojected_pts, measured_pts):

    # the output of cv2.projectPoints may be n x 1 x 2
    reprojected_pts = np.squeeze(reprojected_pts)
    xy_errors = reprojected_pts - measured_pts
    euclidean_error = np.sqrt(np.sum(np.square(xy_errors), 1))

    return euclidean_error


def triangulate_single_point(cal_data, matched_pts):

    cal_data_parent = '/home/levlab/Public/mouse_SR_videos_to_analyze/mouse_SR_calibration_data'

    # first, create projMatr1 and projMatr2 (3 x 4 projection matrices for each camera) for each camera
    mtx = cal_data['mtx']
    dist = cal_data['dist']
    num_cams = len(mtx)

    # undistort points, which should give results in normalized coordinates
    ud_pts = [cv2.undistortPoints(matched_pts[i_cam], mtx[i_cam], dist[i_cam]) for i_cam in range(num_cams)]

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
        try:
            cam_metadata = pickle_metadata[i_cam]
        except:
            pass

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
                # overlay_pts_in_orig_image(pickle_metadata[i_cam], pts_in_calibration_coords, dlc_metadata[i_cam], i_frame,
                #                           rotate_img=False)
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


def overlay_pts_in_orig_image(pickle_metadata, current_coords, dlc_metadata, i_frame, mtx, dist, parent_directories, reprojected_pts=None, rotate_img=False):


    cropped_vids_parent = parent_directories['cropped_vids_parent']
    video_root_folder = parent_directories['video_root_folder']

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

    month_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m')
    day_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, pickle_metadata['mouseID'], month_dir, day_dir)

    cam_dir = day_dir + '_' + 'cam{:02d}'.format(pickle_metadata['cam_num'])
    cropped_vid_folder = os.path.join(cropped_vids_parent, pickle_metadata['mouseID'], month_dir, day_dir, cam_dir)

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
    fig, img_ax = overlay_pts_on_image(cur_img, mtx, dist, current_coords, reprojected_pts, bodyparts, ['o', '+'], jpg_name)

    # new_img = cur_img
    # for i_pt, pt in enumerate(current_coords):
    #     if len(pt) > 0:
    #         try:
    #             x, y = pt[0]
    #         except:
    #             x, y = pt
    #         x = int(round(x))
    #         y = int(round(y))
    #         bp_color = color_from_bodypart(bodyparts[i_pt])
    #         new_img = cv2.circle(new_img, (x, y), 3, bp_color, -1)
    #
    # cv2.imwrite(jpg_name, new_img)


def prepare_img_axes(width, height, scale=1.0, dpi=100, nrows=1, ncols=1):
    fig_width = (width * scale / dpi) * ncols
    fig_height = (width * scale / dpi) * nrows
    fig = plt.figure(
        frameon=False, figsize=(fig_width, fig_height), dpi=dpi
    )

    axs = []
    for i_row in range(nrows):
        ax_row = []
        for i_col in range(ncols):
            idx = (i_row * ncols) + i_col + 1
            ax_row.append(fig.add_subplot(nrows, ncols, idx))
            ax_row[i_col].axis("off")
            ax_row[i_col].set_xlim(0, width)
            ax_row[i_col].set_ylim(0, height)
            ax_row[i_col].invert_yaxis()
        axs.append(ax_row)

    # plt.subplots_adjust(0., 0., 0., 0., 0.)

    return fig, axs


def overlay_pts_on_image(img, mtx, dist, pts, reprojected_pts, bodyparts, markertype, jpg_name):

    dotsize = 6

    h, w, _ = np.shape(img)
    fig, ax = prepare_img_axes(w, h)

    img_ud = cv2.undistort(img, mtx, dist)

    ax[0][0].imshow(img_ud)

    for i_pt, pt in enumerate(pts):

        if len(pt) > 0:
            try:
                x, y = pt[0]
            except:
                x, y = pt
            # x = int(round(x))
            # y = int(round(y))

            pt_ud_norm = np.squeeze(cv2.undistortPoints(np.array([x, y]), mtx, dist))
            pt_ud = cvb.unnormalize_points(pt_ud_norm, mtx)
            bp_color = color_from_bodypart(bodyparts[i_pt])

            ax[0][0].plot(pt_ud[0], pt_ud[1], marker=markertype[0], ms=dotsize, color=bp_color)
            # ax.plot(x, y, marker=markertype[0], ms=dotsize, color=bp_color)

    for i_rpt, rpt in enumerate(reprojected_pts):
        if len(rpt) > 0:
            try:
                x, y = rpt[0]
            except:
                x, y = rpt
            # x = int(round(x))
            # y = int(round(y))
            bp_color = color_from_bodypart(bodyparts[i_rpt])

            ax[0][0].plot(x, y, marker=markertype[1], ms=dotsize, color=bp_color)

    # plt.show()
    fig.savefig(jpg_name)

    return fig, ax


def draw_epipolar_lines(cal_data, frame_pts, reproj_pts, dlc_metadata, pickle_metadata, i_frame, parent_directories, markertype=['o', '+']):

    dotsize = 4
    reproj_pts = np.squeeze(reproj_pts)

    cropped_vids_parent = parent_directories['cropped_vids_parent']
    video_root_folder = parent_directories['video_root_folder']

    bodyparts = dlc_metadata[0]['data']['DLC-model-config file']['all_joints_names']
    mouseID = pickle_metadata[0]['mouseID']
    month_dir = mouseID + '_' + pickle_metadata[0]['trialtime'].strftime('%Y%m')
    day_dir = mouseID + '_' + pickle_metadata[0]['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, mouseID, month_dir, day_dir)

    cam_dirs = [day_dir + '_' + 'cam{:02d}'.format(pickle_metadata[i_cam]['cam_num']) for i_cam in range(2)]

    cropped_vid_folders = [os.path.join(cropped_vids_parent, mouseID, month_dir, day_dir, cam_dir) for cam_dir in cam_dirs]

    orig_vid_names_base = ['_'.join([mouseID,
                              pickle_metadata[0]['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                              '{:d}'.format(pickle_metadata[0]['session_num']),
                              '{:03d}'.format(pickle_metadata[0]['video_number']),
                              'cam{:02d}'.format(pickle_metadata[i_cam]['cam_num'])
                              ]) for i_cam in range(2)]
    orig_vid_names = [os.path.join(orig_vid_folder, orig_vid_name_base + '.avi') for orig_vid_name_base in orig_vid_names_base]

    #read in images from both camera views
    img = []
    for i_cam, orig_vid_name in enumerate(orig_vid_names):
        video_object = cv2.VideoCapture(orig_vid_name)

        video_object.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, cur_img = video_object.read()

        if i_cam == 0:
            cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)
        img.append(cv2.undistort(cur_img, cal_data['mtx'][i_cam], cal_data['dist'][i_cam]))

        video_object.release()

    h, w, _ = np.shape(img[0])
    im_size = (w, h)
    fig, axs = prepare_img_axes(w, h, scale=1.0, dpi=100, nrows=1, ncols=2)
    # fig, ax = prepare_img_axes(w, h)

    for i_cam in range(2):
        axs[0][i_cam].imshow(img[i_cam])

    for i_cam in range(2):
        mtx = cal_data['mtx'][i_cam]
        dist = cal_data['dist'][i_cam]
        points_in_img = frame_pts[i_cam]
        # undistort points
        pt_ud_norm = np.squeeze(cv2.undistortPoints(points_in_img, mtx, dist))
        pt_ud = cvb.unnormalize_points(pt_ud_norm, mtx)
        for i_pt, pt in enumerate(pt_ud):
            if len(pt) > 0:
                try:
                    x, y = pt[0]
                except:
                    x, y = pt
                # x = int(round(x))
                # y = int(round(y))
                bp_color = color_from_bodypart(bodyparts[i_pt])   # undistorted point identified by DLC

                axs[0][i_cam].plot(x, y, marker=markertype[0], ms=dotsize, color=bp_color)
                x2 = points_in_img[i_pt,0]
                y2 = points_in_img[i_pt,1]
                axs[0][i_cam].plot(x2, y2, marker='+', ms=dotsize, color=bp_color)   # point from DLC with original image disortion

                x3 = reproj_pts[i_cam][i_pt, 0]
                y3 = reproj_pts[i_cam][i_pt, 1]
                axs[0][i_cam].plot(x3, y3, marker='*', ms=dotsize, color=bp_color)    # reprojected point

        draw_epipolar_lines_on_img(pt_ud, i_cam+1, cal_data['F'], im_size, bodyparts, axs[0][1-i_cam])

    plt.show()
    pass


def draw_epipolar_lines_on_img(img_pts, whichImage, F, im_size, bodyparts, ax):

    epilines = cv2.computeCorrespondEpilines(img_pts, whichImage, F)

    for i_line, epiline in enumerate(epilines):

        bp_color = color_from_bodypart(bodyparts[i_line])
        epiline = np.squeeze(epiline)
        edge_pts = find_line_edge_coordinates(epiline, im_size)

        if not np.all(edge_pts==0):
            ax.plot(edge_pts[:, 0], edge_pts[:, 1], color=bp_color, ls='-', marker='.')


def find_line_edge_coordinates(line, im_size):

    a, b, c = line
    edge_pts = np.zeros((2, 2))

    x_edge = np.array([0, im_size[0] - 1])
    y_edge = np.array([0, im_size[1] - 1])

    i_pt = 0
    # check the intersection with the left and right image borders unless the line is vertical
    if abs(a) > 0:
        test_y = (-c - a * x_edge[0]) / b
        if y_edge[0] <= test_y <= y_edge[1]:
            # check intersection with left image border
            edge_pts[i_pt, :] = [x_edge[0], test_y]
            i_pt += 1

        test_y = (-c - a * x_edge[1]) / b
        if y_edge[0] <= test_y <= y_edge[1]:
            # check intersection with left image border
            edge_pts[i_pt, :] = [x_edge[1], test_y]
            i_pt += 1

    # check the intersection with the left and right image borders unless the line is horizontal
    if abs(b) > 0:
        if i_pt < 2:
            test_x = (-c - b * y_edge[0]) / a
            if x_edge[0] <= test_x <= x_edge[1]:
                # check intersection with left image border
                edge_pts[i_pt, :] = [test_x, y_edge[0]]
                i_pt += 1

        if i_pt < 2:
            test_x = (-c - b * y_edge[1]) / a
            if x_edge[0] <= test_x <= x_edge[1]:
                # check intersection with left image border
                edge_pts[i_pt, :] = [test_x, y_edge[1]]
                i_pt += 1

    return edge_pts


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

    bp_color = [float(bpc)/255. for bpc in bp_color]

    return bp_color



def refine_trajectories(parent_directories):

    reconstruct_3d_parent = parent_directories['reconstruct3d_parent']
    reconstruct3d_folders = navigation_utilities.find_3dreconstruction_folders(reconstruct_3d_parent)

    for r3df in reconstruct3d_folders:

        r3d_pickles = glob.glob(os.path.join(r3df, '*.pickle'))

        for r3d_file in r3d_pickles:

            r3d_data = skilled_reaching_io.read_pickle(r3d_file)

            #todo:
            # 1) find valid frames on a per-frame basis
            # 2) look for jumps between valid points in adjacent frames
            # 3) look for points that aren't where they should be (for example, if index finger on right paw is too far from middle finger on right paw)

            num_frames = np.shape(worldpoints)[0]

            for i_frame in range(num_frames):

            pass

    pass