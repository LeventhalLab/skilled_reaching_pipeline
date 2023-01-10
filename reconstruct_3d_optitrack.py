import numpy as np
import cv2
import os
import glob

import dlc_utilities
import navigation_utilities
import skilled_reaching_calibration
import skilled_reaching_io
import matplotlib.pyplot as plt
import computer_vision_basics as cvb
import scipy.interpolate
import pandas as pd
import sr_visualization


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
    meta_pickles = []
    for view_dir in view_directories:
        full_pickles.append(glob.glob(os.path.join(view_dir, '*full.pickle')))
        meta_pickles.append(glob.glob(os.path.join(view_dir, '*meta.pickle')))

    pickle_metadata = []
    for cam01_file in full_pickles[0]:
        dlc_output_pickle_metadata = navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam01_file)
        reconstruct3d_parent = parent_directories['reconstruct3d_parent']
        reconstruction3d_fname = navigation_utilities.create_3d_reconstruction_pickle_name(
            dlc_output_pickle_metadata, reconstruct3d_parent)
        if os.path.exists(reconstruction3d_fname):
            print('{} already calculated'.format(reconstruction3d_fname))
            continue

        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(cam01_file))
        calibration_file = navigation_utilities.find_optitrack_calibration_data_name(cal_data_parent, pickle_metadata[0]['trialtime'])
        if calibration_file is None or not os.path.exists(calibration_file):
            # if there is no calibration file for this session, skip
            continue
        cal_data = skilled_reaching_io.read_pickle(calibration_file)

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
        dlc_output = [clean_dlc_output(dlco) for dlco in dlc_output]

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

        frame_num = 0
        frame_str = 'frame{:04d}'.format(frame_num)
        for i_cam in range(2):
            trajectory_metadata = dlc_utilities.extract_trajectory_metadata(dlc_metadata[i_cam], pickle_metadata[i_cam])
            dlc_data = dlc_utilities.extract_data_from_dlc_output(dlc_output[i_cam], trajectory_metadata)

            bp_coords = dlc_utilities.collect_bp_data(dlc_data, 'coordinates')
            mtx = cal_data['mtx'][i_cam]
            dist = cal_data['dist'][i_cam]
            # overlay original dlc output on cropped images to see if problem is with original identification or translating/rotating back into full frame for cam 2 (cam 1 looks good)
        #     overlay_pts_in_cropped_img(pickle_metadata[i_cam], bp_coords, dlc_metadata[i_cam], frame_num, mtx, dist,
        #                                parent_directories, reprojected_pts=None, vid_type='.avi')
        #
        # plt.show()

        reconstruct3d_single_optitrack_video(calibration_file, pts_wrt_orig_img, dlc_conf, pickle_files, dlc_metadata, parent_directories)


def clean_dlc_output(dlc_output):
    # for some reason, dlc occasionally identifies multiple candidate points for a specific bodypart/joint
    frame_list = dlc_output.keys()

    for frame in frame_list:
        if frame[:5] != 'frame':
            continue
        frame_coords = np.squeeze(dlc_output[frame]['coordinates'])
        frame_conf = np.squeeze(dlc_output[frame]['confidence'])
        if frame_coords.ndim == 2:
            # I'm pretty sure this means that all bodyparts are associated with a single point
            dlc_output[frame]['coordinates'] = frame_coords
            dlc_output[frame]['confidence'] = frame_conf

        else:
            # at least one bodypart/joint had multiple points identified for it, figure out which one and pick the highest
            # confidence point for that bodypart/joint

            num_pts = len(frame_coords)
            new_frame_coords = np.zeros((num_pts, 2))
            new_frame_conf = np.zeros(num_pts)
            for i_pt, pt in enumerate(frame_coords):

                if pt.size == 0:
                    # no point identified here, assign [0., 0.] to coordinates and 0. to confidence
                    new_frame_coords[i_pt, :] = np.zeros(2)
                    new_frame_conf[i_pt] = 0.
                elif np.shape(pt)[0] > 1:
                    # multiple points were identified; need to pick one of them
                    pt_conf = np.squeeze(frame_conf[i_pt])
                    max_ptconf = max(pt_conf)
                    max_conf_idx = np.where(pt_conf == max_ptconf)[0][0]
                    # takes the first element of the tuple returned by np.where (possible there are multiple max confidence points, I think?)

                    new_frame_coords[i_pt, :] = pt[max_conf_idx, :]
                    new_frame_conf[i_pt] = max_ptconf
                    pass
                else:
                    # one point was identified for this bodypart, so just keep the old coordinates/confidence
                    new_frame_coords[i_pt, :] = pt
                    new_frame_conf[i_pt] = np.squeeze(frame_conf[i_pt])

            dlc_output[frame]['coordinates'] = new_frame_coords
            dlc_output[frame]['confidence'] = new_frame_conf

    return dlc_output


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

    # todo: try to refine fundamental matrix to see if we can improve calibration accuracy
    cal_metadata = navigation_utilities.parse_optitrack_calibration_data_name(calibration_file)
    # skilled_reaching_calibration.show_cal_images_with_epilines(cal_metadata, parent_directories, plot_undistorted=True)

    video_root_folder = parent_directories['video_root_folder']
    reconstruct3d_parent = parent_directories['reconstruct3d_parent']

    # check if this video has already been reconstructed
    dlc_output_pickle_metadata = [navigation_utilities.parse_dlc_output_pickle_name_optitrack(pf) for pf in pickle_files]
    reconstruction3d_fname = navigation_utilities.create_3d_reconstruction_pickle_name(dlc_output_pickle_metadata[0], reconstruct3d_parent)

    # if os.path.exists(reconstruction3d_fname):
    #     print('{} already calculated'.format(reconstruction3d_fname))
    #     return

    mouseID = dlc_output_pickle_metadata[0]['mouseID']
    session_num = dlc_output_pickle_metadata[0]['session_num']
    vid_num = dlc_output_pickle_metadata[0]['vid_num']
    session_datestring = dlc_output_pickle_metadata[0]['trialtime'].strftime('%m/%d/%Y')

    # read in the calibration file, make sure we have stereo and camera calibrations
    cal_data = skilled_reaching_io.read_pickle(calibration_file)

    # every now and then, a frame gets dropped. assume it's the last frame, so the first num_frames-1 are aligned
    num_cams = len(pts_wrt_orig_img)
    num_cam_vid_frames = [np.shape(pts_wrt_orig_img[i_cam])[0] for i_cam in range(num_cams)]
    num_frames = min(num_cam_vid_frames)
    pts_per_frame = np.shape(dlc_conf[0])[1]

    pickle_metadata = []
    orig_vid_names = []
    for i_cam in range(num_cams):
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(pickle_files[i_cam]))
        orig_vid_names.append(navigation_utilities.find_original_optitrack_videos(video_root_folder, pickle_metadata[i_cam]))

    # perform frame-by-frame reconstruction
    # set up numpy arrays to accept world points, measured points, dlc confidence values, reprojected points, and reprojection errors
    reconstructed_data = {
        'frame_points': np.zeros((num_frames, num_cams, pts_per_frame, 2)),
        'frame_points_ud': np.zeros((num_frames, num_cams, pts_per_frame, 2)),
        'worldpoints': np.zeros((num_frames, pts_per_frame, 3)),
        'reprojected_points': np.zeros((num_frames, num_cams, pts_per_frame, 2)),
        'reprojection_errors': np.zeros((num_frames, num_cams, pts_per_frame)),
        'frame_confidence': np.zeros((num_frames, num_cams, pts_per_frame)),
        'cal_data': cal_data
    }
    for i_frame in range(num_frames):
        print('triangulating frame {:04d} for {}, session number {:d} on {}, video {:03d}'.format(i_frame, mouseID, session_num, session_datestring, vid_num))
        frame_pts = np.zeros((num_cams, pts_per_frame, 2))
        frame_conf = np.zeros((num_cams, pts_per_frame))
        for i_cam in range(num_cams):
            frame_pts[i_cam, :, :] = pts_wrt_orig_img[i_cam][i_frame, :, :]
            frame_conf[i_cam, :] = dlc_conf[i_cam][i_frame, :]

        # frame_pts are the original identified points translated/rotated into the original video image (but turned upright)
        # so... I'm pretty sure they're still distorted.
        frame_worldpoints, frame_reprojected_pts, frame_reproj_errors, frame_pts_ud, valid_frame_points = \
            reconstruct_one_frame(frame_pts, frame_conf, cal_data, dlc_metadata, pickle_metadata, i_frame, parent_directories)
        # at this point, worldpoints is still in units of checkerboards, needs to be scaled by the size of individual checkerboard squares

        reconstructed_data['frame_points'][i_frame, :, :, :] = frame_pts
        reconstructed_data['frame_points_ud'][i_frame, :, :, :] = np.squeeze(frame_pts_ud)
        reconstructed_data['worldpoints'][i_frame, :, :] = frame_worldpoints
        reconstructed_data['reprojected_points'][i_frame, :, :, :] = frame_reprojected_pts
        reconstructed_data['reprojection_errors'][i_frame, :, :] = frame_reproj_errors.T
        reconstructed_data['frame_confidence'][i_frame, :, :] = frame_conf
        reconstructed_data['bodyparts'] = dlc_metadata[0]['data']['DLC-model-config file']['all_joints_names']

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
    # cal_data['T'] = cal_data['T_unit']
    projMatr2 = cvb.P_from_RT(cal_data['R'], cal_data['T'])

    # E, F, R, T = skilled_reaching_calibration.recalculate_E_and_F_from_stereo_matches(cal_data)

    # F_projMatr2 = cvb.P_from_RT(cal_data['R_ffm'], cal_data['T_ffm'])
    # F_projMatr2 = cvb.P_from_RT(R['R_E'], T['t_E'])
    # comment back in to test if frame_pts_ud look correct
    # plot_projpoints(frame_pts_ud, dlc_metadata[0])

    points4D = cv2.triangulatePoints(projMatr1, projMatr2, frame_pts_ud[0], frame_pts_ud[1])
    # points4D_newF = cv2.triangulatePoints(projMatr1, F_projMatr2, frame_pts_ud[0], frame_pts_ud[1])
    worldpoints = np.squeeze(cv2.convertPointsFromHomogeneous(points4D.T)) * cal_data['checkerboard_square_size']
    # worldpoints_newF = np.squeeze(cv2.convertPointsFromHomogeneous(points4D_newF.T))

    # cal_data['F_new'] = F['F_from_E']
    # cal_data['R'] = R['R_E']
    # cal_data['T'] = T['t_E']
    # new_cal_data = cal_data.copy()
    # new_cal_data['R'] = R['R_F']
    # new_cal_data['T'] = T['t_F']
    # alternative calculation of worldpoints using non-opencv triangulation algorithm
    # points4D_new, _ = cvb.linear_LS_triangulation(projPoints[0], projMatr1, projPoints[1], projMatr2)
    # points4D = cv2.triangulatePoints(projMatr1, projMatr2, frame_pts[0], frame_pts[1])
    # worldpoints = points4D_new

    #check that there was good reconstruction of individual points (i.e., the matched points were truly well-matched?)
    frame_pts_ud_unnorm = np.zeros(np.shape(frame_pts))
    for i_cam in range(num_cams):
        frame_pts_ud_unnorm[i_cam, :, :] = cvb.unnormalize_points(np.squeeze(frame_pts_ud[i_cam]), mtx[i_cam])
    # reprojected_pts, reproj_errors = check_3d_reprojection(worldpoints, frame_pts, cal_data, dlc_metadata, pickle_metadata, frame_num, parent_directories)
    reprojected_pts, reproj_errors = check_3d_reprojection(worldpoints, frame_pts, cal_data, dlc_metadata,
                                                           pickle_metadata, frame_num, parent_directories)
    #todo: check reprojected points and reproj_errors, look for mislabeled points
    #also, consider looking across frames for jumps, and checking the dlc confidence values. Finally, need to check if
    #pellet labels swapped between frames

    valid_frame_points = validate_frame_points(reproj_errors, frame_conf, max_reproj_error=20, min_conf=0.9)

    return worldpoints, reprojected_pts, reproj_errors, frame_pts_ud_unnorm, valid_frame_points


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

def plot_worldpoints(worldpoints, dlc_metadata, pickle_metadata, i_frame, parent_directories):

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    dotsize = 3

    video_root_folder = parent_directories['video_root_folder']
    cropped_vids_parent = parent_directories['cropped_vids_parent']

    month_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m')
    day_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, pickle_metadata['mouseID'], month_dir, day_dir)

    cam_dir = day_dir + '_' + 'cam{:02d}'.format(pickle_metadata['cam_num'])
    cropped_vid_folder = os.path.join(cropped_vids_parent, pickle_metadata['mouseID'], month_dir, day_dir)

    orig_vid_name_base = '_'.join([pickle_metadata['prefix'] + pickle_metadata['mouseID'],
                              pickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                              '{:d}'.format(pickle_metadata['session_num']),
                              '{:03d}'.format(pickle_metadata['vid_num']),
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
    ax.set_xlim(20, 60)
    ax.set_ylim(20, 60)
    ax.set_zlim(100, 150)
    ax.invert_yaxis()

    jpg_name = orig_vid_name_base + '_{:04d}'.format(i_frame)
    jpg_name = os.path.join(cropped_vid_folder, jpg_name + '.jpg')

    # plt.savefig(jpg_name)
    plt.show()

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
    frame_pts = np.squeeze(frame_pts)

    #3d plot of worldpoints if needed to check triangulation
    # plot_worldpoints(worldpoints, dlc_metadata[0], pickle_metadata[0], frame_num, parent_directories)

    pts_per_frame = np.shape(frame_pts)[1]
    reproj_errors = np.zeros((pts_per_frame, num_cams))
    dist = [np.zeros((1, 5)) for ii in range(2)]
    dist = np.squeeze(dist)
    projected_pts = reproject_points(worldpoints, cal_data['R'], cal_data['T'], cal_data['mtx'], dist, cal_data['checkerboard_square_size'])

    for i_cam in range(num_cams):
        reproj_errors[:, i_cam] = calculate_reprojection_errors(projected_pts[i_cam], frame_pts[i_cam, :, :])

        # overlay_pts_in_orig_image(pickle_metadata[i_cam], frame_pts[i_cam], dlc_metadata[i_cam], frame_num, mtx, dist, parent_directories, reprojected_pts=projected_pts[i_cam],
        #                           rotate_img=pickle_metadata[i_cam]['isrotated'], plot_undistorted=True)
        # overlay_pts_in_cropped_img(pickle_metadata[i_cam], frame_pts[i_cam], dlc_metadata[i_cam], frame_num, mtx, dist,
        #                            parent_directories, reprojected_pts=None, vid_type='.avi', plot_undistorted=True)

        # overlay_pts_in_orig_image(pickle_metadata[i_cam], frame_pts[i_cam], dlc_metadata[i_cam], frame_num, mtx, dist, parent_directories, reprojected_pts=projected_pts[i_cam],
        #                           rotate_img=pickle_metadata[i_cam]['isrotated'], plot_undistorted=False)
        # overlay_pts_in_cropped_img(pickle_metadata[i_cam], frame_pts[i_cam], dlc_metadata[i_cam], frame_num, mtx, dist,
        #                            parent_directories, reprojected_pts=None, vid_type='.avi', plot_undistorted=False)

    draw_epipolar_lines(cal_data, frame_pts, projected_pts, dlc_metadata, pickle_metadata, frame_num, parent_directories, use_ffm=False, plot_undistorted=True)
    plt.show()

    return projected_pts, reproj_errors


def reproject_points(worldpoints, R, T, mtx, dist, scale_factor):
    '''

    :param worldpoints:
    :param R:
    :param T:
    :param mtx:
    :param dist:
    :param scale_factor: checkerboard size for calibration
    :return:
    '''
    projected_pts = []

    num_cams = np.shape(mtx)[0]

    for i_cam in range(num_cams):
        cur_mtx = mtx[i_cam]
        cur_dist = dist[i_cam]

        if i_cam == 0:
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
            Rmat = np.identity(3)
        else:
            rvec, _ = cv2.Rodrigues(R)
            tvec = T
            Rmat = R

        worldpoints = worldpoints / scale_factor

        C = cvb.P_from_RT(Rmat, tvec)
        wp = worldpoints.T
        wp = np.vstack((wp, np.ones((1, 17))))

        ppts_direct_hom = C @ wp
        ppts_direct = cv2.convertPointsFromHomogeneous(ppts_direct_hom.T)
        ppts_direct = np.squeeze(ppts_direct)
        ppts_direct_unnormalized = cvb.unnormalize_points(ppts_direct, cur_mtx)
        projected_pts.append(ppts_direct_unnormalized)

        # worldpoints = worldpoints / scale_factor
        # ppts, _ = cv2.projectPoints(worldpoints, rvec, tvec, np.identity(3), cur_dist)
        # ppts = np.squeeze(ppts)
        # ppts = cvb.unnormalize_points(ppts, cur_mtx)
        # projected_pts.append(ppts)

    return projected_pts

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
        reshaped_pts[ii][0, :, :] = projPoints_array[ii]
    newpoints[0], newpoints[1] = cv2.correctMatches(cal_data['F'], reshaped_pts[0], reshaped_pts[1])
    newpoints = [np_array.astype('float32') for np_array in newpoints]

    new_cornerpoints = [np.squeeze(newpoints[ii]) for ii in range(2)]


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
                # must be a vector instead of an array
                x, y = pt
            # possible that im_size is width x height or height x width
            try:
                new_x = im_size[0] - x
                new_y = im_size[1] - y
            except:
                pass

            reflected_pts.append([np.array([new_x, new_y])])
        else:
            reflected_pts.append(np.array([]))

    reflected_pts = np.array(reflected_pts)
    reflected_pts = np.squeeze(reflected_pts)

    return reflected_pts


def rotate_translate_optitrack_points(dlc_output, pickle_metadata, dlc_metadata, orig_im_size=(1280, 1024)):

    # note that current algorithm for camera 1 crops, then rotates. We want a rotated, but uncropped transformation of
    # coordinates camera 2 is easy - just crops
    pts_wrt_orig_img = []
    dlc_conf = []
    for i_cam, cam_output in enumerate(dlc_output):
        # cam_output is a dictionary where each entry is 'frame0000', 'frame0001', etc.
        # each frame has keys: 'coordinates', 'confidence', and 'costs'

        cam_metadata = pickle_metadata[i_cam]

        # loop through the frames
        frame_list = list(cam_output.keys())
        frame_list = [fr for fr in frame_list if fr[:5] == 'frame']

        # may need to get num_frames based on number of 'framexxxx' keys in cam_output
        num_frames = cam_output['metadata']['nframes']
        num_joints = len(cam_output['metadata']['all_joints_names'])
        pts_wrt_orig_img.append(np.zeros((num_frames, num_joints, 2)))
        dlc_conf.append(np.zeros((num_frames, num_joints)))

        for i_frame, frame in enumerate(frame_list):
            if frame[:5] != 'frame':
                continue
            current_coords = cam_output[frame]['coordinates']

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

            # store and return the confidence array
            conf = dlc_output[i_cam][frame]['confidence']
            # array_conf = convert_pickle_conf_to_array(conf)

            dlc_conf[i_cam][i_frame, :] = conf

    return pts_wrt_orig_img, dlc_conf


def optitrack_fullframe_to_cropped_coords(fullframe_pts, crop_params, im_size, isrotated):
    '''

    :param fullframe_pts: n x 2 array where n is the number of points to translate/rotate
    :param crop_params:
    :param isrotated:
    :return:
    '''
    # note that current algorithm for camera 1 crops, then rotates. We want a rotated, but uncropped transformation of
    # coordinates camera 2 is easy - just crops

    if isrotated:
        # points are translated into the "upright" (already rotated) video. So if video is rotated, need to reflect
        # the points across the middle of the full frame, then rotate them within the cropped region
        reflected_pts = rotate_pts_180(fullframe_pts, im_size)

        # points are rotated within the full frame. Now subtract the left and top edges of the crop window
        translated_reflected_pts = reflected_pts - np.array([crop_params[0], crop_params[2]])

        # now reflect these points across the center within the cropped frame
        crop_win_size = np.array([crop_params[1] - crop_params[0], crop_params[3] - crop_params[2]])
        translated_pts = rotate_pts_180(translated_reflected_pts, crop_win_size)

    else:
        translated_pts = fullframe_pts - np.array([crop_params[0], crop_params[2]])

    # if fullframe_pt was [0,0], reset translated_pts for that point to [0,0]
    for i_pt, ff_pt in enumerate(fullframe_pts):

        if all(ff_pt == 0):
            translated_pts[i_pt, :] = ff_pt

    return translated_pts


def convert_pts_to_array(pickle_format_pts):
    '''
    helper function to take points from a deeplabcut _full.pickle file and convert to a numpy array.
    :param pickle_format_pts:
    :return:
    '''
    num_joints = len(pickle_format_pts)
    array_pts = np.zeros([num_joints, 2])
    for i_pt, cur_pt in enumerate(pickle_format_pts):
        cur_pt = np.squeeze(cur_pt)
        if len(cur_pt) == 0:
            continue
        else:
            try:
                array_pts[i_pt, :] = cur_pt
            except:
                pass

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
    '''
    move identified points from deeplabcut from the cropped video back to the original video frame coordinates
    :param pickle_metadata:
    :param pts:
    :return:
    '''
    crop_win = pickle_metadata['crop_window']

    translated_pts = []
    for i_pt, pt in enumerate(pts):
        if len(pt) > 0:
            pt = np.squeeze(pt)
            x = pt[0]
            y = pt[1]

            new_x = crop_win[0] + x
            new_y = crop_win[2] + y

            translated_pts.append([np.array([new_x, new_y])])
        else:
            translated_pts.append(np.array([]))

    return translated_pts


def overlay_pts_in_orig_image(pickle_metadata, current_coords, dlc_metadata, i_frame, mtx, dist, parent_directories, reprojected_pts=None, rotate_img=False, plot_undistorted=True):

    cropped_vids_parent = parent_directories['cropped_vids_parent']
    video_root_folder = parent_directories['video_root_folder']

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

    month_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m')
    day_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, pickle_metadata['mouseID'], month_dir, day_dir)

    cam_dir = day_dir + '_' + 'cam{:02d}'.format(pickle_metadata['cam_num'])
    cropped_vid_folder = os.path.join(cropped_vids_parent, pickle_metadata['mouseID'], month_dir, day_dir, cam_dir)

    orig_vid_name_base = '_'.join([pickle_metadata['prefix'] + pickle_metadata['mouseID'],
                              pickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                              '{:d}'.format(pickle_metadata['session_num']),
                              '{:03d}'.format(pickle_metadata['vid_num']),
                              'cam{:02d}'.format(pickle_metadata['cam_num'])
                              ])
    orig_vid_name = os.path.join(orig_vid_folder, orig_vid_name_base + '.avi')
    orig_vid_names = navigation_utilities.find_original_optitrack_videos(video_root_folder, pickle_metadata)

    video_object = cv2.VideoCapture(orig_vid_name)

    video_object.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    ret, cur_img = video_object.read()

    video_object.release()

    jpg_name = orig_vid_name_base + '_{:04d}'.format(i_frame)
    if rotate_img:
        cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)
        # cur_img_ud = cv2.undistort(cur_img, mtx, dist)
        jpg_name = jpg_name + '_rotated'
    if plot_undistorted:
        jpg_name = jpg_name + '_undistorted'
    jpg_name = os.path.join(cropped_vid_folder, jpg_name + '.jpg')

    # overlay points
    fig, img_ax = overlay_pts_on_image(cur_img, mtx, dist, current_coords, reprojected_pts, bodyparts, ['o', 's'], jpg_name, plot_undistorted=plot_undistorted)

    if plot_undistorted:
        fig.suptitle('undistorted images and points. o=undistorted pt, square=reprojected pt')
    else:
        fig.suptitle('original (distorted) images and points. o=original pt, square=reprojected pt')

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


def overlay_pts_in_cropped_img(pickle_metadata, current_coords, dlc_metadata, i_frame, mtx, dist, parent_directories, reprojected_pts=None, vid_type='.avi', plot_undistorted=False):
    # todo: figure out how to account for undistortion in previously cropped image?
    if vid_type[0] != '.':
        vid_type = '.' + vid_type

    cropped_vids_parent = parent_directories['cropped_vids_parent']

    bodyparts = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

    month_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m')
    day_dir = pickle_metadata['mouseID'] + '_' + pickle_metadata['trialtime'].strftime('%Y%m%d')
    cam_dir = day_dir + '_' + 'cam{:02d}'.format(pickle_metadata['cam_num'])
    cropped_vid_folder = os.path.join(cropped_vids_parent, pickle_metadata['mouseID'], month_dir, day_dir, cam_dir)

    cropped_vid_name_base = '_'.join([pickle_metadata['prefix'] + pickle_metadata['mouseID'],
                                   pickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                                   '{:d}'.format(pickle_metadata['session_num']),
                                   '{:03d}'.format(pickle_metadata['vid_num']),
                                   'cam{:02d}'.format(pickle_metadata['cam_num'])])

    cropped_vid_name_search = cropped_vid_name_base + '*' + vid_type
    cropped_vid_name = os.path.join(cropped_vid_folder, cropped_vid_name_search)
    cropped_vid_list = glob.glob(cropped_vid_name)
    video_object = cv2.VideoCapture(cropped_vid_list[0])

    video_object.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    ret, cur_img = video_object.read()

    video_object.release()

    jpg_name = cropped_vid_name_base + '_{:04d}'.format(i_frame)
    # if rotate_img:
    #     cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)
    #     # cur_img_ud = cv2.undistort(cur_img, mtx, dist)
    #     jpg_name = jpg_name + '_rotated'
    jpg_name = os.path.join(cropped_vid_folder, jpg_name + '.jpg')

    # overlay points
    fig, img_ax = overlay_pts_on_image(cur_img, mtx, dist, current_coords, reprojected_pts, bodyparts, ['o', '+'],
                                       jpg_name, plot_undistorted=False)


def prepare_img_axes(width, height, scale=1.0, dpi=100, nrows=1, ncols=1):
    fig_width = (width * scale / dpi) * ncols
    fig_height = (height * scale / dpi) * nrows
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


def overlay_pts_on_image(img, mtx, dist, pts, reprojected_pts, bodyparts, markertype, jpg_name, plot_undistorted=True):

    dotsize = 3

    h, w, _ = np.shape(img)
    fig, ax = prepare_img_axes(w, h)

    if plot_undistorted:
        img_ud = cv2.undistort(img, mtx, dist)
        ax[0][0].imshow(img_ud)
    else:
        ax[0][0].imshow(img)

    for i_pt, pt in enumerate(pts):

        if len(pt) > 0:
            try:
                x, y = pt[0]
            except:
                x, y = pt
            # x = int(round(x))
            # y = int(round(y))

            if plot_undistorted:
                pt_ud_norm = np.squeeze(cv2.undistortPoints(np.array([x, y]), mtx, dist))
                to_plot = cvb.unnormalize_points(pt_ud_norm, mtx)
            else:
                to_plot = np.array([x, y])
            bp_color = color_from_bodypart(bodyparts[i_pt])

            ax[0][0].plot(to_plot[0], to_plot[1], marker=markertype[0], ms=dotsize, color=bp_color)
            # ax.plot(x, y, marker=markertype[0], ms=dotsize, color=bp_color)

    if reprojected_pts is not None:
        for i_rpt, rpt in enumerate(reprojected_pts):
            if len(rpt) > 0:
                try:
                    x, y = rpt[0]
                except:
                    x, y = rpt

                if plot_undistorted:
                    pt_ud_norm = np.squeeze(cv2.undistortPoints(np.array([x, y]), mtx, dist))
                    to_plot = cvb.unnormalize_points(pt_ud_norm, mtx)
                else:
                    to_plot = np.array([x, y])
                # x = int(round(x))
                # y = int(round(y))
                bp_color = color_from_bodypart(bodyparts[i_rpt])

                ax[0][0].plot(to_plot[0], to_plot[1], marker=markertype[1], ms=3, color=bp_color)

    # plt.show()
    # fig.savefig(jpg_name)

    return fig, ax


def draw_epipolar_lines(cal_data, frame_pts, reproj_pts, dlc_metadata, pickle_metadata, i_frame, parent_directories, use_ffm=True, markertype=['o', '+'], plot_undistorted=True):

    '''

    :param cal_data:
    :param frame_pts:
    :param reproj_pts:
    :param dlc_metadata:
    :param pickle_metadata:
    :param i_frame:
    :param parent_directories:
    :param markertype: first element is for point detected by DLC, second is for reprojected point
    :param plot_undistorted:
    :return:
    '''
    dotsize = 3
    reproj_pts = np.squeeze(reproj_pts)

    cropped_vids_parent = parent_directories['cropped_vids_parent']
    video_root_folder = parent_directories['video_root_folder']

    bodyparts = dlc_metadata[0]['data']['DLC-model-config file']['all_joints_names']
    mouseID = pickle_metadata[0]['mouseID']
    month_dir = mouseID + '_' + pickle_metadata[0]['trialtime'].strftime('%Y%m')
    day_dir = mouseID + '_' + pickle_metadata[0]['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, mouseID, month_dir, day_dir)

    cam_dirs = [day_dir + '_' + 'cam{:02d}'.format(pickle_metadata[i_cam]['cam_num']) for i_cam in range(2)]

    # cropped_vid_folders = [os.path.join(cropped_vids_parent, mouseID, month_dir, day_dir, cam_dir) for cam_dir in cam_dirs]

    orig_vid_names_base = ['_'.join([pickle_metadata[0]['prefix'] + mouseID,
                              pickle_metadata[0]['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                              '{:d}'.format(pickle_metadata[0]['session_num']),
                              '{:03d}'.format(pickle_metadata[0]['vid_num']),
                              'cam{:02d}'.format(pickle_metadata[i_cam]['cam_num'])
                              ]) for i_cam in range(2)]
    orig_vid_names = [os.path.join(orig_vid_folder, orig_vid_name_base + '.avi') for orig_vid_name_base in orig_vid_names_base]

    #read in images from both camera views
    img_ud = []
    img = []
    for i_cam, orig_vid_name in enumerate(orig_vid_names):
        video_object = cv2.VideoCapture(orig_vid_name)

        video_object.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, cur_img = video_object.read()

        if i_cam == 0:
            cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)
        img_ud.append(cv2.undistort(cur_img, cal_data['mtx'][i_cam], cal_data['dist'][i_cam]))
        img.append(cur_img)

        video_object.release()

    h, w, _ = np.shape(img_ud[0])
    im_size = (w, h)
    fig, axs = prepare_img_axes(w, h, scale=1.0, dpi=100, nrows=1, ncols=2)
    # fig, ax = prepare_img_axes(w, h)

    for i_cam in range(2):

        if plot_undistorted:
            # undistorted image
            axs[0][i_cam].imshow(img_ud[i_cam])
        else:
            # distorted original image
            axs[0][i_cam].imshow(img[i_cam])

        mtx = cal_data['mtx'][i_cam]
        dist = cal_data['dist'][i_cam]
        points_in_img = frame_pts[i_cam]
        # undistort points
        pt_ud_norm = np.squeeze(cv2.undistortPoints(points_in_img, mtx, dist))
        pt_ud = cvb.unnormalize_points(pt_ud_norm, mtx)
        if np.shape(points_in_img)[0] == 1:
            # only one point
            pt_ud = [pt_ud]

        if plot_undistorted:
            to_plot = pt_ud
        else:
            to_plot = points_in_img

        for i_pt, pt in enumerate(to_plot):
            if len(pt) > 0:
                try:
                    x, y = pt[0]
                except:
                    x, y = pt
                # x = int(round(x))
                # y = int(round(y))
                bp_color = color_from_bodypart(bodyparts[i_pt])   # undistorted point identified by DLC

                axs[0][i_cam].plot(x, y, marker=markertype[0], ms=dotsize, color=bp_color)
                # x2 = points_in_img[i_pt, 0]
                # y2 = points_in_img[i_pt, 1]
                # axs[0][i_cam].plot(x2, y2, marker='+', ms=dotsize, color=bp_color)   # point from DLC with original image disortion

                if reproj_pts[i_cam].ndim == 1:
                    x3 = reproj_pts[i_cam][0]
                    y3 = reproj_pts[i_cam][1]
                else:
                    x3 = reproj_pts[i_cam][i_pt, 0]
                    y3 = reproj_pts[i_cam][i_pt, 1]
                axs[0][i_cam].plot(x3, y3, marker='s', ms=dotsize, color=bp_color)    # reprojected point

        if plot_undistorted:
            if np.shape(points_in_img)[0] == 1:
                to_plot = pt_ud[0].reshape((1, -1, 2))  # needed to get correct array shape for computeCorrespondEpilines in draw_epipolar_lines_on_img
            else:
                to_plot = pt_ud
        else:
            if np.shape(points_in_img)[0] == 1:
                to_plot = points_in_img[1].reshape((1, -1, 2))
            else:
                to_plot = points_in_img
        if use_ffm:
            F = cal_data['F_ffm']
            linestyle = '--'
        else:
            F = cal_data['F']
            linestyle = '-'
        # draw_epipolar_lines_on_img(to_plot, 1+i_cam, cal_data['F'], im_size, bodyparts, axs[0][1-i_cam], linestyle='-')

        imgpts_ud = [cal_data['stereo_imgpoints_ud'][i_cam] for i_cam in range(2)]

        # try recalculating using findFundamentalMat
        # mtx = cal_data['mtx']
        # imgpts_reshaped = [np.reshape(im_pts, (-1, 2)) for im_pts in imgpts_ud]
        # F_new, ffm_mask = cv2.findFundamentalMat(imgpts_reshaped[0], imgpts_reshaped[1], cv2.FM_RANSAC, 0.1, 0.99)
        # dist = np.array([0.,0.,0.,0.,0.])
        # E_new, E_mask = cv2.findEssentialMat(imgpts_reshaped[0], imgpts_reshaped[1], cal_data['mtx'][0], dist, cal_data['mtx'][1], dist, cv2.FM_RANSAC, 0.99, 1)
        #
        # F_from_E = np.linalg.inv(mtx[1].T) @ E_new @ np.linalg.inv(mtx[0])
        # E_from_F = mtx[1].T @ F_new @ mtx[0]

        draw_epipolar_lines_on_img(to_plot, 1 + i_cam, cal_data['F'], im_size, bodyparts, axs[0][1 - i_cam], linestyle='--')

        # draw_epipolar_lines_on_img(to_plot, 1 + i_cam, F_from_E, im_size, bodyparts, axs[0][1 - i_cam], linestyle='dotted')

    plt.show()
    pass


def draw_epipolar_lines_on_img(img_pts, whichImage, F, im_size, bodyparts, ax, lwidth=0.5, linestyle='-'):

    epilines = cv2.computeCorrespondEpilines(img_pts, whichImage, F)

    for i_line, epiline in enumerate(epilines):

        bp_color = color_from_bodypart(bodyparts[i_line])
        epiline = np.squeeze(epiline)
        edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

        if not np.all(edge_pts == 0):
            ax.plot(edge_pts[:, 0], edge_pts[:, 1], color=bp_color, ls=linestyle, marker='.', lw=lwidth)


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


def smooth_3d_trajectory(r3d_data, frame_valid_pts, pt_euc_diffs):

    worldpoints = r3d_data['worldpoints']



def refine_trajectories(parent_directories):

    reconstruct_3d_parent = parent_directories['reconstruct3d_parent']
    reconstruct3d_folders = navigation_utilities.find_3dreconstruction_folders(reconstruct_3d_parent)

    for r3df in reconstruct3d_folders:

        r3d_pickles = glob.glob(os.path.join(r3df, '*.pickle'))

        for r3d_file in r3d_pickles:

            r3d_data = skilled_reaching_io.read_pickle(r3d_file)

            frame_valid_pts = identify_valid_points_in_frames(r3d_data)   # based on reprojection error and dlc confidence
            pt_euc_diffs = calculate_interframe_point_jumps(r3d_data)
            # pt_euc_diffs is num_cameras x num_frames-1 x num_bodyparts numpy array
            correct_pellet_locations(r3d_data, r3d_file, parent_directories, pt_euc_diffs)

            smooth_3d_trajectory(r3d_data, frame_valid_pts, pt_euc_diffs)
            pass

            #todo:
            # 1) find valid points on a per-frame basis (DONE?)
            # 2) look for jumps between valid points in adjacent frames
            # 3) look for points that aren't where they should be (for example, if index finger on right paw is too far from middle finger on right paw)
            # 4) correct pellet locations (maybe do all the pellet locations separately from the other bodyparts?)


def correct_pellet_locations(r3d_data, r3d_file, parent_directories, pt_euc_diffs, max_reproj_error=10, min_conf=0.9):
    video_root_folder = parent_directories['video_root_folder']
    full_pickles, meta_pickles = navigation_utilities.find_dlc_pickles_from_r3d_filename(r3d_file, parent_directories)
    pickle_metadata = []
    for full_pickle in full_pickles:
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(full_pickle))

    month_dir = pickle_metadata[0]['mouseID'] + '_' + pickle_metadata[0]['trialtime'].strftime('%Y%m')
    day_dir = pickle_metadata[0]['mouseID'] + '_' + pickle_metadata[0]['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, pickle_metadata[0]['mouseID'], month_dir, day_dir)

    # r3d_metadata = navigation_utilities.parse_3d_reconstruction_pickle_name(r3d_file)

    dlc_metadata = [skilled_reaching_io.read_pickle(cam_meta_file) for cam_meta_file in meta_pickles]
    pickle_metadata = []
    orig_vid_names = []
    for i_cam in range(len(full_pickles)):
        pickle_metadata.append(navigation_utilities.parse_dlc_output_pickle_name_optitrack(full_pickles[i_cam]))
        orig_vid_names.append(navigation_utilities.find_original_optitrack_video(video_root_folder, pickle_metadata[i_cam]))
    jump_thresh = 20

    cal_data = r3d_data['cal_data']
    projMatr1 = np.eye(3, 4)
    projMatr2 = np.hstack((cal_data['R'], cal_data['T']))
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    pellet_idx = ['pellet' in str for str in r3d_data['bodyparts']]
    num_pellets = len(np.argwhere(pellet_idx))
    pellet_framepoints = r3d_data['frame_points'][:, :, pellet_idx, :]
    pellet_reproj_errors = r3d_data['reprojection_errors'][:, :, pellet_idx]
    pellet_conf = r3d_data['frame_confidence'][:, :, pellet_idx]
    pellet_euc_diffs = pt_euc_diffs[:, :, pellet_idx]

    num_frames = np.shape(pellet_framepoints)[0]
    num_cameras = np.shape(pellet_framepoints)[1]
    # establish final pellet 1 and pellet 2 locations
    validated_pellet_framepoints = np.zeros((num_frames, num_cameras, num_pellets, 2))
    mismatched_pellets = np.zeros(num_frames, dtype=bool)
    for i_frame in range(num_frames):
        print('working on frame {:04d}'.format(i_frame))

        # how confident are we in the location for each pellet?
        frame_pellet_conf = pellet_conf[i_frame, :, :]   # frame_pellet_conf contains dlc confidence values for each pellet in each camera view; shape num_cams x num_pellets
        all_ppts = [np.zeros((2, 2)), np.zeros((2, 2))]
        all_reproj_errors = [np.zeros((2, 2)), np.zeros((2, 2))]
        swapped_points = [False, False]
        for i_pellet in range(num_pellets):
            other_pellet_idx = 1 - i_pellet
            # is the reprojection error small and the confidence high for this pellet in both camera views?
            this_pellet_conf = frame_pellet_conf[:, i_pellet]
            frame_pellet_reproj_error = pellet_reproj_errors[i_frame, :, i_pellet]
            if all(this_pellet_conf > min_conf) and all(frame_pellet_reproj_error < max_reproj_error):
                # the pellet was probably correctly identified in this frame for both views
                for i_cam in range(num_cameras):
                    validated_pellet_framepoints[i_frame, i_cam, i_pellet, :] = pellet_framepoints[i_frame, i_cam, i_pellet, :]
            elif all(frame_pellet_reproj_error < max_reproj_error):
                # even though low confidence in pellet location, they still seem to be a good match, put this case in for possible future modifications
                for i_cam in range(num_cameras):
                    validated_pellet_framepoints[i_frame, i_cam, i_pellet, :] = pellet_framepoints[i_frame, i_cam, i_pellet, :]
            elif frame_pellet_conf[0, i_pellet] > min_conf and frame_pellet_conf[1, other_pellet_idx] > min_conf:
                # high confidence in this pellet location in camera 1 view despite not having a good pellet 1 match in camera 2
                # what if the other pellet in camera 2 is its match? test the reprojection error for this pellet ID in camera 1
                # with the other pellet ID in camera 2

                # undistort this pellet's coordinates in camera 1
                this_pellet_ud = cv2.undistortPoints(pellet_framepoints[i_frame, 0, i_pellet, :], mtx[0], dist[0])
                # undistort the other pellet's coordinates in camera 2
                other_pellet_ud = cv2.undistortPoints(pellet_framepoints[i_frame, 1, other_pellet_idx, :], mtx[0], dist[0])

                point4D = cv2.triangulatePoints(projMatr1, projMatr2, this_pellet_ud[0][0], other_pellet_ud[0][0])
                pellet_wp = np.squeeze(cv2.convertPointsFromHomogeneous(point4D.T))

                test_frame_pts = np.zeros((num_cameras, 1, 2))
                test_frame_pts[0, 0, :] = pellet_framepoints[i_frame, 0, i_pellet, :]
                test_frame_pts[1, 0, :] = pellet_framepoints[i_frame, 1, other_pellet_idx, :]
                ppts, reproj_errors = check_3d_reprojection(pellet_wp, test_frame_pts, cal_data, dlc_metadata, pickle_metadata, i_frame, parent_directories)

                # first element in list is for pellet 1, second is for pellet 2 reprojections; first row is camera 1, second row camera 2 in each ppts array
                all_ppts[i_pellet] = ppts
                all_reproj_errors[i_pellet] = reproj_errors

                if (reproj_errors < max_reproj_error).all():
                    # pellets are accurately labeled in each view, but pellet ID's are swapped in each frame
                    # for now, assume i_pellet was correctly identified in camera 0, so other_pellet_idx should be reassigned to i_pellet in camera 2
                    validated_pellet_framepoints[i_frame, 0, i_pellet, :] = pellet_framepoints[i_frame, 0, i_pellet, :]
                    validated_pellet_framepoints[i_frame, 0, other_pellet_idx, :] = pellet_framepoints[i_frame, 0, other_pellet_idx, :]
                    validated_pellet_framepoints[i_frame, 1, i_pellet, :] = pellet_framepoints[i_frame, 1, other_pellet_idx, :]
                    validated_pellet_framepoints[i_frame, 1, other_pellet_idx, :] = pellet_framepoints[i_frame, 1,
                                                                            i_pellet, :]
                    swapped_points[i_pellet] = True

        if any(swapped_points):
            figs = []
            axs = []
            markersize = 15
            for i_cam in range(2):
                cam_dir = day_dir + '_' + 'cam{:02d}'.format(i_cam)
                # cropped_vid_folder = os.path.join(cropped_vids_parent, pickle_metadata['mouseID'], month_dir, day_dir, cam_dir)

                orig_vid_name_base = '_'.join([pickle_metadata[0]['mouseID'],
                                               pickle_metadata[0]['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                                               '{:d}'.format(pickle_metadata[0]['session_num']),
                                               '{:03d}'.format(pickle_metadata[0]['video_number']),
                                               'cam{:02d}'.format(i_cam+1)
                                               ])
                orig_vid_name = os.path.join(orig_vid_folder, orig_vid_name_base + '.avi')

                video_object = cv2.VideoCapture(orig_vid_name)

                video_object.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
                ret, cur_img = video_object.read()
                width = video_object.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
                height = video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)

                video_object.release()

                if pickle_metadata[i_cam]['isrotated']:
                    cur_img = cv2.rotate(cur_img, cv2.ROTATE_180)

                figs.append(plt.figure())
                axs.append(figs[i_cam].add_subplot(111))
                axs[i_cam].axis("off")
                axs[i_cam].set_xlim(0, width)
                axs[i_cam].set_ylim(0, height)
                axs[i_cam].invert_yaxis()

                axs[i_cam].imshow(cur_img)

                # original pellet 1 as black circle
                axs[i_cam].scatter(pellet_framepoints[i_frame, i_cam, 0, 0],
                                   pellet_framepoints[i_frame, i_cam, 0, 1], s=markersize, facecolors='None', edgecolors='k', marker='o')
                # new pellet 1 as black plus
                axs[i_cam].scatter(validated_pellet_framepoints[i_frame, i_cam, 0, 0], validated_pellet_framepoints[i_frame, i_cam, 0, 1], s=markersize, c='k', marker='+')

                # original pellet 2 as blue circle
                axs[i_cam].scatter(pellet_framepoints[i_frame, i_cam, 1, 0],
                                   pellet_framepoints[i_frame, i_cam, 1, 1], s=markersize, facecolors='None', edgecolors='b', marker='o')
                # new pellet 2 as blue plus
                axs[i_cam].scatter(validated_pellet_framepoints[i_frame, i_cam, 1, 0], validated_pellet_framepoints[i_frame, i_cam, 1, 1], s=markersize, c='b', marker='+')

                # reprojected point 1 as green star
                axs[i_cam].scatter(all_ppts[0][i_cam][0], all_ppts[0][i_cam][1], s=markersize, c='g', marker='*')
                # reprojected point 2 as red star                axs[i_cam].scatter(all_ppts[1][i_cam][0], ppts[1][i_cam][1], s=markersize, c='r', marker='*')
                axs[i_cam].scatter(all_ppts[1][i_cam][0], all_ppts[1][i_cam][1], s=markersize, c='r', marker='*')

                # if i_cam == 0:
                #     axs[i_cam].set_xlim(800, 1100)
                #     axs[i_cam].set_ylim(800, 950)
                # else:
                #     axs[i_cam].set_xlim(850, 1100)
                #     axs[i_cam].set_ylim(400, 600)
                figs[i_cam].suptitle('frame {:d}, camera {:d}'.format(i_frame, i_cam+1))
                # axs[i_cam].invert_yaxis()

            plt.show()

            mismatched_pellets[i_frame] = True

    # now assume that the pellet ID in validated_pellet_framepoints is correct at the end of the recording, and working backwards identify that pellet earlier on

    # find the last 100

    # should now have an array of points for which we have high confidence that corresponding pellet locations in both
    # views have been found

    fig = []
    ax = []
    for i_cam in range(2):
        fig.append(plt.figure())
        ax.append(fig[i_cam].subplots(3, 1))
        ax[i_cam][0].scatter(validated_pellet_framepoints[:, i_cam, 0, 0], validated_pellet_framepoints[:, i_cam, 0, 1])
        ax[i_cam][0].scatter(validated_pellet_framepoints[:, i_cam, 1, 0], validated_pellet_framepoints[:, i_cam, 1, 1])

        if i_cam == 0:
            ax[i_cam][0].set_xlim(800, 1200)
            ax[i_cam][0].set_ylim(700, 1000)
        else:
            ax[i_cam][0].set_xlim(800, 1200)
            ax[i_cam][0].set_ylim(400, 700)
        ax[i_cam][0].invert_yaxis()

        ax[i_cam][1].plot(validated_pellet_framepoints[:, i_cam, 0, 0])  # pellet 1, x
        ax[i_cam][1].plot(validated_pellet_framepoints[:, i_cam, 1, 0])  # pellet 2, x
        ax[i_cam][1].set_ylabel('x')

        ax[i_cam][2].plot(validated_pellet_framepoints[:, i_cam, 0, 1])  # pellet 1, y
        ax[i_cam][2].plot(validated_pellet_framepoints[:, i_cam, 1, 1])  # pellet 2, y
        ax[i_cam][2].set_ylabel('y')

        fig[i_cam].suptitle('camera {:02d}'.format(i_cam + 1))

    bool_fig = plt.figure()
    bool_ax = bool_fig.add_subplot(111)

    bool_ax.plot(mismatched_pellets)
    bool_fig.suptitle('mismatch between cam1 and cam2 pellet IDs')

    wp_fig = plt.figure()
    wp_ax = wp_fig.add_subplot(111, projection='3d')

    plt.show()
    pass


'''
:param frame_pts: num_cams x num_points x 2 numpy array containing (x,y) pairs of deeplabcut output rotated/
        translated into the original video frame so that the images are upright (i.e., if camera 1 is rotated 180
        degrees, the image/coordinates for camera 1 are rotated)
        '''

        # if any(pellet_euc_diffs[0, i_frame, :] > jump_thresh):   # did either pellet location jump a lot in this camera?
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.plot(pellet_euc_diffs[0, :, 0])
        #
        #     fig2 = plt.figure()
        #     ax2 = fig2.add_subplot(111)
        #     ax2.plot(frame_valid_pts[:, 0, 15])
        #     ax2.plot(frame_valid_pts[:, 0, 16])
        #
        #     #pellets 1 and 2 in camera 1
        #     fig3 = plt.figure()
        #     ax3 = fig3.add_subplot(111)
        #     # camera 1, pellet 1
        #     ax3.plot(pellet_framepoints[:, 0, 0, 0], pellet_framepoints[:, 0, 0, 1])
        #     # camera 1, pellet 2
        #     ax3.plot(pellet_framepoints[:, 0, 1, 0], pellet_framepoints[:, 0, 1, 1])
        #     ax3.set_xlim((700, 1200))
        #     ax3.set_ylim((800, 1000))
        #     ax3.invert_yaxis()
        #
        #     #pellets 1 and 2 in camera 2
        #     fig4 = plt.figure()
        #     ax4 = fig4.add_subplot(111)
        #     # camera 2, pellet 1
        #     ax4.plot(pellet_framepoints[:, 1, 0, 0], pellet_framepoints[:, 1, 0, 1])
        #     # camera 2, pellet 2
        #     ax4.plot(pellet_framepoints[:, 1, 1, 0], pellet_framepoints[:, 1, 1, 1])
        #     ax4.set_xlim((700, 1200))
        #     ax4.set_ylim((450, 600))
        #     ax4.invert_yaxis()
        #
        #     plt.show()
        #     pass


def calculate_interframe_point_jumps(r3d_data):
    '''
    calculate the euclidean distance in points for each bodypart in adjacent frames
    :param r3d_data:
    :return:
    '''

    frame_points = r3d_data['frame_points']

    num_frames = np.shape(frame_points)[0]
    num_cams = np.shape(frame_points)[1]
    pts_per_frame = np.shape(frame_points)[2]

    pt_euc_diffs = np.zeros((num_cams, num_frames-1, pts_per_frame))
    for i_cam in range(num_cams):
        # calculate difference across frames in (x, y) point locations
        pt_diffs = np.diff(frame_points[:, i_cam, :, :], n=1, axis=0)

        # calculate euclidean distance between points in adjacent frames
        pt_euc_diffs[i_cam, :, :] = np.sqrt(np.sum(np.square(pt_diffs), axis=2))

    return pt_euc_diffs


def identify_valid_points_in_frames(r3d_data, max_reproj_error=20, min_conf=0.9):
    '''
    identifies invalid points based on whether confidence is too low and/or reprojection error is too large

    :param r3d_data: 3d reconstruction data. dictionary with the following keys:
        frame_points - num_frames x num_cams x num_points x 2 array containing points identified by dlc in each frame
            translated/rotated into the original video frames (but rotated so all frames are upright)
    :param max_reproj_error: maximum allowable projection from world points back into each camera view
    :param min_conf:
    :return: frame_valid_points: num_frames x num_cams x num_points boolean array containing True for each frame-
        camera-view-point that is valid based on reprojection error and dlc confidence
    '''
    reproj_errors = r3d_data['reprojection_errors']
    dlc_conf = r3d_data['frame_confidence']

    num_frames = np.shape(reproj_errors)[0]
    num_cams = np.shape(reproj_errors)[1]
    pts_per_frame = np.shape(reproj_errors)[2]

    frame_valid_points = np.ones((num_frames, num_cams, pts_per_frame), dtype=bool)

    for i_frame in range(num_frames):

        # identify frames/camera views in which reprojection error is too large
        frame_reproj_errors = np.squeeze(reproj_errors[i_frame, :, :])
        invalid_reprojection = frame_reproj_errors > max_reproj_error
        # if reprojection error is too large into either camera view, invalidate point in both frames
        valid_reprojections = np.logical_not(np.any(invalid_reprojection, 0))

        reprojection_valid = np.tile(valid_reprojections, (num_cams, 1))

        # identify valid points based on dlc confidence
        frame_conf = np.squeeze(dlc_conf[i_frame, :, :])
        conf_valid = frame_conf > min_conf

        frame_valid_points[i_frame, : , :] = np.logical_and(reprojection_valid, conf_valid)

        # may have to work separately on the pellets, since pellet1/2 could switch between which pellet was labeled in
        # each frame

    return frame_valid_points


def test_optitrack_reconstruction(parent_directories):

    reconstruct3d_parent = parent_directories['reconstruct3d_parent']

    r3d_directories = navigation_utilities.get_optitrack_r3d_folders(reconstruct3d_parent)

    for rd in r3d_directories:
        test_singlefolder_optitrack_reconstruction(rd, parent_directories)


def test_singlefolder_optitrack_reconstruction(rd, parent_directories):

    r3d_files = navigation_utilities.find_optitrack_r3d_files(rd)

    for r3d_file in r3d_files:
        test_single_optitrack_trajectory(r3d_file, parent_directories)
    pass


def test_single_optitrack_trajectory(r3d_file, parent_directories):

    video_root_folder = parent_directories['video_root_folder']
    cropped_vids_parent = parent_directories['cropped_vids_parent']
    r3d_metadata = navigation_utilities.parse_3d_reconstruction_pickle_name(r3d_file)
    r3d_data = skilled_reaching_io.read_pickle(r3d_file)

    # todo: create a movie of 3d reconstruction with video of points super-imposed on videos (also show reprojection errors?)
    # also pick out some specific frames

    orig_videos = navigation_utilities.find_original_optitrack_videos(video_root_folder, r3d_metadata)
    cropped_videos = navigation_utilities.find_cropped_optitrack_videos(cropped_vids_parent, r3d_metadata)

    # crop regions is a 2-element list of tuples - first tuple is borders for direct view, second set is for mirror view
    # each tuple is four elements: left, right, top, bottom
    # direct_crop = (750, 1250, 500, 900)
    # leftmirror_crop = (0, 400, 400, 800)
    # rightmirror_crop = (1650, 2039, 400, 800)

    sr_visualization.animate_optitrack_vids_plus3d(r3d_data, cropped_videos)
    pass


def projection_from_E(E, cal_data):

    R1, R2, t = cv2.decomposeEssentialMat(E)

    return R1, R2, t



def projection_from_F(F, mtx):

    pass