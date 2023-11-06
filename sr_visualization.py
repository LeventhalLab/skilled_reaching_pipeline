import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import shutil
import navigation_utilities
import reconstruct_3d_optitrack
import computer_vision_basics as cvb
import subprocess

def plot_3d_skeleton(paw_trajectory, bodyparts, ax=None, trail_pts=3):

    pass


def overlay_pts_on_video(paw_trajectory, cal_data, bodyparts, orig_vid_name, crop_region, frame_num, ax=None, trail_pts=3):

    pass

def create_vids_plus_3danimation_figure(figsize=(18, 10), num_views=2, dpi=100.):
    '''

    :param figsize:
    :param dpi:
    :return:
    '''
    fig = plt.figure(figsize=figsize, dpi=dpi)

    axs = []
    for i_ax in range(num_views):
        axs.append(fig.add_subplot(1, num_views+1, i_ax))

    axs.append(fig.add_subplot(1, num_views+1, num_views+1, projection='3d'))

    for ax in axs[:num_views]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    axs[num_views].xaxis.set_ticks([])
    axs[num_views].yaxis.set_ticks([])
    axs[num_views].zaxis.set_ticks([])

    axs[num_views].set_xlabel('x')
    axs[num_views].set_ylabel('y')
    axs[num_views].set_zlabel('z')

    return fig, axs


def animate_vids_plus3d(traj_data, crop_regions, orig_video_name):

    fig, axs = create_vids_plus_3danimation_figure()

    bp_coords_ud = traj_data['bp_coords_ud']
    vid_obj = cv2.VideoCapture(orig_video_name)

    num_vid_frames = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
    num_data_frames = np.shape(bp_coords_ud[0])[1]

    if num_vid_frames != num_data_frames:
        print('frame number mismatch for {}'.format(orig_video_name))
        vid_obj.release()
        return

    cal_data = traj_data['cal_data']
    mtx = cal_data['mtx']
    dist = cal_data['dist']
    num_views = len(crop_regions)
    bodyparts = traj_data['bodyparts']
    bpts2connect = rat_sr_bodyparts2connect()
    for i_frame in range(num_data_frames):

        # read in the first image
        ret, img = vid_obj.read()

        # undistort the image
        img_ud = cv2.undistort(img, mtx, dist)

        for i_view in range(num_views):

            cw = crop_regions[i_view]
            show_crop_frame_with_pts()
            cropped_img = img_ud[cw[2]:cw[3], cw[0]:cw[1], :]


        plt.show()
        pass

    vid_obj.release()


def animate_optitrack_vids_plus3d(r3d_data, orig_videos, cropped_videos, parent_directories):
    '''

    :param r3d_data: dictionary containing the following keys:
        frame_points: undistorted points in the original video coordinate system
    :param cropped_videos:
    :return:
    '''
    reconstruct_3d_parent = parent_directories['reconstruct3d_parent']

    cv_params = [navigation_utilities.parse_cropped_optitrack_video_name(cv_name) for cv_name in cropped_videos]
    animation_name = navigation_utilities.mouse_animation_name(cv_params[0], reconstruct_3d_parent)
    _, an_name = os.path.split(animation_name)

    animation_folder, animation_name_only = os.path.split(animation_name)
    # animation_name_E = animation_name.replace('animation', 'animation_E')
    # animation_name_F = animation_name.replace('animation', 'animation_F')

    animation_name_recal = animation_name.replace('animation', 'animation_recal')
    # comment out to overwrite old videos
    # if os.path.exists(animation_name_E) and os.path.exists(animation_name_F):
    #     print('{} already exists'.format(an_name))
    #     return True   # for now, only make one animation per folder just to get a look at if reconstruction looks good

    if os.path.exists(animation_name_recal):
        print('{} already exists'.format(an_name))
        return True   # for now, only make one animation per folder just to get a look at if reconstruction looks good

    # jpg_folder_E = os.path.join(animation_folder, 'temp_E')
    # jpg_folder_F = os.path.join(animation_folder, 'temp_F')
    jpg_folder_recal = os.path.join(animation_folder, 'temp_recal')
    # if os.path.isdir(jpg_folder_E):
    #     shutil.rmtree(jpg_folder_E)
    # os.mkdir(jpg_folder_E)
    # if os.path.isdir(jpg_folder_F):
    #     shutil.rmtree(jpg_folder_F)
    if os.path.isdir(jpg_folder_recal):
        shutil.rmtree(jpg_folder_recal)
    os.mkdir(jpg_folder_recal)

    num_cams = np.shape(r3d_data['frame_points'])[1]
    show_undistorted = r3d_data['cal_data']['use_undistorted_pts_for_stereo_cal']

    cv_cam_nums = [cvp['cam_num'] for cvp in cv_params]
    im_size = r3d_data['cal_data']['im_size']
    fullframe_pts = [np.squeeze(r3d_data['frame_points'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    fullframe_pts_ud = [np.squeeze(r3d_data['frame_points_ud'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    # reprojected_pts_E = [np.squeeze(r3d_data['reprojected_points_E'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    # reprojected_pts_F = [np.squeeze(r3d_data['reprojected_points_F'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    reprojected_pts_recal = [np.squeeze(r3d_data['reprojected_points_recal'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    # wpts_E = r3d_data['worldpoints_E']
    # wpts_F = r3d_data['worldpoints_F']
    wpts_recal = r3d_data['worldpoints_recal']

    bodyparts = r3d_data['bodyparts']
    num_frames = np.shape(r3d_data['frame_points'])[0]

    # create video capture objects to read frames
    vid_cap_objs = []
    # cropped_im_size = []
    crop_wins = []

    bpts2connect = mouse_sr_bodyparts2connect()
    cropped_vid_metadata = []
    isrotated = []
    im_sizes = []
    for i_cam in range(num_cams):

        # todo: pull image from original video, then undistort, then crop images and overlay undistorted and unnormalized points
        # this should find the index of camera number i_cam + 1 (1 or 2) in the cv_cam_nums list to make sure the vid_cap_objs are in the same order as r3d_data
        # vid_cap_objs.append(cv2.VideoCapture(cropped_videos[cv_cam_nums.index(i_cam + 1)]))

        vid_cap_objs.append(cv2.VideoCapture(orig_videos[cv_cam_nums.index(i_cam + 1)]))

        w = vid_cap_objs[i_cam].get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vid_cap_objs[i_cam].get(cv2.CAP_PROP_FRAME_HEIGHT)
        im_sizes.append((w, h))
        # cropped_im_size.append((w, h))   # may not need this
        cropped_vid_metadata.append(navigation_utilities.parse_cropped_optitrack_video_name(cropped_videos[i_cam]))
        crop_wins.append(cropped_vid_metadata[i_cam]['crop_window'])   # subtract one to line up with python indexing
        # crop_wins.append(np.array([0, w, 0, h], dtype=int))
        isrotated.append(cropped_vid_metadata[i_cam]['isrotated'])

    for i_frame in range(num_frames):
        print('working on {}, frame {:04d}'.format(animation_name_only, i_frame))

        # fig_E, axs_E = create_vids_plus_3danimation_figure()  # todo: add in options to size image axes depending on vid size
        # fig_F, axs_F = create_vids_plus_3danimation_figure()
        fig_recal, axs_recal = create_vids_plus_3danimation_figure()

        fullframe_pts_forthisframe = [fullframe_pts[i_cam][i_frame, :, :] for i_cam in range(num_cams)]
        fullframe_pts_ud_forthisframe = [fullframe_pts_ud[i_cam][i_frame, :, :] for i_cam in range(num_cams)]
        valid_3dpoints = identify_valid_3dpts(fullframe_pts_forthisframe, crop_wins, im_sizes, isrotated)

        # jpg_name_E = os.path.join(jpg_folder_E, 'frame{:04d}.jpg'.format(i_frame))
        # jpg_name_F = os.path.join(jpg_folder_F, 'frame{:04d}.jpg'.format(i_frame))
        jpg_name_recal = os.path.join(jpg_folder_recal, 'frame{:04d}.jpg'.format(i_frame))
        for i_cam in range(num_cams):

            # cur_fullframe_reproj_pts_E = reprojected_pts_E[i_cam][i_frame, :, :]
            # cur_fullframe_reproj_pts_F = reprojected_pts_F[i_cam][i_frame, :, :]
            cur_fullframe_reproj_pts_recal = reprojected_pts_recal[i_cam][i_frame, :, :]
            cur_fullframe_pts = fullframe_pts_ud_forthisframe[i_cam]
            crop_params = cv_params[i_cam]['crop_window']
            translated_frame_points = reconstruct_3d_optitrack.optitrack_fullframe_to_cropped_coords(cur_fullframe_pts, crop_params, im_size[i_cam], isrotated[i_cam])
            # translated_reproj_points_E = reconstruct_3d_optitrack.optitrack_fullframe_to_cropped_coords(cur_fullframe_reproj_pts_E, crop_params, im_size[i_cam], isrotated[i_cam])
            # translated_reproj_points_F = reconstruct_3d_optitrack.optitrack_fullframe_to_cropped_coords(
            #     cur_fullframe_reproj_pts_F, crop_params, im_size[i_cam], isrotated[i_cam])
            translated_reproj_points_recal = reconstruct_3d_optitrack.optitrack_fullframe_to_cropped_coords(
                cur_fullframe_reproj_pts_recal, crop_params, im_size[i_cam], isrotated[i_cam])

            vid_cap_objs[i_cam].set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            ret, img = vid_cap_objs[i_cam].read()

            crop_win = crop_wins[i_cam]
            if show_undistorted:
                mtx = r3d_data['cal_data']['mtx'][i_cam]
                dist = r3d_data['cal_data']['dist'][i_cam]
                cropped_img = undistort2cropped(img, mtx, dist, crop_win, isrotated[i_cam])
            else:
                cropped_img = img[crop_win[2]:crop_win[3], crop_win[0]:crop_win[1], :]
                if isrotated[i_cam]:
                    cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_180)

            # overlay points, check that they match with cropped vids
            cw = [0, np.shape(cropped_img)[1], 0, np.shape(cropped_img)[0]]
            # show_crop_frame_with_pts(cropped_img, cw, translated_frame_points, bodyparts, bpts2connect, valid_3dpoints, axs_E[i_cam], marker='o', s=6)
            # show_crop_frame_with_pts(cropped_img, cw, translated_frame_points, bodyparts, bpts2connect, valid_3dpoints,
            #                          axs_F[i_cam], marker='o', s=6)
            # show_crop_frame_with_pts(cropped_img, cw, translated_reproj_points_E, bodyparts, [], valid_3dpoints,
            #                          axs_E[i_cam], marker='s', s=6)
            # show_crop_frame_with_pts(cropped_img, cw, translated_reproj_points_F, bodyparts, [], valid_3dpoints,
            #                          axs_F[i_cam], marker='s', s=6)

            show_crop_frame_with_pts(cropped_img, cw, translated_frame_points, bodyparts, bpts2connect, valid_3dpoints,
                                     axs_recal[i_cam], marker='o', s=6)
            show_crop_frame_with_pts(cropped_img, cw, translated_reproj_points_recal, bodyparts, [], valid_3dpoints,
                                     axs_recal[i_cam], marker='s', s=6)

        # make the 3d plot
        # cur_wpts_E = np.squeeze(wpts_E[i_frame, :, :])
        # cur_wpts_F = np.squeeze(wpts_F[i_frame, :, :])
        cur_wpts_recal = np.squeeze(wpts_recal[i_frame, :, :])
        bpts2connect_3d = mouse_sr_bodyparts2connect_3d()
        # plot_frame3d(cur_wpts_E, valid_3dpoints, bodyparts, bpts2connect_3d, axs_E[2])
        # plot_frame3d(cur_wpts_F, valid_3dpoints, bodyparts, bpts2connect_3d, axs_F[2])
        plot_frame3d(cur_wpts_recal, valid_3dpoints, bodyparts, bpts2connect_3d, axs_recal[2])

        # fig_E.savefig(jpg_name_E)
        # fig_F.savefig(jpg_name_F)
        fig_recal.savefig(jpg_name_recal)
        plt.close('all')
        # plt.show()

    # # turn the cropped jpegs into a new movie
    # jpg_names_E = os.path.join(jpg_folder_E, 'frame%04d.jpg')
    # command = (
    #     f"ffmpeg -i {jpg_names_E} "
    #     f"-c:v copy {animation_name_E}"
    # )
    # subprocess.call(command, shell=True)
    # 
    # # delete the temp folder to hold frame jpegs
    # shutil.rmtree(jpg_folder_E)

    # turn the cropped jpegs into a new movie
    # jpg_names_F = os.path.join(jpg_folder_F, 'frame%04d.jpg')
    # command = (
    #     f"ffmpeg -i {jpg_names_F} "
    #     f"-c:v copy {animation_name_F}"
    # )
    # subprocess.call(command, shell=True)
    # 
    # # delete the temp folder to hold frame jpegs
    # shutil.rmtree(jpg_folder_F)
    
    # turn the cropped jpegs into a new movie
    jpg_names_recal = os.path.join(jpg_folder_recal, 'frame%04d.jpg')
    command = (
        f"ffmpeg -i {jpg_names_recal} "
        f"-c:v copy {animation_name_recal}"
    )
    subprocess.call(command, shell=True)

    # delete the temp folder to hold frame jpegs
    shutil.rmtree(jpg_folder_recal)

    return True


def undistort2cropped(img, mtx, dist, crop_win, isrotated):
    '''
    undistort frame from original video, then crop
    :param img:
    :param mtx:
    :param dist:
    :param crop_win: should be [left, right, top, bottom]
    :param isrotated:
    :return:
    '''

    if isrotated:
        # rotate before undistorting
        img = cv2.rotate(img, cv2.ROTATE_180)

    img_ud = cv2.undistort(img, mtx, dist)

    if isrotated:
        # rotate back before cropping
        img_ud = cv2.rotate(img_ud, cv2.ROTATE_180)

    cropped_img = img_ud[crop_win[2]:crop_win[3], crop_win[0]:crop_win[1], :]

    if isrotated:
        # rotate back before cropping
        cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_180)

    return cropped_img


def identify_valid_3dpts(framepts_forallcams, crop_wins, im_sizes, isrotated):
    '''

    :param framepts_forallcams:
    :param crop_wins: format [left, right, top, bottom]. This is BEFORE ROTATION of the image if this was an upside-down
        camera
    :param im_sizes:
    :param isrotated: list indicating whether this camera was rotated (currently should be [True, False] since camera 1
        was physically rotated and camera 2 was not
    :return:
    '''
    num_bp = np.shape(framepts_forallcams[0])[0]
    num_cams = len(crop_wins)
    valid_cam_pt = np.zeros((num_bp, 2), dtype=bool)

    for i_cam in range(num_cams):
        if isrotated[i_cam]:
            # if image is rotated, top left corner of cropped image will be bottom right corner of full image
            crop_edge = cvb.rotate_pts_180([crop_wins[i_cam][1], crop_wins[i_cam][3]], im_sizes[i_cam])
        else:
            # if image is not rotated, top left corner of cropped image will be top left corner of full image
            crop_edge = np.array([crop_wins[i_cam][0], crop_wins[i_cam][2]])
        for i_bp in range(num_bp):

            # check each camera view to see if x = y = 0, indicating that point was not correctly identified in that view
            # frame_pt_test = [all(cam_framepts[i_bp, :] - crop_wins[i_cam][:1] == 0) for i_cam, cam_framepts in enumerate(framepts_forallcams)]

            valid_cam_pt[i_bp, i_cam] = any(framepts_forallcams[i_cam][i_bp, :] - crop_edge != 0)

            # if any(frame_pt_test):
            #     continue
            # valid_3dpts[i_bp] = True

    valid_3dpts = np.logical_and(valid_cam_pt[:, 0], valid_cam_pt[:, 1])

    return valid_3dpts


def plot_frame3d(worldpoints, valid_3dpoints, bodyparts, bpts2connect, ax3d, **kwargs):
    bp_c = mouse_bp_colors_3d()
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('s', 3)

    for i_pt, pt in enumerate(worldpoints):

        if valid_3dpoints[i_pt]:

            if len(pt) > 0:
                try:
                    x, y, z = pt[0]
                except:
                    x, y, z = pt
                # x = int(round(x))
                # y = int(round(y))
                kwargs['c'] = bp_c[bodyparts[i_pt]]

                ax3d.scatter(x, y, z, **kwargs)

    connect_bodyparts_3d(worldpoints, bodyparts, bpts2connect, valid_3dpoints, ax3d)

    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    # ax3d.set_xlim(20, 60)
    # ax3d.set_ylim(20, 60)
    # ax3d.set_zlim(100, 150)
    ax3d.invert_yaxis()


def show_crop_frame_with_pts(img, cw, frame_pts, bodyparts, bpts2connect, valid_3dpoints, ax, **kwargs):
    '''

    :param img:
    :param cw: crop window - [left, right, top, bottom]
    :param frame_pts:
    :param bodyparts:
    :param bpts2connect:
    :param valid_3dpoints:
    :param ax:
    :param kwargs:
    :return:
    '''
    if img.ndim == 2:
        # 2-d array for grayscale image
        cropped_img = img[cw[2]:cw[3], cw[0]:cw[1]]
    elif img.ndim == 3:
        # color image
        cropped_img = img[cw[2]:cw[3], cw[0]:cw[1], :]

    ax.imshow(cropped_img)

    overlay_pts(frame_pts, bodyparts, valid_3dpoints, ax, **kwargs)

    connect_bodyparts(frame_pts, bodyparts, bpts2connect, valid_3dpoints, ax)


def overlay_pts(pts, bodyparts, plot_point_bool, ax, **kwargs):
    '''

    :param pts:
    :param bodyparts:
    :param plot_point_bool:
    :param ax:
    :param kwargs:
    :return:
    '''
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('s', 3)
    bp_c = mouse_bp_colors()

    for i_pt, pt in enumerate(pts):
        if plot_point_bool[i_pt]:
            pt = np.squeeze(pt)
            if all(pt == 0):
                continue    # [0, 0] are points that weren't properly identified
            kwargs['c'] = bp_c[bodyparts[i_pt]]
            ax.scatter(pt[0], pt[1], **kwargs)


def draw_epipolar_lines_on_img(img_pts, whichImage, F, im_size, bodyparts, plot_point_bool, ax, lwidth=0.5, linestyle='-'):

    epilines = cv2.computeCorrespondEpilines(img_pts, whichImage, F)
    bp_c = mouse_bp_colors()

    for i_line, epiline in enumerate(epilines):

        if plot_point_bool[i_line]:
            bp_color = bp_c[bodyparts[i_line]]    # color_from_bodypart(bodyparts[i_line])
            epiline = np.squeeze(epiline)
            edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

            if not np.all(edge_pts == 0):
                ax.plot(edge_pts[:, 0], edge_pts[:, 1], color=bp_color, ls=linestyle, marker='.', lw=lwidth)


def overlay_pts_on_original_frame(frame_pts, pts_conf, campickle_metadata, camdlc_metadata, frame_num, cal_data, parent_directories,
                                  ax, plot_undistorted=True, frame_pts_already_undistorted=False, min_conf=0.98, **kwargs):
    '''

    :param frame_pts:
    :param campickle_metadata: a single pickle_metadata structure
    :param camdlc_metadata:
    :param frame_num:
    :param cal_data:
    :param parent_directories:
    :param ax:
    :param kwargs:
    :return:
    '''

    cam_num = campickle_metadata['cam_num']
    video_root_folder = parent_directories['video_root_folder']

    bodyparts = camdlc_metadata['data']['DLC-model-config file']['all_joints_names']
    mouseID = campickle_metadata['mouseID']
    day_dir = mouseID + '_' + campickle_metadata['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, mouseID, day_dir)

    orig_vid_name_base = '_'.join([campickle_metadata['prefix'] + mouseID,
                             campickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                             '{:d}'.format(campickle_metadata['session_num']),
                             '{:03d}'.format(campickle_metadata['vid_num']),
                             'cam{:02d}.avi'.format(cam_num)
                             ])

    orig_vid_name = os.path.join(orig_vid_folder, orig_vid_name_base)

    if not os.path.exists(orig_vid_name):
        # sometimes session number has 2 digits, sometimes one
        orig_vid_name_base = '_'.join([campickle_metadata['prefix'] + mouseID,
                                       campickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                                       '{:02d}'.format(campickle_metadata['session_num']),
                                       '{:03d}'.format(campickle_metadata['vid_num']),
                                       'cam{:02d}.avi'.format(cam_num)
                                       ])
        orig_vid_name = os.path.join(orig_vid_folder, orig_vid_name_base)

    #read in image
    video_object = cv2.VideoCapture(orig_vid_name)

    video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, img = video_object.read()

    mtx = cal_data['mtx'][cam_num-1]
    dist = cal_data['dist'][cam_num-1]

    if cam_num == 1:
        img = cv2.rotate(img, cv2.ROTATE_180)
    img_ud = cv2.undistort(img, mtx, dist)

    video_object.release()

    h, w, _ = np.shape(img_ud)
    im_size = (w, h)

    if plot_undistorted:
        # undistorted image
        ax.imshow(img_ud)
    else:
        # distorted original image
        ax.imshow(img)

    if not frame_pts_already_undistorted:
        pt_ud_norm = np.squeeze(cv2.undistortPoints(frame_pts, mtx, dist))
        pt_ud = cvb.unnormalize_points(pt_ud_norm, mtx)
    else:
        pt_ud = frame_pts

    num_pts = np.shape(frame_pts)[0]
    if num_pts == 1:
        # only one point
        pt_ud = [pt_ud]

    if plot_undistorted:
        to_plot = pt_ud
    else:
        to_plot = frame_pts


    plot_point_bool = pts_conf > min_conf   # np.ones((num_pts, 1), dtype=bool)
    overlay_pts(to_plot, bodyparts, plot_point_bool, ax, **kwargs)

    return im_size


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


def connect_bodyparts(frame_pts, bodyparts, bpts2connect, valid_3dpoints, ax, **kwargs):
    '''
    add lines connecting body parts to video frames showing marked bodypart points
    :param frame_pts: n x 2 numpy array where n is the number of points in the frame
    :param bodyparts: n-element list of bodypart names in order corresponding to frame_pts
    :param bpts2connect: list of 2-element lists containing pairs of body parts to connect with lines (named according to bodyparts)
    :param ax: axes on which to make the plot
    :param linecolor: color of connecting lines, default gray
    :param lwidth: width of connecting lines - default 1.5 (pyplot default)
    :return:
    '''
    kwargs.setdefault('c', (0.5, 0.5, 0.5))
    kwargs.setdefault('lw', 1.5)
    for pt2connect in bpts2connect:

        pt_index = [bodyparts.index(bp_name) for bp_name in pt2connect]

        if all(valid_3dpoints[pt_index]):

            if all(frame_pts[pt_index[0], :] == 0) or all(frame_pts[pt_index[1], :] == 0):
                continue   # one of the points wasn't found
            x = frame_pts[pt_index, 0]
            y = frame_pts[pt_index, 1]
            ax.plot(x, y, **kwargs)


def connect_bodyparts_3d(worldpoints, bodyparts, bpts2connect, valid_3dpoints, ax3d, **kwargs):
    '''
    add lines connecting body parts to video frames showing marked bodypart points
    :param frame_pts: n x 2 numpy array where n is the number of points in the frame
    :param bodyparts: n-element list of bodypart names in order corresponding to frame_pts
    :param bpts2connect: list of 2-element lists containing pairs of body parts to connect with lines (named according to bodyparts)
    :param ax: axes on which to make the plot
    :param linecolor: color of connecting lines, default gray
    :param lwidth: width of connecting lines - default 1.5 (pyplot default)
    :return:
    '''
    kwargs.setdefault('c', (0.5, 0.5, 0.5))
    kwargs.setdefault('lw', 1.5)
    for pt2connect in bpts2connect:

        pt_index = [bodyparts.index(bp_name) for bp_name in pt2connect]

        if all(valid_3dpoints[pt_index]):

            x = worldpoints[pt_index, 0]
            y = worldpoints[pt_index, 1]
            z = worldpoints[pt_index, 2]
            ax3d.plot(x, y, z, **kwargs)


def rat_sr_bodyparts2connect():

    bpts2connect = []

    bpts2connect.append(['leftelbow', 'leftpawdorsum'])

    bpts2connect.append(['leftpawdorsum', 'leftmcp1'])
    bpts2connect.append(['leftpawdorsum', 'leftmcp2'])
    bpts2connect.append(['leftpawdorsum', 'leftmcp3'])
    bpts2connect.append(['leftpawdorsum', 'leftmcp4'])

    bpts2connect.append(['leftmcp1', 'leftpip1'])
    bpts2connect.append(['leftmcp2', 'leftpip2'])
    bpts2connect.append(['leftmcp3', 'leftpip3'])
    bpts2connect.append(['leftmcp4', 'leftpip4'])

    bpts2connect.append(['leftpip1', 'leftdig1'])
    bpts2connect.append(['leftpip2', 'leftdig2'])
    bpts2connect.append(['leftpip3', 'leftdig3'])
    bpts2connect.append(['leftpip4', 'leftdig4'])

    bpts2connect.append(['rightelbow', 'rightpawdorsum'])

    bpts2connect.append(['rightpawdorsum', 'rightmcp1'])
    bpts2connect.append(['rightpawdorsum', 'rightmcp2'])
    bpts2connect.append(['rightpawdorsum', 'rightmcp3'])
    bpts2connect.append(['rightpawdorsum', 'rightmcp4'])

    bpts2connect.append(['rightmcp1', 'rightpip1'])
    bpts2connect.append(['rightmcp2', 'rightpip2'])
    bpts2connect.append(['rightmcp3', 'rightpip3'])
    bpts2connect.append(['rightmcp4', 'rightpip4'])

    bpts2connect.append(['rightpip1', 'rightdig1'])
    bpts2connect.append(['rightpip2', 'rightdig2'])
    bpts2connect.append(['rightpip3', 'rightdig3'])
    bpts2connect.append(['rightpip4', 'rightdig4'])

    return bpts2connect


def mouse_sr_bodyparts2connect():

    bpts2connect = []

    bpts2connect.append(['leftpaw', 'leftdigit1'])
    bpts2connect.append(['leftpaw', 'leftdigit2'])
    bpts2connect.append(['leftpaw', 'leftdigit3'])
    bpts2connect.append(['leftpaw', 'leftdigit4'])

    bpts2connect.append(['rightpaw', 'rightdigit1'])
    bpts2connect.append(['rightpaw', 'rightdigit2'])
    bpts2connect.append(['rightpaw', 'rightdigit3'])
    bpts2connect.append(['rightpaw', 'rightdigit4'])

    return bpts2connect


def mouse_sr_bodyparts2connect_3d():

    bpts2connect = []

    bpts2connect.append(['leftpaw', 'leftdigit1'])
    bpts2connect.append(['leftpaw', 'leftdigit2'])
    bpts2connect.append(['leftpaw', 'leftdigit3'])
    bpts2connect.append(['leftpaw', 'leftdigit4'])

    bpts2connect.append(['rightpaw', 'rightdigit1'])
    bpts2connect.append(['rightpaw', 'rightdigit2'])
    bpts2connect.append(['rightpaw', 'rightdigit3'])
    bpts2connect.append(['rightpaw', 'rightdigit4'])

    bpts2connect.append(['leftear', 'lefteye'])
    bpts2connect.append(['rightear', 'righteye'])
    bpts2connect.append(['lefteye', 'nose'])
    bpts2connect.append(['righteye', 'nose'])

    return bpts2connect


def rat_bp_colors():

    bp_c = {'leftear':(0, 1, 1)}
    bp_c['rightear'] = tuple(np.array(bp_c['leftear']) * 0.5)

    bp_c['lefteye'] = (1, 0, 1)
    bp_c['righteye'] = tuple(np.array(bp_c['lefteye']) * 0.5)

    bp_c['nose'] = (1, 1, 1)

    bp_c['leftelbow'] = (1, 1, 0)
    bp_c['rightelbow'] = tuple(np.array(bp_c['leftelbow']) * 0.5)

    bp_c['rightpawdorsum'] = (0, 0, 1)
    bp_c['rightpalm'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.5)
    bp_c['rightmcp1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightmcp2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightmcp3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightmcp4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['rightpip1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightpip2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightpip3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightpip4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['rightdig1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightdig2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightdig3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightdig4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['leftpawdorsum'] = (1, 0, 0)
    bp_c['leftpalm'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.5)
    bp_c['leftmcp1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftmcp2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftmcp3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftmcp4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['leftpip1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftpip2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftpip3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftpip4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['leftdig1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftdig2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftdig3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftdig4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['pellet1'] = (0, 0, 0)
    bp_c['pellet2'] = (0.1, 0.1, 0.1)
    bp_c['pellet3'] = (0.2, 0.2, 0.2)

    return bp_c


def mouse_bp_colors():

    bp_c = {'leftear':(0, 1, 1)}
    bp_c['rightear'] = tuple(np.array(bp_c['leftear']) * 0.5)

    bp_c['lefteye'] = (1, 0, 1)
    bp_c['righteye'] = tuple(np.array(bp_c['lefteye']) * 0.5)

    bp_c['nose'] = (0, 0, 0)

    bp_c['rightpaw'] = (0, 0, 1)
    bp_c['rightdigit1'] = tuple(np.array(bp_c['rightpaw']) * 0.9)
    bp_c['rightdigit2'] = tuple(np.array(bp_c['rightpaw']) * 0.8)
    bp_c['rightdigit3'] = tuple(np.array(bp_c['rightpaw']) * 0.7)
    bp_c['rightdigit4'] = tuple(np.array(bp_c['rightpaw']) * 0.6)

    bp_c['leftpaw'] = (1, 0, 0)
    bp_c['leftdigit1'] = tuple(np.array(bp_c['leftpaw']) * 0.9)
    bp_c['leftdigit2'] = tuple(np.array(bp_c['leftpaw']) * 0.8)
    bp_c['leftdigit3'] = tuple(np.array(bp_c['leftpaw']) * 0.7)
    bp_c['leftdigit4'] = tuple(np.array(bp_c['leftpaw']) * 0.6)

    bp_c['pellet1'] = (0, 0, 0)
    bp_c['pellet2'] = (0.1, 0.1, 0.1)

    return bp_c


def mouse_bp_colors_3d():

    bp_c = {'leftear':(0, 1, 1)}
    bp_c['rightear'] = tuple(np.array(bp_c['leftear']) * 0.5)

    bp_c['lefteye'] = (1, 0, 1)
    bp_c['righteye'] = tuple(np.array(bp_c['lefteye']) * 0.5)

    bp_c['nose'] = (0, 0, 0)

    bp_c['rightpaw'] = (0, 0, 1)
    bp_c['rightdigit1'] = tuple(np.array(bp_c['rightpaw']) * 0.9)
    bp_c['rightdigit2'] = tuple(np.array(bp_c['rightpaw']) * 0.8)
    bp_c['rightdigit3'] = tuple(np.array(bp_c['rightpaw']) * 0.7)
    bp_c['rightdigit4'] = tuple(np.array(bp_c['rightpaw']) * 0.6)

    bp_c['leftpaw'] = (1, 0, 0)
    bp_c['leftdigit1'] = tuple(np.array(bp_c['leftpaw']) * 0.9)
    bp_c['leftdigit2'] = tuple(np.array(bp_c['leftpaw']) * 0.8)
    bp_c['leftdigit3'] = tuple(np.array(bp_c['leftpaw']) * 0.7)
    bp_c['leftdigit4'] = tuple(np.array(bp_c['leftpaw']) * 0.6)

    bp_c['pellet1'] = (0, 0, 0)
    bp_c['pellet2'] = (0.1, 0.1, 0.1)

    return bp_c