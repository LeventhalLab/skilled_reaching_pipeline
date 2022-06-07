import os
import glob
import sys
import cv2
import pandas as pd
from datetime import datetime


def get_video_folders_to_crop(video_root_folder):
    """
    find all the lowest level directories within video_root_folder, which are presumably the lowest level folders that
    contain the videos to be cropped

    :param video_root_folder: root directory from which to extract the list of folders that contain videos to crop
    :return: crop_dirs - list of lowest level directories within video_root_folder
    """

    crop_dirs = []

    # assume that any directory that does not have a subdirectory contains videos to crop
    for root, dirs, files in os.walk(video_root_folder):
        if not dirs:
            crop_dirs.append(root)

    return crop_dirs


def find_original_optitrack_video(video_root_folder, metadata):

    mouseID = metadata['mouseID']
    trialtime =  metadata['trialtime']
    month_dir = mouseID + '_' + trialtime.strftime('%Y%m')
    date_dir = mouseID + '_' + trialtime.strftime('%Y%m%d')

    vid_name = '_'.join([mouseID,
                         trialtime.strftime('%Y%m%d_%H-%M-%S'),
                         str(metadata['session_num']),
                         '{:03d}'.format(metadata['video_number']),
                         'cam{:02d}.avi'.format(metadata['cam_num'])])

    full_vid_name = os.path.join(video_root_folder, mouseID, month_dir, date_dir, vid_name)

    if not os.path.exists(full_vid_name):
        full_vid_name = None

    return full_vid_name


def create_cropped_video_destination_list(cropped_vids_parent, video_folder_list, view_list):
    """
    create subdirectory trees in which to store the cropped videos. Directory structure is ratID-->sessionID-->
        [sessionID_direct/leftmirror/rightmirror]
    :param cropped_vids_parent: parent directory in which to create directory tree
    :param video_folder_list: list of lowest level directories containing the original videos
    :return: cropped_video_directories
    """

    cropped_video_directories = [[], [], []]
    for crop_dir in video_folder_list:
        _, session_dir = os.path.split(crop_dir)
        ratID, session_name = parse_session_dir_name(session_dir)

        # create direct view directory for this raw video directory
        cropped_vid_dir = session_dir + '_direct'
        direct_view_directory = os.path.join(cropped_vids_parent, ratID, session_dir, cropped_vid_dir)

        # create left mirror view directory for this raw video directory
        cropped_vid_dir = session_dir + '_leftmirror'
        left_view_directory = os.path.join(cropped_vids_parent, ratID, session_dir, cropped_vid_dir)

        # create right mirror view directory for this raw video directory
        cropped_vid_dir = session_dir + '_rightmirror'
        right_view_directory = os.path.join(cropped_vids_parent, ratID, session_dir, cropped_vid_dir)

        cropped_video_directories[0].append(direct_view_directory)
        cropped_video_directories[1].append(left_view_directory)
        cropped_video_directories[2].append(right_view_directory)

    return cropped_video_directories


def trajectory_folder(trajectories_parent, ratID, session_name):

    session_folder = ratID + '_' + session_name
    traj_folder = os.path.join(trajectories_parent, ratID, session_folder)
    if not os.path.isdir(traj_folder):
        os.makedirs(traj_folder)

    return traj_folder


def find_orig_rat_video(video_metadata, video_root_folder, vidtype='.avi'):

    # directory structure:
    #  video_root_folder --> ratID --> ratID_sessiondateX

    rat_folder = os.path.join(video_root_folder, video_metadata['ratID'])
    datestring = date_to_string_for_fname(video_metadata['triggertime'])
    session_folder_test = os.path.join(rat_folder, video_metadata['ratID'] + '_' + datestring + '*')
    session_folder_list = glob.glob(session_folder_test)
    if len(session_folder_list) == 1:
        session_folder = session_folder_list[0]
    elif len(session_folder_list) > 1:
        print('more than one session folder for {} on {}', video_metadata['ratID'], datestring)
        return None
    else:
        print('no session folders for {} on {}', video_metadata['ratID'], datestring)
        return None

    timestring = datetime_to_string_for_fname(video_metadata['triggertime'])
    vid_name = '_'.join((video_metadata['ratID'],
                         timestring,
                         '{:03d}.avi'.format(video_metadata['video_number'])
                         ))
    orig_vid_name = os.path.join(session_folder, vid_name)

    return orig_vid_name


def parse_paw_trajectory_fname(paw_trajectory_fname):

    _, pt_name = os.path.split(paw_trajectory_fname)

def parse_session_dir_name(session_dir):
    """

    :param session_dir - session directory name assumed to be of the form RXXXX_yyyymmddz, where XXXX is the rat number,
        yyyymmdd is the date, and z is a letter identifying distinct sessions on the same day (i.e., "a", "b", etc.)
    :return:
    """

    _, session_dir_name = os.path.split(session_dir)
    dir_name_parts = session_dir_name.split('_')
    ratID = dir_name_parts[0]
    session_name = dir_name_parts[1]

    return ratID, session_name


def find_folders_to_analyze(cropped_videos_parent, view_list=None):
    """
    get the full list of directories containing cropped videos in the videos_to_analyze folder
    :param cropped_videos_parent: parent directory with subfolders direct_view and mirror_views, which have subfolders
        RXXXX-->RXXXXyyyymmddz[direct/leftmirror/rightmirror] (assuming default view list)
    :param view_list:
    :return: folders_to_analyze: dictionary containing a key for each member of view_list. Each key holds a list of
        folders to run through deeplabcut
    """

    if view_list is None:
        view_list = ('direct', 'leftmirror', 'rightmirror')

    folders_to_analyze = dict(zip(view_list, ([] for _ in view_list)))

    rat_folder_list = glob.glob(os.path.join(cropped_videos_parent, 'R*'))
    for rat_folder in rat_folder_list:
        if os.path.isdir(rat_folder):
            # assume the rat_folder directory name is the same as ratID (i.e., form of RXXXX)
            _, ratID = os.path.split(rat_folder)
            session_name = ratID + '_*'
            session_dir_list = glob.glob(rat_folder + '/' + session_name)
            # make sure we only include directories (just in case there are some stray files with the right names)
            session_dir_list = [session_dir for session_dir in session_dir_list if os.path.isdir(session_dir)]
            for session_dir in session_dir_list:
                _, cur_session = os.path.split(session_dir)
                for view in view_list:
                    view_folder = os.path.join(session_dir, cur_session + '_' + view)
                    if os.path.isdir(view_folder):
                        folders_to_analyze[view].extend([view_folder])

    return folders_to_analyze


def find_optitrack_folders_to_analyze(parent_directories, cam_list=(1, 2)):
    """
    get the full list of directories containing cropped videos in the videos_to_analyze folder
    :param cropped_videos_parent: parent directory with subfolders direct_view and mirror_views, which have subfolders
        RXXXX-->RXXXXyyyymmddz[direct/leftmirror/rightmirror] (assuming default view list)
    :param view_list:
    :return: folders_to_analyze: dictionary containing a key for each member of view_list. Each key holds a list of
        folders to run through deeplabcut
    """

    cropped_videos_parent = parent_directories['cropped_vids_parent']

    cam_name_list = ['cam{:02d}'.format(cam_num) for cam_num in cam_list]
    folders_to_analyze = dict.fromkeys(cam_name_list)
    for cam_name in cam_name_list:
        folders_to_analyze[cam_name] = []

    mouse_folder_list = glob.glob(os.path.join(cropped_videos_parent, '*'))
    for mouse_folder in mouse_folder_list:
        if os.path.isdir(mouse_folder):
            # assume the rat_folder directory name is the same as ratID (i.e., form of RXXXX)
            _, mouseID = os.path.split(mouse_folder)
            monthfolder_name = mouseID + '_*'
            month_dir_list = glob.glob(os.path.join(mouse_folder, monthfolder_name))
            # make sure we only include directories (just in case there are some stray files with the right names)
            month_dir_list = [month_dir for month_dir in month_dir_list if os.path.isdir(month_dir)]
            for month_dir in month_dir_list:
                _, cur_month_dir_name = os.path.split(month_dir)
                sessionfolder_name = cur_month_dir_name + '*'
                session_dir_list = glob.glob(os.path.join(month_dir, sessionfolder_name))
                for session_dir in session_dir_list:
                    _, cur_session = os.path.split(session_dir)
                    for cam_name in cam_name_list:
                        cam_folder = os.path.join(session_dir, cur_session + '_' + cam_name)
                        if os.path.isdir(cam_folder):
                            folders_to_analyze[cam_name].extend([cam_folder])

    return folders_to_analyze


def parse_cropped_video_name(cropped_video_name):
    """
    extract metadata information from the video name
    :param cropped_video_name: video name with expected format RXXXX_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'direct', 'leftmirror', or 'rightmirror', and l-r-t-b are left, right, top, and bottom of the
        cropping windows from the original video
    :return: cropped_vid_metadata: dictionary containing the following keys
        ratID - rat ID as a string RXXXX
        boxnum - box number the session was run in. useful for making sure we used the right calibration. If unknown,
            set to 99
        triggertime - datetime object with when the trigger event occurred (date and time)
        video_number - number of the video (ZZZ in the filename). This number is not necessarily unique within a session
            if it had to be restarted partway through
        video_type - video type (e.g., '.avi', '.mp4', etc)
        crop_window - 4-element list [left, right, top, bottom] in pixels
    """

    cropped_vid_metadata = {
        'ratID': '',
        'rat_num': 0,
        'boxnum': 99,
        'triggertime': datetime(1,1,1),
        'video_number': 0,
        'view': '',
        'video_type': '',
        'crop_window': [],
        'cropped_video_name': ''
    }
    _, vid_name = os.path.split(cropped_video_name)
    cropped_vid_metadata['cropped_video_name'] = vid_name
    vid_name, vid_type = os.path.splitext(vid_name)

    metadata_list = vid_name.split('_')

    cropped_vid_metadata['ratID'] = metadata_list[0]
    num_string = ''.join(filter(lambda i: i.isdigit(), cropped_vid_metadata['ratID']))
    cropped_vid_metadata['rat_num'] = int(num_string)

    # if box number is stored in file name, then extract it
    if 'box' in metadata_list[1]:
        cropped_vid_metadata['boxnum'] = int(metadata_list[1][3:])
        next_metadata_idx = 2
    else:
        next_metadata_idx = 1

    datetime_str = metadata_list[next_metadata_idx] + '_' + metadata_list[1+next_metadata_idx]
    cropped_vid_metadata['triggertime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    cropped_vid_metadata['video_number'] = int(metadata_list[next_metadata_idx + 2])
    cropped_vid_metadata['video_type'] = vid_type
    cropped_vid_metadata['view'] = metadata_list[next_metadata_idx + 3]

    left, right, top, bottom = list(map(int, metadata_list[next_metadata_idx + 4].split('-')))
    cropped_vid_metadata['crop_window'].extend(left, right, top, bottom)

    return cropped_vid_metadata


def parse_cropped_optitrack_video_name(cropped_video_name):
    """
    extract metadata information from the video name
    :param cropped_video_name: video name with expected format mouseID_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'direct', 'leftmirror', or 'rightmirror', and l-r-t-b are left, right, top, and bottom of the
        cropping windows from the original video
    :return: cropped_vid_metadata: dictionary containing the following keys
        ratID - rat ID as a string RXXXX
        boxnum - box number the session was run in. useful for making sure we used the right calibration. If unknown,
            set to 99
        triggertime - datetime object with when the trigger event occurred (date and time)
        video_number - number of the video (ZZZ in the filename). This number is not necessarily unique within a session
            if it had to be restarted partway through
        video_type - video type (e.g., '.avi', '.mp4', etc)
        crop_window - 4-element list [left, right, top, bottom] in pixels
    """

    cropped_vid_metadata = {
        'mouseID': '',
        'boxnum': 99,
        'time': datetime(1,1,1),
        'video_number': 0,
        'cam_num': 0,
        'video_type': '',
        'crop_window': [],
        'cropped_video_name': ''
    }
    _, vid_name = os.path.split(cropped_video_name)
    cropped_vid_metadata['cropped_video_name'] = vid_name
    vid_name, vid_type = os.path.splitext(vid_name)

    metadata_list = vid_name.split('_')

    cropped_vid_metadata['mouseID'] = metadata_list[0]

    # # if box number is stored in file name, then extract it
    # if 'box' in metadata_list[1]:
    #     cropped_vid_metadata['boxnum'] = int(metadata_list[1][3:])
    #     next_metadata_idx = 2
    # else:
    #     next_metadata_idx = 1

    datetime_str = metadata_list[next_metadata_idx] + '_' + metadata_list[1+next_metadata_idx]
    cropped_vid_metadata['triggertime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    cropped_vid_metadata['video_number'] = int(metadata_list[next_metadata_idx + 2])
    cropped_vid_metadata['video_type'] = vid_type
    cropped_vid_metadata['view'] = metadata_list[next_metadata_idx + 3]

    left, right, top, bottom = list(map(int, metadata_list[next_metadata_idx + 4].split('-')))
    cropped_vid_metadata['crop_window'].extend(left, right, top, bottom)

    return cropped_vid_metadata


def parse_video_name(video_name):
    """
    extract metadata information from the video name
    :param video_name: video name with expected format RXXXX_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'direct', 'leftmirror', or 'rightmirror', and l-r-t-b are left, right, top, and bottom of the
        cropping windows from the original video
    :return: video_metadata: dictionary containing the following keys
        ratID - rat ID as a string RXXXX
        boxnum - box number the session was run in. useful for making sure we used the right calibration. If unknown,
            set to 99
        triggertime - datetime object with when the trigger event occurred (date and time)
        video_number - number of the video (ZZZ in the filename). This number is not necessarily unique within a session
            if it had to be restarted partway through
        video_type - video type (e.g., '.avi', '.mp4', etc)
    """

    video_metadata = {
        'ratID': '',
        'rat_num': 0,
        'session_name': '',
        'boxnum': 99,
        'triggertime': datetime(1,1,1),
        'video_number': 0,
        'video_type': '',
        'video_name': '',
        'im_size': (1024, 2040)
    }

    if os.path.exists(video_name):
        video_object = cv2.VideoCapture(video_name)
        video_metadata['im_size'] = (int(video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   int(video_object.get(cv2.CAP_PROP_FRAME_WIDTH)))

    vid_path, vid_name = os.path.split(video_name)
    video_metadata['video_name'] = vid_name
    # the last folder in the tree should have the session name
    _, video_metadata['session_name'] = os.path.split(vid_path)
    vid_name, vid_type = os.path.splitext(vid_name)

    metadata_list = vid_name.split('_')

    video_metadata['ratID'] = metadata_list[0]
    num_string = ''.join(filter(lambda i: i.isdigit(), video_metadata['ratID']))
    video_metadata['rat_num'] = int(num_string)

    # if box number is stored in file name, then extract it
    if 'box' in metadata_list[1]:
        video_metadata['boxnum'] = int(metadata_list[1][3:])
        next_metadata_idx = 2
    else:
        next_metadata_idx = 1

    datetime_str = metadata_list[next_metadata_idx] + '_' + metadata_list[1+next_metadata_idx]
    video_metadata['triggertime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    video_metadata['video_number'] = int(metadata_list[next_metadata_idx + 2])
    video_metadata['video_type'] = vid_type

    return video_metadata


def build_video_name(video_metadata, videos_parent):

    video_name = '{}_box{:02d}_{}_{:03d}.avi'.format(video_metadata['ratID'],
                                                  video_metadata['boxnum'],
                                                  video_metadata['triggertime'].strftime('%Y%m%d_%H-%M-%S'),
                                                  video_metadata['video_number'])
    video_name = os.path.join(videos_parent, 'videos_to_crop', video_metadata['ratID'], video_metadata['session_name'], video_name)
    return video_name


def parse_dlc_output_pickle_name(dlc_output_pickle_name):
    """
    extract metadata information from the pickle file name
    :param dlc_output_pickle_name: video name with expected format RXXXX_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'direct', 'leftmirror', or 'rightmirror', and l-r-t-b are left, right, top, and bottom of the
        cropping windows from the original video
    :return: cropped_vid_metadata: dictionary containing the following keys
        ratID - rat ID as a string RXXXX
        boxnum - box number the session was run in. useful for making sure we used the right calibration. If unknown,
            set to 99
        triggertime - datetime object with when the trigger event occurred (date and time)
        video_number - number of the video (ZZZ in the filename). This number is not necessarily unique within a session
            if it had to be restarted partway through
        video_type - video type (e.g., '.avi', '.mp4', etc)
        crop_window - 4-element list [left, right, top, bottom] in pixels
    """

    pickle_metadata = {
        'ratID': '',
        'rat_num': 0,
        'boxnum': 99,
        'triggertime': datetime(1,1,1),
        'video_number': 0,
        'view': '',
        'crop_window': [],
        'scorername': '',
        'pickle_name': ''
    }
    _, pickle_name = os.path.split(dlc_output_pickle_name)
    pickle_metadata['pickle_name'] = pickle_name
    pickle_name, vid_type = os.path.splitext(pickle_name)

    metadata_list = pickle_name.split('_')

    pickle_metadata['ratID'] = metadata_list[0]
    num_string = ''.join(filter(lambda i: i.isdigit(), pickle_metadata['ratID']))
    pickle_metadata['rat_num'] = int(num_string)

    # if box number is stored in file name, then extract it
    if 'box' in metadata_list[1]:
        pickle_metadata['boxnum'] = int(metadata_list[1][3:])
        next_metadata_idx = 2
    else:
        next_metadata_idx = 1

    datetime_str = metadata_list[next_metadata_idx] + '_' + metadata_list[1+next_metadata_idx]
    pickle_metadata['triggertime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    pickle_metadata['video_number'] = int(metadata_list[next_metadata_idx + 2])
    pickle_metadata['view'] = metadata_list[next_metadata_idx + 3]

    # 'DLC' gets appended to the last cropping parameter in the filename by deeplabcut
    crop_window_strings = metadata_list[next_metadata_idx + 4].split('-')
    left, right, top = list(map(int, crop_window_strings[:-1]))

    # find where 'DLC' starts in the last crop_window_string
    dlc_location = crop_window_strings[-1].find('DLC')
    bottom = int(crop_window_strings[-1][:dlc_location])

    pickle_metadata['crop_window'].extend((left, right, top, bottom))

    #todo: write the scorername into the pickle metadata dictionary. It's also in the metadata pickle file
    pickle_metadata['scorername']

    return pickle_metadata


def parse_dlc_output_pickle_name_optitrack(dlc_output_pickle_name):
    """
    extract metadata information from the pickle file name
    :param dlc_output_pickle_name: video name with expected format mouseID_yyyymmdd_HH-MM-SS_[sess#]_ZZZ_camAA_l-r-t-b_[rotated]DLC_dlcrnetms5_mouse_headfixed_skilledreachingNov5shuffle1_150000_full.pickle
        where [sess#] may or may not be present (but generally should be)
    :return: cropped_vid_metadata: dictionary containing the following keys
        mouseID - mouseID as a string (e.g., 'dLightXX')
        trialtime - datetime object with when the trigger event occurred (date and time)
        video_number - number of the video (ZZZ in the filename). This number is not necessarily unique within a session
            if it had to be restarted partway through
        video_type - video type (e.g., '.avi', '.mp4', etc)
        crop_window - 4-element list [left, right, top, bottom] in pixels
    """

    pickle_metadata = {
        'mouseID': '',
        'trialtime': datetime(1,1,1),
        'session_num': 0,
        'video_number': 0,
        'cam_num': 0,
        'crop_window': [],
        'isrotated': False,
        'scorername': '',
        'pickle_name': ''
    }
    _, pickle_name = os.path.split(dlc_output_pickle_name)
    pickle_metadata['pickle_name'] = pickle_name
    pickle_name, vid_type = os.path.splitext(pickle_name)

    metadata_list = pickle_name.split('_')

    pickle_metadata['mouseID'] = metadata_list[0]
    # num_string = ''.join(filter(lambda i: i.isdigit(), pickle_metadata['ratID']))
    # pickle_metadata['rat_num'] = int(num_string)

    # if box number is stored in file name, then extract it
    if 'box' in metadata_list[1]:
        pickle_metadata['boxnum'] = int(metadata_list[1][3:])
        next_metadata_idx = 2
    else:
        next_metadata_idx = 1

    datetime_str = metadata_list[next_metadata_idx] + '_' + metadata_list[1+next_metadata_idx]
    pickle_metadata['trialtime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    pickle_metadata['session_num'] = int(metadata_list[next_metadata_idx + 2])
    pickle_metadata['video_number'] = int(metadata_list[next_metadata_idx + 3])
    pickle_metadata['cam_num'] = int(metadata_list[next_metadata_idx + 4][3:])

    # 'DLC' gets appended to the last cropping parameter in the filename by deeplabcut
    crop_window_strings = metadata_list[next_metadata_idx + 5].split('-')
    left, right, top, bottom = list(map(int, crop_window_strings))

    # find where 'DLC' starts in the last crop_window_string
    # dlc_location = crop_window_strings[-1].find('DLC')
    # bottom = int(crop_window_strings[-1][:dlc_location])

    pickle_metadata['crop_window'].extend((left, right, top, bottom))

    # was this video rotated 180 degrees?
    if metadata_list[next_metadata_idx + 6][:7] == 'rotated':
        pickle_metadata['isrotated'] = True

    #todo: write the scorername into the pickle metadata dictionary. It's also in the metadata pickle file
    pickle_metadata['scorername']

    return pickle_metadata


def create_marked_vids_folder(cropped_vid_folder, cropped_videos_parent, marked_videos_parent):
    """
    :param cropped_vid_folder:
    :param cropped_videos_parent:
    :param marked_videos_parent:
    :return:
    """

    # find the string 'cropped_videos' in cropped_vid_folder; everything after that is the relative path to create the marked_vids_folder
    cropped_vid_relpath = os.path.relpath(cropped_vid_folder, start=cropped_videos_parent)
    marked_vid_relpath = cropped_vid_relpath + '_marked'
    marked_vids_folder = os.path.join(marked_videos_parent, marked_vid_relpath)

    if not os.path.isdir(marked_vids_folder):
        os.makedirs(marked_vids_folder)

    return marked_vids_folder


def create_optitrack_marked_vids_folder(cropped_vid_folder, cropped_videos_parent, marked_videos_parent):
    """
    :param cropped_vid_folder:
    :param cropped_videos_parent:
    :param marked_videos_parent:
    :return:
    """

    # find the string 'cropped_videos' in cropped_vid_folder; everything after that is the relative path to create the marked_vids_folder
    cropped_vid_relpath = os.path.relpath(cropped_vid_folder, start=cropped_videos_parent)
    marked_vid_relpath = cropped_vid_relpath + '_marked'
    marked_vids_folder = os.path.join(marked_videos_parent, marked_vid_relpath)

    if not os.path.isdir(marked_vids_folder):
        os.makedirs(marked_vids_folder)

    return marked_vids_folder


def sort_optitrack_calibration_vid_names_by_camera_number(calibration_vid_list):
    '''
    calibration videos should be collected for each camera. Make sure that corresponding videos collected by multiple
    cameras are sorted by camera number. This way, rotation and translation matrices are consistently calculated for
    camera 2 with respect to camera 1 (not the other way around)
    :param calibration_vid_list: list of videos containing calibration checkerboard images. Should be of the form:
        mouseID_YYmmdd-HH-MM-SS_sessionnum_vidnum_camXX.avi
        sessionnum is the number of the session recorded on that day for that mouse, vidnum is a 3-digit zero-padded number
            with the video number for that session, and XX is the zero-padded camera number (01 or 02)
    :return: sorted_calibration_vid_list - calibration_vid_list sorted by camera number
    '''
    cam_nums = []
    for cal_name in calibration_vid_list:
        cal_metadata = parse_Burgess_calibration_vid_name(cal_name)
        cam_nums.append(cal_metadata['cam_num'])

    # zip calibration_vid_list with camera numbers for each video, sort by camera number, and extract the sorted calibration video name list
    sorted_calibration_vid_list = [cal_vid_name for cal_vid_name, _ in sorted(zip(calibration_vid_list, cam_nums), key=lambda pair: pair[1])]

    return sorted_calibration_vid_list


def create_calibration_file_tree(calibration_parent, vid_metadata):
    """

    :param calibration_parent:
    :param vid_metadata: dictionary containing the following keys
        ratID - rat ID as a string RXXXX
        boxnum - box number the session was run in. useful for making sure we used the right calibration. If unknown,
            set to 99
        triggertime - datetime object with when the trigger event occurred (date and time)
        video_number - number of the video (ZZZ in the filename). This number is not necessarily unique within a session
            if it had to be restarted partway through
        video_type - video type (e.g., '.avi', '.mp4', etc)
        crop_window - 4-element list [left, right, top, bottom] in pixels
    :return:
    """

    year_folder = 'calibration_files_' + datetime.strftime(vid_metadata['triggertime'], '%Y')
    month_folder = 'calibration_files_' + datetime.strftime(vid_metadata['triggertime'], '%Y%m')
    # day_folder = 'calibration_files_' + datetime.strftime(vid_metadata['triggertime'], '%Y%m%d')
    box_folder = month_folder + '_box{:2d}'.format(vid_metadata['boxnum'])

    calibration_file_tree = os.path.join(calibration_parent, year_folder, month_folder, box_folder)

    return calibration_file_tree


def find_calibration_files_folder(session_date, box_num, calibration_parent):

    calibration_metadata = {'time': session_date,
                            'box_num': box_num}

    cal_file_folder = create_calibration_file_folder_name(calibration_metadata, calibration_parent)

    return cal_file_folder


def create_mat_cal_filename(calibration_metadata, basename='SR_boxCalibration'):

    datestring = date_to_string_for_fname(calibration_metadata['time'])
    mat_cal_filename = '_'.join((basename,
                                 'box{:2d}'.format(calibration_metadata['boxnum']),
                                 datestring + '.mat'))

    return mat_cal_filename


def find_calibration_vid_folders(calibration_parent):
    '''
    find all calibration videos. assume directory structure:
        calibration_parent-->calibration_videos__YYYY-->calibration_videos__YYYYMM-->calibration_videos__YYYYMM_boxZZ where
        ZZ is the 2-digit box number
    :param calibration_parent:
    :return:
    '''
    year_folders = glob.glob(os.path.join(calibration_parent, 'calibration_videos_*'))
    month_folders = []
    # for yf in year_folders:
    #     month_folders.extend(glob.glob(os.path.join(yf, 'calibration_videos_*')))
    [month_folders.extend(glob.glob(os.path.join(yf, 'calibration_videos_*'))) for yf in year_folders]

    box_folders = []
    [box_folders.extend(glob.glob(os.path.join(mf, 'calibration_videos_*'))) for mf in month_folders]

    return box_folders


def create_cropped_calib_vid_name(full_calib_vid_name, crop_view, crop_params_dict):
    '''

    :param full_calib_vid_name: full name of calibration video
    :param crop_view:
    :param crop_params_dict:
    :return:
    '''
    full_cropped_path = create_cropped_calib_vids_folder(full_calib_vid_name)

    if not os.path.isdir(full_cropped_path):
        os.makedirs(full_cropped_path)

    _, vid_name = os.path.split(full_calib_vid_name)
    vid_name, ext = os.path.splitext(vid_name)

    cp_strings = [str(cp) for cp in crop_params_dict[crop_view]]
    cp_joined = '-'.join(cp_strings)
    cropped_vid_name = vid_name + '_' + crop_view + '_' + cp_joined + ext

    full_cropped_vid_name = os.path.join(full_cropped_path, cropped_vid_name)

    return full_cropped_vid_name


def create_cropped_calib_vids_folder(full_vid_name):

    full_vid_path, _ = os.path.split(full_vid_name)
    _, folder_name = os.path.split(full_vid_path)

    cropped_folder_name = folder_name + '_cropped'
    full_cropped_path = os.path.join(full_vid_path, cropped_folder_name)

    return full_cropped_path


def find_dlc_output_pickles(video_metadata, marked_videos_parent, view_list=None):
    """

    :param video_metadata:
    :param marked_videos_parent:
    :param view_list:
    :return:
    """
    if view_list is None:
        view_list = ('direct', 'leftmirror', 'rightmirror')

    session_name = video_metadata['session_name']
    rat_pickle_folder = os.path.join(marked_videos_parent, video_metadata['ratID'])
    session_pickle_folder = os.path.join(rat_pickle_folder, session_name)

    dlc_output_pickle_names = {view: None for view in view_list}
    dlc_metadata_pickle_names = {view: None for view in view_list}
    for view in view_list:
        pickle_folder = os.path.join(session_pickle_folder, session_name + '_' + view + '_marked')
        test_string_full, test_string_meta = construct_dlc_output_pickle_names(video_metadata, view)
        test_string_full = os.path.join(pickle_folder, test_string_full)
        test_string_meta = os.path.join(pickle_folder, test_string_meta)

        pickle_full_list = glob.glob(test_string_full)
        pickle_meta_list = glob.glob(test_string_meta)

        if len(pickle_full_list) > 1:
            # ambiguity in which pickle file goes with this video
            sys.exit('Ambiguous dlc output file name for {}'.format(video_metadata['video_name']))

        if len(pickle_meta_list) > 1:
            # ambiguity in which pickle file goes with this video
            sys.exit('Ambiguous dlc output metadata file name for {}'.format(video_metadata['video_name']))

        if len(pickle_full_list) == 0:
            # no pickle file for this view
            print('No dlc output file found for {}, {} view'.format(video_metadata['video_name'], view))
            continue

        if len(pickle_meta_list) == 0:
            # no pickle file for this view
            print('No dlc output metadata file found for {}, {} view'.format(video_metadata['video_name'], view))
            continue

        dlc_output_pickle_names[view] = pickle_full_list[0]
        dlc_metadata_pickle_names[view] = pickle_meta_list[0]

    return dlc_output_pickle_names, dlc_metadata_pickle_names


def construct_dlc_output_pickle_names(video_metadata, view):
    """

    :param video_metadata:
    :param view: string containing 'direct', 'leftmirror', or 'rightmirror'
    :return:
    """
    if video_metadata['boxnum'] == 99:
        pickle_name_full = video_metadata['ratID'] + '_' + \
                           video_metadata['triggertime'].strftime('%Y%m%d_%H-%M-%S') + '_' + \
                           '{:03d}'.format(video_metadata['video_number']) + '_' + \
                           view + '_*_full.pickle'

        pickle_name_meta = video_metadata['ratID'] + '_' + \
                           video_metadata['triggertime'].strftime('%Y%m%d_%H-%M-%S') + '_' + \
                           '{:03d}'.format(video_metadata['video_number']) + '_' + \
                           view + '_*_meta.pickle'
    else:
        pickle_name_full = video_metadata['ratID'] + '_' + \
                      'box{:02d}'.format(video_metadata['boxnum']) + '_' + \
                      video_metadata['triggertime'].strftime('%Y%m%d_%H-%M-%S') + '_' + \
                      '{:03d}'.format(video_metadata['video_number']) + '_' + \
                      view + '_*_full.pickle'

        pickle_name_meta = video_metadata['ratID'] + '_' + \
                           'box{:02d}'.format(video_metadata['boxnum']) + '_' + \
                           video_metadata['triggertime'].strftime('%Y%m%d_%H-%M-%S') + '_' + \
                           '{:03d}'.format(video_metadata['video_number']) + '_' + \
                           view + '_*_meta.pickle'

    return pickle_name_full, pickle_name_meta


def find_calibration_video(video_metadata, calibration_parent):
    """

    :param video_metadata:
    :param calibration_parent:
    :return:
    """
    date_string = video_metadata['triggertime'].strftime('%Y%m%d')
    year_folder = os.path.join(calibration_parent, date_string[0:4])
    month_folder = os.path.join(year_folder, date_string[0:6] + '_calibration')
    calibration_folder = os.path.join(month_folder, date_string[0:6] + '_calibration_videos')

    test_name = 'SR_boxCalibration_box{:02d}_{}.mat'.format(video_metadata['boxnum'], date_string)
    test_name = os.path.join(calibration_folder, test_name)

    if os.path.exists(test_name):
        return test_name
    else:
        return ''
        # sys.exit('No calibration file found for ' + video_metadata['video_name'])


def create_trajectory_filename(video_metadata):

    trajectory_name = video_metadata['ratID'] + '_' + \
        'box{:02d}'.format(video_metadata['boxnum']) + '_' + \
        video_metadata['triggertime'].strftime('%Y%m%d_%H-%M-%S') + '_' + \
        '{:03d}'.format(video_metadata['video_number']) + '_3dtrajectory'

    return trajectory_name


def find_camera_calibration_video(video_metadata, calibration_parent):
    """

    :param video_metadata:
    :param calibration_parent:
    :return:
    """
    date_string = video_metadata['triggertime'].strftime('%Y%m%d')
    year_folder = os.path.join(calibration_parent, date_string[0:4])
    month_folder = os.path.join(year_folder, date_string[0:6] + '_calibration')
    calibration_video_folder = os.path.join(month_folder, 'camera_calibration_videos_' + date_string[0:6])

    test_name = 'CameraCalibration_box{:02d}_{}_*.mat'.format(video_metadata['boxnum'], date_string)
    test_name = os.path.join(calibration_video_folder, test_name)

    calibration_video_list = glob.glob(test_name)

    if len(calibration_video_list) == 0:
        sys.exit('No camera calibration video found for ' + video_metadata['video_name'])

    if len(calibration_video_list) == 1:
        return calibration_video_list[0]

    # more than one potential video was found
    # find the last relevant calibration video collected before the current reaching video
    vid_times = []
    for cal_vid in calibration_video_list:
        cam_cal_md = parse_camera_calibration_video_name(cal_vid)
        vid_times.append(cam_cal_md['time'])

    last_time_prior_to_video = max(d for d in vid_times if d < video_metadata['triggertime'])

    calibration_video_name = calibration_video_list[vid_times.index(last_time_prior_to_video)]

    return calibration_video_name


def parse_camera_calibration_video_name(calibration_video_name):
    """

    :param calibration_video_name: form of GridCalibration_boxXX_YYYYMMDD_HH-mm-ss.avi
    :return:
    """
    camera_calibration_metadata = {
        'boxnum': 99,
        'time': datetime(1, 1, 1)
    }
    _, cal_vid_name = os.path.split(calibration_video_name)
    cal_vid_name, _ = os.path.splitext(cal_vid_name)

    cal_vid_name_parts = cal_vid_name.split('_')

    camera_calibration_metadata['boxnum'] = int(cal_vid_name_parts[1][3:])

    datetime_str = cal_vid_name_parts[2] + '_' + cal_vid_name_parts[3]
    camera_calibration_metadata['time'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    return camera_calibration_metadata


def create_calibration_filename(calibration_metadata):

    # calibration_folder = create_calibration_file_folder_name(calibration_metadata, calibration_parent)
    # if not os.path.isdir(calibration_folder):
    #     os.makedirs(calibration_folder)

    datetime_string = calibration_metadata['time'].strftime('%Y%m%d_%H-%M-%S')

    calibration_name = 'calibration_box{:02d}_{}.pickle'.format(calibration_metadata['boxnum'], datetime_string)
    # calibration_name = os.path.join(calibration_folder, calibration_name)

    return calibration_name


def create_calibration_file_folder_name(calibration_metadata, calibration_parent):
    date_string = calibration_metadata['time'].strftime('%Y%m%d')
    year_folder = os.path.join(calibration_parent, 'calibrationfiles_' + date_string[0:4])
    month_folder = os.path.join(year_folder, 'calibrationfiles_' + date_string[0:6])
    calibration_file_folder_name = os.path.join(month_folder, 'calibrationfiles_' + date_string[0:6] + '_box{:2d}'.format(calibration_metadata['box_num']))

    return calibration_file_folder_name


def create_mat_fname_dlc_output(video_metadata, dlc_mat_output_parent):

    mat_path = os.path.join(dlc_mat_output_parent,
                            video_metadata['ratID'],
                            video_metadata['session_name'])

    if not os.path.isdir(mat_path):
        os.makedirs(mat_path)

    mat_name = '{}_box{:02d}_{}_{:03d}_dlc-out.mat'.format(video_metadata['ratID'],
                                                           video_metadata['boxnum'],
                                                           video_metadata['triggertime'].strftime('%Y%m%d_%H-%M-%S'),
                                                           video_metadata['video_number']
                                                           )
    mat_name = os.path.join(mat_path, mat_name)

    return mat_name


def find_marked_vids_for_3d_reconstruction(marked_vids_parent, dlc_mat_output_parent, rat_df):

    # find marked vids for which we have both relevant views (eventually, need all 3 views)
    marked_rat_folders = glob.glob(os.path.join(marked_vids_parent, 'R*'))

    # return a list of video_metadata dictionaries
    metadata_list = []
    for rat_folder in marked_rat_folders:
        if os.path.isdir(rat_folder):
            _, ratID = os.path.split(rat_folder)
            rat_num = int(ratID[1:])
            paw_pref = rat_df[rat_df['ratID'] == rat_num]['pawPref'].values[0]
            if paw_pref == 'right':
                mirrorview = 'leftmirror'
            else:
                mirrorview = 'rightmirror'
            # find the paw preference for this rat

            session_folders = glob.glob(os.path.join(rat_folder, ratID + '_*'))

            for session_folder in session_folders:
                if os.path.isdir(session_folder):
                    _, session_name = os.path.split(session_folder)

                    # check that there is a direct_marked folder for this session
                    direct_marked_folder = os.path.join(session_folder, session_name + '_direct_marked')
                    mirror_marked_folder = os.path.join(session_folder, session_name + '_' + mirrorview + '_marked')

                    if not os.path.isdir(mirror_marked_folder):
                        continue

                    if os.path.isdir(direct_marked_folder):
                        # find all the full_pickle and metadata_pickle files in the folder, and look to see if there are
                        # matching files in the appropriate mirror view folder
                        test_name = ratID + '_*_full.pickle'
                        full_pickle_list = glob.glob(os.path.join(direct_marked_folder, test_name))

                        for full_pickle_file in full_pickle_list:
                            # is there a matching metadata file, as well as matching metadata files in the mirror folder?
                            _, pickle_name = os.path.split(full_pickle_file)
                            pickle_metadata = parse_dlc_output_pickle_name(pickle_name)
                            # crop_window_string = '{:d}-{:d}-{:d}-{:d}'.format(pickle_metadata['crop_window'][0],
                            #                                                   pickle_metadata['crop_window'][1],
                            #                                                   pickle_metadata['crop_window'][2],
                            #                                                   pickle_metadata['crop_window'][3]
                            #                                                   )
                            meta_direct_file = os.path.join(direct_marked_folder, pickle_name.replace('full', 'meta'))
                            vid_prefix = pickle_name[:pickle_name.find('direct')]
                            test_mirror_name = vid_prefix + '*_full.pickle'
                            full_mirror_name_list = glob.glob(os.path.join(mirror_marked_folder, test_mirror_name))
                            if len(full_mirror_name_list) == 1:
                                full_mirror_file = full_mirror_name_list[0]
                                _, full_mirror_name = os.path.split(full_mirror_file)
                                meta_mirror_file = os.path.join(mirror_marked_folder, full_mirror_name.replace('full', 'meta'))
                                if os.path.exists(meta_direct_file) and os.path.exists(meta_mirror_file):

                                    video_name = '{}_box{:02d}_{}_{:03d}.avi'.format(ratID,
                                                                                     pickle_metadata['boxnum'],
                                                                                     pickle_metadata['triggertime'].strftime('%Y%m%d_%H-%M-%S'),
                                                                                     pickle_metadata['video_number'])
                                    video_metadata = {
                                        'ratID': ratID,
                                        'rat_num': rat_num,
                                        'session_name': session_name,
                                        'boxnum': pickle_metadata['boxnum'],
                                        'triggertime': pickle_metadata['triggertime'],
                                        'video_number': pickle_metadata['video_number'],
                                        'video_type': '.avi',
                                        'video_name': video_name,
                                        'im_size': (1024, 2040)
                                    }
                                    mat_output_name = create_mat_fname_dlc_output(video_metadata, dlc_mat_output_parent)
                                    # check if these files have already been processed
                                    if not os.path.exists(mat_output_name):
                                        # .mat file doesn't already exist
                                        metadata_list.append(video_metadata)

    return metadata_list


def find_Burgess_calibration_folder(calibration_parent, session_datetime):

    session_year = session_datetime.strftime('%Y')
    session_month = session_datetime.strftime('%m')
    session_day = session_datetime.strftime('%d')

    year_folder = 'mouse_SR_calibrations_' + session_year
    month_folder = year_folder + session_month
    day_folder = month_folder + session_day

    calibration_folder = os.path.join(calibration_parent, year_folder, month_folder, day_folder)

    if os.path.exists(calibration_folder):
        return calibration_folder
    else:
        return None


def collect_all_Burgess_calibration_vid_folders(calibration_parent):
    '''
    find all the calibration subfolders
    :param calibration_parent:
    :return:
    '''

    calibration_folders_stem = 'mouse_SR_calibrations'
    # find all folders within the parent calibration folder
    year_folders = [yf for yf in glob.glob(os.path.join(calibration_parent, calibration_folders_stem + '*')) if os.path.isdir(yf)]
    month_folders = []
    for yf in year_folders:
        new_mf = [mf for mf in glob.glob(os.path.join(yf, calibration_folders_stem + '*')) if os.path.isdir(mf)]
        month_folders.extend(new_mf)

    day_folders = []
    for mf in month_folders:
        new_df = [df for df in glob.glob(os.path.join(mf, calibration_folders_stem + '*')) if os.path.isdir(df)]
        day_folders.extend(new_df)

    return day_folders


def find_Burgess_calibration_vids(cal_vid_parent, cam_list=(1, 2), vidtype='.avi'):


    if vidtype[0] != '.':
        vidtype = '.' + vidtype

    cal_vid_folders = collect_all_Burgess_calibration_vid_folders(cal_vid_parent)
    # cal_vid_folder = find_Burgess_calibration_folder(cal_vid_parent, session_datetime)

    basename = 'calibrationvid'
    full_paths = []

    paired_cal_vids = []
    for cal_vid_folder in cal_vid_folders:
        vid_list = glob.glob(os.path.join(cal_vid_folder, basename + '*' + vidtype))

        vids_already_assigned = []
        for vid_name in vid_list:
            if vid_name in vids_already_assigned:
                continue

            vid_metadata = parse_Burgess_calibration_vid_name(vid_name)

            # is there another video for the other camera at this date and time?
            for other_vid_name in vid_list:
                if other_vid_name != vid_name:
                    other_vid_metadata = parse_Burgess_calibration_vid_name(other_vid_name)
                    if vid_metadata['session_datetime'] == other_vid_metadata['session_datetime']:
                        # these calibration videos form a pair
                        paired_cal_vids.append([vid_name, other_vid_name])
                        vids_already_assigned.append(vid_name)
                        vids_already_assigned.append(other_vid_name)

            # for i_cam in cam_list:
            #     vid_name = basename + '_{}_cam{:02d}.avi'.format(datetime_to_string_for_fname(session_datetime), i_cam)
            #     full_paths.append(os.path.join(cal_vid_folder, vid_name))

    return paired_cal_vids


def parse_Burgess_calibration_vid_name(cal_vid_name):
    '''
    calibration file name of form 'calibrationvid_YYYYmmdd_HH-MM-SS_camZZ.avi' where ZZ is '01' or '02'
    :param cal_vid_name:
    :return:
    '''
    _, cal_vid_name = os.path.split(cal_vid_name)
    bare_name, _ = os.path.splitext(cal_vid_name)

    name_parts_list = bare_name.split('_')

    calvid_metadata = {
        'cam_num': int(name_parts_list[3][3:]),
        'session_datetime': fname_string_to_datetime(name_parts_list[1] + '_' + name_parts_list[2]),
        'calvid_name': cal_vid_name
    }

    return calvid_metadata


def create_calibration_data_name(cal_data_parent, session_datetime):

    basename = 'calibration_data'
    cal_data_name = basename + '_' + datetime_to_string_for_fname(session_datetime) + '.pickle'

    cal_data_folder = create_calibration_data_folder(cal_data_parent, session_datetime)
    cal_data_name = os.path.join(cal_data_folder, cal_data_name)

    return cal_data_name


def create_optitrack_calibration_data_name(cal_data_parent, session_datetime, basename='calibrationdata'):
    '''

    :param cal_data_parent: parent directory for folders containing pickle files with calibration results. Has structure:
        cal_data_parent-->calibration_data_YYYY-->calibration_data_YYYYmm
    :param session_datetime:
    :return:
    '''

    cal_data_name = basename + datetime_to_string_for_fname(session_datetime) + '.pickle'

    cal_data_folder = create_optitrack_calibration_data_folder(cal_data_parent, session_datetime)
    cal_data_name = os.path.join(cal_data_folder, cal_data_name)

    return cal_data_name


def create_multiview_calibration_data_name(cal_data_parent, box_num, session_datetime, basename='calibrationdata'):
    '''

    :param cal_data_parent: parent directory for folders containing pickle files with calibration results. Has structure:
        cal_data_parent-->calibration_data_YYYY-->calibration_data_YYYYmm
    :param session_datetime:
    :return:
    '''

    cal_data_name = basename + '_box{:2d}'.format(box_num) + datetime_to_string_for_fname(session_datetime) + '.pickle'

    cal_data_folder = create_calibration_data_folder(cal_data_parent, box_num, session_datetime)
    cal_data_name = os.path.join(cal_data_folder, cal_data_name)

    return cal_data_name


def find_multiview_calibration_data_name(cal_data_parent, session_datetime, box_num, basename='calibrationdata'):

    cal_data_folder = create_calibration_data_folder(cal_data_parent, box_num, session_datetime)

    # search in cal_data_folder for calibration files from this date
    test_name = basename + '_box{:2d}'.format(box_num) + date_to_string_for_fname(session_datetime) + '_*.pickle'

    cal_data_files = glob.glob(os.path.join(cal_data_folder, test_name))
    cal_data_datetimes = []
    for cal_data_file in cal_data_files:
        _, cal_data_root = os.path.split(cal_data_file)
        cal_data_root, _ = os.path.splitext(cal_data_root)
        fparts = cal_data_root.split('_')
        cal_data_datestring = '_'.join(fparts[1:3])

        cal_data_datetimes.append(fname_string_to_datetime(cal_data_datestring))

    # find datetime of calibration data file closest to session_datetime
    nearest_datetime = min(cal_data_datetimes, key=lambda x: abs(x - session_datetime))

    closest_calibration_file = create_multiview_calibration_data_name(cal_data_parent, nearest_datetime, basename=basename)

    return closest_calibration_file


def create_3dreconstruction_folder(metadata, reconstruction3d_parent):
    trialtime = metadata['trialtime']
    trial_datetime_string = datetime_to_string_for_fname(trialtime)
    mouseID = metadata['mouseID']
    reconstruction3d_name = '_'.join(
        [mouseID,
        trial_datetime_string,
        '{:d}'.format(metadata['session_num']),
        '{:03d}'.format(metadata['video_number']),
        '3dreconstruction.pickle']
        )

    month_string = trialtime.strftime('%Y%m')
    date_string = date_to_string_for_fname(trialtime)
    month_folder_name = mouseID + '_' + month_string
    date_folder_name = mouseID + '_' + date_string
    full_path = os.path.join(reconstruction3d_parent, mouseID, month_folder_name, date_folder_name)

    return full_path


def find_3dreconstruction_folders(reconstruction3d_parent):

    mouse_folders = glob.glob(os.path.join(reconstruction3d_parent, '*'))
    reconstruction3d_folders = []
    for mouse_f in mouse_folders:
        if os.path.isdir(mouse_f):
            _, mouseID = os.path.split(mouse_f)    # assume the folder name is the mouse ID

            # search for folders with that mouse ID
            month_folders = glob.glob(os.path.join(mouse_f, mouseID + '_*'))

            for month_f in month_folders:
                if os.path.isdir(month_f):

                    _, mouse_month = os.path.split(month_f)

                    date_folders = glob.glob(os.path.join(month_f, mouse_month + '*'))

                    reconstruction3d_folders.extend([df for df in date_folders if os.path.isdir(df)])

    return reconstruction3d_folders


def create_3d_reconstruction_pickle_name(metadata, reconstruction3d_parent):

    trialtime = metadata['trialtime']
    trial_datetime_string = datetime_to_string_for_fname(trialtime)
    mouseID = metadata['mouseID']
    reconstruction3d_name = '_'.join(
        [mouseID,
        trial_datetime_string,
        '{:d}'.format(metadata['session_num']),
        '{:03d}'.format(metadata['video_number']),
        '3dreconstruction.pickle']
        )

    full_path = create_3dreconstruction_folder(metadata, reconstruction3d_parent)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    full_path_name = os.path.join(full_path, reconstruction3d_name)

    return full_path_name


def parse_3d_reconstruction_pickle_name(r3d_fullpath):

    _, r3d_fname = os.path.split(r3d_fullpath)
    r3d_fname, _ = os.path.splitext(r3d_fname)

    r3d_name_parts = r3d_fname.split('_')

    trial_datetimestring = r3d_name_parts[1] + '_' + r3d_name_parts[2]
    trial_datetime = fname_string_to_datetime(trial_datetimestring)

    session_num = int(r3d_name_parts[3])

    vid_num = int(r3d_name_parts[4])

    r3d_metadata = {
        'mouseID': r3d_name_parts[0],
        'time': trial_datetime,
        'vid_num': vid_num,
        'session_num': session_num,
    }

    return r3d_metadata


def find_folders_to_reconstruct(cropped_videos_parent):
    '''
    find all session folders in cropped_videos_parent that contain.pickle files with labeled coordinates
    :param cropped_videos_parent:
    :return:
    '''
    rat_folder_list = glob.glob(os.path.join(cropped_videos_parent, 'R*'))

    folders_to_reconstruct = []
    for rat_folder in rat_folder_list:
        _, rat_folder_name = os.path.split(rat_folder)
        ratID = rat_folder_name[:5]
        session_folder_list = glob.glob(os.path.join(rat_folder, ratID + '_*'))

        for session_folder in session_folder_list:
            _, session_folder_name = os.path.split(session_folder)

            # check for a direct view folder; if doesn't exist, just continue the loop
            test_direct_folder = os.path.join(session_folder, session_folder_name + '_direct')
            if not os.path.exists(test_direct_folder):
                continue
            # check to see if there are .pickle files containing labeled data
            ratID, session_name = parse_session_dir_name(session_folder_name)
            test_pickle_name = '_'.join((ratID, session_name[:-1], '*', 'full.pickle'))
            full_test_pickle_name = os.path.join(test_direct_folder, test_pickle_name)
            full_pickle_list = glob.glob(full_test_pickle_name)

            if not full_pickle_list:
                # if full_pickle_list is empty, continue the loop; don't need calibration if bodyparts aren't labeled yet
                continue

            pickle_metadata = parse_dlc_output_pickle_name(full_pickle_list[0])
            session_metadata = {'session_folder': session_folder,
                                'session_date': pickle_metadata['triggertime'],
                                'session_box': pickle_metadata['boxnum']}
            folders_to_reconstruct.append(session_metadata)

    return folders_to_reconstruct


def find_dlc_pickles_from_r3d_filename(r3d_file, parent_directories):

    cropped_vids_parent = parent_directories['cropped_vids_parent']

    r3d_metadata = parse_3d_reconstruction_pickle_name(r3d_file)

    mouse_folder = os.path.join(cropped_vids_parent, r3d_metadata['mouseID'])
    month_dirname = r3d_metadata['mouseID'] + '_' + r3d_metadata['time'].strftime('%Y%m')
    day_dirname = r3d_metadata['mouseID'] + '_' + r3d_metadata['time'].strftime('%Y%m%d')

    full_pickles = []
    meta_pickles = []
    for i_cam in range(2):
        cam_dirname = '_'.join([r3d_metadata['mouseID'],
                                date_to_string_for_fname(r3d_metadata['time']),
                                'cam{:02d}'.format(i_cam + 1)
                                ])
        cam_dir = os.path.join(mouse_folder, month_dirname, day_dirname, cam_dirname)
        test_name = '_'.join([r3d_metadata['mouseID'],
                              datetime_to_string_for_fname(r3d_metadata['time']),
                              '{:d}'.format(r3d_metadata['session_num']),
                              '{:03d}'.format(r3d_metadata['vid_num']),
                              'cam{:02d}'.format(i_cam + 1),
                              '*',
                              'full.pickle'])
        full_testname = os.path.join(cam_dir, test_name)
        full_pickle = glob.glob(full_testname)

        if len(full_pickle) == 1:
            full_pickles.append(full_pickle[0])
        elif len(full_pickle) > 1:
            print('more than one camera {:02d} full pickle file found for {}'.format(i_cam + 1, r3d_file))
        else:
            print('no camera {:02d} full pickle files found for {}'.format(i_cam + 1, r3d_file))

        test_name = '_'.join([r3d_metadata['mouseID'],
                              datetime_to_string_for_fname(r3d_metadata['time']),
                              '{:d}'.format(r3d_metadata['session_num']),
                              '{:03d}'.format(r3d_metadata['vid_num']),
                              'cam{:02d}'.format(i_cam + 1),
                              '*',
                              'meta.pickle'])
        meta_testname = os.path.join(cam_dir, test_name)
        meta_pickle = glob.glob(meta_testname)

        if len(meta_pickle) == 1:
            meta_pickles.append(meta_pickle[0])
        elif len(meta_pickle) > 1:
            print('more than one camera {:02d} full pickle file found for {}'.format(i_cam + 1, r3d_file))
        else:
            print('no camera {:02d} full pickle files found for {}'.format(i_cam + 1, r3d_file))

    return full_pickles, meta_pickles




def find_orig_movies_from_r3d_filename(r3d_file, parent_directories):

    video_root_folder = parent_directories['video_root_folder']




def get_Burgess_video_folders_to_crop(video_root_folder):
    """
    find all the lowest level directories within video_root_folder, which are presumably the lowest level folders that
    contain the videos to be cropped

    :param video_root_folder: root directory from which to extract the list of folders that contain videos to crop
    :return: crop_dirs - list of lowest level directories within video_root_folder
    """

    crop_dirs = []

    # assume that any directory that does not have a subdirectory contains videos to crop
    for root, dirs, files in os.walk(video_root_folder):
        if not dirs:
            crop_dirs.append(root)

    return crop_dirs


def create_Burgess_cropped_video_destination_list(cropped_vids_parent, video_folder_list, cam_list):
    """
    create subdirectory trees in which to store the cropped videos. Directory structure is ratID-->sessionID-->
        [sessionID_direct/leftmirror/rightmirror]
    :param cropped_vids_parent: parent directory in which to create directory tree
    :param video_folder_list: list of lowest level directories containing the original videos
    :param cam_list: list/tuple of camera numbers as integers
    :return: cropped_video_directories - list of num_cam element lists containing folders for cropped images from each camera for each session
    """

    num_cams = len(cam_list)
    cam_names = ['cam{:02d}'.format(i_cam) for i_cam in cam_list]
    cropped_video_directories = dict.fromkeys(cam_names)
    for i_cam in cam_list:
        cam_name = 'cam{:02d}'.format(i_cam)
        cropped_video_directories[cam_name] = []
    for i_vidfolder, crop_dir in enumerate(video_folder_list):
        _, session_dir = os.path.split(crop_dir)
        mouseID, session_date_str = parse_session_dir_name(session_dir)
        session_datetime = datetime.strptime(session_date_str, '%Y%m%d')
        session_month = session_datetime.strftime('%Y%m')
        month_folder = mouseID + '_' + session_month

        # create folders for cropped video for each camera
        for i_cam in range(num_cams):
            cam_name = 'cam{:02d}'.format(cam_list[i_cam])
            cropped_vid_dir = session_dir + '_cam{:02d}'.format(cam_list[i_cam])
            cropped_video_directories[cam_name].append(os.path.join(cropped_vids_parent, mouseID, month_folder, session_dir, cropped_vid_dir))

    return cropped_video_directories


def parse_Burgess_vid_name(full_vid_path):
    vid_metadata = {
        'mouseID': '',
        'time': datetime(1, 1, 1),
        'vid_num': 0,
        'session_num': 0,
        'cam_num': 99,
    }
    _, vid_name = os.path.split(full_vid_path)
    vid_name, _ = os.path.splitext(vid_name)

    vid_name_parts = vid_name.split('_')

    vid_metadata['mouseID'] = vid_name_parts[0]

    datetime_str = vid_name_parts[1] + '_' + vid_name_parts[2]
    vid_metadata['time'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    vid_metadata['vid_num'] = int(vid_name_parts[3])

    if len(vid_name_parts) == 6:
        # name includes session # (numbered from the first session overall or first session of the day?)
        vid_metadata['session_num'] = int(vid_name_parts[4])
        vid_metadata['cam_num'] = int(vid_name_parts[5][3:])
    else:
        vid_metadata['cam_num'] = int(vid_name_parts[4][3:])

    return vid_metadata


def create_calibration_data_folder(cal_data_parent, box_num, session_datetime):

    year_folder = 'calibrationfiles_' + session_datetime.strftime('%Y')
    month_folder = year_folder + session_datetime.strftime('%m')
    box_folder = month_folder + '_box{:2d}'.format(box_num)

    cal_data_folder = os.path.join(cal_data_parent, year_folder, month_folder, box_folder)

    if not os.path.exists(cal_data_folder):
        os.makedirs(cal_data_folder)

    return cal_data_folder


def create_optitrack_calibration_data_folder(cal_data_parent, session_datetime):

    year_folder = 'calibrationfiles_' + session_datetime.strftime('%Y')
    month_folder = year_folder + session_datetime.strftime('%m')

    cal_data_folder = os.path.join(cal_data_parent, year_folder, month_folder)

    if not os.path.exists(cal_data_folder):
        os.makedirs(cal_data_folder)

    return cal_data_folder


def date_to_string_for_fname(date_to_covert):

    format_string = '%Y%m%d'

    date_string = date_to_covert.strftime(format_string)

    return date_string


def datetime_to_string_for_fname(datetime_to_convert):

    format_string = '%Y%m%d_%H-%M-%S'

    datetime_string = datetime_to_convert.strftime(format_string)

    return datetime_string


def fname_string_to_datetime(string_to_convert):
    format_string = '%Y%m%d_%H-%M-%S'

    datetime_from_fname = datetime.strptime(string_to_convert, format_string)

    return datetime_from_fname


def fname_string_to_date(string_to_convert):
    format_string = '%Y%m%d'

    date_from_fname = datetime.strptime(string_to_convert, format_string)

    return date_from_fname


def parse_cropped_calibration_video_name(cropped_calibration_vid_name):
    """

    :param cropped_calibration_vid_name: form of GridCalibration_boxXX_YYYYMMDD_HH-mm-ss_view_top-bottom-left-right.avi
    :return:
    """
    cropped_cal_vid_metadata = {
        'boxnum': 99,
        'time': datetime(1, 1, 1)
    }
    _, cropped_cal_vid_name = os.path.split(cropped_calibration_vid_name)
    cropped_cal_vid_name, _ = os.path.splitext(cropped_cal_vid_name)

    cal_vid_name_parts = cropped_cal_vid_name.split('_')

    cropped_cal_vid_metadata['boxnum'] = int(cal_vid_name_parts[1][3:])

    datetime_str = cal_vid_name_parts[2] + '_' + cal_vid_name_parts[3]
    cropped_cal_vid_metadata['time'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    cropped_cal_vid_metadata['view'] = cal_vid_name_parts[4]

    crop_params_strings = cal_vid_name_parts[5].split('-')
    crop_params = [int(cp) for cp in crop_params_strings]
    cropped_cal_vid_metadata['crop_params'] = crop_params

    return cropped_cal_vid_metadata


def create_calibration_file_path(calibration_files_parent, calib_metadata):

    year_str = calib_metadata['time'].strftime('%Y')
    month_str = calib_metadata['time'].strftime('%Y%m')
    year_folder = os.path.join(calibration_files_parent, 'calibration_files_' + year_str)
    month_folder = os.path.join(year_folder, 'calibration_files_' + month_str)
    box_folder = os.path.join(month_folder, 'calibration_files_' + month_str + '_box{:02d}'.format(calib_metadata['boxnum']))

    return box_folder


def create_calibration_summary_name(full_calib_vid_name, calibration_files_parent):
    '''

    :param full_calib_vid_name:
    :return:
    '''

    # store the pickle file with the calibration parameters in the same folder as the calibration video
    calib_vid_path, _ = os.path.split(full_calib_vid_name)
    calib_metadata = parse_camera_calibration_video_name(full_calib_vid_name)

    calib_file_path = create_calibration_file_path(calibration_files_parent, calib_metadata)

    if not os.path.isdir(calib_file_path):
        os.makedirs(calib_file_path)

    calib_fname = 'calibrationdata_' + calib_metadata['time'].strftime('%Y%m%d_%H-%M-%S') + '_box{:02d}'.format(calib_metadata['boxnum']) + '.pickle'

    full_calib_fname = os.path.join(calib_file_path, calib_fname)

    return full_calib_fname