import os
import glob
import sys
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


def create_cropped_video_destination_list(cropped_vids_parent, video_folder_list, view_list):
    """
    create subdirectory trees in which to store the cropped videos. Directory structure is ratID-->[direct_view or
        mirror_views]-->ratID-->[sessionID_direct/leftmirror/rightmirror]
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


def parse_session_dir_name(session_dir):
    """

    :param session_dir - session directory name assumed to be of the form RXXXX_yyyymmddz, where XXXX is the rat number,
        yyyymmdd is the date, and z is a letter identifying distinct sessions on the same day (i.e., "a", "b", etc.)
    :return:
    """

    dir_name_parts = session_dir.split('_')
    ratID = dir_name_parts[0]
    session_name = dir_name_parts[1]

    return ratID, session_name


def find_folders_to_analyze(cropped_videos_parent, view_list=('direct', 'leftmirror', 'rightmirror')):
    """
    get the full list of directories containing cropped videos in the videos_to_analyze folder
    :param cropped_videos_parent: parent directory with subfolders direct_view and mirror_views, which have subfolders
        RXXXX-->RXXXXyyyymmddz[direct/leftmirror/rightmirror] (assuming default view list)
    :param view_list:
    :return: folders_to_analyze: dictionary containing a key for each member of view_list. Each key holds a list of
        folders to run through deeplabcut
    """

    folders_to_analyze = dict(zip(view_list, ([] for _ in view_list)))

    for view in view_list:
        if 'direct' in view:
            view_folder = os.path.join(cropped_videos_parent, 'direct_view')
        elif 'mirror' in view:
            view_folder = os.path.join(cropped_videos_parent, 'mirror_views')
        else:
            print(view + ' does not contain the keyword "direct" or "mirror"')
            continue

        rat_folder_list = glob.glob(os.path.join(view_folder + '/R*'))

        for rat_folder in rat_folder_list:
            # make sure it's actually a folder
            if os.path.isdir(rat_folder):
                # assume the rat_folder directory name is the same as ratID (i.e., form of RXXXX)
                _, ratID = os.path.split(rat_folder)
                session_name = ratID + '_*_' + view
                session_dir_list = glob.glob(rat_folder + '/' + session_name)

                # make sure we only include directories (just in case there are some stray files with the right names)
                session_dir_list = [session_dir for session_dir in session_dir_list if os.path.isdir(session_dir)]

                folders_to_analyze[view].extend(session_dir_list)

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
        'session_name': '',
        'boxnum': 99,
        'triggertime': datetime(1,1,1),
        'video_number': 0,
        'video_type': '',
        'video_name': ''
    }
    vid_path, vid_name = os.path.split(video_name)
    video_metadata['video_name'] = vid_name
    # the last folder in the tree should have the session name
    _, video_metadata['session_name'] = os.path.split(vid_path)
    vid_name, vid_type = os.path.splitext(vid_name)

    metadata_list = vid_name.split('_')

    video_metadata['ratID'] = metadata_list[0]

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

    #todo: write the scorername into the pickle metadata dictionary
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
    day_folder = 'calibration_files_' + datetime.strftime(vid_metadata['triggertime'], '%Y%m%d')
    box_folder = day_folder + '_box{:2d}'.format(vid_metadata['boxnum'])

    calibration_file_tree = os.path.join(calibration_parent, year_folder, month_folder, day_folder, box_folder)

    return calibration_file_tree


def find_dlc_output_pickles(video_metadata, marked_videos_parent, view_list=('direct', 'leftmirror', 'rightmirror')):
    """

    :param video_metadata:
    :param marked_videos_parent:
    :param view_list:
    :return:
    """
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
            sys.exit('No dlc output file found for {}'.format(video_metadata['video_name']))

        if len(pickle_meta_list) == 0:
            # no pickle file for this view
            sys.exit('No dlc output metadata file found for {}'.format(video_metadata['video_name']))

        dlc_output_pickle_names[view] = pickle_full_list[0]
        dlc_metadata_pickle_names[view] = pickle_meta_list[0]

    return dlc_output_pickle_names, dlc_metadata_pickle_names


def construct_dlc_output_pickle_names(video_metadata, view):
    """

    :param video_metadata:
    :param view: string containing 'direct', 'leftmirror', or 'rightmirror'
    :return:
    """
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


def find_calibration_file(video_metadata, calibration_parent):

    date_string = video_metadata['triggertime'].strftime('%Y%m%d')
    year_folder = os.path.join(calibration_parent, date_string[0:4])
    month_folder = os.path.join(year_folder, date_string[0:6] + '_calibration')
    calibration_folder = os.path.join(month_folder, date_string[0:6] + '_calibration_files')

    test_name = 'SR_boxCalibration_box{:02d}_{}.mat'.format(video_metadata['boxnum'], date_string)
    test_name = os.path.join(calibration_folder, test_name)

    if os.path.exists(test_name):
        return test_name
    else:
        sys.exit('No calibration file found for ' + video_metadata[''])


    pass