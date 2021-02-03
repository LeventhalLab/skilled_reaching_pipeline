import os
import glob
from datetime import datetime


def get_video_folders_to_crop(video_root_folder):
    '''
    find all the lowest level directories within video_root_folder, which are presumably the lowest level folders that
    contain the videos to be cropped

    :param video_root_folder: root directory from which to extract the list of folders that contain videos to crop
    :return: crop_dirs - list of lowest level directories within video_root_folder
    '''

    crop_dirs = []

    # assume that any directory that does not have a subdirectory contains videos to crop
    for root, dirs, files in os.walk(video_root_folder):
        if not dirs:
            crop_dirs.append(root)

    return crop_dirs


def create_cropped_video_destination_list(cropped_vids_parent, video_folder_list, view_list):
    '''
    create subdirectory trees in which to store the cropped videos. Directory structure is ratID-->[direct_view or
        mirror_views]-->ratID-->[sessionID_direct/leftmirror/rightmirror]
    :param cropped_vids_parent: parent directory in which to create directory tree
    :param video_folder_list: list of lowest level directories containing the original videos
    :return: cropped_video_directories
    '''

    cropped_video_directories = [[], [], []]
    for crop_dir in video_folder_list:
        _, session_dir = os.path.split(crop_dir)
        ratID, session_name = parse_session_dir_name(session_dir)

        # create direct view directory for this raw video directory
        cropped_vid_dir = session_dir + '_direct'
        direct_view_directory = os.path.join(cropped_vids_parent, 'direct_view', ratID, cropped_vid_dir)

        # create left mirror view directory for this raw video directory
        cropped_vid_dir = session_dir + '_leftmirror'
        left_view_directory = os.path.join(cropped_vids_parent, 'mirror_views', ratID, cropped_vid_dir)

        # create right mirror view directory for this raw video directory
        cropped_vid_dir = session_dir + '_rightmirror'
        right_view_directory = os.path.join(cropped_vids_parent, 'mirror_views', ratID, cropped_vid_dir)

        cropped_video_directories[0].append(direct_view_directory)
        cropped_video_directories[1].append(left_view_directory)
        cropped_video_directories[2].append(right_view_directory)

    return cropped_video_directories


def parse_session_dir_name(session_dir):
    '''

    :param session_dir - session directory name assumed to be of the form RXXXX_yyyymmddz, where XXXX is the rat number,
        yyyymmdd is the date, and z is a letter identifying distinct sessions on the same day (i.e., "a", "b", etc.)
    :return:
    '''

    dir_name_parts = session_dir.split('_')
    ratID = dir_name_parts[0]
    session_name = dir_name_parts[1]

    return ratID, session_name


def find_folders_to_analyze(cropped_vids_parent, view_list=('direct', 'leftmirror', 'rightmirror')):
    '''
    get the full list of directories containing cropped videos in the videos_to_analyze folder
    :param cropped_vids_parent: parent directory with subfolders direct_view and mirror_views, which have subfolders
        RXXXX-->RXXXXyyyymmddz[direct/leftmirror/rightmirror] (assuming default view list)
    :param view_list:
    :return: folders_to_analyze: dictionary containing a key for each member of view_list. Each key holds a list of
        folders to run through deeplabcut
    '''

    folders_to_analyze = dict(zip(view_list, ([] for _ in view_list)))

    for view in view_list:
        if 'direct' in view:
            view_folder = os.path.join(cropped_vids_parent, 'direct_view')
        elif 'mirror' in view:
            view_folder = os.path.join(cropped_vids_parent, 'mirror_views')
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
    '''
    extract metadata information from the video name
    :param cropped_video_name: video name with expected format RXXXX_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'direct', 'leftmirror', or 'rightmirror', and l-r-t-b are left, right, top, and bottom of the
        cropping windows from the original video
    :return: vid_metadata: dictionary containing the following keys
        ratID - rat ID as a string RXXXX
        boxnum - box number the session was run in. useful for making sure we used the right calibration. If unknown,
            set to 99
        triggertime - datetime object with when the trigger event occurred (date and time)
        vid_number - number of the video (ZZZ in the filename). This number is not necessarily unique within a session
            if it had to be restarted partway through
        vid_type - video type (e.g., '.avi', '.mp4', etc)
        crop_window - 4-element list [left, right, top, bottom] in pixels
    '''

    cropped_vid_metadata = {
        'ratID': '',
        'boxnum': 99,
        'triggertime': datetime.datetime(),
        'vid_number': 0,
        'vid_type': '',
        'crop_window': []
    }
    _, vid_name = os.path.split(cropped_video_name)
    vid_name, vid_type = os.path.splitext(vid_name)

    metadata_list = vid_name.split('_')

    cropped_vid_metadata['ratID'] = metadata_list[0]

    # if box number is stored in file name, then extract it
    if 'box' in metadata_list[1]:
        cropped_vid_metadata['boxnum'] = int(metadata_list[1][3:])
        next_metadata_idx = 2
    else:
        next_metadata_idx = 1

    datetime_str = metadata_list[1] + '_' + metadata_list[next_metadata_idx]
    cropped_vid_metadata['triggertime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    cropped_vid_metadata['vid_number'] = int(metadata_list[next_metadata_idx + 1])
    cropped_vid_metadata['vid_type'] = vid_type
    cropped_vid_metadata['view'] = metadata_list[next_metadata_idx + 2]

    left, right, top, bottom = list(map(int, metadata_list[next_metadata_idx + 3].split('-')))
    cropped_vid_metadata['crop_window'].extend(left, right, top, bottom)

    return cropped_vid_metadata


def create_marked_vids_folder(cropped_vid_folder, cropped_vids_parent, marked_vids_parent):
    '''
    :param cropped_vid_folder:
    :param cropped_vids_parent:
    :param marked_vids_parent:
    :return:
    '''

    # find the string 'cropped_videos' in cropped_vid_folder; everything after that is the relative path to create the marked_vids_folder
    cropped_vid_relpath = os.path.relpath(cropped_vid_folder, start=cropped_vids_parent)
    marked_vid_relpath = cropped_vid_relpath + 'marked'
    marked_vids_folder = os.path.join(marked_vids_parent, marked_vid_relpath)

    if not os.path.isdir(marked_vids_folder):
        os.mkdir(marked_vids_folder)

    return marked_vids_folder