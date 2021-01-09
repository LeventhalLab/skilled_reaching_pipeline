import os
import glob


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