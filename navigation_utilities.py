import os
import glob
import sys
import cv2
import numpy as np
import pandas as pd
import shutil
from datetime import datetime, timedelta
import navigation_utilities


def find_rat_cropped_session_folder(session_metadata, parent_directories):
    '''

    :param session_metadata:
    :param parent_directories:
    :return:
    '''
    cropped_vids_parent = parent_directories['cropped_videos_parent']
    ratID = session_metadata['ratID']
    rat_folder = os.path.join(cropped_vids_parent, ratID)

    metadata_keys = session_metadata.keys()
    test_datetime_keys = ['trialtime', 'date', 'time', 'triggertime']
    for test_key in test_datetime_keys:
        if test_key in metadata_keys:
            session_date = session_metadata[test_key]

    session_folder_name = '_'.join((ratID,
                                    fname_date2string(session_date),
                                    session_metadata['task'],
                                    'ses{:02d}'.format(session_metadata['session_num'])))
    session_dir = os.path.join(rat_folder, session_folder_name)

    if os.path.exists(session_dir):
        return session_dir
    else:
        return None


def find_cropped_session_folder(session_metadata, parent_directories):
    '''

    :param session_metadata:
    :param parent_directories:
    :return:
    '''

    cropped_vids_parent = parent_directories['cropped_vids_parent']
    mouseID = session_metadata['mouseID']
    mouse_folder = os.path.join(cropped_vids_parent, mouseID)

    metadata_keys = session_metadata.keys()
    test_datetime_keys = ['trialtime', 'date', 'time']
    for test_key in test_datetime_keys:
        if test_key in metadata_keys:
            session_date = session_metadata[test_key]

    date_dir = os.path.join(mouse_folder, mouseID + '_' + session_date.strftime('%Y%m%d'))

    potential_cam_folders = glob.glob(os.path.join(date_dir, '*_cam*'))
    cam_folders = [pcf for pcf in potential_cam_folders if os.path.isdir(pcf)]

    # make sure the cam_folders are in order of camera number
    cam_nums_from_folders = [int(cf[-2:]) for cf in cam_folders]
    sorted_cam_folders = [scf for (_, scf) in sorted(zip(cam_nums_from_folders, cam_folders), key=lambda x: x[0])]

    return date_dir, sorted_cam_folders


def find_vid_pair_from_session(vid_folder, session_num, vid_type='.avi'):

    test_vid_num = 0
    vid_metadata = parse_session_dir_name(vid_folder)

    vid_pair = []
    for i_cam in range(2):
        # sometimes, the session number is embedded in the filename as a single digit, sometimes as a 2-digit 0-padded
        # number
        test_names = ['_'.join([vid_metadata[0],
                              vid_metadata[1],
                              '*-*-*',
                              '{:d}'.format(session_num),
                              '{:03d}'.format(test_vid_num),
                              'cam{:02d}'.format(i_cam+1) + vid_type
                              ]),
                      '_'.join([vid_metadata[0],
                                vid_metadata[1],
                                '*-*-*',
                                '{:02d}'.format(session_num),
                                '{:03d}'.format(test_vid_num),
                                'cam{:02d}'.format(i_cam + 1) + vid_type
                                ])]
        camvid_lists = [glob.glob(os.path.join(vid_folder, tn)) for tn in test_names]

        if not all(camvid_lists):
            # no files found. maybe this was a stim session?
            test_names = ['_'.join(['stim' + vid_metadata[0],
                                    vid_metadata[1],
                                    '*-*-*',
                                    '{:d}'.format(session_num),
                                    '{:03d}'.format(test_vid_num),
                                    'cam{:02d}'.format(i_cam + 1) + vid_type
                                    ]),
                          '_'.join(['stim' + vid_metadata[0],
                                    vid_metadata[1],
                                    '*-*-*',
                                    '{:02d}'.format(session_num),
                                    '{:03d}'.format(test_vid_num),
                                    'cam{:02d}'.format(i_cam + 1) + vid_type
                                    ])]
        camvid_lists = [glob.glob(os.path.join(vid_folder, tn)) for tn in test_names]

        if not all(camvid_lists):
            return None

        for cvl in camvid_lists:
            if len(cvl) == 1:
                # this must be a valid video name
                vid_pair.append(cvl[0])
                break
    return vid_pair


def sessions_in_optitrack_folder(vid_folder, vid_type='*.avi'):

    vid_list = glob.glob(os.path.join(vid_folder, '*' + vid_type))

    session_nums = []
    for vid in vid_list:
        vid_metadata = parse_Burgess_vid_name(vid)
        session_nums.append(vid_metadata['session_num'])

    session_nums = np.unique(np.array(session_nums))

    return session_nums


# def get_video_folders_to_crop(video_root_folder, rats_to_analyze='all'):
#     """
#     find all the lowest level directories within video_root_folder, which are presumably the lowest level folders that
#     contain the videos to be cropped
#
#     :param video_root_folder: root directory from which to extract the list of folders that contain videos to crop
#     :return: crop_dirs - list of lowest level directories within video_root_folder
#     """
#
#     crop_dirs = []
#
#     # assume that any directory that does not have a subdirectory contains videos to crop
#     for root, dirs, files in os.walk(video_root_folder):
#         if not dirs:
#             crop_dirs.append(root)
#
#     return crop_dirs


def isvalid_ratsessiondate(test_string):

    if len(test_string) != 14:
        return False

    test_string_parts = test_string.split('_')
    if len(test_string_parts) != 2:
        return False

    if is_valid_ratID(test_string_parts[0]) and test_string_parts[1].isdigit():
        return True
    else:
        return False


def is_valid_ratID(test_string):

    if len(test_string) == 5 and test_string[1:].isdigit():
        isvalid = True
    else:
        isvalid = False

    return isvalid


def session_metadata_from_path(full_pathname):
    '''

    :param full_pathname:
    :return:
    '''

    # if path is a folder, assume it is the lowest folder containing session data of the form
    # ratID_YYYYMMDD_task_sessionXX where XX is a 2-digit integer
    if os.path.isfile(full_pathname):
        pname, fname = os.path.split(full_pathname)
        fname_parts = fname.split('_')
        if 'current' in fname_parts[-1]:
            # current was specified at the end of the filename
            current_value = int(fname_parts[-1][7:10]) / 1000
        else:
            current_value = 0.
    else:
        pname = full_pathname
        current_value = 0.
    _, session_folder = os.path.split(pname)

    folder_parts = session_folder.split('_')
    ratID = folder_parts[0]
    rat_num = int(ratID[1:])

    session_date = datetime_from_fname_string(folder_parts[1])

    session_num = int(folder_parts[-1][-2:])
    session_metadata = {
                        'ratID': ratID,
                        'rat_num': rat_num,
                        'date': session_date,
                        'task': folder_parts[2],
                        'session_num': session_num,
                        'current': current_value
    }

    return session_metadata


def get_video_folders_to_crop(video_root_folder, rats_to_analyze='all'):

    rat_folders = glob.glob(os.path.join(video_root_folder, 'R*'))

    vid_folders_to_crop = []
    for rf in rat_folders:
        if os.path.isdir(rf):
            _, ratID = os.path.split(rf)
            if is_valid_ratID(ratID):
                rat_num = int(ratID[1:])
                if rat_num in rats_to_analyze or rats_to_analyze == 'all':
                    date_folders = glob.glob(os.path.join(rf, ratID + '*'))
                    for df in date_folders:
                        if os.path.isdir(df):
                            _, session_name = os.path.split(df)
                            if isvalid_ratsessiondate(session_name):
                                # find valid skilled reaching sessions
                                sr_folders = glob.glob(os.path.join(df, session_name + '_sr_*'))
                                srchrim_folders = glob.glob(os.path.join(df, session_name + '_srchrim_*'))
                                for srf in sr_folders:
                                    if os.path.isdir(srf):
                                        vid_folders_to_crop.append(srf)
                                for srchrimf in srchrim_folders:
                                    if os.path.isdir(srchrimf):
                                        vid_folders_to_crop.append(srchrimf)

    return vid_folders_to_crop



def find_traj_files(traj_directory):

    _, dir_name = os.path.split(traj_directory)
    ratID = dir_name[:5]
    session_datestring = dir_name[6:-1]

    test_fname = '_'.join((ratID,
                            'box*',
                            session_datestring,
                            '*.pickle'))
    test_string = os.path.join(traj_directory, test_fname)
    traj_files = glob.glob(test_string)

    return traj_files


def find_optitrack_r3d_files(r3d_folder):

    _, dir_name = os.path.split(r3d_folder)
    fname_parts = dir_name.split('_')
    mouseID = fname_parts[0]
    session_datestring = fname_parts[1]

    test_fname = '_'.join((mouseID,
                            session_datestring,
                            '*3dreconstruction.pickle'))
    test_string = os.path.join(r3d_folder, test_fname)
    r3d_files = glob.glob(test_string)

    return r3d_files


def find_original_optitrack_videos(video_root_folder, metadata, vidtype='.avi'):

    if vidtype[0] != '.':
        vidtype = '.' + vidtype

    mouseID = metadata['mouseID']
    trialtime = metadata['trialtime']
    month_dir = mouseID + '_' + trialtime.strftime('%Y%m')
    date_dir = mouseID + '_' + trialtime.strftime('%Y%m%d')

    test_vid_name = '_'.join(['*' + mouseID,
                             fname_time2string(trialtime),
                             str(metadata['session_num']),
                             '{:03d}'.format(metadata['vid_num']),
                             'cam*' + vidtype
                            ])
    test_vid_name = os.path.join(video_root_folder, mouseID, date_dir, test_vid_name)

    vid_list = glob.glob(test_vid_name)

    if len(vid_list) == 0:
        test_vid_name = '_'.join(['*' + mouseID,
                                  fname_time2string(trialtime),
                                  '{:02d}'.format(metadata['session_num']),
                                  '{:03d}'.format(metadata['vid_num']),
                                  'cam*' + vidtype
                                  ])
        test_vid_name = os.path.join(video_root_folder, mouseID, date_dir, test_vid_name)

        vid_list = glob.glob(test_vid_name)

    # full_vid_name = os.path.join(video_root_folder, mouseID, month_dir, date_dir, vid_name)
    #
    # if not os.path.exists(full_vid_name):
    #     full_vid_name = None

    return vid_list


def find_cropped_optitrack_videos(cropped_vids_parent, metadata, num_cams=2, vidtype='.avi'):

    if vidtype[0] != '.':
        vidtype = '.' + vidtype

    mouseID = metadata['mouseID']
    trialtime =  metadata['trialtime']
    month_dir = mouseID + '_' + trialtime.strftime('%Y%m')
    date_dir = mouseID + '_' + trialtime.strftime('%Y%m%d')
    cam_dirs = [date_dir + '_cam{:02d}'.format(i_cam + 1) for i_cam in range(num_cams)]

    test_vid_name = '_'.join(['*' + mouseID,
                             fname_time2string(trialtime),
                             '{:02d}'.format(metadata['session_num']),
                             '{:03d}'.format(metadata['vid_num']),
                             'cam*' + vidtype
                            ])
    test_vid_names = [os.path.join(cropped_vids_parent, mouseID, date_dir, cam_dirs[i_cam], test_vid_name) for i_cam in range(num_cams)]

    cropped_vids = []
    for i_cam in range(num_cams):
        vid_list = glob.glob(test_vid_names[i_cam])
        if len(vid_list) == 1:
            cropped_vids.append(vid_list[0])
        else:
            cropped_vids.append(None)

    if all([cropped_vid is None for cropped_vid in cropped_vids]):
        # sometimes the session number is 2 digits (i.e., '05'), and sometimes it's one digit (i.e., '5'). Need to check
        # both possibilities
        test_vid_name = '_'.join(['*' + mouseID,
                                  fname_time2string(trialtime),
                                  '{:d}'.format(metadata['session_num']),
                                  '{:03d}'.format(metadata['vid_num']),
                                  'cam*' + vidtype
                                  ])
        test_vid_names = [os.path.join(cropped_vids_parent, mouseID, date_dir, cam_dirs[i_cam], test_vid_name) for i_cam in range(num_cams)]
        cropped_vids = []
        for i_cam in range(num_cams):
            vid_list = glob.glob(test_vid_names[i_cam])
            if len(vid_list) == 1:
                cropped_vids.append(vid_list[0])
            else:
                cropped_vids.append(None)

    # vid_list = glob.glob(test_vid_name)

    # full_vid_name = os.path.join(video_root_folder, mouseID, month_dir, date_dir, vid_name)
    #
    # if not os.path.exists(full_vid_name):
    #     full_vid_name = None

    return cropped_vids


def create_cropped_video_destination_list(cropped_vids_parent, video_folder_list, view_list):
    """
    create subdirectory trees in which to store the cropped videos. Directory structure is ratID-->sessionID-->
        [sessionID_direct/lm/rm]
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
        cropped_vid_dir = session_dir + '_lm'
        left_view_directory = os.path.join(cropped_vids_parent, ratID, session_dir, cropped_vid_dir)

        # create right mirror view directory for this raw video directory
        cropped_vid_dir = session_dir + '_rm'
        right_view_directory = os.path.join(cropped_vids_parent, ratID, session_dir, cropped_vid_dir)

        cropped_video_directories[0].append(direct_view_directory)
        cropped_video_directories[1].append(left_view_directory)
        cropped_video_directories[2].append(right_view_directory)

    return cropped_video_directories


def create_trajectory_name(h5_metadata, session_metadata, calibration_data, parent_directories):

    # need to create session name here
    session_name = '_'.join((h5_metadata['ratID'],
                             date_to_string_for_fname(h5_metadata['triggertime']),
                             session_metadata['task'],
                             'ses{:02d}'.format(session_metadata['session_num'])))
    traj_folder = os.path.join(parent_directories['trajectories_parent'], h5_metadata['ratID'], session_name)

    if not os.path.exists(traj_folder):
        os.makedirs(traj_folder)

    traj_fname = '_'.join((h5_metadata['ratID'],
                           'b{:02d}'.format(h5_metadata['boxnum']),
                           datetime_to_string_for_fname(h5_metadata['triggertime']),
                           '{:03d}_r3d.pickle'.format(h5_metadata['video_number'])))

    traj_fname = os.path.join(traj_folder, traj_fname)

    return traj_fname


def get_trialsdb_name(parent_directories, ratID, task):
    # rat_folder = os.path.join(parent_directories[source_volume]['data'], ratID)
    task_folder = os.path.join(parent_directories['analysis'], task)
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)

    fname = '_'.join([ratID,
                      task,
                      'trialsdb.pickle'])

    full_name = os.path.join(task_folder, fname)

    return full_name


def get_aggregated_singlerat_data_name(parent_directories, ratID, task):
    # rat_folder = os.path.join(parent_directories[source_volume]['data'], ratID)
    task_folder = os.path.join(parent_directories['analysis'], task)
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)

    fname = '_'.join([ratID,
                      task,
                      'aggregated.pickle'])

    full_name = os.path.join(task_folder, fname)

    return full_name


def create_3dvid_name(traj_metadata, session_metadata, parent_directories):

    traj_summary_folder = get_3dsummaries_folder(session_metadata, parent_directories)

    if not os.path.exists(traj_summary_folder):
        os.makedirs(traj_summary_folder)

    vid_name = '_'.join((traj_metadata['ratID'],
                         'b{:02d}'.format(traj_metadata['boxnum']),
                         datetime_to_string_for_fname(traj_metadata['triggertime']),
                         '{:03d}.avi'.format(traj_metadata['video_number'])
                         ))

    return os.path.join(traj_summary_folder, vid_name)


def get_3dsummaries_folder(session_metadata, parent_directories):

    rat_traj_summary_folder = os.path.join(parent_directories['trajectory_summaries'], session_metadata['ratID'])
    if 'date' in session_metadata.keys():
        session_name = '_'.join((session_metadata['ratID'],
                                 date_to_string_for_fname(session_metadata['date']),
                                 session_metadata['task'],
                                 'ses{:02d}'.format(session_metadata['session_num'])))
    elif 'session_date' in session_metadata.keys():
        session_name = '_'.join((session_metadata['ratID'],
                                 date_to_string_for_fname(session_metadata['session_date']),
                                 session_metadata['task'],
                                 'ses{:02d}'.format(session_metadata['session_num'])))

    session_traj_summary_folder = os.path.join(rat_traj_summary_folder, session_name)

    if not os.path.exists(session_traj_summary_folder):
        os.makedirs(session_traj_summary_folder)

    return session_traj_summary_folder


def get_3dsummaries_basename(traj_metadata, session_metadata, parent_directories):
    session_traj_summary_folder = get_3dsummaries_folder(session_metadata, parent_directories)

    basename = '_'.join((session_metadata['ratID'],
                         'b{:02d}'.format(traj_metadata['boxnum']),
                         datetime_to_string_for_fname(traj_metadata['triggertime']),
                         '{:03d}'.format(traj_metadata['video_number'])))

    basename_3dsummary = os.path.join(session_traj_summary_folder, basename)

    return basename_3dsummary


def parse_trajectory_name(full_traj_path):

    _, traj_name = os.path.split(full_traj_path)
    traj_name, _ = os.path.splitext(traj_name)

    traj_metadata_list = traj_name.split('_')

    trigtime_str = traj_metadata_list[2] + '_' + traj_metadata_list[3]

    traj_metadata = {'ratID': traj_metadata_list[0],
                     'boxnum': int(traj_metadata_list[1][1:]),
                     'triggertime': datetime.strptime(trigtime_str, '%Y%m%d_%H-%M-%S'),
                     'video_number': int(traj_metadata_list[4])
                     }

    return traj_metadata


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
    date_folder_test = os.path.join(rat_folder, video_metadata['ratID'] + '_' + datestring + '*')
    date_folder_list = glob.glob(date_folder_test)
    if len(date_folder_list) == 1:
        date_folder = date_folder_list[0]
    elif len(date_folder_list) > 1:
        print('more than one session folder for {} on {}', video_metadata['ratID'], datestring)
        return None
    else:
        print('no session folders for {} on {}', video_metadata['ratID'], datestring)
        return None

    _, date_foldername = os.path.split(date_folder)
    session_folder = '_'.join((date_foldername, video_metadata['task'], 'ses{:02d}'.format(video_metadata['session_num'])))
    session_folder = os.path.join(date_folder, session_folder)

    if not os.path.exists(session_folder):
        return None

    if 'vid_num' in video_metadata.keys():
        vid_num = video_metadata['vid_num']
    elif 'video_number' in video_metadata.keys():
        vid_num = video_metadata['video_number']

    timestring = datetime_to_string_for_fname(video_metadata['triggertime'])
    vid_name = '_'.join((video_metadata['ratID'],
                         'b{:02d}'.format(video_metadata['boxnum']),
                         timestring,
                         '{:03d}.avi'.format(vid_num)
                         ))
    orig_vid_name = os.path.join(session_folder, vid_name)

    if not os.path.exists(orig_vid_name):
        return None

    return orig_vid_name


def parse_paw_trajectory_fname(paw_trajectory_fname):

    _, pt_name = os.path.split(paw_trajectory_fname)

    fname_parts = pt_name.split('_')

    ratID = fname_parts[0]
    rat_num = int(ratID[1:])

    box_num = int(fname_parts[1][-1:])

    triggertime = fname_string_to_datetime(fname_parts[2] + '_' + fname_parts[3])

    vid_num = int(fname_parts[4])

    traj_metadata = {
        'ratID': ratID,
        'rat_num': rat_num,
        'boxnum': box_num,
        'triggertime': triggertime,
        'vid_num': vid_num
    }

    return traj_metadata


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


def parse_croppedvid_dir_name(session_dir):
    """

    :param session_dir - session directory name assumed to be of the form RXXXX_yyyymmddz, where XXXX is the rat number,
        yyyymmdd is the date, and z is a letter identifying distinct sessions on the same day (i.e., "a", "b", etc.)
    :return:
    """
    _, session_dir_name = os.path.split(session_dir)

    dir_name_parts = session_dir_name.split('_')
    ratID = dir_name_parts[0]
    session_name = '_'.join(dir_name_parts[1:])

    session_date = fname_string_to_date(dir_name_parts[1])

    session_metadata = {'ratID': ratID,
                        'date': session_date,
                        'task': dir_name_parts[2],
                        'session_num': int(dir_name_parts[3][-2:])}

    return session_metadata


def test_dlc_h5_name_from_session_metadata(session_metadata, cam_name, filtered=True):

    if 'date' in session_metadata.keys():
        session_date = session_metadata['date']
    elif 'session_date' in session_metadata.keys():
        session_date = session_metadata['session_date']

    if filtered:
        test_name = '_'.join((session_metadata['ratID'],
                              'b*',
                              date_to_string_for_fname(session_date),
                              '*',
                              cam_name,
                              '*',
                              'el_filtered.h5'))
    else:
        test_name = '_'.join((session_metadata['ratID'],
                              'b*',
                              date_to_string_for_fname(session_date),
                              '*',
                              cam_name,
                              '*',
                              'el.h5'))

    return test_name


def test_dlc_pickle_name_from_session_metadata(session_metadata, cam_name, suffix='full'):

    if 'date' in session_metadata.keys():
        session_date = session_metadata['date']
    elif 'session_date' in session_metadata.keys():
        session_date = session_metadata['session_date']

    test_name = '_'.join((session_metadata['ratID'],
                          'b*',
                          date_to_string_for_fname(session_date),
                          '*',
                          cam_name,
                          '*',
                          suffix + '.pickle'))

    return test_name


def test_dlc_h5_name_from_h5_metadata(h5_metadata, cam_name, filtered=True):

    # crop_string = '-'.join([str(cb) for cb in h5_metadata['crop_window']])
    if filtered:
        test_name = '_'.join((h5_metadata['ratID'],
                              'b{:02d}'.format(h5_metadata['boxnum']),
                              datetime_to_string_for_fname(h5_metadata['triggertime']),
                              '{:03d}'.format(h5_metadata['video_number']),
                              cam_name,
                              '*',
                              'el_filtered.h5'))
    else:
        test_name = '_'.join((h5_metadata['ratID'],
                              'b{:02d}'.format(h5_metadata['boxnum']),
                              datetime_to_string_for_fname(h5_metadata['triggertime']),
                              '{:03d}'.format(h5_metadata['video_number']),
                              cam_name,
                              '*',
                              'el.h5'))

    return test_name


def match_dlc_h5_views(session_metadata):
    pass

def find_folders_to_analyze(cropped_videos_parent, view_list=None):
    """
    get the full list of directories containing cropped videos in the videos_to_analyze folder
    :param cropped_videos_parent: parent directory with subfolders direct_view and mirror_views, which have subfolders
        RXXXX-->RXXXXyyyymmddz[direct/lm/rm] (assuming default view list)
    :param view_list:
    :return: folders_to_analyze: dictionary containing a key for each member of view_list. Each key holds a list of
        folders to run through deeplabcut
    """

    if view_list is None:
        view_list = ('dir', 'lm', 'rm')

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
        RXXXX-->RXXXXyyyymmddz[direct/lm/rm] (assuming default view list)
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
            # monthfolder_name = mouseID + '_*'
            # month_dir_list = glob.glob(os.path.join(mouse_folder, monthfolder_name))
            # # make sure we only include directories (just in case there are some stray files with the right names)
            # month_dir_list = [month_dir for month_dir in month_dir_list if os.path.isdir(month_dir)]
            # for month_dir in month_dir_list:
            #     _, cur_month_dir_name = os.path.split(month_dir)
            sessionfolder_name = mouseID + '*'
            session_dir_list = glob.glob(os.path.join(mouse_folder, sessionfolder_name))
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
        where [view] is 'dir', 'lm', or 'rm', and l-r-t-b are left, right, top, and bottom of the
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
        cropped_vid_metadata['boxnum'] = int(metadata_list[1][-1:])
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
    :param cropped_video_name: video name with expected format RXXXX_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'dir', 'lm', or 'rm', and l-r-t-b are left, right, top, and bottom of the
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
        'triggertime': datetime(1,1,1),
        'video_number': 0,
        'cam_num': '',
        'video_type': '',
        'crop_window': [],
        'cropped_video_name': ''
    }
    _, vid_name = os.path.split(cropped_video_name)
    cropped_vid_metadata['cropped_video_name'] = vid_name
    vid_name, vid_type = os.path.splitext(vid_name)

    metadata_list = vid_name.split('_')

    cropped_vid_metadata['mouseID'] = metadata_list[0]

    datetime_str = metadata_list[1] + '_' + metadata_list[2]
    cropped_vid_metadata['triggertime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    cropped_vid_metadata['session_num'] = int(metadata_list[3])
    cropped_vid_metadata['video_type'] = vid_type
    cropped_vid_metadata['view'] = metadata_list[next_metadata_idx + 3]

    left, right, top, bottom = list(map(int, metadata_list[next_metadata_idx + 4].split('-')))
    cropped_vid_metadata['crop_window'].extend(left, right, top, bottom)

    return cropped_vid_metadata


def parse_cropped_optitrack_video_folder(cropped_video_folder):
    """
    extract metadata information from the video name
    :param cropped_video_name: video name with expected format mouseID_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'dir', 'lm', or 'rm', and l-r-t-b are left, right, top, and bottom of the
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

    cropped_folder_metadata = {
        'mouseID': '',
        'session_date': datetime(1,1,1),
        'cam_num': 0,
        'cropped_video_folder': ''
    }
    try:
        _, folder_name = os.path.split(cropped_video_folder)
    except:
        pass
    cropped_folder_metadata['cropped_video_folder'] = folder_name

    metadata_list = folder_name.split('_')

    if metadata_list[0][0:4] == 'stim':
        cropped_folder_metadata['mouseID'] = metadata_list[0][4:]
    else:
        cropped_folder_metadata['mouseID'] = metadata_list[0]

    cropped_folder_metadata['session_date'] = datetime.strptime(metadata_list[1], '%Y%m%d')

    cropped_folder_metadata['cam_num'] = int(metadata_list[2][3:])

    return cropped_folder_metadata

def parse_cropped_optitrack_video_name(cropped_video_name):
    """
    extract metadata information from the video name
    :param cropped_video_name: video name with expected format mouseID_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'dir', 'lm', or 'rm', and l-r-t-b are left, right, top, and bottom of the
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
        'session_num': 1,
        'triggertime': datetime(1,1,1),
        'video_number': 0,
        'cam_num': 0,
        'video_type': '',
        'crop_window': [],
        'cropped_video_name': '',
        'isrotated': False
    }
    try:
        _, vid_name = os.path.split(cropped_video_name)
    except:
        pass
    try:
        cropped_vid_metadata['cropped_video_name'] = vid_name
    except:
        pass
    vid_name, vid_type = os.path.splitext(vid_name)

    metadata_list = vid_name.split('_')

    if metadata_list[0][0:4] == 'stim':
        cropped_vid_metadata['mouseID'] = metadata_list[0][4:]
    else:
        cropped_vid_metadata['mouseID'] = metadata_list[0]

    datetime_str = metadata_list[1] + '_' + metadata_list[2]
    cropped_vid_metadata['triggertime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    cropped_vid_metadata['session_num'] = int(metadata_list[3])

    cropped_vid_metadata['video_number'] = int(metadata_list[4])

    cropped_vid_metadata['cam_num'] = int(metadata_list[5][3:])

    left, right, top, bottom = list(map(int, metadata_list[6].split('-')))
    cropped_vid_metadata['crop_window'] = (left, right, top, bottom)

    if metadata_list[7] == 'rotated':
        cropped_vid_metadata['isrotated'] = True

    cropped_vid_metadata['video_type'] = vid_type

    return cropped_vid_metadata


def parse_video_name(video_name):
    """
    extract metadata information from the video name
    :param video_name: video name with expected format RXXXX_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'dir', 'lm', or 'rm', and l-r-t-b are left, right, top, and bottom of the
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
        video_metadata['boxnum'] = int(metadata_list[1][-1:])
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
                                                  fname_time2string(video_metadata['triggertime']),
                                                  video_metadata['video_number'])
    video_name = os.path.join(videos_parent, 'videos_to_crop', video_metadata['ratID'], video_metadata['session_name'], video_name)
    return video_name


def parse_dlc_output_h5_name(dlc_output_h5_name):
    """
    extract metadata information from the pickle file name
    :param dlc_output_h5_name: video name with expected format RXXXX_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'dir', 'lm', or 'rm', and l-r-t-b are left, right, top, and bottom of the
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

    h5_metadata = {
        'ratID': '',
        'rat_num': 0,
        'boxnum': 99,
        'triggertime': datetime(1,1,1),
        'video_number': 0,
        'view': '',
        'crop_window': [],
        'scorername': '',
        'h5_name': ''
    }
    view_folder, h5_name = os.path.split(dlc_output_h5_name)
    h5_metadata['h5_name'] = h5_name
    h5_name, vid_type = os.path.splitext(h5_name)

    metadata_list = h5_name.split('_')

    h5_metadata['ratID'] = metadata_list[0]
    num_string = ''.join(filter(lambda i: i.isdigit(), h5_metadata['ratID']))
    h5_metadata['rat_num'] = int(num_string)

    # if box number is stored in file name, then extract it
    if 'b' in metadata_list[1]:
        h5_metadata['boxnum'] = int(metadata_list[1][-1:])
        next_metadata_idx = 2
    else:
        next_metadata_idx = 1

    datetime_str = metadata_list[next_metadata_idx] + '_' + metadata_list[1+next_metadata_idx]
    h5_metadata['triggertime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    h5_metadata['video_number'] = int(metadata_list[next_metadata_idx + 2])
    h5_metadata['view'] = metadata_list[next_metadata_idx + 3]

    # 'DLC' gets appended to the last cropping parameter in the filename by deeplabcut
    crop_window_strings = metadata_list[next_metadata_idx + 4].split('-')
    left, right, top = list(map(int, crop_window_strings[:-1]))

    # find where 'DLC' starts in the last crop_window_string
    dlc_location = crop_window_strings[-1].find('DLC')
    bottom = int(crop_window_strings[-1][:dlc_location])

    h5_metadata['crop_window'].extend((left, right, top, bottom))

    _, view_foldername = os.path.split(view_folder)
    folder_nameparts = view_foldername.split('_')
    h5_metadata['task'] = folder_nameparts[2]
    h5_metadata['session_num'] = int(folder_nameparts[3][-2:])

    h5_metadata['scorername'] = '_'.join(('DLC',
                                          metadata_list[next_metadata_idx + 5],
                                          metadata_list[next_metadata_idx + 6],
                                          metadata_list[next_metadata_idx + 7]))

    return h5_metadata


def parse_dlc_output_pickle_name(dlc_output_pickle_name):
    """
    extract metadata information from the pickle file name
    :param dlc_output_pickle_name: video name with expected format RXXXX_yyyymmdd_HH-MM-SS_ZZZ_[view]_l-r-t-b.avi
        where [view] is 'dir', 'lm', or 'rm', and l-r-t-b are left, right, top, and bottom of the
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
    view_folder, pickle_name = os.path.split(dlc_output_pickle_name)
    pickle_metadata['pickle_name'] = pickle_name
    pickle_name, vid_type = os.path.splitext(pickle_name)

    metadata_list = pickle_name.split('_')

    pickle_metadata['ratID'] = metadata_list[0]
    num_string = ''.join(filter(lambda i: i.isdigit(), pickle_metadata['ratID']))
    pickle_metadata['rat_num'] = int(num_string)

    # if box number is stored in file name, then extract it
    if 'b' in metadata_list[1]:
        pickle_metadata['boxnum'] = int(metadata_list[1][-1:])
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

    _, view_foldername = os.path.split(view_folder)
    folder_nameparts = view_foldername.split('_')
    pickle_metadata['task'] = folder_nameparts[2]
    pickle_metadata['session_num'] = int(folder_nameparts[3][-2:])

    #todo: write the scorername into the pickle metadata dictionary. It's also in the metadata pickle file
    pickle_metadata['scorername'] = '_'.join(('DLC',
                                              metadata_list[next_metadata_idx + 5],
                                              metadata_list[next_metadata_idx + 6],
                                              metadata_list[next_metadata_idx + 7]))

    return pickle_metadata


def scorername_from_fname(fname):

    fpath, fname = os.path.split(fname)
    bare_fname, ext = os.path.splitext(fname)

    # scorername should always start with "DLC", then there are 4 underscores separating the network type, dlc project
    # name and the shuffle
    fname_parts = bare_fname.split('DLC')
    scorername_base = 'DLC' + fname_parts[1]

    scorername_parts = scorername_base.split('_')
    scorername = '_'.join(scorername_parts[:4])

    return scorername


def find_other_optitrack_pickles(pickle_file, parent_directories):

    pickle_metadata = parse_dlc_output_pickle_name_optitrack(pickle_file)
    orig_pickle_folder, orig_pickle_name = os.path.split(pickle_file)

    session_folder, cam_folders = navigation_utilities.find_cropped_session_folder(pickle_metadata, parent_directories)
    # session_folder, _ = os.path.split(orig_pickle_folder)

    _, session_foldername = os.path.split(session_folder)
    test_name = session_foldername + '_*_full.pickle'
    cam_pickles = [glob.glob(os.path.join(cf, test_name)) for cf in cam_folders]

    orig_cam = pickle_metadata['cam_num']
    orig_cam_pickle_stem = orig_pickle_name[:orig_pickle_name.find('cam{:02d}'.format(orig_cam)) + 5]

    cur_cam = 1

    camera_exists = True
    cam_pickle_files = []
    both_files_exist = True
    while cur_cam <= len(cam_pickles) and both_files_exist:
        if orig_cam == cur_cam:
            cam_pickle_files.append(pickle_file)
            cur_cam += 1
            continue

        cam_idx = cur_cam - 1
        curcam_pickle_stem = orig_cam_pickle_stem.replace('cam{:02d}'.format(orig_cam), 'cam{:02d}'.format(cur_cam))

        curcam_pickle = [cc_pickle for cc_pickle in cam_pickles[cam_idx] if curcam_pickle_stem in cc_pickle]

        if len(curcam_pickle) == 1:
            curcam_pickle = curcam_pickle[0]
            cam_pickle_files.append(curcam_pickle)
            cur_cam += 1
            both_files_exist = True
        else:
            print('no matching camera {:d} pickle file for {}'.format(cur_cam, orig_pickle_name))
            both_files_exist = False

    return cam_pickle_files, both_files_exist


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
        'prefix': '',
        'mouseID': '',
        'trialtime': datetime(1,1,1),
        'session_num': 0,
        'vid_num': 0,
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

    if metadata_list[0][:4] == 'stim':
        pickle_metadata['prefix'] = 'stim'
        pickle_metadata['mouseID'] = metadata_list[0][4:]
    elif metadata_list[0][:4] == 'post':
        pickle_metadata['prefix'] = 'poststim'
        pickle_metadata['mouseID'] = metadata_list[0][8:]
    else:
        pickle_metadata['mouseID'] = metadata_list[0]
    # num_string = ''.join(filter(lambda i: i.isdigit(), pickle_metadata['ratID']))
    # pickle_metadata['rat_num'] = int(num_string)

    # if box number is stored in file name, then extract it
    if 'box' in metadata_list[1]:
        pickle_metadata['boxnum'] = int(metadata_list[1][-1:])
        next_metadata_idx = 2
    else:
        next_metadata_idx = 1

    datetime_str = metadata_list[next_metadata_idx] + '_' + metadata_list[1+next_metadata_idx]
    pickle_metadata['trialtime'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    pickle_metadata['session_num'] = int(metadata_list[next_metadata_idx + 2])
    pickle_metadata['vid_num'] = int(metadata_list[next_metadata_idx + 3])
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


# def find_calibration_vid_folders(calibration_parent):
#     '''
#     find all calibration videos. assume directory structure:
#         calibration_parent-->calibration_videos__YYYY-->calibration_videos__YYYYMM-->calibration_videos__YYYYMM_boxZZ where
#         ZZ is the 2-digit box number
#     :param calibration_parent:
#     :return:
#     '''
#     year_folders = glob.glob(os.path.join(calibration_parent, 'calibration_videos_*'))
#     month_folders = []
#     # for yf in year_folders:
#     #     month_folders.extend(glob.glob(os.path.join(yf, 'calibration_videos_*')))
#     [month_folders.extend(glob.glob(os.path.join(yf, 'calibration_videos_*'))) for yf in year_folders]
#
#     box_folders = []
#     [box_folders.extend(glob.glob(os.path.join(mf, 'calibration_videos_*'))) for mf in month_folders]
#
#     return box_folders


def find_calibration_vid_folders(calibration_vids_parent):
    '''
    find all calibration videos. assume directory structure:
        calibration_parent-->calibration_videos__YYYY-->calibration_videos__YYYYMM-->calibration_videos__YYYYMM_boxZZ where
        ZZ is the 2-digit box number
    :param calibration_vids_parent:
    :return:
    '''
    month_folders = glob.glob(os.path.join(calibration_vids_parent, 'calibration_videos_*'))

    # for yf in year_folders:
    #     month_folders.extend(glob.glob(os.path.join(yf, 'calibration_videos_*')))

    month_folders = [mf for mf in month_folders if os.path.isdir(mf)]

    return month_folders


def create_cam_cal_toml_name(cam_cal_vid_name, parent_directories):

    cal_metadata = parse_camera_calibration_video_name(cam_cal_vid_name)

    _, vid_name = os.path.split(cam_cal_vid_name)
    vid_name, _ = os.path.splitext(vid_name)
    toml_name = vid_name + '.toml'

    month_folder = 'calibration_files_{}'.format(cal_metadata['time'].strftime('%Y%m'))
    month_folder = os.path.join(parent_directories['calibration_files_parent'], month_folder)
    single_cam_folder = 'single_camera_calibration_{}'.format(cal_metadata['time'].strftime('%Y%m'))
    single_cam_folder = os.path.join(month_folder, single_cam_folder)

    full_toml_path = os.path.join(single_cam_folder, toml_name)

    return full_toml_path


def create_cam_cal_pickle_name(cam_cal_vid_name, parent_directories):

    cal_metadata = parse_camera_calibration_video_name(cam_cal_vid_name)

    _, vid_name = os.path.split(cam_cal_vid_name)
    vid_name, _ = os.path.splitext(vid_name)
    pickle_name = vid_name + '.pickle'

    month_folder = 'calibration_files_{}'.format(cal_metadata['time'].strftime('%Y%m'))
    month_folder = os.path.join(parent_directories['calibration_files_parent'], month_folder)
    single_cam_folder = 'single_camera_calibration_{}'.format(cal_metadata['time'].strftime('%Y%m'))
    single_cam_folder = os.path.join(month_folder, single_cam_folder)

    full_pickle_path = os.path.join(single_cam_folder, pickle_name)

    return full_pickle_path


def find_camera_calibration_video(cam_cal_vid_name, parent_directories):

    cal_metadata = parse_camera_calibration_video_name(cam_cal_vid_name)
    month_folder = 'calibration_videos_{}'.format(cal_metadata['time'].strftime('%Y%m'))
    month_folder = os.path.join(parent_directories['calibration_vids_parent'], month_folder)
    single_cam_folder = 'single_camera_calibration_{}'.format(cal_metadata['time'].strftime('%Y%m'))
    single_cam_folder = os.path.join(month_folder, single_cam_folder)

    full_cam_cal_vid_name = os.path.join(single_cam_folder, cam_cal_vid_name)

    if not os.path.exists(full_cam_cal_vid_name):
        return None

    return full_cam_cal_vid_name


def cal_frames_folder_from_cal_vids_name(cal_vid_name):

    cal_vid_path, cal_vid_name = os.path.split(cal_vid_name)
    cal_metadata = parse_camera_calibration_video_name(cal_vid_name)
    cal_vid_name, _ = os.path.splitext(cal_vid_name)

    if cal_vid_path[-7:] == 'cropped':
        month_cal_directory, _ = os.path.split(cal_vid_path)
    else:
        month_cal_directory = cal_vid_path

    cal_vid_directory, _ = os.path.split(month_cal_directory)
    parent_directory, _ = os.path.split(cal_vid_directory)

    frames_root = os.path.join(parent_directory, 'calibration_frames')
    month_frames_dir = os.path.join(frames_root, 'calibration_frames_{}'.format(cal_metadata['time'].strftime('%Y%m')))

    name_parts = cal_vid_name.split('_')
    cal_frames_foldername = '_'.join(name_parts[:5])
    cal_frames_folder = os.path.join(month_frames_dir, cal_frames_foldername)

    if not os.path.exists(cal_frames_folder):
        os.makedirs(cal_frames_folder)

    return cal_frames_folder


def find_mirror_calibration_video(mirror_cal_vid_name, parent_directories):

    cal_metadata = parse_camera_calibration_video_name(mirror_cal_vid_name)
    if cal_metadata is None:
        return None
    month_folder = 'calibration_videos_{}'.format(cal_metadata['time'].strftime('%Y%m'))
    month_folder = os.path.join(parent_directories['calibration_vids_parent'], month_folder)

    full_mirror_cal_vid_name = os.path.join(month_folder, mirror_cal_vid_name)

    if not os.path.exists(full_mirror_cal_vid_name):
        return None

    return full_mirror_cal_vid_name


def calib_vid_name_from_cropped_calib_vid_name(cropped_calib_vid_name):

    _, cropped_calib_vid_name = os.path.split(cropped_calib_vid_name)
    cropped_calib_vid_name, ext = os.path.splitext(cropped_calib_vid_name)
    name_parts = cropped_calib_vid_name.split('_')

    base_name = '_'.join(name_parts[:4])

    calib_vid_name = base_name + ext

    return calib_vid_name


def create_cropped_calib_vid_name(full_calib_vid_name, crop_view, crop_params_dict, fliplr):
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

    cp_strings = [str(int(cp)) for cp in crop_params_dict[crop_view]]
    cp_joined = '-'.join(cp_strings)

    cropped_vid_name = vid_name + '_' + crop_view + '_' + cp_joined
    if fliplr:
        cropped_vid_name = cropped_vid_name + '_fliplr'
    cropped_vid_name = cropped_vid_name + ext

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
        view_list = ('dir', 'lm', 'rm')

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
    :param view: string containing 'dir', 'lm', or 'rm'
    :return:
    """
    if video_metadata['boxnum'] == 99:
        pickle_name_full = '_'.join((video_metadata['ratID'],
                                     fname_time2string(video_metadata['triggertime']),
                                     '{:03d}'.format(video_metadata['video_number']),
                                     view,
                                     '*',
                                     'full.pickle'
                                     ))

        pickle_name_meta = '_'.join((video_metadata['ratID'],
                                     fname_time2string(video_metadata['triggertime']),
                                     '{:03d}'.format(video_metadata['video_number']),
                                     view,
                                     '*',
                                     'meta.pickle'
                                     ))
    else:
        pickle_name_full = '_'.join((video_metadata['ratID'],
                                     'box{:02d}'.format(video_metadata['boxnum']),
                                     fname_time2string(video_metadata['triggertime']),
                                     '{:03d}'.format(video_metadata['video_number']),
                                     view,
                                     '*',
                                     'full.pickle'
                                     ))

        pickle_name_meta = '_'.join((video_metadata['ratID'],
                                     'box{:02d}'.format(video_metadata['boxnum']),
                                     fname_time2string(video_metadata['triggertime']),
                                     '{:03d}'.format(video_metadata['video_number']),
                                     view,
                                     '*',
                                     'meta.pickle'
                                     ))

    return pickle_name_full, pickle_name_meta


def find_calibration_video(video_metadata, calibration_parent, vidtype='.avi'):
    """

    :param video_metadata:
    :param calibration_parent:
    :return:
    """
    # date_string = video_metadata['triggertime'].strftime('%Y%m%d')
    # year_folder = os.path.join(calibration_parent, date_string[0:4])
    # month_folder = os.path.join(year_folder, date_string[0:6] + '_calibration')
    # calibration_folder = os.path.join(month_folder, date_string[0:6] + '_calibration_videos')

    if vidtype[0] != '.':
        vidtype = '.' + vidtype

    time_string = fname_time2string(video_metadata['time'])
    month_folder = os.path.join(calibration_parent, 'calibration_videos_{}'.format(time_string[0:6]))

    test_name = 'GridCalibration_b{:02d}_{}{}'.format(video_metadata['boxnum'], time_string, vidtype)
    test_name = os.path.join(month_folder, test_name)

    if os.path.exists(test_name):
        return test_name
    else:
        return None
        # sys.exit('No calibration file found for ' + video_metadata['video_name'])


def find_calibration_videos_optitrack(cal_metadata, calibration_parent, vid_type='.avi'):
    """

    :param video_metadata:
    :param calibration_parent:
    :return:
    """
    cal_vid_folder = find_Burgess_calibration_vid_folder(calibration_parent, cal_metadata['datetime'])

    datetime_string = datetime_to_string_for_fname(cal_metadata['datetime'])

    test_name = 'calibrationvid_{}_cam*{}'.format(datetime_string, vid_type)
    test_name = os.path.join(cal_vid_folder, test_name)

    cal_vid_list = glob.glob(test_name)

    return cal_vid_list

    # if os.path.exists(test_name):
    #     return cal_vid_list
    # else:
    #     return None
    #     # sys.exit('No calibration file found for ' + video_metadata['video_name'])


def create_trajectory_filename(video_metadata):

    trajectory_name = '_'.join((
        video_metadata['ratID'],
        'box{:02d}'.format(video_metadata['boxnum']),
        fname_time2string(video_metadata['triggertime']),
        '{:03d}'.format(video_metadata['video_number']),
        '3dtrajectory.pickle'
    ))

    return trajectory_name


# def find_camera_calibration_video(video_metadata, calibration_parent):
#     """
#
#     :param video_metadata:
#     :param calibration_parent:
#     :return:
#     """
#     date_string = video_metadata['triggertime'].strftime('%Y%m%d')
#     year_folder = os.path.join(calibration_parent, date_string[0:4])
#     month_folder = os.path.join(year_folder, date_string[0:6] + '_calibration')
#     calibration_video_folder = os.path.join(month_folder, 'camera_calibration_videos_' + date_string[0:6])
#
#     test_name = 'CameraCalibration_b{:02d}_{}_*.mat'.format(video_metadata['boxnum'], date_string)
#     test_name = os.path.join(calibration_video_folder, test_name)
#
#     calibration_video_list = glob.glob(test_name)
#
#     if len(calibration_video_list) == 0:
#         sys.exit('No camera calibration video found for ' + video_metadata['video_name'])
#
#     if len(calibration_video_list) == 1:
#         return calibration_video_list[0]
#
#     # more than one potential video was found
#     # find the last relevant calibration video collected before the current reaching video
#     vid_times = []
#     for cal_vid in calibration_video_list:
#         cam_cal_md = parse_camera_calibration_video_name(cal_vid)
#         vid_times.append(cam_cal_md['time'])
#
#     last_time_prior_to_video = max(d for d in vid_times if d < video_metadata['triggertime'])
#
#     calibration_video_name = calibration_video_list[vid_times.index(last_time_prior_to_video)]
#
#     return calibration_video_name


def parse_camera_calibration_video_name(calibration_video_name):
    """

    :param calibration_video_name: form of GridCalibration_bXX_YYYYMMDD_HH-mm-ss.avi
    :return:
    """
    camera_calibration_metadata = {
        'boxnum': 99,
        'time': datetime(1, 1, 1)
    }
    try:
        _, cal_vid_name = os.path.split(calibration_video_name)
    except:
        return None
    cal_vid_name, _ = os.path.splitext(cal_vid_name)

    cal_vid_name_parts = cal_vid_name.split('_')

    camera_calibration_metadata['boxnum'] = int(cal_vid_name_parts[1][-1:])

    datetime_str = cal_vid_name_parts[2] + '_' + cal_vid_name_parts[3]
    camera_calibration_metadata['time'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    return camera_calibration_metadata


def create_calibration_filename(calibration_metadata):

    # calibration_folder = create_calibration_file_folder_name(calibration_metadata, calibration_parent)
    # if not os.path.isdir(calibration_folder):
    #     os.makedirs(calibration_folder)

    datetime_string = fname_time2string(calibration_metadata['time'])

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
                                                           fname_time2string(video_metadata['triggertime']),
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
                mirrorview = 'lm'
            else:
                mirrorview = 'rm'
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
                            vid_prefix = pickle_name[:pickle_name.find('dir')]
                            test_mirror_name = vid_prefix + '*_full.pickle'
                            full_mirror_name_list = glob.glob(os.path.join(mirror_marked_folder, test_mirror_name))
                            if len(full_mirror_name_list) == 1:
                                full_mirror_file = full_mirror_name_list[0]
                                _, full_mirror_name = os.path.split(full_mirror_file)
                                meta_mirror_file = os.path.join(mirror_marked_folder, full_mirror_name.replace('full', 'meta'))
                                if os.path.exists(meta_direct_file) and os.path.exists(meta_mirror_file):

                                    video_name = '{}_box{:02d}_{}_{:03d}.avi'.format(ratID,
                                                                                     pickle_metadata['boxnum'],
                                                                                     fname_time2string(['triggertime']),
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


def processed_data_pickle_name(session_metadata, parent_directories):

    session_folder = find_session_folder(parent_directories, session_metadata)

    formatstring = date_formatstring()
    datestring = fname_datestring_from_datetime(session_metadata['date'], formatstring=formatstring)
    fname = '_'.join([session_metadata['ratID'],
                      datestring,
                      session_metadata['task'],
                      'ses{:02d}_processed.pickle'.format(session_metadata['session_num'])])

    full_path_fname = os.path.join(session_folder, fname)

    return full_path_fname


def date_formatstring():

    formatstring = '%Y%m%d'

    return formatstring


def fname_datestring_from_datetime(dtime, formatstring='%Y%m%d'):

    datestring = datetime.strftime(dtime, formatstring)

    return datestring


def find_session_folder(parent_directories, session_metadata):

    ratID = session_metadata['ratID']
    rat_folder = os.path.join(parent_directories['videos_root_folder'], ratID)

    formatstring = date_formatstring()
    if 'date' in session_metadata.keys():
        date_string = fname_datestring_from_datetime(session_metadata['date'], formatstring=formatstring)
    elif 'session_date' in session_metadata.keys():
        date_string = fname_datestring_from_datetime(session_metadata['session_date'], formatstring=formatstring)

    date_foldername = ratID + '_' + date_string
    date_path = os.path.join(rat_folder, date_foldername)

    session_numstring = 'ses{:02d}'.format(session_metadata['session_num'])

    session_foldername = '_'.join([date_foldername,
                                   session_metadata['task'],
                                   session_numstring])

    full_sessionpath = os.path.join(date_path, session_foldername)

    return full_sessionpath


def find_Burgess_calibration_vid_folder(calibration_parent, session_datetime):

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
    calibration file name of form 'calibrationvid_YYYYmmdd_HH-MM-SS_camZZ.avi' or
        'calibrationvid_YYYYmmdd-HH-MM-SS_camZZ.avi' where ZZ is '01' or '02'
    :param cal_vid_name:
    :return:
    '''
    _, cal_vid_name = os.path.split(cal_vid_name)
    bare_name, _ = os.path.splitext(cal_vid_name)

    name_parts_list = bare_name.split('_')

    if len(name_parts_list[1]) > 8:
        session_datetime = fname_string_to_datetime(name_parts_list[1][:8] + '_' + name_parts_list[1][9:])
        cam_num = int(name_parts_list[2][3:])
    else:
        session_datetime = fname_string_to_datetime(name_parts_list[1] + '_' + name_parts_list[2])
        cam_num = int(name_parts_list[3][3:])

    calvid_metadata = {
        'cam_num': cam_num,
        'session_datetime': session_datetime,
        'calvid_name': cal_vid_name
    }

    return calvid_metadata


def create_calibration_data_name(cal_data_parent, session_datetime):

    basename = 'calibration_data'
    cal_data_name = basename + '_' + datetime_to_string_for_fname(session_datetime) + '.pickle'

    cal_data_folder = create_calibration_data_folder(cal_data_parent, session_datetime)
    cal_data_name = os.path.join(cal_data_folder, cal_data_name)

    return cal_data_name


def find_optitrack_calibration_data_name(cal_data_parent, session_datetime, max_days_to_look_back=5, basename='calibrationdata'):
    '''

    :param cal_data_parent: parent directory for folders containing pickle files with calibration results. Has structure:
        cal_data_parent-->calibration_data_YYYY-->calibration_data_YYYYmm
    :param session_datetime:
    :return:
    '''

    cal_data_folder = create_optitrack_calibration_data_folder(cal_data_parent, session_datetime)
    # search in cal_data_folder for calibration files from this date
    test_name = '_'.join((basename, date_to_string_for_fname(session_datetime) + '_*.pickle'))

    cal_data_files = glob.glob(os.path.join(cal_data_folder, test_name))
    cal_data_datetimes = []

    for cal_data_file in cal_data_files:
        _, cal_data_root = os.path.split(cal_data_file)
        cal_data_root, _ = os.path.splitext(cal_data_root)
        fparts = cal_data_root.split('_')
        cal_data_datestring = '_'.join(fparts[1:3])

        cal_data_datetimes.append(fname_string_to_datetime(cal_data_datestring))

    if not bool(cal_data_datetimes):
        # didn't find any calibration files for this date
        cur_datetime = session_datetime
        session_month = session_datetime.month
        cur_month = session_month

        while not bool(cal_data_datetimes) and session_datetime - cur_datetime <= timedelta(max_days_to_look_back):
            # look back max_days_to_look_back days before giving up
            cur_datetime = cur_datetime - timedelta(1)

            cal_data_folder = create_optitrack_calibration_data_folder(cal_data_parent, cur_datetime)
            test_name = '_'.join((basename, date_to_string_for_fname(cur_datetime) + '_*.pickle'))

            cal_data_files = glob.glob(os.path.join(cal_data_folder, test_name))
            cal_data_datetimes = []
            for cal_data_file in cal_data_files:
                _, cal_data_root = os.path.split(cal_data_file)
                cal_data_root, _ = os.path.splitext(cal_data_root)
                fparts = cal_data_root.split('_')
                cal_data_datestring = '_'.join(fparts[1:3])

                cal_data_datetimes.append(fname_string_to_datetime(cal_data_datestring))

    if not bool(cal_data_datetimes):
        # still haven't found any calibration data; try looking forward in time
        cur_datetime = session_datetime
        session_month = session_datetime.month
        cur_month = session_month
        while not bool(cal_data_datetimes) and cur_datetime - session_datetime <= timedelta(max_days_to_look_back):
            # look forward max_days_to_look_back days before giving up
            cur_datetime = cur_datetime - timedelta(1)

            cal_data_folder = create_optitrack_calibration_data_folder(cal_data_parent, cur_datetime)
            test_name = '_'.join((basename, date_to_string_for_fname(cur_datetime) + '_*.pickle'))

            cal_data_files = glob.glob(os.path.join(cal_data_folder, test_name))
            cal_data_datetimes = []
            for cal_data_file in cal_data_files:
                _, cal_data_root = os.path.split(cal_data_file)
                cal_data_root, _ = os.path.splitext(cal_data_root)
                fparts = cal_data_root.split('_')
                cal_data_datestring = '_'.join(fparts[1:3])

                cal_data_datetimes.append(fname_string_to_datetime(cal_data_datestring))

    if not bool(cal_data_datetimes):
        return None

    # find datetime of calibration data file closest to session_datetime
    nearest_datetime = min(cal_data_datetimes, key=lambda x: abs(x - session_datetime))

    closest_calibration_file = create_optitrack_calibration_data_name(cal_data_parent, nearest_datetime, basename=basename)

    return closest_calibration_file


def get_trajectory_folders(trajectories_parent):
    '''
    get full list of lowest level folders containing trajectory data. File tree structure is:
        trajectory_parent-->ratID-->session_name
    :param trajectories_parent:
    :return:
    '''
    test_string = os.path.join(trajectories_parent, 'R*')
    poss_rat_directories = glob.glob(test_string)
    # only accept directories of the format 'RXXXX'
    rat_directories = []
    for prd in poss_rat_directories:
        _, dir_name = os.path.split(prd)
        if len(dir_name) == 5 and dir_name[1:].isdigit():
            rat_directories.append(prd)

    traj_directories = []
    for rd in rat_directories:
        _, ratID = os.path.split(rd)
        test_string = os.path.join(rd, ratID + '_*')
        poss_session_dirs = glob.glob(test_string)

        for psd in poss_session_dirs:
            _, dir_name = os.path.split(psd)
            if len(dir_name) == 15:   # name format should be RXXXX_YYYYmmddz, where z = 'a', 'b', etc.
                traj_directories.append(psd)

    return traj_directories


def get_optitrack_r3d_folders(reconstruct3d_parent):
    '''
    get full list of lowest level folders containing 3d reconstruction data. File tree structure is:
        reconstruct3d_parent-->ratID-->session_name
    :param reconstruct3d_parent:
    :return:
    '''
    test_string = os.path.join(reconstruct3d_parent, '*')
    poss_mouse_directories = glob.glob(test_string)
    # make sure don't count non-directories
    mouse_directories = [pmd for pmd in poss_mouse_directories if os.path.isdir(pmd)]

    r3d_directories = []
    for md in mouse_directories:
        _, mouseID = os.path.split(md)
        test_string = os.path.join(md, mouseID + '_*')
        # poss_month_directories = glob.glob(test_string)
        # poss_month_directories = [pmd for pmd in poss_month_directories if os.path.isdir(pmd)]
        #
        # for month_dir in poss_month_directories:
        test_string = os.path.join(md, mouseID + '_*')
        poss_day_directories = glob.glob(test_string)
        day_directories = [pdd for pdd in poss_day_directories if os.path.isdir(pdd)]

        if bool(day_directories):
            r3d_directories.extend(day_directories)

    return r3d_directories


def create_optitrack_calibration_data_name(cal_data_parent, session_datetime, basename='calibrationdata'):
    '''

    :param cal_data_parent: parent directory for folders containing pickle files with calibration results. Has structure:
        cal_data_parent-->calibration_data_YYYY-->calibration_data_YYYYmm
    :param session_datetime:
    :return:
    '''

    cal_data_name = '_'.join((basename, datetime_to_string_for_fname(session_datetime) + '.pickle'))

    cal_data_folder = create_optitrack_calibration_data_folder(cal_data_parent, session_datetime)
    cal_data_name = os.path.join(cal_data_folder, cal_data_name)

    return cal_data_name


def parse_optitrack_calibration_data_name(optitrack_cal_data_path):
    '''

    :param optitrack_cal_data_path: parent directory for folders containing pickle files with calibration results. Has structure:
        cal_data_parent-->calibration_data_YYYY-->calibration_data_YYYYmm
    :param session_datetime:
    :return:
    '''

    _, optitrack_cal_data_name = os.path.split(optitrack_cal_data_path)
    optitrack_cal_data_name, _ = os.path.splitext(optitrack_cal_data_name)

    name_parts = optitrack_cal_data_name.split('_')

    cal_datetime_string = name_parts[1] + '_' + name_parts[2]
    cal_datetime = fname_string_to_datetime(cal_datetime_string)

    cal_metadata = {'datetime': cal_datetime}

    return cal_metadata


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
    mouseID = metadata['mouseID']

    month_string = trialtime.strftime('%Y%m')
    date_string = date_to_string_for_fname(trialtime)
    # month_folder_name = mouseID + '_' + month_string
    date_folder_name = mouseID + '_' + date_string
    full_path = os.path.join(reconstruction3d_parent, mouseID, date_folder_name)

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
        '{:03d}'.format(metadata['vid_num']),
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

    if r3d_name_parts[0][:4] == 'stim':
        mouseID = r3d_name_parts[0][4:]
    elif r3d_name_parts[0][:4] == 'post':
        mouseID = r3d_name_parts[0][8:]   # poststim
    else:
        mouseID = r3d_name_parts[0]
    trial_datetimestring = r3d_name_parts[1] + '_' + r3d_name_parts[2]
    trial_datetime = fname_string_to_datetime(trial_datetimestring)

    session_num = int(r3d_name_parts[3])

    vid_num = int(r3d_name_parts[4])

    r3d_metadata = {
        'mouseID': mouseID,
        'trialtime': trial_datetime,
        'vid_num': vid_num,
        'session_num': session_num,
    }

    return r3d_metadata


def find_folders_to_reconstruct(cropped_videos_parent, cam_names):
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
            test_direct_folder = os.path.join(session_folder, session_folder_name + '_dir')
            if not os.path.exists(test_direct_folder):
                continue
            # check to see if there are .h5 files containing labeled data
            session_metadata = parse_croppedvid_dir_name(session_folder_name)

            h5_names = []
            for cam_name in cam_names:
                test_h5_name = test_dlc_h5_name_from_session_metadata(session_metadata, cam_name, filtered=False)
                cam_folder_name = '_'.join((session_folder_name, cam_name))
                full_test_h5_name = os.path.join(session_folder, cam_folder_name, test_h5_name)
                h5_names.append(glob.glob(full_test_h5_name))

            # test_pickle_name = '_'.join((ratID,
            #                              'box01',
            #                              session_name, '*',
            #                              'full.pickle'))
            # full_test_pickle_name = os.path.join(test_direct_folder, test_pickle_name)
            # full_pickle_list = glob.glob(full_test_pickle_name)

            if not all(h5_names):
                # if there aren't matching .h5 files for different views, no point in trying to triangulate
                continue

            h5_metadata = parse_dlc_output_pickle_name(h5_names[0][0])
            session_metadata['boxnum'] = h5_metadata['boxnum']
            folders_to_reconstruct.append(session_metadata)

    return folders_to_reconstruct


def find_dlc_pickles_from_r3d_filename(r3d_file, parent_directories):

    cropped_vids_parent = parent_directories['cropped_vids_parent']

    r3d_metadata = parse_3d_reconstruction_pickle_name(r3d_file)

    mouse_folder = os.path.join(cropped_vids_parent, r3d_metadata['mouseID'])
    try:
        month_dirname = r3d_metadata['mouseID'] + '_' + r3d_metadata['time'].strftime('%Y%m')
        day_dirname = r3d_metadata['mouseID'] + '_' + r3d_metadata['time'].strftime('%Y%m%d')
    except:
        month_dirname = r3d_metadata['mouseID'] + '_' + r3d_metadata['trialtime'].strftime('%Y%m')
        day_dirname = r3d_metadata['mouseID'] + '_' + r3d_metadata['trialtime'].strftime('%Y%m%d')


    full_pickles = []
    meta_pickles = []
    for i_cam in range(2):
        if 'time' in r3d_metadata.keys():
            datestring = date_to_string_for_fname(r3d_metadata['time'])
        elif 'trialtime' in r3d_metadata.keys():
            datestring = date_to_string_for_fname(r3d_metadata['trialtime'])
        else:
            datestring = None
        cam_dirname = '_'.join([r3d_metadata['mouseID'],
                                datestring,
                                'cam{:02d}'.format(i_cam + 1)
                                ])

        cam_dir = os.path.join(mouse_folder, day_dirname, cam_dirname)
        test_name = '_'.join([r3d_metadata['mouseID'],
                              datestring,
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
                              datestring,
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
        [sessionID_direct/lm/rm]
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
        # month_folder = mouseID + '_' + session_month

        # create folders for cropped video for each camera
        for i_cam in range(num_cams):
            cam_name = 'cam{:02d}'.format(cam_list[i_cam])
            cropped_vid_dir = session_dir + '_cam{:02d}'.format(cam_list[i_cam])
            cropped_video_directories[cam_name].append(os.path.join(cropped_vids_parent, mouseID, session_dir, cropped_vid_dir))

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

    if len(vid_name_parts) == 6:
        # name includes session # (numbered from the first session overall or first session of the day?)
        vid_metadata['session_num'] = int(vid_name_parts[3])
        vid_metadata['cam_num'] = int(vid_name_parts[5][3:])
        vid_metadata['vid_num'] = int(vid_name_parts[4])
    else:
        vid_metadata['cam_num'] = int(vid_name_parts[3][3:])

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
    '''
    short function to make sure date-times are formatted in the same way for all filenames
    :param string_to_convert:
    :return:
    '''

    if len(string_to_convert) == 15:
        # most likely, the first 2 digits of the year were left off (should be seventeen characters - 8 for yyyymmdd,
        # one for the underscore, and 8 for HH-MM-SS
        string_to_convert = '20' + string_to_convert
    format_string = '%Y%m%d_%H-%M-%S'

    datetime_from_fname = datetime.strptime(string_to_convert, format_string)

    return datetime_from_fname


def fname_string_to_date(string_to_convert):

    if len(string_to_convert) == 8:
        format_string = '%Y%m%d'
    elif len(string_to_convert) == 6:
        format_string = '%y%m%d'
    else:
        format_string = None

    if format_string is None:
        date_from_fname = None
    else:
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

    cropped_cal_vid_metadata['boxnum'] = int(cal_vid_name_parts[1][-1:])

    datetime_str = cal_vid_name_parts[2] + '_' + cal_vid_name_parts[3]
    cropped_cal_vid_metadata['time'] = datetime.strptime(datetime_str, '%Y%m%d_%H-%M-%S')

    cropped_cal_vid_metadata['view'] = cal_vid_name_parts[4]

    crop_params_strings = cal_vid_name_parts[5].split('-')
    crop_params = [int(cp) for cp in crop_params_strings]
    cropped_cal_vid_metadata['crop_params'] = crop_params

    return cropped_cal_vid_metadata


def create_calibration_file_path(calibration_files_parent, calib_metadata):

    # year_str = calib_metadata['time'].strftime('%Y')
    month_str = calib_metadata['time'].strftime('%Y%m')
    # year_folder = os.path.join(calibration_files_parent, 'calibration_files_' + year_str)
    month_folder = os.path.join(calibration_files_parent, 'calibration_files_' + month_str)
    # box_folder = os.path.join(month_folder, 'calibration_files_' + month_str + '_box{:02d}'.format(calib_metadata['boxnum']))

    return month_folder


def create_calibration_summary_name(full_calib_vid_name, calibration_files_parent):
    '''

    :param full_calib_vid_name:
    :return:
    '''

    # store the pickle file with the calibration parameters in the same folder as the calibration video
    try:
        calib_vid_path, _ = os.path.split(full_calib_vid_name)
    except:
        pass
    calib_metadata = parse_camera_calibration_video_name(full_calib_vid_name)

    calib_file_path = create_calibration_file_path(calibration_files_parent, calib_metadata)

    if not os.path.isdir(calib_file_path):
        os.makedirs(calib_file_path)

    calib_fname = '_'.join(('calibrationdata',
                            fname_time2string(calib_metadata['time']),
                            'b{:02d}'.format(calib_metadata['boxnum']) + '.pickle'
    ))

    full_calib_fname = os.path.join(calib_file_path, calib_fname)

    return full_calib_fname


def create_calibration_toml_name(full_calib_vid_name, calibration_files_parent):
    '''

    :param full_calib_vid_name:
    :return:
    '''

    # store the toml file with the calibration parameters in the same folder as the calibration video
    calib_vid_path, _ = os.path.split(full_calib_vid_name)
    calib_metadata = parse_camera_calibration_video_name(full_calib_vid_name)

    calib_file_path = create_calibration_file_path(calibration_files_parent, calib_metadata)

    if not os.path.isdir(calib_file_path):
        os.makedirs(calib_file_path)

    calib_fname = '_'.join(('calibrationdata',
                            fname_time2string(calib_metadata['time']),
                            'box{:02d}'.format(calib_metadata['boxnum']) + '.toml'
    ))

    full_calib_fname = os.path.join(calib_file_path, calib_fname)

    return full_calib_fname


def fname_time2string(ftime):

    fname_timestring = ftime.strftime('%Y%m%d_%H-%M-%S')

    return fname_timestring


def fname_date2string(ftime):

    fname_datestring = ftime.strftime('%Y%m%d')

    return fname_datestring


def mouse_animation_name(vid_metadata, reconstruct_3d_parent):

    if 'poststim' in vid_metadata['mouseID']:
        mouseID = vid_metadata['mouseID'][8:]
    else:
        mouseID = vid_metadata['mouseID']

    fname = '_'.join((mouseID,
                      fname_time2string(vid_metadata['triggertime']),
                      str(vid_metadata['session_num']),
                      '{:03d}'.format(vid_metadata['video_number']),
                      'animation.mp4'))

    mouse_folder = os.path.join(reconstruct_3d_parent, mouseID)
    # month_folder = '_'.join((mouseID,
    #                          vid_metadata['triggertime'].strftime('%Y%m')))
    # month_folder = os.path.join(mouse_folder, month_folder)
    date_folder = '_'.join((mouseID,
                            fname_date2string(vid_metadata['triggertime'])))
    date_folder = os.path.join(mouse_folder, date_folder)

    fullpath = os.path.join(date_folder, fname)

    return fullpath


def import_scoring_xlsx(scoring_xls_name):

    valid_sheet = True
    day_num = 0

    while valid_sheet:
        day_num += 1
        test_name = 'day{:02d}'.format(day_num)
        if day_num == 1:
            try:
                scores = pd.read_excel(scoring_xls_name, sheet_name=test_name)
            except:
                valid_sheet = False
        else:
            try:
                new_scores = pd.read_excel(scoring_xls_name, sheet_name=test_name)
                scores = scores.append(new_scores)
            except:
                valid_sheet = False

    return scores


def find_manual_scoring_sheet(parent_directories, mouseID):

    test_string = '_'.join([mouseID,
                            'scores.xlsx'])

    test_path = os.path.join(parent_directories['manual_scoring_parent'], test_string)

    scoring_file = glob.glob(test_path)

    if len(scoring_file) == 1:
        scoring_file = scoring_file[0]
    elif len(scoring_file) > 1:
        print('more than one scoring file found for {}'.format(mouseID))
        scoring_file = None
    elif len(scoring_file) == 0:
        print('no scoring file found for {}'.format(mouseID))
        scoring_file = None

    return scoring_file

def get_pickled_metadata_fname(session_metadata, parent_directories):
    session_folder = find_session_folder(parent_directories['data'], session_metadata)

    formatstring = date_formatstring()
    datestring = fname_datestring_from_datetime(session_metadata['date'], formatstring=formatstring)
    fname = '_'.join([session_metadata['ratID'],
                      datestring,
                      session_metadata['task'],
                      'ses{:02d}_metadata.pickle'.format(session_metadata['session_num'])])

    full_path_fname = os.path.join(session_folder, fname)

    return full_path_fname


def get_pickled_analog_processeddata_fname(session_metadata, parent_directories):
    session_folder = find_session_folder(parent_directories['data'], session_metadata)

    formatstring = date_formatstring()
    datestring = fname_datestring_from_datetime(session_metadata['date'], formatstring=formatstring)
    fname = '_'.join([session_metadata['ratID'],
                      datestring,
                      session_metadata['task'],
                      'ses{:02d}_analogprocessed.pickle'.format(session_metadata['session_num'])])

    full_path_fname = os.path.join(session_folder, fname)

    return full_path_fname


def get_pickled_ts_fname(session_metadata, parent_directories):
    session_folder = find_session_folder(parent_directories['data'], session_metadata)

    formatstring = date_formatstring()
    datestring = fname_datestring_from_datetime(session_metadata['date'], formatstring=formatstring)
    fname = '_'.join([session_metadata['ratID'],
                      datestring,
                      session_metadata['task'],
                      'ses{:02d}_timestamps.pickle'.format(session_metadata['session_num'])])

    full_path_fname = os.path.join(session_folder, fname)

    return full_path_fname

def processed_data_pickle_name(session_metadata, parent_directories):

    session_folder = find_session_folder(parent_directories, session_metadata)

    formatstring = date_formatstring()
    datestring = fname_datestring_from_datetime(session_metadata['date'], formatstring=formatstring)
    fname = '_'.join([session_metadata['ratID'],
                      datestring,
                      session_metadata['task'],
                      'ses{:02d}_processed.pickle'.format(session_metadata['session_num'])])

    full_path_fname = os.path.join(session_folder, fname)

    return full_path_fname


def find_baseline_recording(parent_directories, session_metadata):

    session_folder = find_session_folder(parent_directories['data'], session_metadata)

    session_name = session_name_from_metadata(session_metadata)
    test_string = '_'.join([session_name,
                            'baseline',
                            '*.mat'])

    test_path = os.path.join(session_folder, test_string)

    baseline_recordings = glob.glob(test_path)

    baseline_recordings = [br for br in baseline_recordings if '.mat' in br]

    if len(baseline_recordings) == 1:
        baseline_recording = baseline_recordings[0]
    elif len(baseline_recordings) > 1:
        print('more than one baseline file for {}'.format(session_name))
        baseline_recording = None
    else:
        print('no baseline recordings for {}'.format(session_name))
        baseline_recording = None

    return baseline_recording


def find_session_summary_folder(parent_directories, session_metadata):

    summary_parent = parent_directories['summaries']
    ratID = session_metadata['ratID']
    rat_folder = os.path.join(summary_parent, ratID)

    task_folder = os.path.join(rat_folder, session_metadata['task'])

    if not os.path.isdir(task_folder):
        os.makedirs(task_folder)

    # date_string = fname_datestring_from_datetime(session_metadata['date'])
    # date_foldername = ratID + '_' + date_string
    # date_path = os.path.join(rat_folder, date_foldername)
    #
    # session_numstring = 'session{:02d}'.format(session_metadata['session_num'])
    #
    # session_foldername = '_'.join([date_foldername,
    #                                session_metadata['task'],
    #                                session_numstring])
    #
    # full_sessionpath = os.path.join(date_path, session_foldername)

    return task_folder