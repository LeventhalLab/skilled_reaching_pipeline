import numpy as np
import cv2
import navigation_utilities
import skilled_reaching_io

def triangulate_video(video_name, marked_videos_parent, calibration_parent, view_list=('direct', 'leftmirror', 'rightmirror')):

    video_metadata = navigation_utilities.parse_video_name(video_name)
    dlc_output_pickle_names, dlc_metadata_pickle_names = navigation_utilities.find_dlc_output_pickles(video_metadata, marked_videos_parent, view_list=view_list)
    # above line will not complete if all pickle files with DLC output data are not found

    # find the calibration files
    calibration_file = navigation_utilities.find_calibration_file(video_metadata, calibration_parent)
    calibration_params = skilled_reaching_io.read_matlab_calibration(calibration_file)
    # above lines will not complete if a calibration file is not found

    # read in the pickle files
    dlc_output = {view: None for view in view_list}
    dlc_metadata = {view: None for view in view_list}
    pickle_name_metadata = {view: None for view in view_list}
    for view in view_list:
        dlc_output[view] = skilled_reaching_io.read_pickle(dlc_output_pickle_names[view])
        dlc_metadata[view] = skilled_reaching_io.read_pickle(dlc_metadata_pickle_names[view])
        pickle_name_metadata[view] = navigation_utilities.parse_dlc_output_pickle_name(dlc_output_pickle_names[view])

    trajectory_filename = navigation_utilities.create_trajectory_filename(video_metadata)

    trajectory_metadata = extract_trajectory_metadata(dlc_metadata, pickle_name_metadata)

    dlc_data = extract_data_from_dlc_output(dlc_output, trajectory_metadata)
    #todo: preprocessing to get rid of "invalid" points

    # translate and undistort points
    dlc_data = translate_points_to_full_frame(dlc_data, trajectory_metadata)
    pass


def extract_trajectory_metadata(dlc_metadata, name_metadata):

    view_list = dlc_metadata.keys()
    trajectory_metadata = {view: None for view in view_list}

    for view in view_list:
        trajectory_metadata[view] = {'bodyparts': dlc_metadata[view]['data']['DLC-model-config file']['all_joints_names'],
                                     'num_frames': dlc_metadata[view]['data']['nframes'],
                                     'crop_window': name_metadata[view]['crop_window']
                                     }
    # todo:check that number of frames and bodyparts are the same in each view

    return trajectory_metadata


def translate_points_to_full_frame(dlc_data, trajectory_metadata):

    view_list = dlc_metadata.keys()

    for view in view_list:
        for bp in trajectory_metadata[view]['bodyparts']:
            # translate point
            for i_frame in range(trajectory_metadata[view]['num_frames']):
                if not np.all([view][bp]['coordinates'][i_frame] == 0):
                    # a point was found in this frame (coordinate == 0 if no point found)
                    #todo: check that the x's and y's are matching up properly
                    dlc_data[view][bp]['coordinates'][i_frame] += trajectory_metadata[view]['crop_window'][0:2]

    pass


def extract_data_from_dlc_output(dlc_output, trajectory_metadata):

    view_list = dlc_output.keys()

    dlc_data = {view: None for view in view_list}
    for view in view_list:
        # initialize dictionaries for each bodypart
        num_frames = trajectory_metadata[view]['num_frames']
        dlc_data[view] = {bp: None for bp in trajectory_metadata[view]['bodyparts']}
        for i_bp, bp in enumerate(trajectory_metadata[view]['bodyparts']):

            dlc_data[view][bp] = {'coordinates': np.empty((num_frames, 2)),
                                  'confidence': np.empty((num_frames, 1)),
                                  }

            for i_frame in range(num_frames):
                frame_key = 'frame{:04d}'.format(i_frame)

                try:
                    dlc_data[view][bp]['coordinates'][i_frame, :] = dlc_output[view][frame_key]['coordinates'][0][i_bp][0]
                    dlc_data[view][bp]['confidence'][i_frame] = dlc_output[view][frame_key]['confidence'][i_bp][0][0]
                except:
                    # 'coordinates' array and 'confidence' array at this frame are empty - must be a peculiarity of deeplabcut
                    # just leave the dlc_data arrays as empty
                    pass

    return dlc_data