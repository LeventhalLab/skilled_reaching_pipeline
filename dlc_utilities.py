import numpy as np

def collect_bp_data(view_dlc_data, dlc_key):

    bodyparts = tuple(view_dlc_data.keys())

    bp_array_shape = np.shape(view_dlc_data[bodyparts[0]][dlc_key])
    bp_data = np.zeros((len(bodyparts), bp_array_shape[0], bp_array_shape[1]))
    for i_bp, bp in enumerate(bodyparts):
        bp_data[i_bp, :, :] = view_dlc_data[bp][dlc_key]

    return bp_data


def extract_data_from_dlc_output(dlc_output, trajectory_metadata):
    '''

    :param dlc_output: data from dlc _full.pickle files as imported by read_pickle
    :param trajectory_metadata:
    :return:
    '''

    #todo: figure out how to deal with dlc_output as a list of output from multiple views, dictionary of output from multiple views, or as a single view
    if type(dlc_output) is dict:
        dlc_output_keys = tuple(dlc_output.keys())
        # test if the second key contains 'frame' - the first key is now 'metadata' from dlc output
        if 'frame' in dlc_output_keys[1]:
            # dlc_output is the output from a single dlc labeling session, assuming trajectory_metadata also only from a single view
            if trajectory_metadata is None:
                dlc_data = None
                return dlc_data
            num_frames = trajectory_metadata['num_frames']
            dlc_data = {bp: None for bp in trajectory_metadata['bodyparts']}
            for i_bp, bp in enumerate(trajectory_metadata['bodyparts']):

                dlc_data[bp] = {'coordinates': np.zeros((num_frames, 2)),
                                'confidence': np.zeros((num_frames, 1)),
                                }

                for i_frame in range(num_frames):
                    frame_key = 'frame{:04d}'.format(i_frame)

                    try:
                        dlc_data[bp]['coordinates'][i_frame, :] = dlc_output[frame_key]['coordinates'][0][i_bp][0]
                        dlc_data[bp]['confidence'][i_frame] = dlc_output[frame_key]['confidence'][i_bp][0][0]
                    except:
                        try:
                            # if dlc_output was "cleaned" prior to passing into this function, will have squeezed arrays
                            dlc_data[bp]['coordinates'][i_frame, :] = dlc_output[frame_key]['coordinates'][i_bp]
                            dlc_data[bp]['confidence'][i_frame] = dlc_output[frame_key]['confidence'][i_bp]
                        except:
                            # 'coordinates' array and 'confidence' array at this frame are empty - must be a peculiarity of deeplabcut
                            # just leave the dlc_data arrays as empty
                            pass

        else:
            # dlc_output is a dictionary where each entry is dlc output ("full" pickled file) from a different view
            # whose names are the dictionary keys. Return a dlc_data dictionary containing dlc_data for each view
            view_list = dlc_output_keys

            dlc_data = {view: None for view in view_list}
            for view in view_list:
                # initialize dictionaries for each bodypart
                if trajectory_metadata[view] is None:
                    continue
                num_frames = trajectory_metadata[view]['num_frames']
                dlc_data[view] = {bp: None for bp in trajectory_metadata[view]['bodyparts']}
                for i_bp, bp in enumerate(trajectory_metadata[view]['bodyparts']):

                    dlc_data[view][bp] = {'coordinates': np.zeros((num_frames, 2)),
                                          'confidence': np.zeros((num_frames, 1)),
                                          }

                    for i_frame in range(num_frames):
                        frame_key = 'frame{:04d}'.format(i_frame)

                        try:
                            dlc_data[view][bp]['coordinates'][i_frame, :] = \
                            dlc_output[view][frame_key]['coordinates'][0][i_bp][0]
                            dlc_data[view][bp]['confidence'][i_frame] = \
                            dlc_output[view][frame_key]['confidence'][i_bp][0][0]
                        except:
                            # 'coordinates' array and 'confidence' array at this frame are empty - must be a peculiarity of deeplabcut
                            # just leave the dlc_data arrays as empty
                            pass

    elif type(dlc_output) is list:
        # multiple versions of dlc_output stored in a list instead of a dictionary; return dlc_data as a list
        # assume trajectory_metadata is also a list
        dlc_data = []
        for i_dlco, dlco in enumerate(dlc_output):
            # initialize dictionaries for each bodypart
            if trajectory_metadata[i_dlco] is None:
                continue
            num_frames = trajectory_metadata[i_dlco]['num_frames']
            dlc_data.append({bp: None for bp in trajectory_metadata[i_dlco]['bodyparts']})
            for i_bp, bp in enumerate(trajectory_metadata[i_dlco]['bodyparts']):

                dlc_data[i_dlco][bp] = {'coordinates': np.zeros((num_frames, 2)),
                                      'confidence': np.zeros((num_frames, 1)),
                                      }

                for i_frame in range(num_frames):
                    frame_key = 'frame{:04d}'.format(i_frame)

                    try:
                        dlc_data[i_dlco][bp]['coordinates'][i_frame, :] = \
                            dlc_output[i_dlco][frame_key]['coordinates'][0][i_bp][0]
                        dlc_data[i_dlco][bp]['confidence'][i_frame] = \
                            dlc_output[i_dlco][frame_key]['confidence'][i_bp][0][0]
                    except:
                        # 'coordinates' array and 'confidence' array at this frame are empty - must be a peculiarity of deeplabcut
                        # just leave the dlc_data arrays as empty
                        pass

    return dlc_data


def extract_trajectory_metadata(dlc_metadata, name_metadata):
    '''

    :param dlc_metadata:
    :param name_metadata:
    :return:
    '''

    dlc_metadata_keys = tuple(dlc_metadata.keys())
    if 'data' in dlc_metadata_keys:
        # there is only one dlc_metadata data set (as opposed to one for each camera/mirror view)
        trajectory_metadata = {
            'bodyparts': dlc_metadata['data']['DLC-model-config file']['all_joints_names'],
            'num_frames': dlc_metadata['data']['nframes'],
            'crop_window': name_metadata['crop_window']
            }
    else:
        view_list = dlc_metadata_keys
        trajectory_metadata = {view: None for view in view_list}

        for view in view_list:
            if name_metadata[view] is None:
                continue
            trajectory_metadata[view] = {'bodyparts': dlc_metadata[view]['data']['DLC-model-config file']['all_joints_names'],
                                         'num_frames': dlc_metadata[view]['data']['nframes'],
                                         'crop_window': name_metadata[view]['crop_window']
                                         }
            # todo:check that number of frames and bodyparts are the same in each view

    return trajectory_metadata