import skilled_reaching_io
import navigation_utilities
import numpy as np
import os


def analyze_trajectory(trajectory_fname):
    # trajectory_fname = navigation_utilities.create_trajectory_name(h5_metadata, session_metadata, calibration_data,
    #                                                                parent_directories)

    if os.path.exists(trajectory_fname):
        r3d_data = skilled_reaching_io.read_pickle(trajectory_fname)

    bodyparts = r3d_data['dlc_output']['bodyparts']
    paw_pref = r3d_data['rat_info']['pawpref'].values[0]

    f_contact, bp_contact = identify_pellet_contact(r3d_data, paw_pref, pelletname='pellet')
    pass


def identify_pellet_contact(r3d_data, paw_pref, score_threshold=0.95, pelletname='pellet', test_frame_range=(200, 250),
                            pellet_movement_tolerance=1.0, min_paw_pellet_dist=1.0):
    '''

    :param r3d_data:
    :param paw_pref:
    :param score_threshold:
    :param pelletname:
    :param test_frame_range:
    :param pellet_movement_tolerance:
    :param min_paw_pellet_dist:
    :return:
    '''
    # r3d_data = {'points3d': p3ds,
    #             'optim_points3d': optim_p3ds,
    #             'calibration_data': calibration_data,
    #             'h5_group': h5_group,
    #             'anipose_config': anipose_config,
    #             'dlc_output': d}
    bodyparts = r3d_data['dlc_output']['bodyparts']
    pts3d = r3d_data['optim_points3d']   # better with optim_points3d or points3d?

    reaching_pawparts = find_reaching_pawparts(bodyparts, paw_pref)

    initial_pellet_loc = find_initial_pellet_loc(dlc_output, pts3d, score_threshold=score_threshold,
                                         pelletname=pelletname, test_frame_range=test_frame_range)

    if initial_pellet_loc is None:
        # there wasn't a pellet for this video - at least, not one that was reliably identified
        return None

    # now find the bodypart that got closest to the pellet BEFORE the pellet moved, and the frame in which it got that close
    # if the pellet never moved, assume the rat didn't touch it
    did_pellet_move = test_if_pellet_moved(dlc_output, pts3d, initial_pellet_loc, pelletscore_threshold=0.95,
                                           pelletname=pelletname, pellet_movement_tolerance=pellet_movement_tolerance)

    if did_pellet_move:
        # now need to figure out which bodypart made it move, and which frame that was

    pass


def get_reaching_traj(pts3d, dlc_output, reaching_pawparts):
    # working here...
    pass


def find_initial_pellet_loc(dlc_output, pts3d, score_threshold=0.95, pelletname='pellet', test_frame_range=(200, 250)):

    # todo: should we have manual scores loaded in here, too?

    # there should be a pellet that's very visible in all 3 views for at least the early frames
    pellet_idx = dlc_output['bodyparts'].index(pelletname.lower())
    pellet_scores = dlc_output['scores'][:, :, pellet_idx].T

    n_frames = np.shape(pellet_scores)[0]

    valid_pellet_frames = (pellet_scores > score_threshold).all(axis=1)

    if all(valid_pellet_frames[test_frame_range[0] : test_frame_range[1]]):
        initial_pellet_loc = np.mean(pts3d[test_frame_range[0]:test_frame_range[1], pellet_idx, :], axis=0)
    else:
        initial_pellet_loc = None

    return initial_pellet_loc


def test_if_pellet_moved(dlc_output, pts3d, initial_pellet_loc, pelletscore_threshold=0.95, pelletname='pellet', pellet_movement_tolerance=1.0):

    if initial_pellet_loc is None:
        # the pellet wasn't identified in the first place
        return None

    bodyparts = dlc_output['bodyparts']
    pellet_idx = bodyparts.index(pelletname.lower())

    pellet_scores = dlc_output['scores'][:, :, pellet_idx]

    pellet_traj = pts3d[:, pellet_idx, :]
    pellet_diff_from_init = pellet_traj - initial_pellet_loc
    pellet_dist_from_init = np.linalg.norm(pellet_diff_from_init, axis=1)

    if np.mean(pellet_dist_from_init) > pellet_movement_tolerance:
        # is this the right metric? I think better than just any single value far from the initial pellet location.
        # this means it must have moved a fair amount after being touched.
        return True
    else:
        return False

    if (pellet_scores < pelletscore_threshold).any():
        # at some point, the pellet disappeared in at least one view




def find_reaching_pawparts(bodyparts, paw_pref):

    pawparts = ['pawdorsum', 'mcp1', 'mcp2', 'mcp3', 'mcp4', 'pip1', 'pip2', 'pip3', 'pip4', 'dig1', 'dig2', 'dig3', 'dig4', 'palm']

    reaching_pawparts = [paw_pref.lower() + pp for pp in pawparts]

    return reaching_pawparts


def identify_pellet_drop(r3d_data):
    pass


def identify_slot_breach(r3d_data):
    # r3d_data = {'points3d': p3ds,
    #             'optim_points3d': optim_p3ds,
    #             'calibration_data': calibration_data,
    #             'h5_group': h5_group,
    #             'anipose_config': anipose_config,
    #             'dlc_output': d}
    pass

# r3d_data = {'points3d': p3ds,
#             'optim_points3d': optim_p3ds,
#             'calibration_data': calibration_data,
#             'h5_group': h5_group,
#             'anipose_config': anipose_config,
#             'dlc_output': d}
