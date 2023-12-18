import skilled_reaching_io
import navigation_utilities
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
import scipy.stats
import os
import glob


def analyze_trajectories(traj_folder, pellet_score_thresh=0.95):

    traj_files = glob.glob(os.path.join(traj_folder, '*r3d.pickle'))

    # first, estimate the slot position and mean initial pellet location for this session
    trials_slot_z = np.zeros(len(traj_files))
    init_pellet_locs = np.zeros((len(traj_files), 3))
    for i_traj_file, traj_file in enumerate(traj_files):
        r3d_data = skilled_reaching_io.read_pickle(traj_file)
        paw_pref = r3d_data['rat_info']['pawpref'].values[0]
        trials_slot_z[i_traj_file] = find_slot_z(r3d_data, paw_pref)

        trial_pellet_loc = find_initial_pellet_loc(r3d_data['dlc_output'], r3d_data['points3d'], score_threshold=pellet_score_thresh,
                                                   pelletname='pellet', test_frame_range=(200, 250))
        if trial_pellet_loc is None:
            init_pellet_locs[i_traj_file, :] = np.nan
        else:
            init_pellet_locs[i_traj_file, :] = trial_pellet_loc

    mean_init_pellet_loc = np.nanmean(init_pellet_locs, axis=0)
    # need to do something to eliminate outliers
    slot_z_inliers = exclude_outliers_by_zscore(trials_slot_z, max_zscore=3.)
    session_slot_z = np.mean(slot_z_inliers)
    pass

    for traj_file in traj_files:
        analyze_trajectory(traj_file, slot_z=session_slot_z)


def exclude_outliers_by_zscore(data, max_zscore=3.):
    '''
    find outliers in an array by eliminating nan's, then calculating z-scores and removing any
    points whose absolute z-score is > max_zscore
    :param data:
    :param max_zscore:
    :return:
    '''
    data = data[np.logical_not(np.isnan(data))]

    data_zscores = scipy.stats.zscore(data)

    inliers = data[abs(data_zscores) < max_zscore]

    return inliers


def analyze_trajectory(trajectory_fname, slot_z=None,
                       pellet_score_thresh=0.95,
                       pelletname='pellet'):
    # trajectory_fname = navigation_utilities.create_trajectory_name(h5_metadata, session_metadata, calibration_data,
    #                                                                parent_directories)

    if os.path.exists(trajectory_fname):
        r3d_data = skilled_reaching_io.read_pickle(trajectory_fname)

    bodyparts = r3d_data['dlc_output']['bodyparts']
    paw_pref = r3d_data['rat_info']['pawpref'].values[0]

    pts3d = r3d_data['optim_points3d']

    if slot_z is None:
        slot_z = find_slot_z(r3d_data, paw_pref)

    initial_pellet_loc = find_initial_pellet_loc(r3d_data['dlc_output'], pts3d, score_threshold=pellet_score_thresh,
                                         pelletname=pelletname, test_frame_range=[200, 250])

    # slot_z_wrtpellet =
    pts3d_wrt_pellet = pts3d - initial_pellet_loc
    reach_data = identify_reaches(pts3d_wrt_pellet, bodyparts, paw_pref, slot_z)
    reach_data = identify_grasps(pts3d_wrt_pellet, bodyparts, paw_pref, slot_z, reach_data)

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
        pp_idx = [bodyparts.index(pp) for pp in reaching_pawparts]

        pass


def find_slot_z(r3d_data, paw_pref):

    nbins=50
    num_testpts = 1000
    smooth_win = 51

    # don't use optimized points - that includes interpolation through the slot
    pts3d = r3d_data['points3d']

    bodyparts = r3d_data['dlc_output']['bodyparts']
    reaching_pawparts = find_reaching_pawparts(bodyparts, paw_pref)

    pp_idx = [bodyparts.index(pp) for pp in reaching_pawparts]

    paw_z = pts3d[:, pp_idx, 2]
    all_paw_z = np.reshape(paw_z, (-1))
    all_paw_z = all_paw_z[np.logical_not(np.isnan(all_paw_z))]
    zhist = np.histogram(all_paw_z, bins=nbins)
    zhist_dist = scipy.stats.rv_histogram(zhist)

    z_testvals = np.linspace(min(all_paw_z), max(all_paw_z), num_testpts)
    smoothed_dist = smooth(zhist_dist.pdf(z_testvals), smooth_win)

    z_mins_idx, min_props = scipy.signal.find_peaks(-smoothed_dist, prominence=max(smoothed_dist) / 3)

    poss_mins = smoothed_dist[z_mins_idx]
    try:
        z_min_idx = z_mins_idx[np.where(poss_mins == min(poss_mins))[0][0]]
        slot_z = z_testvals[z_min_idx]
    except:
        slot_z = np.nan

    return slot_z

def identify_reaches(pts3d, bodyparts, paw_pref, slot_z, pp2follow='dig2', min_reach_prominence=7):
    pp2follow = paw_pref.lower() + pp2follow
    pp_idx = bodyparts.index(pp2follow)

    all_dig = [paw_pref.lower() + 'dig{:d}'.format(i_dig + 1) for i_dig in range(4)]
    all_dig_idx = [bodyparts.index(dig_name) for dig_name in all_dig]
    pp2follow_z = pts3d[:, pp_idx, 2]

    pd_name = paw_pref.lower() + 'pawdorsum'
    pd_idx = bodyparts.index(pd_name)
    pd_z = pts3d[:, pd_idx, 2]

    # may have to adust/add parameters to find more peaks
    z_mins_idx, min_props = scipy.signal.find_peaks(-pp2follow_z, prominence=min_reach_prominence)
    z_mins = pp2follow_z[z_mins_idx]

    # only take reaches where z_min is less than the slot_z
    reach_z_mins = z_mins[z_mins < slot_z]
    reach_z_mins_idx = z_mins_idx[z_mins < slot_z]

    # from matlab code, need to decide if we need this
    # reaches_to_keep = islocalmin(-pp2follow_z, prominence=1, 'prominencewindow',[0,1000], distance=minGraspSeparation)
    # reachMins = reachMins & reaches_to_keep;

    # matlab code now looks for pawparts being too far apart to be a legit reach; I think
    # that is already taken care of in the optim_3dpts routine from anipose

    # make sure all digits were through the slot at the end of the reach
    all_dig_z = pts3d[:, all_dig_idx, 2]
    valid_reach_ends = []
    for min_idx in reach_z_mins_idx:
        if all(all_dig_z[min_idx, :] < slot_z):
            valid_reach_ends.append(min_idx)

    # find the paw dorsum maxima in between each reach termination
    valid_reach_starts = []
    for i_reach, reach_end in enumerate(valid_reach_ends):
        if i_reach == 0:
            start_frame = 0
        else:
            start_frame = valid_reach_ends[i_reach - 1]
        last_frame = valid_reach_ends[i_reach]
        interval_dig2_z_max = max(pp2follow_z[start_frame : last_frame])
        interval_pd_z_max = max(pd_z[start_frame : last_frame])

        # find the frame where the paw dorsum starts moving forward before this reach ends
        valid_reach_starts.append(np.where(pd_z[start_frame : last_frame] == interval_pd_z_max)[0][0])
        valid_reach_starts[-1] += start_frame

    reach_data = {'start_frames': valid_reach_starts,
                  'end_frames': valid_reach_ends}

    return reach_data


def identify_grasps(pts3d, bodyparts, paw_pref, slot_z, reach_data):
    all_mcp = [paw_pref.lower() + 'mcp{:d}'.format(i_dig + 1) for i_dig in range(4)]
    all_pip = [paw_pref.lower() + 'pip{:d}'.format(i_dig + 1) for i_dig in range(4)]
    all_dig = [paw_pref.lower() + 'dig{:d}'.format(i_dig + 1) for i_dig in range(4)]

    all_parts = all_mcp + all_pip + all_dig
    all_parts.append(paw_pref.lower() + 'pawdorsum')

    all_parts_idx = [bodyparts.index(pp) for pp in all_parts]

    # all_mcp_idx = [bodyparts.index(mcp_name) for mcp_name in all_mcp]
    # all_pip_idx = [bodyparts.index(pip_name) for pip_name in all_pip]
    # all_dig_idx = [bodyparts.index(dig_name) for dig_name in all_dig]

    xyz_coords = pts3d[:, all_parts_idx, :]

    # assume the pellet location has already been subtracted from the trajectory
    dist_from_pellet = np.linalg.norm(xyz_coords, axis=2)

    num_reaches = len(reach_data['start_frames'])

    for i_reach in range(num_reaches):
        start_frame = reach_data['start_frames'][i_reach]
        end_frame = reach_data['end_frames'][i_reach]
    pass



def get_reaching_traj(pts3d, dlc_output, reaching_pawparts):
    # working here...

    bodyparts = dlc_output['bodyparts']
    # get indices in pts3d of reaching_pawparts
    pp_idx = [bodyparts.index(pp) for pp in reaching_pawparts]

    reaching_traj = pt

    pass


def find_initial_pellet_loc(dlc_output, pts3d, score_threshold=0.95, pelletname='pellet', test_frame_range=(200, 250)):

    # should we do a moving average until the pellet stops moving to make sure the pedestal is already up before we estimate its position?
    # todo: should we have manual scores loaded in here, too? that would allow us to check for zeros, indicating there was no pellet

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


def find_reaching_pawparts(bodyparts, paw_pref):

    pawparts = ['pawdorsum', 'mcp1', 'mcp2', 'mcp3', 'mcp4', 'pip1', 'pip2', 'pip3', 'pip4', 'dig1', 'dig2', 'dig3', 'dig4']

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
def smooth(data, span):

    # data: NumPy 1-D array containing the data to be smoothed
    # span: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation

    # todo: change the type of window
    out0 = np.convolve(data, np.ones(span, dtype=int), 'valid') / span
    r = np.arange(1, span - 1, 2)
    start = np.cumsum(data[:span - 1])[::2] / r
    stop = (np.cumsum(data[:-span:-1])[::2] / r)[::-1]

    return np.concatenate((start, out0, stop))