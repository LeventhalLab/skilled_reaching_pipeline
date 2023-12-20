import skilled_reaching_io
import navigation_utilities
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
import scipy.stats
import os
import glob
import compute_angles


def analyze_trajectories(traj_folder, anipose_config, pellet_score_thresh=0.95, init_pellet_frames=(299, 301)):

    traj_files = glob.glob(os.path.join(traj_folder, '*r3d.pickle'))

    # first, estimate the slot position and mean initial pellet location for this session
    trials_slot_z = np.zeros(len(traj_files))
    init_pellet_locs = np.zeros((len(traj_files), 3))
    for i_traj_file, traj_file in enumerate(traj_files):
        r3d_data = skilled_reaching_io.read_pickle(traj_file)
        paw_pref = r3d_data['rat_info']['pawpref'].values[0]
        trials_slot_z[i_traj_file] = find_slot_z(r3d_data, paw_pref)

        trial_pellet_loc = find_initial_pellet_loc(r3d_data['dlc_output'], r3d_data['points3d'], score_threshold=pellet_score_thresh,
                                                   pelletname='pellet', test_frame_range=init_pellet_frames)
        if trial_pellet_loc is None:
            init_pellet_locs[i_traj_file, :] = np.nan
        else:
            init_pellet_locs[i_traj_file, :] = trial_pellet_loc

    # sometimes, the rat reaches before the pellet elevated all the way, so probably should ignore reaches where the pellet was lower
    mean_init_pellet_loc = np.nanmean(init_pellet_locs, axis=0)
    # need to do something to eliminate outliers
    slot_z_inliers = exclude_outliers_by_zscore(trials_slot_z, max_zscore=3.)
    session_slot_z = np.mean(slot_z_inliers)
    pass

    for traj_file in traj_files:
        traj_file = traj_files[-1]
        analyze_trajectory(traj_file, mean_init_pellet_loc, anipose_config, slot_z=session_slot_z)


def exclude_outliers_by_zscore(data, max_zscore=3.):
    '''
    find outliers in an array by eliminating nan's, then calculating z-scores and removing any
    points whose absolute z-score is > max_zscore
    :param data:
    :param max_zscore:
    :return:
    '''
    data = data[np.logical_not(np.isnan(data))]

    if len(data) == 1:
        return data

    data_zscores = scipy.stats.zscore(data)

    inliers = data[abs(data_zscores) < max_zscore]

    return inliers


def analyze_trajectory(trajectory_fname, mean_init_pellet_loc, anipose_config,
                       slot_z=None,
                       pellet_score_thresh=0.95,
                       init_pellet_frames=(299, 301),
                       pelletname='pellet',
                       pellet_movement_tolerance=1.):
    # trajectory_fname = navigation_utilities.create_trajectory_name(h5_metadata, session_metadata, calibration_data,
    #                                                                parent_directories)

    if os.path.exists(trajectory_fname):
        r3d_data = skilled_reaching_io.read_pickle(trajectory_fname)

    bodyparts = r3d_data['dlc_output']['bodyparts']
    paw_pref = r3d_data['rat_info']['pawpref'].values[0]

    pts3d = r3d_data['optim_points3d']

    # construct the vecs dictionary that gets passed to get_angles in anipose
    vecs = dict()
    for bp in bodyparts:
        bp_idx = bodyparts.index(bp)
        vec = pts3d[:, bp_idx, :]
        vecs[bp] = vec

    dig_angles = compute_angles.get_angles(vecs, anipose_config.get('angles', dict()))

    if slot_z is None:
        slot_z = find_slot_z(r3d_data, paw_pref)

    init_pellet_loc = find_initial_pellet_loc(r3d_data['dlc_output'], pts3d, score_threshold=pellet_score_thresh,
                                         pelletname=pelletname, test_frame_range=init_pellet_frames)

    if init_pellet_loc is None:
        init_pellet_loc = mean_init_pellet_loc

    slot_z_wrt_pellet = slot_z - init_pellet_loc[2]
    pts3d_wrt_pellet = pts3d - init_pellet_loc

    pellet_move_frame = find_pellet_movement(r3d_data['dlc_output'], pts3d, np.zeros(3), r3d_data['reprojerr'], pelletscore_threshold=0.95,
                                           pelletname=pelletname,
                                           pellet_movement_tolerance=pellet_movement_tolerance,
                                           init_pellet_loc=init_pellet_loc)
    reach_data = identify_reaches(pts3d_wrt_pellet, bodyparts, paw_pref, slot_z_wrt_pellet)
    reach_data['pellet_move_frame'] = pellet_move_frame
    reach_data = identify_grasps(pts3d_wrt_pellet, r3d_data['dlc_output'], paw_pref, dig_angles, reach_data, r3d_data['reprojerr'], frames2lookforward=40)

    # define retraction as when the pellet comes back inside the chamber (+/- pellet)? then the start of retraction would be when paw
    # starts moving backwards after grasp?
    reach_data = identify_retraction(pts3d_wrt_pellet, slot_z_wrt_pellet, r3d_data['dlc_output'], paw_pref, reach_data, r3d_data['reprojerr'], frames2lookforward=40)

    # calculate aperture, paw orientation
    reach_data = calculate_reach_kinematics(reach_data, paw_pref, pts3d)

    f_contact, bp_contact = identify_pellet_contact(r3d_data, paw_pref, pelletname='pellet')
    pass


def calculate_reach_kinematics(reach_data, paw_pref, pts3d, init_pellet_loc=np.zeros(3)):

    pts3d = pts3d - init_pellet_loc


def calc_paw_orientation(pts3d, paw_pref, bodyparts):
    pass


def calc_aperture(pts3d, paw_pref, bodyparts):
    '''

    :param pts3d: 3d points already normalized to make the pellet the origin
    :param paw_pref:
    :param bodyparts:
    :return:
    '''
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

def identify_reaches(pts3d, bodyparts, paw_pref, slot_z, pp2follow='dig2', min_reach_prominence=7, triggerframe=300):
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

    # only take reaches where z_min is less than the slot_z and it occurred after the trigger frame
    reach_z_mins = z_mins[z_mins < slot_z]
    reach_z_mins_idx = z_mins_idx[z_mins < slot_z]
    reach_z_mins_idx = reach_z_mins_idx[reach_z_mins_idx > triggerframe]

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


def identify_grasps(pts3d, dlc_output, paw_pref, dig_angles, reach_data, reprojerr, init_pellet_loc = np.zeros(3), frames2lookforward=40,
                    pelletname='pellet', pellet_movement_tolerance=1.):
    '''

    :param pts3d: should be with the pellet at the origin
    :param dlc_output:
    :param paw_pref:
    :param reach_data:
    :param reprojerr:
    :param init_pellet_loc:
    :param frames2lookforward:
    :param pelletname:
    :param pellet_movement_tolerance:
    :return:
    '''
    # find the nearest paw part to the initial pellet location to identify end of grasp
    # assume trajectory has already been adjusted to put the pellet at the origin

    bodyparts = dlc_output['bodyparts']
    all_mcp = [paw_pref.lower() + 'mcp{:d}'.format(i_dig + 1) for i_dig in range(4)]
    all_pip = [paw_pref.lower() + 'pip{:d}'.format(i_dig + 1) for i_dig in range(4)]
    all_dig = [paw_pref.lower() + 'dig{:d}'.format(i_dig + 1) for i_dig in range(4)]

    all_parts = all_mcp + all_pip + all_dig
    all_parts.append(paw_pref.lower() + 'pawdorsum')

    all_parts_idx = [bodyparts.index(pp) for pp in all_parts]

    n_frames = np.shape(pts3d)[0]
    n_reach_parts = len(all_parts_idx)
    # all_mcp_idx = [bodyparts.index(mcp_name) for mcp_name in all_mcp]
    # all_pip_idx = [bodyparts.index(pip_name) for pip_name in all_pip]
    # all_dig_idx = [bodyparts.index(dig_name) for dig_name in all_dig]

    pts3d = pts3d - init_pellet_loc
    xyz_coords = pts3d[:, all_parts_idx, :]

    # assume the pellet location has already been subtracted from the trajectory

    # this needs to be modified to use the current pellet location, not the initial pellet location. These should usually
    # be the same thing, but often the rat reaches before the pellet is all the way up, and is therefore still moving
    # when the paw breaches the slot
    pellet_idx = bodyparts.index(pelletname)
    pellet_locs = pts3d[:, pellet_idx, :]
    dist_from_pellet = np.zeros((n_frames, n_reach_parts))
    for i_bpt in range(n_reach_parts):
        dist_from_pellet[:, i_bpt] = np.linalg.norm(xyz_coords[:, i_bpt, :] - pellet_locs, axis=1)

    n_reaches = len(reach_data['start_frames'])
    n_parts = len(all_parts)
    min_dist = np.empty(n_reaches)
    min_dist_partidx = np.zeros(n_reaches, dtype=int)
    min_dist_frame = np.zeros((n_reaches, n_parts), dtype=int)
    all_min_dist = np.empty((n_reaches, n_parts))
    reach_data['grasp_starts'] = []
    reach_data['grasp_ends'] = []
    reach_data['pellet_contact'] = []
    reach_data['min_dist_frame'] = []
    reach_data['min_dist_to_pellet'] = []
    reach_data['reaching_pawparts'] = []
    reach_data['min_dist_partidx'] = []
    for i_reach in range(n_reaches):
        start_frame = reach_data['start_frames'][i_reach]
        end_frame = reach_data['end_frames'][i_reach]

        # make sure the reach ended within frames2lookforward frames of the end of the video
        if end_frame + frames2lookforward > n_frames:
            last_frame2check = n_frames
        else:
            last_frame2check = end_frame + frames2lookforward

        # minimum distance for each paw part from the pellet
        all_min_dist[i_reach, :] = np.min(dist_from_pellet[end_frame : last_frame2check, :], axis=0)
        # minimum distance among all paw parts from the pellet
        min_dist[i_reach] = np.min(all_min_dist[i_reach, :])
        min_dist_partidx[i_reach] = int(np.where(all_min_dist[i_reach, :] == min_dist[i_reach])[0][0])
        min_dist_frame_reach = np.array([np.where(dist_from_pellet[end_frame : last_frame2check, i_part] == all_min_dist[i_reach, i_part]) for i_part in range(n_parts)])
        min_dist_frame_reach = np.squeeze(min_dist_frame_reach) + end_frame
        min_dist_frame[i_reach, :] = min_dist_frame_reach

        if i_reach == n_reaches - 1:
            next_reach_frame = n_frames
        else:
            next_reach_frame = reach_data['start_frames'][i_reach + 1]
        if reach_data['pellet_move_frame'] > start_frame and reach_data['pellet_move_frame'] < next_reach_frame:
            reach_data['pellet_contact'].append(reach_data['pellet_move_frame'])
        else:
            reach_data['pellet_contact'].append(None)

        ''' from Bova et al, 2021
         The start of the grasp was defined as the frame at which flexion of the second digit started to increase after reaching minimum flexion 
         (i.e., maximum extension). Grasp end was defined as the first frame with maximum digit flexion after grasp start. 
         '''
        # find the maximum digit2 extension for this reach
        dig_angle2track = paw_pref + 'dig2_angle'
        dig2_angles = dig_angles[dig_angle2track]

        max_extension = max(dig2_angles[start_frame : end_frame])
        max_ext_frame = np.where(dig2_angles[start_frame : end_frame] == max_extension)[0][0] + start_frame

        # find the next local minimum after max digit extension
        maxflex_idx, maxflex_props = scipy.signal.find_peaks(-dig2_angles[max_ext_frame:], prominence=10)
        reach_data['grasp_starts'].append(max_ext_frame)
        reach_data['grasp_ends'].append(maxflex_idx[0] + max_ext_frame)
        # max_flexion = min(dig2_angles[max_ext_frame : last_frame2check])
        # max_flex_frame = np.where(dig2_angles[max_ext_frame : last_frame2check] == max_flexion)[0][0] + max_ext_frame

        reach_data['min_dist_frame'].append(min_dist_frame)
        reach_data['min_dist_to_pellet'].append(all_min_dist)
        reach_data['reaching_pawparts'].append(all_parts)
        reach_data['min_dist_partidx'].append(min_dist_partidx)

    return reach_data


def identify_retraction(pts3d_wrt_pellet, slot_z, dlc_output, paw_pref, reach_data, reprojerr, fps=300, v_thresh=50, frames2lookforward=40):

    # figure out z-coordinates of reaching paw parts at end of grasp, see when they start moving backwards
    bodyparts = dlc_output['bodyparts']
    all_mcp = [paw_pref.lower() + 'mcp{:d}'.format(i_dig + 1) for i_dig in range(4)]
    all_pip = [paw_pref.lower() + 'pip{:d}'.format(i_dig + 1) for i_dig in range(4)]
    all_dig = [paw_pref.lower() + 'dig{:d}'.format(i_dig + 1) for i_dig in range(4)]

    all_parts = all_mcp + all_pip + all_dig
    all_parts.append(paw_pref.lower() + 'pawdorsum')

    all_parts_idx = [bodyparts.index(pp) for pp in all_parts]

    xyz_coords = pts3d_wrt_pellet[:, all_parts_idx, :]

    n_reaches = len(reach_data['start_frames'])

    reach_data['retract_frames'] = []
    for i_reach in range(n_reaches):

        end_frame = reach_data['grasp_ends'][i_reach]

        # find the paw points at the end of the grasping phase
        graspend_pawpts = xyz_coords[end_frame, :, :]
        graspend_meanloc = np.nanmean(graspend_pawpts, axis=0)

        future_frames_meanloc = np.nanmean(xyz_coords[end_frame:, :, :], axis=1)

        z_v = np.diff(future_frames_meanloc[:, 2]) * fps

        max_v_idx, max_props = scipy.signal.find_peaks(z_v, prominence=10)
        max_v_idx = max_v_idx[z_v[max_v_idx] > v_thresh]
        if len(max_v_idx) > 0:
            max_v_idx = max_v_idx[0]

            v_trough_idx, trough_props = scipy.signal.find_peaks(-z_v[:max_v_idx])
            # find last velocity trough before peak
            if len(v_trough_idx) > 0:
                v_trough_idx = v_trough_idx[-1]
            else:
                v_trough_idx = 0
        else:
            # todo: figure out what to do if no peak velocity is found - maybe lower requirement for max velocity or just use the average z-coordinate?
            pass
        reach_data['retract_frames'].append(v_trough_idx)

    return reach_data


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


def find_pellet_movement(dlc_output, pts3d, initial_pellet_loc, reprojerr, pelletscore_threshold=0.95, pelletname='pellet', pellet_movement_tolerance=1.0,
                         triggerframe=300, init_pellet_loc=np.zeros(3)):

    if initial_pellet_loc is None:
        # the pellet wasn't identified in the first place
        return None

    pts3d = pts3d - init_pellet_loc

    bodyparts = dlc_output['bodyparts']
    pellet_idx = bodyparts.index(pelletname.lower())

    pellet_scores = dlc_output['scores'][:, :, pellet_idx]

    pellet_traj = pts3d[:, pellet_idx, :]
    pellet_diff_from_init = pellet_traj - initial_pellet_loc
    pellet_dist_from_init = np.linalg.norm(pellet_diff_from_init, axis=1)

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.plot(pellet_dist_from_init)
    # plt.show()
    # find the first place the pellet displacement is greater than pellet_movement_tolerance

    pellet_moved_frames = (pellet_dist_from_init > pellet_movement_tolerance)
    first_pellet_moved_frame = np.argmax(pellet_moved_frames)

    if first_pellet_moved_frame == 0:
        # what if there is no clear pellet movement frame? Not sure if I need to worry about this...

        # did the pellet disappear or did it just never move?
        # find the first frame where the pellet could not be found in any view
        max_frame_pellet_scores = np.max(pellet_scores, axis=0)

        pass

    return first_pellet_moved_frame




    if np.mean(pellet_dist_from_init) > pellet_movement_tolerance:
        # is this the right metric? I think better than just any single value far from the initial pellet location.
        # this means it must have moved a fair amount after being touched.
        return True
    else:
        return False


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