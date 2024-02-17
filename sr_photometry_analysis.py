import photometry_analysis as pa
import photometry_analysis_plots as pa_plots
# import photometry_io_skilledreaching as io_utils
import sr_photometry_analysis as srphot_anal
import skilled_reaching_io
import sr_analysis
import navigation_utilities
import sr_analysis
import numpy as np
import math
from scipy.optimize import curve_fit
from datetime import datetime
from matplotlib import pyplot as plt
import os

import matplotlib
matplotlib.use('Qt5Agg')


def resample_photometry_to_video(photometry_signal, trigger_ts, Fs, trigger_frame=300, num_frames=1300, fps=300):

    if np.isnan(trigger_ts):
        # most likely, this video was recorded after the photometry recording completed but before the investigator could
        # stop the skilled reaching task
        return None
    num_points = len(photometry_signal)
    photometry_t = np.linspace(1/Fs, num_points/Fs, num_points)

    phot_vidsignal = np.zeros(num_frames)
    for i_frame in range(num_frames):

        frame_time = trigger_ts + (i_frame - trigger_frame) / fps
        time_diffs = abs(frame_time - photometry_t)

        t_idx = np.where(time_diffs == min(time_diffs))[0][0]
        phot_vidsignal[i_frame] = photometry_signal[t_idx]

    return phot_vidsignal


def add_trialdata_to_srdf(rat_srdf, session_metadata, sr_ts, sr_intervals, session_duration):

    vidtrigger_ts = sr_ts['vidtrigger']
    act3_ts = sr_ts['act3']
    vidtrigger_intervals = sr_intervals['vidtrigger']
    act3_intervals = sr_intervals['act3']
    # find rows
    session_rows = (rat_srdf['session_date'] == session_metadata['date']) & (rat_srdf['date_session_num'] == session_metadata['session_num'])
    session_row_idx = np.where(session_rows)[0]

    # if there are more session rows than vidtrigger events, that's because the task kept running after the photometry recording stopped
    # if n_vidtriggers is the number of vidtrigger events recorded in the photometry stream, assign the first n_vidtrigger rows to have a vidtrigger timestamp

    # also, there are some where there are more vidtrigger events than there are videos. This sometimes happens when there is a "line pop"
    # I think I fixed this, but if it continues to happen, can look at video names for times.
    if len(session_row_idx) < len(vidtrigger_ts):
        pass   # to catch instances when there are more vidtrigger events than there should be

    for i_trial_in_session, row_idx in enumerate(session_row_idx):
        if i_trial_in_session < len(vidtrigger_ts):
            # find the last actuator 3 timestamp before the current reach
            trial_act3_ts = act3_ts[act3_ts < vidtrigger_ts[i_trial_in_session]]
            if len(trial_act3_ts) > 0:
                trial_act3_ts = trial_act3_ts[-1]
                trial_act3_idx = np.where(act3_ts == trial_act3_ts)[0][0]
                rat_srdf.loc[row_idx, 'act3_interval'] = act3_intervals[trial_act3_idx]
            else:
                trial_act3_ts = 0
                rat_srdf.loc[row_idx, 'act3_interval'] = -1
                # something's weird with 453 on 2/23 - actuator3 and vidtriggers overlap, not sure why. Vids look OK, maybe cables were plugged in wrong?

            rat_srdf.loc[row_idx, 'act3_ts'] = trial_act3_ts
            rat_srdf.loc[row_idx, 'vidtrigger_ts'] = vidtrigger_ts[i_trial_in_session]
            rat_srdf.loc[row_idx, 'vidtrigger_interval'] = vidtrigger_intervals[i_trial_in_session]
            rat_srdf.loc[row_idx, 'session_type'] = session_metadata['task']
            rat_srdf.loc[row_idx, 'session_duration'] = session_duration
        else:
            rat_srdf.loc[row_idx, 'session_type'] = session_metadata['task']
            rat_srdf.loc[row_idx, 'session_duration'] = session_duration
            break

    return rat_srdf


def extract_sr_event_ts(processed_phot_data, session_metadata, rat_srdf, perievent_window=(-3, 3), smooth_window=101, f0_pctile=10, expected_baseline=0.2):

    eventlist, event_ts = srphot_anal.get_photometry_events(processed_phot_data, session_metadata)
    Fs = processed_phot_data['Fs']
    n_samples = len(processed_phot_data['t'])
    session_duration = n_samples / Fs

    # collect all actuator3 events
    act3_idx = eventlist.index('Actuator3')
    act3_ts = event_ts[act3_idx]

    # collect all vidtrigger events
    vidtrig_idx = eventlist.index('vid_trigger')
    vidtrigger_ts = event_ts[vidtrig_idx]

    # assign actuator3 and vidtrigger events to valid recording intervals
    act3_intervals = srphot_anal.identify_ts_intervals(act3_ts, processed_phot_data['analysis_intervals'], perievent_window, processed_phot_data['Fs'])
    vidtrigger_intervals = srphot_anal.identify_ts_intervals(vidtrigger_ts, processed_phot_data['analysis_intervals'], perievent_window, processed_phot_data['Fs'])

    sr_ts = {'act3': act3_ts, 'vidtrigger': vidtrigger_ts}
    sr_intervals = {'act3': act3_intervals, 'vidtrigger': vidtrigger_intervals}
    rat_srdf = add_trialdata_to_srdf(rat_srdf, session_metadata, sr_ts, sr_intervals, session_duration)

    # find all early reaches
    earlyreach_ts = srphot_anal.find_early_reaches(eventlist, event_ts)
    earlyreach_intervals = srphot_anal.identify_ts_intervals(earlyreach_ts, processed_phot_data['analysis_intervals'], perievent_window, processed_phot_data['Fs'])

    earlyreach_info = {'earlyreach_ts': earlyreach_ts,
                       'earlyreach_intervals': earlyreach_intervals}

    return rat_srdf, earlyreach_info


def aggregate_data_pre_20230904(processed_phot_data, session_metadata, rat_srdf, smooth_window=101, f0_pctile=10, expected_baseline=0.2, perievent_window=(-5, 5)):

    smoothed_data, detrended_data, session_dff, interval_popt, interval_exp2_fit_successful, baseline_used = srphot_anal.calc_segmented_dff(
        processed_phot_data, processed_phot_data['mean_baseline'],
        smooth_window=smooth_window,
        f0_pctile=f0_pctile,
        expected_baseline=expected_baseline)

    session_zscore = srphot_anal.zscore_full_session(processed_phot_data['analysis_intervals'], session_dff)

    rat_srdf, session_earlyreach_info = extract_sr_event_ts(processed_phot_data, session_metadata, rat_srdf,
                                                            perievent_window=perievent_window,
                                                            smooth_window=smooth_window, f0_pctile=f0_pctile,
                                                            expected_baseline=expected_baseline)
    session_summary = {'sr_dff1': session_dff,
                       'sr_zscores1': session_zscore,
                       'sr_dff2': [],
                       'sr_zscores2': [],
                       'sr_metadata_list': session_metadata,
                       'sr_processed_phot': processed_phot_data,
                       'sr_earlyreach_info': session_earlyreach_info
                       }

    return session_summary, rat_srdf


def aggregate_data_post_20230904(data_files, parent_directories, session_metadata, rat_srdf, smooth_window=101, f0_pctile=10, expected_baseline=0.2, perievent_window=(-5, 5)):

    pickled_metadata_fname = navigation_utilities.get_pickled_metadata_fname(session_metadata, parent_directories)
    pickled_analog_processeddata_fname = navigation_utilities.get_pickled_analog_processeddata_fname(session_metadata, parent_directories)
    pickled_ts_fname = navigation_utilities.get_pickled_ts_fname(session_metadata, parent_directories)


    _, pickle_name = os.path.split(pickled_metadata_fname)
    print('{} already processed'.format(pickle_name))
    phot_metadata = skilled_reaching_io.read_pickle(pickled_metadata_fname)

    processed_analog = skilled_reaching_io.read_pickle(pickled_analog_processeddata_fname)
    ts_dict = skilled_reaching_io.read_pickle(pickled_ts_fname)

    # smoothed_data, detrended_data, session_dff, interval_popt, interval_exp2_fit_successful, baseline_used = srphot_anal.calc_segmented_dff(
    #     processed_phot_data, processed_phot_data['mean_baseline'],
    #     smooth_window=smooth_window,
    #     f0_pctile=f0_pctile,
    #     expected_baseline=expected_baseline)

    # session_zscore = srphot_anal.zscore_full_session(processed_phot_data['analysis_intervals'], session_dff)

    Fs = phot_metadata['Fs']
    n_samples = np.shape(processed_analog['dff'])[0]
    phot_metadata['t'] = np.linspace(1/Fs, n_samples/Fs, n_samples)
    rat_srdf, session_earlyreach_info = extract_sr_event_ts(phot_metadata, session_metadata, rat_srdf,
                                                            perievent_window=perievent_window,
                                                            smooth_window=smooth_window, f0_pctile=f0_pctile,
                                                            expected_baseline=expected_baseline)

    session_summary = {'sr_dff1': processed_analog['dff'][:, 0],
                       'sr_zscores1': processed_analog['session_zscores'][:, 0],
                       'sr_dff2': processed_analog['dff'][:, 1],
                       'sr_zscores2': processed_analog['session_zscores'][:, 1],
                       'sr_metadata_list': session_metadata,
                       'sr_processed_phot': phot_metadata,
                       'sr_earlyreach_info': session_earlyreach_info
                       }

    return session_summary, rat_srdf



def exp1_offset_func(t, a, b, c):
    return a * np.exp(b * t) + c


def exp2_func(t, a, b, c, d):
    return a * np.exp(b * t) + c * np.exp(d * t)


def exp1_func(t, a, b):
    return a * np.exp(b * t)


def align_on_off_idx(pulse_on_idx, pulse_off_idx):
    '''
    make sure that pulse_off indices follow pulse_in indices (i.e., make sure signal didn't turn off before it turned on
    as may happen at the beginning of a recording.
    :param pulse_on_idx:
    :param pulse_off_idx:
    :return: pulse_off_idx: pulse_off_idx array starting at the first "off" event after an "on" event
    '''

    # test if pulse_on_idx has at least two elements
    if np.size(pulse_on_idx) > 1:
        pulse_off_after_first_on = np.squeeze(np.argwhere(pulse_off_idx > pulse_on_idx[0]))
    elif np.size(pulse_on_idx) == 1:
        if np.ndim(pulse_off_idx) == 0:
            pulse_off_idx = np.reshape(pulse_off_idx, 1)
        if np.ndim(pulse_on_idx) == 0:
            pulse_on_idx = np.reshape(pulse_on_idx, 1)
        pulse_off_after_first_on = np.squeeze(np.argwhere(pulse_off_idx > pulse_on_idx))
    else:
        # pulse_on_idx is empty
        return None

    if np.size(pulse_off_after_first_on) > 0:
        if np.size(pulse_off_after_first_on) > 1:
            pulse_off_start_idx = pulse_off_after_first_on[0]
        elif np.size(pulse_off_after_first_on) == 1:
            pulse_off_start_idx = pulse_off_after_first_on
        if np.ndim(pulse_off_after_first_on) == 0:
            pulse_off_start_idx = pulse_off_after_first_on
        pulse_off_idx = pulse_off_idx[pulse_off_start_idx:]
    else:
        pulse_off_idx = None

    return pulse_off_idx


def event_triggered_dff(phot_data, line_labels, event_type, reach_outcomes):
    '''
    :param phot_data: photometry data dictionary, has the keys
        data - photometry data in a num_points x 8 array
        AI_line_desc
        t - time for each point
        virus - string containing name of virus used
        cam_trigger_delay
        cam_trigger_freq
        cam_trigger_pw

    :param line_labels:
    :param event_type:
    :param reach_outcomes:
    :return:
    '''

    # figure out which channel contains event_type, find the timestamps
    eventtype_idx = np.where(line_labels == event_type)[0]
    ts, _ = pa.extract_event_timestamps(phot_data, eventtype_idx, thresh=1.5)

def calculate_zscore(data, baseline_sample_win, data_sample_win):
    '''

    :param data:
    :param baseline_sample_win:
    :param data_sample_win:
    :return:
    '''
    baseline_data = data[baseline_sample_win[0]:baseline_sample_win[1]]
    baseline_mean = np.mean(baseline_data)
    baseline_std = np.std(baseline_data)

    sample_data = data[data_sample_win[0]:data_sample_win[1]]

    sample_zscore = (sample_data - baseline_mean) / baseline_std

    return sample_zscore

def calc_exp2_fit(num_samples, popt2):
    """
    calculate the predicted fit for the double exponential
    :param num_samples:
    :param popt2: 4-element array/list [a,b,c,d] for a * exp(b*t) + c * exp(d*t)
    :return:
    """

    t_ints = list(range(num_samples))
    t_floats = np.float_(t_ints)

    # whichever of b or d is closest to zero represents the offset
    # model is a * exp(b*t) + c * exp(d*t)
    a = popt2[0]
    b = popt2[1]
    c = popt2[2]
    d = popt2[3]

    # if b > d:
    #     # exp2_fit = c * np.exp(t_floats * d)
    #     dc_offset = a
    # else:
    #     # exp2_fit = a * np.exp(t_floats * b)
    #     dc_offset = c

    exp2_fit = a * np.exp(t_floats * b) + c * np.exp(t_floats * d)

    dc_offset = exp2_fit[-1]

    return exp2_fit, dc_offset


def calc_interval_exp2_fit(interval, popt2):
    """
    calculate the predicted fit for the double exponential
    :param num_samples:
    :param popt2: 4-element array/list [a,b,c,d] for a * exp(b*t) + c * exp(d*t)
    :return:
    """

    t_ints = list(range(interval[0], interval[1]))
    t_floats = np.float_(t_ints)

    # whichever of b or d is closest to zero represents the offset
    # model is a * exp(b*t) + c * exp(d*t)
    a = popt2[0]
    b = popt2[1]
    c = popt2[2]
    d = popt2[3]
    # if b > d:
    #     # exp2_fit = c * np.exp(t_floats * d)
    #     dc_offset = a
    # else:
    #     # exp2_fit = a * np.exp(t_floats * b)
    #     dc_offset = c

    exp2_fit = a * np.exp(t_floats * b) + c * np.exp(t_floats * d)

    dc_offset = exp2_fit[-1]

    return exp2_fit, dc_offset


def calc_exp1_offset_fit(num_samples, popt1_offset):
    """
    calculate the predicted fit for the single exponential
    :param num_samples:
    :param popt1:
    :return:
    """

    t_ints = list(range(num_samples))
    t_floats = np.float_(t_ints)

    # model is a * exp(b*t) + c
    a = popt1_offset[0]
    b = popt1_offset[1]
    c = popt1_offset[2]

    exp1_offset_fit = a * np.exp(b * t_floats) + c

    dc_offset = c

    return exp1_offset_fit, dc_offset


def calc_interval_exp1_offset_fit(interval, popt1_offset):
    """
    calculate the predicted fit for the single exponential
    :param num_samples:
    :param popt1:
    :return:
    """

    t_ints = list(range(interval[0], interval[1]))
    t_floats = np.float_(t_ints)

    # model is a * exp(b*t) + c
    a = popt1_offset[0]
    b = popt1_offset[1]
    c = popt1_offset[2]

    exp1_offset_fit = a * np.exp(b * t_floats) + c

    dc_offset = c

    return exp1_offset_fit, dc_offset


def calc_exp1_fit(num_samples, popt1):
    """
    calculate the predicted fit for the single exponential
    :param num_samples:
    :param popt1:
    :return:
    """

    t_ints = list(range(num_samples))
    t_floats = np.float_(t_ints)

    # whichever of b or d is closest to zero represents the offset
    # model is a * exp(b*t) + c
    a = popt1[0]
    b = popt1[1]

    exp1_fit = a * np.exp(b * t_floats)

    return exp1_fit


def extract_perievent_signal_and_zscore(data, ts, t_win, Fs, baseline_window=(-10, 10)):
    """

    :param data:
    :param ts:
    :param t_win:
    :param Fs:
    :param baseline_window: interval in seconds to use as baseline for z-score calculation
    :return:
    """

    #todo: add baseline calculations for z-scores
    if type(ts) is list or type(ts) is np.ndarray:
        event_idx = np.around(np.array(ts) * Fs)
    elif ts is None:
        return None
    else:
        event_idx = np.around(np.array([ts]) * Fs)   # because ts is a number instead of a list (probably)

    idx_win = np.around(np.array(t_win) * Fs)
    baseline_idx_win = np.around(np.array(baseline_window) * Fs)

    window_list = []
    perievent_dff = []
    perievent_zscore = []
    for event in event_idx:
        if event + idx_win[0] >= 0 and event + idx_win[1] <= len(data):
            event_window = event + idx_win
            event_window = event_window.astype(int)
            window_list.append(event_window.tolist())

            temp_data = data[event_window[0]:event_window[1]]
            perievent_dff.append(temp_data.tolist())

        if event + baseline_idx_win[0] >= 0 and event + baseline_idx_win[1] <= len(data):
            basewindow = event + baseline_window
            basewindow = basewindow.astype(int)

            sample_zscore = calculate_zscore(data, basewindow, event_window)
            perievent_zscore.append(sample_zscore.tolist())

    try:
        perievent_dff = np.array(perievent_dff)
    except:
        pass
    perievent_zscore = np.array(perievent_zscore)

    return perievent_dff, perievent_zscore


def extract_perievent_signal(data, ts, t_win, Fs):
    """

    :param data:
    :param ts:
    :param t_win:
    :param Fs:
    :param baseline_window: interval in seconds to use as baseline for z-score calculation
    :return:
    """

    #todo: add baseline calculations for z-scores
    if type(ts) is list or type(ts) is np.ndarray:
        event_indices = np.around(np.array(ts) * Fs)
    elif ts is None:
        return None
    else:
        event_indices = np.around(np.array([ts]) * Fs)   # because ts is a number instead of a list (probably)

    idx_win = np.around(np.array(t_win) * Fs)
    samples_per_event = int(np.diff(idx_win)[0])
    perievent_data = np.empty((len(ts), samples_per_event))
    for i_event, event_idx in enumerate(event_indices):
        if event_idx + idx_win[0] >= 0 and event_idx + idx_win[1] <= len(data):
            event_window = (event_idx + idx_win).astype(int)

            perievent_data[i_event, :] = data[event_window[0]:event_window[1]]

    return perievent_data


def identify_ts_intervals(ts_vector, valid_intervals, t_win, Fs):
    num_events = len(ts_vector)
    event_intervals = np.ones(num_events, dtype=int) * (-1)

    idx_win = np.around(np.array(t_win) * Fs)

    for i_ts, ts in enumerate(ts_vector):
        sample_idx = np.around(ts) * Fs
        event_win = sample_idx + idx_win
        event_win = event_win.astype(int)
        event_interval = test_event_in_intervals(event_win, valid_intervals)

        if np.any(event_interval):
            event_intervals[i_ts] = int(np.where(event_interval)[0][0])

    return event_intervals


def extract_perievent_signal_in_valid_intervals(data, ts, t_win, valid_intervals, Fs, baseline_window=(-10, 10), include_invalid_in_pdata=False):
    """

    :param data:
    :param ts:
    :param t_win:
    :param valid_intervals:
    :param Fs:
    :param baseline_window: interval in seconds to use as baseline for z-score calculation
    :return:
    """

    #todo: add baseline calculations for z-scores
    if type(ts) is list or type(ts) is np.ndarray:
        event_idx = np.around(np.array(ts) * Fs)
    elif ts is None:
        return None, None
    else:
        event_idx = np.around(np.array([ts]) * Fs)   # because ts is a number instead of a list (probably)

    idx_win = np.around(np.array(t_win) * Fs)
    # baseline_idx_win = np.around(np.array(baseline_window) * Fs)

    #todo: working here - need to only retain events within a valid part of the signal
    num_intervals = len(valid_intervals)
    # window_list = [[] for ii in range(num_intervals)]
    if include_invalid_in_pdata:
        perievent_data = []
    else:
        perievent_data = [[] for ii in range(num_intervals)]

    # perievent_zscore = [[] for ii in range(num_intervals)]
    num_events = len(event_idx)
    event_intervals = np.ones(num_events, dtype=int) * (-1)   # use -1 to indicate that the event is not within a valid interval
    for i_event, event in enumerate(event_idx):
        # event is the sample number at which the event occurred; i_event is the number of the event in the list of events
        event_win = event + idx_win
        event_win = event_win.astype(int)
        event_interval = test_event_in_intervals(event_win, valid_intervals)

        if np.any(event_interval):
            event_intervals[i_event] = int(np.where(event_interval)[0][0])

        if not include_invalid_in_pdata:
            if np.any(event_interval):
                # window_list[event_intervals[i_event]].append(event_win.tolist())
                # event_intervals[i_event] = np.where(event_interval)

                temp_data = data[event_win[0]:event_win[1]]
                perievent_data[event_intervals[i_event]].append(temp_data.tolist())
        else:
            # window_list[event_intervals[i_event]].append(event_win.tolist())
            temp_data = data[event_win[0]:event_win[1]]
            perievent_data.append(temp_data.tolist())
            #todo: allow option for storing all perievent data including for invalid trials
            pass

        # if event + idx_win[0] >= 0 and event + idx_win[1] <= len(data):
        #     event_window = event + idx_win
        #     event_window = event_window.astype(int)
        #     window_list.append(event_window.tolist())
        #
        #     temp_data = data[event_window[0]:event_window[1]]
        #     perievent_dff.append(temp_data.tolist())

        # if event + baseline_idx_win[0] >= 0 and event + baseline_idx_win[1] <= len(data):
        #     basewindow = event + baseline_window
        #     basewindow = basewindow.astype(int)
        #
        #     sample_zscore = calculate_zscore(data, basewindow, event_window)
        #     perievent_zscore.append(sample_zscore.tolist())

    if include_invalid_in_pdata:
        perievent_data = np.array(perievent_data)
    else:
        perievent_data = [np.array(interval_event_dff) for interval_event_dff in perievent_data]

    return perievent_data, event_intervals


def test_event_in_intervals(event_win, intervals):

    num_intervals = len(intervals)
    event_interval = np.zeros(num_intervals, dtype=bool)
    for i_interval, interval in enumerate(intervals):

        if (min(event_win) >= interval[0]) and (max(event_win) <= interval[1]):
            event_interval[i_interval] = True

    return event_interval


def extract_photometry_data_from_array(phot_data, chan_num=0):

    data = phot_data['data'][:, chan_num]

    return data


def get_photometry_signal(phot_data, phot_channel=0):
    '''

    :param phot_data: dictionary that includes 'data', which is a list containing multiple data acquisition lines
        also has a key 'AI_line_desc' which should be a list of strings saying what was recorded on each line
    :return:
    '''

    if 'AI_line_desc' in phot_data.keys():
        if 'photometry signal' in phot_data['AI_line_desc']:
            phot_channel = phot_data['AI_line_desc'].index('photometry signal')
        elif 'photometry_signal' in phot_data['AI_line_desc']:
            phot_channel = phot_data['AI_line_desc'].index('photometry_signal')

    data = extract_photometry_data_from_array(phot_data, chan_num=phot_channel)

    return data

def photodetrend(phot_data, phot_channel=0):
    '''

    :param phot_data:
    :param phot_channel:
    :return: photo_detrend2 - raw data minus double exponential fit
    '''
    # assumes photometry signal was recorded on the first channel if not otherwise documented
    # if 'AI_line_desc' in phot_data.keys():
    #     if 'photometry signal' in phot_data['AI_line_desc']:
    #         phot_channel = phot_data['AI_line_desc'].index('photometry signal')
    #     elif 'photometry_signal' in phot_data['AI_line_desc']:
    #         phot_channel = phot_data['AI_line_desc'].index('photometry_signal')
    #
    # data = extract_photometry_data_from_array(phot_data, chan_num=phot_channel)

    data = get_photometry_signal(phot_data, phot_channel=phot_channel)
    num_samples = len(phot_data['t'])

    # fit double exponential; if it throws an error, fit single exponential with offset
    popt, exp2_fit_successful = fit_exponentials(phot_data)
    if popt is None:
        return None, None

    # subtract out the double exponential fit, then add back the asymptote (last value of the fit)
    # would it be better to add back the mean (which Christian does) than the asymptote?

    if exp2_fit_successful:
        exp_fit, dc_offset = calc_exp2_fit(num_samples, popt)
    else:
        exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, popt)

    photo_detrend2 = (data - exp_fit) + exp_fit[-1]

    return photo_detrend2, popt, exp2_fit_successful


def photodetrend_intervals2(phot_metadata, data):
    '''

    :param phot_data:
    :param phot_channel:
    :return: photo_detrend2 - raw data minus double exponential fit
    '''

    intervals = phot_metadata['analysis_intervals']
    num_intervals = len(intervals)
    num_samples = len(data)
    interval_exp2_fit_success = []
    interval_popt = []
    for interval in intervals:
        # fit double exponential; if it throws an error, fit single exponential with offset
        popt, exp2_fit_successful = fit_interval_exponentials(data, interval)

        interval_popt.append(popt)
        interval_exp2_fit_success.append(exp2_fit_successful)

    exp_fit = np.zeros(num_samples)
    dc_offsets = np.zeros(num_intervals)
    photo_detrend2 = np.zeros(num_samples)
    for i_interval, interval in enumerate(intervals):
        if interval_exp2_fit_success[i_interval]:
            interval_fit, dc_offsets[i_interval] = calc_interval_exp2_fit(interval, interval_popt[i_interval])
        else:
            interval_fit, dc_offsets[i_interval] = calc_interval_exp1_offset_fit(interval, interval_popt[i_interval])
        exp_fit[interval[0]:interval[1]] = interval_fit

        photo_detrend2[interval[0]:interval[1]] = (data[interval[0]:interval[1]] - interval_fit) + interval_fit[-1]
        # not sure if the last element of interval_fit is the right thing to add back (maybe dc_offset?)
        # definitely not the last exp fit from the entire signal, in case there was a slip during the recording

    # fit double exponential; if it throws an error, fit single exponential with offset
    # popt, exp2_fit_successful = fit_exponentials(phot_data)
    # if popt is None:
    #     return None, None

    # subtract out the double exponential fit, then add back the asymptote (last value of the fit)
    # would it be better to add back the mean (which Christian does) than the asymptote?

    # if exp2_fit_successful:
    #     exp_fit, dc_offset = calc_exp2_fit(num_samples, popt)
    # else:
    #     exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, popt)
    #
    # photo_detrend2 = (data - exp_fit) + exp_fit[-1]

    return photo_detrend2, exp_fit, interval_popt, interval_exp2_fit_success


def photodetrend_intervals(phot_data, phot_channel=0):
    '''

    :param phot_data:
    :param phot_channel:
    :return: photo_detrend2 - raw data minus double exponential fit
    '''

    data = get_photometry_signal(phot_data, phot_channel=phot_channel)
    intervals = phot_data['analysis_intervals']
    num_intervals = len(intervals)
    num_samples = len(phot_data['t'])
    interval_exp2_fit_success = []
    interval_popt = []
    for interval in intervals:
        # fit double exponential; if it throws an error, fit single exponential with offset
        popt, exp2_fit_successful = fit_interval_exponentials(data, interval)

        interval_popt.append(popt)
        interval_exp2_fit_success.append(exp2_fit_successful)

    exp_fit = np.zeros(num_samples)
    dc_offsets = np.zeros(num_intervals)
    photo_detrend2 = np.zeros(num_samples)
    for i_interval, interval in enumerate(intervals):
        if interval_exp2_fit_success[i_interval]:
            interval_fit, dc_offsets[i_interval] = calc_interval_exp2_fit(interval, interval_popt[i_interval])
        else:
            interval_fit, dc_offsets[i_interval] = calc_interval_exp1_offset_fit(interval, interval_popt[i_interval])
        exp_fit[interval[0]:interval[1]] = interval_fit

        photo_detrend2[interval[0]:interval[1]] = (data[interval[0]:interval[1]] - interval_fit) + interval_fit[-1]
        # not sure if the last element of interval_fit is the right thing to add back (maybe dc_offset?)
        # definitely not the last exp fit from the entire signal, in case there was a slip during the recording

    # fit double exponential; if it throws an error, fit single exponential with offset
    # popt, exp2_fit_successful = fit_exponentials(phot_data)
    # if popt is None:
    #     return None, None

    # subtract out the double exponential fit, then add back the asymptote (last value of the fit)
    # would it be better to add back the mean (which Christian does) than the asymptote?

    # if exp2_fit_successful:
    #     exp_fit, dc_offset = calc_exp2_fit(num_samples, popt)
    # else:
    #     exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, popt)
    #
    # photo_detrend2 = (data - exp_fit) + exp_fit[-1]

    return photo_detrend2, exp_fit, interval_popt, interval_exp2_fit_success


def photodetrend_legacy(phot_data, phot_channel=0):
    '''
    I think this is wrong - the new version more closely mirrors Christian's algorithm
    :param phot_data:
    :param phot_channel:
    :return: photo_detrend2 - raw data minus double exponential fit
    '''
    # assumes photometry signal was recorded on the first channel
    if 'AI_line_desc' in phot_data.keys():
        if 'photometry signal' in phot_data['AI_line_desc']:
            phot_channel = phot_data['AI_line_desc'].index('photometry signal')
        elif 'photometry_signal' in phot_data['AI_line_desc']:
            phot_channel = phot_data['AI_line_desc'].index('photometry_signal')

    data = extract_photometry_data_from_array(phot_data, chan_num=phot_channel)
    num_samples = len(phot_data['t'])

    # left over from when I tried directly fitting the offset
    # popt_offset, popt2 = fit_exponentials(phot_data)

    popt, exp2_fit_successful = fit_exponentials(phot_data)

    if popt is None:
        return None, None

    # subtract out the double exponential fit, then add back the asymptote (by just taking the last element of the calculated exponential fit)
    if exp2_fit_successful:
        exp_fit, dc_offset = calc_exp2_fit(num_samples, popt)
    else:
        exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, popt)

    photo_detrend2 = (data - exp_fit) + exp_fit[-1]

    return photo_detrend2, popt, dc_offset, exp2_fit_successful


def fit_exponentials(phot_data, phot_channel=0):

    # assumes photometry signal was recorded on the first channel
    data = extract_photometry_data_from_array(phot_data, chan_num=phot_channel)
    exp2_fit_successful = True

    t = phot_data['t']
    num_samples = len(t)

    t_ints = list(range(num_samples))

    p0_exp2 = (0.05, -0.00001, 1., -0.00001)
    try:
        popt2, pcov2 = curve_fit(exp2_func, t_ints, data, p0_exp2)
    except RuntimeError:
        print('could not fit double exponential, trying single exponential')
        p0_exp1_offset = (0.05, -0.00001, 1.)
        try:
            popt1_offset, pcov1_offset = curve_fit(exp1_offset_func, t_ints, data, p0_exp1_offset, maxfev=5000)
        except:
            pass
        exp2_fit_successful = False

    if exp2_fit_successful:
        print('succesfully fit double exponential!')
        popt = popt2
    else:
        print('double exponential could not be fit. fit offset single exponential instead')
        popt = popt1_offset

    return popt, exp2_fit_successful


def fit_interval_exponentials(data, interval):

    t_ints = list(range(interval[0], interval[1]))

    ii = 1
    exp2_fit_successful = False
    exp1_fit_successful = False
    while ii < 10:
        p0_exp2 = np.array([0.5, -0.000001 * 10**(-ii), 1., -0.000001 * 10**(-ii)])
        try:
            popt2, pcov2 = curve_fit(exp2_func, t_ints, data[interval[0]:interval[1]], p0_exp2)
            exp2_fit_successful = True
            break
        except RuntimeError:
            ii = ii + 1
            print(ii)
            # # try different starting condition
            # p0_exp2 = (-0.05, 0.00001, -1., 0.00001)
            # try:
            #     popt2, pcov2 = curve_fit(exp2_func, t_ints, data[interval[0]:interval[1]], p0_exp2)
            # except RuntimeError:
            #     print('could not fit double exponential, trying single exponential')
            #     p0_exp1_offset = (0.05, -0.00001, 1.)
    if not exp2_fit_successful:
        try:
            print('could not fit double exponential, trying single exponential')
            p0_exp1_offset = (0.05, -0.00001, 1.)
            popt1_offset, pcov1_offset = curve_fit(exp1_offset_func, t_ints, data[interval[0]:interval[1]], p0_exp1_offset, maxfev=5000)
            exp1_fit_successful = True
        except:
            pass

    if exp2_fit_successful:
        print('succesfully fit double exponential!')
        popt = popt2
    elif exp1_fit_successful:
        print('double exponential could not be fit. fit offset single exponential instead')
        try:
            popt = popt1_offset
        except:
            pass
    else:
        pass

    return popt, exp2_fit_successful


def calc_dff(phot_data, baseline_data, smooth_window=201, f0_pctile=10, expected_baseline_diff=0.2):

    detrended_data, popt, exp2_fit_successful = photodetrend(phot_data)

    if detrended_data is None:
        return None

    # the baseline adjustment should be the background photoreceiver recording (i.e., with tip in black cloth) at the
    # LED current setting used for the recording
    baseline_adjustment = baseline_data['mean_baseline']

    baseline_adjustment2 = np.mean(detrended_data) - expected_baseline_diff

    # another option for the baseline is to use the mean signal minus some constant, which is probably more
    # reasonable since the implanted fiber itself has a lot of autofluorescence (which should probably be
    # measured and documented pre-implantation
    # OR
    # can assume a constant for the implanted fiber autofluorescence above baseline

    # smoothed_data = smooth(detrended_data, smooth_window) - baseline_adjustment
    smoothed_data2 = smooth(detrended_data, smooth_window) - baseline_adjustment2


    # find 10th percentile
    f0 = np.percentile(smoothed_data2, f0_pctile)

    dff = (smoothed_data2 - f0) / f0


    # for now, return smoothed_data2
    return smoothed_data2, detrended_data, dff, popt, exp2_fit_successful


def calc_segmented_signal_dff(phot_metadata, signal, baseline_data, smooth_window=201, f0_pctile=10, expected_baseline=0.2):

    # phot_signal = get_photometry_signal(phot_data, phot_channel=0)
    num_samples = len(signal)
    if 'analysis_intervals' not in phot_metadata.keys():
        phot_metadata['analysis_intervals'] = [[0, num_samples]]

    detrended_data, exp_fit, interval_popt, interval_exp2_fit_success = photodetrend_intervals2(phot_metadata, signal)

    if detrended_data is None:
        return None

    # the baseline adjustment should be the background photoreceiver recording (i.e., with tip in black cloth) at the
    # LED current setting used for the recording
    # baseline_adjustment = baseline_data['mean_baseline']

    baseline_adjustment2 = np.mean(detrended_data) - expected_baseline
    baseline_used = expected_baseline

    # another option for the baseline is to use the mean signal minus some constant, which is probably more
    # reasonable since the implanted fiber itself has a lot of autofluorescence (which should probably be
    # measured and documented pre-implantation
    # OR
    # can assume a constant for the implanted fiber autofluorescence above baseline

    # smoothed_data = smooth(detrended_data, smooth_window) - baseline_adjustment
    smoothed_data2 = smooth(detrended_data, smooth_window) - baseline_adjustment2

    # find f0th percentile
    # only calculate on valid intervals; concatenate valid intervals together
    intervals = phot_metadata['analysis_intervals']
    num_intervals = len(intervals)
    valid_smoothed_data = smoothed_data2[intervals[0][0]:intervals[0][1]]

    if num_intervals > 1:
        for interval in intervals[1:]:
            valid_smoothed_data = np.hstack((valid_smoothed_data, smoothed_data2[interval[0]:interval[1]]))

    f0 = np.percentile(valid_smoothed_data, f0_pctile)

    dff = np.zeros(num_samples)
    for interval in intervals:
        dff[interval[0]:interval[1]] = (smoothed_data2[interval[0]:interval[1]] - f0) / f0

    return smoothed_data2, detrended_data, dff, interval_popt, interval_exp2_fit_success, baseline_used


def calc_segmented_dff(phot_data, baseline_data, smooth_window=201, f0_pctile=10, expected_baseline=0.2):

    phot_signal = get_photometry_signal(phot_data, phot_channel=0)
    num_samples = len(phot_signal)
    if 'analysis_intervals' not in phot_data.keys():
        phot_data['analysis_intervals'] = [[0, num_samples]]

    detrended_data, exp_fit, interval_popt, interval_exp2_fit_success = photodetrend_intervals(phot_data)

    if detrended_data is None:
        return None

    # the baseline adjustment should be the background photoreceiver recording (i.e., with tip in black cloth) at the
    # LED current setting used for the recording
    # baseline_adjustment = baseline_data['mean_baseline']

    baseline_adjustment2 = np.mean(detrended_data) - expected_baseline
    baseline_used = expected_baseline

    # another option for the baseline is to use the mean signal minus some constant, which is probably more
    # reasonable since the implanted fiber itself has a lot of autofluorescence (which should probably be
    # measured and documented pre-implantation
    # OR
    # can assume a constant for the implanted fiber autofluorescence above baseline

    # smoothed_data = smooth(detrended_data, smooth_window) - baseline_adjustment
    smoothed_data2 = smooth(detrended_data, smooth_window) - baseline_adjustment2

    # find f0th percentile
    # only calculate on valid intervals; concatenate valid intervals together
    intervals = phot_data['analysis_intervals']
    num_intervals = len(intervals)
    valid_smoothed_data = smoothed_data2[intervals[0][0]:intervals[0][1]]

    if num_intervals > 1:
        for interval in intervals[1:]:
            valid_smoothed_data = np.hstack((valid_smoothed_data, smoothed_data2[interval[0]:interval[1]]))

    f0 = np.percentile(valid_smoothed_data, f0_pctile)

    dff = np.zeros(num_samples)
    for interval in intervals:
        dff[interval[0]:interval[1]] = (smoothed_data2[interval[0]:interval[1]] - f0) / f0

    return smoothed_data2, detrended_data, dff, interval_popt, interval_exp2_fit_success, baseline_used


def calc_local_dff(phot_data, smooth_window=101, f0_pctile=50):

    detrended_data, popt, exp2_fit_successful = photodetrend(phot_data)

    if detrended_data is None:
        return None

    # does it make sense to use the dc offset as the baseline adjustment? does it really matter?
    baseline_adjustment = 0.0
    # I don't think the baseline_adjustment is needed if the fit is good; just setting to 0.0 in case need to add it back later
    try:
        smoothed_data = smooth(detrended_data, smooth_window) - baseline_adjustment
    except:
        pass

    # find 10th percentile
    f0 = np.percentile(smoothed_data, f0_pctile)

    dff = (smoothed_data - f0) / f0

    return detrended_data, dc_offset, dff, popt, exp2_fit_successful


def smooth(data, span):

    # data: NumPy 1-D array containing the data to be smoothed
    # span: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(data, np.ones(span, dtype=int), 'valid') / span
    r = np.arange(1, span - 1, 2)
    start = np.cumsum(data[:span - 1])[::2] / r
    stop = (np.cumsum(data[:-span:-1])[::2] / r)[::-1]

    return np.concatenate((start, out0, stop))


def get_photometry_events(phot_data, session_metadata):

    if session_metadata['task'].lower() == 'pavlovian':
        eventlist = ['tone', 'pellet']
        if len(phot_data['AI_line_desc']) > 0:
            FED_chan_num = phot_data['AI_line_desc'].index('FED3')
        else:
            FED_chan_num = 1
        event_ts = extract_tone_pellet_FED_ts(phot_data, chan_num=FED_chan_num)
        event_ts = list(event_ts)

    elif session_metadata['task'].lower() in ['sr', 'srchrim', 'srchrimpost']:
        eventlist, event_ts = extract_sr_ts_from_phot_data(phot_data, session_metadata)

    elif session_metadata['task'].lower() == 'chrimsontest':   # may be different versions of the crimson openfield
        eventlist, event_ts, _ = extract_chrimson_events(phot_data, session_metadata)

    # make sure each event_ts is an array (not a scalar, needed for indexing later)
    for i_array, ts_array in enumerate(event_ts):
        if type(ts_array) is np.float64:
            event_ts[i_array] = np.array([ts_array])

    return eventlist, event_ts


def get_photometry_events_from_tsdict(phot_data, ts_dict, session_metadata):

    if session_metadata['task'].lower() == 'pavlovian':
        eventlist = ['tone', 'pellet']
        if len(phot_data['AI_line_desc']) > 0:
            FED_chan_num = phot_data['AI_line_desc'].index('FED3')
        else:
            FED_chan_num = 1
        event_ts = extract_tone_pellet_FED_ts(phot_data, chan_num=FED_chan_num)
        event_ts = list(event_ts)

    elif session_metadata['task'].lower() in ['sr', 'srchrim', 'srchrimpost']:
        eventlist, event_ts = extract_sr_ts_from_phot_data(phot_data, session_metadata)

    elif session_metadata['task'].lower() == 'chrimsontest':   # may be different versions of the crimson openfield
        eventlist, event_ts, _ = extract_chrimson_events(phot_data, session_metadata)

    # make sure each event_ts is an array (not a scalar, needed for indexing later)
    for i_array, ts_array in enumerate(event_ts):
        if type(ts_array) is np.float64:
            event_ts[i_array] = np.array([ts_array])

    return eventlist, event_ts


def identify_TTL_pulses_and_bursts(TTL_data, Fs, TTL_thresh=1., min_pulse_duration=2, inter_burst_thresh=1000):
    '''

    :param TTL_data: numpy array with data stream for TTL pulses - this should be raw voltage data
    :param Fs: data sampling rate in Hz
    :param TTL_thresh: voltage threshold for identifying a TTL pulse
    :param min_pulse_duration: minimum number of ticks before can be considered a separate pulse. sometimes it takes 2
        or so ticks to get from zero to 5 V in the TTL
    :param inter_burst_thresh: minimum number of samples between "on" events to consider a grouping of TTL pulses a
        separate "burst"
    :return:
    '''

    TTL_on = np.diff(TTL_data, prepend=False) > TTL_thresh
    TTL_off = np.diff(TTL_data, prepend=False) < -TTL_thresh

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(TTL_data)
    # plt.show()

    pulse_on_idx = np.squeeze(np.argwhere(TTL_on))
    # eliminate "pulse on's" that are just a continuation of the pulse rise
    pulse_on_intervals = np.diff(pulse_on_idx)
    longrise_idx = np.squeeze(np.argwhere(pulse_on_intervals < min_pulse_duration))
    pulse_on_idx = np.delete(pulse_on_idx, longrise_idx + 1)

    pulse_off_idx = np.squeeze(np.argwhere(TTL_off))
    pulse_off_intervals = np.diff(pulse_off_idx)
    longfall_idx = np.squeeze(np.argwhere(pulse_off_intervals < min_pulse_duration))
    pulse_off_idx = np.delete(pulse_off_idx, longfall_idx + 1)

    # makes sure pulse_off_idx starts after pulse_on_idx
    aligned_pulse_off_idx = align_on_off_idx(pulse_on_idx, pulse_off_idx)

    # make sure there is a pulse_off_idx after the last pulse_on_idx
    valid_pulse_on_idx = find_last_valid_pulse_on_idx(pulse_on_idx, pulse_off_idx)

    burst_on_idx, burst_off_idx, intraburst_freq = detect_bursts(valid_pulse_on_idx, aligned_pulse_off_idx, Fs, inter_burst_thresh=inter_burst_thresh)

    # verify correct pulse/burst identification
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(TTL_data, color='k')
    # ones_array = np.ones((len(burst_on_idx)))
    # ax.plot(burst_on_idx, ones_array, color='g', linestyle='none', marker='*')
    # ax.plot(burst_off_idx, ones_array, color='r', linestyle='none', marker='*')
    # ax.legend()
    # plt.show()

    return burst_on_idx, burst_off_idx, valid_pulse_on_idx, aligned_pulse_off_idx, intraburst_freq


def detect_bursts(pulse_on_idx, aligned_pulse_off_idx, Fs, inter_burst_thresh=1000):
    '''

    :param pulse_on_idx: sample indices in the original TTL data stream of TTL on events
    :param aligned_pulse_off_idx: sample indices in the original TTL data stream of TTL off events, but starting after
        the first "on" event
    :param inter_burst_thresh: minimum number of samples between "on" events to consider a grouping of TTL pulses a
        separate "burst"
    :return burst_on_idx: sample indices in the original TTL stream of TTL burst onsets
    :return burst_off_idx: sample indices in the original TTL stream of TTL burst offsets
    '''

    # find intervals between TTL on times in numbers of samples
    inter_on_intervals = np.diff(pulse_on_idx)

    # find pulses with inter-pulse interval greater than inter_burst_thresh. Add one because difference between the
    # first two "pulse on" indices will have index 0
    inter_burst_idx = np.squeeze(np.argwhere(inter_on_intervals > inter_burst_thresh)) + 1

    # find the index in the original time series of burst onset events
    burst_on_idx = pulse_on_idx[inter_burst_idx]

    # find the index of the burst offset; should be the last "off" event before the start of the next burst
    burst_off_idx = aligned_pulse_off_idx[inter_burst_idx - 1]

    # insert the index of the first "pulse on" event at the beginning of the burst_on_idx array
    burst_on_idx = np.insert(burst_on_idx, 0, pulse_on_idx[0])

    # insert the index of the last "pulse off" event at the end of the burst_off_idx array
    burst_off_idx = np.append(burst_off_idx, aligned_pulse_off_idx[-1])

    intraburst_freq = np.zeros(len(burst_on_idx))
    for i_burst, burst_start_idx in enumerate(burst_on_idx):

        burst_pulse_flags = np.logical_and(pulse_on_idx >= burst_start_idx, pulse_on_idx < burst_off_idx[i_burst])
        burst_pulse_indices = pulse_on_idx[burst_pulse_flags]

        if len(burst_pulse_indices) <= 1:
            continue

        pulse_idx_diffs = np.diff(burst_pulse_indices)
        interpulse_intervals = pulse_idx_diffs / Fs

        intraburst_freq[i_burst] = round(1 / np.median(interpulse_intervals))

    return burst_on_idx, burst_off_idx, intraburst_freq


def sort_bursts_by_frequency(burst_on_idx, burst_off_idx, intraburst_freq):

    unique_freqs = np.unique(intraburst_freq)

    freq_specific_burst_on_ts = []
    freq_specific_burst_off_ts = []
    for freq in unique_freqs:
        freq_specific_burst_on_ts.append(burst_on_idx[intraburst_freq == freq])
        freq_specific_burst_off_ts.append(burst_off_idx[intraburst_freq == freq])

    frequency_specific_bursts = {'intraburst_frequency': unique_freqs,
                                 'freq_specific_burst_on_ts': freq_specific_burst_on_ts,
                                 'freq_specific_burst_off_ts': freq_specific_burst_off_ts
                                 }

    return frequency_specific_bursts


def extract_chrimson_events_digital(ts_dict, min_intraburst_freq=2):

    LED_ts = ts_dict['LED_trigger']

    burst_on_ts, burst_off_ts, pulse_on_idx, aligned_pulse_off_idx, intraburst_freq = bursts_from_pulse_ts(LED_ts)

    frequency_specific_bursts = sort_bursts_by_frequency(burst_on_ts, burst_off_ts, intraburst_freq)

    # from inside sort_bursts_by_frequency function
    # frequency_specific_bursts = {'intraburst_frequency': unique_freqs,
    #                              'freq_specific_burst_on_ts': freq_specific_burst_on_ts,
    #                              'freq_specific_burst_off_ts': freq_specific_burst_off_ts
    #                              }

    eventlist = ['start_{:03d}_Hz'.format(int(bf)) for bf in frequency_specific_bursts['intraburst_frequency'] if bf > min_intraburst_freq]
    for bf in frequency_specific_bursts['intraburst_frequency']:
        if bf > min_intraburst_freq:
            eventlist.append('end_{:03d}_Hz'.format(int(bf)))
    event_ts = []
    for i_event, ts_on in enumerate(frequency_specific_bursts['freq_specific_burst_on_ts']):
        if frequency_specific_bursts['intraburst_frequency'][i_event] > min_intraburst_freq:
            event_ts.append(ts_on)
    for i_event, ts_off in enumerate(frequency_specific_bursts['freq_specific_burst_off_ts']):
        if frequency_specific_bursts['intraburst_frequency'][i_event] > min_intraburst_freq:
            event_ts.append(ts_off)

    return eventlist, event_ts, frequency_specific_bursts


def bursts_from_pulse_ts(LED_ts, inter_burst_thresh=5):
    '''

    :param LED_ts:
    :param inter_burst_thresh: minimum time between bursts in seconds
    :return:
    '''

    pulse_on_ts = LED_ts[:, 0]
    pulse_off_ts = LED_ts[:, 1]

    # makes sure pulse_off_idx starts after pulse_on_idx
    aligned_pulse_off_ts = align_on_off_idx(pulse_on_ts, pulse_off_ts)

    # make sure there is a pulse_off_idx after the last pulse_on_idx
    valid_pulse_on_ts = find_last_valid_pulse_on_idx(pulse_on_ts, pulse_off_ts)

    # set sampling rate to 1 since TTL events have already been converted to timestamps from tick counts
    effective_Fs = 1
    burst_on_idx, burst_off_idx, intraburst_freq = detect_bursts(valid_pulse_on_ts, aligned_pulse_off_ts, effective_Fs, inter_burst_thresh=inter_burst_thresh)

    # verify correct pulse/burst identification
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(TTL_data, color='k')
    # ones_array = np.ones((len(burst_on_idx)))
    # ax.plot(burst_on_idx, ones_array, color='g', linestyle='none', marker='*')
    # ax.plot(burst_off_idx, ones_array, color='r', linestyle='none', marker='*')
    # ax.legend()
    # plt.show()

    return burst_on_idx, burst_off_idx, valid_pulse_on_ts, aligned_pulse_off_ts, intraburst_freq



    pass
def extract_chrimson_events(phot_data, session_metadata):

    if len(phot_data['AI_line_desc']) > 0:
        rlight_chan_num = phot_data['AI_line_desc'].index('LED_trigger')
        try:
            phot_signal_idx = phot_data['AI_line_desc'].index('photometry_signal')
        except:
            phot_signal_idx = phot_data['AI_line_desc'].index('photometry signal')

    elif np.shape(phot_data['data'])[1] < 8:
        rlight_chan_num = 1
        phot_signal_idx = 0
    elif session_metadata['ratID'] == 'R0432' and session_metadata['date'] == datetime(2022, 4, 13, 0, 0):
        rlight_chan_num = 6    # I think just this one session where red light trigger was in position 7 (6 counting from 0)
        phot_signal_idx = 0
    elif session_metadata['rat_num'] < 440:
        rlight_chan_num = 1
        phot_signal_idx = 0
    else:
        rlight_chan_num = 7
        phot_signal_idx = 0

    TTL_data = extract_photometry_data_from_array(phot_data, chan_num=rlight_chan_num)
    burst_on_idx, burst_off_idx, pulse_on_idx, aligned_pulse_off_idx, intraburst_freq = identify_TTL_pulses_and_bursts(
        TTL_data, phot_data['Fs'])

    burst_on_ts = burst_on_idx / phot_data['Fs']
    burst_off_ts = burst_off_idx / phot_data['Fs']
    frequency_specific_bursts = sort_bursts_by_frequency(burst_on_ts, burst_off_ts, intraburst_freq)

    # from inside sort_bursts_by_frequency function
    # frequency_specific_bursts = {'intraburst_frequency': unique_freqs,
    #                              'freq_specific_burst_on_ts': freq_specific_burst_on_ts,
    #                              'freq_specific_burst_off_ts': freq_specific_burst_off_ts
    #                              }
    eventlist = ['start_{:03d}_Hz'.format(int(bf)) for bf in frequency_specific_bursts['intraburst_frequency']]
    for bf in frequency_specific_bursts['intraburst_frequency']:
        eventlist.append('end_{:03d}_Hz'.format(int(bf)))
    event_ts = [ts_on for ts_on in frequency_specific_bursts['freq_specific_burst_on_ts']]
    for ts_off in frequency_specific_bursts['freq_specific_burst_off_ts']:
        event_ts.append(ts_off)

    return eventlist, event_ts, frequency_specific_bursts


def find_last_valid_pulse_on_idx(pulse_on_idx, pulse_off_idx):
    '''
    make sure that each pulse_on_idx has a pulse_off_idx after it. Sometimes at the end of a file there isn't another pulse off
    :param pulse_on_idx:
    :param pulse_off_idx:
    :return:
    '''

    if np.ndim(pulse_on_idx) == 0:
        pulse_on_idx = np.reshape(pulse_on_idx, (-1))
    if np.ndim(pulse_off_idx) == 0:
        pulse_off_idx = np.reshape(pulse_off_idx, (-1))

    if len(pulse_on_idx) == 0 and len(pulse_off_idx) == 0:
        return pulse_on_idx
    if len(pulse_on_idx) == 0 and pulse_off_idx[0] is None:
        # this only happens if there are no pulses at all (I think)
        return pulse_on_idx

    # with new digital lines version, any empty timestamps will have zero in that spot in the vector
    while pulse_off_idx[-1] == 0:
        pulse_off_idx = pulse_off_idx[:-1]
        pulse_on_idx = pulse_on_idx[:-1]

    while pulse_off_idx[-1] <= pulse_on_idx[-1]:
        # the last pulse_off event occurred before the last pulse_on event; remove the last pulse_on event
        pulse_on_idx = pulse_on_idx[:-1]
        pulse_off_idx = pulse_off_idx[:-1]

    return pulse_on_idx


def extract_tone_pellet_FED_ts(phot_data, chan_num=1, TTL_thresh=1.5):
    '''

    :param phot_data:
    :param chan_num:
    :param TTL_thresh:
    :return:
    '''

    TTL_data = extract_photometry_data_from_array(phot_data, chan_num)

    TTL_on = np.diff(TTL_data, prepend=False) > TTL_thresh
    TTL_off = np.diff(TTL_data, prepend=False) < -TTL_thresh

    pulse_on_idx = np.squeeze(np.argwhere(TTL_on))
    pulse_off_idx = np.squeeze(np.argwhere(TTL_off))

    # makes sure pulse_off_idx starts after pulse_on_idx
    aligned_pulse_off_idx = align_on_off_idx(pulse_on_idx, pulse_off_idx)

    # make sure there is a pulse_off_idx after the last pulse_on_idx
    valid_pulse_on_idx = find_last_valid_pulse_on_idx(pulse_on_idx, aligned_pulse_off_idx)

    if np.size(aligned_pulse_off_idx) == np.size(valid_pulse_on_idx):
        pw = aligned_pulse_off_idx - valid_pulse_on_idx
        if not isinstance(pw, np.ndarray):
            pw = np.array(pw)
        if np.ndim(pw) == 0:
            pw = np.reshape(pw, 1)   # need to do this so min(pw) doesn't throw an error if pw is not iterable

        if min(pw) < 150:
            # in earlier sessions, I think tone was a 144 ms pulse, pellet = 200; in later sessions I think tone
            # was 200 ms, pellet = 250
            pw_cutoff = 175
        else:
            pw_cutoff = 225

        # test if pulse_on_idx has at least two elements
        if np.size(valid_pulse_on_idx) > 1:
            pulse_off_after_first_on = np.squeeze(np.argwhere(aligned_pulse_off_idx > valid_pulse_on_idx[0]))
        elif np.size(valid_pulse_on_idx) == 1:
            pulse_off_after_first_on = np.squeeze(np.argwhere(aligned_pulse_off_idx > valid_pulse_on_idx))
        else:
            # pulse_on_idx is empty
            return None

        if np.size(valid_pulse_on_idx) > 1:
            tone_event_idx = valid_pulse_on_idx[pw < pw_cutoff]
            pellet_event_idx = valid_pulse_on_idx[pw > pw_cutoff]
        elif np.size(valid_pulse_on_idx) == 1:
            if pw[0] < pw_cutoff:
                tone_event_idx = np.array(valid_pulse_on_idx)
                pellet_event_idx = None
            else:
                pellet_event_idx = np.array(valid_pulse_on_idx)
                tone_event_idx = None

        if tone_event_idx is None:
            tone_event_ts = None
        else:
            tone_event_ts = tone_event_idx / phot_data['Fs']
        if pellet_event_idx is None:
            pellet_event_ts = None
        else:
            pellet_event_ts = pellet_event_idx / phot_data['Fs']
    else:
        tone_event_ts = None
        pellet_event_ts = None

    return tone_event_ts, pellet_event_ts


def extract_sr_ts_from_phot_data(phot_data, session_metadata, TTL_thresh_a=1., min_pw=5):
    '''

    :param phot_data:
    :return:
    '''
    '''
    AI lines on NIDAQ
    AI0 - photometry signal
    AI1 - FED (IR back if before 4/27/2022)
    AI2 - paw through slot
    AI3 - actuator 3
    AI4 - actuator 2
    AI5 - IR back
    AI6 - video trigger (not early reach)
    AI7 - frame trigger
    '''
    eventlist = ['paw_through_slot', 'Actuator3', 'Actuator2', 'IR_back', 'vid_trigger', 'frame_trigger']
    # eventlist = ['rear_photobeam', 'paw_through_slot', 'actuator3', 'actuator2', 'vid_trigger']
    num_events = len(eventlist)
    event_ts = []
    for i_event, eventname in enumerate(eventlist):
        if eventname == 'Actuator3':
            # workaround - when the Arduino isn't plugged in, I think it's draining current from Actuator3, which is plugged in to trigger the red light
            TTL_thresh = 0.4
        else:
            TTL_thresh = TTL_thresh_a

        nidaq_chan = map_event_to_nidaq_channel(eventname, phot_data, session_metadata)
        if nidaq_chan is None:
            event_ts.append(np.zeros(1))
            continue

        TTL_data = extract_photometry_data_from_array(phot_data, nidaq_chan)

        TTL_on = np.diff(TTL_data, prepend=False) > TTL_thresh
        TTL_off = np.diff(TTL_data, prepend=False) < -TTL_thresh

        if eventname == 'rear_photobeam':
            # exclude photobeam events too close together in time
            pass

        pulse_on_idx = np.squeeze(np.argwhere(TTL_on))
        pulse_off_idx = np.squeeze(np.argwhere(TTL_off))

        # make sure there's at least min_pw ticks between on and off (occasionally there's a brief line pop that's meaningless)
        if np.ndim(pulse_on_idx) == 0:
            # just one pulse_on event
            next_pulseoffs = pulse_off_idx[pulse_off_idx > pulse_on_idx]
            if len(next_pulseoffs) > 0:
                next_pulseoff_idx = next_pulseoffs[0]
            else:
                next_pulseoff_idx = len(TTL_data)

            if next_pulseoff_idx - pulse_on_idx < min_pw:
                final_pulse_on_idx = np.empty(0)
            else:
                final_pulse_on_idx = pulse_on_idx
        else:
            pulseon_to_remove = []
            try:
                for i_pulse, individual_pulseon in enumerate(pulse_on_idx):
                    next_pulseoff_idx = pulse_off_idx[pulse_off_idx > individual_pulseon][0]
                    if next_pulseoff_idx - individual_pulseon < min_pw:
                        # pulse is too short, remove it
                        pulseon_to_remove.append(i_pulse)
            except:
                pass

            final_pulse_on_idx = np.delete(pulse_on_idx, pulseon_to_remove)

        event_ts.append(final_pulse_on_idx / phot_data['Fs'])

    return eventlist, event_ts


def map_event_to_nidaq_channel(eventname, phot_data, session_metadata):
    '''
    AI lines on NIDAQ
    AI0 - photometry signal
    AI1 - IR back
    AI2 - paw through slot
    AI3 - actuator 3
    AI4 - actuator 2
    AI5 - IR back (nose trigger if before 4/27/2022)
    AI6 - video trigger (not early reach)
    AI7 - frame trigger
    '''


    if len(phot_data['AI_line_desc']) > 0:
        try:
            nidaq_chan = phot_data['AI_line_desc'].index(eventname)
        except ValueError:
            nidaq_chan = None
    else:
        # if line ID's weren't written into data file, hardcode the lines in (pre-2023)
        if eventname == 'IR_back':
            if session_metadata['date'] < datetime(2022, 4, 27):
                nidaq_chan = 1
            else:
                nidaq_chan = 5
        elif eventname == 'paw_through_slot':
            nidaq_chan = 2
        elif eventname.lower() == 'actuator3':
            nidaq_chan = 3
        elif eventname.lower() == 'actuator2':
            nidaq_chan = 4
        elif eventname.lower() == 'nose_trigger':   # nose_trigger not in use right now
            nidaq_chan = 5
        elif eventname.lower() == 'vid_trigger':
            nidaq_chan = 6
        elif eventname.lower() == 'frame_trigger':
            nidaq_chan = 7
        else:
            nidaq_chan = None

    return nidaq_chan


def process_session_post_20230904(data_files, session_metadata, parent_directories, smooth_window=101, f0_pctile=10):
    # full_processed_pickle_name = navigation_utilities.processed_data_pickle_name(session_metadata, parent_directories)
    pickled_metadata_fname = navigation_utilities.get_pickled_metadata_fname(session_metadata, parent_directories)
    pickled_analog_processeddata_fname = navigation_utilities.get_pickled_analog_processeddata_fname(session_metadata, parent_directories)
    pickled_ts_fname = navigation_utilities.get_pickled_ts_fname(session_metadata, parent_directories)
    # find the scores file and load it
    if os.path.exists(pickled_metadata_fname):
        _, pickle_name = os.path.split(pickled_metadata_fname)
        print('{} already processed'.format(pickle_name))
        phot_metadata = skilled_reaching_io.read_pickle(pickled_metadata_fname)
        # pa_plots.identify_data_segments(get_photometry_signal(processed_phot_data))
        # intervals_to_analyze, intervals_to_ignore = pa_plots.select_data_segments_by_span(get_photometry_signal(processed_phot_data), session_metadata)

        if 'analysis_intervals' not in phot_metadata.keys():
            analysis_intervals = pa_plots.select_data_segments_by_span(demod_signals[0, :], session_metadata)
            phot_metadata['analysis_intervals'] = analysis_intervals
            skilled_reaching_io.write_pickle(full_processed_pickle_name, phot_metadata)
        elif not phot_metadata['analysis_intervals']:
            # if 'analysis_intervals' is empty, recheck
            analysis_intervals = pa_plots.select_data_segments_by_span(demod_signals[0, :], session_metadata)
            phot_metadata['analysis_intervals'] = analysis_intervals
            skilled_reaching_io.write_pickle(full_processed_pickle_name, phot_metadata)

        processed_analog = skilled_reaching_io.read_pickle(pickled_analog_processeddata_fname)
        ts_dict = skilled_reaching_io.read_pickle(pickled_ts_fname)
    else:
        # find baseline photometry values for each session
        baseline_file = navigation_utilities.find_baseline_recording(parent_directories, session_metadata)
        baseline_data = determine_baseline(baseline_file, session_metadata, parent_directories)

        phot_metadata = skilled_reaching_io.read_photometry_metadata(data_files['metadata'])
        t, analog_data = skilled_reaching_io.read_analog_bin(data_files['analog_bin'], phot_metadata)
        digital_data = skilled_reaching_io.read_digital_bin(data_files['digital_bin'], phot_metadata)

        ts_dict = sr_analysis.extract_digital_timestamps(phot_metadata, digital_data, analog_data)

        num_samples = len(t)

        use_isosbestic = False
        if 'LED1_modulation' in phot_metadata['AI_line_desc']:
            # a modulation signal was provided
            demod_signals = sr_analysis.demodulate_signals(phot_metadata, analog_data)
            # phot_signal = demod_signals[0,:]
            if 405 in phot_metadata['LEDwavelength']:
                use_isosbestic = True

        phot_metadata['session_info'] = session_metadata
        analysis_intervals = pa_plots.select_data_segments_by_span(demod_signals[0, :], session_metadata)
        # analysis_intervals = pa_plots.segment_data(get_photometry_signal(phot_data), session_metadata)
        phot_metadata['analysis_intervals'] = analysis_intervals
        smoothed = np.empty((num_samples, 2))
        detrended = np.empty((num_samples, 2))
        dff = np.empty((num_samples, 2))
        session_zscores = np.empty((num_samples, 2))
        interval_popts = []
        interval_exp2_fits_successful = []
        baselines_used = []
        for i_signal in range(2):
            smoothed[:, i_signal], detrended[:, i_signal], dff[:, i_signal], interval_popt, interval_exp2_fit_successful, baseline_used = calc_segmented_signal_dff(phot_metadata,
                                                                                                                                                    demod_signals[i_signal, :],
                                                                                                                                                    baseline_data,
                                                                                                                                                    smooth_window=smooth_window,
                                                                                                                                                    f0_pctile=f0_pctile,
                                                                                                                                                    expected_baseline=0.2)
            interval_popts.append(interval_popt)
            interval_exp2_fits_successful.append(interval_exp2_fit_successful)
            baselines_used.append(baseline_used)
            session_zscores[:, i_signal] = zscore_full_session(phot_metadata['analysis_intervals'], dff[:, i_signal])

        # todo: extract timestamps based on digital lines and add to metadata dictionary

        # smoothed_data, detrended_data, dff, popt, exp2_fit_successful = calc_dff(phot_data, baseline_data, smooth_window=smooth_window,
        #                                                                      f0_pctile=f0_pctile)

        # if exp2_fit_successful:
        #     fit_coeff, exp_fit = calc_exp2_fit(num_samples, popt)
        # else:
        #     fit_coeff, exp_fit = calc_exp1_offset_fit(num_samples, popt)

        # check this
        # fig, axs = plt.subplots(3, 1)
        # data = get_photometry_signal(phot_data)

        # todo: title figure and save into a folder with all these samples
        # axs[0].plot(t, data)
        # axs[0].plot(t, exp_fit)
        # axs[1].plot(t, detrended_data)
        # axs[2].plot(t, smoothed_data)
        # plt.show()

        phot_metadata['interval_exp2_fits_successful'] = interval_exp2_fits_successful
        phot_metadata['interval_popts'] = interval_popts
        phot_metadata['f0_pctile'] = f0_pctile
        phot_metadata['mean_baseline'] = baseline_data['mean_baseline']
        phot_metadata['baselines_used'] = baselines_used
        phot_metadata['smooth_window'] = smooth_window

        processed_analog = {'dff': dff,
                            'session_zscores': session_zscores}
        processed_analog['dff'] = dff
        processed_analog['session_zscores'] = session_zscores

        # processed_phot_data['smoothed_data'] = smoothed_data
        # processed_phot_data['dff'] = dff
        # processed_phot_data['zscore'] = session_zscore

        skilled_reaching_io.write_pickle(pickled_metadata_fname, phot_metadata)
        skilled_reaching_io.write_pickle(pickled_analog_processeddata_fname, processed_analog)
        skilled_reaching_io.write_pickle(pickled_ts_fname, ts_dict)

    create_session_summary2(phot_metadata, processed_analog, ts_dict, parent_directories, smooth_window=smooth_window)




def process_session(data_files, session_metadata, parent_directories, smooth_window=101, f0_pctile=10):
    full_processed_pickle_name = navigation_utilities.processed_data_pickle_name(session_metadata, parent_directories)

    # find the scores file and load it
    if os.path.exists(full_processed_pickle_name):
        _, pickle_name = os.path.split(full_processed_pickle_name)
        print('{} already processed'.format(pickle_name))
        processed_phot_data = skilled_reaching_io.read_pickle(full_processed_pickle_name)
        # pa_plots.identify_data_segments(get_photometry_signal(processed_phot_data))
        # intervals_to_analyze, intervals_to_ignore = pa_plots.select_data_segments_by_span(get_photometry_signal(processed_phot_data), session_metadata)

        if 'analysis_intervals' not in processed_phot_data.keys():
            analysis_intervals = pa_plots.select_data_segments_by_span(get_photometry_signal(processed_phot_data), session_metadata)
            processed_phot_data['analysis_intervals'] = analysis_intervals
            skilled_reaching_io.write_pickle(full_processed_pickle_name, processed_phot_data)
        elif not processed_phot_data['analysis_intervals']:
            # if 'analysis_intervals' is empty, recheck
            analysis_intervals = pa_plots.select_data_segments_by_span(get_photometry_signal(processed_phot_data), session_metadata)
            processed_phot_data['analysis_intervals'] = analysis_intervals
            skilled_reaching_io.write_pickle(full_processed_pickle_name, processed_phot_data)

    else:
        # find baseline photometry values for each session
        baseline_file = navigation_utilities.find_baseline_recording(parent_directories, session_metadata)
        baseline_data = determine_baseline(baseline_file, session_metadata, parent_directories)

        phot_data = skilled_reaching_io.read_photometry_data(data_files)
        # todo: add in the demodulation step here if needed
        use_isosbestic = False
        if 'LED1_modulation' in phot_data['AI_line_desc']:
            # a modulation signal was provided
            demod_signals = sr_analysis.demodulate_signals(phot_data)
            phot_signal = demod_signals[0,:]
            if 405 in phot_data['LEDwavelength']:
                use_isosbestic = True
        else:
            phot_signal = get_photometry_signal(phot_data)

        analysis_intervals = pa_plots.select_data_segments_by_span(phot_signal, session_metadata)
        # analysis_intervals = pa_plots.segment_data(get_photometry_signal(phot_data), session_metadata)
        phot_data['analysis_intervals'] = analysis_intervals
        smoothed_data, detrended_data, dff, interval_popt, interval_exp2_fit_successful, baseline_used = calc_segmented_dff(phot_data, baseline_data, smooth_window=smooth_window,
                                                                                                             f0_pctile=f0_pctile, expected_baseline=0.2)
        session_zscore = zscore_full_session(phot_data['analysis_intervals'], dff)
        # smoothed_data, detrended_data, dff, popt, exp2_fit_successful = calc_dff(phot_data, baseline_data, smooth_window=smooth_window,
        #                                                                      f0_pctile=f0_pctile)

        # if exp2_fit_successful:
        #     fit_coeff, exp_fit = calc_exp2_fit(num_samples, popt)
        # else:
        #     fit_coeff, exp_fit = calc_exp1_offset_fit(num_samples, popt)

        # check this
        # fig, axs = plt.subplots(3, 1)
        # data = get_photometry_signal(phot_data)

        # todo: title figure and save into a folder with all these samples
        # axs[0].plot(t, data)
        # axs[0].plot(t, exp_fit)
        # axs[1].plot(t, detrended_data)
        # axs[2].plot(t, smoothed_data)
        # plt.show()

        processed_phot_data = phot_data
        processed_phot_data['interval_exp2_fit_successful'] = interval_exp2_fit_successful
        processed_phot_data['interval_popt'] = interval_popt
        processed_phot_data['f0_pctile'] = f0_pctile
        processed_phot_data['mean_baseline'] = baseline_data['mean_baseline']
        processed_phot_data['baseline_used'] = baseline_used
        # processed_phot_data['smoothed_data'] = smoothed_data
        # processed_phot_data['dff'] = dff
        # processed_phot_data['zscore'] = session_zscore
        processed_phot_data['smooth_window'] = smooth_window

    create_session_summary(processed_phot_data, session_metadata, parent_directories, smooth_window=smooth_window)

    skilled_reaching_io.write_pickle(full_processed_pickle_name, processed_phot_data)


def create_session_summary_legacy(phot_mat_name, session_metadata, parent_directories, smooth_window=201, perievent_window=(-5,5)):

    # phot_data = skilled_reaching_io.read_photometry_mat(phot_mat_name)
    #
    # photo_detrend2, popt, exp2_fit_successful = photodetrend(phot_data)
    # detrended_data, dc_offset, dff, popt, exp2_fit_successful = calc_dff(phot_data, smooth_window=smooth_window,
    #                                                                      f0_pctile=10)

    if session_metadata['task'] == 'skilledreaching':
        create_sr_summary(phot_mat_name, session_metadata, parent_directories, smooth_window=smooth_window, perievent_window=perievent_window)
    elif session_metadata['task'] == 'pavlovian':
        create_pavlovian_summary(phot_mat_name, session_metadata, parent_directories, smooth_window=smooth_window, perievent_window=perievent_window)
    elif session_metadata['task'] == 'chrimsontest':   # check spelling - crimson or chrimson?
        pass


def create_session_summary(processed_phot_data, session_metadata, parent_directories, perievent_window=(-5,5), smooth_window=501, f0_pctile=10):

    if session_metadata['task'] in ['skilledreaching', 'srchrimson', 'srchrimpost']:
        create_sr_summary(processed_phot_data, session_metadata, parent_directories, perievent_window=perievent_window, smooth_window=smooth_window, f0_pctile=f0_pctile)
        sr_summary_by_outcome(processed_phot_data, session_metadata, parent_directories, perievent_window=perievent_window, smooth_window=smooth_window, f0_pctile=f0_pctile)
    elif session_metadata['task'] == 'pavlovian':
        create_pavlovian_summary(processed_phot_data, session_metadata, parent_directories, perievent_window=perievent_window, smooth_window=smooth_window, f0_pctile=f0_pctile)
    elif session_metadata['task'] == 'chrimsontest':   # check spelling - crimson or chrimson?
        create_chrimsontest_summary(processed_phot_data, session_metadata, parent_directories, perievent_window=(-5,25), smooth_window=smooth_window, f0_pctile=f0_pctile)


def create_session_summary2(phot_metadata, processed_analog, ts_dict, parent_directories, perievent_window=(-5,5), smooth_window=501, f0_pctile=10):

    if phot_metadata['session_info']['task'] in ['skilledreaching', 'srchrimson', 'srchrim', 'srchrimpost', 'sr']:
        create_sr_summary2(phot_metadata, processed_analog, ts_dict, parent_directories, perievent_window=perievent_window, smooth_window=smooth_window, f0_pctile=f0_pctile)
        # sr_summary_by_outcome2(phot_metadata, processed_analog, ts_dict, parent_directories, perievent_window=perievent_window, smooth_window=smooth_window, f0_pctile=f0_pctile)
    elif phot_metadata['session_info']['task'] == 'pavlovian':
        create_pavlovian_summary2(phot_metadata, session_metadata, parent_directories, perievent_window=perievent_window, smooth_window=smooth_window, f0_pctile=f0_pctile)
    elif phot_metadata['session_info']['task'] == 'chrimsontest':   # check spelling - crimson or chrimson?
        create_chrimsontest_summary2(phot_metadata, processed_analog, ts_dict, parent_directories, perievent_window=(-5,25), smooth_window=smooth_window, f0_pctile=f0_pctile)



def plot_raw_smoothed_dff(processed_phot_data, axs, phot_signal_idx, smooth_window, f0_pctile, TTL_chan, TTL_eventname):
    t = processed_phot_data['t']
    analysis_window = [0., t[-1][0]]

    data = extract_photometry_data_from_array(processed_phot_data, chan_num=phot_signal_idx)
    num_samples = len(data)

    pa_plots.plot_phot_signal(t, data, analysis_window, axs['row1'])
    if 'analysis_intervals' not in processed_phot_data.keys():
        processed_phot_data['analysis_intervals'] = [[0, num_samples]]
    num_intervals = len(processed_phot_data['analysis_intervals'])

    smoothed_data, detrended_data, dff, interval_popt, interval_exp2_fit_successful, baseline_used = calc_segmented_dff(processed_phot_data, processed_phot_data['mean_baseline'],
                                                                                                         smooth_window=smooth_window,
                                                                                                         f0_pctile=f0_pctile,
                                                                                                         expected_baseline=0.2)
    pa_plots.plot_phot_signal(t, smoothed_data, analysis_window, axs['row2'])
    axs['row2'].set_title('(raw data - exp2 fit) + asymptote - baseline, smoothed')
    pa_plots.eliminate_x_labels(axs['row2'])

    pa_plots.plot_phot_signal(t, dff, analysis_window, axs['row3'])
    TTL = extract_photometry_data_from_array(processed_phot_data, chan_num=TTL_chan)
    pa_plots.overlay_TTL(t, TTL, axs['row3'])

    axs['row3'].set_title('dff, {}'.format(TTL_eventname))
    axs['row3'].set_xlabel('time (s)')

    dc_offsets = np.zeros(num_intervals)
    for i_interval, interval in enumerate(processed_phot_data['analysis_intervals']):
        axs['row1'].plot(t[interval[0]:interval[1]], data[interval[0]:interval[1]], color='g')

        if interval_exp2_fit_successful[i_interval]:
            interval_fit, dc_offsets[i_interval] = calc_interval_exp2_fit(interval, interval_popt[i_interval])
        else:
            interval_fit, dc_offsets[i_interval] = calc_interval_exp1_offset_fit(interval, interval_popt[i_interval])

        axs['row1'].plot(t[interval[0]:interval[1]], interval_fit, color='lightgray')
        axs['row2'].plot(t[interval[0]:interval[1]], smoothed_data[interval[0]:interval[1]], color='g')
        axs['row3'].plot(t[interval[0]:interval[1]], dff[interval[0]:interval[1]], color='g')

    axs['row1'].set_title('raw data')
    pa_plots.eliminate_x_labels(axs['row1'])
    axs['row1'].legend(['raw data', 'exp2 fit'])

    # overlay double (or offset single) exponential fit on raw signal
    # if processed_phot_data['exp2_fit_successful']:
    #     exp_fit, dc_offset = calc_exp2_fit(num_samples, processed_phot_data['popt'])
    # else:
    #     exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, processed_phot_data['popt'])
    # axs['row1'].plot(t, exp_fit)

    # detrended_data = (data - exp_fit) + exp_fit[-1]
    # smoothed_data = smooth(detrended_data, smooth_window) - processed_phot_data['mean_baseline']
    # pa_plots.plot_phot_signal(t, smoothed_data, analysis_window, axs['row2'])
    # axs['row2'].set_title('(raw data - exp2 fit) + asymptote - baseline, smoothed')
    # pa_plots.eliminate_x_labels(axs['row2'])

    # f0 = np.percentile(smoothed_data, f0_pctile)
    # dff = (smoothed_data - f0) / f0

    # pa_plots.plot_phot_signal(t, dff, analysis_window, axs['row3'])
    #
    # TTL = extract_photometry_data_from_array(processed_phot_data, chan_num=TTL_chan)
    # pa_plots.overlay_TTL(t, TTL, axs['row3'])
    #
    # axs['row3'].set_title('dff, {}'.format(TTL_eventname))
    # axs['row3'].set_xlabel('time (s)')

    return dff


def create_chrimsontest_summary2(phot_metadata, processed_analog, ts_dict, parent_directories, perievent_window=(-5,25), smooth_window=101, f0_pctile=10,
                                 col_list=['g', 'm'],
                                 alpha_list=[1., 0.5],
                                 ylims=[[-0.2, 0.4], [-5., 10.]]):

    Fs = phot_metadata['Fs']
    num_samples = np.shape(processed_analog['dff'])[0]
    t = np.linspace(1/Fs, num_samples/Fs, num_samples)

    session_metadata = phot_metadata['session_info']
    session_name = navigation_utilities.session_name_from_metadata(session_metadata)

    save_fname = '_'.join([session_name,
                           session_metadata['task'],
                           'session{:02d}'.format(session_metadata['session_num']),
                           'summarysheet_{:d}secwindows'.format(perievent_window[1]),
                           'smooth{:03d}.jpg'.format(smooth_window)]
                          )
    save_folder = navigation_utilities.find_session_summary_folder(parent_directories, session_metadata)
    save_name = os.path.join(save_folder, save_fname)

    allperievent_folder = navigation_utilities.find_perievent_signal_folder(session_metadata, parent_directories)

    # if os.path.exists(save_name) and os.path.exists(allperievent_folder):
    #     print('{} already created'.format(save_fname))
    #     return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    eventlist, event_ts, frequency_specific_bursts = extract_chrimson_events_digital(ts_dict)

    # burst_start_events = [ev for ev in eventlist if 'start' in ev]
    # burststart_ts = [event_ts[i_event] for i_event, ev in enumerate(eventlist) if 'start' in ev]
    # num_events = len(burst_start_events)

    events_to_plot = eventlist
    event_ts_dict = {eventname: ts for (eventname, ts) in zip(events_to_plot, event_ts)}
    num_events = len(events_to_plot)

    session_fig, session_axes = pa_plots.create_single_session_summary_panels(8.5, 11, num_events)
    session_fig.suptitle(session_name + ', {}, smoothing window = {:d} points'.format(session_metadata['task'], smooth_window))

    num_analog_signals = np.shape(processed_analog['dff'])[1]
    analysis_window = [0, t[-1]]
    for i_sig in range(num_analog_signals):
        pa_plots.plot_phot_signal(t, processed_analog['dff'][:, i_sig], analysis_window, session_axes['row1'],
                                  color=col_list[i_sig],
                                  alpha=alpha_list[i_sig])
        session_axes['row1'].set_ylim(ylims[0])
        pa_plots.plot_phot_signal(t, processed_analog['session_zscores'][:, i_sig], analysis_window, session_axes['row2'],
                                  color=col_list[i_sig],
                                  alpha=alpha_list[i_sig])
        session_axes['row2'].set_ylim(ylims[1])

        baseline_window = (-10, 10)

        perievent_dff = extract_perievent_data_digchans(processed_analog['dff'][:, i_sig],
                                                                                  phot_metadata,
                                                                                  event_ts_dict,
                                                                                  events_to_plot)
        perievent_zscore = extract_perievent_data_digchans(processed_analog['session_zscores'][:, i_sig],
                                                                                     phot_metadata,
                                                                                     event_ts_dict,
                                                                                     events_to_plot)
        for i_col in range(num_events):
            axes_name = 'row4_col{:d}'.format(i_col + 1)
            if perievent_dff[i_col] is not None and len(perievent_dff[i_col]) > 0:
                pa_plots.plot_mean_perievent_signal(perievent_dff[i_col], perievent_window,
                                                    session_metadata, events_to_plot[i_col],
                                                    ax=session_axes[axes_name], color=col_list[i_sig],
                                                    alpha=alpha_list[i_sig])

                axes_name = 'row5_col{:d}'.format(i_col + 1)
                pa_plots.plot_mean_perievent_signal(perievent_zscore[i_col], perievent_window,
                                                    session_metadata, events_to_plot[i_col],
                                                    ax=session_axes[axes_name], color=col_list[i_sig],
                                                    alpha=alpha_list[i_sig])
                session_axes[axes_name].set_xlabel('time (s)')

    plt.savefig(save_name, dpi=300)
    plt.close(session_fig)

    # save individual perievent plots
    for i_event, event in enumerate(events_to_plot):
        print('creating peri-event plots for {}'.format(save_fname))
        if event == 'frame_trigger':
            continue
        pa_plots.plot_all_perievent_signals(perievent_dff[i_event], perievent_window, session_metadata, event, parent_directories, num_cols=2, rows_per_page=6)


def create_chrimsontest_summary(processed_phot_data, session_metadata, parent_directories, perievent_window=(-5,15), smooth_window=501, f0_pctile=10):

    session_name = navigation_utilities.session_name_from_metadata(session_metadata)

    save_fname = '_'.join([session_name,
                           session_metadata['task'],
                           'session{:02d}'.format(session_metadata['session_num']),
                           'summarysheet_{:d}secwindows'.format(perievent_window[1]),
                           'smooth{:03d}.jpg'.format(smooth_window)]
                          )
    save_folder = navigation_utilities.find_session_summary_folder(parent_directories, session_metadata)
    save_name = os.path.join(save_folder, save_fname)

    allperievent_folder = navigation_utilities.find_perievent_signal_folder(session_metadata, parent_directories)

    # if os.path.exists(save_name) and os.path.exists(allperievent_folder):
    #     print('{} already created'.format(save_fname))
    #     return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if len(processed_phot_data['AI_line_desc']) > 0:
        rlight_chan_num = processed_phot_data['AI_line_desc'].index('LED_trigger')
        try:
            phot_signal_idx = processed_phot_data['AI_line_desc'].index('photometry_signal')
        except:
            phot_signal_idx = processed_phot_data['AI_line_desc'].index('photometry signal')
    elif np.shape(processed_phot_data['data'])[1] < 8:
        rlight_chan_num = 1
        phot_signal_idx = 0
    elif session_metadata['ratID'] == 'R0432' and session_metadata['date'] == datetime(2022, 4, 13, 0, 0):
        rlight_chan_num = 6  # I think just this one session where red light trigger was in position 7 (6 counting from 0)
        phot_signal_idx = 0
    elif session_metadata['rat_num'] < 440:
        rlight_chan_num = 1
        phot_signal_idx = 0
    else:
        rlight_chan_num = 7
        phot_signal_idx = 0

    eventlist, event_ts = get_photometry_events(processed_phot_data, session_metadata)
    burst_start_events = [ev for ev in eventlist if 'start' in ev]
    burststart_ts = [event_ts[i_event] for i_event, ev in enumerate(eventlist) if 'start' in ev]
    num_events = len(burst_start_events)

    session_fig, session_axes = pa_plots.create_single_session_summary_panels(8.5, 11, num_events)
    session_fig.suptitle(session_name + ', {}, smoothing window = {:d} points'.format(session_metadata['task'], smooth_window))

    dff = plot_raw_smoothed_dff(processed_phot_data, session_axes, phot_signal_idx, smooth_window, f0_pctile, rlight_chan_num, 'LED_trigger')
    session_zscore = zscore_full_session(processed_phot_data['analysis_intervals'], dff)

    perievent_dff = []
    perievent_zscore = []
    baseline_window = (-10, 10)

    for i_event in range(num_events):
        if burststart_ts[i_event] is not None:
            # p_dff, p_zscore = extract_perievent_signal(dff, event_ts[i_event], perievent_window, processed_phot_data['Fs'], baseline_window=baseline_window)
            p_dff, _ = extract_perievent_signal_in_valid_intervals(dff, burststart_ts[i_event], perievent_window, processed_phot_data['analysis_intervals'], processed_phot_data['Fs'],
                                                       baseline_window=baseline_window)
            p_zscore, _ = extract_perievent_signal_in_valid_intervals(session_zscore, burststart_ts[i_event], perievent_window, processed_phot_data['analysis_intervals'], processed_phot_data['Fs'],
                                                       baseline_window=baseline_window)
            perievent_dff.append(p_dff)
            perievent_zscore.append(p_zscore)
        else:
            perievent_dff.append(None)
            perievent_zscore.append(None)

    for i_col in range(num_events):
        axes_name = 'row4_col{:d}'.format(i_col + 1)
        if perievent_dff[i_col] is not None:
            pa_plots.plot_mean_perievent_signal(perievent_dff[i_col], perievent_window,
                                                                 session_metadata, burst_start_events[i_col],
                                                                 ax=session_axes[axes_name], color='g')

            axes_name = 'row5_col{:d}'.format(i_col + 1)
            if len(perievent_zscore[i_col]) > 0:
                pa_plots.plot_mean_perievent_signal(perievent_zscore[i_col], perievent_window,
                                                                     session_metadata, burst_start_events[i_col],
                                                                     ax=session_axes[axes_name], color='g')
            session_axes[axes_name].set_xlabel('time (s)')

    # plt.show()

    plt.savefig(save_name, dpi=300)
    plt.close(session_fig)

    # save individual perievent plots
    for i_event, event in enumerate(burst_start_events):
        print('creating peri-event plots for {}'.format(save_fname))
        pa_plots.plot_all_perievent_signals(perievent_dff[i_event], perievent_window, session_metadata, event,
                                            parent_directories, num_cols=2, rows_per_page=6)


def create_pavlovian_summary(processed_phot_data, session_metadata, parent_directories, perievent_window=(-3, 3), smooth_window=501, f0_pctile=10):

    session_name = navigation_utilities.session_name_from_metadata(session_metadata)

    save_fname = '_'.join([session_name,
                           session_metadata['task'],
                           'session{:02d}'.format(session_metadata['session_num']),
                           'summarysheet_{:d}secwindows'.format(perievent_window[1]),
                           'smooth{:03d}.jpg'.format(smooth_window)]
                          )
    save_folder = navigation_utilities.find_session_summary_folder(parent_directories, session_metadata)
    save_name = os.path.join(save_folder, save_fname)

    allperievent_folder = navigation_utilities.find_perievent_signal_folder(session_metadata, parent_directories)

    # if os.path.exists(save_name) and os.path.exists(allperievent_folder):
    #     print('{} already created'.format(save_fname))
    #     return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if len(processed_phot_data['AI_line_desc']) > 0:
        FED_chan_num = processed_phot_data['AI_line_desc'].index('FED3')
        try:
            phot_signal_idx = processed_phot_data['AI_line_desc'].index('photometry_signal')
        except:
            phot_signal_idx = processed_phot_data['AI_line_desc'].index('photometry signal')
        # todo: read in which line is the photometry data from the nidaq.mat file
    else:
        FED_chan_num = 1
        phot_signal_idx = 0

    eventlist, event_ts = get_photometry_events(processed_phot_data, session_metadata)
    num_events = len(eventlist)

    session_fig, session_axes = pa_plots.create_single_session_summary_panels(8.5, 11, num_events)
    session_fig.suptitle(session_name + ', {}, smoothing window = {:d} points'.format(session_metadata['task'], smooth_window))

    dff = plot_raw_smoothed_dff(processed_phot_data, session_axes, phot_signal_idx, smooth_window, f0_pctile, FED_chan_num, 'FED3')
    session_zscore = zscore_full_session(processed_phot_data['analysis_intervals'], dff)

    perievent_dff = []
    perievent_zscore = []
    baseline_window = (-10, 10)

    for i_event in range(num_events):
        if event_ts[i_event] is not None:
            p_dff, _ = extract_perievent_signal_in_valid_intervals(dff, event_ts[i_event], perievent_window, processed_phot_data['analysis_intervals'], processed_phot_data['Fs'],
                                                       baseline_window=baseline_window)
            p_zscore, _ = extract_perievent_signal_in_valid_intervals(session_zscore, event_ts[i_event], perievent_window, processed_phot_data['analysis_intervals'], processed_phot_data['Fs'],
                                                       baseline_window=baseline_window)
            perievent_dff.append(p_dff)
            perievent_zscore.append(p_zscore)
        else:
            perievent_dff.append(None)
            perievent_zscore.append(None)

    for i_col in range(num_events):
        axes_name = 'row4_col{:d}'.format(i_col + 1)
        if perievent_dff[i_col] is not None:
            pa_plots.plot_mean_perievent_signal(perievent_dff[i_col], perievent_window,
                                                                 session_metadata, eventlist[i_col],
                                                                 ax=session_axes[axes_name], color='g')

            axes_name = 'row5_col{:d}'.format(i_col + 1)
            pa_plots.plot_mean_perievent_signal(perievent_zscore[i_col], perievent_window,
                                                                 session_metadata, eventlist[i_col],
                                                                 ax=session_axes[axes_name], color='g')
            session_axes[axes_name].set_xlabel('time (s)')

    # plt.show()

    plt.savefig(save_name, dpi=300)
    plt.close(session_fig)

    # save individual perievent plots
    for i_event, event in enumerate(eventlist):
        print('creating peri-event plots for {}'.format(save_fname))
        pa_plots.plot_all_perievent_signals(perievent_dff[i_event], perievent_window, session_metadata, event, parent_directories, num_cols=2, rows_per_page=6)


def create_pavlovian_summary_legacy(phot_mat_name, session_metadata, parent_directories, smooth_window=201, perievent_window=(-3, 3)):

    session_name = navigation_utilities.session_name_from_metadata(session_metadata)

    save_fname = '_'.join([session_name,
                           session_metadata['task'],
                           'session{:02d}'.format(session_metadata['session_num']),
                           'summarysheet_{:d}secwindows.jpg'.format(perievent_window[1])]
                          )
    save_folder = navigation_utilities.find_session_summary_folder(parent_directories, session_metadata)
    save_name = os.path.join(save_folder, save_fname)
    if os.path.exists(save_name):
        print('{} already created'.format(save_fname))
        return

    phot_data = skilled_reaching_io.read_photometry_mat(phot_mat_name)
    if len(phot_data['AI_line_desc']) > 0:
        FED_chan_num = phot_data['AI_line_desc'].index('FED3')
        try:
            phot_signal_idx = phot_data['AI_line_desc'].index('photometry_signal')
        except:
            phot_signal_idx = phot_data['AI_line_desc'].index('photometry signal')
        # todo: read in which line is the photometry data from the nidaq.mat file
    else:
        FED_chan_num = 1
        phot_signal_idx = 0

    detrended_data, dc_offset, dff, popt, exp2_fit_successful = calc_dff(phot_data, smooth_window=smooth_window,
                                                                         f0_pctile=10)

    eventlist, event_ts = get_photometry_events(phot_data, session_metadata)
    num_events = len(eventlist)

    session_fig, session_axes = pa_plots.create_single_session_summary_panels(8.5, 11, num_events)


    session_fig.suptitle(session_name + ', {}, smoothing window = {:d} points'.format(session_metadata['task'], smooth_window))

    analysis_window = [0., phot_data['t'][-1][0]]
    t = phot_data['t']

    data = extract_photometry_data_from_array(phot_data, chan_num=phot_signal_idx)
    num_samples = len(data)
    pa_plots.plot_phot_signal(t, data, analysis_window, session_axes['row1'])

    # overlay double (or offset single) exponential fit on raw signal
    if exp2_fit_successful:
        exp_fit, dc_offset = calc_exp2_fit(num_samples, popt)
    else:
        exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, popt)
    session_axes['row1'].plot(t, exp_fit)
    session_axes['row1'].set_title('raw data')
    pa_plots.eliminate_x_labels(session_axes['row1'])
    session_axes['row1'].legend(['raw data', 'exp2 fit'])

    pa_plots.plot_phot_signal(t, (data - exp_fit) + exp_fit[-1], analysis_window, session_axes['row2'])
    session_axes['row2'].set_title('(raw data - exp2 fit) + asymptote')
    pa_plots.eliminate_x_labels(session_axes['row2'])

    pa_plots.plot_phot_signal(t, dff, analysis_window, session_axes['row3'])

    TTL = extract_photometry_data_from_array(phot_data, chan_num=FED_chan_num)
    pa_plots.overlay_TTL(t, TTL, session_axes['row3'])

    session_axes['row3'].set_title('dff')
    session_axes['row3'].set_xlabel('time (s)')

    perievent_dff = []
    perievent_zscore = []
    baseline_window = (-10, 10)

    for i_event in range(num_events):
        if event_ts[i_event] is not None:
            p_dff, p_zscore = extract_perievent_signal_and_zscore(dff, event_ts[i_event], perievent_window, phot_data['Fs'], baseline_window=baseline_window)
            perievent_dff.append(p_dff)
            perievent_zscore.append(p_zscore)
        else:
            perievent_dff.append(None)
            perievent_zscore.append(None)

    for i_col in range(num_events):
        axes_name = 'row4_col{:d}'.format(i_col)
        if perievent_dff[i_col] is not None:
            pa_plots.plot_mean_perievent_signal(perievent_dff[i_col], perievent_window,
                                                                 session_metadata, eventlist[i_col],
                                                                 ax=session_axes[axes_name], color='g')

            axes_name = 'row5_col{:d}'.format(i_col)
            pa_plots.plot_mean_perievent_signal(perievent_zscore[i_col], perievent_window,
                                                                 session_metadata, eventlist[i_col],
                                                                 ax=session_axes[axes_name], color='g')
            session_axes[axes_name].set_xlabel('time (s)')

    plt.savefig(save_name, dpi=300)
    plt.close(session_fig)


def sr_summary_by_outcome(processed_phot_data, session_metadata, parent_directories, perievent_window=(-5,5), smooth_window=251, f0_pctile=0.1):
    '''
    0 – No pellet, mechanical failure
    1 -  First trial success (obtained pellet on initial limb advance). If more than one pellet on pedestal, successfully grabbing any pellet counts as success for scores 1 and 2
    2 -  Success (obtain pellet, but not on first attempt)
    3 -  Forelimb advance - pellet dropped in box
    4 -  Forelimb advance - pellet knocked off shelf
    5 -  Obtained pellet with tongue
    6 -  Walked away without forelimb advance, no forelimb advance
    7 -  Reached, pellet remains on shelf
    8 - Used only contralateral paw
    9 – Laser/video fired at the wrong time
    10 – Used preferred paw after obtaining or moving pellet with tongue
    11 – Obtained pellet with preferred paw after using non-preferred paw
    '''
    num_event_rows = 2
    num_full_row_axes = 3

    session_name = navigation_utilities.session_name_from_metadata(session_metadata)

    save_fname = '_'.join([session_name,
                           session_metadata['task'],
                           'session{:02d}'.format(session_metadata['session_num']),
                           'trialoutcomes_{:d}secwindows'.format(perievent_window[1]),
                           'smooth{:03d}.jpg'.format(smooth_window)]
                          )
    save_folder = navigation_utilities.find_session_summary_folder(parent_directories, session_metadata)
    save_name = os.path.join(save_folder, save_fname)
    # if os.path.exists(save_name):
    #     print('{} already created'.format(save_fname))
    #     return

    xls_score_file = navigation_utilities.find_scores_xlsx(parent_directories)
    sr_scores = skilled_reaching_io.read_xlsx_scores(xls_score_file, session_metadata)

    outcome_groupings = sr_analysis.create_sr_outcome_groupings()
    trials_by_outcome, valid_trials = sr_analysis.extract_sr_trials_by_outcome(sr_scores, session_metadata, outcome_groupings)

    outcome_names = list(trials_by_outcome.keys())
    outcome_names.append('early_reach')
    num_outcomes = len(outcome_names)

    num_event_cols = math.ceil(num_outcomes / num_event_rows)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if len(processed_phot_data['AI_line_desc']) > 0:
        vidtrig_chan_num = processed_phot_data['AI_line_desc'].index('vid_trigger')
        try:
            phot_signal_idx = processed_phot_data['AI_line_desc'].index('photometry_signal')
        except:
            phot_signal_idx = processed_phot_data['AI_line_desc'].index('photometry signal')
    else:
        vidtrig_chan_num = 6
        phot_signal_idx = 0

    session_fig, session_axes = pa_plots.create_single_session_summary_panels(8.5, 11, num_event_cols,
                                                                              num_full_row_axes=num_full_row_axes, num_event_rows=2)
    session_fig.suptitle(session_name + ', {}, smoothing window = {:d} points'.format(session_metadata['task'], smooth_window))

    dff = plot_raw_smoothed_dff(processed_phot_data, session_axes, phot_signal_idx, smooth_window, f0_pctile,
                                vidtrig_chan_num, 'vid_trigger')
    session_zscore = zscore_full_session(processed_phot_data['analysis_intervals'], dff)

    perievent_dff = []
    perievent_zscore = []
    baseline_window = (-10, 10)

    eventlist, event_ts = get_photometry_events(processed_phot_data, session_metadata)
    vid_trig_idx = eventlist.index('vid_trigger')

    for i_outcome, outcome_name in enumerate(outcome_names):
        if outcome_name in trials_by_outcome.keys():
            if trials_by_outcome[outcome_name] is not None:
                ts_idx = np.array([i_trial for i_trial, vid_num in enumerate(valid_trials) if vid_num in trials_by_outcome[outcome_name]])
                ts_idx = ts_idx[ts_idx < len(event_ts[vid_trig_idx])]   # sometimes, some videos were taken after the recording stop; exclude these since there's no photometry for them
                if len(ts_idx) > 0:
                    trial_ts = event_ts[vid_trig_idx][ts_idx]

                    p_dff, p_zscore = extract_perievent_signal_and_zscore(dff, trial_ts, perievent_window, processed_phot_data['Fs'],
                                                                          baseline_window=baseline_window)
                    perievent_dff.append(p_dff)
                    perievent_zscore.append(p_zscore)
                else:
                    perievent_dff.append(None)
                    perievent_zscore.append(None)
            else:
                perievent_dff.append(None)
                perievent_zscore.append(None)
        elif outcome_name == 'early_reach':
            early_reach_ts = find_early_reaches(eventlist, event_ts)
            p_dff, p_zscore = extract_perievent_signal_and_zscore(dff, early_reach_ts, perievent_window, processed_phot_data['Fs'],
                                                                  baseline_window=baseline_window)
            perievent_dff.append(p_dff)
            perievent_zscore.append(p_zscore)

        ax_row = math.floor(i_outcome / num_event_cols) + num_full_row_axes
        ax_col = i_outcome % num_event_cols

        axes_name = 'row{:d}_col{:d}'.format(ax_row+1, ax_col+1)

        if perievent_dff[i_outcome] is not None and len(perievent_dff[i_outcome]) > 0:
            pa_plots.plot_mean_perievent_signal(perievent_dff[i_outcome], perievent_window,
                                                session_metadata, outcome_name,
                                                ax=session_axes[axes_name], color='g')
        else:
            num_trials = 0
            title_string = outcome_name + '\nn = {:d}'.format(num_trials)
            session_axes[axes_name].set_title(title_string, fontsize=10)

        if ax_row == 4:
            session_axes[axes_name].set_xlabel('time (s)')

    plt.savefig(save_name, dpi=300)
    plt.close(session_fig)



def find_early_reaches_legacy(phot_data, session_metadata):

    time_tolerance = 0.001   # not sure if we need this, but will assume anything less than time_tolerance (in sec) apart is the same event
    multiple_reach_time = 5.   # reaches within 5 seconds of a video trigger excluded from "early reach" category

    eventlist, event_ts = get_photometry_events(phot_data, session_metadata)

    # eventlist = ['paw_through_slot', 'Actuator3', 'Actuator2', 'IR_back', 'vid_trigger', 'frame_trigger']

    # an early reach would be a paw_through_slot event after Actuator 2 but before Actuator 3

    pts_idx = eventlist.index('paw_through_slot')
    pts_ts = event_ts[pts_idx]    # should be a numpy array of timestamps
    if not isinstance(pts_ts, np.ndarray):
        pts_ts = np.array([pts_ts])

    vt_idx = eventlist.index('vid_trigger')
    vt_ts = event_ts[vt_idx]
    if not isinstance(vt_ts, np.ndarray):
        vt_ts = np.array([vt_ts])

    a2_idx = eventlist.index('Actuator2')
    a3_idx = eventlist.index('Actuator3')
    a2_ts = event_ts[a2_idx]
    a3_ts = event_ts[a3_idx]
    if not isinstance(a2_ts, np.ndarray):
        a2_ts = np.array([a2_ts])
    if not isinstance(a3_ts, np.ndarray):
        a3_ts = np.array([a3_ts])

    IRback_idx = eventlist.index('IR_back')
    IRback_ts = event_ts[IRback_idx]
    if not isinstance(IRback_ts, np.ndarray):
        IRback_ts = np.array([IRback_ts])
    # if only a single event occurred, need to turn it into an array with the right dimensions for subsequent processing

    # first, eliminate any paw_through_slot events that triggered a video
    # find all elements of pts_ts that are not within time_tolerance of any events in vt_ts
    non_vt_ts = np.array([ts for ts in pts_ts if not any(abs(ts - vt_ts) < time_tolerance)])

    # now find all paw_through_slot events that occurred after Actuator2 but before IR back (or Actuator3, but not recording that right now)
    # or better to look for paw_through_slot events that did not trigger video and occur more than X seconds (maybe 5?)

    # loop through each video trigger timestamp, classify any non-video-trigger timestamps as a multi-reach if they occurred too soon
    multi_reach_mask = np.zeros(len(non_vt_ts), dtype=bool)
    for vidtrig_ts in vt_ts:

        trial_mask = np.asarray(np.logical_and(non_vt_ts > vidtrig_ts, non_vt_ts < vidtrig_ts + multiple_reach_time))
        multi_reach_mask = np.logical_or(multi_reach_mask, trial_mask)

    early_reach_ts = non_vt_ts[np.logical_not(multi_reach_mask)]

    return early_reach_ts


def find_early_reaches(eventlist, event_ts):

    time_tolerance = 0.001   # not sure if we need this, but will assume anything less than time_tolerance (in sec) apart is the same event
    multiple_reach_time = 5.   # reaches within 5 seconds of a video trigger excluded from "early reach" category

    # eventlist, event_ts = get_photometry_events(phot_data, session_metadata)

    # eventlist = ['paw_through_slot', 'Actuator3', 'Actuator2', 'IR_back', 'vid_trigger', 'frame_trigger']

    # an early reach would be a paw_through_slot event after Actuator 2 but before Actuator 3

    pts_idx = eventlist.index('paw_through_slot')
    pts_ts = event_ts[pts_idx]    # should be a numpy array of timestamps
    if not isinstance(pts_ts, np.ndarray):
        pts_ts = np.array([pts_ts])

    vt_idx = eventlist.index('vid_trigger')
    vt_ts = event_ts[vt_idx]
    if not isinstance(vt_ts, np.ndarray):
        vt_ts = np.array([vt_ts])

    a2_idx = eventlist.index('Actuator2')
    a3_idx = eventlist.index('Actuator3')
    a2_ts = event_ts[a2_idx]
    a3_ts = event_ts[a3_idx]
    if not isinstance(a2_ts, np.ndarray):
        a2_ts = np.array([a2_ts])
    if not isinstance(a3_ts, np.ndarray):
        a3_ts = np.array([a3_ts])

    IRback_idx = eventlist.index('IR_back')
    IRback_ts = event_ts[IRback_idx]
    if not isinstance(IRback_ts, np.ndarray):
        IRback_ts = np.array([IRback_ts])
    # if only a single event occurred, need to turn it into an array with the right dimensions for subsequent processing

    # first, eliminate any paw_through_slot events that triggered a video
    # find all elements of pts_ts that are not within time_tolerance of any events in vt_ts
    non_vt_ts = np.array([ts for ts in pts_ts if not any(abs(ts - vt_ts) < time_tolerance)])

    # now find all paw_through_slot events that occurred after Actuator2 but before IR back (or Actuator3, but not recording that right now)
    # or better to look for paw_through_slot events that did not trigger video and occur more than X seconds (maybe 5?)

    # loop through each video trigger timestamp, classify any non-video-trigger timestamps as a multi-reach if they occurred too soon
    multi_reach_mask = np.zeros(len(non_vt_ts), dtype=bool)
    for vidtrig_ts in vt_ts:

        trial_mask = np.asarray(np.logical_and(non_vt_ts > vidtrig_ts, non_vt_ts < vidtrig_ts + multiple_reach_time))
        multi_reach_mask = np.logical_or(multi_reach_mask, trial_mask)

    early_reach_ts = non_vt_ts[np.logical_not(multi_reach_mask)]

    return early_reach_ts


def extract_perievent_data(phot_signal, processed_phot_data, session_metadata, perievent_window = (-3, 3), baseline_window=(-10, 10)):

    eventlist, event_ts = get_photometry_events(processed_phot_data, session_metadata)
    num_events = len(eventlist)

    perievent_signal = []

    for i_event in range(num_events):
        if event_ts[i_event] is not None:
            #todo: account for matching events and timestamps when some are outside of valid recording intervals (ferrule slip, etc)
            p_signal, _ = extract_perievent_signal_in_valid_intervals(phot_signal, event_ts[i_event], perievent_window, processed_phot_data['analysis_intervals'], processed_phot_data['Fs'],
                                                       baseline_window=baseline_window)
            perievent_signal.append(p_signal)
            # perievent_signal is a list of dff values for each event.
            # for each event, there is a list of signals for each interval

        else:
            perievent_signal.append(None)

    return perievent_signal, eventlist, event_ts


def extract_perievent_data_digchans(phot_signal, phot_metadata, ts_dict, events_to_plot, perievent_window=(-3, 3), baseline_window=(-10, 10)):

    num_events = len(events_to_plot)

    perievent_signal = []

    for i_event, event in enumerate(events_to_plot):
        if np.ndim(ts_dict[event]) == 1:
            ts = ts_dict[event]
        else:
            ts = ts_dict[event][:, 0]    # assume we want the "on" time
        # todo: need to find the start of red light pulse trains
        if len(ts) > 0:
            #todo: account for matching events and timestamps when some are outside of valid recording intervals (ferrule slip, etc)
            p_signal, _ = extract_perievent_signal_in_valid_intervals(phot_signal, ts, perievent_window, phot_metadata['analysis_intervals'], phot_metadata['Fs'],
                                                                      baseline_window=baseline_window)
            perievent_signal.append(p_signal)
            # perievent_signal is a list of dff values for each event.
            # for each event, there is a list of signals for each interval

        else:
            perievent_signal.append(None)

    return perievent_signal


def create_sr_summary2(phot_metadata, processed_analog, ts_dict, parent_directories, perievent_window=(-3, 3), smooth_window=501, f0_pctile=10,
                       events_to_plot=['Actuator3', 'paw_through_slot', 'vid_trigger'],
                       col_list=['g', 'm'],
                       alpha_list=[1., 0.5],
                       ylims=[[-0.2,0.4],[-5.,10.]]):

    Fs = phot_metadata['Fs']
    num_samples = np.shape(processed_analog['dff'])[0]
    t = np.linspace(1/Fs, num_samples/Fs, num_samples)

    session_metadata = phot_metadata['session_info']

    num_event_rows = 2
    num_full_row_axes = 3

    session_name = navigation_utilities.session_name_from_metadata(session_metadata)

    save_fname = '_'.join([session_name,
                           session_metadata['task'],
                           'session{:02d}'.format(session_metadata['session_num']),
                           'summarysheet_{:d}secwindows'.format(perievent_window[1]),
                           'smooth{:03d}.jpg'.format(smooth_window)]
                          )
    save_folder = navigation_utilities.find_session_summary_folder(parent_directories, session_metadata)
    save_name = os.path.join(save_folder, save_fname)

    # allperievent_folder = navigation_utilities.find_perievent_signal_folder(session_metadata, parent_directories)

    # if os.path.exists(save_name) and os.path.exists(allperievent_folder):
    #     print('{} already created'.format(save_fname))
    #     return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    num_events = len(events_to_plot)

    session_fig, session_axes = pa_plots.create_single_session_summary_panels(8.5, 11, num_events)
    session_fig.suptitle(session_name + ', {}, smoothing window = {:d} points'.format(session_metadata['task'], smooth_window))

    num_analog_signals = np.shape(processed_analog['dff'])[1]
    analysis_window = [0, t[-1]]
    for i_sig in range(num_analog_signals):
        pa_plots.plot_phot_signal(t, processed_analog['dff'][:, i_sig], analysis_window, session_axes['row1'],
                                  color=col_list[i_sig],
                                  alpha=alpha_list[i_sig])
        session_axes['row1'].set_ylim(ylims[0])
        pa_plots.plot_phot_signal(t, processed_analog['session_zscores'][:, i_sig], analysis_window, session_axes['row2'],
                                  color=col_list[i_sig],
                                  alpha=alpha_list[i_sig])
        session_axes['row2'].set_ylim(ylims[1])

        baseline_window = (-10, 10)

        perievent_dff = extract_perievent_data_digchans(processed_analog['dff'][:, i_sig],
                                                                                  phot_metadata,
                                                                                  ts_dict,
                                                                                  events_to_plot)
        perievent_zscore = extract_perievent_data_digchans(processed_analog['session_zscores'][:, i_sig],
                                                                                     phot_metadata,
                                                                                     ts_dict,
                                                                                     events_to_plot)
        for i_col in range(num_events):
            axes_name = 'row4_col{:d}'.format(i_col + 1)
            if perievent_dff[i_col] is not None and len(perievent_dff[i_col]) > 0:
                pa_plots.plot_mean_perievent_signal(perievent_dff[i_col], perievent_window,
                                                    session_metadata, events_to_plot[i_col],
                                                    ax=session_axes[axes_name], color=col_list[i_sig])

                axes_name = 'row5_col{:d}'.format(i_col + 1)
                pa_plots.plot_mean_perievent_signal(perievent_zscore[i_col], perievent_window,
                                                    session_metadata, events_to_plot[i_col],
                                                    ax=session_axes[axes_name], color=col_list[i_sig])
                session_axes[axes_name].set_xlabel('time (s)')

    plt.savefig(save_name, dpi=300)
    plt.close(session_fig)

    # save individual perievent plots
    for i_event, event in enumerate(events_to_plot):
        print('creating peri-event plots for {}'.format(save_fname))
        if event == 'frame_trigger':
            continue
        pa_plots.plot_all_perievent_signals(perievent_dff[i_event], perievent_window, session_metadata, event, parent_directories, num_cols=2, rows_per_page=6)


def create_sr_summary(processed_phot_data, session_metadata, parent_directories, perievent_window=(-3, 3), smooth_window=501, f0_pctile=10):

    num_event_rows = 2
    num_full_row_axes = 3

    session_name = navigation_utilities.session_name_from_metadata(session_metadata)

    save_fname = '_'.join([session_name,
                           session_metadata['task'],
                           'session{:02d}'.format(session_metadata['session_num']),
                           'summarysheet_{:d}secwindows'.format(perievent_window[1]),
                           'smooth{:03d}.jpg'.format(smooth_window)]
                          )
    save_folder = navigation_utilities.find_session_summary_folder(parent_directories, session_metadata)
    save_name = os.path.join(save_folder, save_fname)

    # allperievent_folder = navigation_utilities.find_perievent_signal_folder(session_metadata, parent_directories)

    # if os.path.exists(save_name) and os.path.exists(allperievent_folder):
    #     print('{} already created'.format(save_fname))
    #     return

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if len(processed_phot_data['AI_line_desc']) > 0:
        vidtrig_chan_num = processed_phot_data['AI_line_desc'].index('vid_trigger')
        try:
            phot_signal_idx = processed_phot_data['AI_line_desc'].index('photometry_signal')
        except:
            phot_signal_idx = processed_phot_data['AI_line_desc'].index('photometry signal')
    else:
        vidtrig_chan_num = 6
        phot_signal_idx = 0

    eventlist, event_ts = get_photometry_events(processed_phot_data, session_metadata)
    num_events = len(eventlist)

    session_fig, session_axes = pa_plots.create_single_session_summary_panels(8.5, 11, num_events)
    session_fig.suptitle(session_name + ', {}, smoothing window = {:d} points'.format(session_metadata['task'], smooth_window))

    dff = plot_raw_smoothed_dff(processed_phot_data, session_axes, phot_signal_idx, smooth_window, f0_pctile,
                                vidtrig_chan_num, 'vid_trigger')
    session_zscore = zscore_full_session(processed_phot_data['analysis_intervals'], dff)

    # perievent_dff = []
    # perievent_zscore = []
    baseline_window = (-10, 10)
    perievent_dff, eventlist, event_ts = extract_perievent_data(dff, processed_phot_data, session_metadata)
    perievent_zscore, eventlist, event_ts = extract_perievent_data(session_zscore, processed_phot_data, session_metadata)
    # for i_event in range(num_events):
    #     if event_ts[i_event] is not None:
    #         p_dff = extract_perievent_signal_in_valid_intervals(dff, event_ts[i_event], perievent_window, processed_phot_data['analysis_intervals'], processed_phot_data['Fs'],
    #                                                    baseline_window=baseline_window)
    #         p_zscore = extract_perievent_signal_in_valid_intervals(session_zscore, event_ts[i_event], perievent_window, processed_phot_data['analysis_intervals'], processed_phot_data['Fs'],
    #                                                    baseline_window=baseline_window)
    #         perievent_dff.append(p_dff)
    #         # perievent_dff is a list of dff values for each event.
    #         # for each event, there is a list of dff's for each interval
    #         perievent_zscore.append(p_zscore)
    #     else:
    #         perievent_dff.append(None)
    #         perievent_zscore.append(None)

    for i_col in range(num_events):
        axes_name = 'row4_col{:d}'.format(i_col + 1)
        if perievent_dff[i_col] is not None and len(perievent_dff[i_col]) > 0:
            pa_plots.plot_mean_perievent_signal(perievent_dff[i_col], perievent_window,
                                                session_metadata, eventlist[i_col],
                                                ax=session_axes[axes_name], color='g')

            axes_name = 'row5_col{:d}'.format(i_col + 1)
            pa_plots.plot_mean_perievent_signal(perievent_zscore[i_col], perievent_window,
                                                session_metadata, eventlist[i_col],
                                                ax=session_axes[axes_name], color='g')
            session_axes[axes_name].set_xlabel('time (s)')

    plt.savefig(save_name, dpi=300)
    plt.close(session_fig)

    # save individual perievent plots
    for i_event, event in enumerate(eventlist):
        print('creating peri-event plots for {}'.format(save_fname))
        if event == 'frame_trigger':
            continue
        pa_plots.plot_all_perievent_signals(perievent_dff[i_event], perievent_window, session_metadata, event, parent_directories, num_cols=2, rows_per_page=6)


def zscore_from_dff(dff):

    mean_dff = np.mean(dff)
    std_dff = np.std(dff)

    zscore_dff = (dff - mean_dff) / std_dff

    return zscore_dff


def zscore_full_session(intervals, dff):
    # working here - todo: calculate z-score for each interval

    num_samples = len(dff)
    session_zscore = np.zeros(num_samples)
    for interval in intervals:
        session_zscore[interval[0]:interval[1]] = zscore_from_dff(dff[interval[0]:interval[1]])

    return session_zscore


def create_sr_summary_legacy(phot_mat_name, session_metadata, parent_directories, smooth_window=201, perievent_window=(-3, 3)):


    session_name = navigation_utilities.session_name_from_metadata(session_metadata)

    save_fname = '_'.join([session_name,
                           session_metadata['task'],
                           'session{:02d}'.format(session_metadata['session_num']),
                           'summarysheet_{:d}secwindows.jpg'.format(perievent_window[1])]
                          )
    save_folder = navigation_utilities.find_session_summary_folder(parent_directories, session_metadata)
    save_name = os.path.join(save_folder, save_fname)
    if os.path.exists(save_name):
        print('{} already created'.format(save_fname))
        return

    phot_data = skilled_reaching_io.read_photometry_mat(phot_mat_name)
    if len(phot_data['AI_line_desc']) > 0:
        # FED_chan_num = phot_data['AI_line_desc'].index('FED3')
        try:
            phot_signal_idx = phot_data['AI_line_desc'].index('photometry_signal')
        except:
            phot_signal_idx = phot_data['AI_line_desc'].index('photometry signal')
    else:
        # FED_chan_num = 1
        phot_signal_idx = 0

    detrended_data, dc_offset, dff, popt, exp2_fit_successful = calc_dff(phot_data, smooth_window=smooth_window,
                                                                         f0_pctile=10)

    eventlist, event_ts = get_photometry_events(phot_data, session_metadata)
    num_events = len(eventlist)

    session_fig, session_axes = pa_plots.create_single_session_summary_panels(8.5, 11, num_events)

    session_fig.suptitle(session_name + ', {}, smoothing window = {:d} points'.format(session_metadata['task'], smooth_window))

    analysis_window = [0., phot_data['t'][-1][0]]
    t = phot_data['t']

    data = extract_photometry_data_from_array(phot_data, chan_num=phot_signal_idx)
    num_samples = len(data)
    pa_plots.plot_phot_signal(t, data, analysis_window, session_axes['row1'], ylim=[0.01, 0.08])

    # overlay double (or offset single) exponential fit on raw signal
    if exp2_fit_successful:
        exp_fit, dc_offset = calc_exp2_fit(num_samples, popt)
    else:
        exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, popt)
    session_axes['row1'].plot(t, exp_fit)
    session_axes['row1'].set_title('raw data')
    pa_plots.eliminate_x_labels(session_axes['row1'])
    session_axes['row1'].legend(['raw data', 'exp2 fit'])

    pa_plots.plot_phot_signal(t, (data - exp_fit) + exp_fit[-1], analysis_window, session_axes['row2'],
                              ylim=[0.01, 0.08])
    session_axes['row2'].set_title('(raw data - exp2 fit) + asymptote')
    pa_plots.eliminate_x_labels(session_axes['row2'])

    pa_plots.plot_phot_signal(t, dff, analysis_window, session_axes['row3'],
                              ylim=[0.01, 0.08])

    # TTL = extract_photometry_data_from_array(phot_data, chan_num=FED_chan_num)
    # pa_plots.overlay_TTL(t, TTL, session_axes['row3'])

    session_axes['row3'].set_title('dff')
    session_axes['row3'].set_xlabel('time (s)')

    perievent_dff = []
    perievent_zscore = []
    baseline_window = (-10, 10)

    for i_event in range(num_events):
        if event_ts[i_event] is not None:
            p_dff, p_zscore = extract_perievent_signal_and_zscore(dff, event_ts[i_event], perievent_window, phot_data['Fs'],
                                                                  baseline_window=baseline_window)
            perievent_dff.append(p_dff)
            perievent_zscore.append(p_zscore)
        else:
            perievent_dff.append(None)
            perievent_zscore.append(None)

    for i_col in range(num_events):
        axes_name = 'row4_col{:d}'.format(i_col)
        if perievent_dff[i_col] is not None and len(perievent_dff[i_col]) > 0:
            pa_plots.plot_mean_perievent_signal(perievent_dff[i_col], perievent_window,
                                                session_metadata, eventlist[i_col],
                                                ax=session_axes[axes_name], color='g')

            axes_name = 'row5_col{:d}'.format(i_col)
            pa_plots.plot_mean_perievent_signal(perievent_zscore[i_col], perievent_window,
                                                session_metadata, eventlist[i_col],
                                                ax=session_axes[axes_name], color='g')
            session_axes[axes_name].set_xlabel('time (s)')

    plt.savefig(save_name, dpi=300)
    plt.close(session_fig)


def determine_baseline(baseline_file, session_metadata, parent_directories):

    baseline_pickle_name = navigation_utilities.create_baseline_pickle_name(baseline_file, session_metadata, parent_directories)
    if os.path.exists(baseline_pickle_name):
        _, baseline_fname = os.path.split(baseline_pickle_name)
        print('baseline data loaded from {}'.format(baseline_fname))
        baseline_data = skilled_reaching_io.read_pickle(baseline_pickle_name)
    elif baseline_file is not None:
        baseline_data = skilled_reaching_io.read_photometry_mat(baseline_file)

        t_min, t_max = pa_plots.select_data_window(baseline_data['t'], baseline_data['data'], session_metadata)
        indmin, indmax = np.searchsorted(baseline_data['t'], (t_min, t_max))
        indmax = min(len(baseline_data['t']) - 1, indmax)

        mean_signal = np.mean(baseline_data['data'][indmin:indmax])

        baseline_data['baseline_range'] = (t_min, t_max)
        baseline_data['mean_baseline'] = mean_signal

    else:
        baseline_data = {'mean_baseline': 0.15,
                         'baseline_range': None
        }

    skilled_reaching_io.write_pickle(baseline_pickle_name, baseline_data)

    return baseline_data