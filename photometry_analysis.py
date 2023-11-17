import skilled_reaching_io
import photometry_analysis_plots
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.optimize import curve_fit
from datetime import datetime
import csv
import scipy.io as sio

def plot_phot_channels(phot_data, chan_to_plot, tlim):

    data = extract_photometry_data_from_array(phot_data)
    t = phot_data['t']
    num_samples = len(t)

    # find the index of the last timestamp before tlim[0]
    start_idx = max((next(i for i in range(num_samples) if phot_data['t'][i] > tlim[0]) - 1, 0))
    # find the index of the first timestamp after tlim[1]
    if tlim[1] > t[-1]:
        end_idx = num_samples
    else:
        end_idx = next(i for i in range(num_samples) if phot_data['t'][i] > tlim[1])

    for chan in chan_to_plot:
        plt.figure(chan)
        data_to_plot = extract_photometry_data_from_array(phot_data, chan)
        plt.plot(t[start_idx:end_idx], data_to_plot[start_idx:end_idx])


def extract_event_timestamps(phot_data, chan_num, thresh=1.5):

    chan_data = extract_photometry_data_from_array(phot_data, chan_num)
    Fs = phot_data['Fs']

    # find all values above thresh and convert booleans to int's (0's and 1's)
    binary_data = np.multiply((np.array(chan_data) > thresh), 1)

    diff = binary_data[1:] - binary_data[:-1]

    # now all values > 0 are onset of pulse, all values < 0 are offset of pulse
    onset_idx = np.where(diff == 1)
    onset_idx = onset_idx[0] + 1
    offset_idx = np.where(diff == -1)
    offset_idx = offset_idx[0] + 1

    ts_on = onset_idx / Fs
    ts_off = offset_idx / Fs

    return ts_on, ts_off


def extract_perievent_signal(data, ts, t_win, Fs, baseline_window=(-10, 10)):
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

    perievent_dff = np.array(perievent_dff)
    perievent_zscore = np.array(perievent_zscore)

    return perievent_dff, perievent_zscore


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

    if b > d:
        # exp2_fit = c * np.exp(t_floats * d)
        dc_offset = a
    else:
        # exp2_fit = a * np.exp(t_floats * b)
        dc_offset = c

    exp2_fit = a * np.exp(t_floats * b) + c * np.exp(t_floats * d)

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

    # whichever of b or d is closest to zero represents the offset
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


def extract_photometry_data_from_array(phot_data, chan_num=0):

    data = phot_data['data'][:, chan_num]

    return data


def photodetrend(phot_data, phot_channel=0):
    '''

    :param phot_data:
    :param phot_channel:
    :return: photo_detrend2 - raw data minus double exponential fit
    '''
    # assumes photometry signal was recorded on the first channel
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

    # fit exponential to photo_detrend2
    # p0_exp1 = [1, -1]
    # popt1, pcov1 = curve_fit(exp1_func, t_ints, photo_detrend2, p0_exp1)
    # exp1_fit = calc_exp1_fit(num_samples, popt1)
    # photo_detrend3 = (photo_detrend2 - exp1_fit) + np.mean(exp1_fit)

    # exp2_fig = plt.figure(1)
    # plt.plot(t_ints, photo_detrend2)
    # plt.plot(t_ints, photo_detrend3)
    # plt.plot(t_ints, photo_detrend3 - dc_offset)

    # exp1_fig = plt.figure(2)
    # plt.plot(t_ints, data)
    # plt.plot(t_ints, photo_detrend1)
    # plt.plot(t_ints, exp1_offset_fit)
    #
    # compare_fig = plt.figure(3)
    # plt.plot(t_ints, photo_detrend1 - photo_detrend3)

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

    # this is left over from trying to directly fit the offset. Decided to use Christian's method
    # of fitting a double exponential, subtracting out the fast decay, then fitting a single exponential

    # p0_exp1_offset = (1, -1, 1)
    # popt_offset, pcov = curve_fit(exp1_offset_func, t_ints, data, p0_exp1_offset)
    # print('succesfully fit single exponential!')

    # return popt_offset, popt2

    return popt, exp2_fit_successful


def calc_dff(phot_data, smooth_window=101, f0_pctile=50):

    detrended_data, popt, dc_offset, exp2_fit_successful = photodetrend(phot_data)

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

    # plt.figure(3)
    # plt.plot(detrended_data)
    # plt.plot(smoothed_data)
    # plt.plot(dff)

    return detrended_data, dc_offset, dff, popt, exp2_fit_successful


def calc_local_dff(phot_data, smooth_window=101, f0_pctile=50):

    detrended_data, popt, dc_offset, exp2_fit_successful = photodetrend(phot_data)

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

    # plt.figure(3)
    # plt.plot(detrended_data)
    # plt.plot(smoothed_data)
    # plt.plot(dff)

    return detrended_data, dc_offset, dff, popt, exp2_fit_successful


def exp1_offset_func(t, a, b, c):
    return a * np.exp(b * t) + c


def exp2_func(t, a, b, c, d):
    return a * np.exp(b * t) + c * np.exp(d * t)


def exp1_func(t, a, b):
    return a * np.exp(b * t)


def smooth(data, span):

    # data: NumPy 1-D array containing the data to be smoothed
    # span: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(data, np.ones(span, dtype=int), 'valid') / span
    r = np.arange(1, span - 1, 2)
    start = np.cumsum(data[:span - 1])[::2] / r
    stop = (np.cumsum(data[:-span:-1])[::2] / r)[::-1]

    return np.concatenate((start, out0, stop))


def get_photometry_events(phot_data, task_name, phot_metadata):


    if task_name.lower() == 'pavlovian':
        eventlist = ['tone', 'pellet']
        event_ts = extract_tone_pellet_FED_ts(phot_data, chan_num=1)

    elif task_name == 'skilledreaching':
        eventlist, event_ts = extract_sr_ts_from_phot_data(phot_data, phot_metadata)

    elif task_name == 'openfield-crimson':

        eventlist, event_ts = extract_crimson_events(phot_data, phot_metadata)

    return eventlist, event_ts


def extract_crimson_events(phot_data, phot_metadata):

    if phot_metadata['session_datetime'] < datetime(2022, 5, 4):
        rlight_chan_num = 1
    else:
        rlight_chan_num = 1

    TTL_data = extract_photometry_data_from_array(phot_data, chan_num=rlight_chan_num)
    burst_on_idx, burst_off_idx, pulse_on_idx, aligned_pulse_off_idx, intraburst_freq = identify_TTL_pulses_and_bursts(
        TTL_data, phot_data['Fs'])

    #todo: rewrite to account for different pulse frequencies
    # todo: separate bursts by intraburst frequency
    burst_on_ts = burst_on_idx / phot_data['Fs']
    burst_off_ts = burst_off_idx / phot_data['Fs']
    frequency_specific_bursts = sort_bursts_by_frequency(burst_on_ts, burst_off_ts, intraburst_freq)

    # from inside sort_bursts_by_frequency function
    # frequency_specific_bursts = {'intraburst_frequency': unique_freqs,
    #                              'freq_specific_burst_on_ts': freq_specific_burst_on_ts,
    #                              'freq_specific_burst_off_ts': freq_specific_burst_off_ts
    #                              }
    eventlist = ['{:.3} Hz on'.format(bf) for bf in frequency_specific_bursts['intraburst_frequency']]
    event_ts = [ts_on for ts_on in frequency_specific_bursts['freq_specific_burst_on_ts']]

    return eventlist, event_ts


def extract_sr_ts_from_phot_data(phot_data, phot_metadata, TTL_thresh=2.5):
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
    eventlist = ['rear_photobeam', 'paw_through_slot', 'actuator3', 'actuator2', 'vid_trigger']
    num_events = len(eventlist)
    event_ts = []
    for i_event, eventname in enumerate(eventlist):
        nidaq_chan = map_event_to_nidaq_channel(eventname, phot_metadata)

        TTL_data = extract_photometry_data_from_array(phot_data, nidaq_chan)

        TTL_on = np.diff(TTL_data, prepend=False) > TTL_thresh
        # TTL_off = np.diff(TTL_data, prepend=False) < -TTL_thresh

        if eventname == 'rear_photobeam':
            # exclude photobeam events too close together in time
            pass

        pulse_on_idx = np.squeeze(np.argwhere(TTL_on))
        # pulse_off_idx = np.squeeze(np.argwhere(TTL_off))

        event_ts.append(pulse_on_idx / phot_data['Fs'])

    return eventlist, event_ts


def map_event_to_nidaq_channel(eventname, phot_metadata):
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

    if eventname == 'rear_photobeam':
        if phot_metadata['session_datetime'] < datetime(2022, 4, 27):
            nidaq_chan = 1
        else:
            nidaq_chan = 5
    elif eventname == 'paw_through_slot':
        nidaq_chan = 2
    elif eventname == 'actuator3':
        nidaq_chan = 3
    elif eventname == 'actuator2':
        nidaq_chan = 4
    elif eventname == 'nose_trigger':   # nose_trigger not in use right now
        nidaq_chan = 5
    elif eventname == 'vid_trigger':
        nidaq_chan = 6
    elif eventname == 'frame_trigger':
        nidaq_chan = 7
    else:
        nidaq_chan = None

    return nidaq_chan


def photometry_session_summary_sheet(photometry_file, save_folder, task_name, smooth_window=201, perievent_window=(-3, 3)):
    '''

    :param photometry_file: full path to the photometry file being analyzed
    :param save_folder:
    :param task_name:
    :param smooth_window:
    :return:
    '''

    phot_metadata = skilled_reaching_io.parse_photometry_fname(photometry_file)

    s_folder, phot_fname = os.path.split(photometry_file)
    _, s_folder_name = os.path.split(s_folder)
    session_name = s_folder_name

    save_name = session_name + '_summarysheet_{:d}secwindows.jpg'.format(perievent_window[1])
    save_name = os.path.join(save_folder, save_name)
    if os.path.exists(save_name):
        print(save_name + ' already exists!')
        return

    phot_data = skilled_reaching_io.read_photometry_mat(photometry_file)
    detrended_data, dc_offset, dff, popt, exp2_fit_successful = calc_dff(phot_data, smooth_window=smooth_window,
                                                                         f0_pctile=10)
    eventlist, event_ts = get_photometry_events(phot_data, task_name, phot_metadata)
    num_events = len(eventlist)

    fig_cols = num_events
    session_fig, session_axes = photometry_analysis_plots.create_single_session_summary_panels(8.5, 11, fig_cols)
    session_fig.suptitle(session_name + ', smoothing window = {:d} points'.format(smooth_window))

    analysis_window = [0., phot_data['t'][-1][0]]

    t = phot_data['t']
    data = extract_photometry_data_from_array(phot_data, chan_num=0)
    num_samples = len(data)
    photometry_analysis_plots.plot_phot_signal(t, data, analysis_window, session_axes['row1'], ylim=[0.01, 0.08])

    # overlay double (or offset single) exponential fit on raw signal
    if exp2_fit_successful:
        exp_fit, dc_offset = calc_exp2_fit(num_samples, popt)
    else:
        exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, popt)
    session_axes['row1'].plot(t, exp_fit)
    session_axes['row1'].set_title('raw data')
    photometry_analysis_plots.eliminate_x_labels(session_axes['row1'])
    session_axes['row1'].legend(['raw data', 'exp2 fit'])

    photometry_analysis_plots.plot_phot_signal(t, (data - exp_fit) + exp_fit[-1], analysis_window, session_axes['row2'],
                                               ylim=[0.01, 0.08])
    session_axes['row2'].set_title('(raw data - exp2 fit) + asymptote')
    photometry_analysis_plots.eliminate_x_labels(session_axes['row2'])

    photometry_analysis_plots.plot_phot_signal(t, dff, analysis_window, session_axes['row3'],
                                               ylim=[0.01, 0.08])

    if task_name.lower() == 'pavlovian':
        TTL = extract_photometry_data_from_array(phot_data, chan_num=1)
    elif task_name.lower() == 'skilledreaching':
        TTL = extract_photometry_data_from_array(phot_data, chan_num=6)
    elif task_name.lower() == 'openfield-crimson':
        TTL = extract_photometry_data_from_array(phot_data, chan_num=1)

    photometry_analysis_plots.overlay_TTL(t, TTL, session_axes['row3'])

    session_axes['row3'].set_title('dff')
    session_axes['row3'].set_xlabel('time (s)')

    # ts_on, ts_off = extract_event_timestamps(phot_data, chan_num=1)
    #
    # # going to try locking analysis to every other TTL pulse because we think the first one is the tone, the second one is pellet retrieval
    # # extract dFF in windows around relevant events
    # odd_ts_on = ts_on[0::2]
    # even_ts_on = ts_on[1::2]


    perievent_dff = []
    perievent_zscore = []
    baseline_window = (-10, 10)

    for i_event in range(num_events):
        if event_ts[i_event] is not None:
            p_dff, p_zscore = extract_perievent_signal(dff, event_ts[i_event], perievent_window, phot_data['Fs'], baseline_window=baseline_window)
            perievent_dff.append(p_dff)
            perievent_zscore.append(p_zscore)
        else:
            perievent_dff.append(None)
            perievent_zscore.append(None)

    for i_col in range(fig_cols):
        axes_name = 'row4_col{:d}'.format(i_col)
        if perievent_dff[i_col] is not None:
            photometry_analysis_plots.plot_mean_perievent_signal(perievent_dff[i_col], perievent_window,
                                                                 phot_metadata, eventlist[i_col],
                                                                 ax=session_axes[axes_name], color='g')

            axes_name = 'row5_col{:d}'.format(i_col)
            photometry_analysis_plots.plot_mean_perievent_signal(perievent_zscore[i_col], perievent_window,
                                                                 phot_metadata, eventlist[i_col],
                                                                 ax=session_axes[axes_name], color='g')
            session_axes[axes_name].set_xlabel('time (s)')

    plt.savefig(save_name)
    plt.close(session_fig)


def create_photometry_session_summary_sheets(ratID, photometry_parent, task_name, dest_folder, smooth_window=201):

    session_folders = skilled_reaching_io.get_session_folders(ratID, photometry_parent)

    save_folder = os.path.join(dest_folder, 'summary_sheets', ratID)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i_folder, s_folder in enumerate(session_folders):

        photometry_file = skilled_reaching_io.find_photometry_file_from_session_folder(s_folder, task_name)
        if photometry_file is None:
            continue
        phot_metadata = skilled_reaching_io.parse_photometry_fname(photometry_file)

        if task_name.lower() == 'pavlovian':
            FED_file = skilled_reaching_io.find_session_FED_file(photometry_parent, phot_metadata)
            if FED_file == None:
                FED_data = None
            elif os.path.exists(FED_file):
                FED_data = skilled_reaching_io.read_FED_csv(FED_file)
            else:
                FED_data = None

        if not os.path.exists(photometry_file):
            continue

        photometry_session_summary_sheet(photometry_file, save_folder, task_name, smooth_window=smooth_window)
            # if summary sheet already exists, skip this one. If need a new sheet, delete/move the old ones


def extract_tone_pellet_FED_ts(phot_data, chan_num=1, TTL_thresh=1.5, pw_cutoff=225):

    TTL_data = extract_photometry_data_from_array(phot_data, chan_num)

    TTL_on = np.diff(TTL_data, prepend=False) > TTL_thresh
    TTL_off = np.diff(TTL_data, prepend=False) < -TTL_thresh

    pulse_on_idx = np.squeeze(np.argwhere(TTL_on))
    pulse_off_idx = np.squeeze(np.argwhere(TTL_off))

    # makes sure pulse_off_idx starts after pulse_on_idx
    aligned_pulse_off_idx = align_on_off_idx(pulse_on_idx, pulse_off_idx)

    if np.size(aligned_pulse_off_idx) == np.size(pulse_on_idx):
        pw = aligned_pulse_off_idx - pulse_on_idx

        if min(pw) < 150:
            # in earlier sessions, I think tone was a 144 ms pulse, pellet = 200; in later sessions I think tone
            # was 200 ms, pellet = 250
            pw_cutoff = 175
        else:
            pw_cutoff = 225

        # test if pulse_on_idx has at least two elements
        if np.size(pulse_on_idx) > 1:
            pulse_off_after_first_on = np.squeeze(np.argwhere(pulse_off_idx > pulse_on_idx[0]))
        elif np.size(pulse_on_idx) == 1:
            pulse_off_after_first_on = np.squeeze(np.argwhere(pulse_off_idx > pulse_on_idx))
        else:
            # pulse_on_idx is empty
            return None


        if np.size(pulse_on_idx) > 1:
            tone_event_idx = pulse_on_idx[pw < pw_cutoff]
            pellet_event_idx = pulse_on_idx[pw > pw_cutoff]
        elif np.size(pulse_on_idx) == 1:
            if pw[0] < pw_cutoff:
                tone_event_idx = np.array(pulse_on_idx)
                pellet_event_idx = None
            else:
                pellet_event_idx = np.array(pulse_on_idx)
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


def identify_TTL_pulses_and_bursts(TTL_data, Fs, TTL_thresh=2., inter_burst_thresh=1000):
    '''

    :param TTL_data: numpy array with data stream for TTL pulses - this should be raw voltage data
    :param Fs: data sampling rate in Hz
    :param TTL_thresh: voltage threshold for identifying a TTL pulse
    :param inter_burst_thresh: minimum number of samples between "on" events to consider a grouping of TTL pulses a
        separate "burst"
    :return:
    '''

    TTL_on = np.diff(TTL_data, prepend=False) > TTL_thresh
    TTL_off = np.diff(TTL_data, prepend=False) < -TTL_thresh

    pulse_on_idx = np.squeeze(np.argwhere(TTL_on))
    pulse_off_idx = np.squeeze(np.argwhere(TTL_off))

    # makes sure pulse_off_idx starts after pulse_on_idx
    aligned_pulse_off_idx = align_on_off_idx(pulse_on_idx, pulse_off_idx)
    burst_on_idx, burst_off_idx, intraburst_freq = detect_bursts(pulse_on_idx, aligned_pulse_off_idx, Fs, inter_burst_thresh=inter_burst_thresh)

    # verify correct pulse/burst identification
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(TTL_data, color='k')
    # ones_array = np.ones((len(burst_on_idx)))
    # ax.plot(burst_on_idx, ones_array, color='g', linestyle='none', marker='*')
    # ax.plot(burst_off_idx, ones_array, color='r', linestyle='none', marker='*')
    # ax.legend()
    # plt.show()

    return burst_on_idx, burst_off_idx, pulse_on_idx, aligned_pulse_off_idx, intraburst_freq


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
        pulse_off_after_first_on = np.squeeze(np.argwhere(pulse_off_idx > pulse_on_idx))
    else:
        # pulse_on_idx is empty
        return None

    if np.size(pulse_off_after_first_on) > 0:
        if np.size(pulse_off_after_first_on) > 1:
            pulse_off_start_idx = pulse_off_after_first_on[0]
        elif np.size(pulse_off_after_first_on) == 1:
            pulse_off_start_idx = pulse_off_after_first_on
        pulse_off_idx = pulse_off_idx[pulse_off_start_idx:]
    else:
        pulse_off_idx = None

    return pulse_off_idx


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

        pulse_idx_diffs = np.diff(burst_pulse_indices)
        interpulse_inverals = pulse_idx_diffs / Fs

        intraburst_freq[i_burst] = round(1 / np.mean(interpulse_inverals))   # assumes integer pulse frequency to allow for inter-pulse intervals being off by 1 sample here and there

    return burst_on_idx, burst_off_idx, intraburst_freq


def photometry_summary_sheets(ratID, photometry_parent, task_name, eventlist, dest_folder):

    session_folders = skilled_reaching_io.get_session_folders(ratID, photometry_parent)

    num_folders = len(session_folders)

    mean_fig_rows = 5
    mean_fig_cols = 3
    plots_per_sheet = mean_fig_cols * mean_fig_rows

    meandff_fig, meandff_axes = photometry_analysis_plots.create_axes_panels(8.5, 11, mean_fig_rows, mean_fig_cols)

    cur_plot_num = 0
    num_sheets = 1
    meandff_fig.suptitle(ratID + ', sheet {:d}'.format(num_sheets))
    for i_folder, s_folder in enumerate(session_folders):
        photometry_file = skilled_reaching_io.find_photometry_file_from_session_folder(s_folder, task_name)

        if photometry_file is None:
            if cur_plot_num == plots_per_sheet - 1 or i_folder == num_folders - 1:
                fname = os.path.join(dest_folder,
                                     ratID + '_' + task_name + '_meandff_sheet{:02d}.pdf'.format(num_sheets))

                plt.savefig(fname)
                plt.close(meandff_fig)

                cur_plot_num = 0
                num_sheets += 1
                if i_folder < num_folders - 1:
                    # meandff_fig, meandff_axes = photometry_analysis_plots.create_axes_panels(8.5, 11, mean_fig_rows,
                    #                                                                          mean_fig_cols)
                    session_fig, session_axes = photometry_analysis_plots.create_single_session_summary_panels(8.5, 11,
                                                                                                               mean_fig_rows,
                                                                                                               mean_fig_cols)
                # cur_row = cur_plot_num % mean_fig_rows
                # cur_col = int(np.floor(cur_plot_num / mean_fig_rows))
            cur_plot_num += 1
            continue

        phot_metadata = skilled_reaching_io.parse_photometry_fname(photometry_file)

        phot_data = skilled_reaching_io.read_photometry_mat(photometry_file)

        if phot_data is None:
            if cur_plot_num == plots_per_sheet - 1 or i_folder == num_folders - 1:
                fname = os.path.join(dest_folder,
                                     ratID + '_' + task_name + '_meandff_sheet{:02d}.pdf'.format(num_sheets))

                plt.savefig(fname)
                plt.close(meandff_fig)

                cur_plot_num = 0
                num_sheets += 1
                if i_folder < num_folders - 1:
                    # meandff_fig, meandff_axes = photometry_analysis_plots.create_axes_panels(8.5, 11, mean_fig_rows,
                    #                                                                          mean_fig_cols)
                    session_fig, session_axes = photometry_analysis_plots.create_single_session_summary_panels(8.5, 11,
                                                                                                               mean_fig_rows,
                                                                                                               mean_fig_cols)
                    meandff_fig.suptitle(ratID + ', sheet {:d}'.format(num_sheets))
                # cur_row = cur_plot_num % mean_fig_rows
                # cur_col = int(np.floor(cur_plot_num / mean_fig_rows))
            cur_plot_num += 1
            continue

        # calculate dff
        detrended_data, dc_offset, dff, _, _, _ = calc_dff(phot_data, smooth_window=smooth_window, f0_pctile=f0_pctile)
        if dff is None:
            if cur_plot_num == plots_per_sheet - 1 or i_folder == num_folders - 1:
                fname = os.path.join(dest_folder,
                                     ratID + '_' + task_name + '_meandff_sheet{:02d}.pdf'.format(num_sheets))

                plt.savefig(fname)
                plt.close(meandff_fig)

                cur_plot_num = 0
                num_sheets += 1
                meandff_fig, meandff_axes = photometry_analysis_plots.create_axes_panels(8.5, 11, mean_fig_rows,
                                                                                         mean_fig_cols)

                meandff_fig.suptitle(ratID + ', sheet {:d}'.format(num_sheets))
                cur_row = cur_plot_num % mean_fig_rows
                cur_col = int(np.floor(cur_plot_num / mean_fig_rows))
            cur_plot_num += 1
            continue

        # chan_num 1 should be the tone TTL
        ts_on, ts_off = extract_event_timestamps(phot_data, chan_num=1)

        # extract dFF in windows around relevant events
        perievent_data = extract_perievent_signal(dff, ts_on, analysis_window, phot_data['Fs'])

        # create a sheet with all peri-event signals for a session

        # create summary sheets with mean peri-event dFF for multiple sessions
        if cur_plot_num == plots_per_sheet-1 or i_folder == num_folders-1:
            fname = os.path.join(dest_folder, ratID + '_' + task_name + '_meandff_sheet{:02d}.pdf'.format(num_sheets))

            plt.savefig(fname)
            plt.close(meandff_fig)

            cur_plot_num = 0
            num_sheets += 1
            if i_folder < num_folders - 1:
                meandff_fig, meandff_axes = photometry_analysis_plots.create_axes_panels(8.5, 11, mean_fig_rows,
                                                                                         mean_fig_cols)
                meandff_fig.suptitle(ratID + ', sheet {:d}'.format(num_sheets))
            cur_row = cur_plot_num % mean_fig_rows
            cur_col = int(np.floor(cur_plot_num / mean_fig_rows))
        else:
            cur_row = cur_plot_num % mean_fig_rows
            cur_col = int(np.floor(cur_plot_num / mean_fig_rows))
            cur_plot_num += 1



        photometry_analysis_plots.plot_mean_perievent_signal(perievent_data,
                                                             analysis_window,
                                                             phot_metadata,
                                                             show_errors='sem',
                                                             ax=meandff_axes[cur_row, cur_col],
                                                             ylim=[0.01, 0.10])

        if cur_row < mean_fig_rows-1:
            photometry_analysis_plots.eliminate_x_labels(meandff_axes[cur_row, cur_col])
        if cur_col > 0:
            photometry_analysis_plots.eliminate_y_labels(meandff_axes[cur_row, cur_col])


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


def create_crimson_session_summary_sheets(ratID, chrimson_parent, dest_folder, light_TTL_chan=6, smooth_window=501):
    '''
    pulses are individual TTL pulses, bursts are clustered groups of TTL pulses
    :param fname:
    :return:
    '''

    session_folders = skilled_reaching_io.get_session_folders(ratID, chrimson_parent)

    save_folder = os.path.join(dest_folder, 'summary_sheets', ratID)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig_rows = 5



    for i_folder, s_folder in enumerate(session_folders):

        crimson_files, pavlov_files = skilled_reaching_io.find_chrimson_files_from_session_folder(s_folder)

        if crimson_files is not None:
            for crimson_file in crimson_files:

                phot_data = skilled_reaching_io.read_photometry_mat(crimson_file)
                phot_metadata = skilled_reaching_io.parse_photometry_fname(crimson_file)

                file_savename = '_'.join((ratID,
                                          skilled_reaching_io.datetime_string(phot_metadata['session_datetime']),
                                          'crimson',
                                          'summarysheet.jpg'))
                save_name = os.path.join(save_folder, file_savename)

                eventlist, event_ts = get_photometry_events(phot_data, task_name, phot_metadata)

                fig_cols = len(eventlist)

                # pulse_on_ts = pulse_on_idx / phot_data['Fs']
                # aligned_pulse_off_ts = aligned_pulse_off_idx / phot_data['Fs']

                detrended_data, dc_offset, dff, popt, exp2_fit_successful = calc_dff(phot_data, smooth_window=smooth_window,
                                                                                     f0_pctile=f0_pctile)

                session_fig, session_axes = photometry_analysis_plots.create_single_session_summary_panels(8.5, 11, fig_cols)

                fig_title = '_'.join((ratID,
                                      skilled_reaching_io.datetime_string(phot_metadata['session_datetime']),
                                      'chrimson, smoothing window = {:d} points'.format(smooth_window)))
                session_fig.suptitle(fig_title)

                analysis_window = [0., phot_data['t'][-1][0]]

                t = phot_data['t']
                data = extract_photometry_data_from_array(phot_data, chan_num=0)
                num_samples = len(data)
                photometry_analysis_plots.plot_phot_signal(t, data, analysis_window, session_axes['row1'],
                                                           ylim=[0.01, 0.08])

                # overlay double (or offset single) exponential fit on raw signal
                if exp2_fit_successful:
                    exp_fit, dc_offset = calc_exp2_fit(num_samples, popt)
                else:
                    exp_fit, dc_offset = calc_exp1_offset_fit(num_samples, popt)
                session_axes['row1'].plot(t, exp_fit)
                session_axes['row1'].set_title('raw data')
                photometry_analysis_plots.eliminate_x_labels(session_axes['row1'])
                session_axes['row1'].legend(['raw data', 'exp2 fit'])

                photometry_analysis_plots.plot_phot_signal(t, (data - exp_fit) + exp_fit[-1], analysis_window,
                                                           session_axes['row2'],
                                                           ylim=[0.01, 0.08])
                session_axes['row2'].set_title('(raw data - exp2 fit) + asymptote')
                photometry_analysis_plots.eliminate_x_labels(session_axes['row2'])

                photometry_analysis_plots.plot_phot_signal(t, dff, analysis_window, session_axes['row3'],
                                                           ylim=[0.01, 0.08])

                ylimits = session_axes['row3'].get_ylim()
                for bon_ts in burst_on_ts:
                    session_axes['row3'].plot([bon_ts,bon_ts],ylimits,color='g')
                for boff_ts in burst_off_ts:
                    session_axes['row3'].plot([boff_ts,boff_ts],ylimits,color='r')


                # photometry_analysis_plots.overlay_TTL(t, TTL, session_axes['row3'])

                peri_event_windows = [(-5, 5), (-5, 5)]
                perievent_dff = []
                perievent_zscore = []
                # perievent_data.append(extract_perievent_signal(dff, burst_on_ts, peri_event_windows[0], phot_data['Fs']))
                # perievent_data.append(extract_perievent_signal(dff, burst_off_ts, peri_event_windows[1], phot_data['Fs']))
                for i_bf, _ in enumerate(frequency_specific_bursts['intraburst_frequency']):
                    p_dff, p_zscore = extract_perievent_signal(dff, frequency_specific_bursts['freq_specific_burst_on_ts'][i_bf], peri_event_windows[i_bf], phot_data['Fs'])
                    perievent_dff.append(p_dff)
                    perievent_zscore.append(p_zscore)

                for i_col in range(fig_cols):
                    axes_name = 'row4_col{:d}'.format(i_col)
                    photometry_analysis_plots.plot_mean_perievent_signal(perievent_dff[i_col], peri_event_windows[i_col], phot_metadata,
                                                                         eventlist[i_col], ax=session_axes[axes_name], color='g')

                    axes_name = 'row5_col{:d}'.format(i_col)
                    photometry_analysis_plots.plot_mean_perievent_signal(perievent_zscore[i_col], peri_event_windows[i_col], phot_metadata,
                                                                         eventlist[i_col], ax=session_axes[axes_name], color='g')
                    session_axes[axes_name].set_xlabel('time (s)')

                # plt.show()
                plt.savefig(save_name)
                plt.close(session_fig)

        if pavlov_files is not None:
            for pavlov_file in pavlov_files:

                eventlist = ['tone', 'pellet']
                task_name = 'pavlovian'
                photometry_session_summary_sheet(pavlov_file, save_folder, task_name, eventlist, smooth_window=201)


def segment_photometry_signal(phot_data, phot_chan):

    #todo: create a version where photometry signal is segmented
    pass


if __name__ == '__main__':
    # parameters for calculating dF/F

    # TTL_lines{'tone': 1,
    #           }
    smooth_window = 201
    f0_pctile = 50

    rat_list = [428, 429, 437, 438, 439]

    analysis_window = (-2.0, 2.0)
    
    crimson_analysis_window = (-2.0, 5.0)
    # fname = '/Volumes/Untitled/photometry_data/crimson_tests/R0433/R0433_20220408_chrimsontest/R0433_20220408_16-21-50_nidaq_open_field_chrimson.mat'
    # analyze_crimson_data(fname)

    # photometry_parent = '/Users/dan/Documents/photometry_analysis'
    # photometry_parent = '/Volumes/Untitled/photometry_data'
    photometry_parent = 'C:\\Users\\dklev\\Dropbox (University of Michigan)\\MED-LeventhalLab\\data\\dLight_photometry'

    crimson_parent = os.path.join(photometry_parent, 'crimson')

    for rat_num in rat_list:
        ratID = 'R{:04d}'.format(rat_num)
        dest_folder = photometry_parent
        create_photometry_session_summary_sheets(ratID, photometry_parent, 'pavlovian', dest_folder, smooth_window=smooth_window)
        create_photometry_session_summary_sheets(ratID, photometry_parent, 'skilledreaching', dest_folder,
                                                 smooth_window=smooth_window)
        create_photometry_session_summary_sheets(ratID, photometry_parent, 'openfield-crimson', dest_folder,
                                                 smooth_window=smooth_window)

        dest_folder = crimson_parent
        # create_crimson_session_summary_sheets(ratID, chrimson_parent, dest_folder, smooth_window=smooth_window)
        # photometry_summary_sheets(ratID, photometry_parent, 'pavlovian', ['tone'], dest_folder)

    # fname = 'R0402_20210729_13-48-25_nidaq_openfield.mat'

    phot_metadata = skilled_reaching_io.parse_photometry_fname(fname)

    FED_full_path = skilled_reaching_io.find_session_FED_file(photometry_parent, phot_metadata)
    FED_data = skilled_reaching_io.read_FED_csv(FED_full_path)

    phot_fname = skilled_reaching_io.find_session_photometry_file(photometry_parent, phot_metadata)

    phot_data = skilled_reaching_io.read_photometry_mat(phot_fname)

    # plot_phot_channels(phot_data, [1], [-1,600000])
    ts_on, ts_off = extract_event_timestamps(phot_data, chan_num=1)
    #todo: figure out what these timestamps correspond to in the .csv file

    # calculate dff
    detrended_data, dc_offset, dff, popt, exp2_fit_successful = calc_dff(phot_data, smooth_window=smooth_window, f0_pctile=f0_pctile)

    # extract dFF in windows around relevant events
    perievent_data = extract_perievent_signal(dff, ts_on, analysis_window, phot_data['Fs'])

    if perievent_data is not None:
        photometry_analysis_plots.plot_mean_perievent_signal(perievent_data, analysis_window, color='g')
        tlim = (min(phot_data['t']), max(phot_data['t']))
        tlim = (-1, 100000)
        plot_phot_channels(phot_data, range(0, 8), tlim)
    pass


def read_photometry_mat(full_path):
    '''
    function to read a .mat file containing photometry data
    assumed to be 8-channel data

    return: dictionary with the following keys
        Fs - float giving the sampling rate in Hz
        current - current applied to the LED
        data - n x 8 numpy array containing the raw data; channel 0 is typically the photometry signal
    '''
    try:
        photometry_data = sio.loadmat(full_path)
    except ValueError:
        # in case of corrupted mat file
        return None

    # reformat into a dictionary for easier use later

    # in some versions, 'current' and 'virus' aren't included in the file
    try:
        phot_data = {
            'Fs': float(photometry_data['Fs'][0][0]),
            'current': photometry_data['current'][0][0],
            'data': photometry_data['data'],
            't': photometry_data['timeStamps'],
            'virus': photometry_data['virus'][0],
            'AI_line_desc': photometry_data['AI_line_desc'][0],
            'cam_trigger_delay': photometry_data['cam_trigger_delay'][0][0],
            'cam_trigger_freq': photometry_data['cam_trigger_freq'][0][0],
            'cam_trigger_pw': photometry_data['cam_trigger_pw'][0][0],
        }
    except KeyError:
        phot_data = {
            'Fs': float(photometry_data['Fs'][0][0]),
            'current': [],
            'data': photometry_data['data'],
            't': photometry_data['timeStamps'],
            'virus': [],
        }

    return phot_data


def read_FED_csv(full_path):
    with open(full_path, newline='') as csvfile:
        FEDreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        FED_datetime = []
        FED_batt_voltage = []
        FED_motor_turns = []
        FED_FR = []
        FED_event = []
        FED_active_poke = []
        FED_left_poke_count = []
        FED_right_poke_count = []
        FED_pellet_count = []
        FED_block_pellet_count = []
        FED_retrieval_time = []
        FED_poke_time = []
        for i_row, row in enumerate(FEDreader):
            if i_row == 0:
                # this is the header row
                pass
            else:
                FED_datetime.append(datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S'))
                FED_version = row[1]
                FED_task = row[2]
                FED_devnum = int(row[3])
                FED_batt_voltage.append(float(row[4]))
                FED_motor_turns.append(int(row[5]))
                FED_FR = int(row[6])  # what does 'FR' stand for?
                FED_event.append(row[7])
                FED_active_poke.append(row[8])
                FED_left_poke_count.append(row[9])
                FED_right_poke_count.append(row[10])
                FED_pellet_count.append(row[11])
                FED_block_pellet_count.append(row[12])
                FED_retrieval_time.append(row[13])
                FED_poke_time.append(row[14])

        FED_data = {
            'datetime': FED_datetime,
            'version': FED_version,
            'task': FED_task,
            'devnum': FED_devnum,
            'batt_voltage': FED_batt_voltage,
            'motor_turns': FED_motor_turns,
            'FR': FED_FR,
            'event': FED_event,
            'active_poke': FED_active_poke,
            'left_poke_count': FED_left_poke_count,
            'right_poke_count': FED_right_poke_count,
            'pellet_count': FED_pellet_count,
            'block_pellet_count': FED_block_pellet_count,
            'retrieval_time': FED_retrieval_time,
            'poke_time': FED_poke_time
        }

        return FED_data


def get_session_folders(ratID, photometry_parent):
    session_folder_list = glob.glob(os.path.join(photometry_parent, ratID, ratID + '_*'))

    return session_folder_list


def find_chrimson_files_from_session_folder(session_folder):
    base_path, session_folder_name = os.path.split(session_folder)

    chrimson_folder = os.path.join(session_folder, session_folder_name + '_chrimson')
    pavlovian_folder = os.path.join(session_folder, session_folder_name + '_postchrimson-pavlovian')

    if not os.path.exists(chrimson_folder):
        chrimson_files = None
    else:
        chrimson_files = glob.glob(os.path.join(chrimson_folder, '*_openfield-chrimson.mat'))

    if not os.path.exists(pavlovian_folder):
        pavlov_files = None
    else:
        pavlov_files = glob.glob(os.path.join(pavlovian_folder, '*_pavlovian.mat'))

    return chrimson_files, pavlov_files


def find_photometry_file_from_session_folder(session_folder, task_name):
    base_path, session_folder_name = os.path.split(session_folder)
    if str.lower(task_name) == 'skilledreaching':
        task_folder = os.path.join(session_folder, session_folder_name + '_skilledreaching')
        test_name = os.path.join(task_folder, '*_nidaq_skilledreaching.mat')
        photometry_file = glob.glob(test_name)
    elif str.lower(task_name) == 'openfield-crimson':
        task_folder = os.path.join(session_folder, session_folder_name + '_openfield-crimson')
        test_name = os.path.join(task_folder, '*_nidaq_openfield-crimson.mat')
        photometry_file = glob.glob(test_name)
    elif str.lower(task_name) == 'pavlovian':
        task_folder = os.path.join(session_folder, session_folder_name + '_pavlovian')
        test_name = os.path.join(task_folder, '*_nidaq_pavlovian.mat')
        photometry_file = glob.glob(test_name)

    try:
        a = photometry_file[0]
    except IndexError:
        print(test_name + ' could not be found.')
        return None

    return photometry_file[0]


def find_session_folder_from_metadata(photometry_parent, photometry_metadata):
    d_string = date_string(photometry_metadata['session_datetime'])
    dt_string = datetime_string(photometry_metadata['session_datetime'])
    session_date_folder = photometry_metadata['ratID'] + '_' + d_string
    # session_name = photometry_metadata['ratID'] + '_' + d_string + '_' + photometry_metadata['task']
    # session_folder = os.path.join(photometry_parent, photometry_metadata['ratID'], session_date_folder, session_name)

    phot_fname = '_'.join([photometry_metadata['ratID'],
                           dt_string,
                           'nidaq',
                           photometry_metadata['task'] + '.mat'])

    full_file_path = glob.glob(os.path.join(photometry_parent, '**', session_date_folder + '*', phot_fname),
                               recursive=True)

    if len(full_file_path) == 1:
        # found exactly one file in these subdirectories
        session_folder, _ = os.path.split(full_file_path[0])
    else:
        session_folder = None

    return session_folder


def find_session_photometry_file(photometry_parent, photometry_metadata):
    session_folder = find_session_folder_from_metadata(photometry_parent, photometry_metadata)
    dt_string = datetime_string(photometry_metadata['session_datetime'])

    fname = '_'.join([photometry_metadata['ratID'], dt_string, 'nidaq', photometry_metadata['task'] + '.mat'])

    session_photometry_file = os.path.join(session_folder, fname)

    return session_photometry_file


def find_session_FED_file(photometry_parent, photometry_metadata):
    session_folder = find_session_folder_from_metadata(photometry_parent, photometry_metadata)
    d_string = date_string(photometry_metadata['session_datetime'])

    test_name = '_'.join([photometry_metadata['ratID'], d_string, 'FED*.csv'])

    session_FED_list = glob.glob(os.path.join(session_folder, test_name))

    if len(session_FED_list) == 1:
        session_FED_file = session_FED_list[0]
    else:
        session_FED_file = None
    # session_FED_file = os.path.join(session_folder, fname)

    return session_FED_file


def datetime_string(session_datetime):
    dt_string = session_datetime.strftime('%Y%m%d_%H-%M-%S')

    return dt_string


def date_string(session_datetime):
    d_string = session_datetime.strftime('%Y%m%d')

    return d_string


def parse_photometry_fname(full_path):
    _, fname_ext = os.path.split(full_path)
    fname, ext = os.path.splitext(fname_ext)

    fileparts = fname.split('_')

    rat_num = int(fileparts[0][1:])

    datestring = fileparts[1] + '_' + fileparts[2]
    session_datetime = datetime.strptime(datestring, '%Y%m%d_%H-%M-%S')

    photometry_metadata = {
        'ratID': fileparts[0],
        'rat_num': rat_num,
        'session_datetime': session_datetime,
        'task': fileparts[4]
    }

    return photometry_metadata

