import numpy as np
import photometry_rat_sr_navigation as prn
import photometry_io_skilledreaching as io_utils
from scipy import signal


def session_type_from_folders(session_folders, search_metadata):
    '''

    :param session_folders: dictionary of folders where each key is a session type ('sr', 'srchrim', etc) that contains a list of session folders
    :param search_metadata: dictionary containing metadata for the session we're looking for
    :return:
    '''
    session_types = list(session_folders.keys())

    for session_type in session_types:

        for session_folder in session_folders[session_type]:
            folder_metadata = prn.session_metadata_from_path(session_folder)
            if search_metadata['date'] == folder_metadata['date'] and search_metadata['session_num'] == folder_metadata['session_num']:
                return session_type

    return None


def outcomes_by_session(sessions_per_rat, parent_directories):

    # load the sr_scores
    sr_scores_xlsx = prn.find_scores_xlsx(parent_directories, xlname='rat_dlight_photometry_SR_sessions.xlsx')

    ratIDs = list(sessions_per_rat.keys())
    sr_behavior_summary = {}
    for ratID in ratIDs:
        sr_scores = io_utils.read_xlsx_scores(sr_scores_xlsx, ratID)

        if not sr_scores.empty:
            sr_behavior_summary[ratID] = single_rat_outcomes(sessions_per_rat[ratID], sr_scores)

    return sr_behavior_summary


def single_rat_outcomes(rat_sessions, sr_scores):
    outcome_groupings = create_sr_outcome_groupings()
    rat_reaching_stats = {'skilledreaching': {},
                          'srchrimson': {}
                         }

    # need to separate by session type - was chrimson activated?
    for session_metadata in rat_sessions:

        # todo: sort sessions by date to make sure everything's in the right order
        task = session_metadata['task']
        if task in ['skilledreaching', 'srchrimson']:
            trials_by_outcome, valid_trials = extract_sr_trials_by_outcome(sr_scores, session_metadata, outcome_groupings)

            if valid_trials is None:
                continue

            session_stats = calculate_sessions_stats(trials_by_outcome, valid_trials)
            reachstat_labels = list(session_stats.keys())

            session_name = outcome_pd_header_from_session_metadata(session_metadata)
            if len(rat_reaching_stats[task]) == 0:
                rat_reaching_stats[task]['session_name'] = [session_name]
                rat_reaching_stats[task]['date'] = [session_metadata['date']]
                for stat_label in reachstat_labels:
                    rat_reaching_stats[task][stat_label] = [session_stats[stat_label]]
            else:
                rat_reaching_stats[task]['session_name'].append(session_name)
                rat_reaching_stats[task]['date'].append(session_metadata['date'])
                for stat_label in reachstat_labels:
                    rat_reaching_stats[task][stat_label].append(session_stats[stat_label])

    return rat_reaching_stats



def calculate_sessions_stats(trials_by_outcome, valid_trials):
    '''

    :param trials_by_outcome:
    :param valid_trials:
    :return:
    '''
    num_reachtrials = len(valid_trials) - len(trials_by_outcome['no_reach'])  # note no_reach includes nontrials (mechanical errors, etc)
    if num_reachtrials < 0:
        pass

    num_firstsuccess = len(trials_by_outcome['first_success'])
    num_anysuccess = len(trials_by_outcome['any_success_prefpaw'])
    num_multisuccess = num_anysuccess - num_firstsuccess

    num_wrongpaw = len(trials_by_outcome['used_contra'])

    if num_reachtrials == 0:
        # avoid division by zero
        session_stats = {
                         'num_reachtrials': num_reachtrials,
                         'num_firstsuccess': num_firstsuccess,
                         'num_anysuccess': num_anysuccess,
                         'num_multisuccess': num_multisuccess,
                         'num_wrongpaw': num_wrongpaw,
                         'firstsuccess_rate': np.NaN,
                         'anysuccess_rate': np.NaN,
                         'multisuccess_rate': np.NaN,
                         'wrongpaw_rate': np.NaN
                         }
    else:
        session_stats = {
                         'num_reachtrials': num_reachtrials,
                         'num_firstsuccess': num_firstsuccess,
                         'num_anysuccess': num_anysuccess,
                         'num_multisuccess': num_multisuccess,
                         'num_wrongpaw': num_wrongpaw,
                         'firstsuccess_rate': num_firstsuccess / num_reachtrials,
                         'anysuccess_rate': num_anysuccess / num_reachtrials,
                         'multisuccess_rate': num_multisuccess / num_reachtrials,
                         'wrongpaw_rate': num_wrongpaw / num_reachtrials
                         }

    return session_stats


def extract_sr_trials_by_outcome(sr_scores, session_metadata, outcome_groupings, outcome_col='session_name'):
    '''
    :param sr_scores:
    :param outcome_groupings: dictionary where each key is the trial type descriptor and the value is a list
                              of outcome scores that go with that outcome type
    :return:

    trial outcomes key as of 3/5/2023
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
    # assume column with scores is the name of the session. But sometimes the column header with the scores will be different,
    # in which case, use the literal string contained in outcome_col as the column header for reaching scores in sr_scores
    if outcome_col == 'session_name':
        outcome_pd_header = outcome_pd_header_from_session_metadata(session_metadata)
    else:
        outcome_pd_header = outcome_col
    trials_by_outcome = dict.fromkeys(outcome_groupings)
    try:
        session_outcomes = sr_scores[outcome_pd_header]
    except:
        print('no scores found for {}'.format(outcome_pd_header))
        valid_trials = None
        return trials_by_outcome, valid_trials

    for key, value in outcome_groupings.items():

        outcome_trials_boolean = session_outcomes.isin(value)
        try:
            outcome_trials = sr_scores['vid_number'][outcome_trials_boolean]
        except:
            outcome_trials = sr_scores['vid_number_in_name'][outcome_trials_boolean]
        trials_by_outcome[key] = outcome_trials.to_numpy()

    try:
        valid_trials_bool = np.logical_not(np.isnan(session_outcomes))
        valid_trials_bool = valid_trials_bool.to_numpy()
        valid_trials = np.where(valid_trials_bool)[0] + 1
        # not sure why, but np.where is returning a tuple containing the array of valid trial numbers
        # add 1 because trials are indexed starting at 1 in the outcomes spreadsheet, but index starts at 0 for python arrays
    except:
        valid_trials = np.array([])

    return trials_by_outcome, valid_trials


def create_sr_outcome_groupings():
    '''
    0 – No pellet
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

    :return:
    '''
    outcome_groupings = {'nontrial': [9],
                         'noreach': [5, 6, 9],
                         'nopellet': [0],
                         'reachprefpaw': [0, 1, 2, 3, 4, 7],
                         'firstsuccess': [1],
                         'anysuccessprefpaw': [1, 2],
                         'dropinbox': [3],
                         'pelletknockedoff': [4],
                         'tonguesuccess': [5],
                         'reachfailure': [4, 7],
                         'usedcontra': [8]}

    return outcome_groupings


def outcome_pd_header_from_session_metadata(session_metadata):

    pd_header = '_'.join([session_metadata['ratID'],
                          session_metadata['date'].strftime('%Y%m%d'),
                          '{:02d}'.format(session_metadata['session_num'])
                          ])

    return pd_header


def demodulate_signals(phot_metadata, analog_data, lpCut=20, filtOrder=6):

    # figure out which lines contain the photometry signals
    if 405 in phot_metadata['LEDwavelength']:
        # this is a green fluorescence with isosbestic control, so only one photometry signal to demodulate
        try:
            mod_idx = phot_metadata['AI_line_desc'].index('photometry_signal')
        except:
            mod_idx = phot_metadata['AI_line_desc'].index('photometry_signal1')

        sig_ref_idx = phot_metadata['AI_line_desc'].index('LED1_modulation')
        sig_carrier_freq = phot_metadata['carrier_freqs'][np.where(phot_metadata['LEDwavelength'] == 470)][0]
        try:
            iso_ref_idx = phot_metadata['AI_line_desc'].index('LED2_modulation')
        except:
            iso_ref_idx = phot_metadata['AI_line_desc'].index('isosbestic_modulation')

        iso_carrier_freq = phot_metadata['carrier_freqs'][np.where(phot_metadata['LEDwavelength'] == 405)][0]

        phot_sig = digitalLIA(analog_data[:, mod_idx],
                              analog_data[:, sig_ref_idx],
                              sig_carrier_freq,
                              phot_metadata['Fs'],
                              lpCut,
                              filtOrder)
        iso_sig = digitalLIA(analog_data[:, mod_idx],
                             analog_data[:, iso_ref_idx],
                             iso_carrier_freq,
                             phot_metadata['Fs'],
                             lpCut,
                             filtOrder)

        demod_sig = np.vstack((phot_sig, iso_sig))

    elif all(phot_metadata['LEDwavelength'] > 0):   # there are two different wavelengths being used but neither is isosbestic

        mod_idx = [phot_metadata['AI_line_desc'].index('photometry_signal1'), phot_metadata['AI_line_desc'].index('photometry_signal2')]
        sig_ref_idx = [phot_metadata['AI_line_desc'].index('LED1_modulation'), phot_metadata['AI_line_desc'].index('LED2_modulation')]
        sig_carrier_freqs = phot_metadata['carrier_freqs']

        demod_sig = []
        for i_signal in range(2):
            demod_sig.append(digitalLIA(analog_data[:, mod_idx[i_signal]],
                                        analog_data[:, sig_ref_idx[i_signal]],
                                        sig_carrier_freqs[i_signal],
                                        phot_metadata['Fs'],
                                        lpCut,
                                        filtOrder))

        demod_sig = np.array(demod_sig)   # verify that this works

    return demod_sig


def pad_sinusoid(signal, freq, cycles_to_pad, Fs):

    pad_length = int((Fs / freq) * cycles_to_pad)

    signal = np.squeeze(signal)

    new_signal = np.hstack((signal[:pad_length], signal, signal[-pad_length:]))

    return new_signal, pad_length


def digitalLIA(modSig,refSig,freq,Fs,lpCut,filtOrder):

    cycles_to_pad = freq    # continue the signal for 1 second in either direction

    if len(modSig) != len(refSig):
        print('modulated and reference signals are not the same length')
        return None

    # in code from Tritsch lab, they look to see if there is 0 lag between modulated and reference signals; however,
    # this will almost never be the case, so not sure what the point of that is...

    # from Tritsch lab code:
    # Normalize Reference Signal and ensure the amplitude goes from +2
    # to -2V --> This step ensures that you are maintaining the original
    # ampltiude from the modulated photometry signal
    refSig = refSig - min(refSig)
    refSig = refSig / max(refSig)
    refSig = refSig * 4
    refSig = refSig - np.mean(refSig)

    padded_refSig, _ = pad_sinusoid(refSig, freq, cycles_to_pad, Fs)
    padded_modSig, pad_length = pad_sinusoid(modSig, freq, cycles_to_pad, Fs)

    # create a bandpass Butterworth filter +/- 10 Hz from the carrier frequency
    bp_sos = signal.butter(filtOrder, [freq-10, freq+10], btype='bandpass', analog=False, output='sos', fs=Fs)
    lp_sos = signal.butter(filtOrder, lpCut, btype='lowpass', analog=False, output='sos', fs=Fs)

    padded_modSig = signal.sosfiltfilt(bp_sos, padded_modSig)
    padded_modSig = padded_modSig - np.mean(padded_modSig)

    padded_refSig_90 = np.append(np.diff(padded_refSig), padded_refSig[-1] - padded_refSig[-2])
    padded_PSD_1 = np.multiply(padded_modSig, padded_refSig)
    padded_PSD_1 = signal.sosfiltfilt(lp_sos, padded_PSD_1)

    padded_PSD_2 = np.multiply(padded_modSig, padded_refSig_90)
    padded_PSD_2 = signal.sosfiltfilt(lp_sos, padded_PSD_2)

    sig = np.hypot(padded_PSD_1[pad_length:-pad_length], padded_PSD_2[pad_length:-pad_length])

    return sig


def extract_digital_timestamps(phot_metadata, digital_data, analog_data, dig_on_analog_lines=['RPi_frame_trigger']):
    '''

    :param phot_metadata:
    :param digital_data:
    :param analog_data:
    :param dig_on_analog_lines: list of strings containing digital signal names that were recorded on analog lines
    :return:
    '''
    analog_thresh = 2

    dig_linenames = phot_metadata['DI_line_desc']
    num_lines = len(dig_linenames)

    # if there are digital signals on the analog lines, include it
    # as of 9/8/2023, this would only be RPi_frame_trigger
    for dig_line_on_analog in dig_on_analog_lines:
        if dig_line_on_analog in phot_metadata['AI_line_desc']:
            dig_linenames.append(dig_line_on_analog)
            idx = phot_metadata['AI_line_desc'].index(dig_line_on_analog)
            dig_from_analog = analog_data[:, idx] > analog_thresh

            dig_from_analog = np.expand_dims(dig_from_analog, 1)   # make it a column vector so it can be stacked onto other digital data
            digital_data = np.hstack((digital_data, dig_from_analog))

    Fs = phot_metadata['Fs']

    # calculate difference along digital_data
    digital_diffs = np.diff(digital_data.astype(np.byte), n=1, axis=0)

    ts_dict = dict.fromkeys(dig_linenames)
    for i_line, linename in enumerate(dig_linenames):
        on_times = np.array(np.where(digital_diffs[:, i_line] == 1)) / Fs
        off_times = np.array(np.where(digital_diffs[:, i_line] == -1)) / Fs

        on_times = np.squeeze(on_times)
        off_times = np.squeeze(off_times)

        num_ts = max(on_times.size, off_times.size)
        ts_dict[linename] = np.zeros((num_ts, 2))

        ts_dict[linename][:on_times.size, 0] = on_times
        ts_dict[linename][:off_times.size, 1] = off_times

    return ts_dict