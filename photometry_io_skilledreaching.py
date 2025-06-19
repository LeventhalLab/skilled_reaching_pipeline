import os
import glob
from datetime import datetime
import scipy.io as sio
import csv
import pandas as pd
import numpy as np
import h5py
import pickle


def read_pickle(filename):
    """ Read the pickle file """
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def write_pickle(filename, data):
    """ Write the pickle file """
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_xlsx_scores(sroutcome_fname, ratID):

    if isinstance(ratID, dict):
        # if ratID contains a session_metadata dictionary, extract the ratID. If it's a string, just use ratID
        ratID = ratID['ratID']

    try:
        df = pd.read_excel(sroutcome_fname, sheet_name=ratID)
    except:
        # most likely, there isn't a sheet for this rat yet
        df = pd.DataFrame()

    return df


def read_photometry_data(data_files):

    if data_files['phot_mat'] is not None:
        # this is an older file saved directly in .mat format
        phot_data = read_photometry_mat(data_files['phot_mat'])
    else:
        md = sio.loadmat(data_files['metadata'])
        if 'LEDwavelength' in md.keys():
            phot_data = {
                'Fs': float(md['Fs'][0][0]),
                'current': md['current'][0][0],
                'virus': md['virus'][0],
                'AI_line_desc': [],
                'DI_line_desc': [],
                'carrier_freqs': md['carrier_freqs'][0],
                'carrier_scale': md['carrier_scale'][0],
                'LEDwavelength': md['LEDwavelength'][0],
                'cam_trigger_delay': md['cam_trigger_delay'][0],
                'cam_trigger_pw': md['cam_trigger_pw'][0],
                'task': md['task']
            }
        else:
            phot_data = {
                'Fs': float(md['Fs'][0][0]),
                'current': md['current'][0][0],
                'virus': md['virus'][0],
                'AI_line_desc': [],
                'DI_line_desc': [],
                'carrier_freqs': md['carrier_freqs'][0],
                'carrier_scale': md['carrier_scale'][0],
                'LEDwavelength': np.array([470, 405]),                # assume this was 470 nm excitation with 405 isosbestic
                'cam_trigger_delay': md['cam_trigger_delay'][0],
                'cam_trigger_pw': md['cam_trigger_pw'][0],
                'task': md['task']
            }

        if 'AI_line_desc' in md.keys():
            line_desc_array = md['AI_line_desc'][0]
            line_desc_list = [ld[0] for ld in line_desc_array]
            phot_data['AI_line_desc'] = line_desc_list

            num_analog_lines = len(md['AI_line_desc'][0])

            phot_data['t'], phot_data['data'] = read_analog_bin(data_files['analog_bin'], phot_data)

        if 'DI_line_desc' in md.keys():
            line_desc_array = md['DI_line_desc'][0]
            line_desc_list = [ld[0] for ld in line_desc_array]
            phot_data['DI_line_desc'] = line_desc_list

            num_digital_lines = len(md['DI_line_desc'][0])

            phot_data['digital_data'] = read_digital_bin(data_files['digital_bin'], phot_data)

    return phot_data


def read_photometry_metadata(metadata_fname):
    md = sio.loadmat(metadata_fname)
    if 'LEDwavelength' in md.keys():
        phot_metadata = {
            'Fs': float(md['Fs'][0][0]),
            'current': md['current'][0][0],
            'virus': md['virus'][0],
            'AI_line_desc': [],
            'DI_line_desc': [],
            'carrier_freqs': md['carrier_freqs'][0],
            'carrier_scale': md['carrier_scale'][0],
            'LEDwavelength': md['LEDwavelength'][0],
            'cam_trigger_delay': md['cam_trigger_delay'][0],
            'cam_trigger_pw': md['cam_trigger_pw'][0],
            'task': md['task']
        }
    else:
        phot_metadata = {
            'Fs': float(md['Fs'][0][0]),
            'current': md['current'][0][0],
            'virus': md['virus'][0],
            'AI_line_desc': [],
            'DI_line_desc': [],
            'carrier_freqs': md['carrier_freqs'][0],
            'carrier_scale': md['carrier_scale'][0],
            'LEDwavelength': np.array([470, 405]),                # assume this was 470 nm excitation with 405 isosbestic
            'cam_trigger_delay': md['cam_trigger_delay'][0],
            'cam_trigger_pw': md['cam_trigger_pw'][0],
            'task': md['task']
        }

    if 'AI_line_desc' in md.keys():
        line_desc_array = md['AI_line_desc'][0]
        line_desc_list = [ld[0] for ld in line_desc_array]
        phot_metadata['AI_line_desc'] = line_desc_list

    if 'DI_line_desc' in md.keys():
        line_desc_array = md['DI_line_desc'][0]
        try:
            line_desc_list = [ld[0] for ld in line_desc_array]
        except:
            # ugly workaround for when line description for matlab acquisition code is an empty string
            line_desc_list = []
            for ld in line_desc_array:
                if len(ld) == 0:
                    line_desc_list.append('empty')
                else:
                    line_desc_list.append(ld[0])

        phot_metadata['DI_line_desc'] = line_desc_list

    return phot_metadata


def read_analog_bin(fname, phot_metadata):
    '''
    read in analog raw data recorded from Matlab nidaq code. Data were stored as double precision floating point. First
    number at each time point is a timestamp, then values are as described by the AI_line_desc array
    :param fname:
    :param num_channels:
    :return:
    '''

    num_channels = len(phot_metadata['AI_line_desc'])

    all_data = np.fromfile(fname, dtype=float, count=-1)

    # reshape based on num_channels. Need to use num_channels+1 because the first column is timestamps
    all_data = np.reshape(all_data, (-1, num_channels+1))
    t = all_data[:, 0]
    analog_data = all_data[:, 1:]

    return t, analog_data


def read_digital_bin(fname, phot_metadata):
    num_channels = len(phot_metadata['DI_line_desc'])

    digital_data = np.fromfile(fname, dtype=np.bool_, count=-1)

    # reshape based on num_channels. Need to use num_channels+1 because the first column is timestamps
    digital_data = np.reshape(digital_data, (-1, num_channels))

    return digital_data


def read_photometry_mat(full_path):
    '''
    function to read a .mat file containing photometry data
    assumed to be 8-channel data
    
    return: dictionary with the following keys
        Fs - float giving the sampling rate in Hz
        current - current applied to the LED
        data - n x 8 numpy array containing the raw data; channel 0 is typically the photometry signal
    '''
    #todo: need to update with new variables being saved in .mat file. Should also add a variable in the
    # recording software to document what each analog line represents
    try:
        photometry_data = sio.loadmat(full_path)

        # in some versions, 'current' and 'virus' aren't included in the file
        try:
            phot_data = {
                'Fs': float(photometry_data['Fs'][0][0]),
                'current': photometry_data['current'][0][0],
                'data': photometry_data['data'],
                't': photometry_data['timeStamps'],
                'virus': photometry_data['virus'][0],
                'AI_line_desc': []
            }
        except KeyError:
            phot_data = {
                'Fs': float(photometry_data['Fs'][0][0]),
                'current': [],
                'data': photometry_data['data'],
                't': photometry_data['timeStamps'],
                'virus': [],
                'AI_line_desc': []
            }

        if 'AI_line_desc' in photometry_data.keys():
            line_desc_array = photometry_data['AI_line_desc'][0]
            line_desc_list = [ld[0] for ld in line_desc_array]
            phot_data['AI_line_desc'] = line_desc_list

    except NotImplementedError:
        # likely a v7.3 .mat file, recorded from the matlab online visualization app
        pfile_data = h5py.File(full_path, 'r')

        phot_data = {
            'data': pfile_data['data'][0],
            'Fs': pfile_data['metadata']['Rate'][0][0],
            'current': [],
            't': pfile_data['timestamps'][0],
            'virus': [],
            'AI_line_desc': ['photometry_signal']
        }
    except ValueError:
        # in case of corrupted mat file
        return None

    # reformat into a dictionary for easier use later
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
                FED_FR = int(row[6])   # what does 'FR' stand for?
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

    full_file_path = glob.glob(os.path.join(photometry_parent, '**', session_date_folder + '*', phot_fname), recursive=True)

    if len(full_file_path)==1:
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

    if len(session_FED_list)==1:
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


def read_csv_scores(csv_score_file):
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

    :param csv_score_file:
    :return:
    '''
    if not os.path.exists(csv_score_file):
        return None

    sr_scores = pd.read_csv(csv_score_file)

    return sr_scores
    # with open(csv_score_file, newline='') as csvfile:
    #     score_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #
    #     # FED_datetime = []
    #     # FED_batt_voltage = []
    #     # FED_motor_turns = []
    #     # FED_FR = []
    #     # FED_event = []
    #     # FED_active_poke = []
    #     # FED_left_poke_count = []
    #     # FED_right_poke_count = []
    #     # FED_pellet_count = []
    #     # FED_block_pellet_count = []
    #     # FED_retrieval_time = []
    #     # FED_poke_time = []
    #     for i_row, row in enumerate(score_reader):
    #         if i_row == 0:
    #             # this is the header row, first column is empty (or at least not a session)
    #             session_list = row[1:]
    #             pass
    #         else:
    #             trial_num.append()
    #             FED_datetime.append(datetime.strptime(row[0], '%m/%d/%Y %H:%M:%S'))
    #             FED_version = row[1]
    #             FED_task = row[2]
    #             FED_devnum = int(row[3])
    #             FED_batt_voltage.append(float(row[4]))
    #             FED_motor_turns.append(int(row[5]))
    #             FED_FR = int(row[6])   # what does 'FR' stand for?
    #             FED_event.append(row[7])
    #             FED_active_poke.append(row[8])
    #             FED_left_poke_count.append(row[9])
    #             FED_right_poke_count.append(row[10])
    #             FED_pellet_count.append(row[11])
    #             FED_block_pellet_count.append(row[12])
    #             FED_retrieval_time.append(row[13])
    #             FED_poke_time.append(row[14])
    #
    #     FED_data = {
    #         'datetime': FED_datetime,
    #         'version': FED_version,
    #         'task': FED_task,
    #         'devnum': FED_devnum,
    #         'batt_voltage': FED_batt_voltage,
    #         'motor_turns': FED_motor_turns,
    #         'FR': FED_FR,
    #         'event': FED_event,
    #         'active_poke': FED_active_poke,
    #         'left_poke_count': FED_left_poke_count,
    #         'right_poke_count': FED_right_poke_count,
    #         'pellet_count': FED_pellet_count,
    #         'block_pellet_count': FED_block_pellet_count,
    #         'retrieval_time': FED_retrieval_time,
    #         'poke_time': FED_poke_time
    #     }
    #
    #     return FED_data