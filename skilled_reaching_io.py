import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import scipy.io as sio
import navigation_utilities
import os
import toml


def read_rat_db(parent_directories, rat_db_fname):

    fname = os.path.join(parent_directories['videos_root_folder'], rat_db_fname)

    _, ext = os.path.splitext(fname)
    if ext in ['.xls', '.xlsx']:
        rat_df = pd.read_excel(fname)
    elif ext == '.csv':
        rat_df = pd.read_csv(fname)

    # convert to all lowercase column headers
    for col_header in rat_df.columns:
        rat_df = rat_df.rename(columns={col_header: col_header.lower()})

    # convert strings in "date" column into datetime objects
    if 'birthdate' in rat_df.columns:
        rat_df['birthdate'] = pd.to_datetime(rat_df['birthdate'], format='%m/%d/%Y').dt.date
    if 'virusdate' in rat_df.columns:
        rat_df['virusdate'] = pd.to_datetime(rat_df['virusdate'], format='%m/%d/%Y').dt.date
    if 'fiberdate' in rat_df.columns:
        rat_df['fiberdate'] = pd.to_datetime(rat_df['fiberdate'], format='%m/%d/%Y').dt.date

    return rat_df


def read_crop_params_csv(crop_params_filepath):
    '''

    :param crop_params_filepath: full path to a .csv file containing cropping regions
    :return:
    crop_params_df - pandas dateframe with the following columns:
        date - date in datetime.date format
        direct_left, direct_right, direct_top, direct_bottom - left, right, top, bottom borders of direct view
        leftmirror_left, leftmirror_right, leftmirror_top, leftmirror_bottom - left, right, top, bottom borders of left mirror view
        rightmirror_left, rightmirror_right, rightmirror_top, rightmirror_bottom - left, right, top, bottom borders of right mirror view
        NOTE - these border coordinates start counting at 1 instead of 0, so should subtract 1 when extracting regions from images
    '''
    crop_params_df = pd.read_csv(crop_params_filepath)

    # convert strings in "date" column into datetime objects
    crop_params_df['date'] = pd.to_datetime(crop_params_df['date'], infer_datetime_format=True).dt.date

    return crop_params_df


def read_session_metadata_xlsx(session_metadata_xlsx_path):

    xl = pd.ExcelFile(session_metadata_xlsx_path)
    sheet_names = xl.sheet_names
    # the sheet names should be 'R0XXX' - should all be in the format of rat IDs; if it's not, ignore it
    xl_ratIDs = [sn for sn in sheet_names if sn[0] == 'R' and len(sn) == 5]

    cal_metadata = dict.fromkeys(xl_ratIDs)

    for ratID in xl_ratIDs:
        cal_metadata[ratID] = pd.read_excel(session_metadata_xlsx_path, sheet_name=ratID)

    return cal_metadata


def read_calibration_metadata_csv(calibration_metadata_csv_path):
    '''

    :param crop_params_filepath: full path to a .csv file containing cropping regions
    :return:
    crop_params_df - pandas dateframe with the following columns:
        date - date in datetime.date format
        direct_left, direct_right, direct_top, direct_bottom - left, right, top, bottom borders of direct view
        leftmirror_left, leftmirror_right, leftmirror_top, leftmirror_bottom - left, right, top, bottom borders of left mirror view
        rightmirror_left, rightmirror_right, rightmirror_top, rightmirror_bottom - left, right, top, bottom borders of right mirror view
        NOTE - these border coordinates start counting at 1 instead of 0, so should subtract 1 when extracting regions from images
    '''
    calibration_metadata_df = pd.read_csv(calibration_metadata_csv_path)

    # convert strings in "date" column into datetime objects
    calibration_metadata_df['date'] = pd.to_datetime(calibration_metadata_df['date'], infer_datetime_format=True).dt.date

    return calibration_metadata_df


def write_toml(filename, data):
    with open(filename, 'w') as f:
        toml.dump(data, f)


def read_toml(filename):
    with open(filename, 'r') as f:
        toml_dict = toml.load(f)

    return toml_dict


def read_pickle(filename):
    """ Read the pickle file """
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def write_pickle(filename, data):
    """ Write the pickle file """
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_matlab_calibration(mat_calibration_name):
    """
    read in matlab calibration file and translate all matrices into opencv versions. For example, Matlab assumes
        s p = Pw [R;t] A
    while opencv assumes
        s p = A [R|t] Pw where s is a scaling factor, p is the 2d projection of the 3d world point Pw, [R|t] are the
            camera extrinsics (rotation and translation), and A is the camera intrinsic matrix
    So, we need to make the following conversions:
        mtx_opencv = transpose(mtx_matlab) where mtx is the camera intrinsic matrix
        Pn_opencv = transpose(Pn_matlab) where Pn is the camera matrix for the virtual camera

    :param mat_calibration_name:
    :return:
    """
    mat_cal = sio.loadmat(mat_calibration_name)

    # transpose intrinsic, camera matrices from matlab file
    # there are 3 camera matrices: top mirror, left mirror, right mirror (stored in that order)
    Pn = np.zeros((3, 4, 3))
    F = np.zeros((3, 3, 3))
    E = np.zeros((3, 3, 3))
    for i_view in range(0, 3):
        Pn[:, :, i_view] = mat_cal['Pn'][:, :, i_view].transpose()
        F[:, :, i_view] = mat_cal['F'][:, :, i_view]            #todo: figure out how to calculate the scale factor
        E[:, :, i_view] = mat_cal['E'][:, :, i_view]

    camera_params = {'mtx': mat_cal['K'].transpose(),
                     'Pn': Pn,
                     'F': F,
                     'E': E,
                     'dist': np.squeeze(mat_cal['dist']),
                     'scalefactor': mat_cal['scaleFactor']
                     }

    return camera_params


def read_rat_csv_database(csv_name):

    rat_db = pd.read_csv(csv_name)

    return rat_db
    # # determine how many lines are in the .csv file
    # with open(csv_name, newline='\n') as csv_file:
    #     num_lines = sum(1 for row in csv_file)
    #
    # with open(csv_name, newline='\n') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     cb_points = np.zeros((num_lines-1, 2))
    #     for i_row, row in enumerate(csv_reader):
    #         if i_row == 0:
    #             # check to make sure the header was read in properly
    #             print(f'Column names are {", ".join(row)}')
    #         else:
    #             # read in the rows of checkerboard points
    #             print('testing')
    #             pass
    #
    # return cb_points


def get_calibration_data(session_date, box_num, cal_file_folder):
    # create the expected .mat file name, and expected .pickle file name - .mat file if calibration was previously
    # done in matlab using the calibration cube, .pickle if calibration was done using video of checkerboard
    calibration_metadata = {'time': session_date,
                            'boxnum': box_num}

    pickle_cal_filename = navigation_utilities.create_calibration_filename(calibration_metadata)
    full_pickle_cal_name = os.path.join(cal_file_folder, pickle_cal_filename)

    mat_cal_filename = navigation_utilities.create_mat_cal_filename(calibration_metadata)
    full_mat_cal_name = os.path.join(cal_file_folder, mat_cal_filename)

    if os.path.exists(full_pickle_cal_name):
        cal_data = read_pickle(full_pickle_cal_name)
    elif os.path.exists(full_mat_cal_name):
        cal_data = read_matlab_calibration(full_mat_cal_name)
    else:
        cal_data = None

    return cal_data


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
                'LEDwavelength': np.array([470, 405]),  # assume this was 470 nm excitation with 405 isosbestic
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
            'LEDwavelength': np.array([470, 405]),  # assume this was 470 nm excitation with 405 isosbestic
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
    all_data = np.reshape(all_data, (-1, num_channels + 1))
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
    # todo: need to update with new variables being saved in .mat file. Should also add a variable in the
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
