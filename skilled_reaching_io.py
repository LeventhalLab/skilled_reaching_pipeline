import pickle
import pandas as pd
import numpy as np
import scipy.io as sio


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
        Pn_opencv = transpoze(Pn_matlab) where Pn is the camera matrix for the virtual camera

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
        F[:, :, i_view] = mat_cal['F'][:, :, i_view]            #todo: check if F and E also should be transposed. I don't think so
        E[:, :, i_view] = mat_cal['E'][:, :, i_view]

    camera_params = {'mtx': mat_cal['K'].transpose(),
                     'Pn': Pn,
                     'F': F,
                     'E': E,
                     'dist': np.squeeze(mat_cal['dist'])
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