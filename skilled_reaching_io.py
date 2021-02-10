import pickle
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

    while opencv assumes

    So, need to take all the matlab matrices, and
    :param mat_calibration_name:
    :return:
    """
    mat_cal = sio.loadmat(mat_calibration_name)

    return mat_cal