import pickle
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
    Pn = np.zeros((3, 4, 4))
    for i_view in range(0, 3):
        Pn[:, :, i_view] = mat_cal['Pn'][:, :, i_view].transpose()

    camera_params = {'mtx': mat_cal['K'].transpose(),
                     'Pn': Pn
                     }

    return camera_params