import numpy as np
import cv2

def unnormalize_points(points2d_norm, mtx):
    '''

    :param points2d: N x 2 array of normalized points
    :param mtx: camera intrinsic matrix.
    :return:
    '''

    if points2d_norm.ndim == 1:
        num_pts = 1
        homogeneous_pts = np.append(points2d_norm, 1.)
        unnorm_pts = np.dot(mtx, homogeneous_pts)
    else:
        num_pts = max(np.shape(points2d_norm))
        homogeneous_pts = np.hstack((points2d_norm, np.ones((num_pts, 1))))
        unnorm_pts = np.dot(mtx, homogeneous_pts.T).T

    if num_pts == 1:
        unnorm_pts = unnorm_pts[:2] / unnorm_pts[-1]
    else:
        unnorm_pts = unnorm_pts[:,:2] / unnorm_pts[:, [-1]]

    return unnorm_pts


def normalize_points(points2d, mtx):
    pass