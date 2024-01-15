import numpy as np
import cv2
import shapely.geometry as sg
import shapely.ops as so
import plot_utilities
import matplotlib.pyplot as plt


def fund_matrix_mirror(x1, x2, min_points=10):
    '''

    :param x1: nx2 array of matched points in one view
    :param x2: nx2 array of matched points in the other view
    :return:
    '''

    x1 = np.squeeze(x1)
    x2 = np.squeeze(x2)
    if np.shape(x2) != np.shape(x1):
        print('matched point arrays are not the same shape')
        return None

    if len(x1) < min_points:
        # if there aren't enough point matches, make F full of NaNs; will have to go back later and recalibrate
        F = np.empty((3, 3))
        F.fill(np.nan)

        return F

    num_points = np.shape(x1)[0]

    A = np.zeros((num_points, 3))

    A[:, 0] = (x2[:, 0] * x1[:, 1]) - (x1[:, 0] * x2[:, 1])
    A[:, 1] = x2[:, 0] - x1[:, 0]
    A[:, 2] = x2[:, 1] - x1[:, 1]

    # solve the linear system of equations A * [f12, f13, f23]' = 0
    _, _, VH = np.linalg.svd(A, full_matrices=False)
    vA = VH.T.conj()
    F = np.zeros((3, 3))
    fvec = vA[:, -1]

    # put the solutions to the constraint equation into F
    F[0, 1] = fvec[0]
    F[0, 2] = fvec[1]
    F[1, 2] = fvec[2]
    F[1, 0] = -F[0, 1]
    F[2, 0] = -F[0, 2]
    F[2, 1] = -F[1, 2]

    return F


def project_points(world_points, proj_matrix, mtx):

    # make sure world_points is a 3 x n array, where n is the number of points
    if world_points.ndim == 1:
        # world_points is a vector
        wp_hom = np.append(world_points, 1.)
    elif np.shape(world_points)[0] != 3:
        wp_hom = np.vstack((world_points.T, np.ones(np.shape(world_points)[0])))
    else:
        wp_hom = np.vstack((world_points, np.ones(np.shape(world_points)[1])))

    pp_hom = np.linalg.multi_dot((mtx, proj_matrix, wp_hom))

    proj_points = pp_hom[:2, :] / pp_hom[-1, :]

    return proj_points


def unnormalize_points(points2d_norm, mtx):
    '''

    :param points2d: N x 2 array of normalized points
    :param mtx: camera intrinsic matrix.
    :return:
    '''
    points2d_norm = np.squeeze(points2d_norm)
    if points2d_norm.ndim == 1:
        num_pts = 1
        homogeneous_pts = np.append(points2d_norm, 1.)
        unnorm_pts = np.dot(mtx, homogeneous_pts)
    else:
        # points2d_norm = np.squeeze(points2d_norm)
        num_pts = max(np.shape(points2d_norm))
        try:
            homogeneous_pts = np.hstack((points2d_norm, np.ones((num_pts, 1))))
        except:
            pass
        unnorm_pts = np.dot(mtx, homogeneous_pts.T).T

    if num_pts == 1:
        unnorm_pts = unnorm_pts[:2] / unnorm_pts[-1]
    else:
        unnorm_pts = unnorm_pts[:, :2] / unnorm_pts[:, [-1]]

    return unnorm_pts


def reflection_matrix(rot, t):

    H = np.hstack((rot, t))
    H = np.vstack(H, np.array([0., 0., 0., 1.]))


def depth_of_points(hom_3dpts, rot, t):
    k, r, c, _, _, _, _ = cv2.decomposeProjectionMatrix(np.hstack((rot, t.T)))
    # for i_col in range(4):
    #     hom_3dpts[:, i_col] = hom_3dpts[:, i_col] / hom_3dpts[:, 3]
    num_pts = np.shape(hom_3dpts)[0]

    d = np.empty(num_pts)
    for ii in range(num_pts):
        # this is just the z-coordinate with respect to the camera described by rot and t
        w = np.dot(rot[2, :], np.squeeze((hom_3dpts[ii, :3] - (c[:3]/c[3]).T)))

        d[ii] = (np.sign(np.linalg.det(rot)) * w) / hom_3dpts[ii, 3] * np.linalg.norm(rot[2, :])

    return d


def normalize_points(points2d, mtx):
    '''

    :param points2d: n x 2 or 3 matrix where n is number of points. If n x 3, that causes assumption that points are in normalized coordinates
    :param mtx:
    :return:
    '''
    points2d = np.squeeze(points2d)   # in case the array is n x 1 x 2 instead of n x 2

    if points2d.ndim == 1:
        num_pts = 1
        if len(points2d) == 2:
            homogeneous_pts = np.append(points2d, 1.)
        else:
            homogeneous_pts = points2d
        norm_pts = np.linalg.solve(mtx, homogeneous_pts)
        norm_pts = norm_pts[:2] / norm_pts[-1]
    else:
        num_pts = max(np.shape(points2d))
        if np.shape(points2d)[1] == 2:
            homogeneous_pts = np.hstack((points2d, np.ones((num_pts, 1))))
        else:
            # already converted to homogeneous points
            homogeneous_pts = points2d
        norm_pts = np.linalg.solve(mtx, homogeneous_pts.T)
        norm_pts = norm_pts[:2, :] / norm_pts[-1, :]

    return norm_pts


def P_from_RT(R, T):
    # create camera projection matrix from rotation vector or matrix and translation vector

    # if R only has one dimension or is 3 x 1, convert vector to matrix using cv2.Rodrigues
    if np.ndim(R) == 1 or (1 in np.shape(R)):
        # R is a vector
        Rmat, _ = cv2.Rodrigues(R)
    else:
        Rmat = R

    if np.ndim(T) == 1:
        # T needs to be 3 x 1 for the horizontal concatenation to work
        T = np.reshape(T, (3,1))

    P = np.hstack((R, T))

    return P

# from Multiple-Quadrotor-SLAM/Work/python_libs/triangulation.py /:
def linear_eigen_triangulation(u1, P1, u2, P2, max_coordinate_value=1.e16):
    """
    Linear Eigenvalue based (using SVD) triangulation.
    Wrapper to OpenCV's "triangulatePoints()" function.
    Relative speed: 1.0

    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "max_coordinate_value" is a threshold to decide whether points are at infinity

    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

    The status-vector is based on the assumption that all 3D points have finite coordinates.
    """
    x = cv2.triangulatePoints(P1[0:3, 0:4], P2[0:3, 0:4], u1.T, u2.T)  # OpenCV's Linear-Eigen triangl

    x[0:3, :] /= x[3:4, :]  # normalize coordinates
    x_status = (np.max(abs(x[0:3, :]), axis=0) <= max_coordinate_value)  # NaN or Inf will receive status False

    return x[0:3, :].T.astype(output_dtype), x_status


def multiview_ls_triangulation(pts, camera_mats):
    '''
    algorithim for multiview triangulation taken from anipose

    :param pts: nx2 array (I think) where n is the number of cameras, and columns are (x,y) pairs
    :param camera_mats: 3x4xn array where n is the number of cameras
    :return:
    '''
    num_cams = np.shape(camera_mats)[2]
    A = np.zeros((num_cams * 2, 4))

    for i_cam in range(num_cams):
        x, y = pts[i_cam]
        mat = camera_mats[:, :, i_cam]
        mat = np.vstack((mat, [0., 0., 0., 1.]))

        A[(i_cam * 2):(i_cam * 2 + 1)] = x * mat[2] - mat[0]
        A[(i_cam * 2 + 1):(i_cam * 2 + 2)] = y * mat[2] - mat[1]

    try:
        u, s, vh = np.linalg.svd(A, full_matrices=True)
    except:
        pass
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d


# from anipose:
#   mat is a 4x4 matrix where mat[:3,:3] is the rotation matrix, mat[:3, 3] is the translation vector, and the bottom row
#   is [0,0,0,1]. In other words, it's the 3 x 4 camera matrix P with with a fourth row of [0,0,0,1]
# def triangulate_simple(points, camera_mats):
#     num_cams = len(camera_mats)
#     A = np.zeros((num_cams * 2, 4))
#     for i in range(num_cams):
#         x, y = points[i]
#         mat = camera_mats[i]
#         A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
#         A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
#     u, s, vh = np.linalg.svd(A, full_matrices=True)
#     p3d = vh[-1]
#     p3d = p3d[:3] / p3d[3]
#     return p3d



# Initialize consts to be used in linear_LS_triangulation()
linear_LS_triangulation_C = -np.eye(2, 3)


def linear_LS_triangulation(u1, P1, u2, P2):
    """
    Linear Least Squares based triangulation.
    Relative speed: 0.1

    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.

    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

    The status-vector will be True for all points.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    # Create array of triangulated points
    x = np.zeros((3, len(u1)))

    # Initialize C matrices
    C1 = np.array(linear_LS_triangulation_C)
    C2 = np.array(linear_LS_triangulation_C)

    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]

        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
        b *= -1

        # Solve for x vector
        cv2.solve(A, b, x[:, i:i + 1], cv2.DECOMP_SVD)

    return x.T.astype(output_dtype), np.ones(len(u1), dtype=bool)


# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)


def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
    """
    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    Relative speed: 0.025

    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "tolerance" is the depth convergence tolerance.

    Additionally returns a status-vector to indicate outliers:
        1: inlier, and in front of both cameras
        0: outlier, but in front of both cameras
        -1: only in front of second camera
        -2: only in front of first camera
        -3: not in front of any camera
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).

    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    # Create array of triangulated points
    x = np.empty((4, len(u1)));
    x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
    x_status = np.empty(len(u1), dtype=int)

    # Initialize C matrices
    C1 = np.array(iterative_LS_triangulation_C)
    C2 = np.array(iterative_LS_triangulation_C)

    for xi in range(len(u1)):
        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[xi, :]
        C2[:, 2] = u2[xi, :]

        # Build A matrix
        A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

        # Build b vector
        b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
        b *= -1

        # Init depths
        d1 = d2 = 1.

        for i in range(10):  # Hartley suggests 10 iterations at most
            # Solve for x vector
            cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

            # Calculate new depths
            d1_new = P1[2, :].dot(x[:, xi])
            d2_new = P2[2, :].dot(x[:, xi])

            # Convergence criterium
            if abs(d1_new - d1) <= tolerance and \
                    abs(d2_new - d2) <= tolerance:
                # if i and np.sum((x[0:3, xi] - x_old)**2) <= 0.0001**2:
                # if abs((d1_new - d1) / d1) <= 3.e-6 and \
                # abs((d2_new - d2) / d2) <= 3.e-6: #and \
                # abs(d1_new - d1) <= tolerance and \
                # abs(d2_new - d2) <= tolerance:
                break

            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d1_new
            A[2:4, :] *= 1 / d2_new
            b[0:2, :] *= 1 / d1_new
            b[2:4, :] *= 1 / d2_new

            # Update depths
            d1 = d1_new
            d2 = d2_new

        # Set status
        x_status[xi] = (i < 10 and  # points should have converged by now
                        (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras
        if d1_new <= 0: x_status[xi] -= 1
        if d2_new <= 0: x_status[xi] -= 2

    return x[0:3, :].T.astype(output_dtype), x_status


def polynomial_triangulation(u1, P1, u2, P2):
    """
    Polynomial (Optimal) triangulation.
    Uses Linear-Eigen for final triangulation.
    Relative speed: 0.1

    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.

    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

    The status-vector is based on the assumption that all 3D points have finite coordinates.
    """
    P1_full = np.eye(4);
    P1_full[0:3, :] = P1[0:3, :]  # convert to 4x4
    P2_full = np.eye(4);
    P2_full[0:3, :] = P2[0:3, :]  # convert to 4x4
    P_canon = P2_full.dot(cv2.invert(P1_full)[1])  # find canonical P which satisfies P2 = P_canon * P1

    # "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
    F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T

    # Other way of calculating "F" [HZ (9.2)]
    # op1 = (P2[0:3, 3:4] - P2[0:3, 0:3] .dot (cv2.invert(P1[0:3, 0:3])[1]) .dot (P1[0:3, 3:4]))
    # op2 = P2[0:3, 0:4] .dot (cv2.invert(P1_full)[1][0:4, 0:3])
    # F = np.cross(op1.reshape(-1), op2, axisb=0).T

    # Project 2D matches to closest pair of epipolar lines
    u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

    # For a purely sideways trajectory of 2nd cam, correctMatches() returns NaN for all possible points!
    if np.isnan(u1_new).all() or np.isnan(u2_new).all():
        F = cv2.findFundamentalMat(u1, u2, cv2.FM_8POINT)[0]  # so use a noisy version of the fund mat
        u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

    # Triangulate using the refined image points
    return linear_eigen_triangulation(u1_new[0], P1, u2_new[0],
                                      P2)  # TODO: replace with linear_LS: better results for points not at Inf


# Attempt to load the optimized triangulation functions
# import triangulation_c
#
# if triangulation_c.loaded:
#     from triangulation_c import linear_LS_triangulation as linear_LS_triangulation_c
#
#
#     def linear_LS_triangulation(*args):
#         x, x_status = linear_LS_triangulation_c(*args)
#         if np.finfo(x.dtype) != np.finfo(output_dtype):
#             x = x.astype(output_dtype)
#         return x, x_status
#
#
#     linear_LS_triangulation.__doc__ = linear_LS_triangulation_c.__doc__
#
#     from triangulation_c import iterative_LS_triangulation as iterative_LS_triangulation_c
#
#
#     def iterative_LS_triangulation(*args, **kwargs):
#         x, x_status = iterative_LS_triangulation_c(*args, **kwargs)
#         if np.finfo(x.dtype) != np.finfo(output_dtype):
#             x = x.astype(output_dtype)
#         return x, x_status
#
#
#     iterative_LS_triangulation.__doc__ = iterative_LS_triangulation_c.__doc__
# else:
#     print
#     'Warning: failed to load the optimized "triangulation_c" module'
#     print
#     "=> falling back to Python-speed for triangulation."

output_dtype = float


def set_triangl_output_dtype(output_dtype_):
    """
    Set the datatype of the triangulated 3D point positions.
    (Default is set to "float")
    """
    global output_dtype
    output_dtype = output_dtype_


def find_line_edge_coordinates(line, im_size):

    a, b, c = line
    edge_pts = np.zeros((2, 2))

    x_edge = np.array([0, im_size[0] - 1])
    y_edge = np.array([0, im_size[1] - 1])

    i_pt = 0
    # check the intersection with the left and right image borders unless the line is vertical
    if abs(a) > 0:
        test_y = (-c - a * x_edge[0]) / b
        if y_edge[0] <= test_y <= y_edge[1]:
            # check intersection with left image border
            edge_pts[i_pt, :] = [x_edge[0], test_y]
            i_pt += 1

        test_y = (-c - a * x_edge[1]) / b
        if y_edge[0] <= test_y <= y_edge[1]:
            # check intersection with left image border
            edge_pts[i_pt, :] = [x_edge[1], test_y]
            i_pt += 1

    # check the intersection with the left and right image borders unless the line is horizontal
    if abs(b) > 0:
        if i_pt < 2:
            test_x = (-c - b * y_edge[0]) / a
            if x_edge[0] <= test_x <= x_edge[1]:
                # check intersection with left image border
                edge_pts[i_pt, :] = [test_x, y_edge[0]]
                i_pt += 1

        if i_pt < 2:
            test_x = (-c - b * y_edge[1]) / a
            if x_edge[0] <= test_x <= x_edge[1]:
                # check intersection with left image border
                edge_pts[i_pt, :] = [test_x, y_edge[1]]
                i_pt += 1

    return edge_pts


def line_polygon_intersect(line_pts, poly_points, tolerance=1):
    '''

    :param line_coeffs: vector [A, B, C] such that Ax + By + C = 0
    :param poly_points:
    :param tolerance:
    :return:
    '''

    epi_line = sg.asLineString(line_pts)
    paw_poly = sg.asPolygon(poly_points)


    # ax = plot_utilities.plot_shapely_point(sh_objects[0], fc='red')
    # plot_utilities.plot_polygon(sh_objects[1], ax=ax)
    # plt.show()
    # find points within tolerance of the line

    if epi_line.intersects(paw_poly):
        epi_paw_intersect = epi_line.intersection(paw_poly)
        pass
    else:
        epi_paw_intersect = None

    return epi_paw_intersect

    # min_x = min(poly_points[:, 0])
    # max_x = max(poly_points[:, 0])
    # min_y = min(poly_points[:, 1])
    # max_y = max(poly_points[:, 1])
    #
    # intersect_points = []
    #
    # iterations = 0
    #
    # for ii in range(round(min_x), round(max_x)):
    #     iterations = iterations + 1
    #     if iterations > 1000:
    #         print('iterations = {:d}'.format(iterations))
    #
    #     iterations2 = 0
    #     for jj in range(round(min_y), round(max_y)):
    #         iterations2 = iterations2 + 1
    #         if iterations2 > 1000:
    #             print('iterations2 = {:d}'.format(iterations2))
    #
    #         line_val = ii * line_coeffs[0] + jj * line_coeffs[1] + line_coeffs[2]
    #
    #         # WORKING HERE...

def find_nearest_neighbor(x, y, num_neighbors=1):
    '''

    :param x: object borders as an m x 2 numpy array, where m is the number of points
    :param y: object borders as an n x 2 numpy array, where n is the number of points
    :param num_neighbors:
    :return:
    '''
    if isinstance(x, sg.Point) or isinstance(x, sg.MultiPoint):
        # x was input as a shapely point/multipoint object
        point_x = x
    elif x.ndim == 1 or np.shape(x)[0] == 1:
        # x is a vector, either shape (2,) or shape (1,2)
        point_x = sg.asPoint(np.squeeze(x))
    else:
        # x is an m x 2 array of multiple points (m > 1)
        point_x = sg.asMultiPoint(x)

    if isinstance(y, sg.Point) or isinstance(y, sg.MultiPoint):
        # y was input as a shapely point/multipoint object
        point_y = y
    elif y.ndim == 1 or np.shape(y)[0] == 1:
        # y is a vector, either shape (2,) or shape (1,2)
        point_y = sg.asPoint(np.squeeze(y))
    else:
        # y is an m y 2 array of multiple points (m > 1)
        point_y = sg.asMultiPoint(y)

    # if type(y) is sg.Point:
    #     point_y = y
    # else:
    #     point_y = sg.asPoint(y)

    near_x, near_y = so.nearest_points(point_x, point_y)
    nn_dist = near_x.distance(near_y)

    # left over from when I thought it would be useful to find the n nearest points. Right now, just finds the nearest points
    # pts_diff = y - x
    #
    # dist_from_x = np.linalg.norm(pts_diff, axis=1)
    #
    # sorted_dist_idx = np.argsort(dist_from_x)
    # sorted_dist = np.sort(dist_from_x)
    #
    # return sorted_dist[:num_neighbors], sorted_dist_idx[:num_neighbors]

    return nn_dist, near_y


def find_nearest_point_on_line(line_pts, pts):

    if type(line_pts) is sg.LineString:
        sg_line = line_pts
    else:
        sg_line = sg.asLineString(line_pts)

    if pts.ndim == 1:
        num_pts = 1
    else:
        num_pts = np.shape(pts)[0]
    if num_pts > 1:
        min_dist = 1000
        for i_pt in range(num_pts):
            test_pt = sg.asPoint(pts[i_pt, :])
            test_dist = test_pt.distance(sg_line)
            if test_dist < min_dist:
                min_dist = test_dist
                closest_pt = test_pt

        sg_point = closest_pt
    else:
        sg_point = sg.asPoint(pts)
    try:
        near_pts = so.nearest_points(sg_line, sg_point)
    except:
        pass


    nndist = near_pts[0].distance(near_pts[1])
    nn_pt = near_pts[0]

    return nndist, nn_pt


def find_nearest_point_to_line(line_pts, pts):

    if type(line_pts) is sg.LineString:
        sg_line = line_pts
    else:
        sg_line = sg.asLineString(line_pts)

    sg_point = sg.asPoint(pts)

    near_pts = so.nearest_points(sg_line, sg_point)
    nndist = near_pts[0].distance(near_pts[1])
    nn_pt = near_pts[1]

    return nndist, nn_pt


def find_line_intersection(line1, line2):

    if type(line1) is sg.LineString:
        sg_line1 = line1
    else:
        sg_line1 = sg.asLineString(line1)

    if type(line2) is sg.LineString:
        sg_line2 = line2
    else:
        sg_line2 = sg.asLineString(line2)

    if sg_line1.intersects(sg_line2):
        line_intersect = sg_line1.intersection(sg_line2)
    else:
        line_intersect = None

    return line_intersect

def line_convex_hull_intersect(line1, points):

    if type(line1) is sg.LineString:
        sg_line1 = line1
    else:
        sg_line1 = sg.asLineString(line1)

    if type(points) is sg.MultiPoint:
        sg_points = points
    else:
        sg_points = sg.asMultiPoint(points)

    cvhull_poly = sg_points.convex_hull

    if cvhull_poly.intersects(sg_line1):
        line_hull_intersect = cvhull_poly.intersection(sg_line1)
    else:
        line_hull_intersect = None

    return line_hull_intersect


def triangulate_points(pts, cal_data):

    num_cams = len(pts)

    projMatr1 = np.eye(3, 4)
    projMatr2 = P_from_RT(cal_data['R'], cal_data['T'])
    mtx = cal_data['mtx']
    dist = cal_data['dist']

    # undistort the points from each camera
    pts_ud = [cv2.undistortPoints(pts[i_cam], mtx[i_cam], dist[i_cam]) for i_cam in range(num_cams)]
    # note that cv2.undistortPoints returns the answer in normalized coordinates

    points4D = cv2.triangulatePoints(projMatr1, projMatr2, pts_ud[0], pts_ud[1])
    world_points = np.squeeze(cv2.convertPointsFromHomogeneous(points4D.T))

    # reproject into original views
    reprojected_pts = []
    for i_cam in range(num_cams):
        dist = np.zeros(5)    # using the points that already have been undistorted against which to compare the reprojections
        if i_cam == 0:
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
        else:
            rvec, _ = cv2.Rodrigues(cal_data['R'])
            tvec = cal_data['T']

        ppts, _ = cv2.projectPoints(world_points, rvec, tvec, mtx[i_cam], dist[i_cam])
        ppts = np.squeeze(ppts)
        reprojected_pts.append(ppts)

    world_points = world_points * cal_data['checkerboard_square_size']

    return world_points, reprojected_pts


def estimate_gamma(img):

    # convert img to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid * 255) / math.log(mean)

    return gamma


def adjust_blacklevel(image, gamma=1.0):
    invGamma = 1.0 / gamma

    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    if image.dtype == 'float64':
        image = image * 255
        image = image.astype('uint8')

    return cv2.LUT(image, table)




def rotate_pts_180(pts, im_size):
    '''

    :param pts:
    :param im_size: 1 x 2 list (height, width) (or should it be width, height?)
    :return:
    '''

    # reflect points around the center

    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)

    if not isinstance(im_size, np.ndarray):
        im_size = np.array(im_size)

    reflected_pts = []
    pts = np.squeeze(pts)
    if np.ndim(pts) == 1:
        try:
            # not quite sure why there are issues with array shape, but this seems to fix it
            pts = np.reshape(pts, (1, 2))
        except:
            pass
    for i_pt, pt in enumerate(pts):
        if len(pt) > 0:
            try:
                x, y = pt[0]
            except:
                # must be a vector instead of an array
                x, y = pt
            # possible that im_size is width x height or height x width
            if all(np.array([x, y]) == 0):
                # if this point wasn't actually found by dlc, keep the coordinate as (0, 0)
                new_x = 0.
                new_y = 0.
            else:
                new_x = im_size[0] - x
                new_y = im_size[1] - y

            reflected_pts.append([np.array([new_x, new_y])])
        else:
            reflected_pts.append(np.array([0., 0.]))

    reflected_pts = np.array(reflected_pts)
    reflected_pts = np.squeeze(reflected_pts)

    return reflected_pts


# some simple geometry functions
def find_nearest_neighbor(test_point, other_points, num_neighbors=1):
    '''
    function to find the point(s) in other_points closest to test_point
    :param test_point: (x,y) pair as a 1 x 2 numpy array
    :param other_points:
    :param num_neighbors:
    :return:
    '''
    num_otherpoints = np.shape(other_points)[0]

    distances = np.linalg.norm(other_points - test_point, axis=1)

    sorted_dist = np.sort(distances)
    sorted_idx = np.argsort(distances)

    nndist = sorted_dist[:num_neighbors]
    nnidx = sorted_idx[:num_neighbors]

    return nndist, nnidx