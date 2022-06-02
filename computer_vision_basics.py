import numpy as np
import cv2
import shapely.geometry as sg
import shapely.ops as so
import plot_utilities
import matplotlib.pyplot as plt


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

    if points2d_norm.ndim == 1:
        num_pts = 1
        homogeneous_pts = np.append(points2d_norm, 1.)
        unnorm_pts = np.dot(mtx, homogeneous_pts)
    else:
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


def normalize_points(points2d, mtx):
    if points2d.ndim == 1:
        num_pts = 1
        homogeneous_pts = np.append(points2d, 1.)
        norm_pts = np.linalg.solve(mtx, homogeneous_pts)
        norm_pts = norm_pts[:2] / norm_pts[-1]
    else:
        num_pts = max(np.shape(points2d))
        homogeneous_pts = np.hstack((points2d, np.ones((num_pts, 1))))
        norm_pts = np.linalg.solve(mtx, homogeneous_pts.T)
        norm_pts = norm_pts[:2, :] / norm_pts[-1, :]

    return norm_pts


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
            # x_old = np.array(x[0:3, xi])    # TODO: remove
            cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

            # Calculate new depths
            d1_new = P1[2, :].dot(x[:, xi])
            d2_new = P2[2, :].dot(x[:, xi])

            # Convergence criterium
            # print i, d1_new - d1, d2_new - d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
            # print i, (d1_new - d1) / d1, (d2_new - d2) / d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
            # print i, np.sqrt(np.sum((x[0:3, xi] - x_old)**2)), (d1_new > 0 and d2_new > 0)    # TODO: remove
            ##print i, u1[xi, :] - P1[0:2, :].dot(x[:, xi]) / d1_new, u2[xi, :] - P2[0:2, :].dot(x[:, xi]) / d2_new    # TODO: remove
            # print bool(i) and ((d1_new - d1) / (d1 - d_old), (d2_new - d2) / (d2 - d1_old), (d1_new > 0 and d2_new > 0))    # TODO: remove
            ##if abs(d1_new - d1) <= tolerance and abs(d2_new - d2) <= tolerance: print "Orig cond met"    # TODO: remove
            if abs(d1_new - d1) <= tolerance and \
                    abs(d2_new - d2) <= tolerance:
                # if i and np.sum((x[0:3, xi] - x_old)**2) <= 0.0001**2:
                # if abs((d1_new - d1) / d1) <= 3.e-6 and \
                # abs((d2_new - d2) / d2) <= 3.e-6: #and \
                # abs(d1_new - d1) <= tolerance and \
                # abs(d2_new - d2) <= tolerance:
                # if i and 1 - abs((d1_new - d1) / (d1 - d_old)) <= 1.e-2 and \    # TODO: remove
                # 1 - abs((d2_new - d2) / (d2 - d1_old)) <= 1.e-2 and \    # TODO: remove
                # abs(d1_new - d1) <= tolerance and \    # TODO: remove
                # abs(d2_new - d2) <= tolerance:    # TODO: remove
                break

            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d1_new
            A[2:4, :] *= 1 / d2_new
            b[0:2, :] *= 1 / d1_new
            b[2:4, :] *= 1 / d2_new

            # Update depths
            # d_old = d1    # TODO: remove
            # d1_old = d2    # TODO: remove
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

    pts_diff = y - x

    dist_from_x = np.linalg.norm(pts_diff, axis=1)

    sorted_dist_idx = np.argsort(dist_from_x)
    sorted_dist = np.sort(dist_from_x)

    return sorted_dist[:num_neighbors], sorted_dist_idx[:num_neighbors]


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

    near_pts = so.nearest_points(sg_line, sg_point)

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