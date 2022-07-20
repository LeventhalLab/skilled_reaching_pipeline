import shapely.geometry as sg
import shapely.ops as so
import cv2
import computer_vision_basics as cvb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def plot_shapely_point(pt, fc='blue', ax=None):

    if ax == None:
        # create figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.scatter(np.array(pt.coords.xy[0]), pt.coords.xy[1], fc=fc)

    return ax


def plot_polygon(poly, ax=None, fc='blue'):

    if ax == None:
        # create figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # a = [poly.exterior.coords[ii] for ii in range(len(poly.exterior.coords))]
    add_polygon_patch(poly.exterior.coords, ax, fc=fc)

    for interior in poly.interiors:
        add_polygon_patch(interior, ax, 'white')

    return ax


def add_polygon_patch(coords, ax, fc='blue'):
    patch = patches.Polygon(np.array(coords.xy).T, fc=fc)
    ax.add_patch(patch)


def draw_epipolar_lines(img, cal_data, cam_num, pts, reproj_pts, markertype=['o', '+'], ax=None, lwidth=0.5):

    cam_idx = cam_num - 1

    if ax is None:
        if len(plt.get_fignums()) == 0:
            # no figures exist
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()

    dotsize = 3
    reproj_pts = np.squeeze(reproj_pts)
    pts = np.squeeze(pts)

    if img.ndim == 2:
        h, w = np.shape(img)
    elif img.ndim == 3:
        h, w, _ = np.shape(img)

    im_size = (w, h)

    mtx = cal_data['mtx'][cam_idx]
    dist = cal_data['dist'][cam_idx]

    F = cal_data['F']

    whichImage = 3 - cam_num

    img_ud = cv2.undistort(img, mtx, dist)
    ax.imshow(img_ud)

    pts_ud_norm = cv2.undistortPoints(pts, mtx, dist)
    pts_ud_norm = np.squeeze(pts_ud_norm)
    pts_ud = cvb.unnormalize_points(pts_ud_norm, mtx)

    draw_cb_epipolar_lines(pts_ud, whichImage, F, im_size, ax, cal_data['cb_size'], lwidth=lwidth)

    return ax

    #
    #
    # for i_cam in range(2):
    #     # distorted original image
    #     axs[0][i_cam].imshow(img[i_cam])
    #
    #
    #     points_in_img = pts[i_cam]
    #     # undistort points
    #     pt_ud_norm = np.squeeze(cv2.undistortPoints(points_in_img, mtx, dist))
    #     pt_ud = cvb.unnormalize_points(pt_ud_norm, mtx)
    #     if np.shape(points_in_img)[0] == 1:
    #         # only one point
    #         pt_ud = [pt_ud]
    #     for i_pt, pt in enumerate(pt_ud):
    #         if len(pt) > 0:
    #             try:
    #                 x, y = pt[0]
    #             except:
    #                 x, y = pt
    #             # x = int(round(x))
    #             # y = int(round(y))
    #             bp_color = color_from_bodypart(bodyparts[i_pt])   # undistorted point identified by DLC
    #
    #             axs[0][i_cam].plot(x, y, marker=markertype[0], ms=dotsize, color=bp_color)
    #             x2 = points_in_img[i_pt, 0]
    #             y2 = points_in_img[i_pt, 1]
    #             axs[0][i_cam].plot(x2, y2, marker='+', ms=dotsize, color=bp_color)   # point from DLC with original image disortion
    #
    #             if reproj_pts[i_cam].ndim == 1:
    #                 x3 = reproj_pts[i_cam][0]
    #                 y3 = reproj_pts[i_cam][1]
    #             else:
    #                 x3 = reproj_pts[i_cam][i_pt, 0]
    #                 y3 = reproj_pts[i_cam][i_pt, 1]
    #             axs[0][i_cam].plot(x3, y3, marker='s', ms=dotsize, color=bp_color)    # reprojected point
    #
    #     if np.shape(points_in_img)[0] == 1:
    #         pt_ud = pt_ud[0].reshape((1, -1, 2))  # needed to get correct array shape for computeCorrespondEpilines in draw_epipolar_lines_on_img
    #     draw_epipolar_lines_on_img(pt_ud, i_cam+1, cal_data['F'], im_size, bodyparts, axs[0][1-i_cam])



def draw_cb_epipolar_lines(img_pts, whichImage, F, im_size, ax, cb_size, lwidth=0.5):

    epilines = cv2.computeCorrespondEpilines(img_pts, whichImage, F)

    # copying color map from opencv drawchessboardcorners (github.com/opencv/opencv/blob/master/modules/calib3d/src/calibinit.cpp#L2156
    line_colors = [[0., 0., 1.],
                [0., 128. / 255., 1.],
                [0., 200. / 255., 200. / 255.],
                [0., 1., 0],
                [200. / 255., 200. / 255., 0.],
                [1., 0., 0.],
                [1., 0., 1.]]

    for i_line, epiline in enumerate(epilines):

        # todo: figure out how to match colors to checkerboard points
        epiline = np.squeeze(epiline)
        edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

        if not np.all(edge_pts==0):
            col_idx = int(i_line / 7.) % 7

            try:
                ax.plot(edge_pts[:, 0], edge_pts[:, 1], color=line_colors[col_idx], ls='-', marker='.', lw=lwidth)
            except:
                pass