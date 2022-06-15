import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_3d_skeleton(paw_trajectory, bodyparts, ax=None, trail_pts=3):

    pass


def overlay_pts_on_video(paw_trajectory, cal_data, bodyparts, orig_vid_name, crop_region, frame_num, ax=None, trail_pts=3):

    pass

def create_vids_plus_3danimation_figure(figsize=(9, 4), dpi=100.):

    fig = plt.figure(figsize=figsize, dpi=dpi)

    axs = []
    axs.append(fig.add_subplot(1, 3, 1))
    axs.append(fig.add_subplot(1, 3, 2))
    axs.append(fig.add_subplot(1, 3, 3, projection='3d'))

    for ax in axs[:2]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    axs[2].xaxis.set_ticks([])
    axs[2].yaxis.set_ticks([])
    axs[2].zaxis.set_ticks([])

    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_zlabel('z')

    return fig, axs


def animate_vids_plus3d(traj_data, crop_regions, orig_video_name):

    fig, axs = create_vids_plus_3danimation_figure()

    bp_coords_ud = traj_data['bp_coords_ud']
    vid_obj = cv2.VideoCapture(orig_video_name)

    num_vid_frames = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
    num_data_frames = np.shape(bp_coords_ud[0])[1]

    if num_vid_frames != num_data_frames:
        print('frame number mismatch for {}'.format(orig_video_name))
        vid_obj.release()
        return

    cal_data = traj_data['cal_data']
    mtx = cal_data['mtx']
    dist = cal_data['dist']
    num_views = len(crop_regions)
    bodyparts = traj_data['bodyparts']
    bpts2connect = bodyparts2connect()
    for i_frame in range(num_data_frames):

        # read in the first image
        ret, img = vid_obj.read()

        # undistort the image
        img_ud = cv2.undistort(img, mtx, dist)

        for i_view in range(num_views):

            cw = crop_regions[i_view]
            show_crop_frame_with_pts()
            cropped_img = img_ud[cw[2]:cw[3], cw[0]:cw[1], :]


        plt.show()
        pass

    vid_obj.release()


def show_crop_frame_with_pts(img, cw, frame_pts, bodyparts, ax):
    cropped_img = img[cw[2]:cw[3], cw[0]:cw[1], :]
    ax.imshow(cropped_img)



def bodyparts2connect():

    bpts2connect = []

    bpts2connect.append(['leftelbow', 'leftpawdorsum'])

    bpts2connect.append(['leftpawdorsum', 'leftmcp1'])
    bpts2connect.append(['leftpawdorsum', 'leftmcp2'])
    bpts2connect.append(['leftpawdorsum', 'leftmcp3'])
    bpts2connect.append(['leftpawdorsum', 'leftmcp4'])

    bpts2connect.append(['leftmcp1', 'leftpip1'])
    bpts2connect.append(['leftmcp2', 'leftpip2'])
    bpts2connect.append(['leftmcp3', 'leftpip3'])
    bpts2connect.append(['leftmcp4', 'leftpip4'])

    bpts2connect.append(['leftpip1', 'leftdig1'])
    bpts2connect.append(['leftpip2', 'leftdig2'])
    bpts2connect.append(['leftpip3', 'leftdig3'])
    bpts2connect.append(['leftpip4', 'leftdig4'])

    bpts2connect.append(['rightelbow', 'rightpawdorsum'])

    bpts2connect.append(['rightpawdorsum', 'rightmcp1'])
    bpts2connect.append(['rightpawdorsum', 'rightmcp2'])
    bpts2connect.append(['rightpawdorsum', 'rightmcp3'])
    bpts2connect.append(['rightpawdorsum', 'rightmcp4'])

    bpts2connect.append(['rightmcp1', 'rightpip1'])
    bpts2connect.append(['rightmcp2', 'rightpip2'])
    bpts2connect.append(['rightmcp3', 'rightpip3'])
    bpts2connect.append(['rightmcp4', 'rightpip4'])

    bpts2connect.append(['rightpip1', 'rightdig1'])
    bpts2connect.append(['rightpip2', 'rightdig2'])
    bpts2connect.append(['rightpip3', 'rightdig3'])
    bpts2connect.append(['rightpip4', 'rightdig4'])

    return bpts2connect


def bp_colors():

    bp_c = {'leftear':(0, 1, 1)}
    bp_c['rightear'] = tuple(np.array(bp_c['leftear']) * 0.5)

    bp_c['lefteye'] = (1, 0, 1)
    bp_c['righteye'] = tuple(np.array(bp_c['lefteye']) * 0.5)

    bp_c['nose'] = (1, 1, 1)

    bp_c['leftelbow'] = (1, 1, 0)
    bp_c['rightelbow'] = tuple(np.array(bp_c['leftelbow']) * 0.5)

    bp_c['rightpawdorsum'] = (0, 0, 1)
    bp_c['rightpalm'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.5)
    bp_c['rightmcp1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightmcp2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightmcp3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightmcp4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['rightpip1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightpip2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightpip3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightpip4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['rightdig1'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.9)
    bp_c['rightdig2'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.8)
    bp_c['rightdig3'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.7)
    bp_c['rightdig4'] = tuple(np.array(bp_c['rightpawdorsum']) * 0.6)

    bp_c['leftpawdorsum'] = (1, 0, 0)
    bp_c['leftpalm'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.5)
    bp_c['leftmcp1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftmcp2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftmcp3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftmcp4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['leftpip1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftpip2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftpip3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftpip4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['leftdig1'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.9)
    bp_c['leftdig2'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.8)
    bp_c['leftdig3'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.7)
    bp_c['leftdig4'] = tuple(np.array(bp_c['leftpawdorsum']) * 0.6)

    bp_c['pellet1'] = (0, 0, 0)
    bp_c['pellet2'] = (0.1, 0.1, 0.1)
    bp_c['pellet3'] = (0.2, 0.2, 0.2)

    return bp_c