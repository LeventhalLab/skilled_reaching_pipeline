import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import cv2
import numpy as np
import os
import stat
import shutil
import navigation_utilities
import reconstruct_3d_optitrack
import computer_vision_basics as cvb
import subprocess
import sr_photometry_analysis as srphot_anal
import skilled_reaching_io
import integrate_phys_kinematics as ipk



def plot_anipose_results(traj3d_fname, session_metadata, rat_df, parent_directories, session_summary, trials_df, test_frame=297, pawparts2plot=['pawdorsum', 'palm', 'dig1', 'dig2','dig3','dig4']):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    col_list = ['k','b','r','g','m','c']

    traj_metadata = navigation_utilities.parse_trajectory_name(traj3d_fname)
    traj_metadata['session_num'] = session_metadata['session_num']
    traj_metadata['task'] = session_metadata['task']

    summary_3dbasename = navigation_utilities.get_3dsummaries_basename(traj_metadata, session_metadata, parent_directories)
    scores_fname = summary_3dbasename + '_scores.pdf'
    pawtraces_fname = summary_3dbasename + '_pawtraces.pdf'
    imgsamp_fname = summary_3dbasename + '_imgsamp.tiff'

    if not (os.path.exists(scores_fname) and os.path.exists(pawtraces_fname) and os.path.exists(imgsamp_fname)):
        _, traj_name = os.path.split(traj3d_fname)
        traj_name, _ = os.path.splitext(traj_name)

        df_row = rat_df[rat_df['ratid'] == session_metadata['ratID']]
        paw_pref = df_row['pawpref'].values[0]
        bpts2plot = [paw_pref.lower() + pawpart for pawpart in pawparts2plot]#  'rightpawdorsum', 'rightdig1', 'rightdig2', 'rightdig3', 'rightdig4']
        num_bpts = len(bpts2plot)
        r3d_data = skilled_reaching_io.read_pickle(traj3d_fname)

        fig_2dproj = plt.figure(figsize=(8.5, 11))
        axs_2dproj = [fig_2dproj.add_subplot(311)]
        axs_2dproj.append(fig_2dproj.add_subplot(312))
        axs_2dproj.append(fig_2dproj.add_subplot(313))

        fig_scores = plt.figure(figsize=(8.5, 11))
        axs_scores = [fig_scores.add_subplot(num_bpts, 1, 1)]
        for i_bpt in range(1, num_bpts):
            axs_scores.append(fig_scores.add_subplot(num_bpts, 1, i_bpt + 1))

        # fig_2dtrack = plt.figure(figsize=(9.4,12))
        # axs_2dtrack = [fig_2dtrack.add_subplot(321)]
        # axs_2dtrack.append([fig_2dtrack.add_subplot(322)])
        # axs_2dtrack.append([fig_2dtrack.add_subplot(323)])
        # axs_2dtrack.append([fig_2dtrack.add_subplot(324)])
        # axs_2dtrack.append([fig_2dtrack.add_subplot(325)])
        # axs_2dtrack.append([fig_2dtrack.add_subplot(326)])

        # fig_3d = plt.figure(figsize=(6, 6))

        bpt_idx = []
        for i_bpt, bpt2plot in enumerate(bpts2plot):
            bpt_idx.append(r3d_data['dlc_output']['bodyparts'].index(bpt2plot))
            cur_bpt_idx = bpt_idx[i_bpt]

            for i_axis in range(3):
                axs_2dproj[i_axis].plot(r3d_data['points3d'][:, bpt_idx, i_axis], linestyle='--')
                axs_2dproj[i_axis].plot(r3d_data['points3d'][:, bpt_idx, i_axis], linestyle='--')

                axs_2dproj[i_axis].plot(r3d_data['optim_points3d'][:, bpt_idx, i_axis])
                axs_2dproj[i_axis].plot(r3d_data['optim_points3d'][:, bpt_idx, i_axis])

                axs_2dproj[i_axis].set_xlim([200, 500])

            for i_cam in range(3):
                axs_scores[i_bpt].plot(np.squeeze(r3d_data['dlc_output']['scores'][i_cam, :, cur_bpt_idx]), color=color_cycle[i_bpt])
            axs_scores[i_bpt].set_xlim([200, 500])
            axs_scores[i_bpt].set_title(bpt2plot)
            if i_bpt < num_bpts:
                axs_scores[i_bpt].tick_params(labelbottom=False)
        axs_scores[num_bpts-1].set_xlabel('frame number')

        axs_2dproj[0].set_title('x')
        axs_2dproj[1].set_title('y')
        axs_2dproj[2].set_title('z')
        axs_2dproj[2].set_xlabel('frame number')

        fig_2dproj.suptitle(traj_name, fontsize=16)
        fig_scores.suptitle(traj_name, fontsize=16)

        plt.figure(fig_2dproj)
        # plt.savefig(pawtraces_fname, format='pdf')
        plt.figure(fig_scores)
        # plt.savefig(scores_fname, format='pdf')

        orig_vid = navigation_utilities.find_orig_rat_video(traj_metadata, parent_directories['videos_root_folder'])

        cap = cv2.VideoCapture(orig_vid)

        cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
        ret, img = cap.read()

        cap.release()

        cam_intrinsics = r3d_data['calibration_data']['cam_intrinsics']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_ud = cv2.undistort(img, cam_intrinsics['mtx'], cam_intrinsics['dist'])

        fig_img = plt.figure()
        ax_img = fig_img.add_subplot()
        ax_img.imshow(img_ud)

        dlc_coords = r3d_data['dlc_output']['points']
        for i_view in range(3):
            for i_bpt, bpt2plot in enumerate(bpts2plot):
                cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)

                ax_img.scatter(dlc_coords[i_view, test_frame, cur_bpt_idx, 0], dlc_coords[i_view, test_frame, cur_bpt_idx, 1], s=3, color=color_cycle[i_bpt])

        plt.savefig(imgsamp_fname, format='tiff', dpi=600)

        plt.close('all')

    create_anipose_vids(traj3d_fname, session_metadata, parent_directories, session_summary, trials_df, paw_pref)



def create_anipose_vids(traj3d_fname, session_metadata, parent_directories, session_summary, trials_df, paw_pref,
                        bpts2plot='reachingpaw', phot_ylim=[-2.5, 5]):

    vid_params = {'lm': 0.05,
                  'rm': 0.,
                  'tm': 0.,
                  'bm': 0.05}
    vid_params['rm'] = 1 - vid_params['lm']
    vid_params['tm'] = 1 - vid_params['bm']

    fps = 300    # need to record this somewhere in the data - maybe in the .log file?

    traj_metadata = navigation_utilities.parse_trajectory_name(traj3d_fname)
    traj_metadata['session_num'] = session_metadata['session_num']
    traj_metadata['task'] = session_metadata['task']

    animation_name = navigation_utilities.create_3dvid_name(traj_metadata, session_metadata, parent_directories)
    if os.path.exists(animation_name):
        return

    print('creating video for {}'.format(animation_name))
    markersize = 5
    cmap = cm.get_cmap('rainbow')

    bpts2connect = rat_sr_bodyparts2connect()

    r3d_data = skilled_reaching_io.read_pickle(traj3d_fname)

    if bpts2plot == 'all':
        bpts2plot = r3d_data['dlc_output']['bodyparts']
    elif bpts2plot == 'reachingpaw':
        bodyparts = r3d_data['dlc_output']['bodyparts']
        bpts2plot = ['leftear', 'rightear', 'lefteye', 'righteye', 'nose', 'pellet']
        mcp_names = ['mcp{:d}'.format(i_dig + 1) for i_dig in range(4)]
        pip_names = ['pip{:d}'.format(i_dig + 1) for i_dig in range(4)]
        dig_names = ['dig{:d}'.format(i_dig + 1) for i_dig in range(4)]

        all_reaching_parts = ['elbow'] + ['pawdorsum'] + mcp_names + pip_names + dig_names
        bpts2plot = bpts2plot + [paw_pref + part_name for part_name in all_reaching_parts]

    num_bpts2plot = len(bpts2plot)
    num_bptstotal = len(r3d_data['dlc_output']['bodyparts'])

    orig_vid = navigation_utilities.find_orig_rat_video(traj_metadata, parent_directories['videos_root_folder'])

    dlc_coords = r3d_data['dlc_output']['points']
    num_frames = np.shape(r3d_data['points3d'])[0]
    cam_intrinsics = r3d_data['calibration_data']['cam_intrinsics']
    scores = r3d_data['dlc_output']['scores']
    min_valid_score = r3d_data['anipose_config']['triangulation']['score_threshold']
    cap = cv2.VideoCapture(orig_vid)

    vidtrigger_ts, vidtrigger_interval = ipk.get_vidtrigger_ts(traj_metadata, trials_df)
    if vidtrigger_ts is None:
        # most likely, trial occurred after photometry recording ended
        return

    Fs = session_summary['sr_processed_phot']['Fs']
    vid_phot_signal = srphot_anal.resample_photometry_to_video(session_summary['sr_zscores1'], vidtrigger_ts, Fs, trigger_frame=300, num_frames=num_frames, fps=fps)
    t = np.linspace(1/fps, num_frames/fps, num_frames)

    session_folder, _ = os.path.split(traj3d_fname)
    jpg_folder = os.path.join(session_folder, 'temp')
    if not os.path.exists(jpg_folder):
        # os.chmod(jpg_folder, stat.S_IWRITE)
        # shutil.rmtree(jpg_folder)
        os.makedirs(jpg_folder)

    # change "optim_points3d" to "points3d" to switch to reprojection from simple triangulation
    pts3d_reproj_key = 'optim_points3d'

    for i_frame in range(num_frames):

        frame_fig = plt.figure(figsize=(20, 10))
        gs = frame_fig.add_gridspec(3, 3, width_ratios=(8, 2, 1), height_ratios=(1, 3, 4), wspace=0.05, hspace=0.02,
                                    left=vid_params['lm'], right=vid_params['rm'], top=vid_params['tm'], bottom=vid_params['bm'])

        vid_ax = frame_fig.add_subplot(gs[1:, 0])
        ax3d = frame_fig.add_subplot(gs[:2, 1], projection='3d')
        ax3d_optim = frame_fig.add_subplot(gs[2, 1], projection='3d')
        legend_ax = frame_fig.add_subplot(gs[:, 2])
        phot_trace_ax = frame_fig.add_subplot(gs[0, 0])

        phot_trace_ax.set_ylim(phot_ylim)
        phot_trace_ax.set_ylabel('DF/F z-score')
        phot_trace_ax.set_xlim([0, max(t)])
        phot_trace_ax.set_xticks([0, 300/fps, max(t)])
        if not vid_phot_signal is None:
            # only plot if a photometry signal was recorded during this trial
            phot_trace_ax.plot(t[:i_frame+1], vid_phot_signal[:i_frame+1], color='g')

        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        img_ud = cv2.undistort(img, cam_intrinsics['mtx'], cam_intrinsics['dist'])
        h, w, _ = np.shape(img_ud)

        if pts3d_reproj_key == 'optim_points3d':
            reproj_text = 'reprojected optimal 3d points'
        else:
            reproj_text = 'reprojected simple triangulation 3d points'
        frame_text = session_metadata['ratID'] + ', ' + 'frame {:04d}'.format(i_frame) + ', ' + reproj_text
        img_ud = cv2.putText(img_ud, frame_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=0, thickness=3)

        for bpt2plot in bpts2plot:
            cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)
            legend_ax.text(0, cur_bpt_idx/num_bptstotal, bpt2plot, color=cmap(cur_bpt_idx / num_bptstotal), transform=legend_ax.transAxes)
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])

        vid_ax.imshow(img_ud)
        for i_view in range(3):
            for i_bpt, bpt2plot in enumerate(bpts2plot):
                cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)

                col = cmap(cur_bpt_idx / num_bptstotal)

                p3d = r3d_data[pts3d_reproj_key][i_frame, cur_bpt_idx, :]
                reproj = np.squeeze(r3d_data['calibration_data']['cgroup'].cameras[i_view].project(p3d).reshape([1, 2]))
                if scores[i_view, i_frame, i_bpt] > min_valid_score:
                    vid_ax.scatter(dlc_coords[i_view, i_frame, cur_bpt_idx, 0],
                                   dlc_coords[i_view, i_frame, cur_bpt_idx, 1], s=markersize, color=col)
                    vid_ax.scatter(reproj[0], reproj[1], s=markersize, color=col, marker='+')

                else:
                    vid_ax.scatter(dlc_coords[i_view, i_frame, cur_bpt_idx, 0],
                                   dlc_coords[i_view, i_frame, cur_bpt_idx, 1], s=markersize, color=col, marker='*')

        for bpt2plot in bpts2plot:
            cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)
            ax3d.scatter(r3d_data['points3d'][i_frame, cur_bpt_idx, 0],
                         r3d_data['points3d'][i_frame, cur_bpt_idx, 2],
                         r3d_data['points3d'][i_frame, cur_bpt_idx, 1],
                         s=markersize,
                         color=cmap(cur_bpt_idx / num_bptstotal))

            ax3d_optim.scatter(r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 0],
                         r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 2],
                         r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 1],
                         s=markersize,
                         color=cmap(cur_bpt_idx / num_bptstotal))
        ax3d.set_title('simple triangulation')
        ax3d_optim.set_title('optimized triangulation')

        connect_3d_bpts(r3d_data['points3d'][i_frame, :, :], r3d_data['dlc_output']['bodyparts'], bpts2plot, bpts2connect, ax3d)
        connect_3d_bpts(r3d_data['optim_points3d'][i_frame, :, :], r3d_data['dlc_output']['bodyparts'], bpts2plot, bpts2connect, ax3d_optim)

        ax3d.set_xlim((-50, 25))
        ax3d.set_ylim((200, 350))  # this is actually z
        ax3d.set_zlim((20, 120))  # this is actually y

        ax3d.set_xlabel('x')
        ax3d.set_ylabel('z')
        ax3d.set_zlabel('y')
        ax3d.invert_zaxis()

        ax3d_optim.set_xlim((-40, 25))
        ax3d_optim.set_ylim((225, 350))  # this is actually z
        ax3d_optim.set_zlim((20, 100))  # this is actually y

        ax3d_optim.set_xlabel('x')
        ax3d_optim.set_ylabel('z')
        ax3d_optim.set_zlabel('y')
        ax3d_optim.invert_zaxis()

        vid_ax.set_xlim((0, w - 1))
        vid_ax.set_ylim((0, h - 1))
        vid_ax.invert_yaxis()
        vid_ax.set_xticks([])
        vid_ax.set_yticks([])

        jpg_name = os.path.join(jpg_folder, 'frame{:04d}.jpg'.format(i_frame))
        plt.savefig(jpg_name, format='jpeg')
        plt.close('all')

    cap.release()


    # turn the cropped jpegs into a new movie
    jpg_names = os.path.join(jpg_folder, 'frame%04d.jpg')
    command = (
        f"ffmpeg -i {jpg_names} "
        f"-c:v copy {animation_name}"
    )
    subprocess.call(command, shell=True)

    # shutil.rmtree(jpg_folder)


# def create_3dgrant_vid(traj3d_fname, session_metadata, parent_directories, session_summary, trials_df, paw_pref,
#                         bpts2plot='all', phot_ylim=[-2.5, 5], cw=[[850, 1250, 450, 900], [175, 600, 475, 825], [1460, 1875, 500, 850]],
#                         lim_3d=[[-30, 30], [0, 80], [280, 340]]):
def create_3dgrant_vid(traj3d_fname, session_metadata, parent_directories, session_summary, trials_df, paw_pref,
                       bpts2plot='all', phot_ylim=[-2.5, 7],
                       cw=[[125, 1875, 450, 890]], lim_3d=[[-30, 30], [0, 80], [280, 340]]):

    vid_params = {'lm': 0.0,
                  'rm': 0.,
                  'tm': 0.,
                  'bm': 0.0}
    vid_params['rm'] = 1 - vid_params['lm']
    vid_params['tm'] = 1 - vid_params['bm']

    fps = 300    # need to record this somewhere in the data - maybe in the .log file?

    traj_metadata = navigation_utilities.parse_trajectory_name(traj3d_fname)
    traj_metadata['session_num'] = session_metadata['session_num']
    traj_metadata['task'] = session_metadata['task']

    animation_name = navigation_utilities.create_cropped_3dvid_name(traj_metadata, session_metadata, parent_directories)
    # if os.path.exists(animation_name):
    #     return

    print('creating video for {}'.format(animation_name))
    vid_markersize = 3
    markersize_skel = 2
    cmap = cm.get_cmap('rainbow')

    bpts2connect = rat_sr_bodyparts2connect()

    r3d_data = skilled_reaching_io.read_pickle(traj3d_fname)

    if bpts2plot == 'all':
        bpts2plot = r3d_data['dlc_output']['bodyparts']
    elif bpts2plot == 'reachingpaw':
        bodyparts = r3d_data['dlc_output']['bodyparts']
        bpts2plot = ['leftear', 'rightear', 'lefteye', 'righteye', 'nose', 'pellet']
        mcp_names = ['mcp{:d}'.format(i_dig + 1) for i_dig in range(4)]
        pip_names = ['pip{:d}'.format(i_dig + 1) for i_dig in range(4)]
        dig_names = ['dig{:d}'.format(i_dig + 1) for i_dig in range(4)]

        all_reaching_parts = ['elbow'] + ['pawdorsum'] + mcp_names + pip_names + dig_names
        bpts2plot = bpts2plot + [paw_pref + part_name for part_name in all_reaching_parts]

    num_bpts2plot = len(bpts2plot)
    num_bptstotal = len(r3d_data['dlc_output']['bodyparts'])

    orig_vid = navigation_utilities.find_orig_rat_video(traj_metadata, parent_directories['videos_root_folder'])

    dlc_coords = r3d_data['dlc_output']['points']
    n_frames = np.shape(r3d_data['points3d'])[0]
    cam_intrinsics = r3d_data['calibration_data']['cam_intrinsics']
    scores = r3d_data['dlc_output']['scores']
    min_valid_score = r3d_data['anipose_config']['triangulation']['score_threshold']
    cap = cv2.VideoCapture(orig_vid)

    vidtrigger_ts, vidtrigger_interval = ipk.get_vidtrigger_ts(traj_metadata, trials_df)
    if vidtrigger_ts is None:
        # most likely, trial occurred after photometry recording ended
        return

    Fs = session_summary['sr_processed_phot']['Fs']
    vid_phot_signal = srphot_anal.resample_photometry_to_video(session_summary['sr_zscores1'], vidtrigger_ts, Fs, trigger_frame=300, n_frames=n_frames, fps=fps)
    # vid_phot_signal = None
    t = np.linspace(1/fps, n_frames/fps, n_frames)

    session_folder, _ = os.path.split(traj3d_fname)
    jpg_folder = os.path.join(session_folder, 'temp')
    if not os.path.exists(jpg_folder):
        # os.chmod(jpg_folder, stat.S_IWRITE)
        # shutil.rmtree(jpg_folder)
        os.makedirs(jpg_folder)

    # change "optim_points3d" to "points3d" to switch to reprojection from simple triangulation
    # pts3d_reproj_key = 'optim_points3d'
    pts3d_reproj_key = 'points3d'

    for i_frame in range(n_frames):

        frame_fig = plt.figure(figsize=(8, 3), dpi=300)
        # gs = frame_fig.add_gridspec(ncols=3, nrows=4, width_ratios=(1, 1, 1), height_ratios=(2, 2, 2, 6), wspace=0.0, hspace=0.02,
        #                             left=vid_params['lm'], right=vid_params['rm'], top=vid_params['tm'], bottom=vid_params['bm'])

        # vid_ax = frame_fig.add_subplot(gs[1:, 0])
        '''MODIFY HERE TO REORGANIZE PLOTS'''
        # img_row = 3
        # view_ax = [frame_fig.add_subplot(gs[img_row, 1])]             # direct view
        # view_ax.append(frame_fig.add_subplot(gs[img_row, 0]))         # left view
        # view_ax.append(frame_fig.add_subplot(gs[img_row, 2]))         # right view

        img_row = 3
        # view_ax = frame_fig.add_subplot(gs[img_row, 1])
        view_ax = frame_fig.add_subplot()

        # ax3d = frame_fig.add_subplot(gs[0:img_row, :2], projection='3d')
        ax3d = view_ax.inset_axes([1075, 0, 200, 200], transform=view_ax.transData, projection='3d')

        # legend_ax = frame_fig.add_subplot(gs[:, 2])
        # phot_trace_ax = frame_fig.add_subplot(gs[1, 2])
        phot_trace_ax = view_ax.inset_axes([400, 0, 400, 150], transform=view_ax.transData)
        # phot_trace_ax = inset_axes(view_ax, width=3., height=1., loc=2)

        phot_trace_ax.set_ylim(phot_ylim)
        # phot_trace_ax.set_ylabel('DF/F z-score')
        phot_trace_ax.set_xlim([0, max(t)])
        phot_trace_ax.set_xticks([0, 300/fps, max(t)])
        phot_trace_ax.set_yticks([])

        # plot a vertical line with DF/F = 2
        scale_bar_x = 0.1
        scale_bar_y = np.array([1, 5])
        phot_trace_ax.plot([scale_bar_x, scale_bar_x], scale_bar_y, color='w', lw=0.5)
        phot_trace_ax.text(scale_bar_x + 0.1, np.mean(scale_bar_y)-1, '{:d}'.format(int(np.diff(scale_bar_y)[0])), color='w', fontsize='x-small')
        phot_trace_ax.text(scale_bar_x - 0.4, np.mean(scale_bar_y)-0.05, 'zDF/F', rotation='vertical', verticalalignment='center', color='w', fontsize='x-small')
        phot_trace_ax.axis('off')
        if not vid_phot_signal is None:
            # only plot if a photometry signal was recorded during this trial
            phot_trace_ax.plot(t[:i_frame+1], vid_phot_signal[:i_frame+1], color='g', lw=1.)

        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        img_ud = cv2.undistort(img, cam_intrinsics['mtx'], cam_intrinsics['dist'])
        h, w, _ = np.shape(img_ud)

        if pts3d_reproj_key == 'optim_points3d':
            reproj_text = 'reprojected optimal 3d points'
        else:
            reproj_text = 'reprojected simple triangulation 3d points'
        frame_text = session_metadata['ratID'] + ', ' + 'frame {:04d}'.format(i_frame) + ', ' + reproj_text
        img_ud = cv2.putText(img_ud, frame_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=0, thickness=3)

        for bpt2plot in bpts2plot:
            cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)
        #     legend_ax.text(0, cur_bpt_idx/num_bptstotal, bpt2plot, color=cmap(cur_bpt_idx / num_bptstotal), transform=legend_ax.transAxes)
        # legend_ax.set_xticks([])
        # legend_ax.set_yticks([])

        # for i_view in range(3):
        #     view_ax[i_view].set_xticks([])
        #     view_ax[i_view].set_yticks([])
        #     view_ax[i_view].imshow(img_ud[cw[i_view][2] : cw[i_view][3], cw[i_view][0] : cw[i_view][1], :])
        view_ax.imshow(img_ud[cw[0][2] : cw[0][3], cw[0][0] : cw[0][1]])
        # lm_cw = cw[1]
        # view_ax[1].imshow(img_ud[lm_cw[2] : lm_cw[3], lm_cw[0] : lm_cw[1], :])
        # dir_cw = cw[0]
        # dir_ax.imshow(img_ud[dir_cw[2]: dir_cw[3], dir_cw[0] : dir_cw[1], :])
        # rm_cw = cw[2]
        # rm_ax.imshow(img_ud[rm_cw[2] : rm_cw[3], rm_cw[0] : rm_cw[1], :])
        for i_view in range(3):
            for i_bpt, bpt2plot in enumerate(bpts2plot):
                cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)

                col = cmap(cur_bpt_idx / num_bptstotal)

                p3d = r3d_data[pts3d_reproj_key][i_frame, cur_bpt_idx, :]
                reproj = np.squeeze(r3d_data['calibration_data']['cgroup'].cameras[i_view].project(p3d).reshape([1, 2]))
                if scores[i_view, i_frame, i_bpt] > min_valid_score:
                    # make sure points are within the crop window
                    x_shifted = dlc_coords[i_view, i_frame, cur_bpt_idx, 0] - cw[0][0]
                    y_shifted = dlc_coords[i_view, i_frame, cur_bpt_idx, 1] - cw[0][2]
                    valid_pt = True
                    if x_shifted < 0 or x_shifted > (cw[0][1] - cw[0][0]):
                        valid_pt = False
                    if y_shifted < 0 or y_shifted > (cw[0][3] - cw[0][2]):
                        valid_pt = False
                    if valid_pt:
                        view_ax.scatter(x_shifted, y_shifted, s=vid_markersize, color=col, edgecolor='none')
                    # vid_ax.scatter(reproj[0], reproj[1], s=markersize, color=col, marker='+')

                else:
                    pass
                    # vid_ax.scatter(dlc_coords[i_view, i_frame, cur_bpt_idx, 0],
                    #                dlc_coords[i_view, i_frame, cur_bpt_idx, 1], s=markersize, color=col, marker='*')

        for bpt2plot in bpts2plot:
            cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)
            # ax3d.scatter(r3d_data['points3d'][i_frame, cur_bpt_idx, 0],
            #              r3d_data['points3d'][i_frame, cur_bpt_idx, 2],
            #              r3d_data['points3d'][i_frame, cur_bpt_idx, 1],
            #              s=markersize,
            #              color=cmap(cur_bpt_idx / num_bptstotal))

            ax3d.scatter(r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 0],
                         r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 2],
                         r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 1],
                         s=markersize_skel,
                         color=cmap(cur_bpt_idx / num_bptstotal), edgecolor='none')

        # ax3d.set_title('optimized triangulation')

        # connect_3d_bpts(r3d_data['points3d'][i_frame, :, :], r3d_data['dlc_output']['bodyparts'], bpts2plot, bpts2connect, ax3d)
        connect_3d_bpts(r3d_data['optim_points3d'][i_frame, :, :], r3d_data['dlc_output']['bodyparts'], bpts2plot, bpts2connect, ax3d)

        ax3d.set_xlim((lim_3d[0][0], lim_3d[0][1]))
        ax3d.set_ylim((lim_3d[2][0], lim_3d[2][1]))  # this is actually z
        ax3d.set_zlim((lim_3d[1][0], lim_3d[1][1]))  # this is actually y

        # ax3d.set_xlabel('x')
        # ax3d.set_ylabel('z')
        # ax3d.set_zlabel('y')
        ax3d.invert_zaxis()
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])

        view_ax.set_xticks([])
        view_ax.set_yticks([])
        view_ax.axis('off')
        # ax3d.axis('off')

        # ax3d.set_xlim((-40, 25))
        # ax3d.set_ylim((225, 350))  # this is actually z
        # ax3d.set_zlim((20, 100))  # this is actually y

        # ax3d.set_xlabel('x')
        # ax3d.set_ylabel('z')
        # ax3d.set_zlabel('y')
        # ax3d.invert_zaxis()

        # vid_ax.set_xlim((0, w - 1))
        # vid_ax.set_ylim((0, h - 1))
        # vid_ax.invert_yaxis()
        # vid_ax.set_xticks([])
        # vid_ax.set_yticks([])

        jpg_name = os.path.join(jpg_folder, 'frame{:04d}.jpg'.format(i_frame))
        plt.savefig(jpg_name, format='jpeg', dpi=300)
        plt.close('all')

    cap.release()


    # turn the cropped jpegs into a new movie
    jpg_names = os.path.join(jpg_folder, 'frame%04d.jpg')
    command = (
        f"ffmpeg -i {jpg_names} "
        f"-c:v copy {animation_name}"
    )
    subprocess.call(command, shell=True)



def create_presentation_vid(traj3d_fname, session_metadata, parent_directories, session_summary, trials_df, paw_pref,
                        bpts2plot='all', phot_ylim=[-2.5, 5], cw=[[850, 1250, 450, 900], [175, 600, 475, 825], [1460, 1875, 500, 850]],
                        lim_3d=[[-30, 30], [0, 80], [280, 340]]):

    vid_params = {'lm': 0.0,
                  'rm': 0.,
                  'tm': 0.,
                  'bm': 0.0}
    vid_params['rm'] = 1 - vid_params['lm']
    vid_params['tm'] = 1 - vid_params['bm']

    fps = 300    # need to record this somewhere in the data - maybe in the .log file?

    traj_metadata = navigation_utilities.parse_trajectory_name(traj3d_fname)
    traj_metadata['session_num'] = session_metadata['session_num']
    traj_metadata['task'] = session_metadata['task']

    animation_name = navigation_utilities.create_cropped_3dvid_name(traj_metadata, session_metadata, parent_directories)
    # if os.path.exists(animation_name):
    #     return

    print('creating video for {}'.format(animation_name))
    markersize = 5
    cmap = cm.get_cmap('rainbow')

    bpts2connect = rat_sr_bodyparts2connect()

    r3d_data = skilled_reaching_io.read_pickle(traj3d_fname)

    if bpts2plot == 'all':
        bpts2plot = r3d_data['dlc_output']['bodyparts']
    elif bpts2plot == 'reachingpaw':
        bodyparts = r3d_data['dlc_output']['bodyparts']
        bpts2plot = ['leftear', 'rightear', 'lefteye', 'righteye', 'nose', 'pellet']
        mcp_names = ['mcp{:d}'.format(i_dig + 1) for i_dig in range(4)]
        pip_names = ['pip{:d}'.format(i_dig + 1) for i_dig in range(4)]
        dig_names = ['dig{:d}'.format(i_dig + 1) for i_dig in range(4)]

        all_reaching_parts = ['elbow'] + ['pawdorsum'] + mcp_names + pip_names + dig_names
        bpts2plot = bpts2plot + [paw_pref + part_name for part_name in all_reaching_parts]

    num_bpts2plot = len(bpts2plot)
    num_bptstotal = len(r3d_data['dlc_output']['bodyparts'])

    orig_vid = navigation_utilities.find_orig_rat_video(traj_metadata, parent_directories['videos_root_folder'])

    dlc_coords = r3d_data['dlc_output']['points']
    n_frames = np.shape(r3d_data['points3d'])[0]
    cam_intrinsics = r3d_data['calibration_data']['cam_intrinsics']
    scores = r3d_data['dlc_output']['scores']
    min_valid_score = r3d_data['anipose_config']['triangulation']['score_threshold']
    cap = cv2.VideoCapture(orig_vid)

    vidtrigger_ts, vidtrigger_interval = ipk.get_vidtrigger_ts(traj_metadata, trials_df)
    if vidtrigger_ts is None:
        # most likely, trial occurred after photometry recording ended
        return

    Fs = session_summary['sr_processed_phot']['Fs']
    vid_phot_signal = srphot_anal.resample_photometry_to_video(session_summary['sr_zscores1'], vidtrigger_ts, Fs, trigger_frame=300, num_frames=n_frames, fps=fps)
    # vid_phot_signal = None
    t = np.linspace(1/fps, n_frames/fps, n_frames)

    session_folder, _ = os.path.split(traj3d_fname)
    jpg_folder = os.path.join(session_folder, 'temp')
    if not os.path.exists(jpg_folder):
        # os.chmod(jpg_folder, stat.S_IWRITE)
        # shutil.rmtree(jpg_folder)
        os.makedirs(jpg_folder)

    # change "optim_points3d" to "points3d" to switch to reprojection from simple triangulation
    # pts3d_reproj_key = 'optim_points3d'
    pts3d_reproj_key = 'points3d'

    for i_frame in range(n_frames):

        frame_fig = plt.figure(figsize=(8, 6))
        gs = frame_fig.add_gridspec(ncols=3, nrows=4, width_ratios=(1, 1, 1), height_ratios=(2, 2, 2, 6), wspace=0.0, hspace=0.02,
                                    left=vid_params['lm'], right=vid_params['rm'], top=vid_params['tm'], bottom=vid_params['bm'])

        # vid_ax = frame_fig.add_subplot(gs[1:, 0])
        img_row = 3
        view_ax = [frame_fig.add_subplot(gs[img_row, 1])]             # direct view
        view_ax.append(frame_fig.add_subplot(gs[img_row, 0]))         # left view
        view_ax.append(frame_fig.add_subplot(gs[img_row, 2]))         # right view

        ax3d = frame_fig.add_subplot(gs[0:img_row, :2], projection='3d')

        # legend_ax = frame_fig.add_subplot(gs[:, 2])
        phot_trace_ax = frame_fig.add_subplot(gs[1, 2])

        phot_trace_ax.set_ylim(phot_ylim)
        # phot_trace_ax.set_ylabel('DF/F z-score')
        phot_trace_ax.set_xlim([0, max(t)])
        phot_trace_ax.set_xticks([0, 300/fps, max(t)])
        phot_trace_ax.set_yticks([])

        # plot a vertical line with DF/F = 2
        scale_bar_x = 0.2
        scale_bar_y = np.array([2, 4])
        phot_trace_ax.plot([scale_bar_x, scale_bar_x], scale_bar_y)
        phot_trace_ax.text(scale_bar_x + 0.05, np.mean(scale_bar_y)-0.05, '{:d}'.format(int(np.diff(scale_bar_y)[0])))
        phot_trace_ax.text(scale_bar_x - 0.2, np.mean(scale_bar_y)-0.05, 'DF/F', rotation='vertical', verticalalignment='center')
        phot_trace_ax.axis('off')
        if not vid_phot_signal is None:
            # only plot if a photometry signal was recorded during this trial
            phot_trace_ax.plot(t[:i_frame+1], vid_phot_signal[:i_frame+1], color='g')

        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        img_ud = cv2.undistort(img, cam_intrinsics['mtx'], cam_intrinsics['dist'])
        h, w, _ = np.shape(img_ud)

        if pts3d_reproj_key == 'optim_points3d':
            reproj_text = 'reprojected optimal 3d points'
        else:
            reproj_text = 'reprojected simple triangulation 3d points'
        frame_text = session_metadata['ratID'] + ', ' + 'frame {:04d}'.format(i_frame) + ', ' + reproj_text
        img_ud = cv2.putText(img_ud, frame_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=0, thickness=3)

        for bpt2plot in bpts2plot:
            cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)
        #     legend_ax.text(0, cur_bpt_idx/num_bptstotal, bpt2plot, color=cmap(cur_bpt_idx / num_bptstotal), transform=legend_ax.transAxes)
        # legend_ax.set_xticks([])
        # legend_ax.set_yticks([])

        for i_view in range(3):
            view_ax[i_view].set_xticks([])
            view_ax[i_view].set_yticks([])
            view_ax[i_view].imshow(img_ud[cw[i_view][2] : cw[i_view][3], cw[i_view][0] : cw[i_view][1], :])
        # lm_cw = cw[1]
        # view_ax[1].imshow(img_ud[lm_cw[2] : lm_cw[3], lm_cw[0] : lm_cw[1], :])
        # dir_cw = cw[0]
        # dir_ax.imshow(img_ud[dir_cw[2]: dir_cw[3], dir_cw[0] : dir_cw[1], :])
        # rm_cw = cw[2]
        # rm_ax.imshow(img_ud[rm_cw[2] : rm_cw[3], rm_cw[0] : rm_cw[1], :])
        # for i_view in range(3):
            for i_bpt, bpt2plot in enumerate(bpts2plot):
                cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)

                col = cmap(cur_bpt_idx / num_bptstotal)

                p3d = r3d_data[pts3d_reproj_key][i_frame, cur_bpt_idx, :]
                reproj = np.squeeze(r3d_data['calibration_data']['cgroup'].cameras[i_view].project(p3d).reshape([1, 2]))
                if scores[i_view, i_frame, i_bpt] > min_valid_score:
                    # make sure points are within the crop window
                    x_shifted = dlc_coords[i_view, i_frame, cur_bpt_idx, 0] - cw[i_view][0]
                    y_shifted = dlc_coords[i_view, i_frame, cur_bpt_idx, 1] - cw[i_view][2]
                    valid_pt = True
                    if x_shifted < 0 or x_shifted > (cw[i_view][1] - cw[i_view][0]):
                        valid_pt = False
                    if y_shifted < 0 or y_shifted > (cw[i_view][3] - cw[i_view][2]):
                        valid_pt = False
                    if valid_pt:
                        view_ax[i_view].scatter(x_shifted, y_shifted, s=markersize, color=col)
                    # vid_ax.scatter(reproj[0], reproj[1], s=markersize, color=col, marker='+')

                else:
                    pass
                    # vid_ax.scatter(dlc_coords[i_view, i_frame, cur_bpt_idx, 0],
                    #                dlc_coords[i_view, i_frame, cur_bpt_idx, 1], s=markersize, color=col, marker='*')

        for bpt2plot in bpts2plot:
            cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)
            # ax3d.scatter(r3d_data['points3d'][i_frame, cur_bpt_idx, 0],
            #              r3d_data['points3d'][i_frame, cur_bpt_idx, 2],
            #              r3d_data['points3d'][i_frame, cur_bpt_idx, 1],
            #              s=markersize,
            #              color=cmap(cur_bpt_idx / num_bptstotal))

            ax3d.scatter(r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 0],
                         r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 2],
                         r3d_data['optim_points3d'][i_frame, cur_bpt_idx, 1],
                         s=markersize,
                         color=cmap(cur_bpt_idx / num_bptstotal))

        # ax3d.set_title('optimized triangulation')

        connect_3d_bpts(r3d_data['points3d'][i_frame, :, :], r3d_data['dlc_output']['bodyparts'], bpts2plot, bpts2connect, ax3d)
        connect_3d_bpts(r3d_data['optim_points3d'][i_frame, :, :], r3d_data['dlc_output']['bodyparts'], bpts2plot, bpts2connect, ax3d)

        ax3d.set_xlim((lim_3d[0][0], lim_3d[0][1]))
        ax3d.set_ylim((lim_3d[2][0], lim_3d[2][1]))  # this is actually z
        ax3d.set_zlim((lim_3d[1][0], lim_3d[1][1]))  # this is actually y

        ax3d.set_xlabel('x')
        ax3d.set_ylabel('z')
        ax3d.set_zlabel('y')
        ax3d.invert_zaxis()
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])

        # ax3d.set_xlim((-40, 25))
        # ax3d.set_ylim((225, 350))  # this is actually z
        # ax3d.set_zlim((20, 100))  # this is actually y

        # ax3d.set_xlabel('x')
        # ax3d.set_ylabel('z')
        # ax3d.set_zlabel('y')
        # ax3d.invert_zaxis()

        # vid_ax.set_xlim((0, w - 1))
        # vid_ax.set_ylim((0, h - 1))
        # vid_ax.invert_yaxis()
        # vid_ax.set_xticks([])
        # vid_ax.set_yticks([])

        jpg_name = os.path.join(jpg_folder, 'frame{:04d}.jpg'.format(i_frame))
        plt.savefig(jpg_name, format='jpeg')
        plt.close('all')

    cap.release()


    # turn the cropped jpegs into a new movie
    jpg_names = os.path.join(jpg_folder, 'frame%04d.jpg')
    command = (
        f"ffmpeg -i {jpg_names} "
        f"-c:v copy {animation_name}"
    )
    subprocess.call(command, shell=True)


def create_presentation_vid_1view(traj3d_fname, session_metadata, parent_directories, session_summary, trials_df, paw_pref,
                        bpts2plot='all', phot_ylim=[-2.5, 5], cw=[[850, 1250, 450, 900], [175, 600, 475, 825], [1460, 1875, 500, 850]],
                        lim_3d=[[-30, 30], [0, 80], [280, 340]], frames2mark = {'reach_on': 290, 'contact': 310, 'drop': 315},
                        frame_marker_colors={'reach_on': 'g', 'contact': 'k', 'drop': 'r', 'retract': 'm'}):

    frame_markers = list(frames2mark.keys())

    vid_params = {'lm': 0.0,
                  'rm': 0.,
                  'tm': 0.,
                  'bm': 0.0}
    vid_params['rm'] = 1 - vid_params['lm']
    vid_params['tm'] = 1 - vid_params['bm']

    fps = 300    # need to record this somewhere in the data - maybe in the .log file?

    traj_metadata = navigation_utilities.parse_trajectory_name(traj3d_fname)
    traj_metadata['session_num'] = session_metadata['session_num']
    traj_metadata['task'] = session_metadata['task']

    animation_name = navigation_utilities.create_cropped1view_3dvid_name(traj_metadata, session_metadata, parent_directories)
    # if os.path.exists(animation_name):
    #     return

    print('creating video for {}'.format(animation_name))
    markersize = 5
    cmap = cm.get_cmap('rainbow')

    bpts2connect = rat_sr_bodyparts2connect()

    r3d_data = skilled_reaching_io.read_pickle(traj3d_fname)

    if bpts2plot == 'all':
        bpts2plot = r3d_data['dlc_output']['bodyparts']
    elif bpts2plot == 'reachingpaw':
        bodyparts = r3d_data['dlc_output']['bodyparts']
        bpts2plot = ['leftear', 'rightear', 'lefteye', 'righteye', 'nose', 'pellet']
        mcp_names = ['mcp{:d}'.format(i_dig + 1) for i_dig in range(4)]
        pip_names = ['pip{:d}'.format(i_dig + 1) for i_dig in range(4)]
        dig_names = ['dig{:d}'.format(i_dig + 1) for i_dig in range(4)]

        all_reaching_parts = ['elbow'] + ['pawdorsum'] + mcp_names + pip_names + dig_names
        bpts2plot = bpts2plot + [paw_pref + part_name for part_name in all_reaching_parts]

    num_bpts2plot = len(bpts2plot)
    num_bptstotal = len(r3d_data['dlc_output']['bodyparts'])

    orig_vid = navigation_utilities.find_orig_rat_video(traj_metadata, parent_directories['videos_root_folder'])

    dlc_coords = r3d_data['dlc_output']['points']
    n_frames = np.shape(r3d_data['points3d'])[0]
    cam_intrinsics = r3d_data['calibration_data']['cam_intrinsics']
    scores = r3d_data['dlc_output']['scores']
    min_valid_score = r3d_data['anipose_config']['triangulation']['score_threshold']
    cap = cv2.VideoCapture(orig_vid)

    vidtrigger_ts, vidtrigger_interval = ipk.get_vidtrigger_ts(traj_metadata, trials_df)
    if vidtrigger_ts is None:
        # most likely, trial occurred after photometry recording ended
        return

    Fs = session_summary['sr_processed_phot']['Fs']
    vid_phot_signal = srphot_anal.resample_photometry_to_video(session_summary['sr_zscores1'], vidtrigger_ts, Fs, trigger_frame=300, num_frames=n_frames, fps=fps)
    # vid_phot_signal = None
    t = np.linspace(1/fps, n_frames/fps, n_frames)

    session_folder, _ = os.path.split(traj3d_fname)
    jpg_folder = os.path.join(session_folder, 'temp')
    if not os.path.exists(jpg_folder):
        # os.chmod(jpg_folder, stat.S_IWRITE)
        # shutil.rmtree(jpg_folder)
        os.makedirs(jpg_folder)

    # change "optim_points3d" to "points3d" to switch to reprojection from simple triangulation
    # pts3d_reproj_key = 'optim_points3d'
    pts3d_reproj_key = 'points3d'

    for i_frame in range(n_frames):

        frame_fig = plt.figure(figsize=(4, 6))
        gs = frame_fig.add_gridspec(ncols=2, nrows=2, width_ratios=(1, 1), height_ratios=(1, 3), wspace=0.0, hspace=0.02,
                                    left=vid_params['lm'], right=vid_params['rm'], top=vid_params['tm'], bottom=vid_params['bm'])

        # vid_ax = frame_fig.add_subplot(gs[1:, 0])
        img_row = 1
        view_ax = [frame_fig.add_subplot(gs[img_row, :])]             # direct view
        # view_ax.append(frame_fig.add_subplot(gs[img_row, 0]))         # left view
        # view_ax.append(frame_fig.add_subplot(gs[img_row, 2]))         # right view

        # ax3d = frame_fig.add_subplot(gs[0:img_row, :2], projection='3d')

        # legend_ax = frame_fig.add_subplot(gs[:, 2])
        phot_trace_ax = frame_fig.add_subplot(gs[0, :])

        phot_trace_ax.set_ylim(phot_ylim)
        # phot_trace_ax.set_ylabel('DF/F z-score')
        phot_trace_ax.set_xlim([0, max(t)])
        phot_trace_ax.set_xticks([0, 300/fps, max(t)])
        phot_trace_ax.set_yticks([])

        # plot a vertical line with DF/F = 2
        scale_bar_x = 0.2
        scale_bar_y = np.array([2, 4])
        phot_trace_ax.plot([scale_bar_x, scale_bar_x], scale_bar_y)
        phot_trace_ax.text(scale_bar_x + 0.03, np.mean(scale_bar_y), '{:d}'.format(int(np.diff(scale_bar_y)[0])), verticalalignment='center')
        phot_trace_ax.text(scale_bar_x - 0.15, np.mean(scale_bar_y), 'DF/F', rotation='vertical', verticalalignment='center')
        phot_trace_ax.axis('off')
        if not vid_phot_signal is None:
            # only plot if a photometry signal was recorded during this trial
            phot_trace_ax.plot(t[:i_frame+1], vid_phot_signal[:i_frame+1], color='g')

            for mark_key in list(frames2mark.keys()):
                if i_frame >= frames2mark[mark_key]:
                    try:
                        phot_trace_ax.axvline(frames2mark[mark_key]/fps, 0, 1, color=frame_marker_colors[mark_key])
                    except:
                        phot_trace_ax.axvline(frames2mark[mark_key]/fps, color='k')


        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        img_ud = cv2.undistort(img, cam_intrinsics['mtx'], cam_intrinsics['dist'])
        h, w, _ = np.shape(img_ud)

        if pts3d_reproj_key == 'optim_points3d':
            reproj_text = 'reprojected optimal 3d points'
        else:
            reproj_text = 'reprojected simple triangulation 3d points'
        frame_text = 'frame {:04d}'.format(i_frame)

        for bpt2plot in bpts2plot:
            cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)
        #     legend_ax.text(0, cur_bpt_idx/num_bptstotal, bpt2plot, color=cmap(cur_bpt_idx / num_bptstotal), transform=legend_ax.transAxes)
        # legend_ax.set_xticks([])
        # legend_ax.set_yticks([])

        for i_view in range(1):
            view_ax[i_view].set_xticks([])
            view_ax[i_view].set_yticks([])
            show_img = img_ud[cw[i_view][2] : cw[i_view][3], cw[i_view][0] : cw[i_view][1], :]
            # show_img = cv2.putText(show_img, frame_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=0, thickness=3)
            view_ax[i_view].imshow(show_img)
        # lm_cw = cw[1]
        # view_ax[1].imshow(img_ud[lm_cw[2] : lm_cw[3], lm_cw[0] : lm_cw[1], :])
        # dir_cw = cw[0]
        # dir_ax.imshow(img_ud[dir_cw[2]: dir_cw[3], dir_cw[0] : dir_cw[1], :])
        # rm_cw = cw[2]
        # rm_ax.imshow(img_ud[rm_cw[2] : rm_cw[3], rm_cw[0] : rm_cw[1], :])
        # for i_view in range(3):
            for i_bpt, bpt2plot in enumerate(bpts2plot):
                cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)

                col = cmap(cur_bpt_idx / num_bptstotal)

                p3d = r3d_data[pts3d_reproj_key][i_frame, cur_bpt_idx, :]
                reproj = np.squeeze(r3d_data['calibration_data']['cgroup'].cameras[i_view].project(p3d).reshape([1, 2]))
                if scores[i_view, i_frame, i_bpt] > min_valid_score:
                    # make sure points are within the crop window
                    x_shifted = dlc_coords[i_view, i_frame, cur_bpt_idx, 0] - cw[i_view][0]
                    y_shifted = dlc_coords[i_view, i_frame, cur_bpt_idx, 1] - cw[i_view][2]
                    valid_pt = True
                    if x_shifted < 0 or x_shifted > (cw[i_view][1] - cw[i_view][0]):
                        valid_pt = False
                    if y_shifted < 0 or y_shifted > (cw[i_view][3] - cw[i_view][2]):
                        valid_pt = False
                    if valid_pt:
                        view_ax[i_view].scatter(x_shifted, y_shifted, s=markersize, color=col)
                    # vid_ax.scatter(reproj[0], reproj[1], s=markersize, color=col, marker='+')

                else:
                    pass

        for bpt2plot in bpts2plot:
            cur_bpt_idx = r3d_data['dlc_output']['bodyparts'].index(bpt2plot)

        jpg_name = os.path.join(jpg_folder, 'frame{:04d}.jpg'.format(i_frame))
        plt.savefig(jpg_name, format='jpeg')
        plt.close('all')

    cap.release()


    # turn the cropped jpegs into a new movie
    jpg_names = os.path.join(jpg_folder, 'frame%04d.jpg')
    command = (
        f"ffmpeg -i {jpg_names} "
        f"-c:v copy {animation_name}"
    )
    subprocess.call(command, shell=True)


def connect_3d_bpts(points3d, bodyparts, bpts2plot, bpts2connect, ax, lw=0.5):

    for bpts_pair in bpts2connect:

        test_bpts = [bpt in bpts2plot for bpt in bpts_pair]
        if all(test_bpts):
            endpt_x = []
            endpt_y = []
            endpt_z = []
            for bpt in bpts_pair:
                bpt_idx = bodyparts.index(bpt)
                endpt_x.append(points3d[bpt_idx, 0])
                endpt_y.append(points3d[bpt_idx, 2])    # plotting y-coord on the z-axis
                endpt_z.append(points3d[bpt_idx, 1])    # plotting z-coord on the y-axis

            ax.plot(endpt_x, endpt_y, endpt_z, color='gray', lw=lw)


def create_vids_plus_3danimation_figure(figsize=(18, 10), num_views=2, dpi=100.):
    '''

    :param figsize:
    :param dpi:
    :return:
    '''
    fig = plt.figure(figsize=figsize, dpi=dpi)

    axs = []
    for i_ax in range(num_views):
        axs.append(fig.add_subplot(1, num_views+1, i_ax))

    axs.append(fig.add_subplot(1, num_views+1, num_views+1, projection='3d'))

    for ax in axs[:num_views]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    axs[num_views].xaxis.set_ticks([])
    axs[num_views].yaxis.set_ticks([])
    axs[num_views].zaxis.set_ticks([])

    axs[num_views].set_xlabel('x')
    axs[num_views].set_ylabel('y')
    axs[num_views].set_zlabel('z')

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
    bpts2connect = rat_sr_bodyparts2connect()
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


def animate_optitrack_vids_plus3d(r3d_data, orig_videos, cropped_videos, parent_directories):
    '''

    :param r3d_data: dictionary containing the following keys:
        frame_points: undistorted points in the original video coordinate system
    :param cropped_videos:
    :return:
    '''
    reconstruct_3d_parent = parent_directories['reconstruct3d_parent']

    cv_params = [navigation_utilities.parse_cropped_optitrack_video_name(cv_name) for cv_name in cropped_videos]
    animation_name = navigation_utilities.mouse_animation_name(cv_params[0], reconstruct_3d_parent)
    _, an_name = os.path.split(animation_name)

    animation_folder, animation_name_only = os.path.split(animation_name)
    # animation_name_E = animation_name.replace('animation', 'animation_E')
    # animation_name_F = animation_name.replace('animation', 'animation_F')

    animation_name_recal = animation_name.replace('animation', 'animation_recal')
    # comment out to overwrite old videos
    # if os.path.exists(animation_name_E) and os.path.exists(animation_name_F):
    #     print('{} already exists'.format(an_name))
    #     return True   # for now, only make one animation per folder just to get a look at if reconstruction looks good

    if os.path.exists(animation_name_recal):
        print('{} already exists'.format(an_name))
        return True   # for now, only make one animation per folder just to get a look at if reconstruction looks good

    # jpg_folder_E = os.path.join(animation_folder, 'temp_E')
    # jpg_folder_F = os.path.join(animation_folder, 'temp_F')
    jpg_folder_recal = os.path.join(animation_folder, 'temp_recal')
    # if os.path.isdir(jpg_folder_E):
    #     shutil.rmtree(jpg_folder_E)
    # os.mkdir(jpg_folder_E)
    # if os.path.isdir(jpg_folder_F):
    #     shutil.rmtree(jpg_folder_F)
    if os.path.isdir(jpg_folder_recal):
        shutil.rmtree(jpg_folder_recal)
    os.mkdir(jpg_folder_recal)

    num_cams = np.shape(r3d_data['frame_points'])[1]
    show_undistorted = r3d_data['cal_data']['use_undistorted_pts_for_stereo_cal']

    cv_cam_nums = [cvp['cam_num'] for cvp in cv_params]
    im_size = r3d_data['cal_data']['im_size']
    fullframe_pts = [np.squeeze(r3d_data['frame_points'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    fullframe_pts_ud = [np.squeeze(r3d_data['frame_points_ud'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    # reprojected_pts_E = [np.squeeze(r3d_data['reprojected_points_E'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    # reprojected_pts_F = [np.squeeze(r3d_data['reprojected_points_F'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    reprojected_pts_recal = [np.squeeze(r3d_data['reprojected_points_recal'][:, i_cam, :, :]) for i_cam in range(num_cams)]
    # wpts_E = r3d_data['worldpoints_E']
    # wpts_F = r3d_data['worldpoints_F']
    wpts_recal = r3d_data['worldpoints_recal']

    bodyparts = r3d_data['bodyparts']
    num_frames = np.shape(r3d_data['frame_points'])[0]

    # create video capture objects to read frames
    vid_cap_objs = []
    # cropped_im_size = []
    crop_wins = []

    bpts2connect = mouse_sr_bodyparts2connect()
    cropped_vid_metadata = []
    isrotated = []
    im_sizes = []
    for i_cam in range(num_cams):

        # todo: pull image from original video, then undistort, then crop images and overlay undistorted and unnormalized points
        # this should find the index of camera number i_cam + 1 (1 or 2) in the cv_cam_nums list to make sure the vid_cap_objs are in the same order as r3d_data
        # vid_cap_objs.append(cv2.VideoCapture(cropped_videos[cv_cam_nums.index(i_cam + 1)]))

        vid_cap_objs.append(cv2.VideoCapture(orig_videos[cv_cam_nums.index(i_cam + 1)]))

        w = vid_cap_objs[i_cam].get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vid_cap_objs[i_cam].get(cv2.CAP_PROP_FRAME_HEIGHT)
        im_sizes.append((w, h))
        # cropped_im_size.append((w, h))   # may not need this
        cropped_vid_metadata.append(navigation_utilities.parse_cropped_optitrack_video_name(cropped_videos[i_cam]))
        crop_wins.append(cropped_vid_metadata[i_cam]['crop_window'])   # subtract one to line up with python indexing
        # crop_wins.append(np.array([0, w, 0, h], dtype=int))
        isrotated.append(cropped_vid_metadata[i_cam]['isrotated'])

    for i_frame in range(num_frames):
        print('working on {}, frame {:04d}'.format(animation_name_only, i_frame))

        # fig_E, axs_E = create_vids_plus_3danimation_figure()  # todo: add in options to size image axes depending on vid size
        # fig_F, axs_F = create_vids_plus_3danimation_figure()
        fig_recal, axs_recal = create_vids_plus_3danimation_figure()

        fullframe_pts_forthisframe = [fullframe_pts[i_cam][i_frame, :, :] for i_cam in range(num_cams)]
        fullframe_pts_ud_forthisframe = [fullframe_pts_ud[i_cam][i_frame, :, :] for i_cam in range(num_cams)]
        valid_3dpoints = identify_valid_3dpts(fullframe_pts_forthisframe, crop_wins, im_sizes, isrotated)

        # jpg_name_E = os.path.join(jpg_folder_E, 'frame{:04d}.jpg'.format(i_frame))
        # jpg_name_F = os.path.join(jpg_folder_F, 'frame{:04d}.jpg'.format(i_frame))
        jpg_name_recal = os.path.join(jpg_folder_recal, 'frame{:04d}.jpg'.format(i_frame))
        for i_cam in range(num_cams):

            # cur_fullframe_reproj_pts_E = reprojected_pts_E[i_cam][i_frame, :, :]
            # cur_fullframe_reproj_pts_F = reprojected_pts_F[i_cam][i_frame, :, :]
            cur_fullframe_reproj_pts_recal = reprojected_pts_recal[i_cam][i_frame, :, :]
            cur_fullframe_pts = fullframe_pts_ud_forthisframe[i_cam]
            crop_params = cv_params[i_cam]['crop_window']
            translated_frame_points = reconstruct_3d_optitrack.optitrack_fullframe_to_cropped_coords(cur_fullframe_pts, crop_params, im_size[i_cam], isrotated[i_cam])
            # translated_reproj_points_E = reconstruct_3d_optitrack.optitrack_fullframe_to_cropped_coords(cur_fullframe_reproj_pts_E, crop_params, im_size[i_cam], isrotated[i_cam])
            # translated_reproj_points_F = reconstruct_3d_optitrack.optitrack_fullframe_to_cropped_coords(
            #     cur_fullframe_reproj_pts_F, crop_params, im_size[i_cam], isrotated[i_cam])
            translated_reproj_points_recal = reconstruct_3d_optitrack.optitrack_fullframe_to_cropped_coords(
                cur_fullframe_reproj_pts_recal, crop_params, im_size[i_cam], isrotated[i_cam])

            vid_cap_objs[i_cam].set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            ret, img = vid_cap_objs[i_cam].read()

            crop_win = crop_wins[i_cam]
            if show_undistorted:
                mtx = r3d_data['cal_data']['mtx'][i_cam]
                dist = r3d_data['cal_data']['dist'][i_cam]
                cropped_img = undistort2cropped(img, mtx, dist, crop_win, isrotated[i_cam])
            else:
                cropped_img = img[crop_win[2]:crop_win[3], crop_win[0]:crop_win[1], :]
                if isrotated[i_cam]:
                    cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_180)

            # overlay points, check that they match with cropped vids
            cw = [0, np.shape(cropped_img)[1], 0, np.shape(cropped_img)[0]]
            # show_crop_frame_with_pts(cropped_img, cw, translated_frame_points, bodyparts, bpts2connect, valid_3dpoints, axs_E[i_cam], marker='o', s=6)
            # show_crop_frame_with_pts(cropped_img, cw, translated_frame_points, bodyparts, bpts2connect, valid_3dpoints,
            #                          axs_F[i_cam], marker='o', s=6)
            # show_crop_frame_with_pts(cropped_img, cw, translated_reproj_points_E, bodyparts, [], valid_3dpoints,
            #                          axs_E[i_cam], marker='s', s=6)
            # show_crop_frame_with_pts(cropped_img, cw, translated_reproj_points_F, bodyparts, [], valid_3dpoints,
            #                          axs_F[i_cam], marker='s', s=6)

            show_crop_frame_with_pts(cropped_img, cw, translated_frame_points, bodyparts, bpts2connect, valid_3dpoints,
                                     axs_recal[i_cam], marker='o', s=6)
            show_crop_frame_with_pts(cropped_img, cw, translated_reproj_points_recal, bodyparts, [], valid_3dpoints,
                                     axs_recal[i_cam], marker='s', s=6)

        # make the 3d plot
        # cur_wpts_E = np.squeeze(wpts_E[i_frame, :, :])
        # cur_wpts_F = np.squeeze(wpts_F[i_frame, :, :])
        cur_wpts_recal = np.squeeze(wpts_recal[i_frame, :, :])
        bpts2connect_3d = mouse_sr_bodyparts2connect_3d()
        # plot_frame3d(cur_wpts_E, valid_3dpoints, bodyparts, bpts2connect_3d, axs_E[2])
        # plot_frame3d(cur_wpts_F, valid_3dpoints, bodyparts, bpts2connect_3d, axs_F[2])
        plot_frame3d(cur_wpts_recal, valid_3dpoints, bodyparts, bpts2connect_3d, axs_recal[2])

        # fig_E.savefig(jpg_name_E)
        # fig_F.savefig(jpg_name_F)
        fig_recal.savefig(jpg_name_recal)
        plt.close('all')
        # plt.show()

    # # turn the cropped jpegs into a new movie
    # jpg_names_E = os.path.join(jpg_folder_E, 'frame%04d.jpg')
    # command = (
    #     f"ffmpeg -i {jpg_names_E} "
    #     f"-c:v copy {animation_name_E}"
    # )
    # subprocess.call(command, shell=True)
    # 
    # # delete the temp folder to hold frame jpegs
    # shutil.rmtree(jpg_folder_E)

    # turn the cropped jpegs into a new movie
    # jpg_names_F = os.path.join(jpg_folder_F, 'frame%04d.jpg')
    # command = (
    #     f"ffmpeg -i {jpg_names_F} "
    #     f"-c:v copy {animation_name_F}"
    # )
    # subprocess.call(command, shell=True)
    # 
    # # delete the temp folder to hold frame jpegs
    # shutil.rmtree(jpg_folder_F)
    
    # turn the cropped jpegs into a new movie
    jpg_names_recal = os.path.join(jpg_folder_recal, 'frame%04d.jpg')
    command = (
        f"ffmpeg -i {jpg_names_recal} "
        f"-c:v copy {animation_name_recal}"
    )
    subprocess.call(command, shell=True)

    # delete the temp folder to hold frame jpegs
    shutil.rmtree(jpg_folder_recal)

    return True


def undistort2cropped(img, mtx, dist, crop_win, isrotated):
    '''
    undistort frame from original video, then crop
    :param img:
    :param mtx:
    :param dist:
    :param crop_win: should be [left, right, top, bottom]
    :param isrotated:
    :return:
    '''

    if isrotated:
        # rotate before undistorting
        img = cv2.rotate(img, cv2.ROTATE_180)

    img_ud = cv2.undistort(img, mtx, dist)

    if isrotated:
        # rotate back before cropping
        img_ud = cv2.rotate(img_ud, cv2.ROTATE_180)

    cropped_img = img_ud[crop_win[2]:crop_win[3], crop_win[0]:crop_win[1], :]

    if isrotated:
        # rotate back before cropping
        cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_180)

    return cropped_img


def identify_valid_3dpts(framepts_forallcams, crop_wins, im_sizes, isrotated):
    '''

    :param framepts_forallcams:
    :param crop_wins: format [left, right, top, bottom]. This is BEFORE ROTATION of the image if this was an upside-down
        camera
    :param im_sizes:
    :param isrotated: list indicating whether this camera was rotated (currently should be [True, False] since camera 1
        was physically rotated and camera 2 was not
    :return:
    '''
    num_bp = np.shape(framepts_forallcams[0])[0]
    num_cams = len(crop_wins)
    valid_cam_pt = np.zeros((num_bp, 2), dtype=bool)

    for i_cam in range(num_cams):
        if isrotated[i_cam]:
            # if image is rotated, top left corner of cropped image will be bottom right corner of full image
            crop_edge = cvb.rotate_pts_180([crop_wins[i_cam][1], crop_wins[i_cam][3]], im_sizes[i_cam])
        else:
            # if image is not rotated, top left corner of cropped image will be top left corner of full image
            crop_edge = np.array([crop_wins[i_cam][0], crop_wins[i_cam][2]])
        for i_bp in range(num_bp):

            # check each camera view to see if x = y = 0, indicating that point was not correctly identified in that view
            # frame_pt_test = [all(cam_framepts[i_bp, :] - crop_wins[i_cam][:1] == 0) for i_cam, cam_framepts in enumerate(framepts_forallcams)]

            valid_cam_pt[i_bp, i_cam] = any(framepts_forallcams[i_cam][i_bp, :] - crop_edge != 0)

            # if any(frame_pt_test):
            #     continue
            # valid_3dpts[i_bp] = True

    valid_3dpts = np.logical_and(valid_cam_pt[:, 0], valid_cam_pt[:, 1])

    return valid_3dpts


def plot_frame3d(worldpoints, valid_3dpoints, bodyparts, bpts2connect, ax3d, **kwargs):
    bp_c = mouse_bp_colors_3d()
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('s', 3)

    for i_pt, pt in enumerate(worldpoints):

        if valid_3dpoints[i_pt]:

            if len(pt) > 0:
                try:
                    x, y, z = pt[0]
                except:
                    x, y, z = pt
                # x = int(round(x))
                # y = int(round(y))
                kwargs['c'] = bp_c[bodyparts[i_pt]]

                ax3d.scatter(x, y, z, **kwargs)

    connect_bodyparts_3d(worldpoints, bodyparts, bpts2connect, valid_3dpoints, ax3d)

    ax3d.set_xlabel('x')
    ax3d.set_ylabel('y')
    ax3d.set_zlabel('z')
    # ax3d.set_xlim(20, 60)
    # ax3d.set_ylim(20, 60)
    # ax3d.set_zlim(100, 150)
    ax3d.invert_yaxis()


def show_crop_frame_with_pts(img, cw, frame_pts, bodyparts, bpts2connect, valid_3dpoints, ax, **kwargs):
    '''

    :param img:
    :param cw: crop window - [left, right, top, bottom]
    :param frame_pts:
    :param bodyparts:
    :param bpts2connect:
    :param valid_3dpoints:
    :param ax:
    :param kwargs:
    :return:
    '''
    if img.ndim == 2:
        # 2-d array for grayscale image
        cropped_img = img[cw[2]:cw[3], cw[0]:cw[1]]
    elif img.ndim == 3:
        # color image
        cropped_img = img[cw[2]:cw[3], cw[0]:cw[1], :]

    ax.imshow(cropped_img)

    overlay_pts(frame_pts, bodyparts, valid_3dpoints, ax, **kwargs)

    connect_bodyparts(frame_pts, bodyparts, bpts2connect, valid_3dpoints, ax)


def overlay_pts(pts, bodyparts, plot_point_bool, ax, **kwargs):
    '''

    :param pts:
    :param bodyparts:
    :param plot_point_bool:
    :param ax:
    :param kwargs:
    :return:
    '''
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('s', 3)
    bp_c = mouse_bp_colors()

    for i_pt, pt in enumerate(pts):
        if plot_point_bool[i_pt]:
            pt = np.squeeze(pt)
            if all(pt == 0):
                continue    # [0, 0] are points that weren't properly identified
            kwargs['c'] = bp_c[bodyparts[i_pt]]
            ax.scatter(pt[0], pt[1], **kwargs)


def draw_epipolar_lines_on_img(img_pts, whichImage, F, im_size, bodyparts, plot_point_bool, ax, lwidth=0.5, linestyle='-'):

    epilines = cv2.computeCorrespondEpilines(img_pts, whichImage, F)
    bp_c = mouse_bp_colors()

    for i_line, epiline in enumerate(epilines):

        if plot_point_bool[i_line]:
            bp_color = bp_c[bodyparts[i_line]]    # color_from_bodypart(bodyparts[i_line])
            epiline = np.squeeze(epiline)
            edge_pts = cvb.find_line_edge_coordinates(epiline, im_size)

            if not np.all(edge_pts == 0):
                ax.plot(edge_pts[:, 0], edge_pts[:, 1], color=bp_color, ls=linestyle, marker='.', lw=lwidth)


def overlay_pts_on_original_frame(frame_pts, pts_conf, campickle_metadata, camdlc_metadata, frame_num, cal_data, parent_directories,
                                  ax, plot_undistorted=True, frame_pts_already_undistorted=False, min_conf=0.98, **kwargs):
    '''

    :param frame_pts:
    :param campickle_metadata: a single pickle_metadata structure
    :param camdlc_metadata:
    :param frame_num:
    :param cal_data:
    :param parent_directories:
    :param ax:
    :param kwargs:
    :return:
    '''

    cam_num = campickle_metadata['cam_num']
    video_root_folder = parent_directories['video_root_folder']

    bodyparts = camdlc_metadata['data']['DLC-model-config file']['all_joints_names']
    mouseID = campickle_metadata['mouseID']
    day_dir = mouseID + '_' + campickle_metadata['trialtime'].strftime('%Y%m%d')
    orig_vid_folder = os.path.join(video_root_folder, mouseID, day_dir)

    orig_vid_name_base = '_'.join([campickle_metadata['prefix'] + mouseID,
                             campickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                             '{:d}'.format(campickle_metadata['session_num']),
                             '{:03d}'.format(campickle_metadata['vid_num']),
                             'cam{:02d}.avi'.format(cam_num)
                             ])

    orig_vid_name = os.path.join(orig_vid_folder, orig_vid_name_base)

    if not os.path.exists(orig_vid_name):
        # sometimes session number has 2 digits, sometimes one
        orig_vid_name_base = '_'.join([campickle_metadata['prefix'] + mouseID,
                                       campickle_metadata['trialtime'].strftime('%Y%m%d_%H-%M-%S'),
                                       '{:02d}'.format(campickle_metadata['session_num']),
                                       '{:03d}'.format(campickle_metadata['vid_num']),
                                       'cam{:02d}.avi'.format(cam_num)
                                       ])
        orig_vid_name = os.path.join(orig_vid_folder, orig_vid_name_base)

    #read in image
    video_object = cv2.VideoCapture(orig_vid_name)

    video_object.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, img = video_object.read()

    mtx = cal_data['mtx'][cam_num-1]
    dist = cal_data['dist'][cam_num-1]

    if cam_num == 1:
        img = cv2.rotate(img, cv2.ROTATE_180)
    img_ud = cv2.undistort(img, mtx, dist)

    video_object.release()

    h, w, _ = np.shape(img_ud)
    im_size = (w, h)

    if plot_undistorted:
        # undistorted image
        ax.imshow(img_ud)
    else:
        # distorted original image
        ax.imshow(img)

    if not frame_pts_already_undistorted:
        pt_ud_norm = np.squeeze(cv2.undistortPoints(frame_pts, mtx, dist))
        pt_ud = cvb.unnormalize_points(pt_ud_norm, mtx)
    else:
        pt_ud = frame_pts

    num_pts = np.shape(frame_pts)[0]
    if num_pts == 1:
        # only one point
        pt_ud = [pt_ud]

    if plot_undistorted:
        to_plot = pt_ud
    else:
        to_plot = frame_pts


    plot_point_bool = pts_conf > min_conf   # np.ones((num_pts, 1), dtype=bool)
    overlay_pts(to_plot, bodyparts, plot_point_bool, ax, **kwargs)

    return im_size


def color_from_bodypart(bodypart):

    if bodypart == 'leftear':
        bp_color = (127,0,0)
    elif bodypart == 'rightear':
        bp_color = (255,0,0)
    elif bodypart == 'lefteye':
        bp_color = (150,150,150)
    elif bodypart == 'righteye':
        bp_color = (200,200,200)
    elif bodypart == 'nose':
        bp_color = (0,0,0)
    elif bodypart == 'leftpaw':
        bp_color = (0,50,0)
    elif bodypart == 'leftdigit1':
        bp_color = (0, 100, 0)
    elif bodypart == 'leftdigit2':
        bp_color = (0,150,0)
    elif bodypart == 'leftdigit3':
        bp_color = (0, 200, 0)
    elif bodypart == 'leftdigit4':
        bp_color = (0,250,0)
    elif bodypart == 'rightpaw':
        bp_color = (0,0,50)
    elif bodypart == 'rightdigit1':
        bp_color = (0, 0, 100)
    elif bodypart == 'rightdigit2':
        bp_color = (0,0,150)
    elif bodypart == 'rightdigit3':
        bp_color = (0, 0, 200)
    elif bodypart == 'rightdigit4':
        bp_color = (0,0,250)
    elif bodypart == 'pellet1':
        bp_color = (100,0,100)
    elif bodypart == 'pellet2':
        bp_color = (200,0,200)
    else:
        bp_color = (0,0,255)

    bp_color = [float(bpc)/255. for bpc in bp_color]

    return bp_color


def connect_bodyparts(frame_pts, bodyparts, bpts2connect, valid_3dpoints, ax, **kwargs):
    '''
    add lines connecting body parts to video frames showing marked bodypart points
    :param frame_pts: n x 2 numpy array where n is the number of points in the frame
    :param bodyparts: n-element list of bodypart names in order corresponding to frame_pts
    :param bpts2connect: list of 2-element lists containing pairs of body parts to connect with lines (named according to bodyparts)
    :param ax: axes on which to make the plot
    :param linecolor: color of connecting lines, default gray
    :param lwidth: width of connecting lines - default 1.5 (pyplot default)
    :return:
    '''
    kwargs.setdefault('c', (0.5, 0.5, 0.5))
    kwargs.setdefault('lw', 1.5)
    for pt2connect in bpts2connect:

        pt_index = [bodyparts.index(bp_name) for bp_name in pt2connect]

        if all(valid_3dpoints[pt_index]):

            if all(frame_pts[pt_index[0], :] == 0) or all(frame_pts[pt_index[1], :] == 0):
                continue   # one of the points wasn't found
            x = frame_pts[pt_index, 0]
            y = frame_pts[pt_index, 1]
            ax.plot(x, y, **kwargs)


def connect_bodyparts_3d(worldpoints, bodyparts, bpts2connect, valid_3dpoints, ax3d, **kwargs):
    '''
    add lines connecting body parts to video frames showing marked bodypart points
    :param frame_pts: n x 2 numpy array where n is the number of points in the frame
    :param bodyparts: n-element list of bodypart names in order corresponding to frame_pts
    :param bpts2connect: list of 2-element lists containing pairs of body parts to connect with lines (named according to bodyparts)
    :param ax: axes on which to make the plot
    :param linecolor: color of connecting lines, default gray
    :param lwidth: width of connecting lines - default 1.5 (pyplot default)
    :return:
    '''
    kwargs.setdefault('c', (0.5, 0.5, 0.5))
    kwargs.setdefault('lw', 1.5)
    for pt2connect in bpts2connect:

        pt_index = [bodyparts.index(bp_name) for bp_name in pt2connect]

        if all(valid_3dpoints[pt_index]):

            x = worldpoints[pt_index, 0]
            y = worldpoints[pt_index, 1]
            z = worldpoints[pt_index, 2]
            ax3d.plot(x, y, z, **kwargs)


def rat_sr_bodyparts2connect():

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

    bpts2connect.append(['leftear', 'lefteye'])
    bpts2connect.append(['rightear', 'righteye'])
    bpts2connect.append(['nose', 'righteye'])
    bpts2connect.append(['nose', 'lefteye'])

    return bpts2connect


def mouse_sr_bodyparts2connect():

    bpts2connect = []

    bpts2connect.append(['leftpaw', 'leftdigit1'])
    bpts2connect.append(['leftpaw', 'leftdigit2'])
    bpts2connect.append(['leftpaw', 'leftdigit3'])
    bpts2connect.append(['leftpaw', 'leftdigit4'])

    bpts2connect.append(['rightpaw', 'rightdigit1'])
    bpts2connect.append(['rightpaw', 'rightdigit2'])
    bpts2connect.append(['rightpaw', 'rightdigit3'])
    bpts2connect.append(['rightpaw', 'rightdigit4'])

    return bpts2connect


def mouse_sr_bodyparts2connect_3d():

    bpts2connect = []

    bpts2connect.append(['leftpaw', 'leftdigit1'])
    bpts2connect.append(['leftpaw', 'leftdigit2'])
    bpts2connect.append(['leftpaw', 'leftdigit3'])
    bpts2connect.append(['leftpaw', 'leftdigit4'])

    bpts2connect.append(['rightpaw', 'rightdigit1'])
    bpts2connect.append(['rightpaw', 'rightdigit2'])
    bpts2connect.append(['rightpaw', 'rightdigit3'])
    bpts2connect.append(['rightpaw', 'rightdigit4'])

    bpts2connect.append(['leftear', 'lefteye'])
    bpts2connect.append(['rightear', 'righteye'])
    bpts2connect.append(['lefteye', 'nose'])
    bpts2connect.append(['righteye', 'nose'])

    return bpts2connect


def rat_bp_colors():

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


def mouse_bp_colors():

    bp_c = {'leftear':(0, 1, 1)}
    bp_c['rightear'] = tuple(np.array(bp_c['leftear']) * 0.5)

    bp_c['lefteye'] = (1, 0, 1)
    bp_c['righteye'] = tuple(np.array(bp_c['lefteye']) * 0.5)

    bp_c['nose'] = (0, 0, 0)

    bp_c['rightpaw'] = (0, 0, 1)
    bp_c['rightdigit1'] = tuple(np.array(bp_c['rightpaw']) * 0.9)
    bp_c['rightdigit2'] = tuple(np.array(bp_c['rightpaw']) * 0.8)
    bp_c['rightdigit3'] = tuple(np.array(bp_c['rightpaw']) * 0.7)
    bp_c['rightdigit4'] = tuple(np.array(bp_c['rightpaw']) * 0.6)

    bp_c['leftpaw'] = (1, 0, 0)
    bp_c['leftdigit1'] = tuple(np.array(bp_c['leftpaw']) * 0.9)
    bp_c['leftdigit2'] = tuple(np.array(bp_c['leftpaw']) * 0.8)
    bp_c['leftdigit3'] = tuple(np.array(bp_c['leftpaw']) * 0.7)
    bp_c['leftdigit4'] = tuple(np.array(bp_c['leftpaw']) * 0.6)

    bp_c['pellet1'] = (0, 0, 0)
    bp_c['pellet2'] = (0.1, 0.1, 0.1)

    return bp_c


def mouse_bp_colors_3d():

    bp_c = {'leftear':(0, 1, 1)}
    bp_c['rightear'] = tuple(np.array(bp_c['leftear']) * 0.5)

    bp_c['lefteye'] = (1, 0, 1)
    bp_c['righteye'] = tuple(np.array(bp_c['lefteye']) * 0.5)

    bp_c['nose'] = (0, 0, 0)

    bp_c['rightpaw'] = (0, 0, 1)
    bp_c['rightdigit1'] = tuple(np.array(bp_c['rightpaw']) * 0.9)
    bp_c['rightdigit2'] = tuple(np.array(bp_c['rightpaw']) * 0.8)
    bp_c['rightdigit3'] = tuple(np.array(bp_c['rightpaw']) * 0.7)
    bp_c['rightdigit4'] = tuple(np.array(bp_c['rightpaw']) * 0.6)

    bp_c['leftpaw'] = (1, 0, 0)
    bp_c['leftdigit1'] = tuple(np.array(bp_c['leftpaw']) * 0.9)
    bp_c['leftdigit2'] = tuple(np.array(bp_c['leftpaw']) * 0.8)
    bp_c['leftdigit3'] = tuple(np.array(bp_c['leftpaw']) * 0.7)
    bp_c['leftdigit4'] = tuple(np.array(bp_c['leftpaw']) * 0.6)

    bp_c['pellet1'] = (0, 0, 0)
    bp_c['pellet2'] = (0.1, 0.1, 0.1)

    return bp_c