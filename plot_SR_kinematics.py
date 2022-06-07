import matplotlib.pyplot as plt
import skilled_reaching_io
import os


def animate_3d_reconstruction(paw_trajectory_fname, video_root_folder):
    # traj_data = package_trajectory_data_for_pickle(paw_trajectory, is_estimate, invalid_points, paw_pref,
    #                                                reproj_error, high_p_invalid, low_p_valid, cal_data)
    # skilled_reaching_io.write_pickle(full_traj_fname, traj_data)

    pt_data = skilled_reaching_io.read_pickle(paw_trajectory_fname)
    pass


if __name__ == '__main__':
    videos_parent = '/home/levlab/Public/rat_SR_videos_to_analyze'   # on the lambda machine
    # videos_parent = '/Users/dan/Documents/deeplabcut/videos_to_analyze'  # on home mac
    # videos_parent = '/Volumes/Untitled/videos_to_analyze'
    video_root_folder = os.path.join(videos_parent, 'videos_to_crop')

    paw_trajectory_fname = '/home/levlab/Public/rat_SR_videos_to_analyze/trajectory_files/R0229/R0229_20181126a/R0229_box99_20181126_16-41-13_002_3dtrajectory'

    animate_3d_reconstruction('/home/levlab/Public/rat_SR_videos_to_analyze/trajectory_files/R0229/R0229_20181126a/R0229_box99_20181126_16-41-13_002_3dtrajectory')

