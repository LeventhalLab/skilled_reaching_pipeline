import glob
import os
import crop_Burgess_videos
import skilled_reaching_calibration
import navigation_utilities
import skilled_reaching_io
from datetime import datetime





if __name__ == '__main__':

    cam_list = (1, 2)

    Burgess_DLC_config_path = '/home/levlab/Public/mouse_headfixed_skilledreaching-DanL-2021-11-05/config.yaml'

    videos_parent = '/home/levlab/Public/mouse_SR_videos_to_analyze'
    video_root_folder = os.path.join(videos_parent, 'mouse_SR_videos_tocrop')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_mouse_SR_videos')
    marked_videos_parent = os.path.join(videos_parent, 'marked_videos')

    cal_vid_parent = os.path.join(videos_parent, 'mouse_SR_calibration_videos')
    cal_data_parent = os.path.join(videos_parent, 'mouse_SR_calibration_data')

    crop_params_csv_path = os.path.join(video_root_folder, 'optitrack_SR_video_crop_regions.csv')

    cb_size = (7, 10)

    # step 1 - run all the calibrations
    # skilled_reaching_calibration.calibrate_all_Burgess_vids(cal_vid_parent, cal_data_parent, cb_size=cb_size)

    # step 2 - crop all videos of mice reaching
    vid_folder_list = navigation_utilities.get_Burgess_video_folders_to_crop(video_root_folder)
    crop_params_df = skilled_reaching_io.read_crop_params_csv(crop_params_csv_path)
    crop_Burgess_videos.preprocess_Burgess_videos(vid_folder_list, cropped_videos_parent, crop_params_df, cam_list, vidtype='avi')

    skilled_reaching_calibration.multi_camera_calibration(cal_vids, cal_data_parent, cb_size=cb_size)