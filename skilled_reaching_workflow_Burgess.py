import glob
import skilled_reaching_calibration
import navigation_utilities
from datetime import datetime





if __name__ == '__main__':

    Burgess_DLC_config_path = '/home/levlab/Public/mouse_headfixed_skilledreaching-DanL-2021-11-05/config.yaml'

    videos_parent = '/home/levlab/Public/mouse_SR_videos_to_analyze'
    video_root_folder = os.path.join(videos_parent, 'videos_to_crop')
    cropped_videos_parent = os.path.join(videos_parent, 'cropped_videos')
    marked_videos_parent = os.path.join(videos_parent, 'marked_videos')

    cal_vid_parent = os.path.join(videos_parent, 'mouse_SR_calibration_videos')
    cal_data_parent = os.path.join(videos_parent, 'mouse_SR_calibration_data')

    cb_size = (7, 10)

    # step 1 - run all the calibrations
    skilled_reaching_calibration.calibrate_all_Burgess_vids(cal_vid_parent, cal_data_parent, cb_size=cb_size)

    # step 2 - crop all videos of mice reaching

    skilled_reaching_calibration.multi_camera_calibration(cal_vids, cal_data_parent, cb_size=cb_size)