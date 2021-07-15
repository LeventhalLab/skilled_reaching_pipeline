import glob
import skilled_reaching_calibration





if __name__ == '__main__':
    cal_vids = ['/Users/dan/Documents/Burgess_sr_videos/calibation_videos/calibration_vids_2021/calibration_vids_202107/calibration_cam01_20210701_15-30-33.avi']
    cal_vids.append('/Users/dan/Documents/Burgess_sr_videos/calibation_videos/calibration_vids_2021/calibration_vids_202107/calibration_cam02_20210701_15-30-33.avi')

    cal_parent = '/Users/dan/Documents/Burgess_sr_videos/calibation_videos'

    cb_size = (10,7)

    skilled_reaching_calibration.multi_camera_calibration(cal_vids, cal_parent, cb_size=cb_size)