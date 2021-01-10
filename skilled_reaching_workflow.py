from crop_videos import preprocess_videos
import navigation_utilities
import glob
import deeplabcut


def analyze_cropped_videos(folders_to_analyze, view_config_paths, cropped_vid_type='*.avi', gputouse=0):
    '''

    :param folders_to_analyze:
    :param view_config_paths:
    :param cropped_vid_type:
    :param gputouse:
    :return:
    '''

    view_list = folders_to_analyze.keys()
    for view in view_list:
        if 'direct' in view:
            config_path = view_config_paths['direct']
        elif 'mirror' in view:
            config_path = view_config_paths['mirror']
        else:
            print(view + ' does not contain the keyword "direct" or "mirror"')
            continue
        current_view_folders = folders_to_analyze[view]

        for current_folder in current_view_folders:
            cropped_video_list = glob.glob(current_folder + '/*' + cropped_vid_type)
            deeplabcut.analyze_videos(config_path,
                                      cropped_video_list,
                                      videotype=cropped_vid_type,
                                      gputouse=gputouse)


if __name__ == '__main__':

    gputouse = 2
    # step 1: preprocess videos to extract left mirror, right mirror, and direct views

    view_list = ('direct', 'leftmirror', 'rightmirror')
    # parameters for cropping
    crop_params_dict = {
        view_list[0]: [700, 1350, 270, 935],
        view_list[1]: [1, 470, 270, 920],
        view_list[2]: [1570, 2040, 270, 920]
    }
    cropped_vid_type = '.avi'

    video_root_folder = '/home/levlab/Public/DLC_DKL/videos_to_analyze/videos_to_crop'
    # vid_folder_list = ['/Users/dan/Documents/deeplabcut/R0382_20200909c','/Users/dan/Documents/deeplabcut/R0230_20181114a']
    video_folder_list = navigation_utilities.get_video_folders_to_crop(video_root_folder)

    cropped_vids_parent = '/home/levlab/Public/DLC_DKL/videos_to_analyze'

    cropped_video_directories = preprocess_videos(video_folder_list, cropped_vids_parent, crop_params_dict, view_list, vidtype='avi')

    # step 2: run the vids through DLC
    # parameters for running DLC
    # need to update these paths when moved to the lambda machine
    view_config_paths = {
        'direct': '/home/levlab/Public/DLC_DKL/skilled_reaching_direct-Dan_Leventhal-2020-10-19/config.yaml',
        'mirror': '/home/levlab/Public/DLC_DKL/skilled_reaching_mirror-Dan_Leventhal-2020-10-19/config.yaml'
    }

    # in case there are some previously cropped videos that need to be analyzed
    folders_to_analyze = navigation_utilities.find_folders_to_analyze(cropped_vids_parent, view_list=view_list)

    analyze_cropped_videos(folders_to_analyze, view_config_paths, cropped_vid_type=cropped_vid_type, gputouse=gputouse)

    # step 3: make sure calibration has been run for these sessions