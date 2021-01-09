from crop_videos import preprocess_videos
import navigation_utilities
import deeplabcut

if __name__ == '__main__':

    # step 1: preprocess videos to extract left mirror, right mirror, and direct views

    view_list = ('direct', 'leftmirror', 'rightmirror')
    # parameters for cropping
    crop_params_dict = {
        view_list[0]: [700, 1350, 270, 935],
        view_list[1]: [1, 470, 270, 920],
        view_list[2]: [1570, 2040, 270, 920]
    }

    video_root_folder = '/home/levlab/Public/DLC_DKL/videos_to_analyze/videos_to_crop'
    # vid_folder_list = ['/Users/dan/Documents/deeplabcut/R0382_20200909c','/Users/dan/Documents/deeplabcut/R0230_20181114a']
    video_folder_list = navigation_utilities.get_video_folders_to_crop(video_root_folder)

    cropped_vids_parent = '/home/levlab/Public/DLC_DKL/videos_to_analyze'

    cropped_video_directories = preprocess_videos(video_folder_list, cropped_vids_parent, crop_params_dict, view_list, vidtype='avi')

    # step 2: run the vids through DLC
    # parameters for running DLC
    # need to update these paths when moved to the lambda machine
    config_path{'direct': '/home/levlab/Public/DLC_DKL/skilled_reaching_direct-Dan_Leventhal-2020-10-19/config.yaml',
                'mirror': '/home/levlab/Public/DLC_DKL/skilled_reaching_mirror-Dan_Leventhal-2020-10-19/config.yaml'}

    folders_to_analyze = navigation_utilities.find_folders_to_analyze(cropped_vids_parent, view_list=view_list)
    #todo: add a key for ratID and session name to go with the folder names in cropped_vid_dirs

    for view in view_list:

        current_view_folders = folders_to_analyze[view]
        for current_view in current_view_folders:
    for rat_session in cropped_vid_dirs:
        direct_view_dir = rat_session['direct']
        deeplabcut.analyze_videos(direct_view_config, [''])



def analyze_cropped_videos(current_view_folders, config_files)