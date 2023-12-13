import skilled_reaching_io
import navigation_utilities
import os


def analyze_trajectory(trajectory_fname):
    # trajectory_fname = navigation_utilities.create_trajectory_name(h5_metadata, session_metadata, calibration_data,
    #                                                                parent_directories)

    if os.path.exists(trajectory_fname):
        r3d_data = skilled_reaching_io.read_pickle(trajectory_fname)

    bodyparts = r3d_data['dlc_output']['bodyparts']

    f_contact, bp_contact = identify_pellet_contact(r3d_data)
    pass


def identify_pellet_contact(r3d_data):
    # r3d_data = {'points3d': p3ds,
    #             'optim_points3d': optim_p3ds,
    #             'calibration_data': calibration_data,
    #             'h5_group': h5_group,
    #             'anipose_config': anipose_config,
    #             'dlc_output': d}
    pass


def identify_pellet_drop(r3d_data):
    pass


def identify_slot_breach(r3d_data):
    # r3d_data = {'points3d': p3ds,
    #             'optim_points3d': optim_p3ds,
    #             'calibration_data': calibration_data,
    #             'h5_group': h5_group,
    #             'anipose_config': anipose_config,
    #             'dlc_output': d}
    pass

# r3d_data = {'points3d': p3ds,
#             'optim_points3d': optim_p3ds,
#             'calibration_data': calibration_data,
#             'h5_group': h5_group,
#             'anipose_config': anipose_config,
#             'dlc_output': d}
