import skilled_reaching_io

r3d_data = {'points3d': p3ds,
            'optim_points3d': optim_p3ds,
            'calibration_data': calibration_data,
            'h5_group': h5_group,
            'anipose_config': anipose_config,
            'dlc_output': d}
skilled_reaching_io.write_pickle(trajectory_fname, r3d_data)