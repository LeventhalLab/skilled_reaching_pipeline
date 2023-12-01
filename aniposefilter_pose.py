from tqdm import tqdm, trange
import os.path, os
import numpy as np
import pandas as pd
from numpy import array as arr
from glob import glob
from scipy import signal, stats
from scipy.interpolate import splev, splrep
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from collections import Counter
from multiprocessing import cpu_count
from multiprocessing import Pool, get_context
import pickle

def filter_pose_medfilt(config, all_points, bodyparts):
    # adapted directly from anipose github file "filter_pose.py"
    n_frames, n_joints, n_possible, _ = all_points.shape

    points_full = all_points[:, :, :, :2]
    scores_full = all_points[:, :, :, 2]

    points = np.full((n_frames, n_joints, 2), np.nan, dtype='float64')
    scores = np.empty((n_frames, n_joints), dtype='float64')

    for bp_ix, bp in enumerate(bodyparts):
        x = points_full[:, bp_ix, 0, 0]
        y = points_full[:, bp_ix, 0, 1]
        score = scores_full[:, bp_ix, 0]

        xmed = signal.medfilt(x, kernel_size=config['filter']['medfilt'])
        ymed = signal.medfilt(y, kernel_size=config['filter']['medfilt'])

        errx = np.abs(x - xmed)
        erry = np.abs(y - ymed)
        err = errx + erry

        bad = np.zeros(len(x), dtype='bool')
        bad[err >= config['filter']['offset_threshold']] = True
        bad[score < config['filter']['score_threshold']] = True

        Xf = arr([x,y]).T
        Xf[bad] = np.nan

        Xfi = np.copy(Xf)

        for i in range(Xf.shape[1]):
            vals = Xfi[:, i]
            nans, ix = nan_helper(vals)
            # some data missing, but not too much
            if np.sum(nans) > 0 and np.mean(~nans) > 0.5 and np.sum(~nans) > 5:
                if config['filter']['spline']:
                    spline = splrep(ix(~nans), vals[~nans], k=3, s=0)
                    vals[nans] = splev(ix(nans), spline)
                else:
                    vals[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
            Xfi[:, i] = vals

        points[:, bp_ix, 0] = Xfi[:, 0]
        points[:, bp_ix, 1] = Xfi[:, 1]
        # dout[scorer, bp, 'interpolated'] = np.isnan(Xf[:, 0])

    scores = scores_full[:, :, 0]

    return points, scores


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]