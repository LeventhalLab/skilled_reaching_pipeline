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
    # where in here does it make use of alternative possible points? does it at all?
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
            # this may need to be adjusted since often many points are missing at the beginning and end of a reaching video
            if np.sum(nans) > 0 and np.mean(~nans) > 0.5 and np.sum(~nans) > 5:
                if config['filter']['spline']:
                    spline = splrep(ix(~nans), vals[~nans], k=3, s=0)
                    vals[nans] = splev(ix(nans), spline)
                else:
                    vals[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
            else:
                pass
            Xfi[:, i] = vals

        points[:, bp_ix, 0] = Xfi[:, 0]
        points[:, bp_ix, 1] = Xfi[:, 1]
        # dout[scorer, bp, 'interpolated'] = np.isnan(Xf[:, 0])

    scores = scores_full[:, :, 0]

    return points, scores


def filter_pose_viterbi(config, all_points, bodyparts):
    # adapted directly from anipose github file "filter_pose.py"
    # where in here does it make use of alternative possible points? does it at all?
    n_frames, n_joints, n_possible, _ = all_points.shape

    points_full = all_points[:, :, :, :2]
    scores_full = all_points[:, :, :, 2]

    points = np.full((n_frames, n_joints, 2), np.nan, dtype='float64')
    scores = np.empty((n_frames, n_joints), dtype='float64')

    if config['filter']['multiprocessing']:
        n_proc_default = max(min(cpu_count() // 2, n_joints), 1)
        n_proc = config['filter'].get('n_proc', n_proc_default)
    else:
        n_proc = 1
    ctx = get_context('spawn')
    pool = ctx.Pool(n_proc)

    max_offset = config['filter']['n_back']
    thres_dist = config['filter']['offset_threshold']

    iterable = [ (jix, points_full[:, jix, :], scores_full[:, jix],
                  max_offset, thres_dist)
                 for jix in range(n_joints) ]

    results = pool.imap_unordered(viterbi_path_wrapper, iterable)

    for jix, pts_new, scs_new in tqdm(results, ncols=70):
        points[:, jix] = pts_new
        scores[:, jix] = scs_new

    pool.close()
    pool.join()

    return points, scores

def viterbi_path_wrapper(args):
    jix, pts, scs, max_offset, thres_dist = args
    pts_new, scs_new = viterbi_path(pts, scs, max_offset, thres_dist)
    return jix, pts_new, scs_new


def viterbi_path(points, scores, n_back=3, thres_dist=30):
    n_frames = points.shape[0]

    points_nans = remove_dups(points, thres=5)
    # points_nans[scores < 0.01] = np.nan

    num_points = np.sum(~np.isnan(points_nans[:, :, 0]), axis=1)
    num_max = np.max(num_points)

    particles = np.zeros((n_frames, num_max * n_back + 1, 3), dtype='float64')
    valid = np.zeros(n_frames, dtype='int64')
    for i in range(n_frames):
        s = 0
        for j in range(n_back):
            if i-j < 0:
                break
            ixs = np.where(~np.isnan(points_nans[i-j, :, 0]))[0]
            n_valid = len(ixs)
            particles[i, s:s+n_valid, :2] = points[i-j, ixs]
            particles[i, s:s+n_valid, 2] = scores[i-j, ixs] * np.power(2.0, -j)
            s += n_valid
        if s == 0:
            particles[i, 0] = [-1, -1, 0.001] # missing point
            s = 1
        valid[i] = s

    ## viterbi algorithm
    n_particles = np.max(valid)

    T_logprob = np.zeros((n_frames, n_particles), dtype='float64')
    T_logprob[:] = -np.inf
    T_back = np.zeros((n_frames, n_particles), dtype='int64')

    T_logprob[0, :valid[0]] = np.log(particles[0, :valid[0], 2])
    T_back[0, :] = -1

    for i in range(1, n_frames):
        va, vb = valid[i-1], valid[i]
        pa = particles[i-1, :va, :2]
        pb = particles[i, :vb, :2]

        dists = cdist(pa, pb)
        cdf_high = stats.norm.logcdf(dists + 2, scale=thres_dist)
        cdf_low = stats.norm.logcdf(dists - 2, scale=thres_dist)
        cdfs = np.array([cdf_high, cdf_low])
        P_trans = logsumexp(cdfs.T, b=[1,-1], axis=2)

        P_trans[P_trans < -100] = -100

        # take care of missing transitions
        P_trans[pb[:, 0] == -1, :] = np.log(0.001)
        P_trans[:, pa[:, 0] == -1] = np.log(0.001)

        pflat = particles[i, :vb, 2]
        possible = T_logprob[i-1, :va] + P_trans

        T_logprob[i, :vb] = np.max(possible, axis=1) + np.log(pflat)
        T_back[i, :vb] = np.argmax(possible, axis=1)

    out = np.zeros(n_frames, dtype='int')
    out[-1] = np.argmax(T_logprob[-1])

    for i in range(n_frames-1, 0, -1):
        out[i-1] = T_back[i, out[i]]

    trace = [particles[i, out[i]] for i in range(n_frames)]
    trace = np.array(trace)

    points_new = trace[:, :2]
    scores_new = trace[:, 2]
    # scores_new[out >= num_points] = 0

    return points_new, scores_new


def remove_dups(pts, thres=7):
    tindex = np.repeat(np.arange(pts.shape[0])[:, None], pts.shape[1], axis=1)*100
    pts_ix = np.dstack([pts, tindex])
    tree = cKDTree(pts_ix.reshape(-1, 3))

    shape = (pts.shape[0], pts.shape[1])
    pairs = tree.query_pairs(thres)
    indices = [b for a, b in pairs]

    if len(pairs) == 0:
        return pts

    i0, i1 = np.unravel_index(indices, shape)
    pts_out = np.copy(pts)
    pts_out[i0, i1] = np.nan

    return pts_out


def filter_pose_autoencoder_scores(config, all_points, bodyparts):
    n_frames, n_joints, n_possible, _ = all_points.shape

    points_full = all_points[:, :, :, :2]
    scores_full = all_points[:, :, :, 2]

    scores_test = all_points[:, :, 0, 2]

    fname_model = config['filter']['autoencoder_path']
    with open(fname_model, 'rb') as f:
        mlp = pickle.load(f)

    scores_pred = mlp.predict_proba(scores_test)
    scores_pred_rep = np.repeat(scores_pred, n_possible, axis=1).reshape(scores_full.shape)

    scores_fixed = np.min([scores_pred_rep, scores_full], axis=0)

    return points_full, scores_fixed


def filter_pose_autoencoder_scores_DL(config, all_points, bodyparts, dlc_proj_name):
    n_frames, n_joints, n_possible, _ = all_points.shape

    ae_fname = '_'.join((dlc_proj_name, 'autoencoder.pickle'))
    ae_fname = os.path.join(config['model_folder'], ae_fname)

    points_full = all_points[:, :, :, :2]
    scores_full = all_points[:, :, :, 2]

    scores_test = all_points[:, :, 0, 2]
    scores_test = np.nan_to_num(scores_test, copy=False, nan=0.0)

    # fname_model = config['filter']['autoencoder_path']
    with open(ae_fname, 'rb') as f:
        mlp = pickle.load(f)

    scores_pred = mlp.predict_proba(scores_test)
    scores_pred_rep = np.repeat(scores_pred, n_possible, axis=1).reshape(scores_full.shape)

    scores_fixed = np.min([scores_pred_rep, scores_full], axis=0)

    return points_full, scores_fixed


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def wrap_points(points, scores):
    if len(points.shape) == 3: # n_possible = 1
        points = points[:, :, None]
        scores = scores[:, :, None]

    n_frames, n_joints, n_possible, _ = points.shape

    all_points = np.full((n_frames, n_joints, n_possible, 3), np.nan, dtype='float64')
    all_points[:, :, :, :2] = points
    all_points[:, :, :, 2] = scores

    return all_points
