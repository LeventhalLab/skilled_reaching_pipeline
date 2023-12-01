import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import whiten
from collections import defaultdict, Counter
import queue
from datetime import datetime
import navigation_utilities
import skilled_reaching_io
import pandas as pd

def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    return out

def get_rtvec(M):
    rvec = cv2.Rodrigues(M[:3, :3])[0].flatten()
    tvec = M[:3, 3].flatten()
    return rvec, tvec

def get_most_common(vals):
    Z = linkage(whiten(vals), 'ward')
    n_clust = max(len(vals)/10, 3)
    clusts = fcluster(Z, t=n_clust, criterion='maxclust')
    cc = Counter(clusts[clusts >= 0])
    most = cc.most_common(n=1)
    top = most[0][0]
    good = clusts == top
    return good

def select_matrices(Ms):
    Ms = np.array(Ms)
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in Ms]
    tvecs = np.array([M[:3, 3] for M in Ms])
    best = get_most_common(np.hstack([rvecs, tvecs]))
    Ms_best = Ms[best]
    return Ms_best


def mean_transform(M_list):
    rvecs = [cv2.Rodrigues(M[:3,:3])[0][:, 0] for M in M_list]
    tvecs = [M[:3, 3] for M in M_list]

    rvec = np.mean(rvecs, axis=0)
    tvec = np.mean(tvecs, axis=0)

    return make_M(rvec, tvec)

def mean_transform_robust(M_list, approx=None, error=0.3):
    if approx is None:
        M_list_robust = M_list
    else:
        M_list_robust = []
        for M in M_list:
            rot_error = (M - approx)[:3,:3]
            m = np.max(np.abs(rot_error))
            if m < error:
                M_list_robust.append(M)
    return mean_transform(M_list_robust)


def get_transform(rtvecs, left, right):
    L = []
    for dix in range(rtvecs.shape[1]):
        d = rtvecs[:, dix]
        good = ~np.isnan(d[:, 0])

        if good[left] and good[right]:
            M_left = make_M(d[left, 0:3], d[left, 3:6])
            M_right = make_M(d[right, 0:3], d[right, 3:6])
            M = np.matmul(M_left, np.linalg.inv(M_right))
            L.append(M)
    L_best = select_matrices(L)
    M_mean = mean_transform(L_best)
    # M_mean = mean_transform_robust(L, M_mean, error=0.5)
    # M_mean = mean_transform_robust(L, M_mean, error=0.2)
    M_mean = mean_transform_robust(L, M_mean, error=0.1)
    return M_mean


def get_connections(xs, cam_names=None, both=True):
    n_cams = xs.shape[0]
    n_points = xs.shape[1]

    if cam_names is None:
        cam_names = np.arange(n_cams)

    connections = defaultdict(int)

    for rnum in range(n_points):
        ixs = np.where(~np.isnan(xs[:, rnum, 0]))[0]
        keys = [cam_names[ix] for ix in ixs]
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a = keys[i]
                b = keys[j]
                connections[(a,b)] += 1
                if both:
                    connections[(b,a)] += 1

    return connections


def get_calibration_graph(rtvecs, cam_names=None):
    n_cams = rtvecs.shape[0]
    n_points = rtvecs.shape[1]

    if cam_names is None:
        cam_names = np.arange(n_cams)

    connections = get_connections(rtvecs, np.arange(n_cams))

    components = dict(zip(np.arange(n_cams), range(n_cams)))
    edges = set(connections.items())

    graph = defaultdict(list)

    for edgenum in range(n_cams-1):
        if len(edges) == 0:
            component_names = dict()
            for k,v in list(components.items()):
                component_names[cam_names[k]] = v
            raise ValueError("""
Could not build calibration graph.
Some group of cameras could not be paired by simultaneous calibration board detections.
Check which cameras have different group numbers below to see the missing edges.
{}""".format(component_names))

        (a, b), weight = max(edges, key=lambda x: x[1])
        graph[a].append(b)
        graph[b].append(a)

        match = components[a]
        replace = components[b]
        for k, v in components.items():
            if match == v:
                components[k] = replace

        for e in edges.copy():
            (a,b), w = e
            if components[a] == components[b]:
                edges.remove(e)

    return graph

def find_calibration_pairs(graph, source=None):
    pairs = []
    explored = set()

    if source is None:
        source = sorted(graph.keys())[0]

    q = queue.deque()
    q.append(source)

    while len(q) > 0:
        item = q.pop()
        explored.add(item)

        for new in graph[item]:
            if new not in explored:
                q.append(new)
                pairs.append( (item, new) )
    return pairs

def compute_camera_matrices(rtvecs, pairs):
    extrinsics = dict()
    source = pairs[0][0]
    extrinsics[source] = np.identity(4)
    for (a,b) in pairs:
        ext = get_transform(rtvecs, b, a)
        extrinsics[b] = np.matmul(ext, extrinsics[a])
    return extrinsics

def get_initial_extrinsics(rtvecs, cam_names=None):
    graph = get_calibration_graph(rtvecs, cam_names)
    pairs = find_calibration_pairs(graph, source=0)
    extrinsics = compute_camera_matrices(rtvecs, pairs)

    n_cams = rtvecs.shape[0]
    rvecs = []
    tvecs = []
    for cnum in range(n_cams):
        rvec, tvec = get_rtvec(extrinsics[cnum])
        rvecs.append(rvec)
        tvecs.append(tvec)
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    return rvecs, tvecs


def fullpickle2h5(fpickle_name, h5_out_name, num_outputs):

    pickle_metadata = navigation_utilities.parse_dlc_output_pickle_name(fpickle_name)
    pickled_data = skilled_reaching_io.read_pickle(fpickle_name)

    all_joints_names = pickled_data['metadata']['all_joints_names']
    nframes = pickled_data['metadata']['nframes']

    xyz_labs_orig = ["x", "y", "likelihood"]
    suffix = [str(s + 1) for s in range(num_outputs)]
    suffix[0] = ""  # first one has empty suffix for backwards compatibility
    xyz_labs = [x + s for s in suffix for x in xyz_labs_orig]

    pdindex = pd.MultiIndex.from_product(
        [[pickle_metadata['scorername']], all_joints_names, xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )

    # now need to take data from full.pickle and rearrange into a numpy array that can be converted to a dataframe
    flattened_data = flatten_pickled_data(pickled_data, num_outputs)
    DataMachine = pd.DataFrame(flattened_data, columns=pdindex, index=range(nframes))

    # h5_name = fpickle_name.split("_full.pickle")[0] + ".h5"

    DataMachine.to_hdf(h5_out_name, "df_with_missing", format="table", mode="w")

    return DataMachine


def flatten_pickled_data(pickled_data, num_outputs):

    nframes = pickled_data['metadata']['nframes']
    nbodyparts = len(pickled_data['metadata']['all_joints_names'])
    ncolumns = nbodyparts * 3 * num_outputs
    flattened_data = np.empty((nframes, ncolumns))
    flattened_data[:] = np.nan

    frame_names = list(pickled_data)
    frames = [name for name in frame_names if name[:5] == 'frame']

    for i_frame, frame in enumerate(frames):
        frame_data = pickled_data[frame]
        frame_coords = frame_data['coordinates'][0]
        frame_conf = frame_data['confidence']
        for i_bp in range(nbodyparts):
            bp_coords = frame_coords[i_bp]
            bp_conf = frame_conf[i_bp]
            for i_out in range(num_outputs):
                start_col_idx = (i_bp * 3 * num_outputs) + (i_out * 3)
                try:
                    flattened_data[i_frame, start_col_idx:start_col_idx+2] = bp_coords[i_out]
                    flattened_data[i_frame, start_col_idx + 2] = bp_conf[i_out]
                except:
                    # if there aren't num_outputs possible values for this bodypart in this frame, just skip
                    pass

    return flattened_data


## convenience function to load a set of DeepLabCut pose-2d files
def load_pose2d_fnames(fname_dict, offsets_dict=None, cam_names=None):
    if cam_names is None:
        cam_names = sorted(fname_dict.keys())
    pose_names = [fname_dict[cname] for cname in cam_names]

    if offsets_dict is None:
        offsets_dict = dict([(cname, (0, 0)) for cname in cam_names])

    datas = []
    for ix_cam, (cam_name, pose_name) in \
            enumerate(zip(cam_names, pose_names)):
        dlabs = pd.read_hdf(pose_name)
        if ix_cam == 0:
            # this ensures that the joint order for the direct view is used uniformly
            bp_index = dlabs.columns.names.index('bodyparts')
            joint_names = list(dlabs.columns.get_level_values(bp_index).unique())
            try:
                ind_index = dlabs.columns.names.index('individuals')
                individuals = list(dlabs.columns.get_level_values(ind_index).unique())
            except:
                # if there is no 'individuals' column header
                ind_index = None
                individuals = None

        if not 'dir' in cam_name:
            # rename from "near/far" to "left/right" for mirror views
            dlabs = rename_mirror_columns(cam_name, dlabs)

        if len(dlabs.columns.levels) > 2:
            scorer = dlabs.columns.levels[0][0]
            dlabs = dlabs.loc[:, scorer]

        dx = offsets_dict[cam_name][0]
        dy = offsets_dict[cam_name][1]

        for ind in individuals:
            for joint in joint_names:
                try:
                    dlabs.loc[:, (ind, joint, 'x')] += dx
                    dlabs.loc[:, (ind, joint, 'y')] += dy
                except KeyError:
                    # for when this joint doesn't exist for a specific individual
                    pass
        datas.append(dlabs)

    n_cams = len(cam_names)
    n_joints = len(joint_names)
    n_frames = min([d.shape[0] for d in datas])

    # frame, camera, bodypart, xy
    points = np.full((n_cams, n_frames, n_joints, 2), np.nan, 'float')
    scores = np.full((n_cams, n_frames, n_joints), np.zeros(1), 'float')

    for cam_ix, dlabs in enumerate(datas):
        for joint_ix, joint_name in enumerate(joint_names):
            for ind in individuals:
                try:
                    # because points and score matrices are filled based on joint name, the joint order in the dataframe does not matter
                    points[cam_ix, :, joint_ix] = np.array(dlabs.loc[:, (ind, joint_name, ('x', 'y'))])[:n_frames]
                    scores[cam_ix, :, joint_ix] = np.array(dlabs.loc[:, (ind, joint_name, ('likelihood'))])[:n_frames].ravel()
                except KeyError:
                    # for when this joint doesn't exist for a specific individual
                    pass

    return {
        'cam_names': cam_names,
        'points': points,
        'scores': scores,
        'bodyparts': joint_names
    }


def datetime64_to_datetime_array(dt64_array):

    dt_list = [datetime64_to_datetime(dt64) for dt64 in dt64_array]
    dt_array = np.array(dt_list)

    return dt_array

def datetime64_to_datetime(dt64):

    ts = pd.to_datetime(dt64)
    year = ts.year
    month = ts.month
    day = ts.day
    hour = ts.hour
    minute = ts.minute
    second = ts.second

    return datetime(year, month, day, hour, minute, second)

def datetime64_to_date_array(dt64_array):

    dt_list = [datetime64_to_date(dt64).date() for dt64 in dt64_array]
    dt_array = np.array(dt_list)

    return dt_array

def datetime64_to_date(dt64):

    ts = pd.to_datetime(dt64)
    year = ts.year
    month = ts.month
    day = ts.day

    return datetime(year, month, day)


def rename_mirror_columns(cam_name, dlabs):

    if 'dir' in cam_name:
        return dlabs

    bp_index = dlabs.columns.names.index('bodyparts')
    ind_index = dlabs.columns.names.index('individuals')
    joint_names = list(dlabs.columns.get_level_values(bp_index).unique())
    ind_names = list(dlabs.columns.get_level_values(ind_index).unique())

    if cam_name == 'lm':
        near_side = 'right'
        far_side = 'left'
    elif cam_name == 'rm':
        near_side = 'left'
        far_side = 'right'

    for individual in ind_names:
        if 'rat' not in individual:
            continue

        for joint in joint_names:
            if 'near' in joint:
                new_joint = joint.replace('near', near_side)
                dlabs.rename(columns={joint: new_joint}, level=bp_index, inplace=True)
            if 'far' in joint:
                new_joint = joint.replace('far', far_side)
                dlabs.rename(columns={joint: new_joint}, level=bp_index, inplace=True)

    return dlabs