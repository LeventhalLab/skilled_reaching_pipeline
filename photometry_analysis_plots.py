import matplotlib.pyplot as plt
import numpy as np
import photometry_analysis
from datetime import datetime


def plot_mean_perievent_signal(perievent_data, analysis_window, phot_metadata, eventname, show_errors='sem', ax=None, ylim=[0.01, 0.08], **kwargs):

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # set some defaults
    kwarg_keys = [str.lower(k) for k in kwargs.keys()]
    if 'color' not in kwarg_keys:
        kwargs['color'] = 'k'
    if 'alpha' not in kwarg_keys:
        kwargs['alpha'] = 0.25

    try:
        num_samples = np.shape(perievent_data)[1]
    except:
        pass
    num_trials = np.shape(perievent_data)[0]
    mean_signal = np.mean(perievent_data, axis=0)
    std_signal = np.std(perievent_data, axis=0)

    if show_errors == 'std':
        ebar = std_signal
    elif show_errors == 'sem':
        ebar = std_signal / np.sqrt(num_trials)

    t = np.linspace(analysis_window[0], analysis_window[1], num_samples)
    try:
        ax.fill_between(t, mean_signal-ebar, mean_signal+ebar, **kwargs)
    except:
        pass

    # main plot is always fully opaque
    kwargs['alpha'] = 1.0
    ax.plot(t, mean_signal, **kwargs)

    ax.set_xlim(analysis_window)
    # ax.set_ylim(ylim)

    # text_string = phot_metadata['session_datetime'].strftime('%m-%d-%Y') + ', n = {:d}'.format(num_trials)
    title_string = eventname + '\nn = {:d}'.format(num_trials)
    ax.set_title(title_string, fontsize=10)
    # ax.text(t[0]+0.1, ylim[0]+0.005,  text_string, fontsize=9)

    return ax


def plot_phot_signal(t, data, analysis_window, ax=None, ylim=[0.01, 0.08], **kwargs):

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # set some defaults
    kwarg_keys = [str.lower(k) for k in kwargs.keys()]
    if 'color' not in kwarg_keys:
        kwargs['color'] = 'k'
    if 'alpha' not in kwarg_keys:
        kwargs['alpha'] = 0.25

    num_samples = len(data)

    ax.plot(t, data, **kwargs)
    ax.set_xlim(analysis_window)

    #todo: set x limits to time limits, limits plot to analysis_window, plot detrended data in next row

    pass



def plot_all_perievent_signals(perievent_data, analysis_window, ax=None):

    pass


def create_single_session_summary_panels(figwidth, figheight, numEvents):
    '''
    first row - raw photometry signal
    second row - detrended signal
    :param figwidth:
    :param figheight:
    :param nrows:
    :param numEvents:
    :return:
    '''

    row_1 = ['row1' for i_col in range(numEvents)]
    row_2 = ['row2' for i_col in range(numEvents)]
    row_3 = ['row3' for i_col in range(numEvents)]
    row_4 = ['row4_col{:d}'.format(i_col) for i_col in range(numEvents)]
    row_5 = ['row5_col{:d}'.format(i_col) for i_col in range(numEvents)]

    fig, axd = plt.subplot_mosaic([row_1, row_2, row_3, row_4, row_5],
                                  figsize=(figwidth, figheight))

    plt.subplots_adjust(hspace=0.5)
    return fig, axd


def create_axes_panels(figwidth, figheight, nrows, ncols):

    f = plt.figure(figsize=(figwidth, figheight))

    axs = f.subplots(nrows=nrows, ncols=ncols)

    return f, axs


def eliminate_x_labels(ax):
    ax.set_xticklabels([])
    ax.set_xlabel('')


def eliminate_y_labels(ax):
    ax.set_yticklabels([])
    ax.set_ylabel('')


def overlay_TTL(t, TTL, ax):

    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    scaled_TTL = (TTL - min(TTL)) / (max(TTL) - min(TTL))
    rescaled_TTL = (scaled_TTL * yrange) + ylim[0]

    ax.plot(t, rescaled_TTL)