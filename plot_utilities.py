import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def plot_shapely_point(pt, fc='blue', ax=None):

    if ax == None:
        # create figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.scatter(np.array(pt.coords.xy[0]), pt.coords.xy[1], fc=fc)

    return ax


def plot_polygon(poly, ax=None, fc='blue'):

    if ax == None:
        # create figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(111)

    a = [poly.exterior.coords[ii] for ii in range(len(poly.exterior.coords))]
    add_polygon_patch(poly.exterior.coords, ax, fc=fc)

    for interior in poly.interiors:
        add_polygon_patch(interior, ax, 'white')

    return ax


def add_polygon_patch(coords, ax, fc='blue'):
    patch = patches.Polygon(np.array(coords.xy).T, fc=fc)
    ax.add_patch(patch)
    ax.set_xlim(1000, 1200)
    ax.set_ylim(700, 900)
    ax.invert_yaxis()



