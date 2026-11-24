#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""scatter_plot_3d plots three variables of a data set against each other in one 3D scatter plot"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, is_color_like
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from .export_image import export_image
from .set_style import set_plot_style
from ..pygeostat_parameters import Parameters



@set_plot_style
def scatter_plot_3d(data, x, y, z, c=None, cat=None, cat_as=None, markers=None, colors=None, s=None,
                    alpha=None, cmap=None, clim=None, cbar=True, xlabel=None,
                    ylabel=None, zlabel=None, title=None,
                    elev=25, azim=-60, grid=True, figsize=None, ax=None, plot_style=None,
                    custom_style=None, output_file=None, **kwargs):
    """
    Plot three variables of a data set against each other in a 3D scatter plot, for example the gold,
    copper and silver grades of the same samples. The variables are the axes, not the coordinates of
    the samples. A fourth variable can set the colour of the points, and a category can set their
    shape (circle, triangle, square, ...), their colour, or both. Values that are NaN in any of the
    variables used are left out.

    The plot is an ordinary Matplotlib 3D axis, so in a notebook with an interactive backend
    (``%matplotlib widget``, which needs ``ipympl``) or in a desktop window (``%matplotlib qt``) it can
    be rotated by dragging. ``elev`` and ``azim`` set the angle of a still picture.

    Please review the documentation of the :func:`gs.set_style()
    <pygeostat.plotting.set_style.set_style>` and :func:`gs.export_image()
    <pygeostat.plotting.export_image.export_image>` functions for details on their parameters so that
    their use in this function can be understood.

    Parameters:
        data: DataFile or DataFrame with the variables
        x, y, z (str): column names of the variables on the three axes
        c (str): column name of a numeric variable that sets the colour of the points. A matplotlib
            colour instead gives all the points that colour. Default: one colour
        cat (str): column name of a category, with a legend. It sets the shape, the colour or both of
            the points, as ``cat_as`` says
        cat_as (str): ``'shape'`` (default), ``'color'`` or ``'both'``: what the category sets. With
            ``'color'`` and ``'both'`` the points cannot also be colored by the variable ``c``
        markers (list): matplotlib marker symbols, one for each category in order of the sorted
            categories. Default: circle, triangle, square, diamond, inverted triangle, plus, cross
        colors (list): matplotlib colors, one for each category in order of the sorted categories.
            Default: matplotlib's ``tab10`` for up to 10 categories, otherwise
            ``gs.Parameters['plotting.cmap_cat']``
        s (float): area of the points. If None, 30
        alpha (float): opacity of the points, from 0 (invisible) to 1. If None, 0.9
        cmap (str): colormap of ``c``. If None, ``gs.Parameters['plotting.scatter_plot.cmap']``
        clim (tuple): limits (min, max) of the colour scale of ``c``. Default: the range of ``c``
        cbar (bool): add a colour bar when ``c`` is a variable
        xlabel, ylabel, zlabel (str): axis labels (default: the column names)
        title (str): title of the plot
        elev (float): elevation of the viewing angle in degrees
        azim (float): azimuth of the viewing angle in degrees
        grid (bool): plots the major grid lines if True
        figsize (tuple): figure size (width, height), if creating a new figure when ``ax`` is None.
            If None, ``gs.Parameters['plotting.scatter_plot_3d.figsize']``
        ax (ax): Matplotlib 3D axis to plot on (made with ``projection='3d'``)
        plot_style (str): Use a predefined set of matplotlib plotting parameters
        custom_style (dict): Alter some of the predefined parameters in the ``plot_style`` selected
        output_file (str): Output figure file name and location
        **kwargs: Optional permissible keyword arguments to pass to
            :func:`gs.export_image() <pygeostat.plotting.export_image.export_image>`

    Returns:
        ax (ax): Matplotlib 3D Axes object with the scatter plot

    **Examples:**

        Three variables on the axes, coloured by a fourth:

        .. plot::

            import pygeostat as gs
            data = gs.ExampleData('3d_correlation')
            gs.scatter_plot_3d(data, 'Var1', 'Var2', 'Var3', c='PC1')

        |

        Three variables with the shape of the points set by a category:

        .. plot::

            import pygeostat as gs
            data = gs.ExampleData('point3d_ind_mv').data.sample(500, random_state=1)
            gs.scatter_plot_3d(data, 'Phi', 'Vsh', 'Sw', cat='Lithofacies')

    |
    """
    from ..data.data import DataFile
    if isinstance(data, DataFile):
        data = data.data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The parameter data must be either a gs.DataFile or a pd.DataFrame")
    cat_as = 'shape' if cat_as is None else cat_as
    axes = [x, y, z]
    colour_var = c if isinstance(c, str) and c in data.columns else None
    used = axes + ([colour_var] if colour_var else []) + ([cat] if cat is not None else [])
    for name in used:
        if name not in data.columns:
            raise ValueError("%r is not a column of the data" % (name,))
    if c is not None and colour_var is None and not is_color_like(c):
        raise ValueError("%r is neither a column of the data nor a matplotlib colour" % (c,))
    if cat_as not in ('shape', 'color', 'both'):
        raise ValueError("`cat_as` must be 'shape', 'color' or 'both', not %r" % (cat_as,))
    if colour_var and cat is not None and cat_as != 'shape':
        raise ValueError("The category sets the colour of the points (cat_as=%r), so they cannot also be "
                         "colored by the variable %r" % (cat_as, colour_var))
    if ax is not None and not hasattr(ax, 'zaxis'):
        raise ValueError("`ax` must be a 3D axis, made with projection='3d'")
    table = data[used].dropna()
    if table.empty:
        raise ValueError("No row has a value in all of %s" % ', '.join(repr(u) for u in used))
    groups = [(None, table)]
    if cat is not None:
        groups = [(value, rows) for value, rows in table.groupby(cat, sort=True, observed=True)]
        if cat_as in ('shape', 'both'):
            markers = ['o', '^', 's', 'D', 'v', 'P', 'X'] if markers is None else markers
            if len(groups) > len(markers):
                raise ValueError("There are %d categories and %d markers: pass `markers` with at least "
                                 "as many symbols" % (len(groups), len(markers)))
        if cat_as in ('color', 'both'):
            if colors is None:
                palette = plt.get_cmap('tab10' if len(groups) <= 10 else Parameters['plotting.cmap_cat'])
                colors = [palette(i) for i in range(len(groups))]
            if len(groups) > len(colors):
                raise ValueError("There are %d categories and %d colors: pass `colors` with at least "
                                 "as many colors" % (len(groups), len(colors)))
    s = 30 if s is None else s
    alpha = 0.9 if alpha is None else alpha
    cmap = Parameters['plotting.scatter_plot.cmap'] if cmap is None else cmap
    if colour_var:
        lo, hi = (table[colour_var].min(), table[colour_var].max()) if clim is None else clim
        norm = Normalize(vmin=lo, vmax=hi)
    figsize = Parameters['plotting.scatter_plot_3d.figsize'] if figsize is None else figsize
    # Made last, so that an error leaves no figure open
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.figure
    size = ax.xaxis.get_ticklabels()[0].get_size()  # the axis names and the legend are as large as the tick labels
    # Points that differ only by their shape are dark with a white edge, which reads better than a pale one
    shape_only = cat is not None and cat_as == 'shape' and not colour_var and c is None
    shape_color, edge = ('0.15', 'w') if shape_only else ('C0', 'k')
    handles = []
    for i, (value, rows) in enumerate(groups):
        marker = markers[i] if cat is not None and cat_as in ('shape', 'both') else 'o'
        style = dict(marker=marker, s=s, alpha=alpha, edgecolors=edge, linewidths=0.4)
        by_color = cat is not None and cat_as in ('color', 'both')
        if colour_var:
            points = ax.scatter(rows[x], rows[y], rows[z], c=rows[colour_var], cmap=cmap, norm=norm,
                                **style)
        else:
            colour = colors[i] if by_color else (c if c is not None else shape_color)
            points = ax.scatter(rows[x], rows[y], rows[z], color=colour, **style)
        if cat is not None:
            handles.append(Line2D([], [], ls='', marker=marker, mfc=colors[i] if by_color else (c if colour_var is None and c is not None else shape_color),
                                  mec=edge, mew=0.4, label=_name(value)))
    if colour_var and cbar:
        colorbar = fig.colorbar(points, ax=ax, shrink=0.6, pad=0.12)
        colorbar.set_label(colour_var, fontsize=size)
        colorbar.ax.tick_params(labelsize=size)
    if handles:
        # the title as large as the entries
        ax.legend(handles=handles, title=cat, loc='upper left', fontsize=size,
                  title_fontsize=size)
    # Few ticks, so that the labels of a corner do not run into each other
    for axis, name, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), axes, (xlabel, ylabel, zlabel)):
        axis.set_major_locator(MaxNLocator(nbins=5))
        axis.labelpad = 6
        axis.set_label_text(name if label is None else label, fontsize=size)
    if title is not None:
        ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(bool(grid))
    if output_file or ('pdfpages' in kwargs):
        export_image(output_file, **kwargs)
    return ax


def _name(value):
    """The name of a category in the legend: 1 for 1.0"""
    return str(int(value)) if isinstance(value, (float, np.floating)) and float(value).is_integer() else str(value)
