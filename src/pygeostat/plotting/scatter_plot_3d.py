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
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from .export_image import export_image
from .set_style import set_plot_style
from .utils import _set_stat_fontsize, _stat_names
from ..pygeostat_parameters import Parameters

_DATASET = '_dataset'  # the column that tells the data sets of a list apart
_STAT_NAMES = ['ndat', 'rho', 'mean', 'var']
_STAT_SETS = {'all': ['ndat', 'rho', 'mean', 'var'], 'minimal': ['rho']}


def _log_ticks(lo, hi):
    """Positions (as log10) and labels of the ticks of a log axis between the log10 values ``lo`` and ``hi``: 1, 2
    and 5 times the powers of ten, or every whole number if that is fewer than three ticks, or only the powers of
    ten for a range of more than three decades. For a range with fewer than two of them, the two ends"""
    decades = np.arange(np.floor(lo), np.ceil(hi) + 1)
    options = [(1,)] if hi - lo > 3 else [(1, 2, 5), tuple(range(1, 10))]
    for steps in options:
        values = np.array([m * 10.0 ** d for d in decades for m in steps])
        values = values[(np.log10(values) >= lo - 1e-9) & (np.log10(values) <= hi + 1e-9)]
        if len(values) >= 3:
            break
    if len(values) < 2:  # a range too narrow for any of those: the two ends
        return np.array(sorted({lo, hi})), ['%.3g' % 10.0 ** v for v in sorted({lo, hi})]
    return np.log10(values), ['%g' % v for v in values]


def _stat_lines(values, stats_on):
    """The statistics asked for, one per line, of a table with the columns x, y and z"""
    lines = []
    if 'ndat' in stats_on:
        lines.append('n = %d' % len(values))
    if 'rho' in stats_on:
        with np.errstate(all='ignore'):
            rho = np.corrcoef(values.to_numpy().T)
        lines.append(r'$\rho$ (xy, xz, yz) = %.2f, %.2f, %.2f' % (rho[0, 1], rho[0, 2], rho[1, 2]))
    if 'mean' in stats_on:
        lines.append('Mean (x, y, z) = %.3g, %.3g, %.3g' % tuple(values.mean()))
    if 'var' in stats_on:
        lines.append('Var (x, y, z) = %.3g, %.3g, %.3g' % tuple(values.var(ddof=0)))
    return lines


@set_plot_style
def scatter_plot_3d(data, x, y, z, c=None, cat=None, cat_as=None, markers=None, colors=None, s=None,
                    alpha=None, cmap=None, clim=None, cbar=True, drop_lines=False, xlabel=None,
                    ylabel=None, zlabel=None, title=None, labels=None,
                    elev=25, azim=-60, grid=True, xlim=None, ylim=None, zlim=None,
                    logx=False, logy=False, logz=False, stat_blk=None, stat_fontsize=None, figsize=None, ax=None,
                    plot_style=None, custom_style=None, output_file=None, **kwargs):
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
        data: DataFile or DataFrame with the variables. A list of them draws several data sets in one plot,
            each in its own colour with a legend (they need the same column names for ``x``, ``y`` and ``z``;
            ``cat`` and ``c`` cannot be used then)
        x, y, z (str): column names of the variables on the three axes
        c (str): column name of a numeric variable that sets the colour of the points. A matplotlib
            colour instead gives all the points that colour. Default: one colour
        cat (str): column name of a category, with a legend. It sets the shape, the colour or both of
            the points, as ``cat_as`` says
        cat_as (str): ``'shape'`` (default), ``'color'`` or ``'both'``: what the category sets. With
            ``'color'`` and ``'both'`` the points cannot also be colored by the variable ``c``. For a
            list of data sets the default is ``'color'``, and ``'both'`` also gives each its own shape
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
        drop_lines (bool): draw a thin line from each point down to the floor of the plot. Together with
            the grid it makes the position of a point easier to read, since the three coordinates of a
            point cannot otherwise be told apart in a flat picture
        xlabel, ylabel, zlabel (str): axis labels (default: the column names)
        title (str): title of the plot
        labels (list): legend name of each data set when ``data`` is a list (default: ``'Data 1'``, ``'Data 2'``, ...)
        elev (float): elevation of the viewing angle in degrees
        azim (float): azimuth of the viewing angle in degrees
        grid (bool): plots the major grid lines if True
        xlim, ylim, zlim (tuple): ``(lower, upper)`` limits of an axis. The range of the data if None
        logx, logy, logz (bool): log scale for an axis, for a very skewed variable. The values must be above
            zero. The axis is drawn as the log10 of the values, with ticks at 1, 2 and 5 times the powers of ten,
            and the limits and the statistics are of the values
        stat_blk (str or list): statistics to write in the upper right corner, in the colour of the data set
            when ``data`` is a list. A preset, ``'all'`` (the number of points ``'ndat'``, the correlations
            ``'rho'`` of the three pairs of variables, and the ``'mean'`` and ``'var'`` of each) or
            ``'minimal'`` (``'rho'``), or a list of those names. ``False``, ``None`` or ``'none'`` writes none
        stat_fontsize (float): font size of the statistics, as in the other plots (below 1, a fraction of the font size).
            If None, ``gs.Parameters['plotting.stat_fontsize']``, or 0.75 if that is None too
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
    several = isinstance(data, (list, tuple))
    if several:
        if cat is not None or c is not None:
            raise ValueError("`cat` and `c` cannot be used with a list of data sets: they are told apart by colour")
        frames = [d.data if isinstance(d, DataFile) else d for d in data]
        if not frames or not all(isinstance(f, pd.DataFrame) for f in frames):
            raise ValueError("A list of data sets must hold gs.DataFile or pd.DataFrame objects")
        names = ['Data %d' % (i + 1) for i in range(len(frames))] if labels is None else list(labels)
        if len(names) != len(frames):
            raise ValueError("`labels` needs one name for each of the %d data sets" % len(frames))
        if len(set(names)) != len(names):
            raise ValueError("`labels` must be different for each data set, not %s" % names)
        for name in [x, y, z]:
            if any(name not in f.columns for f in frames):
                raise ValueError("%r is not a column of every data set" % (name,))
        for f, n in zip(frames, names):
            if f[[x, y, z]].dropna().empty:
                raise ValueError("%r has no row with a value in all of %r, %r and %r" % (n, x, y, z))
        # a categorical column keeps the order of the data sets in the legend
        data = pd.concat([f[[x, y, z]].assign(**{_DATASET: pd.Categorical([n] * len(f), categories=names)})
                          for f, n in zip(frames, names)], ignore_index=True)
        cat = _DATASET
        cat_as = 'color' if cat_as is None else cat_as
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
    stats_on = _stat_names(stat_blk, _STAT_SETS, _STAT_NAMES)
    # smaller than in the flat plots by default: the statistics share the corner with the points
    stat_size = _set_stat_fontsize(0.75 if stat_fontsize is None and Parameters['plotting.stat_fontsize'] is None
                                   else stat_fontsize)
    logs = (logx, logy, logz)
    for name, log, lim in zip(axes, logs, (xlim, ylim, zlim)):
        if log and not (table[name] > 0).all():
            raise ValueError("The log scale of %r needs values above zero" % name)
        if log and lim is not None and len(lim) == 2 and lim[0] <= 0:
            raise ValueError("The lower limit of a log scale must be above zero")
    raw = table
    table = table.copy()
    for name, log in zip(axes, logs):
        if log:
            table[name] = np.log10(table[name])
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
    for name, lim, setter, log in (('xlim', xlim, ax.set_xlim, logx), ('ylim', ylim, ax.set_ylim, logy),
                                   ('zlim', zlim, ax.set_zlim, logz)):
        if lim is not None:
            if len(lim) != 2:
                raise ValueError('{} must be (lower, upper), not {}'.format(name, lim))
            setter(np.log10(lim) if log else lim)
    if drop_lines:
        floor = ax.get_zlim()[0]
        ax.add_collection3d(Line3DCollection([((px, py, floor), (px, py, pz)) for px, py, pz in
                                              zip(table[x], table[y], table[z])], colors='0.55', linewidths=0.4,
                                             alpha=0.6, zorder=0), autolim=False)  # the floor stays where it is
    if colour_var and cbar:
        colorbar = fig.colorbar(points, ax=ax, shrink=0.6, pad=0.12)
        colorbar.set_label(colour_var, fontsize=size)
        colorbar.ax.tick_params(labelsize=size)
    if handles:
        # the title as large as the entries
        ax.legend(handles=handles, title=None if several else cat, loc='upper left', fontsize=size,
                  title_fontsize=size)
    # Few ticks, so that the labels of a corner do not run into each other
    for axis, name, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), axes, (xlabel, ylabel, zlabel)):
        axis.set_major_locator(MaxNLocator(nbins=5))
        axis.labelpad = 6
        axis.set_label_text(name if label is None else label, fontsize=size)
    for axis, log, limits in zip((ax.xaxis, ax.yaxis, ax.zaxis), logs, (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())):
        if log:
            positions, names = _log_ticks(*limits)
            axis.set_major_locator(FixedLocator(positions))
            axis.set_major_formatter(FixedFormatter(names))
    if stats_on:
        if several:
            blocks = [(raw.loc[raw[cat] == value, axes], colors[i] if cat_as in ('color', 'both') else 'k')
                      for i, (value, _) in enumerate(groups)]
        else:
            blocks = [(raw[axes], 'k')]
        offset = 0
        for block, color in blocks:  # the first data set on top
            lines = _stat_lines(block, stats_on)
            ax.annotate('\n'.join(lines), (0.98, 0.98), xycoords='axes fraction', xytext=(0, -offset),
                        textcoords='offset points', ha='right', va='top', fontsize=stat_size, color=color)
            offset += (len(lines) + 0.5) * stat_size * 1.3
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
