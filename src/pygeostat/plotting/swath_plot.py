#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""swath_plot plots the mean of one or more variables along a coordinate direction, with the
number of samples in each slice as bars"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from .utils import format_plot, setup_plot, _format_axes, _format_fontsize, _format_ticks
from .export_image import export_image
from .set_style import set_plot_style
from ..pygeostat_parameters import Parameters

_ORIENTS = ('x', 'y', 'z')


def _decimals(axes, which):
    """The fewest decimals that show every tick of ``which`` ('x' or 'y') on all the axes"""
    ticks = np.concatenate([getattr(a, 'get_%sticks' % which)() for a in axes])
    for decimals in range(4):
        if np.allclose(ticks, np.round(ticks, decimals), atol=1e-9):
            return decimals
    return 3


def _label_lines(ax, means, variables, labels, colors):
    """Write each label next to its line, where the line is furthest from the other lines (above the line
    if it is the higher one there, below it otherwise), so the labels do not cover each other"""
    low, high = ax.get_ylim()
    size = ax.yaxis.get_ticklabels()[0].get_size()
    left, right = ax.get_xlim()
    for i, (v, label, color) in enumerate(zip(variables, labels, colors)):
        others = means[[u for u in variables if u != v]].mean(axis=1)
        gap = ((means[v] - others.fillna(means[v])) / (high - low)).dropna()  # no other line there: no gap
        if gap.empty:  # nothing is drawn for a variable without a value
            continue
        x = gap.abs().idxmax()
        above = gap[x] > 0 or (gap[x] == 0 and i % 2 == 0)  # lines that coincide: alternate above and below
        side = (x - left) / (right - left)  # near an edge, the label grows inwards so it stays inside the plot
        ax.annotate(label, (x, means[v][x]), xytext=(0, 7 if above else -7), textcoords='offset points',
                    ha='right' if side > 0.8 else ('left' if side < 0.2 else 'center'),
                    va='bottom' if above else 'top', color=color, fontsize=size, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75), zorder=5)


@set_plot_style
def swath_plot(data, var, orient='x', x=None, y=None, z=None, nbins=20, width=None, bins=None,
               colors=None, labels=None, bars=True, bar_alpha=None, lw=None, xlabel=None, ylabel=None, title=None, figsize=None,
               legend=True, grid=None, axis_xy=None, count_style='bars', count_label=None, ms=5, ticks=None, fontsize=None,
               xlim=None, ylim=None, background=None, plot_style=None, custom_style=None,
               output_file=None, ax=None, **kwargs):
    """
    Plot a swath plot: the data are cut into slices along ``orient`` and the mean of each variable in
    each slice is drawn as a line (left axis). The bars (right axis) are the number of values in each
    slice, one thin bar per variable so that missing values can be seen. Slices without values are
    left out of the line.

    ``var`` can be a list of variables to compare on one plot (for example the data and an
    estimate), and ``orient`` can be a list of directions, which are stacked in one figure. To compare
    models that are in different files, give ``data`` as a list of data sets: each is cut with the same
    slices, which span the coordinates of all of them, and each is a line.

    Please review the documentation of the :func:`gs.set_style()
    <pygeostat.plotting.set_style.set_style>` and :func:`gs.export_image()
    <pygeostat.plotting.export_image.export_image>` functions for details on their parameters so that
    their use in this function can be understood.

    Parameters:
        data: DataFile or DataFrame with the coordinates and the variables, or a list of them. With a list,
            each data set gives one line, and the coordinates of each are read from it
        var (str or list): column name(s) of the variable(s) to plot. With a list of data sets, one name
            used for all of them, or a list with one name for each
        orient (str or list): ``'x'``, ``'y'`` or ``'z'``, or a list of them. One figure is made for each
        x, y, z (str): column names of the coordinates. Read from a DataFile when it knows them
        nbins (int): number of equal-width slices between the smallest and largest coordinate
        width (float): fixed slice width, in the units of the coordinate, starting at the smallest
            coordinate. Instead of ``nbins``
        bins (array): slice edges, instead of ``nbins`` and ``width``
        colors (list): one matplotlib color for each line (default: matplotlib's colour cycle)
        labels (list): one legend label for each line (default: the column names, or ``'Data 1'``,
            ``'Data 2'``, ... for a list of data sets that use the same column name)
        bars (bool): draw the counts as bars on the right axis
        bar_alpha (float): opacity of the bars, from 0 (invisible) to 1. If None, ``gs.Parameters['plotting.swath_plot.bar_alpha']``
        lw (float): width of the lines. If None, ``gs.Parameters['plotting.swath_plot.lw']``
        xlabel (str): x-axis label (default: ``plotting.xname``, ``yname`` or ``zname`` followed by
            ``plotting.unit`` in brackets, as in the other plots of coordinates; set them with
            ``gs.Parameters`` for the names or the unit of a data set, or pass ``xlabel``)
        ylabel (str): y-axis label of the lines (default: the variable name, or "Value" for several
            variables, followed by "(lines)")
        title (str): title of the plot
        figsize (tuple): size (width, height) of the plot of one direction; with several directions the
            figure is that many times as high. If None, ``gs.Parameters['plotting.swath_plot.figsize']``
        legend (bool): write the label of each line next to it, in its colour, when there is more than one
            variable. There is no legend box: the label is where the line is furthest from the others
        grid (bool): plots the major grid lines if True. Based on Parameters['plotting.grid'] if None.
        axis_xy (bool): GSLIB-style axis visibility if True. Based on Parameters['plotting.axis_xy']
            if None.
        count_style (str): ``'bars'`` draws the counts as bars, ``'points'`` as one point per slice
            (``bar_alpha`` is the opacity of both). Only used if ``bars`` is True
        count_label (str): label of the right axis (default: "Data Count (Bars)" or "Data Count (Points)")
        ms (float): size of the points of ``count_style='points'``
        ticks (bool, float or dict): tick marks on all axes. ``True`` draws them, ``False`` removes them,
            a number sets their length, a dictionary is passed to ``ax.tick_params``. If None,
            ``gs.Parameters['plotting.ticks']``
        fontsize (float or dict): font size of the axis labels, tick labels and title (one number), or a
            dictionary with any of ``'label'``, ``'tick'`` and ``'title'``. The style's sizes if None
        xlim (tuple): ``(lower, upper)`` limits of the x-axis, the same on every direction. The data's if None
        ylim (tuple): ``(lower, upper)`` limits of the left axis (the lines). The data's if None
        background (color): colour of the plotting area, for example ``'lightgrey'`` or ``'white'``. The
            style's if None
        plot_style (str): Use a predefined set of matplotlib plotting parameters
        custom_style (dict): Alter some of the predefined parameters in the ``plot_style`` selected
        output_file (str): Output figure file name and location
        ax (ax): Matplotlib axis to plot on (only for a single ``orient``)
        **kwargs: Optional permissible keyword arguments to pass to
            :func:`gs.export_image() <pygeostat.plotting.export_image.export_image>`

    Returns:
        ax (ax): Matplotlib Axes object with the swath plot, or a list of them (one per direction,
            top to bottom) if ``orient`` is a list

    **Examples:**

        .. plot::

            import pygeostat as gs
            data = gs.ExampleData('3d_estimate')
            gs.swath_plot(data, ['True', 'Estimate'], orient='x', labels=['Data', 'Estimate'])

        |

        Several directions are stacked in one figure, here with slices 1 unit wide:

        .. plot::

            import pygeostat as gs
            data = gs.ExampleData('3d_estimate')
            gs.swath_plot(data, ['True', 'Estimate'], orient=['x', 'y'], width=1,
                          labels=['Data', 'Estimate'])

        |

        A 2D data set with other column names (``Xlocation`` and ``Ylocation``), so only two directions:

        .. plot::

            import pygeostat as gs
            data = gs.ExampleData('cluster')
            gs.swath_plot(data, 'Primary', orient=['x', 'y'], nbins=10)

    |
    """
    from ..data.data import DataFile
    several = isinstance(data, (list, tuple))
    sets = list(data) if several else [data]
    if several and not sets:
        raise ValueError("The list of data sets is empty")
    names = [var] * len(sets) if several and isinstance(var, str) else ([var] if isinstance(var, str) else list(var))
    if several and len(names) != len(sets):
        raise ValueError("`var` needs one name for each of the %d data sets, or one name for all" % len(sets))
    orients = [orient] if isinstance(orient, str) else list(orient)
    for o in orients:
        if o not in _ORIENTS:
            raise ValueError("`orient` must be 'x', 'y' or 'z' (or a list of them), not %r" % (o,))
    # each line is a variable of a data set: its table and the columns of its coordinates
    series = []
    for i, d in enumerate(sets if several else [sets[0]] * len(names)):
        cols = {'x': x, 'y': y, 'z': z}
        if isinstance(d, DataFile):
            cols = {o: (getattr(d, o) if cols[o] is None else cols[o]) for o in _ORIENTS}
            d = d.data
        if not isinstance(d, pd.DataFrame):
            raise ValueError("The parameter data must be a gs.DataFile or a pd.DataFrame, or a list of them")
        for o in orients:
            if cols[o] is None:
                raise ValueError("The column of `%s` is not set and could not be read from the data" % o)
            if cols[o] not in d.columns:
                raise ValueError("%r is not a column of the data" % (cols[o],))
        if names[i] not in d.columns:
            raise ValueError("%r is not a column of the data" % (names[i],))
        series.append((d, cols, names[i]))
    variables = [name for _, _, name in series]
    keys = list(range(len(series)))  # the variable name is the same for lines of different data sets
    if labels is None:
        labels = ['Data %d' % (i + 1) for i in keys] if several and len(set(variables)) < len(variables) else variables
    if colors is None:
        colors = ['C%d' % i for i in keys]
    if len(labels) != len(series) or len(colors) != len(series):
        raise ValueError("`labels` and `colors` need one entry for each line")
    if len(orients) > 1 and ax is not None:
        raise ValueError("`ax` can only be used with a single `orient`")
    if width is not None and not width > 0:
        raise ValueError("`width` must be positive")
    if count_style not in ('bars', 'points'):
        raise ValueError("`count_style` must be 'bars' or 'points', not %r" % (count_style,))
    # the same names the spatial plots use for the axes
    axisnames = {o: Parameters['plotting.%sname' % o] for o in _ORIENTS}
    unit = Parameters['plotting.unit']
    swaths = {}
    for o in orients:
        coord = pd.concat([d[cols[o]] for d, cols, _ in series])
        if bins is not None:
            edges = np.asarray(bins)
        elif width is not None:
            edges = coord.min() + width * np.arange(int(np.ceil((coord.max() - coord.min()) / width)) + 1)
            if len(edges) < 2:
                edges = np.array([coord.min(), coord.min() + width])
        else:
            edges = np.linspace(coord.min(), coord.max(), nbins + 1)
        if bins is None and coord.min() == coord.max():
            raise ValueError("All values of %r are the same, so there is nothing to slice along" % series[0][1][o])
        centers = 0.5 * (edges[1:] + edges[:-1])
        cuts = [pd.cut(d[cols[o]], edges, include_lowest=True) for d, cols, _ in series]
        means = pd.DataFrame({k: d[v].groupby(cut, observed=False).mean() for k, cut, (d, _, v) in zip(keys, cuts, series)})
        counts = pd.DataFrame({k: d[v].groupby(cut, observed=False).count() for k, cut, (d, _, v) in zip(keys, cuts, series)})
        means.index = counts.index = centers
        swaths[o] = (means, counts, edges)
    figsize = Parameters['plotting.swath_plot.figsize'] if figsize is None else figsize
    lw = Parameters['plotting.swath_plot.lw'] if lw is None else lw
    bar_alpha = Parameters['plotting.swath_plot.bar_alpha'] if bar_alpha is None else bar_alpha
    maxcount = max(s[1].values.max() for s in swaths.values())
    if maxcount == 0:
        raise ValueError("None of the variables has a value in the slices")
    # One figure, with the directions stacked, or the axis given. Made last, so an error leaves no
    # figure open
    if ax is not None:
        axes = [ax]
    elif len(orients) == 1:
        axes = [setup_plot(None, figsize=figsize, aspect=False)[1]]
    else:
        fig = setup_plot(None, figsize=(figsize[0], figsize[1] * len(orients)), aspect=False)[0]
        fig.clf()
        axes = list(fig.subplots(len(orients), 1))
        fig.subplots_adjust(hspace=0.35)
    for o, a in zip(orients, axes):
        means, counts, edges = swaths[o]
        width = np.diff(edges).mean()
        for k, label, color in zip(keys, labels, colors):
            valid = means[k].notna()
            a.plot(means.index[valid], means[k][valid], color=color, lw=lw, label=label, zorder=2)
        if legend and len(variables) > 1:
            a.margins(y=0.16)  # room above and below the lines for their labels
        name = axisnames[o] + (' (%s)' % unit if unit else '')
        format_plot(a, xlabel=xlabel if xlabel is not None else name,
                    ylabel=ylabel if ylabel is not None
                    else (variables[0] if len(variables) == 1 else 'Value') + ' (lines)',
                    title=title, grid=grid, axis_xy=axis_xy)
        if bars:
            b = a.twinx()
            barwidth = 0.08 * width
            for i, color in enumerate(colors):
                offset = (i - (len(variables) - 1) / 2) * barwidth
                if count_style == 'bars':
                    b.bar(counts.index + offset, counts[i], width=barwidth, color=color,
                          alpha=bar_alpha, linewidth=0)
                else:
                    b.plot(counts.index + offset, counts[i], 'o', color=color, alpha=bar_alpha,
                           ms=ms, markeredgewidth=0)
            b.set_ylim(0, 1.05 * counts.values.max())
            b.yaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True))  # counts: whole, evenly spaced ticks
            b.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: '{:,.0f}'.format(v)))
            # the lines and their labels are drawn over the bars
            a.set_zorder(b.get_zorder() + 1)
            a.patch.set_visible(False)
            # The right axis gets the sizes of the left one, so the two sides match
            b.set_ylabel(count_label if count_label is not None else
                         'Data Count (Bars)' if count_style == 'bars' else 'Data Count (Points)',
                         fontsize=a.yaxis.label.get_size())
            b.tick_params(axis='y', labelsize=a.yaxis.get_ticklabels()[0].get_size())
            if background is not None:  # the lines' axis is see-through, so the colour goes on this one
                b.set_facecolor(background)
        _format_axes(a, grid=grid, ticks=ticks, fontsize=fontsize, xlim=xlim, ylim=ylim, background=background)
        if bars:
            _format_ticks(b, ticks)
            _format_fontsize(b, fontsize)
    # The same tick format on every panel, so stacked directions read alike
    for a in axes:
        a.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 5, 10], prune='upper'))
        left, right = a.get_xlim()
        # no tick label right at the right edge, where it would run into the label of the right axis
        a.set_xticks([t for t in a.get_xticks() if left <= t <= right - 0.08 * (right - left)])
    xdecimals, ydecimals = _decimals(axes, 'x'), _decimals(axes, 'y')
    for a in axes:
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: '{:,.{}f}'.format(v, xdecimals)))
        a.yaxis.set_major_formatter(FormatStrFormatter('%.{}f'.format(ydecimals)))
    if len(axes) > 1:
        axes[0].figure.align_ylabels(axes)
    if legend and len(variables) > 1:
        for o, a in zip(orients, axes):
            _label_lines(a, swaths[o][0], keys, labels, colors)
    if output_file or ('pdfpages' in kwargs):
        export_image(output_file, **kwargs)
    return axes if len(orients) > 1 else axes[0]
