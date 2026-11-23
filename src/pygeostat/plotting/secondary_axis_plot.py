#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""secondary_axis_plot draws lines against one x axis on a left and a right y axis"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .utils import format_plot, setup_plot, _format_axes, _format_fontsize, _format_ticks
from .export_image import export_image
from .set_style import set_plot_style
from ..pygeostat_parameters import Parameters


def _align_zeros(ax, twin):
    """Extend the bottom of the two y-axes so that their zeros are at the same height. Only when the
    right axis reaches zero and above"""
    (lo_l, hi_l), (lo_r, hi_r) = ax.get_ylim(), twin.get_ylim()
    if hi_r <= 0 or hi_l <= 0 or lo_r > 0:  # values that do not reach zero keep their own range
        return
    f = max(-lo_l / (hi_l - lo_l), -lo_r / (hi_r - lo_r), 0.0)
    if f >= 1:
        return
    ax.set_ylim(-f * hi_l / (1 - f), hi_l)
    twin.set_ylim(-f * hi_r / (1 - f), hi_r)


def _lines(values, n, side):
    """The lines of one side as a list of float arrays, and their names if the table gave them"""
    if isinstance(values, pd.DataFrame):
        arrays, names = [values[c].to_numpy(dtype=float) for c in values.columns], [str(c) for c in values.columns]
    elif isinstance(values, pd.Series):
        arrays, names = [values.to_numpy(dtype=float)], [None if values.name is None else str(values.name)]
    elif isinstance(values, (list, tuple)) and len(values) > 0 and all(np.ndim(v) == 1 for v in values):
        arrays, names = [np.asarray(v, dtype=float) for v in values], [None] * len(values)
    else:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("`%s` must be one line, a table with a line in each column, or a list of lines" % side)
        arrays, names = [values[:, i] for i in range(values.shape[1])], [None] * values.shape[1]
    if not arrays:
        raise ValueError("`%s` has no line" % side)
    for a in arrays:
        if len(a) != n:
            raise ValueError("Every line of `%s` needs one value for each of the %d values of `x`" % (side, n))
    return arrays, names


def _each(value, n, name):
    """A value for each of ``n`` lines, from one value or a list with at least ``n``"""
    if value is None or np.isscalar(value):
        return [value] * n
    value = list(value)
    if len(value) < n:
        raise ValueError("`%s` needs one entry for each of the %d lines" % (name, n))
    return value[:n]


def _secondary_axis(x, y, y2, labels=None, labels2=None, colors=None, colors2=None, ls='-', ls2='--', lw=None,
                    alpha=None, fill=None, fill2=None, fill_label=None, xlabel=None, ylabel=None, ylabel2=None,
                    title=None, xlim=None, ylim=None, ylim2=None, logx=False, align_zeros=True, axis_colors=None,
                    legend=None, figsize=None, ax=None, grid=None, axis_xy=None, ticks=None, fontsize=None,
                    background=None):
    """The drawing of :func:`secondary_axis_plot`, without the plot style and the export, so that a plot built
    on it (such as grade_tonnage_plot) can add to the figure before it is saved. Returns the left axis, with the
    right one as ``ax.twin``"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 1:
        raise ValueError("`x` must be one list of values")
    left, names = _lines(y, len(x), 'y')
    right, names2 = _lines(y2, len(x), 'y2')
    if (labels is not None and len(labels) != len(left)) or (labels2 is not None and len(labels2) != len(right)):
        raise ValueError("`labels` and `labels2` need one entry for each line of `y` and of `y2`")
    labels = [l if l is not None else (n or 'Line %d' % (i + 1)) for i, (l, n) in enumerate(
        zip(labels if labels is not None else [None] * len(left), names))]
    labels2 = [l if l is not None else (n or 'Line %d' % (len(left) + i + 1)) for i, (l, n) in enumerate(
        zip(labels2 if labels2 is not None else [None] * len(right), names2))]
    colors = ['C%d' % i for i in range(len(left))] if colors is None else list(colors)
    colors2 = ['C%d' % (len(left) + i) for i in range(len(right))] if colors2 is None else list(colors2)
    if len(colors) != len(left) or len(colors2) != len(right):
        raise ValueError("`colors` and `colors2` need one color for each line of `y` and of `y2`")
    most = max(len(left), len(right))
    lws, alphas = _each(lw, most, 'lw'), _each(alpha, most, 'alpha')
    if isinstance(axis_colors, str) and axis_colors == 'line':
        if len(left) != 1 or len(right) != 1:
            raise ValueError("axis_colors='line' is for one line on each axis, give a pair of colors for %d and %d "
                             "lines" % (len(left), len(right)))
        axis_colors = (colors[0], colors2[0])
    elif axis_colors is not None and len(axis_colors) != 2:
        raise ValueError("axis_colors must be 'line' or a pair (left, right), not %r" % (axis_colors,))
    for name, lim in (('ylim', ylim), ('ylim2', ylim2)):
        if lim is not None and (len(lim) != 2 or not lim[0] < lim[1]):
            raise ValueError("%s must be (lower, upper), with lower below upper, not %r" % (name, lim))
    for name, band in (('fill', fill), ('fill2', fill2)):
        if band is not None and (len(band) != 2 or any(len(b) != len(x) for b in band)):
            raise ValueError("%s must be (lower, upper), each with one value for each of the %d values of `x`"
                             % (name, len(x)))
    figsize = Parameters['plotting.secondary_axis_plot.figsize'] if figsize is None else figsize
    # Made last, so that an error leaves no figure open
    fig, ax, _ = setup_plot(ax, figsize=figsize, aspect=False)
    twin = ax.twinx()
    for axis, band in ((ax, fill), (twin, fill2)):
        if band is not None:
            axis.fill_between(x, band[0], band[1], color='0.5', alpha=0.35, lw=0)
    handles = []
    for axis, arrays, side_colors, style, side_labels in ((ax, left, colors, ls, labels),
                                                          (twin, right, colors2, ls2, labels2)):
        for i, (values, color, label) in enumerate(zip(arrays, side_colors, side_labels)):
            line_style = {k: v for k, v in (('lw', lws[i]), ('alpha', alphas[i])) if v is not None}
            axis.plot(x, values, color=color, ls=style, **line_style)
            handles.append((Line2D([], [], color=color, ls=style, **line_style), label))
    format_plot(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid, axis_xy=axis_xy, xlim=xlim, logx=logx)
    if align_zeros:
        _align_zeros(ax, twin)
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylim2 is not None:
        twin.set_ylim(ylim2)
    # The right axis gets the sizes of the left one, so the two sides match
    if ylabel2 is not None:
        twin.set_ylabel(ylabel2, fontsize=ax.yaxis.label.get_size())
    twin.tick_params(axis='y', labelsize=ax.yaxis.get_ticklabels()[0].get_size())
    _format_axes(ax, grid=grid, ticks=ticks, fontsize=fontsize, background=background)
    _format_ticks(twin, ticks)
    _format_fontsize(twin, fontsize)
    for axis, side, color in zip((ax, twin), ('left', 'right'), axis_colors or (None, None)):
        if color is not None:
            axis.yaxis.label.set_color(color)
            axis.tick_params(axis='y', colors=color)
            axis.spines[side].set_color(color)
    if len(handles) > 1 if legend is None else legend:
        entries = list(handles)
        if fill_label is not None and (fill is not None or fill2 is not None):
            entries.append((Patch(color='0.5', alpha=0.35, lw=0), fill_label))
        ax.legend([h for h, _ in entries], [l for _, l in entries], loc='upper center',
                  bbox_to_anchor=(0.5, -0.16), ncol=min(3, len(entries)), frameon=False,
                  fontsize=ax.yaxis.get_ticklabels()[0].get_size())
    ax.twin = twin
    return ax


@set_plot_style
def secondary_axis_plot(x, y, y2, labels=None, labels2=None, colors=None, colors2=None, ls='-', ls2='--', lw=None,
                        alpha=None, fill=None, fill2=None, fill_label=None, xlabel=None, ylabel=None, ylabel2=None,
                        title=None, xlim=None, ylim=None, ylim2=None, logx=False, align_zeros=True,
                        axis_colors=None, legend=None, figsize=None, ax=None, grid=None, axis_xy=None, ticks=None,
                        fontsize=None, background=None, plot_style=None, custom_style=None, output_file=None,
                        **kwargs):
    """
    Plot lines against one x axis, some on a left y axis and some on a right y axis with its own scale, for
    example two different properties of the same samples or models. Each axis has its own label, limits and
    colour, so that it is clear which lines it belongs to. :func:`gs.grade_tonnage_plot()
    <pygeostat.plotting.grade_tonnage_plot.grade_tonnage_plot>` is made with it.

    Please review the documentation of the :func:`gs.set_style()
    <pygeostat.plotting.set_style.set_style>` and :func:`gs.export_image()
    <pygeostat.plotting.export_image.export_image>` functions for details on their parameters so that
    their use in this function can be understood.

    Parameters:
        x: the values of the x axis, in order
        y: the line(s) of the left axis, each with one value per ``x``: one list of values, a table (a
            ``pd.DataFrame`` or a 2D array) with a line in each column, or a list of lists of values. A NaN leaves
            a gap in a line
        y2: the line(s) of the right axis, in the same ways as ``y``
        labels, labels2 (list): legend names of the lines of ``y`` and of ``y2``. Default: the column names of a
            table, otherwise ``'Line 1'``, ``'Line 2'``, ... counting the lines of ``y`` first
        colors, colors2 (list): one matplotlib color for each line of ``y`` and of ``y2``. Default: matplotlib's
            colour cycle, continuing from the lines of ``y`` for ``y2``
        ls, ls2 (str): line style of the lines of ``y`` and of ``y2``. By default the right axis is dashed
        lw (float or list): line width, or one for each line (the same list is used for both axes, so it needs
            as many entries as the longer of ``y`` and ``y2``). Default: matplotlib's
        alpha (float or list): opacity of the lines, from 0 to 1, in the same way as ``lw``
        fill, fill2 (tuple): ``(lower, upper)``, each one value per ``x``, a grey band drawn behind the lines of
            the left or of the right axis, for example the range of several realizations
        fill_label (str): legend name of the bands
        xlabel, ylabel, ylabel2 (str): labels of the x axis, the left axis and the right axis
        title (str): title of the plot
        xlim (tuple): ``(lower, upper)`` limits of the x axis. The data's if None
        ylim, ylim2 (tuple): ``(lower, upper)`` limits of the left and of the right axis. They replace the range
            that makes the zeros of the two axes line up, so with both the zeros may not match
        logx (bool): log scale for the x axis
        align_zeros (bool): extend the bottom of the two vertical axes so that their zeros are at the same
            height, when the right axis has values at and above zero
        axis_colors: colour of the two vertical axes (their label, tick marks, tick labels and line), to tell which
            lines they belong to. A pair ``(left, right)``, or ``'line'`` for the colours of the lines when there is
            one line on each axis. By default they keep the colour of the style
        legend (bool): write a legend below the plot. Default: True if there is more than one line
        figsize (tuple): figure size (width, height), if creating a new figure when ``ax`` is None.
            If None, ``gs.Parameters['plotting.secondary_axis_plot.figsize']``
        ax (ax): Matplotlib axis to plot on, which gets the left axis. The right axis is a twin axis
        grid (bool): plots the major grid lines if True. Based on Parameters['plotting.grid'] if None.
        axis_xy (bool): GSLIB-style axis visibility if True. Based on Parameters['plotting.axis_xy']
            if None.
        ticks (bool, float or dict): tick marks on all axes. ``True`` draws them, ``False`` removes them,
            a number sets their length, a dictionary is passed to ``ax.tick_params``. If None,
            ``gs.Parameters['plotting.ticks']``
        fontsize (float or dict): font size of the axis labels, tick labels and title (one number), or a
            dictionary with any of ``'label'``, ``'tick'`` and ``'title'``. The style's sizes if None
        background (color): colour of the plotting area, for example ``'lightgrey'`` or ``'white'``. The
            style's if None
        plot_style (str): Use a predefined set of matplotlib plotting parameters
        custom_style (dict): Alter some of the predefined parameters in the ``plot_style`` selected
        output_file (str): Output figure file name and location
        **kwargs: Optional permissible keyword arguments to pass to
            :func:`gs.export_image() <pygeostat.plotting.export_image.export_image>`

    Returns:
        ax (ax): Matplotlib Axes object of the left axis. The right axis is ``ax.twin``

    **Examples:**

        Two properties of the same samples, each axis coloured like its line:

        .. plot::

            import numpy as np
            import pygeostat as gs
            depth = np.linspace(0, 100, 50)
            gs.secondary_axis_plot(depth, np.exp(-depth / 40), 2 + 0.05 * depth, labels=['Fraction'],
                                   labels2=['Grade'], xlabel='Depth (m)', ylabel='Fraction',
                                   ylabel2='Grade (%)', axis_colors='line')

        |
    """
    ax = _secondary_axis(x, y, y2, labels=labels, labels2=labels2, colors=colors, colors2=colors2, ls=ls, ls2=ls2,
                         lw=lw, alpha=alpha, fill=fill, fill2=fill2, fill_label=fill_label, xlabel=xlabel,
                         ylabel=ylabel, ylabel2=ylabel2, title=title, xlim=xlim, ylim=ylim, ylim2=ylim2, logx=logx,
                         align_zeros=align_zeros, axis_colors=axis_colors, legend=legend, figsize=figsize, ax=ax,
                         grid=grid, axis_xy=axis_xy, ticks=ticks, fontsize=fontsize, background=background)
    if output_file or ('pdfpages' in kwargs):
        export_image(output_file, **kwargs)
    return ax
