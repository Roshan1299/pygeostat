#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cross_plot plots one variable against another with the histogram of each beside it"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter
from .utils import get_label, format_plot, setup_plot, _set_stat_fontsize, _format_axes, _stat_names
from ..pygeostat_parameters import Parameters
from .export_image import export_image
from .set_style import set_plot_style


_STAT_NAMES = ['ndat', 'rho', 'mean', 'var']
_STAT_SETS = {'all': ['ndat', 'rho', 'mean', 'var'], 'minimal': ['rho']}


@set_plot_style
def cross_plot(x, y, nbins=20, lim=None, xlabel=None, ylabel=None, title=None, mc=None, ms=None,
               cmap=None, clim=None, log_color=False, cbar=False, cbar_label=None, hist_color=None, label_size=None,
               line45=False, stat_blk='all', figsize=None, ax=None, grid=None,
               axis_xy=None, x2=None, y2=None, mc2='C1', label=None, label2=None, logx=False, logy=False,
               log_counts=False, ticks=None, fontsize=None, stat_fontsize=None, xlim=None, ylim=None,
               background=None, plot_style=None, custom_style=None, output_file=None, **kwargs):
    """
    Plot any variable ``y`` against any other variable ``x`` as points, with the histogram of ``x`` below and the histogram of ``y``
    on the left, as in the cross plots of the Geostatistics Lessons. The correlation coefficient is
    written on the scatter plot and the mean and variance on each histogram. Pairs where either value
    is NaN are left out.

    Please review the documentation of the :func:`gs.set_style()
    <pygeostat.plotting.set_style.set_style>` and :func:`gs.export_image()
    <pygeostat.plotting.export_image.export_image>` functions for details on their parameters so that
    their use in this function can be understood.

    Parameters:
        x: 1D data for the horizontal axis (pandas Series or DataFrame column, list or numpy array)
        y: 1D data for the vertical axis
        nbins (int): number of histogram bins
        lim (tuple): (min, max) of both axes, so the plot is square. By default each axis fits its data.
            The statistics always use all the values, also those outside ``lim``
        xlabel (str): label of the x axis, which is also the title of its histogram. Taken from the
            name of ``x`` when it has one
        ylabel (str): label of the y axis
        title (str): title of the plot
        mc: color of the points: a matplotlib color, ``'kde'`` to color each point by the density of the
            points around it (as :func:`gs.scatter_plot() <pygeostat.plotting.scatter_plot.scatter_plot>`
            does; one color if all the points are on a line), or a pandas Series or numpy array with a value for each pair to color the points by it
            (for example a third variable). If None, ``gs.Parameters['plotting.cross_plot.c']``
        ms (float): size of the points (area, as in ``matplotlib.pyplot.scatter``). If None,
            ``gs.Parameters['plotting.cross_plot.s']``
        cmap (str): matplotlib colormap for ``'kde'`` or values. If None,
            ``gs.Parameters['plotting.scatter_plot.cmap']``
        clim (tuple): (min, max) of the colors when the points are colored by values or ``'kde'``
        log_color (bool): log scale for the colors of values, for a very skewed variable. Values that are
            not above zero get the lowest color
        cbar (bool): add a colour bar when the points are colored by values or ``'kde'``
        cbar_label (str): label of the colour bar (default: the name of the values, or "Density" for ``'kde'``)
        hist_color (str): matplotlib color of the histogram bars. If None,
            ``gs.Parameters['plotting.cross_plot.histcolor']``. Ignored with two data sets, whose
            histograms take the color of their points
        label_size (float): font size of the axis labels. If None, the one of the tick labels
        line45 (bool): draw the 45 degree line (x = y) and, if ``lim`` is None, use the same limits on both
            axes, to compare an estimate with the true values
        stat_blk (str or list): statistics to write. A preset, ``'all'`` (the number of pairs ``'ndat'``, the
            correlation coefficient ``'rho'``, and the ``'mean'`` and ``'var'`` of each variable) or
            ``'minimal'`` (``'rho'``), or a list of those names. ``False`` or ``'none'`` writes none. The
            number of pairs and the correlation are written on the scatter plot, in the corner with the
            fewest points, and the mean and variance above the histogram of ``x`` and under the one of ``y``
        figsize (tuple): figure size (width, height), if creating a new figure when ``ax`` is None.
            If None, the one of the plot style
        ax (ax): Matplotlib axis for the scatter plot. The histograms are drawn beside it, below and to
            its left, so leave room around it. If None, a new figure is made
        grid (bool): plots the major grid lines if True. Based on Parameters['plotting.grid'] if None.
        axis_xy (bool): GSLIB-style axis visibility if True. Based on Parameters['plotting.axis_xy']
            if None.
        x2, y2: a second data set, drawn over the first in the colour ``mc2``, with its histograms over
            the first ones (both half transparent). Both are needed. The two sets are given a legend, with
            the number of pairs and the correlation coefficient of each (if ``stat_blk`` has them), and
            the mean and variance of each are written in its colour. ``mc`` must then be a single colour (``'C0'`` if None), not ``'kde'`` or values
        mc2 (str): matplotlib color of the second data set
        label, label2 (str): legend labels of the two data sets (default ``'Data 1'`` and ``'Data 2'``)
        logx, logy (bool): log scale of the x or y axis, for the points and the histograms. The values must be
            above zero, and the histogram bins are equal in log scale
        log_counts (bool): log scale for the counts of the two histograms (the axes without tick labels)
        ticks (bool, float or dict): tick marks on the three axes. ``True`` draws them, ``False`` removes
            them, a number sets their length, a dictionary is passed to ``ax.tick_params``. If None,
            ``gs.Parameters['plotting.ticks']``
        fontsize (float or dict): font size of the axis labels, tick labels and title (one number), or a
            dictionary with any of ``'label'``, ``'tick'`` and ``'title'``. The style's sizes if None
        stat_fontsize (float): font size of the statistics (correlation, mean and variance). If None,
            ``gs.Parameters['plotting.stat_fontsize']``, and it does not follow ``fontsize``
        xlim, ylim (tuple): ``(lower, upper)`` limits of one axis, instead of ``lim`` for that axis. The
            data's if None
        background (color): colour of the plotting area of the scatter plot, for example ``'lightgrey'``
            or ``'white'``. The style's if None
        plot_style (str): Use a predefined set of matplotlib plotting parameters
        custom_style (dict): Alter some of the predefined parameters in the ``plot_style`` selected
        output_file (str): Output figure file name and location
        **kwargs: Optional permissible keyword arguments to pass to
            :func:`gs.export_image() <pygeostat.plotting.export_image.export_image>`

    Returns:
        ax (ax): Matplotlib Axes object of the scatter plot. The histogram axes are ``ax.histx`` and
        ``ax.histy``

    **Examples:**

        .. plot::

            import pygeostat as gs
            data = gs.ExampleData('3d_estimate')
            gs.cross_plot(data.data['Estimate'], data.data['True'], line45=True)

    |
    """
    xlabel = get_label(x) if xlabel is None else xlabel
    ylabel = get_label(y) if ylabel is None else ylabel
    for name, v in (('x', x), ('y', y)):
        if getattr(v, 'ndim', 1) > 1 and np.shape(v)[1] != 1:
            raise ValueError("%s must be one variable, not %d columns" % (name, np.shape(v)[1]))
    x = pd.Series(np.ravel(x), dtype=float)
    y = pd.Series(np.ravel(y), dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    two = x2 is not None or y2 is not None
    if two:
        if x2 is None or y2 is None:
            raise ValueError("x2 and y2 must be given together")
        x2 = pd.Series(np.ravel(x2), dtype=float)
        y2 = pd.Series(np.ravel(y2), dtype=float)
        if len(x2) != len(y2):
            raise ValueError("x2 and y2 must have the same length")
        keep2 = x2.notna() & y2.notna()
        x2, y2 = x2[keep2], y2[keep2]
        if len(x2) < 2:
            raise ValueError("At least two pairs without NaN are needed in the second data set")
        if mc is None:
            mc = 'C0'
        elif isinstance(mc, (pd.Series, np.ndarray)) or (isinstance(mc, str) and mc.lower() == 'kde'):
            raise ValueError("mc must be a single colour when x2 and y2 are given")
    mc = Parameters['plotting.cross_plot.c'] if mc is None else mc
    values = None
    if isinstance(mc, (pd.Series, np.ndarray)):
        if np.ndim(mc) != 1 or len(mc) != len(x):
            raise ValueError("The values of mc must be one value for each of the %d pairs" % len(x))
        cbar_label = get_label(mc) if cbar_label is None else cbar_label
        values = pd.Series(np.asarray(mc, dtype=float))
    keep = x.notna() & y.notna() & (True if values is None else values.notna())
    x, y = x[keep], y[keep]
    if len(x) < 2:
        raise ValueError("At least two pairs without NaN are needed")
    density = isinstance(mc, str) and mc.lower() == 'kde'
    if density:
        from scipy.stats import gaussian_kde
        try:
            values = pd.Series(gaussian_kde(np.vstack([x, y])).evaluate(np.vstack([x, y])), index=x.index)
            cbar_label = 'Density' if cbar_label is None else cbar_label
        except (np.linalg.LinAlgError, ValueError):
            values, mc = None, 'C0'  # all the points on a line: there is no density
    elif values is not None:
        values = values[keep]
    if log_color and values is not None and not (values > 0).any():
        raise ValueError("log_color needs at least one value above zero")
    xs = x if x2 is None else pd.concat([x, x2])
    ys = y if y2 is None else pd.concat([y, y2])
    for name, log, v in (('x', logx, xs), ('y', logy, ys)):
        if log and not (v > 0).all():
            raise ValueError("The log scale of %s needs values above zero" % name)
    if line45 and lim is None:
        lim = (min(xs.min(), ys.min()), max(xs.max(), ys.max()))
    xr = lim if xlim is None else xlim
    yr = lim if ylim is None else ylim
    for name, log, r, v in (('x', logx, xr, xs), ('y', logy, yr, ys)):
        if r is not None and len(r) != 2:
            raise ValueError("%slim must be (lower, upper), not %s" % (name, r))
        if r is not None and not r[0] < r[1]:
            raise ValueError("%slim must be increasing, (lower, upper), not %s" % (name, r))
        if log and (r if r is not None else (v.min(),))[0] <= 0:
            raise ValueError("The lower limit of a log scale must be above zero")
    stat_size = _set_stat_fontsize(stat_fontsize)
    stats_on = _stat_names(stat_blk, _STAT_SETS, _STAT_NAMES)
    edges = [(np.geomspace if log else np.linspace)(*(r if r is not None else (v.min(), v.max())), nbins + 1)
             for log, r, v in ((logx, xr, xs), (logy, yr, ys))]
    cmap = Parameters['plotting.scatter_plot.cmap'] if cmap is None else cmap
    ms = Parameters['plotting.cross_plot.s'] if ms is None else ms
    hist_color = Parameters['plotting.cross_plot.histcolor'] if hist_color is None else hist_color
    new_figure = ax is None
    fig, ax, _ = setup_plot(ax, figsize=figsize, aspect=False)
    if new_figure:
        ax.set_axes_locator(None)
        ax.set_position([0.28, 0.28, 0.60 if cbar and values is not None else 0.67, 0.67])
    fmtstr = '{{x:.{}g}}'.format(Parameters.get('plotting.sigfigs') or 3)  # as format_plot, for log axes
    histx = ax.inset_axes([0, -0.28, 1, 0.22], sharex=ax)
    histy = ax.inset_axes([-0.28, 0, 0.22, 1], sharey=ax)
    if two:
        def tag(a, b):
            parts = (['n = %d' % len(a)] if 'ndat' in stats_on else []) + (
                [r'$\rho = %.3f$' % np.corrcoef(a, b)[0, 1]] if 'rho' in stats_on else [])
            return ' (%s)' % ', '.join(parts) if parts else ''
        ax.scatter(x, y, c=mc, s=ms, alpha=0.6, label=(label or 'Data 1') + tag(x, y))
        ax.scatter(x2, y2, c=mc2, s=ms, alpha=0.6, label=(label2 or 'Data 2') + tag(x2, y2))
        ax.legend(loc='best')
    elif values is None:
        points = ax.scatter(x, y, c=mc, s=ms)
    else:
        order = np.argsort(values.to_numpy())  # the densest or highest points are drawn on top
        vmin, vmax = (None, None) if clim is None else clim
        norm = None
        if log_color:
            positive = values[values > 0]
            norm = LogNorm(vmin=positive.min() if vmin is None else vmin,
                           vmax=positive.max() if vmax is None else vmax, clip=True)
            vmin = vmax = None
        points = ax.scatter(x.to_numpy()[order], y.to_numpy()[order], c=values.to_numpy()[order], s=ms,
                            cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
        if cbar:
            colorbar = fig.colorbar(points, cax=ax.inset_axes([1.03, 0, 0.035, 1]))
    if logx:
        ax.set_xscale('log')
        histx.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))  # a few more marks than the decades
        histx.xaxis.set_major_formatter(StrMethodFormatter(fmtstr))
        histx.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.set_yscale('log')
        histy.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))  # a few more marks than the decades
        histy.yaxis.set_major_formatter(StrMethodFormatter(fmtstr))
        histy.yaxis.set_minor_formatter(NullFormatter())
    if xr is not None:
        ax.set_xlim(xr)
    if yr is not None:
        ax.set_ylim(yr)
    format_plot(ax, title=title, grid=grid, axis_xy=axis_xy)
    ax.tick_params(labelleft=False, labelbottom=False)
    bars = dict(alpha=0.6, log=log_counts) if two else dict(log=log_counts)
    histx.hist(x, bins=edges[0], color=mc if two else hist_color, **bars)
    histy.hist(y, bins=edges[1], color=mc if two else hist_color, orientation='horizontal', **bars)
    if two:
        histx.hist(x2, bins=edges[0], color=mc2, **bars)
        histy.hist(y2, bins=edges[1], color=mc2, orientation='horizontal', **bars)
    histy.invert_xaxis()
    if line45:
        ax.plot(lim, lim, color='k', lw=1, ls='--', zorder=1)
    histx.set_ylim(top=histx.get_ylim()[1] * (8.0 if log_counts else 1.6))  # room above the bars for the statistics
    histx.set_yticks([])
    histy.set_xticks([])
    size = histx.xaxis.get_ticklabels()[0].get_size() if label_size is None else label_size
    histx.set_xlabel(xlabel, fontsize=size)
    histy.set_ylabel(ylabel, fontsize=size)
    if values is not None and cbar:
        colorbar.set_label(cbar_label, fontsize=size)
    # a light background keeps the statistics readable where they are over the points or the bars
    box = dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2)
    corner_stats = [t for n, t in (('ndat', 'n = %d' % len(x)), ('rho', r'$\rho = %.3f$' % np.corrcoef(x, y)[0, 1]))
                    if n in stats_on]
    if corner_stats and not two:
        # the corner with the fewest points under the label
        xl, yl = ax.get_xlim(), ax.get_ylim()
        fx = ((np.log10(x) if logx else x) - (np.log10(xl[0]) if logx else xl[0])) / (
            (np.log10(xl[1]) - np.log10(xl[0])) if logx else (xl[1] - xl[0]))
        fy = ((np.log10(y) if logy else y) - (np.log10(yl[0]) if logy else yl[0])) / (
            (np.log10(yl[1]) - np.log10(yl[0])) if logy else (yl[1] - yl[0]))
        corners = [(0.05, 0.95, 'left', 'top'), (0.95, 0.95, 'right', 'top'),
                   (0.05, 0.05, 'left', 'bottom'), (0.95, 0.05, 'right', 'bottom')]

        def points_under(corner):
            inside_x = (fx < 0.35) if corner[2] == 'left' else (fx > 0.65)
            inside_y = (fy > 0.88) if corner[3] == 'top' else (fy < 0.12)
            return (inside_x & inside_y).sum()
        cx, cy, ha, va = min(corners, key=points_under)
        ax.text(cx, cy, '\n'.join(corner_stats), transform=ax.transAxes, ha=ha, va=va, bbox=box,
                fontsize=stat_size)
    hist_stats = [n for n in ('mean', 'var') if n in stats_on]
    if hist_stats:
        lines = {'mean': lambda v: 'Mean = %.2f' % v.mean(), 'var': lambda v: 'Var = %.2f' % v.var(ddof=0)}
        text = lambda v: '\n'.join(lines[n](v) for n in hist_stats)
        for i, (xv, yv, color) in enumerate([(x, y, mc), (x2, y2, mc2)] if two else [(x, y, None)]):
            colour = {} if color is None else {'color': color}
            # the second data set: beside the first above the histogram of x, below it under the one of y
            histx.annotate(text(xv), (0.03 + 0.5 * i, 0.97), xycoords='axes fraction', ha='left', va='top',
                           fontsize=stat_size, **colour)
            histy.annotate(text(yv), (0, -0.02), xycoords='axes fraction', xytext=(0, -i * (len(hist_stats) + 0.6) * stat_size),
                           textcoords='offset points', ha='left', va='top', fontsize=stat_size, **colour)
    _format_axes(ax, grid=grid, ticks=ticks, fontsize=fontsize, background=background)
    for h in (histx, histy):
        _format_axes(h, grid=False, ticks=ticks, fontsize=fontsize)
    ax.histx, ax.histy = histx, histy
    if output_file or ('pdfpages' in kwargs):
        export_image(output_file, **kwargs)
    return ax
