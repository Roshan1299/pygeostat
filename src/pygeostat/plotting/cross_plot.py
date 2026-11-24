#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cross_plot plots one variable against another with the histogram of each beside it"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr
from matplotlib.ticker import LogLocator, NullFormatter
from .utils import get_label, format_plot, setup_plot, _set_stat_fontsize, _format_axes, _stat_names, _marginal_hists, _stat_number, _square_axes
from ..pygeostat_parameters import Parameters
from .export_image import export_image
from .set_style import set_plot_style


_STAT_NAMES = ['ndat', 'rho', 'rho_s', 'mean', 'std', 'var']
_STAT_SETS = {'all': ['ndat', 'rho', 'rho_s', 'mean', 'std'], 'minimal': ['rho']}


@set_plot_style
def cross_plot(x, y, nbins=20, lim=None, xlabel=None, ylabel=None, title=None, mc=None, ms=None,
               cmap=None, clim=None, log_color=False, cbar=False, cbar_label=None, hist_color=None, label_size=None,
               line45=False, stat_blk='all', figsize=None, ax=None, grid=None,
               axis_xy=None, x2=None, y2=None, mc2='C1', label=None, label2=None, logx=False, logy=False,
               log_counts=False, ticks=None, fontsize=None, stat_fontsize=None, xlim=None, ylim=None,
               background=None, alpha=0.6, legend_kws=None, line45_kws=None, stat_bbox=None, plot_style=None,
               custom_style=None, output_file=None, **kwargs):
    """
    Plot any variable ``y`` against any other variable ``x`` as points, with the histogram of ``x`` below and the histogram of ``y``
    to the right (the layout of Figure 9 of Castro Moreira, Costa and Deutsch, 2026). The number of pairs and the
    correlations are written on the scatter plot, and the mean and standard deviation next to each histogram. Pairs
    where either value is NaN are left out. On a log axis the statistics are those of the natural logarithm, which is
    what is plotted (the correlation is then labelled ``rho_ln``).

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
            ``gs.Parameters['plotting.cross_plot.histcolor']``. With two data sets their histograms take the color
            of their points, except a histogram of values that both sets share, which is drawn once in this color
        label_size (float): font size of the axis labels. If None, the one of the tick labels
        line45 (bool): draw the 45 degree line (x = y) and, if ``lim`` is None, use the same limits on both
            axes, to compare an estimate with the true values
        stat_blk (str or list): statistics to write. A preset, ``'all'`` (the number of pairs ``'ndat'``, the
            Pearson and Spearman correlation coefficients ``'rho'`` and ``'rho_s'``, and the ``'mean'`` and
            ``'std'`` of each variable) or ``'minimal'`` (``'rho'``), or a list of those names (``'var'`` is also
            available). ``False`` or ``'none'`` writes none. The number of pairs and the correlations are written on
            the scatter plot, in the corner with the fewest points, and the mean and standard deviation (or variance)
            next to each histogram
        figsize (tuple): figure size (width, height), if creating a new figure when ``ax`` is None.
            If None, the one of the plot style
        ax (ax): Matplotlib axis for the scatter plot. The histograms are drawn outside it, below and to
            its right, so leave room around it. If None, a new figure is made with a square plot
        grid (bool): plots the major grid lines if True. Based on Parameters['plotting.grid'] if None.
        axis_xy (bool): GSLIB-style axis visibility if True. Based on Parameters['plotting.axis_xy']
            if None.
        x2, y2: a second data set, drawn over the first in the colour ``mc2``, with its histograms over the
            first ones (as outlines in the colour of each set, so that neither hides the other). Both are needed.
            The two sets are given a legend, in an empty corner, and the statistics of each are written in its
            colour. When both sets have the same ``x`` (or ``y``) values, that histogram is drawn once.
            ``mc`` must then be a single colour (``'C0'`` if None), not ``'kde'`` or values
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
        alpha (float): opacity of the points of two data sets, from 0 (invisible) to 1. Default: 0.6. Their histograms
            are drawn as outlines in their colour, which are not made transparent
        legend_kws (dict): arguments for ``ax.legend`` (two data sets) that replace the defaults (a compact legend in the
            corner where it covers the fewest points; give ``loc`` to choose the corner)
        line45_kws (dict): arguments for the 45 degree line that replace the defaults (``color='k'``, ``lw=1``,
            ``ls='--'`` and ``zorder=1``)
        stat_bbox (dict): arguments for the box behind the statistics on the scatter plot that replace the
            defaults (a white box, ``facecolor='white'``, ``alpha=0.75``, ``edgecolor='none'`` and ``pad=2``)
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
        _square_axes(ax, colorbar=cbar and values is not None)
    if two:
        ax.scatter(x, y, c=mc, s=ms, alpha=alpha, label=label or 'Data 1')
        ax.scatter(x2, y2, c=mc2, s=ms, alpha=alpha, label=label2 or 'Data 2')
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
            colorbar = fig.colorbar(points, cax=ax.inset_axes([1.03, 0, 0.04, 1]))
    format_plot(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid, axis_xy=axis_xy, logx=logx, logy=logy)
    if logx:
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))  # a few more marks than the decades
        ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if xr is not None:
        ax.set_xlim(xr)
    if yr is not None:
        ax.set_ylim(yr)
    if line45:
        ax.plot(lim, lim, **dict(dict(color='k', lw=1, ls='--', zorder=1), **(line45_kws or {})))
    if label_size is not None:
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
    if values is not None and cbar:
        colorbar.ax.set_title(cbar_label, loc='left', pad=10, **({} if label_size is None else {'fontsize': label_size}))
    hist_stats = [n for n in ('mean', 'std', 'var') if n in stats_on]
    # a histogram of values both data sets share (for example the true values) is drawn once
    same_x = two and len(x) == len(x2) and np.array_equal(x.to_numpy(), x2.to_numpy())
    same_y = two and len(y) == len(y2) and np.array_equal(y.to_numpy(), y2.to_numpy())
    xsets = [(x, mc, None), (x2, mc2, None)] if two and not same_x else [(x, hist_color, None)]
    ysets = [(y, mc, None), (y2, mc2, None)] if two and not same_y else [(y, hist_color, None)]
    histx, histy = _marginal_hists(ax, xsets, ysets, edges[0], edges[1], stats=hist_stats, stat_fontsize=stat_size,
                                   log_counts=log_counts, logx=logx, logy=logy,
                                   gap=0.85 if values is not None and cbar else 0.12)
    # the corner with the fewest points holds the statistics, and the next one the legend
    xl, yl = ax.get_xlim(), ax.get_ylim()
    px = pd.concat([x, x2]) if two else x
    py = pd.concat([y, y2]) if two else y
    fx = ((np.log10(px) if logx else px) - (np.log10(xl[0]) if logx else xl[0])) / (
        (np.log10(xl[1]) - np.log10(xl[0])) if logx else (xl[1] - xl[0]))
    fy = ((np.log10(py) if logy else py) - (np.log10(yl[0]) if logy else yl[0])) / (
        (np.log10(yl[1]) - np.log10(yl[0])) if logy else (yl[1] - yl[0]))
    corners = [(0.05, 0.95, 'left', 'top', 'upper left'), (0.95, 0.95, 'right', 'top', 'upper right'),
               (0.05, 0.05, 'left', 'bottom', 'lower left'), (0.95, 0.05, 'right', 'bottom', 'lower right')]

    # the height of the box of statistics (all the data sets, one under the other), as a part of the plot
    n_lines = sum(n in stats_on for n in ('ndat', 'rho', 'rho_s'))
    box_height = ((n_lines + 0.8) * (2 if two else 1) + 0.6) * stat_size * 1.3 / (
        ax.get_position().height * fig.get_size_inches()[1] * 72)

    box_width = 11 * 0.6 * stat_size / (ax.get_position().width * fig.get_size_inches()[0] * 72)

    def points_under(corner):
        inside_x = (fx < 0.05 + box_width + 0.02) if corner[2] == 'left' else (fx > 0.95 - box_width - 0.02)
        inside_y = (fy > 1 - box_height - 0.05) if corner[3] == 'top' else (fy < box_height + 0.05)
        # the 45 degree line runs through the lower left and the upper right corners
        on_line = line45 and (corner[2], corner[3]) in (('left', 'bottom'), ('right', 'top'))
        return (inside_x & inside_y).sum() + (len(px) if on_line else 0)
    stats_corner = min(corners, key=points_under)
    legend_loc = None
    if two:
        # the legend and the statistics go to the two corners where together they cover the fewest points (and the
        # legend is not over the 45 degree line)
        renderer = fig.canvas.get_renderer() if hasattr(fig.canvas, 'get_renderer') else fig._get_renderer()
        both = ax.transData.transform(np.column_stack([px, py]))
        along = (np.geomspace if logx and logy else np.linspace)(*xl, 100)  # equal steps on the axes
        diagonal = ax.transData.transform(np.column_stack([along] * 2)) if line45 else None
        # a compact legend (the marker needs no long handle), as large as the tick labels
        kws = dict(dict(fontsize=ax.xaxis.get_ticklabels()[0].get_size(), handlelength=1.0, handletextpad=0.5,
                        borderaxespad=0.4), **(legend_kws or {}))
        has_stats = any(n in stats_on for n in ('ndat', 'rho', 'rho_s'))

        def covered(loc):
            legend = ax.legend(loc=loc, **kws)
            legend.draw(renderer)  # a legend that has not been drawn does not know its place yet
            box = legend.get_window_extent(renderer)
            # the box of the legend is wider than its text: what can hide a point is the markers and the text
            right = max(t.get_window_extent(renderer).x1 for t in legend.get_texts()) + 0.8 * stat_size
            legend.remove()
            inside = lambda xy: ((xy[:, 0] > box.x0) & (xy[:, 0] < right) & (xy[:, 1] > box.y0) & (xy[:, 1] < box.y1)).sum()
            return inside(both) + (inside(diagonal) if diagonal is not None else 0)
        if 'loc' in kws:
            legend_loc = kws.pop('loc')
        else:
            score = {c[4]: covered(c[4]) for c in corners}
            pairs = [(score[l[4]] + (points_under(s_) if has_stats else 0), points_under(s_), l, s_)
                     for l in corners for s_ in corners if not (has_stats and l is s_)]
            _, _, legend_corner, stats_corner = min(pairs, key=lambda t: t[:2])
            legend_loc = legend_corner[4]
    cx, cy, ha, va, _ = stats_corner
    # a light background keeps the statistics readable where they are over the points
    box = dict(dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2), **(stat_bbox or {}))
    sets_xy = [(x, y, mc), (x2, y2, mc2)] if two else [(x, y, None)]
    for k, (a, b, colour) in enumerate(sets_xy):
        # on a log axis the statistics are those of the logarithm, which is what is plotted
        a = np.log(a) if logx else a
        b = np.log(b) if logy else b
        lines = [t for n, t in (('ndat', '$n = $%d' % len(a)),
                                ('rho', (r'$\rho_{\ln} = $' if logx or logy else r'$\rho = $') + _stat_number(np.corrcoef(a, b)[0, 1])),
                                ('rho_s', r'$\rho_s = $' + _stat_number(spearmanr(a, b)[0])))
                 if n in stats_on]
        if lines:
            step = (len(lines) + 0.8) * stat_size * (1 if va == 'bottom' else -1)
            # the first data set is on top, as in the legend and beside the histograms
            place = (len(sets_xy) - 1 - k) if va == 'bottom' else k
            ax.annotate('\n'.join(lines), (cx, cy), xycoords='axes fraction', xytext=(0, place * step),
                        textcoords='offset points', ha=ha, va=va, bbox=box, fontsize=stat_size, linespacing=1.0,
                        **({'color': colour} if two else {}))
    if two:
        ax.legend(loc=legend_loc, **kws)
    _format_axes(ax, grid=grid, ticks=ticks, fontsize=fontsize, background=background)
    ax.histx, ax.histy = histx, histy
    if output_file or ('pdfpages' in kwargs):
        export_image(output_file, **kwargs)
    return ax
