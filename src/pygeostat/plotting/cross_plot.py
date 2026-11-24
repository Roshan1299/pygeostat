#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cross_plot plots one variable against another with the histogram of each beside it"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from .utils import get_label, format_plot, setup_plot, _set_stat_fontsize, _stat_names
from ..pygeostat_parameters import Parameters
from .export_image import export_image
from .set_style import set_plot_style


_STAT_NAMES = ['ndat', 'rho', 'mean', 'var']
_STAT_SETS = {'all': ['ndat', 'rho', 'mean', 'var'], 'minimal': ['rho']}


@set_plot_style
def cross_plot(x, y, nbins=20, lim=None, xlabel=None, ylabel=None, title=None, mc=None, ms=None,
               cmap=None, clim=None, log_color=False, cbar=False, cbar_label=None, hist_color=None, label_size=None,
               line45=False, stat_blk='all', figsize=None, ax=None, grid=None,
               axis_xy=None, stat_fontsize=None, plot_style=None, custom_style=None, output_file=None, **kwargs):
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
            ``gs.Parameters['plotting.cross_plot.histcolor']``
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
        stat_fontsize (float): font size of the statistics (correlation, mean and variance). If None,
            ``gs.Parameters['plotting.stat_fontsize']``, and it does not follow ``fontsize``
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
    xs, ys = x, y
    if line45 and lim is None:
        lim = (min(xs.min(), ys.min()), max(xs.max(), ys.max()))
    xr = yr = lim
    for name, r in (('x', xr), ('y', yr)):
        if r is not None and len(r) != 2:
            raise ValueError("%slim must be (lower, upper), not %s" % (name, r))
        if r is not None and not r[0] < r[1]:
            raise ValueError("%slim must be increasing, (lower, upper), not %s" % (name, r))
    stat_size = _set_stat_fontsize(stat_fontsize)
    stats_on = _stat_names(stat_blk, _STAT_SETS, _STAT_NAMES)
    edges = [np.linspace(*(r if r is not None else (v.min(), v.max())), nbins + 1)
             for r, v in ((xr, xs), (yr, ys))]
    cmap = Parameters['plotting.scatter_plot.cmap'] if cmap is None else cmap
    ms = Parameters['plotting.cross_plot.s'] if ms is None else ms
    hist_color = Parameters['plotting.cross_plot.histcolor'] if hist_color is None else hist_color
    new_figure = ax is None
    fig, ax, _ = setup_plot(ax, figsize=figsize, aspect=False)
    if new_figure:
        ax.set_axes_locator(None)
        ax.set_position([0.28, 0.28, 0.60 if cbar and values is not None else 0.67, 0.67])
    histx = ax.inset_axes([0, -0.28, 1, 0.22], sharex=ax)
    histy = ax.inset_axes([-0.28, 0, 0.22, 1], sharey=ax)
    if values is None:
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
    if xr is not None:
        ax.set_xlim(xr)
    if yr is not None:
        ax.set_ylim(yr)
    format_plot(ax, title=title, grid=grid, axis_xy=axis_xy)
    ax.tick_params(labelleft=False, labelbottom=False)
    histx.hist(x, bins=edges[0], color=hist_color)
    histy.hist(y, bins=edges[1], color=hist_color, orientation='horizontal')
    histy.invert_xaxis()
    if line45:
        ax.plot(lim, lim, color='k', lw=1, ls='--', zorder=1)
    histx.set_ylim(top=histx.get_ylim()[1] * 1.6)  # room above the bars for the statistics
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
    if corner_stats:
        # the corner with the fewest points under the label
        xl, yl = ax.get_xlim(), ax.get_ylim()
        fx = (x - xl[0]) / (xl[1] - xl[0])
        fy = (y - yl[0]) / (yl[1] - yl[0])
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
        histx.annotate(text(x), (0.03, 0.97), xycoords='axes fraction', ha='left', va='top', fontsize=stat_size)
        histy.annotate(text(y), (0, -0.02), xycoords='axes fraction', ha='left', va='top', fontsize=stat_size)
    ax.histx, ax.histy = histx, histy
    if output_file or ('pdfpages' in kwargs):
        export_image(output_file, **kwargs)
    return ax
