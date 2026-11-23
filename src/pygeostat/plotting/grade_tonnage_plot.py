#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""grade_tonnage_plot compares the grade-tonnage curves of several models of one variable"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from .secondary_axis_plot import _secondary_axis
from .export_image import export_image
from .set_style import set_plot_style
from ..pygeostat_parameters import Parameters


def _models(data, var, weights, labels):
    """The values, weights and label of each model, as a list of (label, values, weights)"""
    from ..data.data import DataFile
    many = isinstance(data, (list, tuple))
    sources = list(data) if many else [data]
    sources = [s.data if isinstance(s, DataFile) else s for s in sources]
    # a table of values without names: one model for each column
    sources = [pd.DataFrame(s, columns=['Model %d' % (i + 1) for i in range(s.shape[1])])
               if isinstance(s, np.ndarray) and s.ndim == 2 else s for s in sources]
    if many:
        names = [var] * len(sources) if (var is None or isinstance(var, str)) else list(var)
        if len(names) != len(sources):
            raise ValueError("`var` needs one name for each of the %d models, or a single name" % len(sources))
    else:
        if var is None and isinstance(sources[0], pd.DataFrame):
            names = list(sources[0].columns)
        else:
            names = [var] if (var is None or isinstance(var, str)) else list(var)
        sources = sources * len(names)
    models = []
    for i, (source, name) in enumerate(zip(sources, names)):
        if isinstance(source, pd.DataFrame):
            if name is None:
                if source.shape[1] != 1:
                    raise ValueError("`var` is needed to pick the column of a table with several columns")
                name = source.columns[0]
            if name not in source.columns:
                raise ValueError("%r is not a column of the data" % (name,))
            values = source[name]
        else:
            values = pd.Series(np.asarray(source, dtype=float).ravel())
            name = name if name is not None else 'Model %d' % (i + 1)
        if isinstance(weights, str):
            if not isinstance(source, pd.DataFrame) or weights not in source.columns:
                raise ValueError("%r is not a column of the data" % (weights,))
            w = source[weights]
        else:
            w = None if weights is None else pd.Series(np.asarray(weights, dtype=float).ravel())
        if w is not None and len(w) != len(values):
            raise ValueError("`weights` has %d values but the model has %d" % (len(w), len(values)))
        models.append((name, values.to_numpy(dtype=float), None if w is None else w.to_numpy(dtype=float)))
    given = labels is not None
    if labels is None:
        labels = [m[0] for m in models]
    if len(labels) != len(models):
        raise ValueError("`labels` needs one entry for each of the %d models" % len(models))
    return [(label, v, w) for label, (_, v, w) in zip(labels, models)], given


def _each(value, n, name):
    """One value for each of the n models, from a single value or a list"""
    if value is None or np.ndim(value) == 0:
        return [value] * n
    if len(value) != n:
        raise ValueError("`%s` needs one entry for each of the %d models, or a single value" % (name, n))
    return list(value)


def _curve(values, weights, cutoffs):
    """The weight at or above each cutoff, and the weighted mean of the values there"""
    weights = np.ones(len(values)) if weights is None else weights
    valid = np.isfinite(values) & np.isfinite(weights)
    values, weights = values[valid], weights[valid]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    above = np.append(np.cumsum(weights[::-1])[::-1], 0.0)
    metal = np.append(np.cumsum((weights * values)[::-1])[::-1], 0.0)
    first = np.searchsorted(values, cutoffs, side='left')
    with np.errstate(invalid='ignore', divide='ignore'):
        grade = np.where(above[first] > 0, metal[first] / above[first], np.nan)
    return above[first], grade, weights.sum()


def _prepare(values, weights, trim, clip):
    """The values and weights of a model without the values outside ``trim`` (below the lower limit or at
    or above the upper one), with the values above ``clip`` set to ``clip``"""
    if trim is not None:
        keep = ~((values < trim[0]) | (values >= trim[1]))
        values = values[keep]
        weights = None if weights is None else weights[keep]
    if clip is not None:
        values = np.minimum(values, clip)
    return values, weights


def _band(curves, p):
    """The lower and upper limit of the curves (one per row) at each cutoff: their interpolated percentiles
    (as ``numpy.percentile``) for the two p-values. There is none at a cutoff where a curve has no value"""
    low, high = np.full(curves.shape[1], np.nan), np.full(curves.shape[1], np.nan)
    for i in range(curves.shape[1]):
        if np.isfinite(curves[:, i]).all():
            low[i], high[i] = np.percentile(curves[:, i], [100 * p[0], 100 * p[1]])
    return low, high


@set_plot_style
def grade_tonnage_plot(data, var=None, labels=None, weights=None, cutoffs=None, ncut=40,
                       block_tonnes=None, colors=None, lw=None, alpha=None, trim=None, clip=None,
                       cut_range=None, average=False, band=None, realizations=None, xlabel=None,
                       ylabel=None, title=None, xlim=None, ylim=None, ylim2=None, logx=False, legend=None, figsize=None, ax=None,
                       grid=None, axis_xy=None, axis_colors=None, ticks=None, fontsize=None, background=None,
                       ylabel2=None, ls='--', ls2='-', average_color='k', average_lw=2, fill_color='0.5', fill_alpha=0.35,
                       legend_kws=None, plot_style=None, custom_style=None, output_file=None, **kwargs):
    """
    Plot the grade-tonnage curves of one variable for several models, to compare them on one plot.
    For each cutoff grade the dashed lines (left axis) are the tonnage of the blocks at or above the
    cutoff, and the solid lines (right axis) are the mean grade of those blocks. Each model has its own
    colour, so a data set, an estimate and simulated realizations can be compared, drawn in the order
    given.

    The models are either the columns of one table (``var`` is a list of column names, for example
    the realizations of a simulation), or a list of tables with the same or a different column name
    in each (``data`` is a list). Values that are NaN are left out.

    Differences from the CCG program ``gtcurve_sim``: the limits of ``band`` are interpolated percentiles
    and not sorted curves picked by rank, and the average grade (and the band of the grades) stops at the
    cutoff where a model has nothing above it, where ``gtcurve_sim`` uses a grade of 0. The fraction above
    the cutoff counts such a model as 0 in both programs.

    Please review the documentation of the :func:`gs.set_style()
    <pygeostat.plotting.set_style.set_style>` and :func:`gs.export_image()
    <pygeostat.plotting.export_image.export_image>` functions for details on their parameters so that
    their use in this function can be understood.

    Parameters:
        data: DataFile, DataFrame or array, or a list of them (one model each). Without ``var``, each
            column of a single table (or 2D array) is a model
        var (str or list): column name(s). With one table, a list gives one model per column. With a
            list of tables, one name for all of them or one name for each. Not needed for an array
        labels (list): one legend label for each model (default: the column names). A model with the
            label None is drawn but left out of the legend, so the many realizations of a simulation
            can share the one label of the first. With more than 10 models and no ``labels``, the
            legend is left out
        weights (str or array): weight of each value (for example a declustering weight or the volume
            of a block): the name of a column of each table, or one array for all the models. Default:
            equal weights
        cutoffs (array): cutoff grades. If None, ``ncut`` values from the smallest to the largest
            value of all the models (for a log axis, from the smallest positive value)
        ncut (int): number of default cutoffs
        cut_range (tuple): (smallest, largest) default cutoff, instead of the smallest and largest value of all
            the models. The ``ncut`` cutoffs are evenly spaced between them, ends included
        block_tonnes (float): tonnes in one unit of weight (for example the block volume times the
            density). If None, the tonnage is the fraction of the total weight of each model
        colors (list): one matplotlib color for each model (default: matplotlib's colour cycle)
        axis_colors: colour of the two vertical axes (their label, tick marks, tick labels and line), to tell which
            one belongs to which curves. ``'line'`` uses the colour of the line of the model, for a plot of one
            model. Or a pair ``(left, right)`` of matplotlib colors for the tonnage and the grade axes, where None
            keeps the colour of the style for that axis. Black, as in the style, if None
        ticks (bool, float or dict): tick marks on all axes. ``True`` draws them, ``False`` removes them, a
            number sets their length, a dictionary is passed to ``ax.tick_params``. If None,
            ``gs.Parameters['plotting.ticks']``
        fontsize (float or dict): font size of the axis labels, tick labels and title (one number), or a
            dictionary with any of ``'label'``, ``'tick'`` and ``'title'``. The style's sizes if None
        background (color): colour of the plotting area, for example ``'lightgrey'`` or ``'white'``. The
            style's if None
        lw (float or list): width of the lines, one value or one for each model (default: the one of
            the plot style)
        alpha (float or list): opacity of the lines from 0 (invisible) to 1, one value or one for each
            model. Lower it to draw many models, for example the realizations of a simulation
        trim (tuple): (lower, upper) limits of the values: those below the lower limit or at or above the
            upper one are left out, with their weights. Applied before ``clip``. As the trimming limits of
            the CCG program ``gtcurve``
        clip (float): upper limit of the values: the values above it are set to it (the clipping limit of
            ``gtcurve``)
        average (bool): also draw the average of the curves of the models (black), the mean of the
            curves and not the curve of the mean value, as ``gtcurve_sim`` does. A model with nothing
            above a cutoff has no grade there, so the average grade and the band end at the largest cutoff
            where every model has a grade (``gtcurve_sim`` counts that grade as 0, which would pull the
            average down). The tonnage of that model still counts as 0 in the average of the tonnages
        band (tuple): (lower, upper) p-values, for example ``(0.1, 0.9)``: shade the range of the curves
            between them. At each cutoff the band is the interpolated percentile of the curves (as
            ``numpy.percentile``), so with few models the limits do not collapse to the lowest and highest
            curve as the ranks of ``gtcurve_sim`` do
        realizations (list): the positions of the models (0 is the first) that ``average`` and ``band`` are
            made from, for example the realizations of a simulation when the first model is the data. If
            None, all the models
        xlabel (str): x-axis label (default: "Cutoff Grade")
        ylabel (str): label of the tonnage axis (default: "Tonnes Above Cutoff", or
            "Fraction Above Cutoff" without ``block_tonnes``)
        ylabel2 (str): label of the grade axis (default: "Mean Grade Above Cutoff")
        title (str): title of the plot
        xlim (tuple): x-axis limits
        ylim (tuple): limits of the tonnage axis (left), ``(lower, upper)``. They replace the range that makes
            the zeros of the two axes line up, so with ``ylim`` and ``ylim2`` the zeros may not match
        ylim2 (tuple): limits of the grade axis (right), ``(lower, upper)``
        logx (bool): log scale for the cutoff axis
        legend (bool): add a legend below the plot (default: True, except for more than 10 models
            without ``labels``)
        legend_kws (dict): arguments for ``ax.legend`` that replace the defaults (``loc``, ``bbox_to_anchor``,
            ``ncol``, ``frameon`` and ``fontsize``)
        ls, ls2 (str): line style of the tonnage lines (left axis) and of the grade lines (right axis).
            Default: dashed and solid
        average_color (color), average_lw (float): colour and width of the average line. Default: black, 2
        fill_color (color), fill_alpha (float): colour and opacity of the band. Default: grey, '0.5', 0.35
        figsize (tuple): figure size (width, height), if creating a new figure when ``ax`` is None.
            If None, ``gs.Parameters['plotting.grade_tonnage_plot.figsize']``
        ax (ax): Matplotlib axis to plot on, which gets the tonnage. The grade is on a twin axis.
        grid (bool): plots the major grid lines if True. Based on Parameters['plotting.grid'] if None.
        axis_xy (bool): GSLIB-style axis visibility if True. Based on Parameters['plotting.axis_xy']
            if None.
        plot_style (str): Use a predefined set of matplotlib plotting parameters
        custom_style (dict): Alter some of the predefined parameters in the ``plot_style`` selected
        output_file (str): Output figure file name and location
        **kwargs: Optional permissible keyword arguments to pass to
            :func:`gs.export_image() <pygeostat.plotting.export_image.export_image>`

    Returns:
        ax (ax): Matplotlib Axes object with the tonnage (left axis)

    **Examples:**

        Compare the data with an estimate:

        .. plot::

            import pygeostat as gs
            data = gs.ExampleData('3d_estimate')
            gs.grade_tonnage_plot(data, ['True', 'Estimate'], labels=['Data', 'Estimate'])

        |

        The data and the realizations of a simulation, with the average curve and the range between the
        10th and 90th p-values of the realizations:

        .. plot::

            import pygeostat as gs
            data = gs.ExampleData('accuracy_plot')
            gs.grade_tonnage_plot(data, labels=['Data'] + [None] * 100, colors=['C3'] + ['0.6'] * 100,
                                  lw=[2] + [0.5] * 100, alpha=[1] + [0.3] * 100,
                                  realizations=list(range(1, 101)), average=True, band=(0.1, 0.9))

    |
    """
    models, labelled = _models(data, var, weights, labels)
    if trim is not None and not trim[0] < trim[1]:
        raise ValueError("`trim` needs a lower limit below the upper one")
    if cut_range is not None and not cut_range[0] < cut_range[1]:
        raise ValueError("`cut_range` needs a smallest cutoff below the largest one")
    if band is not None and not 0 <= band[0] < band[1] <= 1:
        raise ValueError("`band` needs two p-values from 0 to 1, the lower one below the upper one")
    models = [(label,) + _prepare(v, w, trim, clip) for label, v, w in models]
    chosen = list(range(len(models))) if realizations is None else list(realizations)
    if (average or band is not None) and (not chosen or not all(
            isinstance(i, (int, np.integer)) and 0 <= i < len(models) for i in chosen)):
        raise ValueError("`realizations` needs positions from 0 to %d" % (len(models) - 1))
    legend = (labelled or len(models) <= 10) if legend is None else legend
    if colors is None:
        colors = ['C%d' % i for i in range(len(models))]
    if len(colors) != len(models):
        raise ValueError("`colors` needs one entry for each of the %d models" % len(models))
    if isinstance(axis_colors, str) and axis_colors == 'line':
        if len(models) != 1:
            raise ValueError("axis_colors='line' is for one model, give a pair of colors for %d models" % len(models))
        axis_colors = (colors[0], colors[0])
    elif axis_colors is not None and len(axis_colors) != 2:
        raise ValueError("axis_colors must be 'line' or a pair (left, right), not %r" % (axis_colors,))
    lws, alphas = _each(lw, len(models), 'lw'), _each(alpha, len(models), 'alpha')
    if ncut < 2:
        raise ValueError("`ncut` must be at least 2")
    if cutoffs is None:
        pooled = np.concatenate([m[1][np.isfinite(m[1])] for m in models])
        if pooled.size == 0:
            raise ValueError("None of the models has a value")
        if cut_range is not None:
            low, high = cut_range
        else:
            if logx:
                pooled = pooled[pooled > 0]
                if pooled.size == 0:
                    raise ValueError("A log axis needs positive values")
            low, high = pooled.min(), pooled.max()
        cutoffs = np.geomspace(low, high, ncut) if logx else np.linspace(low, high, ncut)
    cutoffs = np.asarray(cutoffs, dtype=float)
    if logx and (cutoffs <= 0).any():
        raise ValueError("A log axis needs positive cutoffs")
    curves = [_curve(v, w, cutoffs) for _, v, w in models]
    for (label, _, _), (_, _, total) in zip(models, curves):
        if not total > 0:
            raise ValueError("The model %r has no values with a weight above zero" % (label,))
    tonnages = [t * block_tonnes if block_tonnes is not None else t / total for t, _, total in curves]
    grades = [g for _, g, _ in curves]
    if average or band is not None:
        tons_of, grades_of = np.array([tonnages[i] for i in chosen]), np.array([grades[i] for i in chosen])
    figsize = Parameters['plotting.grade_tonnage_plot.figsize'] if figsize is None else figsize
    left, right, line_colors, line_lws, line_alphas = list(tonnages), list(grades), list(colors), list(lws), list(alphas)
    if average:
        mean_grade = np.where(np.isfinite(grades_of).all(axis=0), grades_of.mean(axis=0), np.nan)
        left.append(tons_of.mean(axis=0))
        right.append(mean_grade)
        line_colors.append(average_color)
        line_lws.append(average_lw)
        line_alphas.append(None)
    ax = _secondary_axis(
        cutoffs, left, right, colors=line_colors, colors2=line_colors, ls=ls, ls2=ls2, lw=line_lws, alpha=line_alphas,
        fill=_band(tons_of, band) if band is not None else None,
        fill2=_band(grades_of, band) if band is not None else None,
        xlabel=xlabel if xlabel is not None else 'Cutoff Grade',
        ylabel=ylabel if ylabel is not None else
        ('Fraction Above Cutoff' if block_tonnes is None else 'Tonnes Above Cutoff'),
        ylabel2=ylabel2 if ylabel2 is not None else 'Mean Grade Above Cutoff', title=title, xlim=xlim, ylim=ylim, ylim2=ylim2, logx=logx,
        axis_colors=axis_colors, legend=False, figsize=figsize, ax=ax, grid=grid, axis_xy=axis_xy, ticks=ticks,
        fontsize=fontsize, background=background, fill_color=fill_color, fill_alpha=fill_alpha)
    if block_tonnes is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: '{:,.0f}'.format(v)))
    shown = [i for i, m in enumerate(models) if m[0] is not None]
    what = 'Fraction' if block_tonnes is None else 'Tonnes'
    handles = [Line2D([], [], color=colors[i], ls=style,
                      **{k: v for k, v in (('lw', lws[i]), ('alpha', alphas[i])) if v is not None}) for style in (ls, ls2) for i in shown]
    names = ['%s - %s' % (models[i][0], w) for w in (what, 'Grade') for i in shown]
    if average:
        handles += [Line2D([], [], color=average_color, ls=ls, lw=average_lw), Line2D([], [], color=average_color, ls=ls2, lw=average_lw)]
        names += ['Average - %s' % what, 'Average - Grade']
    if band is not None:
        handles.append(Patch(color=fill_color, alpha=fill_alpha, lw=0))
        names.append('P%d to P%d' % (round(band[0] * 100), round(band[1] * 100)))
    if legend and handles:
        legend_args = dict(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=min(3, len(names)), frameon=False,
                           fontsize=ax.yaxis.get_ticklabels()[0].get_size())
        legend_args.update(legend_kws or {})
        ax.legend(handles, names, **legend_args)
    if output_file or ('pdfpages' in kwargs):
        export_image(output_file, **kwargs)
    return ax
