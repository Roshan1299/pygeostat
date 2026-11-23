#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""grade_tonnage_plot compares the grade-tonnage curves of several models of one variable"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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


@set_plot_style
def grade_tonnage_plot(data, var=None, labels=None, weights=None, cutoffs=None, ncut=40,
                       block_tonnes=None, colors=None, lw=None, alpha=None, xlabel=None,
                       ylabel=None, title=None, xlim=None, ylim=None, ylim2=None, logx=False, legend=None, figsize=None, ax=None,
                       grid=None, axis_xy=None, plot_style=None, custom_style=None, output_file=None, **kwargs):
    """
    Plot the grade-tonnage curves of one variable for several models, to compare them on one plot.
    For each cutoff grade the dashed lines (left axis) are the tonnage of the blocks at or above the
    cutoff, and the solid lines (right axis) are the mean grade of those blocks. Each model has its own
    colour, so a data set, an estimate and simulated realizations can be compared, drawn in the order
    given.

    The models are either the columns of one table (``var`` is a list of column names, for example
    the realizations of a simulation), or a list of tables with the same or a different column name
    in each (``data`` is a list). Values that are NaN are left out.

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
        block_tonnes (float): tonnes in one unit of weight (for example the block volume times the
            density). If None, the tonnage is the fraction of the total weight of each model
        colors (list): one matplotlib color for each model (default: matplotlib's colour cycle)
        lw (float or list): width of the lines, one value or one for each model (default: the one of
            the plot style)
        alpha (float or list): opacity of the lines from 0 (invisible) to 1, one value or one for each
            model. Lower it to draw many models, for example the realizations of a simulation
        xlabel (str): x-axis label (default: "Cutoff Grade")
        ylabel (str): label of the tonnage axis (default: "Tonnes Above Cutoff", or
            "Fraction Above Cutoff" without ``block_tonnes``)
        title (str): title of the plot
        xlim (tuple): x-axis limits
        ylim (tuple): limits of the tonnage axis (left), ``(lower, upper)``. They replace the range that makes
            the zeros of the two axes line up, so with ``ylim`` and ``ylim2`` the zeros may not match
        ylim2 (tuple): limits of the grade axis (right), ``(lower, upper)``
        logx (bool): log scale for the cutoff axis
        legend (bool): add a legend below the plot (default: True, except for more than 10 models
            without ``labels``)
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

    |
    """
    models, labelled = _models(data, var, weights, labels)
    legend = (labelled or len(models) <= 10) if legend is None else legend
    if colors is None:
        colors = ['C%d' % i for i in range(len(models))]
    if len(colors) != len(models):
        raise ValueError("`colors` needs one entry for each of the %d models" % len(models))
    lws, alphas = _each(lw, len(models), 'lw'), _each(alpha, len(models), 'alpha')
    if ncut < 2:
        raise ValueError("`ncut` must be at least 2")
    if cutoffs is None:
        pooled = np.concatenate([m[1][np.isfinite(m[1])] for m in models])
        if pooled.size == 0:
            raise ValueError("None of the models has a value")
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
    figsize = Parameters['plotting.grade_tonnage_plot.figsize'] if figsize is None else figsize
    left, right, line_colors, line_lws, line_alphas = list(tonnages), list(grades), list(colors), list(lws), list(alphas)
    ax = _secondary_axis(
        cutoffs, left, right, colors=line_colors, colors2=line_colors, ls='--', ls2='-', lw=line_lws, alpha=line_alphas,
        xlabel=xlabel if xlabel is not None else 'Cutoff Grade',
        ylabel=ylabel if ylabel is not None else
        ('Fraction Above Cutoff' if block_tonnes is None else 'Tonnes Above Cutoff'),
        ylabel2='Mean Grade Above Cutoff', title=title, xlim=xlim, ylim=ylim, ylim2=ylim2, logx=logx,
        legend=False, figsize=figsize, ax=ax, grid=grid, axis_xy=axis_xy)
    if block_tonnes is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: '{:,.0f}'.format(v)))
    shown = [i for i, m in enumerate(models) if m[0] is not None]
    what = 'Fraction' if block_tonnes is None else 'Tonnes'
    handles = [Line2D([], [], color=colors[i], ls=ls,
                      **{k: v for k, v in (('lw', lws[i]), ('alpha', alphas[i])) if v is not None}) for ls in ('--', '-') for i in shown]
    names = ['%s - %s' % (models[i][0], w) for w in (what, 'Grade') for i in shown]
    if legend and handles:
        ax.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, -0.16),
                  ncol=min(3, len(names)), frameon=False, fontsize=ax.yaxis.get_ticklabels()[0].get_size())
    if output_file or ('pdfpages' in kwargs):
        export_image(output_file, **kwargs)
    return ax
