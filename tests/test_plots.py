#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for plotting functionality in pygeostat"""

import os
import warnings
import tempfile
import pytest
import numpy as np
import pandas as pd
import pygeostat as gs
import matplotlib.markers
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

@pytest.fixture
def plot_output_dir():
    """Create temporal output directory for plots"""
    warnings.simplefilter('ignore', category=ImportWarning)
    out_dir = tempfile.NamedTemporaryFile().name
    gs.mkdir(out_dir)
    yield out_dir
    gs.clrmplmem()
    gs.rmdir(out_dir)

@pytest.fixture
def location_plot_data():
    """Provide plot sample data"""
    return gs.ExampleData('point3d_ind_mv')

@pytest.fixture
def accuracy_plot_data():
    """Prepare data for the accuracy plots."""
    data = gs.ExampleData('accuracy_plot')
    reals = data[list(data.columns[1:])].values
    truth = data[data.columns[0]].values
    return truth, reals

@pytest.fixture
def slice_plot_data():
    """Prepare data for slice plots."""
    griddef = gs.GridDef([40, 0.5, 1, 40, 0.5, 1, 40, 0.5, 1])
    return gs.ExampleData('3d_grid', griddef = griddef)

@pytest.fixture
def histogram_plot_data():
    """Provide data for histogram plots"""
    return gs.ExampleData('point3d_ind_mv')

@pytest.fixture
def histogram_simulation_data(plot_output_dir):
    """Create simulation data for histogram testing."""
    griddef = gs.GridDef([10, 1, 0.5, 10, 1, 0.5, 1, 1, 0.5])
    size = griddef.count()
    mean = 2.5
    scale = 3
    nreal = 100
    ref_data = pd.DataFrame({'value': np.random.normal(mean, scale, size=size)})

    for i, offset in enumerate(np.random.rand(nreal) * 0.04 - 0.02):
        data = pd.DataFrame({'value': np.random.normal(mean + offset, scale, size=size)})
        gs.write_gslib(data, os.path.join(plot_output_dir, f'real{i+1}.out'))

    return griddef, nreal, ref_data


@pytest.fixture
def correlation_matrix_data():
    """Provide correlation matrix data."""
    dfl = gs.ExampleData("point3d_ind_mv")
    data = dfl[dfl.variables]
    return data.corr()


@pytest.fixture
def loadings_plot_data():
    """Provide loadings plot data."""
    refdata = gs.ExampleData('3d_correlation')
    return refdata.data.corr().iloc[3:6, 6:9]


@pytest.fixture
def drill_plot_data():
    """Provide drill plot data."""
    dat = gs.ExampleData('point3d_ind_mv')
    return dat.data[dat.data['HoleID'] == 3]


@pytest.fixture
def contour_plot_data():
    """Provide contour plot data."""
    grid_str = '''120 5.00  10.00 -nx, xmn, xsiz
                110 1205.00  10.00 -ny, ymn, ysiz
                1 0.5 1.0  -nz, zmn, zsiz'''
    griddef = gs.GridDef(grid_str=grid_str)
    return gs.ExampleData("grid2d_surf", griddef=griddef)



def test_location_plot_simple(plot_output_dir, location_plot_data):
    """Test capability of plotting by passing just the data file"""
    output_file = os.path.join(plot_output_dir, "location_plot_test.png")
    try:
        _ = gs.location_plot(location_plot_data, output_file = output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the location plot with simple settings \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_location_plot_cmap(plot_output_dir, location_plot_data):
    """Test capability of plottin by passing argument variable name to get color map"""
    output_file = os.path.join(plot_output_dir, "location_plot_cmap_test.png")
    _, ax = plt.subplots(1,1)

    try:
        _ = gs.location_plot(location_plot_data, var='Phi', output_file=output_file)
        _ = gs.location_plot(location_plot_data, var='Phi', output_file=output_file, ax=ax)

    except Exception as ex:
        pytest.fail(f'Unable to test the location plot with passing variable name\n{str(ex)}')

    assert os.path.isfile(output_file)

def test_location_plot_lines(plot_output_dir, location_plot_data):
    """Test plotting along xz/yz orientation and use the line plots based on drill hole id."""
    output_file = os.path.join(plot_output_dir, "location_plot_lines_test.png")

    try:
        _ = gs.location_plot(
        location_plot_data,
        var = 'Phi',
        orient = 'yz',
        aspect=5,
        plot_collar=True,
        output_file=output_file
    )

    except Exception as ex:
        pytest.fail(f'Unable to test the location plot along yz/xz orientation \n{str(ex)}')

    assert os.path.isfile(output_file)

def test_location_plot_categorical(plot_output_dir, location_plot_data):
    """Test plotting categorical variables along xz/yz orientation."""
    output_file = os.path.join(plot_output_dir, 'location_plot_categorical_test.png')
    _, ax = plt.subplots(1, 1)
    
    try:
        _ = gs.location_plot(
            location_plot_data, 
            var='Lithofacies', 
            orient='yz', 
            aspect=5, 
            plot_collar=True, 
            output_file=output_file
        )
        _ = gs.location_plot(
            location_plot_data, 
            var='Lithofacies', 
            orient='yz', 
            aspect=5, 
            plot_collar=True, 
            output_file=output_file, 
            ax=ax
        )
    except Exception as ex:
        pytest.fail(f'Unable to test the location plot for a categorical variable \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_accuracy_plot(plot_output_dir, accuracy_plot_data):
    """Test the capability of plotting the accuracy with no function parameters"""
    truth, reals = accuracy_plot_data
    output_file = os.path.join(plot_output_dir, 'accuracy_plot_test.png')
    _, ax = plt.subplots(1,1)

    try:
        gs.accuracy_plot(truth=truth, reals=reals, output_file=output_file)
        gs.accuracy_plot(truth=truth, reals=reals, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the accuracy plot with simple settings \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_accuracy_plot_mik(plot_output_dir, accuracy_plot_data):
    """Test the accuracy plot from MIK probabilities at thresholds"""
    truth, reals = accuracy_plot_data
    thresholds = np.quantile(reals, np.linspace(0.05, 0.95, 19))
    mikprobs = (reals[:, :, None] <= thresholds).mean(axis=1)
    output_file = os.path.join(plot_output_dir, 'accuracy_plot_mik_test.png')

    gs.accuracy_plot(truth=truth, reals=mikprobs, mik_thresholds=thresholds,
                     output_file=output_file)

    assert os.path.isfile(output_file)


def test_accuracy_plot_compare(accuracy_plot_data):
    """Test overlaying two models on one axis with labels"""
    truth, reals = accuracy_plot_data
    _, ax = plt.subplots(1, 1)

    gs.accuracy_plot(truth=truth, reals=reals, label='all', ax=ax)
    gs.accuracy_plot(truth=truth, reals=reals[:, :10], label='ten', color='C1',
                     stat_xy=(0.95, 0.4), ax=ax)

    assert ax.get_legend_handles_labels()[1] == ['all', 'ten']
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lines) == 2
    # the second model was given a colour, so all its lines have it
    assert np.allclose(lines[1].get_color()[:, :3], to_rgb('C1'))


def accuracy_plot_colours(truth, reals):
    """Marker colours, line colours and the (x, y) of each point of a default accuracy plot."""
    _, ax = plt.subplots(1, 1)
    gs.accuracy_plot(truth=truth, reals=reals, ax=ax)
    points = [c for c in ax.collections if not isinstance(c, LineCollection)][0]
    lines = [c for c in ax.collections if isinstance(c, LineCollection)][0]
    x, y = points.get_offsets().T
    return points.get_facecolors()[:, :3], lines.get_color()[:, :3], x, y, ax


def test_accuracy_plot_green_red(accuracy_plot_data):
    """By default a point and its line are green at or above the 1:1 line and red below it."""
    truth, reals = accuracy_plot_data
    points, lines, x, y, _ = accuracy_plot_colours(truth, reals)
    expected = np.array([to_rgb('tab:green') if above else to_rgb('tab:red') for above in y >= x])
    assert np.allclose(points, expected)
    assert np.allclose(lines, expected)


def test_accuracy_plot_red_when_below():
    """Truth far from the realizations puts every point below the 1:1 line, so all are red."""
    rng = np.random.default_rng(0)
    reals = rng.normal(0, 1, (200, 50))
    truth = rng.normal(8, 1, 200)
    points, lines, x, y, _ = accuracy_plot_colours(truth, reals)
    assert np.all(y < x)
    assert np.allclose(points, to_rgb('tab:red'))
    assert np.allclose(lines, to_rgb('tab:red'))


def test_accuracy_plot_grid(accuracy_plot_data):
    """A light grey grid is drawn behind the points by default and can be turned off."""
    truth, reals = accuracy_plot_data
    ax = gs.accuracy_plot(truth=truth, reals=reals)
    line = ax.xaxis.get_gridlines()[0]
    assert line.get_visible() and np.allclose(to_rgb(line.get_color()), to_rgb('0.7'))
    assert ax.get_axisbelow() is True
    ax_off = gs.accuracy_plot(truth=truth, reals=reals, grid=False)
    assert not ax_off.xaxis.get_gridlines()[0].get_visible()


def test_slice_plot_simple(plot_output_dir, slice_plot_data):
    """Test the capability of plotting by passing only the data file"""
    output_file = os.path.join(plot_output_dir, 'slice_plot_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        gs.slice_plot(slice_plot_data, griddef=slice_plot_data.griddef, 
                     orient='xy', cmap='viridis', output_file=output_file)
        gs.slice_plot(slice_plot_data, griddef=slice_plot_data.griddef, 
                     orient='xy', cmap='viridis', output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the slice plot with simple settings \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_slice_plot_with_thickness(plot_output_dir, slice_plot_data):
    """Test the capability of plotting with slice thickness parameter"""
    output_file = os.path.join(plot_output_dir, 'slice_plot_thickness_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        gs.slice_plot(slice_plot_data, orient='xy', cmap='viridis', 
                     output_file=output_file, slice_thickness=20)
        gs.slice_plot(slice_plot_data, orient='xy', cmap='viridis', 
                     output_file=output_file, slice_thickness=20, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the slice plot with thickness \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_grid_slice_plot_simple(plot_output_dir, slice_plot_data):
    """Test the capability of grid slice plot with simple parameters"""
    output_file = os.path.join(plot_output_dir, 'grid_slice_plot_test.png')
    
    try:
        _ = gs.grid_slice_plot(slice_plot_data, output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the grid slice plot with simple settings \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_grid_slice_plot_orient(plot_output_dir, slice_plot_data):
    """Test the capability of specifying number of slices and orientation"""
    output_file = os.path.join(plot_output_dir, 'grid_slice_plot_orient_test.png')
    
    try:
        _ = gs.grid_slice_plot(slice_plot_data, orient='xz', n_slice=5, output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the grid slice plot with orientation \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_grid_slice_plot_rows_cols(plot_output_dir, slice_plot_data):
    """Test the capability of specifying number of rows or columns for slices"""
    output_file = os.path.join(plot_output_dir, 'grid_slice_plot_rows_cols_test.png')
    
    try:
        _ = gs.grid_slice_plot(slice_plot_data, orient='xz', n_slice=6, 
                              ncol=2, nrow=3, output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the grid slice plot with rows/columns \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_grid_slice_plot_custom(plot_output_dir, slice_plot_data):
    """Test the ability to specify slice_plot kwargs for consistent formatting"""
    output_file = os.path.join(plot_output_dir, 'grid_slice_plot_custom_test.png')
    
    try:
        _ = gs.grid_slice_plot(slice_plot_data, nrow=2, ncol=5, start_slice=10, 
                              end_slice=25, n_slice=10, cmap='hot', vlim=(-3, 3), 
                              output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the grid slice plot with custom settings \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_histogram_plot_simple(plot_output_dir, histogram_plot_data):
    """Test the capability of plotting a simple histogram"""
    output_file = os.path.join(plot_output_dir, 'histogram_plot_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        gs.histogram_plot(histogram_plot_data, var="Phi", bins=30, output_file=output_file)
        gs.histogram_plot(histogram_plot_data, var="Phi", bins=30, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the histogram plot with simple settings \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_histogram_plot_density_color(plot_output_dir, histogram_plot_data):
    """Test the capability of plotting a histogram with density and custom colour"""
    output_file = os.path.join(plot_output_dir, 'histogram_plot_density_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        gs.histogram_plot(histogram_plot_data, var="Phi", bins=30, color='#c2e1e5', 
                         sigfigs=5, log=True, density=True, output_file=output_file)
        gs.histogram_plot(histogram_plot_data, var="Phi", bins=30, color='#c2e1e5', 
                         sigfigs=5, log=True, density=True, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the histogram plot with density and color \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_histogram_plot_icdf(plot_output_dir, histogram_plot_data):
    """Test the capability of plotting a histogram with icdf and custom stat location"""
    output_file = os.path.join(plot_output_dir, 'histogram_plot_icdf_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        gs.histogram_plot(histogram_plot_data, var="Phi", icdf=True, color=3, 
                         lw=3.5, stat_xy=(1, 0.75), output_file=output_file)
        gs.histogram_plot(histogram_plot_data, var="Phi", icdf=True, color=3, 
                         lw=3.5, stat_xy=(1, 0.75), output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the histogram plot for icdf \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_histogram_plot_categories(plot_output_dir, histogram_plot_data):
    """Test the capability of plotting a histogram with categorical variables"""
    output_file = os.path.join(plot_output_dir, 'histogram_plot_categories_test.png')

    try:
        cats = [1, 2, 3, 4, 5]
        colors = gs.catcmapfromcontinuous("Spectral", 5).colors
        histogram_plot_data.catdict = {c: f"RT {c:02d}" for c in cats}
        colordict = {c: colors[i] for i, c in enumerate(cats)}

        gs.histogram_plot(histogram_plot_data, var='Phi', cat=True, color=colordict, 
                         bins=40, figsize=(8, 4), xlabel=False, output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the histogram plot with categories \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_histogram_plot_cdf_categories(plot_output_dir, histogram_plot_data):
    """Test the capability of plotting cdfs with different categorical variables"""
    output_file = os.path.join(plot_output_dir, 'histogram_plot_cdf_categories_test.png')

    try:
        cats = [1, 2, 3, 4, 5]
        colors = gs.catcmapfromcontinuous("Spectral", 5).colors
        histogram_plot_data.catdict = {c: f"RT {c:02d}" for c in cats}
        colordict = {c: colors[i] for i, c in enumerate(cats)}

        _, axs = plt.subplots(2, 2, figsize=(12, 9))
        axs = axs.flatten()

        for var, ax in zip(histogram_plot_data.variables, axs):
            gs.histogram_plot(histogram_plot_data, var=var, icdf=True, cat=True, 
                             color=colordict, ax=ax, output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the histogram plot for cdf categories \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_histogram_plot_proportions(plot_output_dir, histogram_plot_data):
    """Test the capability of plotting histogram for categorical proportions"""
    output_file = os.path.join(plot_output_dir, 'histogram_plot_proportions_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        cats = [1, 2, 3, 4, 5]
        colors = gs.catcmapfromcontinuous("Spectral", 5).colors
        histogram_plot_data.catdict = {c: f"RT {c:02d}" for c in cats}
        colordict = {c: colors[i] for i, c in enumerate(cats)}

        gs.histogram_plot(histogram_plot_data, cat=True, color=colordict, bins=40, 
                         figsize=(8, 4), xlabel=False, output_file=output_file)
        gs.histogram_plot(histogram_plot_data, cat=True, color=colordict, bins=40, 
                         figsize=(8, 4), xlabel=False, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the histogram plot for proportions\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_histogram_reproduction_plot(plot_output_dir, histogram_simulation_data):
    """Test the capability of plotting histogram reproduction"""
    griddef, nreal, ref_data = histogram_simulation_data
    output_file = os.path.join(plot_output_dir, 'histplt_sim_test.png')

    try:
        gs.histogram_plot_simulation(
            os.path.join(plot_output_dir, 'real*.out'), 
            ref_data, 
            reference_variable='value',
            reference_color='C3', 
            simulated_column=1, 
            griddef=griddef, 
            simulation_color='cyan',
            title='var', 
            grid=True, 
            nreal=nreal, 
            output_file=output_file
        )
    except Exception as ex:
        pytest.fail(f'Unable to test the histogram reproduction plot \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_correlation_matrix_plot_simple(plot_output_dir, correlation_matrix_data):
    """Test the capability of plotting a simple correlation matrix"""
    output_file = os.path.join(plot_output_dir, 'correlation_matrix_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        gs.correlation_matrix_plot(correlation_matrix_data, output_file=output_file)
        gs.correlation_matrix_plot(correlation_matrix_data, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the correlation matrix plot \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_correlation_matrix_plot_lower(plot_output_dir, correlation_matrix_data):
    """Test the capability of plotting lower part of a correlation matrix"""
    output_file = os.path.join(plot_output_dir, 'correlation_matrix_lower_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        gs.correlation_matrix_plot(correlation_matrix_data, lower_matrix=True, 
                                  annotation=True, output_file=output_file)
        gs.correlation_matrix_plot(correlation_matrix_data, lower_matrix=True, 
                                  annotation=True, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the correlation matrix plot with lower matrix\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_scatter_plot_simple(plot_output_dir, histogram_plot_data):
    """Test the capability of plotting a scatter plot with probability density"""
    output_file = os.path.join(plot_output_dir, 'scatter_plot_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        gs.scatter_plot(histogram_plot_data[histogram_plot_data.variables[0]], 
                       histogram_plot_data[histogram_plot_data.variables[1]], 
                       cmap='hot', cbar=True, output_file=output_file)
        gs.scatter_plot(histogram_plot_data[histogram_plot_data.variables[0]], 
                       histogram_plot_data[histogram_plot_data.variables[1]], 
                       cmap='hot', cbar=True, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the scatter plot \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_scatter_plots_multivariate(plot_output_dir, histogram_plot_data):
    """Test scatter plots with probability density for multivariate data"""
    output_file = os.path.join(plot_output_dir, 'scatter_plots_multivariate_test.png')

    try:
        _ = gs.scatter_plots(histogram_plot_data, nmax=1000, stat_xy=(0.95, 0.95), 
                            pad=(-1, -1), s=10, figsize=(10, 10), output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the scatter plots \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_scatter_plots_lu(plot_output_dir, histogram_plot_data):
    """Test scatter plots with upper/lower matrix for comparing datasets"""
    output_file = os.path.join(plot_output_dir, 'scatter_plots_lu_test.png')

    try:
        _ = gs.scatter_plots_lu(
            histogram_plot_data, 
            histogram_plot_data, 
            titles=('Data', 'Realization'), 
            s=10, 
            nmax=1000,
            stat_xy=(0.95, 0.95), 
            pad=(-1, -1), 
            figsize=(10, 10),
            align_orient=True, 
            output_file=output_file
        )
    except Exception as ex:
        pytest.fail(f'Unable to test the scatter plots with upper/lower matrix \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_qq_plot(plot_output_dir):
    """Test the capability of plotting a simple quantile-quantile comparison"""
    output_file = os.path.join(plot_output_dir, 'qq_plot_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        _ = gs.qq_plot(np.random.randn(1000), np.random.randn(1000), 
                      npoints=500, output_file=output_file)
        _ = gs.qq_plot(np.random.randn(1000), np.random.randn(1000), 
                      npoints=500, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the qq_plot \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_probability_plot(plot_output_dir):
    """Test the capability of plotting a CDF with probability scale"""
    output_file = os.path.join(plot_output_dir, 'probability_plot_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        data = gs.ExampleData('oilsands')
        _ = gs.probability_plot(data.data['Fines'], logscale=False, output_file=output_file)
        _ = gs.probability_plot(data.data['Fines'], logscale=False, output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the probability_plot \n{str(ex)}')

    assert os.path.isfile(output_file)


def test_validation_plot_simple(plot_output_dir):
    """Test plotting a simple scatter plot to compare estimate vs truth"""
    output_file = os.path.join(plot_output_dir, 'validation_plot_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        data = gs.ExampleData('3d_estimate')
        _ = gs.validation_plot(data.data['Estimate'], data.data['True'], 
                              stat_blk='minimal', output_file=output_file)
        _ = gs.validation_plot(data.data['Estimate'], data.data['True'], 
                              stat_blk='minimal', output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the validation_plot with simple settings\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_validation_plot_detailed(plot_output_dir):
    """Test plotting a detailed validation plot with statistics"""
    output_file = os.path.join(plot_output_dir, 'validation_plot_detailed_test.png')
    _, ax = plt.subplots(1, 1)

    try:
        mean = [0, 0]
        cov = [[1, 0.95], [0.95, 1]]
        x, y = np.random.multivariate_normal(mean, cov, 5000).T
        
        gs.validation_plot(x, y, vlim=(-3.5, 3.5), grid=True, 
                          stat_xy=(1, 0.68), output_file=output_file)
        gs.validation_plot(x, y, vlim=(-3.5, 3.5), grid=True, 
                          stat_xy=(1, 0.68), output_file=output_file, ax=ax)
    except Exception as ex:
        pytest.fail(f'Unable to test the validation_plot with detailed settings\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_loadings_plot_dataframe(plot_output_dir, loadings_plot_data):
    """Test plotting a loadings plot using pandas dataframe"""
    output_file = os.path.join(plot_output_dir, 'loadings_plot_dataframe_test.png')

    try:
        _ = gs.loadings_plot(loadings_plot_data, output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the loadings_plot with pandas dataframe\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_loadings_plot_array(plot_output_dir, loadings_plot_data):
    """Test plotting a loadings plot using numpy array"""
    output_file = os.path.join(plot_output_dir, 'loadings_plot_array_test.png')

    try:
        _ = gs.loadings_plot(loadings_plot_data.values, output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the loadings_plot with numpy array\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_loadings_plot_labels(plot_output_dir, loadings_plot_data):
    """Test plotting a loadings plot with custom labels"""
    output_file = os.path.join(plot_output_dir, 'loadings_plot_labels_test.png')

    try:
        _ = gs.loadings_plot(
            loadings_plot_data.values, 
            figsize=(4, 4), 
            xticklabels=['PC1', 'PC2', 'PC3'], 
            yticklabels=['InputVariable1', 'InputVariable2', 'InputVariable3'],
            output_file=output_file
        )
    except Exception as ex:
        pytest.fail(f'Unable to test the loadings_plot with custom labels\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_loadings_plot_tick_rotation(plot_output_dir, loadings_plot_data):
    """Test plotting a loadings plot with tick rotation"""
    output_file = os.path.join(plot_output_dir, 'loadings_plot_rotation_test.png')

    try:
        _ = gs.loadings_plot(
            loadings_plot_data.values, 
            figsize=(5, 5), 
            xticklabels=['PC1', 'PC2', 'PC3'],
            yticklabels=['InputVariable1', 'InputVariable2', 'InputVariable3'],
            rotateticks=(False, True), 
            output_file=output_file
        )
    except Exception as ex:
        pytest.fail(f'Unable to test the loadings_plot with tick rotation\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_drill_plot_continuous(plot_output_dir, drill_plot_data):
    """Test plotting a continuous variable in a drill plot"""
    output_file = os.path.join(plot_output_dir, 'drill_plot_continuous_test.png')

    try:
        _ = gs.drill_plot(
            drill_plot_data['Elevation'], 
            drill_plot_data['Sw'], 
            grid=True, 
            output_file=output_file
        )
    except Exception as ex:
        pytest.fail(f'Unable to test the drill_plot for continuous variable\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_drill_plot_categorical(plot_output_dir, drill_plot_data):
    """Test plotting a categorical variable in a drill plot"""
    output_file = os.path.join(plot_output_dir, 'drill_plot_categorical_test.png')

    try:
        _ = gs.drill_plot(
            drill_plot_data['Elevation'], 
            drill_plot_data['Lithofacies'], 
            output_file=output_file
        )
    except Exception as ex:
        pytest.fail(f'Unable to test the drill_plot for categorical variable\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_drill_plot_categorical_custom(plot_output_dir, drill_plot_data):
    """Test plotting a categorical variable with custom dictionary"""
    output_file = os.path.join(plot_output_dir, 'drill_plot_categorical_custom_test.png')

    try:
        cat_dict = {
            1: {'name': 'Sand', 'color': 'gold'},
            3: {'name': 'SHIS', 'color': 'orange'},
            4: {'name': 'MHIS', 'color': 'green'},
            5: {'name': 'MUD', 'color': 'gray'}
        }

        _ = gs.drill_plot(
            drill_plot_data['Elevation'],
            drill_plot_data['Lithofacies'],
            categorical_dictionary=cat_dict,
            output_file=output_file
        )
    except Exception as ex:
        pytest.fail(f'Unable to test the drill_plot with custom categorical dictionary\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_variogram_plot_experimental(plot_output_dir):
    """Test plotting experimental variograms"""
    output_file = os.path.join(plot_output_dir, 'variogram_plot_experimental_test.png')

    try:
        dfl = gs.ExampleData('experimental_variogram')
        _ = gs.variogram_plot(dfl, experimental=True, figsize=(8, 4), output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the variogram_plot for experimental variogram\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_variogram_plot_model(plot_output_dir):
    """Test plotting variogram models"""
    output_file = os.path.join(plot_output_dir, 'variogram_plot_model_test.png')

    try:
        dfl = gs.ExampleData('variogram_model')
        _ = gs.variogram_plot(dfl, experimental=False, lw=3, figsize=(8, 4), output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the variogram_plot for model\n{str(ex)}')

    assert os.path.isfile(output_file)


def test_contour_plot_simple(plot_output_dir, contour_plot_data):
    """Test plotting contours with essential parameters"""
    output_file = os.path.join(plot_output_dir, 'contour_plot_test.png')

    try:
        _ = gs.contour_plot(contour_plot_data, var="Thickness", clabel=True, output_file=output_file)
    except Exception as ex:
        pytest.fail(f'Unable to test the contour plot with simple settings\n{str(ex)}')

    assert os.path.isfile(output_file)


@pytest.mark.skip(reason="Requires additional setup for gmm_file path")
def test_gmm_plotting():
    """Test GMM plotting utilities"""
    # This test depends on locating the gmm_file correctly
    # We'll skip it in this conversion but keep the structure
    
    try:
        # Path needs to be adjusted for your environment
        gmm_file = os.path.join(os.path.dirname(os.getcwd()), 
                    'pygeostat', 'src', 'pygeostat', 'data', 'example_data', 'gmm_fit.out')
        gmm_file = os.path.normpath(gmm_file)
        
        gmm_util = gs.GmmUtility(
            gmm_file=gmm_file,
            data=gs.ExampleData('point2d_mv').data, 
            variable_names=['Var1', 'Var2', 'Var3']
        )
        
        # Test bivariate plot
        gmm_util.bivariate_plot(var_index=[1, 2], cmap='viridis', title='Bivariate Plot')
        
        # Test summary plot
        gmm_util.summary_plot(pad=0.1)
        
        # Test conditional plot
        gmm_util.univariate_conditional_plot(conditioning_data=[0, 0, None])
        
    except Exception as ex:
        pytest.fail(f'Unable to test GMM plotting: {str(ex)}')

def _tick_length(ax):
    ax.figure.canvas.draw()
    return ax.xaxis.get_major_ticks()[0].tick1line.get_markersize()


def _plain_axes():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('t')
    return ax


@pytest.mark.parametrize('ticks, length', [(True, 4), (False, 0), (7, 7), ({'length': 9}, 9),
                                           (np.True_, 4), (np.False_, 0), (np.float64(5), 5)])
def test_format_axes_ticks(ticks, length):
    """True draws 4 point marks, False none, a number sets the length, a dict goes to tick_params."""
    from pygeostat.plotting.utils import _format_axes
    ax = _plain_axes()
    _format_axes(ax, ticks=ticks)
    assert _tick_length(ax) == length
    plt.close(ax.figure)


def test_format_axes_ticks_refuses_text():
    """A string such as '4' is not a tick length."""
    from pygeostat.plotting.utils import _format_axes
    ax = _plain_axes()
    with pytest.raises(TypeError, match='ticks must be'):
        _format_axes(ax, ticks='4')
    plt.close(ax.figure)


def test_format_axes_ticks_default_follows_parameter():
    """ticks=None uses Parameters['plotting.ticks'], so the default can be changed once for all plots."""
    from pygeostat.plotting.utils import _format_axes
    saved = gs.Parameters['plotting.ticks']
    try:
        assert saved is True
        ax = _plain_axes()
        _format_axes(ax)
        assert _tick_length(ax) == 4
        gs.Parameters['plotting.ticks'] = False
        ax2 = _plain_axes()
        _format_axes(ax2)
        assert _tick_length(ax2) == 0
        _format_axes(ax2, ticks=True)
        assert _tick_length(ax2) == 4
    finally:
        gs.Parameters['plotting.ticks'] = saved
        plt.close('all')


def test_format_axes_fontsize():
    """One number sets labels, ticks and title; a dict sets them one by one and keeps the rest."""
    from pygeostat.plotting.utils import _format_axes
    ax = _plain_axes()
    _format_axes(ax, fontsize=20)
    assert (ax.xaxis.label.get_fontsize(), ax.yaxis.label.get_fontsize(), ax.title.get_fontsize()) == (20, 20, 20)
    assert ax.get_xticklabels()[0].get_fontsize() == 20
    ax = _plain_axes()
    title_before = ax.title.get_fontsize()
    _format_axes(ax, fontsize={'label': 15, 'tick': 9})
    assert ax.xaxis.label.get_fontsize() == 15 and ax.get_yticklabels()[0].get_fontsize() == 9
    assert ax.title.get_fontsize() == title_before
    with pytest.raises(ValueError, match="'label', 'tick' or 'title'"):
        _format_axes(ax, fontsize={'size': 3})
    for bad in (0, -3, '12', True, {'label': 0}):
        with pytest.raises(ValueError, match='number above zero'):
            _format_axes(ax, fontsize=bad)
    plt.close('all')


def test_format_axes_limits_grid_and_background():
    """Limits, grid and background are set only when given; a limit that is not a pair is refused."""
    from pygeostat.plotting.utils import _format_axes
    ax = _plain_axes()
    data_xlim = ax.get_xlim()
    _format_axes(ax, ylim=(-1, 10), grid=True, background='lightgrey')
    assert ax.get_ylim() == (-1, 10)
    assert ax.get_xlim() == data_xlim
    assert ax.xaxis.get_gridlines()[0].get_visible()
    assert ax.get_facecolor() == (0.8274509803921568, 0.8274509803921568, 0.8274509803921568, 1.0)
    with pytest.raises(ValueError, match='xlim must be'):
        _format_axes(ax, xlim=(1, 2, 3))
    plt.close('all')


def test_histogram_plot_ticks_fontsize_background(histogram_plot_data):
    """ticks, fontsize and background reach the histogram; stat_fontsize still sizes the statistics."""
    plt.close('all')
    ax = gs.histogram_plot(histogram_plot_data, var='Phi', bins=10, ticks=7,
                           fontsize={'label': 15, 'tick': 8, 'title': 11}, title='t', background='lightgrey',
                           stat_fontsize=13)
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 7
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_markersize() == 7
    assert (ax.xaxis.label.get_fontsize(), ax.yaxis.label.get_fontsize(), ax.title.get_fontsize()) == (15, 15, 11)
    assert ax.get_xticklabels()[0].get_fontsize() == 8
    assert tuple(ax.get_facecolor()[:3]) == (0.8274509803921568,) * 3
    assert {t.get_fontsize() for t in ax.texts} == {13}  # fontsize leaves the statistics alone
    plt.close('all')


def test_histogram_plot_ticks_default_and_off(histogram_plot_data):
    """Tick marks are drawn by default (Parameters['plotting.ticks']) and ticks=False removes them."""
    plt.close('all')
    ax = gs.histogram_plot(histogram_plot_data, var='Phi', bins=10)
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 4
    ax = gs.histogram_plot(histogram_plot_data, var='Phi', bins=10, ticks=False)
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 0
    plt.close('all')


def test_histogram_plot_limits_and_grid(histogram_plot_data):
    """The limits come from the data unless xlim and ylim are given; grid turns the grid lines on or off."""
    plt.close('all')
    ax = gs.histogram_plot(histogram_plot_data, var='Phi', bins=10, xlim=(0.0, 0.5), ylim=(0, 50), grid=True)
    assert ax.get_xlim() == (0.0, 0.5) and ax.get_ylim() == (0, 50)
    assert ax.xaxis.get_gridlines()[0].get_visible()
    data_lim = gs.histogram_plot(histogram_plot_data, var='Phi', bins=10, grid=False).get_xlim()
    assert data_lim[0] <= histogram_plot_data.data['Phi'].min() and data_lim[1] >= histogram_plot_data.data['Phi'].max()
    plt.close('all')

def test_validation_plot_outliers():
    est = pd.Series([0., 1., 2., 3., 4.], index=[10, 11, 12, 13, 14])
    true = pd.Series([0., 1., 2.5, 0., 4.], index=est.index)  # errors 0, 0, 0.5, 3, 0
    ax = gs.validation_plot(est, true, outliers=2, stat_blk=False)
    offsets = {t.get_text(): t.xyann[1] for t in ax.texts}
    assert sorted(offsets) == ['12', '13']          # the two largest |y - x|
    assert offsets['12'] > 0 and offsets['13'] < 0  # 12 is above the line, 13 below
    red = [c for c in ax.collections if len(c.get_offsets()) == 1]
    assert len(red) == 2 and all(tuple(c.get_facecolor()[0][:3]) == (1.0, 0.0, 0.0) for c in red)
    assert not gs.validation_plot(est, true, stat_blk=False).texts  # off by default
    ax = gs.validation_plot(est.values, true.values, outliers=1, stat_blk=False)  # arrays: label by position
    assert [t.get_text() for t in ax.texts] == ['3']
    plt.close('all')


def test_validation_plot_options_for_the_highlighted_points_and_the_regression_line():
    """Highlighted points can have no numbers, the regression line can be left out, and the statistics can go."""
    est = pd.Series([0., 1., 2., 3., 4.])
    true = pd.Series([0., 1., 2.5, 0., 4.])
    ax = gs.validation_plot(est, true, outliers=2, outlier_labels=False, stat_blk=False)
    assert not ax.texts
    assert len([c for c in ax.collections if len(c.get_offsets()) == 1]) == 2  # the red points stay
    red_lines = lambda a: [l for l in a.get_lines() if l.get_color() == 'red']
    assert len(red_lines(ax)) == 1
    plt.close('all')
    ax = gs.validation_plot(est, true, reg_line=False, stat_blk=False)
    assert not red_lines(ax) and len(ax.get_lines()) == 1  # only the 45 degree line
    plt.close('all')


def test_validation_plot_axes_options():
    est = pd.Series([0., 1., 2., 3., 4.])
    true = pd.Series([0., 1., 2.5, 0., 4.])
    ax = gs.validation_plot(est, true, stat_blk=False, ticks=6, fontsize={'label': 16, 'tick': 9},
                            background='lightgrey')
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 6
    assert ax.xaxis.label.get_fontsize() == 16 and ax.get_yticklabels()[0].get_fontsize() == 9
    assert tuple(ax.get_facecolor()[:3]) == (0.8274509803921568,) * 3
    plt.close('all')

def _varsim_data():
    lags = np.arange(1., 11.)
    sim = pd.concat([pd.DataFrame({'Variogram Index': i, 'Variogram Number': 1, 'Lag Distance': lags,
                                   'Variogram Value': 1 - np.exp(-lags / 4) + 0.02 * i, 'Number of Pairs': 100,
                                   'Calculation Azimuth': 0., 'Calculation Dip': 0.}) for i in (1, 2, 3)],
                    ignore_index=True)
    return sim, pd.DataFrame({'Lag Distance': lags, 'Variogram Value': 1 - np.exp(-lags / 4)})


def test_variogram_plot_simulation_options():
    """ticks, fontsize, grid and background reach the variogram reproduction plot; limits were already there."""
    plt.close('all')
    sim, ref = _varsim_data()
    ax = gs.variogram_plot_simulation(sim, 1, reference_data=ref, ticks=7, fontsize={'label': 15, 'tick': 9},
                                      background='lightgrey', grid=True, xlim=(0, 8), ylim=(0, 1.5))
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 7
    assert ax.xaxis.label.get_fontsize() == 15 and ax.get_xticklabels()[0].get_fontsize() == 9
    assert tuple(ax.get_facecolor()[:3]) == (0.8274509803921568,) * 3
    assert ax.xaxis.get_gridlines()[0].get_visible()
    assert ax.get_xlim() == (0, 8) and ax.get_ylim() == (0, 1.5)
    plt.close('all')
    ax = gs.variogram_plot_simulation(sim, 1, reference_data=ref, ticks=False, grid=False)
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 0
    assert not ax.xaxis.get_gridlines()[0].get_visible()
    plt.close('all')


def test_histogram_plot_simulation_options():
    """ticks, fontsize, grid, ylim and background reach the histogram reproduction plot; fontsize leaves the
    statistics alone."""
    plt.close('all')
    sims = np.random.default_rng(0).normal(size=(50, 4))
    ref = np.random.default_rng(1).normal(size=20)
    grid = gs.GridDef([5, 0, 1, 5, 0, 1, 2, 0, 1])
    ax = gs.histogram_plot_simulation(sims, ref, griddef=grid, nreal=4, ticks=6, fontsize={'label': 14, 'tick': 8},
                                      grid=True, ylim=(0, 0.5), xlim=(-2, 2), background='lightgrey')
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 6
    assert ax.xaxis.label.get_fontsize() == 14 and ax.get_xticklabels()[0].get_fontsize() == 8
    assert ax.get_xlim() == (-2, 2) and ax.get_ylim() == (0, 0.5)
    assert ax.xaxis.get_gridlines()[0].get_visible()
    assert tuple(ax.get_facecolor()[:3]) == (0.8274509803921568,) * 3
    default_size = gs.histogram_plot_simulation(sims, ref, griddef=grid, nreal=4).texts[0].get_fontsize()
    assert ax.texts[0].get_fontsize() == default_size  # the statistics keep their own size
    plt.close('all')


def test_histogram_plot_simulation_counts_and_label():
    sims = np.random.default_rng(0).normal(size=(50, 4))
    sims[:5, 0] = np.nan  # one realization has 5 missing values: (45 + 3 x 50) / 4 = 48.75 values per realization
    ref = np.random.default_rng(1).normal(size=20)
    grid = gs.GridDef([5, 0, 1, 5, 0, 1, 2, 0, 1])
    ax = gs.histogram_plot_simulation(sims, ref, griddef=grid, nreal=4)
    assert ax.get_xlabel() == 'Value'  # no name in plain arrays
    assert 'n/real = 49$' in ax.texts[0].get_text()  # 48.75, shown without decimals
    plt.close('all')


def test_reproduction_plots_draw_the_realizations_in_the_same_light_grey():
    """The defaults of both reproduction plots are grey 0.5 with opacity 0.3, and the plot uses the parameter."""
    from pygeostat.pygeostat_parameters import default_parameters
    for name in ('histogram_plot_simulation', 'variogram_plot_simulation'):
        assert default_parameters['plotting.%s.simclr' % name][0] == '0.5'
        assert default_parameters['plotting.%s.alpha' % name][0] == '0.3'
    plt.close('all')
    sim, ref = _varsim_data()
    ax = gs.variogram_plot_simulation(sim, 1, reference_data=ref)
    realization = ax.get_lines()[0]
    assert realization.get_color() == gs.Parameters['plotting.variogram_plot_simulation.simclr']
    assert realization.get_alpha() == gs.Parameters['plotting.variogram_plot_simulation.alpha']
    plt.close('all')


def test_accuracy_plot_shows_a_p_and_g(accuracy_plot_data):
    """The statistics block has A, P and G in percent, as accsim gives them; P and G are n/a without
    Gaussian realizations or the data of the transform, and MIK shows n/a too."""
    truth, reals = accuracy_plot_data
    ax = gs.accuracy_plot(truth=truth, reals=reals)
    text = ax.texts[0].get_text()
    assert '$P_{26} = n/a$' in text and '$G_{26} = n/a$' in text and '$A_{26} = ' in text
    _, stats = gs.accsim(truth, reals, data=truth)
    ax = gs.accuracy_plot(truth=truth, reals=reals, data=truth)
    text = ax.texts[0].get_text()
    assert 'n/a' not in text
    assert '$A_{{26}} = {:0.1f}\\%$'.format(stats['acc']) in text
    assert '$P_{{26}} = {:0.1f}\\%$'.format(stats['pre']) in text
    assert '$G_{{26}} = {:0.1f}\\%$'.format(stats['goo']) in text
    thresholds = np.quantile(reals, np.linspace(0.05, 0.95, 19))
    mikprobs = (reals[:, :, None] <= thresholds).mean(axis=1)
    ax = gs.accuracy_plot(truth=truth, reals=mikprobs, mik_thresholds=thresholds)
    assert '$P_{26} = n/a$' in ax.texts[0].get_text()
    plt.close('all')


def test_accuracy_plot_axis_labels_follow_the_paper(accuracy_plot_data):
    """The axes are titled as in Clayton Deutsch's paper, and the labels can still be set."""
    truth, reals = accuracy_plot_data
    ax = gs.accuracy_plot(truth=truth, reals=reals)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('Predicted Probability', 'True Fraction')
    ax = gs.accuracy_plot(truth=truth, reals=reals, xlabel='p', ylabel='t')
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('p', 't')
    plt.close('all')


def test_validation_plot_shows_the_mean_squared_error_not_its_square_root():
    """The statistics block has MSE (the mean of (x - y)^2), where it had the RMSE."""
    plt.close('all')
    est = pd.Series([0., 1., 2., 3.])
    true = pd.Series([0., 1., 2., 7.])  # errors 0, 0, 0, 4: MSE 4.0, RMSE 2.0
    ax = gs.validation_plot(est, true, stat_blk=['mse'])
    assert ax.texts[0].get_text() == '$MSE = 4.000$'
    plt.close('all')


def test_histogram_plot_simulation_line_width_ylabel_and_stat_fontsize():
    """lw_real, ylabel and stat_fontsize replace the values that were fixed in the code; the defaults are unchanged."""
    plt.close('all')
    sims = np.random.default_rng(0).normal(size=(50, 4))
    ref = np.random.default_rng(1).normal(size=20)
    grid = gs.GridDef([5, 0, 1, 5, 0, 1, 2, 0, 1])
    ax = gs.histogram_plot_simulation(sims, ref, griddef=grid, nreal=4, lw=2)
    widths = sorted(set(line.get_linewidth() for line in ax.lines))
    assert widths == [1.0, 2.0]  # realizations at half of lw, the reference at lw
    assert ax.get_ylabel() == 'Cumulative Frequency'
    default_size = ax.texts[0].get_fontsize()
    plt.close('all')
    ax = gs.histogram_plot_simulation(sims, ref, griddef=grid, nreal=4, lw=2, lw_real=0.25, ylabel='Probability',
                                      stat_fontsize=default_size + 4)
    widths = sorted(set(line.get_linewidth() for line in ax.lines))
    assert widths == [0.25, 2.0]
    assert ax.get_ylabel() == 'Probability'
    assert ax.texts[0].get_fontsize() == default_size + 4
    plt.close('all')


def test_variogram_plot_simulation_line_style_sill_and_realization_label():
    """ls reaches the variograms (it was accepted but ignored); sill_color, sill_lw and real_label replace the values
    that were fixed in the code, and the defaults are unchanged."""
    plt.close('all')
    sim, ref = _varsim_data()
    ax = gs.variogram_plot_simulation(sim, 1, reference_data=ref, legend_label='Reference')
    sill = [line for line in ax.lines if list(line.get_ydata()) == [1, 1]][0]
    assert sill.get_color() == 'k' and sill.get_linewidth() == 1.5
    assert {line.get_linestyle() for line in ax.lines} == {'-'}
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Realizations', 'Reference']
    plt.close('all')
    ax = gs.variogram_plot_simulation(sim, 1, reference_data=ref, legend_label='Reference', ls='--', sill_color='red',
                                      sill_lw=3, real_label='Simulated')
    sill = [line for line in ax.lines if list(line.get_ydata()) == [1, 1]][0]
    assert sill.get_color() == 'red' and sill.get_linewidth() == 3
    others = [line for line in ax.lines if line is not sill]
    assert len(others) == 4 and {line.get_linestyle() for line in others} == {'--'}  # 3 realizations and the reference
    assert sill.get_linestyle() == '-'  # the sill line stays solid
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Simulated', 'Reference']
    plt.close('all')


def test_stat_names_presets_lists_and_none():
    """stat_blk is a preset, a list of names or presets, or off; the names come back in the order of the plot."""
    from pygeostat.plotting.utils import _stat_names
    sets = {'all': ['ndat', 'rho', 'mean'], 'minimal': ['rho']}
    names = ['ndat', 'rho', 'mean']
    assert _stat_names('all', sets, names) == ['ndat', 'rho', 'mean']
    assert _stat_names(True, sets, names) == ['ndat', 'rho', 'mean']
    assert _stat_names('minimal', sets, names) == ['rho']
    assert _stat_names(['mean', 'ndat'], sets, names) == ['ndat', 'mean']
    assert _stat_names(('minimal', 'mean'), sets, names) == ['rho', 'mean']
    for off in (False, None, 'none'):
        assert _stat_names(off, sets, names) == []
    with pytest.raises(ValueError, match="'median' is not a valid stat_blk"):
        _stat_names(['rho', 'median'], sets, names)

def _swath_twin(ax):
    """The axis of the bars, which shares the x axis with the one of the lines"""
    return [a for a in ax.figure.axes if a is not ax and a.get_shared_x_axes().joined(a, ax)][0]


def test_swath_plot_means_and_counts():
    data = pd.DataFrame({'x': [0., 1., 2., 3.], 'y': [0., 0., 1., 1.], 'z': [0.] * 4,
                         'a': [1., 3., 5., np.nan], 'b': [2., 2., 2., 2.]})
    ax = gs.swath_plot(data, ['a', 'b'], orient='x', x='x', y='y', z='z', nbins=2)
    lines = {line.get_label(): line.get_ydata() for line in ax.get_lines()}
    np.testing.assert_allclose(lines['a'], [2., 5.])
    np.testing.assert_allclose(lines['b'], [2., 2.])
    twin = _swath_twin(ax)
    heights = sorted(set(np.round([p.get_height() for p in twin.patches], 3)))
    assert heights == [1.0, 2.0]  # counts 2, 2 for b and 2, 1 for a
    assert twin.get_ylabel() == 'Data Count (Bars)' and ax.get_ylabel() == 'Value (lines)'
    plt.close('all')


def test_swath_plot_orients_and_errors():
    data = gs.ExampleData('3d_estimate')
    axes = gs.swath_plot(data, ['True', 'Estimate'], orient=['x', 'y', 'z'], nbins=5)
    assert len(axes) == 3
    with pytest.raises(ValueError):
        gs.swath_plot(data, 'True', orient='w')
    with pytest.raises(ValueError):
        gs.swath_plot(data, 'nope')
    flat = pd.DataFrame({'x': [0., 1.], 'z': [0., 0.], 'a': [1., 2.]})
    with pytest.raises(ValueError, match="'z'"):
        gs.swath_plot(flat, 'a', orient='z', x='x', y='x', z='z')
    with pytest.raises(ValueError, match='None of the variables'):
        gs.swath_plot(flat.assign(a=np.nan), 'a', x='x', y='x', z='z')
    plt.close('all')


def test_swath_plot_fixed_slice_width():
    data = pd.DataFrame({'x': np.arange(0., 10.5, 0.5), 'y': 0., 'z': 0., 'a': np.arange(21.)})
    ax = gs.swath_plot(data, 'a', orient='x', x='x', y='y', z='z', width=2.5)
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), [1.25, 3.75, 6.25, 8.75])
    ax = gs.swath_plot(data, 'a', orient='x', x='x', y='y', z='z', width=50)
    assert len(ax.get_lines()[0].get_xdata()) == 1
    with pytest.raises(ValueError, match='positive'):
        gs.swath_plot(data, 'a', x='x', y='y', z='z', width=0)
    plt.close('all')


def test_swath_plot_labels():
    data = gs.ExampleData('3d_estimate')
    ax = gs.swath_plot(data, 'True', orient='x')
    assert ax.get_ylabel() == 'True (lines)' and len(ax.collections) == 0  # the values are not drawn as points
    ax = gs.swath_plot(data, ['True', 'Estimate'], orient='x', labels=['Data', 'Estimate'])
    assert ax.get_ylabel() == 'Value (lines)'
    assert [t.get_text() for t in ax.texts] == ['Data', 'Estimate']
    ax = gs.swath_plot(data, ['True', 'Estimate'], orient='x', ylabel='Grade')
    assert ax.get_ylabel() == 'Grade'
    plt.close('all')


def test_swath_plot_draws_on_the_axis_it_is_given():
    """With ax, no figure is made: the axis is returned and its twin is in the same figure."""
    data = gs.ExampleData('3d_estimate')
    fig, axes = plt.subplots(1, 2)
    ax = gs.swath_plot(data, 'True', orient='x', ax=axes[1])
    assert ax is axes[1] and _swath_twin(ax).figure is fig
    assert plt.get_fignums() == [fig.number]
    plt.close('all')


def test_swath_plot_bar_opacity_and_line_width():
    data = gs.ExampleData('3d_estimate')
    ax = gs.swath_plot(data, ['True', 'Estimate'], orient='x')
    twin = _swath_twin(ax)
    assert {p.get_alpha() for p in twin.patches} == {gs.Parameters['plotting.swath_plot.bar_alpha']} and ax.get_lines()[0].get_linewidth() == 2
    ax = gs.swath_plot(data, 'True', orient='x', bar_alpha=0.1, lw=4)
    twin = _swath_twin(ax)
    assert {p.get_alpha() for p in twin.patches} == {0.1} and ax.get_lines()[0].get_linewidth() == 4
    plt.close('all')



def test_swath_plot_follows_the_style_and_the_parameters():
    """The figure size, line width and bar opacity come from gs.Parameters."""
    data = gs.ExampleData('3d_estimate')
    ax = gs.swath_plot(data, 'True', orient='x')
    assert tuple(ax.figure.get_size_inches()) == (7, 4.5)
    saved = {k: gs.Parameters[k] for k in ('plotting.swath_plot.figsize', 'plotting.swath_plot.lw',
                                          'plotting.swath_plot.bar_alpha')}
    try:
        gs.Parameters['plotting.swath_plot.figsize'] = (5, 3)
        gs.Parameters['plotting.swath_plot.lw'] = 5
        gs.Parameters['plotting.swath_plot.bar_alpha'] = 0.6
        ax = gs.swath_plot(data, 'True', orient='x')
        twin = _swath_twin(ax)
        assert tuple(ax.figure.get_size_inches()) == (5, 3)
        assert ax.get_lines()[0].get_linewidth() == 5 and {p.get_alpha() for p in twin.patches} == {0.6}
    finally:
        for k, v in saved.items():
            gs.Parameters[k] = v
    plt.close('all')

def test_swath_plot_stacks_the_directions_in_one_figure(tmp_path):
    data = gs.ExampleData('3d_estimate')
    axes = gs.swath_plot(data, ['True', 'Estimate'], orient=['x', 'y', 'z'], width=2,
                         figsize=(6, 3), output_file=str(tmp_path / 'swath.png'))
    assert len(axes) == 3 and len({a.figure for a in axes}) == 1
    assert tuple(axes[0].figure.get_size_inches()) == (6, 9)
    assert [a.get_xlabel() for a in axes] == ['Easting (m)', 'Northing (m)', 'Elevation (m)']
    assert [f.name for f in tmp_path.iterdir()] == ['swath.png']
    plt.close('all')


def test_swath_plot_stacked_panels_use_one_tick_format():
    data = gs.ExampleData('oilsands')
    axes = gs.swath_plot(data, 'Bitumen', orient=['x', 'y', 'z'])
    plt.gcf().canvas.draw()
    ylabels = [[t.get_text() for t in a.get_yticklabels() if t.get_text()] for a in axes]
    decimals = {len(t.partition('.')[2]) for labels in ylabels for t in labels}
    assert len(decimals) == 1
    xlabels = [t.get_text() for a in axes for t in a.get_xticklabels() if t.get_text()]
    assert any(',' in t for t in xlabels) and all(',' in t for t in xlabels if len(t) > 4)
    plt.close('all')


def test_swath_plot_labels_on_the_lines_and_counts_with_separators():
    data = gs.ExampleData('3d_estimate')
    ax = gs.swath_plot(data, ['True', 'Estimate'], orient='x')
    twin = _swath_twin(ax)
    assert ax.get_legend() is None and ax.get_zorder() > twin.get_zorder()
    for line, text in zip(ax.get_lines(), ax.texts):
        assert text.get_color() == line.get_color()
        assert text.xy in zip(*line.get_data())  # at a point of its own line
    assert twin.yaxis.get_major_formatter()(12345, 0) == '12,345'
    plt.close('all')


def test_swath_plot_both_sides_use_the_same_sizes():
    data = gs.ExampleData('3d_estimate')
    ax = gs.swath_plot(data, ['True', 'Estimate'], orient='x')
    twin = _swath_twin(ax)
    assert twin.yaxis.label.get_size() == ax.yaxis.label.get_size()
    assert twin.get_yticklabels()[0].get_size() == ax.get_yticklabels()[0].get_size()
    plt.close('all')


def test_swath_plot_errors_leave_no_figure_open():
    data = pd.DataFrame({'x': [0., 1.], 'z': [0., 0.], 'a': [1., 2.]})
    plt.close('all')
    for call in (lambda: gs.swath_plot(data, 'a', orient=['x', 'z'], x='x', y='x', z='z'),
                 lambda: gs.swath_plot(data.assign(a=np.nan), 'a', x='x', y='x', z='z'),
                 lambda: gs.swath_plot(data, 'a', x='x', y='x', z='z', labels=['p', 'q'])):
        with pytest.raises(ValueError):
            call()
    assert plt.get_fignums() == []


def test_swath_plot_reads_columns_and_axis_names_of_a_2d_data_set():
    """A data set with other column names and no z: the columns come from the DataFile, the axis names
    from the plotting parameters."""
    data = gs.ExampleData('cluster')  # columns Xlocation and Ylocation, 0.5 to 48.5, 140 samples
    axes = gs.swath_plot(data, 'Primary', orient=['x', 'y'], nbins=10)
    centers = 0.5 + 4.8 * (np.arange(10) + 0.5)
    for ax, name in zip(axes, ('Easting (m)', 'Northing (m)')):
        np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), centers)
        assert ax.get_xlabel() == name and ax.get_ylabel() == 'Primary (lines)'
    with pytest.raises(ValueError, match='`z`'):
        gs.swath_plot(data, 'Primary', orient='z')
    names = {k: gs.Parameters[k] for k in ('plotting.xname', 'plotting.unit')}
    try:
        gs.Parameters['plotting.xname'], gs.Parameters['plotting.unit'] = 'X location', None
        assert gs.swath_plot(data, 'Primary', orient='x').get_xlabel() == 'X location'
    finally:
        for k, v in names.items():
            gs.Parameters[k] = v
    plt.close('all')


def test_swath_plot_even_right_ticks_no_tick_at_the_right_edge_and_labels_on_the_lines():
    data = gs.ExampleData('3d_estimate')
    axes = gs.swath_plot(data, ['True', 'Estimate'], orient=['x', 'y'], nbins=15)
    for a in axes:
        twin = _swath_twin(a)
        ticks = twin.get_yticks()
        ticks = ticks[(ticks >= twin.get_ylim()[0]) & (ticks <= twin.get_ylim()[1])]
        np.testing.assert_allclose(np.diff(ticks), np.diff(ticks)[0])
        np.testing.assert_allclose(ticks, np.round(ticks))  # counts: whole numbers, not 2.5 shown as 2
    for nbins in range(5, 30):
        a = gs.swath_plot(data, ['True', 'Estimate'], nbins=nbins)
        left, right = a.get_xlim()
        assert a.get_xticks().max() <= right - 0.08 * (right - left)
        plt.close('all')
    assert all([t.get_text() for t in a.texts] == ['True', 'Estimate'] for a in axes)  # on every plot
    assert list(gs.swath_plot(data, ['True', 'Estimate'], legend=False).texts) == []
    assert list(gs.swath_plot(data, 'True').texts) == []  # a single line needs no label
    plt.close('all')


def test_swath_plot_label_edge_cases_variable_without_values_coinciding_lines_and_margins():
    from pygeostat.plotting.swath_plot import _label_lines
    data = gs.ExampleData('3d_estimate')
    ax = gs.swath_plot(data.data.assign(Empty=np.nan), ['True', 'Empty'], x=data.x, y=data.y, z=data.z)
    assert [t.get_text() for t in ax.texts] == ['True']  # no value, no label, and no error
    means = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1.0, 2.0, 3.0]}, index=[1, 2, 3])
    fig, ax = plt.subplots()
    _label_lines(ax, means, ['a', 'b'], ['A', 'B'], ['red', 'blue'])
    assert [t.get_va() for t in ax.texts] == ['bottom', 'top']  # not on top of each other
    # the extra room is only made when there are labels
    for ax in (gs.swath_plot(data, ['True', 'Estimate'], legend=False), gs.swath_plot(data, 'True')):
        assert ax.margins()[1] == 0.05
    assert gs.swath_plot(data, ['True', 'Estimate']).margins()[1] == 0.16
    plt.close('all')


def test_swath_plot_label_is_where_its_line_is_furthest_from_the_others_above_if_it_is_the_higher():
    from pygeostat.plotting.swath_plot import _label_lines
    means = pd.DataFrame({'a': [1.0, 1.0, 5.0, 1.0, 1.0], 'b': [1.0, 1.0, 1.0, 1.0, 1.0]}, index=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots()
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    _label_lines(ax, means, ['a', 'b'], ['High', 'Low'], ['red', 'blue'])
    high, low = ax.texts
    assert (high.xy, high.get_va(), high.get_ha()) == ((3, 5.0), 'bottom', 'center')
    assert (low.xy, low.get_va()) == ((3, 1.0), 'top')
    # near the left or right edge the label grows inwards, so it stays inside the plot
    for peak, expected in ([0, 'left'], [4, 'right']):
        ax.clear()
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        values = [1.0] * 5
        values[peak] = 5.0
        _label_lines(ax, pd.DataFrame({'a': values, 'b': [1.0] * 5}, index=[1, 2, 3, 4, 5]), ['a', 'b'],
                     ['Peak', 'Flat'], ['red', 'blue'])
        assert (ax.texts[0].xy[0], ax.texts[0].get_ha()) == (peak + 1, expected)
    plt.close('all')


def test_swath_plot_count_ticks_are_whole_numbers_for_a_small_largest_count():
    data = pd.DataFrame({'x': np.r_[np.zeros(10), np.ones(3)], 'y': 0., 'z': 0., 'a': np.arange(13.)})
    ax = gs.swath_plot(data, 'a', orient='x', x='x', y='y', z='z', nbins=2)
    twin = _swath_twin(ax)
    ticks = twin.get_yticks()
    ticks = ticks[(ticks >= twin.get_ylim()[0]) & (ticks <= twin.get_ylim()[1])]
    np.testing.assert_allclose(ticks, np.round(ticks))
    np.testing.assert_allclose(np.diff(ticks), np.diff(ticks)[0])
    plt.close('all')


def _swath_data():
    return pd.DataFrame({'x': [0., 1., 2., 3.], 'y': [0., 0., 1., 1.], 'z': [0.] * 4,
                         'a': [1., 3., 5., np.nan], 'b': [2., 2., 2., 2.]})


def test_swath_plot_counts_as_points():
    """count_style='points' draws one point per slice and variable, at the same counts as the bars."""
    kwargs = dict(orient='x', x='x', y='y', z='z', nbins=2)
    bars = _swath_twin(gs.swath_plot(_swath_data(), ['a', 'b'], **kwargs))
    bar_heights = sorted(p.get_height() for p in bars.patches)
    plt.close('all')
    ax = gs.swath_plot(_swath_data(), ['a', 'b'], count_style='points', ms=9, **kwargs)
    twin = _swath_twin(ax)
    assert len(twin.patches) == 0 and len(twin.get_lines()) == 2
    assert sorted(np.concatenate([line.get_ydata() for line in twin.get_lines()])) == bar_heights
    assert all(line.get_markersize() == 9 and line.get_linestyle() == 'None' for line in twin.get_lines())
    assert twin.get_ylabel() == 'Data Count (Points)'
    with pytest.raises(ValueError, match="count_style"):
        gs.swath_plot(_swath_data(), 'a', x='x', y='y', z='z', count_style='dots')
    plt.close('all')


def test_swath_plot_axes_options():
    """Ticks, font size, limits and background reach both the line axis and the count axis."""
    ax = gs.swath_plot(_swath_data(), ['a', 'b'], orient='x', x='x', y='y', z='z', nbins=2,
                       ticks=6, fontsize={'label': 17, 'tick': 8}, xlim=(-1, 4), ylim=(0, 9),
                       background='lightgrey')
    twin = _swath_twin(ax)
    ax.figure.canvas.draw()
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_markersize() == 6
    assert twin.yaxis.get_major_ticks()[0].tick2line.get_markersize() == 6  # the count axis is on the right
    for a in (ax, twin):
        assert a.yaxis.label.get_fontsize() == 17
        assert a.get_yticklabels()[0].get_fontsize() == 8
    assert ax.get_xlim() == (-1, 4) and ax.get_ylim() == (0, 9)
    assert twin.patch.get_visible() and twin.get_facecolor()[:3] == (0.8274509803921568,) * 3
    plt.close('all')


def _two_swath_sets():
    first = pd.DataFrame({'x': [0., 1., 2., 3.], 'y': 0., 'z': 0., 'v': [1., 3., 5., 7.]})
    second = pd.DataFrame({'x': [2., 3., 4., 5., 6.], 'y': 0., 'z': 0., 'v': [10., 10., 20., 20., 30.]})
    return first, second


def test_swath_plot_several_data_sets_share_the_slices():
    """Each data set is a line; the slices span the coordinates of all of them; counts are for each data set."""
    plt.close('all')
    first, second = _two_swath_sets()
    ax = gs.swath_plot([first, second], 'v', x='x', y='y', z='z', width=2, labels=['Data', 'Estimate'])
    lines = {line.get_label(): (line.get_xdata(), line.get_ydata()) for line in ax.get_lines()}
    # slices 0-2, 2-4 and 4-6, closed on the right: the first set has no value in the last one
    np.testing.assert_allclose(lines['Data'][0], [1., 3.])
    np.testing.assert_allclose(lines['Data'][1], [3., 7.])
    np.testing.assert_allclose(lines['Estimate'][0], [1., 3., 5.])
    np.testing.assert_allclose(lines['Estimate'][1], [10., 15., 25.])
    twin = _swath_twin(ax)
    assert sorted({p.get_height() for p in twin.patches} - {0}) == [1.0, 2.0, 3.0]  # 0: no value in the last slice
    plt.close('all')
    ax = gs.swath_plot([first, second], 'v', x='x', y='y', z='z', nbins=2)
    assert [l.get_label() for l in ax.get_lines()] == ['Data 1', 'Data 2']  # same name in both, so numbered
    plt.close('all')
    ax = gs.swath_plot([first, second.rename(columns={'v': 'w'})], ['v', 'w'], x='x', y='y', z='z', nbins=2)
    assert [l.get_label() for l in ax.get_lines()] == ['v', 'w']  # different names are the labels
    plt.close('all')


def test_swath_plot_several_data_sets_errors_leave_no_figure():
    plt.close('all')
    first, second = _two_swath_sets()
    args = dict(x='x', y='y', z='z')
    with pytest.raises(ValueError, match='one name for each of the 2 data sets'):
        gs.swath_plot([first, second], ['v', 'v', 'v'], **args)
    with pytest.raises(ValueError, match="'w' is not a column"):
        gs.swath_plot([first, second], ['v', 'w'], **args)
    with pytest.raises(ValueError, match='list of data sets is empty'):
        gs.swath_plot([], 'v', **args)
    with pytest.raises(ValueError, match='gs.DataFile or a pd.DataFrame'):
        gs.swath_plot([first, 3], 'v', **args)
    assert len(plt.get_fignums()) == 0


def test_swath_plot_count_label():
    plt.close('all')
    first, _ = _two_swath_sets()
    ax = gs.swath_plot(first, 'v', x='x', y='y', z='z', nbins=2, count_label='Number of samples', ylabel='Grade')
    assert _swath_twin(ax).get_ylabel() == 'Number of samples' and ax.get_ylabel() == 'Grade'
    plt.close('all')

def test_cross_plot_stats_and_nan():
    x = np.array([0., 1., 2., 3., np.nan])
    y = np.array([0., 2., 4., 6., 9.])
    ax = gs.cross_plot(x, y, nbins=3)
    texts = [t.get_text() for t in ax.texts]
    assert texts == ['n = 4\n$\\rho = 1.000$']  # the pair with the NaN is left out
    assert [t.get_text() for t in ax.histx.texts] == ['Mean = 1.50\nVar = 1.25']
    assert [t.get_text() for t in ax.histy.texts] == ['Mean = 3.00\nVar = 5.00']
    assert len(ax.collections[0].get_offsets()) == 4
    plt.close('all')



def test_cross_plot_follows_the_style_and_the_parameters():
    """The figure size comes from the plot style, and the colours and point size from gs.Parameters."""
    x = np.arange(10.)
    ax = gs.cross_plot(x, x ** 2, custom_style={'figure.figsize': (4, 3)})
    assert tuple(ax.figure.get_size_inches()) == (4, 3)
    saved = {k: gs.Parameters[k] for k in ('plotting.cross_plot.c', 'plotting.cross_plot.s',
                                          'plotting.cross_plot.histcolor')}
    try:
        gs.Parameters['plotting.cross_plot.c'] = 'r'
        gs.Parameters['plotting.cross_plot.s'] = 30
        gs.Parameters['plotting.cross_plot.histcolor'] = 'b'
        ax = gs.cross_plot(x, x ** 2)
        points = ax.collections[0]
        assert tuple(points.get_facecolor()[0][:3]) == (1.0, 0.0, 0.0) and points.get_sizes()[0] == 30
        assert tuple(ax.histx.patches[0].get_facecolor()[:3]) == (0.0, 0.0, 1.0)
    finally:
        for k, v in saved.items():
            gs.Parameters[k] = v
    plt.close('all')


def test_cross_plot_draws_on_the_axis_it_is_given():
    """With ax, nothing new is made: the axis is returned with its histograms inside the same figure."""
    fig, axes = plt.subplots(1, 2)
    x = np.arange(10.)
    ax = gs.cross_plot(x, x ** 2, ax=axes[0])
    assert ax is axes[0] and ax.histx.figure is fig and ax.histy.figure is fig
    assert plt.get_fignums() == [fig.number]
    plt.close('all')

def test_cross_plot_errors():
    with pytest.raises(ValueError):
        gs.cross_plot([1., 2., 3.], [1., 2.])
    with pytest.raises(ValueError):
        gs.cross_plot([1., np.nan], [np.nan, 2.])
    plt.close('all')


def test_cross_plot_rejects_several_columns():
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 1., 3.]})
    with pytest.raises(ValueError, match='one variable'):
        gs.cross_plot(df[['a', 'b']], [1., 2., 3., 4., 5., 6.])
    plt.close('all')


def test_cross_plot_colours_the_points_by_density_by_default():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = x + rng.normal(size=60)
    ax = gs.cross_plot(x, y)
    points = ax.collections[0]
    from scipy.stats import gaussian_kde
    density = gaussian_kde(np.vstack([x, y])).evaluate(np.vstack([x, y]))
    np.testing.assert_allclose(np.sort(points.get_array()), np.sort(density))
    # the densest points are drawn last, so they are on top
    assert points.get_array()[-1] == density.max()
    # all the points on a line: no density, one colour, no error
    ax = gs.cross_plot(x, 2 * x)
    assert ax.collections[0].get_array() is None
    plt.close('all')


def test_cross_plot_colours_the_points_by_values_with_a_colour_bar():
    x = np.arange(6.)
    y = np.array([1., 3., 2., 5., 4., 6.])
    third = pd.Series([10., 20., np.nan, 40., 50., 60.], name='Third')
    ax = gs.cross_plot(x, y, mc=third, cmap='plasma', clim=(0, 100), cbar=True, nbins=3)
    points = ax.collections[0]
    np.testing.assert_allclose(points.get_array(), [10., 20., 40., 50., 60.])  # the pair with the NaN is left out
    assert points.get_clim() == (0, 100) and points.get_cmap().name == 'plasma'
    assert ax.child_axes[-1].get_ylabel() == 'Third'  # the colour bar, after the two histograms
    assert len(gs.cross_plot(x, y, mc=third).child_axes) == 2  # no colour bar by default: only the two histograms
    assert gs.cross_plot(x, y, mc='kde', cbar=True).child_axes[-1].get_ylabel() == 'Density'
    assert gs.cross_plot(x, y, mc='kde', cbar=True, cbar_label='Count').child_axes[-1].get_ylabel() == 'Count'
    ax = gs.cross_plot(x, y, mc='red')
    assert ax.collections[0].get_array() is None and tuple(ax.collections[0].get_facecolor()[0][:3]) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match='one value for each'):
        gs.cross_plot(x, y, mc=np.arange(3.))
    plt.close('all')


def test_cross_plot_axis_names_are_as_large_as_the_tick_labels():
    d = gs.ExampleData('3d_estimate').data
    ax = gs.cross_plot(d['Estimate'], d['True'])
    tick = ax.histx.xaxis.get_ticklabels()[0].get_size()
    assert ax.histx.xaxis.label.get_size() == tick and ax.histy.yaxis.label.get_size() == tick
    assert (ax.histx.get_xlabel(), ax.histy.get_ylabel()) == ('Estimate', 'True')
    ax = gs.cross_plot(d['Estimate'], d['True'], label_size=20)
    assert ax.histx.xaxis.label.get_size() == 20 and ax.histy.yaxis.label.get_size() == 20
    plt.close('all')


def test_cross_plot_45_degree_line_and_log_colours():
    x = np.array([0., 1., 2., 3., 4.])
    y = np.array([0., 3., 1., 4., 8.])
    ax = gs.cross_plot(x, y, line45=True, mc='red')
    assert ax.get_xlim() == ax.get_ylim() == (0., 8.)  # the same limits on both axes
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), line.get_ydata())
    assert not gs.cross_plot(x, y, mc='red').lines  # no line by default
    assert gs.cross_plot(x, y, line45=True, lim=(-1, 9)).get_xlim() == (-1, 9)  # lim is kept
    # values that are not above zero get the lowest color on a log scale
    third = pd.Series([0., 1., 10., 100., 1000.])
    points = gs.cross_plot(x, y, mc=third, log_color=True).collections[0]
    assert type(points.norm).__name__ == 'LogNorm' and (points.norm.vmin, points.norm.vmax) == (1., 1000.)
    np.testing.assert_allclose(points.to_rgba(0.)[:3], points.to_rgba(1.)[:3])
    with pytest.raises(ValueError, match='above zero'):
        gs.cross_plot(x, y, mc=pd.Series(np.zeros(5)), log_color=True)
    plt.close('all')


def test_cross_plot_correlation_is_written_in_the_corner_with_the_fewest_points():
    rng = np.random.default_rng(1)
    x = np.append(rng.uniform(0, 1, 200), [0., 1.])
    y = np.append(rng.uniform(0, 1, 200), [0., 1.])
    keep = ~((x < 0.35) & (y > 0.88))  # an empty top left corner
    ax = gs.cross_plot(x[keep], y[keep])
    label = [t for t in ax.texts if 'rho' in t.get_text()][0]
    assert label.get_position() == (0.05, 0.95)
    keep = ~((x > 0.65) & (y < 0.12))  # an empty bottom right corner
    ax = gs.cross_plot(x[keep], y[keep])
    assert [t for t in ax.texts if 'rho' in t.get_text()][0].get_position() == (0.95, 0.05)
    plt.close('all')


def test_cross_plot_leaves_room_above_the_bars_of_the_x_histogram_for_the_statistics():
    d = gs.ExampleData('3d_estimate').data
    ax = gs.cross_plot(d['Estimate'], d['True'])
    tallest = max(p.get_height() for p in ax.histx.patches)
    assert ax.histx.get_ylim()[1] >= 1.5 * tallest
    plt.close('all')


def _two_sets():
    rng = np.random.RandomState(3)
    x = rng.lognormal(0, 0.5, 60)
    return x, 2 * x + rng.normal(0, 0.2, 60) + 3, x[::-1] * 1.5, rng.lognormal(1, 0.3, 60)


def test_cross_plot_two_data_sets():
    """A second data set gets its own colour, legend entry with its correlation, and histograms and statistics."""
    x, y, x2, y2 = _two_sets()
    ax = gs.cross_plot(x, y, x2=x2, y2=y2, mc2='r', label='Data', label2='Estimate', nbins=5)
    assert len(ax.collections) == 2
    assert tuple(ax.collections[1].get_facecolor()[0][:3]) == (1.0, 0.0, 0.0)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ['Data (n = 60, $\\rho = %.3f$)' % np.corrcoef(x, y)[0, 1],
                      'Estimate (n = 60, $\\rho = %.3f$)' % np.corrcoef(x2, y2)[0, 1]]
    assert len(ax.histx.patches) == 10 and len(ax.histy.patches) == 10  # 5 bars for each set
    assert tuple(ax.histx.patches[-1].get_facecolor()[:3]) == (1.0, 0.0, 0.0)
    assert [t.get_text() for t in ax.histx.texts] == ['Mean = %.2f\nVar = %.2f' % (v.mean(), v.var()) for v in (x, x2)]
    assert [t.get_color() for t in ax.histx.texts] == ['C0', 'r']
    assert not ax.texts  # the correlations are in the legend
    # both sets use the same bins, so the bars of the two sets line up
    np.testing.assert_allclose([p.get_x() for p in ax.histx.patches[:5]], [p.get_x() for p in ax.histx.patches[5:]])
    plt.close('all')


def test_cross_plot_two_data_sets_errors():
    plt.close('all')
    x, y, x2, y2 = _two_sets()
    with pytest.raises(ValueError, match='together'):
        gs.cross_plot(x, y, x2=x2)
    with pytest.raises(ValueError, match='same length'):
        gs.cross_plot(x, y, x2=x2, y2=y2[:5])
    with pytest.raises(ValueError, match='single colour'):
        gs.cross_plot(x, y, x2=x2, y2=y2, mc='kde')
    with pytest.raises(ValueError, match='second data set'):
        gs.cross_plot(x, y, x2=[1., np.nan], y2=[np.nan, 2.])
    assert len(plt.get_fignums()) == 0  # nothing is left open by the errors
    plt.close('all')


def test_cross_plot_log_axes_and_log_counts():
    """logx/logy put the points and the histograms on a log scale with bins equal in log; log_counts is the counts."""
    plt.close('all')
    x, y, _, _ = _two_sets()
    ax = gs.cross_plot(x, y, logx=True, logy=True, nbins=4, mc='C0')
    assert (ax.get_xscale(), ax.get_yscale()) == ('log', 'log')
    assert (ax.histx.get_xscale(), ax.histy.get_yscale()) == ('log', 'log')  # shared with the points
    assert ax.histx.get_yscale() == 'linear'
    edges = np.array([p.get_x() for p in ax.histx.patches] + [ax.histx.patches[-1].get_x() + ax.histx.patches[-1].get_width()])
    np.testing.assert_allclose(np.diff(np.log(edges)), np.diff(np.log(edges))[0])
    plt.close('all')
    ax = gs.cross_plot(x, y, log_counts=True, mc='C0')
    assert (ax.histx.get_yscale(), ax.histy.get_xscale()) == ('log', 'log')
    assert (ax.get_xscale(), ax.get_yscale()) == ('linear', 'linear')
    plt.close('all')
    with pytest.raises(ValueError, match='above zero'):
        gs.cross_plot(x - 5, y, logx=True)
    with pytest.raises(ValueError, match='lower limit'):
        gs.cross_plot(x, y, logx=True, xlim=(0, 5))
    assert len(plt.get_fignums()) == 0
    plt.close('all')


def test_cross_plot_limits_per_axis_ticks_fontsize_background():
    """xlim and ylim replace lim for one axis and set the histogram bins; the options reach the histograms."""
    x, y, _, _ = _two_sets()
    ax = gs.cross_plot(x, y, lim=(0, 20), xlim=(0, 10), nbins=5, mc='C0', ticks=6, fontsize={'label': 13},
                       background='lightgrey')
    assert ax.get_xlim() == (0, 10) and ax.get_ylim() == (0, 20)
    assert ax.histx.patches[0].get_x() == 0 and ax.histy.patches[0].get_y() == 0
    ax.figure.canvas.draw()
    assert ax.histx.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 6
    assert ax.histy.yaxis.get_major_ticks()[0].tick1line.get_markersize() == 6
    assert ax.histx.xaxis.label.get_fontsize() == 13 and ax.histy.yaxis.label.get_fontsize() == 13
    assert tuple(ax.get_facecolor()[:3]) == (0.8274509803921568,) * 3
    with pytest.raises(ValueError, match='ylim must be'):
        gs.cross_plot(x, y, ylim=(1, 2, 3))
    plt.close('all')


@pytest.mark.parametrize('kw', [dict(xlim=(5, 0)), dict(ylim=(3, 3)), dict(lim=(5, 0))])
def test_cross_plot_refuses_limits_that_do_not_increase(kw):
    """Reversed limits give a clear error, and no figure is left open."""
    plt.close('all')
    x, y, _, _ = _two_sets()
    with pytest.raises(ValueError, match='must be increasing'):
        gs.cross_plot(x, y, mc='C0', **kw)
    assert len(plt.get_fignums()) == 0


def test_cross_plot_stat_fontsize_is_separate_from_fontsize():
    """stat_fontsize sizes the correlation, mean and variance; fontsize does not change them."""
    plt.close('all')
    x, y, _, _ = _two_sets()
    ax = gs.cross_plot(x, y, mc='C0', stat_fontsize=17, fontsize=9)
    assert ax.texts[0].get_fontsize() == 17
    assert [t.get_fontsize() for t in ax.histx.texts + ax.histy.texts] == [17, 17]
    assert ax.histx.xaxis.label.get_fontsize() == 9
    plt.close('all')
    with pytest.raises(ValueError, match='stat_fontsize'):
        gs.cross_plot(x, y, mc='C0', stat_fontsize=-1)
    assert len(plt.get_fignums()) == 0


def test_cross_plot_two_data_sets_layout_and_transparency():
    """The points and bars of two sets are transparent; the statistics of the second sit beside and below the first."""
    plt.close('all')
    x, y, x2, y2 = _two_sets()
    ax = gs.cross_plot(x, y, x2=x2, y2=y2)
    assert [c.get_alpha() for c in ax.collections] == [0.6, 0.6]
    assert {p.get_alpha() for p in ax.histx.patches + ax.histy.patches} == {0.6}
    first, second = ax.histx.texts
    assert first.xy[0] < second.xy[0] and first.xy[1] == second.xy[1]  # beside each other
    first, second = ax.histy.texts
    assert first.xyann[1] == 0 and second.xyann[1] < 0  # stacked, the second lower
    plt.close('all')


def test_cross_plot_stat_blk_chooses_the_statistics():
    """all writes every statistic, minimal only the correlation, a list the ones named, none nothing."""
    plt.close('all')
    x, y, x2, y2 = _two_sets()

    def written(**kw):
        ax = gs.cross_plot(x, y, mc='C0', **kw)
        return [t.get_text() for t in ax.texts], [t.get_text() for t in ax.histx.texts + ax.histy.texts]
    rho = r'$\rho = %.3f$' % np.corrcoef(x, y)[0, 1]
    assert written() == (['n = 60\n' + rho], ['Mean = %.2f\nVar = %.2f' % (x.mean(), x.var()),
                                              'Mean = %.2f\nVar = %.2f' % (y.mean(), y.var())])
    assert written(stat_blk='minimal') == ([rho], [])
    assert written(stat_blk=['mean']) == ([], ['Mean = %.2f' % x.mean(), 'Mean = %.2f' % y.mean()])
    assert written(stat_blk='none') == ([], []) and written(stat_blk=False) == ([], [])
    plt.close('all')
    ax = gs.cross_plot(x, y, x2=x2, y2=y2, stat_blk='minimal')
    assert [t.get_text() for t in ax.get_legend().get_texts()][0] == 'Data 1 (%s)' % rho
    assert len(ax.histx.texts) == 0
    plt.close('all')
    ax = gs.cross_plot(x, y, x2=x2, y2=y2, stat_blk=['mean'])
    first, second = ax.histy.texts
    assert second.xyann[1] == -1.6 * second.get_fontsize()  # one line each, so the second sits one line lower
    plt.close('all')
    with pytest.raises(ValueError, match="'median' is not a valid stat_blk"):
        gs.cross_plot(x, y, mc='C0', stat_blk=['median'])
    assert len(plt.get_fignums()) == 0

def _clip_grid():
    grid = gs.GridDef([4, 0.5, 1, 4, 0.5, 1, 1, 0.5, 1])  # cell centres 0.5, 1.5, 2.5, 3.5
    return gs.DataFile(data=pd.DataFrame({'v': np.arange(16.)}), griddef=grid)


def test_clip_plot_cells_by_centre():
    square = [[0.9, 0.9], [3.1, 0.9], [3.1, 3.1], [0.9, 3.1]]  # holds the centres 1.5 and 2.5 only
    ax = gs.clip_plot(_clip_grid(), square, var='v', cbar=False)
    shown = ax.images[0].get_array()
    assert np.isfinite(shown).sum() == 4
    assert np.isfinite(shown[1:3, 1:3]).all()  # the rows with y 1.5 and 2.5, the columns with x 1.5 and 2.5
    plt.close('all')


def test_clip_plot_hull_and_polygon_file(tmp_path):
    points = pd.DataFrame({'X': [0.4, 3.6, 3.6, 0.4, 2.0], 'Y': [0.4, 0.4, 3.6, 3.6, 2.0]})
    hull = gs.convex_hull(points['X'], points['Y'])
    assert len(hull) == 5 and (hull[0] == hull[-1]).all()  # four corners, the interior point is left out
    ax = gs.clip_plot(_clip_grid(), 'hull', var='v', points=points, x='X', y='Y', cbar=False)
    assert np.isfinite(ax.images[0].get_array()).sum() == 16  # the hull holds every cell centre
    path = tmp_path / 'boundary.dat'
    path.write_text('polygon\n2\nx\ny\n0.9 0.9\n3.1 0.9\n3.1 3.1\n0.9 3.1\n')
    ax = gs.clip_plot(_clip_grid(), str(path), var='v', cbar=False)
    assert np.isfinite(ax.images[0].get_array()).sum() == 4
    plt.close('all')


def test_clip_plot_errors():
    with pytest.raises(ValueError, match='points'):
        gs.clip_plot(_clip_grid(), 'hull', var='v')
    with pytest.raises(ValueError, match='three corners'):
        gs.clip_plot(_clip_grid(), [[0, 0], [1, 1]], var='v')
    with pytest.raises(ValueError, match='rows'):
        gs.clip_plot(pd.DataFrame({'v': [1., 2.]}), [[0, 0], [1, 0], [1, 1]], griddef=_clip_grid().griddef, var='v')
    plt.close('all')


def test_clip_plot_points_without_value_and_csv(tmp_path):
    points = pd.DataFrame({'X': [0.4, 3.6, 3.6, 0.4, 2.0], 'Y': [0.4, 0.4, 3.6, 3.6, 2.0]})
    ax = gs.clip_plot(_clip_grid(), 'hull', var='v', points=points, x='X', y='Y', cbar=False)
    dots = ax.collections[-1]
    assert len(dots.get_offsets()) == 5 and tuple(dots.get_facecolor()[0][:3]) == (0., 0., 0.)
    assert ax.get_xlim() == (0.0, 4.0)  # the dots do not change the extent of the grid
    csv = tmp_path / 'boundary.csv'
    csv.write_text('x,y\n0.9,0.9\n3.1,0.9\n3.1,3.1\n0.9,3.1\n')
    ax = gs.clip_plot(_clip_grid(), str(csv), var='v', cbar=False)
    assert np.isfinite(ax.images[0].get_array()).sum() == 4
    with pytest.raises(ValueError, match='one line'):
        gs.convex_hull([0, 1, 2], [0, 1, 2])
    plt.close('all')


def test_clip_plot_default_title_and_colour_bar_label_name_the_variable():
    square = [[0.9, 0.9], [3.1, 0.9], [3.1, 3.1], [0.9, 3.1]]
    ax, colorbar = gs.clip_plot(_clip_grid(), square, var='v', return_cbar=True)
    assert ax.get_title() == 'v Clipped to the Boundary'
    assert colorbar.ax.get_ylabel() == 'v'
    points = pd.DataFrame({'X': [0.4, 3.6, 3.6, 0.4, 2.0], 'Y': [0.4, 0.4, 3.6, 3.6, 2.0]})
    ax = gs.clip_plot(_clip_grid(), 'hull', var='v', points=points, x='X', y='Y')
    assert ax.get_title() == 'v Clipped to the Convex Hull'
    # what the user gives is kept, and False removes the title
    ax, colorbar = gs.clip_plot(_clip_grid(), square, var='v', title='Zinc', cbar_label='Zn (ppm)', return_cbar=True)
    assert ax.get_title() == 'Zinc' and colorbar.ax.get_ylabel() == 'Zn (ppm)'
    assert gs.clip_plot(_clip_grid(), square, var='v', title=False).get_title() == ''
    assert gs.clip_plot(_clip_grid(), square, var='v', title=None).get_title() == 'v Clipped to the Boundary'
    plt.close('all')


def test_clip_plot_colour_bar_label_is_as_large_as_its_tick_labels():
    square = [[0.9, 0.9], [3.1, 0.9], [3.1, 3.1], [0.9, 3.1]]
    ax, colorbar = gs.clip_plot(_clip_grid(), square, var='v', return_cbar=True)
    assert colorbar.ax.yaxis.label.get_size() == colorbar.ax.yaxis.get_ticklabels()[0].get_size()
    # without a colour bar the plot is returned alone, and a colour bar is not asked for
    ax = gs.clip_plot(_clip_grid(), square, var='v', cbar=False)
    assert not isinstance(ax, tuple)
    plt.close('all')


def _hull_by_brute_force(xy):
    """The corners of the convex hull found without a hull routine: a pair of points is an edge of the
    hull when all the other points are on one side of the line through them (n ** 3 steps, small sets)"""
    corners = set()
    for i in range(len(xy)):
        for j in range(len(xy)):
            if i == j or (xy[i] == xy[j]).all():
                continue
            d = xy[j] - xy[i]
            side = d[0] * (xy[:, 1] - xy[i, 1]) - d[1] * (xy[:, 0] - xy[i, 0])
            if (side <= 1e-12 * np.abs(d).max()).all() and (side[np.abs(side) > 1e-12 * np.abs(d).max()] < 0).all():
                on_line = np.abs(side) <= 1e-12 * np.abs(d).max()
                t = ((xy - xy[i]) @ d) / (d @ d)
                if not (on_line & ((t < -1e-12) | (t > 1 + 1e-12))).any():  # no point on the line outside the edge
                    corners.update((tuple(xy[i]), tuple(xy[j])))
    return corners


@pytest.mark.parametrize('name, x, y', [('cluster', 'Xlocation', 'Ylocation'), ('oilsands', 'East', 'North'),
                                        ('point2d_ind', 'X', 'Y')])
def test_convex_hull_of_the_example_data_sets(name, x, y):
    """The hull of a data set is a closed counter-clockwise polygon of data points that holds every point,
    with the same corners as an independent search, and the same area as scipy."""
    from scipy.spatial import ConvexHull
    data = gs.ExampleData(name).data[[x, y]].drop_duplicates().sample(60, random_state=0)
    hull = gs.convex_hull(data[x], data[y])
    xy = data.to_numpy()
    assert (hull[0] == hull[-1]).all()
    assert {tuple(c) for c in hull[:-1]} == _hull_by_brute_force(xy)
    assert {tuple(c) for c in hull[:-1]} <= {tuple(p) for p in xy}
    area = 0.5 * np.sum(hull[:-1, 0] * hull[1:, 1] - hull[1:, 0] * hull[:-1, 1])
    assert area > 0  # counter-clockwise
    assert area == pytest.approx(ConvexHull(xy).volume)
    for start, end in zip(hull[:-1], hull[1:]):  # every point is on the left of every edge, or on it
        side = (end[0] - start[0]) * (xy[:, 1] - start[1]) - (end[1] - start[1]) * (xy[:, 0] - start[0])
        assert (side >= -1e-9 * np.abs(xy).max() ** 2).all()
    plt.close('all')


def test_convex_hull_inputs_duplicates_nan_and_large_coordinates():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 100, (40, 2))
    expected = {tuple(c) for c in gs.convex_hull(xy[:, 0], xy[:, 1])[:-1]}
    for x, y in ((list(xy[:, 0]), list(xy[:, 1])), (pd.Series(xy[:, 0]), pd.Series(xy[:, 1])),
                 (pd.DataFrame(xy[:, :1]), pd.DataFrame(xy[:, 1:]))):
        assert {tuple(c) for c in gs.convex_hull(x, y)[:-1]} == expected
    # repeated locations and a pair with a NaN change nothing
    x = np.append(np.tile(xy[:, 0], 2), np.nan)
    y = np.append(np.tile(xy[:, 1], 2), 5.0)
    assert {tuple(c) for c in gs.convex_hull(x, y)[:-1]} == expected
    # a set moved far away (UTM-size coordinates) and turned has the moved and turned hull
    theta = 0.7
    turned = xy @ np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) + [5.5e5, 6.1e6]
    moved = {tuple(np.round(c, 6)) for c in gs.convex_hull(turned[:, 0], turned[:, 1])[:-1]}
    wanted = {tuple(np.round(np.array(c) @ np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) + [5.5e5, 6.1e6], 6))
              for c in expected}
    assert moved == wanted
    # three points make a triangle; four on a line have no hull
    assert len(gs.convex_hull([0, 1, 0], [0, 0, 1])) == 4
    with pytest.raises(ValueError, match='one line'):
        gs.convex_hull([0, 1, 2, 3], [1, 3, 5, 7])


def _signed_volume(vertices, faces):
    """Volume of a closed surface from its triangles, positive when they face outwards"""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return np.sum(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6


def test_convex_hull_3d_of_a_cube_with_points_inside():
    corners = np.array([[i, j, k] for i in (0, 2) for j in (0, 2) for k in (0, 2)], dtype=float)
    inside = np.random.default_rng(0).uniform(0.1, 1.9, size=(50, 3))
    xyz = np.vstack([inside, corners])
    vertices, faces = gs.convex_hull_3d(*xyz.T)
    assert {tuple(v) for v in vertices} == {tuple(c) for c in corners}
    assert faces.shape == (12, 3)  # two triangles for each of the six sides
    assert _signed_volume(vertices, faces) == pytest.approx(8.0)  # positive: the triangles face outwards


def test_convex_hull_3d_has_exactly_the_points_on_a_sphere_as_corners_and_faces_that_enclose_every_point():
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(1)
    sphere = rng.normal(size=(40, 3))
    sphere = 10 * sphere / np.linalg.norm(sphere, axis=1)[:, None] + [100.0, 200.0, 300.0]
    inside = [100.0, 200.0, 300.0] + rng.uniform(-4, 4, size=(100, 3))  # within 7 of the centre, the sphere has radius 10
    xyz = np.vstack([inside, sphere])
    rng.shuffle(xyz)
    vertices, faces = gs.convex_hull_3d(*xyz.T)
    assert {tuple(v) for v in vertices} == {tuple(p) for p in sphere}  # every point on the sphere, none inside
    assert _signed_volume(vertices, faces) == pytest.approx(ConvexHull(xyz).volume)
    corners = vertices[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    for normal, corner in zip(normals, corners[:, 0]):  # no point is beyond any face
        assert ((xyz - corner) @ normal <= 1e-6 * np.linalg.norm(normal)).all()


def test_convex_hull_3d_inputs_nan_and_errors():
    cube = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
    vertices, faces = gs.convex_hull_3d(list(cube[:, 0]) + [np.nan], list(cube[:, 1]) + [5.0], list(cube[:, 2]) + [5.0])
    assert len(vertices) == 8 and len(faces) == 12  # the location with a NaN is left out
    with pytest.raises(ValueError, match='four'):
        gs.convex_hull_3d([0, 1, 0], [0, 0, 1], [0, 0, 0])
    with pytest.raises(ValueError, match='plane'):
        gs.convex_hull_3d([0, 1, 0, 1], [0, 0, 1, 1], [5, 5, 5, 5])  # all at the same elevation
    with pytest.raises(ValueError, match='plane'):
        gs.convex_hull_3d([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3])  # on one line


def test_clip_plot_axes_options():
    """Ticks, font size (also of the colour bar), limits and the colour of the empty cells can be set."""
    square = [[0.9, 0.9], [3.1, 0.9], [3.1, 3.1], [0.9, 3.1]]
    ax = gs.clip_plot(_clip_grid(), square, var='v', ticks=6, fontsize={'label': 15, 'tick': 9},
                      xlim=(0, 3), ylim=(1, 4), background='lightgrey')
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 6
    assert ax.xaxis.label.get_fontsize() == 15 and ax.get_xticklabels()[0].get_fontsize() == 9
    assert ax.get_xlim() == (0, 3) and ax.get_ylim() == (1, 4)
    assert tuple(ax.get_facecolor()[:3]) == (0.8274509803921568,) * 3
    colorbar_axis = [a for a in ax.figure.axes if a is not ax][0]
    assert colorbar_axis.yaxis.label.get_fontsize() == 15
    assert colorbar_axis.get_yticklabels()[0].get_fontsize() == 9
    plt.close('all')
    ax = gs.clip_plot(_clip_grid(), square, var='v', cbar=False, ticks=False)
    ax.figure.canvas.draw()
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 0
    plt.close('all')


def test_clip_plot_outline_default_is_for_the_hull_only():
    """The convex hull is drawn by default, a given polygon is not; outline overrides both."""
    square = [[0.9, 0.9], [3.1, 0.9], [3.1, 3.1], [0.9, 3.1]]
    points = pd.DataFrame({'x': [1., 3, 2, 1.5], 'y': [1., 1, 3, 2]})
    hull = dict(points=points, x='x', y='y')
    assert len(gs.clip_plot(_clip_grid(), square, var='v', cbar=False).lines) == 0
    assert len(gs.clip_plot(_clip_grid(), 'hull', var='v', cbar=False, **hull).lines) == 1
    assert len(gs.clip_plot(_clip_grid(), square, var='v', cbar=False, outline=True).lines) == 1
    assert len(gs.clip_plot(_clip_grid(), 'hull', var='v', cbar=False, outline=False, **hull).lines) == 0
    plt.close('all')


def test_clip_plot_limits_are_kept_when_points_are_drawn():
    """xlim and ylim also apply when the data locations are drawn on top."""
    square = [[0.9, 0.9], [3.1, 0.9], [3.1, 3.1], [0.9, 3.1]]
    points = pd.DataFrame({'x': [1., 3, 2, 1.5], 'y': [1., 1, 3, 2]})
    ax = gs.clip_plot(_clip_grid(), square, var='v', cbar=False, points=points, x='x', y='y',
                      xlim=(0, 2), ylim=(-1, 5))
    assert ax.get_xlim() == (0, 2) and ax.get_ylim() == (-1, 5)
    plt.close('all')

def _gt_twin(ax):
    """The axis of the grades, which shares the x axis with the one of the tonnages"""
    return [a for a in ax.figure.axes if a is not ax and a.get_shared_x_axes().joined(a, ax)][0]


def test_grade_tonnage_plot_tonnage_and_grade_by_hand():
    # values 1, 2, 3, 4 with weights 1, 1, 2, 4: above 3 are weights 2 and 4 with grades 3 and 4
    data = pd.DataFrame({'g': [1., 2., 3., 4.], 'w': [1., 1., 2., 4.]})
    ax = gs.grade_tonnage_plot(data, 'g', weights='w', cutoffs=[0., 2., 3., 5.])
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [1., 7 / 8, 6 / 8, 0.])
    np.testing.assert_allclose(_gt_twin(ax).get_lines()[0].get_ydata(),
                               [(1 + 2 + 6 + 16) / 8, (2 + 6 + 16) / 7, (6 + 16) / 6, np.nan])
    ax = gs.grade_tonnage_plot(data, 'g', cutoffs=[2.], block_tonnes=10)
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [30.])
    assert ax.get_ylabel() == 'Tonnes Above Cutoff'
    plt.close('all')


def test_grade_tonnage_plot_models_in_the_order_given():
    data = gs.ExampleData('3d_estimate')
    ax = gs.grade_tonnage_plot(data, ['True', 'Estimate'], labels=['Data', 'Estimate'])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        'Data - Fraction', 'Estimate - Fraction', 'Data - Grade', 'Estimate - Grade']
    assert len(ax.get_lines()) == 2 and len(_gt_twin(ax).get_lines()) == 2
    assert ax.get_ylabel() == 'Fraction Above Cutoff' and _gt_twin(ax).get_ylabel() == 'Mean Grade Above Cutoff'
    # a list of tables gives the same curves as the columns of one table
    ax2 = gs.grade_tonnage_plot([data.data, data.data], ['True', 'Estimate'])
    for a, b in zip(ax.get_lines(), ax2.get_lines()):
        np.testing.assert_allclose(a.get_ydata(), b.get_ydata())
    plt.close('all')


def test_grade_tonnage_plot_default_cutoffs_and_log_axis():
    data = pd.DataFrame({'a': [0.5, 1., 2., 8.], 'b': [0.25, 1., 4., 16.]})
    ax = gs.grade_tonnage_plot(data, ['a', 'b'], ncut=5)
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), np.linspace(0.25, 16., 5))
    ax = gs.grade_tonnage_plot(data, ['a', 'b'], ncut=3, logx=True)
    assert ax.get_xscale() == 'log'
    np.testing.assert_allclose(ax.get_lines()[0].get_xdata(), [0.25, 2., 16.])
    plt.close('all')


def test_grade_tonnage_plot_draws_on_the_axis_it_is_given_and_follows_the_parameters():
    data = gs.ExampleData('3d_estimate')
    fig, axes = plt.subplots(1, 2)
    ax = gs.grade_tonnage_plot(data, 'True', ax=axes[1], title='T')
    assert ax is axes[1] and ax.figure is fig and ax.get_title() == 'T'
    plt.close('all')
    ax = gs.grade_tonnage_plot(data, 'True')
    assert tuple(ax.figure.get_size_inches()) == tuple(gs.Parameters['plotting.grade_tonnage_plot.figsize'])
    plt.close('all')


def test_grade_tonnage_plot_each_column_of_a_table_is_a_model():
    values = np.arange(12, dtype=float).reshape(4, 3)
    ax = gs.grade_tonnage_plot(values, cutoffs=[0., 5.])
    assert len(ax.get_lines()) == 3
    ax = gs.grade_tonnage_plot(pd.DataFrame(values, columns=list('abc')))
    assert [t.get_text() for t in ax.get_legend().get_texts()][:3] == ['a - Fraction', 'b - Fraction', 'c - Fraction']
    plt.close('all')


def test_grade_tonnage_plot_without_a_labelled_model_has_no_legend_and_accepts_a_0d_array():
    data = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [2., 3., 4., 5.]})
    assert gs.grade_tonnage_plot(data, ['a', 'b'], labels=[None, None]).get_legend() is None
    assert gs.grade_tonnage_plot(data, 'a', labels=[None]).get_legend() is None
    ax = gs.grade_tonnage_plot(data, ['a', 'b'], lw=np.array(2.), alpha=np.float64(0.5))
    assert ax.get_lines()[1].get_linewidth() == 2 and ax.get_lines()[1].get_alpha() == 0.5
    plt.close('all')


def test_grade_tonnage_plot_errors_leave_no_figure_open():
    data = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    plt.close('all')
    for call in (lambda: gs.grade_tonnage_plot(data, ['a', 'c']),
                 lambda: gs.grade_tonnage_plot(data, ['a', 'b'], labels=['x']),
                 lambda: gs.grade_tonnage_plot(data, ['a', 'b'], colors=['r']),
                 lambda: gs.grade_tonnage_plot(data, ['a', 'b'], weights=[1., 1., 1.]),
                 lambda: gs.grade_tonnage_plot(data, ['a', 'b'], cutoffs=[0., 1.], logx=True),
                 lambda: gs.grade_tonnage_plot(pd.DataFrame({'a': [np.nan] * 3}), 'a', cutoffs=[0.]),
                 lambda: gs.grade_tonnage_plot(data, 'a', weights=[0., 0.], cutoffs=[0.]),
                 lambda: gs.grade_tonnage_plot(data, ['a', 'b'], alpha=[0.5]),
                 lambda: gs.grade_tonnage_plot(data, 'a', ncut=1)):
        with pytest.raises(ValueError):
            call()
    assert plt.get_fignums() == []


def _gt_by_brute_force(values, weights, cutoffs):
    values = np.asarray(values, dtype=float)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    ok = ~np.isnan(values) & ~np.isnan(weights)
    values, weights = values[ok], weights[ok]
    above = [values >= c for c in cutoffs]
    tonnage = [weights[m].sum() / weights.sum() for m in above]
    grade = [(weights[m] * values[m]).sum() / weights[m].sum() if weights[m].sum() > 0 else np.nan for m in above]
    return np.array(tonnage), np.array(grade)


@pytest.mark.parametrize('name, var', [('oilsands', ['Bitumen', 'Fines']), ('3d_estimate', ['True', 'Estimate']),
                                       ('accuracy_plot', None)])
def test_grade_tonnage_plot_matches_a_brute_force_calculation_on_the_example_data(name, var):
    data = gs.ExampleData(name)
    table = data.data if hasattr(data, 'data') and not isinstance(data, pd.DataFrame) else data
    var = list(table.columns) if var is None else var
    ax = gs.grade_tonnage_plot(data, var)
    cutoffs = ax.get_lines()[0].get_xdata()
    assert cutoffs[0] == np.nanmin(table[var].values) and cutoffs[-1] == np.nanmax(table[var].values)
    for i, column in enumerate(var):
        tonnage, grade = _gt_by_brute_force(table[column].values, None, cutoffs)
        np.testing.assert_allclose(ax.get_lines()[i].get_ydata(), tonnage, atol=1e-12)
        np.testing.assert_allclose(_gt_twin(ax).get_lines()[i].get_ydata(), grade, atol=1e-12)
        assert (np.diff(tonnage) <= 0).all() and (np.diff(grade[~np.isnan(grade)]) >= -1e-12).all()
    plt.close('all')


def test_grade_tonnage_plot_with_weights_and_missing_values_and_the_same_result_twice():
    rng = np.random.default_rng(3)
    data = pd.DataFrame({'a': rng.lognormal(size=300), 'b': rng.lognormal(size=300)})
    data.loc[:60, 'b'] = np.nan
    weights = rng.uniform(0.1, 2, 300)
    ax = gs.grade_tonnage_plot(data, ['a', 'b'], weights=weights, ncut=12)
    again = gs.grade_tonnage_plot(data, ['a', 'b'], weights=weights, ncut=12)
    cutoffs = ax.get_lines()[0].get_xdata()
    for i, column in enumerate('ab'):
        tonnage, grade = _gt_by_brute_force(data[column].values, weights, cutoffs)
        np.testing.assert_allclose(ax.get_lines()[i].get_ydata(), tonnage, atol=1e-12)
        np.testing.assert_allclose(_gt_twin(ax).get_lines()[i].get_ydata(), grade, atol=1e-12)
        np.testing.assert_array_equal(ax.get_lines()[i].get_ydata(), again.get_lines()[i].get_ydata())
    plt.close('all')


def test_grade_tonnage_plot_many_realizations_keep_a_short_legend():
    data = gs.ExampleData('accuracy_plot')
    reals = list(data.columns[1:])
    ax = gs.grade_tonnage_plot(data, [data.columns[0]] + reals, colors=['k'] + ['0.6'] * len(reals),
                               labels=['Data', 'Realizations'] + [None] * (len(reals) - 1), alpha=[1.] + [0.4] * len(reals), lw=[2.] + [1.] * len(reals))
    assert len(ax.get_lines()) == 1 + len(reals) and ax.get_lines()[1].get_alpha() == 0.4
    assert ax.get_lines()[0].get_alpha() == 1 and ax.get_lines()[0].get_linewidth() == 2
    assert ax.get_lines()[1].get_linewidth() == 1
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        'Data - Fraction', 'Realizations - Fraction', 'Data - Grade', 'Realizations - Grade']
    assert gs.grade_tonnage_plot(data, list(data.columns)).get_legend() is None  # 101 models, no labels
    assert gs.grade_tonnage_plot(data, list(data.columns), legend=True).get_legend() is not None
    plt.close('all')


def _gtcurve_loop(values, weights, trim, clip, cut_range, ncut):
    """One model as the CCG program gtcurve_sim computes it, value by value (tonnage as a fraction)"""
    kept = [(min(v, clip), w) for v, w in zip(values, weights) if not (v < trim[0] or v >= trim[1]) and w > 1e-10]
    total = sum(w for _, w in kept)
    tonnage, grade = [], []
    for i in range(ncut):
        cutoff = cut_range[0] + i / (ncut - 1) * (cut_range[1] - cut_range[0])
        above = [(v, w) for v, w in kept if v >= cutoff]
        weight = sum(w for _, w in above)
        tonnage.append(weight / total)
        grade.append(sum(v * w for v, w in above) / weight)
    return np.array(tonnage), np.array(grade)


def _gt_curves(ax):
    """The lines of the tonnage (left) and grade (right) axes of grade_tonnage_plot"""
    twin = [a for a in ax.figure.axes if a is not ax and a.get_shared_x_axes().joined(a, ax)][0]
    return ax.get_lines(), twin.get_lines(), twin


def test_grade_tonnage_plot_trim_clip_and_cut_range_as_gtcurve():
    rng = np.random.default_rng(1)
    values, weights = rng.lognormal(0, 0.8, 300) - 0.2, rng.uniform(0.5, 2, 300)
    table = pd.DataFrame({'v': values, 'w': weights})
    ax = gs.grade_tonnage_plot(table, 'v', weights='w', trim=(0.0, 5.0), clip=2.5, cut_range=(0.0, 2.5), ncut=7)
    tonnage, grade, _ = _gt_curves(ax)
    expected_t, expected_g = _gtcurve_loop(values, weights, (0.0, 5.0), 2.5, (0.0, 2.5), 7)
    np.testing.assert_allclose(tonnage[0].get_xdata(), np.linspace(0, 2.5, 7))
    np.testing.assert_allclose(tonnage[0].get_ydata(), expected_t)
    np.testing.assert_allclose(grade[0].get_ydata(), expected_g)
    # a value at the upper trimming limit is left out, one at the lower limit is kept
    ax = gs.grade_tonnage_plot(pd.DataFrame({'v': [1., 2., 3., 4.]}), 'v', trim=(1.0, 4.0), cut_range=(1., 3.), ncut=3)
    np.testing.assert_allclose(_gt_curves(ax)[0][0].get_ydata(), [1., 2 / 3, 1 / 3])
    plt.close('all')


@pytest.mark.parametrize('nsim', [5, 10, 25, 100])
def test_grade_tonnage_plot_average_and_band_as_gtcurve_sim(nsim):
    """The mean of the curves, and the interpolated percentiles of the curves (numpy.percentile), which with
    few realizations are inside the lowest and highest curve."""
    rng = np.random.default_rng(nsim)
    reals = rng.lognormal(0, 0.7, (200, nsim))
    table = pd.DataFrame(reals, columns=['r%d' % k for k in range(nsim)])
    ax = gs.grade_tonnage_plot(table, cut_range=(0.2, 2.0), ncut=9, average=True, band=(0.1, 0.9))
    curves = [_gtcurve_loop(reals[:, k], np.ones(200), (-1e9, 1e9), 1e9, (0.2, 2.0), 9) for k in range(nsim)]
    tons, grades = np.array([c[0] for c in curves]), np.array([c[1] for c in curves])
    lines, grade_lines, twin = _gt_curves(ax)
    np.testing.assert_allclose(lines[-1].get_ydata(), tons.mean(axis=0))
    np.testing.assert_allclose(grade_lines[-1].get_ydata(), grades.mean(axis=0))
    for axis, curve in ((ax, tons), (twin, grades)):
        low, high = np.percentile(curve, [10, 90], axis=0)
        vertices = axis.collections[0].get_paths()[0].vertices
        np.testing.assert_allclose(vertices[1:10, 1], low)
        np.testing.assert_allclose(vertices[11:20, 1][::-1], high)
    plt.close('all')


def test_grade_tonnage_plot_average_of_chosen_models_and_legend():
    data = pd.DataFrame({'data': [1., 2., 3.], 'a': [1., 1., 1.], 'b': [3., 3., 3.]})
    ax = gs.grade_tonnage_plot(data, labels=['Data', 'A', None], realizations=[1, 2], average=True, band=(0.05, 0.95),
                               cut_range=(1., 3.), ncut=3)
    lines, grade_lines, _ = _gt_curves(ax)
    np.testing.assert_allclose(lines[-1].get_ydata(), [1., 0.5, 0.5])  # mean of a (1, 0, 0) and b (1, 1, 1)
    names = [t.get_text() for t in ax.get_legend().get_texts()]
    assert names == ['Data - Fraction', 'A - Fraction', 'Data - Grade', 'A - Grade', 'Average - Fraction',
                     'Average - Grade', 'P5 to P95']
    # a has nothing at or above the cutoffs 2 and 3, so there is no average grade or band of the grades there
    np.testing.assert_allclose(grade_lines[-1].get_ydata(), [2., np.nan, np.nan])
    from pygeostat.plotting.grade_tonnage_plot import _band
    low, high = _band(np.array([[1., np.nan], [2., 3.]]), (0.1, 0.9))
    np.testing.assert_allclose(low, [1.1, np.nan])
    np.testing.assert_allclose(high, [1.9, np.nan])
    plt.close('all')


def test_grade_tonnage_plot_new_options_are_checked_before_a_figure_is_made():
    data = pd.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.]})
    for kwargs, message in (({'trim': (3, 1)}, 'trim'), ({'cut_range': (2, 2)}, 'cut_range'),
                            ({'band': (0.9, 0.1)}, 'band'), ({'band': (0.1, 0.9), 'realizations': [2]}, 'realizations'),
                            ({'average': True, 'realizations': []}, 'realizations')):
        with pytest.raises(ValueError, match=message):
            gs.grade_tonnage_plot(data, **kwargs)
    assert plt.get_fignums() == []


def test_grade_tonnage_plot_zeros_of_the_two_axes_are_at_the_same_height():
    data = pd.DataFrame({'a': [-5., -4., -3., -2., 3.], 'b': [-4., -3., -1., 2., 6.]})
    ax = gs.grade_tonnage_plot(data, cut_range=(-5., 3.), ncut=5)
    twin = ax.figure.axes[-1]
    heights = [-a.get_ylim()[0] / (a.get_ylim()[1] - a.get_ylim()[0]) for a in (ax, twin)]
    assert heights[0] > 0 and heights[0] == pytest.approx(heights[1])
    plt.close('all')


def test_grade_tonnage_plot_keeps_the_range_of_grades_that_do_not_reach_zero():
    data = pd.DataFrame({'a': np.linspace(5, 10, 50), 'b': np.linspace(6, 11, 50)})
    ax = gs.grade_tonnage_plot(data)
    assert ax.figure.axes[-1].get_ylim()[0] > 0
    plt.close('all')


def test_grade_tonnage_plot_axis_colours():
    """'line' colours both vertical axes like the line of one model; a pair sets each axis; None keeps the style."""
    data = pd.DataFrame({'g': [1., 2., 3., 4.]})
    ax = gs.grade_tonnage_plot(data, 'g', colors=['red'], axis_colors='line')
    twin = _gt_twin(ax)
    ax.figure.canvas.draw()
    for axis, side in ((ax, 'left'), (twin, 'right')):
        assert axis.yaxis.label.get_color() == 'red'
        assert axis.get_yticklabels()[0].get_color() == 'red'
        assert axis.yaxis.get_major_ticks()[0].tick1line.get_color() == 'red' or \
            axis.yaxis.get_major_ticks()[0].tick2line.get_color() == 'red'
        assert axis.spines[side].get_edgecolor()[:3] == (1.0, 0.0, 0.0)
    assert ax.xaxis.label.get_color() != 'red'  # only the vertical axes
    plt.close('all')
    ax = gs.grade_tonnage_plot(data, 'g', axis_colors=('navy', None))
    twin = _gt_twin(ax)
    assert ax.yaxis.label.get_color() == 'navy' and twin.yaxis.label.get_color() != 'navy'
    plt.close('all')
    with pytest.raises(ValueError, match="for one model"):
        gs.grade_tonnage_plot(gs.ExampleData('3d_estimate'), ['True', 'Estimate'], axis_colors='line')
    with pytest.raises(ValueError, match='pair'):
        gs.grade_tonnage_plot(data, 'g', axis_colors=('red',))
    assert len(plt.get_fignums()) == 0


def test_grade_tonnage_plot_axes_options():
    data = pd.DataFrame({'g': [1., 2., 3., 4.]})
    ax = gs.grade_tonnage_plot(data, 'g', ticks=6, fontsize={'label': 15, 'tick': 8}, background='lightgrey')
    twin = _gt_twin(ax)
    ax.figure.canvas.draw()
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_markersize() == 6
    assert twin.yaxis.get_major_ticks()[0].tick2line.get_markersize() == 6
    assert ax.yaxis.label.get_fontsize() == 15 and twin.yaxis.label.get_fontsize() == 15
    assert twin.get_yticklabels()[0].get_fontsize() == 8
    assert tuple(ax.get_facecolor()[:3]) == (0.8274509803921568,) * 3
    plt.close('all')


def test_grade_tonnage_plot_limits_of_the_two_vertical_axes():
    """ylim sets the tonnage axis (left) and ylim2 the grade axis (right); without them the zeros still line up."""
    data = pd.DataFrame({'g': [1., 2., 3., 4.]})
    ax = gs.grade_tonnage_plot(data, 'g', ylim=(-0.2, 2), ylim2=(0, 10))
    twin = _gt_twin(ax)
    assert ax.get_ylim() == (-0.2, 2) and twin.get_ylim() == (0, 10)
    assert ax.get_xlim() == gs.grade_tonnage_plot(data, 'g').get_xlim()  # the x axis is not touched
    plt.close('all')
    ax = gs.grade_tonnage_plot(data, 'g', ylim=(0, 3))
    assert ax.get_ylim() == (0, 3) and _gt_twin(ax).get_ylim() != (0, 3)  # only the left one is set
    plt.close('all')
    for kwargs, message in (({'ylim': (2, 1)}, 'ylim must be'), ({'ylim2': (1, 2, 3)}, 'ylim2 must be')):
        with pytest.raises(ValueError, match=message):
            gs.grade_tonnage_plot(data, 'g', **kwargs)
    assert plt.get_fignums() == []

def _scatter_3d_data():
    return pd.DataFrame({'a': [0., 1., 2., 3., 4., np.nan], 'b': [1., 2., 3., 4., 5., 6.],
                         'c': [5., 4., 3., 2., 1., 0.], 'd': [10., 20., 30., 40., 50., 60.],
                         'k': ['p', 'q', 'p', 'q', 'p', 'q']})


def test_scatter_plot_3d_points_and_labels():
    ax = gs.scatter_plot_3d(_scatter_3d_data(), 'a', 'b', 'c')
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ('a', 'b', 'c')
    assert ax._draw_grid
    assert not gs.scatter_plot_3d(_scatter_3d_data(), 'a', 'b', 'c', grid=False)._draw_grid
    x, y, z = ax.collections[0]._offsets3d
    np.testing.assert_allclose(x, [0., 1., 2., 3., 4.])  # the row with a NaN is left out
    np.testing.assert_allclose(y, [1., 2., 3., 4., 5.])
    np.testing.assert_allclose(z, [5., 4., 3., 2., 1.])
    ax = gs.scatter_plot_3d(_scatter_3d_data(), 'a', 'b', 'c', xlabel='A', title='T', elev=10, azim=20)
    assert ax.get_xlabel() == 'A' and ax.get_title() == 'T' and (ax.elev, ax.azim) == (10, 20)
    plt.close('all')


def test_scatter_plot_3d_colour_variable_and_limits():
    ax = gs.scatter_plot_3d(_scatter_3d_data(), 'a', 'b', 'c', c='d', clim=(0, 100))
    points = ax.collections[0]
    assert points.get_clim() == (0, 100)
    np.testing.assert_allclose(points.get_array(), [10., 20., 30., 40., 50.])
    assert ax.figure.axes[-1].get_ylabel() == 'd'  # the colour bar
    ax = gs.scatter_plot_3d(_scatter_3d_data(), 'a', 'b', 'c', c='d', cbar=False)
    assert len(ax.figure.axes) == 1
    ax = gs.scatter_plot_3d(_scatter_3d_data(), 'a', 'b', 'c', c='red')
    assert len(ax.figure.axes) == 1 and ax.collections[0].get_array() is None
    plt.close('all')


def test_scatter_plot_3d_shapes_by_category():
    ax = gs.scatter_plot_3d(_scatter_3d_data(), 'a', 'b', 'c', cat='k')
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['p', 'q']
    assert ax.get_legend().get_title().get_text() == 'k'
    assert [len(p._offsets3d[0]) for p in ax.collections] == [3, 2]  # p: rows 0, 2, 4; q: 1, 3
    shapes = [p.get_paths()[0].vertices for p in ax.collections]
    circle = matplotlib.markers.MarkerStyle('o').get_path().vertices
    triangle = matplotlib.markers.MarkerStyle('^').get_path().vertices
    assert shapes[0].shape == circle.shape and shapes[1].shape == triangle.shape
    ax = gs.scatter_plot_3d(_scatter_3d_data().assign(k=[2., 1., 2., 1., 2., 1.]), 'a', 'b', 'c',
                            cat='k', markers=['s', 'D'])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['1', '2']  # sorted, 1 for 1.0
    assert [len(p._offsets3d[0]) for p in ax.collections] == [2, 3]
    plt.close('all')


def test_scatter_plot_3d_errors_leave_no_figure():
    data = _scatter_3d_data()
    with pytest.raises(ValueError, match="'nope'"):
        gs.scatter_plot_3d(data, 'a', 'b', 'nope')
    with pytest.raises(ValueError, match="'nope'"):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='nope')
    with pytest.raises(ValueError, match='neither a column'):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', c='nope')
    with pytest.raises(ValueError, match='3D axis'):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', ax=plt.figure().add_subplot())
    plt.close('all')
    with pytest.raises(ValueError, match='2 categories and 1 markers'):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k', markers=['o'])
    with pytest.raises(ValueError, match='No row'):
        gs.scatter_plot_3d(data.assign(a=np.nan), 'a', 'b', 'c')
    with pytest.raises(ValueError, match='DataFile or a pd.DataFrame'):
        gs.scatter_plot_3d(data.values, 'a', 'b', 'c')
    assert plt.get_fignums() == []


def test_scatter_plot_3d_axis_datafile_and_output(plot_output_dir):
    fig = plt.figure()
    ax3 = fig.add_subplot(projection='3d')
    out = os.path.join(plot_output_dir, 'scatter3d.png')
    data = gs.ExampleData('3d_correlation')
    ax = gs.scatter_plot_3d(data, 'Var1', 'Var2', 'Var3', c='PC1', ax=ax3, output_file=out)
    assert ax is ax3 and plt.get_fignums() == [fig.number] and os.path.isfile(out)
    assert len(ax.collections[0]._offsets3d[0]) == len(data.data)
    plt.close('all')


def test_scatter_plot_3d_colour_by_category():
    data = _scatter_3d_data().assign(k=['p', 'q', 'r', 'p', 'q', 'r'], a=[0., 1., 2., 3., 4., 5.])
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k', cat_as='color')
    palette = plt.get_cmap('tab10')
    assert [tuple(p.get_facecolor()[0][:3]) for p in ax.collections] == [palette(i)[:3] for i in range(3)]
    # the shape is the same for all the categories, the legend shows each colour
    assert len({p.get_paths()[0].vertices.shape for p in ax.collections}) == 1
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['p', 'q', 'r']
    assert [h.get_markerfacecolor() for h in ax.get_legend().legend_handles] == [palette(i) for i in range(3)]
    # given colours, and colour with shape
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k', cat_as='both', colors=['red', 'green', 'blue'])
    assert [tuple(p.get_facecolor()[0][:3]) for p in ax.collections] == [(1., 0., 0.), (0., 0.5019607843137255, 0.), (0., 0., 1.)]
    assert len({p.get_paths()[0].vertices.shape for p in ax.collections}) == 3
    # shapes are still the default, with one colour
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k')
    assert len({tuple(p.get_facecolor()[0]) for p in ax.collections}) == 1
    plt.close('all')


def test_scatter_plot_3d_colour_by_category_with_many_categories_and_errors():
    data = pd.DataFrame({'a': np.arange(12.), 'b': np.arange(12.) ** 2, 'c': np.arange(12.)[::-1], 'k': np.arange(12)})
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k', cat_as='color')  # more than 10: the colours of plotting.cmap_cat
    cmap = plt.get_cmap(gs.Parameters['plotting.cmap_cat'])
    assert tuple(ax.collections[11].get_facecolor()[0][:3]) == cmap(11)[:3]
    plt.close('all')
    with pytest.raises(ValueError, match='cat_as'):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k', cat_as='size')
    with pytest.raises(ValueError, match='cannot also be colored'):
        gs.scatter_plot_3d(data.assign(d=1.), 'a', 'b', 'c', c='d', cat='k', cat_as='color')
    with pytest.raises(ValueError, match='12 categories and 3 colors'):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k', cat_as='color', colors=['r', 'g', 'b'])
    assert plt.get_fignums() == []


def test_scatter_plot_3d_drop_lines_reach_the_floor():
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    data = _scatter_3d_data()
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c')
    assert not [c for c in ax.collections if isinstance(c, Line3DCollection)]
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', drop_lines=True)
    lines = [c for c in ax.collections if isinstance(c, Line3DCollection)]
    assert len(lines) == 1 and len(lines[0]._segments3d) == 5  # the row with a NaN has no line
    floor = ax.get_zlim()[0]
    for (x0, y0, z0), (x1, y1, z1) in lines[0]._segments3d:
        assert (x0, y0) == (x1, y1) and z0 == floor and z1 >= floor
    np.testing.assert_allclose(sorted(s[1][2] for s in lines[0]._segments3d), [1., 2., 3., 4., 5.])
    plt.close('all')


def test_scatter_plot_3d_shape_only_points_are_dark_and_the_legend_title_is_as_large_as_the_entries():
    data = pd.DataFrame({'a': np.arange(6.), 'b': np.arange(6.) ** 2, 'c': np.arange(6.)[::-1],
                         'k': ['p', 'q'] * 3, 'v': np.arange(6.)})
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k')
    assert {tuple(p.get_facecolor()[0][:3]) for p in ax.collections} == {(0.15, 0.15, 0.15)}
    legend = ax.get_legend()
    sizes = {t.get_fontsize() for t in legend.get_texts()} | {legend.get_title().get_fontsize()}
    assert len(sizes) == 1
    assert sizes == {ax.xaxis.get_ticklabels()[0].get_size()}
    # with a colour the points keep it
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k', c='v')
    assert len({tuple(p.get_facecolor()[0][:3]) for p in ax.collections}) > 1
    plt.close('all')


def test_scatter_plot_3d_colour_bar_is_close_to_the_plot():
    data = pd.DataFrame({'a': np.arange(6.), 'b': np.arange(6.) ** 2, 'c': np.arange(6.)[::-1], 'v': np.arange(6.)})
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', c='v')
    ax.figure.canvas.draw()
    bar = ax.figure.axes[-1]
    assert bar.get_position().x0 - ax.get_position().x1 < 0.1
    plt.close('all')


def test_scatter_plot_3d_legend_shows_the_colour_given_in_c():
    data = pd.DataFrame({'a': np.arange(6.), 'b': np.arange(6.) ** 2, 'c': np.arange(6.)[::-1], 'k': ['p', 'q'] * 3})
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', cat='k', c='red')
    assert {h.get_markerfacecolor() for h in ax.get_legend().legend_handles} == {'red'}
    plt.close('all')


def test_scatter_plot_3d_several_data_sets():
    """A list of data sets gives each its own colour and legend entry, in the order given."""
    first, second = _scatter_3d_data(), _scatter_3d_data().assign(a=lambda d: d['a'] + 10)
    ax = gs.scatter_plot_3d([second, first], 'a', 'b', 'c', labels=['Zeta', 'Alpha'])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Zeta', 'Alpha']  # not alphabetical
    assert ax.get_legend().get_title().get_text() == ''
    assert len(ax.collections) == 2
    colours = [tuple(c.get_facecolor()[0][:3]) for c in ax.collections]
    assert colours[0] != colours[1]
    assert [len(c._offsets3d[0]) for c in ax.collections] == [5, 5]  # the row with a NaN is left out of each
    plt.close('all')
    ax = gs.scatter_plot_3d([first, second], 'a', 'b', 'c')
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Data 1', 'Data 2']
    plt.close('all')
    ax = gs.scatter_plot_3d([first, second], 'a', 'b', 'c', cat_as='both')  # colour and shape
    assert len({c.get_paths()[0].vertices.tobytes() for c in ax.collections}) == 2
    plt.close('all')


def test_scatter_plot_3d_several_data_sets_errors_leave_no_figure():
    plt.close('all')
    first = _scatter_3d_data()
    with pytest.raises(ValueError, match='cannot be used with a list'):
        gs.scatter_plot_3d([first, first], 'a', 'b', 'c', cat='a')
    with pytest.raises(ValueError, match='one name for each'):
        gs.scatter_plot_3d([first, first], 'a', 'b', 'c', labels=['only one'])
    with pytest.raises(ValueError, match='every data set'):
        gs.scatter_plot_3d([first, first.drop(columns='b')], 'a', 'b', 'c')
    with pytest.raises(ValueError, match='must hold'):
        gs.scatter_plot_3d([first, 3], 'a', 'b', 'c')
    with pytest.raises(ValueError, match='different for each'):
        gs.scatter_plot_3d([first, first], 'a', 'b', 'c', labels=['x', 'x'])
    nothing = first.assign(a=np.nan)  # no row is left once the NaN are dropped
    with pytest.raises(ValueError, match="'Data 2' has no row"):
        gs.scatter_plot_3d([first, nothing, first], 'a', 'b', 'c')
    assert len(plt.get_fignums()) == 0


def test_scatter_plot_3d_axes_options():
    """Ticks, font size, limits and the colour of the walls; drop lines go down to the lower z limit."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    data = _scatter_3d_data()
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', ticks=7, fontsize={'label': 14, 'tick': 8}, xlim=(-1, 9),
                            ylim=(-2, 20), zlim=(-5, 10), background='white', drop_lines=True)
    ax.figure.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()) == ((-1, 9), (-2, 20), (-5, 10))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert axis.get_major_ticks()[0].tick1line.get_markersize() == 7
        assert axis.label.get_fontsize() == 14
        assert axis.get_major_ticks()[0].label1.get_fontsize() == 8
        assert tuple(axis.pane.get_facecolor()) == (1.0, 1.0, 1.0, 1.0)
    lines = [c for c in ax.collections if isinstance(c, Line3DCollection)]
    assert {s[0][2] for s in lines[0]._segments3d} == {-5}
    with pytest.raises(ValueError, match='zlim must be'):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', zlim=(1,))
    plt.close('all')


def test_scatter_plot_3d_log_axes():
    """A log axis shows the log10 of the values with the values as tick labels; limits are given as values."""
    plt.close('all')
    data = pd.DataFrame({'a': [0.1, 1., 10., 100.], 'b': [1., 2., 3., 4.], 'c': [1., 10., 100., 1000.]})
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', logx=True, logz=True, xlim=(0.01, 1000))
    ax.figure.canvas.draw()
    np.testing.assert_allclose(ax.get_xlim(), (-2, 3))
    np.testing.assert_allclose(sorted(ax.collections[0]._offsets3d[0]), np.log10(data['a']))  # the points are log10
    np.testing.assert_allclose(sorted(ax.collections[0]._offsets3d[1]), data['b'])  # b is not
    assert [t.get_text() for t in ax.xaxis.get_ticklabels()] == ['0.01', '0.1', '1', '10', '100', '1000']
    assert [t.get_text() for t in ax.zaxis.get_ticklabels()] == ['1', '10', '100', '1000']  # a range of over 3 decades
    assert '.' in ''.join(t.get_text() for t in ax.yaxis.get_ticklabels())  # the other axis is as before
    plt.close('all')
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', logx=True, stat_blk=['mean'])
    assert ax.texts[0].get_text() == 'Mean (x, y, z) = %.3g, %.3g, %.3g' % tuple(data[['a', 'b', 'c']].mean())  # not of the logs
    plt.close('all')
    with pytest.raises(ValueError, match="log scale of 'a' needs values above zero"):
        gs.scatter_plot_3d(data.assign(a=[0., 1, 2, 3]), 'a', 'b', 'c', logx=True)
    with pytest.raises(ValueError, match='lower limit of a log scale'):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', logx=True, xlim=(0, 10))
    assert len(plt.get_fignums()) == 0


def test_scatter_plot_3d_stat_blk_chooses_the_statistics():
    """No statistics by default; all, minimal or a list write them in the corner, one block for each data set."""
    plt.close('all')
    data = _scatter_3d_data().dropna()
    assert len(gs.scatter_plot_3d(data, 'a', 'b', 'c').texts) == 0
    lines = lambda **kw: [t.get_text().split('\n') for t in gs.scatter_plot_3d(data, 'a', 'b', 'c', **kw).texts]
    (all_lines,) = lines(stat_blk='all')
    assert [l.split(' =')[0] for l in all_lines] == ['n', r'$\rho$ (xy, xz, yz)', 'Mean (x, y, z)', 'Var (x, y, z)']
    assert all_lines[0] == 'n = %d' % len(data)
    assert all_lines[2] == 'Mean (x, y, z) = %.3g, %.3g, %.3g' % tuple(data[['a', 'b', 'c']].mean())
    (minimal,) = lines(stat_blk='minimal')
    assert len(minimal) == 1 and minimal[0].startswith(r'$\rho$')
    (named,) = lines(stat_blk=['var', 'ndat'])
    assert [l.split(' =')[0] for l in named] == ['n', 'Var (x, y, z)']
    assert lines(stat_blk='none') == [] and lines(stat_blk=False) == []
    plt.close('all')
    numbers = data[['a', 'b', 'c']]
    ax = gs.scatter_plot_3d([numbers, numbers + 3], 'a', 'b', 'c', stat_blk='minimal', labels=['P', 'Q'])
    assert len(ax.texts) == 2 and ax.texts[0].get_color() != ax.texts[1].get_color()
    assert ax.texts[1].xyann[1] < ax.texts[0].xyann[1] == 0  # the second block is under the first
    plt.close('all')
    with pytest.raises(ValueError, match="'median' is not a valid stat_blk"):
        gs.scatter_plot_3d(data, 'a', 'b', 'c', stat_blk=['median'])
    assert len(plt.get_fignums()) == 0


def test_scatter_plot_3d_log_ticks_for_a_short_range():
    """A range of less than three decades gets ticks at 1, 2 and 5 times the powers of ten."""
    from pygeostat.plotting.scatter_plot_3d import _log_ticks
    positions, labels = _log_ticks(np.log10(0.8), np.log10(60))
    assert labels == ['1', '2', '5', '10', '20', '50']
    np.testing.assert_allclose(positions, np.log10([1, 2, 5, 10, 20, 50]))
    assert _log_ticks(np.log10(1.1), np.log10(3.9))[1] == ['2', '3']


def test_scatter_plot_3d_log_axis_of_a_very_narrow_range_still_has_ticks():
    """A range with no 1, 2 or 5 times a power of ten in it is labelled at its two ends."""
    plt.close('all')
    data = pd.DataFrame({'a': [1.01, 1.015, 1.02], 'b': [1., 2., 3.], 'c': [1., 2., 3.]})
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', logx=True)
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
    assert len(labels) == 2 and 1.0 <= float(labels[0]) <= 1.02 and 1.0 <= float(labels[1]) <= 1.03
    plt.close('all')


def _secondary_twin(ax):
    return [a for a in ax.figure.axes if a is not ax and a.get_shared_x_axes().joined(a, ax)][0]


def test_secondary_axis_plot_lines_on_each_axis():
    """The lines of y are on the left axis, the lines of y2 on the right one, which is ax.twin."""
    plt.close('all')
    x = np.arange(5.)
    ax = gs.secondary_axis_plot(x, [x, x ** 2], 10 * x + 1, labels=['a', 'b'], labels2=['c'], xlabel='X',
                                ylabel='Left', ylabel2='Right')
    twin = _secondary_twin(ax)
    assert ax.twin is twin
    assert len(ax.get_lines()) == 2 and len(twin.get_lines()) == 1
    np.testing.assert_allclose(ax.get_lines()[1].get_ydata(), x ** 2)
    np.testing.assert_allclose(twin.get_lines()[0].get_ydata(), 10 * x + 1)
    assert (ax.get_xlabel(), ax.get_ylabel(), twin.get_ylabel()) == ('X', 'Left', 'Right')
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['a', 'b', 'c']
    # the right axis is dashed and takes the next colours of the cycle by default
    assert twin.get_lines()[0].get_linestyle() == '--' and ax.get_lines()[0].get_linestyle() == '-'
    assert twin.get_lines()[0].get_color() == 'C2'
    plt.close('all')


def test_secondary_axis_plot_table_input_and_default_labels():
    """A table gives a line for each column, named by the column; arrays are numbered, left lines first."""
    plt.close('all')
    x = np.arange(4.)
    ax = gs.secondary_axis_plot(x, pd.DataFrame({'p': x, 'q': 2 * x}), np.column_stack([x, x]))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['p', 'q', 'Line 3', 'Line 4']
    plt.close('all')
    ax = gs.secondary_axis_plot(x, pd.Series(x, name='only'), x + 1)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['only', 'Line 2']
    plt.close('all')
    assert gs.secondary_axis_plot(x, x, x + 1, legend=False).get_legend() is None
    plt.close('all')


def test_secondary_axis_plot_styles_limits_and_axis_colours():
    plt.close('all')
    x = np.arange(5.)
    ax = gs.secondary_axis_plot(x, x, x + 5, colors=['r'], colors2=['b'], ls='-.', ls2=':', lw=[3, 4], alpha=0.5,
                                ylim=(-1, 9), ylim2=(0, 20), xlim=(0, 3), axis_colors='line')
    twin = _secondary_twin(ax)
    left, right = ax.get_lines()[0], twin.get_lines()[0]
    assert (left.get_color(), right.get_color()) == ('r', 'b')
    assert (left.get_linestyle(), right.get_linestyle()) == ('-.', ':')
    assert (left.get_linewidth(), right.get_linewidth(), left.get_alpha()) == (3, 3, 0.5)
    assert ax.get_ylim() == (-1, 9) and twin.get_ylim() == (0, 20) and ax.get_xlim() == (0, 3)
    assert ax.yaxis.label.get_color() == 'r' and twin.yaxis.label.get_color() == 'b'
    assert tuple(ax.spines['left'].get_edgecolor()) == (1.0, 0.0, 0.0, 1.0) and tuple(twin.spines['right'].get_edgecolor()) == (0.0, 0.0, 1.0, 1.0)
    plt.close('all')


def test_secondary_axis_plot_zeros_line_up_unless_told_not_to():
    plt.close('all')
    x = np.arange(5.)
    ax = gs.secondary_axis_plot(x, x - 2, 10 * x)
    (lo_l, hi_l), (lo_r, hi_r) = ax.get_ylim(), _secondary_twin(ax).get_ylim()
    np.testing.assert_allclose(-lo_l / (hi_l - lo_l), -lo_r / (hi_r - lo_r))  # zero at the same height
    plt.close('all')
    ax = gs.secondary_axis_plot(x, x - 2, 10 * x, align_zeros=False)
    (lo_l, hi_l), (lo_r, hi_r) = ax.get_ylim(), _secondary_twin(ax).get_ylim()
    assert not np.isclose(-lo_l / (hi_l - lo_l), -lo_r / (hi_r - lo_r))
    plt.close('all')


def test_secondary_axis_plot_bands_and_axes_options():
    plt.close('all')
    x = np.arange(4.)
    ax = gs.secondary_axis_plot(x, x, x, fill=(x - 1, x + 1), fill_label='Range', ticks=6, fontsize={'label': 14},
                                ylabel='L', ylabel2='R', background='lightgrey', logx=False)
    ax.figure.canvas.draw()
    assert len(ax.collections) == 1 and len(_secondary_twin(ax).collections) == 0
    assert [t.get_text() for t in ax.get_legend().get_texts()][-1] == 'Range'
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_markersize() == 6
    assert _secondary_twin(ax).yaxis.get_major_ticks()[0].tick2line.get_markersize() == 6
    assert ax.yaxis.label.get_fontsize() == 14 and _secondary_twin(ax).yaxis.label.get_fontsize() == 14
    assert tuple(ax.get_facecolor()[:3]) == (0.8274509803921568,) * 3
    plt.close('all')


def test_secondary_axis_plot_errors_leave_no_figure():
    plt.close('all')
    x = np.arange(4.)
    for kwargs, message in (({'y': x[:3]}, 'one value for each of the 4'), ({'y2': []}, 'no line|`y2`'),
                            ({'labels': ['a', 'b']}, 'one entry for each line'),
                            ({'colors': ['r', 'g']}, 'one color for each line'),
                            ({'axis_colors': 'line', 'y': [x, x]}, "axis_colors='line' is for one line"),
                            ({'axis_colors': ('r',)}, 'pair'), ({'ylim': (2, 1)}, 'ylim must be'),
                            ({'ylim2': (1, 2, 3)}, 'ylim2 must be'), ({'lw': [1, 2], 'y': [x, x, x]}, 'one entry'),
                            ({'fill': (x, x[:2])}, 'fill must be')):
        args = dict(x=x, y=x, y2=x + 1)
        args.update(kwargs)
        with pytest.raises(ValueError, match=message):
            gs.secondary_axis_plot(**args)
    assert len(plt.get_fignums()) == 0


def test_secondary_axis_plot_band_colour_and_legend_arguments():
    """fill_color, fill_alpha and legend_kws replace the values that were fixed in the code; the defaults are unchanged."""
    plt.close('all')
    x = np.arange(5.)
    ax = gs.secondary_axis_plot(x, x, 2 * x, fill=(x - 1, x + 1), fill_label='Band', legend=True)
    assert tuple(np.round(ax.collections[0].get_facecolor()[0], 2)) == (0.5, 0.5, 0.5, 0.35)
    assert not ax.get_legend().get_frame_on() and ax.get_legend().get_title().get_text() == ''
    plt.close('all')
    ax = gs.secondary_axis_plot(x, x, 2 * x, fill=(x - 1, x + 1), fill_label='Band', legend=True, fill_color='r',
                                fill_alpha=0.6, legend_kws=dict(frameon=True, title='Lines'))
    assert tuple(np.round(ax.collections[0].get_facecolor()[0], 2)) == (1.0, 0.0, 0.0, 0.6)
    assert tuple(np.round(ax.get_legend().legend_handles[-1].get_facecolor(), 2)) == (1.0, 0.0, 0.0, 0.6)
    assert ax.get_legend().get_frame_on() and ax.get_legend().get_title().get_text() == 'Lines'
    plt.close('all')


def test_grade_tonnage_plot_line_styles_label_average_band_and_legend_arguments():
    """ls, ls2, ylabel2, average_color, average_lw, fill_color, fill_alpha and legend_kws replace the values that were
    fixed in the code; the defaults are unchanged."""
    plt.close('all')
    data = pd.DataFrame({'data': [1., 2., 3.], 'a': [1., 1., 1.], 'b': [3., 3., 3.]})
    base = dict(labels=['Data', 'A', None], realizations=[1, 2], average=True, band=(0.05, 0.95), cut_range=(1., 3.),
                ncut=3)
    ax = gs.grade_tonnage_plot(data, **base)
    lines, grade_lines, twin = _gt_curves(ax)
    assert (lines[0].get_linestyle(), grade_lines[0].get_linestyle()) == ('--', '-')
    assert (lines[-1].get_color(), lines[-1].get_linewidth()) == ('k', 2)
    assert twin.get_ylabel() == 'Mean Grade Above Cutoff'
    assert tuple(np.round(ax.collections[0].get_facecolor()[0], 2)) == (0.5, 0.5, 0.5, 0.35)
    assert not ax.get_legend().get_frame_on()
    plt.close('all')
    ax = gs.grade_tonnage_plot(data, ls=':', ls2='-.', average_color='r', average_lw=3, ylabel2='Grade (%)',
                               fill_color='b', fill_alpha=0.5, legend_kws=dict(frameon=True), **base)
    lines, grade_lines, twin = _gt_curves(ax)
    assert (lines[0].get_linestyle(), grade_lines[0].get_linestyle()) == (':', '-.')
    assert (lines[-1].get_color(), lines[-1].get_linewidth(), grade_lines[-1].get_linestyle()) == ('r', 3, '-.')
    assert twin.get_ylabel() == 'Grade (%)'
    assert tuple(np.round(ax.collections[0].get_facecolor()[0], 2)) == (0.0, 0.0, 1.0, 0.5)
    assert ax.get_legend().get_frame_on()
    assert [h.get_linestyle() for h in ax.get_legend().legend_handles[:2]] == [':', ':']  # the tonnage of Data and A
    plt.close('all')


def test_swath_plot_label_bar_width_spacing_and_margin_arguments():
    """label_kws, bar_width, hspace and label_margin replace the values that were fixed in the code; the defaults are
    unchanged."""
    plt.close('all')
    kwargs = dict(x='x', y='y', z='z', nbins=2)
    ax = gs.swath_plot(_swath_data(), ['a', 'b'], orient='x', **kwargs)
    bars = _swath_twin(ax).patches[0].get_width()
    assert {(t.get_fontweight(), t.get_fontsize() == ax.yaxis.get_ticklabels()[0].get_size()) for t in ax.texts} == {('bold', True)}
    assert ax.margins()[1] == 0.16
    assert gs.swath_plot(_swath_data(), ['a', 'b'], orient=['x', 'y'], **kwargs)[0].figure.subplotpars.hspace == 0.35
    plt.close('all')
    ax = gs.swath_plot(_swath_data(), ['a', 'b'], orient='x', bar_width=0.2, label_margin=0.3,
                       label_kws=dict(fontweight='normal', fontsize=9), **kwargs)
    assert _swath_twin(ax).patches[0].get_width() == pytest.approx(bars * 0.2 / 0.08)
    assert {(t.get_fontweight(), t.get_fontsize()) for t in ax.texts} == {('normal', 9)}
    assert ax.margins()[1] == 0.3
    plt.close('all')
    axes = gs.swath_plot(_swath_data(), ['a', 'b'], orient=['x', 'y'], hspace=0.6, **kwargs)
    assert axes[0].figure.subplotpars.hspace == 0.6
    plt.close('all')


def test_cross_plot_alpha_legend_45_line_and_statistics_box_arguments():
    """alpha, legend_kws, line45_kws and stat_bbox replace the values that were fixed in the code; the defaults are
    unchanged."""
    plt.close('all')
    x, y, x2, y2 = _two_sets()
    ax = gs.cross_plot(x, y, x2=x2, y2=y2, line45=True)
    line = [l for l in ax.lines if l.get_linestyle() == '--'][0]
    assert (line.get_color(), line.get_linewidth(), line.get_zorder()) == ('k', 1, 1)
    assert [c.get_alpha() for c in ax.collections] == [0.6, 0.6] and not ax.get_legend().get_frame_on()
    assert gs.cross_plot(x, y).texts[0].get_bbox_patch().get_facecolor() == (1.0, 1.0, 1.0, 0.75)
    plt.close('all')
    ax = gs.cross_plot(x, y, x2=x2, y2=y2, line45=True, alpha=0.3, legend_kws=dict(frameon=True),
                       line45_kws=dict(color='r', lw=3))
    line = [l for l in ax.lines if l.get_linestyle() == '--'][0]
    assert (line.get_color(), line.get_linewidth(), line.get_zorder()) == ('r', 3, 1)
    assert [c.get_alpha() for c in ax.collections] == [0.3, 0.3] and ax.get_legend().get_frame_on()
    assert {p.get_alpha() for p in ax.histx.patches + ax.histy.patches} == {0.3}
    assert gs.cross_plot(x, y, stat_bbox=dict(facecolor='yellow')).texts[0].get_bbox_patch().get_facecolor() == (1.0, 1.0, 0.0, 0.75)
    plt.close('all')


def test_scatter_plot_3d_edge_drop_line_colourbar_legend_statistics_and_tick_arguments():
    """edge_lw, drop_kws, cbar_kws, legend_kws, stat_xy and nticks replace the values that were fixed in the code; the
    defaults are unchanged."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    plt.close('all')
    data = _scatter_3d_data().dropna()
    kwargs = dict(c='d', cat='k', drop_lines=True, stat_blk='minimal')
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', **kwargs)
    drop = [c for c in ax.collections if isinstance(c, Line3DCollection)][0]
    assert drop.get_linewidths()[0] == 0.4 and ax.collections[0].get_linewidths()[0] == 0.4
    assert not ax.get_legend().get_frame_on() and ax.texts[0].xy == (0.98, 0.98)
    bar_height = ax.figure.axes[-1].get_position().height
    default_ticks = len(ax.get_xticks())
    plt.close('all')
    ax = gs.scatter_plot_3d(data, 'a', 'b', 'c', edge_lw=1.5, drop_kws=dict(linewidths=2), cbar_kws=dict(shrink=1.0),
                            legend_kws=dict(frameon=True), stat_xy=(0.5, 0.9), nticks=2, **kwargs)
    drop = [c for c in ax.collections if isinstance(c, Line3DCollection)][0]
    assert drop.get_linewidths()[0] == 2 and ax.collections[0].get_linewidths()[0] == 1.5
    assert ax.get_legend().get_frame_on() and ax.texts[0].xy == (0.5, 0.9)
    assert ax.figure.axes[-1].get_position().height > bar_height
    assert len(ax.get_xticks()) < default_ticks
    plt.close('all')
