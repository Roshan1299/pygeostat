# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 3D viewer (optional, `pip install pygeostat[view3d]`): `gs.view3d()`, `gs.Viewer`, `gs.drillhole_mesh()`
  and `gs.block_mesh()` for drill holes, block models and sub-blocks, with a user guide page (#147)
- Interactive layer panel for the 3D viewer (optional, `pip install "pygeostat[view3d-panel]"`):
  `Viewer.panel()`, with a slicer (a plane through the scene), and `Viewer.set_layer()`, `set_slicer()`,
  `set_axis_origin()`, `get_view()` and `set_view()` to do the same from a script (#149)
- Display options and picking for the 3D layer panel: colormap, colour by another variable, show only some
  categories, cell edges, tube radius, vertical exaggeration, background, scene boxes and ruler ticks, camera
  views and fit view, picking a drill hole, and screenshot and HTML export; from a script with
  `Viewer.set_background()`, `set_scene()`, `set_axis_ticks()`, `fit_view()`, `set_parallel_projection()`,
  `set_camera_view()`, `interval_info()`, `save_screenshot()` and `export_html()`, new arguments of
  `Viewer.set_layer()` and the `other_vars` argument of `gs.view3d()` (#151)
- Shared plot options in `pygeostat.plotting.utils` (`_format_ticks`, `_format_fontsize`, `_format_axes`) and the parameter `gs.Parameters['plotting.ticks']` (default `True`), so that plots offer `ticks`, `fontsize`, `xlim`, `ylim`, `grid` and `background` with the same meaning
- `histogram_plot()` takes `ticks`, `fontsize` and `background`, next to its existing `xlim`, `ylim` and `grid`
- `validation_plot()` takes `outliers` and `outlier_color` to draw the points furthest from the 45 degree line in colour with their index, `outlier_labels` and `reg_line` to leave out those numbers and the regression line, and the shared options `ticks`, `fontsize`, `background`
- `histogram_plot_simulation()` shows the number of values per realization (`'realn'` in `stat_blk`) and takes `ylim`, `grid`, `ticks`, `fontsize` and `background`; `variogram_plot_simulation()` takes `grid`, `ticks`, `fontsize` and `background`, so the reproduction plots have the same options as the other plots
- `histogram_plot_simulation()` takes `lw_real`, `ylabel` and `stat_fontsize`, and `variogram_plot_simulation()` takes `sill_color`, `sill_lw` and `real_label`, for values that were fixed in the code. The defaults are unchanged
- `accuracy_plot()` takes `color` and `label` to compare models on one axis, `reals_gaussian`, `data`, `weights` and `global_variance` for the precision and goodness
- `gs.cross_plot()`: scatter plot of two variables in the layout of Figure 9 of Castro Moreira, Costa and Deutsch (2026), a square plot with the histogram of x below it and the histogram of y to its right, the number of pairs and the correlations on the plot (in the corner with the fewest points) and the mean and standard deviation beside each histogram (those of the logarithm on a log axis, where the correlation is written `rho_ln`), with an optional second data set (`x2`, `y2`), log scales (`logx`, `logy`, `log_counts`), point size, statistics chosen with `stat_blk` (`'all'`, `'minimal'`, a list or none) and `stat_fontsize`, and the shared options `ticks`, `fontsize`, `xlim`, `ylim`, `grid`, `background`
- `gs.scatter_plot()` takes `hist` to draw the same layout, the histogram of each variable beside the plot with its statistics, and `nbins`, `hist_color` and `hist_stat_blk` for it. It is off by default, so existing scatter plots do not change
- `gs.swath_plot()`: swath plot of the mean of one or more variables along x, y and z, with the data counts as bars or, with `count_style='points'`, as points, a list of data sets to compare models from different files, `count_label` for the right axis, and the shared options `ticks`, `fontsize`, `xlim`, `ylim`, `grid`, `background`
- `gs.clip_plot()`: map of a gridded variable inside a polygon or the convex hull of the data, with `outline` (drawn for the convex hull by default, left out for a given polygon), and the shared options `ticks`, `fontsize`, `xlim`, `ylim`, `grid`, `background`
- `gs.secondary_axis_plot()`: any lines on a left and a right y axis against one x, with a label, limits and colour for each axis, optional grey bands, zeros of the two axes lined up, and the shared options `ticks`, `fontsize`, `xlim`, `grid`, `background`
- `gs.grade_tonnage_plot()` (built on `secondary_axis_plot`): grade-tonnage curves with the grade on a second axis, `axis_colors` to colour each vertical axis (for example like its line), `ylim` and `ylim2`, and the shared options `ticks`, `fontsize`, `xlim`, `grid`, `background`
- `gs.scatter_plot_3d()`: 3D scatter plot with optional `drop_lines`, a list of data sets (each with its own colour and legend entry, `labels`), `zlim`, log scales (`logx`, `logy`, `logz`), statistics chosen with `stat_blk` (`'all'`, `'minimal'`, a list or none), and the shared options `ticks`, `fontsize`, `xlim`, `ylim`, `background`
- The new plots take the values that would otherwise be fixed in the code as arguments, with the previous values as defaults: `legend_kws`, `fill_color` and `fill_alpha`, `line45_kws`, `stat_bbox`, `label_kws`, `drop_kws`, `cbar_kws`, `stat_xy` and others, named in their docstrings

### Fixed
- `variogram_plot_simulation()` applies `ls`, which was accepted but ignored
- Accuracy statistics for MIK (`accmik`): it no longer crashes (missing imports, pandas 3 indexing), the lower tail is padded with 0 instead of 1, the upper tail is padded so true values above the highest threshold are counted, and the average variance is exact instead of a random estimate. A `truth` with several columns is now an error

### Changed
- `accuracy_plot()` follows the updated definitions of accuracy, precision and goodness of Deutsch (2026), Paper 40, CCG Annual Report 28, and the statistics block labels them A26, P26 and G26. It also applies `stat_fontsize` and `axis_xy` (accepted but ignored before), shows the statistics as decimals, and draws coloured deviation lines from each point to the 1:1 line
- The statistics block of `validation_plot()` shows the mean squared error (`MSE`) where it showed the root mean squared error (`RMSE`)
- Plots that use the shared options draw tick marks by default (4 points, pointing outwards). The plot styles had no major tick marks before; set `gs.Parameters['plotting.ticks'] = False` or pass `ticks=False` to get the old look
- The realizations of the histogram and variogram reproduction plots are drawn in a lighter grey by default (`simclr` `'0.5'`, `alpha` `0.3`)

## [1.2.0] - 2025-06-25

### Major Updates
- **Complete documentation overhaul** with modern Sphinx theme
- **Python version support** expanded to include 3.10, 3.11, 3.12, and 3.13
- **Dependency modernization** with updated versions of all major dependencies

### Added
- Documentation with pydata-sphinx-theme
- Modern badge-based README with clear installation instructions
- Support for latest Python versions (3.10-3.13)
- Modern build system using pyproject.toml

### Testing
- **Migrated test suite to pytest** from unittest for better testing experience
- Modern test fixtures and parametrized testing
- Automated testing across multiple Python versions in CI

### Changed
- **Breaking**: Minimum Python version raised to 3.10
- Documentation completely rebuilt with modern structure and examples
- Build system migrated from setup.py to pyproject.toml
- Updated package structure following modern Python standards
- Enhanced example data and tutorials

### Updated Dependencies
- matplotlib >= 3.10.0 (was < 3.0)
- numpy >= 2.1.3 (major version update)
- pandas >= 2.2.3 (major version update)
- scipy >= 1.15.2 (major version update)
- scikit-learn >= 1.6.1 (major version update)
- All other dependencies updated to latest stable versions

## [1.1.1] - 2021-11-13

### Previous Release
- Last stable release before major modernization
- Support for Python 3.6-3.9
- Legacy documentation and build system
- Original CCG geostatistics functionality

---

## Migration Guide

### Upgrading from 1.1.1 to 1.2.1

**Python Version Requirements:**
- **Minimum Python version is now 3.10** (was 3.6)
- Ensure you're using Python 3.10 or newer before upgrading

**Installation:**
```bash
# Uninstall old version
pip uninstall pygeostat

# Install new version
pip install pygeostat

# Verify installation
python -c "import pygeostat; print(pygeostat.__version__)"
```

**Code Compatibility:**
- Most existing code should work without changes
- Some deprecated matplotlib/pandas patterns may need updating
- Check the new documentation for updated examples

