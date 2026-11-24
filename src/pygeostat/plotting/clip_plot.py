#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""clip_plot plots a gridded variable only inside a polygon, or inside the convex hull of the data"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.path import Path
from .export_image import export_image
from .slice_plot import slice_plot


def convex_hull(x, y):
    """
    Convex hull of a set of 2D locations, as a closed polygon.

    Parameters:
        x, y: coordinates of the locations. Pairs where either value is NaN are left out

    Returns:
        polygon (ndarray): ``(n, 2)`` array of the corners of the hull in counter-clockwise order. The
        first corner is repeated at the end, like the polygon files of GSLIB programs

    **Examples:**

        .. plot::

            import pygeostat as gs
            import matplotlib.pyplot as plt
            data = gs.ExampleData('cluster').data
            hull = gs.convex_hull(data['Xlocation'], data['Ylocation'])
            plt.scatter(data['Xlocation'], data['Ylocation'], c='k', s=4)
            plt.plot(hull[:, 0], hull[:, 1], 'C3')

    .. codeauthor:: pygeostat development team
    """
    from scipy.spatial import ConvexHull, QhullError
    xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) < 3:
        raise ValueError("At least three locations are needed for a convex hull")
    try:
        corners = xy[ConvexHull(xy).vertices]
    except QhullError:
        raise ValueError("The locations are on one line, so they have no convex hull")
    return np.vstack([corners, corners[:1]])


def convex_hull_3d(x, y, z):
    """
    Convex hull of a set of 3D locations, as a closed surface of triangles.

    Parameters:
        x, y, z: coordinates of the locations. Locations where any value is NaN are left out

    Returns:
        vertices (ndarray): ``(n, 3)`` array of the locations that are corners of the hull
        faces (ndarray): ``(m, 3)`` array of integer indices into ``vertices``, one row for each triangle of the
        surface. The corners of a triangle are in counter-clockwise order seen from outside the hull

    **Examples:**

        .. plot::

            import pygeostat as gs
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            data = gs.ExampleData('oilsands')
            xyz = data.data[[data.x, data.y, data.z]].dropna().to_numpy()
            vertices, faces = gs.convex_hull_3d(*xyz.T)
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(*xyz.T, c='k', s=1)
            ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor='C3', alpha=0.2, edgecolor='C3', lw=0.3))

    .. codeauthor:: pygeostat development team
    """
    from scipy.spatial import ConvexHull, QhullError
    xyz = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)])
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    if len(xyz) < 4:
        raise ValueError("At least four locations are needed for a 3D convex hull")
    try:
        hull = ConvexHull(xyz)
    except QhullError:
        raise ValueError("The locations are on one plane or one line, so they have no 3D convex hull")
    new_index = np.full(len(xyz), -1)
    new_index[hull.vertices] = np.arange(len(hull.vertices))
    faces = new_index[hull.simplices]
    corners = xyz[hull.simplices]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = (normals * hull.equations[:, :3]).sum(axis=1) < 0  # the equations point away from the hull
    faces[inward] = faces[inward][:, ::-1]
    return xyz[hull.vertices], faces


def _read_polygon(boundary):
    """The corners of the polygon, from a GSLIB polygon file (title, number of columns, their names,
    then one x y pair per line), from a csv file with x and y columns or from an array like object
    with two columns"""
    if isinstance(boundary, str) and boundary.lower().endswith('.csv'):
        boundary = pd.read_csv(boundary).iloc[:, :2].values
    elif isinstance(boundary, str):
        with open(boundary) as file:
            file.readline()
            ncol = int(file.readline().split()[0])
            for _ in range(ncol):
                file.readline()
            boundary = np.loadtxt(file)
    polygon = np.asarray(boundary, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("The boundary needs at least three corners, one x and y pair per row")
    return polygon


def clip_plot(data, boundary, griddef=None, var=None, points=None, pointvar=None, pointkws=None, x=None,
              y=None, ax=None, **kwargs):
    """
    Plot a gridded variable only inside a polygon, as the GSLIB programs clipgrid and pixelplt2008 do.
    The cells outside are NaN, so they are left empty. A cell is inside if its centre is.

    The polygon is given as a GSLIB polygon file, an array of corners or ``'hull'``, the convex hull of
    ``points`` (the locations of the data). The drawing is :func:`gs.slice_plot()
    <pygeostat.plotting.slice_plot.slice_plot>`, so its arguments (``cmap``, ``vlim``, ``cbar``,
    ``title``, ...) can be passed to this function too.

    Unless they are given, the title is the variable with what it is clipped to ("Clipped to the Convex Hull"
    or "Clipped to the Boundary", ``title=False`` for none) and the label of the colour bar is the variable.

    Parameters:
        data: DataFile or DataFrame of the gridded values, with one row per cell
        boundary (str or array): path of a GSLIB polygon file, an ``(n, 2)`` array of corners or
            ``'hull'`` for the convex hull of ``points``
        griddef (GridDef): the grid of ``data``. Read from a DataFile when it has one
        var (str): column to plot. Not needed if ``data`` has only one column
        points: DataFile or DataFrame of data locations to draw on top, coloured by ``pointvar`` with
            the colours of the grid. They are not clipped
        pointvar (str): column of ``points`` that sets their colour. Without it the points are small
            black dots
        pointkws (dict): keyword arguments of ``matplotlib.pyplot.scatter`` for the points, such as ``s``
        x, y (str): column names of the coordinates of ``points``. Read from a DataFile when it knows them
        ax (ax): Matplotlib axis to plot on
        **kwargs: Optional permissible keyword arguments to pass to
            :func:`gs.slice_plot() <pygeostat.plotting.slice_plot.slice_plot>`. ``output_file`` and
            ``out_kws`` save the figure with :func:`gs.export_image()
            <pygeostat.plotting.export_image.export_image>` after the polygon is drawn

    Returns:
        ax (ax): Matplotlib Axes object with the clipped grid

    **Examples:**

        .. plot::

            import pygeostat as gs
            import numpy as np
            grid = gs.GridDef([40, 0.5, 1, 40, 0.5, 1, 1, 0.5, 1])
            xx, yy = np.meshgrid(np.arange(40) + 0.5, np.arange(40) + 0.5)
            values = np.sin(xx / 6) + np.cos(yy / 8)
            data = gs.DataFile(data=__import__('pandas').DataFrame({'value': values.ravel()}), griddef=grid)
            polygon = [[5, 5], [35, 8], [30, 33], [10, 30]]
            gs.clip_plot(data, polygon, var='value')

    |
    """
    from ..data.data import DataFile
    if griddef is None and isinstance(data, DataFile):
        griddef = data.griddef
    if griddef is None:
        raise ValueError("The grid is not known, pass `griddef`")
    if isinstance(data, DataFile):
        values = data.data
    elif isinstance(data, pd.DataFrame):
        values = data
    else:
        raise ValueError("The parameter data must be either a gs.DataFile or a pd.DataFrame")
    if var is None:
        if len(values.columns) != 1:
            raise ValueError("`var` is needed: the data has %d columns" % len(values.columns))
        var = values.columns[0]
    if len(values) != griddef.count():
        raise ValueError("The data has %d rows but the grid has %d cells" % (len(values), griddef.count()))
    if isinstance(points, DataFile):
        x = points.x if x is None else x
        y = points.y if y is None else y
        points = points.data
    if points is not None and (x is None or y is None):
        raise ValueError("The columns of the coordinates of `points` are not set, pass `x` and `y`")
    hull = isinstance(boundary, str) and boundary == 'hull'
    if hull:
        if points is None:
            raise ValueError("boundary='hull' needs `points`, the locations of the data")
        polygon = convex_hull(points[x], points[y])
    else:
        polygon = _read_polygon(boundary)
    # cell centres, x changes fastest as in GSLIB files
    cx = griddef.xmn + griddef.xsiz * np.arange(griddef.nx)
    cy = griddef.ymn + griddef.ysiz * np.arange(griddef.ny)
    xx, yy = np.meshgrid(cx, cy)
    inside = Path(polygon).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
    clipped = values[[var]].copy()
    clipped.loc[~np.tile(inside, griddef.nz), var] = np.nan
    if points is not None and pointvar is not None:
        kwargs['pointdata'] = DataFile(data=points, x=x, y=y)
        kwargs['pointvar'] = pointvar
        kwargs['pointkws'] = {'edgecolors': 'k', 'linewidths': 0.5, **(pointkws or {})}
    if kwargs.get('title') is None:
        kwargs['title'] = '%s Clipped to the %s' % (var, 'Convex Hull' if hull else 'Boundary')
    if kwargs.get('cbar_label') is None:
        kwargs['cbar_label'] = var
    output_file = kwargs.pop('output_file', None)  # saved below, once the polygon is drawn
    out_kws = kwargs.pop('out_kws', None) or {}
    return_cbar = kwargs.pop('return_cbar', False)
    ax, colorbar = slice_plot(clipped, griddef=griddef, var=var, ax=ax, return_cbar=True, **kwargs)
    if colorbar:  # the label of the colour bar as large as its tick labels
        colorbar.ax.yaxis.label.set_size(colorbar.ax.yaxis.get_ticklabels()[0].get_size())
    if points is not None and pointvar is None:
        grid_xlim, grid_ylim = ax.get_xlim(), ax.get_ylim()
        ax.scatter(points[x], points[y], **{'c': 'k', 's': 4, 'linewidths': 0, **(pointkws or {})})
        ax.set_xlim(grid_xlim)
        ax.set_ylim(grid_ylim)
    if output_file or ('pdfpages' in out_kws):
        export_image(output_file, **out_kws)
    return (ax, colorbar) if return_cbar else ax
