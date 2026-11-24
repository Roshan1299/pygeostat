#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""convex_hull and convex_hull_3d find the outline of a set of locations"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np


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
