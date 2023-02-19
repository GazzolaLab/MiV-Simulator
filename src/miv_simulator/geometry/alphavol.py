"""Alpha Shape implementation."""

import itertools
from collections import namedtuple

import networkx as nx
import numpy as np
from numpy.core.umath_tests import inner1d
from scipy.linalg import null_space
from scipy.spatial import Delaunay

AlphaShape = namedtuple("AlphaShape", ["points", "simplices", "bounds"])


def tri_graph(triangles):
    """
    Returns a graph of the triangulation edges.
    """
    G = nx.Graph()
    nodeset = set(list(triangles.ravel()))
    G.add_nodes_from(nodeset)

    for i in range(triangles.shape[0]):
        s = triangles[i, :]
        es = [(s[0], s[1]), (s[0], s[2]), (s[1], s[2])]
        for e in es:
            sl = [i]
            if G.has_edge(e[0], e[1]):
                ed = G.get_edge_data(e[0], e[1])
                ed["triangle"] = sl + ed["triangle_list"]
            else:
                G.add_edge(e[0], e[1], triangle_list=sl)
    return G


def angular_deviation(P):
    """
    Computes angle between two 3d triangles that share two vertices.
    """
    P1, P2, P3, P4 = P
    E = P1 - P2
    Q = null_space(E.T)
    v1 = Q.T * (P3 - P1)
    v2 = Q.T * (P4 - P1)
    theta1 = np.arctan2(v1[1], v1[0])
    theta2 = np.arctan2(v2[1], v2[0])
    theta = np.mod(theta1 - theta2, 2.0 * np.pi)  # radians
    return theta


def feature_edges(G, points, theta=1e-6):
    """
    A feature edge is a triangulation edge that has any of the following attributes:

    - The edge belongs to only one triangle.
    - The edge is shared by more than two triangles.
    - The edge is shared by a pair of triangles with angular deviation greater than the angle theta.
    """

    result = []
    for u, v, triangle_list in G.edges.data("triangle_list"):
        if triangle_list is not None:
            if len(triangle_list) == 1:
                result.append((u, v))
            elif len(triangle_list) > 2:
                result.append((u, v))
            else:
                if len(triangle_list) > 1:
                    triangle_pairs = itertools.combinations(triangle_list, 2)
                    triangle_thetas = map(
                        lambda p: angular_deviation(
                            points[set(list(p[0]) + list(p[1])), :]
                        ),
                        triangle_pairs,
                    )
                    if np.any(triangle_thetas > theta):
                        result.append((u, v))
    return result


def true_boundary(simplices, points):
    G = tri_graph(simplices)

    # Find edges attached to two coplanar faces
    E0 = set(G.edges)
    E1 = set(feature_edges(G, points, 1e-6))
    E2 = E0 - E1
    if len(E2) == 0:
        return simplices


def volumes(simplices, points):
    """Volumes/areas of tetrahedra/triangles."""

    A = points[simplices[:, 0], :]
    B = np.subtract(points[simplices[:, 1], :], A)
    C = np.subtract(points[simplices[:, 2], :], A)

    if points.shape[1] == 3:
        ## 3D Volume
        D = np.subtract(points[simplices[:, 3], :], A)
        BxC = np.cross(B, C, axis=1)
        vol = inner1d(BxC, D)
        vol = np.abs(vol) / 6.0
    else:
        ## 2D Area
        vol = np.subtract(
            np.multiply(B[:, 0], C[:, 1]) - np.multiply(B[:, 1], C[:, 0])
        )
        vol = np.abs(vol) / 2.0

    return vol


def circumcenters(simplices, points):
    """Determine circumcenters of polyhedra as described in the following page:
    http://mathworld.wolfram.com/Circumsphere.html
    """

    n = np.ones((simplices.shape[0], simplices.shape[1], 1))
    spts = points[simplices]
    a = np.linalg.det(np.concatenate((spts, n), axis=2))

    xyz_sqsum = spts[:, :, 0] ** 2 + spts[:, :, 1] ** 2 + spts[:, :, 2] ** 2
    Dx = np.linalg.det(
        np.stack(
            (
                xyz_sqsum,
                spts[:, :, 1],
                spts[:, :, 2],
                np.ones((xyz_sqsum.shape[0], 4)),
            ),
            axis=-1,
        )
    )
    Dy = np.linalg.det(
        np.stack(
            (
                xyz_sqsum,
                spts[:, :, 0],
                spts[:, :, 2],
                np.ones((xyz_sqsum.shape[0], 4)),
            ),
            axis=-1,
        )
    )
    Dz = np.linalg.det(
        np.stack(
            (
                xyz_sqsum,
                spts[:, :, 0],
                spts[:, :, 1],
                np.ones((xyz_sqsum.shape[0], 4)),
            ),
            axis=-1,
        )
    )

    c = np.linalg.det(
        np.stack(
            (xyz_sqsum, spts[:, :, 0], spts[:, :, 1], spts[:, :, 2]), axis=-1
        )
    )
    del xyz_sqsum

    ## circumcenter of the sphere
    x0 = Dx / (2.0 * a)
    y0 = Dy / (2.0 * a)
    z0 = Dz / (2.0 * a)

    ## circumradius
    r = np.sqrt((Dx**2) + (Dy**2) + (Dz**2) - 4.0 * a * c) / (
        2.0 * np.abs(a)
    )

    return ((x0, y0, z0), r)


def free_boundary(simplices):
    """
    Returns the facets that are referenced only by simplex of the given triangulation.
    """

    ## Sort the facet indices in the triangulation
    simplices = np.sort(simplices, axis=1)
    facets = np.vstack(
        (
            simplices[:, [0, 1, 2]],
            simplices[:, [0, 1, 3]],
            simplices[:, [0, 2, 3]],
            simplices[:, [1, 2, 3]],
        )
    )

    ## Find unique facets
    ufacets, counts = np.unique(facets, return_counts=True, axis=0)

    ## Determine which facets are part of only one simplex
    bidxs = np.where(counts == 1)[0]

    if len(bidxs) == 0:
        raise RuntimeError(
            "alpha.free_boundary: unable to determine facets that belong only to one simplex"
        )
    return ufacets[bidxs]


def alpha_shape(pts, radius, tri=None):
    """Alpha shape of 2D or 3D point set.
     V = ALPHAVOL(X,R) gives the area or volume V of the basic alpha shape
     for a 2D or 3D point set. X is a coordinate matrix of size Nx2 or Nx3.

    R is the probe radius with default value R = Inf. In the default case
    the basic alpha shape (or alpha hull) is the convex hull.

    Returns a structure AlphaShape with fields:

    - points    - Triangulation of the alpha shape (Mx3 or Mx4)
    - simplices - Circumradius of simplices in triangulation (Mx1)
    - bounds    - Boundary facets (Px2 or Px3)

    Based on MATLAB code by Jonas Lundgren <splinefit@gmail.com>
    """

    if tri is None:
        assert len(pts) > 0

    ## Check coordinates
    if tri is None:
        dim = pts.shape[1]
    else:
        dim = tri.points.shape[1]

    if dim < 2 or dim > 3:
        raise ValueError("pts must have 2 or 3 columns.")

    ## Check probe radius
    if not (type(radius) == float):
        raise ValueError("radius must be a real number.")

    ## Delaunay triangulation
    if tri is None:
        volpts = self.ev(hru, hrv, hrl).reshape(3, -1).T
        qhull_options = "QJ"
        tri = Delaunay(volpts, qhull_options=qhull_options)

    ## Check for zero volume tetrahedra since
    ## these can be of arbitrary large circumradius
    holes = False
    nz_index = None
    if dim == 3:
        n = tri.simplices.shape[0]
        vol = volumes(tri.simplices, tri.points)
        epsvol = 1e-12 * np.sum(vol) / float(n)
        nz_index = np.argwhere(vol > epsvol).ravel()
        holes = len(nz_index) < n

    ## Limit circumradius of simplices
    nz_simplices = tri.simplices
    if nz_index is not None:
        nz_simplices = tri.simplices[nz_index, :]
    _, rcc = circumcenters(nz_simplices, tri.points)
    rccidxs = np.where(rcc < radius)[0]
    T = nz_simplices[rccidxs, :]
    rcc = rcc[rccidxs]
    bnd = free_boundary(T)
    if holes:
        # The removal of zero volume tetrahedra causes false boundary
        # faces in the interior of the volume. Take care of these.
        bnd = true_boundary(bnd, tri.points)

    return AlphaShape(tri.points, T, bnd)
