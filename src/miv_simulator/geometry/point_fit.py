"""Procedures to fit 3d point clouds to an alpha shape"""

import numpy as np
from past.utils import old_div


##
## Plane fitting to a point cloud:
## https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
##
def PCA(data, correlation=False, sort=True):
    """Applies Principal Component Analysis to the data

    Parameters
    ----------
    data: array
    The array containing the data. The array must have NxM dimensions, where each
    of the N rows represents a different individual record and each of the M columns
    represents a different variable recorded for that individual record.
        array([
        [V11, ... , V1m],
        ...,
        [Vn1, ... , Vnm]])

    correlation(Optional) : bool
        Set the type of matrix to be computed (see Notes):
            If True compute the correlation matrix.
            If False(Default) compute the covariance matrix.

    sort(Optional) : bool
        Set the order that the eigenvalues/vectors will have
            If True(Default) they will be sorted (from higher value to less).
            If False they won't.
    Returns
    -------
    eigenvalues: (1,M) array
    The eigenvalues of the corresponding matrix.

    eigenvector: (M,M) array
    The eigenvectors of the corresponding matrix.

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

    """

    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:
        matrix = np.corrcoef(data_adjust.T)
    else:
        matrix = np.cov(data_adjust.T)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def points_plane_fit(points, equation=False):
    """Computes the best fitting plane of the given points

    Parameters
    ----------
    points: array
    The x,y,z coordinates corresponding to the points from which we want
    to define the best fitting plane. Expected format:
        array([
        [x1,y1,z1],
        ...,
        [xn,yn,zn]])

    equation(Optional) : bool
        Set the oputput plane format:
            If True return the a,b,c,d coefficients of the plane.
            If False(Default) return 1 Point and 1 Normal vector.
    Returns
    -------
    a, b, c, d : float
    The coefficients solving the plane equation.

    or

    point, normal: array
    The plane defined by 1 Point and 1 Normal vector. With format:
    array([Px,Py,Pz]), array([Nx,Ny,Nz])

    """

    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:, 2]

    #: get a point from the plane
    point = np.mean(points, axis=0)

    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d
    else:
        return point, normal


##
## Skew-symmetric cross-product of a 3d vector:
##
def ssc(v):
    return np.stack(([0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]))


##
## Computes a rotation matrix R that rotates unit vector a onto unit vector b.
## Based on code from:
## https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
##
def rotvector3d(A, B):
    AxB = np.cross(A, B)
    s = np.linalg.norm(AxB)
    return (
        np.eye(
            3,
        )
        + ssc(AxB)
        + old_div(np.linalg.matrix_power(ssc(AxB), 2) * (1.0 - np.dot(A, B)), (s**2))
    )
