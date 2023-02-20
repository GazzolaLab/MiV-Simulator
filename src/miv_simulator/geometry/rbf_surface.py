"""Implements a parametric surface as a tuple of RBF instances, one for u and v.
Based on code from bspline_surface.py
"""

import math

import numpy as np
import rbf
import rbf.basis
from rbf.interpolate import RBFInterpolant


def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def cartesian_product(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian_product(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


class RBFSurface:
    def __init__(self, u, v, xyz, order=1, basis=rbf.basis.phs2):
        """Parametric (u,v) 3D surface approximation.

        Parameters
        ----------
        u, v, l : array_like
            1-D arrays of coordinates.
        xyz : array_like
            3-D array of (x, y, z) data with shape (3, u.size, v.size).
        order : int, optional
            Order of interpolation. Default is 1.
        basis: RBF basis function
        """

        self._create_srf(u, v, xyz, order=order, phi=basis)

        self.u = u
        self.v = v
        self.order = order

    def __call__(self, *args, **kwargs):
        """Convenience to allow evaluation of a RBFSurface
        instance via `foo(0, 0)` instead of `foo.ev(0, 0)`.
        """
        return self.ev(*args, **kwargs)

    def _create_srf(self, obs_u, obs_v, xyz, **kwargs):
        # Create surface definitions
        u, v = np.meshgrid(obs_u, obs_v, indexing="ij")
        uv_obs = np.array([u.ravel(), v.ravel()]).T

        xsrf = RBFInterpolant(uv_obs, xyz[:, 0], **kwargs)
        ysrf = RBFInterpolant(uv_obs, xyz[:, 1], **kwargs)
        zsrf = RBFInterpolant(uv_obs, xyz[:, 2], **kwargs)

        usrf = RBFInterpolant(xyz, uv_obs[:, 0], **kwargs)
        vsrf = RBFInterpolant(xyz, uv_obs[:, 1], **kwargs)

        self._xsrf = xsrf
        self._ysrf = ysrf
        self._zsrf = zsrf
        self._usrf = usrf
        self._vsrf = vsrf

    def _resample_uv(self, ures, vres):
        """Helper function to re-sample to u and v parameters
        at the specified resolution
        """
        u, v = self.u, self.v
        lu, lv = len(u), len(v)
        nus = np.array(list(enumerate(u))).T
        nvs = np.array(list(enumerate(v))).T
        newundxs = np.linspace(0, lu - 1, ures * lu - (ures - 1))
        newvndxs = np.linspace(0, lv - 1, vres * lv - (vres - 1))
        hru = np.interp(newundxs, *nus)
        hrv = np.interp(newvndxs, *nvs)
        return hru, hrv

    def ev(self, su, sv, mesh=True, chunk_size=1000):
        """Get point(s) in surface at (su, sv).

        Parameters
        ----------
        u, v : scalar or array-like

        Returns
        -------
        Returns an array of shape len(u) x len(v) x 3
        """

        if mesh:
            U, V = np.meshgrid(su, sv)
        else:
            U = su
            V = sv

        uv_s = np.array([U.ravel(), V.ravel()]).T
        X = self._xsrf(uv_s)
        Y = self._ysrf(uv_s)
        Z = self._zsrf(
            uv_s,
        )

        arr = np.array([X, Y, Z])

        return arr.reshape(3, len(U), -1)

    def inverse(self, xyz):
        """Get parametric coordinates (u, v) that correspond to the given x, y, z.
        May return None if x, y, z are outside the interpolation domain.

        Parameters
        ----------
        xyz : array of coordinates

        Returns
        -------
        Returns an array of shape 3 x len(xyz)
        """

        U = self._usrf(xyz)
        V = self._vsrf(xyz)

        arr = np.array([U, V])
        return arr.T

    def utan(self, su, sv, normalize=True):
        u = np.array([su]).reshape(
            -1,
        )
        v = np.array([sv]).reshape(
            -1,
        )

        dxdu = self._xsrf(u, v, diff=np.asarray([1, 0, 0]))
        dydu = self._ysrf(u, v, diff=np.asarray([1, 0, 0]))
        dzdu = self._zsrf(u, v, diff=np.asarray([1, 0, 0]))

        du = np.array([dxdu, dydu, dzdu]).T

        du = du.swapaxes(0, 1)

        if normalize:
            du /= np.sqrt((du**2).sum(axis=2))[:, :, np.newaxis]

        arr = du.transpose(2, 0, 1)
        return arr

    def vtan(self, su, sv, normalize=True):
        u = np.array([su]).reshape(
            -1,
        )
        v = np.array([sv]).reshape(
            -1,
        )

        dxdv = self._xsrf(u, v, diff=np.asarray([0, 1, 0]))
        dydv = self._ysrf(u, v, diff=np.asarray([0, 1, 0]))
        dzdv = self._zsrf(u, v, diff=np.asarray([0, 1, 0]))
        dv = np.array([dxdv, dydv, dzdv]).T

        dv = dv.swapaxes(0, 1)

        if normalize:
            dv /= np.sqrt((dv**2).sum(axis=2))[:, :, np.newaxis]

        arr = dv.transpose(2, 0, 1)
        return arr

    def normal(self, su, sv):
        """Get normal(s) at (u, v).

        Parameters
        ----------
        u, v : scalar or array-like
            u and v may be scalar or vector (see below)

        Returns
        -------
        Returns an array of shape 3 x len(u) x len(v)
        """

        u = np.array([su]).reshape(
            -1,
        )
        v = np.array([sv]).reshape(
            -1,
        )

        dxdus = self._xsrf(u, v, diff=np.asarray([1, 0, 0]))
        dydus = self._ysrf(u, v, diff=np.asarray([1, 0, 0]))
        dzdus = self._zsrf(u, v, diff=np.asarray([1, 0, 0]))
        dxdvs = self._xsrf(u, v, diff=np.asarray([0, 1, 0]))
        dydvs = self._ysrf(u, v, diff=np.asarray([0, 1, 0]))
        dzdvs = self._zsrf(u, v, diff=np.asarray([0, 1, 0]))

        normals = np.cross(
            [dxdus, dydus, dzdus], [dxdvs, dydvs, dzdvs], axisa=0, axisb=0
        )

        normals /= np.sqrt((normals**2).sum(axis=2))[:, :, np.newaxis]

        arr = normals.transpose(2, 0, 1)
        return arr

    def point_distance(
        self,
        su,
        sv,
        axis=0,
        interp_chunk_size=1000,
        axis_origin=None,
        return_coords=True,
    ):
        """Cumulative distance between pairs of (u, v) coordinates.

        Parameters
        ----------
        u, v : array-like

        axis: axis along which the distance should be computed

        axis_origin: the origin coordinate for the given axes (the left-most coordinate if None)

        return_coords: if True, returns the coordinates for which computed distance (default: True)

        Returns
        -------
        If the lengths of u and v are at least 2, returns the total arc length
        between each u,v pair.
        """
        u = np.array([su]).reshape(
            -1,
        )
        v = np.array([sv]).reshape(
            -1,
        )

        input_axes = [u, v]
        if axis_origin is None:
            axis_origin = input_axes[axis][0]

        c = input_axes[axis]

        cl = (np.sort(c[np.where(c < axis_origin)[0]]))[::-1]
        cr = np.sort(c[np.where(c >= axis_origin)[0]])

        ordered_axes = [
            (-1, [cl if i == axis else x for (i, x) in enumerate(input_axes)]),
            (1, [cr if i == axis else x for (i, x) in enumerate(input_axes)]),
        ]

        aidx = [0, 1]
        aidx.remove(axis)

        distances = []
        coords = [[] for i in range(0, 2)]
        for sgn, axes in ordered_axes:
            npts = axes[axis].shape[0]

            if npts > 1:
                paxes = [axes[i] for i in aidx]
                prod = cartesian_product(paxes)
                for ip, p in enumerate(prod):
                    ecoords = [
                        x if i == axis else p[aidx.index(i)]
                        for (i, x) in enumerate(axes)
                    ]
                    pts = (
                        self.ev(*ecoords, chunk_size=interp_chunk_size)
                        .reshape(3, -1)
                        .T
                    )
                    a = pts[1:, :]
                    b = pts[0 : npts - 1, :]
                    d = np.zeros(
                        npts,
                    )
                    d[1:npts] = np.cumsum(euclidean_distance(a, b))
                    if sgn < 0:
                        distances.append(np.negative(d))
                    else:
                        distances.append(d)
                    if return_coords:
                        pcoords = [
                            x
                            if i == axis
                            else np.repeat(p[aidx.index(i)], npts)
                            for (i, x) in enumerate(axes)
                        ]
                        for i, col in enumerate(pcoords):
                            coords[i].append(col)

        if return_coords:
            return distances, coords
        else:
            return distances

    def mplot_surface(self, ures=8, vres=8, **kwargs):
        """Plot the surface using Mayavi's `mesh()` function

        Parameters
        ----------
        ures, vres : int
            Specifies the oversampling of the original
            surface in u and v directions. For example:
            if `ures` = 2, and `self.u` = [0, 1, 2, 3],
            then the surface will be resampled at
            [0, 0.5, 1, 1.5, 2, 2.5, 3] prior to
            plotting.

        kwargs : dict
            See Mayavi docs for `mesh()`

        Returns
        -------
            None
        """
        from matplotlib.colors import ColorConverter
        from mayavi import mlab

        if not "color" in kwargs:
            # Generate random color
            cvec = np.random.rand(3)
            cvec /= math.sqrt(cvec.dot(cvec))
            kwargs["color"] = tuple(cvec)
        else:
            # The following will convert text strings representing
            # colors into their (r, g, b) equivalents (which is
            # the only way Mayavi will accept them)
            from matplotlib.colors import ColorConverter

            cconv = ColorConverter()
            if kwargs["color"] is not None:
                kwargs["color"] = cconv.to_rgb(kwargs["color"])

        # Make new u and v values of (possibly) higher resolution
        # the original ones.
        hru, hrv = self._resample_uv(ures, vres)
        # Sample the surface at the new u, v values and plot
        meshpts = self.ev(hru, hrv)
        m = mlab.mesh(*meshpts, **kwargs)

        # Turn off perspective
        fig = mlab.gcf()
        fig.scene.camera.trait_set(parallel_projection=1)
        return fig

    def copy(self):
        """Get a copy of the surface"""
        from copy import deepcopy

        return deepcopy(self)


def test_surface(u, v, l):
    import numpy as np

    x = np.array(
        -500.0 * np.cos(u) * (5.3 - np.sin(u) + (1.0 + 0.138 * l) * np.cos(v))
    )
    y = np.array(
        750.0
        * np.sin(u)
        * (5.5 - 2.0 * np.sin(u) + (0.9 + 0.114 * l) * np.cos(v))
    )
    z = np.array(
        2500.0 * np.sin(u)
        + (663.0 + 114.0 * l) * np.sin(v - 0.13 * (np.pi - u))
    )
    return np.array([x, y, z])


def test_uv_isospline():
    obs_u = np.linspace(-0.016 * np.pi, 1.01 * np.pi, 20)
    obs_v = np.linspace(-0.23 * np.pi, 1.425 * np.pi, 20)
    obs_l = 1.0

    u, v = np.meshgrid(obs_u, obs_v, indexing="ij")
    xyz = test_surface(u, v, obs_l).reshape(3, u.size).T

    order = [1]
    for ii in range(len(order)):
        srf = RBFSurface(obs_u, obs_v, xyz, order=order[ii])

        U, V = srf._resample_uv(50, 50)
        L = np.asarray([-1.0])

        nupts = U.shape[0]
        nvpts = V.shape[0]

    from mayavi import mlab

    U, V = srf._resample_uv(10, 10)
    L = np.asarray([1.0])

    nupts = U.shape[0]
    nvpts = V.shape[0]
    # Plot u,v-isosplines on the surface
    upts = srf(U, V[0], L)
    vpts = srf(U[int(nupts / 2)], V, L)

    srf.mplot_surface(color=(0, 1, 0), opacity=1.0, ures=10, vres=10)

    mlab.points3d(*upts, scale_factor=100.0, color=(1, 1, 0))
    mlab.points3d(*vpts, scale_factor=100.0, color=(1, 1, 0))

    mlab.show()


def test_point_distance():
    obs_u = np.linspace(-0.016 * np.pi, 1.01 * np.pi, 20)
    obs_v = np.linspace(-0.23 * np.pi, 1.425 * np.pi, 20)
    obs_l = 1.0

    u, v = np.meshgrid(obs_u, obs_v, indexing="ij")
    xyz = test_surface(u, v, obs_l).reshape(3, u.size).T

    srf = RBFSurface(obs_u, obs_v, xyz, order=2)

    U, V = srf._resample_uv(5, 5)

    dist, coords = srf.point_distance(U, V)
    print(dist)
    print(coords)
    dist, coords = srf.point_distance(U, V[0])
    print(dist)
    print(coords)
    dist, coords = srf.point_distance(U, V[0], axis_origin=np.median(obs_u))
    print(dist)
    print(coords)


if __name__ == "__main__":
    test_uv_isospline()
#    test_point_distance()
