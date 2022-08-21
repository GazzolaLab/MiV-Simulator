import numpy as np
import rbf
from miv_simulator.geometry.rbf_surface import RBFSurface
from miv_simulator.geometry.rbf_volume import RBFVolume

max_u = 11690.0
max_v = 2956.0


def DG_volume(u, v, l, rotate=None):
    """Parametric equations of the dentate gyrus volume."""

    u = np.array([u]).reshape(
        -1,
    )
    v = np.array([v]).reshape(
        -1,
    )
    l = np.array([l]).reshape(
        -1,
    )

    if rotate is not None:
        rot = make_rotate3d(rotate)
    else:
        rot = None

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

    pts = np.array([x, y, z]).reshape(3, u.size)

    if rot is not None:
        xyz = np.dot(rot, pts).T
    else:
        xyz = pts.T

    return xyz


def DG_meshgrid(
    extent_u,
    extent_v,
    extent_l,
    resolution=[30, 30, 10],
    rotate=None,
    return_uvl=False,
):

    ures, vres, lres = resolution

    obs_u = np.linspace(extent_u[0], extent_u[1], num=ures)
    obs_v = np.linspace(extent_v[0], extent_v[1], num=vres)
    obs_l = np.linspace(extent_l[0], extent_l[1], num=lres)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing="ij")
    xyz = DG_volume(u, v, l, rotate=rotate)

    if return_uvl:
        return xyz, obs_u, obs_v, obs_l
    else:
        return xyz


def make_DG_volume(
    extent_u,
    extent_v,
    extent_l,
    rotate=None,
    basis=rbf.basis.phs3,
    order=2,
    resolution=[30, 30, 10],
    return_xyz=False,
):
    """Creates an RBF volume based on the parametric equations of the dentate volume."""

    xyz, obs_u, obs_v, obs_l = DG_meshgrid(
        extent_u,
        extent_v,
        extent_l,
        rotate=rotate,
        resolution=resolution,
        return_uvl=True,
    )
    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, basis=basis, order=order)

    if return_xyz:
        return vol, xyz
    else:
        return vol


def make_DG_surface(
    extent_u,
    extent_v,
    obs_l,
    rotate=None,
    basis=rbf.basis.phs2,
    order=1,
    resolution=[33, 30],
):
    """Creates an RBF surface based on the parametric equations of the dentate volume."""
    ures = resolution[0]
    vres = resolution[1]

    obs_u = np.linspace(extent_u[0], extent_u[1], num=ures)
    obs_v = np.linspace(extent_v[0], extent_v[1], num=vres)

    u, v = np.meshgrid(obs_u, obs_v, indexing="ij")
    xyz = DG_volume(u, v, obs_l, rotate=rotate)

    srf = RBFSurface(obs_u, obs_v, xyz, basis=basis, order=order)

    return srf
