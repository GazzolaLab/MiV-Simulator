import sys
import numpy as np
from neural_geometry.geometry import transform_volume, make_alpha_shape
from neural_geometry.linear_volume import LinearVolume


max_u = 6000.
max_v = 2500.

def network_volume_transform(u, v, l):
    return u, v, l

def network_volume(u, v, l, rotate=None):
    """Parametric equations of the anatomical volume."""

    return transform_volume(network_volume_transform, u, v, l, rotate=rotate)


def network_meshgrid(extent_u, extent_v, extent_l, resolution=[3, 3, 3], rotate=None, return_uvl=False):

    ures, vres, lres = resolution

    obs_u = np.linspace(extent_u[0], extent_u[1], num=ures)
    obs_v = np.linspace(extent_v[0], extent_v[1], num=vres)
    obs_l = np.linspace(extent_l[0], extent_l[1], num=lres)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = network_volume(u, v, l, rotate=rotate)

    if return_uvl:
        return xyz, obs_u, obs_v, obs_l
    else:
        return xyz


def make_network_volume(extent_u, extent_v, extent_l, rotate=None, resolution=[3, 3, 3], return_xyz=False):
    """Creates an linear volume based on the parametric equations of the network volume."""

    xyz, obs_u, obs_v, obs_l = network_meshgrid(extent_u, extent_v, extent_l, \
                                            rotate=rotate, resolution=resolution,
                                            return_uvl=True)
    vol = LinearVolume(obs_u, obs_v, obs_l, xyz)

    if return_xyz:
        return vol, xyz
    else:
        return vol



def test_mplot_volume():

    extent_u = [0.0, 4000.0]
    extent_v = [0.0, 1250.0]
    extent_l = [0.0, 100.0]
    
    vol = make_network_volume(extent_u, extent_v, extent_l, resolution=[3, 3, 3])

    from mayavi import mlab

    vol.mplot_volume(color=(0, 1, 0), opacity=1.0, ures=1, vres=1)

    mlab.show()


def test_tri():

    extent_u = [0.0, 4000.0]
    extent_v = [0.0, 1250.0]
    extent_l = [0.0, 100.0]
    
    vol = make_network_volume(extent_u, extent_v, extent_l, resolution=[3, 3, 3])

    tri = vol.create_triangulation(ures=1, vres=1, lres=1)
    
    return vol, tri
    

def test_alpha_shape():

    extent_u = [0.0, 4000.0]
    extent_v = [0.0, 1250.0]
    extent_l = [0.0, 100.0]
    
    vol = make_network_volume(extent_u, extent_v, extent_l, resolution=[3, 3, 3])

    alpha = make_alpha_shape(vol, alpha_radius=1200.)
    
    return vol, alpha
    
if __name__ == '__main__':
    test_alpha_shape()

    # test_mplot_volume()
    # vol, tri = test_tri()
    # points = tri.points
    # simplices = tri.simplices
    # import matplotlib.pyplot as plt
    # import mpl_toolkits.mplot3d as plt3d
    # axes = plt3d.Axes3D(plt.figure())
    # vts = points[simplices, :]
    # poly = plt3d.art3d.Poly3DCollection(vts)
    # poly.set_alpha(0.2)
    # poly.set_color('grey')
    # axes.add_collection3d(poly)
    # axes.plot(points[:,0], points[:,1], points[:,2], 'ko')
    # axes.set_aspect('equal')
    # plt.show()


