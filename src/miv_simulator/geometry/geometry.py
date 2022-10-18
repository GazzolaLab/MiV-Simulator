"""Classes and procedures related to neuronal geometry and distance calculation."""

import logging
import math
import numpy as np
import rbf
from miv_simulator.geometry.alphavol import alpha_shape
from miv_simulator.geometry.rbf_volume import RBFVolume
from mpi4py import MPI

## This logger will inherit its setting from its root logger
logger = logging.getLogger(f"{__name__}")


def rotate2d(theta):
    """
    Returns the 2D rotation matrix associated with counterclockwise rotation around the origin
    by theta radians.
    """
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array(((c, -s), (s, c)))
    return rot


def rotate3d(axis, theta):
    """
    Returns the 3D rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def make_rotate3d(rotate):
    """Creates a rotation matrix based on angles in degrees."""
    rot = None
    for i in range(0, 3):
        if rotate[i] != 0.0:
            a = float(np.deg2rad(rotate[i]))
            rot = rotate3d([1 if i == j else 0 for j in range(0, 3)], a)

    return rot


def transform_volume(transform, u, v, l, rotate=None):
    """Transform to Euclidean volume."""

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

    x, y, z = transform(u, v, l)

    pts = np.array([x, y, z]).reshape(3, u.size)

    if rot is not None:
        xyz = np.dot(rot, pts).T
    else:
        xyz = pts.T

    return xyz


def get_layer_extents(layer_extents, layer):
    min_u, max_u = 0.0, 0.0
    min_v, max_v = 0.0, 0.0
    min_l, max_l = 0.0, 0.0
    for current_layer, extent in layer_extents.items():
        if current_layer == layer:
            min_u, max_u = extent[0][0], extent[1][0]
            min_v, max_v = extent[0][1], extent[1][1]
            min_l, max_l = extent[0][2], extent[1][2]
    return ((min_u, max_u), (min_v, max_v), (min_l, max_l))


def get_total_extents(layer_extents):

    min_u = float("inf")
    max_u = 0.0

    min_v = float("inf")
    max_v = 0.0

    min_l = float("inf")
    max_l = 0.0

    for layer, extent in layer_extents.items():
        min_u = min(extent[0][0], min_u)
        min_v = min(extent[0][1], min_v)
        min_l = min(extent[0][2], min_l)
        max_u = max(extent[1][0], max_u)
        max_v = max(extent[1][1], max_v)
        max_l = max(extent[1][2], max_l)

    return ((min_u, max_u), (min_v, max_v), (min_l, max_l))


def uvl_in_bounds(uvl_coords, layer_extents, pop_layers):
    for layer, count in pop_layers.items():
        if count > 0:
            min_extent = layer_extents[layer][0]
            max_extent = layer_extents[layer][1]
            result = (
                (uvl_coords[0] < (max_extent[0] + 0.001))
                and (uvl_coords[0] > (min_extent[0] - 0.001))
                and (uvl_coords[1] < (max_extent[1] + 0.001))
                and (uvl_coords[1] > (min_extent[1] - 0.001))
                and (uvl_coords[2] < (max_extent[2] + 0.001))
                and (uvl_coords[2] > (min_extent[2] - 0.001))
            )
            if result:
                return True
    return False


def make_alpha_shape(vol, alpha_radius=120.0):

    logger.info("Constructing volume triangulation...")

    tri = vol.create_triangulation()

    logger.info("Constructing alpha shape...")

    alpha = alpha_shape([], alpha_radius, tri=tri)

    return alpha


def save_alpha_shape(file_path, dataset_path, alpha_shape):
    import h5py

    f = h5py.File(file_path)
    grp = f.create_group(dataset_path)
    grp["points"] = alpha_shape.points
    grp["simplices"] = alpha_shape.simplices
    grp["bounds"] = alpha_shape.bounds
    f.close()


def load_alpha_shape(file_path, dataset_path):
    import h5py

    f = h5py.File(file_path)
    alpha_shape = None
    if dataset_path in f:
        points = f[dataset_path]["points"][:]
        simplices = f[dataset_path]["simplices"][:]
        bounds = f[dataset_path]["bounds"][:]
        alpha_shape = dentate.alphavol.AlphaShape(points, simplices, bounds)
    f.close()
    return alpha_shape


def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def make_uvl_distance(vol, xyz_coords, rotate=None):
    f = lambda u, v, l: euclidean_distance(
        vol(u, v, l, rotate=rotate), xyz_coords
    )
    return f


def optimize_inverse_uvl_coords(
    vol, xyz_coords, rotate, layer_extents, pop_layers, optiter=100
):
    import dlib

    f_uvl_distance = make_uvl_distance(vol, xyz_coords, rotate=rotate)
    for layer, count in pop_layers.items():
        if count > 0:
            min_extent = layer_extents[layer][0]
            max_extent = layer_extents[layer][1]
            uvl_coords, dist = dlib.find_min_global(
                f_uvl_distance, min_extent, max_extent, optiter
            )
            if uvl_in_bounds(uvl_coords, layer_extents, {layer: count}):
                return uvl_coords
    return None


def generate_nodes(alpha, nsample, nodeitr):
    from rbf.pde.geometry import contains
    from rbf.pde.nodes import min_energy_nodes

    N = nsample * 2  # total number of nodes
    node_count = 0
    itr = nodeitr

    vert = alpha.points
    smp = np.asarray(alpha.bounds, dtype=np.int64)

    while node_count < nsample:
        logger.info("Generating %i nodes (%i iterations)..." % (N, itr))
        # create N quasi-uniformly distributed nodes
        out = min_energy_nodes(N, (vert, smp), iterations=itr)
        nodes = out[0]

        # remove nodes outside of the domain
        in_nodes = nodes[contains(nodes, vert, smp)]

        node_count = len(in_nodes)
        N = int(1.5 * N)

    logger.info(
        "%i interior nodes generated (%i iterations)" % (node_count, itr)
    )

    xyz_coords = in_nodes.reshape(-1, 3)

    return xyz_coords


def get_volume_distances(
    ip_vol,
    origin_spec=None,
    nsample=1000,
    alpha_radius=120.0,
    nodeitr=20,
    comm=None,
):
    """Computes arc-distances along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    ip_vol : RBFVolume
        An interpolated volume instance of class RBFVolume.
    origin_coords : array(float)
        Origin point to use for distance computation.
    nsample : int
        Number of points to sample inside the volume.
    alpha_radius : float
        Parameter for creation of alpha volume that encloses the
        RBFVolume. Smaller values improve the quality of alpha volume,
        but increase the time for sampling points inside the volume.
    nodeitr : int
        Number of iterations for distributing sampled points inside the volume.
    comm : MPIComm (optional)
        mpi4py MPI communicator: if provided, node generation and distance computation will be performed in parallel
    Returns
    -------
    (Y1, X1, ... , YN, XN) where N is the number of dimensions of the volume.
    X : array of coordinates
        The sampled coordinates.
    Y : array of distances
        The arc-distance from the starting index of the coordinate space to the corresponding coordinates in X.

    """

    size = 1
    rank = 0
    if comm is not None:
        rank = comm.rank
        size = comm.size

    uvl_coords = None
    origin_extent = None
    origin_ranges = None
    origin_pos_um = None
    if rank == 0:
        boundary_uvl_coords = np.array(
            [
                [ip_vol.u[0], ip_vol.v[0], ip_vol.l[0]],
                [ip_vol.u[0], ip_vol.v[-1], ip_vol.l[0]],
                [ip_vol.u[-1], ip_vol.v[0], ip_vol.l[0]],
                [ip_vol.u[-1], ip_vol.v[-1], ip_vol.l[0]],
                [ip_vol.u[0], ip_vol.v[0], ip_vol.l[-1]],
                [ip_vol.u[0], ip_vol.v[-1], ip_vol.l[-1]],
                [ip_vol.u[-1], ip_vol.v[0], ip_vol.l[-1]],
                [ip_vol.u[-1], ip_vol.v[-1], ip_vol.l[-1]],
            ]
        )

        resample = 10
        span_U, span_V, span_L = ip_vol._resample_uvl(
            resample, resample, resample
        )

        if origin_spec is None:
            origin_coords = np.asarray(
                [np.median(span_U), np.median(span_V), np.max(span_L)]
            )
        else:
            origin_coords = np.asarray(
                [
                    origin_spec["U"](span_U),
                    origin_spec["V"](span_V),
                    origin_spec["L"](span_L),
                ]
            )

        logger.info(
            "Origin coordinates: %f %f %f"
            % (origin_coords[0], origin_coords[1], origin_coords[2])
        )

        pos, extents = ip_vol.point_position(
            origin_coords[0], origin_coords[1], origin_coords[2]
        )

        origin_pos = pos[0]
        origin_extent = extents[0]
        origin_pos_um = (
            origin_pos[0] * origin_extent[0],
            origin_pos[1] * origin_extent[1],
        )
        origin_ranges = (
            (
                -(origin_pos[0] * origin_extent[0]),
                (1.0 - origin_pos[0]) * origin_extent[0],
            ),
            (
                -(origin_pos[1] * origin_extent[1]),
                (1.0 - origin_pos[1]) * origin_extent[1],
            ),
        )

        logger.info(
            "Origin position: %f %f extent: %f %f"
            % (origin_pos[0], origin_pos[1], origin_extent[0], origin_extent[1])
        )
        logger.info(
            "Origin ranges: %f : %f %f : %f"
            % (
                origin_ranges[0][0],
                origin_ranges[0][1],
                origin_ranges[1][0],
                origin_ranges[1][1],
            )
        )

        logger.info("Creating volume triangulation...")
        tri = ip_vol.create_triangulation()

        logger.info("Constructing alpha shape...")
        alpha = alpha_shape([], alpha_radius, tri=tri)

        xyz_coords = generate_nodes(alpha, nsample, nodeitr)

        logger.info("Inverse interpolation of UVL coordinates...")
        uvl_coords_interp = ip_vol.inverse(xyz_coords)
        xyz_coords_interp = (
            ip_vol(
                uvl_coords_interp[:, 0],
                uvl_coords_interp[:, 1],
                uvl_coords_interp[:, 2],
                mesh=False,
            )
            .reshape(3, -1)
            .T
        )

        xyz_error_interp = np.abs(np.subtract(xyz_coords, xyz_coords_interp))

        node_uvl_coords = uvl_coords_interp
        uvl_coords = np.vstack([boundary_uvl_coords, node_uvl_coords])

    comm.barrier()
    if comm is not None:
        uvl_coords = comm.bcast(uvl_coords, root=0)
        origin_extent, origin_pos_um, origin_ranges = comm.bcast(
            (origin_extent, origin_pos_um, origin_ranges), root=0
        )

    if rank == 0:
        logger.info("Computing volume distances...")

    ldists_u = []
    ldists_v = []
    obs_uvls = []
    for i, uvl in enumerate(uvl_coords):
        if i % size == rank:
            sample_U = uvl[0]
            sample_V = uvl[1]
            sample_L = uvl[2]
            pos, extent = ip_vol.point_position(sample_U, sample_V, sample_L)

            uvl_pos = pos[0]
            uvl_extent = extent[0]

            obs_uvls.append(uvl)
            ldists_u.append(uvl_pos[0] * origin_extent[0] - origin_pos_um[0])
            ldists_v.append(uvl_pos[1] * origin_extent[1] - origin_pos_um[1])

    distances_u = np.asarray(ldists_u, dtype=np.float32)
    distances_v = np.asarray(ldists_v, dtype=np.float32)
    obs_uvl = np.asarray(np.vstack(obs_uvls), dtype=np.float32)

    return (origin_ranges, obs_uvl, distances_u, distances_v)


def interp_soma_distances(
    comm,
    ip_dist_u,
    ip_dist_v,
    soma_coords,
    layer_extents,
    population_layers,
    interp_chunk_size=1000,
    populations=None,
    allgather=False,
):
    """Interpolates path lengths of cell coordinates along the dimensions of an `RBFVolume` instance.

    Parameters
    ----------
    comm : MPIComm
        mpi4py MPI communicator
    ip_dist_u : RBFInterpolant
        Interpolation function for computing arc distances along the first dimension of the volume.
    ip_dist_v : RBFInterpolant
        Interpolation function for computing arc distances along the second dimension of the volume.
    soma_coords : { population_name : coords_dict }
        | A dictionary that maps each cell population name to a dictionary of coordinates. The dictionary of coordinates must have the following type:
        | coords_dict : { gid : (u, v, l) }
        | where:
        |    - gid: cell identifier
        |    - u, v, l: floating point coordinates
    population_layers: { population_name : layers }
        | A dictionary of population count per layer
        | Argument layers has the following type: { layer_name: count }
    allgather: boolean (default: False)
        if True, the results are gathered from all ranks and combined

    Returns
    -------
    A dictionary of the form:

      { population: { gid: (distance_U, distance_V } }

    """

    rank = comm.rank
    size = comm.size

    if populations is None:
        populations = sorted(soma_coords.keys())

    soma_distances = {}
    for pop in populations:
        coords_dict = soma_coords[pop]
        if rank == 0:
            logger.info(
                "Computing soma distances for %d cells from population %s..."
                % (len(coords_dict), pop)
            )
        count = 0
        local_dist_dict = {}
        pop_layer = population_layers[pop]
        u_obs = []
        v_obs = []
        gids = []
        for gid, coords in coords_dict.items():
            if gid % size == rank:
                soma_u, soma_v, soma_l = coords
                try:
                    assert uvl_in_bounds(coords, layer_extents, pop_layer)
                except Exception as e:
                    logger.error(
                        "gid %i: out of limits error for coordinates: %f %f %f)"
                        % (gid, soma_u, soma_v, soma_l)
                    )
                    raise e

                u_obs.append(np.array([soma_u, soma_v, soma_l]).ravel())
                v_obs.append(np.array([soma_u, soma_v, soma_l]).ravel())
                gids.append(gid)
        if len(u_obs) > 0:
            u_obs_array = np.vstack(u_obs)
            v_obs_array = np.vstack(v_obs)
            distance_u_obs = ip_dist_u(u_obs_array).reshape(-1, 1)
            distance_v_obs = ip_dist_v(v_obs_array).reshape(-1, 1)
            distance_u = np.mean(distance_u_obs, axis=1)
            distance_v = np.mean(distance_v_obs, axis=1)
            try:
                assert np.all(np.isfinite(distance_u))
                assert np.all(np.isfinite(distance_v))
            except Exception as e:
                u_nan_idxs = np.where(np.isnan(distance_u))[0]
                v_nan_idxs = np.where(np.isnan(distance_v))[0]
                logger.error(
                    "Invalid distances: u: %s; v: %s",
                    str(u_obs_array[u_nan_idxs]),
                    str(v_obs_array[v_nan_idxs]),
                )
                raise e

        for (i, gid) in enumerate(gids):
            local_dist_dict[gid] = (distance_u[i], distance_v[i])
            if rank == 0:
                logger.info(
                    "gid %i: distances: %f %f"
                    % (gid, distance_u[i], distance_v[i])
                )
        if allgather:
            dist_dicts = comm.allgather(local_dist_dict)
            combined_dist_dict = {}
            for dist_dict in dist_dicts:
                for k, v in dist_dict.items():
                    combined_dist_dict[k] = v
            soma_distances[pop] = combined_dist_dict
        else:
            soma_distances[pop] = local_dist_dict

    return soma_distances


def make_distance_interpolant(
    comm, geometry_config, make_volume, resolution=[30, 30, 10], nsample=1000
):
    from rbf.interpolate import RBFInterpolant

    rank = comm.rank

    layer_extents = geometry_config["Parametric Surface"]["Layer Extents"]
    (extent_u, extent_v, extent_l) = get_total_extents(layer_extents)

    rotate = geometry_config["Parametric Surface"]["Rotation"]
    origin = geometry_config["Parametric Surface"]["Origin"]

    min_u, max_u = extent_u
    min_v, max_v = extent_v
    min_l, max_l = extent_l

    ## This parameter is used to expand the range of L and avoid
    ## situations where the endpoints of L end up outside of the range
    ## of the distance interpolant
    safety = 0.01

    ip_volume = None
    if rank == 0:
        logger.info(f"Creating volume: min_l = {min_l:f} max_l = {max_l:f}...")
        ip_volume = make_volume(
            (min_u - safety, max_u + safety),
            (min_v - safety, max_v + safety),
            (min_l - safety, max_l + safety),
            resolution=resolution,
            rotate=rotate,
        )

    ip_volume = comm.bcast(ip_volume, root=0)

    interp_sigma = 0.01
    interp_basis = rbf.basis.ga
    interp_order = 1

    if rank == 0:
        logger.info("Computing reference distances...")

    vol_dist = get_volume_distances(
        ip_volume, origin_spec=origin, nsample=nsample, comm=comm
    )
    (origin_ranges, obs_uvl, dist_u, dist_v) = vol_dist

    if rank == 0:
        logger.info("Done computing reference distances...")

    sendcounts = np.array(comm.gather(len(obs_uvl), root=0))
    displs = np.concatenate([np.asarray([0]), np.cumsum(sendcounts)[:-1]])

    obs_uvs = None
    dist_us = None
    dist_vs = None
    if rank == 0:
        obs_uvs = np.zeros((np.sum(sendcounts), 3), dtype=np.float32)
        dist_us = np.zeros(np.sum(sendcounts), dtype=np.float32)
        dist_vs = np.zeros(np.sum(sendcounts), dtype=np.float32)

    uvl_datatype = MPI.FLOAT.Create_contiguous(3).Commit()
    comm.Gatherv(
        sendbuf=obs_uvl,
        recvbuf=(obs_uvs, sendcounts, displs, uvl_datatype),
        root=0,
    )
    uvl_datatype.Free()
    comm.Gatherv(
        sendbuf=dist_u, recvbuf=(dist_us, sendcounts, displs, MPI.FLOAT), root=0
    )
    comm.Gatherv(
        sendbuf=dist_v, recvbuf=(dist_vs, sendcounts, displs, MPI.FLOAT), root=0
    )

    ip_dist_u = None
    ip_dist_v = None
    if rank == 0:
        logger.info("Computing U volume distance interpolants...")
        ip_dist_u = RBFInterpolant(
            obs_uvs,
            dist_us,
            order=interp_order,
            phi=interp_basis,
            sigma=interp_sigma,
        )
        logger.info("Computing V volume distance interpolants...")
        ip_dist_v = RBFInterpolant(
            obs_uvs,
            dist_vs,
            order=interp_order,
            phi=interp_basis,
            sigma=interp_sigma,
        )

    origin_ranges = comm.bcast(origin_ranges, root=0)
    ip_dist_u = comm.bcast(ip_dist_u, root=0)
    ip_dist_v = comm.bcast(ip_dist_v, root=0)

    return origin_ranges, ip_dist_u, ip_dist_v


def measure_distances(
    comm,
    geometry_config,
    soma_coords,
    ip_dist,
    resolution=[30, 30, 10],
    interp_chunk_size=1000,
    allgather=False,
):

    rank = comm.rank

    layer_extents = geometry_config["Parametric Surface"]["Layer Extents"]
    cell_distribution = geometry_config["Cell Distribution"]
    (extent_u, extent_v, extent_l) = get_total_extents(layer_extents)

    rotate = geometry_config["Parametric Surface"]["Rotation"]
    origin = geometry_config["Parametric Surface"]["Origin"]

    origin_ranges, ip_dist_u, ip_dist_v = ip_dist

    origin_ranges = comm.bcast(origin_ranges, root=0)
    ip_dist_u = comm.bcast(ip_dist_u, root=0)
    ip_dist_v = comm.bcast(ip_dist_v, root=0)

    soma_distances = interp_soma_distances(
        comm,
        ip_dist_u,
        ip_dist_v,
        soma_coords,
        layer_extents,
        cell_distribution,
        interp_chunk_size=interp_chunk_size,
        allgather=allgather,
    )

    return soma_distances


def measure_distance_extents(geometry_config, volume):

    layer_extents = geometry_config["Parametric Surface"]["Layer Extents"]

    (extent_u, extent_v, extent_l) = get_total_extents(layer_extents)

    min_u, max_u = extent_u
    min_v, max_v = extent_v
    min_l, max_l = extent_l

    rotate = geometry_config["Parametric Surface"]["Rotation"]
    origin_spec = geometry_config["Parametric Surface"]["Origin"]

    span_U = np.linspace(min_u, max_u, num=2000)
    span_V = np.linspace(min_v, max_v, num=1000)
    span_L = np.linspace(min_l, max_l, num=1000)

    if origin_spec is None:
        coord_U = np.median(span_U)
        coord_V = np.median(span_V)
        coord_L = np.median(span_L)
    else:
        coord_U = origin_spec["U"](span_U)
        coord_V = origin_spec["V"](span_V)
        coord_L = origin_spec["L"](span_L)

    span1_U = np.linspace(min_u, coord_U, num=2000)
    span2_U = np.linspace(coord_U, max_u, num=2000)
    span1_V = np.linspace(min_v, coord_V, num=1000)
    span2_V = np.linspace(coord_V, max_v, num=1000)

    u, v, l = np.meshgrid(span1_U, coord_V, coord_L, indexing="ij")
    u1_xyz = volume(u, v, l, rotate=rotate)
    u, v, l = np.meshgrid(span2_U, coord_V, coord_L, indexing="ij")
    u2_xyz = volume(u, v, l, rotate=rotate)
    u_dist_extent = (
        -np.sum(euclidean_distance(u1_xyz[:-1, :], u1_xyz[1:, :])),
        np.sum(euclidean_distance(u2_xyz[:-1, :], u2_xyz[1:, :])),
    )

    u, v, l = np.meshgrid(coord_U, span1_V, coord_L, indexing="ij")
    v1_xyz = volume(u, v, l, rotate=rotate)
    u, v, l = np.meshgrid(coord_U, span2_V, coord_L, indexing="ij")
    v2_xyz = volume(u, v, l, rotate=rotate)

    v_dist_extent = (
        -np.sum(euclidean_distance(v1_xyz[:-1, :], v1_xyz[1:, :])),
        np.sum(euclidean_distance(v2_xyz[:-1, :], v2_xyz[1:, :])),
    )

    return u_dist_extent, v_dist_extent


def icp_transform(
    comm,
    geometry_config,
    volume,
    make_surface,
    soma_coords,
    projection_ls,
    population_extents,
    rotate=None,
    populations=None,
    icp_iter=1000,
    opt_iter=100,
):
    """
    Uses the iterative closest point (ICP) algorithm of the PCL library to transform soma coordinates onto a surface for a particular L value.
    http://pointclouds.org/documentation/tutorials/iterative_closest_point.php#iterative-closest-point

    """

    import dlib
    import pcl

    rank = comm.rank
    size = comm.size

    if populations is None:
        populations = list(soma_coords.keys())

    srf_resample = 25

    layer_extents = geometry_config["Parametric Surface"]["Layer Extents"]

    (extent_u, extent_v, extent_l) = get_total_extents(layer_extents)

    min_u, max_u = extent_u
    min_v, max_v = extent_v
    min_l, max_l = extent_l

    ## This parameter is used to expand the range of L and avoid
    ## situations where the endpoints of L end up outside of the range
    ## of the distance interpolant
    safety = 0.01

    extent_u = (min_u - safety, max_u + safety)
    extent_v = (min_v - safety, max_v + safety)

    projection_ptclouds = []
    for obs_l in projection_ls:
        srf = make_surface(extent_u, extent_v, obs_l, rotate=rotate)
        U, V = srf._resample_uv(srf_resample, srf_resample)
        meshpts = srf.ev(U, V)
        projection_ptcloud = pcl.PointCloud()
        projection_ptcloud.from_array(meshpts)
        projection_ptclouds.append(projection_ptcloud)

    soma_coords_dict = {}
    for pop in populations:
        coords_dict = soma_coords[pop]
        if rank == 0:
            logger.info(
                f"Computing point transformation for population {pop}..."
            )
        count = 0
        xyz_coords = []
        gids = []
        for gid, coords in coords_dict.items():
            if gid % size == rank:
                soma_u, soma_v, soma_l = coords
                xyz_coords.append(volume(soma_u, soma_v, soma_l, rotate=rotate))
                gids.append(gid)
        xyz_pts = np.vstack(xyz_coords)

        cloud_in = pcl.PointCloud()
        cloud_in.from_array(xyz_pts)

        icp = cloud_in.make_IterativeClosestPoint()

        all_est_xyz_coords = []
        all_est_uvl_coords = []
        all_interp_err = []

        for (k, cloud_prj) in enumerate(projection_ls):
            k_est_xyz_coords = np.zeros((len(gids), 3))
            k_est_uvl_coords = np.zeros((len(gids), 3))
            interp_err = np.zeros((len(gids),))
            converged, transf, estimate, fitness = icp.icp(
                cloud_in, cloud_prj, max_iter=icp_iter
            )
            logger.info(
                f"Transformation of population {pop} has converged: "
                + str(converged)
                + f" score: {fitness:f}"
            )
            for i, gid in zip(list(range(0, estimate.size)), gids):
                est_xyz_coords = estimate[i]
                k_est_xyz_coords[i, :] = est_xyz_coords
                f_uvl_distance = make_uvl_distance(
                    est_xyz_coords, rotate=rotate
                )
                uvl_coords, err = dlib.find_min_global(
                    f_uvl_distance, limits[0], limits[1], opt_iter
                )
                k_est_uvl_coords[i, :] = uvl_coords
                interp_err[
                    i,
                ] = err
                if rank == 0:
                    logger.info(
                        "gid %i: u: %f v: %f l: %f"
                        % (gid, uvl_coords[0], uvl_coords[1], uvl_coords[2])
                    )
            all_est_xyz_coords.append(k_est_xyz_coords)
            all_est_uvl_coords.append(k_est_uvl_coords)
            all_interp_err.append(interp_err)

        coords_dict = {}
        for (i, gid) in enumerate(gids):
            coords_dict[gid] = {
                "X Coordinate": np.asarray(
                    [col[i, 0] for col in all_est_xyz_coords], dtype="float32"
                ),
                "Y Coordinate": np.asarray(
                    [col[i, 1] for col in all_est_xyz_coords], dtype="float32"
                ),
                "Z Coordinate": np.asarray(
                    [col[i, 2] for col in all_est_xyz_coords], dtype="float32"
                ),
                "U Coordinate": np.asarray(
                    [col[i, 0] for col in all_est_uvl_coords], dtype="float32"
                ),
                "V Coordinate": np.asarray(
                    [col[i, 1] for col in all_est_uvl_coords], dtype="float32"
                ),
                "L Coordinate": np.asarray(
                    [col[i, 2] for col in all_est_uvl_coords], dtype="float32"
                ),
                "Interpolation Error": np.asarray(
                    [err[i] for err in all_interp_err], dtype="float32"
                ),
            }

        soma_coords_dict[pop] = coords_dict

    return soma_coords_dict




if __name__ == "__main__":
    #     test_alphavol()
    test_nodes()
