import base64
import gc
import pickle
import sys

import h5py
import numpy as np
from miv_simulator import utils
from miv_simulator.env import Env
from miv_simulator.geometry.geometry import make_distance_interpolant
from miv_simulator.geometry.geometry import (
    measure_distances as geometry_measure_distances,
)
from miv_simulator.volume import make_network_volume
from mpi4py import MPI
from neuroh5.io import append_cell_attributes, bcast_cell_attributes

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


def measure_distances(
    config,
    coords_path,
    coords_namespace,
    geometry_path,
    populations,
    resolution,
    nsample,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    verbose,
    config_prefix="",
):
    utils.config_logging(verbose)
    logger = utils.get_script_logger(__file__)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config=config, config_prefix=config_prefix)
    output_path = coords_path

    soma_coords = {}

    if rank == 0:
        logger.info("Reading population coordinates...")

    for population in sorted(populations):
        coords = bcast_cell_attributes(
            coords_path, population, 0, namespace=coords_namespace, comm=comm
        )

        soma_coords[population] = {
            k: (
                v["U Coordinate"][0],
                v["V Coordinate"][0],
                v["L Coordinate"][0],
            )
            for (k, v) in coords
        }
        del coords
        gc.collect()

    has_ip_dist = False
    origin_ranges = None
    ip_dist_u = None
    ip_dist_v = None
    ip_dist_path = "Distance Interpolant/%d/%d/%d" % tuple(resolution)
    if rank == 0:
        if geometry_path is not None:
            f = h5py.File(geometry_path, "a")
            pkl_path = f"{ip_dist_path}/ip_dist.pkl"
            if pkl_path in f:
                has_ip_dist = True
                ip_dist_dset = f[pkl_path]
                origin_ranges, ip_dist_u, ip_dist_v = pickle.loads(
                    base64.b64decode(ip_dist_dset[()])
                )
            f.close()
    has_ip_dist = env.comm.bcast(has_ip_dist, root=0)

    if not has_ip_dist:
        if rank == 0:
            logger.info("Creating distance interpolant...")
        (origin_ranges, ip_dist_u, ip_dist_v) = make_distance_interpolant(
            env.comm,
            geometry_config=env.geometry,
            make_volume=make_network_volume,
            resolution=resolution,
            nsample=nsample,
        )
        if rank == 0:
            if geometry_path is not None:
                f = h5py.File(geometry_path, "a")
                pkl_path = f"{ip_dist_path}/ip_dist.pkl"
                pkl = pickle.dumps((origin_ranges, ip_dist_u, ip_dist_v))
                pklstr = base64.b64encode(pkl)
                f[pkl_path] = pklstr
                f.close()

    ip_dist = (origin_ranges, ip_dist_u, ip_dist_v)
    if rank == 0:
        logger.info("Measuring soma distances...")

    soma_distances = geometry_measure_distances(
        env.comm, env.geometry, soma_coords, ip_dist, resolution=resolution
    )

    for population in list(sorted(soma_distances.keys())):

        if rank == 0:
            logger.info(f"Writing distances for population {population}...")

        dist_dict = soma_distances[population]
        attr_dict = {}
        for k, v in dist_dict.items():
            attr_dict[k] = {
                "U Distance": np.asarray([v[0]], dtype=np.float32),
                "V Distance": np.asarray([v[1]], dtype=np.float32),
            }
        append_cell_attributes(
            output_path,
            population,
            attr_dict,
            namespace="Arc Distances",
            comm=comm,
            io_size=io_size,
            chunk_size=chunk_size,
            value_chunk_size=value_chunk_size,
            cache_size=cache_size,
        )
        if rank == 0:
            f = h5py.File(output_path, "a")
            f["Populations"][population]["Arc Distances"].attrs[
                "Reference U Min"
            ] = origin_ranges[0][0]
            f["Populations"][population]["Arc Distances"].attrs[
                "Reference U Max"
            ] = origin_ranges[0][1]
            f["Populations"][population]["Arc Distances"].attrs[
                "Reference V Min"
            ] = origin_ranges[1][0]
            f["Populations"][population]["Arc Distances"].attrs[
                "Reference V Max"
            ] = origin_ranges[1][1]
            f.close()

    comm.Barrier()
