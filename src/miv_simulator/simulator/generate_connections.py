import gc
import os
import sys
import random
import h5py
from miv_simulator import utils
from miv_simulator.connections import (
    ConnectionProb,
    generate_uv_distance_connections,
)
from miv_simulator import config
from miv_simulator.env import Env
from miv_simulator.utils.neuron import configure_hoc
from mpi4py import MPI
from neuroh5.io import (
    read_cell_attributes,
    read_population_names,
    read_population_ranges,
)
from typing import Optional, Union, Tuple, Dict

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


# !for imperative API, use distance connections instead
def generate_distance_connections(
    config,
    include,
    forest_path,
    connectivity_path,
    connectivity_namespace,
    coords_path,
    coords_namespace,
    synapses_namespace,
    distances_namespace,
    resolution,
    interp_chunk_size,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    write_size,
    verbose,
    dry_run,
    debug,
    config_prefix="",
):
    utils.config_logging(verbose)

    env = Env(comm=MPI.COMM_WORLD, config=config, config_prefix=config_prefix)

    configure_hoc(
        dt=env.dt,
        tstop=env.tstop,
        celsius=env.globals.get("celsius", None),
    )

    synapse_seed = int(
        env.model_config["Random Seeds"]["Synapse Projection Partitions"]
    )

    connectivity_seed = int(
        env.model_config["Random Seeds"]["Distance-Dependent Connectivity"]
    )
    cluster_seed = int(env.model_config["Random Seeds"]["Connectivity Clustering"])

    return generate_connections(
        filepath=coords_path,
        forest_filepath=forest_path,
        include_forest_populations=include,
        synapses=env.connection_config,
        axon_extents=env.connection_extents,
        output_filepath=connectivity_path,
        connectivity_namespace=connectivity_namespace,
        coordinates_namespace=coords_namespace,
        synapses_namespace=synapses_namespace,
        distances_namespace=distances_namespace,
        populations_dict=env.Populations,
        io_size=io_size,
        chunk_size=chunk_size,
        value_chunk_size=value_chunk_size,
        cache_size=cache_size,
        write_size=write_size,
        dry_run=dry_run,
        seeds=(synapse_seed, connectivity_seed, cluster_seed),
    )


def generate_connections(
    filepath: str,
    forest_filepath: str,
    include_forest_populations: Optional[list],
    synapses: config.Synapses,
    axon_extents: config.AxonExtents,
    output_filepath: Optional[str],
    connectivity_namespace: str,
    coordinates_namespace: str,
    synapses_namespace: str,
    distances_namespace: str,
    populations_dict: Dict[str, int],
    io_size: int,
    chunk_size: int,
    value_chunk_size: int,
    cache_size: int,
    write_size: int,
    dry_run: bool,
    seeds: Union[Tuple[int], int, None],
):
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    configure_hoc()

    if (not dry_run) and (rank == 0):
        if not os.path.isfile(output_filepath):
            input_file = h5py.File(filepath, "r")
            output_file = h5py.File(output_filepath, "w")
            input_file.copy("/H5Types", output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    population_ranges = read_population_ranges(filepath)[0]
    populations = sorted(list(population_ranges.keys()))

    color = 0
    if rank == 0:
        color = 1
    comm0 = comm.Split(color, 0)

    soma_distances = {}
    soma_coords = {}
    for population in populations:
        if rank == 0:
            logger.info(f"Reading {population} coordinates...")
            coords_iter = read_cell_attributes(
                filepath,
                population,
                comm=comm0,
                mask={"U Coordinate", "V Coordinate", "L Coordinate"},
                namespace=coordinates_namespace,
            )
            distances_iter = read_cell_attributes(
                filepath,
                population,
                comm=comm0,
                mask={"U Distance", "V Distance"},
                namespace=distances_namespace,
            )

            soma_coords[population] = {
                k: (
                    float(v["U Coordinate"][0]),
                    float(v["V Coordinate"][0]),
                    float(v["L Coordinate"][0]),
                )
                for (k, v) in coords_iter
            }

            distances = {
                k: (float(v["U Distance"][0]), float(v["V Distance"][0]))
                for (k, v) in distances_iter
            }

            if len(distances) > 0:
                soma_distances[population] = distances

            gc.collect()

    comm.barrier()
    comm0.Free()

    soma_distances = comm.bcast(soma_distances, root=0)
    soma_coords = comm.bcast(soma_coords, root=0)

    forest_populations = sorted(read_population_names(forest_filepath))
    if (include_forest_populations is None) or (len(include_forest_populations) == 0):
        destination_populations = forest_populations
    else:
        destination_populations = []
        for p in include_forest_populations:
            if p in forest_populations:
                destination_populations.append(p)
    if rank == 0:
        logger.info(
            f"Generating connectivity for populations {destination_populations}..."
        )

    # !for imperative API, does not seem applicable any longer
    # if len(soma_distances) == 0:
    #     (origin_ranges, ip_dist_u, ip_dist_v) = make_distance_interpolant(
    #         env, resolution=resolution, nsample=interp_chunk_size
    #     )
    #     ip_dist = (origin_ranges, ip_dist_u, ip_dist_v)
    #     soma_distances = measure_distances(
    #         env, soma_coords, ip_dist, resolution=resolution
    #     )

    for destination_population in destination_populations:
        if rank == 0:
            logger.info(
                f"Generating connection probabilities for population {destination_population}..."
            )

        connection_prob = ConnectionProb(
            destination_population,
            soma_coords,
            soma_distances,
            axon_extents,
        )

        if rank == 0:
            logger.info(
                f"Generating connections for population {destination_population}..."
            )

        if seeds is None or isinstance(seeds, int):
            r = random.Random(seeds)
            seeds = [r.randint(0, 2**32 - 1) for _ in range(3)]

        generate_uv_distance_connections(
            comm,
            populations_dict,
            synapses,
            connection_prob,
            forest_filepath,
            seeds[0],
            seeds[1],
            seeds[2],
            synapses_namespace,
            connectivity_namespace,
            output_filepath,
            io_size,
            chunk_size,
            value_chunk_size,
            cache_size,
            write_size,
            dry_run=dry_run,
            debug=False,
        )
    MPI.Finalize()
