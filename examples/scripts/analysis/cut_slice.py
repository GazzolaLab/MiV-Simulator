import os
import sys
from collections import defaultdict

import click
import numpy as np
import yaml
from miv_simulator import utils
from miv_simulator.env import Env
from miv_simulator.utils import io as io_utils
from mpi4py import MPI
from neuroh5.io import read_cell_attributes, read_population_ranges

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


@click.command()
@click.option(
    "--arena-id",
    required=False,
    type=str,
    help="name of arena used for spatial stimulus",
)
@click.option("--config", "-c", required=True, type=str)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
    help="path to directory containing network config files",
)
@click.option(
    "--dataset-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory containing required neuroh5 data files",
)
@click.option("--distances-namespace", "-n", type=str, default="Arc Distances")
@click.option("--distance-limits", type=(float, float))
@click.option(
    "--spike-input-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="path to file for input spikes when cell selection is specified",
)
@click.option(
    "--spike-input-namespace",
    required=False,
    type=str,
    help="namespace for input spikes when cell selection is specified",
)
@click.option(
    "--spike-input-attr",
    required=False,
    type=str,
    help="attribute name for input spikes when cell selection is specified",
)
@click.option(
    "--output-path",
    "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option("--io-size", type=int, default=-1)
@click.option(
    "--trajectory-id",
    required=True,
    type=str,
    help="name of trajectory used for spatial stimulus",
)
@click.option("--write-selection", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def main(
    arena_id,
    config,
    config_prefix,
    dataset_prefix,
    distances_namespace,
    distance_limits,
    spike_input_path,
    spike_input_namespace,
    spike_input_attr,
    output_path,
    io_size,
    trajectory_id,
    write_selection,
    verbose,
):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank
    if io_size == -1:
        io_size = comm.size

    env = Env(
        comm=comm,
        config_file=config,
        config_prefix=config_prefix,
        dataset_prefix=dataset_prefix,
        results_path=output_path,
        spike_input_path=spike_input_path,
        spike_input_namespace=spike_input_namespace,
        spike_input_attr=spike_input_attr,
        arena_id=arena_id,
        trajectory_id=trajectory_id,
        io_size=io_size,
    )

    if rank == 0:
        logger.info("%i ranks have been allocated" % comm.size)

    pop_ranges, pop_size = read_population_ranges(
        env.connectivity_file_path, comm=comm
    )

    distance_U_dict = {}
    distance_V_dict = {}
    range_U_dict = {}
    range_V_dict = {}

    selection_dict = defaultdict(set)

    comm0 = env.comm.Split(2 if rank == 0 else 0, 0)

    if rank == 0:
        for population in pop_ranges:
            distances = read_cell_attributes(
                env.data_file_path,
                population,
                namespace=distances_namespace,
                comm=comm0,
            )
            soma_distances = {
                k: (v["U Distance"][0], v["V Distance"][0])
                for (k, v) in distances
            }
            del distances

            numitems = len(list(soma_distances.keys()))
            logger.info(
                "read %s distances (%i elements)" % (population, numitems)
            )

            if numitems == 0:
                continue

            distance_U_array = np.asarray(
                [soma_distances[gid][0] for gid in soma_distances]
            )
            distance_V_array = np.asarray(
                [soma_distances[gid][1] for gid in soma_distances]
            )

            U_min = np.min(distance_U_array)
            U_max = np.max(distance_U_array)
            V_min = np.min(distance_V_array)
            V_max = np.max(distance_V_array)

            range_U_dict[population] = (U_min, U_max)
            range_V_dict[population] = (V_min, V_max)

            distance_U = {gid: soma_distances[gid][0] for gid in soma_distances}
            distance_V = {gid: soma_distances[gid][1] for gid in soma_distances}

            distance_U_dict[population] = distance_U
            distance_V_dict[population] = distance_V

            min_dist = U_min
            max_dist = U_max
            if distance_limits:
                min_dist = distance_limits[0]
                max_dist = distance_limits[1]

            selection_dict[population] = {
                k
                for k in distance_U
                if (distance_U[k] >= min_dist) and (distance_U[k] <= max_dist)
            }

        yaml_output_dict = {}
        for k, v in utils.viewitems(selection_dict):
            yaml_output_dict[k] = list(v)

        yaml_output_path = f"{output_path}/DG_slice.yaml"
        with open(yaml_output_path, "w") as outfile:
            yaml.dump(yaml_output_dict, outfile)

        del yaml_output_dict

    env.comm.barrier()

    write_selection_file_path = None
    if write_selection:
        write_selection_file_path = (
            f"{env.results_path}/{env.modelName}_selection.h5"
        )

    if write_selection_file_path is not None:
        if rank == 0:
            io_utils.mkout(env, write_selection_file_path)
        env.comm.barrier()
        selection_dict = env.comm.bcast(dict(selection_dict), root=0)
        env.cell_selection = selection_dict
        io_utils.write_cell_selection(env, write_selection_file_path)
        input_selection = io_utils.write_connection_selection(
            env, write_selection_file_path
        )
        io_utils.write_input_cell_selection(
            env, input_selection, write_selection_file_path
        )


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                utils.list_find(
                    lambda x: os.path.basename(x) == os.path.basename(__file__),
                    sys.argv,
                )
                + 1
            ) :
        ]
    )
