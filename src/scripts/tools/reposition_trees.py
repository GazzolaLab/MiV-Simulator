import os, sys, gc, logging, random
import click
import numpy as np
from mpi4py import MPI
from miv_simulator.simulator.reposition_tree import reposition_tree
from miv_simulator.utils import config_logging, get_script_logger, list_find
from miv_simulator.env import Env
from neuroh5.io import (
    NeuroH5TreeGen,
    append_cell_attributes,
    append_cell_trees,
    scatter_read_cell_attributes,
    read_population_ranges,
)
import h5py

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


@click.command()
@click.option("--config", required=True, type=str)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
)
@click.option("--population", required=True, type=str)
@click.option(
    "--forest-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--coords-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--coords-namespace", type=str, default="Coordinates")
@click.option("--template-path", type=str)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--dry-run", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def main(
    config,
    config_prefix,
    population,
    forest_path,
    coords_path,
    coords_namespace,
    template_path,
    output_path,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    dry_run,
    verbose,
):
    """

    :param population: str
    :param forest_path: str (path)
    :param output_path: str (path)
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param verbose: bool
    """

    config_logging(verbose)
    logger = get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info("%i ranks have been allocated" % comm.size)

    env = Env(
        comm=comm,
        config_file=config,
        config_prefix=config_prefix,
        template_paths=template_path,
    )

    if rank == 0:
        if not os.path.isfile(output_path):
            input_file = h5py.File(forest_path, "r")
            output_file = h5py.File(output_path, "w")
            input_file.copy("/H5Types", output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    (forest_pop_ranges, _) = read_population_ranges(forest_path)
    (forest_population_start, forest_population_count) = forest_pop_ranges[
        population
    ]

    (pop_ranges, _) = read_population_ranges(output_path)

    (population_start, population_count) = pop_ranges[population]

    color = 0
    if rank == 0:
        color = 1
    comm0 = comm.Split(color, 0)

    soma_coords = None
    if rank == 0:
        logger.info(f"Reading {population} coordinates...")
        coords_iter = scatter_read_cell_attributes(
            coords_path,
            population,
            comm=comm0,
            mask={"X Coordinate", "Y Coordinate", "Z Coordinate"},
            namespaces=[coords_namespace],
            io_size=io_size,
        )
        soma_coords = {}
        soma_coords[population] = {
            k: (
                float(v["X Coordinate"][0]),
                float(v["Y Coordinate"][0]),
                float(v["Z Coordinate"][0]),
            )
            for (k, v) in coords_iter[coords_namespace]
        }

    comm.barrier()
    comm0.Free()

    soma_coords = comm.bcast(soma_coords, root=0)

    new_trees_dict = {}
    for gid, tree_dict in NeuroH5TreeGen(
        forest_path,
        population,
        io_size=io_size,
        comm=comm,
        cache_size=cache_size,
        topology=False,
    ):
        if gid is not None:
            logger.info("Rank %d received gid %d" % (rank, gid))
            cell_coords = soma_coords[population][gid]
            new_tree_dict = reposition_tree(
                tree_dict, cell_coords, env.SWC_Types
            )
            new_trees_dict[gid] = new_tree_dict

    if not dry_run:
        if (not dry_run) and (rank == 0):
            logger.info(f"Appending repositioned trees to {output_path}...")
        append_cell_trees(
            output_path, population, new_trees_dict, io_size=io_size, comm=comm
        )

    comm.barrier()
    if (not dry_run) and (rank == 0):
        logger.info(
            f"Appended {len(new_trees_dict)} repositioned trees to {output_path}"
        )


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                list_find(
                    lambda x: os.path.basename(x) == os.path.basename(__file__),
                    sys.argv,
                )
                + 1
            ) :
        ]
    )
