##
## Generates distance-weighted random connectivity between the specified populations.
##

import os
import os.path
import sys
import click
from mpi4py import MPI
import h5py
import miv_simulator.utils as utils
from miv_simulator.env import Env
from miv_simulator.gapjunctions import generate_gj_connections
from miv_simulator.utils.neuron import configure_hoc_env
from neuroh5.io import (
    read_cell_attributes,
    read_population_ranges,
)

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--template-path", type=str, default="templates")
@click.option(
    "--types-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--forest-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--connectivity-path", required=True, type=click.Path())
@click.option("--connectivity-namespace", type=str, default="Gap Junctions")
@click.option(
    "--coords-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--coords-namespace", type=str, default="Coordinates")
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(
    config,
    template_path,
    types_path,
    forest_path,
    connectivity_path,
    connectivity_namespace,
    coords_path,
    coords_namespace,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    write_size,
    verbose,
    dry_run,
):
    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config=config, template_paths=template_path)
    configure_hoc_env(env)

    gj_config = env.gapjunctions
    gj_seed = int(env.model_config["Random Seeds"]["Gap Junctions"])

    soma_coords = {}

    if (not dry_run) and (rank == 0):
        if not os.path.isfile(connectivity_path):
            input_file = h5py.File(types_path, "r")
            output_file = h5py.File(connectivity_path, "w")
            input_file.copy("/H5Types", output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    population_ranges = read_population_ranges(coords_path)[0]
    populations = sorted(population_ranges.keys())

    if rank == 0:
        logger.info("Reading population coordinates...")

    color = 0
    if rank == 0:
        color = 1
    comm0 = comm.Split(color, 0)

    soma_coords = {}
    for population in populations:
        if rank == 0:
            logger.info(f"Reading {population} coordinates...")
            coords_iter = read_cell_attributes(
                coords_path,
                population,
                comm=comm0,
                mask=set(["X Coordinate", "Y Coordinate", "Z Coordinate"]),
                namespace=coords_namespace,
            )

            soma_coords[population] = {
                k: (
                    float(v["X Coordinate"][0]),
                    float(v["Y Coordinate"][0]),
                    float(v["Z Coordinate"][0]),
                )
                for (k, v) in coords_iter
            }

    comm.barrier()
    comm0.Free()

    soma_coords = comm.bcast(soma_coords, root=0)

    generate_gj_connections(
        env,
        forest_path,
        soma_coords,
        gj_config,
        gj_seed,
        connectivity_namespace,
        connectivity_path,
        io_size,
        chunk_size,
        value_chunk_size,
        cache_size,
        dry_run=dry_run,
    )

    MPI.Finalize()


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
