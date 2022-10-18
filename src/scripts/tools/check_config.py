import os
import os.path
import random
import sys
import logging

import click
from miv_simulator.env import Env
from miv_simulator.utils import config_logging, get_script_logger, list_find
from mpi4py import MPI

script_name = os.path.basename(__file__)
logger = get_script_logger(script_name)


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


@click.command()
@click.option("--config", required=True, type=str)
@click.option(
    "--config-prefix",
    required=False,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, config_prefix, verbose):
    """
    Check configuration by loading and instantiate env.

    .. code-block:: bash

       % check-config --config=Microcircuit_Small.yaml --config-prefix=config

    If the simulation configuration is initialized successfully,the code runs without raising any error.
    """

    config_logging(verbose)
    logger = get_script_logger(script_name)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    env = Env(comm=comm, config=config, config_prefix=config_prefix)
    logging.debug(
        f"The environment is loaded successfully from the configuration {config=}, {config_prefix=}."
    )
