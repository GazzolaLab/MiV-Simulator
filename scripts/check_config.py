##
## Load configuration and instantiate env.
##

import os, sys, os.path, itertools, random, pickle, logging, click, gc
import math
from mpi4py import MPI
import h5py
import numpy as np
from neuroh5.io import append_cell_attributes, read_population_ranges
from biophys_microcircuit.env import Env
from biophys_microcircuit.utils import get_script_logger, config_logging, list_find, viewitems

script_name = os.path.basename(__file__)
logger = get_script_logger(script_name)

def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook

def random_subset( iterator, K ):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item

    return result


@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=False, type=click.Path(exists=True, file_okay=False, dir_okay=True), default="config")
@click.option("--verbose", '-v', type=bool, default=False, is_flag=True)
def main(config, config_prefix, verbose):

    config_logging(verbose)
    logger = get_script_logger(script_name)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    env = Env(comm=comm, config_file=config, config_prefix=config_prefix)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
