import sys
import numpy as np
import click
from biophys_microcircuit import plot, utils
from mpi4py import MPI


def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-path", '-c', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", '-n', type=str, default='Generated Coordinates')
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--subpopulation", required=False, default=-1, type=int)
@click.option("--scale", type=float, default=25.0)
@click.option("--subvol", type=bool, default=False, is_flag=True)
@click.option("--mayavi", "-m", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(config, coords_path, coords_namespace, populations, subpopulation, scale, subvol, mayavi, verbose):

    utils.config_logging(verbose)
    plot.plot_coords_in_volume (populations, coords_path, coords_namespace, config, \
                                subpopulation=subpopulation, subvol=subvol, scale=scale, verbose=verbose, mayavi=mayavi)
        

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("plot_coords_in_volume.py") != -1,sys.argv)+1):])
