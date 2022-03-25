import sys, os
import click
from biophys_microcircuit import plot
from neuroh5.io import read_population_ranges, read_tree_selection

script_name = os.path.basename(__file__)


def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

@click.command()
@click.option("--forest-path", '-p', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--population", '-i', type=str, required=True)
@click.option("--gid", '-g', type=int, required=True)
@click.option("--conn-loc/--no-conn-loc", default=True)
@click.option("--line-width", type=float, default=1.0)
@click.option("--color-edge-scalars/--no-color-edge-scalars", default=True)
@click.option("--mayavi", "-m", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(forest_path, population, gid, conn_loc, line_width, color_edge_scalars, mayavi, verbose):

    (tree_iter, _) = read_tree_selection(forest_path, population, selection=[gid])
    (gid,tree_dict) = next(tree_iter)
        
    plot.plot_cell_tree (population, gid, tree_dict, \
                         line_width=line_width,  \
                         color_edge_scalars=color_edge_scalars,
                         conn_loc=conn_loc, mayavi=mayavi)

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
