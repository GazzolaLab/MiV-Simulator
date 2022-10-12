import gc, math, os, sys
import click
from mpi4py import MPI
from miv_simulator import plotting as plot
from miv_simulator import utils
from miv_simulator.env import Env

script_name = os.path.basename(__file__)


@click.command()
@click.option("--config", required=True, type=str)
@click.option(
    "--config-prefix",
    default="config",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option("--connectivity-path", "-p", required=True, type=click.Path())
@click.option("--coords-path", "-c", required=True, type=click.Path())
@click.option("--distances-namespace", type=str, default="Arc Distances")
@click.option("--target-gid", "-g", type=int)
@click.option("--destination", "-d", type=str)
@click.option("--source", "-s", type=str)
@click.option("--extent-type", "-t", type=str, default="global")
@click.option("--direction", type=str, default="in")
@click.option("--normed", type=bool, default=False, is_flag=True)
@click.option("--bin-size", type=float, default=20.0)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-format", type=str, default="png")
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(
    config,
    config_prefix,
    connectivity_path,
    coords_path,
    distances_namespace,
    target_gid,
    destination,
    source,
    extent_type,
    direction,
    normed,
    bin_size,
    font_size,
    fig_format,
    verbose,
):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(script_name))

    env = Env(config_file=config, config_prefix=config_prefix)

    plot.plot_single_vertex_dist(
        env,
        connectivity_path,
        coords_path,
        distances_namespace,
        target_gid,
        destination,
        source,
        direction=direction,
        normed=normed,
        extent_type=extent_type,
        bin_size=bin_size,
        fontSize=font_size,
        saveFig=True,
        figFormat=fig_format,
    )


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                utils.list_find(
                    lambda x: os.path.basename(x) == script_name, sys.argv
                )
                + 1
            ) :
        ]
    )
