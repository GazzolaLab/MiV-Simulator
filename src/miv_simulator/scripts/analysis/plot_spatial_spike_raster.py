import os
import sys
import click
from miv_simulator import plotting as plot
from miv_simulator import utils

script_name = os.path.basename(__file__)


@click.command()
@click.option("--spike-events-path", "-p", required=True, type=click.Path())
@click.option("--spike-events-namespace", "-n", type=str, default="Spike Events")
@click.option("--coords-path", "-c", required=True, type=click.Path())
@click.option("--distances-namespace", "-d", type=str, default="Arc Distances")
@click.option("--populations", "-i", type=str, multiple=True)
@click.option(
    "--include-artificial/--exclude-artificial",
    type=bool,
    default=True,
    is_flag=True,
)
@click.option("--max-spikes", type=int, default=int(1e6))
@click.option("--t-variable", type=str, default="t")
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--t-step", type=float, default=5.0)
@click.option("--font-size", type=float, default=14)
@click.option("--save-fig", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(
    spike_events_path,
    spike_events_namespace,
    coords_path,
    distances_namespace,
    populations,
    include_artificial,
    max_spikes,
    t_variable,
    t_max,
    t_min,
    t_step,
    font_size,
    save_fig,
    verbose,
):
    utils.config_logging(verbose)

    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    if not populations:
        populations = ["eachPop"]

    plot.plot_spatial_spike_raster(
        spike_events_path,
        spike_events_namespace,
        coords_path,
        distances_namespace,
        include=populations,
        include_artificial=include_artificial,
        time_range=time_range,
        time_variable=t_variable,
        time_step=t_step,
        max_spikes=max_spikes,
        fontSize=font_size,
        saveFig=save_fig,
    )


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                utils.list_find(lambda x: os.path.basename(x) == script_name, sys.argv)
                + 1
            ) :
        ]
    )
