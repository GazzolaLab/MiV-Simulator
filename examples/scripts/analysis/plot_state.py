import os
import re
import sys

import click
from miv_simulator import plottting as plot
from miv_simulator import statedata, utils

script_name = os.path.basename(__file__)


@click.command()
@click.option("--state-path", "-p", required=True, type=click.Path())
@click.option("--state-namespace", "-n", type=str)
@click.option("--state-namespace-pattern", type=str)
@click.option("--populations", "-i", type=str, multiple=True)
@click.option("--max-units", type=int, default=1)
@click.option("--gid", "-g", type=int, default=None, multiple=True)
@click.option("--t-variable", type=str, default="t")
@click.option("--state-variable", type=str, default="v")
@click.option("--t-max", type=float)
@click.option("--t-min", type=float)
@click.option("--font-size", type=float, default=14)
@click.option("--colormap", type=str)
@click.option("--lowpass-plot", type=bool, default=False, is_flag=True)
@click.option("--query", "-q", type=bool, default=False, is_flag=True)
@click.option("--reduce", type=bool, default=False, is_flag=True)
@click.option("--distance", type=bool, default=False, is_flag=True)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(
    state_path,
    state_namespace,
    state_namespace_pattern,
    populations,
    max_units,
    gid,
    t_variable,
    state_variable,
    t_max,
    t_min,
    font_size,
    colormap,
    lowpass_plot,
    query,
    reduce,
    distance,
    verbose,
):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)

    if reduce and distance:
        raise RuntimeError(
            "Options --reduce and --distance are mutually exclusive"
        )

    if t_max is None:
        time_range = None
    else:
        if t_min is None:
            time_range = [0.0, t_max]
        else:
            time_range = [t_min, t_max]

    if not populations:
        populations = ["eachPop"]
    else:
        populations = list(populations)

    namespace_id_lst, attr_info_dict = statedata.query_state(
        state_path, populations
    )

    if query:
        for population in populations:
            for this_namespace_id in namespace_id_lst:
                if this_namespace_id not in attr_info_dict[population]:
                    continue
                print(
                    f"Population {population}; Namespace: {str(this_namespace_id)}"
                )
                for attr_name, attr_cell_index in attr_info_dict[population][
                    this_namespace_id
                ]:
                    print(f"\tAttribute: {str(attr_name)}")
                    for i in attr_cell_index:
                        print("\t%d" % i)
        sys.exit()

    state_namespaces = []
    if state_namespace is not None:
        state_namespaces.append(state_namespace)

    if state_namespace_pattern is not None:
        for namespace_id in namespace_id_lst:
            m = re.match(state_namespace_pattern, namespace_id)
            if m:
                state_namespaces.append(namespace_id)

    if len(gid) == 0:
        gid = None

    kwargs = {}
    if colormap is not None:
        kwargs["colormap"] = colormap

    plot.plot_intracellular_state(
        state_path,
        state_namespaces,
        include=populations,
        time_range=time_range,
        time_variable=t_variable,
        state_variable=state_variable,
        lowpass_plot=lowpass_plot,
        max_units=max_units,
        gid_set=gid,
        reduce=reduce,
        distance=distance,
        fontSize=font_size,
        saveFig=True,
        **kwargs,
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
