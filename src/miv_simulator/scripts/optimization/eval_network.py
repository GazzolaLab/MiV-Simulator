import click
from miv_simulator.eval_network import eval_network


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--config-path",
    required=True,
    type=str,
    help="Operational optimization configuration file (YAML)",
)
@click.option(
    "--params-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="JSON file containing optimized network parameters",
)
@click.option(
    "--params-label",
    default=None,
    type=str,
    help="Label (top-level key) to read from the JSON file (default: first key)",
)
@click.option(
    "--t-start",
    default=50.0,
    type=float,
    help="Start time (ms) for feature extraction (default: 50.0)",
)
@click.option(
    "--output-path",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Optional path to write JSON evaluation results",
)
@click.option("--verbose", "-v", is_flag=True)
def main_cli(config_path, params_path, params_label, t_start, output_path, verbose):
    network_args = click.get_current_context().args
    network_config = {}
    for arg in network_args:
        kv = arg.split("=")
        if len(kv) > 1:
            k, v = kv
            network_config[k.replace("--", "").replace("-", "_")] = v
        else:
            k = kv[0]
            network_config[k.replace("--", "").replace("-", "_")] = True

    eval_network(
        config_path=config_path,
        params_path=params_path,
        params_label=params_label,
        t_start=t_start,
        output_path=output_path,
        verbose=verbose,
        **network_config,
    )
