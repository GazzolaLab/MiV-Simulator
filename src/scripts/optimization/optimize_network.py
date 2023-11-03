import os
import sys
import click
from miv_simulator import utils
from miv_simulator.optimize_network import optimize_network


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
    help="optimization configuration file name",
)
@click.option(
    "--target-features-path",
    required=False,
    type=click.Path(),
    help="path to neuroh5 file containing target rate maps used for rate optimization",
)
@click.option(
    "--target-features-namespace",
    type=str,
    required=False,
    default="Input Spikes",
    help="namespace containing target rate maps used for rate optimization",
)
@click.option(
    "--optimize-file-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="results",
)
@click.option(
    "--optimize-file-name",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option("--nprocs-per-worker", type=int, default=1)
@click.option("--n-epochs", type=int, default=1)
@click.option("--n-initial", type=int, default=30)
@click.option("--initial-maxiter", type=int, default=50)
@click.option("--initial-method", type=str, default="glp")
@click.option("--optimizer-method", type=str, default="nsga2")
@click.option("--population-size", type=int, default=100)
@click.option("--num-generations", type=int, default=200)
@click.option("--resample-fraction", type=float)
@click.option("--mutation-rate", type=float)
@click.option("--collective-mode", type=str, default="gather")
@click.option("--spawn-startup-wait", type=int, default=3)
@click.option("--spawn-workers", is_flag=True)
@click.option("--get-best", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def main_cli(
    config_path,
    target_features_path,
    target_features_namespace,
    optimize_file_dir,
    optimize_file_name,
    nprocs_per_worker,
    n_epochs,
    n_initial,
    initial_maxiter,
    initial_method,
    optimizer_method,
    population_size,
    num_generations,
    resample_fraction,
    mutation_rate,
    collective_mode,
    spawn_startup_wait,
    spawn_workers,
    get_best,
    verbose,
):
    optimize_network(
        config_path,
        target_features_path,
        target_features_namespace,
        optimize_file_dir,
        optimize_file_name,
        nprocs_per_worker,
        n_epochs,
        n_initial,
        initial_maxiter,
        initial_method,
        optimizer_method,
        population_size,
        num_generations,
        resample_fraction,
        mutation_rate,
        collective_mode,
        spawn_startup_wait,
        spawn_workers,
        get_best,
        verbose,
    )
