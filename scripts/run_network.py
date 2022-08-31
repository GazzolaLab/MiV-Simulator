#!/usr/bin/env python
"""
Network model simulation script.
"""
__author__ = "See AUTHORS.md"

import os
import sys

import click
import numpy as np
from miv_simulator import network
from miv_simulator.env import Env
from miv_simulator.utils import config_logging, list_find
from mpi4py import MPI


def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


@click.command()
@click.option(
    "--arena-id",
    required=False,
    type=str,
    help="name of arena used for stimulus",
)
@click.option(
    "--cell-selection-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="name of file specifying subset of cells gids to be instantiated",
)
@click.option(
    "--config-file",
    required=True,
    type=str,
    help="model configuration file name",
)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
    help="path to directory containing network and cell mechanism config files",
)
@click.option(
    "--template-paths",
    type=str,
    default="templates",
    help="colon-separated list of paths to directories containing hoc cell templates",
)
@click.option(
    "--hoc-lib-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=".",
    help="path to directory containing required hoc libraries",
)
@click.option(
    "--dataset-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory containing required neuroh5 data files",
)
@click.option(
    "--results-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory where output files will be written",
)
@click.option(
    "--results-id",
    type=str,
    required=False,
    default=None,
    help="identifier that is used to name neuroh5 namespaces that contain output spike and "
    "intracellular trace data",
)
@click.option(
    "--node-rank-file",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="name of file specifying assignment of cell gids to MPI ranks",
)
@click.option(
    "--io-size",
    type=int,
    default=0,
    help="the number of MPI ranks to be used for I/O operations",
)
@click.option(
    "--use-cell-attr-gen",
    is_flag=True,
    help="whether to use cell attribute generator for I/O operations",
)
@click.option(
    "--cell-attr-gen-cache-size",
    type=int,
    default=10,
    help="cell attribute generator cache readahead size",
)
@click.option(
    "--recording-profile",
    type=str,
    default="Network default",
    help="intracellular recording profile to use",
)
@click.option(
    "--output-syn-spike-count",
    is_flag=True,
    help="record the per-cell number of spikes received from each pre-synaptic source",
)
@click.option(
    "--use-coreneuron", is_flag=True, help="use CoreNEURON for simulation"
)
@click.option(
    "--stimulus-id", required=False, type=str, help="name of input stimulus"
)
@click.option(
    "--tstop", type=int, default=1, help="physical time to simulate (ms)"
)
@click.option(
    "--v-init",
    type=float,
    default=-75.0,
    help="initialization membrane potential (mV)",
)
@click.option(
    "--stimulus-onset",
    type=float,
    default=1.0,
    help="starting time of stimulus (ms)",
)
@click.option(
    "--max-walltime-hours",
    type=float,
    default=1.0,
    help="maximum wall time (hours)",
)
@click.option(
    "--microcircuit-inputs",
    is_flag=True,
    help="initialize intrinsic microcircuit inputs (True by default when cell selection is provided)",
)
@click.option(
    "--checkpoint-clear-data",
    is_flag=True,
    help="delete simulation data from memory after it has been saved",
)
@click.option(
    "--checkpoint-interval",
    type=float,
    default=500.0,
    help="checkpoint interval in ms of simulation time",
)
@click.option(
    "--results-write-time",
    type=float,
    default=360.0,
    help="time to write out results at end of simulation",
)
@click.option(
    "--spike-input-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="path to file for input spikes when cell selection is specified",
)
@click.option(
    "--spike-input-namespace",
    required=False,
    type=str,
    help="namespace for input spikes when cell selection is specified",
)
@click.option(
    "--spike-input-attr",
    required=False,
    type=str,
    help="attribute name for input spikes when cell selection is specified",
)
@click.option("--dt", type=float, default=0.025, help="")
@click.option(
    "--ldbal",
    is_flag=True,
    help="estimate load balance based on cell complexity",
)
@click.option(
    "--lptbal",
    is_flag=True,
    help="optimize load balancing assignment with LPT algorithm",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help="delete from memory the synapse attributes metadata after specifying connections",
)
@click.option(
    "--profile-memory",
    is_flag=True,
    help="calculate and print heap usage while constructing the network",
)
@click.option(
    "--write-selection",
    is_flag=True,
    help="write out cell and connectivity data for selection",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="print verbose diagnostic messages while constructing the network",
)
@click.option("--debug", is_flag=True, help="enable debug mode")
@click.option(
    "--dry-run",
    is_flag=True,
    help="whether to actually execute simulation after building network",
)
def main(
    arena_id,
    cell_selection_path,
    config_file,
    config_prefix,
    template_paths,
    hoc_lib_path,
    dataset_prefix,
    results_path,
    results_id,
    node_rank_file,
    io_size,
    use_cell_attr_gen,
    cell_attr_gen_cache_size,
    recording_profile,
    output_syn_spike_count,
    use_coreneuron,
    stimulus_id,
    tstop,
    v_init,
    stimulus_onset,
    max_walltime_hours,
    microcircuit_inputs,
    checkpoint_clear_data,
    checkpoint_interval,
    results_write_time,
    spike_input_path,
    spike_input_namespace,
    spike_input_attr,
    dt,
    ldbal,
    lptbal,
    cleanup,
    profile_memory,
    write_selection,
    verbose,
    debug,
    dry_run,
):

    profile_time = False
    config_logging(verbose)

    comm = MPI.COMM_WORLD
    np.seterr(all="raise")
    params = dict(locals())
    params["config"] = params.pop("config_file")
    env = Env(**params)

    if profile_time:
        import cProfile

        cProfile.runctx(
            "init(env)", None, locals(), filename="MiV_profile_init"
        )
        if not dry_run:
            cProfile.runctx(
                "run(env)", None, locals(), filename="MiV_profile_run"
            )
    else:
        network.init(env)
        if not dry_run:
            network.run(env, output_syn_spike_count=output_syn_spike_count)


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                list_find(
                    lambda x: os.path.basename(x) == os.path.basename(__file__),
                    sys.argv,
                )
                + 1
            ) :
        ]
    )
