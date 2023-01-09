import os
import sys
import click
from miv_simulator import utils
from miv_simulator.clamps import network


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--config-file",
    "-c",
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
    "--population",
    "-p",
    required=True,
    type=str,
    default="GC",
    help="target population",
)
@click.option(
    "--gid", "-g", required=True, type=int, default=0, help="target cell gid"
)
@click.option(
    "--arena-id",
    "-a",
    required=False,
    type=str,
    help="arena id for input stimulus",
)
@click.option(
    "--stimulus-id",
    "-s",
    required=False,
    type=str,
    help="input stimulus id",
)
@click.option(
    "--template-paths",
    type=str,
    required=True,
    help="colon-separated list of paths to directories containing hoc cell templates",
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
    "--spike-events-path",
    "-s",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="path to neuroh5 file containing spike times",
)
@click.option(
    "--spike-events-namespace",
    type=str,
    default="Spike Events",
    help="namespace containing spike times",
)
@click.option(
    "--spike-events-t",
    required=False,
    type=str,
    default="t",
    help="name of variable containing spike times",
)
@click.option(
    "--input-features-path",
    required=False,
    type=click.Path(),
    help="path to neuroh5 file containing input selectivity features",
)
@click.option(
    "--input-features-namespaces",
    type=str,
    multiple=True,
    required=False,
    default=["Place Selectivity", "Grid Selectivity"],
    help="namespace containing input selectivity features",
)
@click.option("--use-coreneuron", is_flag=True, help="enable use of CoreNEURON")
@click.option(
    "--plot-cell",
    is_flag=True,
    help="plot the distribution of weight and g_unit synaptic parameters",
)
@click.option(
    "--write-cell",
    is_flag=True,
    help="write out selected cell tree morphology and connections",
)
@click.option(
    "--profile-memory",
    is_flag=True,
    help="calculate and print heap usage after the simulation is complete",
)
@click.option(
    "--recording-profile",
    type=str,
    default="Network clamp default",
    help="recording profile to use",
)
def show(
    config_file,
    config_prefix,
    population,
    gid,
    arena_id,
    stimulus_id,
    template_paths,
    dataset_prefix,
    results_path,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    input_features_path,
    input_features_namespaces,
    use_coreneuron,
    plot_cell,
    write_cell,
    profile_memory,
    recording_profile,
):
    network.show(
        config_file,
        config_prefix,
        population,
        gid,
        arena_id,
        stimulus_id,
        template_paths,
        dataset_prefix,
        results_path,
        spike_events_path,
        spike_events_namespace,
        spike_events_t,
        input_features_path,
        input_features_namespaces,
        use_coreneuron,
        plot_cell,
        write_cell,
        profile_memory,
        recording_profile,
    )


@click.command()
@click.option(
    "--config-file",
    "-c",
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
    "--population",
    "-p",
    required=True,
    type=str,
    default="PYR",
    help="target population",
)
@click.option("--dt", required=False, type=float, help="simulation time step")
@click.option("--gids", "-g", type=int, multiple=True, help="target cell gid")
@click.option(
    "--gid-selection-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="file containing target cell gids",
)
@click.option(
    "--arena-id",
    "-a",
    required=False,
    type=str,
    help="arena id for input stimulus",
)
@click.option(
    "--stimulus-id",
    "-s",
    required=False,
    type=str,
    help="input stimulus id",
)
@click.option(
    "--generate-weights",
    "-w",
    required=False,
    type=str,
    multiple=True,
    help="generate weights for the given presynaptic population",
)
@click.option(
    "--t-max", "-t", type=float, default=150.0, help="simulation end time"
)
@click.option("--t-min", type=float)
@click.option(
    "--template-paths",
    type=str,
    required=True,
    help="colon-separated list of paths to directories containing hoc cell templates",
)
@click.option(
    "--dataset-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory containing required neuroh5 data files",
)
@click.option(
    "--spike-events-path",
    "-s",
    type=click.Path(),
    help="path to neuroh5 file containing spike times",
)
@click.option(
    "--spike-events-namespace",
    type=str,
    default="Spike Events",
    help="namespace containing spike times",
)
@click.option(
    "--spike-events-t",
    required=False,
    type=str,
    default="t",
    help="name of variable containing spike times",
)
@click.option(
    "--coords-path",
    type=click.Path(),
    help="path to neuroh5 file containing cell positions (required for phase-modulated input)",
)
@click.option(
    "--distances-namespace",
    type=str,
    default="Arc Distances",
    help="namespace containing soma distances (required for phase-modulated inputs)",
)
@click.option("--phase-mod", is_flag=True, help="enable phase-modulated inputs")
@click.option(
    "--input-features-path",
    required=False,
    type=click.Path(),
    help="path to neuroh5 file containing input selectivity features",
)
@click.option(
    "--input-features-namespaces",
    type=str,
    multiple=True,
    required=False,
    default=["Place Selectivity", "Grid Selectivity"],
    help="namespace containing input selectivity features",
)
@click.option(
    "--n-trials",
    required=False,
    type=int,
    default=1,
    help="number of trials for input stimulus",
)
@click.option(
    "--params-path",
    required=False,
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="optional path to parameters generated by optimize",
)
@click.option(
    "--params-id",
    required=False,
    multiple=True,
    type=int,
    help="optional ids to parameters contained in parameters path",
)
@click.option(
    "--results-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory where output files will be written",
)
@click.option(
    "--results-file-id",
    type=str,
    required=False,
    default=None,
    help="identifier that is used to name neuroh5 files that contain output spike and intracellular trace data",
)
@click.option(
    "--results-namespace-id",
    type=str,
    required=False,
    default=None,
    help="identifier that is used to name neuroh5 namespaces that contain output spike and intracellular trace data",
)
@click.option("--use-coreneuron", is_flag=True, help="enable use of CoreNEURON")
@click.option(
    "--plot-cell",
    is_flag=True,
    help="plot the distribution of weight and g_unit synaptic parameters",
)
@click.option(
    "--write-cell",
    is_flag=True,
    help="write out selected cell tree morphology and connections",
)
@click.option(
    "--profile-memory",
    is_flag=True,
    help="calculate and print heap usage after the simulation is complete",
)
@click.option(
    "--recording-profile",
    type=str,
    default="Network clamp default",
    help="recording profile to use",
)
@click.option(
    "--input-seed", type=int, help="seed for generation of spike trains"
)
def go(
    config_file,
    config_prefix,
    population,
    dt,
    gids,
    gid_selection_file,
    arena_id,
    stimulus_id,
    generate_weights,
    t_max,
    t_min,
    template_paths,
    dataset_prefix,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    coords_path,
    distances_namespace,
    phase_mod,
    input_features_path,
    input_features_namespaces,
    n_trials,
    params_path,
    params_id,
    results_path,
    results_file_id,
    results_namespace_id,
    use_coreneuron,
    plot_cell,
    write_cell,
    profile_memory,
    recording_profile,
    input_seed,
):
    network.go(
        config_file,
        config_prefix,
        population,
        dt,
        gids,
        gid_selection_file,
        arena_id,
        stimulus_id,
        generate_weights,
        t_max,
        t_min,
        template_paths,
        dataset_prefix,
        spike_events_path,
        spike_events_namespace,
        spike_events_t,
        coords_path,
        distances_namespace,
        phase_mod,
        input_features_path,
        input_features_namespaces,
        n_trials,
        params_path,
        params_id,
        results_path,
        results_file_id,
        results_namespace_id,
        use_coreneuron,
        plot_cell,
        write_cell,
        profile_memory,
        recording_profile,
        input_seed,
    )


@click.command()
@click.option(
    "--config-file",
    "-c",
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
    "--population",
    "-p",
    required=True,
    type=str,
    default="PYR",
    help="target population",
)
@click.option("--dt", type=float, help="simulation time step")
@click.option("--gids", "-g", type=int, multiple=True, help="target cell gid")
@click.option(
    "--gid-selection-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="file containing target cell gids",
)
@click.option("--arena-id", "-a", type=str, required=False, help="arena id")
@click.option(
    "--stimulus-id", "-s", type=str, required=False, help="stimulus id"
)
@click.option(
    "--generate-weights",
    "-w",
    required=False,
    type=str,
    multiple=True,
    help="generate weights for the given presynaptic population",
)
@click.option(
    "--t-max", "-t", type=float, default=150.0, help="simulation end time"
)
@click.option("--t-min", type=float)
@click.option(
    "--nprocs-per-worker",
    type=int,
    default=1,
    help="number of processes per worker",
)
@click.option(
    "--opt-epsilon", type=float, default=1e-2, help="local convergence epsilon"
)
@click.option(
    "--opt-seed",
    type=int,
    help="seed for random sampling of optimization parameters",
)
@click.option(
    "--opt-iter", type=int, default=10, help="number of optimization iterations"
)
@click.option(
    "--template-paths",
    type=str,
    required=True,
    help="colon-separated list of paths to directories containing hoc cell templates",
)
@click.option(
    "--dataset-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory containing required neuroh5 data files",
)
@click.option(
    "--param-config-name",
    type=str,
    help="parameter configuration name to use for optimization (defined in config file)",
)
@click.option(
    "--param-type",
    type=str,
    default="synaptic",
    help="parameter type to use for optimization (synaptic)",
)
@click.option("--recording-profile", type=str, help="recording profile to use")
@click.option(
    "--results-file", required=False, type=str, help="optimization results file"
)
@click.option(
    "--results-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory where output files will be written",
)
@click.option(
    "--spike-events-path",
    type=click.Path(),
    required=False,
    help="path to neuroh5 file containing spike times",
)
@click.option(
    "--spike-events-namespace",
    type=str,
    required=False,
    default="Spike Events",
    help="namespace containing input spike times",
)
@click.option(
    "--spike-events-t",
    required=False,
    type=str,
    default="t",
    help="name of variable containing spike times",
)
@click.option(
    "--coords-path",
    type=click.Path(),
    help="path to neuroh5 file containing cell positions (required for phase-modulated input)",
)
@click.option(
    "--distances-namespace",
    type=str,
    default="Arc Distances",
    help="namespace containing soma distances (required for phase-modulated inputs)",
)
@click.option("--phase-mod", is_flag=True, help="enable phase-modulated inputs")
@click.option(
    "--input-features-path",
    required=False,
    type=click.Path(),
    help="path to neuroh5 file containing input selectivity features",
)
@click.option(
    "--input-features-namespaces",
    type=str,
    multiple=True,
    required=False,
    default=["Place Selectivity", "Grid Selectivity"],
    help="namespace containing input selectivity features",
)
@click.option(
    "--n-trials",
    required=False,
    type=int,
    default=1,
    help="number of trials for input stimulus",
)
@click.option(
    "--trial-regime",
    required=False,
    type=str,
    default="mean",
    help="trial aggregation regime (mean or best)",
)
@click.option(
    "--problem-regime",
    required=False,
    type=str,
    default="every",
    help="problem regime (independently evaluate every problem or mean or max aggregate evaluation)",
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
    "--target-state-variable",
    type=str,
    required=False,
    help="name of state variable used for state optimization",
)
@click.option(
    "--target-state-filter",
    type=str,
    required=False,
    help="optional filter for state values used for state optimization",
)
@click.option("--use-coreneuron", is_flag=True, help="enable use of CoreNEURON")
@click.option(
    "--cooperative-init",
    is_flag=True,
    help="use a single worker to read model data then send to the remaining workers",
)
@click.argument("target")  # help='rate, rate_dist, state'
def optimize(
    config_file,
    config_prefix,
    population,
    dt,
    gids,
    gid_selection_file,
    arena_id,
    stimulus_id,
    generate_weights,
    t_max,
    t_min,
    nprocs_per_worker,
    opt_epsilon,
    opt_seed,
    opt_iter,
    template_paths,
    dataset_prefix,
    param_config_name,
    param_type,
    recording_profile,
    results_file,
    results_path,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    coords_path,
    distances_namespace,
    phase_mod,
    input_features_path,
    input_features_namespaces,
    n_trials,
    trial_regime,
    problem_regime,
    target_features_path,
    target_features_namespace,
    target_state_variable,
    target_state_filter,
    use_coreneuron,
    cooperative_init,
    target,
):
    network.optimize(
        config_file,
        config_prefix,
        population,
        dt,
        gids,
        gid_selection_file,
        arena_id,
        stimulus_id,
        generate_weights,
        t_max,
        t_min,
        nprocs_per_worker,
        opt_epsilon,
        opt_seed,
        opt_iter,
        template_paths,
        dataset_prefix,
        param_config_name,
        param_type,
        recording_profile,
        results_file,
        results_path,
        spike_events_path,
        spike_events_namespace,
        spike_events_t,
        coords_path,
        distances_namespace,
        phase_mod,
        input_features_path,
        input_features_namespaces,
        n_trials,
        trial_regime,
        problem_regime,
        target_features_path,
        target_features_namespace,
        target_state_variable,
        target_state_filter,
        use_coreneuron,
        cooperative_init,
        target,
    )


cli.add_command(show)
cli.add_command(go)
cli.add_command(optimize)


def main_cli():

    cli(sys.argv[1:])
