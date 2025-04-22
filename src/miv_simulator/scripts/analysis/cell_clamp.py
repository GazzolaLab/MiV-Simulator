#!/usr/bin/env python3


import click
from miv_simulator.clamps.cell import cell_clamps


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
@click.option("--erev", type=float, help="synaptic reversal potential")
@click.option(
    "--population",
    "-p",
    required=True,
    type=str,
    default="GC",
    help="target population",
)
@click.option("--presyn-name", type=str, help="presynaptic population")
@click.option("--gid", "-g", required=True, type=int, default=0, help="target cell gid")
@click.option("--load-weights", "-w", is_flag=True)
@click.option(
    "--measurements",
    "-m",
    type=str,
    default="passive,fi,ap,ap_rate",
    help="measurements to perform",
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
    required=False,
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
@click.option("--syn-mech-name", type=str, help="synaptic mechanism name")
@click.option("--syn-weight", type=float, help="synaptic weight")
@click.option("--syn-count", type=int, default=1, help="synaptic count")
@click.option("--swc-type", type=str, help="synaptic swc type")
@click.option("--syn-layer", type=str, help="synaptic layer name")
@click.option(
    "--stim-amp",
    type=float,
    default=0.1,
    help="current stimulus amplitude (nA)",
)
@click.option(
    "--v-init",
    type=float,
    default=-75.0,
    help="initialization membrane potential (mV)",
)
@click.option("--dt", type=float, default=0.025, help="simulation timestep (ms)")
@click.option("--use-cvode", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def main(
    config_file,
    config_prefix,
    erev,
    population,
    presyn_name,
    gid,
    load_weights,
    measurements,
    template_paths,
    dataset_prefix,
    results_path,
    results_file_id,
    results_namespace_id,
    syn_mech_name,
    syn_weight,
    syn_count,
    syn_layer,
    swc_type,
    stim_amp,
    v_init,
    dt,
    use_cvode,
    verbose,
):
    cell_clamps(
        config_file,
        erev,
        population,
        presyn_name,
        gid,
        load_weights,
        measurements,
        template_paths,
        dataset_prefix,
        results_path,
        results_file_id,
        results_namespace_id,
        syn_mech_name,
        syn_weight,
        syn_count,
        syn_layer,
        swc_type,
        stim_amp,
        v_init,
        dt,
        use_cvode,
        verbose,
        config_prefix=config_prefix,
    )
