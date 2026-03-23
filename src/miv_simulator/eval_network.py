#!/usr/bin/env python
"""
Network model evaluation script: applies optimized parameters from a JSON file,
simulates the network, and reports evaluation metrics.
"""

import datetime
import json
import sys

from mpi4py import MPI

from miv_simulator import network
from miv_simulator.utils import (
    config_logging,
    get_module_logger,
    read_from_yaml,
)
from miv_simulator.utils import io as io_utils
from miv_simulator.optimization import (
    network_features,
    optimization_params,
    update_network_params,
)
from miv_simulator.optimize_network import compute_objectives, init_network

logger = get_module_logger(__name__)


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


def eval_network(
    config_path,
    params_path,
    params_label=None,
    t_start=50.0,
    output_path=None,
    verbose=False,
    **network_config,
):
    """
    Applies optimized synaptic parameters from a JSON file to the network model,
    simulates, and evaluates the results.

    :param config_path: path to operational YAML configuration file
    :param params_path: path to JSON file containing optimized parameters
    :param params_label: key within the JSON file (default: first key found)
    :param t_start: start time (ms) for feature extraction (default: 50.0)
    :param output_path: optional path to write JSON evaluation results
    :param verbose: enable verbose logging
    :param network_config: additional keyword arguments forwarded to Env
    """
    config_logging(verbose)

    run_ts = datetime.datetime.today().strftime("%Y%m%d_%H%M")

    # Load operational config
    operational_config = read_from_yaml(config_path)
    network_config.update(operational_config.get("kwargs", {}))

    target_populations = operational_config["target_populations"]
    param_config_name = operational_config["param_config_name"]
    objective_names = operational_config["objective_names"]

    # Set results file id
    network_config.setdefault("results_file_id", f"eval_network_{run_ts}")

    # Initialize the network
    comm = MPI.COMM_WORLD
    env = init_network(comm=comm, subworld_size=None, kwargs=network_config)

    rank = int(env.pc.id())

    if env.cleanup:
        raise RuntimeError(
            "eval_network requires cleanup=False. "
            "With cleanup=True, network.init() deletes biophys_cells after wiring each "
            "gid, so update_network_params() cannot apply the optimized parameters."
        )

    # Load optimized parameters from JSON
    params_dict = None
    if rank == 0:
        if params_path is not None:
            with open(params_path) as f:
                all_params = json.load(f)
            if params_label is None:
                params_label = next(iter(all_params))
            logger.info(f"Loading parameters from label '{params_label}' in {params_path}")

            params_entry = all_params[params_label]
            params_dict = params_entry["parameters"]
            logger.info(f"Loaded {len(params_dict)} optimized parameters")
    env.pc.barrier()
    params_label, params_dict = env.comm.bcast((params_label, params_dict), root=0)

    # Build optimization parameter structure from network config
    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        target_populations,
        param_config_name=param_config_name,
        phenotype_dict=env.phenotype_ids,
    )
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples
    opt_targets = opt_param_config.opt_targets

    # Map parameter names to (param_tuple, value) pairs
    if params_dict is not None:
        param_tuple_values = []
        for param_name, param_tuple in zip(param_names, param_tuples):
            p = param_tuple
            if param_name in params_dict:
                param_value = params_dict[param_name]
            else:
                param_value = params_dict[p.population][p.source][str(p.sec_type)][
                    p.syn_name
                ][p.param_path]
            param_tuple_values.append((param_tuple, param_value))

        # Apply parameters to the network
        if rank == 0:
            logger.info("Applying optimized parameters to network")
        update_network_params(env, param_tuple_values)

    # Run without checkpoint output so spike vectors remain in memory for feature extraction.
    # network.run(output=True) calls spikeout(clear_data=env.checkpoint_clear_data) at each
    # checkpoint segment; with the default checkpoint_clear_data=True this empties env.t_vec
    # and env.id_vec, causing network_features() to return all-zero rates.
    if rank == 0:
        logger.info(f"Running simulation (t_stop={env.tstop} ms)")
    network.run(env, output=False)

    # Extract features from in-memory spike data before any output flushing
    t_stop = env.tstop
    features = network_features(env, t_start, t_stop, target_populations)

    # Write simulation output to disk
    if rank == 0:
        logger.info(f"Writing output to {env.results_file_path}")
        io_utils.mkout(env, env.results_file_path)
    env.pc.barrier()
    io_utils.spikeout(env, env.results_file_path)
    if env.recording_profile is not None:
        io_utils.recsout(env, env.results_file_path)
    if rank == 0:
        io_utils.lfpout(env, env.results_file_path)

    # Compute objectives using same reduction as the optimizer controller
    result = compute_objectives([{0: features}], operational_config, opt_targets)
    objectives_arr, features_arr, constraints_arr = result[0]

    # Log results
    if rank == 0:
        logger.info("=== Evaluation Results ===")
        for name, val in zip(objective_names, objectives_arr.tolist()):
            logger.info(f"  objective  {name}: {val:.6f}")
        for name, val in zip(objective_names, features_arr[0].tolist()):
            logger.info(f"  feature    {name}: {val:.6f}")
        for pop_name, val in zip(target_populations, constraints_arr.tolist()):
            logger.info(f"  constraint {pop_name} positive rate: {val:.6f}")

        if output_path is not None:
            output_data = {
                params_label: {
                    "parameters": params_dict,
                    "objectives": dict(
                        zip(objective_names, [float(v) for v in objectives_arr])
                    ),
                    "features": dict(
                        zip(objective_names, [float(v) for v in features_arr[0]])
                    ),
                    "constraints": {
                        f"{pop} positive rate": float(c)
                        for pop, c in zip(target_populations, constraints_arr)
                    },
                }
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=4)
            logger.info(f"Wrote evaluation results to {output_path}")
    env.pc.barrier()

    network.shutdown(env)
