#!/usr/bin/env python
"""
Network model optimization script for optimization with dmosopt
"""

import os
import sys
import datetime
import copy

os.environ["DISTWQ_CONTROLLER_RANK"] = "-1"

from functools import partial
import click
import numpy as np
from neuron import h
from miv_simulator import network
from miv_simulator.env import Env
from miv_simulator.mechanisms import compile_and_load
from miv_simulator.utils import (
    read_from_yaml,
    write_to_yaml,
    get_module_logger,
)
from miv_simulator.synapses import (
    syn_param_from_dict,
)
from miv_simulator.optimization import (
    optimization_params,
    update_network_params,
    network_features,
)
from miv_simulator.network_objectives import (
    load_network_opt_config,
    merge_raw_features,
)

from dmosopt import dmosopt
from dmosopt.MOASMO import get_best
from mpi4py import MPI

logger = get_module_logger(__name__)


def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


def dmosopt_get_best(file_path, opt_id):
    (
        _,
        max_epoch,
        old_evals,
        param_names,
        is_int,
        lo_bounds,
        hi_bounds,
        objective_names,
        feature_names,
        constraint_names,
        problem_parameters,
        problem_ids,
    ) = dmosopt.init_from_h5(file_path, None, opt_id, None)

    problem_id = 0
    old_eval_epochs = [e.epoch for e in old_evals[problem_id]]
    old_eval_xs = [e.parameters for e in old_evals[problem_id]]
    old_eval_ys = [e.objectives for e in old_evals[problem_id]]
    x = np.vstack(old_eval_xs)
    y = np.vstack(old_eval_ys)
    old_eval_fs = None
    f = None
    if feature_names is not None:
        old_eval_fs = [e.features for e in old_evals[problem_id]]
        f = np.concatenate(old_eval_fs, axis=None)

    old_eval_cs = None
    c = None
    if constraint_names is not None:
        old_eval_cs = [e.constraints for e in old_evals[problem_id]]
        c = np.vstack(old_eval_cs)

    x = np.vstack(old_eval_xs)
    y = np.vstack(old_eval_ys)

    if len(old_eval_epochs) > 0 and old_eval_epochs[0] is not None:
        epochs = np.concatenate(old_eval_epochs, axis=None)

    best_x, best_y, best_f, best_c, best_epoch, _ = get_best(
        x,
        y,
        f,
        c,
        len(param_names),
        len(objective_names),
        epochs=epochs,
        feasible=True,
    )
    best_x_items = tuple((param_names[i], best_x[:, i]) for i in range(best_x.shape[1]))
    best_y_items = tuple(
        (objective_names[i], best_y[:, i]) for i in range(best_y.shape[1])
    )
    return (best_x_items, best_y_items, best_f, best_c)


def init_controller(subworld_size, use_coreneuron):
    h.nrnmpi_init()
    h("objref pc, cvode")
    h.cvode = h.CVode()
    h.pc = h.ParallelContext()
    h.pc.subworlds(subworld_size)
    if use_coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True
        coreneuron.verbose = 0
        h.cvode.cache_efficient(1)
        h.finitialize(-65)
        h.pc.set_maxstep(10)
        h.pc.psolve(0.05)


def optimize_network(
    config_path,
    optimize_file_dir,
    optimize_file_name,
    nprocs_per_worker,
    n_epochs,
    n_initial,
    initial_maxiter,
    initial_method,
    optimizer_method,
    surrogate_method,
    surrogate_method_kwargs,
    population_size,
    num_generations,
    resample_fraction,
    mutation_rate,
    collective_mode,
    spawn_startup_wait,
    spawn_workers,
    get_best,
    verbose,
    surrogate_custom_training=None,
    surrogate_custom_training_kwargs=None,
):
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

    run_ts = datetime.datetime.today().strftime("%Y%m%d_%H%M")

    if optimize_file_name is None:
        optimize_file_name = f"dmosopt.optimize_network_{run_ts}.h5"
    operational_config = read_from_yaml(config_path)
    operational_config["run_ts"] = run_ts
    operational_config["nprocs_per_worker"] = nprocs_per_worker

    network_config.update(operational_config.get("kwargs", {}))
    env = Env(**network_config)

    param_config_name = operational_config["param_config_name"]

    # Load optimization config from operational_config (derives target_populations
    # from Objectives and Constraints in the YAML)
    opt_config = load_network_opt_config(operational_config)
    opt_config.validate_picklable()

    target_populations = opt_config.target_populations()
    # Store for downstream consumers (e.g. init_network_objfun)
    operational_config["target_populations"] = target_populations

    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        target_populations,
        param_config_name=param_config_name,
        phenotype_dict=env.phenotype_ids,
    )

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples
    hyperprm_space = {
        param_pattern: [param_tuple.param_range[0], param_tuple.param_range[1]]
        for param_pattern, param_tuple in zip(param_names, param_tuples)
    }

    init_objfun = "init_network_objfun"
    init_params = {
        "operational_config": operational_config,
        "opt_targets": opt_targets,
        "param_tuples": [param_tuple._asdict() for param_tuple in param_tuples],
        "param_names": param_names,
    }
    init_params.update(network_config.items())

    nworkers = env.comm.size - 1
    if resample_fraction is None:
        resample_fraction = float(nworkers) / float(population_size)
    if resample_fraction > 1.0:
        resample_fraction = 1.0
    if resample_fraction < 0.1:
        resample_fraction = 0.1

    feature_dtypes = opt_config.feature_dtypes(target_populations)
    constraint_names = opt_config.constraint_names()
    objective_names_opt = opt_config.objective_names()

    surrogate_method_kwargs = copy.copy(surrogate_method_kwargs)
    if "batch_size" not in surrogate_method_kwargs:
        surrogate_method_kwargs["batch_size"] = 400

    surrogate_config = operational_config.get("Surrogate", {}) or {}
    if surrogate_custom_training is None:
        surrogate_custom_training = surrogate_config.get("custom_training")
    custom_kwargs = dict(surrogate_config.get("kwargs", {}) or {})
    custom_kwargs.update(surrogate_custom_training_kwargs or {})
    weights = custom_kwargs.get("feasibility_constraint_weights")
    if isinstance(weights, dict):
        resolved = {}
        for entry, weight in weights.items():
            if isinstance(entry, str):
                if entry not in constraint_names:
                    raise RuntimeError(
                        f"feasibility_constraint_weights names constraint {entry!r}, "
                        f"which is not among the constraints {constraint_names}"
                    )
                resolved[constraint_names.index(entry)] = float(weight)
            else:
                resolved[int(entry)] = float(weight)
        custom_kwargs["feasibility_constraint_weights"] = resolved

    dmosopt_params = {
        "opt_id": "miv_simulator.optimize_network",
        "obj_fun_init_name": f"miv_simulator.optimize_network.{init_objfun}",
        "obj_fun_init_args": init_params,
        "controller_init_fun_name": "miv_simulator.optimize_network.init_controller",
        "controller_init_fun_args": {
            "subworld_size": nprocs_per_worker,
            "use_coreneuron": network_config.get("use_coreneuron", False),
        },
        "reduce_fun_name": "miv_simulator.optimize_network.compute_objectives",
        "reduce_fun_args": (operational_config, opt_targets, opt_config),
        "problem_parameters": {},
        "space": hyperprm_space,
        "objective_names": objective_names_opt,
        "feature_dtypes": feature_dtypes,
        "constraint_names": constraint_names,
        "n_initial": n_initial,
        "initial_maxiter": initial_maxiter,
        "initial_method": initial_method,
        "optimizer_name": optimizer_method,
        "surrogate_method_name": None
        if surrogate_custom_training
        else surrogate_method,
        "surrogate_method_kwargs": {}
        if surrogate_custom_training
        else surrogate_method_kwargs,
        "surrogate_custom_training": surrogate_custom_training,
        "surrogate_custom_training_kwargs": custom_kwargs
        if surrogate_custom_training
        else None,
        "n_epochs": n_epochs,
        "population_size": population_size,
        "num_generations": num_generations,
        "resample_fraction": resample_fraction,
        "mutation_rate": mutation_rate,
        "file_path": os.path.join(optimize_file_dir, optimize_file_name),
        "termination_conditions": True,
        "save_surrogate_eval": True,
        "sensitivity_method": "dgsm",
        "save": True,
        "save_eval": 5,
    }

    if get_best:
        best = dmosopt_get_best(dmosopt_params["file_path"], dmosopt_params["opt_id"])
    else:
        best = dmosopt.run(
            dmosopt_params,
            spawn_workers=spawn_workers,
            spawn_startup_wait=spawn_startup_wait,
            nprocs_per_worker=nprocs_per_worker,
            collective_mode=collective_mode,
            verbose=True,
            worker_debug=True,
        )

    if best is not None:
        if optimize_file_dir is not None:
            results_file_id = f"optimize_network_{run_ts}"
            yaml_file_path = os.path.join(
                optimize_file_dir, f"optimize_network.{results_file_id}.yaml"
            )
            prms = best[0]
            prms_dict = dict(prms)
            n_res = prms[0][1].shape[0]
            results_config_dict = {}
            for i in range(n_res):
                result_param_list = []
                for param_pattern, param_tuple in zip(param_names, param_tuples):
                    result_param_list.append(
                        (
                            param_tuple.population,
                            param_tuple.source,
                            param_tuple.sec_type,
                            param_tuple.syn_name,
                            param_tuple.param_path,
                            param_tuple.phenotype,
                            float(prms_dict[param_pattern][i]),
                        )
                    )
                results_config_dict[i] = result_param_list
            write_to_yaml(yaml_file_path, results_config_dict)


def init_network_objfun(
    operational_config, opt_targets, param_names, param_tuples, worker, **kwargs
):
    param_tuples = [syn_param_from_dict(param_tuple) for param_tuple in param_tuples]

    target_populations = operational_config["target_populations"]
    kwargs["results_file_id"] = (
        f"optimize_network_{worker.worker_id}_{operational_config['run_ts']}"
    )
    nprocs_per_worker = operational_config["nprocs_per_worker"]
    env = init_network(
        comm=worker.merged_comm, subworld_size=nprocs_per_worker, kwargs=kwargs
    )
    if kwargs.get("use_coreneuron", False):
        h.cvode.cache_efficient(1)
        h.pc.set_maxstep(10)
        h.pc.psolve(0.05)

    t_start = float(operational_config.get("t_start", 50.0))
    t_stop = env.tstop

    def from_param_dict(params_dict):
        result = []
        for param_name, param_tuple in zip(param_names, param_tuples):
            p = param_tuple
            if param_name in params_dict:
                param_value = params_dict[param_name]
            else:
                param_value = params_dict[p.population][p.source][str(p.sec_type)][
                    p.syn_name
                ][p.param_path]
            result.append((param_tuple, param_value))
        return result

    return partial(
        network_objfun,
        env,
        operational_config,
        opt_targets,
        from_param_dict,
        t_start,
        t_stop,
        target_populations,
    )


def init_network(comm, subworld_size, kwargs):
    np.seterr(all="raise")
    env = Env(comm=comm, **kwargs)
    compile_and_load(directory=env.mechanisms_path, comm=env.comm)
    network.init(env, subworld_size=subworld_size)
    return env


def network_objfun(
    env,
    operational_config,
    opt_targets,
    from_param_dict,
    t_start,
    t_stop,
    target_populations,
    x,
):
    param_tuple_values = from_param_dict(x)
    update_network_params(env, param_tuple_values)

    env.tstop = t_stop
    network.run(env, output=False)

    return network_features(env, t_start, t_stop, target_populations)


def _merge_pop_features(pop_features_dicts):
    """Aggregate per-worker per-population raw spike dicts into one.

    Sums n_total and n_active; merges spike_density_dict and spike_times by
    gid; uses first worker's time_bins.
    """
    if not pop_features_dicts:
        return {}
    result = {
        "n_total": 0,
        "n_active": 0,
        "time_bins": None,
        "spike_density_dict": {},
    }
    result.update(merge_raw_features(pop_features_dicts))
    return result


def compute_objectives(local_features, operational_config, opt_targets, opt_config):
    all_features_dict = {}

    target_populations = opt_config.target_populations()

    for pop_name in target_populations:
        pop_features_dicts = [
            features_dict[0][pop_name] for features_dict in local_features
        ]
        merged = _merge_pop_features(pop_features_dicts)
        for feature in opt_config.features:
            if feature.populations is not None and pop_name not in feature.populations:
                continue
            for fname, fval in feature.compute(pop_name, merged).items():
                all_features_dict[f"{pop_name}.{fname}"] = fval

    objectives = [-obj.compute(all_features_dict) for obj in opt_config.objectives]
    feature_dtypes = opt_config.feature_dtypes(target_populations)
    features = tuple(all_features_dict.get(fname, 0.0) for fname, _ in feature_dtypes)
    constraints = [c.compute(all_features_dict) for c in opt_config.constraints]

    result = (
        np.asarray(objectives, dtype=np.float32),
        np.array([features], dtype=np.dtype(feature_dtypes))
        if feature_dtypes
        else np.array([]),
        np.asarray(constraints, dtype=np.float32),
    )
    return {0: result}
