#!/usr/bin/env python

"""
Routines and objective functions related to optimization of synaptic parameters.
"""

import os, sys, logging
import click
import numpy as np
from mpi4py import MPI
from collections import defaultdict, namedtuple
from enum import Enum, IntEnum, unique
from miv_simulator.env import Env
from miv_simulator import cells, synapses, spikedata, stimulus
from miv_simulator.utils import (
    get_module_logger,
)
from miv_simulator.synapses import SynParam


# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)

OptResult = namedtuple("OptResult", ["parameters", "objectives", "features"])


OptConfig = namedtuple(
    "OptConfig",
    [
        "param_bounds",
        "param_names",
        "param_initial_dict",
        "param_tuples",
        "opt_targets",
    ],
)


@unique
class TrialRegime(IntEnum):
    mean = 0
    best = 1


@unique
class ProblemRegime(IntEnum):
    every = 0
    mean = 1
    max = 2


@unique
class InitMessageTag(IntEnum):
    cell = 0
    input = 1


def parse_optimization_param_dict(
    pop_name,
    param_dict,
    phenotype_id=None,
    keyfun=lambda kv: str(kv[0]),
    param_tuples=[],
    param_initial_dict={},
    param_bounds={},
    param_names=[],
):
    for source, source_dict in sorted(param_dict.items(), key=keyfun):
        for sec_type, sec_type_dict in sorted(source_dict.items(), key=keyfun):
            for syn_name, syn_mech_dict in sorted(
                sec_type_dict.items(), key=keyfun
            ):
                for param_fst, param_rst in sorted(
                    syn_mech_dict.items(), key=keyfun
                ):
                    if isinstance(param_rst, dict):
                        for const_name, const_range in sorted(
                            param_rst.items()
                        ):
                            param_path = (param_fst, const_name)
                            param_tuples.append(
                                SynParam(
                                    pop_name,
                                    source,
                                    sec_type,
                                    syn_name,
                                    param_path,
                                    const_range,
                                    phenotype_id,
                                )
                            )
                            if phenotype_id is not None:
                                param_key = "%s.%d.%s.%s.%s.%s.%s" % (
                                    pop_name,
                                    phenotype_id,
                                    str(source),
                                    sec_type,
                                    syn_name,
                                    param_fst,
                                    const_name,
                                )
                            else:
                                param_key = "%s.%s.%s.%s.%s.%s" % (
                                    pop_name,
                                    str(source),
                                    sec_type,
                                    syn_name,
                                    param_fst,
                                    const_name,
                                )

                            param_initial_value = (
                                const_range[1] - const_range[0]
                            ) / 2.0
                            param_initial_dict[param_key] = param_initial_value
                            param_bounds[param_key] = const_range
                            param_names.append(param_key)
                    else:
                        param_name = param_fst
                        param_range = param_rst
                        param_tuples.append(
                            SynParam(
                                pop_name,
                                source,
                                sec_type,
                                syn_name,
                                param_name,
                                param_range,
                                phenotype_id,
                            )
                        )
                        if phenotype_id is not None:
                            param_key = "%s.%d.%s.%s.%s.%s" % (
                                pop_name,
                                phenotype_id,
                                source,
                                sec_type,
                                syn_name,
                                param_name,
                            )
                        else:
                            param_key = "%s.%s.%s.%s.%s" % (
                                pop_name,
                                source,
                                sec_type,
                                syn_name,
                                param_name,
                            )

                        param_initial_value = (
                            param_range[1] - param_range[0]
                        ) / 2.0
                        param_initial_dict[param_key] = param_initial_value
                        param_bounds[param_key] = param_range
                        param_names.append(param_key)


def parse_optimization_param_entries(
    pop_name,
    param_entries,
    phenotype_dict={},
    keyfun=lambda kv: str(kv[0]),
    param_tuples=[],
    param_initial_dict={},
    param_bounds={},
    param_names=[],
):
    if isinstance(param_entries, dict):
        parse_optimization_param_dict(
            pop_name,
            param_entries,
            keyfun=keyfun,
            param_tuples=param_tuples,
            param_initial_dict=param_initial_dict,
            param_bounds=param_bounds,
            param_names=param_names,
        )
    elif isinstance(param_entries, list):
        for param_entry in param_entries:
            if isinstance(param_entry, dict):
                parse_optimization_param_dict(
                    pop_name,
                    param_entry,
                    keyfun=keyfun,
                    param_tuples=param_tuples,
                    param_initial_dict=param_initial_dict,
                    param_bounds=param_bounds,
                    param_names=param_names,
                )
                continue
            optimization_options = param_entry[0]
            param_dict = param_entry[1]
            # Instantiate parameter entries for each phenotype
            if optimization_options.get("phenotype", False):
                phenotype_ids = phenotype_dict[pop_name]
                if len(phenotype_ids) > 1:
                    for phenotype_id in phenotype_ids:
                        parse_optimization_param_dict(
                            pop_name,
                            param_dict,
                            keyfun=keyfun,
                            phenotype_id=phenotype_id,
                            param_tuples=param_tuples,
                            param_initial_dict=param_initial_dict,
                            param_bounds=param_bounds,
                            param_names=param_names,
                        )
                    else:
                        parse_optimization_param_dict(
                            pop_name,
                            param_dict,
                            keyfun=keyfun,
                            param_tuples=param_tuples,
                            param_initial_dict=param_initial_dict,
                            param_bounds=param_bounds,
                            param_names=param_names,
                        )
    else:
        raise RuntimeError(
            f"Invalid optimization parameter object: {param_entries}"
        )


def optimization_params(
    optimization_config,
    pop_names,
    param_config_name,
    phenotype_dict={},
    param_type="synaptic",
):
    """Constructs a flat list representation of synaptic optimization parameters based on network clamp optimization configuration."""

    param_bounds = {}
    param_names = []
    param_initial_dict = {}
    param_tuples = []
    opt_targets = {}

    for pop_name in pop_names:
        if param_type == "synaptic":
            if pop_name in optimization_config["synaptic"]:
                opt_params = optimization_config["synaptic"][pop_name]
                param_ranges = opt_params["Parameter ranges"][param_config_name]
            else:
                raise RuntimeError(
                    "optimization_params: population %s does not have optimization configuration"
                    % pop_name
                )
            for target_name, target_val in opt_params["Targets"].items():
                opt_targets[f"{pop_name} {target_name}"] = target_val

            parse_optimization_param_entries(
                pop_name,
                param_ranges,
                phenotype_dict=phenotype_dict,
                param_bounds=param_bounds,
                param_names=param_names,
                param_initial_dict=param_initial_dict,
                param_tuples=param_tuples,
            )

        else:
            raise RuntimeError(
                "optimization_params: unknown parameter type %s" % param_type
            )

    return OptConfig(
        param_bounds=param_bounds,
        param_names=param_names,
        param_initial_dict=param_initial_dict,
        param_tuples=param_tuples,
        opt_targets=opt_targets,
    )


def update_network_params(env, param_tuples):
    for population in env.biophys_cells:
        synapse_config = env.celltypes[population].get("synapses", {})
        phenotype_dict = env.phenotype_dict.get(population, None)
        weights_dict = synapse_config.get("weights", {})

        for param_tuple, param_value in param_tuples:
            if param_tuple.population != population:
                continue

            source = param_tuple.source
            sec_type = param_tuple.sec_type
            syn_name = param_tuple.syn_name
            param_path = param_tuple.param_path
            param_phenotype = param_tuple.phenotype

            if isinstance(param_path, list) or isinstance(param_path, tuple):
                p, s = param_path
            else:
                p, s = param_path, None

            sources = None
            if isinstance(source, list) or isinstance(source, tuple):
                sources = source
            else:
                if source is not None:
                    sources = [source]

            if isinstance(sec_type, list) or isinstance(sec_type, tuple):
                sec_types = sec_type
            else:
                sec_types = [sec_type]

            biophys_cell_dict = env.biophys_cells[population]
            for gid in biophys_cell_dict:
                if (phenotype_dict is not None) and (
                    param_phenotype is not None
                ):
                    gid_phenotype = phenotype_dict.get(gid, None)
                    if gid_phenotype is not None:
                        if gid_phenotype != param_phenotype:
                            continue

                biophys_cell = biophys_cell_dict[gid]
                is_reduced = False
                if hasattr(biophys_cell, "is_reduced"):
                    is_reduced = biophys_cell.is_reduced

                for this_sec_type in sec_types:
                    synapses.modify_syn_param(
                        biophys_cell,
                        env,
                        this_sec_type,
                        syn_name,
                        param_name=p,
                        value={s: param_value}
                        if (s is not None)
                        else param_value,
                        filters={"sources": sources}
                        if sources is not None
                        else None,
                        origin=None if is_reduced else "soma",
                        update_targets=True,
                    )


def update_run_params(env, param_tuples):
    for population in env.biophys_cells:
        synapse_config = env.celltypes[population].get("synapses", {})
        weights_dict = synapse_config.get("weights", {})

        for param_tuple, param_value in param_tuples:
            if param_tuple.population != population:
                continue

            source = param_tuple.source
            sec_type = param_tuple.sec_type
            syn_name = param_tuple.syn_name
            param_path = param_tuple.param_path

            if isinstance(param_path, list) or isinstance(param_path, tuple):
                p, s = param_path
            else:
                p, s = param_path, None

            sources = None
            if isinstance(source, list) or isinstance(source, tuple):
                sources = source
            else:
                if source is not None:
                    sources = [source]

            if isinstance(sec_type, list) or isinstance(sec_type, tuple):
                sec_types = sec_type
            else:
                sec_types = [sec_type]

            biophys_cell_dict = env.biophys_cells[population]
            for gid in biophys_cell_dict:
                biophys_cell = biophys_cell_dict[gid]
                is_reduced = False
                if hasattr(biophys_cell, "is_reduced"):
                    is_reduced = biophys_cell.is_reduced

                for this_sec_type in sec_types:
                    synapses.modify_syn_param(
                        biophys_cell,
                        env,
                        this_sec_type,
                        syn_name,
                        param_name=p,
                        value={s: param_value}
                        if (s is not None)
                        else param_value,
                        filters={"sources": sources}
                        if sources is not None
                        else None,
                        origin=None if is_reduced else "soma",
                        update_targets=True,
                    )


def network_features(
    env, target_trj_rate_map_dict, t_start, t_stop, target_populations
):
    features_dict = dict()

    temporal_resolution = float(env.stimulus_config["Temporal Resolution"])
    time_bins = np.arange(t_start, t_stop, temporal_resolution)

    pop_spike_dict = spikedata.get_env_spike_dict(env, include_artificial=False)

    for pop_name in target_populations:
        has_target_trj_rate_map = pop_name in target_trj_rate_map_dict

        n_active = 0
        sum_mean_rate = 0.0
        spike_density_dict = spikedata.spike_density_estimate(
            pop_name, pop_spike_dict[pop_name], time_bins
        )
        for gid, dens_dict in spike_density_dict.items():
            mean_rate = np.mean(dens_dict["rate"])
            sum_mean_rate += mean_rate
            if mean_rate > 0.0:
                n_active += 1

        n_total = len(env.cells[pop_name]) - len(env.artificial_cells[pop_name])

        n_target_rate_map = 0
        sum_snr = None
        if has_target_trj_rate_map:
            pop_target_trj_rate_map_dict = target_trj_rate_map_dict[pop_name]
            n_target_rate_map = len(pop_target_trj_rate_map_dict)
            snrs = []
            for gid in pop_target_trj_rate_map_dict:
                target_trj_rate_map = pop_target_trj_rate_map_dict[gid]
                rate_map_len = len(target_trj_rate_map)
                if gid in spike_density_dict:
                    measured_rate = spike_density_dict[gid]["rate"][
                        :rate_map_len
                    ]
                    ref_signal = target_trj_rate_map - np.mean(
                        target_trj_rate_map
                    )
                    signal = measured_rate - np.mean(measured_rate)
                    noise = signal - ref_signal
                    snr = np.var(signal) / max(np.var(noise), 1e-6)
                else:
                    snr = 0.0
                snrs.append(snr)
            sum_snr = np.sum(snrs)

        pop_features_dict = {}
        pop_features_dict["n_total"] = n_total
        pop_features_dict["n_active"] = n_active
        pop_features_dict["n_target_rate_map"] = n_target_rate_map
        pop_features_dict["sum_mean_rate"] = sum_mean_rate
        pop_features_dict["sum_snr"] = sum_snr

        features_dict[pop_name] = pop_features_dict

    return features_dict


def distgfs_broker_bcast(broker, tag):
    data_dict = None
    if broker.worker_id == 1:
        status = MPI.Status()
        nprocs = broker.nprocs_per_worker
        data_dict = {}
        while len(data_dict) < nprocs:
            if broker.merged_comm.Iprobe(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG
            ):
                data = broker.merged_comm.recv(
                    source=MPI.ANY_SOURCE, tag=tag, status=status
                )
                source = status.Get_source()
                data_dict[source] = data
            else:
                time.sleep(1)
    if broker.worker_id == 1:
        broker.group_comm.bcast(data_dict, root=0)
    else:
        data_dict = broker.group_comm.bcast(None, root=0)
    broker.group_comm.barrier()
    return data_dict


def distgfs_broker_init(broker, *args):
    cell_data_dict = distgfs_broker_bcast(broker, InitMessageTag["cell"])
    if broker.worker_id != 1:
        reqs = []
        for i in cell_data_dict:
            reqs.append(broker.merged_comm.isend(cell_data_dict[i], dest=i))
        MPI.Request.Waitall(reqs)
    cell_data_dict.clear()

    input_data_dict = distgfs_broker_bcast(broker, InitMessageTag["input"])
    if broker.worker_id != 1:
        reqs = []
        for i in input_data_dict:
            reqs.append(broker.merged_comm.isend(input_data_dict[i], dest=i))
        MPI.Request.Waitall(reqs)


def opt_reduce_every(xs):
    result = {}
    for d in xs:
        result.update(d)
    return result


def opt_reduce_every_features(items):
    result = {}
    for xd in items:
        for k in xd:
            yd, fd = xd[k]
            result[k] = (yd, fd)
    return result


def opt_reduce_every_features_constraints(items):
    result = {}
    for xd in items:
        for k in xd:
            yd, fd, cd = xd[k]
            result[k] = (yd, fd, cd)
    return result


def opt_reduce_every_constraints(items):
    result = {}
    features = {}
    for yd, cd in items:
        for k in yd:
            result[k] = (yd[k], cd[k])
    return result


def opt_reduce_mean(xs):
    ks = list(xs[0].keys())
    vs = {k: [] for k in ks}
    for d in xs:
        for k in ks:
            v = d[k]
            if not np.isnan(v):
                vs[k].append(v)
    return {k: np.mean(vs[k]) for k in ks}


def opt_reduce_mean_features(xs, index, feature_dtypes):
    ks = index
    vs = []
    fs = []
    ax = {}
    for x in xs:
        ax.update(x[0])
    for k in index:
        v = ax[k][0]
        f = ax[k][1]
        vs.append(v)
        fs.append(f)
    cval = np.concatenate(fs)
    fval = np.empty((1,), dtype=feature_dtypes)
    for fld in fs[0].dtype.fields:
        fval[fld] = cval[fld].reshape((-1, 1))
    return {0: (np.mean(vs), fval)}


def opt_reduce_max(xs):
    ks = list(xs[0].keys())
    vs = {k: [] for k in ks}
    for d in xs:
        for k in ks:
            v = d[k]
            if not np.isnan(v):
                vs[k].append(v)
    return {k: np.max(vs[k]) for k in ks}


def opt_eval_fun(
    problem_regime, cell_index_set, eval_problem_fun, feature_dtypes=None
):
    problem_regime = ProblemRegime[problem_regime]

    def f(pp, **kwargs):
        if problem_regime == ProblemRegime.every:
            results_dict = eval_problem_fun(pp, **kwargs)
        elif (
            problem_regime == ProblemRegime.mean
            or problem_regime == ProblemRegime.max
        ):
            mpp = {gid: pp for gid in cell_index_set}
            results_dict = eval_problem_fun(mpp, **kwargs)
        else:
            raise RuntimeError(
                "opt_eval_fun: unknown problem regime %s" % str(problem_regime)
            )

        return results_dict

    return f
