__doc__ = """
Routines for Network Clamp simulation.
"""
import gc
import os
import pprint
import sys
import time
from collections import defaultdict

import click
import h5py
import numpy as np
from miv_simulator import io_utils, spikedata, stimulus, synapses
from miv_simulator.cell_clamp import init_biophys_cell
from miv_simulator.cells import (
    h,
    is_cell_registered,
    load_biophys_cell_dicts,
    make_input_cell,
    record_cell,
    register_cell,
    report_topology,
)
from miv_simulator.env import Env
from miv_simulator.neuron_utils import configure_hoc_env, h
from miv_simulator.stimulus import oscillation_phase_mod_config, rate_maps_from_features
from miv_simulator.utils import (
    Context,
    config_logging,
    generate_results_file_id,
    get_low_pass_filtered_trace,
    get_module_logger,
    get_trial_time_indices,
    is_interactive,
    list_find,
    list_index,
    read_from_yaml,
    write_to_yaml,
)
from mpi4py import MPI
from neuroh5.io import (
    bcast_cell_attributes,
    read_cell_attribute_info,
    scatter_read_cell_attribute_selection,
)

# This logger will inherit its settings from the root logger, created in miv_simulator.env
logger = get_module_logger(__name__)

context = Context()
env = None


def set_union(s, t, datatype):
    return s.union(t)


mpi_op_set_union = MPI.Op.Create(set_union, commute=True)


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


def generate_weights(env, weight_source_rules, this_syn_attrs):
    """
    Generates synaptic weights according to the rules specified in the
    Weight Generator section of network clamp configuration.
    """
    weights_dict = {}

    if len(weight_source_rules) > 0:

        for presyn_id, weight_rule in weight_source_rules.items():
            source_syn_dict = defaultdict(list)

            for syn_id, syn in this_syn_attrs.items():
                this_presyn_id = syn.source.population
                this_presyn_gid = syn.source.gid
                if this_presyn_id == presyn_id:
                    source_syn_dict[this_presyn_gid].append(syn_id)

            if weight_rule["class"] == "Sparse":
                weights_name = weight_rule["name"]
                rule_params = weight_rule["params"]
                fraction = rule_params["fraction"]
                seed_offset = int(
                    env.model_config["Random Seeds"]["Sparse Weights"]
                )
                seed = int(seed_offset + 1)
                weights_dict[presyn_id] = synapses.generate_sparse_weights(
                    weights_name, fraction, seed, source_syn_dict
                )
            elif weight_rule["class"] == "Log-Normal":
                weights_name = weight_rule["name"]
                rule_params = weight_rule["params"]
                mu = rule_params["mu"]
                sigma = rule_params["sigma"]
                clip = None
                if "clip" in rule_params:
                    clip = rule_params["clip"]
                seed_offset = int(
                    env.model_config["Random Seeds"]["GC Log-Normal Weights 1"]
                )
                seed = int(seed_offset + 1)
                weights_dict[presyn_id] = synapses.generate_log_normal_weights(
                    weights_name, mu, sigma, seed, source_syn_dict, clip=clip
                )
            elif weight_rule["class"] == "Normal":
                weights_name = weight_rule["name"]
                rule_params = weight_rule["params"]
                mu = rule_params["mu"]
                sigma = rule_params["sigma"]
                seed_offset = int(
                    env.model_config["Random Seeds"]["GC Normal Weights"]
                )
                seed = int(seed_offset + 1)
                weights_dict[presyn_id] = synapses.generate_normal_weights(
                    weights_name, mu, sigma, seed, source_syn_dict
                )
            else:
                raise RuntimeError(
                    "network_clamp.generate_weights: unknown weight generator rule class "
                    f'{weight_rule["class"]}'
                )

    return weights_dict


def init_inputs_from_spikes(
    env,
    presyn_sources,
    time_range,
    spike_events_path,
    spike_events_namespace,
    arena_id,
    trajectory_id,
    spike_train_attr_name="t",
    n_trials=1,
):
    """Initializes presynaptic spike sources from a file with spikes times."""
    populations = sorted(presyn_sources.keys())

    equilibration_duration = float(
        env.stimulus_config.get("Equilibration Duration", 0.0)
    )
    if time_range is not None:
        spkdata_time_range = (
            time_range[0] - equilibration_duration,
            time_range[1],
        )

    if arena_id and trajectory_id:
        this_spike_events_namespace = (
            f"{spike_events_namespace} {arena_id} {trajectory_id}"
        )
    elif arena_id:
        this_spike_events_namespace = f"{spike_events_namespace} {arena_id}"
    else:
        this_spike_events_namespace = spike_events_namespace

    ## Load spike times of presynaptic cells
    spkdata = spikedata.read_spike_events(
        spike_events_path,
        populations,
        this_spike_events_namespace,
        spike_train_attr_name=spike_train_attr_name,
        time_range=spkdata_time_range,
        n_trials=n_trials,
        merge_trials=True,
        comm=env.comm,
        io_size=env.io_size,
    )

    spkindlst = spkdata["spkindlst"]
    spktlst = spkdata["spktlst"]
    spkpoplst = spkdata["spkpoplst"]
    pop_active_cells = spkdata["pop_active_cells"]

    ## Organize spike times by index of presynaptic population and gid
    input_source_dict = {}
    for population in populations:

        pop_index = int(env.Populations[population])
        spk_pop_index = list_index(population, spkpoplst)
        if spk_pop_index is None:
            logger.warning(
                f"No spikes found for population {population} in file {spike_events_path}"
            )
            continue
        spk_inds = spkindlst[spk_pop_index]
        spk_ts = spktlst[spk_pop_index]

        spikes_attr_dict = {}
        gid_range = presyn_sources[population]
        logger.info(
            f"Initializing {len(gid_range)} spike sources for population {population}."
        )

        for gid in gid_range:
            this_spk_inds = np.argwhere(spk_inds == gid)
            if len(this_spk_inds) > 0:
                ts = spk_ts[this_spk_inds] + equilibration_duration
                spikes_attr_dict[gid] = {spike_train_attr_name: ts}

        input_source_dict[pop_index] = {"spiketrains": spikes_attr_dict}

    return input_source_dict


def init_inputs_from_features(
    env,
    presyn_sources,
    time_range,
    input_features_path,
    input_features_namespaces,
    arena_id,
    trajectory_id,
    phase_mod=False,
    soma_positions_dict=None,
    spike_train_attr_name="t",
    n_trials=1,
    seed=None,
):
    """Initializes presynaptic spike sources from a file with input selectivity features represented as firing rates."""

    logger.info(f"init_inputs_from_features: seed = {seed}")
    populations = sorted(presyn_sources.keys())

    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    equilibration_duration = float(
        env.stimulus_config["Equilibration Duration"]
    )
    spatial_resolution = float(env.stimulus_config["Spatial Resolution"])
    temporal_resolution = float(env.stimulus_config["Temporal Resolution"])

    this_input_features_namespaces = [
        f"{input_features_namespace} {arena_id}"
        for input_features_namespace in input_features_namespaces
    ]

    input_features_attr_names = [
        "Selectivity Type",
        "Num Fields",
        "Field Width",
        "Peak Rate",
        "Module ID",
        "Grid Spacing",
        "Grid Orientation",
        "Field Width Concentration Factor",
        "X Offset",
        "Y Offset",
    ]

    selectivity_type_names = {i: n for n, i in env.selectivity_types.items()}

    arena = env.stimulus_config["Arena"][arena_id]
    arena_x, arena_y = stimulus.get_2D_arena_spatial_mesh(
        arena=arena, spatial_resolution=spatial_resolution
    )

    trajectory = arena.trajectories[trajectory_id]
    t, x, y, d = stimulus.generate_linear_trajectory(
        trajectory,
        temporal_resolution=temporal_resolution,
        equilibration_duration=equilibration_duration,
    )
    if time_range is not None:
        t_range_inds = np.where(
            (t <= time_range[1]) & (t >= time_range[0] - equilibration_duration)
        )[0]
        t = t[t_range_inds]
        x = x[t_range_inds]
        y = y[t_range_inds]
        d = d[t_range_inds]
    trajectory = t, x, y, d

    equilibrate = stimulus.get_equilibration(env)

    input_source_dict = {}
    for population in populations:
        pop_start = int(env.celltypes[population]["start"])
        num_cells = int(env.celltypes[population]["num"])

        selection = list(presyn_sources[population])
        logger.info(
            f"generating spike trains in time range {time_range} "
            f"for {len(selection)} inputs from presynaptic population {population}..."
        )

        phase_mod_config_dict = None
        if phase_mod:
            phase_mod_config_dict = oscillation_phase_mod_config(
                env, population, soma_positions_dict[population]
            )

        pop_index = int(env.Populations[population])
        spikes_attr_dict = {}
        for input_features_namespace in this_input_features_namespaces:
            logger.info(
                f"reading input features namespace {input_features_namespace}..."
            )
            input_features_iter = scatter_read_cell_attribute_selection(
                input_features_path,
                population,
                selection=selection,
                namespace=input_features_namespace,
                mask=set(input_features_attr_names),
                comm=env.comm,
            )
            for gid, selectivity_attr_dict in input_features_iter:

                phase_mod_config = None
                if phase_mod_config_dict is not None:
                    phase_mod_config = phase_mod_config_dict[gid]

                spikes_attr_dict[gid] = stimulus.generate_input_spike_trains(
                    env,
                    population,
                    selectivity_type_names,
                    trajectory,
                    gid,
                    selectivity_attr_dict,
                    equilibrate=equilibrate,
                    phase_mod_config=phase_mod_config,
                    spike_train_attr_name=spike_train_attr_name,
                    n_trials=n_trials,
                    return_selectivity_features=False,
                    merge_trials=True,
                    time_range=time_range,
                    comm=env.comm,
                    seed=seed,
                )
                spikes_attr_dict[gid][
                    spike_train_attr_name
                ] += equilibration_duration

        input_source_dict[pop_index] = {"spiketrains": spikes_attr_dict}

    return input_source_dict


def init(
    env,
    pop_name,
    cell_index_set,
    arena_id=None,
    trajectory_id=None,
    n_trials=1,
    spike_events_path=None,
    spike_events_namespace="Spike Events",
    spike_train_attr_name="t",
    input_features_path=None,
    input_features_namespaces=None,
    coords_path=None,
    distances_namespace="Arc Distances",
    phase_mod=False,
    generate_weights_pops=set(),
    t_min=None,
    t_max=None,
    write_cell=False,
    plot_cell=False,
    input_seed=None,
    cooperative_init=False,
    worker=None,
):
    """
    Instantiates a cell and all its synapses and connections and loads
    or generates spike times for all synaptic connections.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid_set: cell gids
    :param spike_events_path:

    """

    if phase_mod and coords_path is None:
        raise RuntimeError(
            "network_clamp.init: when phase_mod is True, coords_path must be provided"
        )

    if env.cell_selection is None:
        env.cell_selection = {}
    selection = env.cell_selection.get(pop_name, [])
    env.cell_selection[pop_name] = list(cell_index_set) + [selection]

    ## If specified, presynaptic spikes that only fall within this time range
    ## will be loaded or generated
    if t_max is None:
        t_range = None
    else:
        if t_min is None:
            t_range = [0.0, t_max]
        else:
            t_range = [t_min, t_max]

    ## Attribute namespace that contains recorded spike events
    namespace_id = spike_events_namespace

    my_cell_index_list = []
    for i, gid in enumerate(cell_index_set):
        if i % env.comm.size == env.comm.rank:
            my_cell_index_list.append(gid)
    my_cell_index_set = set(my_cell_index_list)

    data_dict = None
    cell_dict = None

    if (worker is not None) and cooperative_init:
        if worker.worker_id == 1:
            cell_dict = load_biophys_cell_dicts(
                env, pop_name, my_cell_index_set
            )
            req = worker.merged_comm.isend(
                cell_dict, tag=InitMessageTag["cell"].value, dest=0
            )
            req.wait()
        else:
            cell_dict = worker.merged_comm.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG
            )
    else:
        cell_dict = load_biophys_cell_dicts(env, pop_name, my_cell_index_set)

    ## Load cell gid and its synaptic attributes and connection data
    for gid in my_cell_index_set:
        cell = init_biophys_cell(
            env, pop_name, gid, cell_dict=cell_dict[gid], write_cell=write_cell
        )
        del cell_dict[gid]

    pop_index_dict = {ind: name for name, ind in env.Populations.items()}

    ## Determine presynaptic populations that connect to this cell type
    presyn_names = sorted(env.projection_dict[pop_name])
    all_populations = [pop_name] + presyn_names

    weight_source_dict = {}
    for presyn_name in presyn_names:
        presyn_index = int(env.Populations[presyn_name])

        if presyn_name in generate_weights_pops:
            if presyn_name in env.netclamp_config.weight_generators[pop_name]:
                weight_rule = env.netclamp_config.weight_generators[pop_name][
                    presyn_name
                ]
            else:
                raise RuntimeError(
                    f"network_clamp.init: no weights generator rule specified for population {presyn_name}"
                )
        else:
            weight_rule = None

        if weight_rule is not None:
            weight_source_dict[presyn_index] = weight_rule

    min_delay = float("inf")
    syn_attrs = env.synapse_attributes
    presyn_sources = {presyn_name: set() for presyn_name in presyn_names}

    for gid in my_cell_index_set:
        this_syn_attrs = syn_attrs[gid]
        for syn_id, syn in this_syn_attrs.items():
            presyn_id = syn.source.population
            if presyn_id is None:
                raise RuntimeError(
                    f"gid {gid} synapse {syn_id} presyn_id is None: syn = {syn}"
                )
            presyn_name = pop_index_dict[presyn_id]
            presyn_gid = syn.source.gid
            presyn_sources[presyn_name].add(presyn_gid)

    for presyn_name in presyn_names:

        presyn_gid_set = env.comm.reduce(
            presyn_sources[presyn_name], root=0, op=mpi_op_set_union
        )
        env.comm.barrier()
        if env.comm.rank == 0:
            presyn_gid_rank_dict = {
                rank: set() for rank in range(env.comm.size)
            }
            for i, gid in enumerate(presyn_gid_set):
                rank = i % env.comm.size
                presyn_gid_rank_dict[rank].add(gid)
            presyn_sources[presyn_name] = env.comm.scatter(
                [
                    presyn_gid_rank_dict[rank]
                    for rank in sorted(presyn_gid_rank_dict)
                ],
                root=0,
            )
        else:
            presyn_sources[presyn_name] = env.comm.scatter(None, root=0)
        env.comm.barrier()

    soma_positions_dict = None
    if coords_path is not None:
        soma_positions_dict = {}
        for population in all_populations:
            reference_u_arc_distance_bounds = None
            if env.comm.rank == 0:
                with h5py.File(coords_path, "r") as coords_f:
                    reference_u_arc_distance_bounds = (
                        coords_f["Populations"][population][
                            distances_namespace
                        ].attrs["Reference U Min"],
                        coords_f["Populations"][population][
                            distances_namespace
                        ].attrs["Reference U Max"],
                    )
            env.comm.barrier()
            reference_u_arc_distance_bounds = env.comm.bcast(
                reference_u_arc_distance_bounds, root=0
            )
            distances = bcast_cell_attributes(
                coords_path, population, namespace=distances_namespace, root=0
            )
            abs_positions = {
                k: v["U Distance"][0] - reference_u_arc_distance_bounds[0]
                for (k, v) in distances
            }
            soma_positions_dict[population] = abs_positions
            del distances

    input_source_dict = None
    if (worker is not None) and cooperative_init:
        if worker.worker_id == 1:
            if spike_events_path is not None:
                input_source_dict = init_inputs_from_spikes(
                    env,
                    presyn_sources,
                    t_range,
                    spike_events_path,
                    spike_events_namespace,
                    arena_id,
                    trajectory_id,
                    spike_train_attr_name,
                    n_trials,
                )
            elif input_features_path is not None:
                input_source_dict = init_inputs_from_features(
                    env,
                    presyn_sources,
                    t_range,
                    input_features_path,
                    input_features_namespaces,
                    arena_id=arena_id,
                    trajectory_id=trajectory_id,
                    spike_train_attr_name=spike_train_attr_name,
                    n_trials=n_trials,
                    seed=input_seed,
                    phase_mod=phase_mod,
                    soma_positions_dict=soma_positions_dict,
                )
            else:
                raise RuntimeError(
                    "network_clamp.init: neither input spikes nor input features are provided"
                )
            req = worker.merged_comm.isend(
                input_source_dict, tag=InitMessageTag["input"].value, dest=0
            )
            req.wait()
        else:
            input_source_dict = worker.merged_comm.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG
            )
    else:
        if spike_events_path is not None:
            input_source_dict = init_inputs_from_spikes(
                env,
                presyn_sources,
                t_range,
                spike_events_path,
                spike_events_namespace,
                arena_id,
                trajectory_id,
                spike_train_attr_name,
                n_trials,
            )
        elif input_features_path is not None:
            input_source_dict = init_inputs_from_features(
                env,
                presyn_sources,
                t_range,
                input_features_path,
                input_features_namespaces,
                arena_id=arena_id,
                trajectory_id=trajectory_id,
                spike_train_attr_name=spike_train_attr_name,
                n_trials=n_trials,
                seed=input_seed,
                phase_mod=phase_mod,
                soma_positions_dict=soma_positions_dict,
            )

        else:
            raise RuntimeError(
                "network_clamp.init: neither input spikes nor input features are provided"
            )

    if t_range is not None:
        env.tstart = t_range[0]
        env.tstop = t_range[1]

    env.comm.barrier()

    for presyn_name in presyn_names:
        presyn_gids = presyn_sources[presyn_name]
        presyn_id = int(env.Populations[presyn_name])
        for presyn_gid in presyn_gids:
            ## Load presynaptic spike times into the VecStim for stimulus gid;
            ## if spike_generator_dict contains an entry for the respective presynaptic population,
            ## then use the given generator to generate spikes.
            if not (
                (presyn_gid in env.gidset)
                or (is_cell_registered(env, presyn_gid))
            ):
                cell = make_input_cell(
                    env,
                    presyn_gid,
                    presyn_id,
                    input_source_dict,
                    spike_train_attr_name=spike_train_attr_name,
                )
                register_cell(env, presyn_name, presyn_gid, cell)

    for gid in my_cell_index_set:
        synapses.config_biophys_cell_syns(
            env, gid, pop_name, insert=True, insert_netcons=True, verbose=True
        )
        record_cell(env, pop_name, gid)
    gc.collect()

    if plot_cell:
        from miv_simulator.plot import plot_synaptic_attribute_distribution

        syn_attrs = env.synapse_attributes
        syn_name = "AMPA"
        syn_mech_name = syn_attrs.syn_mech_names[syn_name]
        for gid in my_cell_index_set:
            biophys_cell = env.biophys_cells[pop_name][gid]
            for param_name in ["weight", "g_unit"]:
                param_label = f"{syn_name}; {syn_mech_name}; {param_name};"
                plot_synaptic_attribute_distribution(
                    biophys_cell,
                    env,
                    syn_name,
                    param_name,
                    filters=None,
                    from_mech_attrs=True,
                    from_target_attrs=True,
                    param_label=param_label,
                    export=f"syn_params_{gid}.h5",
                    description="network_clamp",
                    show=False,
                    svg_title=f"Synaptic parameters for gid {gid}",
                    output_dir=env.results_path,
                )

    if env.verbose:
        for gid in my_cell_index_set:
            if is_cell_registered(env, gid):
                cell = env.pc.gid2cell(gid)
                for sec in list(
                    cell.hoc_cell.all if hasattr(cell, "hoc_cell") else cell.all
                ):
                    h.psection(sec=sec)
            break

    mindelay = env.pc.set_maxstep(10)

    if is_interactive:
        context.update(locals())

    env.comm.barrier()
    return my_cell_index_set


def run(env, cvode=False, pc_runworker=False):
    """
    Runs network clamp simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env: instance of env.Env
    :param cvode: whether to use adaptive integration
    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    rec_dt = None
    if env.recording_profile is not None:
        rec_dt = env.recording_profile.get("dt", None)
    if env.recs_count == 0:
        ## placeholder compartment to allow recording of time below
        h("""create soma""")
    if rec_dt is None:
        env.t_rec.record(h._ref_t)
    else:
        env.t_rec.record(h._ref_t, rec_dt)
    env.t_vec.resize(0)
    env.id_vec.resize(0)

    st_comptime = env.pc.step_time()

    h.cvode_active(1 if cvode else 0)

    h.t = float(env.tstart)
    h.dt = env.dt
    tstop = float(env.tstop)
    if "Equilibration Duration" in env.stimulus_config:
        tstop += float(env.stimulus_config["Equilibration Duration"])
    h.tstop = float(env.n_trials) * tstop

    h.finitialize(env.v_init)

    if rank == 0:
        logger.info(
            f"*** Running simulation with dt = {h.dt:.04f} and tstop = {h.tstop:.02f}"
        )

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if rank == 0:
        logger.info("*** Simulation completed")
    env.pc.barrier()

    comptime = env.pc.step_time() - st_comptime
    avgcomp = env.pc.allreduce(comptime, 1) / nhosts
    maxcomp = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info(f"Host {rank} ran simulation in {comptime:.02f} seconds")

    if pc_runworker:
        env.pc.runworker()
    env.pc.done()

    return spikedata.get_env_spike_dict(env, include_artificial=None)


def update_params(env, pop_param_dict):

    for population, param_tuple_dict in pop_param_dict.items():

        synapse_config = env.celltypes[population]["synapses"]
        weights_dict = synapse_config.get("weights", {})
        biophys_cell_dict = env.biophys_cells[population]
        for gid, param_tuples in param_tuple_dict.items():

            if gid not in biophys_cell_dict:
                continue
            biophys_cell = biophys_cell_dict[gid]
            is_reduced = False
            if hasattr(biophys_cell, "is_reduced"):
                is_reduced = biophys_cell.is_reduced

            for (param_tuple, param_value) in param_tuples:

                assert population == param_tuple.population

                source = param_tuple.source
                sec_type = param_tuple.sec_type
                syn_name = param_tuple.syn_name
                param_path = param_tuple.param_path

                if isinstance(param_path, list) or isinstance(
                    param_path, tuple
                ):
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


def run_with(env, param_dict, cvode=False, pc_runworker=False):
    """
    Runs network clamp simulation with the specified parameters for the given gid(s).
    Assumes that procedure `init` has been called with
    the network configuration provided by the `env` argument.

    :param env: instance of env.Env
    :param param_dict: dictionary { gid: params }
    :param cvode: whether to use adaptive integration
    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    stash_id_dict = defaultdict(lambda: dict())
    syn_attrs = env.synapse_attributes
    for pop_name in param_dict:
        for gid in param_dict[pop_name]:
            stash_id = syn_attrs.stash_mech_attrs(pop_name, gid)
            stash_id_dict[pop_name][gid] = stash_id
    update_params(env, param_dict)

    rec_dt = None
    if env.recording_profile is not None:
        rec_dt = env.recording_profile.get("dt", None)

    if env.recs_count == 0:
        ## placeholder compartment to allow recording of time below
        h("""create soma""")

    if rec_dt is None:
        env.t_rec.record(h._ref_t)
    else:
        env.t_rec.record(h._ref_t, rec_dt)

    env.t_vec.resize(0)
    env.id_vec.resize(0)

    st_comptime = env.pc.step_time()

    h.cvode_active(1 if cvode else 0)

    h.t = float(env.tstart)
    h.dt = env.dt
    tstop = float(env.tstop)
    if "Equilibration Duration" in env.stimulus_config:
        tstop += float(env.stimulus_config["Equilibration Duration"])
    h.tstop = float(env.n_trials) * tstop

    h.finitialize(env.v_init)
    h.finitialize(env.v_init)

    if rank == 0:
        logger.info(
            f"*** Running simulation with dt = {h.dt:.04f} and tstop = {h.tstop:.02f}"
        )
        logger.info(f"*** Parameters: {pprint.pformat(param_dict)}")

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if rank == 0:
        logger.info("*** Simulation completed")
    env.pc.barrier()

    comptime = env.pc.step_time() - st_comptime
    avgcomp = env.pc.allreduce(comptime, 1) / nhosts
    maxcomp = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info(f"Host {rank} ran simulation in {comptime:.02f} seconds")

    if pc_runworker:
        env.pc.runworker()
    env.pc.done()

    for pop_name in param_dict:
        for gid in param_dict[pop_name]:
            stash_id = stash_id_dict[pop_name][gid]
            syn_attrs.restore_mech_attrs(pop_name, gid, stash_id)
            synapses.config_biophys_cell_syns(env, gid, pop_name, insert=False)

    return spikedata.get_env_spike_dict(env, include_artificial=None)


def init_state_objfun(
    config_file,
    population,
    cell_index_set,
    arena_id,
    trajectory_id,
    generate_weights,
    t_max,
    t_min,
    opt_iter,
    template_paths,
    dataset_prefix,
    results_path,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    input_features_path,
    input_features_namespaces,
    coords_path,
    distances_namespace,
    phase_mod,
    n_trials,
    trial_regime,
    problem_regime,
    param_type,
    param_config_name,
    recording_profile,
    state_variable,
    state_filter,
    target_value,
    use_coreneuron,
    cooperative_init,
    dt,
    worker,
    **kwargs,
):

    params = dict(locals())
    params["config"] = params.pop("config_file")
    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env, bcast_template=True)

    my_cell_index_set = init(
        env,
        population,
        cell_index_set,
        arena_id,
        trajectory_id,
        n_trials,
        spike_events_path,
        spike_events_namespace=spike_events_namespace,
        spike_train_attr_name=spike_events_t,
        coords_path=coords_path,
        distances_namespace=distances_namespace,
        phase_mod=phase_mod,
        input_features_path=input_features_path,
        input_features_namespaces=input_features_namespaces,
        generate_weights_pops=set(generate_weights),
        t_min=t_min,
        t_max=t_max,
        cooperative_init=cooperative_init,
        worker=worker,
    )

    time_step = env.stimulus_config["Temporal Resolution"]
    equilibration_duration = float(
        env.stimulus_config["Equilibration Duration"]
    )

    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        [population],
        param_config_name,
        param_type,
    )

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples

    recording_profile = {
        "label": f"network_clamp.state.{state_variable}",
        "dt": None if use_coreneuron else 0.1,
        "section quantity": {state_variable: {"swc types": ["soma"]}},
    }
    env.recording_profile = recording_profile
    state_recs_dict = {}
    for gid in my_cell_index_set:
        state_recs_dict[gid] = record_cell(
            env, population, gid, recording_profile=recording_profile
        )

    def from_param_dict(params_dict):
        result = []
        for param_pattern, param_tuple in zip(param_names, param_tuples):
            result.append((param_tuple, params_dict[param_pattern]))
        return result

    def gid_state_values(spkdict, t_offset, n_trials, t_rec, state_recs_dict):
        t_vec = np.asarray(t_rec.to_python(), dtype=np.float32)
        t_trial_inds = get_trial_time_indices(t_vec, n_trials, t_offset)
        results_dict = {}
        filter_fun = None
        if state_filter == "lowpass":
            filter_fun = lambda x, t: get_low_pass_filtered_trace(x, t)
        for gid in state_recs_dict:
            state_values = []
            state_recs = state_recs_dict[gid]
            for rec in state_recs:
                vec = np.asarray(rec["vec"].to_python(), dtype=np.float32)
                if filter_fun is None:
                    data = np.asarray(
                        [np.mean(vec[t_inds]) for t_inds in t_trial_inds]
                    )
                else:
                    data = np.asarray(
                        [
                            np.mean(filter_fun(vec[t_inds], t_vec[t_inds]))
                            for t_inds in t_trial_inds
                        ]
                    )
                state_values.append(np.mean(data))
            results_dict[gid] = state_values
        return results_dict

    def eval_problem(cell_param_dict, **kwargs):
        state_values_dict = gid_state_values(
            run_with(
                env,
                {
                    population: {
                        gid: from_param_dict(cell_param_dict[gid])
                        for gid in my_cell_index_set
                    }
                },
            ),
            equilibration_duration,
            n_trials,
            env.t_rec,
            state_recs_dict,
        )
        if trial_regime == "mean":
            return {
                gid: -abs(np.mean(state_values_dict[gid]) - target_value)
                for gid in my_cell_index_set
            }
        elif trial_regime == "best":
            return {
                gid: -(
                    np.min(
                        np.abs(
                            np.asarray(state_values_dict[gid]) - target_value
                        )
                    )
                )
                for gid in my_cell_index_set
            }
        else:
            raise RuntimeError(
                f"state_objfun: unknown trial regime {trial_regime}"
            )

    return opt_eval_fun(problem_regime, my_cell_index_set, eval_problem)


def init_rate_objfun(
    config_file,
    population,
    cell_index_set,
    arena_id,
    trajectory_id,
    n_trials,
    trial_regime,
    problem_regime,
    generate_weights,
    t_max,
    t_min,
    opt_iter,
    template_paths,
    dataset_prefix,
    results_path,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    coords_path,
    distances_namespace,
    phase_mod,
    input_features_path,
    input_features_namespaces,
    param_type,
    param_config_name,
    recording_profile,
    target_rate,
    use_coreneuron,
    cooperative_init,
    dt,
    worker,
    **kwargs,
):

    params = dict(locals())
    params["config"] = params.pop("config_file")
    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env, bcast_template=True)

    my_cell_index_set = init(
        env,
        population,
        cell_index_set,
        arena_id,
        trajectory_id,
        n_trials,
        spike_events_path=spike_events_path,
        spike_events_namespace=spike_events_namespace,
        spike_train_attr_name=spike_events_t,
        coords_path=coords_path,
        distances_namespace=distances_namespace,
        phase_mod=phase_mod,
        input_features_path=input_features_path,
        input_features_namespaces=input_features_namespaces,
        generate_weights_pops=set(generate_weights),
        t_min=t_min,
        t_max=t_max,
        cooperative_init=cooperative_init,
        worker=worker,
    )

    time_range = (t_min if t_min is not None else 0.0, t_max)
    tsecs = (time_range[1] - time_range[0]) / 1e3

    time_step = env.stimulus_config["Temporal Resolution"]
    equilibration_duration = float(
        env.stimulus_config.get("Equilibration Duration", 0.0)
    )

    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        [population],
        param_config_name,
        param_type,
    )

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples

    recording_profile = {
        "label": f"network_clamp.rate.v",
        "dt": None if use_coreneuron else 0.1,
        "section quantity": {"v": {"swc types": ["soma"]}},
    }
    env.recording_profile = recording_profile
    state_recs_dict = {}
    for gid in my_cell_index_set:
        state_recs_dict[gid] = record_cell(
            env, population, gid, recording_profile=recording_profile
        )

    target_v_threshold = opt_targets[f"{population} state"]["v"].get(
        "threshold", None
    )
    target_v_margin = opt_targets[f"{population} state"]["v"].get(
        "margin", -1.0
    )
    if target_v_threshold is None:
        raise RuntimeError(
            f"network_clamp: network clamp optimization configuration for population {population} "
            f"must have state variable v threshold setting in section Targets"
        )

    def from_param_dict(params_dict):
        result = []
        for param_pattern, param_tuple in zip(param_names, param_tuples):
            result.append((param_tuple, params_dict[param_pattern]))
        return result

    def gid_firing_rate(spkdict, cell_index_set):
        rates_dict = defaultdict(list)
        mean_rates_dict = {}
        for i in range(n_trials):
            spkdict1 = {}
            for gid in cell_index_set:
                if gid in spkdict[population]:
                    spk_ts = spkdict[population][gid][i]
                    spkdict1[gid] = spk_ts
                else:
                    spkdict1[gid] = np.asarray([], dtype=np.float32)

            for gid in cell_index_set:
                this_rate = len(spkdict1[gid]) / tsecs
                logger.info(
                    f"firing rate objective: spike times of gid {gid}: {pprint.pformat(spkdict1[gid])}"
                )
                logger.info(
                    f"firing rate objective: rate of gid {gid} is {this_rate:.02f}"
                )
                rates_dict[gid].append(this_rate)

        return rates_dict

    def gid_mean_v(t_offset, v_threshold, n_trials, t_rec, state_recs_dict):
        t_vec = np.asarray(t_rec.to_python(), dtype=np.float32)
        t_trial_inds = get_trial_time_indices(t_vec, n_trials, t_offset)
        results_dict = {}
        for gid in state_recs_dict:
            state_values = []
            state_recs = state_recs_dict[gid]
            for rec in state_recs:
                vec = np.asarray(rec["vec"].to_python(), dtype=np.float32)
                data = np.asarray(
                    [
                        np.mean(np.clip(vec[t_inds], None, v_threshold))
                        for t_inds in t_trial_inds
                    ]
                )
                state_values.append(data)
            results_dict[gid] = state_values
        return results_dict

    def mean_rate_diff(gid, rates, target_rate):
        rates_array = np.asarray(rates)
        nz_idxs = np.argwhere(
            np.logical_not(np.isclose(rates_array, 0.0, rtol=1e-4, atol=1e-4))
        )
        mean_rate = 0.0
        if len(nz_idxs) > 0:
            mean_rate = np.mean(rates_array[nz_idxs])
        return abs(mean_rate - target_rate)

    def best_rate_diff(gid, rates, target_rate):
        rates_array = np.asarray(rates)
        nz_idxs = np.argwhere(
            np.logical_not(np.isclose(rates_array, 0.0, rtol=1e-4, atol=1e-4))
        )
        max_rate = 0.0
        if len(nz_idxs) > 0:
            max_rate = np.max(rates_array[nz_idxs])
        return abs(max_rate - target_rate)

    def eval_problem(cell_param_dict, **kwargs):
        spkdict = run_with(
            env,
            {
                population: {
                    gid: from_param_dict(cell_param_dict[gid])
                    for gid in my_cell_index_set
                }
            },
        )
        firing_rates_dict = gid_firing_rate(spkdict, my_cell_index_set)
        mean_v_dict = gid_mean_v(
            equilibration_duration,
            target_v_threshold,
            n_trials,
            env.t_rec,
            state_recs_dict,
        )
        if trial_regime == "mean":
            objectives_dict = {
                gid: -mean_rate_diff(gid, firing_rates_dict[gid], target_rate)
                for gid in my_cell_index_set
            }
        elif trial_regime == "best":
            objectives_dict = {
                gid: -best_rate_diff(gid, firing_rates_dict[gid], target_rate)
                for gid in my_cell_index_set
            }
        else:
            raise RuntimeError(
                f"rate_objfun: unknown trial regime {trial_regime}"
            )

        N_objectives = 1
        opt_rate_feature_dtypes = [
            ("mean_rate", (np.float32, (1,))),
            ("trial_objs", (np.float32, (N_objectives, n_trials))),
            ("mean_v", (np.float32, (n_trials,))),
        ]

        features_dict = {}
        constraints_dict = {}

        for gid in my_cell_index_set:
            feature_array = np.empty(
                shape=(1,), dtype=np.dtype(opt_rate_feature_dtypes)
            )
            rates_array = np.asarray(firing_rates_dict[gid])
            nz_idxs = np.argwhere(
                np.logical_not(
                    np.isclose(rates_array, 0.0, rtol=1e-4, atol=1e-4)
                )
            )
            feature_array["mean_rate"] = 0.0
            if len(nz_idxs) > 0:
                feature_array["mean_rate"] = np.mean(rates_array[nz_idxs])
            for i in range(N_objectives):
                feature_array["trial_objs"][i, :] = rates_array
            feature_array["mean_v"] = mean_v_dict[gid]
            features_dict[gid] = feature_array

        for gid in my_cell_index_set:
            constraints_dict[gid] = np.asarray([1], dtype=np.int8)
            if (
                np.mean(features_dict[gid]["mean_v"])
                >= target_v_threshold - target_v_margin
            ):
                objectives_dict[gid] -= 1e6
                constraints_dict[gid][0] = -1

        return objectives_dict, features_dict, constraints_dict

    return opt_eval_fun(problem_regime, my_cell_index_set, eval_problem)


def init_rate_dist_objfun(
    config_file,
    population,
    cell_index_set,
    arena_id,
    trajectory_id,
    n_trials,
    trial_regime,
    problem_regime,
    generate_weights,
    t_max,
    t_min,
    opt_iter,
    template_paths,
    dataset_prefix,
    results_path,
    spike_events_path,
    spike_events_namespace,
    spike_events_t,
    coords_path,
    distances_namespace,
    phase_mod,
    input_features_path,
    input_features_namespaces,
    param_type,
    param_config_name,
    recording_profile,
    target_features_path,
    target_features_namespace,
    target_features_arena,
    target_features_trajectory,
    use_coreneuron,
    cooperative_init,
    dt,
    worker,
    **kwargs,
):

    params = dict(locals())
    params["config"] = params.pop("config_file")
    env = Env(**params)
    env.results_file_path = None
    configure_hoc_env(env, bcast_template=True)

    my_cell_index_set = init(
        env,
        population,
        cell_index_set,
        arena_id,
        trajectory_id,
        n_trials,
        spike_events_path,
        spike_events_namespace=spike_events_namespace,
        spike_train_attr_name=spike_events_t,
        coords_path=coords_path,
        distances_namespace=distances_namespace,
        phase_mod=phase_mod,
        input_features_path=input_features_path,
        input_features_namespaces=input_features_namespaces,
        generate_weights_pops=set(generate_weights),
        t_min=t_min,
        t_max=t_max,
        cooperative_init=cooperative_init,
        worker=worker,
    )

    time_step = env.stimulus_config["Temporal Resolution"]

    target_rate_vector_dict = rate_maps_from_features(
        env,
        population,
        cell_index_set=my_cell_index_set,
        input_features_path=target_features_path,
        input_features_namespace=target_features_namespace,
        time_range=None,
        arena_id=arena_id,
    )
    for gid, target_rate_vector in target_rate_vector_dict.items():
        target_rate_vector[
            np.isclose(target_rate_vector, 0.0, atol=1e-3, rtol=1e-3)
        ] = 0.0

    trj_x, trj_y, trj_d, trj_t = stimulus.read_trajectory(
        input_features_path
        if input_features_path is not None
        else spike_events_path,
        target_features_arena,
        target_features_trajectory,
    )
    time_range = (0.0, min(np.max(trj_t), t_max))
    time_bins = np.arange(time_range[0], time_range[1] + time_step, time_step)

    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        [pop_name],
        param_config_name,
        param_type,
    )

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples

    def from_param_dict(params_dict):
        result = []
        for param_pattern, param_tuple in zip(param_names, param_tuples):
            result.append((param_tuple, params_dict[param_pattern]))
        return result

    def gid_firing_rate_vectors(spkdict, cell_index_set):
        rates_dict = defaultdict(list)
        for i in range(n_trials):
            spkdict1 = {}
            for gid in cell_index_set:
                if gid in spkdict[population]:
                    spkdict1[gid] = spkdict[population][gid][i]
                else:
                    spkdict1[gid] = np.asarray([], dtype=np.float32)
            spike_density_dict = spikedata.spike_density_estimate(
                population, spkdict1, time_bins
            )
            for gid in cell_index_set:
                rate_vector = spike_density_dict[gid]["rate"]
                idxs = np.where(
                    np.isclose(rate_vector, 0.0, atol=1e-3, rtol=1e-3)
                )[0]
                rate_vector[idxs] = 0.0
                rates_dict[gid].append(rate_vector)
            for gid in spkdict[population]:
                logger.info(
                    f"firing rate objective: trial {i} firing rate of gid {gid}: {spike_density_dict[gid]}"
                )
                logger.info(
                    f"firing rate objective: trial {i} firing rate min/max of gid {gid}: "
                    f"{np.min(rates_dict[gid]):.02f} / {np.max(rates_dict[gid]):.02f} Hz"
                )

        return rates_dict

    def mean_trial_rate_mse(gid, rate_vectors, target_rate_vector):
        mean_rate_vector = np.mean(np.row_stack(rate_vectors), axis=0)
        logger.info(
            f"firing rate objective: mean firing rate min/max of gid {gid}: "
            f"{np.min(mean_rate_vector):.02f} / {np.max(mean_rate_vector):.02f} Hz"
        )

        return np.square(
            np.subtract(mean_rate_vectore, target_rate_vector)
        ).mean()

    def best_trial_rate_mse(gid, rate_vectors, target_rate_vector):
        mses = []
        for rate_vector in rate_vectors:
            mse = np.square(np.subtract(rate_vector, target_rate_vector)).mean()
            mses.append(mse)

        min_mse_index = np.argmin(mses)
        min_mse = mses[max_mse_index]

        logger.info(
            f"firing rate objective: max firing rate min/max of gid {gid}: "
            f"{np.min(rate_vector[min_mse_index]):.02f} / {np.max(rate_vectors[min_mse_index]):.02f} Hz"
        )

        return min_mse

    def eval_problem(cell_param_dict, **kwargs):
        firing_rate_vectors_dict = gid_firing_rate_vectors(
            run_with(
                env,
                {
                    population: {
                        gid: from_param_dict(cell_param_dict[gid])
                        for gid in my_cell_index_set
                    }
                },
            ),
            my_cell_index_set,
        )
        if trial_regime == "mean":
            return {
                gid: -mean_trial_rate_mse(
                    gid,
                    firing_rate_vectors_dict[gid],
                    target_rate_vector_dict[gid],
                )
                for gid in my_cell_index_set
            }
        elif trial_regime == "best":
            return {
                gid: -best_trial_rate_mse(
                    gid,
                    firing_rate_vectors_dict[gid],
                    target_rate_vector_dict[gid],
                )
                for gid in my_cell_index_set
            }
        else:
            raise RuntimeError(
                f"firing_rate_dist: unknown trial regime {trial_regime}"
            )

    return opt_eval_fun(problem_regime, my_cell_index_set, eval_problem)


def optimize_run(
    env,
    pop_name,
    param_config_name,
    init_objfun,
    problem_regime,
    nprocs_per_worker=1,
    opt_iter=10,
    solver_epsilon=1e-2,
    opt_seed=None,
    param_type="synaptic",
    init_params={},
    feature_dtypes=None,
    constraint_names=None,
    results_file=None,
    cooperative_init=False,
    verbose=False,
):
    import distgfs

    opt_param_config = optimization_params(
        env.netclamp_config.optimize_parameters,
        [pop_name],
        param_config_name,
        param_type,
    )

    opt_targets = opt_param_config.opt_targets
    param_names = opt_param_config.param_names
    param_tuples = opt_param_config.param_tuples

    hyperprm_space = {
        param_pattern: [param_tuple.param_range[0], param_tuple.param_range[1]]
        for param_pattern, param_tuple in zip(param_names, param_tuples)
    }

    problem_metadata = np.array(
        [
            tuple(
                opt_targets[k]
                for k in sorted(opt_targets)
                if isinstance(opt_targets[k], float)
            )
        ],
        dtype=[
            (f"Target {k}", np.float32, (1,))
            for k in sorted(opt_targets)
            if isinstance(opt_targets[k], float)
        ],
    )

    if results_file is None:
        if env.results_path is not None:
            file_path = f"{env.results_path}/distgfs.network_clamp.{env.results_file_id}.h5"
        else:
            file_path = f"distgfs.network_clamp.{env.results_file_id}.h5"
    else:
        file_path = f"{env.results_path}/{results_file}"
    problem_ids = None
    reduce_fun_name = None
    if ProblemRegime[problem_regime] == ProblemRegime.every:
        if feature_dtypes is None:
            reduce_fun_name = "opt_reduce_every"
        elif (feature_dtypes is not None) and (constraint_names is None):
            reduce_fun_name = "opt_reduce_every_features"
        elif (feature_dtypes is not None) and (constraint_names is not None):
            reduce_fun_name = "opt_reduce_every_features_constraints"
        elif (feature_dtypes is None) and (constraint_names is not None):
            reduce_fun_name = "opt_reduce_every_constraints"
        problem_ids = init_params.get("cell_index_set", None)
    elif ProblemRegime[problem_regime] == ProblemRegime.mean:
        reduce_fun_name = "opt_reduce_mean"
        feature_dtypes = None
    elif ProblemRegime[problem_regime] == ProblemRegime.max:
        reduce_fun_name = "opt_reduce_max"
        feature_dtypes = None
    else:
        raise RuntimeError(
            f"optimize_run: unknown problem regime {problem_regime}"
        )

    distgfs_params = {
        "opt_id": "network_clamp.optimize",
        "problem_ids": problem_ids,
        "obj_fun_init_name": init_objfun,
        "obj_fun_init_module": "miv_simulator.network_clamp",
        "obj_fun_init_args": init_params,
        "reduce_fun_name": reduce_fun_name,
        "reduce_fun_module": "miv_simulator.optimization",
        "problem_parameters": {},
        "space": hyperprm_space,
        "feature_dtypes": feature_dtypes,
        "constraint_names": constraint_names,
        "file_path": file_path,
        "save": True,
        "n_iter": opt_iter,
        "seed": opt_seed,
        "solver_epsilon": solver_epsilon,
        "metadata": problem_metadata,
    }

    if cooperative_init:
        distgfs_params["broker_fun_name"] = "distgfs_broker_init"
        distgfs_params["broker_module_name"] = "miv_simulator.optimization"

    opt_results = distgfs.run(
        distgfs_params,
        verbose=verbose,
        collective_mode="sendrecv",
        spawn_workers=True,
        nprocs_per_worker=nprocs_per_worker,
    )
    if opt_results is not None:
        if ProblemRegime[problem_regime] == ProblemRegime.every:
            gid_results_config_dict = {}
            for gid, opt_result in opt_results.items():
                params_dict = dict(opt_result[0])
                result_value = opt_result[1]
                results_config_tuples = []
                for param_pattern, param_tuple in zip(
                    param_names, param_tuples
                ):
                    results_config_tuples.append(
                        (
                            param_tuple.population,
                            param_tuple.source,
                            param_tuple.sec_type,
                            param_tuple.syn_name,
                            param_tuple.param_path,
                            params_dict[param_pattern],
                        )
                    )
                gid_results_config_dict[int(gid)] = results_config_tuples

            logger.info(
                "Optimized parameters and objective function: "
                f"{pprint.pformat(gid_results_config_dict)} @"
                f"{result_value}"
            )
            return {pop_name: gid_results_config_dict}
        else:
            params_dict = dict(opt_results[0])
            result_value = opt_results[1]
            results_config_tuples = []
            for param_pattern, param_tuple in zip(param_names, param_tuples):
                results_config_tuples.append(
                    (
                        param_tuple.population,
                        param_tuple.source,
                        param_tuple.sec_type,
                        param_tuple.syn_name,
                        param_tuple.param_path,
                        params_dict[param_pattern],
                    )
                )
            logger.info(
                "Optimized parameters and objective function: "
                f"{pprint.pformat(results_config_tuples)} @"
                f"{result_value}"
            )
            return {pop_name: results_config_tuples}
    else:
        return None


def dist_ctrl(
    controller, init_params, cell_index_set, param_path, pop_param_tuple_dicts
):
    """Controller for distributed network clamp runs."""
    task_ids = []
    results_file_id = init_params.get("results_file_id", None)
    if len(param_path) > 0:
        for this_param_path, pop_param_tuple_dict in zip(
            param_path, pop_param_tuple_dicts
        ):
            params_basename = os.path.splitext(
                os.path.basename(this_param_path)
            )[0]
            this_results_file_id = f"{results_file_id}_{params_basename}"
            task_id = controller.submit_call(
                "dist_run",
                module_name="miv_simulator.network_clamp",
                args=(
                    init_params,
                    cell_index_set,
                    this_results_file_id,
                    pop_param_tuple_dict,
                ),
            )
            task_ids.append(task_id)
    else:
        task_id = controller.submit_call(
            "dist_run",
            module_name="miv_simulator.network_clamp",
            args=(init_params, cell_index_set, None, None),
        )
        task_ids.append(task_id)

    for task_id in task_ids:
        task_id, res = controller.get_next_result()

    controller.info()


def dist_run(
    init_params, cell_index_set, results_file_id=None, pop_param_tuple_dict=None
):
    """Initialize workers for distributed network clamp runs."""

    if results_file_id is None:
        results_file_id = init_params.get("results_file_id", None)
    if results_file_id is None:
        population = init_params["population"]
        results_file_id = generate_results_file_id(
            population, seed=init_params.get("opt_seed", None)
        )
        init_params["results_file_id"] = results_file_id

    global env
    if env is None:
        init_params["config"] = init_params.pop("config_file", None)
        env = Env(**init_params)
        configure_hoc_env(env, bcast_template=True)
    env.clear()

    env.results_file_id = results_file_id
    env.results_file_path = (
        f"{env.results_path}/{env.modelName}_results_{env.results_file_id}.h5"
    )

    population = init_params["population"]
    arena_id = init_params["arena_id"]
    trajectory_id = init_params["trajectory_id"]
    spike_events_path = init_params["spike_events_path"]
    spike_events_namespace = init_params["spike_events_namespace"]
    spike_events_t = init_params["spike_events_t"]
    coords_path = init_params["coords_path"]
    distances_namespace = init_params["distances_namespace"]
    phase_mod = init_params["phase_mod"]
    input_features_path = init_params["input_features_path"]
    input_features_namespaces = init_params["input_features_namespaces"]
    generate_weights = init_params.get("generate_weights", [])
    t_min = init_params["t_min"]
    t_max = init_params["t_max"]
    n_trials = init_params["n_trials"]
    input_seed = init_params.get("input_seed", None)

    init(
        env,
        population,
        cell_index_set,
        arena_id,
        trajectory_id,
        n_trials,
        spike_events_path,
        spike_events_namespace=spike_events_namespace,
        coords_path=coords_path,
        distances_namespace=distances_namespace,
        phase_mod=phase_mod,
        spike_train_attr_name=spike_events_t,
        input_features_path=input_features_path,
        input_features_namespaces=input_features_namespaces,
        generate_weights_pops=set(generate_weights),
        input_seed=input_seed,
        t_min=t_min,
        t_max=t_max,
    )

    if pop_param_tuple_dict is not None:
        run_with(env, pop_param_tuple_dict)
        write_output(env)
        write_params(env, pop_param_tuple_dict)
    else:
        run(env)
        write_output(env)

    return None


def write_output(env):
    rank = env.comm.rank
    if rank == 0:
        io_utils.mkout(env, env.results_file_path)
    env.comm.barrier()

    if rank == 0:
        logger.info("*** Writing spike data")
    io_utils.spikeout(env, env.results_file_path)
    if rank == 0:
        logger.info("*** Writing intracellular data")
    io_utils.recsout(
        env,
        env.results_file_path,
        write_cell_location_data=True,
        write_trial_data=True,
    )
    if rank == 0:
        logger.info("*** Writing synapse spike counts")
    for pop_name in sorted(env.biophys_cells.keys()):
        presyn_names = sorted(env.projection_dict[pop_name])
        synapses.write_syn_spike_count(
            env,
            pop_name,
            env.results_file_path,
            filters={"sources": presyn_names},
            write_kwds={"io_size": env.io_size},
        )


def write_params(env, pop_params_dict):
    rank = env.comm.rank
    if rank == 0:
        logger.info("*** Writing synapse parameters")
        output_pop_params_dict = {}
        for this_pop_name, this_pop_param_dict in pop_params_dict.items():
            this_pop_output_params_dict = {}
            for this_gid, this_gid_param_list in this_pop_param_dict.items():
                this_gid_param_dicts = []
                for this_gid_param in this_gid_param_list:
                    syn_param, param_val = this_gid_param
                    this_gid_param_dicts.append(
                        {
                            "population": syn_param.population,
                            "source": syn_param.source,
                            "sec_type": syn_param.sec_type,
                            "syn_name": syn_param.syn_name,
                            "param_path": syn_param.param_path,
                            "param_val": float(param_val),
                        }
                    )
                this_pop_output_params_dict[this_gid] = this_gid_param_dicts
            output_pop_params_dict[this_pop_name] = this_pop_output_params_dict
        io_utils.write_params(env.results_file_path, output_pop_params_dict)
    env.comm.barrier()


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
    "--trajectory-id",
    "-t",
    required=False,
    type=str,
    help="trajectory id for input stimulus",
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
    population,
    gid,
    arena_id,
    trajectory_id,
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
    """
    Show configuration for the specified cell.
    """

    np.seterr(all="raise")

    verbose = True
    init_params = dict(locals())
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        comm0 = comm.Split(2 if rank == 0 else 1, 0)
        init_params["config"] = init_params.pop("config_file", None)
        env = Env(**init_params, comm=comm0)
        configure_hoc_env(env)

        init(
            env,
            population,
            {gid},
            arena_id,
            trajectory_id,
            spike_events_path=spike_events_path,
            spike_events_namespace=spike_events_namespace,
            spike_train_attr_name=spike_events_t,
            input_features_path=input_features_path,
            input_features_namespaces=input_features_namespaces,
            plot_cell=plot_cell,
            write_cell=write_cell,
        )

        cell = env.biophys_cells[population][gid]
        logger.info(pprint.pformat(report_topology(cell, env)))

        if env.profile_memory:
            profile_memory(logger)

    comm.barrier()


@click.command()
@click.option(
    "--config-file",
    "-c",
    required=True,
    type=str,
    help="model configuration file name",
)
@click.option(
    "--population",
    "-p",
    required=True,
    type=str,
    default="GC",
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
    "--trajectory-id",
    "-t",
    required=False,
    type=str,
    help="trajectory id for input stimulus",
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
    population,
    dt,
    gids,
    gid_selection_file,
    arena_id,
    trajectory_id,
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

    """
    Runs network clamp simulation for the specified gid, or for all gids found in the input data file.
    """

    init_params = dict(locals())
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    np.seterr(all="raise")
    verbose = True
    init_params["verbose"] = verbose
    config_logging(verbose)

    pop_params_tuple_dicts = None
    if rank == 0:
        if results_file_id is None:
            results_file_id = generate_results_file_id(
                population, seed=input_seed
            )
        if len(params_path) > 0:
            pop_params_tuple_dicts = []
            if len(params_id) == 0:
                params_id = [None] * len(params_path)
            for this_params_path, this_param_id in zip(params_path, params_id):
                pop_params_dict = read_from_yaml(this_params_path)
                pop_params_tuple_dict = {}
                for (
                    this_pop_name,
                    this_pop_param_dict,
                ) in pop_params_dict.items():
                    this_pop_params_tuple_dict = defaultdict(list)
                    for (
                        this_gid,
                        this_gid_params,
                    ) in this_pop_param_dict.items():
                        if this_param_id is not None:
                            this_gid_params_list = this_gid_params[
                                this_param_id
                            ]
                        else:
                            this_gid_params_list = this_gid_params
                        for this_gid_param in this_gid_params_list:
                            (
                                this_population,
                                source,
                                sec_type,
                                syn_name,
                                param_path,
                                param_val,
                            ) = this_gid_param
                            syn_param = SynParam(
                                this_population,
                                source,
                                sec_type,
                                syn_name,
                                param_path,
                                None,
                            )
                            this_pop_params_tuple_dict[this_gid].append(
                                (syn_param, param_val)
                            )
                    pop_params_tuple_dict[this_pop_name] = dict(
                        this_pop_params_tuple_dict
                    )
                pop_params_tuple_dicts.append(pop_params_tuple_dict)
    results_file_id = comm.bcast(results_file_id, root=0)
    init_params["results_file_id"] = results_file_id
    pop_params_tuple_dicts = comm.bcast(pop_params_tuple_dicts, root=0)

    cell_index_set = set()
    if gid_selection_file is not None:
        with open(gid_selection_file) as f:
            lines = f.readlines()
            for line in lines:
                gid = int(line)
                cell_index_set.add(gid)
    elif gids is not None:
        for gid in gids:
            cell_index_set.add(gid)
    else:
        comm.barrier()
        comm0 = comm.Split(2 if rank == 0 else 1, 0)
        if rank == 0:
            init_params["config"] = init_params.pop("config_file", None)
            env = Env(**init_params, comm=comm0)
            attr_info_dict = read_cell_attribute_info(
                env.data_file_path,
                populations=[population],
                read_cell_index=True,
                comm=comm0,
            )
            cell_index = None
            attr_name, attr_cell_index = next(
                iter(attr_info_dict[population]["Trees"])
            )
            cell_index_set = set(attr_cell_index)
        comm.barrier()
        cell_index_set = comm.bcast(cell_index_set, root=0)
        comm.barrier()
        comm0.Free()

    if size > 1:
        import distwq

        if distwq.is_controller:
            distwq.run(
                fun_name="dist_ctrl",
                module_name="miv_simulator.network_clamp",
                verbose=True,
                args=(
                    init_params,
                    cell_index_set,
                    params_path,
                    pop_params_tuple_dicts,
                ),
                spawn_workers=True,
                nprocs_per_worker=1,
            )

        else:
            distwq.run(verbose=True, spawn_workers=True, nprocs_per_worker=1)
    else:
        init_params["config"] = init_params.pop("config_file", None)
        env = Env(**init_params, comm=comm)
        configure_hoc_env(env)
        init(
            env,
            population,
            cell_index_set,
            arena_id,
            trajectory_id,
            n_trials,
            spike_events_path,
            spike_events_namespace=spike_events_namespace,
            spike_train_attr_name=spike_events_t,
            coords_path=coords_path,
            distances_namespace=distances_namespace,
            phase_mod=phase_mod,
            input_features_path=input_features_path,
            input_features_namespaces=input_features_namespaces,
            generate_weights_pops=set(generate_weights),
            t_min=t_min,
            t_max=t_max,
            input_seed=input_seed,
            plot_cell=plot_cell,
            write_cell=write_cell,
        )
        if pop_params_tuple_dicts is not None:
            for this_params_path, pop_params_tuple_dict in zip(
                params_path, pop_params_tuple_dicts
            ):
                params_basename = os.path.splitext(
                    os.path.basename(this_params_path)
                )[0]
                env.results_file_id = f"{results_file_id}_{params_basename}"
                env.results_file_path = f"{env.results_path}/{env.modelName}_results_{env.results_file_id}.h5"
                run_with(env, pop_params_tuple_dict)
                write_output(env)
                write_params(env, pop_params_tuple_dict)
        else:
            run(env)
            write_output(env)
        if env.profile_memory:
            profile_memory(logger)


@click.command()
@click.option(
    "--config-file",
    "-c",
    required=True,
    type=str,
    help="model configuration file name",
)
@click.option(
    "--population",
    "-p",
    required=True,
    type=str,
    default="GC",
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
    "--trajectory-id", "-t", type=str, required=False, help="trajectory id"
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
    population,
    dt,
    gids,
    gid_selection_file,
    arena_id,
    trajectory_id,
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
    """
    Optimize the firing rate of the specified cell in a network clamp configuration.
    """
    init_params = dict(locals())

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    results_file_id = None
    if rank == 0:
        ts = time.strftime("%Y%m%d_%H%M%S")
        opt_seed_lab = (
            f"NOS{np.random.randint(99999999):08d}"
            if opt_seed is None
            else f"{opt_seed:08d}"
        )
        results_file_id = f"{population!s}_{ts!s}_{opt_seed_lab!s}"

    results_file_id = comm.bcast(results_file_id, root=0)
    comm.barrier()

    np.seterr(all="raise")
    verbose = True
    cache_queries = True

    cell_index_set = set()
    if gid_selection_file is not None:
        with open(gid_selection_file) as f:
            lines = f.readlines()
            for line in lines:
                gid = int(line)
                cell_index_set.add(gid)
    elif gids is not None:
        for gid in gids:
            cell_index_set.add(gid)
    else:
        comm.barrier()
        comm0 = comm.Split(2 if rank == 0 else 1, 0)
        if rank == 0:
            init_params["config"] = init_params.pop("config_file", None)
            env = Env(**init_params, comm=comm0)
            attr_info_dict = read_cell_attribute_info(
                env.data_file_path,
                populations=[population],
                read_cell_index=True,
                comm=comm0,
            )
            cell_index = None
            attr_name, attr_cell_index = next(
                iter(attr_info_dict[population]["Trees"])
            )
            cell_index_set = set(attr_cell_index)
        comm.barrier()
        cell_index_set = comm.bcast(cell_index_set, root=0)
        comm.barrier()
        comm0.Free()
    init_params["cell_index_set"] = cell_index_set
    del init_params["gids"]

    N_objectives = 1
    opt_rate_feature_dtypes = [
        ("mean_rate", (np.float32, (1,))),
        ("trial_objs", (np.float32, (N_objectives, n_trials))),
        ("mean_v", (np.float32, (n_trials,))),
    ]
    params = dict(locals())
    params["config"] = params.pop("config_file", None)
    env = Env(**params)
    if size == 1:
        configure_hoc_env(env)
        init(
            env,
            population,
            cell_index_set,
            arena_id,
            trajectory_id,
            n_trials,
            spike_events_path,
            spike_events_namespace=spike_events_namespace,
            spike_train_attr_name=spike_events_t,
            coords_path=coords_path,
            distances_namespace=distances_namespace,
            phase_mod=phase_mod,
            input_features_path=input_features_path,
            input_features_namespaces=input_features_namespaces,
            generate_weights_pops=set(generate_weights),
            t_min=t_min,
            t_max=t_max,
        )

    if population in env.netclamp_config.optimize_parameters[param_type]:
        opt_params = env.netclamp_config.optimize_parameters[param_type][
            population
        ]
    else:
        raise RuntimeError(
            f"network_clamp.optimize: population {population} does not have optimization configuration"
        )

    if target == "rate":
        opt_target = opt_params["Targets"]["firing rate"]
        init_params["target_rate"] = opt_target
        init_objfun_name = "init_rate_objfun"
        feature_dtypes = opt_rate_feature_dtypes
        constraint_names = ["mean_v_below_threshold"]
    elif target == "state":
        assert target_state_variable is not None
        opt_target = opt_params["Targets"]["state"][target_state_variable][
            "mean"
        ]
        init_params["target_value"] = opt_target
        init_params["state_variable"] = target_state_variable
        init_params["state_filter"] = target_state_filter
        init_objfun_name = "init_state_objfun"
        feature_dtypes = None
        constraint_names = None
    elif target == "ratedist" or target == "rate_dist":
        init_params["target_features_arena"] = arena_id
        init_params["target_features_trajectory"] = trajectory_id
        init_objfun_name = "init_rate_dist_objfun"
        feature_dtypes = None
        constraint_names = None
    else:
        raise RuntimeError(
            f"network_clamp.optimize: unknown optimization target {target}"
        )

    results_config_dict = optimize_run(
        env,
        population,
        param_config_name,
        init_objfun_name,
        problem_regime=problem_regime,
        opt_iter=opt_iter,
        solver_epsilon=opt_epsilon,
        opt_seed=opt_seed,
        param_type=param_type,
        init_params=init_params,
        feature_dtypes=feature_dtypes,
        constraint_names=constraint_names,
        results_file=results_file,
        nprocs_per_worker=nprocs_per_worker,
        cooperative_init=cooperative_init,
        verbose=verbose,
    )
    if results_config_dict is not None:
        if results_path is not None:
            file_path = (
                f"{results_path}/network_clamp.optimize.{results_file_id}.yaml"
            )
            write_to_yaml(file_path, results_config_dict)
    comm.barrier()


cli.add_command(show)
cli.add_command(go)
cli.add_command(optimize)

if __name__ == "__main__":

    cli(
        args=sys.argv[
            (
                list_find(
                    lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv
                )
                + 1
            ) :
        ],
        standalone_mode=False,
    )
