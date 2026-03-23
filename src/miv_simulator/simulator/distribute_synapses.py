import gc
import math
import os
import sys
import time
from collections import defaultdict
from functools import reduce

import h5py
import numpy as np
from miv_simulator import cells, utils
from miv_simulator import config
from miv_simulator.cells import make_section_graph
from miv_simulator.utils.neuron import configure_hoc, interplocs, load_template
from mpi4py import MPI
from neuroh5.io import (
    NeuroH5TreeGen,
    append_cell_attributes,
    read_population_ranges,
)
from typing import Optional, Tuple, Literal, Dict

logger = utils.get_module_logger(__name__)

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


# !for imperative API, use update_synapse_statistics instead
def update_syn_stats(env, syn_stats_dict, syn_dict):
    return update_synapse_statistics(syn_dict, syn_stats_dict)


def update_synapse_statistics(syn_dict, syn_stats_dict):
    this_syn_stats_dict = {
        "section": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
        "layer": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
        "swc_type": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
        "total": {"excitatory": 0, "inhibitory": 0},
    }

    for syn_id, syn_sec, syn_type, swc_type, syn_layer in zip(
        syn_dict["syn_ids"],
        syn_dict["syn_secs"],
        syn_dict["syn_types"],
        syn_dict["swc_types"],
        syn_dict["syn_layers"],
    ):
        if syn_type == config.SynapseTypesDef.excitatory:
            syn_type_str = "excitatory"
        elif syn_type == config.SynapseTypesDef.inhibitory:
            syn_type_str = "inhibitory"
        else:
            raise ValueError(f"Unknown synapse type {str(syn_type)}")

        syn_stats_dict["section"][syn_sec][syn_type_str] += 1
        syn_stats_dict["layer"][syn_layer][syn_type_str] += 1
        syn_stats_dict["swc_type"][swc_type][syn_type_str] += 1
        syn_stats_dict["total"][syn_type_str] += 1

        this_syn_stats_dict["section"][syn_sec][syn_type_str] += 1
        this_syn_stats_dict["layer"][syn_layer][syn_type_str] += 1
        this_syn_stats_dict["swc_type"][swc_type][syn_type_str] += 1
        this_syn_stats_dict["total"][syn_type_str] += 1

    return this_syn_stats_dict


def global_syn_summary(comm, syn_stats, gid_count, root):
    global_count = comm.gather(gid_count, root=root)
    global_count = np.sum(global_count)
    res = []
    for population in sorted(syn_stats):
        pop_syn_stats = syn_stats[population]
        for part in ["layer", "swc_type"]:
            syn_stats_dict = pop_syn_stats[part]
            for part_name in syn_stats_dict:
                for syn_type in syn_stats_dict[part_name]:
                    global_syn_count = comm.gather(
                        syn_stats_dict[part_name][syn_type], root=root
                    )
                    if comm.rank == root:
                        res.append(
                            f"{population} {part} {part_name}: mean {syn_type} synapses per cell: {np.sum(global_syn_count) / global_count:.2f}"
                        )
        total_syn_stats_dict = pop_syn_stats["total"]
        for syn_type in total_syn_stats_dict:
            global_syn_count = comm.gather(total_syn_stats_dict[syn_type], root=root)
            if comm.rank == root:
                res.append(
                    f"{population}: mean {syn_type} synapses per cell: {np.sum(global_syn_count) / global_count:.2f}"
                )

    return global_count, str.join("\n", res)


def local_syn_summary(syn_stats_dict):
    res = []
    for part_name in ["layer", "swc_type"]:
        for part_type in syn_stats_dict[part_name]:
            syn_count_dict = syn_stats_dict[part_name][part_type]
            for syn_type, syn_count in list(syn_count_dict.items()):
                res.append(
                    "%s %i: %s synapses: %i"
                    % (part_name, part_type, syn_type, syn_count)
                )
    return str.join("\n", res)


# !for imperative API, use check_synapses instead
def check_syns(
    gid,
    morph_dict,
    syn_stats_dict,
    seg_density_per_sec,
    layer_set_dict,
    swc_set_dict,
    env,
    logger,
):
    return check_synapses(
        gid,
        morph_dict,
        syn_stats_dict,
        seg_density_per_sec,
        layer_set_dict,
        swc_set_dict,
        env.layers,
        logger,
    )


def check_synapses(
    gid,
    morph_dict,
    syn_stats_dict,
    seg_density_per_sec,
    layer_set_dict,
    swc_set_dict,
    swc_defs,
    layer_defs,
    logger,
):
    layer_stats = syn_stats_dict["layer"]
    swc_stats = syn_stats_dict["swc_type"]

    warning_flag = False
    incomplete_layers = []
    for syn_type, layer_set in list(layer_set_dict.items()):
        for layer in layer_set:
            layer_index = layer_defs[layer]
            if layer_index in layer_stats:
                if layer_stats[layer_index][syn_type] <= 0:
                    incomplete_layers.append(layer)
                    warning_flag = True
            else:
                incomplete_layers.append(layer)
                warning_flag = True
    if warning_flag:
        logger.warning(
            f"Rank {MPI.COMM_WORLD.Get_rank()}: incomplete synapse layer set for cell {gid}: "
            f"  incomplete layers: {incomplete_layers}\n"
            f"  populated layers: {layer_stats}\n"
            f"  layer_set_dict: {layer_set_dict}\n"
            f"  seg_density_per_sec: {seg_density_per_sec}\n"
            f"  morph_dict: {morph_dict}"
        )
    for syn_type, swc_set in swc_set_dict.items():
        for swc_type in swc_set:
            swc_type_index = swc_defs[swc_type]
            if swc_type_index in swc_stats:
                if swc_stats[swc_type_index][syn_type] <= 0:
                    warning_flag = True
            else:
                warning_flag = True
    if warning_flag:
        logger.warning(
            f"Rank {MPI.COMM_WORLD.Get_rank()}: incomplete synapse swc type set for cell {gid}: {swc_stats}"
            f"  swc_set_dict: {swc_set_dict.items}\n"
            f"  seg_density_per_sec: {seg_density_per_sec}\n"
            f"   morph_dict: {morph_dict}"
        )


def get_node_attribute(name, content, sec, secnodes, x=None):
    if name in content:
        if x is None:
            return content[name]
        elif sec.n3d() == 0:
            return content[name][0]
        else:
            prev = None
            for i in range(sec.n3d()):
                pos = sec.arc3d(i) / sec.L
                if pos >= x:
                    if (prev is None) or (abs(pos - x) < abs(prev - x)):
                        return content[name][secnodes[i]]
                    else:
                        return content[name][secnodes[i - 1]]
                else:
                    prev = pos
    else:
        return None


def synapse_seg_density(
    syn_type_dict,
    layer_dict,
    layer_density_dicts,
    seg_dict,
    ran,
    neurotree_dict=None,
    max_density_retries: int = 1000,
):
    """
    Computes per-segment density of synapse placement.

    :param max_density_retries: Maximum attempts to draw a positive density from the
        normal distribution for each segment before raising RuntimeError.  The default
        (1000) is generous for any physically reasonable configuration; a RuntimeError
        almost always indicates a misconfiguration (e.g. non-positive mean with zero
        variance).
    """
    segdensity_dict = {}
    layers_dict = {}

    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict["section_topology"]["nodes"]
    else:
        secnodes_dict = None
    for syn_type_label, layer_density_dict in layer_density_dicts.items():
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for layer_label, density_dict in layer_density_dict.items():
            if layer_label == "default" or layer_label == -1:
                layer = "default"
            else:
                layer = int(layer_dict[layer_label])
            rans[layer] = ran
        segdensity = defaultdict(list)
        layers = defaultdict(list)
        total_seg_density = 0.0
        for sec_index, seg_list in seg_dict.items():
            for seg in seg_list:
                if neurotree_dict is not None:
                    secnodes = secnodes_dict[sec_index]
                    layer = get_node_attribute(
                        "layer", neurotree_dict, seg.sec, secnodes, seg.x
                    )
                else:
                    layer = -1
                layers[sec_index].append(layer)

                this_ran = None

                if layer > -1:
                    if layer in rans:
                        this_ran = rans[layer]
                    elif "default" in rans:
                        this_ran = rans["default"]
                    else:
                        this_ran = None
                elif "default" in rans:
                    this_ran = rans["default"]
                else:
                    this_ran = None
                if this_ran is not None:
                    for _attempt in range(max_density_retries):
                        dens = this_ran.normal(
                            density_dict["mean"], density_dict["variance"]
                        )
                        if dens > 0.0:
                            break
                    else:
                        raise RuntimeError(
                            f"synapse_seg_density: could not draw a positive density "
                            f"for synapse type '{syn_type_label}' after "
                            f"{max_density_retries} attempts "
                            f"(mean={density_dict['mean']}, "
                            f"variance={density_dict['variance']}). "
                            f"Check density configuration for section index {sec_index}."
                        )
                else:
                    dens = 0.0
                total_seg_density += dens
                segdensity[sec_index].append(dens)

        if total_seg_density < 1e-6:
            logger.warning(
                f"sections with zero {syn_type_label} "
                f"synapse density: {segdensity}; rans: {rans}; "
                f"density_dict: {density_dict}; layers: {layers} "
                f"morphology: {neurotree_dict}"
            )

        segdensity_dict[syn_type] = segdensity
        layers_dict[syn_type] = layers
    return (segdensity_dict, layers_dict)


def synapse_seg_counts(
    syn_type_dict,
    layer_dict,
    layer_density_dicts,
    sec_index_dict,
    seg_dict,
    ran,
    neurotree_dict=None,
):
    """
    Computes per-segment relative counts of synapse placement.
    """
    segcounts_dict = {}
    layers_dict = {}
    segcount_total = 0
    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict["section_topology"]["nodes"]
    else:
        secnodes_dict = None
    for syn_type_label, layer_density_dict in layer_density_dicts.items():
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for layer_label, density_dict in layer_density_dict.items():
            if layer_label == "default" or layer_label == -1:
                layer = "default"
            else:
                layer = layer_dict[layer_label]

            rans[layer] = ran
        segcounts = []
        layers = []
        for sec_index, seg_list in seg_dict.items():
            for seg in seg_list:
                L = seg.sec.L
                nseg = seg.sec.nseg
                if neurotree_dict is not None:
                    secnodes = secnodes_dict[sec_index]
                    layer = get_node_attribute(
                        "layer", neurotree_dict, seg.sec, secnodes, seg.x
                    )
                else:
                    layer = -1
                layers.append(layer)

                ran = None

                if layer > -1:
                    if layer in rans:
                        ran = rans[layer]
                    elif "default" in rans:
                        ran = rans["default"]
                    else:
                        ran = None
                elif "default" in rans:
                    ran = rans["default"]
                else:
                    ran = None
                if ran is not None:
                    pos = L / nseg
                    dens = ran.normal(density_dict["mean"], density_dict["variance"])
                    rc = dens * pos
                    segcount_total += rc
                    segcounts.append(rc)
                else:
                    segcounts.append(0)

            segcounts_dict[syn_type] = segcounts
            layers_dict[syn_type] = layers
    return (segcounts_dict, segcount_total, layers_dict)


def distribute_uniform_synapses(
    density_seed,
    syn_type_dict,
    swc_type_dict,
    layer_dict,
    sec_layer_density_dict,
    neurotree_dict,
    cell_sec_dict,
    cell_secidx_dict,
):
    """
    Computes uniformly-spaced synapse locations.
    """
    syn_ids = []
    syn_locs = []
    syn_cdists = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    r = np.random.RandomState()

    sec_interp_loc_dict = {}
    segcounts_per_sec = {}
    for sec_name, layer_density_dict in sec_layer_density_dict.items():
        sec_index_dict = cell_secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        L_total = 0
        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        for sec, idx in zip(seclst, secidxlst):
            sec_interp_loc_dict[idx] = interplocs(sec)
        sec_dict = {int(idx): sec for sec, idx in zip(seclst, secidxlst)}
        seg_dict = {}
        for sec_index, sec in sec_dict.items():
            seg_list = []
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if (
                        seg.x < 1.0
                        and seg.x > 0.0
                        and ((L_total + sec.L * seg.x) <= maxdist)
                    ):
                        seg_list.append(seg)
            L_total += sec.L
            seg_dict[sec_index] = seg_list
        segcounts_dict, total, layers_dict = synapse_seg_counts(
            syn_type_dict,
            layer_dict,
            layer_density_dict,
            sec_index_dict=sec_index_dict,
            seg_dict=seg_dict,
            ran=r,
            neurotree_dict=neurotree_dict,
        )
        segcounts_per_sec[sec_name] = segcounts_dict
        for syn_type_label, _ in layer_density_dict.items():
            syn_type = syn_type_dict[syn_type_label]
            segcounts = segcounts_dict[syn_type]
            layers = layers_dict[syn_type]
            for sec_index, seg_list in seg_dict.items():
                interp_loc = sec_interp_loc_dict[sec_index]
                for seg, layer, seg_count in zip(seg_list, layers, segcounts):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end = seg.x + (0.5 / seg.sec.nseg)
                    seg_range = seg_end - seg_start
                    int_seg_count = math.floor(seg_count)
                    syn_count = 0
                    while syn_count < int_seg_count:
                        syn_loc = seg_start + seg_range * (syn_count + 1) / math.ceil(
                            seg_count
                        )
                        assert (syn_loc <= 1) & (syn_loc >= 0)
                        if syn_loc < 1.0:
                            syn_cdist = math.sqrt(
                                reduce(
                                    lambda a, b: a + b,
                                    (interp_loc[i](syn_loc) ** 2 for i in range(3)),
                                )
                            )
                            syn_cdists.append(syn_cdist)
                            syn_locs.append(syn_loc)
                            syn_ids.append(syn_index)
                            syn_secs.append(sec_index_dict[seg.sec])
                            syn_layers.append(layer)
                            syn_types.append(syn_type)
                            swc_types.append(swc_type)
                            syn_index += 1
                        syn_count += 1

    assert len(syn_ids) > 0
    syn_dict = {
        "syn_ids": np.asarray(syn_ids, dtype="uint32"),
        "syn_cdists": np.asarray(syn_cdists, dtype="float32"),
        "syn_locs": np.asarray(syn_locs, dtype="float32"),
        "syn_secs": np.asarray(syn_secs, dtype="uint32"),
        "syn_layers": np.asarray(syn_layers, dtype="int8"),
        "syn_types": np.asarray(syn_types, dtype="uint8"),
        "swc_types": np.asarray(swc_types, dtype="uint8"),
    }

    return (syn_dict, segcounts_per_sec)


def distribute_poisson_synapses(
    density_seed,
    syn_type_dict,
    swc_type_dict,
    layer_dict,
    sec_layer_density_dict,
    neurotree_dict,
    cell_sec_dict,
    cell_secidx_dict,
    max_density_retries: int = 1000,
    max_placement_retries: int = 100_000,
):
    """
    Computes synapse locations distributed according to a Poisson distribution.

    :param max_density_retries: Forwarded to synapse_seg_density; maximum attempts to
        draw a positive per-segment density before raising RuntimeError.
    :param max_placement_retries: Maximum attempts to draw an exponential inter-arrival
        sample that falls within the first segment's bounds when placing the very first
        synapse on a section.  The default (100 000) guards against degenerate
        configurations where the mean inter-arrival distance (1/density) far exceeds the
        segment length.  A RuntimeError here indicates a misconfiguration.
    """
    import networkx as nx

    syn_ids = []
    syn_cdists = []
    syn_locs = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    sec_graph = make_section_graph(neurotree_dict)

    debug_flag = False
    secnodes_dict = neurotree_dict["section_topology"]["nodes"]
    for sec, secnodes in secnodes_dict.items():
        if len(secnodes) < 2:
            debug_flag = True

    if debug_flag:
        logger.debug(f"sec_graph: {str(list(sec_graph.edges))}")
        logger.debug(f"neurotree_dict: {str(neurotree_dict)}")

    sec_interp_loc_dict = {}
    seg_density_per_sec = {}
    r = np.random.RandomState()
    r.seed(int(density_seed))
    for sec_name, layer_density_dict in sec_layer_density_dict.items():
        swc_type = swc_type_dict[sec_name]
        seg_dict = {}
        L_total = 0

        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        for sec, idx in zip(seclst, secidxlst):
            sec_interp_loc_dict[idx] = interplocs(sec)
        sec_dict = {int(idx): sec for sec, idx in zip(seclst, secidxlst)}
        if len(sec_dict) > 1:
            sec_subgraph = sec_graph.subgraph(list(sec_dict.keys()))
            if len(sec_subgraph.edges()) > 0:
                sec_roots = [n for n, d in sec_subgraph.in_degree() if d == 0]
                sec_edges = []
                for sec_root in sec_roots:
                    sec_edges.append(list(nx.dfs_edges(sec_subgraph, sec_root)))
                    sec_edges.append([(None, sec_root)])
                sec_edges = [val for sublist in sec_edges for val in sublist]
            else:
                sec_edges = [(None, idx) for idx in list(sec_dict.keys())]
        else:
            sec_edges = [(None, idx) for idx in list(sec_dict.keys())]
        for sec_index, sec in sec_dict.items():
            seg_list = []
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if (
                        seg.x < 1.0
                        and seg.x > 0.0
                        and ((L_total + sec.L * seg.x) <= maxdist)
                    ):
                        seg_list.append(seg)
            seg_dict[sec_index] = seg_list
            L_total += sec.L
        seg_density_dict, layers_dict = synapse_seg_density(
            syn_type_dict,
            layer_dict,
            layer_density_dict,
            seg_dict,
            r,
            neurotree_dict=neurotree_dict,
            max_density_retries=max_density_retries,
        )
        seg_density_per_sec[sec_name] = seg_density_dict
        for syn_type_label, _ in layer_density_dict.items():
            syn_type = syn_type_dict[syn_type_label]
            seg_density = seg_density_dict[syn_type]
            layers = layers_dict[syn_type]
            end_distance = {}
            for sec_parent, sec_index in sec_edges:
                interp_loc = sec_interp_loc_dict[sec_index]
                seg_list = seg_dict[sec_index]
                sec_seg_layers = layers[sec_index]
                sec_seg_density = seg_density[sec_index]
                interval = 0.0
                syn_loc = 0.0
                for seg, layer, density in zip(
                    seg_list, sec_seg_layers, sec_seg_density
                ):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end = seg.x + (0.5 / seg.sec.nseg)
                    L = seg.sec.L
                    L_seg_start = seg_start * L
                    L_seg_end = seg_end * L
                    if density > 0.0:
                        beta = 1.0 / density
                        if interval > 0.0:
                            sample = r.exponential(beta)
                        else:
                            for _attempt in range(max_placement_retries):
                                sample = r.exponential(beta)
                                if (sample >= L_seg_start) and (sample < L_seg_end):
                                    break
                            else:
                                raise RuntimeError(
                                    f"distribute_poisson_synapses: could not place "
                                    f"initial synapse in segment "
                                    f"[{L_seg_start:.4g}, {L_seg_end:.4g}) "
                                    f"for section index {sec_index} after "
                                    f"{max_placement_retries} attempts "
                                    f"(density={density:.4g}, beta={beta:.4g}). "
                                    f"Consider increasing density or reducing nseg."
                                )
                        interval += sample
                        while interval < L_seg_end:
                            if interval >= L_seg_start:
                                syn_loc = interval / L
                                assert (syn_loc <= 1) and (syn_loc >= seg_start)
                                if syn_loc < 1.0:
                                    syn_cdist = math.sqrt(
                                        reduce(
                                            lambda a, b: a + b,
                                            (
                                                interp_loc[i](syn_loc) ** 2
                                                for i in range(3)
                                            ),
                                        )
                                    )
                                    syn_cdists.append(syn_cdist)
                                    syn_locs.append(syn_loc)
                                    syn_ids.append(syn_index)
                                    syn_secs.append(sec_index)
                                    syn_layers.append(layer)
                                    syn_types.append(syn_type)
                                    swc_types.append(swc_type)
                                    syn_index += 1
                            interval += r.exponential(beta)
                    else:
                        interval = seg_end * L
                end_distance[sec_index] = (1.0 - syn_loc) * L

    assert len(syn_ids) > 0
    syn_dict = {
        "syn_ids": np.asarray(syn_ids, dtype="uint32"),
        "syn_cdists": np.asarray(syn_cdists, dtype="float32"),
        "syn_locs": np.asarray(syn_locs, dtype="float32"),
        "syn_secs": np.asarray(syn_secs, dtype="uint32"),
        "syn_layers": np.asarray(syn_layers, dtype="int8"),
        "syn_types": np.asarray(syn_types, dtype="uint8"),
        "swc_types": np.asarray(swc_types, dtype="uint8"),
    }

    return (syn_dict, seg_density_per_sec)


# !for imperative API, use distribute_synapses instead
def distribute_synapse_locations(
    config,
    template_path,
    output_path,
    forest_path,
    populations,
    distribution,
    io_size,
    chunk_size,
    value_chunk_size,
    write_size,
    verbose,
    dry_run,
    debug,
    config_prefix="",
    mechanisms_path=None,
):
    if mechanisms_path is None:
        mechanisms_path = "./mechanisms"

    utils.config_logging(verbose)
    from miv_simulator.env import Env

    env = Env(
        comm=MPI.COMM_WORLD,
        config=config,
        template_paths=template_path,
        config_prefix=config_prefix,
    )

    configure_hoc(
        template_directory=template_path,
        use_coreneuron=env.use_coreneuron,
        mechanisms_directory=mechanisms_path,
        dt=env.dt,
        tstop=env.tstop,
        celsius=env.globals.get("celsius", None),
    )

    return distribute_synapses(
        forest_filepath=forest_path,
        cell_types=env.celltypes,
        swc_defs=env.SWC_Types,
        synapse_defs=env.Synapse_Types,
        layer_defs=env.layers,
        populations=populations,
        distribution=distribution,
        template_path=template_path,
        io_size=io_size,
        output_filepath=output_path,
        write_size=write_size,
        chunk_size=chunk_size,
        value_chunk_size=value_chunk_size,
        seed=env.model_config["Random Seeds"]["Synapse Locations"],
        dry_run=dry_run,
    )


def distribute_synapses(
    forest_filepath: str,
    cell_types: config.CellTypes,
    swc_defs: Dict[str, int],
    synapse_defs: Dict[str, int],
    layer_defs: Dict[str, int],
    populations: Tuple[str, ...],
    distribution: Literal["uniform", "poisson"],
    template_path: str,
    output_filepath: Optional[str],
    io_size: int,
    write_size: int,
    chunk_size: int,
    value_chunk_size: int,
    seed: Optional[int],
    dry_run: bool,
):
    logger = utils.get_script_logger(os.path.basename(__file__))
    write_size = 1
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        logger.info(f"{comm.size} ranks have been allocated")

    configure_hoc()

    if io_size == -1:
        io_size = comm.size

    if output_filepath is None:
        output_filepath = forest_filepath

    if not dry_run:
        if rank == 0:
            if not os.path.isfile(output_filepath):
                input_file = h5py.File(forest_filepath, "r")
                output_file = h5py.File(output_filepath, "w")
                input_file.copy("/H5Types", output_file)
                input_file.close()
                output_file.close()
        comm.barrier()

    (pop_ranges, _) = read_population_ranges(forest_filepath, comm=comm)
    start_time = time.time()
    syn_stats = dict()
    for population in populations:
        syn_stats[population] = {
            "section": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "layer": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "swc_type": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "total": {"excitatory": 0, "inhibitory": 0},
        }

    for population in populations:
        logger.info(f"Rank {rank} population: {population}")
        (population_start, _) = pop_ranges[population]
        template_class = load_template(
            population_name=population,
            template_name=cell_types[population]["template"],
            template_path=template_path,
        )

        density_dict = cell_types[population]["synapses"]["density"]
        layer_set_dict = defaultdict(set)
        swc_set_dict = defaultdict(set)
        for sec_name, sec_dict in density_dict.items():
            for syn_type, syn_dict in sec_dict.items():
                swc_set_dict[syn_type].add(sec_name)
                for layer_name in syn_dict:
                    if layer_name != "default":
                        layer_set_dict[syn_type].add(layer_name)

        syn_stats_dict = {
            "section": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "layer": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "swc_type": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "total": {"excitatory": 0, "inhibitory": 0},
        }

        count = 0
        gid_count = 0
        synapse_dict = {}
        for gid, morph_dict in NeuroH5TreeGen(
            forest_filepath,
            population,
            io_size=io_size,
            comm=comm,
            topology=True,
        ):
            local_time = time.time()
            if gid is not None:
                cell = cells.make_neurotree_hoc_cell(
                    template_class, neurotree_dict=morph_dict, gid=gid
                )
                cell_sec_dict = {
                    "apical": (cell.apical_list, None),
                    "basal": (cell.basal_list, None),
                    "soma": (cell.soma_list, None),
                    "ais": (cell.ais_list, None),
                    "hillock": (cell.hillock_list, None),
                }
                cell_secidx_dict = {
                    "apical": cell.apicalidx,
                    "basal": cell.basalidx,
                    "soma": cell.somaidx,
                    "ais": cell.aisidx,
                    "hillock": cell.hilidx,
                }

                random_seed = (seed or 0) + gid
                if distribution == "uniform":
                    (
                        syn_dict,
                        seg_density_per_sec,
                    ) = distribute_uniform_synapses(
                        random_seed,
                        synapse_defs,
                        swc_defs,
                        layer_defs,
                        density_dict,
                        morph_dict,
                        cell_sec_dict,
                        cell_secidx_dict,
                    )

                elif distribution == "poisson":
                    (
                        syn_dict,
                        seg_density_per_sec,
                    ) = distribute_poisson_synapses(
                        random_seed,
                        synapse_defs,
                        swc_defs,
                        layer_defs,
                        density_dict,
                        morph_dict,
                        cell_sec_dict,
                        cell_secidx_dict,
                    )
                else:
                    raise Exception(f"Unknown distribution type: {distribution}")

                synapse_dict[gid] = syn_dict
                this_syn_stats = update_synapse_statistics(syn_dict, syn_stats_dict)
                check_synapses(
                    gid,
                    morph_dict,
                    this_syn_stats,
                    seg_density_per_sec,
                    layer_set_dict,
                    swc_set_dict,
                    swc_defs,
                    layer_defs,
                    logger,
                )

                del cell
                num_syns = len(synapse_dict[gid]["syn_ids"])
                logger.info(
                    f"Rank {rank} took {time.time() - local_time:.2f} s to compute {num_syns} synapse locations for {population} gid: {gid}\n"
                    f"{local_syn_summary(this_syn_stats)}"
                )
                gid_count += 1
            else:
                logger.info(f"Rank {rank} gid is None")
            gc.collect()

            # Check if any rank has reached the write_size threshold and
            #  ensure that all ranks in that group call collectively
            local_should_write = (
                (not dry_run) and (write_size > 0) and (gid_count % write_size == 0)
            )
            global_should_write = comm.allreduce(local_should_write, op=MPI.LOR)

            if global_should_write:
                append_cell_attributes(
                    output_filepath,
                    population,
                    synapse_dict,
                    namespace="Synapse Attributes",
                    comm=comm,
                    io_size=io_size,
                    chunk_size=chunk_size,
                    value_chunk_size=value_chunk_size,
                )
                synapse_dict = {}
            syn_stats[population] = syn_stats_dict
            count += 1

        # Final write for any remaining synapse data - allreduce to ensure all ranks participate
        local_should_write_final = not dry_run
        global_should_write_final = comm.allreduce(
            local_should_write_final, op=MPI.LAND
        )

        if global_should_write_final:
            append_cell_attributes(
                output_filepath,
                population,
                synapse_dict,
                namespace="Synapse Attributes",
                comm=comm,
                io_size=io_size,
                chunk_size=chunk_size,
                value_chunk_size=value_chunk_size,
            )

        global_count, summary = global_syn_summary(comm, syn_stats, gid_count, root=0)
        if rank == 0:
            logger.info(
                f"Population: {population}, {comm.size} ranks took {time.time() - start_time:.2f} s "
                f"to compute synapse locations for {np.sum(global_count)} cells"
            )
            logger.info(summary)
