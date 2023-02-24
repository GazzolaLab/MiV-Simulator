__doc__ = """
Network initialization routines.
"""

from typing import Dict, Union
import gc
import os
import sys
import pprint
import time

import numpy as np
from miv_simulator import cells, lfp, lpt, synapses
from miv_simulator.env import Env
from miv_simulator.utils import (
    Promise,
    compose_iter,
    get_module_logger,
    imapreduce,
)
from miv_simulator.utils import io as io_utils
from miv_simulator.utils import neuron as neuron_utils
from miv_simulator.utils import profile_memory, simtime, zip_longest
from miv_simulator.utils.neuron import h
from miv_simulator.opto.run import *

if hasattr(h, "nrnmpi_init"):
    h.nrnmpi_init()

from mpi4py import MPI
from neuroh5.io import (
    NeuroH5CellAttrGen,
    bcast_graph,
    read_cell_attribute_selection,
    read_graph_selection,
    read_tree_selection,
    scatter_read_cell_attribute_selection,
    scatter_read_cell_attributes,
    scatter_read_graph,
    scatter_read_trees,
)
from numpy import ndarray

# This logger will inherit its settings from the root logger, created in miv_simulator.env
logger = get_module_logger(__name__)


def set_union(a, b, datatype):
    return a.union(b)


mpi_op_set_union = MPI.Op.Create(set_union, commute=True)


def ld_bal(env):
    """
    For given cxvec on each rank, calculates the fractional load balance.

    :param env: an instance of the `miv_simulator.Env` class.
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    cxvec = env.cxvec
    sum_cx = sum(cxvec)
    max_sum_cx = env.pc.allreduce(sum_cx, 2)
    sum_cx = env.pc.allreduce(sum_cx, 1)
    if rank == 0:
        logger.info(
            f"*** expected load balance {(((sum_cx / nhosts) / max_sum_cx)):.2f}"
        )


def lpt_bal(env):
    """
    Load-balancing based on the LPT algorithm.
    Each rank has gidvec, cxvec: gather everything to rank 0, do lpt
    algorithm and write to a balance file.

    :param env: an instance of the `miv_simulator.Env` class.
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    cxvec = env.cxvec
    gidvec = list(env.gidset)
    src = [None] * nhosts
    src[0] = list(zip(cxvec, gidvec))
    dest = env.pc.py_alltoall(src)
    del src

    if rank == 0:
        allpairs = sum(dest, [])
        del dest
        parts = lpt.lpt(allpairs, nhosts)
        lpt.statistics(parts)
        part_rank = 0
        with open(f"parts.{nhosts}", "w") as fp:
            for part in parts:
                for x in part[1]:
                    fp.write("%d %d\n" % (x[1], part_rank))
                part_rank = part_rank + 1
    env.pc.barrier()


def connect_cells(env: Env) -> None:
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    synapse and network connection mechanisms for each postsynaptic cell.

    :param env: an instance of the `miv_simulator.Env` class
    """
    connectivity_file_path = env.connectivity_file_path
    forest_file_path = env.forest_file_path
    rank = int(env.pc.id())
    syn_attrs = env.synapse_attributes

    if rank == 0:
        logger.info(f"*** Connectivity file path is {connectivity_file_path}")
        logger.info("*** Reading projections: ")

    biophys_cell_count = 0
    for postsyn_name, presyn_names in sorted(env.projection_dict.items()):
        if rank == 0:
            logger.info(f"*** Reading projections of population {postsyn_name}")

        synapse_config = env.celltypes[postsyn_name]["synapses"]
        if "correct_for_spines" in synapse_config:
            correct_for_spines = synapse_config["correct_for_spines"]
        else:
            correct_for_spines = False

        if "unique" in synapse_config:
            unique = synapse_config["unique"]
        else:
            unique = False
        weight_dicts = []
        has_weights = False
        if "weights" in synapse_config:
            has_weights = True
            weight_dicts = synapse_config["weights"]

        if rank == 0:
            logger.info(
                f"*** Reading synaptic attributes of population {postsyn_name}"
            )

        cell_attr_namespaces = ["Synapse Attributes"]

        if env.use_cell_attr_gen:
            synapses_attr_gen = None
            if env.node_allocation is None:
                synapses_attr_gen = NeuroH5CellAttrGen(
                    forest_file_path,
                    postsyn_name,
                    namespace="Synapse Attributes",
                    comm=env.comm,
                    return_type="tuple",
                    io_size=env.io_size,
                    cache_size=env.cell_attr_gen_cache_size,
                )
            else:
                synapses_attr_gen = NeuroH5CellAttrGen(
                    forest_file_path,
                    postsyn_name,
                    namespace="Synapse Attributes",
                    comm=env.comm,
                    return_type="tuple",
                    io_size=env.io_size,
                    cache_size=env.cell_attr_gen_cache_size,
                    node_allocation=env.node_allocation,
                )

            for iter_count, (gid, gid_attr_data) in enumerate(
                synapses_attr_gen
            ):
                if gid is not None:
                    (attr_tuple, attr_tuple_index) = gid_attr_data
                    syn_ids_ind = attr_tuple_index.get("syn_ids", None)
                    syn_locs_ind = attr_tuple_index.get("syn_locs", None)
                    syn_layers_ind = attr_tuple_index.get("syn_layers", None)
                    syn_types_ind = attr_tuple_index.get("syn_types", None)
                    swc_types_ind = attr_tuple_index.get("swc_types", None)
                    syn_secs_ind = attr_tuple_index.get("syn_secs", None)
                    syn_locs_ind = attr_tuple_index.get("syn_locs", None)

                    syn_ids = attr_tuple[syn_ids_ind]
                    syn_layers = attr_tuple[syn_layers_ind]
                    syn_types = attr_tuple[syn_types_ind]
                    swc_types = attr_tuple[swc_types_ind]
                    syn_secs = attr_tuple[syn_secs_ind]
                    syn_locs = attr_tuple[syn_locs_ind]

                    syn_attrs.init_syn_id_attrs(
                        gid,
                        syn_ids,
                        syn_layers,
                        syn_types,
                        swc_types,
                        syn_secs,
                        syn_locs,
                    )
        else:
            if env.node_allocation is None:
                cell_attributes_dict = scatter_read_cell_attributes(
                    forest_file_path,
                    postsyn_name,
                    namespaces=sorted(cell_attr_namespaces),
                    mask={
                        "syn_ids",
                        "syn_locs",
                        "syn_secs",
                        "syn_layers",
                        "syn_types",
                        "swc_types",
                    },
                    comm=env.comm,
                    io_size=env.io_size,
                    return_type="tuple",
                )
            else:
                cell_attributes_dict = scatter_read_cell_attributes(
                    forest_file_path,
                    postsyn_name,
                    namespaces=sorted(cell_attr_namespaces),
                    mask={
                        "syn_ids",
                        "syn_locs",
                        "syn_secs",
                        "syn_layers",
                        "syn_types",
                        "swc_types",
                    },
                    comm=env.comm,
                    node_allocation=env.node_allocation,
                    io_size=env.io_size,
                    return_type="tuple",
                )

                syn_attrs_iter, syn_attrs_info = cell_attributes_dict[
                    "Synapse Attributes"
                ]
                syn_attrs.init_syn_id_attrs_from_iter(
                    syn_attrs_iter,
                    attr_type="tuple",
                    attr_tuple_index=syn_attrs_info,
                    debug=(rank == 0),
                )
                del cell_attributes_dict
                gc.collect()

        weight_attr_mask = list(syn_attrs.syn_mech_names)
        weight_attr_mask.append("syn_id")

        if has_weights:
            for weight_dict in weight_dicts:
                expr_closure = weight_dict.get("closure", None)
                weights_namespaces = weight_dict["namespace"]

                if rank == 0:
                    logger.info(
                        f"*** Reading synaptic weights of population {postsyn_name} from namespaces {weights_namespaces}"
                    )

                if env.node_allocation is None:
                    weight_attr_dict = scatter_read_cell_attributes(
                        forest_file_path,
                        postsyn_name,
                        namespaces=weights_namespaces,
                        mask=set(weight_attr_mask),
                        comm=env.comm,
                        io_size=env.io_size,
                        return_type="tuple",
                    )
                else:
                    weight_attr_dict = scatter_read_cell_attributes(
                        forest_file_path,
                        postsyn_name,
                        namespaces=weights_namespaces,
                        mask=set(weight_attr_mask),
                        comm=env.comm,
                        node_allocation=env.node_allocation,
                        io_size=env.io_size,
                        return_type="tuple",
                    )

                append_weights = False
                multiple_weights = "error"
                for weights_namespace in weights_namespaces:
                    syn_weights_iter, weight_attr_info = weight_attr_dict[
                        weights_namespace
                    ]
                    first_gid = None
                    syn_id_index = weight_attr_info.get("syn_id", None)
                    syn_name_inds = [
                        (syn_name, attr_index)
                        for syn_name, attr_index in sorted(
                            weight_attr_info.items()
                        )
                        if syn_name != "syn_id"
                    ]
                    for gid, cell_weights_tuple in syn_weights_iter:
                        if first_gid is None:
                            first_gid = gid
                        weights_syn_ids = cell_weights_tuple[syn_id_index]
                        for syn_name, syn_name_index in syn_name_inds:
                            if syn_name not in syn_attrs.syn_mech_names:
                                if rank == 0 and first_gid == gid:
                                    logger.warning(
                                        f"*** connect_cells: population: {postsyn_name}; gid: {gid}; syn_name: {syn_name} "
                                        "not found in network configuration"
                                    )
                            else:
                                weights_values = cell_weights_tuple[
                                    syn_name_index
                                ]
                                assert len(weights_syn_ids) == len(
                                    weights_values
                                )
                                syn_attrs.add_mech_attrs_from_iter(
                                    gid,
                                    syn_name,
                                    zip_longest(
                                        weights_syn_ids,
                                        [
                                            {
                                                "weight": Promise(
                                                    expr_closure, [x]
                                                )
                                            }
                                            for x in weights_values
                                        ]
                                        if expr_closure
                                        else [
                                            {"weight": x}
                                            for x in weights_values
                                        ],
                                    ),
                                    multiple=multiple_weights,
                                    append=append_weights,
                                )
                                if rank == 0 and gid == first_gid:
                                    logger.info(
                                        f"*** connect_cells: population: {postsyn_name}; gid: {gid}; found {len(weights_values)} {syn_name} synaptic weights ({weights_namespace})"
                                    )
                    expr_closure = None
                    append_weights = True
                    multiple_weights = "overwrite"
                    del weight_attr_dict[weights_namespace]

        env.edge_count[postsyn_name] = 0
        for presyn_name in presyn_names:
            env.comm.barrier()
            if rank == 0:
                logger.info(
                    f"Rank {rank}: Reading projection {presyn_name} -> {postsyn_name}"
                )
            if env.node_allocation is None:
                (graph, a) = scatter_read_graph(
                    connectivity_file_path,
                    comm=env.comm,
                    io_size=env.io_size,
                    projections=[(presyn_name, postsyn_name)],
                    namespaces=["Synapses", "Connections"],
                )
            else:
                (graph, a) = scatter_read_graph(
                    connectivity_file_path,
                    comm=env.comm,
                    io_size=env.io_size,
                    node_allocation=env.node_allocation,
                    projections=[(presyn_name, postsyn_name)],
                    namespaces=["Synapses", "Connections"],
                )
            if rank == 0:
                logger.info(
                    f"Rank {rank}: Read projection {presyn_name} -> {postsyn_name}"
                )
            edge_iter = graph[postsyn_name][presyn_name]

            last_time = time.time()
            if env.microcircuit_inputs:
                presyn_input_sources = env.microcircuit_input_sources.get(
                    presyn_name, set()
                )
                syn_edge_iter = compose_iter(
                    lambda edgeset: presyn_input_sources.update(edgeset[1][0]),
                    edge_iter,
                )
                env.microcircuit_input_sources[
                    presyn_name
                ] = presyn_input_sources
            else:
                syn_edge_iter = edge_iter
            syn_attrs.init_edge_attrs_from_iter(
                postsyn_name, presyn_name, a, syn_edge_iter
            )
            if rank == 0:
                logger.info(
                    f"Rank {rank}: took {(time.time() - last_time):.02f} s to initialize edge attributes for projection {presyn_name} -> {postsyn_name}"
                )
                del graph[postsyn_name][presyn_name]

        first_gid = None
        if postsyn_name in env.biophys_cells:
            for gid in env.biophys_cells[postsyn_name]:
                if env.node_allocation is not None:
                    assert gid in env.node_allocation
                if first_gid is None:
                    first_gid = gid
                try:
                    biophys_cell = env.biophys_cells[postsyn_name][gid]
                    cells.init_biophysics(
                        biophys_cell,
                        env=env,
                        reset_cable=True,
                        correct_cm=correct_for_spines,
                        correct_g_pas=correct_for_spines,
                        verbose=((rank == 0) and (first_gid == gid)),
                    )
                    synapses.init_syn_mech_attrs(biophys_cell, env)
                except KeyError:
                    raise KeyError(
                        f"*** connect_cells: population: {postsyn_name}; gid: {gid}; could not initialize biophysics"
                    )

    gc.collect()

    ##
    ## This section instantiates cells that are not part of the
    ## network, but are presynaptic sources for cells that _are_
    ## part of the network. It is necessary to create cells at
    ## this point, as NEURON's ParallelContext does not allow the
    ## creation of gids after netcons including those gids are
    ## created.
    ##
    if env.microcircuit_inputs:
        make_input_cell_selection(env)
    gc.collect()

    first_gid = None
    start_time = time.time()

    gids = list(syn_attrs.gids())
    comm0 = env.comm.Split(2 if len(gids) > 0 else 0, 0)

    first_gid_set = set([])
    for gid in gids:
        if not env.pc.gid_exists(gid):
            logger.info(f"connect_cells: rank {rank}: gid {gid} does not exist")
        assert gid in env.gidset
        assert env.pc.gid_exists(gid)
        postsyn_cell = env.pc.gid2cell(gid)
        postsyn_name = find_gid_pop(env.celltypes, gid)

        first_gid = None
        if postsyn_name not in first_gid_set:
            first_gid = gid
            first_gid_set.add(postsyn_name)

        if rank == 0 and gid == first_gid:
            logger.info(f"Rank {rank}: configuring synapses for gid {gid}")

        last_time = time.time()

        syn_count, mech_count, nc_count = synapses.config_hoc_cell_syns(
            env,
            gid,
            postsyn_name,
            cell=postsyn_cell.hoc_cell
            if hasattr(postsyn_cell, "hoc_cell")
            else postsyn_cell,
            unique=unique,
            insert=True,
            insert_netcons=True,
        )

        if rank == 0 and gid == first_gid:
            logger.info(
                f"Rank {rank}: took {(time.time() - last_time):.02f} s to configure {syn_count} synapses, {mech_count} synaptic mechanisms, {nc_count} network "
                f"connections for gid {gid} from population {postsyn_name}"
            )
            hoc_cell = env.pc.gid2cell(gid)
            if hasattr(hoc_cell, "all"):
                if gid in env.biophys_cells[postsyn_name]:
                    biophys_cell = env.biophys_cells[postsyn_name][gid]
                for sec in list(hoc_cell.all):
                    logger.info(pprint.pformat(sec.psection()))

        env.edge_count[postsyn_name] += syn_count

        if gid in env.recording_sets.get(postsyn_name, {}):
            cells.record_cell(env, postsyn_name, gid)

        if env.cleanup:
            syn_attrs.del_syn_id_attr_dict(gid)
            if gid in env.biophys_cells[postsyn_name]:
                del env.biophys_cells[postsyn_name][gid]

    comm0.Free()
    gc.collect()

    if rank == 0:
        logger.info(
            f"Rank {rank}: took {(time.time() - start_time):.02f} s to configure all synapses"
        )


def find_gid_pop(
    celltypes: Dict[
        str,
        Dict[
            str,
            Union[
                str,
                Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]],
                int,
                Dict[
                    str,
                    Dict[
                        str,
                        Union[
                            Dict[str, Dict[str, float]],
                            Dict[str, Dict[str, int]],
                        ],
                    ],
                ],
                Dict[str, str],
            ],
        ],
    ],
    gid: int,
) -> str:
    """
    Given a celltypes structure and a gid, find the population to which the gid belongs.
    """
    for pop_name in celltypes:
        start = celltypes[pop_name]["start"]
        num = celltypes[pop_name]["num"]
        if (start <= gid) and (gid < (start + num)):
            return pop_name

    return None


def connect_cell_selection(env):
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    synapse and network connection mechanisms for the selected postsynaptic cells.

    :param env: an instance of the `miv_simulator.Env` class
    """
    connectivity_file_path = env.connectivity_file_path
    forest_file_path = env.forest_file_path
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    syn_attrs = env.synapse_attributes

    if rank == 0:
        logger.info(f"*** Connectivity file path is {connectivity_file_path}")
        logger.info("*** Reading projections: ")

    selection_pop_names = sorted(env.cell_selection.keys())
    biophys_cell_count = 0
    for postsyn_name in sorted(env.projection_dict.keys()):
        if rank == 0:
            logger.info(f"*** Postsynaptic population: {postsyn_name}")

        if postsyn_name not in selection_pop_names:
            continue

        presyn_names = sorted(env.projection_dict[postsyn_name])

        gid_range = [
            gid
            for gid in env.cell_selection[postsyn_name]
            if env.pc.gid_exists(gid)
        ]

        synapse_config = env.celltypes[postsyn_name]["synapses"]
        if "correct_for_spines" in synapse_config:
            correct_for_spines = synapse_config["correct_for_spines"]
        else:
            correct_for_spines = False

        if "unique" in synapse_config:
            unique = synapse_config["unique"]
        else:
            unique = False

        weight_dicts = []
        has_weights = False
        if "weights" in synapse_config:
            has_weights = True
            weight_dicts = synapse_config["weights"]

        if rank == 0:
            logger.info(
                f"*** Reading synaptic attributes of population {postsyn_name}"
            )

        syn_attrs_iter, syn_attrs_info = read_cell_attribute_selection(
            forest_file_path,
            postsyn_name,
            selection=gid_range,
            namespace="Synapse Attributes",
            comm=env.comm,
            mask={
                "syn_ids",
                "syn_locs",
                "syn_secs",
                "syn_layers",
                "syn_types",
                "swc_types",
            },
            return_type="tuple",
        )

        syn_attrs.init_syn_id_attrs_from_iter(
            syn_attrs_iter, attr_type="tuple", attr_tuple_index=syn_attrs_info
        )
        del syn_attrs_iter

        weight_attr_mask = list(syn_attrs.syn_mech_names)
        weight_attr_mask.append("syn_id")

        if has_weights:
            for weight_dict in weight_dicts:
                expr_closure = weight_dict.get("closure", None)
                weights_namespaces = weight_dict["namespace"]

                if rank == 0:
                    logger.info(
                        f"*** Reading synaptic weights of population {postsyn_name} from namespaces {weights_namespaces}"
                    )
                append_weights = False
                multiple_weights = "error"

                for weights_namespace in weights_namespaces:
                    (
                        syn_weights_iter,
                        weight_attr_info,
                    ) = read_cell_attribute_selection(
                        forest_file_path,
                        postsyn_name,
                        selection=gid_range,
                        mask=set(weight_attr_mask),
                        namespace=weights_namespace,
                        comm=env.comm,
                        return_type="tuple",
                    )

                    first_gid = None
                    syn_id_index = weight_attr_info.get("syn_id", None)
                    syn_name_inds = [
                        (syn_name, attr_index)
                        for syn_name, attr_index in sorted(
                            weight_attr_info.items()
                        )
                        if syn_name != "syn_id"
                    ]

                    for gid, cell_weights_tuple in syn_weights_iter:
                        if first_gid is None:
                            first_gid = gid
                        weights_syn_ids = cell_weights_tuple[syn_id_index]
                        for syn_name, syn_name_index in syn_name_inds:
                            if syn_name not in syn_attrs.syn_mech_names:
                                if rank == 0 and first_gid == gid:
                                    logger.warning(
                                        f"*** connect_cells: population: {postsyn_name}; gid: {gid}; syn_name: {syn_name} "
                                        "not found in network configuration"
                                    )
                            else:
                                weights_values = cell_weights_tuple[
                                    syn_name_index
                                ]
                                syn_attrs.add_mech_attrs_from_iter(
                                    gid,
                                    syn_name,
                                    zip_longest(
                                        weights_syn_ids,
                                        [
                                            {
                                                "weight": Promise(
                                                    expr_closure, [x]
                                                )
                                            }
                                            for x in weights_values
                                        ]
                                        if expr_closure
                                        else [
                                            {"weight": x}
                                            for x in weights_values
                                        ],
                                    ),
                                    multiple=multiple_weights,
                                    append=append_weights,
                                )

                                if rank == 0 and gid == first_gid:
                                    logger.info(
                                        f"*** connect_cells: population: {postsyn_name}; gid: {gid}; "
                                        f"found {len(weights_values)} {syn_name} synaptic weights ({weights_namespace})"
                                    )
                multiple_weights = "overwrite"
                append_weights = True
                del syn_weights_iter

        (graph, a) = read_graph_selection(
            connectivity_file_path,
            selection=gid_range,
            projections=[
                (presyn_name, postsyn_name)
                for presyn_name in sorted(presyn_names)
            ],
            comm=env.comm,
            namespaces=["Synapses", "Connections"],
        )

        env.edge_count[postsyn_name] = 0
        if postsyn_name in graph:
            for presyn_name in presyn_names:
                logger.info(f"*** Connecting {presyn_name} -> {postsyn_name}")

                edge_iter = graph[postsyn_name][presyn_name]
                presyn_input_sources = env.microcircuit_input_sources.get(
                    presyn_name, set()
                )
                syn_edge_iter = compose_iter(
                    lambda edgeset: presyn_input_sources.update(edgeset[1][0]),
                    edge_iter,
                )
                syn_attrs.init_edge_attrs_from_iter(
                    postsyn_name, presyn_name, a, syn_edge_iter
                )
                env.microcircuit_input_sources[
                    presyn_name
                ] = presyn_input_sources
                del graph[postsyn_name][presyn_name]

        first_gid = None
        if postsyn_name in env.biophys_cells:
            biophys_cell_count += len(env.biophys_cells[postsyn_name])
            for gid in env.biophys_cells[postsyn_name]:
                if env.node_allocation is not None:
                    assert gid in env.node_allocation
                if first_gid is None:
                    first_gid = gid
                try:
                    if syn_attrs.has_gid(gid):
                        biophys_cell = env.biophys_cells[postsyn_name][gid]
                        cells.init_biophysics(
                            biophys_cell,
                            reset_cable=True,
                            correct_cm=correct_for_spines,
                            correct_g_pas=correct_for_spines,
                            env=env,
                            verbose=((rank == 0) and (first_gid == gid)),
                        )
                        synapses.init_syn_mech_attrs(biophys_cell, env)
                except KeyError:
                    raise KeyError(
                        f"connect_cells: population: {postsyn_name}; gid: {gid}; could not initialize biophysics"
                    )

    ##
    ## This section instantiates cells that are not part of the
    ## selection, but are presynaptic sources for cells that _are_
    ## part of the selection. It is necessary to create cells at this
    ## point, as NEURON's ParallelContext does not allow the creation
    ## of gids after netcons including those gids are created.
    ##
    make_input_cell_selection(env)

    ##
    ## This section instantiates the synaptic mechanisms and netcons for each connection.
    ##
    first_gid = None
    gids = list(syn_attrs.gids())
    assert len(gids) == biophys_cell_count

    for gid in gids:
        last_time = time.time()
        if first_gid is None:
            first_gid = gid

        cell = env.pc.gid2cell(gid)
        pop_name = find_gid_pop(env.celltypes, gid)

        syn_count, mech_count, nc_count = synapses.config_hoc_cell_syns(
            env,
            gid,
            pop_name,
            cell=cell.hoc_cell if hasattr(cell, "hoc_cell") else cell,
            unique=unique,
            insert=True,
            insert_netcons=True,
        )

        if rank == 0 and gid == first_gid:
            logger.info(
                f"Rank {rank}: took {time.time() - last_time:.02f} s to configure {syn_count} synapses, {mech_count} synaptic mechanisms, "
                f"{nc_count} network connections for gid {gid}; cleanup flag is {env.cleanup}"
            )
            hoc_cell = env.pc.gid2cell(gid)
            if hasattr(hoc_cell, "all"):
                for sec in list(hoc_cell.all):
                    logger.info(pprint.pformat(sec.psection()))

        if gid in env.recording_sets.get(pop_name, {}):
            cells.record_cell(env, pop_name, gid)

        env.edge_count[pop_name] += syn_count
        if env.cleanup:
            syn_attrs.del_syn_id_attr_dict(gid)
            if gid in env.biophys_cells[pop_name]:
                del env.biophys_cells[pop_name][gid]


def connect_gjs(env: Env) -> None:
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    half-gap mechanisms on the pre- and post-junction cells.

    :param env: an instance of the `miv_simulator.Env` class

    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    dataset_path = os.path.join(env.dataset_prefix, env.datasetName)

    gapjunctions = env.gapjunctions
    gapjunctions_file_path = env.gapjunctions_file_path

    num_gj = 0
    num_gj_intra = 0
    num_gj_inter = 0
    if gapjunctions_file_path is not None:
        (graph, a) = bcast_graph(
            gapjunctions_file_path,
            namespaces=["Coupling strength", "Location"],
            comm=env.comm,
        )

        ggid = 2e6
        for name in sorted(gapjunctions.keys()):
            if rank == 0:
                logger.info(f"*** Creating gap junctions {name}")
            prj = graph[name[0]][name[1]]
            attrmap = a[(name[1], name[0])]
            cc_src_idx = attrmap["Coupling strength"]["Source"]
            cc_dst_idx = attrmap["Coupling strength"]["Destination"]
            dstsec_idx = attrmap["Location"]["Destination section"]
            dstpos_idx = attrmap["Location"]["Destination position"]
            srcsec_idx = attrmap["Location"]["Source section"]
            srcpos_idx = attrmap["Location"]["Source position"]

            for src in sorted(prj.keys()):
                edges = prj[src]
                destinations = edges[0]
                cc_dict = edges[1]["Coupling strength"]
                loc_dict = edges[1]["Location"]
                srcweights = cc_dict[cc_src_idx]
                dstweights = cc_dict[cc_dst_idx]
                dstposs = loc_dict[dstpos_idx]
                dstsecs = loc_dict[dstsec_idx]
                srcposs = loc_dict[srcpos_idx]
                srcsecs = loc_dict[srcsec_idx]
                for i in range(0, len(destinations)):
                    dst = destinations[i]
                    srcpos = srcposs[i]
                    srcsec = srcsecs[i]
                    dstpos = dstposs[i]
                    dstsec = dstsecs[i]
                    wgt = srcweights[i] * 0.001
                    if env.pc.gid_exists(src):
                        if rank == 0:
                            logger.info(
                                "host %d: gap junction: gid = %d sec = %d coupling = %g "
                                "sgid = %d dgid = %d\n"
                                % (rank, src, srcsec, wgt, ggid, ggid + 1)
                            )
                        cell = env.pc.gid2cell(src)
                        gj = neuron_utils.mkgap(
                            env, cell, src, srcpos, srcsec, ggid, ggid + 1, wgt
                        )
                    if env.pc.gid_exists(dst):
                        if rank == 0:
                            logger.info(
                                "host %d: gap junction: gid = %d sec = %d coupling = %g "
                                "sgid = %d dgid = %d\n"
                                % (rank, dst, dstsec, wgt, ggid + 1, ggid)
                            )
                        cell = env.pc.gid2cell(dst)
                        gj = neuron_utils.mkgap(
                            env, cell, dst, dstpos, dstsec, ggid + 1, ggid, wgt
                        )
                    ggid = ggid + 2
                    if env.pc.gid_exists(src) or env.pc.gid_exists(dst):
                        num_gj += 1
                        if env.pc.gid_exists(src) and env.pc.gid_exists(dst):
                            num_gj_intra += 1
                        else:
                            num_gj_inter += 1

            del graph[name[0]][name[1]]

        logger.info(
            f"*** rank {rank}: created total {num_gj} gap junctions: {num_gj_intra} intraprocessor {num_gj_inter} interprocessor"
        )


def make_cells(env: Env) -> None:
    """
    Instantiates cell templates according to population ranges and NeuroH5 morphology if present.

    :param env: an instance of the `miv_simulator.Env` class
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    recording_seed = int(
        env.model_config["Random Seeds"]["Intracellular Recording Sample"]
    )
    ranstream_recording = np.random.RandomState()
    ranstream_recording.seed(recording_seed)

    dataset_path = env.dataset_path
    data_file_path = env.data_file_path
    pop_names = sorted(env.celltypes.keys())

    if rank == 0:
        logger.info(
            f"Population attributes: {pprint.pformat(env.cell_attribute_info)}"
        )
    for pop_name in pop_names:
        if rank == 0:
            logger.info(f"*** Creating population {pop_name}")

        template_name = env.celltypes[pop_name].get("template", None)
        if template_name is None:
            continue

        template_name_lower = template_name.lower()
        if template_name_lower != "vecstim":
            neuron_utils.load_cell_template(env, pop_name, bcast_template=True)

        mech_dict = None
        mech_file_path = None
        if "mech_file_path" in env.celltypes[pop_name]:
            mech_dict = env.celltypes[pop_name]["mech_dict"]
            mech_file_path = env.celltypes[pop_name]["mech_file_path"]
            if rank == 0:
                logger.info(
                    f"*** Mechanism file for population {pop_name} is {mech_file_path}"
                )

        is_BRK = template_name.lower() == "brk_nrn"
        is_PR = template_name.lower() == "pr_nrn"
        is_SC = template_name.lower() == "sc_nrn"
        is_reduced = is_BRK or is_PR or is_SC

        num_cells = 0
        if (pop_name in env.cell_attribute_info) and (
            "Trees" in env.cell_attribute_info[pop_name]
        ):
            if rank == 0:
                logger.info(f"*** Reading trees for population {pop_name}")

            if env.node_allocation is None:
                (trees, forestSize) = scatter_read_trees(
                    data_file_path, pop_name, comm=env.comm, io_size=env.io_size
                )
            else:
                (trees, forestSize) = scatter_read_trees(
                    data_file_path,
                    pop_name,
                    comm=env.comm,
                    io_size=env.io_size,
                    node_allocation=env.node_allocation,
                )
            if rank == 0:
                logger.info(f"*** Done reading trees for population {pop_name}")

            first_gid = None
            for i, (gid, tree) in enumerate(trees):
                if rank == 0:
                    logger.info(f"*** Creating {pop_name} gid {gid}")

                if first_gid is None:
                    first_gid = gid

                if is_SC:
                    cell = cells.make_SC_cell(
                        gid=gid, pop_name=pop_name, env=env, mech_dict=mech_dict
                    )
                elif is_PR:
                    cell = cells.make_PR_cell(
                        gid=gid, pop_name=pop_name, env=env, mech_dict=mech_dict
                    )
                elif is_BRK:
                    cell = cells.make_BRK_cell(
                        gid=gid, pop_name=pop_name, env=env, mech_dict=mech_dict
                    )
                else:
                    hoc_cell = cells.make_hoc_cell(
                        env, pop_name, gid, neurotree_dict=tree
                    )
                    cell = cells.make_biophys_cell(
                        gid=gid,
                        population_name=pop_name,
                        hoc_cell=hoc_cell,
                        env=env,
                        tree_dict=tree,
                        mech_dict=mech_dict,
                    )
                    # cells.init_spike_detector(biophys_cell)
                    if (
                        rank == 0
                        and gid == first_gid
                        and mech_file_path is not None
                    ):
                        logger.info(
                            f"*** make_cells: population: {pop_name}; gid: {gid}; loaded biophysics from path: {mech_file_path}"
                        )

                if is_reduced:
                    soma_xyz = cells.get_soma_xyz(tree, env.SWC_Types)
                    cell.position(soma_xyz[0], soma_xyz[1], soma_xyz[2])
                if rank == 0 and first_gid == gid:
                    if hasattr(cell, "hoc_cell"):
                        hoc_cell = cell.hoc_cell
                        if hasattr(hoc_cell, "all"):
                            for sec in list(hoc_cell.all):
                                logger.info(pprint.pformat(sec.psection()))
                cells.register_cell(env, pop_name, gid, cell)
                num_cells += 1
            del trees

        elif (pop_name in env.cell_attribute_info) and (
            "Coordinates" in env.cell_attribute_info[pop_name]
        ):
            if rank == 0:
                logger.info(
                    f"*** Reading coordinates for population {pop_name}"
                )

            if env.node_allocation is None:
                cell_attr_dict = scatter_read_cell_attributes(
                    data_file_path,
                    pop_name,
                    namespaces=["Coordinates"],
                    comm=env.comm,
                    io_size=env.io_size,
                    return_type="tuple",
                )
            else:
                cell_attr_dict = scatter_read_cell_attributes(
                    data_file_path,
                    pop_name,
                    namespaces=["Coordinates"],
                    node_allocation=env.node_allocation,
                    comm=env.comm,
                    io_size=env.io_size,
                    return_type="tuple",
                )
            if rank == 0:
                logger.info(
                    f"*** Done reading coordinates for population {pop_name}"
                )

            coords_iter, coords_attr_info = cell_attr_dict["Coordinates"]

            x_index = coords_attr_info.get("X Coordinate", None)
            y_index = coords_attr_info.get("Y Coordinate", None)
            z_index = coords_attr_info.get("Z Coordinate", None)
            for i, (gid, cell_coords) in enumerate(coords_iter):
                if rank == 0:
                    logger.info(f"*** Creating {pop_name} gid {gid}")

                cell_x = cell_coords[x_index][0]
                cell_y = cell_coords[y_index][0]
                cell_z = cell_coords[z_index][0]

                cell = None
                if is_SC:
                    cell = cells.make_SC_cell(
                        gid=gid, pop_name=pop_name, env=env, mech_dict=mech_dict
                    )
                elif is_PR:
                    cell = cells.make_PR_cell(
                        gid=gid, pop_name=pop_name, env=env, mech_dict=mech_dict
                    )
                elif is_BRK:
                    cell = cells.make_BRK_cell(
                        gid=gid, pop_name=pop_name, env=env, mech_dict=mech_dict
                    )
                else:
                    cell = cells.make_hoc_cell(env, pop_name, gid)
                cell.position(cell_x, cell_y, cell_z)
                cells.register_cell(env, pop_name, gid, cell)
                num_cells += 1
        else:
            raise RuntimeError(
                f"make_cells: unknown cell configuration type for cell type {pop_name}"
            )

        h.define_shape()

        recording_set = set()
        pop_biophys_gids = list(env.biophys_cells[pop_name].keys())
        pop_biophys_gids_per_rank = env.comm.gather(pop_biophys_gids, root=0)
        if rank == 0:
            if env.recording_profile is not None:
                recording_fraction = env.recording_profile.get("fraction", 1.0)
                recording_limit = env.recording_profile.get("limit", -1)
                all_pop_biophys_gids = sorted(
                    item
                    for sublist in pop_biophys_gids_per_rank
                    for item in sublist
                )
                for gid in all_pop_biophys_gids:
                    if ranstream_recording.uniform() <= recording_fraction:
                        recording_set.add(gid)
                    if (recording_limit > 0) and (
                        len(recording_set) > recording_limit
                    ):
                        break
                logger.info(f"recording_set = {recording_set}")
        recording_set = env.comm.bcast(recording_set, root=0)
        env.recording_sets[pop_name] = recording_set
        del pop_biophys_gids_per_rank
        logger.info(
            f"*** Rank {rank}: Created {num_cells} cells from population {pop_name}"
        )

    # if node rank map has not been created yet, create it now
    if env.node_allocation is None:
        env.node_allocation = set()
        for gid in env.gidset:
            env.node_allocation.add(gid)


def make_cell_selection(env):
    """
    Instantiates cell templates for the selected cells according to
    population ranges and NeuroH5 morphology if present.

    :param env: an instance of the `miv_simulator.Env` class
    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    dataset_path = env.dataset_path
    data_file_path = env.data_file_path

    pop_names = sorted(env.cell_selection.keys())

    for pop_name in pop_names:
        if rank == 0:
            logger.info(
                f"*** Creating selected cells from population {pop_name}"
            )

        template_name = env.celltypes[pop_name]["template"]
        template_name_lower = template_name.lower()
        if template_name_lower != "vecstim":
            neuron_utils.load_cell_template(env, pop_name, bcast_template=True)

        templateClass = getattr(h, env.celltypes[pop_name]["template"])

        gid_range = [
            gid for gid in env.cell_selection[pop_name] if gid % nhosts == rank
        ]

        if "mech_file_path" in env.celltypes[pop_name]:
            mech_dict = env.celltypes[pop_name]["mech_dict"]
        else:
            mech_dict = None

        is_BRK = template_name.lower() == "brk_nrn"
        is_PR = template_name.lower() == "pr_nrn"
        is_SC = template_name.lower() == "sc_nrn"
        is_reduced = is_BRK or is_PR or is_SC

        num_cells = 0
        if (pop_name in env.cell_attribute_info) and (
            "Trees" in env.cell_attribute_info[pop_name]
        ):
            if rank == 0:
                logger.info(f"*** Reading trees for population {pop_name}")

            (trees, _) = read_tree_selection(
                data_file_path, pop_name, gid_range, comm=env.comm
            )
            if rank == 0:
                logger.info(f"*** Done reading trees for population {pop_name}")

            first_gid = None
            cell = None
            for i, (gid, tree) in enumerate(trees):
                if rank == 0:
                    logger.info(f"*** Creating {pop_name} gid {gid}")
                if first_gid == None:
                    first_gid = gid

                if is_SC:
                    cell = cells.make_SC_cell(
                        gid=gid,
                        pop_name=pop_name,
                        env=env,
                        param_dict=mech_dict,
                    )
                elif is_PR:
                    cell = cells.make_PR_cell(
                        gid=gid,
                        pop_name=pop_name,
                        env=env,
                        param_dict=mech_dict,
                    )
                elif is_BRK:
                    cell = cells.make_BRK_cell(
                        gid=gid,
                        pop_name=pop_name,
                        env=env,
                        param_dict=mech_dict,
                    )
                else:
                    hoc_cell = cells.make_hoc_cell(
                        env, pop_name, gid, neurotree_dict=tree
                    )
                    if mech_file_path is None:
                        cell = hoc_cell
                    else:
                        cell = cells.make_biophys_cell(
                            gid=gid,
                            pop_name=pop_name,
                            hoc_cell=hoc_cell,
                            env=env,
                            tree_dict=tree,
                            mech_dict=mech_dict,
                        )
                        # cells.init_spike_detector(biophys_cell)
                        if rank == 0 and gid == first_gid:
                            logger.info(
                                f"*** make_cell_selection: population: {pop_name}; gid: {gid}; loaded biophysics from path: {mech_file_path}"
                            )

                if is_reduced:
                    soma_xyz = cells.get_soma_xyz(tree, env.SWC_Types)
                    cell.position(soma_xyz[0], soma_xyz[1], soma_xyz[2])

                if rank == 0 and first_gid == gid:
                    if hasattr(cell, "hoc_cell"):
                        hoc_cell = cell.hoc_cell
                        if hasattr(hoc_cell, "all"):
                            for sec in list(hoc_cell.all):
                                logger.info(pprint.pformat(sec.psection()))
                cells.register_cell(env, pop_name, gid, cell)

                num_cells += 1

        elif (pop_name in env.cell_attribute_info) and (
            "Coordinates" in env.cell_attribute_info[pop_name]
        ):
            if rank == 0:
                logger.info(
                    f"*** Reading coordinates for population {pop_name}"
                )

            coords_iter, coords_attr_info = read_cell_attribute_selection(
                data_file_path,
                pop_name,
                selection=gid_range,
                namespace="Coordinates",
                comm=env.comm,
                return_type="tuple",
            )
            x_index = coords_attr_info.get("X Coordinate", None)
            y_index = coords_attr_info.get("Y Coordinate", None)
            z_index = coords_attr_info.get("Z Coordinate", None)

            if rank == 0:
                logger.info(
                    f"*** Done reading coordinates for population {pop_name}"
                )

            for i, (gid, cell_coords_tuple) in enumerate(coords_iter):
                if rank == 0:
                    logger.info(f"*** Creating {pop_name} gid {gid}")

                cell = None
                if is_SC:
                    cell = cells.make_SC_cell(
                        gid=gid,
                        pop_name=pop_name,
                        env=env,
                        param_dict=mech_dict,
                    )
                    cells.register_cell(env, pop_name, gid, SC_cell)
                elif is_PR:
                    cell = cells.make_PR_cell(
                        gid=gid,
                        pop_name=pop_name,
                        env=env,
                        param_dict=mech_dict,
                    )
                    cells.register_cell(env, pop_name, gid, PR_cell)
                elif is_BRK:
                    cell = cells.make_BRK_cell(
                        gid=gid,
                        pop_name=pop_name,
                        env=env,
                        param_dict=mech_dict,
                    )
                else:
                    hoc_cell = cells.make_hoc_cell(env, pop_name, gid)
                    if mech_file_path is None:
                        cell = hoc_cell
                    else:
                        cell = cells.make_biophys_cell(
                            gid=gid,
                            pop_name=pop_name,
                            hoc_cell=hoc_cell,
                            env=env,
                            tree_dict=tree,
                            mech_dict=mech_dict,
                        )

                cell_x = cell_coords_tuple[x_index][0]
                cell_y = cell_coords_tuple[y_index][0]
                cell_z = cell_coords_tuple[z_index][0]
                hoc_cell.position(cell_x, cell_y, cell_z)
                cells.register_cell(env, pop_name, gid, cell)
                num_cells += 1

        h.define_shape()
        logger.info(
            f"*** Rank {rank}: Created {num_cells} cells from population {pop_name}"
        )

    if env.node_allocation is None:
        env.node_allocation = set()
        for gid in env.gidset:
            env.node_allocation.add(gid)


def make_input_cell_selection(env):
    """
    Creates cells with predefined spike patterns when only a subset of the network is instantiated.

    :param env: an instance of the `miv_simulator.Env` class
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    created_input_sources = {
        pop_name: set() for pop_name in env.celltypes.keys()
    }
    for pop_name, input_gid_range in sorted(
        env.microcircuit_input_sources.items()
    ):
        pop_index = int(env.Populations[pop_name])

        has_spike_train = False
        if (env.spike_input_attribute_info is not None) and (
            env.spike_input_ns is not None
        ):
            if (pop_name in env.spike_input_attribute_info) and (
                env.spike_input_ns in env.spike_input_attribute_info[pop_name]
            ):
                has_spike_train = True

        if has_spike_train:
            spike_generator = None
        else:
            if env.netclamp_config is None:
                logger.warning(
                    f"make_input_cell_selection: population {pop_name} has neither input spike trains nor input generator configuration"
                )
                spike_generator = None
            else:
                if pop_name in env.netclamp_config.input_generators:
                    spike_generator = env.netclamp_config.input_generators[
                        pop_name
                    ]
                else:
                    raise RuntimeError(
                        f"make_input_cell_selection: population {pop_name} has neither input spike trains nor input generator configuration"
                    )

        if spike_generator is not None:
            input_source_dict = {pop_index: {"generator": spike_generator}}
        else:
            input_source_dict = {pop_index: {"spiketrains": {}}}

        if (env.cell_selection is not None) and (
            pop_name in env.cell_selection
        ):
            local_input_gid_range = input_gid_range.difference(
                set(env.cell_selection[pop_name])
            )
        else:
            local_input_gid_range = input_gid_range
        input_gid_ranges = env.comm.allreduce(
            local_input_gid_range, op=mpi_op_set_union
        )

        created_input_gids = []
        for i, gid in enumerate(input_gid_ranges):
            if (i % nhosts == rank) and not env.pc.gid_exists(gid):
                input_cell = cells.make_input_cell(
                    env, gid, pop_index, input_source_dict
                )
                cells.register_cell(env, pop_name, gid, input_cell)
                created_input_gids.append(gid)
        created_input_sources[pop_name] = set(created_input_gids)

        if rank == 0:
            logger.info(
                f"*** Rank {rank}: created {pop_name} input sources for gids {created_input_gids}"
            )

    env.microcircuit_input_sources = created_input_sources

    if env.node_allocation is None:
        env.node_allocation = set()
    for _, this_gidset in env.microcircuit_input_sources.items():
        for gid in this_gidset:
            env.node_allocation.add(gid)


def merge_spiketrain_trials(
    spiketrain: ndarray,
    trial_index: ndarray,
    trial_duration: ndarray,
    n_trials: int,
) -> ndarray:
    if (trial_index is not None) and (trial_duration is not None):
        trial_spiketrains = []
        for trial_i in range(n_trials):
            trial_spiketrain_i = spiketrain[np.where(trial_index == trial_i)[0]]
            trial_spiketrain_i += np.sum(trial_duration[:trial_i])
            trial_spiketrains.append(trial_spiketrain_i)
        spiketrain = np.concatenate(trial_spiketrains)
    spiketrain.sort()
    return spiketrain


def init_input_cells(env: Env) -> None:
    """
    Initializes cells with predefined spike patterns.

    :param env: an instance of the `miv_simulator.Env` class
    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    if rank == 0:
        logger.info(f"*** Stimulus onset is {env.stimulus_onset} ms")

    dataset_path = env.dataset_path
    input_file_path = env.data_file_path

    pop_names = sorted(env.celltypes.keys())

    trial_index_attr = "Trial Index"
    trial_dur_attr = "Trial Duration"
    for pop_name in pop_names:
        if "spike train" in env.celltypes[pop_name]:
            if env.arena_id and env.stimulus_id:
                vecstim_namespace = f"{env.celltypes[pop_name]['spike train']['namespace']} {env.arena_id} {env.stimulus_id}"
            else:
                vecstim_namespace = env.celltypes[pop_name]["spike train"][
                    "namespace"
                ]
            vecstim_attr = env.celltypes[pop_name]["spike train"]["attribute"]

            has_vecstim = False
            vecstim_source_loc = []
            if (env.spike_input_attribute_info is not None) and (
                env.spike_input_ns is not None
            ):
                if (pop_name in env.spike_input_attribute_info) and (
                    env.spike_input_ns
                    in env.spike_input_attribute_info[pop_name]
                ):
                    has_vecstim = True
                    vecstim_source_loc.append(
                        (
                            env.spike_input_path,
                            env.spike_input_ns,
                            env.spike_input_attr,
                        )
                    )
            if (env.cell_attribute_info is not None) and (
                vecstim_namespace is not None
            ):
                if (pop_name in env.cell_attribute_info) and (
                    vecstim_namespace in env.cell_attribute_info[pop_name]
                ):
                    has_vecstim = True
                    vecstim_source_loc.append(
                        (input_file_path, vecstim_namespace, vecstim_attr)
                    )

            if has_vecstim:
                for input_path, input_ns, input_attr in vecstim_source_loc:
                    if rank == 0:
                        logger.info(
                            f"*** Initializing stimulus population {pop_name} from input path {input_path} namespace {vecstim_namespace}"
                        )

                    if env.cell_selection is None:
                        if env.node_allocation is None:
                            cell_vecstim_dict = scatter_read_cell_attributes(
                                input_path,
                                pop_name,
                                namespaces=[input_ns],
                                mask={
                                    input_attr,
                                    vecstim_attr,
                                    trial_index_attr,
                                    trial_dur_attr,
                                },
                                comm=env.comm,
                                io_size=env.io_size,
                                return_type="tuple",
                            )
                        else:
                            cell_vecstim_dict = scatter_read_cell_attributes(
                                input_path,
                                pop_name,
                                namespaces=[input_ns],
                                node_allocation=env.node_allocation,
                                mask={
                                    input_attr,
                                    vecstim_attr,
                                    trial_index_attr,
                                    trial_dur_attr,
                                },
                                comm=env.comm,
                                io_size=env.io_size,
                                return_type="tuple",
                            )

                        vecstim_iter, vecstim_attr_info = cell_vecstim_dict[
                            input_ns
                        ]
                    else:
                        if pop_name in env.cell_selection:
                            gid_range = [
                                gid
                                for gid in env.cell_selection[pop_name]
                                if env.pc.gid_exists(gid)
                            ]
                            (
                                vecstim_iter,
                                vecstim_attr_info,
                            ) = scatter_read_cell_attribute_selection(
                                input_path,
                                pop_name,
                                gid_range,
                                namespace=input_ns,
                                selection=list(gid_range),
                                mask={
                                    input_attr,
                                    vecstim_attr,
                                    trial_index_attr,
                                    trial_dur_attr,
                                },
                                comm=env.comm,
                                io_size=env.io_size,
                                return_type="tuple",
                            )
                        else:
                            vecstim_iter = []

                    vecstim_attr_index = vecstim_attr_info.get(
                        vecstim_attr, None
                    )
                    trial_index_attr_index = vecstim_attr_info.get(
                        trial_index_attr, None
                    )
                    trial_dur_attr_index = vecstim_attr_info.get(
                        trial_dur_attr, None
                    )
                    for gid, vecstim_tuple in vecstim_iter:
                        if not (env.pc.gid_exists(gid)):
                            continue

                        cell = env.artificial_cells[pop_name][gid]

                        spiketrain = vecstim_tuple[vecstim_attr_index]
                        trial_duration = None
                        trial_index = None
                        if trial_index_attr_index is not None:
                            trial_index = vecstim_tuple[trial_index_attr_index]
                            trial_duration = vecstim_tuple[trial_dur_attr_index]
                        if len(spiketrain) > 0:
                            spiketrain = merge_spiketrain_trials(
                                spiketrain,
                                trial_index,
                                trial_duration,
                                env.n_trials,
                            )
                            spiketrain += (
                                float(
                                    env.stimulus_config[
                                        "Equilibration Duration"
                                    ]
                                )
                                + env.stimulus_onset
                            )
                            if len(spiketrain) > 0:
                                cell.play(
                                    h.Vector(spiketrain.astype(np.float64))
                                )
                                if rank == 0:
                                    logger.info(
                                        f"*** Spike train for {pop_name} gid {gid} is of length {len(spiketrain)} ({spiketrain[0]} : {spiketrain[-1]} ms)"
                                    )

    gc.collect()

    if env.microcircuit_inputs:
        for pop_name in sorted(env.microcircuit_input_sources.keys()):
            gid_range = env.microcircuit_input_sources.get(pop_name, set())

            if (env.cell_selection is not None) and (
                pop_name in env.cell_selection
            ):
                this_gid_range = gid_range.difference(
                    set(env.cell_selection[pop_name])
                )
            else:
                this_gid_range = gid_range

            has_spike_train = False
            spike_input_source_loc = []
            if (env.spike_input_attribute_info is not None) and (
                env.spike_input_ns is not None
            ):
                if (pop_name in env.spike_input_attribute_info) and (
                    env.spike_input_ns
                    in env.spike_input_attribute_info[pop_name]
                ):
                    has_spike_train = True
                    spike_input_source_loc.append(
                        (env.spike_input_path, env.spike_input_ns)
                    )
            if (env.cell_attribute_info is not None) and (
                env.spike_input_ns is not None
            ):
                if (pop_name in env.cell_attribute_info) and (
                    env.spike_input_ns in env.cell_attribute_info[pop_name]
                ):
                    has_spike_train = True
                    spike_input_source_loc.append(
                        (input_file_path, env.spike_input_ns)
                    )

            if rank == 0:
                logger.info(
                    f"*** Initializing input source {pop_name} from locations {spike_input_source_loc}"
                )
            if has_spike_train:
                vecstim_attr_set = {"t", trial_index_attr, trial_dur_attr}
                if env.spike_input_attr is not None:
                    vecstim_attr_set.add(env.spike_input_attr)
                if pop_name in env.celltypes:
                    if "spike train" in env.celltypes[pop_name]:
                        vecstim_attr_set.add(
                            env.celltypes[pop_name]["spike train"]["attribute"]
                        )
                cell_spikes_items = []
                for input_path, input_ns in spike_input_source_loc:
                    item = scatter_read_cell_attribute_selection(
                        input_path,
                        pop_name,
                        list(this_gid_range),
                        namespace=input_ns,
                        mask=vecstim_attr_set,
                        comm=env.comm,
                        io_size=env.io_size,
                        return_type="tuple",
                    )
                    cell_spikes_items.append(item)
                for (
                    cell_spikes_iter,
                    cell_spikes_attr_info,
                ) in cell_spikes_items:
                    if len(cell_spikes_attr_info) == 0:
                        continue
                    trial_index_attr_index = cell_spikes_attr_info.get(
                        trial_index_attr, None
                    )
                    trial_dur_attr_index = cell_spikes_attr_info.get(
                        trial_dur_attr, None
                    )
                    if (env.spike_input_attr is not None) and (
                        env.spike_input_attr in cell_spikes_attr_info
                    ):
                        spike_train_attr_index = cell_spikes_attr_info.get(
                            env.spike_input_attr, None
                        )
                    elif "t" in cell_spikes_attr_info.keys():
                        spike_train_attr_index = cell_spikes_attr_info.get(
                            "t", None
                        )
                    elif "Spike Train" in cell_spikes_attr_info.keys():
                        spike_train_attr_index = cell_spikes_attr_info.get(
                            "Spike Train", None
                        )
                    elif len(this_gid_range) > 0:
                        raise RuntimeError(
                            f"init_input_cells: unable to determine spike train attribute for population {pop_name} in spike input file {env.spike_input_path};"
                            f" namespace {env.spike_input_ns}; attr keys {list(cell_spikes_attr_info.keys())}"
                        )
                    for gid, cell_spikes_tuple in cell_spikes_iter:
                        if not (env.pc.gid_exists(gid)):
                            continue
                        if gid not in env.artificial_cells[pop_name]:
                            logger.info(
                                f"init_input_cells: Rank {rank}: env.artificial_cells[{pop_name}] = {env.artificial_cells[pop_name]} this_gid_range = {this_gid_range}"
                            )
                        input_cell = env.artificial_cells[pop_name][gid]

                        spiketrain = cell_spikes_tuple[spike_train_attr_index]
                        trial_index = None
                        trial_duration = None
                        if trial_index_attr_index is not None:
                            trial_index = vecstim_tuple[trial_index_attr_index]
                            trial_duration = vecstim_tuple[trial_dur_attr_index]
                        if len(spiketrain) > 0:
                            spiketrain = merge_spiketrain_trials(
                                spiketrain,
                                trial_index,
                                trial_duration,
                                env.n_trials,
                            )
                            spiketrain += (
                                float(
                                    env.stimulus_config[
                                        "Equilibration Duration"
                                    ]
                                )
                                + env.stimulus_onset
                            )
                            if len(spiketrain) > 0:
                                input_cell.play(
                                    h.Vector(spiketrain.astype(np.float64))
                                )
                                if rank == 0:
                                    logger.info(
                                        f"*** Spike train for {pop_name} gid {gid} is of length {len(spiketrain)} ({spiketrain[0]} : {spiketrain[-1]} ms)"
                                    )

            else:
                if rank == 0:
                    logger.warning(
                        f"No spike train data found for population {pop_name} in spike input file {env.spike_input_path}; "
                        f"namespace: {env.spike_input_ns}"
                    )

    gc.collect()


def init(env: Env) -> None:
    """
    Initializes the network by calling make_cells, init_input_cells, connect_cells, connect_gjs.
    If env.optldbal or env.optlptbal are specified, performs load balancing.

    :param env: an instance of the `miv_simulator.Env` class
    """
    neuron_utils.configure_hoc_env(env)

    assert env.data_file_path
    assert env.connectivity_file_path
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    if env.optldbal or env.optlptbal:
        lb = h.LoadBalance()
        if not os.path.isfile("mcomplex.dat"):
            lb.ExperimentalMechComplex()

    if rank == 0:
        logger.info("*** Creating cells...")
    st = time.time()

    if env.cell_selection is None:
        make_cells(env)
    else:
        make_cell_selection(env)
    if env.profile_memory and rank == 0:
        profile_memory(logger)

    env.mkcellstime = time.time() - st
    if rank == 0:
        logger.info(f"*** Cells created in {env.mkcellstime:.02f} s")
    local_num_cells = imapreduce(
        env.cells.items(), lambda kv: len(kv[1]), lambda ax, x: ax + x
    )
    logger.info(f"*** Rank {rank} created {local_num_cells} cells")
    if env.cell_selection is None:
        st = time.time()
        connect_gjs(env)
        env.pc.setup_transfer()
        env.connectgjstime = time.time() - st
        if rank == 0:
            logger.info(
                f"*** Gap junctions created in {env.connectgjstime:.02f} s"
            )

    if env.opsin_config is not None:
        st = time.time()
        opsin_pop_dict = {
            pop_name: set(env.cells[pop_name].keys()).difference(
                set(env.artificial_cells[pop_name].keys())
            )
            for pop_name in env.cells.keys()
        }
        rho_params = env.opsin_config["rho parameters"]
        protocol_params = env.opsin_config["protocol parameters"]
        env.opto_stim = OptoStim(
            env.pc,
            opsin_pop_dict,
            model_nstates=env.opsin_config["nstates"],
            opsin_type=env.opsin_config["opsin type"],
            protocol=env.opsin_config["protocol"],
            protocol_params=protocol_params,
            rho_params=rho_params,
            seed=int(env.model_config["Random Seeds"].get("Opsin", None)),
        )
        env.optotime = time.time() - st
        if rank == 0:
            logger.info(
                "*** Opsin configuration instantiated in {env.optotime:.02f} s"
            )

    if env.profile_memory and rank == 0:
        profile_memory(logger)

    st = time.time()
    if (not env.use_coreneuron) and (len(env.LFP_config) > 0):
        lfp_pop_dict = {
            pop_name: set(env.cells[pop_name].keys()).difference(
                set(env.artificial_cells[pop_name].keys())
            )
            for pop_name in env.cells.keys()
        }
        for lfp_label, lfp_config_dict in sorted(env.LFP_config.items()):
            env.lfp[lfp_label] = lfp.LFP(
                lfp_label,
                env.pc,
                lfp_pop_dict,
                lfp_config_dict["position"],
                rho=lfp_config_dict["rho"],
                dt_lfp=lfp_config_dict["dt"],
                fdst=lfp_config_dict["fraction"],
                maxEDist=lfp_config_dict["maxEDist"],
                seed=int(
                    env.model_config["Random Seeds"]["Local Field Potential"]
                ),
            )
        if rank == 0:
            logger.info("*** LFP objects instantiated")
    lfp_time = time.time() - st

    st = time.time()
    if rank == 0:
        logger.info(f"*** Creating connections: time = {st:.02f} s")
    if env.cell_selection is None:
        connect_cells(env)
    else:
        connect_cell_selection(env)
    env.pc.set_maxstep(10.0)

    env.connectcellstime = time.time() - st

    if rank == 0:
        logger.info(
            f"*** Done creating connections: time = {time.time():.02f} s"
        )
    if rank == 0:
        logger.info(f"*** Connections created in {env.connectcellstime:.02f} s")
    edge_count = int(sum(env.edge_count[dest] for dest in env.edge_count))
    logger.info(f"*** Rank {rank} created {edge_count} connections")
    if env.profile_memory and rank == 0:
        profile_memory(logger)

    st = time.time()
    init_input_cells(env)
    env.mkstimtime = time.time() - st
    if rank == 0:
        logger.info(f"*** Stimuli created in {env.mkstimtime:.02f} s")
    setup_time = (
        env.mkcellstime
        + env.mkstimtime
        + env.connectcellstime
        + env.connectgjstime
        + lfp_time
    )
    max_setup_time = env.pc.allreduce(setup_time, 2)  ## maximum value
    equilibration_duration = float(
        env.stimulus_config.get("Equilibration Duration", 0.0)
    )
    tstop = (env.tstop + equilibration_duration) * float(env.n_trials)
    if not env.use_coreneuron:
        env.simtime = simtime.SimTimeEvent(
            env.pc,
            tstop,
            env.max_walltime_hours,
            env.results_write_time,
            max_setup_time,
        )
    h.v_init = env.v_init
    h.stdinit()
    if env.optldbal or env.optlptbal:
        lpt.cx(env)
        ld_bal(env)
        if env.optlptbal:
            lpt_bal(env)
        h.cvode.cache_efficient(1)
        h.cvode.use_fast_imem(1)


def shutdown(env: Env):
    """
    Forces NEURON to make it delete its MPI communicator and shut down properly.

    TODO: This may no longer be required in more recent versions of neurons
    """
    env.pc.runworker()
    env.pc.done()
    h.quit()


def run(
    env: Env,
    output: bool = True,
    output_syn_spike_count: bool = False,
):
    """
    Runs network simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env: an instance of the `miv_simulator.Env` class
    :param output: if True, output spike and cell voltage trace data
    :param output_syn_spike_count: if True, output spike counts per pre-synaptic source for each gid
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    if output_syn_spike_count and env.cleanup:
        raise RuntimeError(
            "Unable to compute synapse spike counts when cleanup is True"
        )
    gc.collect()

    if rank == 0:
        if output:
            logger.info(f"Creating results file {env.results_file_path}")
            io_utils.mkout(env, env.results_file_path)

    if rank == 0:
        logger.info(
            f"*** Running simulation; recording profile is {pprint.pformat(env.recording_profile)}"
        )

    rec_dt = None
    if env.recording_profile is not None:
        rec_dt = env.recording_profile.get("dt", None)
    if rec_dt is None:
        env.t_rec.record(h._ref_t)
    else:
        env.t_rec.record(h._ref_t, rec_dt)

    env.t_rec.resize(0)
    env.t_vec.resize(0)
    env.id_vec.resize(0)

    h.t = env.tstart
    if env.simtime is not None:
        env.simtime.reset()
    h.secondorder = 2
    h.finitialize(env.v_init)
    h.finitialize(env.v_init)
    gc.collect()

    if rank == 0:
        logger.info("*** Completed finitialize")

    equilibration_duration = float(
        env.stimulus_config.get("Equilibration Duration", 0.0)
    )
    tstop = (env.tstop + equilibration_duration) * float(env.n_trials)

    if env.checkpoint_interval is not None:
        if env.checkpoint_interval > 1.0:
            tsegments = np.concatenate(
                (
                    np.arange(env.tstart, tstop, env.checkpoint_interval)[1:],
                    np.asarray([tstop]),
                )
            )
        else:
            raise RuntimeError("Invalid checkpoint interval length")
    else:
        tsegments = np.asarray([tstop])

    for tstop_i in tsegments:
        if (h.t + env.dt) > env.tstop:
            break
        elif tstop_i < env.tstop:
            h.tstop = tstop_i
        else:
            h.tstop = env.tstop
        if rank == 0:
            logger.info(f"*** Running simulation up to {h.tstop:.2f} ms")
        env.pc.timeout(env.nrn_timeout)
        env.pc.psolve(h.tstop)
        while h.t < h.tstop - h.dt / 2:
            env.pc.psolve(h.t + 1.0)
        if output:
            if rank == 0:
                logger.info(f"*** Writing spike data up to {h.t:.2f} ms")
            io_utils.spikeout(
                env,
                env.results_file_path,
                t_start=env.last_checkpoint,
                clear_data=env.checkpoint_clear_data,
            )
            if env.recording_profile is not None:
                if rank == 0:
                    logger.info(
                        f"*** Writing intracellular data up to {h.t:.2f} ms"
                    )
                io_utils.recsout(
                    env,
                    env.results_file_path,
                    t_start=env.last_checkpoint,
                    clear_data=env.checkpoint_clear_data,
                )
            env.last_checkpoint = h.t
        if env.simtime is not None:
            env.tstop = env.simtime.tstop
    if output_syn_spike_count:
        for pop_name in sorted(env.biophys_cells.keys()):
            presyn_names = sorted(env.projection_dict[pop_name])
            synapses.write_syn_spike_count(
                env,
                pop_name,
                env.results_file_path,
                filters={"sources": presyn_names},
                write_kwds={"io_size": env.io_size},
            )
    if rank == 0:
        logger.info("*** Simulation completed")

    if rank == 0 and output:
        io_utils.lfpout(env, env.results_file_path)
    if shutdown:
        del env.cells

    comptime = env.pc.step_time()
    cwtime = comptime + env.pc.step_wait()
    maxcw = env.pc.allreduce(cwtime, 2)
    meancomp = env.pc.allreduce(comptime, 1) / nhosts
    maxcomp = env.pc.allreduce(comptime, 2)

    gjtime = env.pc.vtransfer_time()

    gjvect = h.Vector()
    env.pc.allgather(gjtime, gjvect)
    meangj = gjvect.mean()
    maxgj = gjvect.max()

    summary = {
        "rank": rank,
        "cell_creation": env.mkcellstime,
        "cell_connection": env.connectcellstime,
        "gap_junctions": env.connectgjstime,
        "run_simulation": env.pc.step_time(),
        "spike_communication": env.pc.send_time(),
        "event_handling": env.pc.event_time(),
        "numerical_integration": env.pc.integ_time(),
        "voltage_transfer": gjtime,
        "load_balance": (meancomp / maxcw),
        "mean_voltage_transfer_time": meangj,
        "max_voltage_transfer_time": maxgj,
    }

    if rank == 0:
        logger.info(
            f"Execution time summary for host {rank}: \n"
            f"  created cells in {env.mkcellstime:.02f} s\n"
            f"  connected cells in {env.connectcellstime:.02f} s\n"
            f"  created gap junctions in {env.connectgjstime:.02f} s\n"
            f"  ran simulation in {comptime:.02f} s\n"
            f"  spike communication time: {env.pc.send_time():.02f} s\n"
            f"  event handling time: {env.pc.event_time():.02f} s\n"
            f"  numerical integration time: {env.pc.integ_time():.02f} s\n"
            f"  voltage transfer time: {gjtime:.02f} s\n"
        )
        if maxcw > 0:
            logger.info(f"Load balance = {(meancomp / maxcw):.02f}\n")
        if meangj > 0:
            logger.info(
                "Mean/max voltage transfer time: {meangj:.02f} / {maxgj:.02f} s\n"
            )
            for i in range(nhosts):
                logger.debug(
                    "Voltage transfer time on host {i}: {gjvect.x[i]:.02f} s\n"
                )

    return summary
