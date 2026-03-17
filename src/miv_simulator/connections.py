"""Classes and procedures related to neuronal connectivity generation."""

from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Union

import gc
import math
import pprint
import time
from collections import defaultdict

import numpy as np
from miv_simulator.env import SynapseConfig
from miv_simulator.utils import (
    get_module_logger,
    list_find_all,
    random_choice_w_replacement,
    random_clustered_shuffle,
)
from mpi4py.MPI import Intracomm
from neuroh5.io import NeuroH5CellAttrGen, append_graph
from numpy import int8, ndarray, uint8
from numpy.random.mtrand import RandomState
from scipy.stats import norm

## This logger will inherit its setting from its root logger,
## which is created in module env
logger = get_module_logger(__name__)


class ConnectionProbability:
    """An object of this class will instantiate functions that describe
    the connection probabilities for each presynaptic population. These
    functions can then be used to get the distribution of connection
    probabilities across all possible source neurons, given the soma
    coordinates of a destination (post-synaptic) neuron.
    """

    def __init__(
        self,
        destination_population: str,
        soma_coords: Dict[str, Dict[int, Tuple[float, float, float]]],
        soma_distances: Dict[str, Dict[int, Tuple[float, float]]],
        extents: Dict[str, Dict[str, Dict[str, List[Union[float, int]]]]],
    ) -> None:
        """
        Warning: This method does not produce an absolute probability.
        It must be normalized so that the total area (volume) under the distribution is 1 before sampling.
        :param destination_population: post-synaptic population name
        :param soma_distances: a dictionary that contains per-population dicts of u, v distances of cell somas
        :param extent: dict: {source: 'width': (tuple of float), 'offset': (tuple of float)}
        """
        self.destination_population = destination_population
        self.soma_coords = soma_coords
        self.soma_distances = soma_distances
        self.p_dist = defaultdict(dict)
        self.width = defaultdict(dict)
        self.offset = defaultdict(dict)
        self.scale_factor = defaultdict(dict)

        for source_population, layer_extents in extents.items():
            for layer, extents in layer_extents.items():
                extent_width = extents["width"]
                if "offset" in extents:
                    extent_offset = extents["offset"]
                else:
                    extent_offset = (0.0, 0.0)

                u_extent = (float(extent_width[0]) / 2.0) - float(extent_offset[0])
                v_extent = (float(extent_width[1]) / 2.0) - float(extent_offset[1])
                self.width[source_population][layer] = {
                    "u": u_extent,
                    "v": v_extent,
                }

                self.scale_factor[source_population][layer] = {
                    axis: self.width[source_population][layer][axis] / 3.0
                    for axis in self.width[source_population][layer]
                }

                if extent_offset is None:
                    self.offset[source_population][layer] = {"u": 0.0, "v": 0.0}
                else:
                    self.offset[source_population][layer] = {
                        "u": float(extent_offset[0]),
                        "v": float(extent_offset[1]),
                    }

                self.p_dist[source_population][layer] = (
                    lambda source_population, layer: np.vectorize(
                        lambda distance_u, distance_v: (
                            norm.pdf(
                                np.abs(distance_u)
                                - self.offset[source_population][layer]["u"],
                                scale=self.scale_factor[source_population][layer]["u"],
                            )
                            * norm.pdf(
                                np.abs(distance_v)
                                - self.offset[source_population][layer]["v"],
                                scale=self.scale_factor[source_population][layer]["v"],
                            )
                        ),
                        otypes=[float],
                    )
                )(source_population, layer)

                logger.info(
                    f"population {source_population}: layer: {layer}: \n"
                    f"u width: {self.width[source_population][layer]['u']}\n"
                    f"v width: {self.width[source_population][layer]['v']}\n"
                    f"u scale_factor: {self.scale_factor[source_population][layer]['u']}\n"
                    f"v scale_factor: {self.scale_factor[source_population][layer]['v']}\n"
                )

    def filter_by_distance(
        self, destination_gid: int, source_population: str, source_layer: int
    ) -> Tuple[float, float, ndarray, ndarray, ndarray, ndarray, ndarray]:
        """
        Given the id of a target neuron, returns the distances along u and v
        and the gids of source neurons whose axons potentially contact the target neuron.

        :param destination_gid: int
        :param source_population: string
        :return: tuple of array of int
        """
        destination_coords = self.soma_coords[self.destination_population][
            destination_gid
        ]
        source_coords = self.soma_coords[source_population]

        destination_distances = self.soma_distances[self.destination_population][
            destination_gid
        ]

        source_distances = self.soma_distances[source_population]

        destination_u, destination_v, destination_l = destination_coords
        destination_distance_u, destination_distance_v = destination_distances

        distance_u_lst = []
        distance_v_lst = []
        source_u_lst = []
        source_v_lst = []
        source_gid_lst = []

        if source_layer in self.width[source_population]:
            layer_key = source_layer
        elif "default" in self.width[source_population]:
            layer_key = "default"
        else:
            raise RuntimeError(
                f"connections.get_probability: gid {destination_gid}: missing configuration for {source_population} layer {source_layer}"
            )

        source_width = self.width[source_population][layer_key]
        source_offset = self.offset[source_population][layer_key]

        max_distance_u = source_width["u"] + source_offset["u"]
        max_distance_v = source_width["v"] + source_offset["v"]

        for source_gid, coords in source_coords.items():
            source_u, source_v, source_l = coords

            source_distance_u, source_distance_v = source_distances[source_gid]

            distance_u = abs(destination_distance_u - source_distance_u)
            distance_v = abs(destination_distance_v - source_distance_v)

            if ((max_distance_u - distance_u) > 0.0) and (
                (max_distance_v - distance_v) > 0.0
            ):
                source_u_lst.append(source_u)
                source_v_lst.append(source_v)
                distance_u_lst.append(distance_u)
                distance_v_lst.append(distance_v)
                source_gid_lst.append(source_gid)

        return (
            destination_u,
            destination_v,
            np.asarray(source_u_lst),
            np.asarray(source_v_lst),
            np.asarray(distance_u_lst),
            np.asarray(distance_v_lst),
            np.asarray(source_gid_lst, dtype=np.uint32),
        )

    def get_probability(
        self, destination_gid: int, source: str, source_layers: List[int]
    ) -> Dict[int, Tuple[ndarray, ndarray, ndarray, ndarray]]:
        """
        Given the soma coordinates of a destination neuron and a
        population source, return an array of connection probabilities
        and an array of corresponding source gids.

        :param destination_gid: int
        :param source: string
        :return: array of float, array of int

        """
        prob_dict = {}
        for layer in source_layers:
            (
                destination_u,
                destination_v,
                source_u,
                source_v,
                distance_u,
                distance_v,
                source_gid,
            ) = self.filter_by_distance(destination_gid, source, layer)
            if layer in self.p_dist[source]:
                layer_key = layer
            elif "default" in self.p_dist[source]:
                layer_key = "default"
            else:
                raise RuntimeError(
                    f"connections.get_probability: gid {destination_gid}: missing configuration for {source} layer {layer}"
                )
            p = self.p_dist[source][layer_key](distance_u, distance_v)
            psum = np.sum(p)
            assert (p >= 0.0).all() and (p <= 1.0).all()
            if psum > 0.0:
                pn = p / p.sum()
            else:
                pn = p
            prob_dict[layer] = (
                pn.ravel(),
                source_gid.ravel(),
                distance_u.ravel(),
                distance_v.ravel(),
            )
        return prob_dict


def choose_synapse_projection(
    ranstream_syn: RandomState,
    syn_layer: int8,
    swc_type: uint8,
    syn_type: uint8,
    population_dict: Dict[str, int],
    projection_synapse_dict: Dict[
        str, Tuple[int, List[int], List[int], List[float], int]
    ],
    log: bool = False,
) -> str:
    """
    Given a synapse projection, SWC synapse location, and synapse type,
    chooses a projection from the given projection dictionary based on
    1) whether the projection properties match the given synapse
    properties and 2) random choice between all the projections that
    satisfy the given criteria.

    :param ranstream_syn: random state object
    :param syn_layer: synapse layer
    :param swc_type: SWC location for synapse (soma, axon, apical, basal)
    :param syn_type: synapse type (excitatory, inhibitory, neuromodulatory)
    :param population_dict: mapping of population names to population indices
    :param projection_synapse_dict: mapping of projection names to a tuple of the form: <type, layers, swc sections, proportions>

    """
    ivd = {v: k for k, v in population_dict.items()}
    projection_lst = []
    projection_prob_lst = []
    for k, (
        syn_config_type,
        syn_config_layers,
        syn_config_sections,
        syn_config_proportions,
        syn_config_contacts,
    ) in projection_synapse_dict.items():
        if (syn_type == syn_config_type) and (swc_type in syn_config_sections):
            ord_indices = list_find_all(lambda x: x == swc_type, syn_config_sections)
            for ord_index in ord_indices:
                if syn_layer == syn_config_layers[ord_index]:
                    if k not in population_dict:
                        raise RuntimeError(
                            f"Key {k} is not in population_dict; "
                            f" population_dict is {population_dict}"
                        )
                    projection_lst.append(population_dict[k])
                    projection_prob_lst.append(syn_config_proportions[ord_index])
    if len(projection_lst) > 1:
        candidate_projections = np.asarray(projection_lst)
        candidate_probs = np.asarray(projection_prob_lst)
        projection = ranstream_syn.choice(candidate_projections, 1, p=candidate_probs)[
            0
        ]
    elif len(projection_lst) > 0:
        projection = projection_lst[0]
    else:
        projection = None

    if projection is None:
        logger.error(
            f"Projection is none for syn_type {syn_type}, syn_layer {syn_layer} swc_type {swc_type}\n"
            f"projection synapse dict: {pprint.pformat(projection_synapse_dict)}"
        )

    if projection is not None:
        return ivd[projection]
    else:
        return None


def _partition_synapses_to_projections(
    gid: int,
    ranstream_syn: RandomState,
    synapse_dict: Dict[str, ndarray],
    population_dict: Dict[str, int],
    projection_synapse_dict: Dict[
        str, Tuple[int, List[int], List[int], List[float], int]
    ],
    rank: int,
) -> Tuple[Dict[int, float], DefaultDict]:
    """
    Assign every synapse to a named projection by matching each synapse's
    (layer, swc_type, syn_type) triple against projection_synapse_dict, then
    randomly choosing among matching projections weighted by their proportions.

    Retries up to 10 times until every projection has at least one synapse.
    Validates the result and raises AssertionError if any projection is empty.

    :param gid: GID of the destination neuron (for log messages).
    :param ranstream_syn: RandomState for synapse-to-projection sampling.
    :param synapse_dict: dict with arrays 'syn_ids', 'syn_cdists', 'syn_types',
        'swc_types', 'syn_layers'.
    :param population_dict: mapping of population name -> population index.
    :param projection_synapse_dict: mapping of projection name ->
        (syn_type, layers, sections, proportions, contacts).
    :param rank: MPI rank (for log messages).
    :return: (syn_cdist_dict, synapse_prj_partition) where syn_cdist_dict maps
        syn_id -> cable distance and synapse_prj_partition is a nested
        defaultdict: projection -> layer -> [syn_id, ...].
    :raises AssertionError: if any projection ends up with an empty synapse list.
    """
    num_projections = len(projection_synapse_dict)
    source_populations = sorted(projection_synapse_dict)
    prj_pop_index = {population: i for (i, population) in enumerate(source_populations)}
    synapse_prj_counts = np.zeros((num_projections,))
    synapse_prj_partition = defaultdict(lambda: defaultdict(list))
    syn_cdist_dict: Dict[int, float] = {}
    maxit = 10
    it = 0

    while (np.count_nonzero(synapse_prj_counts) < num_projections) and (it < maxit):
        log_flag = it > 1
        if log_flag:
            logger.info(
                f"generate_synaptic_connections: gid {gid}: iteration {it}: "
                f"source_populations = {source_populations}\n"
                f"synapse_prj_counts = {synapse_prj_counts}\n"
                f"number of synapses in synapse_dict = {len(synapse_dict['syn_ids'])}"
            )
        synapse_prj_counts.fill(0)
        synapse_prj_partition.clear()
        for syn_id, syn_cdist, syn_type, swc_type, syn_layer in zip(
            synapse_dict["syn_ids"],
            synapse_dict["syn_cdists"],
            synapse_dict["syn_types"],
            synapse_dict["swc_types"],
            synapse_dict["syn_layers"],
        ):
            syn_cdist_dict[syn_id] = syn_cdist
            projection = choose_synapse_projection(
                ranstream_syn,
                syn_layer,
                swc_type,
                syn_type,
                population_dict,
                projection_synapse_dict,
                log=log_flag,
            )
            if log_flag:
                logger.info(
                    f"generate_synaptic_connections: {gid=}: "
                    f"{syn_id=} {syn_type=} {swc_type=} "
                    f"{syn_layer=} {projection=} "
                    f"{ranstream_syn=}"
                )
            log_flag = False
            assert projection is not None, (
                f"generate_synaptic_connections: {gid=}: {syn_id=} {syn_type=} {swc_type=} {syn_layer=} {projection=} {ranstream_syn=} {population_dict=} {projection_synapse_dict=}\n {synapse_dict['syn_types']=}"
            )
            synapse_prj_counts[prj_pop_index[projection]] += 1
            synapse_prj_partition[projection][syn_layer].append(syn_id)
        it += 1

    empty_projections = []
    for projection in projection_synapse_dict:
        logger.debug(
            f"Rank {rank}: gid {gid}: source {projection} has {len(synapse_prj_partition[projection])} synapses"
        )
        if not (len(synapse_prj_partition[projection]) > 0):
            empty_projections.append(projection)
    if len(empty_projections) > 0:
        logger.warning(
            f"Rank {rank}: gid {gid}: projections {empty_projections} have an empty synapse list; "
            f"swc types are {set(synapse_dict['swc_types'].flat)} layers are {set(synapse_dict['syn_layers'].flat)}"
        )
    assert len(empty_projections) == 0

    return syn_cdist_dict, synapse_prj_partition


def _select_layer_source_vertices(
    rank: int,
    destination_gid: int,
    projection: str,
    prj_layer: int,
    syn_ids: List[int],
    syn_config_contacts: int,
    syn_cdist_dict: Dict[int, float],
    source_probs: ndarray,
    source_gids: ndarray,
    distances_u: ndarray,
    distances_v: ndarray,
    ranstream_con: RandomState,
    cluster_seed: int,
    random_choice: Callable,
) -> Tuple[ndarray, List[int], ndarray]:
    """
    Select source GIDs for a single (projection, layer) group of synapses
    using distance-weighted probability sampling and spatial clustering.

    Synapses are sorted by cable distance, grouped into
    ceil(n_syn / contacts) sampling groups, and source GIDs are assigned
    via random_choice + random_clustered_shuffle.

    Returns empty arrays when source_gids is empty (no candidates in extent).

    :param rank: MPI rank (for log messages).
    :param destination_gid: GID of the post-synaptic neuron.
    :param projection: source population name (for log messages).
    :param prj_layer: layer index (for log messages).
    :param syn_ids: synapse IDs assigned to this projection+layer.
    :param syn_config_contacts: number of contacts per source neuron.
    :param syn_cdist_dict: mapping syn_id -> cable distance.
    :param source_probs: distance-weighted sampling probabilities.
    :param source_gids: candidate source GID array.
    :param distances_u: u-axis distances for each source GID.
    :param distances_v: v-axis distances for each source GID.
    :param ranstream_con: RandomState for source GID sampling.
    :param cluster_seed: seed for random_clustered_shuffle.
    :param random_choice: callable (ranstream, n, probs) -> count array.
    :return: (source_vertices, ordered_syn_ids, distances) — all empty when
        there are no candidate source GIDs.
    """
    if len(source_gids) == 0:
        return (
            np.asarray([], dtype=np.uint32),
            [],
            np.asarray([], dtype=np.float32),
        )

    distance_dict = {
        gid: du + dv for gid, du, dv in zip(source_gids, distances_u, distances_v)
    }
    ordered_syn_ids = sorted(syn_ids, key=lambda x: syn_cdist_dict[x])
    n_syn_groups = int(math.ceil(float(len(syn_ids)) / float(syn_config_contacts)))
    source_gid_counts = random_choice(ranstream_con, n_syn_groups, source_probs)

    if syn_config_contacts > 1:
        ncontacts = int(math.ceil(syn_config_contacts))
        for i in range(len(source_gid_counts)):
            if source_gid_counts[i] > 0:
                source_gid_counts[i] *= ncontacts

    if len(source_gid_counts) == 0:
        logger.warning(
            f"Rank {rank}: source vertices list is empty for gid: {destination_gid} "
            f"source: {projection} layer: {prj_layer} "
            f"source probs: {source_probs} distances_u: {distances_u} distances_v: {distances_v}"
        )

    source_vertices = np.asarray(
        random_clustered_shuffle(
            len(source_gids),
            source_gid_counts,
            center_ids=source_gids,
            cluster_std=2.0,
            random_seed=cluster_seed,
        ),
        dtype=np.uint32,
    )[: len(syn_ids)]
    assert len(source_vertices) == len(syn_ids)

    distances = np.asarray(
        [distance_dict[gid] for gid in source_vertices], dtype=np.float32
    ).reshape(-1)

    return source_vertices, ordered_syn_ids, distances


def _consolidate_projection_connections(
    rank: int,
    destination_gid: int,
    projection: str,
    prj_source_vertices: List[ndarray],
    prj_syn_ids: List[List[int]],
    prj_distances: List[ndarray],
    gid_dict: Dict,
) -> int:
    """
    Concatenate per-layer accumulator lists, write the result into
    gid_dict[destination_gid], and return the number of connections created.

    :param rank: MPI rank (for log/assert messages).
    :param destination_gid: GID of the post-synaptic neuron.
    :param projection: source population name (for log messages).
    :param prj_source_vertices: per-layer source GID arrays.
    :param prj_syn_ids: per-layer cable-distance-ordered synapse ID lists.
    :param prj_distances: per-layer source-to-destination distance arrays.
    :param gid_dict: connection_dict[projection], mutated in place.
    :return: number of connections written.
    :raises AssertionError: if the resulting source vertex array is empty.
    """
    sv_arr = (
        np.concatenate(prj_source_vertices)
        if prj_source_vertices
        else np.asarray([], dtype=np.uint32)
    )
    si_arr = (
        np.concatenate(prj_syn_ids) if prj_syn_ids else np.asarray([], dtype=np.uint32)
    )
    dist_arr = (
        np.concatenate(prj_distances)
        if prj_distances
        else np.asarray([], dtype=np.float32)
    )

    if len(sv_arr) == 0:
        logger.warning(
            f"Rank {rank}: source gid list is empty for gid: {destination_gid} source: {projection}"
        )
    assert len(sv_arr) > 0, (
        f"Rank {rank}: source gid list is empty. The cell density might be too small."
    )

    gid_dict[destination_gid] = (
        sv_arr,
        {
            "Synapses": {"syn_id": np.asarray(si_arr, dtype=np.uint32)},
            "Connections": {"distance": dist_arr},
        },
    )
    return len(sv_arr)


def generate_synaptic_connections(
    rank: int,
    gid: int,
    ranstream_syn: RandomState,
    ranstream_con: RandomState,
    cluster_seed: int,
    destination_gid: int,
    synapse_dict: Dict[str, ndarray],
    population_dict: Dict[str, int],
    projection_synapse_dict: Dict[
        str, Tuple[int, List[int], List[int], List[float], int]
    ],
    projection_prob_dict: Dict[
        str, Dict[int, Tuple[ndarray, ndarray, ndarray, ndarray]]
    ],
    connection_dict: DefaultDict[Any, Any],
    random_choice: Callable = random_choice_w_replacement,
    debug_flag: bool = False,
) -> int:
    """
    Given a set of synapses for a particular gid, projection
    configuration, projection and connection probability dictionaries,
    generates a set of possible connections for each synapse. The
    procedure first assigns each synapse to a projection, using the
    given proportions of each synapse type, and then chooses source
    gids for each synapse using the given projection probability
    dictionary.

    :param ranstream_syn: random stream for the synapse partitioning step
    :param ranstream_con: random stream for the choosing source gids step
    :param destination_gid: destination gid
    :param synapse_dict: synapse configurations, a dictionary with fields: 1) syn_ids (synapse ids) 2) syn_types (excitatory, inhibitory, etc).,
                        3) swc_types (SWC types(s) of synapse location in the neuronal morphological structure 3) syn_layers (synapse layer placement)
    :param population_dict: mapping of population names to population indices
    :param projection_synapse_dict: mapping of projection names to a tuple of the form: <syn_layer, swc_type, syn_type, syn_proportion>
    :param projection_prob_dict: mapping of presynaptic population names to sets of source probabilities and source gids
    :param connection_dict: output connection dictionary
    :param random_choice: random choice procedure (default uses np.ranstream.multinomial)

    """
    syn_cdist_dict, synapse_prj_partition = _partition_synapses_to_projections(
        gid=gid,
        ranstream_syn=ranstream_syn,
        synapse_dict=synapse_dict,
        population_dict=population_dict,
        projection_synapse_dict=projection_synapse_dict,
        rank=rank,
    )

    count = 0
    for projection, prj_layer_dict in synapse_prj_partition.items():
        (_, _, _, _, syn_config_contacts) = projection_synapse_dict[projection]
        gid_dict = connection_dict[projection]
        prj_source_vertices: List[ndarray] = []
        prj_syn_ids: List[List[int]] = []
        prj_distances: List[ndarray] = []

        for prj_layer, syn_ids in prj_layer_dict.items():
            source_probs, source_gids, distances_u, distances_v = projection_prob_dict[
                projection
            ][prj_layer]
            source_vertices, ordered_syn_ids, distances = _select_layer_source_vertices(
                rank=rank,
                destination_gid=destination_gid,
                projection=projection,
                prj_layer=prj_layer,
                syn_ids=syn_ids,
                syn_config_contacts=syn_config_contacts,
                syn_cdist_dict=syn_cdist_dict,
                source_probs=source_probs,
                source_gids=source_gids,
                distances_u=distances_u,
                distances_v=distances_v,
                ranstream_con=ranstream_con,
                cluster_seed=cluster_seed,
                random_choice=random_choice,
            )
            if len(source_vertices) > 0:
                prj_source_vertices.append(source_vertices)
                prj_syn_ids.append(ordered_syn_ids)
                prj_distances.append(distances)
                cluster_seed += 1

        count += _consolidate_projection_connections(
            rank=rank,
            destination_gid=destination_gid,
            projection=projection,
            prj_source_vertices=prj_source_vertices,
            prj_syn_ids=prj_syn_ids,
            prj_distances=prj_distances,
            gid_dict=gid_dict,
        )

    return count


def generate_uv_distance_connections(
    comm: Intracomm,
    population_dict: Dict[str, int],
    connection_config: Dict[str, Dict[str, SynapseConfig]],
    connection_prob: ConnectionProbability,
    forest_path: str,
    synapse_seed: int,
    connectivity_seed: int,
    cluster_seed: int,
    synapse_namespace: str,
    connectivity_namespace: str,
    connectivity_path: str,
    io_size: int,
    chunk_size: int,
    value_chunk_size: int,
    cache_size: int,
    write_size: int = 1,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    """
    Generates connectivity based on U, V distance-weighted probabilities.

    :param comm: mpi4py MPI communicator
    :param connection_config: connection configuration object (instance of env.ConnectionConfig)
    :param connection_prob: ConnectionProbability instance
    :param forest_path: location of file with neuronal trees and synapse information
    :param synapse_seed: random seed for synapse partitioning
    :param connectivity_seed: random seed for connectivity generation
    :param cluster_seed: random seed for determining connectivity clustering for repeated connections from the same source
    :param synapse_namespace: namespace of synapse properties
    :param connectivity_namespace: namespace of connectivity attributes
    :param io_size: number of I/O ranks to use for parallel connectivity append
    :param chunk_size: HDF5 chunk size for connectivity file (pointer and index datasets)
    :param value_chunk_size: HDF5 chunk size for connectivity file (value datasets)
    :param cache_size: how many cells to read ahead
    :param write_size: how many cells to write out at the same time
    """

    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info(f"{comm.size} ranks have been allocated")

    start_time = time.time()

    ranstream_syn = np.random.RandomState()
    ranstream_con = np.random.RandomState()

    destination_population = connection_prob.destination_population

    source_populations = sorted(connection_config[destination_population].keys())

    for source_population in source_populations:
        if rank == 0:
            logger.info(
                f"{source_population} -> {destination_population}: \n"
                f"{pprint.pformat(connection_config[destination_population][source_population])}"
            )

    projection_config = connection_config[destination_population]
    projection_synapse_dict = {
        source_population: (
            projection_config[source_population].type,
            projection_config[source_population].layers,
            projection_config[source_population].sections,
            projection_config[source_population].proportions,
            projection_config[source_population].contacts,
        )
        for source_population in source_populations
    }

    comm.barrier()

    it_count = 0
    total_count = 0
    gid_count = 0
    connection_dict = defaultdict(lambda: {})
    projection_dict = {}
    for destination_gid, synapse_dict in NeuroH5CellAttrGen(
        forest_path,
        destination_population,
        namespace=synapse_namespace,
        comm=comm,
        io_size=io_size,
        cache_size=cache_size,
    ):
        if destination_gid is None:
            logger.info(f"Rank {rank} destination gid is None")
        else:
            logger.info(
                f"Rank {rank} received attributes for destination: {destination_population}, gid: {destination_gid}"
            )

            ranstream_con.seed(destination_gid + connectivity_seed)
            ranstream_syn.seed(destination_gid + synapse_seed)
            last_gid_time = time.time()

            projection_prob_dict = {}

            for source_population in source_populations:
                source_layers = projection_config[source_population].layers
                projection_prob_dict[source_population] = (
                    connection_prob.get_probability(
                        destination_gid, source_population, source_layers
                    )
                )

                for layer, (
                    probs,
                    source_gids,
                    distances_u,
                    distances_v,
                ) in projection_prob_dict[source_population].items():
                    if len(distances_u) > 0:
                        max_u_distance = np.max(distances_u)
                        min_u_distance = np.min(distances_u)
                        if rank == 0:
                            logger.info(
                                f"Rank {rank} has {len(source_gids)} possible sources from population {source_population} "
                                f"for destination: {destination_population}, layer {layer}, gid: {destination_gid}; "
                                f"max U distance: {max_u_distance:.2f} min U distance: {min_u_distance:.2f}"
                            )
                    else:
                        logger.warning(
                            f"Rank {rank} has {len(source_gids)} possible sources from population {source_population} "
                            f"for destination: {destination_population}, layer {layer}, gid: {destination_gid}"
                        )

            count = generate_synaptic_connections(
                rank,
                destination_gid,
                ranstream_syn,
                ranstream_con,
                cluster_seed + destination_gid,
                destination_gid,
                synapse_dict,
                population_dict,
                projection_synapse_dict,
                projection_prob_dict,
                connection_dict,
                debug_flag=debug,
            )
            total_count += count

            logger.info(
                f"Rank {rank} took {time.time() - last_gid_time:.2f} s to compute {count} edges for destination: {destination_population}, gid: {destination_gid}"
            )

        if (write_size > 0) and (gid_count % write_size == 0):
            if len(connection_dict) > 0:
                projection_dict = {destination_population: connection_dict}
            else:
                projection_dict = {}
            if not dry_run:
                last_time = time.time()
                append_graph(
                    connectivity_path,
                    projection_dict,
                    io_size=io_size,
                    comm=comm,
                    chunk_size=value_chunk_size,
                )
                if rank == 0:
                    if connection_dict:
                        logger.info(
                            f"Appending connectivity for {len(connection_dict)} projections took {time.time() - last_time:.2f} s"
                        )
            projection_dict.clear()
            connection_dict.clear()
            gc.collect()

        gid_count += 1
        it_count += 1
        if (it_count > 1) and debug:
            break

    gc.collect()
    last_time = time.time()
    if len(connection_dict) > 0:
        projection_dict = {destination_population: connection_dict}
    else:
        projection_dict = {}
    if not dry_run:
        append_graph(
            connectivity_path,
            projection_dict,
            io_size=io_size,
            comm=comm,
            chunk_size=value_chunk_size,
        )
        if rank == 0:
            if connection_dict:
                logger.info(
                    f"Appending connectivity for {len(connection_dict)} projections took {time.time() - last_time:.2f} s"
                )

    global_count = comm.gather(total_count, root=0)
    if rank == 0:
        logger.info(
            f"{comm.size} ranks took {time.time() - start_time:.2f} s to generate {np.sum(global_count)} edges"
        )
