"""Procedures related to gap junction connectivity generation. """

import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

from miv_simulator import cells
from miv_simulator.utils.neuron import h, load_cell_template
from miv_simulator.utils import (
    get_module_logger,
)
from neuroh5.io import append_graph, read_tree_selection

## This logger will inherit its setting from its root logger,
## which is created in module env
logger = get_module_logger(__name__)


def filter_by_distance(gids_a, coords_a, gids_b, coords_b, bounds, params):
    coords_tree_a = cKDTree(coords_a)
    coords_tree_b = cKDTree(coords_b)

    res = coords_tree_a.query_ball_tree(coords_tree_b, bounds[1])

    res_dict = {}
    for i, nns in enumerate(res):
        gid_a = gids_a[i]
        nngids = []
        nndists = []
        nnprobs = []
        for nn in nns:
            nndist = euclidean(coords_a[i], coords_b[nn])
            if nndist > 0.0:
                nndists.append(nndist)
                nnprobs.append(np.polyval(params, nndist))
                nngids.append(gids_b[nn])
        if len(nngids) > 0:
            res_dict[gid_a] = (
                np.asarray(nngids, dtype=np.uint32),
                np.asarray(nndists, dtype=np.float32),
                np.asarray(nnprobs, dtype=np.float32),
            )

    return res_dict


def distance_to_root(root, sec, loc):
    """
    Returns the distance from the given section location to the middle of the given root section.
    """
    distance = 0.0
    if sec is root:
        return distance

    distance += loc * sec.L

    while h.SectionRef(sec=sec).has_parent == 1:
        sec = h.SectionRef(sec=sec).parent
        distance += sec.L
    distance -= 0.5 * sec.L

    return distance


def choose_gj_locations(ranstream_gj, cell_a, cell_b):
    apical_sections_a = cell_a.apicalidx.to_python()
    basal_sections_a = cell_a.basalidx.to_python()
    apical_sections_b = cell_b.apicalidx.to_python()
    basal_sections_b = cell_b.basalidx.to_python()

    if (
        (len(apical_sections_a) > 0)
        and (len(basal_sections_a) > 0)
        and (len(apical_sections_b) > 0)
        and (len(basal_sections_b) > 0)
    ):
        sec_type = ranstream_gj.random_sample()
        if sec_type > 0.5:
            sectionidx_a = int(
                apical_sections_a[ranstream_gj.randint(len(apical_sections_a))]
            )
            sectionidx_b = int(
                apical_sections_b[ranstream_gj.randint(len(apical_sections_b))]
            )
        else:
            sectionidx_a = int(
                basal_sections_a[ranstream_gj.randint(len(basal_sections_a))]
            )
            sectionidx_b = int(
                basal_sections_b[ranstream_gj.randint(len(basal_sections_b))]
            )
    elif (len(apical_sections_a) > 0) and (len(apical_sections_b) > 0):
        sectionidx_a = int(
            apical_sections_a[ranstream_gj.randint(len(apical_sections_a))]
        )
        sectionidx_b = int(
            apical_sections_b[ranstream_gj.randint(len(apical_sections_b))]
        )
    elif (len(basal_sections_a) > 0) and (len(basal_sections_b) > 0):
        sectionidx_a = int(
            basal_sections_a[ranstream_gj.randint(len(basal_sections_a))]
        )
        sectionidx_b = int(
            basal_sections_b[ranstream_gj.randint(len(basal_sections_b))]
        )
    else:
        raise ValueError("Cells with incompatible section types")

    section_a = list(cell_a.sections)[sectionidx_a]
    section_b = list(cell_b.sections)[sectionidx_b]

    position_a = max(ranstream_gj.random_sample(), 0.01)
    position_b = max(ranstream_gj.random_sample(), 0.01)

    soma_a = cell_a.soma_list if hasattr(cell_a, "soma_list") else cell_a.soma
    soma_b = cell_b.soma_list if hasattr(cell_b, "soma_list") else cell_b.soma
    distance_a = distance_to_root(soma_a, section_a, position_a)
    distance_b = distance_to_root(soma_b, section_b, position_b)

    return (
        sectionidx_a,
        position_a,
        distance_a,
        sectionidx_b,
        position_b,
        distance_b,
    )


def generate_gap_junctions(
    connection_prob,
    coupling_coeffs,
    coupling_params,
    ranstream_gj,
    gids_a,
    gids_b,
    gj_probs,
    gj_distances,
    cell_dict_a,
    cell_dict_b,
    gj_dict,
):
    k = int(round(connection_prob * len(gj_distances)))
    selected = ranstream_gj.choice(
        np.arange(0, len(gj_distances)), size=k, replace=False, p=gj_probs
    )
    count = len(selected)

    gid_dict = defaultdict(list)
    for i in selected:
        gid_a = gids_a[i]
        gid_b = gids_b[i]
        gid_dict[gid_a].append(gid_b)

    for gid_a, gids_b in gid_dict.items():
        sections_a = []
        positions_a = []
        sections_b = []
        positions_b = []
        couplings_a = []
        couplings_b = []

        cell_a = cell_dict_a[gid_a]

        for gid_b in gids_b:
            cell_b = cell_dict_b[gid_b]

            (
                section_a,
                position_a,
                distance_a,
                section_b,
                position_b,
                distance_b,
            ) = choose_gj_locations(ranstream_gj, cell_a, cell_b)

            sections_a.append(section_a)
            positions_a.append(position_a)

            sections_b.append(section_b)
            positions_b.append(position_b)

            coupling_weight_a = np.polyval(coupling_params, distance_a)
            coupling_weight_b = np.polyval(coupling_params, distance_b)

            coupling_a = coupling_coeffs * coupling_weight_a
            coupling_b = coupling_coeffs * coupling_weight_b

            couplings_a.append(coupling_a)
            couplings_b.append(coupling_b)

        if len(gids_b) > 0:
            gj_dict[gid_a] = (
                np.asarray(gids_b, dtype=np.uint32),
                {
                    "Location": {
                        "Source section": np.asarray(
                            sections_a, dtype=np.uint32
                        ),
                        "Source position": np.asarray(
                            positions_a, dtype=np.float32
                        ),
                        "Destination section": np.asarray(
                            sections_b, dtype=np.uint32
                        ),
                        "Destination position": np.asarray(
                            positions_b, dtype=np.float32
                        ),
                    },
                    "Coupling strength": {
                        "Source": np.asarray(couplings_a, dtype=np.float32),
                        "Destination": np.asarray(
                            couplings_b, dtype=np.float32
                        ),
                    },
                },
            )

    return count


def generate_gj_connections(
    env,
    forest_path,
    soma_coords_dict,
    gj_config_dict,
    gj_seed,
    connectivity_namespace,
    connectivity_path,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    dry_run=False,
):
    """Generates gap junction connectivity based on Euclidean-distance-weighted probabilities.
    :param gj_config: connection configuration object (instance of env.GapjunctionConfig)
    :param gj_seed: random seed for determining gap junction connectivity
    :param connectivity_namespace: namespace of gap junction connectivity attributes
    :param connectivity_path: path to gap junction connectivity file
    :param io_size: number of I/O ranks to use for parallel connectivity append
    :param chunk_size: HDF5 chunk size for connectivity file (pointer and index datasets)
    :param value_chunk_size: HDF5 chunk size for connectivity file (value datasets)
    :param cache_size: how many cells to read ahead
    """

    comm = env.comm

    rank = comm.rank
    size = comm.size

    if io_size == -1:
        io_size = comm.size

    start_time = time.time()

    ranstream_gj = np.random.RandomState(gj_seed)
    population_pairs = list(gj_config_dict.keys())

    for pp in population_pairs:
        if rank == 0:
            logger.info("%s <-> %s" % (pp[0], pp[1]))

    total_count = 0
    gid_count = 0

    for i, (pp, gj_config) in enumerate(sorted(gj_config_dict.items())):
        if rank == 0:
            logger.info(
                f"Generating gap junction connections between populations {pp[0]} and {pp[1]}..."
            )

        ranstream_gj.seed(gj_seed + i)

        coupling_params = np.asarray(gj_config.coupling_parameters)
        coupling_coeffs = np.asarray(gj_config.coupling_coefficients)
        connection_prob = gj_config.connection_probability
        connection_params = np.asarray(gj_config.connection_parameters)
        connection_bounds = np.asarray(gj_config.connection_bounds)

        population_a = pp[0]
        population_b = pp[1]

        template_name_a = env.celltypes[population_a]["template"]
        template_name_b = env.celltypes[population_b]["template"]

        load_cell_template(env, population_a, bcast_template=True)
        load_cell_template(env, population_b, bcast_template=True)
        template_class_a = getattr(h, template_name_a)
        template_class_b = getattr(h, template_name_b)

        clst_a = []
        gid_a = []
        for gid, coords in soma_coords_dict[population_a].items():
            clst_a.append(np.asarray(coords))
            gid_a.append(gid)
        gid_a = np.asarray(gid_a)

        sortidx_a = np.argsort(gid_a)
        coords_a = np.asarray([clst_a[i] for i in sortidx_a])

        clst_b = []
        gid_b = []
        for gid, coords in soma_coords_dict[population_b].items():
            clst_b.append(np.asarray(coords))
            gid_b.append(gid)
        gid_b = np.asarray(gid_b)

        sortidx_b = np.argsort(gid_b)
        coords_b = np.asarray([clst_b[i] for i in sortidx_b])

        gj_prob_dict = filter_by_distance(
            gid_a[sortidx_a],
            coords_a,
            gid_b[sortidx_b],
            coords_b,
            connection_bounds,
            connection_params,
        )

        gj_probs = []
        gj_distances = []
        gids_a = []
        gids_b = []
        for gid, v in gj_prob_dict.items():
            if gid % size == rank:
                (nngids, nndists, nnprobs) = v
                gids_a.append(np.full(nngids.shape, gid, np.int32))
                gids_b.append(nngids)
                gj_probs.append(nnprobs)
                gj_distances.append(nndists)

        gids_a = np.concatenate(gids_a)
        gids_b = np.concatenate(gids_b)
        gj_probs = np.concatenate(gj_probs)
        gj_probs = gj_probs / gj_probs.sum()
        gj_distances = np.concatenate(gj_distances)
        gids_a = np.asarray(gids_a, dtype=np.uint32)
        gids_b = np.asarray(gids_b, dtype=np.uint32)

        cell_dict_a = {}
        selection_a = set(gids_a)
        if rank == 0:
            logger.info(
                f"Reading tree selection of population {pp[0]} ({len(selection_a)} cells)..."
            )
        (tree_iter_a, _) = read_tree_selection(
            forest_path, population_a, list(selection_a)
        )
        for gid, tree_dict in tree_iter_a:
            cell_dict_a[gid] = cells.make_neurotree_hoc_cell(
                template_class_a, neurotree_dict=tree_dict, gid=gid
            )

        cell_dict_b = {}
        selection_b = set(gids_b)
        if rank == 0:
            logger.info(
                f"Reading tree selection of population {pp[1]} ({len(selection_b)} cells)..."
            )

        (tree_iter_b, _) = read_tree_selection(
            forest_path, population_b, list(selection_b)
        )
        for gid, tree_dict in tree_iter_b:
            cell_dict_b[gid] = cells.make_neurotree_hoc_cell(
                template_class_b, neurotree_dict=tree_dict, gid=gid
            )

        if rank == 0:
            logger.info(
                f"Generating gap junction pairs between populations {pp[0]} and {pp[1]}..."
            )

        gj_dict = {}
        count = generate_gap_junctions(
            connection_prob,
            coupling_coeffs,
            coupling_params,
            ranstream_gj,
            gids_a,
            gids_b,
            gj_probs,
            gj_distances,
            cell_dict_a,
            cell_dict_b,
            gj_dict,
        )

        gj_graph_dict = {pp[0]: {pp[1]: gj_dict}}

        if not dry_run:
            append_graph(
                connectivity_path, gj_graph_dict, io_size=io_size, comm=comm
            )

        total_count += count

    global_count = comm.gather(total_count, root=0)
    if rank == 0:
        logger.info(
            "%i ranks took %i s to generate %i edges"
            % (comm.size, time.time() - start_time, np.sum(global_count))
        )
