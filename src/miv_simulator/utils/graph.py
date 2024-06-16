"""Classes and procedures related to neuronal connectivity graph analysis. """

import gc, math, time, pickle, base64
from collections import defaultdict, ChainMap
import numpy as np
from mpi4py import MPI
from miv_simulator.utils import (
    Struct,
    get_module_logger,
    add_bins,
    update_bins,
    finalize_bins,
)
from neuroh5.io import (
    NeuroH5ProjectionGen,
    bcast_cell_attributes,
    read_cell_attributes,
    read_population_names,
    read_population_ranges,
    read_projection_names,
)
import h5py

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = get_module_logger(__name__)


def vertex_distribution(
    connectivity_path,
    coords_path,
    distances_namespace,
    destination,
    sources,
    bin_size=20.0,
    cache_size=100,
    comm=None,
):
    """
    Obtain spatial histograms of source vertices connecting to a given destination population.

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace:
    :param destination:
    :param source:

    """

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    color = 0
    if rank == 0:
        color = 1
    comm0 = comm.Split(color, 0)

    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    destination_soma_distances = {}
    if rank == 0:
        logger.info(f"Reading {destination} distances...")
        distances_iter = read_cell_attributes(
            coords_path,
            destination,
            comm=comm0,
            mask=set(["U Distance", "V Distance"]),
            namespace=distances_namespace,
        )

        destination_soma_distances = {
            k: (float(v["U Distance"][0]), float(v["V Distance"][0]))
            for (k, v) in distances_iter
        }

        gc.collect()

    comm.barrier()

    destination_soma_distances = comm.bcast(destination_soma_distances, root=0)
    destination_soma_distance_U = {}
    destination_soma_distance_V = {}
    for k, v in viewitems(destination_soma_distances):
        destination_soma_distance_U[k] = v[0]
        destination_soma_distance_V[k] = v[1]

    del destination_soma_distances

    if sources == ():
        sources = []
        for src, dst in read_projection_names(connectivity_path):
            if dst == destination:
                sources.append(src)

    source_soma_distances = {}
    if rank == 0:
        for s in sources:
            logger.info(f"Reading {s} distances...")
            distances_iter = read_cell_attributes(
                coords_path,
                s,
                comm=comm0,
                mask=set(["U Distance", "V Distance"]),
                namespace=distances_namespace,
            )

            source_soma_distances[s] = {
                k: (float(v["U Distance"][0]), float(v["V Distance"][0]))
                for (k, v) in distances_iter
            }

            gc.collect()

    comm.barrier()
    comm0.Free()

    source_soma_distances = comm.bcast(source_soma_distances, root=0)

    source_soma_distance_U = {}
    source_soma_distance_V = {}
    for s in sources:
        this_source_soma_distance_U = {}
        this_source_soma_distance_V = {}
        for k, v in viewitems(source_soma_distances[s]):
            this_source_soma_distance_U[k] = v[0]
            this_source_soma_distance_V[k] = v[1]
        source_soma_distance_U[s] = this_source_soma_distance_U
        source_soma_distance_V[s] = this_source_soma_distance_V
    del source_soma_distances

    if rank == 0:
        logger.info(
            "Reading connections %s -> %s..." % (str(sources), destination)
        )

    dist_bins = defaultdict(dict)
    dist_u_bins = defaultdict(dict)
    dist_v_bins = defaultdict(dict)

    gg = [
        NeuroH5ProjectionGen(
            connectivity_path,
            source,
            destination,
            cache_size=cache_size,
            comm=comm,
        )
        for source in sources
    ]

    for prj_gen_tuple in zip_longest(*gg):
        destination_gid = prj_gen_tuple[0][0]
        if rank == 0 and destination_gid is not None:
            logger.info("%d" % destination_gid)
        if not all(
            [prj_gen_elt[0] == destination_gid for prj_gen_elt in prj_gen_tuple]
        ):
            raise RuntimeError(
                "destination %s: destination gid %i not matched across multiple projection generators: "
                "%s"
                % (
                    destination,
                    destination_gid,
                    [prj_gen_elt[0] for prj_gen_elt in prj_gen_tuple],
                )
            )

        if destination_gid is not None:
            for source, (this_destination_gid, rest) in zip_longest(
                sources, prj_gen_tuple
            ):
                this_source_soma_distance_U = source_soma_distance_U[source]
                this_source_soma_distance_V = source_soma_distance_V[source]
                this_dist_bins = dist_bins[source]
                this_dist_u_bins = dist_u_bins[source]
                this_dist_v_bins = dist_v_bins[source]
                (source_indexes, attr_dict) = rest
                dst_U = destination_soma_distance_U[destination_gid]
                dst_V = destination_soma_distance_V[destination_gid]
                for source_gid in source_indexes:
                    dist_u = dst_U - this_source_soma_distance_U[source_gid]
                    dist_v = dst_V - this_source_soma_distance_V[source_gid]
                    dist = abs(dist_u) + abs(dist_v)

                    update_bins(this_dist_bins, bin_size, dist)
                    update_bins(this_dist_u_bins, bin_size, dist_u)
                    update_bins(this_dist_v_bins, bin_size, dist_v)

    add_bins_op = MPI.Op.Create(add_bins, commute=True)
    for source in sources:
        dist_bins[source] = comm.reduce(dist_bins[source], op=add_bins_op)
        dist_u_bins[source] = comm.reduce(dist_u_bins[source], op=add_bins_op)
        dist_v_bins[source] = comm.reduce(dist_v_bins[source], op=add_bins_op)

    dist_hist_dict = defaultdict(dict)
    dist_u_hist_dict = defaultdict(dict)
    dist_v_hist_dict = defaultdict(dict)

    if rank == 0:
        for source in sources:
            dist_hist_dict[destination][source] = finalize_bins(
                dist_bins[source], bin_size
            )
            dist_u_hist_dict[destination][source] = finalize_bins(
                dist_u_bins[source], bin_size
            )
            dist_v_hist_dict[destination][source] = finalize_bins(
                dist_v_bins[source], bin_size
            )

    return {
        "Total distance": dist_hist_dict,
        "U distance": dist_u_hist_dict,
        "V distance": dist_v_hist_dict,
    }


def spatial_bin_graph(
    connectivity_path,
    coords_path,
    distances_namespace,
    destination,
    sources,
    extents,
    bin_size=20.0,
    cache_size=100,
    comm=None,
):
    """
    Obtain reduced graphs of the specified projections by binning nodes according to their spatial position.

    :param connectivity_path:
    :param coords_path:
    :param distances_namespace:
    :param destination:
    :param source:

    """

    import networkx as nx

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    (population_ranges, _) = read_population_ranges(coords_path)

    destination_start = population_ranges[destination][0]
    destination_count = population_ranges[destination][1]

    if rank == 0:
        logger.info("reading %s distances..." % destination)

    destination_soma_distances = bcast_cell_attributes(
        coords_path,
        destination,
        namespace=distances_namespace,
        comm=comm,
        root=0,
    )

    ((x_min, x_max), (y_min, y_max)) = extents
    u_bins = np.arange(x_min, x_max, bin_size)
    v_bins = np.arange(y_min, y_max, bin_size)

    dest_u_bins = {}
    dest_v_bins = {}
    destination_soma_distance_U = {}
    destination_soma_distance_V = {}
    for k, v in destination_soma_distances:
        dist_u = v["U Distance"][0]
        dist_v = v["V Distance"][0]
        dest_u_bins[k] = np.searchsorted(u_bins, dist_u, side="left")
        dest_v_bins[k] = np.searchsorted(v_bins, dist_v, side="left")
        destination_soma_distance_U[k] = dist_u
        destination_soma_distance_V[k] = dist_v

    del destination_soma_distances

    if (sources == ()) or (sources == []) or (sources is None):
        sources = []
        for src, dst in read_projection_names(connectivity_path):
            if dst == destination:
                sources.append(src)

    source_soma_distances = {}
    for s in sources:
        if rank == 0:
            logger.info("reading %s distances..." % s)
        source_soma_distances[s] = bcast_cell_attributes(
            coords_path, s, namespace=distances_namespace, comm=comm, root=0
        )

    source_u_bins = {}
    source_v_bins = {}
    source_soma_distance_U = {}
    source_soma_distance_V = {}
    for s in sources:
        this_source_soma_distance_U = {}
        this_source_soma_distance_V = {}
        this_source_u_bins = {}
        this_source_v_bins = {}
        for k, v in source_soma_distances[s]:
            dist_u = v["U Distance"][0]
            dist_v = v["V Distance"][0]
            this_source_u_bins[k] = np.searchsorted(u_bins, dist_u, side="left")
            this_source_v_bins[k] = np.searchsorted(v_bins, dist_v, side="left")
            this_source_soma_distance_U[k] = dist_u
            this_source_soma_distance_V[k] = dist_v
        source_soma_distance_U[s] = this_source_soma_distance_U
        source_soma_distance_V[s] = this_source_soma_distance_V
        source_u_bins[s] = this_source_u_bins
        source_v_bins[s] = this_source_v_bins
    del source_soma_distances

    if rank == 0:
        logger.info(
            "reading connections %s -> %s..." % (str(sources), destination)
        )
    gg = [
        NeuroH5ProjectionGen(
            connectivity_path,
            source,
            destination,
            cache_size=cache_size,
            comm=comm,
        )
        for source in sources
    ]

    dist_bins = defaultdict(dict)
    dist_u_bins = defaultdict(dict)
    dist_v_bins = defaultdict(dict)

    local_u_bin_graph = defaultdict(dict)
    local_v_bin_graph = defaultdict(dict)

    for prj_gen_tuple in zip_longest(*gg):
        destination_gid = prj_gen_tuple[0][0]
        if not all(
            [prj_gen_elt[0] == destination_gid for prj_gen_elt in prj_gen_tuple]
        ):
            raise RuntimeError(
                "destination %s: destination_gid %i not matched across multiple projection generators: "
                "%s"
                % (
                    destination,
                    destination_gid,
                    [prj_gen_elt[0] for prj_gen_elt in prj_gen_tuple],
                )
            )

        if destination_gid is not None:
            dest_u_bin = dest_u_bins[destination_gid]
            dest_v_bin = dest_v_bins[destination_gid]
            for source, (this_destination_gid, rest) in zip_longest(
                sources, prj_gen_tuple
            ):
                this_source_u_bins = source_u_bins[source]
                this_source_v_bins = source_v_bins[source]
                (source_indexes, attr_dict) = rest
                source_u_bin_dict = defaultdict(int)
                source_v_bin_dict = defaultdict(int)
                for source_gid in source_indexes:
                    source_u_bin = this_source_u_bins[source_gid]
                    source_v_bin = this_source_v_bins[source_gid]
                    source_u_bin_dict[source_u_bin] += 1
                    source_v_bin_dict[source_v_bin] += 1
                local_u_bin_graph[dest_u_bin][source] = source_u_bin_dict
                local_v_bin_graph[dest_v_bin][source] = source_v_bin_dict

    local_u_bin_graphs = comm.gather(dict(local_u_bin_graph), root=0)
    local_v_bin_graphs = comm.gather(dict(local_v_bin_graph), root=0)

    u_bin_graph = None
    v_bin_graph = None
    nu = None
    nv = None

    if rank == 0:

        u_bin_edges = {destination: dict(ChainMap(*local_u_bin_graphs))}
        v_bin_edges = {destination: dict(ChainMap(*local_v_bin_graphs))}

        nu = len(u_bins)
        u_bin_graph = nx.Graph()
        for pop in [destination] + list(sources):
            for i in range(nu):
                u_bin_graph.add_node((pop, i))

        for i, ss in viewitems(u_bin_edges[destination]):
            for source, ids in viewitems(ss):
                u_bin_graph.add_weighted_edges_from(
                    [
                        ((source, j), (destination, i), count)
                        for j, count in viewitems(ids)
                    ]
                )

        nv = len(v_bins)
        v_bin_graph = nx.Graph()
        for pop in [destination] + list(sources):
            for i in range(nv):
                v_bin_graph.add_node((pop, i))

        for i, ss in viewitems(v_bin_edges[destination]):
            for source, ids in viewitems(ss):
                v_bin_graph.add_weighted_edges_from(
                    [
                        ((source, j), (destination, i), count)
                        for j, count in viewitems(ids)
                    ]
                )

    label = "%s to %s" % (str(sources), destination)

    return {
        "label": label,
        "bin size": bin_size,
        "destination": destination,
        "sources": sources,
        "U graph": u_bin_graph,
        "V graph": v_bin_graph,
    }


def save_spatial_bin_graph(output_path, graph_dict):

    bin_size = graph_dict["bin size"]
    label_pkl = pickle.dumps(graph_dict["label"])
    label_pkl_str = base64.b64encode(label_pkl)
    destination_pkl = pickle.dumps(graph_dict["destination"])
    destination_pkl_str = base64.b64encode(destination_pkl)
    sources_pkl = pickle.dumps(graph_dict["sources"])
    sources_pkl_str = base64.b64encode(sources_pkl)
    u_bin_graph = graph_dict["U graph"]
    u_bin_graph_pkl = pickle.dumps(u_bin_graph)
    u_bin_graph_pkl_str = base64.b64encode(u_bin_graph_pkl)
    v_bin_graph = graph_dict["V graph"]
    v_bin_graph_pkl = pickle.dumps(v_bin_graph)
    v_bin_graph_pkl_str = base64.b64encode(v_bin_graph_pkl)

    f = h5py.File(output_path)
    dataset_path = "Spatial Bin Graph/%.02f" % bin_size
    grp = f.create_group(dataset_path)
    grp["bin size"] = bin_size
    grp["label.pkl"] = label_pkl_str
    grp["destination.pkl"] = destination_pkl_str
    grp["sources.pkl"] = sources_pkl_str
    grp["U graph.pkl"] = u_bin_graph_pkl_str
    grp["V graph.pkl"] = v_bin_graph_pkl_str
    f.close()


def load_spatial_bin_graph(input_path, dataset_path):

    f = h5py.File(input_path, "r")
    grp = f[dataset_path]
    bin_size = grp["bin size"][()]
    label_dset = grp["label.pkl"]
    destination_dset = grp["destination.pkl"]
    sources_dset = grp["sources.pkl"]
    u_bin_graph_dset = grp["U graph.pkl"]
    v_bin_graph_dset = grp["V graph.pkl"]

    label = pickle.loads(base64.b64decode(label_dset[()]))
    destination = pickle.loads(base64.b64decode(destination_dset[()]))
    sources = pickle.loads(base64.b64decode(sources_dset[()]))

    u_bin_graph = pickle.loads(base64.b64decode(u_bin_graph_dset[()]))
    v_bin_graph = pickle.loads(base64.b64decode(v_bin_graph_dset[()]))
    f.close()

    return {
        "label": label,
        "bin size": bin_size,
        "destination": destination,
        "sources": sources,
        "U graph": u_bin_graph,
        "V graph": v_bin_graph,
    }
