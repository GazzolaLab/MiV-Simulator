import gc
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
from miv_simulator import cells, synapses, utils
from miv_simulator import config
from miv_simulator.utils.neuron import configure_hoc, load_template
from mpi4py import MPI
from neuroh5.io import (
    NeuroH5TreeGen,
    append_cell_attributes,
    read_population_ranges,
)
from typing import Optional, Tuple, Literal

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
            global_syn_count = comm.gather(
                total_syn_stats_dict[syn_type], root=root
            )
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
        logger,
    )


def check_synapses(
    gid,
    morph_dict,
    syn_stats_dict,
    seg_density_per_sec,
    layer_set_dict,
    swc_set_dict,
    logger,
):
    layer_stats = syn_stats_dict["layer"]
    swc_stats = syn_stats_dict["swc_type"]

    warning_flag = False
    for syn_type, layer_set in list(layer_set_dict.items()):
        for layer in layer_set:
            if layer in layer_stats:
                if layer_stats[layer][syn_type] <= 0:
                    warning_flag = True
            else:
                warning_flag = True
    if warning_flag:
        logger.warning(
            f"Rank {MPI.COMM_WORLD.Get_rank()}: incomplete synapse layer set for cell {gid}: {layer_stats}"
            f"  layer_set_dict: {layer_set_dict}\n"
            f"  seg_density_per_sec: {seg_density_per_sec}\n"
            f"  morph_dict: {morph_dict}"
        )
    for syn_type, swc_set in swc_set_dict.items():
        for swc_type in swc_set:
            if swc_type in swc_stats:
                if swc_stats[swc_type][syn_type] <= 0:
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
        dt=env.dt,
        tstop=env.tstop,
        celsius=env.globals.get("celsius", None),
    )

    return distribute_synapses(
        forest_filepath=forest_path,
        cell_types=env.celltypes,
        populations=populations,
        distribution=distribution,
        mechanisms_path=mechanisms_path,
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
    populations: Tuple[str, ...],
    distribution: Literal["uniform", "poisson"],
    mechanisms_path: str,
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

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        logger.info(f"{comm.size} ranks have been allocated")

    configure_hoc(mechanisms_directory=mechanisms_path)

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
                    ) = synapses.distribute_uniform_synapses(
                        random_seed,
                        config.SynapseTypesDef.__members__,
                        config.SWCTypesDef.__members__,
                        config.LayersDef.__members__,
                        density_dict,
                        morph_dict,
                        cell_sec_dict,
                        cell_secidx_dict,
                    )

                elif distribution == "poisson":
                    (
                        syn_dict,
                        seg_density_per_sec,
                    ) = synapses.distribute_poisson_synapses(
                        random_seed,
                        config.SynapseTypesDef.__members__,
                        config.SWCTypesDef.__members__,
                        config.LayersDef.__members__,
                        density_dict,
                        morph_dict,
                        cell_sec_dict,
                        cell_secidx_dict,
                    )
                else:
                    raise Exception(
                        f"Unknown distribution type: {distribution}"
                    )

                synapse_dict[gid] = syn_dict
                this_syn_stats = update_synapse_statistics(
                    syn_dict, syn_stats_dict
                )
                check_synapses(
                    gid,
                    morph_dict,
                    this_syn_stats,
                    seg_density_per_sec,
                    layer_set_dict,
                    swc_set_dict,
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
            if (
                (not dry_run)
                and (write_size > 0)
                and (gid_count % write_size == 0)
            ):
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

        if not dry_run:
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

        global_count, summary = global_syn_summary(
            comm, syn_stats, gid_count, root=0
        )
        if rank == 0:
            logger.info(
                f"Population: {population}, {comm.size} ranks took {time.time() - start_time:.2f} s "
                f"to compute synapse locations for {np.sum(global_count)} cells"
            )
            logger.info(summary)

        comm.barrier()

    MPI.Finalize()
