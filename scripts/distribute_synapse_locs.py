import os, sys, gc, logging, string, time, itertools
from mpi4py import MPI
from neuroh5.io import NeuroH5TreeGen, append_cell_attributes, read_population_ranges
import click
from collections import defaultdict
import numpy as np
import biophys_microcircuit
from biophys_microcircuit import cells, neuron_utils, synapses, utils
from biophys_microcircuit.env import Env
from biophys_microcircuit.neuron_utils import configure_hoc_env
from biophys_microcircuit.cells import load_cell_template, make_section_graph
from biophys_microcircuit.utils import *
import h5py

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


def update_syn_stats(env, syn_stats_dict, syn_dict):

    syn_type_excitatory = env.Synapse_Types['excitatory']
    syn_type_inhibitory = env.Synapse_Types['inhibitory']

    this_syn_stats_dict = { 'section': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
        'layer': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
        'swc_type': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
        'total': { 'excitatory': 0, 'inhibitory': 0 } }

    for (syn_id,syn_sec,syn_type,swc_type,syn_layer) in \
        zip(syn_dict['syn_ids'],
                       syn_dict['syn_secs'],
                       syn_dict['syn_types'],
                       syn_dict['swc_types'],
                       syn_dict['syn_layers']):
        
        if syn_type == syn_type_excitatory:
            syn_type_str = 'excitatory'
        elif syn_type == syn_type_inhibitory:
            syn_type_str = 'inhibitory'
        else:
            raise ValueError('Unknown synapse type %s' % str(syn_type))

        syn_stats_dict['section'][syn_sec][syn_type_str] += 1
        syn_stats_dict['layer'][syn_layer][syn_type_str] += 1
        syn_stats_dict['swc_type'][swc_type][syn_type_str] += 1
        syn_stats_dict['total'][syn_type_str] += 1

        this_syn_stats_dict['section'][syn_sec][syn_type_str] += 1
        this_syn_stats_dict['layer'][syn_layer][syn_type_str] += 1
        this_syn_stats_dict['swc_type'][swc_type][syn_type_str] += 1
        this_syn_stats_dict['total'][syn_type_str] += 1

    return this_syn_stats_dict


def global_syn_summary(comm, syn_stats, gid_count, root):
    global_count = comm.gather(gid_count, root=root)
    global_count = np.sum(global_count)
    res = []
    for population in sorted(syn_stats):
        pop_syn_stats = syn_stats[population]
        for part in ['layer', 'swc_type']:
            syn_stats_dict = pop_syn_stats[part]
            for part_name in syn_stats_dict:
                for syn_type in syn_stats_dict[part_name]:
                    global_syn_count = comm.gather(syn_stats_dict[part_name][syn_type], root=root)
                    if comm.rank == root:
                        res.append(f"{population} {part} {part_name}: mean {syn_type} synapses per cell: {np.sum(global_syn_count) / global_count:.2f}")
        total_syn_stats_dict = pop_syn_stats['total']
        for syn_type in total_syn_stats_dict:
            global_syn_count = comm.gather(total_syn_stats_dict[syn_type], root=root)
            if comm.rank == root:
                res.append(f"{population}: mean {syn_type} synapses per cell: {np.sum(global_syn_count) / global_count:.2f}")
        
    return global_count, str.join('\n', res)

def local_syn_summary(syn_stats_dict):
    res = []
    for part_name in ['layer','swc_type']:
        for part_type in syn_stats_dict[part_name]:
            syn_count_dict = syn_stats_dict[part_name][part_type]
            for syn_type, syn_count in list(syn_count_dict.items()):
                res.append("%s %i: %s synapses: %i" % (part_name, part_type, syn_type, syn_count))
    return str.join('\n', res)


def check_syns(gid, morph_dict, syn_stats_dict, seg_density_per_sec, layer_set_dict, swc_set_dict, env, logger):

    layer_stats = syn_stats_dict['layer']
    swc_stats = syn_stats_dict['swc_type']

    warning_flag = False
    for syn_type, layer_set in list(layer_set_dict.items()):
        for layer in layer_set:
            if layer in layer_stats:
                if layer_stats[layer][syn_type] <= 0:
                    warning_flag = True
            else:
                warning_flag = True
    if warning_flag:
        logger.warning(f'Rank {env.comm.Get_rank()}: incomplete synapse layer set for cell {gid}: {layer_stats}'
                       f'  layer_set_dict: {layer_set_dict}\n'
                       f'  seg_density_per_sec: {seg_density_per_sec}\n'
                       f'  morph_dict: {morph_dict}')
    for syn_type, swc_set in viewitems(swc_set_dict):
        for swc_type in swc_set:
            if swc_type in swc_stats:
                if swc_stats[swc_type][syn_type] <= 0:
                    warning_flag = True
            else:
                warning_flag = True
    if warning_flag:
        logger.warning(f'Rank {env.comm.Get_rank()}: incomplete synapse swc type set for cell {gid}: {swc_stats}'
                       f'  swc_set_dict: {swc_set_dict.items}\n'
                       f'  seg_density_per_sec: {seg_density_per_sec}\n'
                       f'   morph_dict: {morph_dict}')
                

            
        
@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option("--template-path", type=str)
@click.option("--output-path", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--distribution", type=str, default='uniform')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--debug", is_flag=True)
def main(config, config_prefix, template_path, output_path, forest_path, populations, distribution, io_size, chunk_size, value_chunk_size,
         write_size, verbose, dry_run, debug):
    """
    :param config:
    :param config_prefix:
    :param template_path:
    :param forest_path:
    :param populations:
    :param distribution:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))
        
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        logger.info(f'{comm.size} ranks have been allocated')

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix, template_paths=template_path)

    configure_hoc_env(env)
    
    if io_size == -1:
        io_size = comm.size

    if output_path is None:
        output_path = forest_path

    if not dry_run:
        if rank==0:
            if not os.path.isfile(output_path):
                input_file  = h5py.File(forest_path,'r')
                output_file = h5py.File(output_path,'w')
                input_file.copy('/H5Types',output_file)
                input_file.close()
                output_file.close()
        comm.barrier()
        
    (pop_ranges, _) = read_population_ranges(forest_path, comm=comm)
    start_time = time.time()
    syn_stats = dict()
    for population in populations:
        syn_stats[population] = { 'section': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                                  'layer': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                                  'swc_type': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                                  'total': { 'excitatory': 0, 'inhibitory': 0 } }


    for population in populations:
        logger.info(f'Rank {rank} population: {population}')
        (population_start, _) = pop_ranges[population]
        template_class = load_cell_template(env, population, bcast_template=True)

        density_dict = env.celltypes[population]['synapses']['density']
        layer_set_dict = defaultdict(set)
        swc_set_dict = defaultdict(set)
        for sec_name, sec_dict in viewitems(density_dict):
            for syn_type, syn_dict in viewitems(sec_dict):
                swc_set_dict[syn_type].add(env.SWC_Types[sec_name])
                for layer_name in syn_dict:
                    if layer_name != 'default':
                        layer = env.layers[layer_name]
                        layer_set_dict[syn_type].add(layer)
        
        syn_stats_dict = { 'section': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                           'layer': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                           'swc_type': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                           'total': { 'excitatory': 0, 'inhibitory': 0 } }

        count = 0
        gid_count = 0
        synapse_dict = {}
        for gid, morph_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, comm=comm, topology=True):
            local_time = time.time()
            if gid is not None:
                logger.info(f'Rank {rank} gid: {gid}: {morph_dict}')
                cell = cells.make_neurotree_hoc_cell(template_class, neurotree_dict=morph_dict, gid=gid)
                cell_sec_dict = {'apical': (cell.apical_list, None), 'basal': (cell.basal_list, None),
                                 'soma': (cell.soma_list, None), 'ais': (cell.ais_list, None),
                                     'hillock': (cell.hillock_list, None)}
                cell_secidx_dict = {'apical': cell.apicalidx, 'basal': cell.basalidx,
                                    'soma': cell.somaidx, 'ais': cell.aisidx, 'hillock': cell.hilidx}

                random_seed = env.model_config['Random Seeds']['Synapse Locations'] + gid
                if distribution == 'uniform':
                    syn_dict, seg_density_per_sec = synapses.distribute_uniform_synapses(random_seed, env.Synapse_Types, env.SWC_Types, env.layers,
                                                                                         density_dict, morph_dict,
                                                                                         cell_sec_dict, cell_secidx_dict)
                                                                    
                    
                elif distribution == 'poisson':
                    syn_dict, seg_density_per_sec = synapses.distribute_poisson_synapses(random_seed, env.Synapse_Types, env.SWC_Types, env.layers,
                                                                                         density_dict, morph_dict,
                                                                                         cell_sec_dict, cell_secidx_dict)
                else:
                    raise Exception('Unknown distribution type: %s' % distribution)

                synapse_dict[gid] = syn_dict
                this_syn_stats = update_syn_stats (env, syn_stats_dict, syn_dict)
                check_syns(gid, morph_dict, this_syn_stats, seg_density_per_sec, layer_set_dict, swc_set_dict, env, logger)
                
                del cell
                num_syns = len(synapse_dict[gid]['syn_ids'])
                logger.info(f'Rank {rank} took {time.time() - local_time:.2f} s to compute {num_syns} synapse locations for {population} gid: {gid}\n'
                            f'{local_syn_summary(this_syn_stats)}')
                gid_count += 1
            else:
                logger.info(f'Rank {rank} gid is None')
            gc.collect()
            if (not dry_run) and (write_size > 0) and (gid_count % write_size == 0):
                append_cell_attributes(output_path, population, synapse_dict,
                                       namespace='Synapse Attributes', comm=comm, io_size=io_size, 
                                       chunk_size=chunk_size, value_chunk_size=value_chunk_size)
                synapse_dict = {}
            syn_stats[population] = syn_stats_dict
            count += 1
            if debug and count == 5:
                break

        if not dry_run:
            append_cell_attributes(output_path, population, synapse_dict,
                                   namespace='Synapse Attributes', comm=comm, io_size=io_size, 
                                   chunk_size=chunk_size, value_chunk_size=value_chunk_size)

        global_count, summary = global_syn_summary(comm, syn_stats, gid_count, root=0)
        if rank == 0:
            logger.info(f'Population: {population}, {comm.size} ranks took {time.time() - start_time:.2f} s '
                        f'to compute synapse locations for {np.sum(global_count)} cells')
            logger.info(summary)

        comm.barrier()
            
    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
