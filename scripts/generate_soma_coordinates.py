##
## Generate soma coordinates within layer-specific volume.
##

import os, sys, os.path, itertools, random, pickle, logging, click, gc
from collections import defaultdict
import math
from mpi4py import MPI
import h5py
import numpy as np
from neuroh5.io import append_cell_attributes, read_population_ranges
import rbf
from rbf.pde.geometry import contains
from rbf.pde.nodes import min_energy_nodes, disperse
from rbf.pde.sampling import rejection_sampling
from scipy.spatial import cKDTree
from ca1.env import Env
from neural_geometry.alphavol import alpha_shape
from neural_geometry.geometry import make_uvl_distance, make_alpha_shape, load_alpha_shape, save_alpha_shape, get_total_extents, get_layer_extents, uvl_in_bounds
from ca1.CA1_volume import make_CA1_volume, CA1_volume, CA1_volume_transform
from ca1.utils import get_script_logger, config_logging, list_find, viewitems

script_name = os.path.basename(__file__)
logger = get_script_logger(script_name)

def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


#sys_excepthook = sys.excepthook
#sys.excepthook = mpi_excepthook

def random_subset( iterator, K ):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item

    return result


def gen_min_energy_nodes(count, domain, constraint, nodeiter, dispersion_delta, snap_delta):

    N = int(count*2) # layer-specific number of nodes
    node_count = 0

    while node_count < count:
        # create N quasi-uniformly distributed nodes
        def rho(x):
            return np.ones(x.shape[0])
        #nodes = rejection_sampling(N, rho, (vert, smp), start=0)
                    
        out = min_energy_nodes(N, domain, iterations=nodeiter, 
                               **{'dispersion_delta':dispersion_delta, 'snap_delta': snap_delta})
        nodes = out[0]

        # remove nodes with nan
        nodes1 = nodes[~np.logical_or.reduce((np.isnan(nodes[:,0]), np.isnan(nodes[:,1]), np.isnan(nodes[:,2])))]
                    
        # remove nodes outside of the domain
        vert, smp = domain
        in_nodes = nodes[contains(nodes1, vert, smp)]
        valid_idxs = None
        if constraint is not None:
            valid_idxs = []
            current_xyz = in_nodes.reshape(-1,3)
            for i in range(len(current_xyz)):
                if current_xyz[i][2] >= constraint[0] and current_xyz[i][2] <= constraint[1]:
                    valid_idxs.append(i)
            in_nodes = in_nodes[valid_idxs]
        node_count = len(in_nodes)
        N = int(1.5*N)
        logger.info("%i interior nodes out of %i nodes generated" % (node_count, len(nodes)))
    
    return in_nodes

@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=False, type=click.Path(exists=True, file_okay=False, dir_okay=True), default="config")
@click.option("--types-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--geometry-path", required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--output-path", required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--output-namespace", type=str, default='Generated Coordinates')
@click.option("--populations", '-i', type=str, multiple=True)
@click.option("--resolution", type=(int,int,int), default=(3,3,3))
@click.option("--alpha-radius", type=float, default=2500.)
@click.option("--nodeiter", type=int, default=10)
@click.option("--dispersion-delta", type=float, default=0.1)
@click.option("--snap-delta", type=float, default=0.01)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--verbose", '-v', type=bool, default=False, is_flag=True)
def main(config, config_prefix, types_path, geometry_path, output_path, output_namespace, populations, resolution, alpha_radius, nodeiter, dispersion_delta, snap_delta, io_size, chunk_size, value_chunk_size, verbose):

    config_logging(verbose)
    logger = get_script_logger(script_name)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    np.seterr(all='raise')
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)


    if rank==0:
        if not os.path.isfile(output_path):
            input_file  = h5py.File(types_path,'r')
            output_file = h5py.File(output_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix)

    random_seed = int(env.model_config['Random Seeds']['Soma Locations'])
    random.seed(random_seed)
    
    layer_extents = env.geometry['Parametric Surface']['Layer Extents']
    rotate = env.geometry['Parametric Surface']['Rotation']

    (extent_u, extent_v, extent_l) = get_total_extents(layer_extents)
    vol = make_CA1_volume(extent_u, extent_v, extent_l,
                          rotate=rotate, resolution=resolution)
    layer_alpha_shape_path = 'Layer Alpha Shape/%d/%d/%d' % resolution
    if rank == 0:
        logger.info("Constructing alpha shape for volume: extents: %s..." % str((extent_u, extent_v, extent_l)))
        vol_alpha_shape_path = '%s/all' % (layer_alpha_shape_path)
        if geometry_path:
            vol_alpha_shape = load_alpha_shape(geometry_path, vol_alpha_shape_path)
        else:
            vol_alpha_shape = make_alpha_shape(vol, alpha_radius=alpha_radius)
            if geometry_path:
                save_alpha_shape(geometry_path, vol_alpha_shape_path, vol_alpha_shape)
        vert = vol_alpha_shape.points
        smp  = np.asarray(vol_alpha_shape.bounds, dtype=np.int64)
        vol_domain = (vert, smp)
    
    layer_alpha_shapes = {}
    layer_extent_vals = {}
    layer_extent_transformed_vals = {}
    if rank == 0:
        for layer, extents in viewitems(layer_extents):
            (extent_u, extent_v, extent_l) = get_layer_extents(layer_extents, layer)
            layer_extent_vals[layer] = (extent_u, extent_v, extent_l)
            layer_extent_transformed_vals[layer] = CA1_volume_transform(extent_u, extent_v, extent_l)
            has_layer_alpha_shape = False
            if geometry_path:
                this_layer_alpha_shape_path = '%s/%s' % (layer_alpha_shape_path, layer)
                this_layer_alpha_shape = load_alpha_shape(geometry_path, this_layer_alpha_shape_path)
                layer_alpha_shapes[layer] = this_layer_alpha_shape
                if this_layer_alpha_shape is not None:
                    has_layer_alpha_shape = True
            if not has_layer_alpha_shape:
                logger.info("Constructing alpha shape for layers %s: extents: %s..." % (layer, str(extents)))
                layer_vol = make_CA1_volume(extent_u, extent_v, extent_l,
                                            rotate=rotate, resolution=resolution)
                this_layer_alpha_shape = make_alpha_shape(layer_vol, alpha_radius=alpha_radius)
                layer_alpha_shapes[layer] = this_layer_alpha_shape
                if geometry_path:
                    save_alpha_shape(geometry_path, this_layer_alpha_shape_path, this_layer_alpha_shape)

    comm.barrier()
    population_ranges = read_population_ranges(output_path, comm)[0]
    if len(populations) == 0:
        populations = sorted(population_ranges.keys())
        
    total_count = 0
    for population in populations:
        (population_start, population_count) = population_ranges[population]
        total_count += population_count
        
    all_xyz_coords1 = None
    generated_coords_count_dict = defaultdict(int)
    if rank == 0:
        all_xyz_coords_lst = []
        for population in populations:
            gc.collect()

            (population_start, population_count) = population_ranges[population]

            pop_layers     = env.geometry['Cell Distribution'][population]
            pop_constraint = None
            if 'Cell Constraints' in env.geometry:
                if population in env.geometry['Cell Constraints']:
                    pop_constraint = env.geometry['Cell Constraints'][population]
            if rank == 0:
                logger.info("Population %s: layer distribution is %s" % (population, str(pop_layers)))
            
            pop_layer_count = 0
            for layer, count in viewitems(pop_layers):
                pop_layer_count += count
            assert(population_count == pop_layer_count)

            xyz_coords_lst = []
            for layer, count in viewitems(pop_layers):
                if count <= 0:
                    continue
                
                alpha = layer_alpha_shapes[layer]

                vert = alpha.points
                smp  = np.asarray(alpha.bounds, dtype=np.int64)
                
                extents_xyz = layer_extent_transformed_vals[layer]                              
                for (vvi,vv) in enumerate(vert):
                    for (vi,v) in enumerate(vv):
                        if v < extents_xyz[vi][0]: vert[vvi][vi] = extents_xyz[vi][0]
                        elif v > extents_xyz[vi][1]: vert[vvi][vi] = extents_xyz[vi][1]

                N = int(count*2) # layer-specific number of nodes
                node_count = 0

                logger.info("Generating %i nodes in layer %s for population %s..." % (N, layer, population))
                if verbose:
                    rbf_logger = logging.Logger.manager.loggerDict['rbf.pde.nodes']
                    rbf_logger.setLevel(logging.DEBUG)

                min_energy_constraint = None
                if pop_constraint is not None and layer in pop_constraint:
                    min_energy_constraint = pop_constraint[layer]
                    
                nodes = gen_min_energy_nodes(count, (vert, smp), 
                                             min_energy_constraint, 
                                             nodeiter, dispersion_delta, snap_delta)
                #nodes = gen_min_energy_nodes(count, (vert, smp), 
                #                             pop_constraint[layer] if pop_constraint is not None else None, 
                #                             nodeiter, dispersion_delta, snap_delta)
                
                xyz_coords_lst.append(nodes.reshape(-1,3))

            for this_xyz_coords in xyz_coords_lst:
                all_xyz_coords_lst.append(this_xyz_coords)
                generated_coords_count_dict[population] += len(this_xyz_coords)

        # Additional dispersion step to ensure no overlapping cell positions
        all_xyz_coords = np.row_stack(all_xyz_coords_lst)
        mask = np.ones((all_xyz_coords.shape[0],), dtype=np.bool)
        # distance to nearest neighbor
        while True:
            kdt = cKDTree(all_xyz_coords[mask, :])
            nndist, nnindices = kdt.query(all_xyz_coords[mask, :], k=2)
            nndist, nnindices = nndist[:, 1:], nnindices[:, 1:]

            zindices = nnindices[np.argwhere(np.isclose(nndist, 0.0, atol=1e-3, rtol=1e-3))]
            if len(zindices) > 0:
                mask[np.argwhere(mask)[zindices]] = False
            else:
                break

        coords_offset = 0
        for population in populations:
            pop_coords_count = generated_coords_count_dict[population]
            pop_mask = mask[coords_offset:coords_offset + pop_coords_count]
            generated_coords_count_dict[population] = np.count_nonzero(pop_mask)
            coords_offset += pop_coords_count

        logger.info("Dispersion of %i nodes..." % np.count_nonzero(mask))
        all_xyz_coords1 = disperse(all_xyz_coords[mask, :], vol_domain, delta=dispersion_delta)

    if rank == 0:
        logger.info("Computing UVL coordinates of %i nodes..." % len(all_xyz_coords1))

    all_xyz_coords_interp = None
    all_uvl_coords_interp = None

    if rank == 0:
        all_uvl_coords_interp = vol.inverse(all_xyz_coords1)
        all_xyz_coords_interp = vol(all_uvl_coords_interp[:,0],
                                    all_uvl_coords_interp[:,1],
                                    all_uvl_coords_interp[:,2],mesh=False).reshape(3,-1).T

    if rank == 0:
        logger.info("Broadcasting generated nodes...")

    xyz_coords = comm.bcast(all_xyz_coords1, root=0)
    all_xyz_coords_interp = comm.bcast(all_xyz_coords_interp, root=0)
    all_uvl_coords_interp = comm.bcast(all_uvl_coords_interp, root=0)
    generated_coords_count_dict = comm.bcast(dict(generated_coords_count_dict), root=0)

    coords_offset = 0
    pop_coords_dict = {}
    for population in populations:
        xyz_error = np.asarray([0.0, 0.0, 0.0])

        pop_layers  = env.geometry['Cell Distribution'][population]

        pop_start, pop_count = population_ranges[population]
        coords = []
        
        gen_coords_count = generated_coords_count_dict[population]

        for i, coord_ind in enumerate(range(coords_offset, coords_offset+gen_coords_count)):

            if i % size == rank:

                uvl_coords  = all_uvl_coords_interp[coord_ind,:].ravel()
                xyz_coords1 = all_xyz_coords_interp[coord_ind,:].ravel()
                if uvl_in_bounds(all_uvl_coords_interp[coord_ind,:], layer_extents, pop_layers):
                    xyz_error   = np.add(xyz_error, np.abs(np.subtract(xyz_coords[coord_ind,:], xyz_coords1)))

                    logger.info('Rank %i: %s cell %i: %f %f %f' % (rank, population, i, uvl_coords[0], uvl_coords[1], uvl_coords[2]))

                    coords.append((xyz_coords1[0],xyz_coords1[1],xyz_coords1[2],
                                  uvl_coords[0],uvl_coords[1],uvl_coords[2]))
                else:
                    logger.debug('Rank %i: %s cell %i not in bounds: %f %f %f' % (rank, population, i, uvl_coords[0], uvl_coords[1], uvl_coords[2]))
                    uvl_coords = None
                    xyz_coords1 = None

        
        total_xyz_error = np.zeros((3,))
        comm.Allreduce(xyz_error, total_xyz_error, op=MPI.SUM)

        coords_count = 0
        coords_count = np.sum(np.asarray(comm.allgather(len(coords))))

        mean_xyz_error = np.asarray([(total_xyz_error[0] / coords_count), \
                                     (total_xyz_error[1] / coords_count), \
                                     (total_xyz_error[2] / coords_count)])

        pop_coords_dict[population] = coords
        coords_offset += gen_coords_count
        
        if rank == 0:
            logger.info('Total %i coordinates generated for population %s: mean XYZ error: %f %f %f' %
                        (coords_count, population,
                         mean_xyz_error[0], mean_xyz_error[1], mean_xyz_error[2]))

    if rank == 0:
        color = 1
    else:
        color = 0

    ## comm0 includes only rank 0
    comm0 = comm.Split(color, 0)

    for population in populations:

        pop_start, pop_count = population_ranges[population]
        pop_layers     = env.geometry['Cell Distribution'][population]
        pop_constraint = None
        if 'Cell Constraints' in env.geometry:
            if population in env.geometry['Cell Constraints']:
                pop_constraint = env.geometry['Cell Constraints'][population]

        coords_lst = comm.gather(pop_coords_dict[population], root=0)
        if rank == 0:
            all_coords = []
            for sublist in coords_lst:
                for item in sublist:
                    all_coords.append(item)
            coords_count = len(all_coords)
                    
            if coords_count < pop_count:
                logger.warning("Generating additional %i coordinates for population %s..." % (pop_count - len(all_coords), population))

                safety = 0.01
                delta = pop_count - len(all_coords)
                for i in range(delta):
                    for layer, count in viewitems(pop_layers):
                        if count > 0:
                            min_extent = layer_extents[layer][0]
                            max_extent = layer_extents[layer][1]
                            coord_u = np.random.uniform(min_extent[0] + safety, max_extent[0] - safety)
                            coord_v = np.random.uniform(min_extent[1] + safety, max_extent[1] - safety)
                            if pop_constraint is None:
                                coord_l = np.random.uniform(min_extent[2] + safety, max_extent[2] - safety)
                            else:
                                coord_l = np.random.uniform(pop_constraint[layer][0] + safety, pop_constraint[layer][1] - safety)
                            xyz_coords = CA1_volume(coord_u, coord_v, coord_l, rotate=rotate).ravel()
                            all_coords.append((xyz_coords[0],xyz_coords[1],xyz_coords[2],
                                              coord_u, coord_v, coord_l))

            sampled_coords = random_subset(all_coords, int(pop_count))
            sampled_coords.sort(key=lambda coord: coord[3]) ## sort on U coordinate
            
            coords_dict = { pop_start+i :  { 'X Coordinate': np.asarray([x_coord],dtype=np.float32),
                                             'Y Coordinate': np.asarray([y_coord],dtype=np.float32),
                                             'Z Coordinate': np.asarray([z_coord],dtype=np.float32),
                                             'U Coordinate': np.asarray([u_coord],dtype=np.float32),
                                             'V Coordinate': np.asarray([v_coord],dtype=np.float32),
                                             'L Coordinate': np.asarray([l_coord],dtype=np.float32) }
                            for (i,(x_coord,y_coord,z_coord,u_coord,v_coord,l_coord)) in enumerate(sampled_coords) }

            append_cell_attributes(output_path, population, coords_dict,
                                   namespace=output_namespace,
                                   io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size,comm=comm0)

        comm.barrier()

    comm0.Free()

if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
