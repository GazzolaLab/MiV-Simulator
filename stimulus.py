import os, sys, gc, copy, time
import numpy as np
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
from collections import defaultdict, ChainMap, namedtuple
from MiV.utils import get_module_logger, Struct, gauss2d, gaussian, viewitems, mpi_op_set_union
from MiV.stgen import get_inhom_poisson_spike_times_by_thinning
from neuroh5.io import read_cell_attributes, append_cell_attributes, NeuroH5CellAttrGen, scatter_read_cell_attribute_selection
from mpi4py import MPI
import h5py


## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = get_module_logger(__name__)


PhaseModConfig = namedtuple('PhaseModConfig',
                           ['phase_range',
                            'phase_pref',
                            'phase_offset',
                            'mod_depth',
                            'frequency'])


    
class ConstantInputCellConfig:
    def __init__(self, selectivity_type=None, arena=None, 
                 peak_rate=None, local_random=None, selectivity_attr_dict=None, phase_mod_config=None):
        """
        :param selectivity_type: int
        :param arena: namedtuple
        :param peak_rate: float
        :param local_random: :class:'np.random.RandomState'
        :param selectivity_attr_dict: dict
        """

        self.phase_mod_function = None
        if phase_mod_config is not None:
            phase_range = phase_mod_config.phase_range
            phase_pref = phase_mod_config.phase_pref
            phase_offset = phase_mod_config.phase_offset
            mod_depth = phase_mod_config.mod_depth
            freq = phase_mod_config.frequency
            
            self.phase_mod_function = lambda t, initial_phase=0.: stationary_phase_mod(t, phase_range, phase_pref, phase_offset+initial_phase, mod_depth, freq)

        
        if selectivity_attr_dict is not None:
            self.init_from_attr_dict(selectivity_attr_dict)
        elif any([arg is None for arg in [selectivity_type, peak_rate, arena]]):
            raise RuntimeError('ConstantInputCellConfig: missing argument(s) required for object construction')
        else:
            if local_random is None:
                local_random = np.random.RandomState()
            self.selectivity_type = selectivity_type
            self.peak_rate = peak_rate

    def init_from_attr_dict(self, selectivity_attr_dict):
        self.selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        self.peak_rate = selectivity_attr_dict['Peak Rate'][0]

    def get_selectivity_attr_dict(self):
        return {'Selectivity Type': np.array([self.selectivity_type], dtype=np.uint8),
                'Peak Rate': np.array([self.peak_rate], dtype=np.float32),
                }

    def get_rate_map(self, x, y, velocity=None, initial_phase=0.0):
        """

        :param x: array
        :param y: array
        :return: array
        """

        if (velocity is None) and (self.phase_mod_function is not None):
            raise RuntimeError("ConstantInputCellConfig.get_rate_map: when phase config is provided, get_rate_map must be provided with velocity")
        
        rate_array = np.ones_like(x, dtype=np.float32) * self.peak_rate
        mean_rate = np.mean(rate_array)
        if (velocity is not None) and (self.phase_mod_function is not None):
            d = np.insert(np.cumsum(np.sqrt(np.diff(x) ** 2. + np.diff(y) ** 2.)), 0, 0.)
            t = d/velocity
            rate_array *= self.phase_mod_function(t, initial_phase=initial_phase)
            mean_rate_mod = np.mean(rate_array)
            if mean_rate_mod > 0.:
                rate_array *= mean_rate / mean_rate_mod

        return rate_array

    
    


def get_input_cell_config(selectivity_type, selectivity_type_names, population=None, stimulus_config=None,
                          arena=None, distance=None, local_random=None,
                          selectivity_attr_dict=None, phase_mod_config=None, noise_gen_dict=None, comm=None):
    """

    :param selectivity_type: int
    :param selectivity_type_names: dict: {int: str}
    :param population: str
    :param stimulus_config: dict
    :param arena: namedtuple
    :param distance: float; u arc distance normalized to reference layer
    :param local_random: :class:'np.random.RandomState'
    :param selectivity_attr_dict: dict
    :param phase_mod_config: dict; oscillatory phase modulation configuration
    :return: instance of one of various InputCell classes
    """
    if selectivity_type not in selectivity_type_names:
        raise RuntimeError('get_input_cell_config: enumerated selectivity type: %i not recognized' % selectivity_type)
    selectivity_type_name = selectivity_type_names[selectivity_type]

    if selectivity_attr_dict is not None:
        if selectivity_type_name == 'constant':
            input_cell_config = ConstantInputCellConfig(selectivity_attr_dict=selectivity_attr_dict,
                                                        phase_mod_config=phase_mod_config)
        else:
            RuntimeError('get_input_cell_config: selectivity type %s is not supported' % selectivity_type_name)
    elif any([arg is None for arg in [population, stimulus_config, arena]]):
        raise RuntimeError(f'get_input_cell_config: missing argument(s) required to construct {selectivity_type_name} cell config object: population: {population} arena: {arena} stimulus_config: {stimulus_config}')
    else:
        if population not in stimulus_config['Peak Rate'] or \
                selectivity_type not in stimulus_config['Peak Rate'][population]:
            raise RuntimeError('get_input_cell_config: peak rate not specified for population: %s, selectivity type: '
                               '%s' % (population, selectivity_type_name))
        peak_rate = stimulus_config['Peak Rate'][population][selectivity_type]

        if selectivity_type_name == 'constant':
            input_cell_config = ConstantInputCellConfig(selectivity_type=selectivity_type, arena=arena,
                                                        peak_rate=peak_rate,
                                                        phase_mod_config=phase_mod_config)
        else:
            RuntimeError(f'get_input_cell_config: selectivity type: {selectivity_type_name} not implemented')

    return input_cell_config


def get_equilibration(env):
    if 'Equilibration Duration' in env.stimulus_config and env.stimulus_config['Equilibration Duration'] > 0.:
        equilibrate_len = int(env.stimulus_config['Equilibration Duration'] /
                              env.stimulus_config['Temporal Resolution'])
        from scipy.signal import hann
        equilibrate_hann = hann(2 * equilibrate_len)[:equilibrate_len]
        equilibrate = (equilibrate_hann, equilibrate_len)
    else:
        equilibrate = None

    return equilibrate



def get_2D_arena_bounds(arena, margin=0., margin_fraction=None):
    """

    :param arena: namedtuple
    :return: tuple of (tuple of float)
    """

    vertices_x = np.asarray([v[0] for v in arena.domain.vertices], dtype=np.float32)
    vertices_y = np.asarray([v[1] for v in arena.domain.vertices], dtype=np.float32)
    if margin_fraction is not None:
        extent_x = np.abs(np.max(vertices_x) - np.min(vertices_x))
        extent_y = np.abs(np.max(vertices_y) - np.min(vertices_y))
        margin = max(margin_fraction*extent_x, margin_fraction*extent_y)
    arena_x_bounds = (np.min(vertices_x) - margin, np.max(vertices_x) + margin)
    arena_y_bounds = (np.min(vertices_y) - margin, np.max(vertices_y) + margin)

    return arena_x_bounds, arena_y_bounds


def get_2D_arena_extents(arena):
    """

    :param arena: namedtuple
    :return: tuple of (tuple of float)
    """

    vertices_x = np.asarray([v[0] for v in arena.domain.vertices], dtype=np.float32)
    vertices_y = np.asarray([v[1] for v in arena.domain.vertices], dtype=np.float32)
    extent_x = np.abs(np.max(vertices_x) - np.min(vertices_x))
    extent_y = np.abs(np.max(vertices_y) - np.min(vertices_y))

    return extent_x, extent_y


def get_2D_arena_spatial_mesh(arena, spatial_resolution=5., margin=0., indexing='ij'):
    """

    :param arena: namedtuple
    :param spatial_resolution: float (cm)
    :param margin: float
    :return: tuple of array
    """
    arena_x_bounds, arena_y_bounds = get_2D_arena_bounds(arena=arena, margin=margin)
    arena_x = np.arange(arena_x_bounds[0], arena_x_bounds[1] + spatial_resolution / 2., spatial_resolution)
    arena_y = np.arange(arena_y_bounds[0], arena_y_bounds[1] + spatial_resolution / 2., spatial_resolution)

    return np.meshgrid(arena_x, arena_y, indexing=indexing)


def get_2D_arena_grid(arena, spatial_resolution=5., margin=0., indexing='ij'):
    """

    :param arena: namedtuple
    :param spatial_resolution: float (cm)
    :param margin: float
    :return: tuple of array
    """
    arena_x_bounds, arena_y_bounds = get_2D_arena_bounds(arena=arena, margin=margin)
    arena_x = np.arange(arena_x_bounds[0], arena_x_bounds[1] + spatial_resolution / 2., spatial_resolution)
    arena_y = np.arange(arena_y_bounds[0], arena_y_bounds[1] + spatial_resolution / 2., spatial_resolution)

    return arena_x, arena_y


def generate_linear_trajectory(trajectory, temporal_resolution=1., equilibration_duration=None):
    """
    Construct coordinate arrays for a spatial trajectory, considering run velocity to interpolate at the specified
    temporal resolution. Optionally, the trajectory can be prepended with extra distance traveled for a specified
    network equilibration time, with the intention that the user discards spikes generated during this period before
    analysis.
    :param trajectory: namedtuple
    :param temporal_resolution: float (ms)
    :param equilibration_duration: float (ms)
    :return: tuple of array
    """
    velocity = trajectory.velocity  # (cm / s)
    spatial_resolution = velocity / 1000. * temporal_resolution
    x = trajectory.path[:, 0]
    y = trajectory.path[:, 1]

    if equilibration_duration is not None:
        equilibration_distance = velocity / 1000. * equilibration_duration
        x = np.insert(x, 0, x[0] - equilibration_distance)
        y = np.insert(y, 0, y[0])
    else:
        equilibration_duration = 0.
        equilibration_distance = 0.

    segment_lengths = np.sqrt(np.diff(x) ** 2. + np.diff(y) ** 2.)
    distance = np.insert(np.cumsum(segment_lengths), 0, 0.)

    interp_distance = np.arange(distance.min(), distance.max() + spatial_resolution / 2., spatial_resolution)
    interp_x = np.interp(interp_distance, distance, x)
    interp_y = np.interp(interp_distance, distance, y)
    t = interp_distance / (velocity / 1000.)  # ms

    t = np.subtract(t, equilibration_duration)
    interp_distance -= equilibration_distance

    return t, interp_x, interp_y, interp_distance


def generate_input_spike_trains(env, population, selectivity_type_names, trajectory, gid, selectivity_attr_dict, spike_train_attr_name='Spike Train',
                                selectivity_type_name=None, spike_hist_resolution=1000, equilibrate=None, phase_mod_config=None, initial_phases=None,
                                spike_hist_sum=None, return_selectivity_features=True, n_trials=1, merge_trials=True, time_range=None,
                                comm=None, seed=None, debug=False):
    """
    Generates spike trains for the given gid according to the
    input selectivity rate maps contained in the given selectivity
    file, and returns a dictionary with spike trains attributes.

    :param env:
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.rank

    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    t, x, y, d = trajectory
    abs_d = d - d[0]
    abs_t = (t - t[0])/1000.
    velocity = np.insert(abs_d[1:]/abs_t[1:], 0, abs_d[1]/abs_t[1])
    
    equilibration_duration = float(env.stimulus_config['Equilibration Duration'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])

    local_random = np.random.RandomState()
    input_spike_train_seed = int(env.model_config['Random Seeds']['Input Spiketrains'])

    if phase_mod_config is not None:
        if (n_trials > 1) and (initial_phases is None):
            initial_phases = global_oscillation_initial_phases(env, n_trials)
            
    if seed is None:
        local_random.seed(int(input_spike_train_seed + gid))
    else:
        local_random.seed(int(seed))

    this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
    this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
    if selectivity_type_name is None:
        selectivity_type_name = this_selectivity_type_name

    input_cell_config = get_input_cell_config(selectivity_type=this_selectivity_type,
                                              selectivity_type_names=selectivity_type_names,
                                              selectivity_attr_dict=selectivity_attr_dict,
                                              phase_mod_config=phase_mod_config)

    trial_duration = np.max(t) - np.min(t)
    if time_range is not None:
        trial_duration = max(trial_duration, (time_range[1] - time_range[0]) + equilibration_duration )
        
    spike_trains = []
    trial_indices = []
    initial_phase = 0.
    for i in range(n_trials):
        if initial_phases is not None:
            initial_phase = initial_phases[i]
        rate_map = input_cell_config.get_rate_map(x=x, y=y, velocity=velocity if phase_mod_config is not None else None,
                                                  initial_phase=initial_phase)
        if (selectivity_type_name != 'constant') and (equilibrate is not None):
            equilibrate_filter, equilibrate_len = equilibrate
            rate_map[:equilibrate_len] = np.multiply(rate_map[:equilibrate_len], equilibrate_filter)

        spike_train = np.asarray(get_inhom_poisson_spike_times_by_thinning(rate_map, t, dt=temporal_resolution,
                                                                           generator=local_random),
                                 dtype=np.float32)
        if merge_trials:
            spike_train += float(i)*trial_duration
        spike_trains.append(spike_train)
        trial_indices.append(np.ones((spike_train.shape[0],), dtype=np.uint8) * i)

    if debug and rank == 0:
        callback, context = debug
        this_context = Struct(**dict(context()))
        this_context.update(dict(locals()))
        callback(this_context)

    spikes_attr_dict = dict()
    if merge_trials:
        spikes_attr_dict[spike_train_attr_name] = np.asarray(np.concatenate(spike_trains), dtype=np.float32)
        spikes_attr_dict['Trial Index'] = np.asarray(np.concatenate(trial_indices), dtype=np.uint8)
    else:
        spikes_attr_dict[spike_train_attr_name] = spike_trains
        spikes_attr_dict['Trial Index'] = trial_indices
        
    spikes_attr_dict['Trial Duration'] = np.asarray([trial_duration]*n_trials, dtype=np.float32)
    
    if return_selectivity_features:
        spikes_attr_dict['Selectivity Type'] = np.array([this_selectivity_type], dtype=np.uint8)
        spikes_attr_dict['Trajectory Rate Map'] = np.asarray(rate_map, dtype=np.float32)

    if spike_hist_sum is not None:
        spike_hist_edges = np.linspace(min(t), max(t), spike_hist_resolution + 1)
        hist, edges = np.histogram(spike_train, bins=spike_hist_edges)
        spike_hist_sum[this_selectivity_type_name] = np.add(spike_hist_sum[this_selectivity_type_name], hist)

    return spikes_attr_dict



def choose_input_selectivity_type(p, local_random):
    """

    :param p: dict: {str: float}
    :param local_random: :class:'np.random.RandomState'
    :return: str
    """
    if len(p) == 1:
        return list(p.keys())[0]
    return local_random.choice(list(p.keys()), p=list(p.values()))


def generate_input_features(env, population, arena, arena_x, arena_y,
                            gid, norm_distances, 
                            selectivity_type_names,
                            selectivity_type_namespaces, noise_gen_dict=None,
                            rate_map_sum=None, debug=False):
    """
    Generates input features for the given population and
    returns the selectivity type-specific dictionary provided through
    argument selectivity_type_namespaces.  The set of selectivity
    attributes is determined by procedure get_selectivity_attr_dict in
    the respective input cell configuration class
    (e.g. ConstantInputCellConfig).

    :param env
    :param population: str
    :param arena: str
    :param gid: int
    :param distances: (float, float)
    :param selectivity_type_names: 
    :param selectivity_type_namespaces: 
    :param debug: bool
    """

    if env.comm is not None:
        rank = env.comm.rank
    else:
        rank = 0
    
    norm_u_arc_distance = norm_distances[0]
    selectivity_seed_offset = int(env.model_config['Random Seeds']['Input Selectivity'])

    local_random = np.random.RandomState()
    local_random.seed(int(selectivity_seed_offset + gid))
    this_selectivity_type = \
     choose_input_selectivity_type(p=env.stimulus_config['Selectivity Type Probabilities'][population],
                                   local_random=local_random)

    
    
    input_cell_config = get_input_cell_config(population=population,
                                              selectivity_type=this_selectivity_type,
                                              selectivity_type_names=selectivity_type_names,
                                              stimulus_config=env.stimulus_config,
                                              arena=arena,
                                              distance=norm_u_arc_distance,
                                              local_random=local_random,
                                              noise_gen_dict=noise_gen_dict,
                                              comm=env.comm)
    
    this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
    selectivity_attr_dict = input_cell_config.get_selectivity_attr_dict()
    rate_map = input_cell_config.get_rate_map(x=arena_x, y=arena_y)

    if debug and rank == 0:
        callback, context = debug
        this_context = Struct(**dict(context()))
        this_context.update(dict(locals()))
        callback(this_context)
        
    if rate_map_sum is not None:
        rate_map_sum[this_selectivity_type_name] += rate_map

    return this_selectivity_type_name, selectivity_attr_dict


def read_stimulus(stimulus_path, stimulus_namespace, population, module=None):
    ratemap_lst = []
    module_gid_set = set()
    if module is not None:
        if not isinstance(module, int):
            raise Exception('module variable must be an integer')
        gid_module_gen = read_cell_attributes(stimulus_path, population, namespace='Cell Attributes')
        for (gid, attr_dict) in gid_module_gen:
            this_module = attr_dict['Module'][0]
            if this_module == module:
                module_gid_set.add(gid)

    attr_gen = read_cell_attributes(stimulus_path, population, namespace=stimulus_namespace)
    for gid, stimulus_dict in attr_gen:
        if gid in module_gid_set or len(module_gid_set) == 0:
            rate = stimulus_dict['Trajectory Rate Map']
            spiketrain = stimulus_dict['Spike Train']
            peak_index = np.where(rate == np.max(rate))[0][0]
            ratemap_lst.append((gid, rate, spiketrain, peak_index))

    ## sort by peak_index
    ratemap_lst.sort(key=lambda item: item[3])
    return ratemap_lst


def read_feature(feature_path, feature_namespace, population):
    feature_lst = []

    attr_gen = read_cell_attributes(feature_path, population, namespace=feature_namespace)
    for gid, feature_dict in attr_gen:
        if 'Module ID' in feature_dict:
            gid_module = feature_dict['Module ID'][0]
        else:
            gid_module = None
        rate = feature_dict['Arena Rate Map']
        feature_lst.append((gid, rate, gid_module))

    return feature_lst


def bin_stimulus_features(features, t, bin_size, time_range):
    """
    Continuous stimulus feature binning.

    Parameters
    ----------
    features: matrix of size "number of times each feature was recorded" x "number of features"
    t: a vector of size "number of times each feature was recorded"
    bin_size: size of time bins
    time_range: the start and end times for binning the stimulus


    Returns
    -------
    matrix of size "number of time bins" x "number of features in the output"
        the average value of each output feature in every time bin
    """

    t_start, t_end = time_range

    edges = np.arange(t_start, t_end, bin_size)
    nbins = edges.shape[0] - 1
    nfeatures = features.shape[1]
    binned_features = np.empty([nbins, nfeatures])
    for i in range(nbins):
        for j in range(nfeatures):
            delta = edges[i + 1] - edges[i]
            bin_range = np.arange(edges[i], edges[i + 1], delta / 5.)
            ip_vals = np.interp(bin_range, t, features[:, j])
            binned_features[i, j] = np.mean(ip_vals)

    return binned_features




def rate_maps_from_features (env, population, cell_index_set, input_features_path=None, input_features_namespace=None, 
                             input_features_dict=None, arena_id=None, trajectory_id=None, time_range=None,
                             include_time=False, phase_mod_config=None):
    
    """Initializes presynaptic spike sources from a file with input selectivity features represented as firing rates."""

    if input_features_dict is not None:
        if (input_features_path is not None) or  (input_features_namespace is not None):
            raise RuntimeError("rate_maps_from_features: when input_features_dict is provided, input_features_path must be None")
    else:
        if (input_features_path is None) or  (input_features_namespace is None):
            raise RuntimeError("rate_maps_from_features: either input_features_dict has to be provided, or input_features_path and input_features_namespace")
    
    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    if arena_id is None:
        arena_id = env.arena_id
    if trajectory_id is None:
        trajectory_id = env.trajectory_id

    spatial_resolution = float(env.stimulus_config['Spatial Resolution'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])

    input_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate',
                                 'Module ID', 'Grid Spacing', 'Grid Orientation',
                                 'Field Width Concentration Factor', 
                                 'X Offset', 'Y Offset']
    
    selectivity_type_names = { i: n for n, i in viewitems(env.selectivity_types) }

    arena = env.stimulus_config['Arena'][arena_id]
    
    trajectory = arena.trajectories[trajectory_id]
    equilibration_duration = float(env.stimulus_config.get('Equilibration Duration', 0.))

    t, x, y, d = generate_linear_trajectory(trajectory, temporal_resolution=temporal_resolution,
                                            equilibration_duration=equilibration_duration)
    if time_range is not None:
        t_range_inds = np.where((t < time_range[1]) & (t >= time_range[0]))[0] 
        t = t[t_range_inds]
        x = x[t_range_inds]
        y = y[t_range_inds]
        d = d[t_range_inds]

    input_rate_map_dict = {}

    if len(d) == 0:
        return input_rate_map_dict

    abs_d = d - d[0]
    abs_t = (t - t[0])/1000.
    velocity = np.insert(abs_d[1:]/abs_t[1:], 0, abs_d[1]/abs_t[1])

    pop_index = int(env.Populations[population])

    if input_features_path is not None:
        this_input_features_namespace = f'{input_features_namespace} {arena_id}'
        input_features_iter = scatter_read_cell_attribute_selection(input_features_path, population,
                                                                    selection=cell_index_set,
                                                                    namespace=this_input_features_namespace,
                                                                    mask=set(input_features_attr_names), 
                                                                    comm=env.comm, io_size=env.io_size)
    else:
        input_features_iter = viewitems(input_features_dict)
        
    for gid, selectivity_attr_dict in input_features_iter:
        
        this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        this_selectivity_type_name = selectivity_type_names[this_selectivity_type]

        input_cell_config = get_input_cell_config(selectivity_type=this_selectivity_type,
                                                  selectivity_type_names=selectivity_type_names,
                                                  selectivity_attr_dict=selectivity_attr_dict,
                                                  phase_mod_config=phase_mod_config)
        rate_map = input_cell_config.get_rate_map(x=x, y=y, velocity=velocity if phase_mod_config is not None else None)
        rate_map[np.isclose(rate_map, 0., atol=1e-3, rtol=1e-3)] = 0.

        if include_time:
            input_rate_map_dict[gid] = (t, rate_map)
        else:
            input_rate_map_dict[gid] = rate_map
            
    return input_rate_map_dict


def arena_rate_maps_from_features (env, population, input_features_path, input_features_namespace, cell_index_set,
                                   arena_id=None, time_range=None, n_trials=1):
    
    """Initializes presynaptic spike sources from a file with input selectivity features represented as firing rates."""
        
    if time_range is not None:
        if time_range[0] is None:
            time_range[0] = 0.0

    if arena_id is None:
        arena_id = env.arena_id

    spatial_resolution = float(env.stimulus_config['Spatial Resolution'])
    temporal_resolution = float(env.stimulus_config['Temporal Resolution'])
    
    this_input_features_namespace = f'{input_features_namespace} {arena_id}'
    
    input_features_attr_names = ['Selectivity Type', 'Num Fields', 'Field Width', 'Peak Rate',
                                 'Module ID', 'Grid Spacing', 'Grid Orientation',
                                 'Field Width Concentration Factor', 
                                 'X Offset', 'Y Offset']
    
    selectivity_type_names = { i: n for n, i in viewitems(env.selectivity_types) }

    arena = env.stimulus_config['Arena'][arena_id]
    arena_x, arena_y = get_2D_arena_spatial_mesh(arena=arena, spatial_resolution=spatial_resolution)
    
    input_rate_map_dict = {}
    pop_index = int(env.Populations[population])

    input_features_iter = scatter_read_cell_attribute_selection(input_features_path, population,
                                                                selection=cell_index_set,
                                                                namespace=this_input_features_namespace,
                                                                mask=set(input_features_attr_names), 
                                                                comm=env.comm, io_size=env.io_size)
    for gid, selectivity_attr_dict in input_features_iter:

        this_selectivity_type = selectivity_attr_dict['Selectivity Type'][0]
        this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
        input_cell_config = get_input_cell_config(population=population,
                                                  selectivity_type=this_selectivity_type,
                                                  selectivity_type_names=selectivity_type_names,
                                                  selectivity_attr_dict=selectivity_attr_dict)
        if input_cell_config.num_fields > 0:
            rate_map = input_cell_config.get_rate_map(x=arena_x, y=arena_y)
            rate_map[np.isclose(rate_map, 0., atol=1e-3, rtol=1e-3)] = 0.
            input_rate_map_dict[gid] = rate_map
            
    return input_rate_map_dict




def oscillation_phase_mod_config(env, population, soma_positions, local_random=None):
    """
    Obtains phase modulation configuration for a given neuronal population.
    """
    global_oscillation_config = env.stimulus_config['Global Oscillation']
    frequency = global_oscillation_config['frequency']
    phase_mod_config = global_oscillation_config['Phase Modulation'][population]
    phase_range = phase_mod_config['phase range']
    mod_depth = phase_mod_config['depth']

    population_phase_prefs = global_oscillation_phase_pref(env, population,
                                                           num_cells=len(soma_positions),
                                                           local_random=local_random)
    position_array = np.asarray([ soma_positions[k] for k in sorted(soma_positions) ])
    population_phase_shifts = global_oscillation_phase_shift(env, position_array)
        
    phase_mod_config_dict = {}
    for i, gid in enumerate(sorted(soma_positions)):
        phase_mod_config_dict[gid] = PhaseModConfig(phase_range,
                                                    population_phase_prefs[i],
                                                    population_phase_shifts[i],
                                                    mod_depth, frequency)

    return phase_mod_config_dict
    
def global_oscillation_phase_shift(env, position):
    """
    Computes the phase shift of the global oscillatory signal for the given position, assumed to be on the long axis. 
    Uses the "Global Oscillation" entry in the input configuration. 
    See `global_oscillation_signal` for a description of the configuration format.
    """

    global_oscillation_config = env.stimulus_config['Global Oscillation']
    phase_dist_config = global_oscillation_config['Phase Distribution']
    phase_slope = phase_dist_config['slope']
    phase_offset = phase_dist_config['offset']
    x = position / 1000.

    return x*phase_slope + phase_offset


def global_oscillation_phase_pref(env, population, num_cells, local_random=None):
    """
    Computes oscillatory phase preferences for all cells in the given population.
    Uses the "Global Oscillation" entry in the input configuration. 
    See `global_oscillation_signal` for a description of the configuration format.

    Returns: an array of phase preferences of length equal to the population size.
    """

    seed = int(env.model_config['Random Seeds']['Phase Preference'])

    if local_random is None:
        local_random = np.random.RandomState(seed)
    
    global_oscillation_config = env.stimulus_config['Global Oscillation']
    phase_mod_config = global_oscillation_config['Phase Modulation'][population]
    phase_range = phase_mod_config['phase range']
    phase_loc = (phase_range[1] - phase_range[0]) / 2.
    fw = 2. * np.sqrt(2. * np.log(100.))
    phase_scale = (phase_range[1] - phase_range[0]) / fw
    s = local_random.normal(loc=phase_loc, scale=phase_scale, size=num_cells)
    s = np.mod(s, 360)
    
    return s


def global_oscillation_initial_phases(env, n_trials, local_random=None):
    """
    Computes initial oscillatory phases for multiple trials.
    Uses the "Global Oscillation" entry in the input configuration. 
    See `global_oscillation_signal` for a description of the configuration format.

    Returns: an array of phases in radians of length equal to n_trials.
    """

    seed = int(env.model_config['Random Seeds']['Initial Phase'])

    if local_random is None:
        local_random = np.random.RandomState(seed)

    s = [0.]
    if n_trials > 1:
        for i in range(n_trials-1):
            s.append(local_random.uniform(0.0, 360.0))

    a = np.deg2rad(np.asarray(s))
    
    return a

    

def stationary_phase_mod(t, phase_range, phase_pref, phase_offset, mod_depth, freq):
    """
    Computes stationary oscillatory phase modulation with the given parameters.
    """

    r = phase_range[1] - phase_range[0]
    delta = 2*np.pi - np.deg2rad(phase_pref)
    s = np.cos(2*np.pi*freq*(t) - np.deg2rad(phase_offset) + delta) + 1

    d = np.clip(mod_depth, 0., 1.)
    mod = s*mod_depth/2. + (1. - mod_depth)

    return mod


def spatial_phase_mod(x, velocity, field_width, phase_range, phase_entry, phase_offset, mod_depth, freq):
    ''' Non-stationary phase modulation for spatial receptive fields.
        Calculates modulation according to the equation:

         s = cos(r*x/field_width + 2*pi*freq*x/velocity - phase_entry - phase_offset + r/2.) + 1
         mod =  s*mod_depth/2. + (1. - mod_depth)

        - position: spatial position
        - velocity: movement velocity
        - field_width: receptive field
        - phase_range: range of preferred phase
        - phase_entry: receptive field entry phase 
        - mod_depth: modulation depth
        - freq: frequency of global oscillation
    '''
    r = np.deg2rad(phase_range[1] - phase_range[0])
    s = np.cos(r*x/field_width + 2*np.pi*freq*x/velocity - np.deg2rad(phase_entry + phase_offset) + r/2.) + 1
    d = np.clip(mod_depth, 0., 1.)
    m = s*mod_depth/2. + (1. - mod_depth)

    return m


def spatial2d_phase_mod(x, y, velocity, field_width, phase_range, phase_entry, phase_offset, mod_depth, freq):

    x_mod = spatial_phase_mod(x, velocity, field_width, phase_range, phase_entry, phase_offset, 
                              mod_depth, freq)
    y_mod = spatial_phase_mod(y, velocity, field_width, phase_range, phase_entry, phase_offset, 
                              mod_depth, freq)

    m = (x_mod + y_mod) / 2.

    return m

