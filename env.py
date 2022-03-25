
import os, pprint, logging
from collections import defaultdict, namedtuple
import numpy as np
from mpi4py import MPI
import yaml
from neuroh5.io import read_cell_attribute_info, read_population_names, read_population_ranges, read_projection_names
from biophys_microcircuit.utils import IncludeLoader, ExprClosure, get_root_logger, str, viewitems, zip, read_from_yaml
from dentate.synapses import SynapseAttributes, get_syn_filter_dict

SynapseConfig = namedtuple('SynapseConfig',
                           ['type',
                            'sections',
                            'layers',
                            'proportions',
                            'contacts',
                            'mechanisms'])

GapjunctionConfig = namedtuple('GapjunctionConfig',
                               ['sections',
                                'connection_probability',
                                'connection_parameters',
                                'connection_bounds',
                                'coupling_coefficients',
                                'coupling_parameters',
                                'coupling_bounds'])

NetclampConfig = namedtuple('NetclampConfig',
                            ['template_params',
                             'weight_generators',
                             'optimize_parameters'])

ArenaConfig = namedtuple('Arena',
                         ['name',
                          'domain',
                          'trajectories',
                          'properties'])

DomainConfig = namedtuple('Domain',
                          ['vertices',
                           'simplices'])

TrajectoryConfig = namedtuple('Trajectory',
                              ['velocity',
                               'path'])



class Env(object):
    """
    Network model configuration.
    """

    def __init__(self, comm=None, config_file=None, template_paths="templates", hoc_lib_path=None,
                 dataset_prefix=None, config_prefix=None,
                 results_path=None, results_file_id=None, results_namespace_id=None, 
                 node_rank_file=None, node_allocation=None, io_size=0, recording_profile=None, recording_fraction=1.0,
                 tstop=0., v_init=-65, stimulus_onset=0.0, n_trials=1,
                 max_walltime_hours=0.5, checkpoint_interval=500.0, checkpoint_clear_data=True, 
                 results_write_time=0, dt=None, ldbal=False, lptbal=False, 
                 cell_selection_path=None, microcircuit_inputs=False,
                 spike_input_path=None, spike_input_namespace=None, spike_input_attr=None,
                 cleanup=True, cache_queries=False, profile_memory=False, use_coreneuron=False,
                 transfer_debug=False, verbose=False, **kwargs):
        """
        :param comm: :class:'MPI.COMM_WORLD'
        :param config_file: str; model configuration file name
        :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
        :param hoc_lib_path: str; path to directory containing required hoc libraries
        :param dataset_prefix: str; path to directory containing required neuroh5 data files
        :param config_prefix: str; path to directory containing network and cell mechanism config files
        :param results_path: str; path to directory to export output files
        :param results_file_id: str; label for neuroh5 files to write spike and voltage trace data
        :param results_namespace_id: str; label for neuroh5 namespaces to write spike and voltage trace data
        :param node_rank_file: str; name of file specifying assignment of node gids to MPI ranks
        :param node_allocation: iterable; gids assigned to the current MPI ranks; cannot be specified together with node_rank_file
        :param io_size: int; the number of MPI ranks to be used for I/O operations
        :param recording_profile: str; intracellular recording configuration to use
        :param tstop: int; physical time to simulate (ms)
        :param v_init: float; initialization membrane potential (mV)
        :param stimulus_onset: float; starting time of stimulus (ms)
        :param max_walltime_hours: float; maximum wall time (hours)
        :param results_write_time: float; time to write out results at end of simulation
        :param dt: float; simulation time step
        :param ldbal: bool; estimate load balance based on cell complexity
        :param lptbal: bool; calculate load balance with LPT algorithm
        :param cleanup: bool; clean up auxiliary cell and synapse structures after network init
        :param profile: bool; profile memory usage
        :param cache_queries: bool; whether to use a cache to speed up queries to filter_synapses
        :param verbose: bool; print verbose diagnostic messages while constructing the network
        """
        self.kwargs = kwargs
        
        self.SWC_Types = {}
        self.SWC_Type_index = {}
        self.Synapse_Types = {}
        self.Synapse_Type_index = {}
        self.layers = {}
        self.layer_type_index = {}
        self.globals = {}

        self.gidset = set([])
        self.gjlist = []
        self.cells = defaultdict(lambda: dict())
        self.artificial_cells = defaultdict(lambda: dict())
        self.biophys_cells = defaultdict(lambda: dict())
        self.spike_onset_delay = {}
        self.recording_sets = {}

        self.pc = None
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        rank = self.comm.Get_rank()

        if rank == 0:
            color = 1
        else:
            color = 0
        ## comm0 includes only rank 0
        comm0 = self.comm.Split(color, 0)

        self.use_coreneuron = use_coreneuron

        # If true, the biophysical cells and synapses dictionary will be freed
        # as synapses and connections are instantiated.
        self.cleanup = cleanup

        # If true, compute and print memory usage at various points
        # during simulation initialization
        self.profile_memory = profile_memory

        # print verbose diagnostic messages
        self.verbose = verbose
        self.logger = get_root_logger()
        if self.verbose:
            self.logger.setLevel(logging.INFO)

        # Directories for cell templates
        if template_paths is not None:
            self.template_paths = template_paths.split(':')
        else:
            self.template_paths = []
        self.template_dict = {}
            
        # The location of required hoc libraries
        self.hoc_lib_path = hoc_lib_path

        # Checkpoint interval in ms of simulation time
        self.checkpoint_clear_data = checkpoint_clear_data
        self.last_checkpoint = 0.
        if checkpoint_interval > 0.:
            self.checkpoint_interval = max(float(checkpoint_interval), 1.0)
        else:
            self.checkpoint_interval = None
        
        # The location of all datasets
        self.dataset_prefix = dataset_prefix

        # The path where results files should be written
        self.results_path = results_path

        # Identifier used to construct results data namespaces
        self.results_namespace_id = results_namespace_id
        # Identifier used to construct results data files
        self.results_file_id = results_file_id

        # Number of MPI ranks to be used for I/O operations
        self.io_size = int(io_size)

        # Initialization voltage
        self.v_init = float(v_init)

        # simulation time [ms]
        self.tstop = float(tstop)

        # stimulus onset time [ms]
        self.stimulus_onset = float(stimulus_onset)

        # number of trials
        self.n_trials = int(n_trials)

        # maximum wall time in hours
        self.max_walltime_hours = float(max_walltime_hours)

        # time to write out results at end of simulation
        self.results_write_time = float(results_write_time)

        # time step
        self.dt = float(dt if dt is not None else 0.025)

        # used to estimate cell complexity
        self.cxvec = None

        # measure/perform load balancing
        self.optldbal = ldbal
        self.optlptbal = lptbal

        self.transfer_debug = transfer_debug
            
        # cache queries to filter_synapses
        self.cache_queries = cache_queries

        self.config_prefix = config_prefix
        if config_file is not None:
            if config_prefix is not None:
                config_file_path = self.config_prefix + '/' + config_file
            else:
                config_file_path = config_file
            self.model_config = None
            if rank == 0:
                if not os.path.isfile(config_file_path):
                    raise RuntimeError("configuration file %s was not found" % config_file_path)
                with open(config_file_path) as fp:
                    self.model_config = yaml.load(fp, IncludeLoader)
            self.model_config = self.comm.bcast(self.model_config, root=0)
        else:
            raise RuntimeError("missing configuration file")

        if 'Definitions' in self.model_config:
            self.parse_definitions()
            self.SWC_Type_index = dict([(item[1], item[0]) for item in viewitems(self.SWC_Types)])
            self.Synapse_Type_index = dict([(item[1], item[0]) for item in viewitems(self.Synapse_Types)])
            self.layer_type_index = dict([(item[1], item[0]) for item in viewitems(self.layers)])

            
        if 'Global Parameters' in self.model_config:
            self.parse_globals()

        self.geometry = None
        if 'Geometry' in self.model_config:
            self.geometry = self.model_config['Geometry']

        if 'Origin' in self.geometry['Parametric Surface']:
            self.parse_origin_coords()

        self.celltypes = self.model_config['Cell Types']
        self.cell_attribute_info = {}

        # The name of this model
        if 'Model Name' in self.model_config:
            self.modelName = self.model_config['Model Name']
        # The dataset to use for constructing the network
        if 'Dataset Name' in self.model_config:
            self.datasetName = self.model_config['Dataset Name']

        if rank == 0:
            self.logger.info('env.dataset_prefix = %s' % str(self.dataset_prefix))

        # Cell selection for simulations of subsets of the network
        self.cell_selection = None
        self.cell_selection_path = cell_selection_path
        if rank == 0:
            self.logger.info('env.cell_selection_path = %s' % str(self.cell_selection_path))
            if cell_selection_path is not None:
                with open(cell_selection_path) as fp:
                    self.cell_selection = yaml.load(fp, IncludeLoader)
        self.cell_selection = self.comm.bcast(self.cell_selection, root=0)
        

        # Spike input path
        self.spike_input_path = spike_input_path
        self.spike_input_ns = spike_input_namespace
        self.spike_input_attr = spike_input_attr
        self.spike_input_attribute_info = None
        if self.spike_input_path is not None:
            if rank == 0:
                self.logger.info('env.spike_input_path = %s' % str(self.spike_input_path))
                self.spike_input_attribute_info = \
                     read_cell_attribute_info(self.spike_input_path, sorted(self.Populations.keys()), comm=comm0)
                self.logger.info('env.spike_input_attribute_info = %s' % str(self.spike_input_attribute_info))
            self.spike_input_attribute_info = self.comm.bcast(self.spike_input_attribute_info, root=0)

        if results_path:
            if self.results_file_id is None:
                self.results_file_path = "%s/%s_results.h5" % (self.results_path, self.modelName)
            else:
                self.results_file_path = "%s/%s_results_%s.h5" % (self.results_path, self.modelName, self.results_file_id)
        else:
            if self.results_file_id is None:
                self.results_file_path = "%s_results.h5" % (self.modelName)
            else:
                self.results_file_path = "%s_results_%s.h5" % (self.modelName, self.results_file_id)
            
        if 'Connection Generator' in self.model_config:
            self.parse_connection_config()
            self.parse_gapjunction_config()

        if self.dataset_prefix is not None:
            self.dataset_path = os.path.join(self.dataset_prefix, self.datasetName)
            if 'Cell Data' in self.model_config:
                self.data_file_path = os.path.join(self.dataset_path, self.model_config['Cell Data'])
                self.forest_file_path = os.path.join(self.dataset_path, self.model_config['Cell Data'])
                self.load_celltypes()
            else:
                self.data_file_path = None
                self.forest_file_path = None
            if rank == 0:
                self.logger.info('env.data_file_path = %s' % self.data_file_path)
            if 'Connection Data' in self.model_config:		
                self.connectivity_file_path = os.path.join(self.dataset_path, self.model_config['Connection Data'])
            else:
                self.connectivity_file_path = None
            if 'Gap Junction Data' in self.model_config:
                self.gapjunctions_file_path = os.path.join(self.dataset_path, self.model_config['Gap Junction Data'])
            else:
                self.gapjunctions_file_path = None
        else:
            self.dataset_path = None
            self.data_file_path = None
            self.connectivity_file_path = None
            self.forest_file_path = None
            self.gapjunctions_file_path = None

        self.node_allocation = None
        if node_rank_file and node_allocation:
            raise RuntimeError("Only one of node_rank_file and node_allocation must be specified.")
        if node_rank_file:
            self.load_node_rank_map(node_rank_file)
        if node_allocation:
            self.node_allocation = set(node_allocation)

        self.netclamp_config = None
        if 'Network Clamp' in self.model_config:
            self.parse_netclamp_config()

        self.stimulus_config = None
        self.arena_id = None
        self.trajectory_id = None
        if 'Stimulus' in self.model_config:
            self.parse_stimulus_config()
            self.init_stimulus_config(**kwargs)
            
        self.analysis_config = None
        if 'Analysis' in self.model_config:
            self.analysis_config = self.model_config['Analysis']

        self.projection_dict = None
        if self.dataset_prefix is not None:
            if rank == 0:
                projection_dict = defaultdict(list)
                self.logger.info('env.connectivity_file_path = %s' % str(self.connectivity_file_path))
                if self.connectivity_file_path is not None:
                    for (src, dst) in read_projection_names(self.connectivity_file_path, comm=comm0):
                        projection_dict[dst].append(src)
                self.projection_dict = dict(projection_dict)
                self.logger.info('projection_dict = %s' % str(self.projection_dict))
            self.projection_dict = self.comm.bcast(self.projection_dict, root=0)

        # If True, instantiate as spike source those cells that do not
        # have data in the input data file
        self.microcircuit_inputs = microcircuit_inputs or (self.cell_selection is not None)
        self.microcircuit_input_sources = { pop_name: set([]) for pop_name in self.celltypes.keys() }
            
        # Configuration profile for recording intracellular quantities
        assert((recording_fraction >= 0.0) and (recording_fraction <= 1.0))
        self.recording_fraction = recording_fraction
        self.recording_profile = None
        if ('Recording' in self.model_config) and (recording_profile is not None):
            self.recording_profile = self.model_config['Recording']['Intracellular'][recording_profile]
            self.recording_profile['label'] = recording_profile
            for recvar, recdict  in viewitems(self.recording_profile.get('synaptic quantity', {})):
                filters = {}
                if 'syn types' in recdict:
                    filters['syn_types'] = recdict['syn types']
                if 'swc types' in recdict:
                    filters['swc_types'] = recdict['swc types']
                if 'layers' in recdict:
                    filters['layers'] = recdict['layers']
                if 'sources' in recdict:
                    filters['sources'] = recdict['sources']
                syn_filters = get_syn_filter_dict(self, filters, convert=True)
                recdict['syn_filters'] = syn_filters
        
            if self.use_coreneuron:
                self.recording_profile['dt'] = None

        # Configuration profile for recording local field potentials
        self.LFP_config = {}
        if 'Recording' in self.model_config:
            for label, config in viewitems(self.model_config['Recording']['LFP']):
                self.LFP_config[label] = {'position': tuple(config['position']),
                                         'maxEDist': config['maxEDist'],
                                         'fraction': config['fraction'],
                                         'rho': config['rho'],
                                         'dt': config['dt']}


        self.t_vec = None
        self.id_vec = None
        self.t_rec = None
        self.recs_dict = {}  # Intracellular samples on this host
        self.recs_count = 0
        for pop_name, _ in viewitems(self.Populations):
            self.recs_dict[pop_name] = defaultdict(list)
            
        # used to calculate model construction times and run time
        self.mkcellstime = 0
        self.mkstimtime = 0
        self.connectcellstime = 0
        self.connectgjstime = 0

        self.simtime = None
        self.lfp = {}

        self.edge_count = defaultdict(dict)
        self.syns_set = defaultdict(set)
        
        comm0.Free()


    def parse_arena_domain(self, config):
        vertices = config['vertices']
        simplices = config['simplices']

        return DomainConfig(vertices, simplices)

    def parse_arena_trajectory(self, config):
        velocity = float(config['run velocity'])
        path_config = config['path']

        path_x = []
        path_y = []
        for v in path_config:
            path_x.append(v[0])
            path_y.append(v[1])

        path = np.column_stack((np.asarray(path_x, dtype=np.float32),
                                np.asarray(path_y, dtype=np.float32)))

        return TrajectoryConfig(velocity, path)

    def init_stimulus_config(self, arena_id=None, trajectory_id=None, **kwargs):
        if arena_id is not None:
            if arena_id in self.stimulus_config['Arena']:
                self.arena_id = arena_id
            else:
                raise RuntimeError('init_stimulus_config: arena id parameter not found in stimulus configuration')
            if trajectory_id is None:
                self.trajectory_id = None
            else:
                if trajectory_id in self.stimulus_config['Arena'][arena_id].trajectories:
                    self.trajectory_id = trajectory_id
                else:
                    raise RuntimeError('init_stimulus_config: trajectory id parameter not found in stimulus configuration')

    def parse_stimulus_config(self):
        stimulus_dict = self.model_config['Stimulus']
        stimulus_config = {}

        for k, v in viewitems(stimulus_dict):
            if k == 'Selectivity Type Probabilities':
                selectivity_type_prob_dict = {}
                for (pop, dvals) in viewitems(v):
                    pop_selectivity_type_prob_dict = {}
                    for (selectivity_type_name, selectivity_type_prob) in viewitems(dvals):
                        pop_selectivity_type_prob_dict[int(self.selectivity_types[selectivity_type_name])] = \
                            float(selectivity_type_prob)
                    selectivity_type_prob_dict[pop] = pop_selectivity_type_prob_dict
                stimulus_config['Selectivity Type Probabilities'] = selectivity_type_prob_dict
            elif k == 'Peak Rate':
                peak_rate_dict = {}
                for (pop, dvals) in viewitems(v):
                    pop_peak_rate_dict = {}
                    for (selectivity_type_name, peak_rate) in viewitems(dvals):
                        pop_peak_rate_dict[int(self.selectivity_types[selectivity_type_name])] = float(peak_rate)
                    peak_rate_dict[pop] = pop_peak_rate_dict
                stimulus_config['Peak Rate'] = peak_rate_dict
            elif k == 'Arena':
                stimulus_config['Arena'] = {}
                for arena_id, arena_val in viewitems(v):
                    arena_properties = {}
                    arena_domain = None
                    arena_trajectories = {}
                    for kk, vv in viewitems(arena_val):
                        if kk == 'Domain':
                            arena_domain = self.parse_arena_domain(vv)
                        elif kk == 'Trajectory':
                            for name, trajectory_config in viewitems(vv):
                                trajectory = self.parse_arena_trajectory(trajectory_config)
                                arena_trajectories[name] = trajectory
                        else:
                            arena_properties[kk] = vv
                    stimulus_config['Arena'][arena_id] = ArenaConfig(arena_id, arena_domain,
                                                                  arena_trajectories, arena_properties)
            else:
                stimulus_config[k] = v

        self.stimulus_config = stimulus_config

        
    def parse_netclamp_config(self):
        """

        :return:
        """
        netclamp_config_dict = self.model_config['Network Clamp']
        weight_generator_dict = netclamp_config_dict['Weight Generator']
        template_param_rules_dict = netclamp_config_dict['Template Parameter Rules']
        opt_param_rules_dict = {}
        if 'Synaptic Optimization' in netclamp_config_dict:
            opt_param_rules_dict['synaptic'] = netclamp_config_dict['Synaptic Optimization']

        template_params = {}
        for (template_name, params) in viewitems(template_param_rules_dict):
            template_params[template_name] = params

        self.netclamp_config = NetclampConfig(template_params,
                                              weight_generator_dict,
                                              opt_param_rules_dict)

    def parse_origin_coords(self):
        origin_spec = self.geometry['Parametric Surface']['Origin']

        coords = {}
        for key in ['U', 'V', 'L']:
            spec = origin_spec[key]
            if isinstance(spec, float):
                coords[key] = lambda x: spec
            elif spec == 'median':
                coords[key] = lambda x: np.median(x)
            elif spec == 'mean':
                coords[key] = lambda x: np.mean(x)
            elif spec == 'min':
                coords[key] = lambda x: np.min(x)
            elif spec == 'max':
                coords[key] = lambda x: np.max(x)
            else:
                raise ValueError
        self.geometry['Parametric Surface']['Origin'] = coords

    def parse_definitions(self):
        defs = self.model_config['Definitions']
        self.Populations = defs['Populations']
        self.SWC_Types = defs['SWC Types']
        self.Synapse_Types = defs['Synapse Types']
        self.layers = defs['Layers']
        self.selectivity_types = defs['Input Selectivity Types']

    def parse_globals(self):
        self.globals = self.model_config['Global Parameters']

    def parse_syn_mechparams(self, mechparams_dict):
        res = {}
        for mech_name, mech_params in viewitems(mechparams_dict):
            mech_params1 = {}
            for k, v in viewitems(mech_params):
                if isinstance(v, dict):
                    if 'expr' in v:
                        mech_params1[k] = ExprClosure([v['parameter']], v['expr'],
                                                       v.get('const', None), ['x'])
                    else:
                        raise RuntimeError('parse_syn_mechparams: unknown parameter type %s' % str(v))
                else:
                    mech_params1[k] = v
            res[mech_name] = mech_params1
        return res

    def parse_connection_config(self):
        """

        :return:
        """
        connection_config = self.model_config['Connection Generator']

        self.connection_velocity = connection_config['Connection Velocity']

        syn_mech_names = connection_config['Synapse Mechanisms']
        syn_param_rules = connection_config['Synapse Parameter Rules']

        self.synapse_attributes = SynapseAttributes(self, syn_mech_names, syn_param_rules)

        extent_config = connection_config['Axon Extent']
        self.connection_extents = {}

        for population in extent_config:

            pop_connection_extents = {}
            for layer_name in extent_config[population]:

                if layer_name == 'default':
                    pop_connection_extents[layer_name] = \
                        {'width': extent_config[population][layer_name]['width'], \
                         'offset': extent_config[population][layer_name]['offset']}
                else:
                    layer_index = self.layers[layer_name]
                    pop_connection_extents[layer_index] = \
                        {'width': extent_config[population][layer_name]['width'], \
                         'offset': extent_config[population][layer_name]['offset']}

            self.connection_extents[population] = pop_connection_extents

        synapse_config = connection_config['Synapses']
        connection_dict = {}

        for (key_postsyn, val_syntypes) in viewitems(synapse_config):
            connection_dict[key_postsyn] = {}

            for (key_presyn, syn_dict) in viewitems(val_syntypes):
                val_type = syn_dict['type']
                val_synsections = syn_dict['sections']
                val_synlayers = syn_dict['layers']
                val_proportions = syn_dict['proportions']
                if 'contacts' in syn_dict:
                    val_contacts = syn_dict['contacts']
                else:
                    val_contacts = 1
                mechparams_dict = None
                swctype_mechparams_dict = None
                if 'mechanisms' in syn_dict:
                    mechparams_dict = syn_dict['mechanisms']
                else:
                    swctype_mechparams_dict = syn_dict['swctype mechanisms']

                res_type = self.Synapse_Types[val_type]
                res_synsections = []
                res_synlayers = []
                res_mechparams = {}

                for name in val_synsections:
                    res_synsections.append(self.SWC_Types[name])
                for name in val_synlayers:
                    res_synlayers.append(self.layers[name])
                if swctype_mechparams_dict is not None:
                    for swc_type in swctype_mechparams_dict:
                        swc_type_index = self.SWC_Types[swc_type]
                        res_mechparams[swc_type_index] = self.parse_syn_mechparams(swctype_mechparams_dict[swc_type])
                else:
                    res_mechparams['default'] = self.parse_syn_mechparams(mechparams_dict)

                connection_dict[key_postsyn][key_presyn] = \
                    SynapseConfig(res_type, res_synsections, res_synlayers, val_proportions, val_contacts, \
                                  res_mechparams)

            config_dict = defaultdict(lambda: 0.0)
            for (key_presyn, conn_config) in viewitems(connection_dict[key_postsyn]):
                for (s, l, p) in zip(conn_config.sections, conn_config.layers, conn_config.proportions):
                    config_dict[(conn_config.type, s, l)] += p

            for (k, v) in viewitems(config_dict):
                try:
                    assert (np.isclose(v, 1.0))
                except Exception as e:
                    self.logger.error(f'Connection configuration: probabilities for {key_postsyn} do not sum to 1: type: {self.Synapse_Type_index[k[0]]} section: {self.SWC_Type_index[k[1]]}  layer {self.layer_type_index[k[2]]} = {v}')
                    raise e

        self.connection_config = connection_dict

    def parse_gapjunction_config(self):
        """

        :return:
        """
        connection_config = self.model_config['Connection Generator']
        if 'Gap Junctions' in connection_config:
            gj_config = connection_config['Gap Junctions']

            gj_sections = gj_config['Locations']
            sections = {}
            for pop_a, pop_dict in viewitems(gj_sections):
                for pop_b, sec_names in viewitems(pop_dict):
                    pair = (pop_a, pop_b)
                    sec_idxs = []
                    for sec_name in sec_names:
                        sec_idxs.append(self.SWC_Types[sec_name])
                    sections[pair] = sec_idxs

            gj_connection_probs = gj_config['Connection Probabilities']
            connection_probs = {}
            for pop_a, pop_dict in viewitems(gj_connection_probs):
                for pop_b, prob in viewitems(pop_dict):
                    pair = (pop_a, pop_b)
                    connection_probs[pair] = float(prob)

            connection_weights_x = []
            connection_weights_y = []
            gj_connection_weights = gj_config['Connection Weights']
            for x in sorted(gj_connection_weights.keys()):
                connection_weights_x.append(x)
                connection_weights_y.append(gj_connection_weights[x])

            connection_params = np.polyfit(np.asarray(connection_weights_x), \
                                           np.asarray(connection_weights_y), \
                                           3)
            connection_bounds = [np.min(connection_weights_x), \
                                 np.max(connection_weights_x)]

            gj_coupling_coeffs = gj_config['Coupling Coefficients']
            coupling_coeffs = {}
            for pop_a, pop_dict in viewitems(gj_coupling_coeffs):
                for pop_b, coeff in viewitems(pop_dict):
                    pair = (pop_a, pop_b)
                    coupling_coeffs[pair] = float(coeff)

            gj_coupling_weights = gj_config['Coupling Weights']
            coupling_weights_x = []
            coupling_weights_y = []
            for x in sorted(gj_coupling_weights.keys()):
                coupling_weights_x.append(x)
                coupling_weights_y.append(gj_coupling_weights[x])

            coupling_params = np.polyfit(np.asarray(coupling_weights_x), \
                                         np.asarray(coupling_weights_y), \
                                         3)
            coupling_bounds = [np.min(coupling_weights_x), \
                               np.max(coupling_weights_x)]
            coupling_params = coupling_params
            coupling_bounds = coupling_bounds

            self.gapjunctions = {}
            for pair, sec_idxs in viewitems(sections):
                self.gapjunctions[pair] = GapjunctionConfig(sec_idxs, \
                                                            connection_probs[pair], \
                                                            connection_params, \
                                                            connection_bounds, \
                                                            coupling_coeffs[pair], \
                                                            coupling_params, \
                                                            coupling_bounds)
        else:
            self.gapjunctions = None

            
    def load_node_rank_map(self, node_rank_file):

        rank = 0
        if self.comm is not None:
            rank = self.comm.Get_rank()

        node_rank_map = None
        if rank == 0:
            with open(node_rank_file) as fp:
                dval = {}
                lines = fp.readlines()
                for l in lines:
                    a = l.split(' ')
                    dval[int(a[0])] = int(a[1])
                node_rank_map = dval
        node_rank_map = self.comm.bcast(node_rank_map, root=0)

        pop_names = sorted(self.celltypes.keys())

        self.node_allocation = set([])
        for pop_name in pop_names:
            present = False
            num = self.celltypes[pop_name]['num']
            start = self.celltypes[pop_name]['start']
            for gid in range(start, start+num):
                if gid in node_rank_map:
                    present = True
                if node_rank_map[gid] == rank:
                    self.node_allocation.add(gid)
            if not present:
                if rank == 0:
                    self.logger.warning('load_node_rank_map: gids assigned to population %s are not present in node ranks file %s; '
                                        'gid to rank assignment will not be used'  % (pop_name, node_rank_file))
                self.node_allocation = None
                break

            
    def load_celltypes(self):
        """

        :return:
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        celltypes = self.celltypes
        typenames = sorted(celltypes.keys())

        if rank == 0:
            color = 1
        else:
            color = 0
        ## comm0 includes only rank 0
        comm0 = self.comm.Split(color, 0)

        if rank == 0:
            self.logger.info('env.data_file_path = %s' % str(self.data_file_path))

        self.cell_attribute_info = None
        population_ranges = None
        population_names = None
        if rank == 0:
            population_names = read_population_names(self.data_file_path, comm0)
            (population_ranges, _) = read_population_ranges(self.data_file_path, comm0)
            self.cell_attribute_info = read_cell_attribute_info(self.data_file_path, population_names, comm=comm0)
            self.logger.info('population_names = %s' % str(population_names))
            self.logger.info('population_ranges = %s' % str(population_ranges))
            self.logger.info('attribute info: %s' % str(self.cell_attribute_info))
        population_ranges = self.comm.bcast(population_ranges, root=0)
        population_names = self.comm.bcast(population_names, root=0)
        self.cell_attribute_info = self.comm.bcast(self.cell_attribute_info, root=0)
        comm0.Free()

        for k in typenames:
            population_range = population_ranges.get(k, None)
            if population_range is not None:
                celltypes[k]['start'] = population_ranges[k][0]
                celltypes[k]['num'] = population_ranges[k][1]
                if 'mechanism file' in celltypes[k]:
                    celltypes[k]['mech_file_path'] = '%s/%s' % (self.config_prefix, celltypes[k]['mechanism file'])
                    mech_dict = None
                    if rank == 0:
                        mech_dict = read_from_yaml(celltypes[k]['mech_file_path'])
                    mech_dict = self.comm.bcast(mech_dict, root=0)
                    celltypes[k]['mech_dict'] = mech_dict
                if 'synapses' in celltypes[k]:
                    synapses_dict = celltypes[k]['synapses']
                    if 'weights' in synapses_dict:
                        weights_config = synapses_dict['weights']
                        if isinstance(weights_config, list):
                            weights_dicts =  weights_config
                        else:
                            weights_dicts =  [weights_config]
                        for weights_dict in weights_dicts:
                            if 'expr' in weights_dict:
                                expr = weights_dict['expr']
                                parameter = weights_dict['parameter']
                                const = weights_dict.get('const', {})
                                clos = ExprClosure(parameter, expr, const)
                                weights_dict['closure'] = clos
                        synapses_dict['weights'] = weights_dicts
                        


    def clear(self):
        self.gidset = set([])
        self.gjlist = []
        self.cells = defaultdict(dict)
        self.artificial_cells = defaultdict(dict)
        self.biophys_cells = defaultdict(dict)
        self.recording_sets = {}
        if self.pc is not None:
            self.pc.gid_clear()
        if self.t_vec is not None:
            self.t_vec.resize(0)
        if self.id_vec is not None:
            self.id_vec.resize(0)
        if self.t_rec is not None:
            self.t_rec.resize(0)
        self.recs_dict = {}
        self.recs_count = 0
        for pop_name, _ in viewitems(self.Populations):
            self.recs_dict[pop_name] = defaultdict(list)
