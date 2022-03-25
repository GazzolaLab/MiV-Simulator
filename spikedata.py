import sys, math, copy
from collections import defaultdict
import numpy as np
from neuroh5.io import scatter_read_cell_attributes, read_cell_attributes, read_population_names, read_population_ranges, write_cell_attributes
import biophys_microcircuit
from biophys_microcircuit.utils import get_module_logger, viewitems, zip, Struct, get_trial_time_ranges, baks

## This logger will inherit its setting from its root logger
## which is created in module env
logger = get_module_logger(__name__)

default_baks_analysis_options = Struct(**{'BAKS Alpha': 4.77,
                                          'BAKS Beta': None})


def get_env_spike_dict(env, include_artificial=True):
    """
    Constructs  a dictionary with per-gid per-trial spike times from the output vectors with spike times and gids contained in env.
    """
    equilibration_duration = float(env.stimulus_config['Equilibration Duration'])
    n_trials = env.n_trials

    t_vec = np.array(env.t_vec, dtype=np.float32)
    id_vec = np.array(env.id_vec, dtype=np.uint32)

    trial_time_ranges = get_trial_time_ranges(env.t_rec.to_python(), env.n_trials)
    trial_time_bins = [ t_trial_start for t_trial_start, t_trial_end in trial_time_ranges ] 
    trial_dur = np.asarray([env.tstop + equilibration_duration] * n_trials, dtype=np.float32)
    
    binlst = []
    typelst = sorted(env.celltypes.keys())
    binvect = np.asarray([env.celltypes[k]['start'] for k in typelst ])
    sort_idx = np.argsort(binvect, axis=0)
    pop_names = [typelst[i] for i in sort_idx]
    bins = binvect[sort_idx][1:]
    inds = np.digitize(id_vec, bins)

    pop_spkdict = {}
    for i, pop_name in enumerate(pop_names):
        spkdict = {}
        sinds = np.where(inds == i)
        if len(sinds) > 0:
            ids = id_vec[sinds]
            ts = t_vec[sinds]
            for j in range(0, len(ids)):
                gid = ids[j]
                t = ts[j]
                if (not include_artificial) and (gid in env.artificial_cells[pop_name]):
                    continue
                if gid in spkdict:
                   spkdict[gid].append(t)
                else:
                   spkdict[gid] = [t]
            for gid in spkdict:
                spiketrain = np.array(spkdict[gid], dtype=np.float32)
                if gid in env.spike_onset_delay:
                    spiketrain -= env.spike_onset_delay[gid]
                trial_bins = np.digitize(spiketrain, trial_time_bins) - 1
                trial_spikes = [np.copy(spiketrain[np.where(trial_bins == trial_i)[0]])
                                for trial_i in range(env.n_trials)]
                for trial_i, trial_spiketrain in enumerate(trial_spikes):
                    trial_spiketrain -= np.sum(trial_dur[:(trial_i)]) + equilibration_duration
                spkdict[gid] = trial_spikes
        pop_spkdict[pop_name] = spkdict

    return pop_spkdict

def read_spike_events(input_file, population_names, namespace_id, spike_train_attr_name='t', time_range=None,
                      max_spikes=None, n_trials=-1, merge_trials=False, comm=None, io_size=0, include_artificial=True):
    """
    Reads spike trains from a NeuroH5 file, and returns a dictionary with spike times and cell indices.
    :param input_file: str (path to file)
    :param population_names: list of str
    :param namespace_id: str
    :param spike_train_attr_name: str
    :param time_range: list of float
    :param max_spikes: float
    :param n_trials: int
    :param merge_trials: bool
    :return: dict
    """
    assert((n_trials >= 1) | (n_trials == -1))

    trial_index_attr = 'Trial Index'
    trial_dur_attr = 'Trial Duration'
    artificial_attr = 'artificial'
    
    spkpoplst = []
    spkindlst = []
    spktlst = []
    spktrials = []
    num_cell_spks = {}
    pop_active_cells = {}

    tmin = float('inf')
    tmax = 0.

    for pop_name in population_names:

        if time_range is None or time_range[1] is None:
            logger.info('Reading spike data for population %s...' % pop_name)
        else:
            logger.info('Reading spike data for population %s in time range %s...' % (pop_name, str(time_range)))

        spike_train_attr_set = set([spike_train_attr_name, trial_index_attr, trial_dur_attr, artificial_attr])
        spkiter_dict = scatter_read_cell_attributes(input_file, pop_name, namespaces=[namespace_id], 
                                                    mask=spike_train_attr_set, comm=comm, io_size=io_size)
        spkiter = spkiter_dict[namespace_id]
        
        this_num_cell_spks = 0
        active_set = set([])

        pop_spkindlst = []
        pop_spktlst = []
        pop_spktriallst = []

        logger.info('Read spike cell attributes for population %s...' % pop_name)

        # Time Range
        if time_range is not None:
            if time_range[0] is None:
                time_range[0] = 0.0

        for spkind, spkattrs in spkiter:
            is_artificial_flag = spkattrs.get(artificial_attr, None)
            is_artificial = (is_artificial_flag[0] > 0) if is_artificial_flag is not None else None
            if is_artificial is not None:
                if is_artificial and (not include_artificial):
                    continue
            slen = len(spkattrs[spike_train_attr_name])
            trial_dur = spkattrs.get(trial_dur_attr, np.asarray([0.]))
            trial_ind = spkattrs.get(trial_index_attr, np.zeros((slen,),dtype=np.uint8))
            if n_trials == -1:
                n_trials = len(set(trial_ind))
            filtered_spk_idxs_by_trial = np.argwhere(trial_ind <= n_trials).ravel()
            filtered_spkts = spkattrs[spike_train_attr_name][filtered_spk_idxs_by_trial]
            filtered_trial_ind = trial_ind[filtered_spk_idxs_by_trial]
            if time_range is not None:
                filtered_spk_idxs_by_time = np.argwhere(np.logical_and(filtered_spkts >= time_range[0],
                                                                       filtered_spkts <= time_range[1])).ravel()
                filtered_spkts = filtered_spkts[filtered_spk_idxs_by_time]
                filtered_trial_ind = filtered_trial_ind[filtered_spk_idxs_by_time]
            pop_spkindlst.append(np.repeat([spkind], len(filtered_spkts)).astype(np.uint32))
            pop_spktriallst.append(filtered_trial_ind)
            this_num_cell_spks += len(filtered_spkts)
            active_set.add(spkind)
            for i, spkt in enumerate(filtered_spkts):
                trial_i = filtered_trial_ind[i]
                if merge_trials:
                    spkt += np.sum(trial_dur[:trial_i])
                pop_spktlst.append(spkt)
                tmin = min(tmin, spkt)
                tmax = max(tmax, spkt)

        pop_active_cells[pop_name] = active_set
        num_cell_spks[pop_name] = this_num_cell_spks

        if not active_set:
            continue

        pop_spkts = np.asarray(pop_spktlst, dtype=np.float32)
        del (pop_spktlst)
        pop_spkinds = np.concatenate(pop_spkindlst, dtype=np.uint32)
        del (pop_spkindlst)
        pop_spktrials = np.concatenate(pop_spktriallst, dtype=np.uint32)
        del (pop_spktriallst)

        # Limit to max_spikes
        if (max_spikes is not None) and (len(pop_spkts) > max_spikes):
            logger.warn(' Reading only randomly sampled %i out of %i spikes for population %s' %
                        (max_spikes, len(pop_spkts), pop_name))
            sample_inds = np.random.randint(0, len(pop_spkinds) - 1, size=int(max_spikes))
            pop_spkts = pop_spkts[sample_inds]
            pop_spkinds = pop_spkinds[sample_inds]
            pop_spktrials = pop_spkinds[sample_inds]
            tmax = max(tmax, max(pop_spkts))

        spkpoplst.append(pop_name)
        pop_trial_spkindlst = []
        pop_trial_spktlst = []
        for trial_i in range(n_trials):
            trial_idxs = np.where(pop_spktrials == trial_i)[0]
            sorted_trial_idxs = np.argsort(pop_spkts[trial_idxs])
            pop_trial_spktlst.append(np.take(pop_spkts[trial_idxs], sorted_trial_idxs))
            pop_trial_spkindlst.append(np.take(pop_spkinds[trial_idxs], sorted_trial_idxs))
                
        del pop_spkts
        del pop_spkinds
        del pop_spktrials

        if merge_trials:
            pop_spkinds = np.concatenate(pop_trial_spkindlst)
            pop_spktlst = np.concatenate(pop_trial_spktlst)
            spkindlst.append(pop_spkinds)
            spktlst.append(pop_spktlst)
        else:
            spkindlst.append(pop_trial_spkindlst)
            spktlst.append(pop_trial_spktlst)
            

        logger.info(' Read %i spikes and %i trials for population %s' % (this_num_cell_spks, n_trials, pop_name))

    return {'spkpoplst': spkpoplst, 'spktlst': spktlst, 'spkindlst': spkindlst,
            'tmin': tmin, 'tmax': tmax,
            'pop_active_cells': pop_active_cells, 'num_cell_spks': num_cell_spks,
            'n_trials': n_trials}

def make_spike_dict(spkinds, spkts):
    """
    Given arrays with cell indices and spike times, returns a dictionary with per-cell spike times.
    """
    spk_dict = defaultdict(list)
    for spkind, spkt in zip(np.nditer(spkinds), np.nditer(spkts)):
        spk_dict[int(spkind)].append(float(spkt))
    return spk_dict


def spike_density_estimate(population, spkdict, time_bins, arena_id=None, trajectory_id=None, output_file_path=None,
                            progress=False, inferred_rate_attr_name='Inferred Rate Map', **kwargs):
    """
    Calculates spike density function for the given spike trains.
    :param population:
    :param spkdict:
    :param time_bins:
    :param arena_id: str
    :param trajectory_id: str
    :param output_file_path:
    :param progress:
    :param inferred_rate_attr_name: str
    :param kwargs: dict
    :return: dict
    """
    if progress:
        from tqdm import tqdm

    analysis_options = copy.copy(default_baks_analysis_options)
    analysis_options.update(kwargs)

    def make_spktrain(lst, t_start, t_stop):
        spkts = np.asarray(lst, dtype=np.float32)
        return spkts[(spkts >= t_start) & (spkts <= t_stop)]

    
    t_start = time_bins[0]
    t_stop = time_bins[-1]

    spktrains = {ind: make_spktrain(lst, t_start, t_stop) for (ind, lst) in viewitems(spkdict)}
    baks_args = dict()
    baks_args['a'] = analysis_options['BAKS Alpha']
    baks_args['b'] = analysis_options['BAKS Beta']
    
    if progress:
        seq = tqdm(viewitems(spktrains))
    else:
        seq = viewitems(spktrains)
        
    spk_rate_dict = {ind: baks(spkts / 1000., time_bins / 1000., **baks_args)[0].reshape((-1,))
                     if len(spkts) > 1 else np.zeros(time_bins.shape)
                     for ind, spkts in seq}

    if output_file_path is not None:
        if arena_id is None or trajectory_id is None:
            raise RuntimeError('spike_density_estimate: arena_id and trajectory_id required to write Spike Density'
                               'Function namespace')
        namespace = 'Spike Density Function %s %s' % (arena_id, trajectory_id)
        attr_dict = {ind: {inferred_rate_attr_name: np.asarray(spk_rate_dict[ind], dtype='float32')}
                     for ind in spk_rate_dict}
        write_cell_attributes(output_file_path, population, attr_dict, namespace=namespace)

    result = {ind: {'rate': rate, 'time': time_bins} for ind, rate in viewitems(spk_rate_dict)}

        
    result = { ind: { 'rate': rate, 'time': time_bins }
              for ind, rate in viewitems(spk_rate_dict) }
    
    return result

def spike_bin_counts(spkdict, time_bins):
    bin_dict = {}
    for (ind, lst) in viewitems(spkdict):

        if len(lst) > 0:
            spkts = np.asarray(lst, dtype=np.float32)
            bins, bin_edges = np.histogram(spkts, bins=time_bins)
            
            bin_dict[ind] = bins

    return bin_dict
