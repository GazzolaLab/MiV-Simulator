from typing import Dict, List, Optional, Tuple

from collections import namedtuple

import numpy as np
from miv_simulator.utils import AbstractEnv, Struct, get_module_logger
from miv_simulator.spike_encoder import (
    RateEncoder,
    PoissonRateEncoder,
    RankOrderEncoder,
)
from miv_simulator.stgen import get_inhom_poisson_spike_times_by_thinning
from mpi4py import MPI
from mpi4py.MPI import Intracomm
from neuroh5.io import (
    read_cell_attributes,
    scatter_read_cell_attribute_selection,
)
from numpy import ndarray, uint8

## This logger will inherit its setting from its root logger, dentate,
## which is created in module env
logger = get_module_logger(__name__)


PhaseModConfig = namedtuple(
    "PhaseModConfig",
    ["phase_range", "phase_pref", "phase_offset", "mod_depth", "frequency"],
)


class InputSource:
    pass


class ConstantInputSource(InputSource):
    def __init__(
        self,
        selectivity_type: None = None,
        peak_rate: None = None,
        local_random: None = None,
        selectivity_attr_dict: Optional[Dict[str, ndarray]] = None,
        phase_mod_config: None = None,
    ) -> None:
        """
        :param selectivity_type: int
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

            self.phase_mod_function = (
                lambda t, initial_phase=0.0: stationary_phase_mod(
                    t,
                    phase_range,
                    phase_pref,
                    phase_offset + initial_phase,
                    mod_depth,
                    freq,
                )
            )

        if selectivity_attr_dict is not None:
            self.init_from_attr_dict(selectivity_attr_dict)
        elif any([arg is None for arg in [selectivity_type, peak_rate]]):
            raise RuntimeError(
                "ConstantInputSource: missing argument(s) required for object construction"
            )
        else:
            if local_random is None:
                local_random = np.random.RandomState()
            self.selectivity_type = selectivity_type
            self.peak_rate = peak_rate

    def init_from_attr_dict(
        self, selectivity_attr_dict: Dict[str, ndarray]
    ) -> None:
        self.selectivity_type = selectivity_attr_dict["Selectivity Type"][0]
        self.peak_rate = selectivity_attr_dict["Peak Rate"][0]

    def get_selectivity_attr_dict(self):
        return {
            "Selectivity Type": np.array(
                [self.selectivity_type], dtype=np.uint8
            ),
            "Peak Rate": np.array([self.peak_rate], dtype=np.float32),
        }

    def call(
        self,
        signal: ndarray,
        initial_phase: float = 0.0,
    ) -> ndarray:
        """

        :param x: array
        :param y: array
        :return: array
        """

        rate_array = np.ones_like(signal, dtype=np.float32) * self.peak_rate
        mean_rate = np.mean(rate_array)
        if self.phase_mod_function is not None:
            # t = TODO
            rate_array *= self.phase_mod_function(
                t, initial_phase=initial_phase
            )
            mean_rate_mod = np.mean(rate_array)
            if mean_rate_mod > 0.0:
                rate_array *= mean_rate / mean_rate_mod

        return rate_array


def make_input_source(
    selectivity_type: uint8,
    selectivity_type_names: Dict[int, str],
    population: None = None,
    stimulus_config: None = None,
    distance: None = None,
    local_random: None = None,
    selectivity_attr_dict: Optional[Dict[str, ndarray]] = None,
    phase_mod_config: None = None,
    noise_gen_dict: None = None,
    comm: None = None,
) -> InputSource:
    """

    :param selectivity_type: int
    :param selectivity_type_names: dict: {int: str}
    :param population: str
    :param stimulus_config: dict
    :param distance: float; u arc distance normalized to reference layer
    :param local_random: :class:'np.random.RandomState'
    :param selectivity_attr_dict: dict
    :param phase_mod_config: dict; oscillatory phase modulation configuration
    :return: instance of one of various InputSource classes
    """
    if selectivity_type not in selectivity_type_names:
        raise RuntimeError(
            "make_input_source: enumerated selectivity type: %i not recognized"
            % selectivity_type
        )
    selectivity_type_name = selectivity_type_names[selectivity_type]

    if selectivity_attr_dict is not None:
        if selectivity_type_name == "constant":
            input_source = ConstantInputSource(
                selectivity_attr_dict=selectivity_attr_dict,
                phase_mod_config=phase_mod_config,
            )
        else:
            RuntimeError(
                "make_input_source: selectivity type %s is not supported"
                % selectivity_type_name
            )
    elif any([arg is None for arg in [population, stimulus_config]]):
        raise RuntimeError(
            f"make_input_source: missing argument(s) required to construct {selectivity_type_name} cell config object: population: {population} stimulus_config: {stimulus_config}"
        )
    else:
        if (
            population not in stimulus_config["Peak Rate"]
            or selectivity_type not in stimulus_config["Peak Rate"][population]
        ):
            raise RuntimeError(
                "make_input_source: peak rate not specified for population: %s, selectivity type: "
                "%s" % (population, selectivity_type_name)
            )
        peak_rate = stimulus_config["Peak Rate"][population][selectivity_type]

        if selectivity_type_name == "constant":
            input_source = ConstantInputSource(
                selectivity_type=selectivity_type,
                peak_rate=peak_rate,
                phase_mod_config=phase_mod_config,
            )
        else:
            RuntimeError(
                f"make_input_source: selectivity type: {selectivity_type_name} not implemented"
            )

    return input_source


def get_equilibration(env: AbstractEnv) -> Tuple[ndarray, int]:
    if (
        "Equilibration Duration" in env.stimulus_config
        and env.stimulus_config["Equilibration Duration"] > 0.0
    ):
        equilibrate_len = int(
            env.stimulus_config["Equilibration Duration"]
            / env.stimulus_config["Temporal Resolution"]
        )
        from scipy.signal import hann

        equilibrate_hann = hann(2 * equilibrate_len)[:equilibrate_len]
        equilibrate = (equilibrate_hann, equilibrate_len)
    else:
        equilibrate = None

    return equilibrate


def generate_input_spike_trains(
    env: AbstractEnv,
    population: str,
    selectivity_type_names: Dict[int, str],
    signal: ndarray,
    gid: int,
    selectivity_attr_dict: Dict[str, ndarray],
    spike_train_attr_name: str = "Spike Train",
    selectivity_type_name: None = None,
    spike_hist_resolution: int = 1000,
    equilibrate: Optional[Tuple[ndarray, int]] = None,
    phase_mod_config: None = None,
    initial_phases: ndarray = None,
    spike_hist_sum: None = None,
    return_selectivity_features: bool = True,
    n_trials: int = 1,
    merge_trials: bool = True,
    time_range: Optional[List[float]] = None,
    comm: Optional[Intracomm] = None,
    seed: None = None,
    debug: bool = False,
) -> Dict[str, ndarray]:
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

    t, d = trajectory
    abs_t = (t - t[0]) / 1000.0

    equilibration_duration = float(
        env.stimulus_config["Equilibration Duration"]
    )
    temporal_resolution = float(env.stimulus_config["Temporal Resolution"])

    local_random = np.random.RandomState()
    input_spike_train_seed = int(
        env.model_config["Random Seeds"]["Input Spiketrains"]
    )

    if phase_mod_config is not None:
        if (n_trials > 1) and (initial_phases is None):
            initial_phases = global_oscillation_initial_phases(env, n_trials)

    if seed is None:
        local_random.seed(int(input_spike_train_seed + gid))
    else:
        local_random.seed(int(seed))

    this_selectivity_type = selectivity_attr_dict["Selectivity Type"][0]
    this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
    if selectivity_type_name is None:
        selectivity_type_name = this_selectivity_type_name

    input_source = make_input_source(
        selectivity_type=this_selectivity_type,
        selectivity_type_names=selectivity_type_names,
        selectivity_attr_dict=selectivity_attr_dict,
        phase_mod_config=phase_mod_config,
    )

    trial_duration = np.max(t) - np.min(t)
    if time_range is not None:
        trial_duration = max(
            trial_duration,
            (time_range[1] - time_range[0]) + equilibration_duration,
        )

    spike_trains = []
    trial_indices = []
    initial_phase = 0.0
    for i in range(n_trials):
        if initial_phases is not None:
            initial_phase = initial_phases[i]
        rate_map = input_source.get_rate_map(
            d=d,
            initial_phase=initial_phase,
        )
        if (selectivity_type_name != "constant") and (equilibrate is not None):
            equilibrate_filter, equilibrate_len = equilibrate
            rate_map[:equilibrate_len] = np.multiply(
                rate_map[:equilibrate_len], equilibrate_filter
            )

        spike_train = np.asarray(
            get_inhom_poisson_spike_times_by_thinning(
                rate_map, t, dt=temporal_resolution, generator=local_random
            ),
            dtype=np.float32,
        )
        if merge_trials:
            spike_train += float(i) * trial_duration
        spike_trains.append(spike_train)
        trial_indices.append(
            np.ones((spike_train.shape[0],), dtype=np.uint8) * i
        )

    if debug and rank == 0:
        callback, context = debug
        this_context = Struct(**dict(context()))
        this_context.update(dict(locals()))
        callback(this_context)

    spikes_attr_dict = dict()
    if merge_trials:
        spikes_attr_dict[spike_train_attr_name] = np.asarray(
            np.concatenate(spike_trains), dtype=np.float32
        )
        spikes_attr_dict["Trial Index"] = np.asarray(
            np.concatenate(trial_indices), dtype=np.uint8
        )
    else:
        spikes_attr_dict[spike_train_attr_name] = spike_trains
        spikes_attr_dict["Trial Index"] = trial_indices

    spikes_attr_dict["Trial Duration"] = np.asarray(
        [trial_duration] * n_trials, dtype=np.float32
    )

    if return_selectivity_features:
        spikes_attr_dict["Selectivity Type"] = np.array(
            [this_selectivity_type], dtype=np.uint8
        )
        spikes_attr_dict["Trajectory Rate Map"] = np.asarray(
            rate_map, dtype=np.float32
        )

    if spike_hist_sum is not None:
        spike_hist_edges = np.linspace(
            min(t), max(t), spike_hist_resolution + 1
        )
        hist, edges = np.histogram(spike_train, bins=spike_hist_edges)
        spike_hist_sum[this_selectivity_type_name] = np.add(
            spike_hist_sum[this_selectivity_type_name], hist
        )

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


def generate_input_features(
    env,
    population,
    gid,
    norm_distances,
    selectivity_type_names,
    selectivity_type_namespaces,
    noise_gen_dict=None,
    debug=False,
):
    """
    Generates input features for the given population and
    returns the selectivity type-specific dictionary provided through
    argument selectivity_type_namespaces.  The set of selectivity
    attributes is determined by procedure get_selectivity_attr_dict in
    the respective input cell configuration class
    (e.g. ConstantInputSource).

    :param env
    :param population: str
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
    selectivity_seed_offset = int(
        env.model_config["Random Seeds"]["Input Selectivity"]
    )

    local_random = np.random.RandomState()
    local_random.seed(int(selectivity_seed_offset + gid))
    this_selectivity_type = choose_input_selectivity_type(
        p=env.stimulus_config["Selectivity Type Probabilities"][population],
        local_random=local_random,
    )

    input_source = make_input_source(
        population=population,
        selectivity_type=this_selectivity_type,
        selectivity_type_names=selectivity_type_names,
        stimulus_config=env.stimulus_config,
        distance=norm_u_arc_distance,
        local_random=local_random,
        noise_gen_dict=noise_gen_dict,
        comm=env.comm,
    )

    this_selectivity_type_name = selectivity_type_names[this_selectivity_type]
    selectivity_attr_dict = input_source.get_selectivity_attr_dict()

    if debug and rank == 0:
        callback, context = debug
        this_context = Struct(**dict(context()))
        this_context.update(dict(locals()))
        callback(this_context)

    return this_selectivity_type_name, selectivity_attr_dict


def read_stimulus(stimulus_path, stimulus_namespace, population, module=None):
    ratemap_lst = []
    module_gid_set = set()
    if module is not None:
        if not isinstance(module, int):
            raise Exception("module variable must be an integer")
        gid_module_gen = read_cell_attributes(
            stimulus_path, population, namespace="Cell Attributes"
        )
        for gid, attr_dict in gid_module_gen:
            this_module = attr_dict["Module"][0]
            if this_module == module:
                module_gid_set.add(gid)

    attr_gen = read_cell_attributes(
        stimulus_path, population, namespace=stimulus_namespace
    )
    for gid, stimulus_dict in attr_gen:
        if gid in module_gid_set or len(module_gid_set) == 0:
            rate = stimulus_dict["Trajectory Rate Map"]
            spiketrain = stimulus_dict["Spike Train"]
            peak_index = np.where(rate == np.max(rate))[0][0]
            ratemap_lst.append((gid, rate, spiketrain, peak_index))

    ## sort by peak_index
    ratemap_lst.sort(key=lambda item: item[3])
    return ratemap_lst


def oscillation_phase_mod_config(
    env, population, soma_positions, local_random=None
):
    """
    Obtains phase modulation configuration for a given neuronal population.
    """
    global_oscillation_config = env.stimulus_config["Global Oscillation"]
    frequency = global_oscillation_config["frequency"]
    phase_mod_config = global_oscillation_config["Phase Modulation"][population]
    phase_range = phase_mod_config["phase range"]
    mod_depth = phase_mod_config["depth"]

    population_phase_prefs = global_oscillation_phase_pref(
        env,
        population,
        num_cells=len(soma_positions),
        local_random=local_random,
    )
    position_array = np.asarray(
        [soma_positions[k] for k in sorted(soma_positions)]
    )
    population_phase_shifts = global_oscillation_phase_shift(
        env, position_array
    )

    phase_mod_config_dict = {}
    for i, gid in enumerate(sorted(soma_positions)):
        phase_mod_config_dict[gid] = PhaseModConfig(
            phase_range,
            population_phase_prefs[i],
            population_phase_shifts[i],
            mod_depth,
            frequency,
        )

    return phase_mod_config_dict


def global_oscillation_phase_shift(env, position):
    """
    Computes the phase shift of the global oscillatory signal for the given position, assumed to be on the long axis.
    Uses the "Global Oscillation" entry in the input configuration.
    See `global_oscillation_signal` for a description of the configuration format.
    """

    global_oscillation_config = env.stimulus_config["Global Oscillation"]
    phase_dist_config = global_oscillation_config["Phase Distribution"]
    phase_slope = phase_dist_config["slope"]
    phase_offset = phase_dist_config["offset"]
    x = position / 1000.0

    return x * phase_slope + phase_offset


def global_oscillation_phase_pref(
    env, population, num_cells, local_random=None
):
    """
    Computes oscillatory phase preferences for all cells in the given population.
    Uses the "Global Oscillation" entry in the input configuration.
    See `global_oscillation_signal` for a description of the configuration format.

    Returns: an array of phase preferences of length equal to the population size.
    """

    seed = int(env.model_config["Random Seeds"]["Phase Preference"])

    if local_random is None:
        local_random = np.random.RandomState(seed)

    global_oscillation_config = env.stimulus_config["Global Oscillation"]
    phase_mod_config = global_oscillation_config["Phase Modulation"][population]
    phase_range = phase_mod_config["phase range"]
    phase_loc = (phase_range[1] - phase_range[0]) / 2.0
    fw = 2.0 * np.sqrt(2.0 * np.log(100.0))
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

    seed = int(env.model_config["Random Seeds"]["Initial Phase"])

    if local_random is None:
        local_random = np.random.RandomState(seed)

    s = [0.0]
    if n_trials > 1:
        for i in range(n_trials - 1):
            s.append(local_random.uniform(0.0, 360.0))

    a = np.deg2rad(np.asarray(s))

    return a


def stationary_phase_mod(
    t, phase_range, phase_pref, phase_offset, mod_depth, freq
):
    """
    Computes stationary oscillatory phase modulation with the given parameters.
    """

    r = phase_range[1] - phase_range[0]
    delta = 2 * np.pi - np.deg2rad(phase_pref)
    s = np.cos(2 * np.pi * freq * (t) - np.deg2rad(phase_offset) + delta) + 1

    d = np.clip(mod_depth, 0.0, 1.0)
    mod = s * mod_depth / 2.0 + (1.0 - mod_depth)

    return mod
