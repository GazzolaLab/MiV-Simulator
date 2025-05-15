from typing import Tuple
from collections import namedtuple
import numpy as np
from miv_simulator.utils import AbstractEnv, get_module_logger
from numpy import ndarray

## This logger will inherit its setting from its root logger,
## which is created in module env
logger = get_module_logger(__name__)


PhaseModConfig = namedtuple(
    "PhaseModConfig",
    ["phase_range", "phase_pref", "phase_offset", "mod_depth", "frequency"],
)


def get_equilibration(env: AbstractEnv) -> Tuple[ndarray, int]:
    if (
        "Equilibration Duration" in env.stimulus_config
        and env.stimulus_config["Equilibration Duration"] > 0.0
    ):
        equilibrate_len = int(
            env.stimulus_config["Equilibration Duration"]
            / env.stimulus_config["Temporal Resolution"]
        )
        from scipy.signal.windows import hann

        equilibrate_hann = hann(2 * equilibrate_len)[:equilibrate_len]
        equilibrate = (equilibrate_hann, equilibrate_len)
    else:
        equilibrate = None

    return equilibrate


def create_time_array(signal, time_config):
    """
    Create a time array representing the duration of the input signal.

    Args:
        signal: Input signal array, shape [n_samples, n_features]
        time_config: EncoderTimeConfig object with duration and step size information

    Returns:
        t: Time array in milliseconds matching the length of the signal
    """
    if len(signal.shape) != 2:
        raise ValueError(
            f"Expected signal with shape [n_samples, n_features], got {signal.shape}"
        )

    n_samples = signal.shape[0]

    # Case 1: If signal length matches the time_config's expected number of steps
    if n_samples == time_config.num_steps:
        return time_config.get_time_vector_ms()

    # Case 2: Signal length differs from time_config's steps (common when signal is from external source)
    # Create a time array that spans the duration but has exactly n_samples points
    if n_samples > 1:
        t = np.linspace(0, time_config.duration_ms, n_samples)
    else:
        # Handle edge case of a single-sample signal
        t = np.array([0])

    return t


def oscillation_phase_mod_config(env, population, soma_positions, local_random=None):
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
    position_array = np.asarray([soma_positions[k] for k in sorted(soma_positions)])
    population_phase_shifts = global_oscillation_phase_shift(env, position_array)

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


def global_oscillation_phase_pref(env, population, num_cells, local_random=None):
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


def stationary_phase_mod(t, phase_range, phase_pref, phase_offset, mod_depth, freq):
    """
    Computes stationary oscillatory phase modulation with the given parameters.
    """

    delta = 2 * np.pi - np.deg2rad(phase_pref)
    s = np.cos(2 * np.pi * freq * (t) - np.deg2rad(phase_offset) + delta) + 1

    mod = s * mod_depth / 2.0 + (1.0 - mod_depth)

    return mod


def spatial_phase_mod(
    x,
    velocity,
    field_width,
    phase_range,
    phase_entry,
    phase_offset,
    mod_depth,
    freq,
):
    """Non-stationary phase modulation for spatial receptive fields.
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
    """
    r = np.deg2rad(phase_range[1] - phase_range[0])
    s = (
        np.cos(
            r * x / field_width
            + 2 * np.pi * freq * x / velocity
            - np.deg2rad(phase_entry + phase_offset)
            + r / 2.0
        )
        + 1
    )
    m = s * mod_depth / 2.0 + (1.0 - mod_depth)

    return m


def spatial2d_phase_mod(
    x,
    y,
    velocity,
    field_width,
    phase_range,
    phase_entry,
    phase_offset,
    mod_depth,
    freq,
):
    x_mod = spatial_phase_mod(
        x,
        velocity,
        field_width,
        phase_range,
        phase_entry,
        phase_offset,
        mod_depth,
        freq,
    )
    y_mod = spatial_phase_mod(
        y,
        velocity,
        field_width,
        phase_range,
        phase_entry,
        phase_offset,
        mod_depth,
        freq,
    )

    m = (x_mod + y_mod) / 2.0

    return m
