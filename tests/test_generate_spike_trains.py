import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from typing import Tuple, List, Callable, Optional
from miv_simulator.env import Env
from miv_simulator.input_features import (
    EncoderTimeConfig,
    CoordinateSystemConfig,
    InputModality,
    InputFeature,
    InputFeaturePopulation,
    FeatureSpace,
)
from miv_simulator.input_spike_trains import generate_input_spike_trains
from mpi4py import MPI
import h5py


# Define a custom reduction operation that concatenates lists
def list_concat(a, b, datatype):
    """Concatenate two lists of lists"""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


# Create MPI concatenation operation
list_concat_op = MPI.Op.Create(list_concat, commute=True)


class TemporalModality(InputModality):
    """Temporal modality for processing 1D time series data."""

    def __init__(
        self,
        name: str = "temporal",
        input_shape: Tuple[int] = (1000,),  # (samples,)
        temporal_bounds: Tuple[float, float] = (0, 1),  # Normalized time
        frequency_bounds: Tuple[float, float] = (1, 100),  # Hz
        sample_rate: int = 1000,  # 1kHz default
    ):
        # Simple 2D feature coordinate system:
        # - time position (when in the signal this feature responds)
        # - frequency preference (what frequency this feature detects)
        feature_coordinate_system = CoordinateSystemConfig(
            dimensions=2,
            bounds=[
                temporal_bounds,  # Time position (normalized)
                frequency_bounds,  # Frequency preference (Hz)
            ],
            units=["normalized_time", "Hz"],
        )

        super().__init__(
            name,
            "temporal",
            feature_coordinate_system,
            input_shape=input_shape,  # (time_samples,)
        )

        self.sample_rate = sample_rate  # Samples per second

    def preprocess_signal(self, stimulus: np.ndarray) -> np.ndarray:
        """Preprocess temporal signal (normalize amplitude)"""
        # Ensure input is 1D or 2D
        if len(stimulus.shape) == 1:
            # Single channel - reshape to (time, channels)
            processed = stimulus.reshape(-1, 1)
        elif len(stimulus.shape) == 2:
            # Already in correct format (time, channels)
            processed = stimulus.copy()
        else:
            raise ValueError(f"Expected 1D or 2D signal, got shape {stimulus.shape}")

        # Normalize amplitude to [-1, 1]
        if processed.max() > 1.0 or processed.min() < -1.0:
            processed = processed / np.max(np.abs(processed))

        return processed

    def to_feature_coordinates(self, modality_coordinates: np.ndarray) -> np.ndarray:
        """Convert (time, frequency) to feature space."""
        if len(modality_coordinates) != 2:
            raise ValueError(f"Expected 2 coordinates, got {len(modality_coordinates)}")
        return modality_coordinates

    def from_feature_coordinates(self, feature_coordinates: np.ndarray) -> np.ndarray:
        """Convert feature space coordinates to modality-specific coordinates."""
        if len(feature_coordinates) != 2:
            raise ValueError(
                f"Expected 2D feature coordinates, got {len(feature_coordinates)}"
            )
        return feature_coordinates

    def create_input_filter(self, position: np.ndarray) -> Callable:
        """Create a temporal input filter based on feature coordinates."""
        # Extract feature coordinates
        time_pos, preferred_freq = position

        def frequency_filter(signal: np.ndarray) -> np.ndarray:
            """Apply a frequency filter across the entire signal duration for all channels."""
            # Handle different input shapes
            original_shape = signal.shape
            if len(original_shape) == 1:
                signal_reshaped = signal.reshape(-1, 1)
            else:
                signal_reshaped = signal

            time_points, num_channels = signal_reshaped.shape
            filtered_signal = np.zeros_like(signal_reshaped, dtype=np.float32)

            # Adaptive window size based on frequency
            cycles_to_capture = 3  # We want at least 3 cycles for reliable detection
            min_window_size = 50  # Minimum window size in ms

            # Calculate window size based on frequency (in samples)
            window_size = max(
                int(cycles_to_capture * self.sample_rate / preferred_freq),
                int(min_window_size * self.sample_rate / 1000),
            )

            # FFT-based filtering
            from numpy.fft import rfft, irfft, rfftfreq

            # Process each channel
            for channel in range(num_channels):
                channel_signal = signal_reshaped[:, channel]

                # Apply FFT
                fft_values = rfft(channel_signal)
                freqs = rfftfreq(len(channel_signal), d=1.0 / self.sample_rate)

                # Create bandpass parameters
                bandwidth = max(2.0, preferred_freq * 0.3)  # Hz
                low_cutoff = max(0.5, preferred_freq - bandwidth / 2)
                high_cutoff = min(
                    preferred_freq + bandwidth / 2, self.sample_rate / 2 * 0.95
                )

                # Create bandpass filter in frequency domain
                filter_mask = np.zeros_like(freqs, dtype=np.float32)
                mask_indices = np.where((freqs >= low_cutoff) & (freqs <= high_cutoff))
                filter_mask[mask_indices] = 1.0

                # Smooth the edges of the filter mask to reduce ringing
                if len(mask_indices[0]) > 2:
                    transition_width = max(2, int(len(freqs) * 0.01))

                    # Smooth the lower edge
                    lower_idx = mask_indices[0][0]
                    if lower_idx > transition_width:
                        lower_transition = np.hanning(2 * transition_width)[
                            :transition_width
                        ]
                        filter_mask[lower_idx - transition_width : lower_idx] = (
                            lower_transition
                        )

                    # Smooth the upper edge
                    upper_idx = mask_indices[0][-1]
                    if upper_idx + transition_width < len(filter_mask):
                        upper_transition = np.hanning(2 * transition_width)[
                            transition_width:
                        ]
                        filter_mask[
                            upper_idx + 1 : upper_idx + transition_width + 1
                        ] = upper_transition

                # Apply the filter in frequency domain
                filtered_fft = fft_values * filter_mask

                # Inverse FFT to get filtered signal
                filtered_channel = irfft(filtered_fft, n=len(channel_signal))

                # Calculate energy using a properly sized window
                energy = np.zeros_like(channel_signal)

                for i in range(time_points):
                    half_window = window_size // 2
                    start_idx = max(0, i - half_window)
                    end_idx = min(time_points, i + half_window)

                    window_data = filtered_channel[start_idx:end_idx]
                    if len(window_data) > 0:
                        # Use RMS energy for better stability
                        energy[i] = np.sqrt(np.mean(window_data**2))

                # Apply time-based weighting
                if 0 <= time_pos <= 1:
                    time_indices = np.linspace(0, 1, time_points)
                    # Use a Gaussian window to weight by time
                    time_weight = np.exp(
                        -(((time_indices - time_pos) / 0.1) ** 2)
                    )  # Narrower window for better contrast
                    energy = energy * time_weight

                filtered_signal[:, channel] = energy

            # Normalize to a consistent range
            for channel in range(num_channels):
                channel_max = np.max(filtered_signal[:, channel])
                if channel_max > 0:
                    filtered_signal[:, channel] /= channel_max

            # Reshape output to match input
            if len(original_shape) == 1:
                return filtered_signal.flatten()

            return filtered_signal

        return frequency_filter

    def generate_feature_distribution(
        self, n_features: int, local_random: Optional[np.random.RandomState] = None
    ) -> List[np.ndarray]:
        """Generate a distribution of feature coordinates appropriate for this modality."""
        if local_random is None:
            local_random = np.random.RandomState()

        positions = []
        # Time bounds and frequency bounds
        t_min, t_max = self.feature_coordinate_system.bounds[0]
        f_min, f_max = self.feature_coordinate_system.bounds[1]

        # Create a grid of features over the feature space
        # Distribute time positions uniformly
        n_time_positions = int(np.sqrt(n_features))
        n_frequency_positions = n_features // n_time_positions

        # Ensure we get the requested number of features
        remaining = n_features - (n_time_positions * n_frequency_positions)

        # Create log-spaced frequencies for better coverage
        log_f_min = np.log(max(1.0, f_min))
        log_f_max = np.log(f_max)

        # Generate positions on a grid
        for i in range(n_time_positions):
            time_pos = t_min + (t_max - t_min) * (i + 0.5) / n_time_positions

            for j in range(n_frequency_positions + (1 if i < remaining else 0)):
                # Log spacing in frequency
                log_freq = (
                    log_f_min
                    + (log_f_max - log_f_min) * (j + 0.5) / n_frequency_positions
                )
                freq = np.exp(log_freq)

                # Add small jitter to avoid exact grid alignment
                jitter_t = (
                    local_random.uniform(-0.5, 0.5)
                    * (t_max - t_min)
                    / (2 * n_time_positions)
                )
                jitter_f = local_random.uniform(-0.5, 0.5) * (
                    freq * 0.1
                )  # 10% jitter in frequency

                positions.append(np.array([time_pos + jitter_t, freq + jitter_f]))

        return positions


# Create a population using the generate_feature_distribution method
class TemporalFeaturePopulation(InputFeaturePopulation):
    def generate_features(
        self,
        start_gid: int = 0,
        n_features: Optional[int] = None,
        local_random: Optional[np.random.RandomState] = None,
        rank: Optional[int] = None,
        size: Optional[int] = None,
    ) -> List[InputFeature]:
        """Generate a population of features using the modality's distribution method."""

        if local_random is None:
            local_random = np.random.RandomState()

        if n_features is None:
            n_features = self.n_features
        else:
            n_features = min(n_features, self.n_features)

        # Use the modality's generate_feature_distribution method
        positions = self.modality.generate_feature_distribution(
            n_features, local_random
        )

        if (rank is not None) and (size is not None):
            feature_indices = list(range(rank, self.n_features, size))
        else:
            feature_indices = list(range(self.n_features))

        features = []

        for i in feature_indices:
            gid = start_gid + i

            position = positions[i]

            # Generate encoding based on distribution
            encoding = self._generate_encoding(position, local_random)

            # Create input filter specific to this feature's position
            input_filter = self.modality.create_input_filter(position)

            # Create feature with filter
            feature = InputFeature(
                gid=gid,
                position=position,
                encoding=encoding,
                input_filter=input_filter,
            )
            features.append(feature)
            self.features[gid] = feature

        return features


def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


if __name__ == "__main__":
    dry_run = False
    plot = False

    comm = MPI.COMM_WORLD
    rank = comm.rank

    logging.basicConfig(level=logging.INFO)

    # np.seterr(all="raise")
    dataset_prefix = "/scratch1/03320/iraikov/striped2/MiV/results/livn"
    config = {}
    params = dict(locals())
    # params["config"] = params.pop("config_file")
    params["Model Name"] = "temporal_input_features"
    env = Env(**params)

    # Set up parameters
    sample_dt_ms = 1.0
    sample_rate = 1000.0 / sample_dt_ms  # Sample rate [Hz]
    duration = 10.0  # Overall signal duration [s]
    n_features = 1000  # Number of features in the population

    # Create an instance of the temporal modality
    temporal_modality = TemporalModality(
        name="temporal",
        input_shape=(
            int(duration * sample_rate),
        ),  # duration seconds of data at sample_rate
        temporal_bounds=(0, 1),  # Normalized time (0-1)
        frequency_bounds=(1, 100),  # 1-100 Hz
        sample_rate=sample_rate,
    )

    # Create a feature space and register the modality
    feature_space = FeatureSpace(name="feature_space")
    feature_space.register_modality(temporal_modality)

    # Create the population
    temporal_feature_population = TemporalFeaturePopulation(
        name="temporal_neurons",
        feature_space=feature_space,
        n_features=n_features,
        modality=temporal_modality,
        encoding_distribution={
            "feature_type": "linear_rate",
            "peak_rate": 100.0,
        },
    )

    max_pop_enum = 0
    for _, pop_enum in env.Populations.items():
        max_pop_enum = max(pop_enum, max_pop_enum)

    pop_id = max_pop_enum + 1
    env.Populations[temporal_feature_population.name] = pop_id
    cell_distribution = {}
    if "Cell Distribution" in env.geometry:
        cell_distribution = env.geometry["Cell Distribution"]
    else:
        env.geometry["Cell Distribution"] = cell_distribution
    cell_distribution[temporal_feature_population.name] = {"All": n_features}

    # Generate the features
    features = temporal_feature_population.generate_features(
        rank=comm.rank, size=comm.size
    )

    # Create test signals with different frequencies
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Generate a signal that changes frequency over time
    stimulus = np.zeros_like(t)
    segment_length = len(t) // 4

    # 5Hz in first quarter
    stimulus[:segment_length] = np.sin(2 * np.pi * 5.0 * t[:segment_length])

    # 10Hz in second quarter
    stimulus[segment_length : 2 * segment_length] = np.sin(
        2 * np.pi * 10.0 * t[:segment_length]
    )

    # 20Hz in third quarter
    stimulus[2 * segment_length : 3 * segment_length] = np.sin(
        2 * np.pi * 20.0 * t[:segment_length]
    )

    # 40Hz in fourth quarter
    stimulus[3 * segment_length :] = np.sin(2 * np.pi * 40.0 * t[:segment_length])

    # Process the stimulus using the modality
    processed_stimulus = temporal_modality.preprocess_signal(stimulus)

    # Initialize encoders and get responses
    dt_ms = 1.0  # Encoder timestep [ms]
    sample_duration_ms = dt_ms  # Duration of one sample [ms]

    # Initialize the encoders with appropriate time config
    time_config = EncoderTimeConfig(duration_ms=sample_duration_ms, dt_ms=dt_ms)

    # Initialize all encoders
    for feature in temporal_feature_population.features.values():
        feature.initialize_encoder(time_config)

    # Get responses for each feature
    local_activations = []
    local_spike_responses = []

    local_positions = list(
        [feature.position for feature in temporal_feature_population.features.values()]
    )

    local_times = np.vstack(local_positions)[:, 0]

    for feature in temporal_feature_population.features.values():
        # Get the activation level from the input filter
        activation = feature.input_filter(processed_stimulus)
        local_activations.append(activation)

        # Get spike response
        response = feature.get_response(processed_stimulus)
        local_spike_responses.append(response)

    spike_responses = comm.reduce(local_spike_responses, op=list_concat_op, root=0)
    activations = comm.reduce(local_activations, op=list_concat_op, root=0)
    positions = np.array(comm.reduce(local_positions, op=list_concat_op, root=0))

    if plot and (rank == 0):

        # Visualize the results
        # Plot for the input signal
        fig, axs = plt.subplots(
            3, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [2, 3, 3]}
        )

        # Plot the input signal
        axs[0].plot(t, stimulus)
        axs[0].set_title("Input Signal: Frequency Changes Over Time")
        axs[0].set_ylabel("Amplitude")

        # Add frequency transition markers and labels
        for i, freq in enumerate([5, 10, 20, 40]):
            pos = i * segment_length
            axs[0].axvline(x=t[pos], color="r", linestyle="--", alpha=0.5)
            axs[0].text(
                t[pos + segment_length // 2],
                1.1,
                f"{freq} Hz",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Plot the feature positions in feature space
        times = positions[:, 0]
        freqs = positions[:, 1]

        # Create a color map based on mean activation levels
        mean_activations = np.array([np.mean(act) for act in activations])
        normalized_activations = (
            mean_activations / np.max(mean_activations)
            if np.max(mean_activations) > 0
            else mean_activations
        )

        # Plot feature positions colored by activation level
        scatter = axs[1].scatter(
            times,
            freqs,
            c=normalized_activations,
            cmap="viridis",
            s=100,
            alpha=0.8,
            edgecolors="k",
        )
        axs[1].set_title("Feature Population Distribution")
        axs[1].set_xlabel("Time Position")
        axs[1].set_ylabel("Frequency Preference (Hz)")
        cbar = plt.colorbar(scatter, ax=axs[1])
        cbar.set_label("Mean Activation Level")

        # Highlight the key frequency bands
        for i, freq in enumerate([5, 10, 20, 40]):
            axs[1].axhline(y=freq, color="r", linestyle="--", alpha=0.3)
            axs[1].text(
                1.02,
                freq,
                f"{freq} Hz",
                horizontalalignment="left",
                verticalalignment="center",
            )

        # Plot spike raster
        # Sort features by frequency for better visualization
        sorted_indices = np.argsort(freqs)
        sorted_responses = [spike_responses[i] for i in sorted_indices]
        sorted_freqs = freqs[sorted_indices]

        # Create a color map for frequencies
        # Use the log scale for better color distribution across frequency range
        log_freqs = np.log(sorted_freqs)
        norm = mcolors.Normalize(vmin=np.min(log_freqs), vmax=np.max(log_freqs))
        cmap = plt.get_cmap("plasma")

        # Create spike raster
        for i, response in enumerate(sorted_responses):
            if isinstance(response, list):
                # If the response is already spike times, use directly
                spike_times = response
            # Convert to spike times (assuming binary spike array)
            elif len(response.shape) >= 3:  # [samples, timesteps, neurons]
                binary_spikes = response[0, :, 0]  # Take first sample, first neuron
                spike_times = t[binary_spikes > 0]
            else:
                # Use all responses above threshold as "spikes"
                spike_times = t[response > 0.5] if len(response) > 0 else []

            # Get color for this feature based on its frequency
            log_freq = np.log(sorted_freqs[i])
            color = cmap(norm(log_freq))

            # Plot spike times as vertical lines with frequency-based color
            for spike_time in spike_times[0]:
                norm_spike_time = spike_time / 1000.0
                axs[2].plot(
                    [norm_spike_time, norm_spike_time],
                    [i - 0.4, i + 0.4],
                    color=color,
                    linewidth=2.0,
                )

        # Create frequency range labels for y-axis
        # Define frequency bins (can be adjusted based on your needs)
        freq_bins = [1, 5, 10, 20, 40, 100]  # Bin edges
        n_ticks = 6  # Number of tick marks to show

        # Find which bin each feature belongs to
        bin_indices = np.digitize(sorted_freqs, freq_bins) - 1
        bin_indices = np.clip(
            bin_indices, 0, len(freq_bins) - 2
        )  # Ensure valid bin indices

        # Calculate positions for tick marks
        tick_positions = []
        tick_labels = []

        for i in range(len(freq_bins) - 1):
            # Find features in this bin
            features_in_bin = np.where(bin_indices == i)[0]

            if len(features_in_bin) > 0:
                # Put tick at the middle of this frequency range
                tick_pos = np.mean(features_in_bin)
                tick_positions.append(tick_pos)

                # Create label for this range
                if i == len(freq_bins) - 2:  # Last bin
                    tick_labels.append(f"{freq_bins[i]}-{freq_bins[i + 1]} Hz")
                else:
                    tick_labels.append(f"{freq_bins[i]}-{freq_bins[i + 1]} Hz")

        # Add boundary ticks
        tick_positions.insert(0, 0)
        tick_labels.insert(0, f"< {freq_bins[0]} Hz")
        tick_positions.append(len(positions) - 1)
        tick_labels.append(f"> {freq_bins[-1]} Hz")

        # Set the y-axis ticks and labels
        axs[2].set_yticks(tick_positions)
        axs[2].set_yticklabels(tick_labels, fontsize=9)

        # Add horizontal lines to separate frequency ranges
        for i in range(1, len(freq_bins)):
            # Find the boundary between bins
            boundary_idx = np.where(sorted_freqs >= freq_bins[i])[0]
            if len(boundary_idx) > 0:
                boundary_pos = boundary_idx[0] - 0.5
                axs[2].axhline(
                    y=boundary_pos,
                    color="gray",
                    linestyle="-",
                    alpha=0.5,
                    linewidth=0.5,
                )

        # Add a colorbar to show the frequency mapping
        # Create a mappable object for the colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # This is required for the colorbar to work

        # Add colorbar to the right of the spike raster
        cbar = plt.colorbar(sm, ax=axs[2])
        cbar.set_label("Frequency Preference (Hz)")

        # Set the colorbar ticks to show actual frequency values instead of log values
        freq_ticks = np.array(
            [1, 5, 10, 20, 40, 100]
        )  # Choose meaningful frequency values
        log_freq_ticks = np.log(freq_ticks)
        # Only use ticks that are within the data range
        valid_ticks = freq_ticks[
            (log_freq_ticks >= np.min(log_freqs))
            & (log_freq_ticks <= np.max(log_freqs))
        ]
        valid_log_ticks = np.log(valid_ticks)
        cbar.set_ticks(valid_log_ticks)
        cbar.set_ticklabels([f"{int(f)} Hz" for f in valid_ticks])

        plt.tight_layout()
        plt.show()

        # Print a summary of the results
        print("\nFeature Population Summary:")
        print(f"Total features: {len(positions)}")

        # Group features by frequency ranges
        freq_ranges = [(0, 7.5), (7.5, 15), (15, 30), (30, 60), (60, 100)]
        for freq_range in freq_ranges:
            features_in_range = [
                position
                for position in positions
                if freq_range[0] <= position[1] < freq_range[1]
            ]
            if features_in_range:
                print(
                    f"\nFeatures tuned to {freq_range[0]}-{freq_range[1]} Hz: {len(features_in_range)}"
                )
                mean_acts = [
                    np.mean(activations[i])
                    for i, position in enumerate(positions)
                    if freq_range[0] <= position[1] < freq_range[1]
                ]
                print(f"  Mean activation: {np.mean(mean_acts):.4f}")

        # Find features with highest activations
        feature_indices = list(range(len(activations)))
        feature_indices.sort(key=lambda i: np.mean(activations[i]), reverse=True)

        print("\nTop 5 most active features:")
        for i in range(min(5, len(feature_indices))):
            idx = feature_indices[i]
            position = positions[idx]
            print(
                f"  Feature at position {position}: mean activation = {np.mean(activations[idx]):.4f}"
            )
    comm.barrier()

    signal_id = "test_temporal_features_20240510"
    output_path = os.path.join(dataset_prefix, "temporal_input_spike_trains_n1000_10s.h5")

    if not dry_run:
        generate_input_spike_trains(
            env,
            temporal_feature_population,
            signal=stimulus,
            signal_id=signal_id,
            coords_path=None,
            output_path=output_path,
            output_spikes_namespace="Temporal Feature Spikes",
            output_spike_train_attr_name="Spike Train",
            io_size=1,
            write_size=50,
            chunk_size=10000,
            value_chunk_size=10000,
        )

    # Save the input signal and metadata to the output file on rank 0
    if comm.rank == 0 and not dry_run:
        print(f"Saving input signal to {output_path}")
        with h5py.File(output_path, "a") as f:
            # Create a group for signals if it doesn't exist
            if "Signals" not in f:
                signals_group = f.create_group("Signals")
            else:
                signals_group = f["Signals"]

            # Create a group for the specific signal
            if signal_id in signals_group:
                del signals_group[signal_id]
            signal_group = signals_group.create_group(signal_id)

            signal_group.create_dataset("data", data=stimulus, compression="gzip")

            # Save metadata
            signal_group.attrs["duration"] = duration
            signal_group.attrs["sample_rate"] = sample_rate
            signal_group.attrs["sample_dt_ms"] = sample_dt_ms
            signal_group.attrs["description"] = (
                "Synthetic signal with changing frequency (5Hz, 10Hz, 20Hz, 40Hz)"
            )

            # Save frequency information for each segment
            frequencies = np.array([5.0, 10.0, 20.0, 40.0])
            signal_group.create_dataset("frequencies", data=frequencies)

            # Save time points
            signal_group.create_dataset("time", data=t, compression="gzip")

            # Save segment information
            segment_starts = np.array(
                [0, segment_length, 2 * segment_length, 3 * segment_length]
            )
            segment_ends = np.array(
                [segment_length, 2 * segment_length, 3 * segment_length, len(stimulus)]
            )
            signal_group.create_dataset("segment_starts", data=segment_starts)
            signal_group.create_dataset("segment_ends", data=segment_ends)

    comm.barrier()
