import gc
import math
import os
import sys
import time
import h5py
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from miv_simulator.utils import config_logging, get_script_logger
from miv_simulator.stimulus import stationary_phase_mod
from mpi4py import MPI
from neuroh5.io import (
    NeuroH5CellAttrGen,
    append_cell_attributes,
    read_population_ranges,
)
from spike_encoder import (
    EncoderTimeConfig,
    LinearRateEncoder,
    ReceptiveFieldEncoder,
    PoissonSpikeGenerator,
    EncodingPipeline,
)

logger = get_script_logger(os.path.basename(__file__))

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


@dataclass
class CoordinateSystemConfig:
    """Configuration for a coordinate system."""

    dimensions: int
    bounds: List[Tuple[float, float]]  # Min and max for each dimension
    resolution: Optional[List[float]] = None  # Resolution per dimension
    units: Optional[List[str]] = None  # Units for each dimension
    transform_matrix: Optional[np.ndarray] = None  # For transforming coordinates


class InputModality(ABC):
    """Abstract base class for input modalities."""

    def __init__(
        self,
        name: str,
        stimulus_type: str,
        feature_coordinate_system: CoordinateSystemConfig,
        input_shape: Tuple[int, ...] = None,
    ):
        self.name = name
        self.stimulus_type = stimulus_type
        self.feature_coordinate_system = feature_coordinate_system
        # self.feature_coordinate_system.validate()
        self.input_shape = input_shape

    @abstractmethod
    def preprocess_signal(self, stimulus: np.ndarray) -> np.ndarray:
        """Preprocess a raw signal into a format suitable for encoders."""
        pass

    @abstractmethod
    def to_feature_coordinates(self, modality_coordinates: np.ndarray) -> np.ndarray:
        """Convert modality-specific coordinates to feature space coordinates."""
        pass

    @abstractmethod
    def from_feature_coordinates(self, feature_coordinates: np.ndarray) -> np.ndarray:
        """Convert feature space coordinates to modality-specific coordinates."""
        pass


class LinearRateInput:
    def __init__(
        self,
        time_config: EncoderTimeConfig,
        feature_type: Optional[str] = None,
        peak_rate: Optional[float] = None,
        local_random: Optional[np.random.RandomState] = None,
        phase_mod_config: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        :param feature_type: int
        :param local_random: :class:'np.random.RandomState'
        :param phase_mod_config: dict; oscillatory phase modulation configuration
        """
        self.time_config = time_config

        self.phase_mod_function = None
        if phase_mod_config is not None:
            phase_range = phase_mod_config.phase_range
            phase_pref = phase_mod_config.phase_pref
            phase_offset = phase_mod_config.phase_offset
            mod_depth = phase_mod_config.mod_depth
            freq = phase_mod_config.frequency

            self.phase_mod_function = lambda t, initial_phase=0.0: stationary_phase_mod(
                t,
                phase_range,
                phase_pref,
                phase_offset + initial_phase,
                mod_depth,
                freq,
            )

        if local_random is None:
            local_random = np.random.RandomState()
        self.feature_type = feature_type
        self.peak_rate = peak_rate

        self.encoder = EncodingPipeline(
            [
                LinearRateEncoder(
                    time_config=time_config,
                    max_firing_rate_hz=peak_rate,
                ),
                PoissonSpikeGenerator(
                    time_config=time_config, random_seed=local_random
                ),
            ],
            time_config=self.time_config,
        )

    def get_response(
        self, signal: ndarray, initial_phase: Optional[float] = None
    ) -> np.ndarray:
        """Compute response given input signal."""
        response = self.encoder.encode(signal, return_times=True)
        return response


class ReceptiveFieldInput:
    def __init__(
        self,
        time_config: EncoderTimeConfig,
        feature_type: Optional[str] = None,
        peak_rate: Optional[float] = None,
        tuning_width: Optional[float] = None,
        local_random: Optional[np.random.RandomState] = None,
        phase_mod_config: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        :param feature_type: int
        :param local_random: :class:'np.random.RandomState'
        :param attr_dict: dict
        :param phase_mod_config: dict; oscillatory phase modulation configuration
        """
        self.time_config = time_config

        self.phase_mod_function = None
        if phase_mod_config is not None:
            phase_range = phase_mod_config.phase_range
            phase_pref = phase_mod_config.phase_pref
            phase_offset = phase_mod_config.phase_offset
            mod_depth = phase_mod_config.mod_depth
            freq = phase_mod_config.frequency

            self.phase_mod_function = lambda t, initial_phase=0.0: stationary_phase_mod(
                t,
                phase_range,
                phase_pref,
                phase_offset + initial_phase,
                mod_depth,
                freq,
            )

        if local_random is None:
            local_random = np.random.RandomState()
        self.feature_type = feature_type
        self.peak_rate = peak_rate

        self.encoder = EncodingPipeline(
            [
                ReceptiveFieldEncoder(
                    time_config=time_config,
                    input_range=(0, 1),
                    max_firing_rate_hz=self.peak_rate,
                    neurons_per_dim=1,
                    tuning_width=self.tuning_width,
                ),
                PoissonSpikeGenerator(
                    time_config=time_config, random_seed=local_random
                ),
            ]
        )

    def get_response(
        self, signal: ndarray, initial_phase: Optional[float] = None
    ) -> np.ndarray:
        """Compute response given input signal."""
        return self.encoder.encode(signal)


class FeatureEncoding:
    """Specification for a feature encoder."""

    def __init__(
        self,
        feature_type: str,
        encoder_params: Dict[str, Any],
        tuning_params: Optional[Dict[str, Any]] = None,
        phase_mod_config: Optional[Dict[str, float]] = None,
    ):
        self.feature_type = feature_type
        self.encoder_params = encoder_params
        self.tuning_params = tuning_params or {}
        self.phase_mod_config = phase_mod_config

    def create_encoder(
        self,
        time_config: EncoderTimeConfig,
        local_random: Optional[np.random.RandomState] = None,
    ):
        """Create an encoder instance based on this specification."""
        if self.feature_type == "linear_rate":
            input_encoder = LinearRateInput(
                time_config=time_config,
                local_random=local_random,
                feature_type=self.feature_type,
                phase_mod_config=self.phase_mod_config,
                **self.encoder_params,
                **self.tuning_params,
            )
        elif self.feature_type == "receptive_field":
            input_encoder = ReceptiveFieldInput(
                time_config=time_config,
                local_random=local_random,
                feature_type=self.feature_type,
                phase_mod_config=self.phase_mod_config,
                **self.encoder_params,
                **self.tuning_params,
            )
        else:
            RuntimeError(
                "create_encoder: feature type {feature_type_name} is not supported"
            )
        return input_encoder


class InputFeature:
    """Individual feature within a population."""

    def __init__(
        self,
        gid: int,
        position: np.ndarray,
        encoding: FeatureEncoding,
        input_filter: Optional[Callable] = None,  # Function mapping input to encoder
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.gid = gid
        self.position = position
        self.encoding = encoding
        self.kwargs = kwargs or {}
        self.input_filter = input_filter
        self._encoder = None

    def initialize_encoder(
        self,
        time_config: EncoderTimeConfig,
        local_random: Optional[np.random.RandomState] = None,
    ):
        """Initialize the encoder for this feature."""
        if self._encoder is None:
            self._encoder = self.encoding.create_encoder(time_config, local_random)

    def get_response(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Get the feature's response to a stimulus."""
        if self._encoder is None:
            raise RuntimeError(
                "Encoder not initialized. Call initialize_encoder first."
            )

        # Apply feature's input filter if defined
        filtered_input = signal
        if self.input_filter is not None:
            filtered_input = self.input_filter(signal)

        return self._encoder.get_response(filtered_input, **kwargs)[0]

    def to_attribute_dict(self) -> Dict[str, np.ndarray]:
        """Convert to attribute dictionary for storage."""
        attr_dict = {
            "Feature Type": np.array([self.encoding.feature_type], dtype=np.uint8),
            "Position": self.position.astype(np.float32),
        }
        # Add encoder-specific attributes
        attr_dict.update(
            {
                k: np.array([v], dtype=np.float32)
                for k, v in self.encoding.encoder_params.items()
            }
        )
        # Add tuning-specific attributes
        attr_dict.update(
            {
                k: np.array([v], dtype=np.float32)
                for k, v in self.encoding.tuning_params.items()
                if isinstance(v, (int, float))
            }
        )
        # Add metadata
        attr_dict.update(
            {
                k: np.array(
                    [v], dtype=np.float32 if isinstance(v, (int, float)) else object
                )
                for k, v in self.metadata.items()
                if isinstance(v, (int, float, str))
            }
        )
        return attr_dict

    @classmethod
    def from_attribute_dict(
        cls,
        attr_dict: Dict[str, np.ndarray],
        feature_type_mapping: Dict[int, str],
        gid: int,
        input_filter: Optional[Callable] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "InputFeature":
        """Create an InputFeature from an attribute dictionary.

        Args:
            attr_dict: Dictionary containing feature attributes as NumPy arrays
            gid: Global ID for the feature
            input_filter: Optional function to filter input signals
            kwargs: Optional additional keyword arguments for the feature

        Returns:
            A new InputFeature instance
        """
        # Extract feature type
        feature_type_arr = attr_dict["Feature Type"]
        # Handle different ways feature_type might be stored
        if feature_type_arr.dtype == np.uint8:
            # Convert uint8 to string
            feature_type = feature_type_mapping[feature_type_arr[0]]
        else:
            feature_type = feature_type_arr[0]
            # Map numeric codes to feature type strings if needed
            if feature_type == 0:
                feature_type = "linear_rate"
            elif feature_type == 1:
                feature_type = "receptive_field"
            else:
                raise RuntimeError(f"Unknown feature type {feature_type}")

        # Extract position
        position = attr_dict["Position"]

        # Extract encoder and tuning parameters
        encoder_params = {}
        tuning_params = {}

        # Known parameter categories TODO
        known_encoder_params = ["peak_rate"]
        known_tuning_params = ["tuning_width"]

        for key, value in attr_dict.items():
            if key not in ["Feature Type", "Position"]:
                # Convert to scalar if it's a 1-element array
                param_value = value[0] if len(value) == 1 else value

                if key in known_encoder_params:
                    encoder_params[key] = param_value
                elif key in known_tuning_params:
                    tuning_params[key] = param_value
                else:
                    # Default to encoder parameters for unknown keys
                    encoder_params[key] = param_value

        # Create encoding
        encoding = FeatureEncoding(
            feature_type=feature_type,
            encoder_params=encoder_params,
            tuning_params=tuning_params,
        )

        # Create and return feature
        return cls(
            gid=gid,
            position=position,
            encoding=encoding,
            input_filter=input_filter,
            kwargs=kwargs or {},
        )


class InputFeaturePopulation:
    """Collection of features within a feature space."""

    def __init__(
        self,
        name: str,
        feature_space: "FeatureSpace",
        n_features: int,
        modality: InputModality,
        density_function: Optional[Callable] = None,
        encoding_distribution: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.feature_space = feature_space
        self.n_features = n_features
        self.modality = modality
        self.density_function = density_function or (
            lambda x: 1.0
        )  # Uniform by default
        self.encoding_distribution = encoding_distribution or {}
        self.features: Dict[int, InputFeature] = {}

    def generate_features(
        self,
        start_gid: int = 0,
        local_random: Optional[np.random.RandomState] = None,
        rank: Optional[int] = None,
        size: Optional[int] = None,
    ) -> List[InputFeature]:
        """Generate a population of features with appropriate spatial distribution."""

        if local_random is None:
            local_random = np.random.RandomState()

        features = []
        feature_indices = None
        if (rank is not None) and (size is not None):
            feature_indices = list(range(rank, self.n_features, size))
        else:
            feature_indices = list(range(self.n_features))

        for i in feature_indices:
            gid = start_gid + i

            # Generate position based on density function
            position = self._generate_position(local_random)

            # Generate encoding based on distribution
            encoding = self._generate_encoding(position, local_random)

            # Create feature
            feature = InputFeature(gid, position, encoding)
            features.append(feature)
            self.features[gid] = feature

        return features

    def _generate_position(self, local_random: np.random.RandomState) -> np.ndarray:
        """Generate a position in feature space according to the density function."""
        # TODO: use rejection sampling based on the provided density function
        dimensions = self.modality.coordinate_system.dimensions
        bounds = self.modality.coordinate_system.bounds

        position = np.zeros(dimensions)
        for d in range(dimensions):
            position[d] = local_random.uniform(bounds[d][0], bounds[d][1])

        return position

    def _generate_encoding(
        self, position: np.ndarray, local_random: np.random.RandomState
    ) -> FeatureEncoding:
        """Generate an encoding specification based on position and distribution."""
        # Default to a basic receptive field encoding if nothing else specified
        feature_type = self.encoding_distribution.get("feature_type", "receptive_field")

        # Basic encoder params
        encoder_params = {
            "peak_rate": self.encoding_distribution.get("peak_rate", 100.0),
        }

        # Add tuning params for receptive fields
        tuning_params = {}
        if feature_type == "receptive_field":
            tuning_params["tuning_width"] = self.encoding_distribution.get(
                "tuning_width", 0.2
            )

        return FeatureEncoding(feature_type, encoder_params, tuning_params)

    def get_features_in_region(
        self, center: np.ndarray, radius: float
    ) -> List[InputFeature]:
        """Get features within a specified region."""
        features = []
        for feature in self.features.values():
            distance = np.linalg.norm(feature.position - center)
            if distance <= radius:
                features.append(feature)
        return features

    def get_attribute_dicts(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Get attribute dictionaries for all features in the population."""
        return {
            gid: feature.to_attribute_dict() for gid, feature in self.features.items()
        }

    @classmethod
    def from_attribute_dicts(
        cls,
        attr_dicts: Dict[int, Dict[str, np.ndarray]],
        feature_type_mapping: Dict[int, str],
        feature_space: "FeatureSpace",
        modality_name: str,
        name: str = "InputPopulation",
        create_input_filter: Optional[Callable] = None,
        density_function: Optional[Callable] = None,
        encoding_distribution: Optional[Dict[str, Any]] = None,
    ) -> "InputFeaturePopulation":
        """Create an InputFeaturePopulation from attribute dictionaries.

        Args:
            attr_dicts: Dictionary mapping GIDs to attribute dictionaries
            feature_space: The FeatureSpace containing this population
            modality_name: Name of the modality for this population
            name: Name for the population
            create_input_filter: Optional function to create input filters based on feature positions
            density_function: Optional function specifying spatial density of features
            encoding_distribution: Optional distribution of encoding parameters

        Returns:
            A new InputFeaturePopulation instance
        """
        if modality_name not in feature_space.modalities:
            raise ValueError(f"Modality '{modality_name}' not found in feature space")

        modality = feature_space.modalities[modality_name]

        # Create population without generating features
        population = cls(
            name=name,
            feature_space=feature_space,
            n_features=0,  # Add features from attributes instead of generating them
            modality=modality,
            density_function=density_function,
            encoding_distribution=encoding_distribution or {},
        )

        # Add features from attribute dictionaries
        for gid, attr_dict in attr_dicts.items():
            input_filter = None
            if create_input_filter is not None:
                position = attr_dict["Position"]
                input_filter = create_input_filter(position)

            feature = InputFeature.from_attribute_dict(
                feature_type_mapping=feature_type_mapping,
                attr_dict=attr_dict,
                gid=gid,
                input_filter=input_filter,
            )

            population.features[gid] = feature

        # Update n_features to reflect the actual number of features
        # TODO: update for distributed case
        population.n_features = len(population.features)

        return population


class FeatureSpace:
    """Top-level container for feature populations across modalities."""

    def __init__(self, name: str, common_embedding_dim: Optional[int] = None):
        self.name = name
        self.common_embedding_dim = common_embedding_dim
        self.modalities: Dict[str, InputModality] = {}
        self.populations: Dict[str, InputFeaturePopulation] = {}

    def register_modality(self, modality: InputModality) -> None:
        """Register a sensory modality with this feature space."""
        if modality.name in self.modalities:
            raise ValueError(f"Modality '{modality.name}' already registered")

        # If using common embedding, ensure mapping methods exist
        if self.common_embedding_dim is not None:
            if not hasattr(modality, "to_embedding") or not hasattr(
                modality, "from_embedding"
            ):
                raise ValueError(
                    "When using common embedding, modality must implement mapping methods"
                )

        self.modalities[modality.name] = modality

    def create_population(
        self,
        name: str,
        modality_name: str,
        n_features: int,
        density_function: Optional[Callable] = None,
        encoding_distribution: Optional[Dict[str, Any]] = None,
        start_gid: int = 0,
        local_random: Optional[np.random.RandomState] = None,
        rank: Optional[int] = None,
        size: Optional[int] = None,
    ) -> InputFeaturePopulation:
        """Create a new feature population."""
        if name in self.populations:
            raise ValueError(f"Population '{name}' already exists")
        if modality_name not in self.modalities:
            raise ValueError(f"Modality '{modality_name}' not registered")

        modality = self.modalities[modality_name]
        population = InputFeaturePopulation(
            name=name,
            feature_space=self,
            n_features=n_features,
            modality=modality,
            density_function=density_function,
            encoding_distribution=encoding_distribution,
        )

        # Generate features
        population.generate_features(start_gid, local_random, rank=rank, size=size)

        self.populations[name] = population
        return population

    def encode_signal(
        self,
        signal: np.ndarray,
        modality_name: str,
        population_names: Optional[List[str]] = None,
        time_config: Optional[EncoderTimeConfig] = None,
        **kwargs,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Encode a signal across selected populations."""

        if modality_name not in self.modalities:
            raise ValueError(f"Modality '{modality_name}' not registered")

        modality = self.modalities[modality_name]

        # Process signal for this modality
        processed_signal = modality.preprocess_stimulus(signal)

        # If population_names is None, use all populations with this modality
        if population_names is None:
            population_names = [
                name
                for name, pop in self.populations.items()
                if pop.modality.name == modality_name
            ]

        # Create time config if not provided
        if time_config is None:
            stimulus_duration = processed_signal.shape[0]
            time_config = EncoderTimeConfig(
                duration_ms=stimulus_duration,
                dt_ms=1.0,  # Default value
            )

        responses = {}
        for population_name in population_names:
            if population_name not in self.populations:
                raise ValueError(f"Population '{population_name}' not found")

            population = self.populations[population_name]

            # Skip populations from other modalities
            if population.modality.name != modality_name:
                continue

            # Initialize encoders for all features
            for feature in population.features.values():
                feature.initialize_encoder(time_config)

            # Get responses for each feature
            population_responses = {}
            for gid, feature in population.features.items():
                response = feature.get_response(processed_signal, **kwargs)
                population_responses[gid] = response

            responses[population_name] = population_responses

        return responses


def test_temporal_feature_encoding():
    import matplotlib.pyplot as plt

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

        def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
            """Preprocess temporal signal (normalize amplitude)"""
            # Ensure input is 1D or 2D
            if len(signal.shape) == 1:
                # Single channel - reshape to (time, channels)
                processed = signal.reshape(-1, 1)
            elif len(signal.shape) == 2:
                # Already in correct format (time, channels)
                processed = signal.copy()
            else:
                raise ValueError(f"Expected 1D or 2D signal, got shape {signal.shape}")

            # Normalize amplitude to [-1, 1]
            if processed.max() > 1.0 or processed.min() < -1.0:
                processed = processed / np.max(np.abs(processed))

            return processed

        def to_feature_coordinates(
            self, modality_coordinates: np.ndarray
        ) -> np.ndarray:
            """Convert (time, frequency) to feature space."""
            if len(modality_coordinates) != 2:
                raise ValueError(
                    f"Expected 2 coordinates, got {len(modality_coordinates)}"
                )
            return modality_coordinates

        def from_feature_coordinates(
            self, feature_coordinates: np.ndarray
        ) -> np.ndarray:
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
                cycles_to_capture = (
                    3  # We want at least 3 cycles for reliable detection
                )
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
                    mask_indices = np.where(
                        (freqs >= low_cutoff) & (freqs <= high_cutoff)
                    )
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

            # Create log-spaced frequencies (similar to auditory system)
            log_f_min = np.log(max(1.0, f_min))
            log_f_max = np.log(f_max)

            for i in range(n_features):
                # Linear spacing in time
                time_pos = t_min + (t_max - t_min) * (i / n_features)

                # Log spacing in frequency
                log_freq = log_f_min + (log_f_max - log_f_min) * (i % 10) / 10
                freq = np.exp(log_freq)

                positions.append(np.array([time_pos, freq]))

            return positions

    sample_dt_ms = 1.0
    sample_rate = 1000.0 / sample_dt_ms  # sample rate [Hz]
    duration = 1.0  # overall signal duration [s]

    # Create an instance of the temporal modality
    temporal_modality = TemporalModality(
        name="temporal",
        input_shape=(
            int(duration * sample_rate),
        ),  # duration second of data at sample_rate
        temporal_bounds=(0, 1),  # Normalized time (0-1)
        frequency_bounds=(1, 100),  # 1-100 Hz
        sample_rate=sample_rate,
    )

    # Use with the feature space
    feature_space = FeatureSpace(name="feature_space")
    feature_space.register_modality(temporal_modality)

    # Create several temporal features with different frequency tuning
    features = []
    target_frequencies = [5.0, 10.0, 20.0, 40.0]
    time_positions = [0.2, 0.4, 0.6, 0.8]  # Time positions in the signal

    for i, (freq, time_pos) in enumerate(zip(target_frequencies, time_positions)):
        position = np.array([time_pos, freq])
        feature = InputFeature(
            gid=i + 1,
            position=position,
            encoding=FeatureEncoding(
                feature_type="linear_rate", encoder_params={"peak_rate": 100.0}
            ),
            input_filter=temporal_modality.create_input_filter(position),
        )
        features.append(feature)

    dt_ms = 1.0  # encoder timestep [ms]
    sample_duration_ms = dt_ms  # duration of one sample [ms]

    # Initialize the encoders with appropriate time config
    time_config = EncoderTimeConfig(duration_ms=sample_duration_ms, dt_ms=dt_ms)
    for feature in features:
        feature.initialize_encoder(time_config)

    # Create test signals with different frequencies
    def create_test_sine_wave(
        frequency=10.0, duration=duration, sample_rate=sample_rate
    ):
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        return np.sin(2 * np.pi * frequency * t)

    # Create signals
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

    # Process the stimulus
    processed_stimulus = temporal_modality.preprocess_signal(stimulus)

    # Get spike responses for each feature
    spike_responses = []
    activations = []

    for feature in features:
        # Get the activation level from the input filter
        activation = feature.input_filter(processed_stimulus)
        activations.append(activation)

        # Get spike response
        response = feature.get_response(processed_stimulus)
        spike_responses.append(response)

        # Visualize the results
    fig, axs = plt.subplots(
        6, 1, figsize=(12, 16), gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 2]}
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

    # Plot the filtered signals for each feature
    for i, (feature, freq) in enumerate(zip(features, target_frequencies)):
        # Apply input filter to get filtered signal
        filtered_signal = feature.input_filter(processed_stimulus)

        # Plot filtered signal
        axs[i + 1].plot(t, filtered_signal)
        axs[i + 1].set_title(f"Filter Response: {freq} Hz Tuned Feature")
        axs[i + 1].set_ylabel("Response")

        # Add vertical lines at feature's preferred time position
        time_pos = feature.position[0]
        axs[i + 1].axvline(x=time_pos, color="g", linestyle="--", alpha=0.5)

        # Highlight the segment where this frequency appears
        highlight_idx = np.where(freq == np.array([5, 10, 20, 40]))[0][0]
        highlight_start = highlight_idx * segment_length
        highlight_end = (highlight_idx + 1) * segment_length

        axs[i + 1].axvspan(
            t[highlight_start], t[highlight_end - 1], color="y", alpha=0.2
        )

    # Plot spike raster
    for i, (response, freq) in enumerate(zip(spike_responses, target_frequencies)):
        if isinstance(response, list):
            # If the response is already spike times, use directly
            spike_times = response
        # Convert to spike times (assuming binary spike array)
        elif len(response.shape) >= 3:  # [samples, timesteps, neurons]
            binary_spikes = response[0, :, 0]  # Take first sample, first neuron
            spike_times = t[binary_spikes > 0]
        else:
            raise RuntimeError(f"Invalid response {response}")

        # Plot spike times as vertical lines
        for spike_time in spike_times[0]:
            norm_spike_time = spike_time / 1000.0
            axs[5].plot(
                [norm_spike_time, norm_spike_time],
                [i - 0.4, i + 0.4],
                "k-",
                linewidth=1.5,
            )

    axs[5].set_yticks(range(len(target_frequencies)))
    axs[5].set_yticklabels([f"{freq} Hz tuned" for freq in target_frequencies])
    axs[5].set_title("Spike Responses")
    axs[5].set_xlabel("Time (s)")
    axs[5].set_xlim(0, duration)

    plt.tight_layout()
    plt.show()

    # Print a summary of the results
    print("\nFeature Response Summary:")
    for i, (freq, activation) in enumerate(zip(target_frequencies, activations)):
        print(f"Feature tuned to {freq} Hz: mean activation = {np.mean(activation)}")


def generate_input_features(
    env,
    population: InputFeaturePopulation,
    coords_path: str,
    distances_namespace: str,
    output_path: str,
    output_feature_namespace: str,
    io_size: int,
    chunk_size: int,
    value_chunk_size: int,
    cache_size: int,
    write_size: int,
    dry_run: bool,
    verbose: bool,
    debug: bool,
    debug_count: int,
):
    """
    :param env: env.Env
    :param population: InputFeaturePopulation
    :param coords_path: str (path to file)
    :param distances_namespace: str
    :param output_path: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param write_size: int
    :param verbose: bool
    :param debug: bool
    :param debug_count: int
    :param dry_run: bool
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config_logging(verbose)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info(f"{comm.size} ranks have been allocated")

    if not dry_run and rank == 0:
        if output_path is None:
            raise RuntimeError("generate_input_features: missing output_path")
        if not os.path.isfile(output_path):
            input_file = h5py.File(coords_path, "r")
            output_file = h5py.File(output_path, "w")
            input_file.copy("/H5Types", output_file)
            input_file.close()
            output_file.close()
    comm.barrier()

    population_ranges = read_population_ranges(coords_path, comm)[0]
    population_name = population.name

    reference_u_arc_distance_bounds_dict = {}
    if rank == 0:
        if population_name not in population_ranges:
            raise RuntimeError(
                f"generate_input_features: specified population: {population_name} not found in "
                f"provided coords_path: {coords_path}"
            )
        with h5py.File(coords_path, "r") as coords_f:
            pop_size = population_ranges[population_name][1]
            unique_gid_count = len(
                set(
                    coords_f["Populations"][population_name][distances_namespace][
                        "U Distance"
                    ]["Cell Index"][:]
                )
            )
            if pop_size != unique_gid_count:
                raise RuntimeError(
                    f"generate_input_features: only {unique_gid_count}/{pop_size} unique cell indexes found "
                    f"for specified population: {population_name} in provided coords_path: {coords_path}"
                )
            try:
                reference_u_arc_distance_bounds_dict[population_name] = (
                    coords_f["Populations"][population_name][distances_namespace].attrs[
                        "Reference U Min"
                    ],
                    coords_f["Populations"][population_name][distances_namespace].attrs[
                        "Reference U Max"
                    ],
                )
            except Exception:
                raise RuntimeError(
                    f"generate_input_features: problem locating attributes "
                    f"containing reference bounds in namespace: "
                    f"{distances_namespace} for population: {population_name} from "
                    f"coords_path: {coords_path}"
                )
    comm.barrier()
    reference_u_arc_distance_bounds_dict = comm.bcast(
        reference_u_arc_distance_bounds_dict, root=0
    )

    local_random = np.random.RandomState()

    pop_norm_distances = {}

    if rank == 0:
        logger.info(f"Generating normalized distances for population {population}...")

    reference_u_arc_distance_bounds = reference_u_arc_distance_bounds_dict[population]

    this_pop_norm_distances = {}

    start_time = time.time()
    gid_count = defaultdict(lambda: 0)
    distances_attr_gen = NeuroH5CellAttrGen(
        coords_path,
        population,
        namespace=distances_namespace,
        comm=comm,
        io_size=io_size,
        cache_size=cache_size,
    )

    ## Normalize population distances
    for iter_count, (gid, distances_attr_dict) in enumerate(distances_attr_gen):
        req = comm.Ibarrier()
        if gid is not None:
            if rank == 0:
                logger.info(
                    f"Rank {rank} generating selectivity features for gid {gid}..."
                )
            u_arc_distance = distances_attr_dict["U Distance"][0]
            # v_arc_distance = distances_attr_dict["V Distance"][0]
            norm_u_arc_distance = (
                u_arc_distance - reference_u_arc_distance_bounds[0]
            ) / (
                reference_u_arc_distance_bounds[1] - reference_u_arc_distance_bounds[0]
            )

            this_pop_norm_distances[gid] = norm_u_arc_distance

        features = population.generate_features(
            start_gid=rank, rank=rank, size=comm.size, local_random=local_random
        )

        write_every = max(1, int(math.floor(write_size / comm.size)))

        feature_attr_dict = {}
        n_iter = comm.allreduce(len(features), op=MPI.MAX)
        gid_count = 0
        for i in range(n_iter):
            feature = None
            if i < len(features):
                feature = features[i]
                feature_attr_dict[feature.gid] = feature.to_attribute_dict()
                gid_count += 1
            if (i > 0 and i % write_every == 0) or (debug and i == debug_count):
                req = comm.Ibarrier()
                total_gid_count = comm.reduce(gid_count, root=0, op=MPI.SUM)
                req.wait()
                if rank == 0:
                    logger.info(
                        f"generated {gid_count} features in population {population_name} in "
                        f"{(time.time() - start_time):.2f} s"
                    )

                if not dry_run:
                    req = comm.Ibarrier()
                    if rank == 0:
                        logger.info(
                            f"writing selectivity features for {population_name}..."
                        )
                    append_cell_attributes(
                        output_path,
                        population_name,
                        feature_attr_dict,
                        namespace=output_feature_namespace,
                        comm=comm,
                        io_size=io_size,
                        chunk_size=chunk_size,
                        value_chunk_size=value_chunk_size,
                    )
                    req.wait()
                    del feature_attr_dict
                    feature_attr_dict = {}
                    gc.collect()

            if debug and iter_count >= debug_count:
                break

        pop_norm_distances[population] = this_pop_norm_distances

        total_gid_count = 0
        req = comm.Ibarrier()
        total_gid_count = comm.reduce(gid_count, root=0, op=MPI.SUM)
        req.wait()

        if rank == 0:
            logger.info(
                f"generated selectivity features for {total_gid_count} {population_name} cells in "
                f"{(time.time() - start_time):.2f} s"
            )

        if not dry_run:
            req = comm.Ibarrier()
            if rank == 0:
                logger.info(f"writing selectivity features for {population}...")
            append_cell_attributes(
                output_path,
                population,
                feature_attr_dict,
                namespace=output_feature_namespace,
                comm=comm,
                io_size=io_size,
                chunk_size=chunk_size,
                value_chunk_size=value_chunk_size,
            )
            req.wait()
            del feature_attr_dict
            gc.collect()
        req = comm.Ibarrier()
        req.wait()
