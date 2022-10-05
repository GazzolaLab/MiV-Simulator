from dataclasses import dataclass
from typing import Tuple, Optional

from machinable import Experiment
from machinable.config import Field
from miv_simulator.simulator import generate_input_spike_trains

from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


class InputSpikeTrains(HandlesYAMLConfig, Experiment):
    @dataclass
    class Config(FromYAMLConfig):
        input_features: str = Field("???")
        coordinates: Optional[str] = None
        distances_namespace: str = "Arc Distances"
        arena_id: str = "A"
        populations: Tuple[str] = None
        n_trials: int = 1
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        cache_size: int = 50
        write_size: int = 10000
        gather: bool = True
        spikes_namespace: str = "Input Spikes"
        spike_train_attr_name: str = "Spike Train"

    def on_execute(self):
        generate_input_spike_trains(
            config=self.config.network,
            selectivity_path=self.config.input_features,
            selectivity_namespace="Selectivity",
            coords_path=self.config.coordinates,
            distances_namespace=self.config.distances_namespace,
            arena_id=self.config.arena_id,
            populations=self.config.populations,
            n_trials=self.config.n_trials,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            cache_size=self.config.cache_size,
            write_size=self.config.write_size,
            output_path=self.output_filepath,
            spikes_namespace=self.config.spikes_namespace,
            spike_train_attr_name=self.config.spike_train_attr_name,
            phase_mod=False,
            gather=self.config.gather,
            debug=False,
            plot=False,
            show_fig=False,
            save_fig=False,
            save_fig_dir=".",
            font_size=14.0,
            fig_format="svg",
            verbose=True,
            dry_run=False,
        )

    @property
    def output_filepath(self):
        return self.local_directory("data/network_input_spike_trains.h5")
