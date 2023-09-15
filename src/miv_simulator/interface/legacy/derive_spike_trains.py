from typing import Dict, List, Optional, Tuple, Union

import arrow
import numpy as np
from machinable import Component
from pydantic import Field, BaseModel
from miv_simulator.simulator import (
    generate_input_spike_trains,
    import_input_spike_train,
)


class DeriveSpikeTrains(Component):
    class Config(BaseModel):
        config_filepath: str = Field("???")
        input_features: str = Field("???")
        coordinates: Optional[str] = None
        distances_namespace: str = "Arc Distances"
        arena_id: str = "A"
        populations: Tuple[str, ...] = ()
        n_trials: int = 1
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        cache_size: int = 50
        write_size: int = 10000
        gather: bool = True
        spikes_namespace: str = "Input Spikes"
        spike_train_attr_name: str = "Spike Train"
        # resources
        ranks_: int = 1

    def on_instantiate(self):
        self.active_spike_input_namespace = None
        self.active_spike_input_attr = None

    def __call__(self):
        generate_input_spike_trains(
            config=self.config.config_filepath,
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
            save_fig=None,
            save_fig_dir=".",
            font_size=14.0,
            fig_format="svg",
            verbose=True,
            dry_run=False,
        )

    def from_numpy(
        self,
        spike_trains: Dict[int, Union[List, np.ndarray]],
        namespace: str = "Custom",
        attr_name: str = "Input Spikes",
    ) -> "DeriveSpikeTrains":
        import_input_spike_train(
            spike_trains,
            namespace=namespace,
            attr_name=attr_name,
            output_filepath=self.output_filepath,
        )

        self.active_spike_input_namespace = namespace
        self.active_spike_input_attr = attr_name

        self.save_file("refreshed_at", str(arrow.now()))

        return self

    @property
    def refreshed_at(self):
        refreshed_at = self.load_file("refreshed_at")
        if refreshed_at is not None:
            return arrow.get(refreshed_at)

        return self.finished_at()

    @property
    def output_filepath(self):
        return self.local_directory("network_input_spike_trains.h5")
