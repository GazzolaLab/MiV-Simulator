from dataclasses import dataclass
from typing import Tuple, Optional, Union, List, Dict
from collections import defaultdict

import numpy as np

from machinable import Experiment
from machinable.config import Field
from miv_simulator.simulator import (
    generate_input_spike_trains,
    import_input_spike_train,
)

from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


class DeriveSpikeTrains(HandlesYAMLConfig, Experiment):
    @dataclass
    class Config(FromYAMLConfig):
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

    def on_execute(self):
        generate_input_spike_trains(
            config=self.config.blueprint,
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
        spike_train: Union[List, np.ndarray],
        namespace: str = "Custom",
        attr_name: str = "Input Spikes",
    ) -> "DeriveSpikeTrains":
        self.set_custom(namespace, attr_name)

        import_input_spike_train(
            spike_train,
            namespace=namespace,
            attr_name=attr_name,
            output_filepath=self.output_filepath,
        )

        return self

    def set_custom(
        self,
        namespace: str = "Custom",
        attr_name: str = "Input Spikes",
    ):
        self.save_data(
            "custom_spike_train_meta.json",
            {
                "namespace": namespace,
                "attr_name": attr_name,
            },
        )

    @property
    def custom_spike_train_meta(self) -> Optional[Dict]:
        return self.load_data(
            "custom_spike_train_meta.json", defaultdict(lambda: None)
        )

    @property
    def output_filepath(self):
        return self.local_directory("data/network_input_spike_trains.h5")
