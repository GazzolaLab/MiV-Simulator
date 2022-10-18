from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from machinable import Experiment
from machinable.config import Field
from machinable.element import normversion
from machinable.types import VersionType
from miv_simulator.simulator import generate_input_features
from neuroh5.io import append_cell_attributes

from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


class InputFeatures(HandlesYAMLConfig, Experiment):
    @dataclass
    class Config(FromYAMLConfig):
        coordinates: str = Field("???")
        distances_namespace: str = "Arc Distances"
        arena_id: str = "A"
        populations: Tuple[str, ...] = ()
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        cache_size: int = 50
        write_size: int = 10000
        gather: bool = True

    def on_execute(self):
        generate_input_features(
            config=self.config.network,
            coords_path=self.config.coordinates,
            distances_namespace=self.config.distances_namespace,
            output_path=self.output_filepath,
            arena_id=self.config.arena_id,
            populations=self.config.populations,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            cache_size=self.config.cache_size,
            write_size=self.config.write_size,
            verbose=True,
            gather=self.config.gather,
            interactive=False,
            debug=False,
            debug_count=10,
            plot=False,
            show_fig=False,
            save_fig=None,
            save_fig_dir=".",
            font_size=14,
            fig_format="png",
            dry_run=False,
        )

    @property
    def output_filepath(self):
        return self.local_directory("data/network_input_features.h5")

    def spike_trains(self, version: VersionType = None):
        return self.derive_singleton(
            "miv_simulator.experiment.input_spike_trains",
            [
                {
                    "network": self.config.network,
                    "coordinates": self.config.coordinates,
                    "input_features": self.output_filepath,
                }
            ]
            + normversion(version),
        )

    def append_inputs(
        self,
        data: Dict,
        spike_train_namespace="MNIST",
        population="STIM",
        spike_train_attr_name="Spike Train",
        chunk_size=1000,
        value_chunk_size=1000,
    ):
        data[0]["SPIKE_TRAIN"]

        trial_duration = 500.0

        spike_attr_dict = defaultdict(list)
        for trial in sorted(data.keys()):
            for gid in data[trial]["SPIKE_TRAIN"]:
                spiketrain = (
                    data[trial]["SPIKE_TRAIN"][gid] * 1000.0
                    + trial * trial_duration
                )
                if len(spiketrain) > 0:
                    spike_attr_dict[gid].append(spiketrain)

        output_spike_attr_dict = dict(
            {
                k: {
                    spike_train_attr_name: np.concatenate(
                        spike_attr_dict[k], dtype=np.float32
                    )
                }
                for k in spike_attr_dict
            }
        )

        append_cell_attributes(
            self.output_filepath,
            population,
            output_spike_attr_dict,
            namespace=spike_train_namespace,
        )
