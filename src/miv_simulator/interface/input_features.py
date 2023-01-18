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
from miv_simulator.interface.config import BaseConfig


class InputFeatures(Experiment):
    @dataclass
    class Config(BaseConfig):
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
        # resources
        ranks_: int = 1

    def on_execute(self):
        generate_input_features(
            config=self.config.blueprint,
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
        return (
            self.local_directory("data/", create=True)
            + "network_input_features.h5"
        )

    def derive_spike_trains(self, version: VersionType = None):
        return self.derive(
            "miv_simulator.interface.derive_spike_trains",
            [
                {
                    "blueprint": self.config.blueprint,
                    "coordinates": self.config.coordinates,
                    "input_features": self.output_filepath,
                }
            ]
            + normversion(version),
        )
