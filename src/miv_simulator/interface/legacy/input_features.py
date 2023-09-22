from typing import Dict, Tuple

from machinable import Component
from pydantic import Field, BaseModel
from machinable.element import normversion
from machinable.types import VersionType
from miv_simulator.input_features import generate_input_features


class InputFeatures(Component):
    class Config(BaseModel):
        config_filepath: str = Field("???")
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

    def __call__(self):
        generate_input_features(
            config=self.config.config_filepath,
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
        return self.local_directory("network_input_features.h5")

    def derive_spike_trains(self, version: VersionType = None):
        return self.derive(
            "miv_simulator.interface.legacy.derive_spike_trains",
            [
                {
                    "config_filepath": self.config.config_filepath,
                    "coordinates": self.config.coordinates,
                    "input_features": self.output_filepath,
                }
            ]
            + normversion(version),
            uses=self,
        )
