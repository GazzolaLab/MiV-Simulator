import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from machinable import Experiment
from machinable.config import Field
from miv_simulator.simulator import measure_distances
from miv_simulator.interface.config import BaseConfig


class MeasureDistances(Experiment):
    @dataclass
    class Config(BaseConfig):
        coordinates: str = Field("???")
        geometry: Optional[str] = None
        output_namespace: str = "Generated Coordinates"
        populations: Tuple[str, ...] = ("PYR", "PVBC", "OLM", "STIM")
        interp_chunk_size: int = 1000
        alpha_radius: float = 120.0
        resolution: Tuple[int, int, int] = (30, 30, 10)
        nsample: int = 1000
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        cache_size: int = 50
        ranks_: int = 8

    def on_execute(self):
        logging.basicConfig(level=logging.INFO)
        measure_distances(
            config=self.config.blueprint,
            coords_path=self.config.coordinates,
            coords_namespace=self.config.output_namespace,
            geometry_path=self.config.geometry,
            populations=self.config.populations,
            resolution=self.config.resolution,
            nsample=self.config.nsample,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            cache_size=self.config.cache_size,
            verbose=False,
        )
