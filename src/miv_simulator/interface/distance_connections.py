from typing import List, Tuple

import logging
from dataclasses import dataclass

from machinable import Component
from machinable.config import Field
from miv_simulator.config import Blueprint
from miv_simulator.simulator import generate_distance_connections


class DistanceConnections(Component):
    @dataclass
    class Config:
        blueprint: Blueprint = Field(default_factory=Blueprint)
        coordinates: str = Field("???")
        forest: str = Field("???")
        include: List[str] = Field(default_factory=lambda: [])
        connectivity_namespace: str = "Connectivity"
        coordinates_namespace: str = "Coordinates"
        synapses_namespace: str = "Synapse Attributes"
        distances_namespace: str = "Arc Distances"
        resolution: Tuple[int, int, int] = (30, 30, 10)
        interp_chunk_size: int = 1000
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        write_size: int = 1
        cache_size: int = 1
        # resources
        ranks_: int = 8

    @property
    def output_filepath(self):
        return self.local_directory("data/", create=True) + "connectivity.h5"

    def __call__(self):
        logging.basicConfig(level=logging.INFO)

        generate_distance_connections(
            config=self.config.blueprint,
            include=self.config.include,
            forest_path=self.config.forest,
            connectivity_path=self.output_filepath,
            connectivity_namespace=self.config.connectivity_namespace,
            coords_path=self.config.coordinates,
            coords_namespace=self.config.coordinates_namespace,
            synapses_namespace=self.config.synapses_namespace,
            distances_namespace=self.config.distances_namespace,
            resolution=self.config.resolution,
            interp_chunk_size=self.config.interp_chunk_size,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            cache_size=self.config.cache_size,
            write_size=self.config.write_size,
            verbose=True,
            dry_run=False,
            debug=False,
        )
