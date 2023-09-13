from typing import List, Tuple

import logging

from machinable import Component
from pydantic import BaseModel, Field
from miv_simulator import config
from typing import Optional, Dict
from miv_simulator.simulator import distance_connections
from miv_simulator.utils import from_yaml


class DistanceConnections(Component):
    class Config(BaseModel):
        filepath: str = Field("???")
        forest_filepath: str = Field("???")
        axon_extents: config.AxonExtents = Field("???")
        synapses: config.Synapses = Field("???")
        include_forest_populations: Optional[list] = None
        template_path: str = "./templates"
        use_coreneuron: bool = False
        dt: float = 0.025
        tstop: float = 0.0
        celsius: Optional[float] = 35.0
        connectivity_namespace: str = "Connections"
        coordinates_namespace: str = "Coordinates"
        synapses_namespace: str = "Synapse Attributes"
        distances_namespace: str = "Arc Distances"
        resolution: Tuple[int, int, int] = (30, 30, 10)
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        cache_size: int = 1
        write_size: int = 1
        ranks_: int = 8

    def config_from_file(self, filename: str) -> Dict:
        return from_yaml(filename)

    @property
    def output_filepath(self):
        return self.local_directory("connectivity.h5")

    def __call__(self):
        logging.basicConfig(level=logging.INFO)
        distance_connections(
            filepath=self.config.filepath,
            forest_filepath=self.config.forest_filepath,
            include_forest_populations=self.config.include_forest_populations,
            synapses=self.config.synapses,
            axon_extents=self.config.axon_extents,
            template_path=self.config.template_path,
            use_coreneuron=self.config.use_coreneuron,
            dt=self.config.dt,
            tstop=self.config.tstop,
            celsius=self.config.celsius,
            output_filepath=self.output_filepath,
            connectivity_namespace=self.config.connectivity_namespace,
            coordinates_namespace=self.config.coordinates_namespace,
            synapses_namespace=self.config.synapses_namespace,
            distances_namespace=self.config.distances_namespace,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            cache_size=self.config.cache_size,
            write_size=self.config.write_size,
            dry_run=False,
            seeds=self.seed
        )
