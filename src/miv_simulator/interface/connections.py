from typing import List, Tuple

import logging

from machinable import Component
from pydantic import BaseModel, Field, ConfigDict
from miv_simulator import config, simulator
from typing import Optional, Dict
from miv_simulator.utils import from_yaml
from mpi4py import MPI


class Connections(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        filepath: str = Field("???")
        forest_filepath: str = Field("???")
        axon_extents: config.AxonExtents = Field("???")
        synapses: config.Synapses = Field("???")
        include_forest_populations: Optional[list] = None
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
        ranks: int = 8
        nodes: str = "1"

    def config_from_file(self, filename: str) -> Dict:
        return from_yaml(filename)

    @property
    def output_filepath(self):
        return self.local_directory("connectivity.h5")

    def __call__(self):
        logging.basicConfig(level=logging.INFO)
        simulator.generate_connections(
            filepath=self.config.filepath,
            forest_filepath=self.config.forest_filepath,
            include_forest_populations=self.config.include_forest_populations,
            synapses=self.config.synapses,
            axon_extents=self.config.axon_extents,
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
            seeds=self.seed,
        )

    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0

    def compute_context(self):
        context = super().compute_context()
        del context["config"]["filepath"]
        del context["config"]["forest_filepath"]
        del context["config"]["io_size"]
        del context["config"]["chunk_size"]
        del context["config"]["value_chunk_size"]
        del context["config"]["cache_size"]
        del context["config"]["write_size"]
        del context["config"]["ranks"]
        del context["config"]["nodes"]
        context["predicate"]["uses"] = sorted([u.hash for u in self.uses])
        return context
