import logging

from machinable import Component
from miv_simulator import Config
from miv_simulator.simulator import distribute_synapse_locations
from pydantic import BaseModel, Field


class DistributeSynapseLocations(Component):
    class Config(BaseModel):
        blueprint: Blueprint = Field(default_factory=Blueprint)
        population: str = Field("???")
        coordinates: str = Field("???")
        forest: str = Field("???")
        templates: str = "templates"
        mechanisms: str = "./mechanisms"
        distribution: str = "uniform"
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        write_size: int = 1
        # resources
        ranks_: int = 8
        nodes_: int = 1

    def __call__(self):
        logging.basicConfig(level=logging.INFO)
        distribute_synapse_locations(
            config=self.config.blueprint,
            template_path=self.config.templates,
            output_path=self.config.forest,  # modify in-place
            forest_path=self.config.forest,
            populations=[self.config.population],
            distribution=self.config.distribution,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            write_size=self.config.write_size,
            verbose=True,
            dry_run=False,
            debug=False,
            mechanisms_path=self.config.mechanisms,
        )
