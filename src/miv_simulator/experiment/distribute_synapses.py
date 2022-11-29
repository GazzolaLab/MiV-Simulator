import logging
from dataclasses import dataclass

from machinable import Experiment
from machinable.config import Field
from miv_simulator.simulator import distribute_synapse_locations
from miv_simulator.mechanisms import compile_and_load


from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


class DistributeSynapseLocations(HandlesYAMLConfig, Experiment):
    @dataclass
    class Config(FromYAMLConfig):
        population: str = Field("???")
        coordinates: str = Field("???")
        forest: str = Field("???")
        templates: str = "templates"
        distribution: str = "uniform"
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        write_size: int = 1
        # resources
        ranks_: int = 8

    def on_execute(self):
        # compile_and_load()
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
        )
