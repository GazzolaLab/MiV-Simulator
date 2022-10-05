import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import miv_simulator
from machinable import Experiment
from machinable.config import Field
from miv_simulator.mechanisms import compile_and_load
from miv_simulator.simulator import distribute_synapse_locations

from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


class SynapseLocations(HandlesYAMLConfig, Experiment):
    @dataclass
    class Config(FromYAMLConfig):
        population: str = Field("???")
        coordinates: str = Field("???")
        forest: str = Field("???")
        templates: Optional[str] = None
        distribution: str = "uniform"
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        write_size: int = 1

    @property
    def output_filepath(self):
        return self.local_directory(f"data/forest_{self.config.population}.h5")

    def on_execute(self):
        logging.basicConfig(level=logging.INFO)
        templates = self.config.templates
        if templates is None:
            templates = os.path.join(
                os.path.dirname(miv_simulator.__file__), "templates"
            )
        compile_and_load()
        distribute_synapse_locations(
            config=self.config.network,
            template_path=templates,
            output_path=self.output_filepath,
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
