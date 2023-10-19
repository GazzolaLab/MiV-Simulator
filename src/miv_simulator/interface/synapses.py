import logging

from machinable import Component
from miv_simulator import config
from miv_simulator import simulator
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict
from miv_simulator.utils import from_yaml
from mpi4py import MPI


class Synapses(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        forest_filepath: str = Field("???")
        cell_types: config.CellTypes = Field("???")
        population: str = Field("???")
        distribution: str = "uniform"
        mechanisms_path: str = "./mechanisms"
        template_path: str = "./templates"
        io_size: int = -1
        write_size: int = 1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        ranks_: int = 8
        nodes_: int = 1

    def config_from_file(self, filename: str) -> Dict:
        return from_yaml(filename)

    @property
    def output_filepath(self) -> str:
        return self.local_directory("synapses.h5")

    def __call__(self):
        logging.basicConfig(level=logging.INFO)
        simulator.distribute_synapses(
            forest_filepath=self.config.forest_filepath,
            cell_types=self.config.cell_types,
            populations=(self.config.population,),
            distribution=self.config.distribution,
            mechanisms_path=self.config.mechanisms_path,
            template_path=self.config.template_path,
            output_filepath=self.output_filepath,
            io_size=self.config.io_size,
            write_size=self.config.write_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            seed=self.seed,
            dry_run=False,
        )

    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0
