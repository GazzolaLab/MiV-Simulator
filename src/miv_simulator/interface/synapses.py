import logging

from machinable import Component
from miv_simulator import config
from miv_simulator import simulator
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Dict
from miv_simulator.utils import from_yaml
from mpi4py import MPI
from miv_simulator import mechanisms


class Synapses(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        forest_filepath: str = Field("???")
        cell_types: config.CellTypes = Field("???")
        population: str = Field("???")
        distribution: str = "uniform"
        mechanisms_path: str = "./mechanisms/compiled"
        template_path: str = "./templates"
        io_size: int = -1
        write_size: int = 1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        ranks: int = 8

        @field_validator("cell_types")
        @classmethod
        def template_must_not_be_reduced(cls, v):
            for population, d in v.items():
                if d["template"].lower() in ["brk_nrn", "pr_nrn", "sc_nrn"]:
                    raise ValueError(
                        f"Reduced template {d['template']} for population {population}. A non-reduced template is required for synapse generation."
                    )

            return v

    def config_from_file(self, filename: str) -> Dict:
        return from_yaml(filename)

    @property
    def output_filepath(self) -> str:
        return self.local_directory("synapses.h5")

    def __call__(self):
        logging.basicConfig(level=logging.INFO)
        mechanisms.load(self.config.mechanisms_path)
        simulator.distribute_synapses(
            forest_filepath=self.config.forest_filepath,
            cell_types=self.config.cell_types,
            populations=(self.config.population,),
            distribution=self.config.distribution,
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
