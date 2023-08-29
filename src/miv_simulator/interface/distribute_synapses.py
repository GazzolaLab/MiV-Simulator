import logging

from machinable import Component
from miv_simulator import config
from miv_simulator import simulator
from pydantic import BaseModel, Field
from typing import Optional


class DistributeSynapses(Component):
    class Config(BaseModel):
        forest_filepath: str = Field("???")
        cell_types: config.CellTypes = {}
        population: str = Field("???")
        distribution: str = "uniform"
        mechanisms_path: str = "./mechanisms"
        template_path: str = "./templates"
        dt: float = 0.025
        tstop: float = (0.0,)
        celsius: Optional[float] = 35.0
        write_size: int = 1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        use_coreneuron: bool = False
        ranks_: int = 8
        nodes_: int = 1

    def __call__(self):
        logging.basicConfig(level=logging.INFO)
        simulator.distribute_synapses(
            forest_filepath=self.config.forest_filepath,
            cell_types=self.config.cell_types,
            populations=(self.config.population,),
            distribution=self.config.distribution,
            mechanisms_path=self.config.mechanisms_path,
            template_path=self.config.template_path,
            dt=self.config.dt,
            tstop=self.config.tstop,
            celcius=self.config.celsius,
            output_filepath=None,  # modify in-place
            write_size=self.config.write_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            use_coreneuron=self.config.use_coreneuron,
            seed=self.seed,
            io_size=self.config.io_size,
            dry_run=False,
        )
