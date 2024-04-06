from machinable import Component
from pydantic import BaseModel, Field, ConfigDict
from miv_simulator import config
from mpi4py import MPI
from miv_simulator.utils import io as io_utils, from_yaml
from typing import Dict, Optional


class H5Types(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        cell_distributions: config.CellDistributions = Field("???")
        projections: config.SynapticProjections = Field("???")
        population_definitions: Dict[str, int] = Field("???")
        ranks: int = 1
        nodes: str = "1"

    def config_from_file(self, filename: str) -> Dict:
        return from_yaml(filename)

    @property
    def output_filepath(self) -> str:
        return self.local_directory("neuro.h5")

    def __call__(self) -> None:
        if MPI.COMM_WORLD.rank == 0:
            io_utils.create_neural_h5(
                self.output_filepath,
                self.config.cell_distributions,
                synapses={
                    post: {pre: True for pre in v}
                    for post, v in self.config.projections.items()
                },
                population_definitions=self.config.population_definitions,
            )
        MPI.COMM_WORLD.barrier()

    def compute_context(self):
        context = super().compute_context()
        del context["config"]["ranks"]
        del context["config"]["nodes"]
        return context
