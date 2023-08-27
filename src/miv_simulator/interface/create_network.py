from machinable import Component
from miv_simulator.utils import io as io_utils
from miv_simulator import config
from pydantic import BaseModel
from machinable.types import VersionType
from machinable.element import normversion


class CreateNetwork(Component):
    class Config(BaseModel):
        cell_distributions: config.CellDistributions = {}
        synapses: config.Synapses = {}

    @property
    def output_filepath(self) -> str:
        return self.local_directory("network.h5")

    def __call__(self) -> None:
        io_utils.create_h5types(
            self.output_filepath,
            self.config.cell_distributions,
            self.config.synapses,
            # todo: add back support for gap-junctions
            gap_junctions=None,
        )

    def soma_coordinates(self, version: VersionType = None) -> "Component":
        return self.derive(
            "miv_simulator.interface.soma_coordinates",
            [
                {
                    "blueprint": {},  # todo!
                    "h5types": self.output_filepath,
                }
            ]
            + normversion(version),
            uses=self,
        )

    def synapse_forest(self, version: VersionType = None) -> "Component":
        return self.derive(
            "miv_simulator.interface.synapse_forest",
            [
                {
                    "h5types": self.output_filepath,
                }
            ]
            + normversion(version),
            uses=self,
        )
