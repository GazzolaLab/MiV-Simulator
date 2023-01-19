import logging
import os
from dataclasses import dataclass

from machinable import Experiment
from machinable.element import normversion
from machinable.config import Field
from machinable.types import VersionType
from miv_simulator.simulator import make_h5types
from miv_simulator.config import Blueprint


class MakeNetwork(Experiment):
    @dataclass
    class Config:
        blueprint: Blueprint = Field(default_factory=Blueprint)
        gap_junctions: bool = False
        ranks_: int = 1

    def version_microcircuit(self):
        return {
            "blueprint": {
                "Geometry": {
                    "Parametric Surface": {
                        "Layer Extents": {
                            "SO": [[0.0, 0.0, 0.0], [1000.0, 1000.0, 100.0]],
                            "SP": [[0.0, 0.0, 100.0], [1000.0, 1000.0, 150.0]],
                            "SR": [[0.0, 0.0, 150.0], [1000.0, 1000.0, 350.0]],
                            "SLM": [[0.0, 0.0, 350.0], [1000.0, 1000.0, 450.0]],
                        }
                    },
                    "Cell Distribution": {
                        "STIM": {"SO": 0, "SP": 10, "SR": 0, "SLM": 0},
                        "PYR": {"SO": 0, "SP": 80, "SR": 0, "SLM": 0},
                        "PVBC": {"SO": 35, "SP": 10, "SR": 8, "SLM": 0},
                        "OLM": {"SO": 44, "SP": 0, "SR": 0, "SLM": 0},
                    },
                }
            }
        }

    def on_execute(self) -> None:
        logging.basicConfig(level=logging.INFO)
        make_h5types(
            self.config.blueprint,
            self.output_filepath,
            self.config.gap_junctions,
        )

    @property
    def output_filepath(self) -> str:
        return self.local_directory("data/", create=True) + "network_h5types.h5"

    def soma_coordinates(self, version: VersionType = None) -> "Experiment":
        return self.derive(
            "miv_simulator.interface.soma_coordinates",
            [
                {
                    "blueprint": self.config.blueprint,
                    "h5types": self.output_filepath,
                }
            ]
            + normversion(version),
        )

    def synapse_forest(self, version: VersionType = None) -> "Experiment":
        return self.derive(
            "miv_simulator.interface.synapse_forest",
            [
                {
                    "h5types": self.output_filepath,
                }
            ]
            + normversion(version),
        )
