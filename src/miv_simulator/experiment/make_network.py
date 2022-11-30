import logging
import os
import shutil
from dataclasses import dataclass

import commandlib
import miv_simulator
from machinable import Experiment
from machinable.element import normversion
from machinable.types import VersionType
from miv_simulator.simulator import make_h5types
import h5py
from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


def _bin_check(bin: str) -> None:
    if not shutil.which(bin):
        raise FileNotFoundError(f"{bin} not found. Did you add it to the PATH?")


class MakeNetwork(HandlesYAMLConfig, Experiment):
    @dataclass
    class Config(FromYAMLConfig):
        gap_junctions: bool = False
        # resources
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
        return self.local_directory("data/network_h5types.h5")

    def soma_coordinates(self, version: VersionType = None) -> Experiment:
        return self.derive_singleton(
            "miv_simulator.experiment.soma_coordinates",
            [
                {
                    "blueprint": self.config.blueprint,
                    "h5types": self.output_filepath,
                }
            ]
            + normversion(version),
        )

    def dentric_trees(self, population: str) -> str:
        if not self.is_finished():
            raise RuntimeError(
                "You need to execute network generation before creating the dendric trees"
            )
        if population not in ["OLM", "PVBC", "PYR"]:
            raise ValueError("Invalid population")
        # create trees
        h5 = self.local_directory(f"data/dentric_tree_{population}.h5")
        if not os.path.isfile(h5):
            _bin_check("neurotrees_import")
            # generate file
            src = os.path.join(
                os.path.dirname(miv_simulator.__file__),
                "morphology",
                f"{population}.swc",
            )
            commandlib.Command("neurotrees_import", population, h5, src).run()
            commandlib.Command(
                "h5copy",
                "-p",
                "-s",
                "/H5Types",
                "-d",
                "/H5Types",
                "-i",
                self.output_filepath,
                "-o",
                h5,
            ).run()
        return h5

    def synapse_forest(self, population: str) -> str:
        tree = self.dentric_trees(population)
        h5 = self.local_directory(f"data/forest_{population}.h5")
        if not os.path.isfile(h5):
            # determine population ranges
            with h5py.File(self.output_filepath, "r") as f:
                idx = list(
                    reversed(
                        f["H5Types"]["Population labels"].dtype.metadata["enum"]
                    )
                ).index(population)
                offset = f["H5Types"]["Populations"][idx][0]

            _bin_check("neurotrees_copy")
            commandlib.Command(
                "neurotrees_copy",
                "--fill",
                "--output",
                h5,
                tree,
                population,
                offset,
            ).run()

        return h5
