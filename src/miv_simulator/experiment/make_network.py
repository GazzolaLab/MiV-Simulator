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

from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


def _bin_check(bin: str) -> None:
    if not shutil.which(bin):
        raise FileNotFoundError(f"{bin} not found. Did you add it to the PATH?")


class MakeNetwork(HandlesYAMLConfig, Experiment):
    @dataclass
    class Config(FromYAMLConfig):
        gap_junctions: bool = False

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
                "datasets",
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
            _bin_check("neurotrees_copy")
            commandlib.Command(
                "neurotrees_copy",
                "--fill",
                "--output",
                h5,
                tree,
                population,
                {"PYR": 10, "PVBC": 90, "OLM": 143}[population],
            ).run()

        return h5
