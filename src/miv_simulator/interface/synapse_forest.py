import os
import shutil
import subprocess

import h5py
from machinable import Component
from miv_simulator import config
from pydantic import BaseModel, Field


def _bin_check(bin: str) -> None:
    if not shutil.which(bin):
        raise FileNotFoundError(f"{bin} not found. Did you add it to the PATH?")


SWCFilePath = str


class GenerateSynapseForest(Component):
    class Config(BaseModel):
        filepath: str = Field("???")
        population: config.PopulationName = Field("???")
        morphology: SWCFilePath = Field("???")

    @property
    def tree_output_filepath(self) -> str:
        return self.local_directory("dentric_tree.h5")

    @property
    def output_filepath(self) -> str:
        return self.local_directory("forest.h5")

    def __call__(self) -> None:
        # create tree
        if not os.path.isfile(self.tree_output_filepath):
            _bin_check("neurotrees_import")
            assert (
                subprocess.run(
                    [
                        "neurotrees_import",
                        self.config.population,
                        self.tree_output_filepath,
                        self.config.morphology,
                    ]
                ).returncode
                == 0
            )
            assert (
                subprocess.run(
                    [
                        "h5copy",
                        "-p",
                        "-s",
                        "/H5Types",
                        "-d",
                        "/H5Types",
                        "-i",
                        self.config.filepath,
                        "-o",
                        self.tree_output_filepath,
                    ]
                ).returncode
                == 0
            )

        if not os.path.isfile(self.output_filepath):
            # determine population ranges
            with h5py.File(self.config.filepath, "r") as f:
                idx = list(
                    reversed(
                        f["H5Types"]["Population labels"].dtype.metadata["enum"]
                    )
                ).index(self.config.population)
                offset = f["H5Types"]["Populations"][idx][0]

            _bin_check("neurotrees_copy")
            assert (
                subprocess.run(
                    [
                        "neurotrees_copy",
                        "--fill",
                        "--output",
                        self.output_filepath,
                        self.tree_output_filepath,
                        self.config.population,
                        str(offset),
                    ]
                ).returncode
                == 0
            )
