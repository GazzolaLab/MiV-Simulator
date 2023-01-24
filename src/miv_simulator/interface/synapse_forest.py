import shutil
import os

import commandlib
from machinable import Experiment
from dataclasses import dataclass
from machinable.config import Field
import h5py


def _bin_check(bin: str) -> None:
    if not shutil.which(bin):
        raise FileNotFoundError(f"{bin} not found. Did you add it to the PATH?")


class GenerateSynapseForest(Experiment):
    @dataclass
    class Config:
        h5types: str = Field("???")
        population: str = Field("???")
        morphology: str = Field("???")

    @property
    def tree_output_filepath(self) -> str:
        return self.local_directory("data/", create=True) + "dentric_tree.h5"

    @property
    def output_filepath(self) -> str:
        return self.local_directory("data/", create=True) + "forest.h5"

    def on_execute(self) -> None:
        # create tree
        if not os.path.isfile(self.tree_output_filepath):
            _bin_check("neurotrees_import")
            commandlib.Command(
                "neurotrees_import",
                self.config.population,
                self.tree_output_filepath,
                self.config.morphology,
            ).run()
            commandlib.Command(
                "h5copy",
                "-p",
                "-s",
                "/H5Types",
                "-d",
                "/H5Types",
                "-i",
                self.config.h5types,
                "-o",
                self.tree_output_filepath,
            ).run()

        if not os.path.isfile(self.output_filepath):
            # determine population ranges
            with h5py.File(self.config.h5types, "r") as f:
                idx = list(
                    reversed(
                        f["H5Types"]["Population labels"].dtype.metadata["enum"]
                    )
                ).index(self.config.population)
                offset = f["H5Types"]["Populations"][idx][0]

            _bin_check("neurotrees_copy")
            commandlib.Command(
                "neurotrees_copy",
                "--fill",
                "--output",
                self.output_filepath,
                self.tree_output_filepath,
                self.config.population,
                offset,
            ).run()
