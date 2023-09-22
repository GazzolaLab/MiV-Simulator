import os
import shutil
import subprocess

import h5py
from miv_simulator import config


def _bin_check(bin: str) -> None:
    if not shutil.which(bin):
        raise FileNotFoundError(f"{bin} not found. Did you add it to the PATH?")


def generate_synapse_forest(
    filepath: str,
    tree_output_filepath: str,
    output_filepath: str,
    population: config.PopulationName,
    morphology: config.SWCFilePath,
) -> None:
    # create tree
    if not os.path.isfile(tree_output_filepath):
        _bin_check("neurotrees_import")
        assert (
            subprocess.run(
                [
                    "neurotrees_import",
                    population,
                    tree_output_filepath,
                    morphology,
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
                    filepath,
                    "-o",
                    tree_output_filepath,
                ]
            ).returncode
            == 0
        )

    if not os.path.isfile(output_filepath):
        # determine population ranges
        with h5py.File(filepath, "r") as f:
            idx = list(
                reversed(
                    f["H5Types"]["Population labels"].dtype.metadata["enum"]
                )
            ).index(population)
            offset = f["H5Types"]["Populations"][idx][0]

        _bin_check("neurotrees_copy")
        assert (
            subprocess.run(
                [
                    "neurotrees_copy",
                    "--fill",
                    "--output",
                    output_filepath,
                    tree_output_filepath,
                    population,
                    str(offset),
                ]
            ).returncode
            == 0
        )
