import os
import shutil
import shlex
import subprocess

import h5py
from miv_simulator import config


def _bin_check(bin: str) -> None:
    if not shutil.which(bin):
        raise FileNotFoundError(f"{bin} not found. Did you add it to the PATH?")


def _sh(cmd, spawn_process=False):
    if spawn_process:
        try:
            subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            error_message = e.output.decode()
            print(f"{os.getcwd()}$:")
            print(" ".join(cmd))
            print("Error:", error_message)
            raise subprocess.CalledProcessError(
                e.returncode, e.cmd, output=error_message
            )
    else:
        cmdq = " ".join([shlex.quote(c) for c in cmd])
        if os.system(cmdq) != 0:
            raise RuntimeError(f"Error running {cmdq}")


def generate_synapse_forest(
    filepath: str,
    tree_output_filepath: str,
    output_filepath: str,
    population: config.PopulationName,
    morphology: config.SWCFilePath,
    _run=_sh,
) -> None:
    # create tree
    if not os.path.isfile(tree_output_filepath):
        _bin_check("neurotrees_import")
        _run(
            [
                "mpirun -n 1 neurotrees_import",
                population,
                tree_output_filepath,
                morphology,
            ]
        )

        _run(
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
        )

    if not os.path.isfile(output_filepath):
        # determine population ranges
        with h5py.File(filepath, "r") as f:
            h5type_num = f["H5Types"]["Population labels"].dtype.metadata[
                "enum"
            ][population]
            population_range = [
                p for p in f["H5Types"]["Populations"] if p[2] == h5type_num
            ]
            assert len(population_range) == 1
            offset = population_range[0][0]

        _bin_check("neurotrees_copy")
        _run(
            [
                "mpirun -n 1 neurotrees_copy",
                "--fill",
                "--output",
                output_filepath,
                tree_output_filepath,
                population,
                str(offset),
            ]
        )
