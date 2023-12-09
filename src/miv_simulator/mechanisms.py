import os
import shutil
from glob import glob

import subprocess
from mpi4py import MPI
from neuron import h

if hasattr(h, "nrnmpi_init"):
    h.nrnmpi_init()

from typing import Optional


def compile(directory: str = "./mechanisms", force: bool = False) -> str:
    """
    Compile NEURON NMODL files

    Parameters
    ----------
    directory:
        Directory for the mechanism source files. Defaults to ./mechanisms
    force : bool
        Force recompile

    Returns
    -------
    str: compilation path
    """
    src = os.path.abspath(directory)

    if not os.path.isdir(src):
        raise FileNotFoundError(f"Mechanism directory does not exists at {src}")

    # attempt to automatically compile
    compiled = os.path.join(src, "compiled")
    if force and os.path.isdir(compiled):
        # remove compiled directory
        shutil.rmtree(compiled)
    if not os.path.isdir(compiled):
        print("Attempting to compile *.mod files via nrnivmodl")
        # move into compiled directory
        os.makedirs(compiled)
        for m in glob(os.path.join(src, "**/*.mod"), recursive=True):
            shutil.copyfile(m, os.path.join(compiled, os.path.basename(m)))
        # compile
        if not shutil.which("nrnivmodl"):
            raise ModuleNotFoundError(
                "nrnivmodl not found. Did you add it to the PATH?"
            )
        subprocess.run(["nrnivmodl"], cwd=compiled)

    return compiled


_loaded = {}


def load(directory: str = "./mechanisms", force: bool = False) -> str:
    """
    Load the output DLL file into NEURON.
    """
    if not force and directory in _loaded:
        # already loaded
        return _loaded[directory]

    src = os.path.abspath(directory)
    compiled = os.path.join(src, "compiled")

    dll_path = os.path.join(compiled, "x86_64", ".libs", "libnrnmech.so")
    assert os.path.exists(
        dll_path
    ), f"libnrnmech.so file is not found properly. {dll_path}"
    h(f'nrn_load_dll("{dll_path}")')

    _loaded[directory] = dll_path

    return dll_path


def compile_and_load(
    directory: str = "./mechanisms", force: bool = False
) -> str:
    """
    Compile mechanism file on the processor 0, and load the output DLL file into NEURON.

    WARNING: The used MPI barriers might cause trouble if this
             function is called within an MPI process.
    """
    if not force and directory in _loaded:
        # already loaded
        return _loaded[directory]

    comm = MPI.COMM_WORLD
    rank = comm.rank

    subcomm = comm.Split(color=rank == 0, key=rank)

    if rank == 0:
        src = compile(directory, force)
    else:
        src = None
    subcomm.barrier()
    src = subcomm.bcast(src, root=0)

    return load(directory, force)
