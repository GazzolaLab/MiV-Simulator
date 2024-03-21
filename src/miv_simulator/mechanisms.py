import os
import shutil
from glob import glob

import subprocess
from mpi4py import MPI
from neuron import h
import hashlib

if hasattr(h, "nrnmpi_init"):
    h.nrnmpi_init()


def compile(
    source: str = "./mechanisms",
    output_path: str = "${source}/compiled",
    force: bool = False,
    recursive: bool = True,
    return_hash: bool = False,
) -> str:
    """
    Compile NEURON NMODL files

    Parameters
    ----------
    source:
        Directory for the mechanism source files. Defaults to ./mechanisms
    output_path:
        Target directory. Defaults to `${source}/compiled`
    force:
        If True, recompile even if source is unchanged
    recursive:
        If True, subdirectories are included
    return_hash:
        If True, only the compilation hash is being returned

    Returns
    -------
    str: compilation path
    """
    src = os.path.abspath(source)
    output_path = os.path.abspath(output_path.replace("${source}", source))

    if not os.path.isdir(src):
        raise FileNotFoundError(f"Mechanism directory does not exists at {src}")

    # compute mechanism hash
    hash_object = hashlib.sha256()
    file_data = {}
    if recursive:
        mod_files = glob(os.path.join(src, "**/*.mod"), recursive=True)
        # ignore output_path if part of this directory
        mod_files = [f for f in mod_files if output_path not in f]
    else:
        mod_files = glob(os.path.join(src, "*.mod"))
    for m in sorted(mod_files, key=lambda x: x.replace(src, "")):
        with open(m, "r") as fm:
            data = fm.read()
            hash_object.update(data.encode())
            file_data[m] = data
    hex_dig = hash_object.hexdigest()

    compiled = os.path.join(output_path, hex_dig)

    # attempt to automatically compile
    if force and os.path.isdir(compiled):
        # remove compiled directory
        shutil.rmtree(compiled)
    if not os.path.isdir(compiled):
        if not shutil.which("nrnivmodl"):
            raise ModuleNotFoundError(
                "nrnivmodl not found. Did you add it to the PATH?"
            )

        try:
            print("Compiling *.mod files via nrnivmodl")
            os.makedirs(compiled)
            for m, data in file_data.items():
                with open(
                    os.path.join(compiled, os.path.basename(m)), "w"
                ) as f:
                    f.write(data)

            subprocess.run(["nrnivmodl"], cwd=compiled, check=True)
        except subprocess.CalledProcessError:
            print("Compilation failed, reverting ...")
            shutil.rmtree(compiled, ignore_errors=True)

    if return_hash:
        return hex_dig

    return compiled


_loaded = {}


def load(directory: str, force: bool = False) -> str:
    """
    Load the output DLL file into NEURON.
    """
    if not force and directory in _loaded:
        # already loaded
        return _loaded[directory]

    dll_path = os.path.join(
        os.path.abspath(directory), "x86_64", ".libs", "libnrnmech.so"
    )
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"{dll_path} does not exists.")

    h(f'nrn_load_dll("{dll_path}")')

    _loaded[directory] = dll_path

    return dll_path


def compile_and_load(
    directory: str = "./mechanisms",
    output_path: str = "${source}/compiled",
    force: bool = False,
    recursive: bool = True,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> str:
    """
    Compile mechanism file on the processor 0, and load the output DLL file into NEURON.
    """
    rank = comm.rank
    if rank == 0:
        compiled = compile(directory, output_path, force, recursive)
    else:
        compiled = None
    comm.barrier()
    compiled = comm.bcast(compiled, root=0)

    return load(compiled)
