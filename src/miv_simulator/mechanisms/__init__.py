import os
import shutil
from glob import glob

import commandlib
import miv_simulator
from neuron import h

from typing import Optional


def compile(directory: Optional[str] = None, force: bool = False) -> str:
    """
    Compile NEURON NMODL files

    Parameters
    ----------
    directory:
        Optional directory of the source files. If None, package default machanism will be used
    force : bool
        Force recompile

    Returns
    -------
    str: compilation path
    """
    # infer compilation source
    default_directory = os.path.join(
        os.path.dirname(miv_simulator.__file__), "mechanisms"
    )
    if directory is None:
        src = default_directory
    else:
        src = os.path.abspath(directory)

    # attempt to automatically compile
    compiled = os.path.join(src, "compiled")
    if force:
        # remove compiled directory
        remove_cmd = commandlib.Command("rm", "-rf")
        remove_cmd(compiled).run()
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
        nrnivmodl = commandlib.Command("nrnivmodl").in_dir(compiled)
        nrnivmodl.run()

    return compiled


def compile_and_load(
    directory: Optional[str] = None, force: bool = False
) -> str:
    """Compile and load dll file into NEURON"""
    src = compile(directory, force)
    dll_path = os.path.join(src, "x86_64", ".libs", "libnrnmech.so")
    assert os.path.exists(dll_path), "libnrnmech.so file is not found properly."
    h(f'nrn_load_dll("{dll_path}")')

    return dll_path
