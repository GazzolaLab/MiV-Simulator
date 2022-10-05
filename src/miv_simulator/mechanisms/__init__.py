import os
import shutil
from glob import glob

import commandlib
import miv_simulator
from neuron import h

from typing import Optional


def compile(source_directory: Optional[str] = None) -> str:
    # attempt to automatically compile
    if source_directory is None:
        source_directory = os.path.join(
            os.path.dirname(miv_simulator.__file__), "mechanisms"
        )
    compiled = os.path.join(source_directory, "compiled")
    if not os.path.isdir(compiled):
        print("Attempting to compile *.mod files via nrnivmodl")
        # move into compiled directory
        os.makedirs(compiled)
        for m in glob(
            os.path.join(source_directory, "**/*.mod"), recursive=True
        ):
            shutil.copyfile(m, os.path.join(compiled, os.path.basename(m)))
        # compile
        if not shutil.which("nrnivmodl"):
            raise ModuleNotFoundError(
                "nrnivmodl not found. Did you add it to the PATH?"
            )
        nrnivmodl = commandlib.Command("nrnivmodl").in_dir(compiled)
        nrnivmodl.run()

    return compiled


def compile_and_load():
    src = compile()
    h(
        f"nrn_load_dll(\"{os.path.join(src, 'x86_64', '.libs', 'libnrnmech.so')}\")"
    )
