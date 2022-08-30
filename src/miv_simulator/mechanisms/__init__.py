import os
import shutil
from glob import glob

import commandlib
import miv_simulator
from neuron import h


def compile() -> str:
    # attempt to automatically compile
    src = os.path.join(os.path.dirname(miv_simulator.__file__), "mechanisms")
    compiled = os.path.join(src, "compiled")
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


def compile_and_load():
    src = compile()
    h(
        f"nrn_load_dll(\"{os.path.join(src, 'x86_64', '.libs', 'libnrnmech.so')}\")"
    )
