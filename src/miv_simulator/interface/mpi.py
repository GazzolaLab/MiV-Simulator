from dataclasses import dataclass
import subprocess
import sys
import shutil
from machinable import Execution


class LocalExecution(Execution):
    @dataclass
    class Config:
        mpi: str = "mpirun"

    def on_dispatch(self):
        for experiment in self.experiments:
            if experiment.config.get("ranks_", 0) > 0:
                # run using MPI
                script = self.save_file(
                    f"mpi-{experiment.experiment_id}.sh",
                    experiment.dispatch_code(),
                )
                cmd = [
                    shutil.which(self.config.mpi),
                    "-n",
                    str(experiment.config.ranks_),
                    script,
                ]
                # capture output to file and stream it
                with open(experiment.local_directory("output.log"), "wb") as f:
                    p = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                    )
                    for o in iter(p.stdout.readline, b""):
                        sys.stdout.write(o.decode("utf-8"))
                        f.write(o)
            else:
                # default single-threaded execution
                experiment()
