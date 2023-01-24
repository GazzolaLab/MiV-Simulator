import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from machinable import Experiment
from machinable.config import Field, validator
import miv_simulator.network
from miv_simulator.env import Env
from miv_simulator.utils import config_logging
from mpi4py import MPI
from miv_simulator.mechanisms import compile_and_load
from miv_simulator.config import Blueprint


def h5_copy_dataset(f_src, f_dst, dset_path):
    print(f"Copying {dset_path} from {f_src} to {f_dst} ...")
    target_path = str(pathlib.Path(dset_path).parent)
    f_src.copy(f_src[dset_path], f_dst[target_path])


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stderr.flush()
    sys.stdout.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


class RunNetwork(Experiment):
    @dataclass
    class Config:
        blueprint: Blueprint = Field(default_factory=Blueprint)
        cells: str = Field("???")
        connections: str = Field("???")
        spike_input_path: str = Field("???")
        spike_input_namespace: Optional[str] = None
        spike_input_attr: Optional[str] = None
        record_syn_spike_count: bool = False
        mechanisms: str = "./mechanisms"
        t_stop: int = 1
        v_init: float = -75.0
        stimulus_onset: float = 1.0
        results_write_time: float = 360.0
        dt: float = 0.025
        ranks_: int = 8
        nodes_: int = 1

    def on_create(self):
        compile_and_load(self.config.mechanisms)

    def on_execute(self):
        self.local_directory("data", create=True)
        config_logging(True)
        np.seterr(all="raise")

        data_configuration = {
            "Model Name": "simulation",
            "Dataset Name": "simulation",
            "Cell Data": self.config.cells,
            "Connection Data": self.config.connections,
        }

        blueprint = self.config.blueprint or {}

        self.env = env = Env(
            comm=MPI.COMM_WORLD,
            config={**blueprint, **data_configuration},
            template_paths="templates",
            hoc_lib_path=None,
            dataset_prefix="",
            results_path=self.local_directory("data"),
            results_file_id=None,
            results_namespace_id=None,
            node_rank_file=None,
            node_allocation=None,
            io_size=0,
            use_cell_attr_gen=False,
            cell_attr_gen_cache_size=10,
            recording_profile=None,
            tstart=0.0,
            tstop=self.config.t_stop,
            v_init=self.config.v_init,
            stimulus_onset=self.config.stimulus_onset,
            n_trials=1,
            max_walltime_hours=1.0,
            checkpoint_interval=500.0,
            checkpoint_clear_data=True,
            nrn_timeout=600.0,
            results_write_time=self.config.results_write_time,
            dt=self.config.dt,
            ldbal=False,
            lptbal=False,
            cell_selection_path=None,
            microcircuit_inputs=False,
            spike_input_path=self.config.spike_input_path,
            spike_input_namespace=self.config.spike_input_namespace,
            spike_input_attr=self.config.spike_input_attr,
            cleanup=True,
            cache_queries=False,
            profile_memory=False,
            use_coreneuron=False,
            transfer_debug=False,
            verbose=False,
        )

        miv_simulator.network.init(env)

        summary = miv_simulator.network.run(
            env,
            output_syn_spike_count=self.config.record_syn_spike_count,
            output=False,
        )

        if self.on_write_meta_data() is not False:
            self.save_data(
                f"timing_summary_rank{int(env.pc.id())}.json", summary
            )

    def on_write_meta_data(self) -> Optional[bool]:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        return rank == 0

    def on_after_dispatch(self):
        return miv_simulator.network.shutdown(self.env)
