import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from machinable import Experiment
from machinable.config import Field
import miv_simulator.network
from miv_simulator.env import Env
from miv_simulator.utils import config_logging
from mpi4py import MPI
from miv_simulator.mechanisms import compile_and_load


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
        using: List[str] = Field(default_factory=lambda: [])
        record_syn_spike_count: bool = False
        t_stop: int = 1
        v_init: float = -75.0
        stimulus_onset: float = 1.0
        results_write_time: float = 360.0
        dt: float = 0.025
        ranks_: int = 8

    def on_create(self):
        compile_and_load()

        self.using = []
        for use in self.config.using:
            experiment = Experiment.find(use)
            if not experiment:
                raise ValueError(f"Invalid use: {use}")
            self.using.append(experiment)

    def dependencies(self, kind: Optional[str] = None):
        return list(filter(lambda x: x.module == kind, self.using))

    def on_execute(self):
        self.local_directory("data", create=True)
        config_logging(True)
        np.seterr(all="raise")

        network = self.dependencies("miv_simulator.experiment.make_network")[0]
        spike_trains = self.dependencies(
            "miv_simulator.experiment.derive_spike_trains"
        )[0]
        custom_spike_train_meta = spike_trains.custom_spike_train_meta
        data = Experiment.singleton(
            "miv_simulator.experiment.prepare_data",
            {"using": self.config.using},
        ).execute()

        if data.is_finished():
            print(
                f"Using existing H5 data from {data.experiment_id} at {data.local_directory()}"
            )

        data_configuration = {
            "Model Name": "simulation",
            "Dataset Name": "simulation",
            "Cell Data": "cells.h5",
            "Connection Data": "connections.h5",
        }

        env = Env(
            comm=MPI.COMM_WORLD,
            config={**network.config.blueprint, **data_configuration},
            template_paths="templates",
            hoc_lib_path=None,
            dataset_prefix=data.local_directory("data"),
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
            spike_input_path=spike_trains.output_filepath,
            spike_input_namespace=custom_spike_train_meta["namespace"],
            spike_input_attr=custom_spike_train_meta["attr_name"],
            cleanup=True,
            cache_queries=False,
            profile_memory=False,
            use_coreneuron=False,
            transfer_debug=False,
            verbose=False,
        )

        miv_simulator.network.init(env)
        miv_simulator.network.run(
            env,
            output_syn_spike_count=self.config.record_syn_spike_count,
            output=False,
        )
