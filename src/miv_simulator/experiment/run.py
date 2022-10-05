import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

import commandlib
import h5py
import numpy as np
from machinable import Experiment
from machinable.config import Field
from miv_simulator import network
from miv_simulator.env import Env
from miv_simulator.utils import config_logging
from mpi4py import MPI

from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


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
        record_syn_spike_count: bool = False
        t_stop: int = 1
        v_init: float = -75.0
        stimulus_onset: float = 1.0
        results_write_time: float = 360.0
        dt: float = 0.025

    def output_filepath(self, path: str = "cells") -> str:
        return self.local_directory(f"data/{path}.h5")

    def dataset(self, kind: str) -> Optional["Experiment"]:
        return self.elements.filter(
            lambda x: x.__model__.module == kind
        ).first()

    def on_execute(self):
        # consolidate generated data files into unified H5
        self.prepare_data()
        # launch simulation
        self.run()

    def run(self):
        config_logging(True)

        np.seterr(all="raise")

        h5types = self.dataset("miv_simulator.experiment.make_h5types")
        input_spike_trains = self.dataset(
            "miv_simulator.experiment.input_spike_trains"
        )

        env = Env(
            comm=MPI.COMM_WORLD,
            config=h5types.config.network,
            template_paths="templates",
            hoc_lib_path=None,
            dataset_prefix=self.local_directory("data"),
            results_path=self.local_directory("data/results", create=True),
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
            spike_input_path=input_spike_trains.output_filepath,
            spike_input_namespace=None,
            spike_input_attr=None,
            cleanup=True,
            cache_queries=False,
            profile_memory=False,
            use_coreneuron=False,
            transfer_debug=False,
            verbose=False,
        )

        network.init(env)
        network.run(
            env, output_syn_spike_count=self.config.record_syn_spike_count
        )

    def prepare_data(self):
        # todo: cache dataset generation across run configurations
        self.local_directory("data", create=True)

        h5types = self.dataset("miv_simulator.experiment.make_h5types")
        soma_coordinates = self.dataset(
            "miv_simulator.experiment.soma_coordinates"
        )
        input_spike_trains = self.dataset(
            "miv_simulator.experiment.input_spike_trains"
        )

        # todo: this should not be hardcoded but inferred from the
        # config, e.g. h5types.config.network...
        MiV_populations = ["PYR", "OLM", "PVBC", "STIM"]
        MiV_IN_populations = ["OLM", "PVBC"]
        MiV_EXT_populations = ["STIM"]

        print("Import H5Types")
        with h5py.File(self.output_filepath("cells"), "w") as f:
            input_file = h5py.File(h5types.output_filepath, "r")
            h5_copy_dataset(input_file, f, "/H5Types")
            input_file.close()

        print("Import coordinate entries")
        with h5py.File(self.output_filepath("cells"), "a") as f_dst:

            grp = f_dst.create_group("Populations")

            for p in MiV_populations:
                grp.create_group(p)

            for p in MiV_populations:
                coords_dset_path = f"/Populations/{p}/Generated Coordinates"
                coords_output_path = f"/Populations/{p}/Coordinates"
                distances_dset_path = f"/Populations/{p}/Arc Distances"
                with h5py.File(soma_coordinates.output_filepath, "r") as f_src:
                    h5_copy_dataset(f_src, f_dst, coords_dset_path)
                    h5_copy_dataset(f_src, f_dst, distances_dset_path)

        print("Create forest entries and synapse attributes")

        def _run(commands):
            cmd = " ".join(commands)
            print(cmd)
            try:
                commandlib.Command(*commands).run()
            except commandlib.exceptions.CommandError as _ex:
                print(f"Error for {p}")
                raise _ex

        for p in MiV_populations:
            if p not in ["OLM", "PVBC", "PYR"]:
                continue
            forest_file = h5types.synapse_forest(p)
            forest_syns_file = h5types.synapse_forest(p)
            forest_dset_path = f"/Populations/{p}/Trees"
            forest_syns_dset_path = f"/Populations/{p}/Synapse Attributes"

            cmd = [
                "h5copy",
                "-p",
                "-s",
                f'"{forest_dset_path}"',
                "-d",
                f'"{forest_dset_path}"',
                "-i",
                f'"{forest_file}"',
                "-o",
                f'"{self.output_filepath()}"',
            ]
            _run(cmd)

        print("Create vector stimulus entries")
        vecstim_file_dict = {"A Diag": input_spike_trains.output_filepath}

        vecstim_dict = {
            f"Input Spikes {stim_id}": stim_file
            for stim_id, stim_file in vecstim_file_dict.items()
        }
        for (vecstim_ns, vecstim_file) in vecstim_dict.items():
            for p in MiV_EXT_populations:
                vecstim_dset_path = f"/Populations/{p}/{vecstim_ns}"
                cmd = [
                    "h5copy",
                    "-p",
                    "-s",
                    f'"{vecstim_dset_path}"',
                    "-d",
                    f'"{vecstim_dset_path}"',
                    "-i",
                    f'"{vecstim_file}"',
                    "-o",
                    f'"{self.output_filepath()}"',
                ]
                _run(cmd)

        with h5py.File(self.output_filepath("connections"), "w") as f:
            input_file = h5py.File(h5types.output_filepath, "r")
            h5_copy_dataset(input_file, f, "/H5Types")
            input_file.close()

        print("Create connectivity entries")
        for p in MiV_populations:
            if p in ["PYR", "PVBC", "OLM", "STIM"]:
                connectivity_file = soma_coordinates.output_filepath
                projection_dset_path = f"/Projections/{p}"
                cmd = [
                    "h5copy",
                    "-p",
                    "-s",
                    f'"{projection_dset_path}"',
                    "-d",
                    f'"{projection_dset_path}"',
                    "-i",
                    f'"{connectivity_file}"',
                    "-o",
                    self.output_filepath("connections"),
                ]
                _run(cmd)
