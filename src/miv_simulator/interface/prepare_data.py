import os
import pathlib
import sys
from typing import List

import commandlib
import h5py
from machinable import Experiment
from neuroh5.io import read_population_names


def h5_copy_dataset(f_src, f_dst, dset_path):
    print(f"Copying {dset_path} from {f_src} to {f_dst} ...")
    target_path = str(pathlib.Path(dset_path).parent)
    f_src.copy(f_src[dset_path], f_dst[target_path])


class PrepareData(Experiment):
    def output_filepath(self, path: str = "cells") -> str:
        return self.local_directory(f"data/simulation/{path}.h5")

    def on_compute_predicate(self):
        return {
            "uses*": sorted(
                map(
                    lambda x: x.experiment_id,
                    self.uses,
                )
            )
        }

    def on_create(self):
        # load dependencies
        self.soma_coordinates = None
        self.network = None
        self.spike_trains = None
        self.distance_connections = {}
        self.synapse_forest = {}
        for dependency in self.uses:
            if dependency.module == "miv_simulator.interface.soma_coordinates":
                self.soma_coordinates = dependency
            elif dependency.module == "miv_simulator.interface.make_network":
                self.network = dependency
            elif (
                dependency.module
                == "miv_simulator.interface.derive_spike_trains"
            ):
                self.spike_trains = dependency
            elif (
                dependency.module
                == "miv_simulator.interface.distance_connections"
            ):
                populations = read_population_names(dependency.config.forest)
                for p in populations:
                    if p in self.distance_connections:
                        # check for duplicates
                        raise ValueError(
                            f"Redundant distance connection specification for population {p}"
                            f"Found duplicate in {dependency.config.forest}, while already "
                            f"defined in {self.distance_connections[p].config.forest}"
                        )
                    self.distance_connections[p] = dependency
            elif dependency.module == "miv_simulator.interface.synapse_forest":
                if dependency.config.population in self.synapse_forest:
                    # check for duplicates
                    raise ValueError(
                        f"Redundant distance connection specification for population {dependency.config.population}"
                        f"Found duplicate in {dependency}, while already "
                        f"defined in {self.synapse_forest[dependency.config.population]}"
                    )
                self.synapse_forest[dependency.config.population] = dependency

    def on_execute(self):
        print(f"Consolidating generated data files into unified H5")

        self.local_directory("data/simulation", create=True)

        MiV_populations = ["PYR", "OLM", "PVBC", "STIM"]
        MiV_IN_populations = ["OLM", "PVBC"]
        MiV_EXT_populations = ["STIM"]

        print("Import H5Types")
        with h5py.File(self.output_filepath("cells"), "w") as f:
            input_file = h5py.File(self.network.output_filepath, "r")
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
                with h5py.File(
                    self.soma_coordinates.output_filepath, "r"
                ) as f_src:
                    h5_copy_dataset(f_src, f_dst, coords_dset_path)
                    h5_copy_dataset(f_src, f_dst, distances_dset_path)

        print("Create forest entries and synapse attributes")

        def _run(commands):
            cmd = " ".join(commands)
            print(cmd)
            print(commandlib.Command(*commands).output())

        for p in MiV_populations:
            if p not in ["OLM", "PVBC", "PYR"]:
                continue
            forest_file = self.synapse_forest[p].output_filepath
            forest_syns_file = self.synapse_forest[p].output_filepath
            forest_dset_path = f"/Populations/{p}/Trees"
            forest_syns_dset_path = f"/Populations/{p}/Synapse Attributes"

            cmd = [
                "h5copy",
                "-p",
                "-s",
                forest_dset_path,
                "-d",
                forest_dset_path,
                "-i",
                forest_file,
                "-o",
                self.output_filepath(),
            ]
            _run(cmd)

            cmd = [
                "h5copy",
                "-p",
                "-s",
                forest_syns_dset_path,
                "-d",
                forest_syns_dset_path,
                "-i",
                forest_file,
                "-o",
                self.output_filepath(),
            ]
            _run(cmd)

        print("Create vector stimulus entries")
        vecstim_file_dict = {"A Diag": self.spike_trains.output_filepath}

        vecstim_dict = {
            f"Input Spikes {stim_id}": stim_file
            for stim_id, stim_file in vecstim_file_dict.items()
        }
        for vecstim_ns, vecstim_file in vecstim_dict.items():
            for p in MiV_EXT_populations:
                vecstim_dset_path = f"/Populations/{p}/{vecstim_ns}"
                cmd = [
                    "h5copy",
                    "-p",
                    "-s",
                    vecstim_dset_path,
                    "-d",
                    vecstim_dset_path,
                    "-i",
                    vecstim_file,
                    "-o",
                    self.output_filepath(),
                ]
                _run(cmd)

        print("Copy coordinates for STIM cells")
        cmd = [
            "h5copy",
            "-p",
            "-s",
            "/Populations/STIM/Generated Coordinates",
            "-d",
            "/Populations/STIM/Coordinates",
            "-i",
            self.output_filepath(),
            "-o",
            self.output_filepath(),
        ]
        _run(cmd)

        with h5py.File(self.output_filepath("connections"), "w") as f:
            input_file = h5py.File(self.network.output_filepath, "r")
            h5_copy_dataset(input_file, f, "/H5Types")
            input_file.close()

        print("Create connectivity entries")
        for p, e in self.distance_connections.items():
            connectivity_file = e.output_filepath
            print(connectivity_file)
            projection_dset_path = f"/Projections/{p}"
            cmd = [
                "h5copy",
                "-p",
                "-s",
                projection_dset_path,
                "-d",
                projection_dset_path,
                "-i",
                connectivity_file,
                "-o",
                self.output_filepath("connections"),
            ]
            _run(cmd)
