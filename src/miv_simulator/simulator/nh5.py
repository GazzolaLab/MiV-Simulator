from typing import Tuple

import os
import pathlib

import subprocess
import h5py


def _run(commands):
    cmd = " ".join(commands)
    print(cmd)
    subprocess.check_output(commands)


def copy_dataset(f_src: h5py.File, f_dst: h5py.File, dset_path: str) -> None:
    print(f"Copying {dset_path} from {f_src} to {f_dst} ...")
    target_path = str(pathlib.Path(dset_path).parent)
    f_src.copy(f_src[dset_path], f_dst[target_path])


class Graph:
    """Utility to manage NeuroH5 graph data"""

    def __init__(self, directory: str):
        self.directory = directory

    def local_directory(self, *append: str, create: bool = False) -> str:
        d = os.path.join(os.path.abspath(self.directory), *append)
        if create:
            os.makedirs(d, exist_ok=True)
        return d

    @property
    def cells_filepath(self) -> str:
        return self.local_directory("cells.h5")

    @property
    def connections_filepath(self) -> str:
        return self.local_directory("connections.h5")

    def import_h5types(self, src: str):
        with h5py.File(self.cells_filepath, "w") as f:
            input_file = h5py.File(src, "r")
            copy_dataset(input_file, f, "/H5Types")
            input_file.close()

        with h5py.File(self.connections_filepath, "w") as f:
            input_file = h5py.File(src, "r")
            copy_dataset(input_file, f, "/H5Types")
            input_file.close()

    def import_soma_coordinates(self, src: str, populations: Tuple[str] = ()):
        with h5py.File(self.cells_filepath, "a") as f_dst:
            grp = f_dst.create_group("Populations")

            for p in populations:
                grp.create_group(p)

            for p in populations:
                coords_dset_path = f"/Populations/{p}/Generated Coordinates"
                coords_output_path = f"/Populations/{p}/Coordinates"
                distances_dset_path = f"/Populations/{p}/Arc Distances"
                with h5py.File(src, "r") as f_src:
                    copy_dataset(f_src, f_dst, coords_dset_path)
                    copy_dataset(f_src, f_dst, distances_dset_path)

    def import_synapse_attributes(
        self, population: str, forest_file: str, synapses_file: str
    ):
        forest_dset_path = f"/Populations/{population}/Trees"
        forest_syns_dset_path = f"/Populations/{population}/Synapse Attributes"

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
            self.cells_filepath,
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
            synapses_file,
            "-o",
            self.cells_filepath,
        ]
        _run(cmd)

    def import_projections(self, population: str, src: str):
        projection_dset_path = f"/Projections/{population}"
        cmd = [
            "h5copy",
            "-p",
            "-s",
            projection_dset_path,
            "-d",
            projection_dset_path,
            "-i",
            src,
            "-o",
            self.connections_filepath,
        ]
        _run(cmd)
