from typing import Optional, Tuple, Dict
import os
import logging

from machinable import Component
from machinable.element import normversion
from machinable.types import VersionType
from miv_simulator import config
from miv_simulator.simulator.soma_coordinates import (
    generate as generate_soma_coordinates,
)
from miv_simulator.utils import io as io_utils, from_yaml
from mpi4py import MPI
from pydantic import BaseModel, Field


class CreateNetwork(Component):
    """Creates neural H5 type definitions and soma coordinates within specified layer geometry."""

    class Config(BaseModel):
        cell_distributions: config.CellDistributions = Field("???")
        synapses: config.Synapses = Field("???")
        layer_extents: config.LayerExtents = Field("???")
        rotation: config.Rotation = (0.0, 0.0, 0.0)
        cell_constraints: config.CellConstraints = {}
        populations: Optional[Tuple[str, ...]] = None
        geometry_filepath: Optional[str] = None
        coordinate_namespace: str = "Generated Coordinates"
        resolution: Tuple[int, int, int] = (3, 3, 3)
        alpha_radius: float = 2500.0
        nodeiter: int = 10
        dispersion_delta: float = 0.1
        snap_delta: float = 0.01
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        ranks_: int = 1

    def config_from_file(self, filename: str) -> Dict:
        return from_yaml(filename)

    @property
    def output_filepath(self) -> str:
        return self.local_directory("network.h5")

    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0

    def __call__(self) -> None:
        logging.basicConfig(level=logging.INFO)

        if MPI.COMM_WORLD.rank == 0:
            io_utils.create_neural_h5(
                self.output_filepath,
                self.config.cell_distributions,
                self.config.synapses,
            )
        MPI.COMM_WORLD.barrier()

        generate_soma_coordinates(
            output_filepath=self.output_filepath,
            cell_distributions=self.config.cell_distributions,
            layer_extents=self.config.layer_extents,
            rotation=self.config.rotation,
            cell_constraints=self.config.cell_constraints,
            output_namespace=self.config.coordinate_namespace,
            geometry_filepath=self.config.geometry_filepath,
            populations=self.config.populations,
            resolution=self.config.resolution,
            alpha_radius=self.config.alpha_radius,
            nodeiter=self.config.nodeiter,
            dispersion_delta=self.config.dispersion_delta,
            snap_delta=self.config.snap_delta,
            h5_types_filepath=None,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
        )

    def measure_distances(self, version: VersionType = None):
        return self.derive(
            "miv_simulator.interface.measure_distances",
            [
                {
                    "filepath": self.output_filepath,
                    "cell_distributions": self.config.cell_distributions,
                    "layer_extents": self.config.layer_extents,
                    "rotation": self.config.rotation,
                    "geometry_filepath": self.config.geometry_filepath,
                    "coordinate_namespace": self.config.coordinate_namespace,
                    "resolution": self.config.resolution,
                    "alpha_radius": self.config.alpha_radius,
                    "io_size": self.config.io_size,
                    "chunk_size": self.config.chunk_size,
                    "value_chunk_size": self.config.value_chunk_size,
                }
            ]
            + normversion(version),
            uses=self,
        )

    def synapse_forest(self, version: VersionType = None) -> "Component":
        return self.derive(
            "miv_simulator.interface.synapse_forest",
            [
                {
                    "filepath": self.output_filepath,
                }
            ]
            + normversion(version),
            uses=self,
        )

    def distribute_synapses(self, version: VersionType = None):
        return self.derive(
            "miv_simulator.interface.distribute_synapses",
            [] + normversion(version),
            uses=self,
        )

    def distance_connections(self, version: VersionType = None):
        return self.derive(
            "miv_simulator.interface.distance_connections",
            [
                {
                    "blueprint": self.config.blueprint,
                    "coordinates": self.output_filepath,
                }
            ]
            + normversion(version),
            uses=self,
        )

    def input_features(self, version: VersionType = None):
        return self.derive(
            "miv_simulator.interface.input_features",
            [
                {
                    "blueprint": self.config.blueprint,
                    "coordinates": self.output_filepath,
                }
            ]
            + normversion(version),
            uses=self,
        )
