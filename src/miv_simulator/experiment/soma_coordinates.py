import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from machinable import Experiment
from machinable.config import Field
from machinable.element import normversion
from machinable.types import VersionType
from miv_simulator import plotting
from miv_simulator.simulator import generate_soma_coordinates

from miv_simulator.experiment.config import FromYAMLConfig, HandlesYAMLConfig


class SomaCoordinates(HandlesYAMLConfig, Experiment):
    """Generate soma coordinates within layer-specific volume."""

    @dataclass
    class Config(FromYAMLConfig):
        h5types: str = Field("???")
        geometry: Optional[str] = None
        output_namespace: str = "Generated Coordinates"
        populations: Tuple[str, ...] = ()
        resolution: Tuple[int, int, int] = (3, 3, 3)
        alpha_radius: float = 2500.0
        nodeiter: int = 10
        dispersion_delta: float = 0.1
        snap_delta: float = 0.01
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000

    @property
    def output_filepath(self):
        return self.local_directory("data/coordinates.h5")

    def on_execute(self):
        logging.basicConfig(level=logging.INFO)
        self.local_directory("data", create=True)
        generate_soma_coordinates(
            config=self.config.network,
            types_path=self.config.h5types,
            output_path=self.output_filepath,
            geometry_path=self.config.geometry,
            output_namespace=self.config.output_namespace,
            populations=self.config.populations,
            resolution=self.config.resolution,
            alpha_radius=self.config.alpha_radius,
            nodeiter=self.config.nodeiter,
            dispersion_delta=self.config.dispersion_delta,
            snap_delta=self.config.snap_delta,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            verbose=False,
        )

    def plot_in_volume(
        self,
        populations: Tuple[str, ...],
        scale: float = 25.0,
        subpopulation: int = -1,
        subvol: bool = False,
        mayavi: bool = False,
    ):
        plotting.plot_coords_in_volume(
            populations=populations,
            coords_path=self.output_filepath,
            coords_namespace=self.config.output_namespace,
            config=self.config.network,
            scale=scale,
            subpopulation=subpopulation,
            subvol=subvol,
            mayavi=mayavi,
        )

    def measure_distances(self, version: VersionType = None):
        return self.derive_singleton(
            "miv_simulator.experiment.measure_distances",
            [
                {
                    "network": self.config.network,
                    "coordinates": self.output_filepath,
                    "output_namespace": self.config.output_namespace,
                }
            ]
            + normversion(version),
        )

    def distribute_synapses(self, version: VersionType = None):
        return self.derive_singleton(
            "miv_simulator.experiment.synapse_locations",
            [
                {
                    "network": self.config.network,
                    "coordinates": self.output_filepath,
                }
            ]
            + normversion(version),
        )

    def distance_connections(self, version: VersionType = None):
        return self.derive_singleton(
            "miv_simulator.experiment.distance_connections",
            [
                {
                    "network": self.config.network,
                    "coordinates": self.output_filepath,
                }
            ]
            + normversion(version),
        )

    def input_features(self, version: VersionType = None):
        return self.derive_singleton(
            "miv_simulator.experiment.input_features",
            [
                {
                    "network": self.config.network,
                    "coordinates": self.output_filepath,
                }
            ]
            + normversion(version),
        )
