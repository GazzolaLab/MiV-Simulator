from typing import Optional, Tuple

import logging
from pydantic import BaseModel, ConfigDict, Field

from machinable import Component
from miv_simulator import config, simulator
from mpi4py import MPI


class MeasureDistances(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        filepath: str = Field("???")
        cell_distributions: config.CellDistributions = Field("???")
        layer_extents: config.LayerExtents = Field("???")
        rotation: config.Rotation = (0.0, 0.0, 0.0)
        geometry_filepath: Optional[str] = None
        coordinate_namespace: str = "Generated Coordinates"
        resolution: Tuple[int, int, int] = (30, 30, 10)
        populations: Optional[Tuple[str, ...]] = None
        origin: config.Origin = {"U": "median", "V": "median", "L": "max"}
        interp_chunk_size: int = 1000
        alpha_radius: float = 120.0
        n_sample: int = 1000
        io_size: int = -1
        chunk_size: int = 1000
        value_chunk_size: int = 1000
        cache_size: int = 50
        ranks: int = -1
        nodes: str = "1"

    def __call__(self):
        logging.basicConfig(level=logging.INFO)
        simulator.measure_distances(
            filepath=self.config.filepath,
            geometry_filepath=self.config.geometry_filepath,
            coordinate_namespace=self.config.coordinate_namespace,
            resolution=self.config.resolution,
            populations=self.config.populations,
            cell_distributions=self.config.cell_distributions,
            layer_extents=self.config.layer_extents,
            rotation=self.config.rotation,
            origin=config.Origin(**self.config.origin).as_spec(),
            n_sample=self.config.n_sample,
            alpha_radius=self.config.alpha_radius,
            io_size=self.config.io_size,
            chunk_size=self.config.chunk_size,
            value_chunk_size=self.config.value_chunk_size,
            cache_size=self.config.cache_size,
        )

    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0

    def compute_context(self):
        context = super().compute_context()
        del context["config"]["filepath"]
        del context["config"]["io_size"]
        del context["config"]["chunk_size"]
        del context["config"]["value_chunk_size"]
        del context["config"]["cache_size"]
        del context["config"]["ranks"]
        del context["config"]["nodes"]
        context["predicate"]["uses"] = sorted([u.hash for u in self.uses])
        return context
