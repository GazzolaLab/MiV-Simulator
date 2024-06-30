import os

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict
from machinable import Interface, get
from miv_simulator.config import Config
from miv_simulator import mechanisms


def _lp(x1, x2, y1, y2, x) -> int:
    q = ((y2 - y1) * x + x2 * y1 - x1 * y2) / (x2 - x1)
    q = max(x1, q)
    q = min(x2, q)
    return int(q)


class Network(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        config_filepath: str = Field("???")
        mechanisms_path: str = ("./mechanisms",)
        template_path: str = ("./templates",)
        morphology_path: str = "./morphology"
        populations: Optional[list[str]] = None

    def launch(self):
        self.source_config = config = Config.from_yaml(
            self.config.config_filepath
        )

        populations = self.config.populations
        if populations is None:
            populations = list(config.synapses.keys())

        self.h5_types = get(
            "miv_simulator.interface.h5_types",
            [
                {
                    "projections": config.projections,
                    "cell_distributions": config.cell_distributions,
                    "population_definitions": config.definitions.populations,
                },
            ],
        )
        # eager execute to avoid scheduling overheads
        if not self.h5_types.cached():
            self.h5_types.commit().dispatch()

        self.architecture = get(
            "miv_simulator.interface.architecture",
            [
                {
                    "filepath": self.h5_types.output_filepath,
                    "cell_distributions": self.h5_types.config.cell_distributions,
                    "layer_extents": config.layer_extents,
                },
            ],
            uses=self.h5_types,
        ).launch()

        self.distances = self.architecture.measure_distances().launch()

        self.synapse_forest = {
            population: self.architecture.generate_synapse_forest(
                {
                    "population": population,
                    "morphology": os.path.join(
                        self.config.morphology_path, f"{population}.swc"
                    ),
                },
                uses=self.distances,
            ).launch()
            for population in [p for p in config.synapses if p in populations]
        }

        self.synapses = {
            population: self.architecture.distribute_synapses(
                {
                    "forest_filepath": self.synapse_forest[
                        population
                    ].output_filepath,
                    "cell_types": config.cell_types,
                    "population": population,
                    "layer_definitions": config.definitions.layers,
                    "distribution": "poisson",
                    "mechanisms_path": self.config.mechanisms_path,
                    "template_path": self.config.template_path,
                    # apply heuristic based on number of cells
                    "io_size": _lp(
                        0,
                        5e5,
                        1,
                        30,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "write_size": _lp(
                        0,
                        5e5,
                        1,
                        100,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "chunk_size": _lp(
                        0,
                        5e5,
                        1000,
                        10000,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "value_chunk_size": _lp(
                        0,
                        5e5,
                        1000,
                        200000,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "nodes": str(
                        _lp(
                            0,
                            5e5,
                            1,
                            25,
                            sum(config.cell_distributions[population].values()),
                        )
                    ),
                },
                uses=self.synapse_forest[population],
            ).launch()
            for population in self.synapse_forest
        }

        self.connections = {
            population: self.architecture.generate_connections(
                {
                    "synapses": config.synapses,
                    "forest_filepath": self.synapses[
                        population
                    ].output_filepath,
                    "axon_extents": config.axon_extents,
                    "population_definitions": config.definitions.populations,
                    "layer_definitions": config.definitions.layers,
                    "io_size": _lp(
                        0,
                        5e5,
                        1,
                        40,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "cache_size": _lp(
                        0,
                        5e5,
                        1,
                        20,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "write_size": _lp(
                        0,
                        5e5,
                        1,
                        250,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "chunk_size": _lp(
                        0,
                        5e5,
                        1000,
                        10000,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "value_chunk_size": _lp(
                        0,
                        5e5,
                        1000,
                        640000,
                        sum(config.cell_distributions[population].values()),
                    ),
                    "nodes": str(
                        _lp(
                            0,
                            5e5,
                            1,
                            64,
                            sum(config.cell_distributions[population].values()),
                        )
                    ),
                },
                uses=self.synapses[population],
            ).launch()
            for population in self.synapses
        }

        self.neural_h5 = get(
            "miv_simulator.interface.neuroh5_graph",
            uses=[
                self.architecture,
                self.distances,
                *self.synapse_forest.values(),
                *self.synapses.values(),
                *self.connections.values(),
            ],
        ).launch()

        return self

    def version_from_config(self, config_filepath: str):
        source = os.path.dirname(os.path.dirname(config_filepath))
        return {
            "config_filepath": f"{config_filepath}",
            "mechanisms_path": mechanisms.compile(f"{source}/mechanisms"),
            "template_path": f"{source}/templates",
            "morphology_path": f"{source}/morphology",
        }

    def compute_context(self):
        context = super().compute_context()
        del context["config"]
        return context
