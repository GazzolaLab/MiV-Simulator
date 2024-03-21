import os

from pydantic import BaseModel, Field, ConfigDict
from machinable import Interface, get
from miv_simulator.config import Config


class Network(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        config_filepath: str = Field("???")
        mechanisms_path: str = ("./mechanisms",)
        template_path: str = ("./templates",)
        morphology_path: str = "./morphology"

    def launch(self):
        config = Config.from_yaml(self.config.config_filepath)

        self.h5_types = get(
            "miv_simulator.interface.h5_types",
            [
                {
                    "projections": config.projections,
                    "cell_distributions": config.cell_distributions,
                },
            ],
        ).launch()

        self.network = get(
            "miv_simulator.interface.network_architecture",
            [
                {
                    "filepath": self.h5_types.output_filepath,
                    "cell_distributions": self.h5_types.config.cell_distributions,
                    "layer_extents": config.layer_extents,
                },
            ],
            uses=self.h5_types,
        ).launch()

        self.distances = self.network.measure_distances().launch()

        self.synapse_forest = {
            population: self.network.generate_synapse_forest(
                {
                    "population": population,
                    "morphology": os.path.join(
                        self.config.morphology_path, f"{population}.swc"
                    ),
                },
                uses=self.distances,
            ).launch()
            for population in config.synapses
        }

        self.synapses = {
            population: self.network.distribute_synapses(
                {
                    "forest_filepath": self.synapse_forest[
                        population
                    ].output_filepath,
                    "cell_types": config.cell_types,
                    "population": population,
                    "distribution": "poisson",
                    "mechanisms_path": self.config.mechanisms_path,
                    "template_path": self.config.template_path,
                    "io_size": 1,
                    "write_size": 0,
                },
                uses=self.synapse_forest[population],
            ).launch()
            for population in config.synapses
        }

        self.connections = {
            population: self.network.generate_connections(
                {
                    "synapses": config.synapses,
                    "forest_filepath": self.synapses[
                        population
                    ].output_filepath,
                    "axon_extents": config.axon_extents,
                    "io_size": 1,
                    "cache_size": 20,
                    "write_size": 100,
                },
                uses=self.synapses[population],
            ).launch()
            for population in config.synapses
        }

        self.neural_h5 = get(
            "miv_simulator.interface.neuroh5_graph",
            uses=[
                self.network,
                self.distances,
                *self.synapse_forest.values(),
                *self.synapses.values(),
                *self.connections.values(),
            ],
        ).launch()

        return self

    def compute_context(self):
        context = super().compute_context()
        del context["config"]
        return context
