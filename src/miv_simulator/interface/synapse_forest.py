from machinable import Component
from miv_simulator import config, simulator
from pydantic import BaseModel, Field, ConfigDict
from machinable.utils import file_hash

class GenerateSynapseForest(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        filepath: str = Field("???")
        population: config.PopulationName = Field("???")
        morphology: config.SWCFilePath = Field("???")

    @property
    def tree_output_filepath(self) -> str:
        return self.local_directory("dentric_tree.h5")

    @property
    def output_filepath(self) -> str:
        return self.local_directory("forest.h5")

    def __call__(self) -> None:
        simulator.generate_synapse_forest(
            filepath=self.config.filepath,
            tree_output_filepath=self.tree_output_filepath,
            output_filepath=self.output_filepath,
            population=self.config.population,
            morphology=self.config.morphology,
        )
        print("generate_synapse_forest() completed")

    def compute_context(self):
        # remove filepath in favor of uses
        context = super().compute_context()
        del context['config']['filepath']
        context['config']['morphology'] = file_hash(context['config']['morphology'])
        context['predicate']['uses'] = sorted([u.hash for u in self.uses])
        return context
