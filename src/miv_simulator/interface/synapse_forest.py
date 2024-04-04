from machinable import Component
from miv_simulator import config, simulator
from pydantic import BaseModel, Field, ConfigDict
from machinable.utils import file_hash
from typing import Optional


class GenerateSynapseForest(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        filepath: str = Field("???")
        population: config.PopulationName = Field("???")
        morphology: config.SWCFilePath = Field("???")
        mpi_args: Optional[str] = "-n 1"
        nodes: str = "1"

    @property
    def tree_output_filepath(self) -> str:
        return self.local_directory("dentric_tree.h5")

    @property
    def output_filepath(self) -> str:
        return self.local_directory("forest.h5")

    def __call__(self, runner=None) -> None:
        kwargs = {}
        if runner is not None:
            kwargs["_run"] = runner
        simulator.generate_synapse_forest(
            filepath=self.config.filepath,
            tree_output_filepath=self.tree_output_filepath,
            output_filepath=self.output_filepath,
            population=self.config.population,
            morphology=self.config.morphology,
            **kwargs,
        )

    def dispatch_code(
        self,
        inline: bool = True,
        project_directory: Optional[str] = None,
        python: Optional[str] = None,
    ) -> Optional[str]:
        # since generate_synapse_forest just subprocesses
        # the calls, we can return the code directly to
        # avoid overheads

        cmds = []
        self.__call__(cmds.append)
        lines = [" ".join(cmd) for cmd in cmds]

        if isinstance(python, str):
            runner, _, python = python.rpartition(" ")
            if runner:
                lines = [runner + " " + line for line in lines]

        cache_marker = self.local_directory("cached")
        lines.append(f'echo "finished" > {cache_marker}')

        return "\n".join(lines)

    def compute_context(self):
        context = super().compute_context()
        del context["config"]["mpi_args"]
        del context["config"]["nodes"]
        del context["config"]["filepath"]
        context["config"]["morphology"] = file_hash(
            context["config"]["morphology"]
        )
        context["predicate"]["uses"] = sorted([u.hash for u in self.uses])
        return context
