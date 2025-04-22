from machinable import Component
from neuroh5.io import read_population_names
from typing import Dict
from miv_simulator.utils.io import H5FileManager


class NeuroH5Graph(Component):
    class Config:
        ranks: int = 1
        nodes: str = "1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = None

    @property
    def graph(self) -> None:
        if self._graph is None:
            self._graph = H5FileManager(self.local_directory())
        return self._graph

    def __call__(self) -> None:
        print("Merging H5 data")
        self.architecture = None
        self.distances = None
        self.synapses = {}
        self.synapse_forest = {}
        self.connections = {}
        populations = []
        for u in self.uses:
            name = u.module.replace("miv_simulator.interface.", "")
            if name == "architecture":
                self.architecture = u
            elif name == "distances":
                self.distances = u
            elif name == "connections":
                for p in read_population_names(u.config.forest_filepath):
                    populations.append(p)
                    if p in self.connections:
                        raise ValueError(
                            f"Redundant distance connection specification for population {p}. "
                            f"Found duplicate in {u.config.forest_filepath}, while already "
                            f"defined in {self.connections[p].config.forest_filepath} ({populations})"
                        )
                    self.connections[p] = u
            elif name == "synapse_forest":
                if u.config.population in self.synapse_forest:
                    raise ValueError(
                        f"Redundant distance connection specification for population {u.config.population}"
                        f"Found duplicate in {u}, while already "
                        f"defined in {self.synapse_forest[u.config.population]}"
                    )
                self.synapse_forest[u.config.population] = u
            elif name == "synapses":
                if u.config.population in self.synapses:
                    raise ValueError(
                        f"Redundant specification for population {u.config.population}"
                        f"Found duplicate in {u}, while already "
                        f"defined in {self.synapses[u.config.population]}"
                    )
                self.synapses[u.config.population] = u

        self.graph.import_h5types(self.architecture.config.filepath)
        self.graph.import_soma_coordinates(
            self.architecture.config.filepath,
            populations=list(populations),
        )
        for p in self.synapse_forest.keys():
            self.graph.import_synapse_attributes(
                p,
                self.synapse_forest[p].output_filepath,
                self.synapses[p].output_filepath,
            )
        self.graph.copy_stim_coordinates()

        for p, c in self.connections.items():
            self.graph.import_projections(p, c.output_filepath)

        # serialize sources
        self.save_file(
            "graph.json",
            {
                "architecture": self.architecture.serialize(),
                "distances": self.distances.serialize(),
                "connections": {k: v.serialize() for k, v in self.connections.items()},
                "synapse_forest": {
                    k: v.serialize() for k, v in self.synapse_forest.items()
                },
                "synapses": {k: v.serialize() for k, v in self.synapses.items()},
            },
        )

    def files(self) -> Dict[str, str]:
        return {
            "cells": self.graph.cells_filepath,
            "connections": self.graph.connections_filepath,
        }

    def compute_context(self):
        context = super().compute_context()
        del context["config"]["ranks"]
        del context["config"]["nodes"]
        context["predicate"]["uses"] = sorted([u.hash for u in self.uses])
        return context
