from machinable import Component
from neuroh5.io import read_population_names
from typing import Dict
from miv_simulator.utils.io import H5FileManager


class NeuroH5Graph(Component):
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
        self.network = None
        self.distances = None
        self.synapses = {}
        self.synapse_forest = {}
        self.connections = {}
        populations = []
        for u in self.uses:
            name = u.module.replace("miv_simulator.interface.", "")
            if name == "network_architecture":
                self.network = u
            elif name == "distances":
                self.distances = u
            elif name == "connections":
                for p in read_population_names(u.config.forest_filepath):
                    populations.append(p)
                    if p in self.connections:
                        raise ValueError(
                            f"Redundant distance connection specification for population {p}"
                            f"Found duplicate in {u.config.forest_filepath}, while already "
                            f"defined in {self.connections[p].config.forest_filepath}"
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

        self.graph.import_h5types(self.network.config.filepath)
        self.graph.import_soma_coordinates(
            self.network.config.filepath,
            populations=list(populations) + ["STIM"],
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
                "network": self.network.serialize(),
                "distances": self.distances.serialize(),
                "connections": {
                    k: v.serialize() for k, v in self.connections.items()
                },
                "synapse_forest": {
                    k: v.serialize() for k, v in self.synapse_forest.items()
                },
                "synapses": {
                    k: v.serialize() for k, v in self.synapses.items()
                },
            },
        )

    def on_compute_predicate(self):
        def generate_uid(use):
            if getattr(use, "refreshed_at", None) is not None:
                return f"{use.uuid}-{use.refreshed_at}"
            return use.uuid

        return {
            "uses": sorted(
                map(
                    generate_uid,
                    self.uses,
                )
            )
        }

    def files(self) -> Dict[str, str]:
        return {
            "cells": self.graph.cells_filepath,
            "connections": self.graph.connections_filepath,
        }
