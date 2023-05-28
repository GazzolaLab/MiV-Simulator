from typing import Dict, List, Optional, Tuple, Union

from dataclasses import dataclass

from machinable import Component
from machinable.config import Field
from miv_simulator.clamps import network
from miv_simulator.config import Blueprint


class ClampGo(Component):
    @dataclass
    class Config:
        blueprint: Blueprint = Field(default_factory=Blueprint)
        population: str = "PYR"
        dt: Optional[float] = None
        gids: List[int] = Field(default_factory=[])
        gid_selection_file: Optional[str] = None
        area_id: Optional[str] = None
        stimulus_id: Optional[str] = None
        generate_weights: set[str] = set()
        t_max: Optional[float] = 150.0
        t_min: Optional[float] = None
        templates: str = "templates"
        dataset_path: Optional[str] = None
        spike_events_path: Optional[str] = None
        spike_events_namespace: str = "Spike Events"
        spike_events_t: str = "t"
        coordinates: str = Field("???")
        distances_namespace: str = ("Arc Distances",)
        phase_mod: bool = (False,)
        input_features_path: Optional[str] = None
        input_features_namespaces: List[str] = Field(
            default_factory=lambda: ["Place Selectivity", "Grid Selectivity"]
        )
        n_trials: int = 1
        params_path: List[str] = Field(default_factory=[])
        params_id: List[int] = Field(default_factory=[])
        results_namespace_id: Optional[str] = None
        use_coreneuron: bool = False
        plot_cell: bool = False
        write_cell: bool = False
        profile_memory: bool = False
        recording_profile: Optional[str] = None
        input_seed: Optional[int] = None

    def __call__(self):
        network.go(
            config_file=self.config.blueprint,
            config_prefix=None,
            population=self.config.population,
            dt=self.config.dt,
            gids=self.config.gids,
            gid_selection_file=self.config.gid_selection_file,
            arena_id=self.config.arena_id,
            stimulus_id=self.config.stimulus_id,
            generate_weights=self.config.generate_weights,
            t_max=self.config.t_max,
            t_min=self.config.t_min,
            template_paths=self.config.templates,
            dataset_prefix=self.config.dataset_path,
            spike_events_path=self.config.spike_events_path,
            spike_events_namespace=self.config.spike_events_namespace,
            spike_events_t=self.config.spike_events_t,
            coords_path=self.config.coordinates,
            distances_namespace=self.config.distances_namespace,
            phase_mod=self.config.phase_mod,
            input_features_path=self.config.input_features_path,
            input_features_namespaces=self.config.input_features_namespaces,
            n_trials=self.config.n_trials,
            params_path=self.config.params_path,
            params_id=self.config.params_id,
            results_path=self.local_directory("data/results"),
            results_file_id=self.config.results_namespace_id,
            results_namespace_id=self.config.results_namespace_id,
            use_coreneuron=self.config.use_coreneuron,
            plot_cell=self.config.plot_cell,
            write_cell=self.config.write_cell,
            profile_memory=self.config.profile_memory,
            recording_profile=self.config.recording_profile,
            input_seed=self.config.input_seed,
        )
