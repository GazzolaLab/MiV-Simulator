from miv_simulator.clamps import network

from dataclasses import dataclass
from typing import Tuple, Optional, Union, List, Dict

from machinable import Interface
from machinable.config import Field
from miv_simulator.interface.config import BaseConfig


class ClampShow(Interface):
    @dataclass
    class Config(BaseConfig):
        population: str = "GC"
        gid: int = 0
        area_id: Optional[str] = None
        stimulus_id: Optional[str] = None
        templates: str = "templates"
        dataset_path: Optional[str] = None
        spike_events_path: Optional[str] = None
        spike_events_namespace: str = "Spike Events"
        spike_events_t: str = "t"
        input_features_path: Optional[str] = None
        input_features_namespaces: List[str] = Field(
            default_factory=lambda: ["Place Selectivity", "Grid Selectivity"]
        )
        use_coreneuron: bool = False
        plot_cell: bool = False
        write_cell: bool = False
        profile_memory: bool = False
        recording_profile: Optional[str] = None

    def on_execute(self):
        network.show(
            config_file=self.config.blueprint,
            config_prefix=None,
            population=self.config.population,
            gid=self.config.gid,
            arena_id=self.config.arena_id,
            stimulus_id=self.config.stimulus_id,
            template_paths=self.config.templates,
            dataset_prefix=self.config.dataset_path,
            results_path=self.local_directory("data/results", create=True),
            spike_events_path=self.config.spike_events_path,
            spike_events_namespace=self.config.spike_events_namespace,
            spike_events_t=self.config.spike_events_t,
            input_features_path=self.config.input_features_path,
            input_features_namespaces=self.config.input_features_namespaces,
            use_coreneuron=self.config.use_coreneuron,
            plot_cell=self.config.plot_cell,
            write_cell=self.config.write_cell,
            profile_memory=self.config.profile_memory,
            recording_profile=self.config.recording_profile,
        )
