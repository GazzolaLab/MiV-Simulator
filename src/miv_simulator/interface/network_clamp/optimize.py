from dataclasses import dataclass
from typing import Tuple, Optional, Union, List, Dict

from machinable import Experiment
from machinable.config import Field

from miv_simulator.clamps import network
from miv_simulator.config import Blueprint


class ClampOptimize(Experiment):
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
        nprocs_per_worker_: int = 1
        opt_epsilon: float = 1e-2
        opt_seed: Optional[int] = None
        opt_iter: int = 10
        templates: str = "templates"
        dataset_path: Optional[str] = None
        param_config_name: Optional[str] = None
        param_type: str = "synaptic"
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
        trial_regime: str = ("mean",)
        problem_regime: str = "every"
        target_features_path: Optional[str] = None
        target_features_namespace: str = "Input Spikes"
        target_state_variable: Optional[str] = None
        target_state_filter: Optional[str] = None
        use_coreneuron: bool = False
        cooperative_init: bool = False
        target: str = "rate"

    def on_execute(self):
        network.optimize(
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
            nprocs_per_worker=self.config.nprocs_per_worker_,
            opt_epsilon=self.config.opt_epsilon,
            opt_seed=self.config.opt_seed,
            opt_iter=self.config.opt_iter,
            template_paths=self.config.templates,
            dataset_prefix=self.config.dataset_path,
            param_config_name=self.config.param_config_name,
            param_type=self.config.param_type,
            recording_profile=self.config.recording_profile,
            results_file=None,
            results_path=self.local_directory("data/results", create=True),
            spike_events_path=self.config.spike_events_path,
            spike_events_namespace=self.config.spike_events_namespace,
            spike_events_t=self.config.spike_events_t,
            coords_path=self.config.coordinates,
            distances_namespace=self.config.distances_namespace,
            phase_mod=self.config.phase_mod,
            input_features_path=self.config.input_features_path,
            input_features_namespaces=self.config.input_features_namespaces,
            n_trials=self.config.n_trials,
            trial_regime=self.config.trial_regime,
            problem_regime=self.config.problem_regime,
            target_features_path=self.config.target_features_path,
            target_features_namespace=self.config.target_features_namespace,
            target_state_variable=self.config.target_state_variable,
            target_state_filter=self.config.target_state_filter,
            use_coreneuron=self.config.use_coreneuron,
            cooperative_init=self.config.cooperative_init,
            target=self.config.target,
        )
