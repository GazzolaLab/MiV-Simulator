#!/usr/bin/env python
"""
Network optimization objectives and constraints framework.
"""

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Union, Set

import numpy as np


class NetworkFeature(ABC):
    """Abstract base class for network features."""

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Unnamespaced feature names. Framework applies "{pop_name}.{name}" namespacing."""
        ...

    @property
    def populations(self) -> Optional[List[str]]:
        """None = apply to all target_populations."""
        return None

    @abstractmethod
    def compute(
        self,
        pop_name: str,
        pop_features_dict: Dict,
    ) -> Dict[str, float]:
        """Returns {feature_name: scalar_value}."""
        ...


def merge_raw_features(dicts: List[Dict]) -> Dict:
    result: Dict = {}
    for d in dicts:
        for key, value in d.items():
            if key == "time_bins":
                if result.get(key) is None:
                    result[key] = value
            elif isinstance(value, dict):
                result.setdefault(key, {}).update(value)
            elif isinstance(value, np.ndarray):
                result[key] = value if key not in result else result[key] + value
            elif isinstance(value, (int, float, np.integer, np.floating)):
                result[key] = result.get(key, 0) + value
            elif key not in result:
                result[key] = value
    return result


class NetworkObjective(ABC):
    """Abstract base class for network objectives."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name used as dmosopt objective key."""
        ...

    @property
    @abstractmethod
    def required_features(self) -> List[str]:
        """
        Namespaced feature keys, e.g. ["CA3.mean_rate", "DG.rate_cv"].
        Can span multiple populations.
        """
        ...

    @abstractmethod
    def compute(self, feature_values: Dict[str, float]) -> float:
        """
        Receives the full all_features_dict (all populations, all features).
        Returns score (higher = better). Framework negates before dmosopt.
        """
        ...


class NetworkConstraint(ABC):
    """Abstract base class for network constraints."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def required_features(self) -> List[str]: ...

    @abstractmethod
    def compute(self, feature_values: Dict[str, float]) -> float:
        """c_i > 0 -> feasible (dmosopt convention: np.all(C > 0.0, axis=1))."""
        ...


class FeatureRegistry:
    """Registry for NetworkFeature classes."""

    _registry: Dict[str, NetworkFeature] = {}

    @classmethod
    def register(cls, feature: NetworkFeature) -> None:
        for name in feature.feature_names:
            if name in cls._registry:
                raise ValueError(f"Feature {name} already registered")
            cls._registry[name] = feature

    @classmethod
    def get(cls, name: str) -> NetworkFeature:
        return cls._registry[name]

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


@dataclass
class NetworkOptimizationConfig:
    """Configuration for network optimization."""

    features: List[NetworkFeature] = field(default_factory=list)
    objectives: List[NetworkObjective] = field(default_factory=list)
    constraints: List[NetworkConstraint] = field(default_factory=list)

    def objective_names(self) -> List[str]:
        return [obj.name for obj in self.objectives]

    def constraint_names(self) -> List[str]:
        return [c.name for c in self.constraints]

    def feature_dtypes(
        self, target_populations: Optional[List[str]] = None
    ) -> List[tuple]:
        """
        Returns [(namespaced_feature_name, np.float32), ...] for all features
        produced by every NetworkFeature object across all target_populations.

        If target_populations is None, falls back to the populations derived
        from objectives/constraints via target_populations().
        """
        pops = (
            target_populations
            if target_populations is not None
            else self.target_populations()
        )
        dtypes: List[tuple] = []
        for pop in pops:
            for feature in self.features:
                if feature.populations is not None and pop not in feature.populations:
                    continue
                for fname in feature.feature_names:
                    dtypes.append((f"{pop}.{fname}", np.float32))
        return dtypes

    def target_populations(self) -> List[str]:
        """Derive unique population names from objectives and constraints."""
        pops: Set[str] = set()
        for obj in self.objectives:
            pops.update(self._extract_populations(obj))
        for c in self.constraints:
            pops.update(self._extract_populations(c))
        return sorted(pops)

    @staticmethod
    def _extract_populations(
        item: Union[NetworkObjective, NetworkConstraint],
    ) -> List[str]:
        """Extract population names from an objective or constraint instance."""
        pops: List[str] = []

        # Direct population attributes on built-ins
        if hasattr(item, "_pop_name") and item._pop_name is not None:
            pops.append(item._pop_name)
        if hasattr(item, "_pop_names") and item._pop_names is not None:
            pops.extend(item._pop_names)
        if hasattr(item, "_num_pop") and item._num_pop is not None:
            pops.append(item._num_pop)
        if hasattr(item, "_denom_pop") and item._denom_pop is not None:
            pops.append(item._denom_pop)

        # Fallback: parse required_features namespace prefixes
        if not pops:
            for feat in item.required_features:
                if "." in feat:
                    # Namespaced like "CA3.mean_rate" -> "CA3"
                    pop = feat.split(".")[0]
                    pops.append(pop)

        return pops

    def validate_picklable(self) -> None:
        """Validate that this config is picklable."""
        try:
            pickle.dumps(self)
        except (pickle.PickleError, TypeError) as e:
            raise TypeError(f"NetworkOptimizationConfig is not picklable: {e}") from e


class MeanFiringRateFeature(NetworkFeature):
    """Mean firing rate feature."""

    @property
    def feature_names(self) -> List[str]:
        return ["mean_rate"]

    def compute(
        self,
        pop_name: str,
        pop_features_dict: Dict,
    ) -> Dict[str, float]:
        spike_density_dict = pop_features_dict["spike_density_dict"]
        if not spike_density_dict:
            return {"mean_rate": 0.0}

        total_rate = 0.0
        n_active = 0
        for gid, dens_dict in spike_density_dict.items():
            mean_rate = float(np.mean(dens_dict["rate"]))
            if mean_rate > 0.0:
                total_rate += mean_rate
                n_active += 1

        if n_active > 0:
            mean_rate = total_rate / n_active
        else:
            mean_rate = 0.0

        return {"mean_rate": mean_rate}


class PopulationFiringRateFeature(NetworkFeature):
    """Mean firing rate over every cell of the population, silent ones included.

    ``MeanFiringRateFeature`` averages over active cells only, so a population in
    which a tenth of the cells fire fast reads as a fast population. This
    feature divides the same per-cell rates by the population's cell count
    instead, so a largely silent population reads as slow.
    """

    @property
    def feature_names(self) -> List[str]:
        return ["population_rate"]

    def compute(
        self,
        pop_name: str,
        pop_features_dict: Dict,
    ) -> Dict[str, float]:
        n_total = pop_features_dict["n_total"]
        if n_total <= 0:
            return {"population_rate": 0.0}
        spike_density_dict = pop_features_dict["spike_density_dict"]
        total_rate = sum(
            float(np.mean(dens_dict["rate"]))
            for dens_dict in spike_density_dict.values()
        )
        return {"population_rate": total_rate / float(n_total)}


class FractionActiveFeature(NetworkFeature):
    """Fraction of active neurons feature."""

    @property
    def feature_names(self) -> List[str]:
        return ["fraction_active"]

    def compute(
        self,
        pop_name: str,
        pop_features_dict: Dict,
    ) -> Dict[str, float]:
        n_total = pop_features_dict["n_total"]
        n_active = pop_features_dict["n_active"]

        if n_total > 0:
            fraction_active = n_active / n_total
        else:
            fraction_active = 0.0

        return {"fraction_active": fraction_active}


class FiringRateStabilityFeature(NetworkFeature):
    """Firing rate stability feature with multiple metrics."""

    def __init__(
        self,
        active_threshold: Union[float, Dict[str, float]] = 0.01,
        temporal_resolution: float = 2.0,
    ):
        self.active_threshold = active_threshold
        self.temporal_resolution = temporal_resolution

    def threshold(self, pop_name: str) -> float:
        if isinstance(self.active_threshold, dict):
            return float(self.active_threshold.get(pop_name, 0.01))
        return float(self.active_threshold)

    @property
    def feature_names(self) -> List[str]:
        return [
            "mean_fraction_active_per_bin",
            "std_fraction_active_per_bin",
            "rate_cv",
        ]

    def compute(
        self,
        pop_name: str,
        pop_features_dict: Dict,
    ) -> Dict[str, float]:
        time_bins = pop_features_dict["time_bins"]
        spike_density_dict = pop_features_dict["spike_density_dict"]
        n_total = pop_features_dict["n_total"]

        t_start = time_bins[0]
        t_end = time_bins[-1] + (time_bins[1] - time_bins[0])
        fr_time_bins = np.arange(t_start, t_end, self.temporal_resolution)
        fr_time_centers = (fr_time_bins + self.temporal_resolution / 2).astype(
            np.float32
        )
        sum_active_per_bin = np.zeros_like(fr_time_centers, dtype=np.float32)

        for gid, dens_dict in spike_density_dict.items():
            ip_rate = np.interp(
                fr_time_centers,
                time_bins,
                dens_dict["rate"].astype(np.float32),
            ).astype(np.float32)
            active_per_bin = ip_rate > self.threshold(pop_name)
            sum_active_per_bin += active_per_bin

        if n_total > 0:
            mean_fraction_active_per_bin = float(
                np.mean(sum_active_per_bin / float(n_total))
            )
            std_fraction_active_per_bin = float(
                np.std(sum_active_per_bin / float(n_total))
            )
        else:
            mean_fraction_active_per_bin = 0.0
            std_fraction_active_per_bin = 0.0

        if mean_fraction_active_per_bin > 0:
            rate_cv = std_fraction_active_per_bin / mean_fraction_active_per_bin
        else:
            rate_cv = 0.0

        return {
            "mean_fraction_active_per_bin": mean_fraction_active_per_bin,
            "std_fraction_active_per_bin": std_fraction_active_per_bin,
            "rate_cv": rate_cv,
        }


class PopulationSynchronyFeature(NetworkFeature):
    """Population synchrony feature using pairwise cross-correlation."""

    def __init__(self, max_pairs: int = 200):
        self.max_pairs = max_pairs

    @property
    def feature_names(self) -> List[str]:
        return ["pairwise_synchrony"]

    def compute(
        self,
        pop_name: str,
        pop_features_dict: Dict,
    ) -> Dict[str, float]:
        spike_density_dict = pop_features_dict["spike_density_dict"]
        if len(spike_density_dict) < 2:
            return {"pairwise_synchrony": 0.0}

        gids = list(spike_density_dict.keys())
        n_pairs = min(self.max_pairs, len(gids) * (len(gids) - 1) // 2)

        if n_pairs == 0:
            return {"pairwise_synchrony": 0.0}

        np.random.seed(42)
        selected_pairs = []
        gids_set = set(gids)
        while len(selected_pairs) < n_pairs and len(gids_set) >= 2:
            gid1 = np.random.choice(list(gids_set))
            remaining = list(gids_set - {gid1})
            if not remaining:
                break
            gid2 = np.random.choice(remaining)
            selected_pairs.append((gid1, gid2))
            if len(gids_set) > 2:
                gids_set -= {gid1, gid2}

        correlations = []
        for gid1, gid2 in selected_pairs:
            rate1 = spike_density_dict[gid1]["rate"]
            rate2 = spike_density_dict[gid2]["rate"]
            if np.std(rate1) > 0 and np.std(rate2) > 0:
                corr = np.corrcoef(rate1, rate2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            pairwise_synchrony = float(np.mean(correlations))
        else:
            pairwise_synchrony = 0.0

        return {"pairwise_synchrony": pairwise_synchrony}


class ParticipationRatioFeature(NetworkFeature):
    """How much of a population fires together."""

    def __init__(self, bin_size: float = 50.0):
        self.bin_size = float(bin_size)

    @property
    def feature_names(self) -> List[str]:
        return ["peak_active_fraction", "participation_ratio"]

    def compute(self, pop_name: str, pop_features_dict: Dict) -> Dict[str, float]:
        n_total = pop_features_dict.get("n_total", 0)
        spike_times = pop_features_dict.get("spike_times") or {}
        time_bins = pop_features_dict.get("time_bins")
        if n_total <= 0 or time_bins is None or len(time_bins) == 0:
            return {"peak_active_fraction": 0.0, "participation_ratio": 0.0}
        t_start = float(time_bins[0])
        t_stop = float(time_bins[-1]) + (
            float(time_bins[1] - time_bins[0]) if len(time_bins) > 1 else 0.0
        )
        n_bins = max(1, int(round((t_stop - t_start) / self.bin_size)))
        active = np.zeros(n_bins, dtype=np.int64)
        for ts in spike_times.values():
            idx = np.floor((np.asarray(ts, dtype=float) - t_start) / self.bin_size)
            idx = np.unique(idx[(idx >= 0) & (idx < n_bins)]).astype(np.int64)
            active[idx] += 1
        fraction = active / float(n_total)
        mean = float(fraction.mean())
        peak = float(fraction.max())
        return {
            "peak_active_fraction": peak,
            "participation_ratio": peak / mean if mean > 0 else 0.0,
        }


class TargetRateObjective(NetworkObjective):
    """Objective to match a target firing rate for a population."""

    def __init__(self, pop_name: str, target_rate: float, name: Optional[str] = None):
        self._pop_name = pop_name
        self._target_rate = target_rate
        self._name = name or f"{pop_name}_target_rate"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.mean_rate"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        rate = feature_values.get(f"{self._pop_name}.mean_rate", 0.0)
        return -((rate - self._target_rate) ** 2)


class TargetLogRateObjective(NetworkObjective):
    """Objective to match a target firing rate on a logarithmic scale."""

    def __init__(
        self,
        pop_name: str,
        target_rate: float,
        feature: str = "population_rate",
        eps: float = 1e-3,
        name: Optional[str] = None,
    ):
        self._pop_name = pop_name
        self._target_rate = target_rate
        self._feature = feature
        self._eps = eps
        self._name = name or f"{pop_name}_target_log_rate"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.{self._feature}"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        rate = feature_values.get(f"{self._pop_name}.{self._feature}", 0.0)
        ratio = (max(rate, 0.0) + self._eps) / (self._target_rate + self._eps)
        return -(float(np.log(ratio)) ** 2)


class TargetFractionActiveObjective(NetworkObjective):
    """Objective to match a target fraction of active neurons."""

    def __init__(
        self, pop_name: str, target_fraction: float, name: Optional[str] = None
    ):
        self._pop_name = pop_name
        self._target_fraction = target_fraction
        self._name = name or f"{pop_name}_target_fraction_active"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.fraction_active"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        frac = feature_values.get(f"{self._pop_name}.fraction_active", 0.0)
        return -((frac - self._target_fraction) ** 2)


class TargetMeanFractionActiveObjective(NetworkObjective):
    """Objective to match a target mean fraction active per time bin."""

    def __init__(self, pop_name: str, target_value: float, name: Optional[str] = None):
        self._pop_name = pop_name
        self._target_value = target_value
        self._name = name or f"{pop_name} mean fraction active per time bin"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.mean_fraction_active_per_bin"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        val = feature_values.get(f"{self._pop_name}.mean_fraction_active_per_bin", 0.0)
        return -((val - self._target_value) ** 2)


class TargetStdFractionActiveObjective(NetworkObjective):
    """Objective to match a target std fraction active per time bin."""

    def __init__(self, pop_name: str, target_value: float, name: Optional[str] = None):
        self._pop_name = pop_name
        self._target_value = target_value
        self._name = name or f"{pop_name} std fraction active per time bin"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.std_fraction_active_per_bin"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        val = feature_values.get(f"{self._pop_name}.std_fraction_active_per_bin", 0.0)
        return -((val - self._target_value) ** 2)


class SteadyFiringObjective(NetworkObjective):
    """Objective to encourage steady firing (low CV)."""

    def __init__(self, pop_name: str, name: Optional[str] = None):
        self._pop_name = pop_name
        self._name = name or f"{pop_name}_steady_firing"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.rate_cv"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        rate_cv = feature_values.get(f"{self._pop_name}.rate_cv", 0.0)
        return -rate_cv


class MultiPopSteadyFiringObjective(NetworkObjective):
    """Objective for steady firing across multiple populations."""

    def __init__(self, pop_names: List[str], name: Optional[str] = None):
        self._pop_names = pop_names
        self._name = name or "multi_pop_steady_firing"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{p}.rate_cv" for p in self._pop_names]

    def compute(self, feature_values: Dict[str, float]) -> float:
        cvs = [feature_values.get(f"{p}.rate_cv", 0.0) for p in self._pop_names]
        return -np.mean(cvs)


class PopulationRateRatioObjective(NetworkObjective):
    """Objective to match a ratio between two populations' rates."""

    def __init__(
        self,
        num_pop: str,
        denom_pop: str,
        target_ratio: float,
        name: Optional[str] = None,
    ):
        self._num_pop = num_pop
        self._denom_pop = denom_pop
        self._target_ratio = target_ratio
        self._name = name or f"{num_pop}_to_{denom_pop}_ratio"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._num_pop}.mean_rate", f"{self._denom_pop}.mean_rate"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        num_rate = feature_values.get(f"{self._num_pop}.mean_rate", 0.0)
        denom_rate = feature_values.get(f"{self._denom_pop}.mean_rate", 0.0)
        if denom_rate > 0:
            actual_ratio = num_rate / denom_rate
        else:
            actual_ratio = 0.0
        return -((actual_ratio - self._target_ratio) ** 2)


class MaximizeFeatureObjective(NetworkObjective):
    """Objective to maximize a feature (e.g. PVBC fraction active)."""

    def __init__(
        self,
        required_features: List[str] = None,
        pop_name: str = None,
        name: Optional[str] = None,
    ):
        self._pop_name = pop_name
        self._required_features = required_features
        self._name = name

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        if self._required_features:
            return self._required_features[0].replace(".", " ")
        return "maximize_feature"

    @property
    def required_features(self) -> List[str]:
        return self._required_features

    def compute(self, feature_values: Dict[str, float]) -> float:
        val = feature_values.get(self._required_features[0], 0.0)
        return val  # higher is better; framework negates for dmosopt


class CustomNetworkObjective(NetworkObjective):
    """Custom objective with user-defined function."""

    def __init__(
        self,
        name: str,
        required_features: List[str],
        fn: Callable[[Dict[str, float]], float],
    ):
        self._name = name
        self._required_features = required_features
        self._fn = fn
        self._validate_callable()

    def _validate_callable(self) -> None:
        if isinstance(self._fn, type(lambda: None)) and self._fn.__name__ == "<lambda>":
            raise TypeError("CustomNetworkObjective.fn must be a module-level function")

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return self._required_features

    def compute(self, feature_values: Dict[str, float]) -> float:
        return self._fn(feature_values)


class FiringRateBoundConstraint(NetworkConstraint):
    """Constraint on firing rate bounds."""

    def __init__(
        self,
        pop_name: str,
        min_rate: float = 0.0,
        max_rate: float = float("inf"),
        feature: str = "mean_rate",
        name: Optional[str] = None,
    ):
        self._pop_name = pop_name
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._feature = feature
        self._name = name or f"{pop_name}_rate_bound"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.{self._feature}"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        rate = feature_values.get(f"{self._pop_name}.{self._feature}", 0.0)
        c1 = rate - self._min_rate
        c2 = self._max_rate - rate
        return min(c1, c2)


class MinActiveFractionConstraint(NetworkConstraint):
    """Constraint on minimum fraction of active neurons."""

    def __init__(self, pop_name: str, min_fraction: float, name: Optional[str] = None):
        self._pop_name = pop_name
        self._min_fraction = min_fraction
        self._name = name or f"{pop_name}_min_active_fraction"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.fraction_active"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        frac = feature_values.get(f"{self._pop_name}.fraction_active", 0.0)
        return frac - self._min_fraction


class ActiveFractionBoundConstraint(NetworkConstraint):
    """Constraint on the fraction of active cells, with a lower and upper bound."""

    def __init__(
        self,
        pop_name: str,
        min_fraction: float = 0.0,
        max_fraction: float = float("inf"),
        feature: str = "fraction_active",
        name: Optional[str] = None,
    ):
        self._pop_name = pop_name
        self._min_fraction = min_fraction
        self._max_fraction = max_fraction
        self._feature = feature
        self._name = name or f"{pop_name}_active_fraction_bound"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.{self._feature}"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        frac = feature_values.get(f"{self._pop_name}.{self._feature}", 0.0)
        return min(frac - self._min_fraction, self._max_fraction - frac)


class SteadyFiringConstraint(NetworkConstraint):
    """Constraint on maximum allowed CV for steady firing."""

    def __init__(self, pop_name: str, max_cv: float, name: Optional[str] = None):
        self._pop_name = pop_name
        self._max_cv = max_cv
        self._name = name or f"{pop_name}_steady_firing"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.rate_cv"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        rate_cv = feature_values.get(f"{self._pop_name}.rate_cv", 0.0)
        return self._max_cv - rate_cv


class SynchronyBoundConstraint(NetworkConstraint):
    """Participation ratio at most ``max_ratio``."""

    def __init__(self, pop_name: str, max_ratio: float, name: Optional[str] = None):
        self._pop_name = pop_name
        self._max_ratio = max_ratio
        self._name = name or f"{pop_name}_synchrony_bound"

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> List[str]:
        return [f"{self._pop_name}.participation_ratio"]

    def compute(self, feature_values: Dict[str, float]) -> float:
        return self._max_ratio - feature_values.get(
            f"{self._pop_name}.participation_ratio", 0.0
        )


def load_network_opt_config(
    config: Dict,
) -> NetworkOptimizationConfig:
    """
    Reads the "Network Optimization" namespace from the given config dict and
    builds a NetworkOptimizationConfig by importing classes via importlib.

    If the namespace is absent, returns an empty config (user must populate it).
    """

    network_opt_config = config.get("Network Optimization", {})

    if not network_opt_config:
        return NetworkOptimizationConfig()

    feature_entries = network_opt_config.get("Features", [])
    objective_entries = network_opt_config.get("Objectives", [])
    constraint_entries = network_opt_config.get("Constraints", [])

    features = _load_feature_list(feature_entries)
    objectives = _load_objective_list(objective_entries)
    constraints = _load_constraint_list(constraint_entries)

    return NetworkOptimizationConfig(
        features=features,
        objectives=objectives,
        constraints=constraints,
    )


def _load_feature_list(entries: List[Dict]) -> List[NetworkFeature]:
    import importlib

    features = []
    for entry in entries:
        if isinstance(entry, dict):
            class_path = entry.get("class")
            if not class_path:
                raise ValueError("Feature entry must have 'class' key")
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls(**entry.get("kwargs", {}))
            if "populations" in entry:
                instance.populations = entry["populations"]
            features.append(instance)
        else:
            raise ValueError(f"Invalid feature entry: {entry}")
    return features


def _load_objective_list(entries: List[Dict]) -> List[NetworkObjective]:
    import importlib

    objectives = []
    for entry in entries:
        if isinstance(entry, dict):
            class_path = entry.get("class")
            if not class_path:
                raise ValueError("Objective entry must have 'class' key")
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # Promote pop_name / pop_names from top-level YAML to kwargs
            kwargs = entry.get("kwargs", {}).copy()
            if "pop_name" in entry:
                kwargs["pop_name"] = entry["pop_name"]
            if "pop_names" in entry:
                kwargs["pop_names"] = entry["pop_names"]
            if "num_pop" in entry:
                kwargs["num_pop"] = entry["num_pop"]
            if "denom_pop" in entry:
                kwargs["denom_pop"] = entry["denom_pop"]

            instance = cls(**kwargs)
            if "name" in entry:
                instance._name = entry["name"]
            objectives.append(instance)
        else:
            raise ValueError(f"Invalid objective entry: {entry}")
    return objectives


def _load_constraint_list(entries: List[Dict]) -> List[NetworkConstraint]:
    import importlib

    constraints = []
    for entry in entries:
        if isinstance(entry, dict):
            class_path = entry.get("class")
            if not class_path:
                raise ValueError("Constraint entry must have 'class' key")
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            # Promote pop_name from top-level YAML to kwargs
            kwargs = entry.get("kwargs", {}).copy()
            if "pop_name" in entry:
                kwargs["pop_name"] = entry["pop_name"]

            instance = cls(**kwargs)
            if "name" in entry:
                instance._name = entry["name"]
            constraints.append(instance)
        else:
            raise ValueError(f"Invalid constraint entry: {entry}")
    return constraints
