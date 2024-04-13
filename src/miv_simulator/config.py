import copy
from pydantic import (
    BaseModel as _BaseModel,
    AfterValidator,
)
from typing import Literal, Dict, List, Tuple, Optional, Union, Callable
from enum import IntEnum
from collections import defaultdict
import numpy as np
from typing_extensions import Annotated

# Definitions

SWCFilePath = str


class SWCTypesDef(IntEnum):
    soma = 1
    axon = 2
    basal = 3
    apical = 4
    trunk = 5
    tuft = 6
    ais = 7
    hillock = 8


SWCTypesLiteral = Literal[
    "soma", "axon", "basal", "apical", "trunk", "tuft", "ais", "hillock"
]


class SynapseTypesDef(IntEnum):
    excitatory = 0
    inhibitory = 1
    modulatory = 2


SynapseTypesLiteral = Literal["excitatory", "inhibitory", "modulatory"]

# Synapses

SynapseMechanismName = str

# Population

PopulationName = str
PostSynapticPopulationName = PopulationName
PreSynapticPopulationName = PopulationName


# Geometry

X_Coordinate = float
Y_Coordinate = float
Z_Coordinate = float
U_Coordinate = float
V_Coordinate = float
L_Coordinate = float

ParametricCoordinate = Tuple[U_Coordinate, V_Coordinate, L_Coordinate]
Rotation = Tuple[float, float, float]


LayerName = str
"""One of the layers defined in the LayersDef enum."""

CellDistribution = Dict[LayerName, int]
"""For a given neuron kind, this defines the distribution (i.e. numbers) of neurons accross the different layers.

Example:
```python
{
    "STIM": {"SO":  0, "SP":  64, "SR": 0, "SLM": 0},
    "PYR":  {"SO":  0, "SP": 223, "SR": 0, "SLM": 0},
    "PVBC": {"SO": 35, "SP":  50, "SR": 8, "SLM": 0},
    "OLM":  {"SO": 21, "SP":   0, "SR": 0, "SLM": 0},
}
```
"""

LayerExtents = Dict[LayerName, List[ParametricCoordinate]]
"""Describes a volume extent

Example:
```python
{
    "SO": [[0.0, 0.0, 0.0], [200.0, 200.0, 5.0]],
    "SP": [[0.0, 0.0, 5.0], [200.0, 200.0, 50.0]],
    "SR": [[0.0, 0.0, 50.0], [200.0, 200.0, 100.0]],
    "SLM": [[0.0, 0.0, 100.0], [200.0, 200.0, 150.0]],
}
```
"""

CellConstraints = Optional[
    Dict[PopulationName, Dict[LayerName, Tuple[float, float]]]
]
"""Describes constraints on the distribution of neurons in a given layer.

Example:
```python
{
    "PC": {"SP": [100, 120]},
    "PVBC": {"SR": [150, 200]},
}
```
"""


# Pydantic data models


class BaseModel(_BaseModel):
    """Hack to ensure dict-access backwards-compatibility"""

    def __getitem__(self, item):
        return getattr(self, item)


class Mechanism(BaseModel):
    g_unit: Optional[float] = None
    weight: Optional[float] = None
    tau_rise: Optional[float] = None
    tau_decay: Optional[float] = None
    e: Optional[int] = None


class Synapse(BaseModel):
    type: SynapseTypesLiteral
    sections: List[SWCTypesLiteral]
    layers: List[LayerName]
    proportions: list[float]
    mechanisms: Dict[SynapseMechanismName, Dict[Union[str, int], Mechanism]]
    contacts: int = 1

    def to_config(self, layer_definitions: Dict[LayerName, int]):
        return type(
            "SynapseConfig",
            (),
            {
                "type": SynapseTypesDef.__members__[self.type],
                "sections": list(
                    map(SWCTypesDef.__members__.get, self.sections)
                ),
                "layers": list(map(layer_definitions.get, self.layers)),
                "proportions": self.proportions,
                "mechanisms": self.mechanisms,
                "contacts": self.contacts,
            },
        )


def _origin_value_to_callable(value: Union[str, float]) -> Callable:
    if isinstance(value, float):
        return lambda _: value

    return getattr(np, value)


class Origin(BaseModel):
    U: Union[str, float]
    V: Union[str, float]
    L: Union[str, float]

    def as_spec(self):
        return {
            "U": _origin_value_to_callable(self.U),
            "V": _origin_value_to_callable(self.V),
            "L": _origin_value_to_callable(self.L),
        }


class ParametricSurface(BaseModel):
    Origin: Origin
    Layer_Extents: LayerExtents
    Rotation: List[float]


class CellType(BaseModel):
    template: str
    synapses: Dict[
        Literal["density"],
        Dict[
            SWCTypesLiteral,
            Dict[
                SynapseTypesLiteral,
                Dict[
                    str,
                    Dict[Literal["mean", "variance"], float],
                ],
            ],
        ],
    ]
    mechanism: Optional[Dict] = None


CellTypes = Dict[str, CellType]


class AxonExtent(BaseModel):
    width: Tuple[float, float]
    offset: Tuple[float, float]


AxonExtents = Dict[str, Dict[LayerName, AxonExtent]]


def probabilities_sum_to_one(x):
    sums = defaultdict(lambda: 0.0)
    for key_presyn, conn_config in x.items():
        for s, l, p in zip(
            conn_config.sections,
            conn_config.layers,
            conn_config.proportions,
        ):
            sums[(conn_config.type, s, l)] += p

    for k, v in sums.items():
        if not np.isclose(v, 1.0):
            raise ValueError(
                f"Invalid connection configuration: probabilities do not sum to 1 ({k}={v})"
            )

    return x


Synapses = Dict[
    PostSynapticPopulationName,
    Annotated[
        Dict[PreSynapticPopulationName, Synapse],
        AfterValidator(probabilities_sum_to_one),
    ],
]


CellDistributions = Dict[PopulationName, CellDistribution]

SynapticProjections = Dict[
    PostSynapticPopulationName,
    List[PreSynapticPopulationName],
]

sentinel = object()


class Definitions(BaseModel):
    swc_types: Dict[str, int]
    synapse_types: Dict[str, int]
    synapse_mechanisms: Dict[str, int]
    layers: Dict[str, int]
    populations: Dict[str, int]
    input_selectivity_types: Dict[str, int]


class Config:
    def __init__(self, data: Dict) -> None:
        self._data = copy.deepcopy(data)
        self._definitions = None

        # compatibility
        self.get("Cell Types.STIM", {}).setdefault("synapses", {})

    @property
    def data(self) -> Dict:
        return self._data

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        from miv_simulator.utils import from_yaml

        return cls(from_yaml(filepath))

    @classmethod
    def from_json(cls, filepath: str) -> "Config":
        import json

        with open(filepath, "r") as f:
            return cls(json.load(f))

    def get(self, path: str, default=sentinel, splitter: str = "."):
        d = self._data
        for key in path.split(splitter):
            if key not in d:
                if default is not sentinel:
                    return default

            d = d[key]

        return d

    @property
    def cell_distributions(self) -> CellDistributions:
        return self.get("Geometry.Cell Distribution")

    @property
    def layer_extents(self) -> LayerExtents:
        return self.get("Geometry.Parametric Surface.Layer Extents")

    @property
    def geometry_rotation(self) -> Rotation:
        return self.get("Geometry.Parametric Surface.Rotation", (0.0, 0.0, 0.0))

    @property
    def geometry_origin(self) -> Origin:
        return self.get(
            "Geometry.Parametric Surface.Origin",
            {"U": "median", "V": "median", "L": "max"},
        )

    @property
    def axon_extents(self) -> AxonExtents:
        return self.get("Connection Generator.Axon Extent")

    @property
    def synapses(self) -> Synapses:
        synapses = {}
        for post, v in self.get("Connection Generator.Synapses").items():
            synapses[post] = {}
            for pre, syn in v.items():
                synapse = copy.copy(syn)
                if "swctype mechanisms" in synapse:
                    synapse["mechanisms"] = {
                        SWCTypesDef.__members__[swc_type]: c
                        for swc_type, c in synapse["swctype mechanisms"].items()
                    }
                    del synapse["swctype mechanisms"]
                elif "mechanisms" in synapse:
                    synapse["mechanisms"] = {"default": synapse["mechanisms"]}
                synapses[post][pre] = synapse
        return synapses

    @property
    def projections(self) -> SynapticProjections:
        return {post: list(pre.keys()) for post, pre in self.synapses.items()}

    @property
    def cell_types(self) -> CellTypes:
        return self.get("Cell Types")

    @property
    def clamp(self) -> Optional[Dict]:
        return self.get("Network Clamp", None)

    @property
    def definitions(self) -> Definitions:
        if self._definitions is None:
            self._definitions = Definitions(
                swc_types=self.get("Definitions.SWC Types", {}),
                synapse_types=self.get("Definitions.Synapse Types", {}),
                synapse_mechanisms=self.get(
                    "Definitions.Synapse Mechanisms", {}
                ),
                layers=self.get("Definitions.Layers", {}),
                populations=self.get("Definitions.Populations", {}),
                input_selectivity_types=self.get(
                    "Definitions.Input Selectivity Types", {}
                ),
            )
        return self._definitions
