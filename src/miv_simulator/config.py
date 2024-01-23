import copy
from pydantic import (
    BaseModel as _BaseModel,
    Field,
    conlist,
    AfterValidator,
    BeforeValidator,
)
from typing import Literal, Dict, Any, List, Tuple, Optional, Union, Callable
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


class SynapseMechanismsDef(IntEnum):
    AMPA = 0
    GABA_A = 1
    GABA_B = 2
    NMDA = 30


SynapseMechanismsLiteral = Literal["AMPA", "GABA_A", "GABA_B", "NMDA"]


class LayersDef(IntEnum):
    default = -1
    Hilus = 0
    GCL = 1  # Granule cell
    IML = 2  # Inner molecular
    MML = 3  # Middle molecular
    OML = 4  # Outer molecular
    SO = 5  # Oriens
    SP = 6  # Pyramidale
    SL = 7  # Lucidum
    SR = 8  # Radiatum
    SLM = 9  # Lacunosum-moleculare


LayersLiteral = Literal[
    "default",
    "Hilus",
    "GCL",
    "IML",
    "MML",
    "OML",
    "SO",
    "SP",
    "SL",
    "SR",
    "SLM",
]


class InputSelectivityTypesDef(IntEnum):
    random = 0
    constant = 1


class PopulationsDef(IntEnum):
    STIM = 0  # Stimulus
    PYR = 100  # PYR distal dendrites
    PVBC = 101  # Basket cells expressing parvalbumin
    OLM = 102  # GABAergic oriens-lacunosum/moleculare


PopulationsLiteral = Literal["STIM", "PYR", "PVBC", "OLM"]


def AllowStringsFrom(enum):
    """For convenience, allows users to specify enum values using their string name"""

    def _cast(v) -> int:
        if isinstance(v, str):
            try:
                return enum.__members__[v]
            except KeyError:
                raise ValueError(
                    f"'{v}'. Must be one of {tuple(enum.__members__.keys())}"
                )
        return v

    return BeforeValidator(_cast)


# Population

SynapseTypesDefOrStr = Annotated[
    SynapseTypesDef, AllowStringsFrom(SynapseTypesDef)
]
SWCTypesDefOrStr = Annotated[SWCTypesDef, AllowStringsFrom(SWCTypesDef)]
LayersDefOrStr = Annotated[LayersDef, AllowStringsFrom(LayersDef)]
SynapseMechanismsDefOrStr = Annotated[
    SynapseMechanismsDef, AllowStringsFrom(SynapseMechanismsDef)
]
PopulationsDefOrStr = Annotated[
    PopulationsDef, AllowStringsFrom(PopulationsDef)
]


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
    g_unit: float
    weight: float
    tau_rise: Optional[float] = None
    tau_decay: Optional[float] = None
    e: Optional[int] = None


class Synapse(BaseModel):
    type: SynapseTypesDefOrStr
    sections: conlist(SWCTypesDefOrStr)
    layers: conlist(LayersDefOrStr)
    proportions: conlist(float)
    mechanisms: Dict[SynapseMechanismsLiteral, Mechanism]
    contacts: int = 1


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
                    LayersLiteral,
                    Dict[Literal["mean", "variance"], float],
                ],
            ],
        ],
    ]


CellTypes = Dict[PopulationsLiteral, CellType]


class AxonExtent(BaseModel):
    width: Tuple[float, float]
    offset: Tuple[float, float]


AxonExtents = Dict[PopulationsLiteral, Dict[LayerName, AxonExtent]]


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


class Config:
    def __init__(self, data: Dict) -> None:
        self._data = copy.deepcopy(data)

        # compatibility
        self.get("Cell Types.STIM", {}).setdefault("synapses", {})

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        from miv_simulator.utils import from_yaml

        return cls(from_yaml(filepath))

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
        return self.get("Connection Generator.Synapses")

    @property
    def projections(self) -> SynapticProjections:
        return {post: list(pre.keys()) for post, pre in self.synapses.items()}

    @property
    def cell_types(self) -> CellTypes:
        return self.get("Cell Types")
