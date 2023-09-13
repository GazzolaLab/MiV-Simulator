import os
from pydantic import (
    BaseModel as _BaseModel,
    Field,
    conlist,
)
from typing import Literal, Dict, Any, List, Tuple, Optional, Union, Callable
from enum import IntEnum
from collections import defaultdict
import numpy as np
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator, BeforeValidator

# Definitions


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
    "default", "Hilus", "GCL", "IML", "MML", "OML", "SO", "SP", "SL", "SR", "SLM"
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

SynapseTypesDefOrStr = Annotated[SynapseTypesDef, AllowStringsFrom(SynapseTypesDef)]
SWCTypesDefOrStr = Annotated[SWCTypesDef, AllowStringsFrom(SWCTypesDef)]
LayersDefOrStr = Annotated[LayersDef, AllowStringsFrom(LayersDef)]
SynapseMechanismsDefOrStr = Annotated[
    SynapseMechanismsDef, AllowStringsFrom(SynapseMechanismsDef)
]
PopulationsDefOrStr = Annotated[PopulationsDef, AllowStringsFrom(PopulationsDef)]


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

"""One of the layers defined in the LayersDef enum."""
LayerName = str
"""For a given neuron kind, this defines the distribution (i.e. numbers) of neurons accross the different layers."""
CellDistribution = Dict[LayerName, int]
"""Describes a volume extent"""
LayerExtents = Dict[LayerName, List[ParametricCoordinate]]
"""Describes constraints on the distribution of neurons in a given layer."""
CellConstraints = Optional[Dict[PopulationName, Dict[LayerName, Tuple[float, float]]]]


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


def _origin_value_to_callable(value: Union[str, float]) -> Callable:
    if isinstance(value, (float, int)):
        return lambda _: value

    return getattr(np, value)


class Origin(BaseModel):
    U: Union[str, float, int]
    V: Union[str, float, int]
    L: Union[str, float, int]

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


def path(*append) -> str:
    return os.path.join(os.path.dirname(__file__), *append)
