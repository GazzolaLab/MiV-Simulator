import os
from pydantic import (
    BaseModel as _BaseModel,
    Field,
    conlist,
    GetCoreSchemaHandler,
)
from typing import Literal, Dict, Any
from enum import IntEnum
from collections import defaultdict
import numpy as np
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator, BeforeValidator
from pydantic_core import CoreSchema, core_schema

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


class SynapseTypesDef(IntEnum):
    excitatory = 0
    inhibitory = 1
    modulatory = 2


class SynapseMechanismsDef(IntEnum):
    AMPA = 0
    GABA_A = 1
    GABA_B = 2
    NMDA = 30


class LayersDef(IntEnum):
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


class InputSelectivityTypesDef(IntEnum):
    random = 0
    constant = 1


class PopulationsDef(IntEnum):
    STIM = 0  # Stimulus
    PYR = 100  # PYR distal dendrites
    PVBC = 101  # Basket cells expressing parvalbumin
    OLM = 102  # GABAergic oriens-lacunosum/moleculare


class AllowStringsFrom:
    """For convenience, allows users to specify enum values using their string name"""

    def __init__(self, enum):
        self.enum = enum

    def cast(self, v):
        if isinstance(v, str):
            try:
                return self.enum.__members__[v]
            except KeyError:
                raise ValueError(
                    f"'{v}'. Must be one of {tuple(self.enum.__members__.keys())}"
                )
        return v

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_before_validator_function(
            self.cast, handler(source_type)
        )


SynapseTypesDefOrStr = Annotated[
    SynapseTypesDef, AllowStringsFrom(SynapseTypesDef)
]
SWCTypesDefOrStr = Annotated[SWCTypesDef, AllowStringsFrom(SWCTypesDef)]
LayersDefOrStr = Annotated[LayersDef, AllowStringsFrom(LayersDef)]
SynapseMechanismsDefOrStr = Annotated[
    SynapseMechanismsDef, AllowStringsFrom(SynapseMechanismsDef)
]


PopulationName = Literal["STIM", "PYR", "PVBC", "OLM"]
PostSynapticPopulationName = PopulationName
PreSynapticPopulationName = PopulationName


# Pydantic data models


class BaseModel(_BaseModel):
    """Hack to ensure dict-access backwards-compatibility"""

    def __getitem__(self, item):
        return getattr(self, item)


class CellDistribution(BaseModel):
    """For a given neuron kind, this defines the distribution
    (i.e. numbers) of neurons accross the different layers."""

    SO: int = Field(title="Stratum oriens")
    SP: int = Field(title="Stratum pyramidale")
    SR: int = Field(title="Stratum radiatum")
    SLM: int = Field(title="Stratum lacunosum-moleculare")


class Mechanism(BaseModel):
    tau_rise: float = None
    tau_decay: float = None
    e: int = None
    g_unit: float
    weight: float


class Synapse(BaseModel):
    type: SynapseTypesDefOrStr
    sections: conlist(SWCTypesDefOrStr)
    layers: conlist(LayersDefOrStr)
    proportions: conlist(float)
    mechanisms: Dict[SynapseMechanismsDefOrStr, Mechanism]


# Composed


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


def config_path(*append) -> str:
    return os.path.join(os.path.dirname(__file__), *append)
