__doc__ = """Contains the end-user public API of the MiV-Simulator"""
__all__ = [
    "distribute_synapses",
    "generate_connections",
    "generate_network_architecture",
    "generate_synapse_forest",
    "measure_distances",
    "ExecutionEnvironment",
    "create_neural_h5",
    "configure_hoc",
]
from neuron import h
from miv_simulator.simulator.distribute_synapses import distribute_synapses
from miv_simulator.simulator.generate_connections import generate_connections
from miv_simulator.simulator.generate_network_architecture import (
    generate_network_architecture,
)
from miv_simulator.simulator.generate_synapse_forest import (
    generate_synapse_forest,
)
from miv_simulator.simulator.measure_distances import measure_distances
from miv_simulator.simulator.execution_environment import ExecutionEnvironment
from miv_simulator.utils.io import create_neural_h5
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuron.hoc import HocObject


def configure_hoc(
    coreneuron: bool = False,
    force: bool = False,
    **optional_attrs,
) -> "HocObject":
    if not force and hasattr(h, "pc"):
        # already configured
        return

    h.load_file("stdrun.hoc")
    h.load_file("loadbal.hoc")
    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)
    h("objref pc, nc, nil")
    h("strdef dataset_path")

    if coreneuron:
        from neuron import coreneuron

        coreneuron.enable = True
        coreneuron.verbose = 0

    h.pc = h.ParallelContext()
    h.pc.gid_clear()

    # set optional settings like celsius, dt, etc.
    for k, v in optional_attrs.items():
        setattr(h, k, v)

    # more accurate integration of synaptic discontinuities
    if hasattr(h, "nrn_netrec_state_adjust"):
        h.nrn_netrec_state_adjust = 1

    # sparse parallel transfer
    if hasattr(h, "nrn_sparse_partrans"):
        h.nrn_sparse_partrans = 1

    return h
