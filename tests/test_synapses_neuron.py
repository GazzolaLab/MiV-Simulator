"""Unit tests for NEURON-object-dependent routines in miv_simulator.synapses.

These tests create real NEURON objects (h.Section, h.ExpSyn, h.Exp2Syn,
h.NetCon) and exercise the functions that interact with them.  They must be
run with the NEURON Python extension on the path:

    PYTHONPATH=/home/igr/bin/nrnpython3/lib/python:$PYTHONPATH \\
        /home/igr/venv/bin/pytest tests/test_synapses_neuron.py -v

Pure Python / NumPy functionality is tested separately in test_synapses.py.
"""

from collections import defaultdict

import numpy as np
import pytest

neuron = pytest.importorskip("neuron")
from neuron import h  # noqa: E402 (after importorskip guard)

from miv_simulator.synapses import (  # noqa: E402
    SynapseManager,
    config_syn,
    make_shared_synapse_mech,
    make_syn_mech,
    make_unique_synapse_mech,
    syn_in_seg,
)
from miv_simulator.utils import AbstractEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYN_EXC = 0
SYN_INH = 1
SWC_APICAL = 4
GID = 42
POP_GC = 0
POP_MC = 1

SYN_MECH_NAMES = {"AMPA": "ExpSyn", "GABA": "Exp2Syn"}
SYN_PARAM_RULES = {
    "ExpSyn": {"mech_params": ["tau", "e"], "netcon_params": {"weight": 0}},
    "Exp2Syn": {"mech_params": ["tau1", "tau2", "e"], "netcon_params": {"weight": 0}},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class MockEnv(AbstractEnv):
    def __init__(self, Populations=None, celltypes=None):
        self.Synapse_Types = {"excitatory": 0, "inhibitory": 1}
        self.SWC_Types = {"soma": 1, "apical": 4}
        self.layers = {"default": -1, "Layer1": 1}
        self.Populations = Populations or {"GC": 0, "MC": 1}
        self.celltypes = celltypes or {}
        self.synapse_manager = None


def make_real_section(name="test", L=100.0, diam=1.0, nseg=5):
    """Create a real NEURON Section with geometry."""
    sec = h.Section(name=name)
    sec.L = L
    sec.diam = diam
    sec.nseg = nseg
    return sec


def syns_dict_factory():
    """Return a fresh nested defaultdict for use with syn_in_seg / make_shared_synapse_mech."""
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sec():
    s = make_real_section()
    yield s
    # No explicit cleanup needed; NEURON GC handles it


@pytest.fixture
def mid_seg(sec):
    """Middle segment of a 5-segment section."""
    return sec(0.5)


@pytest.fixture
def syn_manager():
    env = MockEnv()
    return SynapseManager(env, SYN_MECH_NAMES, SYN_PARAM_RULES)


@pytest.fixture
def populated_manager(syn_manager):
    n = 6
    syn_ids = np.arange(n, dtype=np.uint32)
    syn_manager.init_syn_id_attrs(
        GID,
        syn_ids,
        np.zeros(n, dtype=np.int8),
        np.zeros(n, dtype=np.uint8),
        np.full(n, SWC_APICAL, dtype=np.uint8),
        np.zeros(n, dtype=np.uint16),
        np.full(n, 0.5, dtype=np.float32),
    )
    return syn_manager


# ===========================================================================
# TestSynInSeg
# ===========================================================================


class TestSynInSeg:
    def test_returns_none_for_empty_syns_dict(self, mid_seg):
        sd = syns_dict_factory()
        assert syn_in_seg("AMPA", mid_seg, sd) is None

    def test_returns_mechanism_when_present(self, sec, mid_seg):
        sd = syns_dict_factory()
        syn = h.ExpSyn(mid_seg)
        sd[sec][mid_seg.x]["AMPA"] = syn
        result = syn_in_seg("AMPA", mid_seg, sd)
        assert result is syn

    def test_returns_none_for_different_mechanism_name(self, sec, mid_seg):
        sd = syns_dict_factory()
        syn = h.ExpSyn(mid_seg)
        sd[sec][mid_seg.x]["AMPA"] = syn
        # Search for GABA, which is not stored
        assert syn_in_seg("GABA", mid_seg, sd) is None

    def test_returns_none_for_wrong_segment_position(self, sec):
        sd = syns_dict_factory()
        seg_at_02 = sec(0.2)
        seg_at_07 = sec(0.7)
        syn = h.ExpSyn(seg_at_02)
        sd[sec][seg_at_02.x]["AMPA"] = syn
        # Searching at 0.7 must not find the mechanism at 0.2
        result = syn_in_seg("AMPA", seg_at_07, sd)
        assert result is None


# ===========================================================================
# TestMakeSynMech
# ===========================================================================


class TestMakeSynMech:
    def test_creates_point_process_of_correct_type(self, mid_seg):
        syn = make_syn_mech("ExpSyn", mid_seg)
        assert hasattr(syn, "tau")

    def test_point_process_attached_to_correct_segment(self, sec, mid_seg):
        syn = make_syn_mech("ExpSyn", mid_seg)
        # The point process should be in the correct segment
        assert syn.get_segment() == mid_seg

    def test_invalid_mechanism_name_raises(self, mid_seg):
        with pytest.raises((AttributeError, Exception)):
            make_syn_mech("NoSuchMech_xyz_999", mid_seg)


# ===========================================================================
# TestMakeSharedSynapseMech
# ===========================================================================


class TestMakeSharedSynapseMech:
    def test_creates_new_when_empty(self, mid_seg):
        sd = syns_dict_factory()
        syn = make_shared_synapse_mech("AMPA", mid_seg, sd, mech_names=SYN_MECH_NAMES)
        assert syn is not None
        assert hasattr(syn, "tau")

    def test_returns_same_object_for_same_segment(self, mid_seg):
        sd = syns_dict_factory()
        syn1 = make_shared_synapse_mech("AMPA", mid_seg, sd, mech_names=SYN_MECH_NAMES)
        syn2 = make_shared_synapse_mech("AMPA", mid_seg, sd, mech_names=SYN_MECH_NAMES)
        assert syn1 is syn2

    def test_creates_new_for_different_segment(self, sec):
        sd = syns_dict_factory()
        seg_a = sec(0.3)
        seg_b = sec(0.7)
        syn_a = make_shared_synapse_mech("AMPA", seg_a, sd, mech_names=SYN_MECH_NAMES)
        syn_b = make_shared_synapse_mech("AMPA", seg_b, sd, mech_names=SYN_MECH_NAMES)
        assert syn_a is not syn_b

    def test_syns_dict_populated_after_creation(self, sec, mid_seg):
        sd = syns_dict_factory()
        make_shared_synapse_mech("AMPA", mid_seg, sd, mech_names=SYN_MECH_NAMES)
        assert sd[sec][mid_seg.x]["AMPA"] is not None

    def test_mech_names_dict_resolved(self, mid_seg):
        sd = syns_dict_factory()
        syn = make_shared_synapse_mech("AMPA", mid_seg, sd, mech_names=SYN_MECH_NAMES)
        # ExpSyn has tau; Exp2Syn has tau1/tau2
        assert hasattr(syn, "tau")
        assert not hasattr(syn, "tau1")


# ===========================================================================
# TestMakeUniqueSynapseMech
# ===========================================================================


class TestMakeUniqueSynapseMech:
    def test_always_creates_new_object(self, mid_seg):
        syn1 = make_unique_synapse_mech("AMPA", mid_seg, mech_names=SYN_MECH_NAMES)
        syn2 = make_unique_synapse_mech("AMPA", mid_seg, mech_names=SYN_MECH_NAMES)
        assert syn1 is not syn2

    def test_mech_names_dict_resolved(self, mid_seg):
        syn = make_unique_synapse_mech("GABA", mid_seg, mech_names=SYN_MECH_NAMES)
        # Exp2Syn has tau1 and tau2
        assert hasattr(syn, "tau1")
        assert hasattr(syn, "tau2")

    def test_no_syns_dict_update_when_provided(self, sec, mid_seg):
        sd = syns_dict_factory()
        make_unique_synapse_mech(
            "AMPA", mid_seg, syns_dict=sd, mech_names=SYN_MECH_NAMES
        )
        # unique mode must not populate syns_dict
        assert sd[sec][mid_seg.x]["AMPA"] is None


# ===========================================================================
# TestConfigSyn
# ===========================================================================


class TestConfigSyn:
    """Tests using real h.Section, h.ExpSyn, and h.NetCon objects."""

    @pytest.fixture
    def syn_and_nc(self, mid_seg):
        syn = h.ExpSyn(mid_seg)
        nc = h.NetCon(None, syn)
        return syn, nc

    def test_sets_mech_param_on_syn(self, syn_and_nc):
        syn, nc = syn_and_nc
        mech_param_set, nc_param_set = config_syn(
            "AMPA", SYN_PARAM_RULES, mech_names=SYN_MECH_NAMES, syn=syn, tau=5.0
        )
        assert mech_param_set is True
        assert pytest.approx(syn.tau) == 5.0

    def test_sets_netcon_weight(self, syn_and_nc):
        syn, nc = syn_and_nc
        mech_param_set, nc_param_set = config_syn(
            "AMPA", SYN_PARAM_RULES, mech_names=SYN_MECH_NAMES, nc=nc, weight=0.001
        )
        assert nc_param_set is True
        assert pytest.approx(nc.weight[0]) == 0.001

    def test_returns_correct_flags(self, syn_and_nc):
        syn, nc = syn_and_nc
        mech_set, nc_set = config_syn(
            "AMPA",
            SYN_PARAM_RULES,
            mech_names=SYN_MECH_NAMES,
            syn=syn,
            nc=nc,
            tau=2.0,
            weight=0.002,
        )
        assert mech_set is True
        assert nc_set is True

    def test_none_value_skipped(self, syn_and_nc):
        syn, nc = syn_and_nc
        original_tau = syn.tau
        config_syn(
            "AMPA", SYN_PARAM_RULES, mech_names=SYN_MECH_NAMES, syn=syn, tau=None
        )
        assert pytest.approx(syn.tau) == original_tau

    def test_unknown_param_raises(self, syn_and_nc):
        syn, nc = syn_and_nc
        with pytest.raises(RuntimeError, match="Unknown parameter"):
            config_syn(
                "AMPA",
                SYN_PARAM_RULES,
                mech_names=SYN_MECH_NAMES,
                syn=syn,
                no_such_param=1.0,
            )

    def test_syn_none_with_mech_param_raises(self, syn_and_nc):
        _, nc = syn_and_nc
        with pytest.raises(RuntimeError, match="mechanism object required"):
            config_syn(
                "AMPA",
                SYN_PARAM_RULES,
                mech_names=SYN_MECH_NAMES,
                syn=None,
                nc=nc,
                tau=5.0,
            )

    def test_nc_none_with_netcon_param_raises(self, syn_and_nc):
        syn, _ = syn_and_nc
        with pytest.raises(RuntimeError, match="NetCon object required"):
            config_syn(
                "AMPA",
                SYN_PARAM_RULES,
                mech_names=SYN_MECH_NAMES,
                syn=syn,
                nc=None,
                weight=0.001,
            )

    def test_list_value_of_length_one_unwrapped(self, syn_and_nc):
        syn, nc = syn_and_nc
        config_syn(
            "AMPA", SYN_PARAM_RULES, mech_names=SYN_MECH_NAMES, nc=nc, weight=[0.5]
        )
        assert pytest.approx(nc.weight[0]) == 0.5

    def test_list_value_of_length_gt_one_raises(self, syn_and_nc):
        syn, nc = syn_and_nc
        with pytest.raises(RuntimeError, match="length > 1"):
            config_syn(
                "AMPA",
                SYN_PARAM_RULES,
                mech_names=SYN_MECH_NAMES,
                nc=nc,
                weight=[0.5, 0.6],
            )


# ===========================================================================
# TestSynapseManagerInitEdgeAttrsWithNeuron
# ===========================================================================


class TestSynapseManagerInitEdgeAttrsWithNeuron:
    """Tests init_edge_attrs with delays=None, which requires h.dt."""

    def test_none_delays_uses_h_dt(self, syn_manager):
        n = 4
        syn_ids = np.arange(n, dtype=np.uint32)
        syn_manager.init_syn_id_attrs(
            GID,
            syn_ids,
            np.zeros(n, dtype=np.int8),
            np.zeros(n, dtype=np.uint8),
            np.full(n, SWC_APICAL, dtype=np.uint8),
            np.zeros(n, dtype=np.uint16),
            np.full(n, 0.5, dtype=np.float32),
        )
        # Pass delays=None — uses 2.0 * h.dt
        syn_manager.init_edge_attrs(
            GID, "GC", np.arange(n, dtype=np.int32), syn_ids, delays=None
        )
        expected_delay = 2.0 * h.dt
        for sid in syn_ids:
            view = syn_manager.get_synapse(GID, int(sid))
            assert pytest.approx(float(view.source.delay), abs=1e-6) == expected_delay


# ===========================================================================
# TestSynapseManagerRealPointProcesses
# ===========================================================================


class TestSynapseManagerRealPointProcesses:
    """Verify that pps_dict correctly stores real NEURON HocObjects."""

    def test_add_and_get_real_expsyn(self, syn_manager, mid_seg):
        syn = h.ExpSyn(mid_seg)
        syn.tau = 3.0
        syn_manager.add_point_process(GID, syn_id=0, syn_name="AMPA", pps=syn)
        retrieved = syn_manager.get_point_process(GID, syn_id=0, syn_name="AMPA")
        assert pytest.approx(retrieved.tau) == 3.0

    def test_add_and_get_real_netcon(self, syn_manager, mid_seg):
        syn = h.ExpSyn(mid_seg)
        nc = h.NetCon(None, syn)
        nc.weight[0] = 0.007
        syn_manager.add_netcon(GID, syn_id=0, syn_name="AMPA", nc=nc)
        retrieved = syn_manager.get_netcon(GID, syn_id=0, syn_name="AMPA")
        assert pytest.approx(retrieved.weight[0]) == 0.007

    def test_add_and_get_real_exp2syn(self, syn_manager, mid_seg):
        syn = h.Exp2Syn(mid_seg)
        syn.tau1 = 0.5
        syn.tau2 = 10.0
        syn_manager.add_point_process(GID, syn_id=1, syn_name="GABA", pps=syn)
        retrieved = syn_manager.get_point_process(GID, syn_id=1, syn_name="GABA")
        assert pytest.approx(retrieved.tau1) == 0.5
        assert pytest.approx(retrieved.tau2) == 10.0
