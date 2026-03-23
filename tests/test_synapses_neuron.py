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
    config_cell_syns,
    config_syn,
    insert_cell_syns,
    make_shared_synapse_mech,
    make_syn_mech,
    make_unique_synapse_mech,
    modify_syn_param,
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
        # Pass delays=None (default is to use 2.0 * h.dt)
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


# ===========================================================================
# Helpers shared by TestInsertCellSyns and TestConfigCellSyns
# ===========================================================================

# SWC type table required by insert_cell_syns (six compartment types).
_FULL_SWC_TYPES = {
    "soma": 1,
    "axon": 2,
    "basal": 3,
    "apical": SWC_APICAL,
    "ais": 8,
    "hillock": 9,
}

# Connection-config mechanism params used in insert_cell_syns tests.
# Keys are synapse labels ("AMPA"), matching syn_mech_names.
_MECH_PARAMS = {"tau": 5.0, "e": 0.0}
_DEFAULT_SYN_PARAMS = {"default": {"AMPA": _MECH_PARAMS}}


class _MechConfig:
    """Stand-in for env.connection_config[postsyn][presyn]."""

    def __init__(self, mechanisms=None):
        self.mechanisms = mechanisms or _DEFAULT_SYN_PARAMS


class _MockPC:
    """Minimal ParallelContext stub; only id() is required by config_cell_syns."""

    def id(self):
        return 0


def _make_insert_env(syn_manager, sections, gid=GID, presyn="GC", postsyn="GC"):
    """Return a minimal AbstractEnv for insert_cell_syns."""

    class _Env(AbstractEnv):
        def __init__(self):
            self.Synapse_Types = {"excitatory": SYN_EXC, "inhibitory": SYN_INH}
            self.SWC_Types = _FULL_SWC_TYPES
            # layer 0 is used by np.zeros(n, int8) in init_syn_id_attrs
            self.layers = {"default": 0}
            self.Populations = {"GC": POP_GC, "MC": POP_MC}
            self.celltypes = {postsyn: {"synapses": {}}}
            self.connection_config = {postsyn: {presyn: _MechConfig()}}
            self.biophys_cells = {postsyn: {gid: _CellObj(sections)}}
            self.synapse_manager = syn_manager

    class _CellObj:
        def __init__(self, secs):
            self.sections = secs

    return _Env()


def _make_config_env(syn_manager, postsyn="GC"):
    """Return a minimal AbstractEnv for config_cell_syns (insert=False)."""

    class _Env(AbstractEnv):
        def __init__(self):
            self.Synapse_Types = {"excitatory": SYN_EXC, "inhibitory": SYN_INH}
            self.SWC_Types = _FULL_SWC_TYPES
            self.layers = {"default": 0}
            self.Populations = {"GC": POP_GC, "MC": POP_MC}
            self.celltypes = {postsyn: {"synapses": {}}}
            self.synapse_manager = syn_manager
            self.pc = _MockPC()

    return _Env()


def _init_synapses(syn_manager, gid, n, presyn="GC", swc_type=SWC_APICAL):
    """Register n apical synapses for gid, all at syn_loc=0.5 in section 0."""
    syn_ids = np.arange(n, dtype=np.uint32)
    syn_manager.init_syn_id_attrs(
        gid,
        syn_ids,
        np.zeros(n, dtype=np.int8),  # syn_layers   = 0
        np.zeros(n, dtype=np.uint8),  # syn_types    = excitatory
        np.full(n, swc_type, dtype=np.uint8),  # swc_types
        np.zeros(n, dtype=np.uint16),  # syn_secs     = section 0
        np.full(n, 0.5, dtype=np.float32),  # syn_locs     = 0.5
    )
    syn_manager.init_edge_attrs(
        gid, presyn, np.arange(n, dtype=np.int32), syn_ids, delays=[2.0] * n
    )
    return syn_ids


# ===========================================================================
# TestInsertCellSyns
# ===========================================================================


class TestInsertCellSyns:
    """Verify that insert_cell_syns creates real NEURON point processes and
    writes the mechanism parameter values from connection_config.mechanisms
    directly onto those NEURON objects."""

    @pytest.fixture
    def cell_sec(self):
        return make_real_section(name="apical_0")

    @pytest.fixture
    def insert_env(self, syn_manager, cell_sec):
        """Env + registered syn_ids for a 3-synapse insert_cell_syns scenario."""
        syn_ids = _init_synapses(syn_manager, GID, n=3)
        env = _make_insert_env(syn_manager, sections=[cell_sec])
        return env, syn_ids

    # ------------------------------------------------------------------
    # Parameter propagation to NEURON mechanism

    def test_tau_written_to_point_process(self, insert_env):
        env, syn_ids = insert_env
        insert_cell_syns(env, GID, "GC", "GC", syn_ids)
        pp = env.synapse_manager.get_point_process(GID, int(syn_ids[0]), "AMPA")
        assert pytest.approx(pp.tau) == _MECH_PARAMS["tau"]

    def test_e_written_to_point_process(self, insert_env):
        env, syn_ids = insert_env
        insert_cell_syns(env, GID, "GC", "GC", syn_ids)
        pp = env.synapse_manager.get_point_process(GID, int(syn_ids[0]), "AMPA")
        assert pytest.approx(pp.e) == _MECH_PARAMS["e"]

    def test_all_syn_ids_get_point_processes(self, insert_env):
        env, syn_ids = insert_env
        insert_cell_syns(env, GID, "GC", "GC", syn_ids)
        for sid in syn_ids:
            assert env.synapse_manager.has_point_process(GID, int(sid), "AMPA")

    # ------------------------------------------------------------------
    # Return-value shape

    def test_returns_correct_syn_count(self, insert_env):
        env, syn_ids = insert_env
        syn_count, _, _ = insert_cell_syns(env, GID, "GC", "GC", syn_ids)
        assert syn_count == len(syn_ids)

    def test_returns_correct_mech_count(self, insert_env):
        """One AMPA mechanism inserted per syn_id."""
        env, syn_ids = insert_env
        _, mech_count, _ = insert_cell_syns(env, GID, "GC", "GC", syn_ids)
        assert mech_count == len(syn_ids)

    def test_nc_count_zero_without_insert_netcons(self, insert_env):
        env, syn_ids = insert_env
        _, _, nc_count = insert_cell_syns(env, GID, "GC", "GC", syn_ids)
        assert nc_count == 0

    # ------------------------------------------------------------------
    # Shared vs. unique mechanism objects

    def test_shared_mode_same_segment_reuses_hoc_object(self, insert_env):
        """unique=False: two syn_ids at the same segment share one h.ExpSyn."""
        env, syn_ids = insert_env
        insert_cell_syns(env, GID, "GC", "GC", syn_ids[:2], unique=False)
        mgr = env.synapse_manager
        pp0 = mgr.get_point_process(GID, int(syn_ids[0]), "AMPA")
        pp1 = mgr.get_point_process(GID, int(syn_ids[1]), "AMPA")
        assert pp0 is pp1

    def test_unique_mode_different_hoc_objects_per_syn_id(self, insert_env):
        """unique=True: every syn_id gets its own h.ExpSyn even at the same segment."""
        env, syn_ids = insert_env
        insert_cell_syns(env, GID, "GC", "GC", syn_ids[:2], unique=True)
        mgr = env.synapse_manager
        pp0 = mgr.get_point_process(GID, int(syn_ids[0]), "AMPA")
        pp1 = mgr.get_point_process(GID, int(syn_ids[1]), "AMPA")
        assert pp0 is not pp1

    # ------------------------------------------------------------------
    # Error handling

    def test_missing_biophys_cell_raises(self, insert_env):
        env, syn_ids = insert_env
        with pytest.raises(KeyError):
            insert_cell_syns(env, 9999, "GC", "GC", syn_ids)


# ===========================================================================
# TestConfigCellSyns
# ===========================================================================


class TestConfigCellSyns:
    """Verify that config_cell_syns reads parameters from the synapse manager
    and writes them onto existing NEURON point process objects."""

    @pytest.fixture
    def config_setup(self, syn_manager):
        """One synapse with a pre-inserted h.ExpSyn and default params stored
        under the mechanism name ("ExpSyn") as expected by
        get_effective_mechanism_parameters.

        Uses yield so that the local `sec` variable stays alive for the
        full duration of each test (NEURON deletes a Section when its Python
        wrapper is GC'd, which would detach the ExpSyn).
        """
        gid = GID
        syn_ids = _init_synapses(syn_manager, gid, n=1)

        # Insert a real ExpSyn with a sentinel tau that config_cell_syns must overwrite
        sec = make_real_section(name="cfg_sec")
        pp = h.ExpSyn(sec(0.5))
        pp.tau = 1.0  # sentinel that must be replaced
        pp.e = 0.0
        syn_manager.add_point_process(gid, 0, "AMPA", pp)

        # Store defaults under the mechanism name ("ExpSyn"), which is the key
        # used internally by get_effective_mechanism_parameters.
        param_store = syn_manager.syn_store.param_store
        param_store.set_default_value(gid, "ExpSyn", "tau", 7.5)
        param_store.set_default_value(gid, "ExpSyn", "e", -5.0)

        env = _make_config_env(syn_manager)
        yield env, syn_ids, pp
        # sec remains in scope until here, keeping the ExpSyn attached

    # ------------------------------------------------------------------
    # Parameter propagation from synapse manager to NEURON mechanism

    def test_tau_applied_to_existing_point_process(self, config_setup):
        env, syn_ids, pp = config_setup
        config_cell_syns(env, GID, "GC", syn_ids=syn_ids)
        assert pytest.approx(pp.tau) == 7.5

    def test_e_applied_to_existing_point_process(self, config_setup):
        env, syn_ids, pp = config_setup
        config_cell_syns(env, GID, "GC", syn_ids=syn_ids)
        assert pytest.approx(pp.e) == -5.0

    # ------------------------------------------------------------------
    # Return-value shape

    def test_returns_one_syn_count(self, config_setup):
        env, syn_ids, _ = config_setup
        syn_count, _, _ = config_cell_syns(env, GID, "GC", syn_ids=syn_ids)
        assert syn_count == 1

    def test_returns_nonzero_mech_count(self, config_setup):
        env, syn_ids, _ = config_setup
        _, mech_count, _ = config_cell_syns(env, GID, "GC", syn_ids=syn_ids)
        assert mech_count >= 1

    def test_nc_count_zero_without_netcon(self, config_setup):
        env, syn_ids, _ = config_setup
        _, _, nc_count = config_cell_syns(env, GID, "GC", syn_ids=syn_ids)
        assert nc_count == 0

    # ------------------------------------------------------------------
    # Graceful handling of missing point processes

    def test_skips_synapse_without_point_process(self, syn_manager):
        """config_cell_syns silently skips a synapse that has no point process."""
        gid = 300
        syn_ids = _init_synapses(syn_manager, gid, n=1)
        env = _make_config_env(syn_manager)
        # No point process added; should not raise
        syn_count, mech_count, nc_count = config_cell_syns(
            env, gid, "GC", syn_ids=syn_ids
        )
        assert mech_count == 0


# ===========================================================================
# Helpers for TestModifySynParamUpdateTargets
# ===========================================================================


class _NodeForModify:
    """Minimal section-node stub for modify_syn_param: only .index is needed."""

    def __init__(self, index=0):
        self.index = index


class _ModifyCellNeuron:
    """Minimal cell for modify_syn_param NEURON tests."""

    def __init__(self, gid, population_name, sec_type):
        self.gid = gid
        self.population_name = population_name
        self.nodes = {sec_type: [_NodeForModify(index=0)]}
        self.mech_dict = {}
        self.is_reduced = False


def _make_full_env(syn_manager, postsyn="GC", presyn="GC"):
    """Return an env suitable for modify_syn_param with update_targets=True.

    The same env object is stored in syn_manager.env (created externally) and
    returned here so that modify_mechanism_parameters' connection_config lookup
    and config_cell_syns' pc/celltypes access both hit the same object.
    """
    env = MockEnv(
        Populations={"GC": POP_GC, "MC": POP_MC},
        celltypes={postsyn: {"synapses": {}}},
    )
    env.SWC_Types = _FULL_SWC_TYPES
    env.layers = {"default": 0}
    env.pc = _MockPC()
    env.cache_queries = False
    # Empty mechanism dict means modify_mechanism_parameters uses the passed value directly.
    env.connection_config = {postsyn: {presyn: _MechConfig(mechanisms={"default": {}})}}
    env.synapse_manager = syn_manager
    return env


# ===========================================================================
# TestModifySynParamUpdateTargets
# ===========================================================================


class TestModifySynParamUpdateTargets:
    """Verify that modify_syn_param with update_targets=True pushes parameter
    values onto live NEURON point process and NetCon objects via config_cell_syns.

    Each fixture creates a fresh (env, syn_manager) pair where the env carries
    connection_config so that modify_mechanism_parameters can resolve the presyn
    name, and where partition_synapses_by_source can find the synapse (requires
    init_edge_attrs to be called so that source_population != -1).
    """

    @pytest.fixture
    def msp_setup(self):
        """One apical synapse with a pre-inserted h.ExpSyn.

        Uses yield to keep the local h.Section alive for the full test; NEURON
        deletes a Section when its Python wrapper is GC'd, which would detach
        the point process.
        """
        # Build env first, then syn_manager wired to that same env.
        env = MockEnv(
            Populations={"GC": POP_GC, "MC": POP_MC},
            celltypes={"GC": {"synapses": {}}},
        )
        env.SWC_Types = _FULL_SWC_TYPES
        env.layers = {"default": 0}
        env.pc = _MockPC()
        env.cache_queries = False
        env.connection_config = {"GC": {"GC": _MechConfig(mechanisms={"default": {}})}}
        syn_manager = SynapseManager(env, SYN_MECH_NAMES, SYN_PARAM_RULES)
        env.synapse_manager = syn_manager

        gid = GID
        _init_synapses(syn_manager, gid, n=1)  # sets source_population = POP_GC
        cell = _ModifyCellNeuron(gid, "GC", "apical")

        sec = make_real_section(name="msp_sec")
        pp = h.ExpSyn(sec(0.5))
        pp.tau = 1.0  # sentinel; must be overwritten
        pp.e = 0.0  # sentinel
        syn_manager.add_point_process(gid, 0, "AMPA", pp)

        yield env, cell, pp
        # sec stays in scope until here, keeping the ExpSyn attached

    @pytest.fixture
    def msp_setup_with_nc(self, msp_setup):
        """Extend msp_setup with a NetCon so weight can be tested."""
        env, cell, pp = msp_setup
        nc = h.NetCon(None, pp)
        nc.weight[0] = 0.0  # sentinel
        env.synapse_manager.add_netcon(GID, 0, "AMPA", nc)
        yield env, cell, pp, nc

    # ------------------------------------------------------------------
    # Mechanism parameters

    def test_tau_applied_to_point_process(self, msp_setup):
        env, cell, pp = msp_setup
        modify_syn_param(
            cell,
            env,
            "apical",
            "AMPA",
            param_name="tau",
            value=3.5,
            update_targets=True,
        )
        assert pytest.approx(pp.tau) == 3.5

    def test_e_applied_to_point_process(self, msp_setup):
        env, cell, pp = msp_setup
        modify_syn_param(
            cell, env, "apical", "AMPA", param_name="e", value=-7.0, update_targets=True
        )
        assert pytest.approx(pp.e) == -7.0

    # ------------------------------------------------------------------
    # NetCon parameter

    def test_netcon_weight_applied(self, msp_setup_with_nc):
        env, cell, pp, nc = msp_setup_with_nc
        modify_syn_param(
            cell,
            env,
            "apical",
            "AMPA",
            param_name="weight",
            value=0.005,
            update_targets=True,
        )
        assert pytest.approx(nc.weight[0]) == 0.005
