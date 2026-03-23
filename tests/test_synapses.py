"""Unit tests for non-NEURON-object-dependent routines in miv_simulator.synapses.

The module-level ``from neuron import h`` in synapses.py means NEURON must be
importable even here.  Run with:

    PYTHONPATH=/home/igr/bin/nrnpython3/lib/python:$PYTHONPATH \\
        /home/igr/venv/bin/pytest tests/test_synapses.py -v

These tests exercise only pure Python / NumPy functionality and do NOT create
real NEURON objects (Sections, Segments, point processes, NetCons).  Tests that
require real NEURON objects live in test_synapses_neuron.py.
"""

import numpy as np
import pytest

from miv_simulator.synapses import (
    SYNAPSE_CORE_DTYPE,
    SynapseMechanismParameterStore,
    SynapseStore,
    SynapseManager,
    SynapseView,
    _build_synapse_filter_mask,
    get_mech_rules_dict,
    get_syn_filter_dict,
    modify_syn_param,
    syn_param_from_dict,
    validate_syn_mech_param,
)
from miv_simulator.utils import AbstractEnv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYN_EXC = 0
SYN_INH = 1
SWC_SOMA = 1
SWC_APICAL = 4
LAYER_DEFAULT = -1
LAYER_1 = 1
POP_GC = 0
POP_MC = 1
GID = 42

SYN_MECH_NAMES = {"AMPA": "ExpSyn", "GABA": "Exp2Syn"}
SYN_PARAM_RULES = {
    "ExpSyn": {"mech_params": ["tau", "e"], "netcon_params": {"weight": 0}},
    "Exp2Syn": {"mech_params": ["tau1", "tau2", "e"], "netcon_params": {"weight": 0}},
}
MECH_PARAM_SPECS = {
    "ExpSyn": {"tau": 0, "e": 1, "weight": 2},
    "Exp2Syn": {"tau1": 0, "tau2": 1, "e": 2, "weight": 3},
}


# ---------------------------------------------------------------------------
# Shared helpers and mock objects
# ---------------------------------------------------------------------------


def make_core_attrs(n):
    """Build a structured NumPy array of SYNAPSE_CORE_DTYPE with n entries.

    Layout:
      syn_id           = 0 .. n-1
      syn_type         = EXC for even, INH for odd
      swc_type         = APICAL for first half, SOMA for second half
      syn_layer        = LAYER_DEFAULT for even, LAYER_1 for odd
      syn_loc          = 0.5
      syn_section      = index % 3
      source_population = GC for first half, MC for second half
      source_gid       = 100 + i
      delay            = 2.0
    """
    attrs = np.zeros(n, dtype=SYNAPSE_CORE_DTYPE)
    half = n // 2
    for i in range(n):
        attrs[i]["syn_id"] = i
        attrs[i]["syn_type"] = SYN_EXC if i % 2 == 0 else SYN_INH
        attrs[i]["swc_type"] = SWC_APICAL if i < half else SWC_SOMA
        attrs[i]["syn_layer"] = LAYER_DEFAULT if i % 2 == 0 else LAYER_1
        attrs[i]["syn_loc"] = 0.5
        attrs[i]["syn_section"] = i % 3
        attrs[i]["source_population"] = POP_GC if i < half else POP_MC
        attrs[i]["source_gid"] = 100 + i
        attrs[i]["delay"] = 2.0
    return attrs


class MockEnv(AbstractEnv):
    """Minimal environment for testing."""

    def __init__(
        self,
        Synapse_Types=None,
        SWC_Types=None,
        layers=None,
        Populations=None,
        celltypes=None,
    ):
        self.Synapse_Types = Synapse_Types or {"excitatory": 0, "inhibitory": 1}
        self.SWC_Types = SWC_Types or {"soma": 1, "apical": 4}
        self.layers = layers or {"default": -1, "Layer1": 1}
        self.Populations = Populations or {"GC": 0, "MC": 1}
        self.celltypes = celltypes or {}
        self.synapse_manager = None


class MockSynapseManagerStub:
    """Minimal stub satisfying validate_syn_mech_param requirements."""

    def __init__(self):
        self.syn_mech_names = {"AMPA": "ExpSyn", "GABA": "Exp2Syn"}
        self.syn_param_rules = SYN_PARAM_RULES


class MockCell:
    """Minimal cell mock for get_mech_rules_dict."""

    def __init__(self, nodes=None):
        self.nodes = nodes if nodes is not None else {"soma": [object()], "apical": []}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_env():
    return MockEnv()


@pytest.fixture
def mock_env_with_stub(mock_env):
    mock_env.synapse_manager = MockSynapseManagerStub()
    return mock_env


@pytest.fixture
def param_store():
    return SynapseMechanismParameterStore(MECH_PARAM_SPECS)


@pytest.fixture
def syn_store():
    return SynapseStore(MECH_PARAM_SPECS, initial_size=20)


@pytest.fixture
def syn_manager(mock_env):
    return SynapseManager(mock_env, SYN_MECH_NAMES, SYN_PARAM_RULES)


@pytest.fixture
def populated_manager(syn_manager):
    """SynapseManager with GID=42 loaded with 12 mixed synapses and edges."""
    n = 12
    syn_ids = np.arange(n, dtype=np.uint32)
    syn_layers = np.array(
        [LAYER_DEFAULT if i % 2 == 0 else LAYER_1 for i in range(n)], dtype=np.int8
    )
    syn_types = np.array(
        [SYN_EXC if i % 2 == 0 else SYN_INH for i in range(n)], dtype=np.uint8
    )
    swc_types = np.array(
        [SWC_APICAL if i < 6 else SWC_SOMA for i in range(n)], dtype=np.uint8
    )
    syn_secs = (np.arange(n) % 3).astype(np.uint16)
    syn_locs = np.full(n, 0.5, dtype=np.float32)
    syn_manager.init_syn_id_attrs(
        GID, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs
    )
    # Provide explicit delays so h.dt is not required
    syn_manager.init_edge_attrs(
        GID, "GC", np.arange(6, dtype=np.int32), syn_ids[:6], delays=[2.0] * 6
    )
    syn_manager.init_edge_attrs(
        GID, "MC", np.arange(6, 12, dtype=np.int32), syn_ids[6:], delays=[3.0] * 6
    )
    return syn_manager


# ===========================================================================
# TestSynParamFromDict
# ===========================================================================


class TestSynParamFromDict:
    def test_all_fields_populated(self):
        d = {
            "population": "GC",
            "source": "MC",
            "sec_type": "apical",
            "syn_name": "AMPA",
            "param_path": ["weight"],
            "param_range": (0.0, 1.0),
            "phenotype": None,
        }
        sp = syn_param_from_dict(d)
        assert sp.population == "GC"
        assert sp.source == "MC"
        assert sp.sec_type == "apical"
        assert sp.syn_name == "AMPA"
        assert sp.param_path == ["weight"]
        assert sp.param_range == (0.0, 1.0)
        assert sp.phenotype is None

    def test_missing_key_raises(self):
        d = {"population": "GC"}  # missing all other required keys
        with pytest.raises(KeyError):
            syn_param_from_dict(d)

    def test_extra_keys_ignored(self):
        d = {
            "population": "GC",
            "source": "MC",
            "sec_type": "apical",
            "syn_name": "AMPA",
            "param_path": None,
            "param_range": None,
            "phenotype": None,
            "extra_key": "should_be_ignored",
        }
        sp = syn_param_from_dict(d)
        assert sp.population == "GC"
        assert not hasattr(sp, "extra_key")


# ===========================================================================
# TestGetMechRulesDict
# ===========================================================================


class TestGetMechRulesDict:
    def test_value_only(self):
        cell = MockCell()
        result = get_mech_rules_dict(cell, value=1.0)
        assert result == {"value": 1.0}

    def test_origin_valid_section(self):
        cell = MockCell(nodes={"soma": [object()], "apical": []})
        result = get_mech_rules_dict(cell, origin="soma")
        assert result == {"origin": "soma"}

    def test_origin_empty_section_raises(self):
        cell = MockCell(nodes={"soma": [object()], "apical": []})
        with pytest.raises(ValueError, match="invalid origin type"):
            get_mech_rules_dict(cell, origin="apical")

    def test_origin_parent_always_valid(self):
        cell = MockCell(nodes={"soma": [], "apical": []})
        result = get_mech_rules_dict(cell, origin="parent")
        assert result == {"origin": "parent"}

    def test_origin_branch_origin_always_valid(self):
        cell = MockCell(nodes={"soma": [], "apical": []})
        result = get_mech_rules_dict(cell, origin="branch_origin")
        assert result == {"origin": "branch_origin"}

    def test_origin_invalid_raises(self):
        cell = MockCell(nodes={"soma": [object()]})
        with pytest.raises(ValueError, match="invalid origin type"):
            get_mech_rules_dict(cell, origin="not_a_section_type")

    def test_none_values_excluded(self):
        cell = MockCell()
        result = get_mech_rules_dict(cell, value=None, origin=None)
        assert result == {}

    def test_both_value_and_origin(self):
        cell = MockCell(nodes={"soma": [object()], "apical": []})
        result = get_mech_rules_dict(cell, value=5.0, origin="soma")
        assert result == {"value": 5.0, "origin": "soma"}


# ===========================================================================
# TestBuildSynapseFilterMask
# ===========================================================================


class TestBuildSynapseFilterMask:
    """Uses make_core_attrs(10) for all tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = 10
        self.attrs = make_core_attrs(self.n)

    def _mask(self, **kwargs):
        return _build_synapse_filter_mask(self.attrs, self.n, **kwargs)

    def test_no_filters_selects_all(self):
        mask = self._mask()
        assert mask.sum() == self.n
        assert mask.all()

    def test_syn_sections_filter(self):
        mask = self._mask(syn_sections=[0])
        # section = i % 3; indices 0, 3, 6, 9 have section 0
        expected = np.array([i for i in range(self.n) if i % 3 == 0])
        assert np.array_equal(np.where(mask)[0], expected)

    def test_syn_indexes_filter(self):
        mask = self._mask(syn_indexes=[2, 5])
        assert mask.sum() == 2
        assert mask[2] and mask[5]
        assert not mask[0] and not mask[1]

    def test_syn_types_excitatory_only(self):
        mask = self._mask(syn_types=[SYN_EXC])
        # even indices are excitatory
        assert all(self.attrs["syn_type"][i] == SYN_EXC for i in np.where(mask)[0])
        assert mask.sum() == sum(1 for i in range(self.n) if i % 2 == 0)

    def test_layers_filter(self):
        mask = self._mask(layers=[LAYER_1])
        # odd indices have LAYER_1
        assert all(self.attrs["syn_layer"][i] == LAYER_1 for i in np.where(mask)[0])

    def test_sources_filter(self):
        half = self.n // 2
        mask = self._mask(sources=[POP_GC])
        assert mask.sum() == half
        assert np.where(mask)[0].tolist() == list(range(half))

    def test_swc_types_filter(self):
        half = self.n // 2
        mask = self._mask(swc_types=[SWC_SOMA])
        assert mask.sum() == self.n - half
        assert all(i >= half for i in np.where(mask)[0])

    def test_combined_syn_type_and_layer(self):
        mask = self._mask(syn_types=[SYN_EXC], layers=[LAYER_DEFAULT])
        # EXC = even indices; LAYER_DEFAULT = even indices; intersection = even indices
        result_indices = np.where(mask)[0]
        for i in result_indices:
            assert self.attrs["syn_type"][i] == SYN_EXC
            assert self.attrs["syn_layer"][i] == LAYER_DEFAULT

    def test_empty_result_when_no_match(self):
        mask = self._mask(syn_types=[99])
        assert mask.sum() == 0

    def test_duplicate_values_deduplicated(self):
        mask_dup = self._mask(syn_sections=[0, 0, 0])
        mask_single = self._mask(syn_sections=[0])
        assert np.array_equal(mask_dup, mask_single)

    def test_valid_count_limits_scope(self):
        # Only examine first 3 entries
        mask = _build_synapse_filter_mask(self.attrs, 3)
        assert mask.sum() == 3
        assert len(mask) == 3


# ===========================================================================
# TestSynapseMechanismParameterStoreBasics
# ===========================================================================


class TestSynapseMechanismParameterStoreBasics:
    def test_ensure_gid_storage_creates_entry(self, param_store):
        param_store._ensure_gid_storage(gid=1)
        assert 1 in param_store.gid_data
        data = param_store.gid_data[1]
        assert "arrays" in data
        assert "has_mech" in data
        assert "synapse_count" in data

    def test_ensure_gid_storage_idempotent(self, param_store):
        param_store._ensure_gid_storage(gid=1, synapse_count=5)
        param_store.gid_data[1]["synapse_count"] = 99  # mutate
        param_store._ensure_gid_storage(gid=1)  # second call
        # Original data should not be reset
        assert param_store.gid_data[1]["synapse_count"] == 99

    def test_ensure_mech_storage_creates_has_mech_array(self, param_store):
        param_store._ensure_mech_storage(1, "ExpSyn")
        assert "ExpSyn" in param_store.gid_data[1]["has_mech"]
        arr = param_store.gid_data[1]["has_mech"]["ExpSyn"]
        assert arr.dtype == bool

    def test_ensure_param_array_fills_with_nan(self, param_store):
        param_store._ensure_param_array(1, "ExpSyn", "tau")
        arr = param_store.gid_data[1]["arrays"]["ExpSyn"]["tau"]
        assert np.all(np.isnan(arr))

    def test_ensure_param_array_idempotent(self, param_store):
        param_store._ensure_param_array(1, "ExpSyn", "tau")
        arr_id = id(param_store.gid_data[1]["arrays"]["ExpSyn"]["tau"])
        param_store._ensure_param_array(1, "ExpSyn", "tau")  # second call
        assert id(param_store.gid_data[1]["arrays"]["ExpSyn"]["tau"]) == arr_id

    def test_has_mechanism_false_before_set(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=5)
        assert not param_store.has_mechanism(1, 0, "ExpSyn")

    def test_has_mechanism_true_after_set(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=5)
        param_store.set_synapse_parameter(1, 0, "ExpSyn", "tau", 3.0)
        assert param_store.has_mechanism(1, 0, "ExpSyn")


# ===========================================================================
# TestSynapseMechanismParameterStoreSetGet
# ===========================================================================


class TestSynapseMechanismParameterStoreSetGet:
    def test_set_and_get_numeric_value(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=5)
        result = param_store.set_synapse_parameter(1, 0, "ExpSyn", "tau", 5.0)
        assert result is True
        val = param_store.get_synapse_parameter(1, 0, "ExpSyn", "tau")
        assert pytest.approx(val) == 5.0

    def test_set_and_get_complex_value(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=5)
        complex_val = [1.0, 2.0, 3.0]
        param_store.set_synapse_parameter(1, 0, "ExpSyn", "tau", complex_val)
        # Mark mechanism
        result_val = param_store.get_synapse_parameter(1, 0, "ExpSyn", "tau")
        assert result_val == complex_val

    def test_unknown_mechanism_returns_false(self, param_store):
        result = param_store.set_synapse_parameter(1, 0, "NoSuchMech", "tau", 1.0)
        assert result is False

    def test_unknown_param_returns_false(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=5)
        result = param_store.set_synapse_parameter(1, 0, "ExpSyn", "no_such_param", 1.0)
        assert result is False

    def test_resize_triggered_on_out_of_bounds_index(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=5)
        # Index 100 is beyond initial size
        param_store.set_synapse_parameter(1, 100, "ExpSyn", "tau", 7.5)
        val = param_store.get_synapse_parameter(1, 100, "ExpSyn", "tau")
        assert pytest.approx(val) == 7.5

    def test_resize_preserves_existing_values(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=5)
        param_store.set_synapse_parameter(1, 0, "ExpSyn", "tau", 3.0)
        # Trigger resize with a high index
        param_store.set_synapse_parameter(1, 200, "ExpSyn", "tau", 9.0)
        # Original value at index 0 must still be readable
        val0 = param_store.get_synapse_parameter(1, 0, "ExpSyn", "tau")
        assert pytest.approx(val0) == 3.0


# ===========================================================================
# TestSynapseMechanismParameterStoreDefaults
# ===========================================================================


class TestSynapseMechanismParameterStoreDefaults:
    def test_global_default_none_selector(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        param_store.set_default_value(1, "ExpSyn", "tau", 4.0, synapse_selector=None)
        val = param_store.get_default_value(1, "ExpSyn", "tau", syn_id=0, syn_index=0)
        assert pytest.approx(val) == 4.0
        val2 = param_store.get_default_value(1, "ExpSyn", "tau", syn_id=7, syn_index=7)
        assert pytest.approx(val2) == 4.0

    def test_list_selector_applies_only_to_listed_ids(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        param_store.set_default_value(1, "ExpSyn", "tau", 2.0, synapse_selector=[0, 1])
        assert (
            pytest.approx(param_store.get_default_value(1, "ExpSyn", "tau", 0, 0))
            == 2.0
        )
        assert (
            pytest.approx(param_store.get_default_value(1, "ExpSyn", "tau", 1, 1))
            == 2.0
        )
        assert param_store.get_default_value(1, "ExpSyn", "tau", 2, 2) is None

    def test_callable_selector(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        # The _create_selector wrapper calls selector_spec(syn_id) with 1 arg
        param_store.set_default_value(
            1, "ExpSyn", "tau", 6.0, synapse_selector=lambda sid: sid % 2 == 0
        )
        assert (
            pytest.approx(param_store.get_default_value(1, "ExpSyn", "tau", 0, 0))
            == 6.0
        )
        assert param_store.get_default_value(1, "ExpSyn", "tau", 1, 1) is None

    def test_most_recent_default_takes_precedence(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        param_store.set_default_value(1, "ExpSyn", "tau", 1.0, synapse_selector=None)
        param_store.set_default_value(1, "ExpSyn", "tau", 9.0, synapse_selector=None)
        val = param_store.get_default_value(1, "ExpSyn", "tau", 0, 0)
        assert pytest.approx(val) == 9.0

    def test_has_default_params_true_when_matching(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        param_store.set_default_value(1, "ExpSyn", "tau", 3.0, synapse_selector=None)
        assert param_store.has_default_params(
            1, syn_id=0, syn_index=0, mech_name="ExpSyn"
        )

    def test_has_default_params_false_when_none_match(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        param_store.set_default_value(1, "ExpSyn", "tau", 3.0, synapse_selector=[5, 6])
        assert not param_store.has_default_params(
            1, syn_id=0, syn_index=0, mech_name="ExpSyn"
        )

    def test_get_default_params_returns_correct_set(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        param_store.set_default_value(1, "ExpSyn", "tau", 2.0, synapse_selector=None)
        param_store.set_default_value(1, "ExpSyn", "e", -70.0, synapse_selector=None)
        names = param_store.get_default_params(
            1, syn_id=0, syn_index=0, mech_name="ExpSyn"
        )
        assert "tau" in names
        assert "e" in names

    def test_hierarchy_specific_overrides_default(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        param_store.set_default_value(1, "ExpSyn", "tau", 1.0, synapse_selector=None)
        param_store.set_synapse_parameter(1, 0, "ExpSyn", "tau", 99.0)
        val, source = param_store.get_parameter_value_hierarchy(
            1, 0, "ExpSyn", "tau", syn_id=0
        )
        assert pytest.approx(val) == 99.0
        assert source == "specific"

    def test_hierarchy_fallback_to_default(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        param_store.set_default_value(1, "ExpSyn", "tau", 5.0, synapse_selector=None)
        val, source = param_store.get_parameter_value_hierarchy(
            1, 0, "ExpSyn", "tau", syn_id=0
        )
        assert pytest.approx(val) == 5.0
        assert source == "default"

    def test_hierarchy_no_value_returns_none(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        val, source = param_store.get_parameter_value_hierarchy(
            1, 0, "ExpSyn", "tau", syn_id=0
        )
        assert val is None
        assert source is None

    def test_set_default_unknown_mech_returns_false(self, param_store):
        result = param_store.set_default_value(1, "NoSuchMech", "tau", 1.0)
        assert result is False

    def test_complex_value_stored_and_retrieved(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=10)
        obj = {"nested": [1, 2, 3]}
        param_store.set_default_value(1, "ExpSyn", "tau", obj, synapse_selector=None)
        val = param_store.get_default_value(1, "ExpSyn", "tau", 0, 0)
        assert val == obj


# ===========================================================================
# TestResizeArrays
# ===========================================================================


class TestResizeArrays:
    def test_resize_extends_has_mech_array(self, param_store):
        param_store._ensure_mech_storage(1, "ExpSyn")
        original_len = len(param_store.gid_data[1]["has_mech"]["ExpSyn"])
        param_store._resize_arrays(1, original_len + 50)
        new_len = len(param_store.gid_data[1]["has_mech"]["ExpSyn"])
        assert new_len >= original_len + 50

    def test_resize_extends_param_arrays_with_nan(self, param_store):
        param_store._ensure_param_array(1, "ExpSyn", "tau")
        old_size = len(param_store.gid_data[1]["arrays"]["ExpSyn"]["tau"])
        new_size = old_size + 100
        param_store._resize_arrays(1, new_size)
        arr = param_store.gid_data[1]["arrays"]["ExpSyn"]["tau"]
        assert len(arr) >= new_size
        assert np.all(np.isnan(arr[old_size:new_size]))

    def test_resize_preserves_existing_bool_flags(self, param_store):
        param_store._ensure_gid_storage(1, synapse_count=5)
        param_store.set_synapse_parameter(1, 0, "ExpSyn", "tau", 1.0)
        assert param_store.gid_data[1]["has_mech"]["ExpSyn"][0]
        old_size = len(param_store.gid_data[1]["has_mech"]["ExpSyn"])
        param_store._resize_arrays(1, old_size + 50)
        assert param_store.gid_data[1]["has_mech"]["ExpSyn"][0]

    def test_resize_noop_when_new_size_smaller(self, param_store):
        param_store._ensure_mech_storage(1, "ExpSyn")
        old_len = len(param_store.gid_data[1]["has_mech"]["ExpSyn"])
        param_store._resize_arrays(1, old_len - 1)
        assert len(param_store.gid_data[1]["has_mech"]["ExpSyn"]) == old_len

    def test_gid_not_present_is_noop(self, param_store):
        param_store._resize_arrays(999, 100)  # must not raise


# ===========================================================================
# TestCreateSelector
# ===========================================================================


class TestCreateSelector:
    def test_none_selector_matches_all(self, param_store):
        sel = param_store._create_selector(gid=1, selector_spec=None)
        assert sel(0, 0) is True
        assert sel(999, 999) is True

    def test_list_selector_matches_only_listed_ids(self, param_store):
        sel = param_store._create_selector(gid=1, selector_spec=[1, 3, 5])
        assert sel(1, 0) is True
        assert sel(3, 0) is True
        assert sel(2, 0) is False

    def test_ndarray_selector(self, param_store):
        sel = param_store._create_selector(gid=1, selector_spec=np.array([10, 20]))
        assert sel(10, 0) is True
        assert sel(15, 0) is False

    def test_callable_selector_delegates(self, param_store):
        # The wrapper calls selector_spec(syn_id) with a single positional arg,
        # so the user-supplied callable must accept exactly one argument.
        sel = param_store._create_selector(gid=1, selector_spec=lambda sid: sid > 10)
        assert sel(11, 0) is True
        assert sel(10, 0) is False


# ===========================================================================
# TestSynapseStore
# ===========================================================================


class TestSynapseStore:
    def _add(self, store, gid=GID, n=5):
        syn_ids = np.arange(n, dtype=np.uint32)
        syn_layers = np.zeros(n, dtype=np.int8)
        syn_types = np.array(
            [SYN_EXC if i % 2 == 0 else SYN_INH for i in range(n)], dtype=np.uint8
        )
        swc_types = np.full(n, SWC_APICAL, dtype=np.uint8)
        syn_secs = (np.arange(n) % 3).astype(np.uint16)
        syn_locs = np.full(n, 0.5, dtype=np.float32)
        store.add_synapses_from_arrays(
            gid, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs
        )
        return syn_ids

    def test_add_synapses_from_arrays_stores_core_fields(self, syn_store):
        syn_ids = self._add(syn_store)
        view = syn_store.get_synapse(GID, syn_ids[2])
        assert view is not None
        assert int(view.syn_type) == SYN_EXC

    def test_add_synapses_initializes_source_defaults(self, syn_store):
        syn_ids = self._add(syn_store)
        view = syn_store.get_synapse(GID, syn_ids[0])
        assert int(view.source.gid) == -1
        assert int(view.source.population) == -1
        assert float(view.source.delay) == 0.0

    def test_get_synapse_returns_none_for_unknown_gid(self, syn_store):
        assert syn_store.get_synapse(999, 0) is None

    def test_get_synapse_returns_none_for_unknown_syn_id(self, syn_store):
        self._add(syn_store)
        assert syn_store.get_synapse(GID, 9999) is None

    def test_get_synapse_view_fields_match_stored(self, syn_store):
        n = 6
        syn_ids = np.arange(n, dtype=np.uint32)
        syn_layers = np.array(
            [LAYER_DEFAULT if i % 2 == 0 else LAYER_1 for i in range(n)], dtype=np.int8
        )
        syn_types = np.array(
            [SYN_EXC if i % 2 == 0 else SYN_INH for i in range(n)], dtype=np.uint8
        )
        swc_types = np.array(
            [SWC_APICAL if i < 3 else SWC_SOMA for i in range(n)], dtype=np.uint8
        )
        syn_secs = (np.arange(n) % 3).astype(np.uint16)
        syn_locs = np.full(n, 0.75, dtype=np.float32)
        syn_store.add_synapses_from_arrays(
            GID, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs
        )
        for i in range(n):
            view = syn_store.get_synapse(GID, i)
            assert int(view.syn_type) == (SYN_EXC if i % 2 == 0 else SYN_INH)
            assert int(view.swc_type) == (SWC_APICAL if i < 3 else SWC_SOMA)
            assert int(view.syn_layer) == (LAYER_DEFAULT if i % 2 == 0 else LAYER_1)
            assert pytest.approx(float(view.syn_loc)) == 0.75

    def test_get_synapses_by_filter_single_field(self, syn_store):
        self._add(syn_store, n=6)
        results = syn_store.get_synapses_by_filter(GID, syn_types=[SYN_EXC])
        for view in results:
            assert int(view.syn_type) == SYN_EXC

    def test_get_synapses_by_filter_multiple_values(self, syn_store):
        self._add(syn_store, n=6)
        results = syn_store.get_synapses_by_filter(GID, syn_types=[SYN_EXC, SYN_INH])
        assert len(results) == 6

    def test_get_synapses_by_filter_empty_result(self, syn_store):
        self._add(syn_store, n=4)
        results = syn_store.get_synapses_by_filter(GID, syn_types=[99])
        assert len(results) == 0

    def test_keys_yields_all_syn_ids(self, syn_store):
        n = 5
        syn_ids = self._add(syn_store, n=n)
        assert set(syn_store.keys(GID)) == set(int(s) for s in syn_ids)

    def test_items_yields_all_synapse_views(self, syn_store):
        n = 4
        syn_ids = self._add(syn_store, n=n)
        pairs = list(syn_store.items(GID))
        assert len(pairs) == n
        yielded_ids = {s for s, _ in pairs}
        assert yielded_ids == set(int(sid) for sid in syn_ids)
        for _, view in pairs:
            assert isinstance(view, SynapseView)

    def test_keys_empty_for_unknown_gid(self, syn_store):
        assert list(syn_store.keys(999)) == []

    def test_ensure_capacity_grows_on_overflow(self, syn_store):
        # Add more synapses than initial_size=20 in two batches
        n1 = 15
        n2 = 15  # total 30 > 20
        ids1 = np.arange(n1, dtype=np.uint32)
        ids2 = np.arange(n1, n1 + n2, dtype=np.uint32)

        def add(ids):
            syn_store.add_synapses_from_arrays(
                GID,
                ids,
                np.zeros(len(ids), dtype=np.int8),
                np.zeros(len(ids), dtype=np.uint8),
                np.zeros(len(ids), dtype=np.uint8),
                np.zeros(len(ids), dtype=np.uint16),
                np.full(len(ids), 0.5, dtype=np.float32),
            )

        add(ids1)
        add(ids2)
        for sid in list(ids1) + list(ids2):
            assert syn_store.get_synapse(GID, sid) is not None


# ===========================================================================
# TestSynapseView
# ===========================================================================


class TestSynapseView:
    @pytest.fixture
    def view_and_store(self):
        store = SynapseStore(MECH_PARAM_SPECS, initial_size=10)
        syn_ids = np.array([7], dtype=np.uint32)
        syn_layers = np.array([LAYER_1], dtype=np.int8)
        syn_types = np.array([SYN_INH], dtype=np.uint8)
        swc_types = np.array([SWC_SOMA], dtype=np.uint8)
        syn_secs = np.array([2], dtype=np.uint16)
        syn_locs = np.array([0.3], dtype=np.float32)
        store.add_synapses_from_arrays(
            GID, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs
        )
        # Set source info
        store.attrs[GID]["source_gid"][0] = 55
        store.attrs[GID]["source_population"][0] = POP_MC
        store.attrs[GID]["delay"][0] = 1.5
        view = store.get_synapse(GID, 7)
        return view, store

    def test_property_reads_correct_values(self, view_and_store):
        view, _ = view_and_store
        assert int(view.syn_type) == SYN_INH
        assert int(view.swc_type) == SWC_SOMA
        assert int(view.syn_layer) == LAYER_1
        assert pytest.approx(float(view.syn_loc)) == 0.3
        assert int(view.syn_section) == 2

    def test_property_setter_mutates_backing_array(self, view_and_store):
        view, store = view_and_store
        view.syn_type = SYN_EXC
        assert int(store.attrs[GID]["syn_type"][0]) == SYN_EXC

    def test_source_view_reads_gid_population_delay(self, view_and_store):
        view, _ = view_and_store
        assert int(view.source.gid) == 55
        assert int(view.source.population) == POP_MC
        assert pytest.approx(float(view.source.delay)) == 1.5

    def test_source_view_setter_persists(self, view_and_store):
        view, store = view_and_store
        view.source.delay = 9.9
        assert pytest.approx(float(store.attrs[GID]["delay"][0])) == 9.9


# ===========================================================================
# TestGetSynFilterDict
# ===========================================================================


class TestGetSynFilterDict:
    def test_valid_syn_types_no_convert(self, mock_env):
        rules = {"syn_types": ["excitatory"]}
        result = get_syn_filter_dict(mock_env, rules, convert=False)
        assert result["syn_types"] == ["excitatory"]

    def test_valid_syn_types_with_convert(self, mock_env):
        rules = {"syn_types": ["excitatory"]}
        result = get_syn_filter_dict(mock_env, rules, convert=True)
        assert result["syn_types"] == [0]

    def test_unknown_syn_type_raises(self, mock_env):
        with pytest.raises(ValueError, match="syn_type"):
            get_syn_filter_dict(mock_env, {"syn_types": ["unknown_type"]})

    def test_valid_swc_type_with_convert(self, mock_env):
        result = get_syn_filter_dict(mock_env, {"swc_types": ["apical"]}, convert=True)
        assert result["swc_types"] == [4]

    def test_unknown_swc_type_raises(self, mock_env):
        with pytest.raises(ValueError, match="swc_type"):
            get_syn_filter_dict(mock_env, {"swc_types": ["dendrite_xyz"]})

    def test_valid_layer_with_convert(self, mock_env):
        result = get_syn_filter_dict(mock_env, {"layers": ["Layer1"]}, convert=True)
        assert result["layers"] == [1]

    def test_unknown_layer_raises(self, mock_env):
        with pytest.raises(ValueError, match="layer"):
            get_syn_filter_dict(mock_env, {"layers": ["Layer99"]})

    def test_valid_source_with_convert(self, mock_env):
        result = get_syn_filter_dict(mock_env, {"sources": ["GC"]}, convert=True)
        assert result["sources"] == [0]

    def test_unknown_source_raises(self, mock_env):
        with pytest.raises(ValueError, match="population"):
            get_syn_filter_dict(mock_env, {"sources": ["UnknownPop"]})

    def test_unrecognized_filter_name_raises_by_default(self, mock_env):
        with pytest.raises(ValueError, match="unrecognized filter"):
            get_syn_filter_dict(mock_env, {"invalid_key": ["x"]})

    def test_unrecognized_filter_name_allowed_when_check_false(self, mock_env):
        result = get_syn_filter_dict(
            mock_env, {"invalid_key": ["x"]}, check_valid=False
        )
        assert "invalid_key" in result

    def test_input_deep_copied(self, mock_env):
        rules = {"syn_types": ["excitatory"]}
        _ = get_syn_filter_dict(mock_env, rules, convert=True)
        # Original must not be mutated
        assert rules["syn_types"] == ["excitatory"]

    def test_empty_rules_returns_empty_dict(self, mock_env):
        assert get_syn_filter_dict(mock_env, {}) == {}


# ===========================================================================
# TestValidateSynMechParam
# ===========================================================================


class TestValidateSynMechParam:
    def test_valid_mech_param_returns_true(self, mock_env_with_stub):
        assert validate_syn_mech_param(mock_env_with_stub, "AMPA", "tau") is True

    def test_valid_netcon_param_returns_true(self, mock_env_with_stub):
        assert validate_syn_mech_param(mock_env_with_stub, "AMPA", "weight") is True

    def test_unknown_syn_name_returns_false(self, mock_env_with_stub):
        assert validate_syn_mech_param(mock_env_with_stub, "NMDA", "tau") is False

    def test_unknown_param_name_returns_false(self, mock_env_with_stub):
        assert validate_syn_mech_param(mock_env_with_stub, "AMPA", "gmax") is False


# ===========================================================================
# TestSynapseManagerInitSynIdAttrs
# ===========================================================================


class TestSynapseManagerInitSynIdAttrs:
    def _make_arrays(self, n=6):
        syn_ids = np.arange(n, dtype=np.uint32)
        syn_layers = np.zeros(n, dtype=np.int8)
        syn_types = np.array(
            [SYN_EXC if i % 2 == 0 else SYN_INH for i in range(n)], dtype=np.uint8
        )
        swc_types = np.full(n, SWC_APICAL, dtype=np.uint8)
        syn_secs = (np.arange(n) % 3).astype(np.uint16)
        syn_locs = np.full(n, 0.5, dtype=np.float32)
        return syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs

    def test_init_stores_all_synapses(self, syn_manager):
        arrays = self._make_arrays(6)
        syn_ids = arrays[0]
        syn_manager.init_syn_id_attrs(GID, *arrays)
        for sid in syn_ids:
            view = syn_manager.get_synapse(GID, int(sid))
            assert view is not None

    def test_duplicate_gid_raises(self, syn_manager):
        arrays = self._make_arrays(4)
        syn_manager.init_syn_id_attrs(GID, *arrays)
        with pytest.raises(RuntimeError, match="exists"):
            syn_manager.init_syn_id_attrs(GID, *arrays)

    def test_empty_syn_ids_is_noop(self, syn_manager):
        empty = np.array([], dtype=np.uint32)
        syn_manager.init_syn_id_attrs(
            GID,
            empty,
            empty.astype(np.int8),
            empty.astype(np.uint8),
            empty.astype(np.uint8),
            empty.astype(np.uint16),
            empty.astype(np.float32),
        )
        # GID should not be present in storage
        assert GID not in syn_manager.syn_store.attrs

    def test_init_from_iter_dict_mode(self, syn_manager):
        arrays = self._make_arrays(4)
        syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs = arrays
        attr_dict = {
            "syn_ids": syn_ids,
            "syn_layers": syn_layers,
            "syn_types": syn_types,
            "swc_types": swc_types,
            "syn_secs": syn_secs,
            "syn_locs": syn_locs,
        }
        syn_manager.init_syn_id_attrs_from_iter([(GID, attr_dict)], attr_type="dict")
        for sid in syn_ids:
            assert syn_manager.get_synapse(GID, int(sid)) is not None


# ===========================================================================
# TestSynapseManagerInitEdgeAttrs
# ===========================================================================


class TestSynapseManagerInitEdgeAttrs:
    def _init_synapses(self, manager, n=6):
        syn_ids = np.arange(n, dtype=np.uint32)
        manager.init_syn_id_attrs(
            GID,
            syn_ids,
            np.zeros(n, dtype=np.int8),
            np.zeros(n, dtype=np.uint8),
            np.full(n, SWC_APICAL, dtype=np.uint8),
            np.zeros(n, dtype=np.uint16),
            np.full(n, 0.5, dtype=np.float32),
        )
        return syn_ids

    def test_sets_source_population(self, syn_manager):
        syn_ids = self._init_synapses(syn_manager, n=4)
        presyn_gids = np.arange(4, dtype=np.int32)
        syn_manager.init_edge_attrs(GID, "GC", presyn_gids, syn_ids, delays=[2.0] * 4)
        for i, sid in enumerate(syn_ids):
            view = syn_manager.get_synapse(GID, int(sid))
            assert int(view.source.population) == POP_GC
            assert int(view.source.gid) == i

    def test_sets_explicit_delays(self, syn_manager):
        syn_ids = self._init_synapses(syn_manager, n=3)
        delays = [1.0, 2.0, 3.0]
        syn_manager.init_edge_attrs(
            GID, "GC", np.arange(3, dtype=np.int32), syn_ids, delays=delays
        )
        for i, sid in enumerate(syn_ids):
            view = syn_manager.get_synapse(GID, int(sid))
            assert pytest.approx(float(view.source.delay)) == delays[i]

    def test_uninitialized_gid_raises(self, syn_manager):
        with pytest.raises(RuntimeError, match="not been initialized"):
            syn_manager.init_edge_attrs(
                999,
                "GC",
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.uint32),
                delays=[2.0],
            )

    def test_unknown_syn_id_raises(self, syn_manager):
        self._init_synapses(syn_manager, n=4)
        with pytest.raises(RuntimeError, match="not been initialized"):
            syn_manager.init_edge_attrs(
                GID,
                "GC",
                np.array([0], dtype=np.int32),
                np.array([999], dtype=np.uint32),  # non-existent syn_id
                delays=[2.0],
            )


# ===========================================================================
# TestSynapseManagerFilterSynapses
# ===========================================================================


class TestSynapseManagerFilterSynapses:
    def test_filter_by_syn_type(self, populated_manager):
        results = list(populated_manager.filter_synapses(GID, syn_types=[SYN_EXC]))
        assert len(results) == 6  # half of 12 are EXC
        for _, view in results:
            assert int(view.syn_type) == SYN_EXC

    def test_filter_by_layer(self, populated_manager):
        results = list(populated_manager.filter_synapses(GID, layers=[LAYER_1]))
        assert len(results) == 6
        for _, view in results:
            assert int(view.syn_layer) == LAYER_1

    def test_filter_by_sources(self, populated_manager):
        results = list(populated_manager.filter_synapses(GID, sources=[POP_GC]))
        assert len(results) == 6
        for _, view in results:
            assert int(view.source.population) == POP_GC

    def test_filter_by_swc_type(self, populated_manager):
        results = list(populated_manager.filter_synapses(GID, swc_types=[SWC_APICAL]))
        assert len(results) == 6

    def test_filter_combined_criteria(self, populated_manager):
        results = list(
            populated_manager.filter_synapses(
                GID, syn_types=[SYN_EXC], layers=[LAYER_DEFAULT]
            )
        )
        for _, view in results:
            assert int(view.syn_type) == SYN_EXC
            assert int(view.syn_layer) == LAYER_DEFAULT

    def test_filter_empty_gid_returns_empty(self, syn_manager):
        results = list(syn_manager.filter_synapses(999))
        assert results == []

    def test_filter_no_filters_returns_all(self, populated_manager):
        results = list(populated_manager.filter_synapses(GID))
        assert len(results) == 12

    def test_filter_synapse_ids_dtype(self, populated_manager):
        ids = populated_manager.filter_synapse_ids(GID, syn_types=[SYN_EXC])
        assert ids.dtype == np.uint32

    def test_filter_synapse_ids_no_filters_returns_all(self, populated_manager):
        ids = populated_manager.filter_synapse_ids(GID)
        assert len(ids) == 12

    def test_cache_returns_same_result(self, populated_manager):
        r1 = populated_manager.filter_synapses(GID, syn_types=[SYN_EXC], cache=True)
        r2 = populated_manager.filter_synapses(GID, syn_types=[SYN_EXC], cache=True)
        ids1 = {sid for sid, _ in r1}
        ids2 = {sid for sid, _ in r2}
        assert ids1 == ids2


# ===========================================================================
# TestSynapseManagerPartitionBySource
# ===========================================================================


class TestSynapseManagerPartitionBySource:
    def test_all_populations_present_in_result(self, populated_manager):
        result = populated_manager.partition_synapses_by_source(GID)
        assert "GC" in result
        assert "MC" in result

    def test_gc_yields_correct_syn_ids(self, populated_manager):
        result = populated_manager.partition_synapses_by_source(GID)
        gc_pairs = list(result["GC"])
        assert len(gc_pairs) == 6
        for _, view in gc_pairs:
            assert int(view.source.population) == POP_GC

    def test_mc_yields_correct_syn_ids(self, populated_manager):
        result = populated_manager.partition_synapses_by_source(GID)
        mc_pairs = list(result["MC"])
        assert len(mc_pairs) == 6
        for _, view in mc_pairs:
            assert int(view.source.population) == POP_MC

    def test_partition_with_explicit_syn_ids(self, populated_manager):
        # Only partition the first 4 synapses (all from GC)
        subset = list(range(4))
        result = populated_manager.partition_synapses_by_source(GID, syn_ids=subset)
        gc_pairs = list(result["GC"])
        # generator_ifempty returns None (not an empty iterator) for empty populations
        assert len(gc_pairs) == 4
        assert result["MC"] is None

    def test_unknown_gid_returns_empty_iterators(self, syn_manager):
        result = syn_manager.partition_synapses_by_source(999)
        for pop_name in result:
            assert list(result[pop_name]) == []


# ===========================================================================
# TestSynapseManagerPointProcessStorage
# ===========================================================================


class TestSynapseManagerPointProcessStorage:
    """Tests the dict-layer add/has/get/del operations.

    Plain Python objects are used as stand-ins; no real NEURON objects are
    created or required.
    """

    def test_add_and_has_point_process(self, syn_manager):
        fake_pp = object()
        syn_manager.add_point_process(GID, syn_id=0, syn_name="AMPA", pps=fake_pp)
        assert syn_manager.has_point_process(GID, syn_id=0, syn_name="AMPA")

    def test_get_point_process(self, syn_manager):
        fake_pp = object()
        syn_manager.add_point_process(GID, 0, "AMPA", fake_pp)
        retrieved = syn_manager.get_point_process(GID, 0, "AMPA")
        assert retrieved is fake_pp

    def test_duplicate_point_process_raises(self, syn_manager):
        syn_manager.add_point_process(GID, 0, "AMPA", object())
        with pytest.raises(RuntimeError, match="already has mechanism"):
            syn_manager.add_point_process(GID, 0, "AMPA", object())

    def test_get_missing_throw_false_returns_none(self, syn_manager):
        assert syn_manager.get_point_process(GID, 0, "AMPA", throw_error=False) is None

    def test_add_has_get_netcon(self, syn_manager):
        fake_nc = object()
        syn_manager.add_netcon(GID, 0, "AMPA", fake_nc)
        assert syn_manager.has_netcon(GID, 0, "AMPA")
        assert syn_manager.get_netcon(GID, 0, "AMPA") is fake_nc

    def test_del_netcon_removes_entry(self, syn_manager):
        syn_manager.add_netcon(GID, 0, "AMPA", object())
        syn_manager.del_netcon(GID, 0, "AMPA")
        assert not syn_manager.has_netcon(GID, 0, "AMPA")

    def test_del_netcon_throw_false_when_absent(self, syn_manager):
        syn_manager.del_netcon(GID, 0, "AMPA", throw_error=False)  # must not raise

    def test_add_has_get_vecstim(self, syn_manager):
        fake_vs = object()
        fake_nc = object()
        syn_manager.add_vecstim(GID, 0, "AMPA", fake_vs, fake_nc)
        assert syn_manager.has_vecstim(GID, 0, "AMPA")
        vs, nc = syn_manager.get_vecstim(GID, 0, "AMPA")
        assert vs is fake_vs
        assert nc is fake_nc


# ===========================================================================
# TestAddDefaultMechanismParameters
# ===========================================================================


class TestAddDefaultMechanismParameters:
    """Verify that add_default_mechanism_parameters stores defaults under the
    mechanism name (e.g. 'ExpSyn'), not the synapse label (e.g. 'AMPA'), so
    that get_effective_mechanism_parameters can retrieve them."""

    def test_default_params_readable_via_get_effective(self, populated_manager):
        """Defaults written via add_default_mechanism_parameters must be
        returned by get_effective_mechanism_parameters for the same synapse."""
        populated_manager.add_default_mechanism_parameters(
            GID, "AMPA", {"tau": 3.0, "e": -70.0}, syn_ids=None
        )
        params = populated_manager.get_effective_mechanism_parameters(GID, 0, "AMPA")
        assert pytest.approx(params["tau"]) == 3.0
        assert pytest.approx(params["e"]) == -70.0

    def test_default_applies_to_all_synapses(self, populated_manager):
        populated_manager.add_default_mechanism_parameters(
            GID, "AMPA", {"tau": 5.0}, syn_ids=None
        )
        for sid in range(6):  # first 6 are GC (AMPA)
            params = populated_manager.get_effective_mechanism_parameters(
                GID, sid, "AMPA"
            )
            assert pytest.approx(params["tau"]) == 5.0

    def test_gaba_label_resolves_to_exp2syn(self, populated_manager):
        """The GABA label must map to Exp2Syn parameters correctly."""
        populated_manager.add_default_mechanism_parameters(
            GID, "GABA", {"tau1": 1.0, "tau2": 8.0}, syn_ids=None
        )
        params = populated_manager.get_effective_mechanism_parameters(GID, 0, "GABA")
        assert pytest.approx(params["tau1"]) == 1.0
        assert pytest.approx(params["tau2"]) == 8.0


# ===========================================================================
# TestAddMechanismParametersFromIter
# ===========================================================================


class TestAddMechanismParametersFromIter:
    """Tests for add_mechanism_parameters_from_iter / _process_mech_attrs_batch."""

    @pytest.fixture
    def single_manager(self, syn_manager):
        """SynapseManager with one synapse registered for GID."""
        n = 3
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
        syn_manager.init_edge_attrs(
            GID, "GC", np.arange(n, dtype=np.int32), syn_ids, delays=[2.0] * n
        )
        return syn_manager

    def test_sets_parameter_via_iter(self, single_manager):
        single_manager.add_mechanism_parameters_from_iter(
            GID, "AMPA", iter([(0, {"tau": 4.0})])
        )
        params = single_manager.get_effective_mechanism_parameters(GID, 0, "AMPA")
        assert pytest.approx(params["tau"]) == 4.0

    def test_duplicate_raises_in_error_mode(self, single_manager):
        single_manager.add_mechanism_parameters_from_iter(
            GID, "AMPA", iter([(0, {"tau": 4.0})])
        )
        with pytest.raises(RuntimeError):
            single_manager.add_mechanism_parameters_from_iter(
                GID, "AMPA", iter([(0, {"tau": 9.0})])
            )

    def test_duplicate_skipped_in_skip_mode(self, single_manager):
        single_manager.add_mechanism_parameters_from_iter(
            GID, "AMPA", iter([(0, {"tau": 4.0})])
        )
        single_manager.add_mechanism_parameters_from_iter(
            GID, "AMPA", iter([(0, {"tau": 9.0})]), multiple="skip"
        )
        params = single_manager.get_effective_mechanism_parameters(GID, 0, "AMPA")
        assert pytest.approx(params["tau"]) == 4.0

    def test_duplicate_overwritten_in_overwrite_mode(self, single_manager):
        single_manager.add_mechanism_parameters_from_iter(
            GID, "AMPA", iter([(0, {"tau": 4.0})])
        )
        single_manager.add_mechanism_parameters_from_iter(
            GID, "AMPA", iter([(0, {"tau": 9.0})]), multiple="overwrite"
        )
        params = single_manager.get_effective_mechanism_parameters(GID, 0, "AMPA")
        assert pytest.approx(params["tau"]) == 9.0

    def test_unknown_gid_raises(self, single_manager):
        with pytest.raises(RuntimeError):
            single_manager.add_mechanism_parameters_from_iter(
                999, "AMPA", iter([(0, {"tau": 1.0})])
            )


# ===========================================================================
# TestGetMechanismParameters
# ===========================================================================


class TestGetMechanismParameters:
    """Tests for SynapseManager.get_mechanism_parameters."""

    @pytest.fixture
    def manager_with_params(self, syn_manager):
        n = 2
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
        syn_manager.init_edge_attrs(
            GID, "GC", np.arange(n, dtype=np.int32), syn_ids, delays=[2.0] * n
        )
        syn_manager.add_mechanism_parameters(GID, 0, "AMPA", {"tau": 6.0, "e": -5.0})
        return syn_manager

    def test_returns_set_value(self, manager_with_params):
        result = manager_with_params.get_mechanism_parameters(GID, 0, "AMPA")
        assert result is not None
        assert pytest.approx(result["tau"]) == 6.0
        assert pytest.approx(result["e"]) == -5.0

    def test_returns_none_for_unknown_syn_id(self, manager_with_params):
        result = manager_with_params.get_mechanism_parameters(
            GID, 999, "AMPA", throw_error_on_missing_id=False
        )
        assert result is None

    def test_raises_for_unknown_syn_id(self, manager_with_params):
        with pytest.raises(RuntimeError):
            manager_with_params.get_mechanism_parameters(
                GID, 999, "AMPA", throw_error_on_missing_id=True
            )

    def test_returns_none_when_no_params_set(self, manager_with_params):
        # syn_id=1 exists but has no specific parameters
        result = manager_with_params.get_mechanism_parameters(
            GID, 1, "AMPA", throw_error_on_missing_param=False
        )
        assert result is None


# ===========================================================================
# TestModifySynParam
# ===========================================================================


class _Node:
    """Minimal section-node stub: only .index is needed by apply_syn_mech_rules."""

    def __init__(self, index=0):
        self.index = index


class _ModifyCell:
    """Minimal cell for modify_syn_param tests."""

    def __init__(self, gid, population_name, sec_type, n_nodes=1, mech_dict=None):
        self.gid = gid
        self.population_name = population_name
        self.nodes = {sec_type: [_Node(index=i) for i in range(n_nodes)]}
        self.mech_dict = mech_dict if mech_dict is not None else {}
        self.is_reduced = False


class TestModifySynParam:
    """Tests for modify_syn_param: validation, cell.mech_dict updates, and
    param_store updates (update_targets=False, no NEURON required).

    Synapses are intentionally initialised without init_edge_attrs so that
    source_population stays at -1.  modify_mechanism_parameters then finds
    no presyn_name entry and skips the connection_config lookup, thus no
    connection_config is needed in the env.
    """

    @pytest.fixture
    def setup(self, syn_manager):
        """Two apical synapses at section 0, env wired for modify_syn_param."""
        n = 2
        syn_ids = np.arange(n, dtype=np.uint32)
        syn_manager.init_syn_id_attrs(
            GID,
            syn_ids,
            np.zeros(n, dtype=np.int8),
            np.zeros(n, dtype=np.uint8),
            np.full(n, SWC_APICAL, dtype=np.uint8),
            np.zeros(n, dtype=np.uint16),  # syn_section = 0
            np.full(n, 0.5, dtype=np.float32),
        )
        # source_population = -1 (default) -> presyn_name = None -> no connection_config needed
        env = MockEnv()
        env.synapse_manager = syn_manager
        env.cache_queries = False
        cell = _ModifyCell(GID, "GC", "apical")
        return env, cell, syn_ids

    # ------------------------------------------------------------------
    # Validation errors

    def test_raises_when_sec_type_not_in_cell(self, setup):
        env, cell, _ = setup
        with pytest.raises(ValueError, match="sec_type"):
            modify_syn_param(
                cell, env, "no_such_sec_type", "AMPA", param_name="tau", value=1.0
            )

    def test_raises_when_param_name_is_none(self, setup):
        env, cell, _ = setup
        with pytest.raises(ValueError):
            modify_syn_param(cell, env, "apical", "AMPA", param_name=None, value=1.0)

    def test_raises_when_value_is_none(self, setup):
        env, cell, _ = setup
        with pytest.raises(ValueError):
            modify_syn_param(cell, env, "apical", "AMPA", param_name="tau", value=None)

    def test_raises_when_param_not_recognized(self, setup):
        env, cell, _ = setup
        with pytest.raises(ValueError, match="not recognized"):
            modify_syn_param(
                cell, env, "apical", "AMPA", param_name="no_such_param", value=1.0
            )

    # ------------------------------------------------------------------
    # cell.mech_dict update

    def test_populates_mech_dict_on_first_call(self, setup):
        env, cell, _ = setup
        modify_syn_param(cell, env, "apical", "AMPA", param_name="tau", value=3.0)
        rule = cell.mech_dict["apical"]["synapses"]["AMPA"]["tau"]
        assert pytest.approx(rule["value"]) == 3.0

    def test_second_call_overwrites_mech_dict_value(self, setup):
        env, cell, _ = setup
        modify_syn_param(cell, env, "apical", "AMPA", param_name="tau", value=3.0)
        modify_syn_param(cell, env, "apical", "AMPA", param_name="tau", value=9.0)
        rule = cell.mech_dict["apical"]["synapses"]["AMPA"]["tau"]
        assert pytest.approx(rule["value"]) == 9.0

    # ------------------------------------------------------------------
    # param_store update

    def test_tau_stored_for_all_syn_ids(self, setup):
        env, cell, syn_ids = setup
        modify_syn_param(cell, env, "apical", "AMPA", param_name="tau", value=4.5)
        for syn_id in syn_ids:
            params = env.synapse_manager.get_effective_mechanism_parameters(
                GID, int(syn_id), "AMPA"
            )
            assert pytest.approx(params["tau"]) == 4.5

    def test_netcon_weight_stored_for_all_syn_ids(self, setup):
        env, cell, syn_ids = setup
        modify_syn_param(cell, env, "apical", "AMPA", param_name="weight", value=0.003)
        for syn_id in syn_ids:
            params = env.synapse_manager.get_effective_mechanism_parameters(
                GID, int(syn_id), "AMPA"
            )
            assert pytest.approx(params["weight"]) == 0.003

    def test_second_call_overwrites_param_store_value(self, setup):
        env, cell, syn_ids = setup
        modify_syn_param(cell, env, "apical", "AMPA", param_name="tau", value=4.0)
        modify_syn_param(cell, env, "apical", "AMPA", param_name="tau", value=9.0)
        params = env.synapse_manager.get_effective_mechanism_parameters(
            GID, int(syn_ids[0]), "AMPA"
        )
        assert pytest.approx(params["tau"]) == 9.0
