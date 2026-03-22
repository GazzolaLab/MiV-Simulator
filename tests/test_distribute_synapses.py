"""Unit tests for synapse distribution routines in
miv_simulator.simulator.distribute_synapses.

Synapse-type and SWC-type constants (integers matching a typical config):
    syn_type  0 = excitatory
    syn_type  1 = inhibitory
    swc_type  4 = apical

NEURON is required for these tests.  Run with:
    PYTHONPATH=/home/igr/bin/nrnpython3/lib/python:$PYTHONPATH \\
        /home/igr/venv/bin/pytest tests/test_distribute_synapses.py -v
"""

from collections import defaultdict

import networkx as nx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYN_EXC = 0  # excitatory synapse type
SYN_INH = 1  # inhibitory synapse type
SWC_APICAL = 4


# ---------------------------------------------------------------------------
# Minimal mock infrastructure for NEURON Section / Segment objects
# ---------------------------------------------------------------------------


class MockSegment:
    """Minimal stand-in for a NEURON Segment."""

    def __init__(self, x, sec):
        self.x = x
        self.sec = sec


class MockSection:
    """Minimal stand-in for a NEURON Section.

    Provides all methods consumed by the real ``interplocs`` function so that
    tests can call the genuine implementation without patching:
      - n3d / arc3d / x3d / y3d / z3d / diam3d
    The section is modelled as a straight line along the x-axis.
    """

    def __init__(self, L=100.0, nseg=5):
        self.L = L
        self.nseg = nseg
        # Two 3D points: one at origin, one at (L, 0, 0)
        self._n3d = 2
        self._segs = [MockSegment(x=(i + 0.5) / nseg, sec=self) for i in range(nseg)]

    # --- geometry methods required by interplocs ---
    def n3d(self):
        return self._n3d

    def arc3d(self, i):
        """Arc length (un-normalised) at 3D point i."""
        return i * self.L  # 0 at start, L at end

    def x3d(self, i):
        return i * self.L  # x increases linearly with arc

    def y3d(self, i):
        return 0.0

    def z3d(self, i):
        return 0.0

    def diam3d(self, i):
        return 1.0

    def __iter__(self):
        return iter(self._segs)


class SectionIndexMap:
    """Dual-use object: yields integer indices when iterated, maps sections to
    indices when subscripted.  Mimics cell.apicalidx / cell.somaidx.
    """

    def __init__(self, pairs):
        """pairs: list of (MockSection, int) tuples in section order."""
        self._sec_to_idx = {sec: idx for sec, idx in pairs}
        self._indices = [idx for _, idx in pairs]

    def __iter__(self):
        return iter(self._indices)

    def __getitem__(self, key):
        return self._sec_to_idx[key]


def _make_interplocs_patch(n_coords=3):
    """Return a replacement for interplocs that gives linear coordinate funcs.

    For a section with L=100 the returned functions produce:
        coord_i(loc) = loc * 100
    giving syn_cdist = sqrt(n_coords) * loc * 100.
    """

    def fake_interplocs(sec):
        return [lambda loc, _L=sec.L: loc * _L for _ in range(n_coords)]

    return fake_interplocs


def _make_section_graph_patch(secnodes_dict):
    """Return a replacement for make_section_graph producing a trivial graph."""

    def fake_make_section_graph(_neurotree_dict):
        g = nx.DiGraph()
        for sec_index in secnodes_dict:
            g.add_node(sec_index)
        return g

    return fake_make_section_graph


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def syn_type_dict():
    return {"excitatory": SYN_EXC, "inhibitory": SYN_INH}


@pytest.fixture
def swc_type_dict():
    return {"apical": SWC_APICAL}


@pytest.fixture
def layer_dict():
    return {"default": -1, "Layer1": 1}


@pytest.fixture
def density_dict_default():
    """Single section 'apical', excitatory synapses, default layer."""
    return {
        "apical": {
            "excitatory": {
                "default": {"mean": 0.5, "variance": 0.0},
            }
        }
    }


@pytest.fixture
def mock_section():
    return MockSection(L=100.0, nseg=5)


@pytest.fixture
def neurotree_dict(mock_section):
    """Minimal neurotree_dict for a single apical section (index 0).

    Includes the ``src``/``dst``/``loc`` arrays required by
    ``make_section_graph`` (all empty for a single isolated section) as well
    as the ``nodes`` mapping consumed by the density helpers.
    """
    return {
        "section_topology": {
            "nodes": {0: [0, 1]},  # section 0 → node indices 0 and 1
            "src": [],
            "dst": [],
            "loc": [],
        },
        "layer": [-1, -1],  # both nodes in default (−1) layer
    }


@pytest.fixture
def cell_fixtures(mock_section, neurotree_dict):
    """cell_sec_dict and cell_secidx_dict for a single apical section."""
    sec_index = 0
    sec_idx_map = SectionIndexMap([(mock_section, sec_index)])
    cell_sec_dict = {"apical": ([mock_section], None)}
    cell_secidx_dict = {"apical": sec_idx_map}
    return cell_sec_dict, cell_secidx_dict


# ---------------------------------------------------------------------------
# Tests: update_synapse_statistics
# ---------------------------------------------------------------------------


class TestUpdateSynapseStatistics:
    def test_excitatory_counts_accumulate(self):
        from miv_simulator.simulator.distribute_synapses import (
            update_synapse_statistics,
        )

        syn_dict = {
            "syn_ids": np.array([0, 1], dtype=np.uint32),
            "syn_secs": np.array([0, 0], dtype=np.uint32),
            "syn_types": np.array([SYN_EXC, SYN_EXC], dtype=np.uint8),
            "swc_types": np.array([SWC_APICAL, SWC_APICAL], dtype=np.uint8),
            "syn_layers": np.array([1, 1], dtype=np.int8),
        }
        stats = {
            "section": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "layer": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "swc_type": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "total": {"excitatory": 0, "inhibitory": 0},
        }

        result = update_synapse_statistics(syn_dict, stats)

        assert result["total"]["excitatory"] == 2
        assert result["total"]["inhibitory"] == 0
        assert stats["total"]["excitatory"] == 2

    def test_mixed_types_accumulate_separately(self):
        from miv_simulator.simulator.distribute_synapses import (
            update_synapse_statistics,
        )

        syn_dict = {
            "syn_ids": np.array([0, 1, 2], dtype=np.uint32),
            "syn_secs": np.array([0, 1, 1], dtype=np.uint32),
            "syn_types": np.array([SYN_EXC, SYN_INH, SYN_INH], dtype=np.uint8),
            "swc_types": np.array([SWC_APICAL] * 3, dtype=np.uint8),
            "syn_layers": np.array([1, 0, 0], dtype=np.int8),
        }
        stats = {
            "section": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "layer": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "swc_type": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "total": {"excitatory": 0, "inhibitory": 0},
        }

        result = update_synapse_statistics(syn_dict, stats)

        assert result["total"]["excitatory"] == 1
        assert result["total"]["inhibitory"] == 2

    def test_unknown_syn_type_raises(self):
        from miv_simulator.simulator.distribute_synapses import (
            update_synapse_statistics,
        )

        syn_dict = {
            "syn_ids": np.array([0], dtype=np.uint32),
            "syn_secs": np.array([0], dtype=np.uint32),
            "syn_types": np.array([99], dtype=np.uint8),
            "swc_types": np.array([SWC_APICAL], dtype=np.uint8),
            "syn_layers": np.array([1], dtype=np.int8),
        }
        stats = {
            "section": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "layer": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "swc_type": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "total": {"excitatory": 0, "inhibitory": 0},
        }

        with pytest.raises(ValueError, match="Unknown synapse type"):
            update_synapse_statistics(syn_dict, stats)


# ---------------------------------------------------------------------------
# Tests: check_synapses (no NEURON)
# ---------------------------------------------------------------------------


class TestCheckSynapses:
    def _make_stats(self, exc_count=2, inh_count=0, layer=1, swc_type=SWC_APICAL):
        stats = {
            "section": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "layer": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "swc_type": defaultdict(lambda: {"excitatory": 0, "inhibitory": 0}),
            "total": {"excitatory": exc_count, "inhibitory": inh_count},
        }
        stats["layer"][layer]["excitatory"] = exc_count
        stats["swc_type"][swc_type]["excitatory"] = exc_count
        return stats

    def test_no_warning_when_layers_populated(self, caplog):
        from miv_simulator.simulator.distribute_synapses import check_synapses

        stats = self._make_stats(exc_count=3, layer=1)
        layer_set_dict = {"excitatory": {"Layer1"}}
        swc_set_dict = {"excitatory": {"apical"}}
        swc_defs = {"apical": SWC_APICAL}
        layer_defs = {"Layer1": 1}

        with caplog.at_level("WARNING"):
            check_synapses(
                gid=0,
                morph_dict={},
                syn_stats_dict=stats,
                seg_density_per_sec={},
                layer_set_dict=layer_set_dict,
                swc_set_dict=swc_set_dict,
                swc_defs=swc_defs,
                layer_defs=layer_defs,
                logger=__import__("logging").getLogger("test"),
            )

        assert "incomplete" not in caplog.text.lower()

    def test_warning_when_layer_missing(self, caplog):
        from miv_simulator.simulator.distribute_synapses import check_synapses

        # Stats have layer 1, but layer_set_dict requires layer 2 (absent)
        stats = self._make_stats(exc_count=3, layer=1)
        stats["layer"][2]["excitatory"] = 0
        layer_set_dict = {"excitatory": {"Layer2"}}
        swc_set_dict = {"excitatory": {"apical"}}
        swc_defs = {"apical": SWC_APICAL}
        layer_defs = {"Layer2": 2}

        import logging

        logger = logging.getLogger("test_warn")
        with caplog.at_level("WARNING", logger="test_warn"):
            check_synapses(
                gid=42,
                morph_dict={},
                syn_stats_dict=stats,
                seg_density_per_sec={},
                layer_set_dict=layer_set_dict,
                swc_set_dict=swc_set_dict,
                swc_defs=swc_defs,
                layer_defs=layer_defs,
                logger=logger,
            )

        assert "incomplete" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Tests: local_syn_summary (no NEURON)
# ---------------------------------------------------------------------------


class TestLocalSynSummary:
    def test_output_contains_layer_and_swc_type_counts(self):
        from miv_simulator.simulator.distribute_synapses import local_syn_summary

        stats = {
            "layer": {1: {"excitatory": 5, "inhibitory": 2}},
            "swc_type": {SWC_APICAL: {"excitatory": 5, "inhibitory": 2}},
            "total": {"excitatory": 5, "inhibitory": 2},
        }

        output = local_syn_summary(stats)

        assert "layer" in output
        assert "swc_type" in output
        assert "5" in output
        assert "2" in output


# ---------------------------------------------------------------------------
# Tests: get_node_attribute
# ---------------------------------------------------------------------------


class TestGetNodeAttribute:
    def test_returns_none_when_name_absent(self, mock_section):
        from miv_simulator.simulator.distribute_synapses import get_node_attribute

        result = get_node_attribute("nonexistent", {}, mock_section, [0, 1], x=0.5)
        assert result is None

    def test_returns_value_when_x_is_none(self, mock_section):
        from miv_simulator.simulator.distribute_synapses import get_node_attribute

        content = {"layer": [1, 2]}
        result = get_node_attribute("layer", content, mock_section, [0, 1], x=None)
        assert result == [1, 2]

    def test_returns_first_element_when_no_3d_points(self):
        from miv_simulator.simulator.distribute_synapses import get_node_attribute

        class FlatSection:
            L = 100.0

            def n3d(self):
                return 0

        content = {"layer": [7, 8]}
        result = get_node_attribute("layer", content, FlatSection(), [0, 1], x=0.5)
        assert result == 7

    def test_interpolates_at_position(self, mock_section):
        from miv_simulator.simulator.distribute_synapses import get_node_attribute

        # mock_section has 2 3d points at arc=0 and arc=100 (L=100)
        # secnodes = [10, 20]; content["layer"][10]=3, content["layer"][20]=4
        content = {"layer": {10: 3, 20: 4}}
        secnodes = [10, 20]
        # x=0.5 → pos=0.5 for arc3d(1)/L=100/100=1.0, but arc3d(0)/L=0.0
        # The loop: i=0 → pos=0.0 < 0.5, prev=0.0; i=1 → pos=1.0 >= 0.5
        # prev=0.0, |1.0-0.5|=0.5, |0.0-0.5|=0.5 → equal, takes secnodes[1]=20
        result = get_node_attribute("layer", content, mock_section, secnodes, x=0.5)
        assert result in (3, 4)  # either node, depending on tie-breaking


# ---------------------------------------------------------------------------
# Tests: synapse_seg_density
# ---------------------------------------------------------------------------


class TestSynapseSegDensity:
    def test_returns_positive_density_for_nonzero_mean(self, syn_type_dict, layer_dict):
        from miv_simulator.simulator.distribute_synapses import synapse_seg_density

        sec = MockSection(L=100.0, nseg=3)
        seg_dict = {0: [seg for seg in sec if 0.0 < seg.x < 1.0]}
        ran = np.random.RandomState(42)
        layer_density_dicts = {
            "excitatory": {"default": {"mean": 1.0, "variance": 0.0}}
        }

        segdensity_dict, layers_dict = synapse_seg_density(
            syn_type_dict,
            layer_dict,
            layer_density_dicts,
            seg_dict,
            ran,
            neurotree_dict=None,
        )

        assert SYN_EXC in segdensity_dict
        densities = segdensity_dict[SYN_EXC][0]
        assert all(d > 0.0 for d in densities)

    def test_layers_dict_has_correct_structure(self, syn_type_dict, layer_dict):
        from miv_simulator.simulator.distribute_synapses import synapse_seg_density

        sec = MockSection(L=50.0, nseg=2)
        seg_dict = {0: list(sec)}
        ran = np.random.RandomState(0)
        layer_density_dicts = {
            "excitatory": {"default": {"mean": 0.5, "variance": 0.0}}
        }

        _, layers_dict = synapse_seg_density(
            syn_type_dict, layer_dict, layer_density_dicts, seg_dict, ran
        )

        assert SYN_EXC in layers_dict
        # Without neurotree_dict each seg gets layer=-1
        assert layers_dict[SYN_EXC][0] == [-1, -1]


# ---------------------------------------------------------------------------
# Tests: synapse_seg_counts
# ---------------------------------------------------------------------------


class TestSynapseSegCounts:
    def test_returns_positive_count_for_nonzero_mean(self, syn_type_dict, layer_dict):
        from miv_simulator.simulator.distribute_synapses import synapse_seg_counts

        sec = MockSection(L=100.0, nseg=3)
        seg_dict = {0: list(sec)}
        ran = np.random.RandomState(7)
        layer_density_dicts = {
            "excitatory": {"default": {"mean": 1.0, "variance": 0.0}}
        }

        segcounts_dict, total, layers_dict = synapse_seg_counts(
            syn_type_dict,
            layer_dict,
            layer_density_dicts,
            sec_index_dict=None,
            seg_dict=seg_dict,
            ran=ran,
            neurotree_dict=None,
        )

        assert SYN_EXC in segcounts_dict
        assert total > 0
        assert all(c > 0 for c in segcounts_dict[SYN_EXC])

    def test_zero_count_for_zero_mean(self, syn_type_dict, layer_dict):
        from miv_simulator.simulator.distribute_synapses import synapse_seg_counts

        sec = MockSection(L=100.0, nseg=2)
        seg_dict = {0: list(sec)}
        ran = np.random.RandomState(0)
        # No layer matches → ran is set to None → segcount = 0
        layer_density_dicts = {"excitatory": {"Layer1": {"mean": 1.0, "variance": 0.0}}}

        segcounts_dict, total, _ = synapse_seg_counts(
            syn_type_dict,
            layer_dict,
            layer_density_dicts,
            sec_index_dict=None,
            seg_dict=seg_dict,
            ran=ran,
            neurotree_dict=None,
        )

        # layer=-1, rans has only Layer1 (int 1) → no match → count=0
        assert segcounts_dict[SYN_EXC] == [0, 0]


# ---------------------------------------------------------------------------
# Tests: distribute_uniform_synapses
# ---------------------------------------------------------------------------


class TestDistributeUniformSynapses:
    def _run(self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default):
        from miv_simulator.simulator.distribute_synapses import (
            distribute_uniform_synapses,
        )

        sec = MockSection(L=100.0, nseg=10)
        sec_idx_map = SectionIndexMap([(sec, 0)])
        cell_sec_dict = {"apical": ([sec], None)}
        cell_secidx_dict = {"apical": sec_idx_map}
        neurotree = {
            "section_topology": {"nodes": {0: [0, 1]}},
            "layer": [-1, -1],
        }

        syn_dict, segcounts_per_sec = distribute_uniform_synapses(
            density_seed=42,
            syn_type_dict=syn_type_dict,
            swc_type_dict=swc_type_dict,
            layer_dict=layer_dict,
            sec_layer_density_dict=density_dict_default,
            neurotree_dict=neurotree,
            cell_sec_dict=cell_sec_dict,
            cell_secidx_dict=cell_secidx_dict,
        )
        return syn_dict, segcounts_per_sec

    def test_returns_all_required_keys(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default
        )
        for key in (
            "syn_ids",
            "syn_locs",
            "syn_cdists",
            "syn_secs",
            "syn_layers",
            "syn_types",
            "swc_types",
        ):
            assert key in syn_dict, f"Missing key: {key}"

    def test_syn_ids_are_contiguous_from_zero(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default
        )
        ids = syn_dict["syn_ids"]
        assert len(ids) > 0
        assert list(ids) == list(range(len(ids)))

    def test_syn_locs_in_unit_interval(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default
        )
        locs = syn_dict["syn_locs"]
        assert np.all(locs >= 0.0)
        assert np.all(locs < 1.0)

    def test_swc_type_matches_section(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default
        )
        assert np.all(syn_dict["swc_types"] == SWC_APICAL)

    def test_syn_types_match_excitatory(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default
        )
        assert np.all(syn_dict["syn_types"] == SYN_EXC)


# ---------------------------------------------------------------------------
# Tests: distribute_poisson_synapses
# ---------------------------------------------------------------------------


class TestDistributePoissonSynapses:
    def _run(
        self,
        syn_type_dict,
        swc_type_dict,
        layer_dict,
        density_dict_default,
        seed=42,
    ):
        from miv_simulator.simulator.distribute_synapses import (
            distribute_poisson_synapses,
        )

        sec = MockSection(L=100.0, nseg=10)
        sec_idx_map = SectionIndexMap([(sec, 0)])
        cell_sec_dict = {"apical": ([sec], None)}
        cell_secidx_dict = {"apical": sec_idx_map}
        neurotree = {
            "section_topology": {
                "nodes": {0: [0, 1]},
                "src": [],
                "dst": [],
                "loc": [],
            },
            "layer": [-1, -1],
        }

        syn_dict, seg_density_per_sec = distribute_poisson_synapses(
            density_seed=seed,
            syn_type_dict=syn_type_dict,
            swc_type_dict=swc_type_dict,
            layer_dict=layer_dict,
            sec_layer_density_dict=density_dict_default,
            neurotree_dict=neurotree,
            cell_sec_dict=cell_sec_dict,
            cell_secidx_dict=cell_secidx_dict,
        )
        return syn_dict, seg_density_per_sec

    def test_returns_all_required_keys(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default
        )
        for key in (
            "syn_ids",
            "syn_locs",
            "syn_cdists",
            "syn_secs",
            "syn_layers",
            "syn_types",
            "swc_types",
        ):
            assert key in syn_dict, f"Missing key: {key}"

    def test_syn_locs_in_unit_interval(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default
        )
        locs = syn_dict["syn_locs"]
        assert np.all(locs >= 0.0)
        assert np.all(locs < 1.0)

    def test_swc_type_matches_section(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default
        )
        assert np.all(syn_dict["swc_types"] == SWC_APICAL)

    def test_reproducibility_with_same_seed(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict_a, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default, seed=7
        )
        syn_dict_b, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default, seed=7
        )
        np.testing.assert_array_equal(syn_dict_a["syn_locs"], syn_dict_b["syn_locs"])
        np.testing.assert_array_equal(
            syn_dict_a["syn_cdists"], syn_dict_b["syn_cdists"]
        )

    def test_different_seeds_give_different_results(
        self, syn_type_dict, swc_type_dict, layer_dict, density_dict_default
    ):
        syn_dict_a, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default, seed=1
        )
        syn_dict_b, _ = self._run(
            syn_type_dict, swc_type_dict, layer_dict, density_dict_default, seed=2
        )
        # Different seeds should (almost certainly) produce different placements
        assert not np.array_equal(syn_dict_a["syn_locs"], syn_dict_b["syn_locs"])


# ---------------------------------------------------------------------------
# Tests: safety guard in synapse_seg_density (max_density_retries)
# ---------------------------------------------------------------------------


class TestSynapseSegDensityGuard:
    def test_raises_after_max_retries_when_density_nonpositive(
        self, syn_type_dict, layer_dict
    ):
        """mean=0 variance=0 can never yield a positive draw; guard must fire."""
        from miv_simulator.simulator.distribute_synapses import synapse_seg_density

        sec = MockSection(L=100.0, nseg=3)
        seg_dict = {0: list(sec)}
        ran = np.random.RandomState(0)
        layer_density_dicts = {
            "excitatory": {"default": {"mean": 0.0, "variance": 0.0}}
        }

        with pytest.raises(RuntimeError, match="positive density"):
            synapse_seg_density(
                syn_type_dict,
                layer_dict,
                layer_density_dicts,
                seg_dict,
                ran,
                neurotree_dict=None,
                max_density_retries=5,
            )

    def test_error_message_contains_retry_count(self, syn_type_dict, layer_dict):
        from miv_simulator.simulator.distribute_synapses import synapse_seg_density

        sec = MockSection(L=50.0, nseg=1)
        seg_dict = {0: list(sec)}
        ran = np.random.RandomState(1)
        layer_density_dicts = {
            "excitatory": {"default": {"mean": 0.0, "variance": 0.0}}
        }

        with pytest.raises(RuntimeError, match="10 attempts"):
            synapse_seg_density(
                syn_type_dict,
                layer_dict,
                layer_density_dicts,
                seg_dict,
                ran,
                neurotree_dict=None,
                max_density_retries=10,
            )


# ---------------------------------------------------------------------------
# Tests: safety guard in distribute_poisson_synapses (max_placement_retries)
# ---------------------------------------------------------------------------


class TestDistributePoissonSynapsesGuard:
    def test_raises_after_max_placement_retries_when_segment_unreachable(
        self, syn_type_dict, swc_type_dict, layer_dict
    ):
        """Very low density on a very short section: the exponential inter-arrival
        distance (beta = 1/density >> seg_length) almost never falls inside the first
        segment window.  With max_placement_retries=3 the guard fires immediately.
        """
        from miv_simulator.simulator.distribute_synapses import (
            distribute_poisson_synapses,
        )

        # Very short section (L=0.001) but low density (mean=1e-6 → beta=1e6).
        # Probability of a single exponential sample landing in [0, 0.001):
        #   P ≈ 1 - exp(-0.001 / 1e6) ≈ 1e-9  (virtually zero)
        sec = MockSection(L=0.001, nseg=1)
        sec_idx_map = SectionIndexMap([(sec, 0)])
        cell_sec_dict = {"apical": ([sec], None)}
        cell_secidx_dict = {"apical": sec_idx_map}
        neurotree = {
            "section_topology": {
                "nodes": {0: [0, 1]},
                "src": [],
                "dst": [],
                "loc": [],
            },
            "layer": [-1, -1],
        }
        density_dict = {
            "apical": {"excitatory": {"default": {"mean": 1e-6, "variance": 0.0}}}
        }

        with pytest.raises(RuntimeError, match="initial synapse"):
            distribute_poisson_synapses(
                density_seed=0,
                syn_type_dict=syn_type_dict,
                swc_type_dict=swc_type_dict,
                layer_dict=layer_dict,
                sec_layer_density_dict=density_dict,
                neurotree_dict=neurotree,
                cell_sec_dict=cell_sec_dict,
                cell_secidx_dict=cell_secidx_dict,
                max_placement_retries=3,
            )

    def test_error_message_contains_retry_count(
        self, syn_type_dict, swc_type_dict, layer_dict
    ):
        from miv_simulator.simulator.distribute_synapses import (
            distribute_poisson_synapses,
        )

        sec = MockSection(L=0.001, nseg=1)
        sec_idx_map = SectionIndexMap([(sec, 0)])
        cell_sec_dict = {"apical": ([sec], None)}
        cell_secidx_dict = {"apical": sec_idx_map}
        neurotree = {
            "section_topology": {
                "nodes": {0: [0, 1]},
                "src": [],
                "dst": [],
                "loc": [],
            },
            "layer": [-1, -1],
        }
        density_dict = {
            "apical": {"excitatory": {"default": {"mean": 1e-6, "variance": 0.0}}}
        }

        with pytest.raises(RuntimeError, match="7 attempts"):
            distribute_poisson_synapses(
                density_seed=0,
                syn_type_dict=syn_type_dict,
                swc_type_dict=swc_type_dict,
                layer_dict=layer_dict,
                sec_layer_density_dict=density_dict,
                neurotree_dict=neurotree,
                cell_sec_dict=cell_sec_dict,
                cell_secidx_dict=cell_secidx_dict,
                max_placement_retries=7,
            )
