"""Unit tests for miv_simulator.connections.

Synapse-type and SWC-type constants used throughout (integers that
match what a real config would produce):
    syn_type  0 = excitatory
    syn_type  1 = inhibitory
    swc_type  1 = soma
    swc_type  4 = apical
"""

from collections import defaultdict

import numpy as np
import pytest
from numpy.random import RandomState

from miv_simulator.connections import (
    ConnectionProbability,
    _consolidate_projection_connections,
    _partition_synapses_to_projections,
    _select_layer_source_vertices,
    choose_synapse_projection,
    generate_synaptic_connections,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYN_EXC = np.uint8(0)
SYN_INH = np.uint8(1)
SWC_SOMA = np.uint8(1)
SWC_APICAL = np.uint8(4)
LAYER_SOMA = np.int8(0)
LAYER_APICAL = np.int8(1)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def population_dict():
    return {"PopExc": 0, "PopInh": 1}


@pytest.fixture
def projection_synapse_dict():
    """Two non-overlapping projections: one excitatory (apical L1),
    one inhibitory (soma L0)."""
    return {
        "PopExc": (SYN_EXC, [LAYER_APICAL], [SWC_APICAL], [1.0], 1),
        "PopInh": (SYN_INH, [LAYER_SOMA], [SWC_SOMA], [1.0], 1),
    }


@pytest.fixture
def synapse_dict():
    """4 excitatory apical-L1 synapses + 2 inhibitory soma-L0 synapses."""
    return {
        "syn_ids": np.array([1, 2, 3, 4, 5, 6], dtype=np.uint32),
        "syn_cdists": np.array([0.5, 1.0, 1.5, 2.0, 0.3, 0.7], dtype=np.float32),
        "syn_types": np.array([SYN_EXC, SYN_EXC, SYN_EXC, SYN_EXC, SYN_INH, SYN_INH]),
        "swc_types": np.array(
            [SWC_APICAL, SWC_APICAL, SWC_APICAL, SWC_APICAL, SWC_SOMA, SWC_SOMA]
        ),
        "syn_layers": np.array(
            [
                LAYER_APICAL,
                LAYER_APICAL,
                LAYER_APICAL,
                LAYER_APICAL,
                LAYER_SOMA,
                LAYER_SOMA,
            ]
        ),
    }


@pytest.fixture
def projection_prob_dict():
    """Uniform source probabilities for PopExc (2 sources) and PopInh (3 sources)."""
    exc_probs = np.array([0.5, 0.5])
    exc_gids = np.array([10, 20], dtype=np.uint32)
    exc_du = np.array([0.1, 0.2])
    exc_dv = np.array([0.1, 0.2])

    inh_probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    inh_gids = np.array([30, 40, 50], dtype=np.uint32)
    inh_du = np.array([0.05, 0.1, 0.15])
    inh_dv = np.array([0.05, 0.1, 0.15])

    return {
        "PopExc": {LAYER_APICAL: (exc_probs, exc_gids, exc_du, exc_dv)},
        "PopInh": {LAYER_SOMA: (inh_probs, inh_gids, inh_du, inh_dv)},
    }


# ---------------------------------------------------------------------------
# Tests: choose_synapse_projection
# ---------------------------------------------------------------------------


class TestChooseSynapseProjection:
    def test_exact_match_returns_population_name(
        self, population_dict, projection_synapse_dict
    ):
        result = choose_synapse_projection(
            ranstream_syn=RandomState(0),
            syn_layer=LAYER_APICAL,
            swc_type=SWC_APICAL,
            syn_type=SYN_EXC,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
        )
        assert result == "PopExc"

    def test_inhibitory_match_returns_population_name(
        self, population_dict, projection_synapse_dict
    ):
        result = choose_synapse_projection(
            ranstream_syn=RandomState(0),
            syn_layer=LAYER_SOMA,
            swc_type=SWC_SOMA,
            syn_type=SYN_INH,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
        )
        assert result == "PopInh"

    def test_wrong_swc_type_returns_none(
        self, population_dict, projection_synapse_dict
    ):
        # Excitatory but on soma section: no projection matches
        result = choose_synapse_projection(
            ranstream_syn=RandomState(0),
            syn_layer=LAYER_APICAL,
            swc_type=SWC_SOMA,
            syn_type=SYN_EXC,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
        )
        assert result is None

    def test_wrong_layer_returns_none(self, population_dict, projection_synapse_dict):
        # Excitatory, correct section, but wrong layer
        result = choose_synapse_projection(
            ranstream_syn=RandomState(0),
            syn_layer=LAYER_SOMA,
            swc_type=SWC_APICAL,
            syn_type=SYN_EXC,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
        )
        assert result is None

    def test_multiple_candidates_selects_one(self, population_dict):
        """Two projections both receive excitatory apical-L1 synapses.
        The function must pick exactly one, respecting the proportions."""
        overlapping = {
            "PopA": (SYN_EXC, [LAYER_APICAL], [SWC_APICAL], [0.6], 1),
            "PopB": (SYN_EXC, [LAYER_APICAL], [SWC_APICAL], [0.4], 1),
        }
        pop = {"PopA": 0, "PopB": 1}
        results = set()
        for seed in range(20):
            r = choose_synapse_projection(
                ranstream_syn=RandomState(seed),
                syn_layer=LAYER_APICAL,
                swc_type=SWC_APICAL,
                syn_type=SYN_EXC,
                population_dict=pop,
                projection_synapse_dict=overlapping,
            )
            assert r in ("PopA", "PopB")
            results.add(r)
        # Over 20 seeds, both populations must be selected at least once
        assert results == {"PopA", "PopB"}


# ---------------------------------------------------------------------------
# Tests: _partition_synapses_to_projections
# ---------------------------------------------------------------------------


class TestPartitionSynapsesToProjections:
    def test_all_synapses_assigned(
        self, synapse_dict, population_dict, projection_synapse_dict
    ):
        cdist, partition = _partition_synapses_to_projections(
            gid=100,
            ranstream_syn=RandomState(42),
            synapse_dict=synapse_dict,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
            rank=0,
        )
        total_assigned = sum(
            len(ids) for layer_dict in partition.values() for ids in layer_dict.values()
        )
        assert total_assigned == len(synapse_dict["syn_ids"])

    def test_excitatory_synapses_go_to_PopExc(
        self, synapse_dict, population_dict, projection_synapse_dict
    ):
        _, partition = _partition_synapses_to_projections(
            gid=100,
            ranstream_syn=RandomState(42),
            synapse_dict=synapse_dict,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
            rank=0,
        )
        exc_ids = partition["PopExc"][LAYER_APICAL]
        # syn_ids 1-4 are excitatory
        assert sorted(exc_ids) == [1, 2, 3, 4]

    def test_inhibitory_synapses_go_to_PopInh(
        self, synapse_dict, population_dict, projection_synapse_dict
    ):
        _, partition = _partition_synapses_to_projections(
            gid=100,
            ranstream_syn=RandomState(42),
            synapse_dict=synapse_dict,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
            rank=0,
        )
        inh_ids = partition["PopInh"][LAYER_SOMA]
        assert sorted(inh_ids) == [5, 6]

    def test_cdist_dict_populated_for_all_synapses(
        self, synapse_dict, population_dict, projection_synapse_dict
    ):
        cdist, _ = _partition_synapses_to_projections(
            gid=100,
            ranstream_syn=RandomState(42),
            synapse_dict=synapse_dict,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
            rank=0,
        )
        for syn_id, cdist_val in zip(
            synapse_dict["syn_ids"], synapse_dict["syn_cdists"]
        ):
            assert cdist[syn_id] == pytest.approx(cdist_val)

    def test_assertion_fails_when_projection_is_empty(self, population_dict):
        """A synapse dict with only excitatory synapses will leave PopInh empty."""
        exc_only = {
            "syn_ids": np.array([1, 2], dtype=np.uint32),
            "syn_cdists": np.array([0.5, 1.0], dtype=np.float32),
            "syn_types": np.array([SYN_EXC, SYN_EXC]),
            "swc_types": np.array([SWC_APICAL, SWC_APICAL]),
            "syn_layers": np.array([LAYER_APICAL, LAYER_APICAL]),
        }
        proj = {
            "PopExc": (SYN_EXC, [LAYER_APICAL], [SWC_APICAL], [1.0], 1),
            "PopInh": (SYN_INH, [LAYER_SOMA], [SWC_SOMA], [1.0], 1),
        }
        with pytest.raises(AssertionError):
            _partition_synapses_to_projections(
                gid=1,
                ranstream_syn=RandomState(0),
                synapse_dict=exc_only,
                population_dict=population_dict,
                projection_synapse_dict=proj,
                rank=0,
            )


# ---------------------------------------------------------------------------
# Tests: _select_layer_source_vertices
# ---------------------------------------------------------------------------


def _make_uniform_random_choice(ranstream, n, p):
    """Deterministic mock: spreads n samples as evenly as possible."""
    counts = np.zeros(len(p), dtype=int)
    for i in range(n):
        counts[i % len(p)] += 1
    return counts


class TestSelectLayerSourceVertices:
    def test_empty_source_gids_returns_empty_arrays(self):
        sv, si, dist = _select_layer_source_vertices(
            rank=0,
            destination_gid=1,
            projection="PopExc",
            prj_layer=LAYER_APICAL,
            syn_ids=[1, 2, 3],
            syn_config_contacts=1,
            syn_cdist_dict={1: 0.5, 2: 1.0, 3: 1.5},
            source_probs=np.array([]),
            source_gids=np.array([], dtype=np.uint32),
            distances_u=np.array([]),
            distances_v=np.array([]),
            ranstream_con=RandomState(0),
            cluster_seed=42,
            random_choice=_make_uniform_random_choice,
        )
        assert len(sv) == 0
        assert len(si) == 0
        assert len(dist) == 0

    def test_output_length_matches_syn_ids(self):
        source_gids = np.array([10, 20, 30], dtype=np.uint32)
        source_probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        syn_ids = [1, 2, 3]

        sv, si, dist = _select_layer_source_vertices(
            rank=0,
            destination_gid=1,
            projection="PopExc",
            prj_layer=LAYER_APICAL,
            syn_ids=syn_ids,
            syn_config_contacts=1,
            syn_cdist_dict={1: 0.5, 2: 1.0, 3: 1.5},
            source_probs=source_probs,
            source_gids=source_gids,
            distances_u=np.array([0.1, 0.2, 0.3]),
            distances_v=np.array([0.1, 0.2, 0.3]),
            ranstream_con=RandomState(42),
            cluster_seed=7,
            random_choice=_make_uniform_random_choice,
        )
        assert len(sv) == len(syn_ids)
        assert len(si) == len(syn_ids)
        assert len(dist) == len(syn_ids)

    def test_source_vertices_are_valid_gids(self):
        source_gids = np.array([10, 20, 30], dtype=np.uint32)
        source_probs = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        sv, _, _ = _select_layer_source_vertices(
            rank=0,
            destination_gid=1,
            projection="PopExc",
            prj_layer=LAYER_APICAL,
            syn_ids=[1, 2, 3],
            syn_config_contacts=1,
            syn_cdist_dict={1: 0.5, 2: 1.0, 3: 1.5},
            source_probs=source_probs,
            source_gids=source_gids,
            distances_u=np.array([0.1, 0.2, 0.3]),
            distances_v=np.array([0.1, 0.2, 0.3]),
            ranstream_con=RandomState(42),
            cluster_seed=7,
            random_choice=_make_uniform_random_choice,
        )
        assert all(gid in source_gids for gid in sv)

    def test_syn_ids_ordered_by_cable_distance(self):
        """ordered_syn_ids must be sorted ascending by syn_cdist_dict value."""
        # Deliberate reverse-order cdists
        syn_ids = [10, 20, 30]
        cdist = {10: 3.0, 20: 1.0, 30: 2.0}
        source_gids = np.array([100, 200, 300], dtype=np.uint32)

        _, ordered, _ = _select_layer_source_vertices(
            rank=0,
            destination_gid=1,
            projection="PopExc",
            prj_layer=LAYER_APICAL,
            syn_ids=syn_ids,
            syn_config_contacts=1,
            syn_cdist_dict=cdist,
            source_probs=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
            source_gids=source_gids,
            distances_u=np.array([0.1, 0.2, 0.3]),
            distances_v=np.array([0.1, 0.2, 0.3]),
            ranstream_con=RandomState(0),
            cluster_seed=0,
            random_choice=_make_uniform_random_choice,
        )
        assert ordered == [20, 30, 10]

    def test_distances_are_sum_of_u_and_v(self):
        """Each returned distance must equal du + dv for the chosen source GID."""
        source_gids = np.array([10, 20], dtype=np.uint32)
        du = np.array([0.1, 0.3])
        dv = np.array([0.2, 0.4])
        expected_distance = {10: 0.3, 20: 0.7}

        sv, _, dist = _select_layer_source_vertices(
            rank=0,
            destination_gid=1,
            projection="PopExc",
            prj_layer=LAYER_APICAL,
            syn_ids=[1, 2],
            syn_config_contacts=1,
            syn_cdist_dict={1: 0.5, 2: 1.0},
            source_probs=np.array([0.5, 0.5]),
            source_gids=source_gids,
            distances_u=du,
            distances_v=dv,
            ranstream_con=RandomState(7),
            cluster_seed=0,
            random_choice=_make_uniform_random_choice,
        )
        for gid, d in zip(sv, dist):
            assert d == pytest.approx(expected_distance[int(gid)])

    def test_contacts_greater_than_1_multiplies_counts(self):
        """With contacts=2, source_gid_counts entries are doubled before shuffle."""
        call_log = []

        def recording_choice(ranstream, n, p):
            counts = _make_uniform_random_choice(ranstream, n, p)
            call_log.append((n, counts.copy()))
            return counts

        source_gids = np.array([10, 20], dtype=np.uint32)
        # 4 synapses, contacts=2 → n_syn_groups = ceil(4/2) = 2
        sv, _, _ = _select_layer_source_vertices(
            rank=0,
            destination_gid=1,
            projection="PopExc",
            prj_layer=LAYER_APICAL,
            syn_ids=[1, 2, 3, 4],
            syn_config_contacts=2,
            syn_cdist_dict={1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0},
            source_probs=np.array([0.5, 0.5]),
            source_gids=source_gids,
            distances_u=np.array([0.1, 0.2]),
            distances_v=np.array([0.1, 0.2]),
            ranstream_con=RandomState(0),
            cluster_seed=0,
            random_choice=recording_choice,
        )
        assert len(call_log) == 1
        n_groups_called, _ = call_log[0]
        # n_syn_groups = ceil(4/2) = 2
        assert n_groups_called == 2
        assert len(sv) == 4


# ---------------------------------------------------------------------------
# Tests: _consolidate_projection_connections
# ---------------------------------------------------------------------------


class TestConsolidateProjectionConnections:
    def test_empty_lists_raise_assertion(self):
        gid_dict = {}
        with pytest.raises(AssertionError):
            _consolidate_projection_connections(
                rank=0,
                destination_gid=1,
                projection="PopExc",
                prj_source_vertices=[],
                prj_syn_ids=[],
                prj_distances=[],
                gid_dict=gid_dict,
            )

    def test_single_layer_writes_gid_dict_entry(self):
        gid_dict = {}
        sv = np.array([10, 20], dtype=np.uint32)
        si = [1, 2]
        dist = np.array([0.3, 0.7], dtype=np.float32)

        count = _consolidate_projection_connections(
            rank=0,
            destination_gid=42,
            projection="PopExc",
            prj_source_vertices=[sv],
            prj_syn_ids=[si],
            prj_distances=[dist],
            gid_dict=gid_dict,
        )

        assert count == 2
        assert 42 in gid_dict
        result_sv, attrs = gid_dict[42]
        np.testing.assert_array_equal(result_sv, sv)
        np.testing.assert_array_equal(
            attrs["Synapses"]["syn_id"], np.array([1, 2], dtype=np.uint32)
        )
        np.testing.assert_array_almost_equal(attrs["Connections"]["distance"], dist)

    def test_returns_correct_count(self):
        gid_dict = {}
        count = _consolidate_projection_connections(
            rank=0,
            destination_gid=1,
            projection="PopExc",
            prj_source_vertices=[np.array([10, 20, 30], dtype=np.uint32)],
            prj_syn_ids=[[1, 2, 3]],
            prj_distances=[np.array([0.1, 0.2, 0.3], dtype=np.float32)],
            gid_dict=gid_dict,
        )
        assert count == 3

    def test_multi_layer_concatenates_arrays(self):
        gid_dict = {}
        sv1 = np.array([10, 20], dtype=np.uint32)
        sv2 = np.array([30, 40, 50], dtype=np.uint32)
        si1 = [1, 2]
        si2 = [3, 4, 5]
        d1 = np.array([0.1, 0.2], dtype=np.float32)
        d2 = np.array([0.3, 0.4, 0.5], dtype=np.float32)

        count = _consolidate_projection_connections(
            rank=0,
            destination_gid=7,
            projection="PopExc",
            prj_source_vertices=[sv1, sv2],
            prj_syn_ids=[si1, si2],
            prj_distances=[d1, d2],
            gid_dict=gid_dict,
        )
        assert count == 5
        result_sv, attrs = gid_dict[7]
        assert len(result_sv) == 5
        assert len(attrs["Synapses"]["syn_id"]) == 5
        assert len(attrs["Connections"]["distance"]) == 5

    def test_syn_id_dtype_is_uint32(self):
        gid_dict = {}
        _consolidate_projection_connections(
            rank=0,
            destination_gid=1,
            projection="PopExc",
            prj_source_vertices=[np.array([10], dtype=np.uint32)],
            prj_syn_ids=[[99]],
            prj_distances=[np.array([0.5], dtype=np.float32)],
            gid_dict=gid_dict,
        )
        _, attrs = gid_dict[1]
        assert attrs["Synapses"]["syn_id"].dtype == np.uint32


# ---------------------------------------------------------------------------
# Tests: ConnectionProbability
# ---------------------------------------------------------------------------


def _make_soma_data(destination_population, source_population):
    """Shared soma coords and distances: dest at (0.5, 0.5), sources at 0.1-step grid."""
    soma_coords = {
        destination_population: {100: (0.5, 0.5, 0.0)},
        source_population: {
            0: (0.1, 0.1, 0.0),
            1: (0.3, 0.3, 0.0),
            2: (0.5, 0.5, 0.0),
            3: (0.7, 0.7, 0.0),
            4: (0.9, 0.9, 0.0),
        },
    }
    soma_distances = {
        destination_population: {100: (0.5, 0.5)},
        source_population: {
            0: (0.1, 0.1),
            1: (0.3, 0.3),
            2: (0.5, 0.5),
            3: (0.7, 0.7),
            4: (0.9, 0.9),
        },
    }
    return soma_coords, soma_distances


def _make_connection_probability_narrow():
    """Narrow extent (width=0.6): only gids 1, 2, 3 fall within +/-0.3 of dest.

    NOT suitable for get_probability: the small scale_factor (0.1) causes
    norm.pdf values > 1 which violates the internal assertion in that method.
    Use only for filter_by_distance tests.
    """
    destination_population = "dest_pop"
    source_population = "src_pop"
    soma_coords, soma_distances = _make_soma_data(
        destination_population, source_population
    )
    # width=0.6 → u_extent = 0.3, scale_factor = 0.1
    extents = {source_population: {"default": {"width": [0.6, 0.6]}}}
    cp = ConnectionProbability(
        destination_population=destination_population,
        soma_coords=soma_coords,
        soma_distances=soma_distances,
        extents=extents,
    )
    return cp, source_population


def _make_connection_probability_wide():
    """Wide extent (width=9.0): all 5 sources included."""
    destination_population = "dest_pop"
    source_population = "src_pop"
    soma_coords, soma_distances = _make_soma_data(
        destination_population, source_population
    )
    extents = {source_population: {"default": {"width": [9.0, 9.0]}}}
    cp = ConnectionProbability(
        destination_population=destination_population,
        soma_coords=soma_coords,
        soma_distances=soma_distances,
        extents=extents,
    )
    return cp, source_population


def _make_connection_probability_integer_layer():
    """cp whose only layer key is integer 1 (no 'default' fallback).

    Calling filter_by_distance with any other layer raises RuntimeError.
    """
    destination_population = "dest_pop"
    source_population = "src_pop"
    soma_coords, soma_distances = _make_soma_data(
        destination_population, source_population
    )
    extents = {source_population: {1: {"width": [9.0, 9.0]}}}
    cp = ConnectionProbability(
        destination_population=destination_population,
        soma_coords=soma_coords,
        soma_distances=soma_distances,
        extents=extents,
    )
    return cp, source_population


class TestConnectionProbability:
    def test_width_stored_correctly(self):
        cp, src = _make_connection_probability_narrow()
        assert cp.width[src]["default"]["u"] == pytest.approx(0.3)
        assert cp.width[src]["default"]["v"] == pytest.approx(0.3)

    def test_scale_factor_is_width_over_3(self):
        cp, src = _make_connection_probability_narrow()
        assert cp.scale_factor[src]["default"]["u"] == pytest.approx(0.1)
        assert cp.scale_factor[src]["default"]["v"] == pytest.approx(0.1)

    def test_filter_by_distance_excludes_distant_sources(self):
        # narrow extent: only gids 1, 2, 3 are within ±0.3 of destination
        cp, src = _make_connection_probability_narrow()
        dest_u, dest_v, *_, source_gids = cp.filter_by_distance(
            destination_gid=100, source_population=src, source_layer="default"
        )
        assert 0 not in source_gids
        assert 4 not in source_gids

    def test_filter_by_distance_includes_nearby_sources(self):
        cp, src = _make_connection_probability_narrow()
        dest_u, dest_v, *_, source_gids = cp.filter_by_distance(
            destination_gid=100, source_population=src, source_layer="default"
        )
        assert 1 in source_gids
        assert 2 in source_gids
        assert 3 in source_gids

    def test_get_probability_probs_sum_to_one(self):
        # wide extent so scale_factor is large enough for norm.pdf values <= 1
        cp, src = _make_connection_probability_wide()
        prob_dict = cp.get_probability(
            destination_gid=100, source=src, source_layers=["default"]
        )
        probs, _, _, _ = prob_dict["default"]
        if len(probs) > 0:
            assert np.sum(probs) == pytest.approx(1.0)

    def test_get_probability_probs_are_non_negative(self):
        cp, src = _make_connection_probability_wide()
        prob_dict = cp.get_probability(
            destination_gid=100, source=src, source_layers=["default"]
        )
        probs, _, _, _ = prob_dict["default"]
        assert (probs >= 0.0).all()

    def test_missing_layer_config_raises_runtime_error(self):
        # cp has only integer layer 1; requesting layer 99 (no "default") must raise
        cp, src = _make_connection_probability_integer_layer()
        with pytest.raises(RuntimeError):
            cp.filter_by_distance(
                destination_gid=100,
                source_population=src,
                source_layer=99,
            )


# ---------------------------------------------------------------------------
# Tests: generate_synaptic_connections  (integration)
# ---------------------------------------------------------------------------


class TestGenerateSynapticConnections:
    def test_connection_dict_contains_both_projections(
        self,
        synapse_dict,
        population_dict,
        projection_synapse_dict,
        projection_prob_dict,
    ):
        connection_dict = defaultdict(dict)
        generate_synaptic_connections(
            rank=0,
            gid=100,
            ranstream_syn=RandomState(1),
            ranstream_con=RandomState(2),
            cluster_seed=99,
            destination_gid=100,
            synapse_dict=synapse_dict,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
            projection_prob_dict=projection_prob_dict,
            connection_dict=connection_dict,
            random_choice=_make_uniform_random_choice,
        )
        assert "PopExc" in connection_dict
        assert "PopInh" in connection_dict

    def test_total_count_matches_number_of_synapses(
        self,
        synapse_dict,
        population_dict,
        projection_synapse_dict,
        projection_prob_dict,
    ):
        connection_dict = defaultdict(dict)
        count = generate_synaptic_connections(
            rank=0,
            gid=100,
            ranstream_syn=RandomState(1),
            ranstream_con=RandomState(2),
            cluster_seed=99,
            destination_gid=100,
            synapse_dict=synapse_dict,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
            projection_prob_dict=projection_prob_dict,
            connection_dict=connection_dict,
            random_choice=_make_uniform_random_choice,
        )
        assert count == len(synapse_dict["syn_ids"])

    def test_gid_dict_structure_is_correct(
        self,
        synapse_dict,
        population_dict,
        projection_synapse_dict,
        projection_prob_dict,
    ):
        connection_dict = defaultdict(dict)
        generate_synaptic_connections(
            rank=0,
            gid=100,
            ranstream_syn=RandomState(1),
            ranstream_con=RandomState(2),
            cluster_seed=99,
            destination_gid=100,
            synapse_dict=synapse_dict,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
            projection_prob_dict=projection_prob_dict,
            connection_dict=connection_dict,
            random_choice=_make_uniform_random_choice,
        )
        for proj in ("PopExc", "PopInh"):
            assert 100 in connection_dict[proj]
            sv, attrs = connection_dict[proj][100]
            assert sv.dtype == np.uint32
            assert "Synapses" in attrs
            assert "Connections" in attrs
            assert "syn_id" in attrs["Synapses"]
            assert "distance" in attrs["Connections"]

    def test_exc_connections_sourced_from_exc_gids(
        self,
        synapse_dict,
        population_dict,
        projection_synapse_dict,
        projection_prob_dict,
    ):
        connection_dict = defaultdict(dict)
        generate_synaptic_connections(
            rank=0,
            gid=100,
            ranstream_syn=RandomState(1),
            ranstream_con=RandomState(2),
            cluster_seed=99,
            destination_gid=100,
            synapse_dict=synapse_dict,
            population_dict=population_dict,
            projection_synapse_dict=projection_synapse_dict,
            projection_prob_dict=projection_prob_dict,
            connection_dict=connection_dict,
            random_choice=_make_uniform_random_choice,
        )
        exc_source_gids = projection_prob_dict["PopExc"][LAYER_APICAL][1]
        sv_exc, _ = connection_dict["PopExc"][100]
        assert all(gid in exc_source_gids for gid in sv_exc)

    def test_deterministic_with_fixed_seeds(
        self,
        synapse_dict,
        population_dict,
        projection_synapse_dict,
        projection_prob_dict,
    ):
        def run():
            cd = defaultdict(dict)
            return generate_synaptic_connections(
                rank=0,
                gid=100,
                ranstream_syn=RandomState(7),
                ranstream_con=RandomState(13),
                cluster_seed=42,
                destination_gid=100,
                synapse_dict=synapse_dict,
                population_dict=population_dict,
                projection_synapse_dict=projection_synapse_dict,
                projection_prob_dict=projection_prob_dict,
                connection_dict=cd,
                random_choice=_make_uniform_random_choice,
            ), cd

        count1, cd1 = run()
        count2, cd2 = run()
        assert count1 == count2
        for proj in ("PopExc", "PopInh"):
            np.testing.assert_array_equal(cd1[proj][100][0], cd2[proj][100][0])
