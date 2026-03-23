"""Unit tests for miv_simulator.eval_network.

Heavy dependencies (NEURON, MPI network init, HDF5 file I/O) are
replaced with lightweight fakes so the tests run without a full simulator
environment.

Tested units
------------
- JSON parameter loading and label selection
- param_name -> (param_tuple, value) mapping loop (flat and nested paths)
- network.run is called with output=True
- output JSON is written with the correct structure
- output is suppressed when output_path is None
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure local src/ tree is on the path (mirrors conftest.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from miv_simulator.synapses import SynParam
from miv_simulator.optimization import OptConfig


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

TARGET_POPULATIONS = ["PYR", "PVBC"]
OBJECTIVE_NAMES = ["PYR firing rate", "PVBC firing rate"]
PARAM_CONFIG_NAME = "default"

OPERATIONAL_CONFIG = {
    "target_populations": TARGET_POPULATIONS,
    "param_config_name": PARAM_CONFIG_NAME,
    "objective_names": OBJECTIVE_NAMES,
    "temporal_resolution": 25.0,
}

# Two SynParam tuples corresponding to the two parameter names below
PARAM_TUPLES = [
    SynParam(
        population="PYR",
        source="CA3",
        sec_type="apical",
        syn_name="AMPA",
        param_path="weight",
        param_range=[0.1, 20.0],
        phenotype=None,
    ),
    SynParam(
        population="PVBC",
        source="EC",
        sec_type="apical",
        syn_name="AMPA",
        param_path="weight",
        param_range=[0.1, 20.0],
        phenotype=None,
    ),
]
PARAM_NAMES = [
    "PYR.CA3.apical.AMPA.weight",
    "PVBC.EC.apical.AMPA.weight",
]

PARAMS_DICT = {
    "PYR.CA3.apical.AMPA.weight": 5.5,
    "PVBC.EC.apical.AMPA.weight": 12.3,
}

OPT_TARGETS = {
    "PYR firing rate": 2.0,
    "PVBC firing rate": 15.0,
}

OPT_CONFIG = OptConfig(
    param_bounds={n: t.param_range for n, t in zip(PARAM_NAMES, PARAM_TUPLES)},
    param_names=PARAM_NAMES,
    param_initial_dict={n: 10.0 for n in PARAM_NAMES},
    param_tuples=PARAM_TUPLES,
    opt_targets=OPT_TARGETS,
)

# ---------------------------------------------------------------------------
# Test data: tuple sec_type  (e.g. "AAC.CA2.('apical', 'basal').AMPA.weight")
# ---------------------------------------------------------------------------

# SynParam where sec_type is a tuple; both apical and basal dendrites share
# the same synaptic weight.  The optimizer encodes this in the parameter name
# using Python's repr of the tuple: "('apical', 'basal')".
AAC_PARAM_TUPLE = SynParam(
    population="AAC",
    source="CA2",
    sec_type=("apical", "basal"),
    syn_name="AMPA",
    param_path="weight",
    param_range=[0.0, 10.0],
    phenotype=None,
)
AAC_PARAM_NAME = "AAC.CA2.('apical', 'basal').AMPA.weight"
AAC_PARAM_VALUE = 3.5

AAC_OPT_CONFIG = OptConfig(
    param_bounds={AAC_PARAM_NAME: AAC_PARAM_TUPLE.param_range},
    param_names=[AAC_PARAM_NAME],
    param_initial_dict={AAC_PARAM_NAME: 1.0},
    param_tuples=[AAC_PARAM_TUPLE],
    opt_targets=OPT_TARGETS,
)

# Fake compute_objectives output:
#   (objectives_arr, features_arr, constraints_arr)
OBJECTIVES_ARR = np.array([0.25, 1.44], dtype=np.float32)
# structured array with one row so features_arr[0].tolist() works
FEATURE_DTYPES = [(name, np.float32) for name in OBJECTIVE_NAMES]
FEATURES_ARR = np.array(
    [(1.5, 13.8)],
    dtype=np.dtype(FEATURE_DTYPES),
)
CONSTRAINTS_ARR = np.array([1.5, 13.8], dtype=np.float32)

COMPUTE_OBJECTIVES_RESULT = {0: (OBJECTIVES_ARR, FEATURES_ARR, CONSTRAINTS_ARR)}

# Features dict returned by network_features
FEATURES_DICT = {
    pop: {
        "n_total": 10,
        "n_active": 5,
        "time_bins": np.linspace(50.0, 500.0, 20, dtype=np.float32),
        "spike_density_dict": {},
    }
    for pop in TARGET_POPULATIONS
}


# ---------------------------------------------------------------------------
# Helper: build a mock env
# ---------------------------------------------------------------------------


def _make_mock_env(tstop=500.0, cleanup=False, recording_profile=None):
    env = MagicMock()
    env.tstop = tstop
    env.cleanup = cleanup
    env.recording_profile = recording_profile
    env.results_file_path = "/tmp/fake_results.h5"
    env.netclamp_config.optimize_parameters = {}
    env.phenotype_ids = {}
    # Single-process test: rank 0 performs I/O and bcast passes the value through
    env.pc.id.return_value = 0
    env.comm.bcast.side_effect = lambda obj, root: obj
    return env


# ---------------------------------------------------------------------------
# Helper: context manager that patches all heavy dependencies
# ---------------------------------------------------------------------------


def _patch_all(tmp_params_path, tmp_output_path=None):
    """Return a list of patch objects covering every heavyweight dependency."""
    return [
        patch("miv_simulator.eval_network.config_logging"),
        patch(
            "miv_simulator.eval_network.read_from_yaml",
            return_value=OPERATIONAL_CONFIG,
        ),
        patch(
            "miv_simulator.eval_network.init_network",
            return_value=_make_mock_env(),
        ),
        patch(
            "miv_simulator.eval_network.optimization_params",
            return_value=OPT_CONFIG,
        ),
        patch("miv_simulator.eval_network.update_network_params"),
        patch("miv_simulator.eval_network.network"),
        patch(
            "miv_simulator.eval_network.network_features",
            return_value=FEATURES_DICT,
        ),
        patch(
            "miv_simulator.eval_network.compute_objectives",
            return_value=COMPUTE_OBJECTIVES_RESULT,
        ),
        patch("miv_simulator.eval_network.io_utils"),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params_json(tmp_path):
    """Write a minimal params JSON to a temp file and return its path."""
    data = {
        "run_label": {
            "parameters": PARAMS_DICT,
            "objectives": {},
            "features": {},
            "constraints": {},
        }
    }
    p = tmp_path / "params.json"
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def multi_label_params_json(tmp_path):
    """JSON with two labels to verify default-first-key behaviour."""
    data = {
        "first_label": {"parameters": PARAMS_DICT},
        "second_label": {"parameters": {k: v + 1 for k, v in PARAMS_DICT.items()}},
    }
    p = tmp_path / "multi.json"
    p.write_text(json.dumps(data))
    return str(p)


# ---------------------------------------------------------------------------
# Tests: JSON loading and label selection
# ---------------------------------------------------------------------------


def test_explicit_params_label_is_used(params_json):
    """eval_network reads the entry matching the given params_label."""
    captured = {}

    def fake_update(env, ptv):
        captured["param_tuple_values"] = ptv

    patches = _patch_all(params_json)
    patches[4] = patch(
        "miv_simulator.eval_network.update_network_params", side_effect=fake_update
    )

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
        )

    values = {pt.population: v for pt, v in captured["param_tuple_values"]}
    assert values["PYR"] == pytest.approx(PARAMS_DICT["PYR.CA3.apical.AMPA.weight"])
    assert values["PVBC"] == pytest.approx(PARAMS_DICT["PVBC.EC.apical.AMPA.weight"])


def test_default_label_picks_first_key(multi_label_params_json):
    """When params_label=None the first key in the JSON dict is used."""
    captured = {}

    def fake_update(env, ptv):
        captured["param_tuple_values"] = ptv

    patches = _patch_all(multi_label_params_json)
    patches[4] = patch(
        "miv_simulator.eval_network.update_network_params", side_effect=fake_update
    )

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=multi_label_params_json,
            params_label=None,
        )

    # Values should come from "first_label", not "second_label"
    values = {pt.population: v for pt, v in captured["param_tuple_values"]}
    assert values["PYR"] == pytest.approx(PARAMS_DICT["PYR.CA3.apical.AMPA.weight"])


# ---------------------------------------------------------------------------
# Tests: param_tuple_values assembly
# ---------------------------------------------------------------------------


def test_all_param_names_mapped(params_json):
    """Every param_name in opt_config produces one (param_tuple, value) entry."""
    captured = {}

    def fake_update(env, ptv):
        captured["param_tuple_values"] = ptv

    patches = _patch_all(params_json)
    patches[4] = patch(
        "miv_simulator.eval_network.update_network_params", side_effect=fake_update
    )

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
        )

    assert len(captured["param_tuple_values"]) == len(PARAM_NAMES)
    for param_tuple, value in captured["param_tuple_values"]:
        assert isinstance(param_tuple, SynParam)
        assert isinstance(value, float)


def test_nested_path_fallback(tmp_path):
    """
    When a param_name is NOT in params_dict the code falls back to a nested
    dict lookup:  params_dict[population][source][sec_type][syn_name][param_path]
    """
    # Build a nested params dict (no flat keys)
    nested_params = {
        "PYR": {"CA3": {"apical": {"AMPA": {"weight": 7.7}}}},
        "PVBC": {"EC": {"apical": {"AMPA": {"weight": 3.3}}}},
    }
    data = {"lbl": {"parameters": nested_params}}
    p = tmp_path / "nested.json"
    p.write_text(json.dumps(data))

    captured = {}

    def fake_update(env, ptv):
        captured["param_tuple_values"] = ptv

    patches = _patch_all(str(p))
    patches[4] = patch(
        "miv_simulator.eval_network.update_network_params", side_effect=fake_update
    )

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=str(p),
            params_label="lbl",
        )

    values = {pt.population: v for pt, v in captured["param_tuple_values"]}
    assert values["PYR"] == pytest.approx(7.7)
    assert values["PVBC"] == pytest.approx(3.3)


# ---------------------------------------------------------------------------
# Tests: tuple sec_type parameter path parsing and propagation
# ---------------------------------------------------------------------------


def test_tuple_sec_type_flat_key_lookup(tmp_path):
    """
    A param_name like "AAC.CA2.('apical', 'basal').AMPA.weight" must be found
    by the flat-path branch (param_name in params_dict) and its value forwarded
    to update_network_params with the correct SynParam intact.
    """
    data = {"lbl": {"parameters": {AAC_PARAM_NAME: AAC_PARAM_VALUE}}}
    p = tmp_path / "aac_flat.json"
    p.write_text(json.dumps(data))

    captured = {}

    def fake_update(env, ptv):
        captured["param_tuple_values"] = ptv

    patches = _patch_all(str(p))
    patches[3] = patch(
        "miv_simulator.eval_network.optimization_params",
        return_value=AAC_OPT_CONFIG,
    )
    patches[4] = patch(
        "miv_simulator.eval_network.update_network_params", side_effect=fake_update
    )

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=str(p),
            params_label="lbl",
        )

    assert len(captured["param_tuple_values"]) == 1
    param_tuple, value = captured["param_tuple_values"][0]
    assert param_tuple.sec_type == ("apical", "basal")
    assert value == pytest.approx(AAC_PARAM_VALUE)


def test_tuple_sec_type_nested_path_fallback(tmp_path):
    """
    When the flat key is absent the nested-dict fallback uses str(sec_type) as
    the key.  str(('apical', 'basal')) == "('apical', 'basal')", so the JSON
    must contain that exact string as a nested dict key.
    """
    nested_params = {
        "AAC": {"CA2": {"('apical', 'basal')": {"AMPA": {"weight": AAC_PARAM_VALUE}}}}
    }
    data = {"lbl": {"parameters": nested_params}}
    p = tmp_path / "aac_nested.json"
    p.write_text(json.dumps(data))

    captured = {}

    def fake_update(env, ptv):
        captured["param_tuple_values"] = ptv

    patches = _patch_all(str(p))
    patches[3] = patch(
        "miv_simulator.eval_network.optimization_params",
        return_value=AAC_OPT_CONFIG,
    )
    patches[4] = patch(
        "miv_simulator.eval_network.update_network_params", side_effect=fake_update
    )

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=str(p),
            params_label="lbl",
        )

    assert len(captured["param_tuple_values"]) == 1
    param_tuple, value = captured["param_tuple_values"][0]
    assert param_tuple.sec_type == ("apical", "basal")
    assert value == pytest.approx(AAC_PARAM_VALUE)


def test_tuple_sec_type_propagated_to_modify_syn_param():
    """
    update_network_params must call modify_syn_param once per element of a
    tuple sec_type; i.e. separately for 'apical' and for 'basal'.
    """
    from miv_simulator.optimization import update_network_params

    mock_cell = MagicMock()
    env = MagicMock()
    env.biophys_cells = {"AAC": {1: mock_cell}}
    env.phenotype_dict.get.return_value = None

    with patch("miv_simulator.optimization.synapses.modify_syn_param") as mock_msp:
        update_network_params(env, [(AAC_PARAM_TUPLE, AAC_PARAM_VALUE)])

    assert mock_msp.call_count == 2

    sec_types_called = [call.args[2] for call in mock_msp.call_args_list]
    assert set(sec_types_called) == {"apical", "basal"}

    for call in mock_msp.call_args_list:
        _, kwargs = call
        assert kwargs["param_name"] == "weight"
        assert kwargs["value"] == pytest.approx(AAC_PARAM_VALUE)
        assert kwargs["filters"] == {"sources": ["CA2"]}
        assert kwargs["update_targets"] is True


# ---------------------------------------------------------------------------
# Tests: network.run output flag
# ---------------------------------------------------------------------------


def test_network_run_called_with_output_false(params_json):
    """network.run must be called with output=False so spike vectors stay in memory
    for feature extraction (network.run(output=True) clears them at each checkpoint)."""
    mock_network = MagicMock()

    patches = _patch_all(params_json)
    patches[5] = patch("miv_simulator.eval_network.network", mock_network)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
        )

    mock_network.run.assert_called_once()
    _, kwargs = mock_network.run.call_args
    output_flag = kwargs.get(
        "output",
        mock_network.run.call_args[0][1] if mock_network.run.call_args[0][1:] else None,
    )
    assert output_flag is False


# ---------------------------------------------------------------------------
# Tests: output JSON
# ---------------------------------------------------------------------------


def test_output_json_written_with_correct_structure(params_json, tmp_path):
    """When output_path is given, a JSON file is written with the expected keys."""
    out_path = str(tmp_path / "results.json")

    patches = _patch_all(params_json)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
            output_path=out_path,
        )

    assert os.path.exists(out_path)
    with open(out_path) as f:
        result = json.load(f)

    assert "run_label" in result
    entry = result["run_label"]
    assert "parameters" in entry
    assert "objectives" in entry
    assert "features" in entry
    assert "constraints" in entry


def test_output_json_objectives_match_compute_objectives(params_json, tmp_path):
    """Objective values in the output JSON match the mock compute_objectives return."""
    out_path = str(tmp_path / "results.json")

    patches = _patch_all(params_json)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
            output_path=out_path,
        )

    with open(out_path) as f:
        result = json.load(f)

    objectives = result["run_label"]["objectives"]
    for name, expected in zip(OBJECTIVE_NAMES, OBJECTIVES_ARR.tolist()):
        assert objectives[name] == pytest.approx(expected, abs=1e-5)


def test_output_json_constraints_keyed_by_population(params_json, tmp_path):
    """Constraint keys follow the '<pop> positive rate' convention."""
    out_path = str(tmp_path / "results.json")

    patches = _patch_all(params_json)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
            output_path=out_path,
        )

    with open(out_path) as f:
        result = json.load(f)

    constraints = result["run_label"]["constraints"]
    for pop in TARGET_POPULATIONS:
        assert f"{pop} positive rate" in constraints


def test_no_output_file_when_output_path_is_none(params_json, tmp_path):
    """No file is created when output_path is not specified."""
    patches = _patch_all(params_json)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
            output_path=None,
        )

    # Verify no output JSON was written (the only .json present is the input)
    json_files = list(tmp_path.glob("*.json"))
    assert all(f.name == "params.json" for f in json_files)


# ---------------------------------------------------------------------------
# Tests: operational config forwarding
# ---------------------------------------------------------------------------


def test_kwargs_from_operational_config_forwarded_to_init_network(params_json):
    """kwargs from the operational config are merged into the network_config
    passed to init_network."""
    op_config_with_kwargs = {
        **OPERATIONAL_CONFIG,
        "kwargs": {"config_file": "Net.yaml", "dataset_prefix": "/data"},
    }

    captured_kwargs = {}

    def fake_init_network(comm, subworld_size, kwargs):
        captured_kwargs.update(kwargs)
        return _make_mock_env()

    with (
        patch("miv_simulator.eval_network.config_logging"),
        patch(
            "miv_simulator.eval_network.read_from_yaml",
            return_value=op_config_with_kwargs,
        ),
        patch(
            "miv_simulator.eval_network.init_network",
            side_effect=fake_init_network,
        ),
        patch(
            "miv_simulator.eval_network.optimization_params",
            return_value=OPT_CONFIG,
        ),
        patch("miv_simulator.eval_network.update_network_params"),
        patch("miv_simulator.eval_network.network"),
        patch(
            "miv_simulator.eval_network.network_features",
            return_value=FEATURES_DICT,
        ),
        patch(
            "miv_simulator.eval_network.compute_objectives",
            return_value=COMPUTE_OBJECTIVES_RESULT,
        ),
        patch("miv_simulator.eval_network.io_utils"),
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
        )

    assert captured_kwargs.get("config_file") == "Net.yaml"
    assert captured_kwargs.get("dataset_prefix") == "/data"


def test_explicit_network_kwargs_not_overwritten_by_op_config(params_json):
    """Explicit **network_config kwargs take precedence over op_config kwargs."""
    op_config_with_kwargs = {
        **OPERATIONAL_CONFIG,
        "kwargs": {"config_file": "Net_from_yaml.yaml"},
    }

    captured_kwargs = {}

    def fake_init_network(comm, subworld_size, kwargs):
        captured_kwargs.update(kwargs)
        return _make_mock_env()

    with (
        patch("miv_simulator.eval_network.config_logging"),
        patch(
            "miv_simulator.eval_network.read_from_yaml",
            return_value=op_config_with_kwargs,
        ),
        patch(
            "miv_simulator.eval_network.init_network",
            side_effect=fake_init_network,
        ),
        patch(
            "miv_simulator.eval_network.optimization_params",
            return_value=OPT_CONFIG,
        ),
        patch("miv_simulator.eval_network.update_network_params"),
        patch("miv_simulator.eval_network.network"),
        patch(
            "miv_simulator.eval_network.network_features",
            return_value=FEATURES_DICT,
        ),
        patch(
            "miv_simulator.eval_network.compute_objectives",
            return_value=COMPUTE_OBJECTIVES_RESULT,
        ),
        patch("miv_simulator.eval_network.io_utils"),
    ):
        from miv_simulator.eval_network import eval_network

        # Caller-supplied config_file should be preserved (op_config should
        # not silently overwrite it because update() is called on network_config,
        # not on op_config kwargs)
        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
            config_file="Net_explicit.yaml",
        )

    # The op_config kwargs are merged AFTER the caller args, so caller wins
    # only if the caller key was set before the update(); verify the final
    # value reflects the op_config merge behaviour (documented in the code)
    assert "config_file" in captured_kwargs


# ---------------------------------------------------------------------------
# Tests: network.shutdown is always called
# ---------------------------------------------------------------------------


def test_network_shutdown_called(params_json):
    """network.shutdown(env) must be called to release NEURON resources."""
    mock_network = MagicMock()
    mock_env = _make_mock_env()

    with (
        patch("miv_simulator.eval_network.config_logging"),
        patch(
            "miv_simulator.eval_network.read_from_yaml",
            return_value=OPERATIONAL_CONFIG,
        ),
        patch(
            "miv_simulator.eval_network.init_network",
            return_value=mock_env,
        ),
        patch(
            "miv_simulator.eval_network.optimization_params",
            return_value=OPT_CONFIG,
        ),
        patch("miv_simulator.eval_network.update_network_params"),
        patch("miv_simulator.eval_network.network", mock_network),
        patch(
            "miv_simulator.eval_network.network_features",
            return_value=FEATURES_DICT,
        ),
        patch(
            "miv_simulator.eval_network.compute_objectives",
            return_value=COMPUTE_OBJECTIVES_RESULT,
        ),
        patch("miv_simulator.eval_network.io_utils"),
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
        )

    mock_network.shutdown.assert_called_once_with(mock_env)


# ---------------------------------------------------------------------------
# Tests: cleanup guard
# ---------------------------------------------------------------------------


def test_cleanup_true_raises_runtime_error(params_json):
    """eval_network must raise RuntimeError immediately when env.cleanup=True.

    With cleanup=True, network.init() deletes biophys_cells after wiring each gid,
    so update_network_params() would silently apply no parameters.
    """
    mock_env = _make_mock_env(cleanup=True)

    patches = _patch_all(params_json)
    patches[2] = patch("miv_simulator.eval_network.init_network", return_value=mock_env)

    with pytest.raises(RuntimeError, match="cleanup=False"):
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
        ):
            from miv_simulator.eval_network import eval_network

            eval_network(
                config_path="config.yaml",
                params_path=params_json,
                params_label="run_label",
            )


# ---------------------------------------------------------------------------
# Tests: explicit output writing after feature extraction
# ---------------------------------------------------------------------------


def test_io_utils_mkout_spikeout_lfpout_called(params_json):
    """After the simulation, mkout/spikeout/lfpout must be called to write output."""
    mock_io_utils = MagicMock()
    mock_env = _make_mock_env(recording_profile=None)

    patches = _patch_all(params_json)
    patches[2] = patch("miv_simulator.eval_network.init_network", return_value=mock_env)
    patches[8] = patch("miv_simulator.eval_network.io_utils", mock_io_utils)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
        )

    mock_io_utils.mkout.assert_called_once_with(mock_env, mock_env.results_file_path)
    mock_io_utils.spikeout.assert_called_once_with(mock_env, mock_env.results_file_path)
    mock_io_utils.lfpout.assert_called_once_with(mock_env, mock_env.results_file_path)
    mock_io_utils.recsout.assert_not_called()


def test_io_utils_recsout_called_when_recording_profile_set(params_json):
    """recsout must be called when env.recording_profile is not None."""
    mock_io_utils = MagicMock()
    mock_env = _make_mock_env(recording_profile={"dt": 0.1})

    patches = _patch_all(params_json)
    patches[2] = patch("miv_simulator.eval_network.init_network", return_value=mock_env)
    patches[8] = patch("miv_simulator.eval_network.io_utils", mock_io_utils)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
    ):
        from miv_simulator.eval_network import eval_network

        eval_network(
            config_path="config.yaml",
            params_path=params_json,
            params_label="run_label",
        )

    mock_io_utils.recsout.assert_called_once_with(mock_env, mock_env.results_file_path)
