from miv_simulator import coding as t
import numpy as np
import miv_simulator.typing as st


def test_coding_spike_times_vs_binary_sparse_spike_train():
    for a, b in [
        ([0.1, 0.3, 0.4, 0.85], [1, 1]),
        ([0.8], [0, 1]),
    ]:
        result = t.spike_times_2_binary_sparse_spike_train(a, 0.5)
        expected = np.array(b, dtype=np.int8)
        assert np.array_equal(result, expected)

    for a, b in [
        ([1, 0, 1], [0.0, 1.0]),
        ([0, 1], [0.5]),
    ]:
        spike_train = np.array(a, dtype=np.int8)
        result = t.binary_sparse_spike_train_2_spike_times(spike_train, 0.5)
        expected = np.array(b)
        assert np.array_equal(result, expected)


def test_coding_adjust_temporal_resolution():
    spike_train = np.array([0, 1, 0, 1, 0], dtype=np.int8)

    # identity
    adjusted = t.adjust_temporal_resolution(spike_train, 1, 1)
    assert np.array_equal(adjusted, spike_train)

    # up
    adjusted = t.adjust_temporal_resolution(spike_train, 0.5, 1)
    expected = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    assert np.array_equal(adjusted, expected)

    # down
    adjusted = t.adjust_temporal_resolution(spike_train, 2, 1)
    expected = np.array([1, 1], dtype=np.int8)
    assert np.array_equal(adjusted, expected)
