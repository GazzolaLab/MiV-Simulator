from miv_simulator import typing as t
import numpy as np


def test_typing_cast():
    assert t.cast_spike_times(0.5).shape == (1,)
    assert t.cast_spike_times([0.5, 0.1])[1] == 0.5
    assert t.cast_spike_times(int(1))[0] == float(1.0)

    assert t.cast_binary_sparse_spike_train(0.1)[0] == 0
