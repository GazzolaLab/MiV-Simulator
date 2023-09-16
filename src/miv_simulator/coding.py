from miv_simulator import typing as st
import numpy as np


def spike_times_2_binary_sparse_spike_train(
    array: st.SpikeTimesLike, temporal_resolution: float
) -> st.BinarySparseSpikeTrain:
    a = st.cast_spike_times(array)
    bins = np.floor(a / temporal_resolution).astype(int)
    # since a is sorted, maximum is last value
    spike_train = np.zeros(bins[-1] + 1, dtype=np.int8)
    spike_train[bins] = 1
    return spike_train


def binary_sparse_spike_train_2_spike_times(
    array: st.BinarySparseSpikeTrainLike, temporal_resolution: float
) -> st.SpikeTimes:
    a = st.cast_binary_sparse_spike_train(array)
    spike_indices = np.where(a == 1)[0]
    spike_times = spike_indices * temporal_resolution
    return spike_times


def adjust_temporal_resolution(
    array: st.BinarySparseSpikeTrainLike,
    original_resolution: float,
    target_resolution: float,
) -> st.BinarySparseSpikeTrain:
    a = st.cast_binary_sparse_spike_train(array)

    ratio = target_resolution / original_resolution
    if ratio == 1:
        return a

    new_length = int(a.shape[0] * ratio)
    new_spike_train = np.zeros(new_length, dtype=np.int8)

    # up
    if ratio > 1:
        for idx, val in enumerate(a):
            start = int(idx * ratio)
            end = int((idx + 1) * ratio)
            new_spike_train[start:end] = val

    # down
    elif ratio < 1:
        for idx in range(0, len(a), int(1 / ratio)):
            if np.any(a[idx : idx + int(1 / ratio)]):
                new_spike_train[idx // int(1 / ratio)] = 1

    return new_spike_train
