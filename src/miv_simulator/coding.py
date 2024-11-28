import numpy as np
from numpy.typing import NDArray
from typing import Annotated as EventArray, Dict

SpikeTimesLike = EventArray[NDArray[np.float_], "SpikeTimesLike ..."]
"""Potentially unsorted or scalar data that can be transformed into `SpikeTimes`"""

SpikeTimes = EventArray[NDArray[np.float64], "SpikeTimes T ..."]
"""Sorted array of absolute spike times"""


# Spike train encodings (RLE, delta encoding, variable time binning etc.)

BinarySparseSpikeTrainLike = EventArray[NDArray, "BinarySparseSpikeTrainLike ..."]
"""Binary data that can be cast to the `BinarySparseSpikeTrain` format"""


BinarySparseSpikeTrain = EventArray[
    NDArray[np.int8], "BinarySparseSpikeTrain t_bin ..."
]
"""Binary spike train representation for a given temporal resolution"""


def _inspect(type_) -> Dict:
    annotation = type_.__metadata__[0]
    name, *dims = annotation.split(" ")

    return {
        "annotation": annotation,
        "name": name,
        "dims": dims,
        "dtype": type_.__origin__.__args__[1].__args__[0],
    }


def _cast(a, a_type, r_type):  # -> r_type
    a_t, r_t = _inspect(a_type), _inspect(r_type)
    if a_t["name"].replace("Like", "") != r_t["name"]:
        raise ValueError(
            f"Expected miv_simulator.typing.{r_t['name']}Like but found {a_t['name']}"
        )
    v = np.array(a, dtype=r_t["dtype"])
    if len(v.shape) == 0:
        return np.reshape(
            v,
            [
                1,
            ],
        )
    return v


def cast_spike_times(a: SpikeTimesLike) -> SpikeTimes:
    return np.sort(_cast(a, SpikeTimesLike, SpikeTimes), axis=0)


def cast_binary_sparse_spike_train(
    a: BinarySparseSpikeTrainLike,
) -> BinarySparseSpikeTrain:
    return _cast(a, BinarySparseSpikeTrainLike, BinarySparseSpikeTrain)


def spike_times_2_binary_sparse_spike_train(
    array: SpikeTimesLike, temporal_resolution: float
) -> BinarySparseSpikeTrain:
    a = cast_spike_times(array)
    bins = np.floor(a / temporal_resolution).astype(int)
    # since a is sorted, maximum is last value
    spike_train = np.zeros(bins[-1] + 1, dtype=np.int8)
    spike_train[bins] = 1
    return spike_train


def binary_sparse_spike_train_2_spike_times(
    array: BinarySparseSpikeTrainLike, temporal_resolution: float
) -> SpikeTimes:
    a = cast_binary_sparse_spike_train(array)
    spike_indices = np.where(a == 1)[0]
    spike_times = spike_indices * temporal_resolution
    return spike_times


def adjust_temporal_resolution(
    array: BinarySparseSpikeTrainLike,
    original_resolution: float,
    target_resolution: float,
) -> BinarySparseSpikeTrain:
    a = cast_binary_sparse_spike_train(array)

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
