from numpy.typing import NDArray
import numpy as np
from typing import Annotated as EventArray, Dict

"""Potentially unsorted or scalar data that can be transformed into `SpikeTimes`"""
SpikeTimesLike = EventArray[NDArray[np.float_], "SpikeTimesLike ..."]

"""Sorted array of absolute spike times"""
SpikeTimes = EventArray[NDArray[np.float_], "SpikeTimes T ..."]

# spike train encodings (RLE, delta encoding, variable time binning etc.)

"""Binary data that can be cast to the `BinarySparseSpikeTrain` format"""
BinarySparseSpikeTrainLike = EventArray[
    NDArray, "BinarySparseSpikeTrainLike ..."
]

"""Binary spike train representation for a given temporal resolution"""
BinarySparseSpikeTrain = EventArray[
    NDArray[np.int8], "BinarySparseSpikeTrain t_bin ..."
]


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
