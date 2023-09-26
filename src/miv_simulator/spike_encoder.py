import numpy as np
from numpy import ndarray
from typing import Tuple, Optional, Iterable, Iterator

def rate_generator(
    signal: Union[ndarray, Iterable[ndarray]],
    time_window: int = 100,
    dt: float = 0.02,
    **kwargs,
) -> Iterator[ndarray]:
    """
    Lazily invokes ``RateEncoder`` to iteratively encode a sequence of
    data.

    :param data: NDarray of shape ``[n_samples, n_1, ..., n_k]``.
    :param time_window: Length of Poisson spike train per input variable.
    :param dt: Spike generator time step.
    :return: NDarray of shape ``[time, n_1, ..., n_k]`` of rate-encoded spikes.
    """
    encoder = RateEncoder(time_window=time_window, dt=dt)
    for chunk in signal:
        yield encoder.encode(chunk)


def poisson_rate_generator(
    signal: Union[ndarray, Iterable[ndarray]],
    time_window: int = 100,
    dt: float = 0.02,
    **kwargs,
) -> Iterator[ndarray]:
    """
    Lazily invokes ``PoissonEncoder`` to iteratively encode a sequence of
    data.

    :param data: NDarray of shape ``[n_samples, n_1, ..., n_k]``.
    :param time_window: Length of Poisson spike train per input variable.
    :param dt: Spike generator time step.
    :return: NDarray of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    encoder = PoissonRateEncoder(time_window=time_window, dt=dt)
    for chunk in signal:
        yield encoder.encode(chunk)




class RateEncoder:
    def __init__(
        self,
        dt: float = 0.02,
        input_range: Tuple[int, int] = (0, 255),
        output_freq_range: Tuple[int, int] = (0, 200),
    ) -> None:
        assert input_range[1] - input_range[0] > 0
        assert output_freq_range[1] - output_freq_range[0] > 0
        assert time_window > 0
        self.min_input, self.max_input = input_range[0], input_range[1]
        self.min_output, self.max_output = (
            output_freq_range[0],
            output_freq_range[1],
        )
        self.time_window = time_window
        self.ndim = 1

    def encode(self, signal: ndarray) -> ndarray:
        assert (
            len(signal.shape) == 2
        ), "encode requires input signal of shape number_samples x input_dimensions"

        nsamples = signal.shape[0]
        ndim = signal.shape[1]
        assert (
            ndim == self.ndim
        ), f"input signal has dimension {ndim} but encoder has input dimension {self.ndim}"

        freq = np.interp(
            signal,
            [self.min_input, self.max_input],
            [self.min_output, self.max_output],
        )
        nz = np.argwhere(freq > 0)
        period = np.zeros(nsamples)
        period[nz] = (1 / freq[nz]) * 1000  # ms
        spikes = np.zeros((nsamples, self.time_window))
        for i in range(nsamples):
            if period[i] > 0:
                stride = int(period[i])
                spikes[i, 0 : self.time_window : stride] = 1

        return spikes


class PoissonRateEncoder:
    def __init__(
        self,
        time_window: int = 100,
        input_range: Tuple[int, int] = (0, 255),
        output_freq_range: Tuple[int, int] = (0, 200),
        generator: Optional[np.random.RandomState] = None,
    ) -> None:
        assert input_range[1] - input_range[0] > 0
        assert output_freq_range[1] - output_freq_range[0] > 0
        assert time_window > 0
        self.min_input, self.max_input = input_range[0], input_range[1]
        self.min_output, self.max_output = (
            output_freq_range[0],
            output_freq_range[1],
        )
        self.time_window = time_window
        if generator is None:
            generator = np.random
        self.generator = generator
        self.ndim = 1

    def encode(self, signal: ndarray) -> ndarray:
        assert (
            len(signal.shape) == 2
        ), "encode requires input signal of shape number_samples x input_dimensions"

        nsamples = signal.shape[0]
        ndim = signal.shape[1]
        assert (
            ndim == self.ndim
        ), f"input signal has dimension {ndim} but encoder has input dimension {self.ndim}"

        spike_train = []
        freq = np.interp(
            signal,
            [self.min_input, self.max_input],
            [self.min_output, self.max_output],
        )

        spikes = self.generator.uniform(
            0, 1, nsamples * self.time_window
        ).reshape((nsamples, self.time_window))
        dt = 0.001  # second
        for i in range(nsamples):
            spikes[i, np.where(spikes < freq[i] * dt)] = 1
            spikes[i, np.where(spikes[i] != 1)] = 0

        return spikes
