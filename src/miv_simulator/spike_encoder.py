import numpy as np
from numpy import ndarray
from typing import Tuple, Optional, Iterable, Iterator


def rate_generator(
    signal: Union[ndarray, Iterable[ndarray]],
    t_start: float = 0.0,
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
    t_start_ = t_start
    encoder = RateEncoder(time_window=time_window, dt=dt, **kwargs)
    for chunk in signal:
        output, t_next = encoder.encode(chunk, t_start=t_start_)
        yield output
        t_start_ = t_next


def poisson_rate_generator(
    signal: Union[ndarray, Iterable[ndarray]],
    t_start: float = 0.0,
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
    t_start_ = t_start
    encoder = PoissonRateEncoder(time_window=time_window, dt=dt, **kwargs)
    for chunk in signal:
        output, t_next = encoder.encode(chunk, t_start=t_start_)
        yield output
        t_start_ = t_next


class RateEncoder:
    def __init__(
        self,
        time_window: int = 100,
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
        self.spike_array = None

    def encode(
        self,
        signal: ndarray,
        return_spike_array: bool = False,
        t_start: Optional[float] = None,
    ) -> ndarray:
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
        if (
            (self.spike_array is None)
            or (self.spike_array.shape[0] != nsamples)
            or (self.spike_array.shape[1] != ndim)
        ):
            self.spike_array = np.zeros(
                (nsamples, self.time_window), dtype=bool
            )
        else:
            self.spike_array.fill(0)
        for i in range(nsamples):
            if period[i] > 0:
                stride = int(period[i])
                self.spike_array[i, 0 : self.time_window : stride] = 1

        t_next = None
        if t_start is not None:
            t_next = t_start + self.time_window * nsamples * self.dt

        if return_spike_array:
            return np.copy(self.spike_array), t_next
        else:
            if t_start is None:
                t_start = 0.0
            spike_times = []
            spike_inds = np.argwhere[spike_array[i] == 1]
            for i in range(nsamples):
                this_spike_inds = spike_inds[
                    np.argwhere(spike_inds[:, 0] == i).flat
                ]
                this_spike_times = []
                if len(this_spike_inds) > 0:
                    this_spike_times = (
                        t_start
                        + np.asarray(this_spike_inds[:, 1], dtype=np.float32)
                        * self.dt
                    )
                spike_times.append(this_spike_times)
            return spike_times, t_next


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

    def encode(
        self,
        signal: ndarray,
        return_spike_array: bool = False,
        t_start: Optional[float] = None,
    ) -> ndarray:
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

        spike_array = self.generator.uniform(
            0, 1, nsamples * self.time_window
        ).reshape((nsamples, self.time_window))
        dt = 0.001  # second
        for i in range(nsamples):
            spike_array[i, np.where(spike_array < freq[i] * dt)] = 1
            spike_array[i, np.where(spike_array[i] != 1)] = 0

        t_next = None
        if t_start is not None:
            t_next = t_start + self.time_window * nsamples * self.dt

        if return_spike_array:
            return np.copy(self.spike_array), t_next
        else:
            if t_start is None:
                t_start = 0.0
            spike_times = []
            spike_inds = np.argwhere[spike_array[i] == 1]
            for i in range(nsamples):
                this_spike_inds = spike_inds[
                    np.argwhere(spike_inds[:, 0] == i).flat
                ]
                this_spike_times = []
                if len(this_spike_inds) > 0:
                    this_spike_times = (
                        t_start
                        + np.asarray(this_spike_inds[:, 1], dtype=np.float32)
                        * self.dt
                    )
                spike_times.append(this_spike_times)
            return spike_times, t_next
