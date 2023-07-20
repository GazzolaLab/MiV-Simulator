import numpy as np
from numpy import ndarray
from typing import Tuple


class RateEncoder:
    def __init__(
        self,
        time_window: int = 100,
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

    def encode(self, signal: ndarray) -> ndarray:
        assert (
            len(signal.shape) == 2
        ), "encode requires input signal of shape number_samples x input_dimensions"

        nsamples = signal.shape[0]
        ndim = signal.shape[1]

        total_spikes = []
        for r in range(ndim):
            s = signal[:, r]
            freq = np.interp(
                s,
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
            total_spikes.append(spikes)

        return np.stack(total_spikes, axis=1)


class PoissonRateEncoder:
    def __init__(
        self,
        time_window: int = 100,
        input_range: Tuple[int, int] = (0, 255),
        output_freq_range: Tuple[int, int] = (0, 200),
        generator: None = None,
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

    def encode(self, signal: ndarray) -> ndarray:
        assert (
            len(signal.shape) == 2
        ), "encode requires input signal of shape number_samples x input_dimensions"

        nsamples = signal.shape[0]
        ndim = signal.shape[1]

        total_spikes = []
        for r in range(ndim):
            spike_train = []
            s = signal[:, r]
            freq = np.interp(
                s,
                [self.min_input, self.max_input],
                [self.min_output, self.max_output],
            )

            spikes = np.random.uniform(
                0, 1, nsamples * self.time_window
            ).reshape((nsamples, self.time_window))
            dt = 0.001  # second
            for i in range(nsamples):
                spikes[i, np.where(spikes < freq[i] * dt)] = 1
                spikes[i, np.where(spikes[i] != 1)] = 0

            total_spikes.append(spikes)

        return np.stack(total_spikes, axis=1)
