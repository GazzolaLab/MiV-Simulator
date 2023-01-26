"""Simulation of opsin receptors at the channel, cell, and network level. 
Based on code from the PyRhO: A Multiscale Optogenetics Simulation Platform
https://github.com/ProjectPyRhO/PyRhO.git
"""

from typing import TYPE_CHECKING, Dict, Set, Tuple
from collections import OrderedDict, defaultdict
import abc
import copy
import logging
import math
import numpy as np
from neuron import h
from miv_simulator.utils import (
    get_module_logger,
)

if TYPE_CHECKING:
    from neuron.hoc import HocObject

logger = get_module_logger(__name__)


def cycles2times(cycles, Dt_delay):
    r"""
    Convert pulse cycles (durations) to times (absolute events).

    Parameters
    ----------
    cycles : list or array
        Array of cycles and delay duration i.e.:
        :math:`[[\Delta t_{on,0}, \Delta t_{off,0}],
                ...,
                [\Delta t_{on,N-1}, \Delta t_{off,N-1}]]`.
    Dt_delay : float
        Delay duration, :math:`\Delta t_{delay}`

    Returns
    -------
    tuple
        List of pulse times and total protocol time aligned at t0 = 0 i.e.:
        :math:`([[t_{on,0}, t_{off,0}], ..., [t_{on,N-1}, t_{off,N-1}]], \Delta t_{total})`.
    """

    # TODO: Generalise to Dt_delays c.f. recovery
    cycles = np.array(cycles)
    nPulses = cycles.shape[0]
    assert cycles.shape[1] <= 2
    times = np.cumsum(np.r_[Dt_delay, cycles.ravel()])
    Dt_total = times[-1]
    times = times[:-1].reshape((nPulses, 2))  # Trim the final Dt_off & reshape
    # times = np.zeros((nPulses, 2)) #[Dt_delay,Dt_delay+cycles[row,0] for row in pulses]
    # Dt_total = Dt_delay
    # for p in range(nPulses):
    #    times[p, 0] = Dt_total
    #    times[p, 1] = Dt_total+cycles[p, 0]
    #    Dt_total += sum(cycles[p, :])
    return (times, Dt_total)


default_opsin_type = "ChR2"
rho_type = default_opsin_type
