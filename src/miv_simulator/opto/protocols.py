"""
Stimulation protocols to run on the opsin models
    * Neuro-engineering stimuli: ``step``, ``sinusoid``, ``ramp``, ``delta``
    * Opsin-specific protocols: ``shortPulse``, ``recovery``.
    * The ``custom`` protocol can be used with arbitrary interpolation fuctions

Based on code from the PyRhO: A Multiscale Optogenetics Simulation Platform
https://github.com/ProjectPyRhO/PyRhO.git
"""

import warnings
import logging
import os
import abc
from collections import OrderedDict
import numpy as np
import quantities as pq
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from miv_simulator.utils import (
    Struct,
    get_module_logger,
)
from miv_simulator.opto.core import cycles2times


logger = get_module_logger(__name__)


class Protocol(Struct):
    """Common base class for all protocols."""

    __metaclass__ = abc.ABCMeta

    protocol = None
    nRuns = None
    Dt_delay = None
    cycles = None
    Dt_total = None
    dt = None
    phis = None

    def __init__(self, params=None):
        if params is None:
            params = default_protocol_parameters[self.protocol]
        self.update(params)
        self.prepare()
        self.t_start, self.t_end = 0, self.Dt_total
        self.phi_ts = None
        self.lam = 470  # Default wavelength [nm]

    def __str__(self):
        return self.protocol

    def __repr__(self):
        return "<PyRhO {} Protocol object (nRuns={}, nPhis={})>".format(
            self.protocol, self.nRuns, self.nPhis
        )

    def __iter__(self):
        """Iterator to return the pulse sequence for the next trial."""
        self.run = 0
        self.phiInd = 0
        self.vInd = 0
        return self

    def __next__(self):
        """Iterator to return the pulse sequence for the next trial."""
        self.run += 1
        if self.run > self.nRuns:
            raise StopIteration
        return self.getRunCycles(self.run - 1)

    def prepare(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """

        if np.isscalar(self.cycles):  # Only 'on' duration specified
            Dt_on = self.cycles
            if hasattr(self, "Dt_total"):
                Dt_off = self.Dt_total - Dt_on - self.Dt_delay
            else:
                Dt_off = 0
            self.cycles = np.asarray([[Dt_on, Dt_off]])
        elif isinstance(self.cycles, (list, tuple, np.ndarray)):
            if np.isscalar(self.cycles[0]):
                self.cycles = [self.cycles]  # Assume only one pulse
        else:
            raise TypeError(
                "Unexpected type for cycles - expected a list or array!"
            )

        self.cycles = np.asarray(self.cycles)
        self.nPulses = self.cycles.shape[0]
        self.pulses, self.Dt_total = cycles2times(self.cycles, self.Dt_delay)
        self.Dt_delays = np.array(
            [pulse[0] for pulse in self.pulses], copy=True
        )  # pulses[:,0]    # Delay Durations #self.Dt_delays = np.array([self.Dt_delay] * self.nRuns)
        self.Dt_ons = np.array(
            self.cycles[:, 0]
        )  # self.Dt_ons = np.array([cycle[0] for cycle in self.cycles])
        self.Dt_offs = np.array(
            self.cycles[:, 1]
        )  # self.Dt_offs = np.array([cycle[1] for cycle in self.cycles])

        if np.isscalar(self.phis):
            self.phis = [self.phis]  # np.asarray([self.phis])
        logger.info(f"self.phis = {self.phis}")
        self.phis.sort(reverse=True)
        self.nPhis = len(self.phis)

        self.extraPrep()
        return

    def extraPrep(self):
        pass

    def getShortestPeriod(self):
        # min(self.Dt_delay, min(min(self.cycles)))
        return np.amin(self.cycles[self.cycles.nonzero()])

    def getRunCycles(self, run):
        return (self.cycles, self.Dt_delay)

    def genPulseSet(self, genPulse=None):
        """Function to generate a set of spline functions to phi(t) simulations."""
        if genPulse is None:  # Default to square pulse generator
            genPulse = self.genPulse
        phi_ts = [
            [
                [None for pulse in range(self.nPulses)]
                for phi in range(self.nPhis)
            ]
            for run in range(self.nRuns)
        ]
        for run in range(self.nRuns):
            cycles, Dt_delay = self.getRunCycles(run)
            pulses, Dt_total = cycles2times(cycles, Dt_delay)
            for phiInd, phi in enumerate(self.phis):
                for pInd, pulse in enumerate(pulses):
                    phi_ts[run][phiInd][pInd] = genPulse(run, phi, pulse)
        self.phi_ts = phi_ts
        return phi_ts

    def genPulse(self, run, phi, pulse):
        """Default interpolation function for square pulses."""
        pStart, pEnd = pulse
        phi_t = spline([pStart, pEnd], [phi, phi], k=1, ext=1)
        return phi_t

    def getStimArray(self, run, phiInd, dt):  # phi_ts, Dt_delay, cycles, dt):
        """Return a stimulus array (not spline) with the same sampling rate as
        the photocurrent.
        """

        cycles, Dt_delay = self.getRunCycles(run)
        phi_ts = self.phi_ts[run][phiInd][:]

        nPulses = cycles.shape[0]
        assert len(phi_ts) == nPulses

        # start, end = RhO.t[0], RhO.t[0]+Dt_delay #start, end = 0.00, Dt_delay
        start, end = 0, Dt_delay
        nSteps = int(round(((end - start) / dt) + 1))
        t = np.linspace(start, end, nSteps, endpoint=True)
        phi_tV = np.zeros_like(t)
        # _idx_pulses_ = np.empty([0,2],dtype=int) # Light on and off indexes for each pulse

        for p in range(nPulses):
            start = end
            Dt_on, Dt_off = cycles[p, 0], cycles[p, 1]
            end = start + Dt_on + Dt_off
            nSteps = int(round(((end - start) / dt) + 1))
            tPulse = np.linspace(start, end, nSteps, endpoint=True)
            phi_t = phi_ts[p]
            phiPulse = phi_t(
                tPulse
            )  # -tPulse[0] # Align time vector to 0 for phi_t to work properly

            # onInd = len(t) - 1 # Start of on-phase
            # offInd = onInd + int(round(Dt_on/dt))
            # _idx_pulses_ = np.vstack((_idx_pulses_, [onInd,offInd]))

            # t = np.r_[t, tPulse[1:]]

            phi_tV = np.r_[phi_tV, phiPulse[1:]]

        phi_tV[np.ma.where(phi_tV < 0)] = 0  # Safeguard for negative phi values
        return phi_tV  # , t, _idx_pulses_


class ProtCustom(Protocol):
    """Present a time-varying stimulus defined by a spline function."""

    # Class attributes
    protocol = "custom"
    squarePulse = False
    # custPulseGenerator = None
    phi_ft = None

    def extraPrep(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """
        self.nRuns = 1  # nRuns ### TODO: Reconsider this...

        # self.custPulseGenerator = self.phi_ft
        if not hasattr(self, "phi_ts") or self.phi_ts is None:
            # self.phi_ts = self.genPulseSet()
            # self.genPulseSet(self.custPulseGenerator)
            self.genPulseSet(self.phi_ft)


class ProtSinusoid(Protocol):
    """Present oscillating stimuli over a range of frequencies to find the
    resonant frequency.
    """

    protocol = "sinusoid"
    squarePulse = False
    startOn = False
    phi0 = 0

    def extraPrep(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """

        self.fs = np.sort(np.array(self.fs))  # Frequencies [Hz]
        self.ws = (
            2 * np.pi * self.fs / (1000)
        )  # Frequencies [rads/ms] (scaled from /s to /ms
        self.sr = max(
            10000, int(round(10 * max(self.fs)))
        )  # Nyquist frequency - sampling rate (10*f) >= 2*f >= 10/ms
        # self.dt = 1000/self.sr # dt is set by simulator but used for plotting
        self.nRuns = len(self.ws)

        if (1000) / min(self.fs) > min(self.Dt_ons):
            warnings.warn(
                "Warning: The period of the lowest frequency is longer than the stimulation time!"
            )

        if isinstance(self.phi0, (int, float, complex)):
            self.phi0 = np.ones(self.nRuns) * self.phi0
        elif isinstance(self.phi0, (list, tuple, np.ndarray)):
            if len(self.phi0) != self.nRuns:
                self.phi0 = np.ones(self.nRuns) * self.phi0[0]
        else:
            warnings.warn("Unexpected data type for phi0: ", type(self.phi0))

        assert len(self.phi0) == self.nRuns

        self.t_start, self.t_end = 0, self.Dt_total
        self.phi_ts = self.genPulseSet()
        self.runLabels = [
            r"$f={}\mathrm{{Hz}}$ ".format(round_sig(f, 3)) for f in self.fs
        ]

    def getShortestPeriod(self):
        return 1000 / self.sr  # dt [ms]

    def genPulse(self, run, phi, pulse):
        pStart, pEnd = pulse
        Dt_on = pEnd - pStart
        t = np.linspace(
            0.0, Dt_on, int(round((Dt_on * self.sr / 1000)) + 1), endpoint=True
        )  # Create smooth series of time points to interpolate between
        if self.startOn:  # Generalise to phase offset
            phi_t = spline(
                pStart + t,
                self.phi0[run] + 0.5 * phi * (1 + np.cos(self.ws[run] * t)),
                ext=1,
                k=5,
            )
        else:
            phi_t = spline(
                pStart + t,
                self.phi0[run] + 0.5 * phi * (1 - np.cos(self.ws[run] * t)),
                ext=1,
                k=5,
            )

        return phi_t


class ProtStep(Protocol):
    """Present a step (Heaviside) pulse."""

    protocol = "step"
    squarePulse = True
    nRuns = 1

    def extraPrep(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """
        self.nRuns = 1
        self.phi_ts = self.genPulseSet()


class ProtRamp(Protocol):
    """Linearly increasing pulse."""

    protocol = "ramp"
    squarePulse = False
    nRuns = 1
    phi0 = 0

    def extraPrep(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """
        self.nRuns = 1  # nRuns # Make len(phi_ton)?
        self.cycles = np.column_stack((self.Dt_ons, self.Dt_offs))
        self.phi_ts = self.genPulseSet()

    def genPulse(self, run, phi, pulse):
        """Generate spline for a particular pulse. phi0 is the offset so
        decreasing ramps can be created with negative phi values.
        """
        pStart, pEnd = pulse
        phi_t = spline([pStart, pEnd], [self.phi0, self.phi0 + phi], k=1, ext=1)
        return phi_t


class ProtDelta(Protocol):
    # One very short, saturation intensity pulse e.g. 10 ns @ 100 mW*mm^-2 for wild type ChR
    # Used to calculate gbar, assuming that O(1)-->1 as Dt_on-->0 and phi-->inf
    protocol = "delta"
    squarePulse = True
    nRuns = 1
    Dt_on = 0

    def prepare(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """
        assert self.Dt_total >= self.Dt_delay + self.Dt_on  # ==> Dt_off >= 0
        self.cycles = np.asarray(
            [[self.Dt_on, self.Dt_total - self.Dt_delay - self.Dt_on]]
        )
        self.nPulses = self.cycles.shape[0]
        self.pulses, self.Dt_total = cycles2times(self.cycles, self.Dt_delay)
        self.Dt_delays = np.array(
            [row[0] for row in self.pulses], copy=True
        )  # pulses[:,0]    # Delay Durations
        self.Dt_ons = [
            row[1] - row[0] for row in self.pulses
        ]  # pulses[:,1] - pulses[:,0]   # Pulse Durations
        self.Dt_offs = (
            np.append(self.pulses[1:, 0], self.Dt_total) - self.pulses[:, 1]
        )

        if np.isscalar(self.phis):
            self.phis = np.asarray([self.phis])
        self.phis.sort(reverse=True)
        self.nPhis = len(self.phis)

        self.addStimulus = config.addStimulus
        self.extraPrep()
        return

    def extraPrep(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """
        self.nRuns = 1
        self.phi_ts = self.genPulseSet()


class ProtShortPulse(Protocol):
    # Vary pulse length - See Nikolic et al. 2009, Fig. 2 & 9
    protocol = "shortPulse"
    squarePulse = True
    nPulses = 1  # Fixed at 1

    def prepare(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """
        self.pDs = np.sort(np.array(self.pDs))
        self.nRuns = len(self.pDs)
        self.Dt_delays = np.ones(self.nRuns) * self.Dt_delay
        self.Dt_ons = self.pDs
        self.Dt_offs = (
            (np.ones(self.nRuns) * self.Dt_total) - self.Dt_delays - self.Dt_ons
        )
        self.cycles = np.column_stack((self.Dt_ons, self.Dt_offs))
        self.phis.sort(reverse=True)
        self.nPhis = len(self.phis)
        self.phi_ts = self.genPulseSet()
        self.runLabels = [
            r"$\mathrm{{Pulse}}={}\mathrm{{ms}}$ ".format(pD) for pD in self.pDs
        ]

    def getRunCycles(self, run):
        return (
            np.asarray([[self.Dt_ons[run], self.Dt_offs[run]]]),
            self.Dt_delays[run],
        )


class ProtRecovery(Protocol):
    """Two pulse stimulation protocol with varying inter-pulse interval to
    determine the dark recovery rate.
    """

    # Vary Inter-Pulse-Interval
    protocol = "recovery"
    squarePulse = True
    nPulses = 2  # Fixed at 2 for this protocol

    Dt_on = 0
    # def __next__(self):
    # if self.run >= self.nRuns:
    # raise StopIteration
    # return np.asarray[self.pulses[self.run]]

    def prepare(self):
        """Function to set-up additional variables and make parameters
        consistent after any changes.
        """
        self.Dt_IPIs = np.sort(np.asarray(self.Dt_IPIs))

        self.nRuns = len(self.Dt_IPIs)
        self.Dt_delays = np.ones(self.nRuns) * self.Dt_delay
        self.Dt_ons = np.ones(self.nRuns) * self.Dt_on
        self.Dt_offs = self.Dt_IPIs
        # [:,0] = on phase duration; [:,1] = off phase duration
        self.cycles = np.column_stack((self.Dt_ons, self.Dt_offs))

        self.pulses, _ = cycles2times(self.cycles, self.Dt_delay)
        self.runCycles = np.zeros((self.nPulses, 2, self.nRuns))
        for run in range(self.nRuns):
            self.runCycles[:, :, run] = np.asarray(
                [
                    [self.Dt_ons[run], self.Dt_offs[run]],
                    [self.Dt_ons[run], self.Dt_offs[run]],
                ]
            )

        self.t_start = 0
        self.t_end = self.Dt_total
        IPIminD = (
            max(self.Dt_delays) + (2 * max(self.Dt_ons)) + max(self.Dt_IPIs)
        )
        if self.t_end < IPIminD:
            warnings.warn("Insufficient run time for all stimulation periods!")
        else:
            self.runCycles[-1, 1, :] = self.Dt_total - IPIminD

        self.IpIPI = np.zeros(self.nRuns)
        self.tpIPI = np.zeros(self.nRuns)

        if np.isscalar(self.phis):
            self.phis = np.asarray([self.phis])
        self.phis.sort(reverse=True)
        self.nPhis = len(self.phis)

        self.phi_ts = self.genPulseSet()
        self.runLabels = [
            r"$\mathrm{{IPI}}={}\mathrm{{ms}}$ ".format(IPI)
            for IPI in self.Dt_IPIs
        ]

    def getRunCycles(self, run):
        return self.runCycles[:, :, run], self.Dt_delays[run]


protocols = OrderedDict(
    [
        ("step", ProtStep),
        ("delta", ProtDelta),
        ("sinusoid", ProtSinusoid),
        ("ramp", ProtRamp),
        ("recovery", ProtRecovery),
        ("shortPulse", ProtShortPulse),
        ("custom", ProtCustom),
    ]
)

default_protocol_parameters = OrderedDict([])


def make_param_dict(*xs):
    result = {}
    for param in xs:
        result[param[0]] = param[1]
    return result


default_protocol_parameters["custom"] = make_param_dict(
    (
        "phis",
        [1e16, 1e17],
        0,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  #'photons/s/mm^2'
    (
        "Dt_delay",
        25,
        0,
        1e9,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  #'ms'
    (
        "cycles",
        [[150.0, 50.0]],
        0,
        None,
        pq.ms,
        "cycles",
        "List of [on, off] durations for each pulse",
    ),
)  # , #'ms'#,

default_protocol_parameters["step"] = make_param_dict(
    (
        "phis",
        [1e16, 1e17],
        0,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  #'photons/s/mm^2'
    (
        "Dt_delay",
        25,
        0,
        1e9,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  #'ms'
    (
        "cycles",
        [[150.0, 100.0]],
        0,
        None,
        pq.ms,
        "cycles",
        "List of [on, off] durations for each pulse",
    ),
)  #'ms'

default_protocol_parameters["sinusoid"] = make_param_dict(
    (
        "phis",
        [1e12],
        0,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  #'photons/s/mm^2'
    (
        "phi0",
        [0],
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\phi_0",
        "Constant offset for flux",
    ),  #'photons/s/mm^2'
    (
        "startOn",
        True,
        False,
        True,
        1,
        "\phi_{t=0}>0",
        "Start at maximum flux (else minimum)",
    ),
    (
        "fs",
        [0.1, 0.5, 1, 5, 10],
        0,
        None,
        pq.Hz,
        "\mathbf{f}",
        "List of modulation frequencies",
    ),  #'pq.Hz' #50, 100, 500, 1000
    (
        "Dt_delay",
        25,
        0,
        1e9,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  #'ms'
    (
        "cycles",
        [[10000.0, 50.0]],
        0,
        None,
        pq.ms,
        "cycles",
        "List of [on, off] durations for each pulse",
    ),
)  #'ms'

default_protocol_parameters["chirp"] = make_param_dict(
    (
        "phis",
        [1e12],
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  # 'photons/s/mm^2'
    (
        "phi0",
        [0],
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\phi_0",
        "Constant offset for flux",
    ),  # 'photons/s/mm^2'
    (
        "linear",
        True,
        False,
        True,
        1,
        "linear",
        "Linear frequency sweep (else exponential)",
    ),  # False := exponential
    (
        "startOn",
        False,
        False,
        True,
        1,
        "\phi_{t=0}>0",
        "Start at maximum flux (else minimum)",
    ),
    (
        "Dt_delay",
        100,
        0,
        1e9,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  # 'ms'
    (
        "cycles",
        [[10000.0, 100.0]],
        0,
        None,
        pq.ms,
        "cycles",
        "List of [on, off] durations for each pulse",
    ),  # 'ms'
    ("f0", 0.1, 0, None, pq.Hz, "f_0", "Starting frequency"),  # 'pq.Hz'
    ("fT", 1000, 0, None, pq.Hz, "f_T", "Ending frequency"),
)  # 'pq.Hz'

default_protocol_parameters["ramp"] = make_param_dict(
    (
        "phis",
        [1e16, 1e17, 1e18],
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  # 'photons/s/mm^2' #1e12,1e13,1e14,1e15,
    (
        "phi0",
        0,
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\phi_0",
        "Constant offset for flux",
    ),  # 'photons/s/mm^2'
    (
        "Dt_delay",
        25,
        0,
        1e9,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  # 'ms'
    (
        "cycles",
        [[250.0, 25.0]],
        0,
        None,
        pq.ms,
        "cycles",
        "List of [on, off] durations for each pulse",
    ),
)  # 'ms'#,

default_protocol_parameters["delta"] = make_param_dict(
    (
        "phis",
        [1e20],
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  # 'photons/s/mm^2'
    (
        "Dt_delay",
        5,
        0,
        1e9,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  # 'ms'
    (
        "Dt_on",
        1e-3,
        0,
        1e9,
        pq.ms,
        "\Delta t_{on}",
        "On-phase duration",
    ),  # 'ms'
    (
        "Dt_total",
        25.0,
        0,
        None,
        pq.ms,
        "T_{total}",
        "Total simulation duration",
    ),
)  # 'ms'

default_protocol_parameters["rectifier"] = make_param_dict(
    (
        "phis",
        [1e16],
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  # 'photons/s/mm^2' # Change to 1e17?
    (
        "Dt_delay",
        50,
        0,
        1e9,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  # 'ms'
    (
        "cycles",
        [[250.0, 100.0]],
        None,
        None,
        pq.ms,
        "cycles",
        "List of [on, off] durations for each pulse",
    ),
)  # 'ms' #,

default_protocol_parameters["shortPulse"] = make_param_dict(
    (
        "phis",
        [1.5e15],
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  # 'photons/s/mm^2' #1e12
    (
        "Dt_delay",
        25,
        0,
        None,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  # 'ms'
    (
        "pDs",
        [1, 2, 3, 5, 8, 10, 20],
        0,
        None,
        pq.ms,
        "\mathbf{\Delta t_{on}}",
        "List of pulse on-phase durations",
    ),  # 'ms' # [0.1, 0.2, 0.5, 1, 2, 5, 10]
    (
        "Dt_total",
        100.0,
        0,
        None,
        pq.ms,
        "T_{total}",
        "Total simulation duration",
    ),
)  # 'ms'

default_protocol_parameters["recovery"] = make_param_dict(
    (
        "phis",
        [1e17],
        None,
        None,
        pq.mole * pq.mm**-2 * pq.second**-1,
        "\mathbf{\phi}",
        "List of flux values",
    ),  # 'photons/s/mm^2'
    (
        "Dt_delay",
        100,
        0,
        None,
        pq.ms,
        "\Delta t_{delay}",
        "Delay duration before the first pulse",
    ),  # 'ms'
    (
        "Dt_on",
        100,
        0,
        None,
        pq.ms,
        "\Delta t_{on}",
        "On-phase duration",
    ),  # 'ms'
    (
        "Dt_IPIs",
        [500, 1000, 1500, 2500, 5000, 7500, 10000],
        None,
        None,
        pq.ms,
        "\mathbf{\Delta t_{off}}",
        "List of pulse off-phase durations",
    ),  # 'ms'
    # ('Dt_IPIs',[0.5,1,1.5,2.5,5,7.5,10],None,None,seconds), # 'ms'
    (
        "Dt_total",
        12000,
        0,
        None,
        pq.ms,
        "T_{total}",
        "Total simulation duration",
    ),
)  # 'ms'


# E.g.
# protocols['shortPulse']([1e12], [-70], 25, [1,2,3,5,8,10,20], 100, 0.1)

# squarePulses = [protocol for protocol in protocols if protocol.squarePulse]
# arbitraryPulses = [protocol for protocol in protocols if not protocol.squarePulse]
# squarePulses = {'custom': True, 'delta': True, 'step': True, 'rectifier': True, 'shortPulse': True, 'recovery': True}
# arbitraryPulses = {'custom': True, 'sinusoid': True, 'chirp': True, 'ramp':True} # Move custom here
# smallSignalAnalysis = {'sinusoid': True, 'step': True, 'delta': True}


def select_protocol(protocol, params=None):
    """Protocol selection function"""
    if protocol in protocols:
        run_params = {}
        run_params.update(default_protocol_parameters[protocol])
        if params is not None:
            run_params.update(params)
        return protocols[protocol](run_params)
    else:
        raise NotImplementedError(protocol)
