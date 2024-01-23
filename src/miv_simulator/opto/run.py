"""Simulation of opsin receptors at the channel, cell, and network level.
Based on code from the PyRhO: A Multiscale Optogenetics Simulation Platform
https://github.com/ProjectPyRhO/PyRhO.git
"""

from typing import TYPE_CHECKING, Optional, Any, Dict, Set, Tuple
from collections import defaultdict
import copy
import logging
import math
import numpy as np
from neuron import h
from miv_simulator.utils import (
    get_module_logger,
)
from miv_simulator.opto.core import cycles2times
from miv_simulator.opto.models import select_model
from miv_simulator.opto.protocols import select_protocol

if TYPE_CHECKING:
    from neuron.hoc import HocObject

logger = get_module_logger(__name__)


class OptoStim:

    """Class for cellular level optogenetic simulations with NEURON."""

    mechanisms = {3: "RhO3c", 4: "RhO4c", 6: "RhO6c"}

    def __init__(
        self,
        pc: "HocObject",
        pop_gid_dict: Dict[str, Set[int]],
        model_nstates: int,
        opsin_type: str,
        rho_params: Dict[str, Any],
        protocol: str,
        protocol_params: Optional[Dict[str, Any]] = None,
        max_gid_rec_count: int = 1,
        sec_rec_count: int = 1,
        seed: int = 1,
    ):
        self.rng = np.random.RandomState(seed)

        self.max_gid_rec_count = max_gid_rec_count
        self.sec_rec_count = sec_rec_count
        self.pc = pc
        self.protocol = select_protocol(protocol, params=protocol_params)
        self.rho_params = copy.deepcopy(rho_params)
        self.model = select_model(model_nstates, opsin_type)
        self.pop_gid_dict = pop_gid_dict

        self.pop_rec_dict = defaultdict(lambda: defaultdict(list))
        self.pop_rho_dict = defaultdict(lambda: defaultdict(list))
        self.pop_sec_dict = defaultdict(lambda: defaultdict(list))

        self.protocol.prepare()
        self.init_mechanisms()

        self.rho_tvec = None
        self.init_recording()

        self.dt = None
        self.init_dt()

        self.phiVec = None
        self.tvec = None

        ## TODO: support for multiple runs
        run = 0
        cycles, Dt_delay = self.protocol.getRunCycles(run)
        ## TODO: support for multiple phi indices
        phi_index = 0
        phi_ts = self.protocol.phi_ts[run][phi_index][:]
        self.setup_stim(phi_ts, Dt_delay, cycles)

    def init_dt(self):
        """Function to set the time step according to the protocol and rhodopsin"""

        self.dt = self.protocol.getShortestPeriod()
        if self.dt < h.dt:
            h.dt = self.dt
            logger.info(f"Time step reduced to {h.dt} by optogenetic protocol")

    def init_mechanisms(self):
        for pop_name in self.pop_gid_dict:
            gid_set = self.pop_gid_dict[pop_name]
            sec_dict = self.pop_sec_dict[pop_name]
            rho_dict = self.pop_rho_dict[pop_name]
            mech = getattr(h, self.mechanisms[self.model.nStates])

            expProb = self.rho_params.get("expProb", 1.0)

            for gid in gid_set:
                if not self.pc.gid_exists(gid):
                    continue

                cell = self.pc.gid2cell(gid)
                assert cell is not None

                is_art = False
                if hasattr(cell, "is_art"):
                    is_art = cell.is_art() > 0
                if is_art:
                    continue

                sec_list = sec_dict[gid]
                rho_list = rho_dict[gid]
                for sec in cell.all:
                    nseg = sec.nseg
                    for x in np.linspace(0.1, 0.9, nseg) if nseg > 1 else [0.5]:
                        if (expProb >= 1) or (self.rng.random() <= expProb):
                            # Insert a rhodopsin mechanism and append it to rho_list for that cell
                            rho = mech(sec(x))
                            rho_list.append(rho)
                            sec_list.append(sec)

    def init_recording(self):
        # Record time points
        self.rho_tvec = h.Vector()
        self.rho_tvec.record(h._ref_t)

        for pop_name in self.pop_gid_dict:
            gid_set = self.pop_gid_dict[pop_name]
            sec_dict = self.pop_sec_dict[pop_name]
            rho_dict = self.pop_rho_dict[pop_name]
            rec_dict = self.pop_rec_dict[pop_name]
            this_gid_rec_count = 0

            for gid in gid_set:
                if not self.pc.gid_exists(gid):
                    continue

                if gid not in sec_dict:
                    continue

                sec_list = sec_dict[gid]
                rho_list = rho_dict[gid]

                rec_rho_list = [
                    {"section": sec} for sec in rho_list[: self.sec_rec_count]
                ]
                rec_dict[gid] = rec_rho_list

                for rec in rec_rho_list:
                    sec = rec["section"]

                    # Record photocurrent
                    Iphi = h.Vector()
                    Iphi.record(sec._ref_i)
                    rec["Iphi"] = Iphi

                    # Record state variables according to RhO model
                    for s in self.model.stateVars:
                        state_vec = h.Vector()
                        state_vec.record(getattr(sec, f"_ref_{s}"))
                        rec[s] = state_vec

                this_gid_rec_count += 1
                if self.max_gid_rec_count < this_gid_rec_count:
                    break

    def setup_stim(self, phi_ts, Dt_delay, cycles):
        """Main routine for simulating a pulse train."""

        # TODO: Rename RhOnc.mod to RhOn.mod and RhOn.mod to RhOnD.mod (discrete) and similar for runTrialPhi_t
        # Make RhO.mod a continuous function of light - 9.12: Discontinuities 9.13: Time-dependent parameters p263
        # - Did cubic spline interpolation ever get implemented for vec.play(&rangevar, tvec, 1)?
        # - http://www.neuron.yale.edu/neuron/faq#vecplay
        # - http://www.neuron.yale.edu/neuron/static/new_doc/programming/math/vector.html#Vector.play
        # - Ramp example: http://www.neuron.yale.edu/phpbb/viewtopic.php?f=15&t=2602

        nPulses = cycles.shape[0]
        assert len(phi_ts) == nPulses

        times, Dt_total = cycles2times(cycles, Dt_delay)

        info = "Simulating pulse cycles: [Dt_delay={:.4g}ms".format(Dt_delay)
        for p in range(nPulses):
            info += "; [Dt_on={:.4g}ms; Dt_off={:.4g}ms]".format(
                cycles[p, 0], cycles[p, 1]
            )
        info += "]"
        logger.info(info)

        # Set simulation run time
        logger.info(f"Total optogenetic stimulus time is {Dt_total} ms")

        ### Delay phase (to allow the system to settle)
        phi = 0
        self.model.initStates(
            phi=phi
        )  # Reset state and time arrays from previous runs
        self.model.s0 = self.model.states[-1, :]  # Store initial state used

        logger.info(f"Optogenetic initial conditions: {self.model.s0}")

        start, end = (
            self.model.t[0],
            self.model.t[0] + Dt_delay,
        )  # start, end = 0.00, Dt_delay
        nSteps = int(round(((end - start) / self.dt) + 1))
        t = np.linspace(start, end, nSteps, endpoint=True)
        phi_tV = np.zeros_like(t)

        discontinuities = np.asarray([len(t) - 1])
        for p in range(nPulses):
            start = end
            Dt_on, Dt_off = cycles[p, 0], cycles[p, 1]
            end = start + Dt_on + Dt_off
            nSteps = int(round(((end - start) / self.dt) + 1))
            tPulse = np.linspace(start, end, nSteps, endpoint=True)
            phi_t = phi_ts[p]
            phiPulse = phi_t(
                tPulse
            )  # -tPulse[0] # Align time vector to 0 for phi_t to work properly
            discontinuities = np.r_[discontinuities, len(tPulse) - 1]

            onInd = len(t) - 1  # Start of on-phase
            offInd = onInd + int(round(Dt_on / self.dt))
            self.model.pulseInd = np.vstack(
                (self.model.pulseInd, [onInd, offInd])
            )

            t = np.r_[t, tPulse[1:]]
            phi_tV = np.r_[phi_tV, phiPulse[1:]]

            self.model.ssInf.append(
                self.model.calcSteadyState(phi_t(end - Dt_off))
            )

        self.tvec = h.Vector(t)
        self.tvec.label("Time [ms]")
        phi_tV[np.ma.where(phi_tV < 0)] = 0  # Safeguard for negative phi values
        self.phiVec = h.Vector(phi_tV)
        self.phiVec.label("phi [ph./mm^2/s]")

        # Iterate over all opsin mechanisms and initialize instantaneous fluxes
        for pop_name in self.pop_gid_dict:
            gid_set = self.pop_gid_dict[pop_name]
            rho_dict = self.pop_rho_dict[pop_name]

            for gid in gid_set:
                if gid not in rho_dict:
                    continue

                rho_list = rho_dict[gid]
                for rho in rho_list:
                    self.phiVec.play(
                        rho._ref_phi, self.tvec, 1, discontinuities
                    )

    def sample(self, rec_index=0):
        t_rec = np.array(self.rho_tvec.to_python(), copy=True)

        pop_result_dict = defaultdict(lambda: {})

        stored = False

        for pop_name in self.pop_gid_dict:
            gid_set = self.pop_gid_dict[pop_name]
            rec_dict = self.pop_rec_dict[pop_name]

            for gid in gid_set:
                if not self.pc.gid_exists(gid):
                    continue

                if gid not in rec_dict:
                    continue

                rec = rec_dict[gid]
                Iphi = np.array(rec[rec_index]["Iphi"].to_python(), copy=True)

                # Get solution variables
                soln = np.zeros((len(t_rec), self.model.nStates))
                for sInd, s in enumerate(self.model.stateVars):
                    soln[:, sInd] = np.array(rec[rec_index][s].to_python())

                if not stored:
                    self.model.storeStates(soln[1:], t_rec[1:])
                    stored = True

                pop_result_dict[pop_name][gid] = {"Iphi": Iphi, "states": soln}

        return t_rec, pop_result_dict
