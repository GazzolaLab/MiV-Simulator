"""Simulation of opsin receptors at the channel, cell, and network level. 
Based on code from the PyRhO: A Multiscale Optogenetics Simulation Platform
https://github.com/ProjectPyRhO/PyRhO.git
"""

from typing import TYPE_CHECKING, Dict, Set, Tuple
import copy
import logging
import math
import numpy as np
from neuron import h
from miv_simulator.utils import (
    get_module_logger,
)
from miv_simulator.opsin.models import select_model

if TYPE_CHECKING:
    from neuron.hoc import HocObject

logger = get_module_logger(__name__)


class Opsin:
    
    """Class for cellular level opsin simulations with NEURON."""

    mechanisms = {3: 'RhO3c', 4: 'RhO4c', 6: 'RhO6c'}

    def __init__(self, nstates: int,
                 protocol, rhoParams,
                 rec_index: int = 0,
                 seed: int = 1, ):

        self.rng = np.random.RandomState()
        
        self.protocol = protocol
        self.rhoParams = copy.deepcopy(rhoParams)
        self.model = select_model(nstates)
        
        # Selects the appropriate mechanism for insertion
        self.rho_list = []
        self.sec_list = []
        self.init_mechanisms()
        self.set_opsin_params()

        self.rhoRec = None
        self.Iphi = h.Vector()
        self.rho_tvec = h.Vector()
        self.init_recording()

        self.dt = None
        self.init_dt()
        
    def init_dt(self):
        
        """Function to prepare the simulator according to the protocol and rhodopsin"""

        self.dt = get_shortest_period(self.protocol)
        if self.dt < h.dt:
            h.dt = self.dt

        return

    def init_mechanisms(self):

        self.sec_list = h.List()
        self.rho_list = h.List()
        mech = getattr(h, self.mechanisms[self.model.nStates])
        
        # TODO: manage list of cells
        expProb = self.rhoParams.expProb
        for sec in self.cell.all:
            self.sec_list.append(sec)
            if (expProb >= 1) or (self.rng.random() <= expProb):  # Insert a rhodopsin and append it to rho_list
                rho = mech(sec(0.5))
                self.rho_list.append(rho)

    def set_opsin_params(self, rho_list, params):
        for rho in rho_list:
            for p in params:
                setattr(rho, p, params[p])

    def get_opsin_params(self):
        # TODO
        pass

    def set_phi(self, phiOn, Dt_delay, Dt_on, Dt_off, nPulses):
        for rho in self.rho_list:
            rho.phiOn = phiOn
            rho.Dt_delay = Dt_delay
            rho.Dt_on = Dt_on
            rho.Dt_off = Dt_off
            rho.nPulses = nPulses

    def init_recording(self):

        self.rhoRec = self.rho_list[rec_index]

        self.Iphi = h.Vector()
        self.rho_tvec = h.Vector()
        # Record time points
        self.rho_tvec.record(h._ref_t)  
        # Record photocurrent
        self.Iphi.record(rhoRec._ref_i)

        # Save state variables according to RhO model
        for s in self.model.stateVars:
            h(f'objref {s}Vec')
            h.rho = rhoRec
            h(f'{s}Vec = new Vector()')
            h(f'{s}Vec.record(&rho.{s})')
            

    def setup_pulses(self, RhO, phi_ts, Dt_delay, cycles, dt):
        """Main routine for simulating a pulse train."""
        
        # TODO: Rename RhOnc.mod to RhOn.mod and RhOn.mod to RhOnD.mod (discrete) and similar for runTrialPhi_t
        # Make RhO.mod a continuous function of light - 9.12: Discontinuities 9.13: Time-dependent parameters p263
            #- Did cubic spline interpolation ever get implemented for vec.play(&rangevar, tvec, 1)?
            #- http://www.neuron.yale.edu/neuron/faq#vecplay
            #- http://www.neuron.yale.edu/neuron/static/new_doc/programming/math/vector.html#Vector.play
            #- Ramp example: http://www.neuron.yale.edu/phpbb/viewtopic.php?f=15&t=2602

        nPulses = cycles.shape[0]
        assert(len(phi_ts) == nPulses)

        times, Dt_total = cycles2times(cycles, Dt_delay)

        info = "Simulating pulse cycles: [Dt_delay={:.4g}ms".format(Dt_delay)
        for p in range(nPulses):
            info += "; [Dt_on={:.4g}ms; Dt_off={:.4g}ms]".format(cycles[p, 0], cycles[p, 1])
        info += "]"
        logger.info(info)

        # Set simulation run time
        h.tstop = Dt_total  # Dt_delay + np.sum(cycles) #nPulses*(Dt_on+Dt_off) + padD

        ### Delay phase (to allow the system to settle)
        phi = 0
        self.model.initStates(phi)  # Reset state and time arrays from previous runs
        self.model.s0 = self.model.states[-1, :]   # Store initial state used

        logger.info(f"Trial initial conditions:{self.model.s0}")

        start, end = self.model.t[0], self.model.t[0]+Dt_delay  # start, end = 0.00, Dt_delay
        nSteps = int(round(((end-start)/dt)+1))
        t = np.linspace(start, end, nSteps, endpoint=True)
        phi_tV = np.zeros_like(t)

        discontinuities = np.asarray([len(t) - 1])
        for p in range(nPulses):
            start = end
            Dt_on, Dt_off = cycles[p, 0], cycles[p, 1]
            end = start + Dt_on + Dt_off
            nSteps = int(round(((end-start)/dt)+1))
            tPulse = np.linspace(start, end, nSteps, endpoint=True)
            phi_t = phi_ts[p]
            phiPulse = phi_t(tPulse) # -tPulse[0] # Align time vector to 0 for phi_t to work properly
            discontinuities = np.r_[discontinuities, len(tPulse) - 1]

            onInd = len(t) - 1  # Start of on-phase
            offInd = onInd + int(round(Dt_on/dt))
            self.model.pulseInd = np.vstack((self.model.pulseInd, [onInd, offInd]))

            t = np.r_[t, tPulse[1:]]
            phi_tV = np.r_[phi_tV, phiPulse[1:]]

            self.model.ssInf.append(self.model.calcSteadyState(phi_t(end-Dt_off)))

        tvec = h.Vector(t)
        tvec.label('Time [ms]')
        phi_tV[np.ma.where(phi_tV < 0)] = 0  # Safeguard for negative phi values
        phiVec = h.Vector(phi_tV)
        phiVec.label('phi [ph./mm^2/s]')
        phiVec.play(self.rhoRec._ref_phi, tvec, 1, discontinuities)

        
    def sample(self):
        
        I_RhO = np.array(h.Iphi.to_python(), copy=True)
        t = np.array(h.rho_tvec.to_python(), copy=True)
        self.t = t

        # Get solution variables N.B. NEURON changes the sampling rate
        soln = np.zeros((len(t), self.model.nStates))
        for sInd, s in enumerate(self.model.stateVars):
            h('objref tmpVec')
            h('tmpVec = {}Vec'.format(s))
            soln[:, sInd] = np.array(h.tmpVec.to_python(), copy=True)
            h('{}Vec.clear()'.format(s))
            h('{}Vec.resize(0)'.format(s))

        # Reset vectors
        h.rho_tvec.clear()
        h.rho_tvec.resize(0)
        h.Iphi.clear()
        h.Iphi.resize(0)

        self.model.storeStates(soln[1:], t[1:])

        self.dt = h.dt
        self.protocol.dt = h.dt

        ### Calculate photocurrent
        states, t = self.model.getStates()

        return I_RhO, t, soln
