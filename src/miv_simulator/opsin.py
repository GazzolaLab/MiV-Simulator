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

if TYPE_CHECKING:
    from neuron.hoc import HocObject

logger = logging.getLogger(__name__)

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
    assert(cycles.shape[1] <= 2)
    times = np.cumsum(np.r_[Dt_delay, cycles.ravel()])
    Dt_total = times[-1]
    times = times[:-1].reshape((nPulses, 2))  # Trim the final Dt_off & reshape
    #times = np.zeros((nPulses, 2)) #[Dt_delay,Dt_delay+cycles[row,0] for row in pulses]
    #Dt_total = Dt_delay
    #for p in range(nPulses):
    #    times[p, 0] = Dt_total
    #    times[p, 1] = Dt_total+cycles[p, 0]
    #    Dt_total += sum(cycles[p, :])
    return (times, Dt_total)




class Opsin:
    
    """Class for cellular level opsin simulations with NEURON."""

    mechanisms = {3: 'RhO3c', 4: 'RhO4c', 6: 'RhO6c'}

    def __init__(self, Prot, RhO, params,
                 rec_index: int = 0,
                 seed: int = 1,
                 ):

        self.Prot = Prot
        self.RhO = RhO

        ### TODO: Move into prepare() in order to insert the correct type of mechanism (continuous or discrete) according to the protocol
        #for states, mod in self.mechanisms.items():
        #    self.mechanisms[states] += 'c'
        
        self.mod = self.mechanisms[self.RhO.nStates]  # Use this to select the appropriate mod file for insertion
        #if not Prot.squarePulse:
        #self.mod += 'c'

        self.init_mechanisms(self.RhO, expProb=params['expProb'].value)
        self.rhoParams = copy.deepcopy(modelParams[str(self.RhO.nStates)])
        self.RhO.exportParams(self.rhoParams)
        self.set_opsin_params(self.rho_list, self.rhoParams)  # self.RhO, modelParams[str(self.RhO.nStates)])

        self.rhoRec = self.rho_list[rec_index]  # TODO: move to init_recording Choose a rhodopsin to record
        self.init_recording(self.rhoRec, params['Vcomp'].value, self.RhO)

    def init_dt(self, Prot):
        
        """Function to prepare the simulator according to the protocol and rhodopsin"""
        
        Prot.prepare()

        dt = Prot.getShortestPeriod()
        Prot.dt = self.checkDt(dt)
        if self.dt < h.dt:
            h.dt = self.dt

        return

    def init_mechanisms(self, RhO, expProb=1):

        self.sec_list = h.List()
        self.rho_list = h.List()
        mech = getattr(h, self.mechanisms[RhO.nStates])
        
        # TODO: manage list of cells
        for sec in self.cell:  # h.allsec(): # Loop over every section in the cell
            self.sec_list.append(sec)
            if (expProb >= 1) or (self.rng.random() <= expProb):  # Insert a rhodopsin and append it to rho_list
                rho = mech(sec(0.5))
                self.rho_list.append(rho)

    def set_opsin_params(self, rho_list, pSet):
        for rho in rho_list:  # not self.rhoList so that subsets can be passed
            for p in pSet:
                setattr(rho, p, pSet[p].value)

    def getOpsinParams(self):
        # TODO
        pass

    def init_recording(self, rhoRec, Vcomp, RhO):

        self.Iphi = h.Vector()
        self.rho_tvec = h.Vector()
        # Record time points
        self.rho_tvec.record(h._ref_t)  

        # Record photocurrent
        self.Iphi.record(rhoRec._ref_i)

        # Save state variables according to RhO model
        for s in RhO.stateVars:
            h('objref {}Vec'.format(s))
            h.rho = rhoRec                             # TODO: Check this works with multiple sections/rhodopsins
            h('{}Vec = new Vector()'.format(s))
            h('{}Vec.record(&rho.{})'.format(s, s))

    # TODO: init experiment / init stimulus
    def set_phi(self, phiOn, Dt_delay, Dt_on, Dt_off, nPulses):
        for rho in self.rho_list:
            rho.phiOn = phiOn
            rho.Dt_delay = Dt_delay
            rho.Dt_on = Dt_on
            rho.Dt_off = Dt_off
            rho.nPulses = nPulses

    def run(self, RhO, phi_ts, V, Dt_delay, cycles, dt, verbose=config.verbose):
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

        if verbose > 1:
            if V is not None:
                Vstr = 'V = {:+}mV, '.format(V)
            else:
                Vstr = ''
            info = "Simulating experiment {}pulse cycles: [Dt_delay={:.4g}ms".format(Vstr, Dt_delay)
            for p in range(nPulses):
                info += "; [Dt_on={:.4g}ms; Dt_off={:.4g}ms]".format(cycles[p, 0], cycles[p, 1])
            info += "]"
            print(info)

        # Set simulation run time
        h.tstop = Dt_total  # Dt_delay + np.sum(cycles) #nPulses*(Dt_on+Dt_off) + padD

        ### Delay phase (to allow the system to settle)
        phi = 0
        RhO.initStates(phi)  # Reset state and time arrays from previous runs
        RhO.s0 = RhO.states[-1, :]   # Store initial state used

        if verbose > 1:
            print("Trial initial conditions:{}".format(RhO.s0))

        start, end = RhO.t[0], RhO.t[0]+Dt_delay  # start, end = 0.00, Dt_delay
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
            RhO.pulseInd = np.vstack((RhO.pulseInd, [onInd, offInd]))

            t = np.r_[t, tPulse[1:]]
            phi_tV = np.r_[phi_tV, phiPulse[1:]]

            RhO.ssInf.append(RhO.calcSteadyState(phi_t(end-Dt_off)))

        tvec = h.Vector(t)
        tvec.label('Time [ms]')
        phi_tV[np.ma.where(phi_tV < 0)] = 0  # Safeguard for negative phi values
        phiVec = h.Vector(phi_tV)
        phiVec.label('phi [ph./mm^2/s]')

        phiVec.play(self.rhoRec._ref_phi, tvec, 1, discontinuities)

    def collect(self):
        
        # Collect data
        I_RhO = np.array(h.Iphi.to_python(), copy=True)
        t = np.array(h.rho_tvec.to_python(), copy=True)
        self.t = t

        # Get solution variables N.B. NEURON changes the sampling rate
        soln = np.zeros((len(t), RhO.nStates))
        for sInd, s in enumerate(RhO.stateVars):
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

        RhO.storeStates(soln[1:], t[1:])

        self.dt = h.dt
        self.Prot.dt = h.dt

        ### Calculate photocurrent
        states, t = RhO.getStates()

        return I_RhO, t, soln
