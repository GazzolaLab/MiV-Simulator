#!/usr/bin/env python

import sys
import os
import numpy as np
from neuron import h
from mpi4py import MPI
import matplotlib.pyplot as plt

from miv_simulator.mechanisms import compile_and_load
from miv_simulator.opto.run import OptoSim

h.nrnmpi_init()

pc = h.ParallelContext()
nranks = int(pc.nhost())
myrank = int(pc.id())

ngids = 2

cells = []
vrecs = []
stims = []

def run():
    
#    compile_and_load(force=True)

    for gid in range(ngids):
        
        if gid % nranks == myrank:
            
            print(f"Creating gid {gid} on rank {myrank}")
            
            sec=h.Section()
            sec.insert("hh")
            for seg in sec:
                seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
                seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
                seg.hh.gl = 0.0003  # Leak conductance in S/cm2
                seg.hh.el = -54.3  # Reversal potential in mV
            
            pc.set_gid2node(gid, myrank)
            nc = h.NetCon(sec(1.0)._ref_v, None, sec=sec)
            pc.cell(gid, nc)
            cells.append(sec)
            
            # Current injection into section
            stim = h.IClamp(sec(0.5))
            stim.delay = 50
            stim.amp = 4
            stim.dur = 100
            stims.append(stim)
            
            # Record membrane potential
            v = h.Vector()
            v.record(sec(0.5)._ref_v)
            vrecs.append(v)

    rec_t = h.Vector()
    rec_t.record(h._ref_t)

    pop_gid_dict = { 'default': [0, 1] }
    opto = OptoSim(pc=pc,
                   pop_gid_dict=pop_gid_dict,
                   rho_params = {'expProb': 0.9},
                   nstates=3,
                   protocol="step")
    
    h.dt = 0.25
    pc.set_maxstep(10)
    h.finitialize(-65)
    pc.psolve(200)
    
    pc.runworker()
    pc.done()

    if myrank == 0:
        plt.plot(rec_t, vrecs[0])
        plt.show()
    
    h.quit()

run()
