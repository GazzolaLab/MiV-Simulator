#!/usr/bin/env python

import sys
import os
import numpy as np
from neuron import h
from mpi4py import MPI
import matplotlib.pyplot as plt

from miv_simulator.mechanisms import compile_and_load
from miv_simulator.opto.run import OptoStim
from miv_simulator.utils import config_logging

h.nrnmpi_init()
config_logging(True)

pc = h.ParallelContext()
nranks = int(pc.nhost())
myrank = int(pc.id())

ngids = 2

cells = []
vrecs = []
stims = []

opto_stim = True


class Cell:
    def __init__(self, gid):
        self._gid = gid
        self._setup_morphology()
        self.all = self.soma.wholetree()
        self._setup_biophysics()

    def __repr__(self):
        return f"{self.name}[{self._gid}]"


class BallAndStick(Cell):
    name = "BallAndStick"

    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1.0
        self.dend.nseg = 3

        # everything below here in this method is NEW
        self._spike_detector = h.NetCon(
            self.soma(0.5)._ref_v, None, sec=self.soma
        )

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100  # Axial resistance in Ohm * cm
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = 0.18  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -54.3  # Reversal potential in mV

        # Insert passive current in the dendrite
        self.dend.insert("pas")
        for seg in self.dend:
            seg.pas.g = 0.0005  # Passive conductance in S/cm2
            seg.pas.e = -65  # Leak reversal potential mV


def run():

    compile_and_load(force=True)

    for gid in range(ngids):

        if gid % nranks == myrank:

            print(f"Creating gid {gid} on rank {myrank}")

            cell = BallAndStick(gid)
            cells.append(cell)

            pc.set_gid2node(gid, pc.id())
            pc.cell(cell._gid, cell._spike_detector)
            pc.outputcell(gid)

            # Current injection into section
            stim = h.IClamp(cell.soma(0.5))
            stim.delay = 50
            stim.amp = 0
            stim.dur = 100
            stims.append(stim)

            # Record membrane potential
            v = h.Vector()
            v.record(cell.soma(0.5)._ref_v)
            vrecs.append(v)

    rec_t = h.Vector()
    rec_t.record(h._ref_t)

    pop_gid_dict = {"default": [0, 1]}
    ncycles = 10
    cycles = np.asarray([[10, 50] * ncycles]).reshape((-1, 2))
    if opto_stim:
        opto = OptoStim(
            pc=pc,
            pop_gid_dict=pop_gid_dict,
            rho_params={"expProb": 0.95},
            model_nstates=6,
            opsin_type="ChR2",
            protocol="step",
            protocol_params=(
                ("Dt_delay", 150),
                ("phis", [1e16, 1e16]),
                ("cycles", cycles),
            ),
        )
    cell = pc.gid2cell(1)
    h.psection(sec=cell.soma)
    h.psection(sec=cell.dend)

    h.dt = 0.025
    pc.set_maxstep(10)
    h.finitialize(-65)
    pc.psolve(1000)

    pc.runworker()
    pc.done()

    h.quit()
