
## Test of Gfluct3 mechanism, a noise source that emulates fluctuating channel conductances.
## The conductances are sampled from a Gaussian distribution where
## negative values are replaced with 0.

from neuron import h, gui

soma = h.Section(name='soma')

soma.L = soma.diam = 12.6157 # Makes a soma of 500 microns squared.

for sec in h.allsec():
    sec.Ra = 100    # Axial resistance in Ohm * cm
    sec.cm = 1      # Membrane capacitance in micro Farads / cm^2

# Insert active Hodgkin-Huxley current in the soma
soma.insert('hh')
for seg in soma:
    seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
    seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
    seg.hh.gl = 0.0003    # Leak conductance in S/cm2
    seg.hh.el = -54.3     # Reversal potential in mV

# Insert fluctuating conductance in soma
fl = h.Gfluct3(soma(0.5))
fl.g_e0 = 0.023
fl.h = 0.25
fl.on = 1


h.psection(sec=soma)

v_vec = h.Vector()        # Membrane potential vector
t_vec = h.Vector()        # Time stamp vector
v_vec.record(soma(0.5)._ref_v)
t_vec.record(h._ref_t)
simdur = 25.0

h.tstop = simdur
h.run()

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4)) # Default figsize is (8,6)
plt.plot(t_vec, v_vec)
plt.xlabel('time (ms)')
plt.ylabel('mV')
plt.show()

