## Test of Gfluct3 mechanism, a noise source that emulates fluctuating channel conductances.
## The conductances are sampled from a Gaussian distribution where
## negative values are replaced with 0.

from neuron import h, gui
from miv_simulator.mechanisms import compile_and_load

import numpy as np

import pytest


def test_run_with_Gfluct3():
    compile_and_load("tests/mechanisms", force=True)

    soma = h.Section(name="soma")
    soma.L = soma.diam = 12.6157  # Makes a soma of 500 microns squared.

    for sec in h.allsec():
        sec.Ra = 100  # Axial resistance in Ohm * cm
        sec.cm = 1  # Membrane capacitance in micro Farads / cm^2

    # Insert active Hodgkin-Huxley current in the soma
    soma.insert("hh")
    for seg in soma:
        seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
        seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
        seg.hh.gl = 0.0003  # Leak conductance in S/cm2
        seg.hh.el = -54.3  # Reversal potential in mV

    # Insert fluctuating conductance in soma
    fl = h.Gfluct3(soma(0.5))
    fl.g_e0 = 0.023
    fl.h = 0.25
    fl.on = 1

    h.psection(sec=soma)

    v_vec = h.Vector()  # Membrane potential vector
    t_vec = h.Vector()  # Time stamp vector
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)
    simdur = 1.0  # 25.0

    h.tstop = simdur
    h.run()

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,4)) # Default figsize is (8,6)
    # plt.plot(t_vec, v_vec)
    # plt.xlabel('time (ms)')
    # plt.ylabel('mV')
    # plt.show()

    expected_v_vec = np.array(
        [
            -65.0,
            -61.74941016,
            -59.45556825,
            -57.8345957,
            -56.68642231,
            -55.86994955,
            -55.28566643,
            -54.86343537,
            -54.55388574,
            -54.32233507,
            -54.14448734,
            -53.84814417,
            -53.62091752,
            -53.44142031,
            -53.2946901,
            -53.17030741,
            -53.06105699,
            -52.96197643,
            -52.8696811,
            -52.78188629,
            -52.69706986,
            -52.56121319,
            -52.44204488,
            -52.33435402,
            -52.23457321,
            -52.14027276,
            -52.04980785,
            -51.96207291,
            -51.87633119,
            -51.79209688,
            -51.70905378,
            -51.65701963,
            -51.59665146,
            -51.53078034,
            -51.46135745,
            -51.38972573,
            -51.31680826,
            -51.24323869,
            -51.16945141,
            -51.09574371,
            -51.02231867,
        ]
    )  # Empirical
    np.testing.assert_allclose(v_vec.to_python(), expected_v_vec)
