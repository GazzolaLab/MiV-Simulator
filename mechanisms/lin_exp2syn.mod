TITLE Shared synaptic conductance with per-stream linear summation

COMMENT

Milstein, 2018

Rise and decay kinetics are shared across all presynaptic sources. Conductances are linearly summed across sources, and
updated at every time step. Each source stores an independent weight multiplier and a unitary peak conductance (g_unit).
Each stream will sum events linearly without saturation during repetitive activation.

Implementation informed by:

The NEURON Book: Chapter 10, N.T. Carnevale and M.L. Hines, 2004

ENDCOMMENT

NEURON {
	POINT_PROCESS LinExp2Syn
	RANGE g, i, tau_rise, tau_decay, e
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
}

PARAMETER {
	tau_rise		= 1.	(ms) 	: time constant of exponential rise
	tau_decay 		= 5. 	(ms) 	: time constant of exponential decay
	e 				= 0. 	(mV) 	: reversal potential
}


ASSIGNED {
	v			(mV)		: postsynaptic voltage
	i 			(nA)		: current = g*(v - Erev)
	g 			(umho)		: conductance
    factor 					: normalization factor

}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	if (tau_rise/tau_decay > 0.9999) {
		tau_rise = 0.9999*tau_decay
	}
	if (tau_rise/tau_decay < 1e-9) {
		tau_rise = tau_decay*1e-9
	}
	A = 0
	B = 0
	tp = (tau_rise*tau_decay)/(tau_decay - tau_rise) * log(tau_decay/tau_rise)
	factor = -exp(-tp/tau_rise) + exp(-tp/tau_decay)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	i = g * (v - e)
}

DERIVATIVE state {
	A' = -A/tau_rise
	B' = -B/tau_decay
}

NET_RECEIVE(weight, g_unit (umho)) {
	INITIAL {}
	A = A + weight * g_unit * factor
	B = B + weight* g_unit * factor
}
