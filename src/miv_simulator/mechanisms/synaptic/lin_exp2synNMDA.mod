COMMENT
Two state kinetic scheme synapse described by rise time tau_rise,
and decay time constant tau_decay. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau_rise and 1/tau_decay is
 A = a*exp(-t/tau_rise) and
 G = a*tau_decay/(tau_decay-tau_rise)*(-exp(-t/tau_rise) + exp(-t/tau_decay))
	where tau_rise < tau_decay

If tau_decay-tau_rise -> 0 then we have a alphasynapse.
and if tau_rise -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
	POINT_PROCESS LinExp2SynNMDA
        RANGE vshift, Kd, gamma, mg
	RANGE tau_rise, tau_decay, e, i
	RANGE g, pnmda
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
}

PARAMETER {
	tau_rise=10. (ms) <1e-9,1e9>
	tau_decay = 35. (ms) <1e-9,1e9>
	e=0	(mV)
	mg=1    (mM)		: external magnesium concentration
        vshift = 0 (mV)      : positive left-shifts mg unblock (reduces mg block at more negative voltages)
        Kd   = 3.57 	(mM) 	: modulate Mg concentration dependence
        gamma = 0.062 (/mV)	: modulate slope of Mg sensitivity

}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	factor
	pnmda
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	if (tau_rise/tau_decay > .9999) {
		tau_rise = .9999*tau_decay
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
	pnmda = mgblock(v)
	i = g*pnmda*(v - e)
}

DERIVATIVE state {
	A' = -A/tau_rise
	B' = -B/tau_decay
}

FUNCTION mgblock(v(mV)) {

	: from Jahr & Stevens
	mgblock = 1 / (1 + exp(gamma * -(v+vshift)) * (mg / Kd))
}

NET_RECEIVE(weight, g_unit (uS)) {
	INITIAL {}
	A = A + weight*g_unit*factor
	B = B + weight*g_unit*factor
}
