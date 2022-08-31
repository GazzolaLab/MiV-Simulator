TITLE A-type potassium channel (voltage dependent, for O-LM cell)

COMMENT
A-type potassium channel (voltage dependent, for O-LM cell)

Ions: k

Style: quasi-ohmic

From: 1.	Zhang, L. and McBain, J. Voltage-gated potassium currents in
	stratum oriens-alveus inhibitory neurons of the rat CA1
	hippocampus, J. Physiol. 488.3:647-660, 1995.

		Activation V1/2 = -14 mV
		slope = 16.6
		activation t = 5 ms
		Inactivation V1/2 = -71 mV
		slope = 7.3
		inactivation t = 15 ms
		recovery from inactivation = 142 ms

2.	Martina, M. et al. Functional and Molecular Differences between
	Voltage-gated K+ channels of fast-spiking interneurons and pyramidal
	neurons of rat hippocampus, J. Neurosci. 18(20):8111-8125, 1998.	
	(only the gmax is from this paper)

		gmax = 0.0175 mho/cm2
		Activation V1/2 = -6.2 +/- 3.3 mV
		slope = 23.0 +/- 0.7 mV
		Inactivation V1/2 = -75.5 +/- 2.5 mV
		slope = 8.5 +/- 0.8 mV
		recovery from inactivation t = 165 +/- 49 ms  

3.	Warman, E.N. et al.  Reconstruction of Hippocampal CA1 pyramidal
	cell electrophysiology by computer simulation, J. Neurophysiol.
	71(6):2033-2045, 1994.

		gmax = 0.01 mho/cm2
		(number taken from the work by Numann et al. in guinea pig
		CA1 neurons)

Updates:
2014 December (Marianne Bezaire): documented
ENDCOMMENT


UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
}
 
NEURON {
        SUFFIX ch_KvAolm
        USEION k READ ek WRITE ik
        RANGE gmax,ik
        GLOBAL ainf, binf, aexp, bexp, tau_b
        RANGE myi, g
}
 
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}
 
PARAMETER {
        v (mV)
        p = 5 (degC)
        dt (ms)
        gmax = 0.0165 (mho/cm2)	:from Martina et al.
        ek = -90 (mV)
	tau_a = 5 (ms)
}
 
STATE {
        a b
}
 
ASSIGNED {
        ik (mA/cm2)
	ainf binf aexp bexp
	tau_b
	myi (mA/cm2)
	g (mho/cm2)
}
 
BREAKPOINT {
	SOLVE deriv METHOD derivimplicit
	g = gmax*a*b
	ik = g*(v - ek)
	myi = ik
}
 
INITIAL {
	rates(v)
	a = ainf
	b = binf
}

DERIVATIVE deriv {  :Computes state variables m, h, and n rates(v)      
	: at the current v and dt.
	rates(v) : fixed a typo found by of Andres Ecker
	a' = (ainf - a)/(tau_a)
	b' = (binf - b)/(tau_b)
}
 
PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL alpha_b, beta_b
	TABLE ainf, aexp, binf, bexp, tau_a, tau_b  DEPEND dt, p FROM -200 TO 100 WITH 300
	alpha_b = 0.000009/exp((v-26)/18.5)
	beta_b = 0.014/(exp((v+70)/(-11))+0.2)
        ainf = 1/(1 + exp(-(v + 14)/16.6))
        aexp = 1 - exp(-dt/(tau_a))
	tau_b = 1/(alpha_b + beta_b)
        binf = 1/(1 + exp((v + 71)/7.3))
        bexp = 1 - exp(-dt/(tau_b))
}
 
UNITSON

