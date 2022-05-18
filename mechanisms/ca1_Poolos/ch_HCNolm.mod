TITLE Hyperpolarization-activated, CN-gated channel (voltage dependent, for O-LM cells)

COMMENT
Hyperpolarization-activated, CN-gated channel (voltage dependent, for O-LM cells)

Ions: non-specific

Style: quasi-ohmic

From: 1.	Maccaferri, G. and McBain, C.J. The hyperpolarization-activated current
	(Ih) and its contribution to pacemaker activity in rat CA1 hippocampal
	stratum oriens-alveus interneurons, J. Physiol. 497.1:119-130,
	1996.

		V1/2 = -84.1 mV
		   k = 10.2
		reversal potential = -32.9 +/- 1.1 mV

at -70 mV, currents were fitted by a single exponetial of t = 2.8+/- 0.76 s
at -120 mV, two exponentials were required, t1 = 186.3+/-33.6 ms 
t2 = 1.04+/-0.16 s


2.	Maccaferri, G. et al. Properties of the
	Hyperpoarization-activated current in rat hippocampal CA1 Pyramidal
	cells. J. Neurophysiol. Vol. 69 No. 6:2129-2136, 1993.

		V1/2 = -97.9 mV
		   k = 13.4
		reversal potential = -18.3 mV

3.	Pape, H.C.  Queer current and pacemaker: The
	hyperpolarization-activated cation current in neurons, Annu. Rev. 
	Physiol. 58:299-327, 1996.

		single channel conductance is around 1 pS
		average channel density is below 0.5 um-2
		0.5 pS/um2 = 0.00005 mho/cm2 = 0.05 umho/cm2		
4.	Magee, J.C. Dendritic Hyperpolarization-Activated Currents Modify
	the Integrative Properties of Hippocampal CA1 Pyramidal Neurons, J.
	Neurosci., 18(19):7613-7624, 1998

Deals with Ih in CA1 pyramidal cells.  Finds that conductance density
increases with distance from the soma.

soma g = 0.0013846 mho/cm2
dendrite g (300-350 um away) = 0.0125 mho/cm2
see Table 1 in th paper

Updates:
2014 December (Marianne Bezaire): documented
ENDCOMMENT



 UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
}
 
NEURON {
        SUFFIX ch_HCNolm
        USEION h READ eh WRITE ih VALENCE 1
        RANGE gmax,ih, g
        GLOBAL rinf, rexp, tau_r
        RANGE myi
}
 
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}
 
PARAMETER {
        v (mV)
        dt (ms)
        gmax = 0.001385 (mho/cm2)
		g (mho/cm2)
        eh = -32.9 (mV)
}
 
STATE {
        r
}
 
ASSIGNED {
        ih (mA/cm2)
	rinf rexp
	tau_r
	myi (mA/cm2)
}
 
BREAKPOINT {
	SOLVE deriv METHOD derivimplicit
	g = gmax*r
	ih = g*(v - eh)
	myi = ih
}
 
INITIAL {
	rates(v)
	r = rinf
}

DERIVATIVE deriv { :Computes state variable h at current v and dt.
	rates(v)
	r' = (rinf - r)/tau_r
}


PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        TABLE rinf, tau_r, rexp DEPEND dt FROM -200
TO 100 WITH 300
	rinf = 1/(1 + exp((v+84.1)/10.2))
	tau_r = 100 + 1/(exp(-17.9-0.116*v)+exp(-1.84+0.09*v))
	rexp = 1 - exp(-dt/(tau_r))
}

FUNCTION efun(z) {
  if (fabs(z) < 1e-4) {
    efun = 1 - z/2
  } else {
    efun = z/(exp(z) - 1)
  }
}

 
UNITSON

