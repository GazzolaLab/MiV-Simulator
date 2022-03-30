TITLE Fast delayed rectifier potassium channel (voltage dependent)

COMMENT
Fast delayed rectifier potassium channel (voltage dependent)

Ions: k

Style: quasi-ohmic

From: Yuen and Durand, 1991 (squid axon)

Updates:
2014 December (Marianne Bezaire): documented
? ? ?: further shifted the voltage dependence by 65 mV
? ? (Aradi): shifted the voltage dependence by 16 mV
ENDCOMMENT


VERBATIM
#include <stdlib.h> /* 	Include this library so that the following
						(innocuous) warning does not appear:
						 In function '_thread_cleanup':
						 warning: incompatible implicit declaration of 
						          built-in function 'free'  */
ENDVERBATIM
 
UNITS {
	(mA) =(milliamp)
	(mV) =(millivolt)
	(uF) = (microfarad)
	(molar) = (1/liter)
	(nA) = (nanoamp)
	(mM) = (millimolar)
	(um) = (micron)
	FARADAY = 96520 (coul)
	R = 8.3134	(joule/degC)
}
 
NEURON { 
	SUFFIX ch_Kdrfast 
	USEION k READ ek WRITE ik VALENCE 1
	RANGE g, gmax, ninf, ntau, ik
	RANGE myi
	THREADSAFE
}
 
PARAMETER {
	v (mV) 
	celsius (degC) : temperature - set in hoc; default is 6.3
	dt (ms) 

	ek  (mV)
	gmax (mho/cm2)
}
 
STATE {
	n	
}
 
ASSIGNED {		     
	g (mho/cm2)
	ik (mA/cm2)
	ninf
	ntau (ms)
	nexp
	myi (mA/cm2)
} 

BREAKPOINT {
	SOLVE states
	g = gmax*n*n*n*n
	ik = g*(v-ek)
	myi =  ik
}
 
UNITSOFF
 
INITIAL {
	trates(v)

	n = ninf
}

PROCEDURE states() {	:Computes state variables m, h, and n 
	trates(v)	:      at the current v and dt.       
	n = n + nexp*(ninf-n)
}
 
LOCAL q10
PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
	LOCAL  alpha, beta, sum
	:q10 = 3^((celsius - 6.3)/10)
	q10 = 3^((celsius - 34)/10)
		
	:"nf" fKDR activation system
	alpha = -0.07*vtrap((v+65-47),-6)
	beta = 0.264/exp((v+65-22)/40)
	sum = alpha+beta        
	ntau = 1/sum
	ninf = alpha/sum	
}
 
PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
	LOCAL tinc
	TABLE ninf, nexp, ntau
	DEPEND dt, celsius
	FROM -100 TO 100 WITH 200
						   
	rates(v)	: not consistently executed from here if usetable_hh == 1
	: so don't expect the tau values to be tracking along with
	: the inf values in hoc

	tinc = -dt * q10
	nexp = 1 - exp(tinc/ntau)
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{  
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON

