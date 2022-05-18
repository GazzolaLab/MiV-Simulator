TITLE Hyperpolarization-activated, CN-gated channel (voltage dependent)

COMMENT
Hyperpolarization-activated, CN-gated channel (voltage dependent)

Ions: non-specific

Style: quasi-ohmic

From: Magee 1998 for distal dendrites, default values are for dendrites and low Na

Updates:
2014 December (Marianne Bezaire): documented
ENDCOMMENT


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {
	v 		(mV)
	e  		(mV)        
	celsius 	(degC)
	gmax=.0001 	(mho/cm2)
	vhalfl=-90   	(mV)
	vhalft=-75   	(mV)
	a0t=0.011      	(/ms)
	zetal=4    	(1)
	zetat=2.2    	(1)
	gmt=.4   	(1)
	q10=4.5
	qtl=1
}


NEURON {
	SUFFIX ch_HCNp
	NONSPECIFIC_CURRENT i
	RANGE gmax, vhalfl, myi, e, g
	GLOBAL linf,taul
}

STATE {
	l
}

ASSIGNED {
	i (mA/cm2)
	myi (mA/cm2)
	linf      
	taul
	g
}

INITIAL {
	rate(v)
	l=linf
}


BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gmax*l
	i = g*(v-e)
	myi = i
}


FUNCTION alpl(v(mV)) {
	alpl = exp(0.0378*zetal*(v-vhalfl)) 
}

FUNCTION alpt(v(mV)) {
	alpt = exp(0.0378*zetat*(v-vhalft)) 
}

FUNCTION bett(v(mV)) {
	bett = exp(0.0378*zetat*gmt*(v-vhalft)) 
}

DERIVATIVE states {     : exact when v held constant; integrates over dt step
	rate(v)
	l' =  (linf - l)/taul
}

PROCEDURE rate(v (mV)) { :callable from hoc
	LOCAL a,qt
	qt=q10^((celsius-33)/10)
	a = alpt(v)
	linf = 1/(1+ alpl(v))
	taul = bett(v)/(qtl*qt*a0t*(1+a))
}
