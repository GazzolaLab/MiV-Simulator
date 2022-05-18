TITLE sodium channel (voltage dependent, for axons)

COMMENT
sodium channel (voltage dependent, for axons)
Meant for axons, no slow inactivation.

Ions: na

Style: quasi-ohmic

From: M.Migliore Jul. 1997

Updates:
2002 April (Michele Migliore): added sh to account for higher threshold
2014 December (Marianne Bezaire): documented
ENDCOMMENT


NEURON {
	SUFFIX ch_Navaxonp
	USEION na READ ena WRITE ina
	RANGE  gmax, myi, e, g
	GLOBAL minf, hinf, mtau, htau,thinf, qinf
}

PARAMETER {
	sh   = 15	 (mV)
	gmax = 0.010 (mho/cm2)	
								
	tha  =  -30	(mV)		: v 1/2 for act	
	qa   = 7.2	(mV)		: act slope (4.5)		
	Ra   = 0.4	(/ms)		: open (v)		
	Rb   = 0.124 	(/ms)		: close (v)		

	thi1  = -45	(mV)		: v 1/2 for inact 	
	thi2  = -45 (mV)		: v 1/2 for inact 	
	qd   = 1.5	(mV)	        : inact tau slope
	qg   = 1.5  (mV)
	mmin = 0.02	
	hmin = 0.5			
	q10 = 2
	Rg   = 0.01 (/ms)		: inact recov (v) 	
	Rd   = .03 	(/ms)		: inact (v)	

	thinf = -50 (mV)		: inact inf slope	
	qinf  = 4 	(mV)		: inact inf slope 

	ena		(mV)            : must be explicitly def. in hoc
	celsius
	v 		(mV)
	e
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ina 		(mA/cm2)
	myi 		(mA/cm2)
	g		(mho/cm2)
	minf 		hinf 		
	mtau (ms)	htau (ms) 	
}
 

STATE { m h}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gmax*m*m*m*h
	ina = g * (v - ena)
	myi = ina
} 

INITIAL {
	trates(v)
	m=minf  
	h=hinf
}

DERIVATIVE states {   
	trates(v)      
	m' = (minf-m)/mtau
	h' = (hinf-h)/htau
}

PROCEDURE trates(vm) {  
        LOCAL  a, b, qt
        qt=q10^((celsius-24)/10)
	a = trap0(vm,tha+sh,Ra,qa)
	b = trap0(-vm,-tha-sh,Rb,qa)
	mtau = 1/(a+b)/qt
        if (mtau<mmin) {mtau=mmin}
	minf = a/(a+b)

	a = trap0(vm,thi1+sh,Rd,qd)
	b = trap0(-vm,-thi2-sh,Rg,qg)
	htau =  1/(a+b)/qt
        if (htau<hmin) {htau=hmin}
	hinf = 1/(1+exp((vm-thinf-sh)/qinf))
}

FUNCTION trap0(v,th,a,q) {
	if (fabs(v-th) > 1e-6) {
	        trap0 = a * (v - th) / (1 - exp(-(v - th)/q))
	} else {
	        trap0 = a * q
 	}
}	

        

