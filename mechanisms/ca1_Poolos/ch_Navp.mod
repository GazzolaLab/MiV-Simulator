TITLE sodium channel (voltage dependent)

COMMENT
sodium channel (voltage dependent)

Ions: na

Style: quasi-ohmic

From: modified from Jeff Magee. M.Migliore may97

Updates:
2002 April (Michele Migliore): added sh to account for higher threshold
2014 December (Marianne Bezaire): documented
ENDCOMMENT

NEURON {
	SUFFIX ch_Navp
	USEION na READ ena WRITE ina
	RANGE  gmax, ar2, myi, e, g
	GLOBAL minf, hinf, mtau, htau, sinf, taus,qinf, thinf
}

PARAMETER {
	sh   = 15		(mV)
	gmax = 0.010	(mho/cm2)	
								
	tha  = -30 		(mV)
	qa   = 7.2		(mV)	: act slope		
	Ra   = 0.4		(/ms)	: open (v)		
	Rb   = 0.124 	(/ms)	: close (v)		

	thi1  = -45		(mV)	: v 1/2 for inact 	
	thi2  = -45 	(mV)	: v 1/2 for inact 	
	qd   = 1.5		(mV)    : inact tau slope
	qg   = 1.5      (mV)
	mmin = 0.02	
	hmin = 0.5			
	q10  = 2
	Rg   = 0.01 	(/ms)	: inact recov (v) 	
	Rd   = 0.03 	(/ms)	: inact (v)	
	qq   = 10		(mV)
	tq   = -55      (mV)

	thinf  = -50 	(mV)	: inact inf slope	
	qinf  = 4 		(mV)	: inact inf slope 

	vhalfs = -60	(mV)	: slow inact.
	a0s = 0.0003	(ms)	: a0s=b0s
	zetas = 12		(1)
	gms = 0.2		(1)
	smax = 10		(ms)
	vvh = -58		(mV) 
	vvs = 2			(mV)
	ar2 = 1			(1)		: 1=no inact., 0=max inact.
	ena				(mV)    : must be explicitly def. in hoc
	celsius
	v 				(mV)
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
	g			(mho/cm2)
	minf
	hinf 		
	sinf
	mtau		(ms)
	htau		(ms) 	
	taus		(ms)
}
 

STATE { m h s}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gmax*m*m*m*h*s
	ina = g * (v - ena)
	myi = ina
} 

INITIAL {
	trates(v,ar2)
	m=minf  
	h=hinf
	s=sinf
}


FUNCTION alpv(v(mV)) {
	alpv = 1/(1+exp((v-vvh-sh)/vvs))
}
        
FUNCTION alps(v(mV)) {  
	alps = exp(1.e-3*zetas*(v-vhalfs-sh)*9.648e4/(8.315*(273.16+celsius)))
}

FUNCTION bets(v(mV)) {
	bets = exp(1.e-3*zetas*gms*(v-vhalfs-sh)*9.648e4/(8.315*(273.16+celsius)))
}

LOCAL mexp, hexp, sexp

DERIVATIVE states {   
	trates(v,ar2)      
	m' = (minf-m)/mtau
	h' = (hinf-h)/htau
	s' = (sinf - s)/taus
}

PROCEDURE trates(vm,a2) {  
	LOCAL  a, b, c, qt
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
	c=alpv(vm)
	sinf = c+a2*(1-c)
	taus = bets(vm)/(a0s*(1+alps(vm)))
	if (taus<smax) {taus=smax}
}

FUNCTION trap0(v,th,a,q) {
	if (fabs(v-th) > 1e-6) {
	    trap0 = a * (v - th) / (1 - exp(-(v - th)/q))
	} else {
	    trap0 = a * q
 	}
}	
