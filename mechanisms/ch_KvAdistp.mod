TITLE A-type potassium channel (voltage dependent)

COMMENT
A-type potassium channel (voltage dependent)

Ions: k

Style: quasi-ohmic

From: Klee Ficker and Heinemann

Updates:
2014 December (Marianne Bezaire): documented
1997 June (Michele Migliore): modified to account for Dax A Current
ENDCOMMENT


UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
}

PARAMETER {
	celsius
        v (mV)
        gmax=.008 (mho/cm2)
        vhalfn=-1   (mV)
        vhalfl=-56   (mV)
        a0l=0.05      (/ms)
        a0n=.1    (/ms)
        zetan=-1.8    (1)
        zetal=3    (1)
        gmn=0.39   (1)
        gml=1   (1)
        lmin=2  (mS)
        nmin=0.2  (mS)
        pw=-1    (1)
        tq=-40
        qq=5
        q10=5
        qtl=1
	ek
	e
}


NEURON {
        SUFFIX ch_KvAdistp :kad
        USEION k READ ek WRITE ik
        RANGE gmax, myi, e, g
        GLOBAL ninf,linf,taul,taun,lmin
}

STATE {
	n
	l
}

ASSIGNED {
	ik (mA/cm2)
	myi (mA/cm2)
	ninf
	linf      
	taul
	taun
	g
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gmax*n*l
	ik = g*(v-ek)
	myi = ik
}

INITIAL {
	rates(v)
	n=ninf
	l=linf
}


FUNCTION alpn(v(mV)) {
LOCAL zeta
  zeta=zetan+pw/(1+exp((v-tq)/qq))
  alpn = exp(1.e-3*zeta*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betn(v(mV)) {
LOCAL zeta
  zeta=zetan+pw/(1+exp((v-tq)/qq))
  betn = exp(1.e-3*zeta*gmn*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION alpl(v(mV)) {
  alpl = exp(1.e-3*zetal*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) 
}

FUNCTION betl(v(mV)) {
  betl = exp(1.e-3*zetal*gml*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius)))
 
}

DERIVATIVE states {  
        rates(v)
        n' = (ninf - n)/taun
        l' = (linf - l)/taul
}

PROCEDURE rates(v (mV)) { :callable from hoc
        LOCAL a,qt
        qt=q10^((celsius-24)/10)
        a = alpn(v)
        ninf = 1/(1 + a)
        taun = betn(v)/(qt*a0n*(1+a))
        if (taun<nmin) {taun=nmin}
        a = alpl(v)
        linf = 1/(1+ a)
        taul = 0.26*(v+50)/qtl
        if (taul<lmin/qtl) {taul=lmin/qtl}
}

