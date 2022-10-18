TITLE Fluctuating conductances

COMMENT
-----------------------------------------------------------------------------

	Fluctuating conductance model for synaptic bombardment
	======================================================

THEORY

  Synaptic bombardment is represented by a stochastic model containing
  two fluctuating conductances g_e(t) and g_i(t) descibed by:

     Isyn = g_e(t) * [V - E_e] + g_i(t) * [V - E_i]
     d g_e / dt = -(g_e - g_e0) / tau_e + sqrt(D_e) * Ft
     d g_i / dt = -(g_i - g_i0) / tau_i + sqrt(D_i) * Ft

  where E_e, E_i are the reversal potentials, g_e0, g_i0 are the average
  conductances, tau_e, tau_i are time constants, D_e, D_i are noise diffusion
  coefficients and Ft is a gaussian white noise of unit standard deviation.

  g_e and g_i are described by an Ornstein-Uhlenbeck (OU) stochastic process
  where tau_e and tau_i represent the "correlation" (if tau_e and tau_i are 
  zero, g_e and g_i are white noise).  The estimation of OU parameters can
  be made from the power spectrum:

     S(w) =  2 * D * tau^2 / (1 + w^2 * tau^2)

  and the diffusion coeffient D is estimated from the variance:

     D = 2 * sigma^2 / tau


NUMERICAL RESOLUTION

  The numerical scheme for integration of OU processes takes advantage 
  of the fact that these processes are gaussian, which led to an exact
  update rule independent of the time step dt (see Gillespie DT, Am J Phys 
  64: 225, 1996):

     x(t+dt) = x(t) * exp(-dt/tau) + A * N(0,1)

  where A = sqrt( D*tau/2 * (1-exp(-2*dt/tau)) ) and N(0,1) is a normal
  random number (avg=0, sigma=1)


IMPLEMENTATION

  This mechanism is implemented as a nonspecific current defined as a
  point process.
  
  Modified 4/7/2015 by Ted Carnevale 
  
  Uses events to control times at which conductances are updated, so
  this mechanism can be used with adaptive integration.  Produces
  exactly same results as original Gfluct2 does if h is set equal to
  dt by a hoc or Python statement and fixed dt integration is used.

  Modified 8/3/2016 by Michael Hines to work with CoreNEURON
  copy/paste/modify fragments from netstim.mod

PARAMETERS

  The mechanism takes the following parameters:

     E_e = 0  (mV)		: reversal potential of excitatory conductance
     E_i = -75 (mV)		: reversal potential of inhibitory conductance

     g_e0 = 0.0121 (umho)	: average excitatory conductance
     g_i0 = 0.0573 (umho)	: average inhibitory conductance

     std_e = 0.0030/2 (umho)	: standard dev of excitatory conductance
     std_i = 0.0066/2 (umho)	: standard dev of inhibitory conductance

     tau_e = 2.728 (ms)		: time constant of excitatory conductance
     tau_i = 10.49 (ms)		: time constant of inhibitory conductance


Gfluct2: conductance cannot be negative


REFERENCE

  Destexhe, A., Rudolph, M., Fellous, J-M. and Sejnowski, T.J.  
  Fluctuating synaptic conductances recreate in-vivo--like activity in
  neocortical neurons. Neuroscience 107: 13-24 (2001).

  (electronic copy available at http://cns.iaf.cnrs-gif.fr)


  A. Destexhe, 1999

-----------------------------------------------------------------------------
ENDCOMMENT




NEURON {
	POINT_PROCESS Gfluct3
	RANGE h, on, g_e, g_i, E_e, E_i, g_e0, g_i0, g_e1, g_i1
	RANGE std_e, std_i, tau_e, tau_i, D_e, D_i
	RANGE new_seed
	NONSPECIFIC_CURRENT i
	THREADSAFE
	BBCOREPOINTER donotuse
}

UNITS {
	(nA) = (nanoamp) 
	(mV) = (millivolt)
	(umho) = (micromho)
}

PARAMETER {
    
     on = 0
     h = 0.025 (ms) : interval at which conductances are to be updated
		    : for fixed dt simulation, should be an integer multiple of dt

     E_e = 0  (mV)		: reversal potential of excitatory conductance
     E_i = -75 (mV)		: reversal potential of inhibitory conductance

     g_e0 = 0.0121 (umho)	: average excitatory conductance
     g_i0 = 0.0573 (umho)	: average inhibitory conductance

     std_e = 0.0030 (umho)	: standard dev of excitatory conductance
     std_i = 0.0066 (umho)	: standard dev of inhibitory conductance

     tau_e = 2.728 (ms)		: time constant of excitatory conductance
     tau_i = 10.49 (ms)		: time constant of inhibitory conductance


}

ASSIGNED {
	v	(mV)		: membrane voltage
	i 	(nA)		: fluctuating current
	ival 	(nA)		: fluctuating current
	g_e	(umho)		: total excitatory conductance
	g_i	(umho)		: total inhibitory conductance
	g_e1	(umho)		: fluctuating excitatory conductance
	g_i1	(umho)		: fluctuating inhibitory conductance
	D_e	(umho umho /ms) : excitatory diffusion coefficient
	D_i	(umho umho /ms) : inhibitory diffusion coefficient
	exp_e
	exp_i
	amp_e	(umho)
	amp_i	(umho)
	donotuse
}

VERBATIM
#if NRNBBCORE /* running in CoreNEURON */

#define IFNEWSTYLE(arg) arg

#else /* running in NEURON */

/*
   1 means noiseFromRandom was called when _ran_compat was previously 0 .
   2 means noiseFromRandom123 was called when _ran_compat was
previously 0.
*/
static int _ran_compat; /* specifies the noise style for all instances */
#define IFNEWSTYLE(arg) if(_ran_compat == 2) { arg }

#endif /* running in NEURON */ 
ENDVERBATIM  

INITIAL {

	VERBATIM
	  if (_p_donotuse) {
	    /* only this style initializes the stream on finitialize */
	    IFNEWSTYLE(nrnran123_setseq((nrnran123_State*)_p_donotuse, 0, 0);)
	  }
	ENDVERBATIM

	g_e1 = 0
	g_i1 = 0
	if(tau_e != 0) {
		D_e = 2 * std_e * std_e / tau_e
		exp_e = exp(-h/tau_e)
		amp_e = std_e * sqrt( (1-exptrap(1, -2*h/tau_e)) )
	}
	if(tau_i != 0) {
		D_i = 2 * std_i * std_i / tau_i
		exp_i = exp(-h/tau_i)
		amp_i = std_i * sqrt( (1-exptrap(2, -2*h/tau_i)) )
	    }
       if ((tau_e != 0) || (tau_i != 0)) {
	   net_send(h, 1)
       }
}

VERBATIM
#include "nrnran123.h"

#if !NRNBBCORE
/* backward compatibility */
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
int nrn_random_isran123(void* r, uint32_t* id1, uint32_t* id2, uint32_t* id3);
#endif
ENDVERBATIM

FUNCTION mynormrand(mean, std) {
VERBATIM
	if (_p_donotuse) {
		// corresponding hoc Random distrubution must be Random.normal(0,1)
		double x;
#if !NRNBBCORE
		if (_ran_compat == 2) {
			x = nrnran123_normal((nrnran123_State*)_p_donotuse);
		}else{		
			x = nrn_random_pick(_p_donotuse);
		}
#else
		#pragma acc routine(nrnran123_normal) seq
		x = nrnran123_normal((nrnran123_State*)_p_donotuse);
#endif
		x = _lmean + _lstd*x;
		return x;
	}
#if !NRNBBCORE
ENDVERBATIM
	mynormrand = normrand(mean, std)
VERBATIM
#endif
ENDVERBATIM
}

:BEFORE BREAKPOINT {
PROCEDURE foo() {
	if (on > 0) {
		g_e = g_e0 + g_e1
		if (g_e < 0) { g_e = 0 }
		g_i = g_i0 + g_i1
		if (g_i < 0) { g_i = 0 }
		ival = g_e * (v - E_e) + g_i * (v - E_i)
	} else {
		ival = 0
	}
	
	:printf("on = %g t = %g v = %g i = %g g_e = %g g_i = %g E_e = %g E_i = %g\n", on, t, v, i, g_e, g_i, E_e, E_i)
    }


BREAKPOINT {
      foo()
      i = ival   
    }


PROCEDURE oup() {		: use Scop function normrand(mean, std_dev)
   if(tau_e!=0) {
	g_e1 =  exp_e * g_e1 + amp_e * mynormrand(0,1)
   }
   if(tau_i!=0) {
	g_i1 =  exp_i * g_i1 + amp_i * mynormrand(0,1)
   }
}


PROCEDURE new_seed(seed) {		: procedure to set the seed
VERBATIM
#if !NRNBBCORE
ENDVERBATIM
	set_seed(seed)
	VERBATIM
	  printf("Setting random generator with seed = %g\n", _lseed);
	ENDVERBATIM
VERBATIM  
#endif
ENDVERBATIM
}

PROCEDURE noiseFromRandom() {
VERBATIM
#if !NRNBBCORE
 {
	void** pv = (void**)(&_p_donotuse);
	if (_ran_compat == 2) {
		fprintf(stderr, "Gfluct3.noiseFromRandom123 was previously called\n");
		assert(0);
	} 
	_ran_compat = 1;
	if (ifarg(1)) {
		*pv = nrn_random_arg(1);
	}else{
		*pv = (void*)0;
	}
 }
#endif
ENDVERBATIM
}

PROCEDURE noiseFromRandom123() {
VERBATIM
#if !NRNBBCORE
 {
	nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);
	if (_ran_compat == 1) {
		fprintf(stderr, "Gfluct3.noiseFromRandom was previously called\n");
		assert(0);
	}
	_ran_compat = 2;
	if (*pv) {
		nrnran123_deletestream(*pv);
		*pv = (nrnran123_State*)0;
	}
	if (ifarg(3)) {
		*pv = nrnran123_newstream3((uint32_t)*getarg(1), (uint32_t)*getarg(2), (uint32_t)*getarg(3));
	}else if (ifarg(2)) {
		*pv = nrnran123_newstream((uint32_t)*getarg(1), (uint32_t)*getarg(2));
	}
 }
#endif
ENDVERBATIM
}

NET_RECEIVE (w) {
    if (flag==1) {
	oup()
	net_send(h, 1)
    }
}


FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    :printf("exptrap Gfluct3 [%f]: x = %f\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}

VERBATIM
#if !NRNBBCORE
static void bbcore_write(double* x, int* d, int* xx, int *offset, _threadargsproto_) {
	/* error if using the legacy normrand */
	if (!_p_donotuse) {
		fprintf(stderr, "Gfluct3: cannot use the legacy normrand generator for the random stream.\n");
		assert(0);
	}
	if (d) {
		uint32_t* di = ((uint32_t*)d) + *offset;
		if (_ran_compat == 1) { 
			void** pv = (void**)(&_p_donotuse);
			/* error if not using Random123 generator */
			if (!nrn_random_isran123(*pv, di, di+1, di+2)) {
				fprintf(stderr, "Gfluct3: Random123 generator is required\n");
				assert(0);
			}
		}else{
			nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);
			nrnran123_getids3(*pv, di, di+1, di+2);
		}
		/*printf("Gfluct3 bbcore_write %d %d %d\n", di[0], di[1], di[3]);*/
	}
	*offset += 3;
}
#endif

static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
	assert(!_p_donotuse);
	uint32_t* di = ((uint32_t*)d) + *offset;
	nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);
	*pv = nrnran123_newstream3(di[0], di[1], di[2]);
	*offset += 3;  
}
ENDVERBATIM

