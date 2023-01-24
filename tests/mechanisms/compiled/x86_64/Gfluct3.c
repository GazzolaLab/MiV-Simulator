/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__Gfluct3
#define _nrn_initial _nrn_initial__Gfluct3
#define nrn_cur _nrn_cur__Gfluct3
#define _nrn_current _nrn_current__Gfluct3
#define nrn_jacob _nrn_jacob__Gfluct3
#define nrn_state _nrn_state__Gfluct3
#define _net_receive _net_receive__Gfluct3 
#define foo foo__Gfluct3 
#define noiseFromRandom123 noiseFromRandom123__Gfluct3 
#define noiseFromRandom noiseFromRandom__Gfluct3 
#define new_seed new_seed__Gfluct3 
#define oup oup__Gfluct3 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define on _p[0]
#define on_columnindex 0
#define h _p[1]
#define h_columnindex 1
#define E_e _p[2]
#define E_e_columnindex 2
#define E_i _p[3]
#define E_i_columnindex 3
#define g_e0 _p[4]
#define g_e0_columnindex 4
#define g_i0 _p[5]
#define g_i0_columnindex 5
#define std_e _p[6]
#define std_e_columnindex 6
#define std_i _p[7]
#define std_i_columnindex 7
#define tau_e _p[8]
#define tau_e_columnindex 8
#define tau_i _p[9]
#define tau_i_columnindex 9
#define i _p[10]
#define i_columnindex 10
#define g_e _p[11]
#define g_e_columnindex 11
#define g_i _p[12]
#define g_i_columnindex 12
#define g_e1 _p[13]
#define g_e1_columnindex 13
#define g_i1 _p[14]
#define g_i1_columnindex 14
#define D_e _p[15]
#define D_e_columnindex 15
#define D_i _p[16]
#define D_i_columnindex 16
#define ival _p[17]
#define ival_columnindex 17
#define exp_e _p[18]
#define exp_e_columnindex 18
#define exp_i _p[19]
#define exp_i_columnindex 19
#define amp_e _p[20]
#define amp_e_columnindex 20
#define amp_i _p[21]
#define amp_i_columnindex 21
#define v _p[22]
#define v_columnindex 22
#define _g _p[23]
#define _g_columnindex 23
#define _tsav _p[24]
#define _tsav_columnindex 24
#define _nd_area  *_ppvar[0]._pval
#define donotuse	*_ppvar[2]._pval
#define _p_donotuse	_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  2;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_exptrap(void*);
 static double _hoc_foo(void*);
 static double _hoc_mynormrand(void*);
 static double _hoc_noiseFromRandom123(void*);
 static double _hoc_noiseFromRandom(void*);
 static double _hoc_new_seed(void*);
 static double _hoc_oup(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "exptrap", _hoc_exptrap,
 "foo", _hoc_foo,
 "mynormrand", _hoc_mynormrand,
 "noiseFromRandom123", _hoc_noiseFromRandom123,
 "noiseFromRandom", _hoc_noiseFromRandom,
 "new_seed", _hoc_new_seed,
 "oup", _hoc_oup,
 0, 0
};
#define exptrap exptrap_Gfluct3
#define mynormrand mynormrand_Gfluct3
 extern double exptrap( _threadargsprotocomma_ double , double );
 extern double mynormrand( _threadargsprotocomma_ double , double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "h", "ms",
 "E_e", "mV",
 "E_i", "mV",
 "g_e0", "umho",
 "g_i0", "umho",
 "std_e", "umho",
 "std_i", "umho",
 "tau_e", "ms",
 "tau_i", "ms",
 "i", "nA",
 "g_e", "umho",
 "g_i", "umho",
 "g_e1", "umho",
 "g_i1", "umho",
 "D_e", "umho umho /ms",
 "D_i", "umho umho /ms",
 0,0
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Gfluct3",
 "on",
 "h",
 "E_e",
 "E_i",
 "g_e0",
 "g_i0",
 "std_e",
 "std_i",
 "tau_e",
 "tau_i",
 0,
 "i",
 "g_e",
 "g_i",
 "g_e1",
 "g_i1",
 "D_e",
 "D_i",
 0,
 0,
 "donotuse",
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 25, _prop);
 	/*initialize range parameters*/
 	on = 0;
 	h = 0.025;
 	E_e = 0;
 	E_i = -75;
 	g_e0 = 0.0121;
 	g_i0 = 0.0573;
 	std_e = 0.003;
 	std_i = 0.0066;
 	tau_e = 2.728;
 	tau_i = 10.49;
  }
 	_prop->param = _p;
 	_prop->param_size = 25;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[3]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void bbcore_write(double*, int*, int*, int*, _threadargsproto_);
 extern void hoc_reg_bbcore_write(int, void(*)(double*, int*, int*, int*, _threadargsproto_));
 static void bbcore_read(double*, int*, int*, int*, _threadargsproto_);
 extern void hoc_reg_bbcore_read(int, void(*)(double*, int*, int*, int*, _threadargsproto_));
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Gfluct3_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
   hoc_reg_bbcore_write(_mechtype, bbcore_write);
   hoc_reg_bbcore_read(_mechtype, bbcore_read);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 25, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "bbcorepointer");
  hoc_register_dparam_semantics(_mechtype, 3, "netsend");
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Gfluct3 /work2/08818/fg14/frontera/mind-in-vitro/MiV-Simulator/tests/mechanisms/compiled/Gfluct3.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Fluctuating conductances";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int foo(_threadargsproto_);
static int noiseFromRandom123(_threadargsproto_);
static int noiseFromRandom(_threadargsproto_);
static int new_seed(_threadargsprotocomma_ double);
static int oup(_threadargsproto_);
 
/*VERBATIM*/
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
 
/*VERBATIM*/
#include "nrnran123.h"
 
double mynormrand ( _threadargsprotocomma_ double _lmean , double _lstd ) {
   double _lmynormrand;
 
/*VERBATIM*/
	if (_p_donotuse) {
		// corresponding hoc Random distrubution must be Random.normal(0,1)
		double x;
#if !NRNBBCORE
		if (_ran_compat == 2) {
			x = nrnran123_normal((nrnran123_State*)_p_donotuse);
		}else{		
			x = nrn_random_pick((Rand*)_p_donotuse);
		}
#else
		#pragma acc routine(nrnran123_normal) seq
		x = nrnran123_normal((nrnran123_State*)_p_donotuse);
#endif
		x = _lmean + _lstd*x;
		return x;
	}
#if !NRNBBCORE
 _lmynormrand = normrand ( _lmean , _lstd ) ;
   
/*VERBATIM*/
#endif
 
return _lmynormrand;
 }
 
static double _hoc_mynormrand(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  mynormrand ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}
 
static int  foo ( _threadargsproto_ ) {
   if ( on > 0.0 ) {
     g_e = g_e0 + g_e1 ;
     if ( g_e < 0.0 ) {
       g_e = 0.0 ;
       }
     g_i = g_i0 + g_i1 ;
     if ( g_i < 0.0 ) {
       g_i = 0.0 ;
       }
     ival = g_e * ( v - E_e ) + g_i * ( v - E_i ) ;
     }
   else {
     ival = 0.0 ;
     }
    return 0; }
 
static double _hoc_foo(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 foo ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int  oup ( _threadargsproto_ ) {
   if ( tau_e  != 0.0 ) {
     g_e1 = exp_e * g_e1 + amp_e * mynormrand ( _threadargscomma_ 0.0 , 1.0 ) ;
     }
   if ( tau_i  != 0.0 ) {
     g_i1 = exp_i * g_i1 + amp_i * mynormrand ( _threadargscomma_ 0.0 , 1.0 ) ;
     }
    return 0; }
 
static double _hoc_oup(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 oup ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int  new_seed ( _threadargsprotocomma_ double _lseed ) {
   
/*VERBATIM*/
#if !NRNBBCORE
 set_seed ( _lseed ) ;
   
/*VERBATIM*/
	  printf("Setting random generator with seed = %g\n", _lseed);
 
/*VERBATIM*/
 
#endif
  return 0; }
 
static double _hoc_new_seed(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 new_seed ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static int  noiseFromRandom ( _threadargsproto_ ) {
   
/*VERBATIM*/
#if !NRNBBCORE
 {
	Rand** pv = (Rand**)(&_p_donotuse);
	if (_ran_compat == 2) {
		fprintf(stderr, "Gfluct3.noiseFromRandom123 was previously called\n");
		assert(0);
	} 
	_ran_compat = 1;
	if (ifarg(1)) {
		*pv = nrn_random_arg(1);
	}else{
		*pv = (Rand*)0;
	}
 }
#endif
  return 0; }
 
static double _hoc_noiseFromRandom(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 noiseFromRandom ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int  noiseFromRandom123 ( _threadargsproto_ ) {
   
/*VERBATIM*/
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
  return 0; }
 
static double _hoc_noiseFromRandom123(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 noiseFromRandom123 ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 1.0 ) {
     oup ( _threadargs_ ) ;
     net_send ( _tqitem, _args, _pnt, t +  h , 1.0 ) ;
     }
   } }
 
double exptrap ( _threadargsprotocomma_ double _lloc , double _lx ) {
   double _lexptrap;
 if ( _lx >= 700.0 ) {
     _lexptrap = exp ( 700.0 ) ;
     }
   else {
     _lexptrap = exp ( _lx ) ;
     }
   
return _lexptrap;
 }
 
static double _hoc_exptrap(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  exptrap ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}
 
/*VERBATIM*/
static void bbcore_write(double* x, int* d, int* xx, int *offset, _threadargsproto_) {
	/* error if using the legacy normrand */
	if (!_p_donotuse) {
		fprintf(stderr, "Gfluct3: cannot use the legacy normrand generator for the random stream.\n");
		assert(0);
	}
	if (d) {
		uint32_t* di = ((uint32_t*)d) + *offset;
#if !NRNBBCORE
		if (_ran_compat == 1) { 
			char which;
			Rand** pv = (Rand**)(&_p_donotuse);
			/* error if not using Random123 generator */
			if (!nrn_random_isran123(*pv, di, di+1, di+2)) {
				fprintf(stderr, "Gfluct3: Random123 generator is required\n");
				assert(0);
			}
			/* because coreneuron psolve may not start at t=0 also need the sequence */
			nrn_random123_getseq(*pv, di+3, &which);
			di[4] = (int)which;
		}else{
#else
	{
#endif
			char which;
			nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);
			nrnran123_getids3(*pv, di, di+1, di+2);
			nrnran123_getseq(*pv, di+3, &which);
			di[4] = (int)which;
		}
		/*printf("Gfluct3 bbcore_write %d %d %d %d %d\n", di[0], di[1], di[2], di[3], di[4]);*/
	}
	*offset += 5;
}

static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
	uint32_t* di = ((uint32_t*)d) + *offset;
	nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);
#if !NRNBBCORE
	assert(_ran_compat == 2);
#endif
	if (pv) {
		nrnran123_deletestream(*pv);
	}
	*pv = nrnran123_newstream3(di[0], di[1], di[2]);
	nrnran123_setseq(*pv, di[3], (char)di[4]);
	*offset += 5;
}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
 {
   
/*VERBATIM*/
	  if (_p_donotuse) {
	    /* only this style initializes the stream on finitialize */
	    IFNEWSTYLE(nrnran123_setseq((nrnran123_State*)_p_donotuse, 0, 0);)
	  }
 g_e1 = 0.0 ;
   g_i1 = 0.0 ;
   if ( tau_e  != 0.0 ) {
     D_e = 2.0 * std_e * std_e / tau_e ;
     exp_e = exp ( - h / tau_e ) ;
     amp_e = std_e * sqrt ( ( 1.0 - exptrap ( _threadargscomma_ 1.0 , - 2.0 * h / tau_e ) ) ) ;
     }
   if ( tau_i  != 0.0 ) {
     D_i = 2.0 * std_i * std_i / tau_i ;
     exp_i = exp ( - h / tau_i ) ;
     amp_i = std_i * sqrt ( ( 1.0 - exptrap ( _threadargscomma_ 2.0 , - 2.0 * h / tau_i ) ) ) ;
     }
   if ( ( tau_e  != 0.0 )  || ( tau_i  != 0.0 ) ) {
     net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  h , 1.0 ) ;
     }
   }

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   foo ( _threadargs_ ) ;
   i = ival ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/work2/08818/fg14/frontera/mind-in-vitro/MiV-Simulator/tests/mechanisms/compiled/Gfluct3.mod";
static const char* nmodl_file_text = 
  "TITLE Fluctuating conductances\n"
  "\n"
  "COMMENT\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "	Fluctuating conductance model for synaptic bombardment\n"
  "	======================================================\n"
  "\n"
  "THEORY\n"
  "\n"
  "  Synaptic bombardment is represented by a stochastic model containing\n"
  "  two fluctuating conductances g_e(t) and g_i(t) descibed by:\n"
  "\n"
  "     Isyn = g_e(t) * [V - E_e] + g_i(t) * [V - E_i]\n"
  "     d g_e / dt = -(g_e - g_e0) / tau_e + sqrt(D_e) * Ft\n"
  "     d g_i / dt = -(g_i - g_i0) / tau_i + sqrt(D_i) * Ft\n"
  "\n"
  "  where E_e, E_i are the reversal potentials, g_e0, g_i0 are the average\n"
  "  conductances, tau_e, tau_i are time constants, D_e, D_i are noise diffusion\n"
  "  coefficients and Ft is a gaussian white noise of unit standard deviation.\n"
  "\n"
  "  g_e and g_i are described by an Ornstein-Uhlenbeck (OU) stochastic process\n"
  "  where tau_e and tau_i represent the \"correlation\" (if tau_e and tau_i are \n"
  "  zero, g_e and g_i are white noise).  The estimation of OU parameters can\n"
  "  be made from the power spectrum:\n"
  "\n"
  "     S(w) =  2 * D * tau^2 / (1 + w^2 * tau^2)\n"
  "\n"
  "  and the diffusion coeffient D is estimated from the variance:\n"
  "\n"
  "     D = 2 * sigma^2 / tau\n"
  "\n"
  "\n"
  "NUMERICAL RESOLUTION\n"
  "\n"
  "  The numerical scheme for integration of OU processes takes advantage \n"
  "  of the fact that these processes are gaussian, which led to an exact\n"
  "  update rule independent of the time step dt (see Gillespie DT, Am J Phys \n"
  "  64: 225, 1996):\n"
  "\n"
  "     x(t+dt) = x(t) * exp(-dt/tau) + A * N(0,1)\n"
  "\n"
  "  where A = sqrt( D*tau/2 * (1-exp(-2*dt/tau)) ) and N(0,1) is a normal\n"
  "  random number (avg=0, sigma=1)\n"
  "\n"
  "\n"
  "IMPLEMENTATION\n"
  "\n"
  "  This mechanism is implemented as a nonspecific current defined as a\n"
  "  point process.\n"
  "  \n"
  "  Modified 4/7/2015 by Ted Carnevale \n"
  "  \n"
  "  Uses events to control times at which conductances are updated, so\n"
  "  this mechanism can be used with adaptive integration.  Produces\n"
  "  exactly same results as original Gfluct2 does if h is set equal to\n"
  "  dt by a hoc or Python statement and fixed dt integration is used.\n"
  "\n"
  "  Modified 8/3/2016 by Michael Hines to work with CoreNEURON\n"
  "  copy/paste/modify fragments from netstim.mod\n"
  "\n"
  "PARAMETERS\n"
  "\n"
  "  The mechanism takes the following parameters:\n"
  "\n"
  "     E_e = 0  (mV)		: reversal potential of excitatory conductance\n"
  "     E_i = -75 (mV)		: reversal potential of inhibitory conductance\n"
  "\n"
  "     g_e0 = 0.0121 (umho)	: average excitatory conductance\n"
  "     g_i0 = 0.0573 (umho)	: average inhibitory conductance\n"
  "\n"
  "     std_e = 0.0030/2 (umho)	: standard dev of excitatory conductance\n"
  "     std_i = 0.0066/2 (umho)	: standard dev of inhibitory conductance\n"
  "\n"
  "     tau_e = 2.728 (ms)		: time constant of excitatory conductance\n"
  "     tau_i = 10.49 (ms)		: time constant of inhibitory conductance\n"
  "\n"
  "\n"
  "Gfluct2: conductance cannot be negative\n"
  "\n"
  "\n"
  "REFERENCE\n"
  "\n"
  "  Destexhe, A., Rudolph, M., Fellous, J-M. and Sejnowski, T.J.  \n"
  "  Fluctuating synaptic conductances recreate in-vivo--like activity in\n"
  "  neocortical neurons. Neuroscience 107: 13-24 (2001).\n"
  "\n"
  "  (electronic copy available at http://cns.iaf.cnrs-gif.fr)\n"
  "\n"
  "\n"
  "  A. Destexhe, 1999\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS Gfluct3\n"
  "	RANGE h, on, g_e, g_i, E_e, E_i, g_e0, g_i0, g_e1, g_i1\n"
  "	RANGE std_e, std_i, tau_e, tau_i, D_e, D_i\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	THREADSAFE\n"
  "	BBCOREPOINTER donotuse\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp) \n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    \n"
  "     on = 0\n"
  "     h = 0.025 (ms) : interval at which conductances are to be updated\n"
  "		    : for fixed dt simulation, should be an integer multiple of dt\n"
  "\n"
  "     E_e = 0  (mV)		: reversal potential of excitatory conductance\n"
  "     E_i = -75 (mV)		: reversal potential of inhibitory conductance\n"
  "\n"
  "     g_e0 = 0.0121 (umho)	: average excitatory conductance\n"
  "     g_i0 = 0.0573 (umho)	: average inhibitory conductance\n"
  "\n"
  "     std_e = 0.0030 (umho)	: standard dev of excitatory conductance\n"
  "     std_i = 0.0066 (umho)	: standard dev of inhibitory conductance\n"
  "\n"
  "     tau_e = 2.728 (ms)		: time constant of excitatory conductance\n"
  "     tau_i = 10.49 (ms)		: time constant of inhibitory conductance\n"
  "\n"
  "\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	v	(mV)		: membrane voltage\n"
  "	i 	(nA)		: fluctuating current\n"
  "	ival 	(nA)		: fluctuating current\n"
  "	g_e	(umho)		: total excitatory conductance\n"
  "	g_i	(umho)		: total inhibitory conductance\n"
  "	g_e1	(umho)		: fluctuating excitatory conductance\n"
  "	g_i1	(umho)		: fluctuating inhibitory conductance\n"
  "	D_e	(umho umho /ms) : excitatory diffusion coefficient\n"
  "	D_i	(umho umho /ms) : inhibitory diffusion coefficient\n"
  "	exp_e\n"
  "	exp_i\n"
  "	amp_e	(umho)\n"
  "	amp_i	(umho)\n"
  "	donotuse\n"
  "}\n"
  "\n"
  "VERBATIM\n"
  "#if NRNBBCORE /* running in CoreNEURON */\n"
  "\n"
  "#define IFNEWSTYLE(arg) arg\n"
  "\n"
  "#else /* running in NEURON */\n"
  "\n"
  "/*\n"
  "   1 means noiseFromRandom was called when _ran_compat was previously 0 .\n"
  "   2 means noiseFromRandom123 was called when _ran_compat was\n"
  "previously 0.\n"
  "*/\n"
  "static int _ran_compat; /* specifies the noise style for all instances */\n"
  "#define IFNEWSTYLE(arg) if(_ran_compat == 2) { arg }\n"
  "\n"
  "#endif /* running in NEURON */ \n"
  "ENDVERBATIM  \n"
  "\n"
  "INITIAL {\n"
  "\n"
  "	VERBATIM\n"
  "	  if (_p_donotuse) {\n"
  "	    /* only this style initializes the stream on finitialize */\n"
  "	    IFNEWSTYLE(nrnran123_setseq((nrnran123_State*)_p_donotuse, 0, 0);)\n"
  "	  }\n"
  "	ENDVERBATIM\n"
  "\n"
  "	g_e1 = 0\n"
  "	g_i1 = 0\n"
  "	if(tau_e != 0) {\n"
  "		D_e = 2 * std_e * std_e / tau_e\n"
  "		exp_e = exp(-h/tau_e)\n"
  "		amp_e = std_e * sqrt( (1-exptrap(1, -2*h/tau_e)) )\n"
  "	}\n"
  "	if(tau_i != 0) {\n"
  "		D_i = 2 * std_i * std_i / tau_i\n"
  "		exp_i = exp(-h/tau_i)\n"
  "		amp_i = std_i * sqrt( (1-exptrap(2, -2*h/tau_i)) )\n"
  "	    }\n"
  "       if ((tau_e != 0) || (tau_i != 0)) {\n"
  "	   net_send(h, 1)\n"
  "       }\n"
  "}\n"
  "\n"
  "VERBATIM\n"
  "#include \"nrnran123.h\"\n"
  "ENDVERBATIM\n"
  "\n"
  "FUNCTION mynormrand(mean, std) {\n"
  "VERBATIM\n"
  "	if (_p_donotuse) {\n"
  "		// corresponding hoc Random distrubution must be Random.normal(0,1)\n"
  "		double x;\n"
  "#if !NRNBBCORE\n"
  "		if (_ran_compat == 2) {\n"
  "			x = nrnran123_normal((nrnran123_State*)_p_donotuse);\n"
  "		}else{		\n"
  "			x = nrn_random_pick((Rand*)_p_donotuse);\n"
  "		}\n"
  "#else\n"
  "		#pragma acc routine(nrnran123_normal) seq\n"
  "		x = nrnran123_normal((nrnran123_State*)_p_donotuse);\n"
  "#endif\n"
  "		x = _lmean + _lstd*x;\n"
  "		return x;\n"
  "	}\n"
  "#if !NRNBBCORE\n"
  "ENDVERBATIM\n"
  "	mynormrand = normrand(mean, std)\n"
  "VERBATIM\n"
  "#endif\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  ":BEFORE BREAKPOINT {\n"
  "PROCEDURE foo() {\n"
  "	if (on > 0) {\n"
  "		g_e = g_e0 + g_e1\n"
  "		if (g_e < 0) { g_e = 0 }\n"
  "		g_i = g_i0 + g_i1\n"
  "		if (g_i < 0) { g_i = 0 }\n"
  "		ival = g_e * (v - E_e) + g_i * (v - E_i)\n"
  "	} else {\n"
  "		ival = 0\n"
  "	}\n"
  "	\n"
  "	:printf(\"on = %g t = %g v = %g i = %g g_e = %g g_i = %g E_e = %g E_i = %g\\n\", on, t, v, i, g_e, g_i, E_e, E_i)\n"
  "    }\n"
  "\n"
  "\n"
  "BREAKPOINT {\n"
  "      foo()\n"
  "      i = ival   \n"
  "    }\n"
  "\n"
  "\n"
  "PROCEDURE oup() {		: use Scop function normrand(mean, std_dev)\n"
  "   if(tau_e!=0) {\n"
  "	g_e1 =  exp_e * g_e1 + amp_e * mynormrand(0,1)\n"
  "   }\n"
  "   if(tau_i!=0) {\n"
  "	g_i1 =  exp_i * g_i1 + amp_i * mynormrand(0,1)\n"
  "   }\n"
  "}\n"
  "\n"
  "\n"
  "PROCEDURE new_seed(seed) {		: procedure to set the seed\n"
  "VERBATIM\n"
  "#if !NRNBBCORE\n"
  "ENDVERBATIM\n"
  "	set_seed(seed)\n"
  "	VERBATIM\n"
  "	  printf(\"Setting random generator with seed = %g\\n\", _lseed);\n"
  "	ENDVERBATIM\n"
  "VERBATIM  \n"
  "#endif\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "PROCEDURE noiseFromRandom() {\n"
  "VERBATIM\n"
  "#if !NRNBBCORE\n"
  " {\n"
  "	Rand** pv = (Rand**)(&_p_donotuse);\n"
  "	if (_ran_compat == 2) {\n"
  "		fprintf(stderr, \"Gfluct3.noiseFromRandom123 was previously called\\n\");\n"
  "		assert(0);\n"
  "	} \n"
  "	_ran_compat = 1;\n"
  "	if (ifarg(1)) {\n"
  "		*pv = nrn_random_arg(1);\n"
  "	}else{\n"
  "		*pv = (Rand*)0;\n"
  "	}\n"
  " }\n"
  "#endif\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "PROCEDURE noiseFromRandom123() {\n"
  "VERBATIM\n"
  "#if !NRNBBCORE\n"
  " {\n"
  "	nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);\n"
  "	if (_ran_compat == 1) {\n"
  "		fprintf(stderr, \"Gfluct3.noiseFromRandom was previously called\\n\");\n"
  "		assert(0);\n"
  "	}\n"
  "	_ran_compat = 2;\n"
  "	if (*pv) {\n"
  "		nrnran123_deletestream(*pv);\n"
  "		*pv = (nrnran123_State*)0;\n"
  "	}\n"
  "	if (ifarg(3)) {\n"
  "		*pv = nrnran123_newstream3((uint32_t)*getarg(1), (uint32_t)*getarg(2), (uint32_t)*getarg(3));\n"
  "	}else if (ifarg(2)) {\n"
  "		*pv = nrnran123_newstream((uint32_t)*getarg(1), (uint32_t)*getarg(2));\n"
  "	}\n"
  " }\n"
  "#endif\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "    if (flag==1) {\n"
  "	oup()\n"
  "	net_send(h, 1)\n"
  "    }\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION exptrap(loc,x) {\n"
  "  if (x>=700.0) {\n"
  "    :printf(\"exptrap Gfluct3 [%f]: x = %f\\n\", loc, x)\n"
  "    exptrap = exp(700.0)\n"
  "  } else {\n"
  "    exptrap = exp(x)\n"
  "  }\n"
  "}\n"
  "\n"
  "VERBATIM\n"
  "static void bbcore_write(double* x, int* d, int* xx, int *offset, _threadargsproto_) {\n"
  "	/* error if using the legacy normrand */\n"
  "	if (!_p_donotuse) {\n"
  "		fprintf(stderr, \"Gfluct3: cannot use the legacy normrand generator for the random stream.\\n\");\n"
  "		assert(0);\n"
  "	}\n"
  "	if (d) {\n"
  "		uint32_t* di = ((uint32_t*)d) + *offset;\n"
  "#if !NRNBBCORE\n"
  "		if (_ran_compat == 1) { \n"
  "			char which;\n"
  "			Rand** pv = (Rand**)(&_p_donotuse);\n"
  "			/* error if not using Random123 generator */\n"
  "			if (!nrn_random_isran123(*pv, di, di+1, di+2)) {\n"
  "				fprintf(stderr, \"Gfluct3: Random123 generator is required\\n\");\n"
  "				assert(0);\n"
  "			}\n"
  "			/* because coreneuron psolve may not start at t=0 also need the sequence */\n"
  "			nrn_random123_getseq(*pv, di+3, &which);\n"
  "			di[4] = (int)which;\n"
  "		}else{\n"
  "#else\n"
  "	{\n"
  "#endif\n"
  "			char which;\n"
  "			nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);\n"
  "			nrnran123_getids3(*pv, di, di+1, di+2);\n"
  "			nrnran123_getseq(*pv, di+3, &which);\n"
  "			di[4] = (int)which;\n"
  "		}\n"
  "		/*printf(\"Gfluct3 bbcore_write %d %d %d %d %d\\n\", di[0], di[1], di[2], di[3], di[4]);*/\n"
  "	}\n"
  "	*offset += 5;\n"
  "}\n"
  "\n"
  "static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {\n"
  "	uint32_t* di = ((uint32_t*)d) + *offset;\n"
  "	nrnran123_State** pv = (nrnran123_State**)(&_p_donotuse);\n"
  "#if !NRNBBCORE\n"
  "	assert(_ran_compat == 2);\n"
  "#endif\n"
  "	if (pv) {\n"
  "		nrnran123_deletestream(*pv);\n"
  "	}\n"
  "	*pv = nrnran123_newstream3(di[0], di[1], di[2]);\n"
  "	nrnran123_setseq(*pv, di[3], (char)di[4]);\n"
  "	*offset += 5;\n"
  "}\n"
  "ENDVERBATIM\n"
  "\n"
  ;
#endif
