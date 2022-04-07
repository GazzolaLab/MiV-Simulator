
import numpy as np
from numpy import array
from numpy import log
from scipy.interpolate import Akima1DInterpolator


"""
neurotools.stgen
================

A collection of tools for stochastic process generation.


Classes
-------

StGen - Object to generate stochastic processes of various kinds
and return them as SpikeTrain or AnalogSignal objects.


Functions
---------

shotnoise_fromspikes - Convolves the provided spike train with shot decaying exponential.

gamma_hazard - Compute the hazard function for a gamma process with parameters a,b.
"""


def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None):
    """
    Given a time series of instantaneous spike rates in Hz, produce a spike train consistent 
    with an inhomogeneous Poisson process with a refractory period after each spike.
    :param rate: instantaneous rates in time (Hz)
    :param t: corresponding time values (ms)
    :param dt: temporal resolution for spike times (ms)
    :param refractory: absolute deadtime following a spike (ms)
    :param generator: :class:'np.random.RandomState()'
    :return: list of m spike times (ms)
    """
    if generator is None:
        generator = random
    interp_t = np.arange(t[0], t[-1] + dt, dt)
#    try:
    rate[np.isclose(rate, 0., atol=1e-3, rtol=1e-3)] = 0.
    rate_ip = Akima1DInterpolator(t, rate)
    interp_rate = rate_ip(interp_t)
#    except Exception:
#        print('t shape: %s rate shape: %s' % (str(t.shape), str(rate.shape)))
    interp_rate /= 1000.
    spike_times = []
    non_zero = np.where(interp_rate > 1.e-100)[0]
    if len(non_zero) == 0:
        return spike_times
    interp_rate[non_zero] = 1. / (1. / interp_rate[non_zero] - refractory)
    max_rate = np.max(interp_rate)
    if not max_rate > 0.:
        return spike_times
    i = 0
    ISI_memory = 0.
    while i < len(interp_t):
        x = generator.uniform(0.0, 1.0)
        if x > 0.:
            ISI = -np.log(x) / max_rate
            i += int(ISI / dt)
            ISI_memory += ISI
            if (i < len(interp_t)) and (generator.uniform(0.0, 1.0) <= (interp_rate[i] / max_rate)) and ISI_memory >= 0.:
                spike_times.append(interp_t[i])
                ISI_memory = -refractory
    return spike_times


class StGen(object):

    def __init__(self, rng=None, seed=None):
        """ 
        Stochastic Process Generator
        ============================

        Object to generate stochastic processes of various kinds
        and return them as SpikeTrain or AnalogSignal objects.
      

        Inputs:
            rng - The random number generator state object (optional). Can be None, or 
            a np.random.RandomState object, or an object with the same 
            interface.

            seed - A seed for the rng (optional).

        If rng is not None, the provided rng will be used to generate random numbers, 
        otherwise StGen will create its own random number generator.
        If a seed is provided, it is passed to rng.seed(seed)

        Examples:
            >> x = StGen()



        StGen Methods:

        Spiking point processes:
        ------------------------
 
        poisson_generator - homogeneous Poisson process
        inh_poisson_generator - inhomogeneous Poisson process (time varying rate)
        inh_adaptingmarkov_generator - inhomogeneous adapting markov process (time varying)
        inh_2Dadaptingmarkov_generator - inhomogeneous adapting and 
        refractory markov process (time varying)

        Continuous time processes:
        --------------------------

        OU_generator - Ohrnstein-Uhlenbeck process
        

        See also:
          shotnoise_fromspikes

        """

        if rng == None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if seed != None:
            self.rng.seed(seed)
        self.rpy_checked = False

    def seed(self, seed):
        """ seed the rng with a given seed """
        self.rng.seed(seed)

    def poisson_generator(self, rate, t_start=0.0, t_stop=1000.0, debug=False):
        """
        Returns a SpikeTrain whose spikes are a realization of a Poisson process
        with the given rate (Hz) and stopping time t_stop (milliseconds).

        Note: t_start is always 0.0, thus all realizations are as if 
        they spiked at t=0.0, though this spike is not included in the SpikeList.

        Inputs:
            rate    - the rate of the discharge (in Hz)
            t_start - the beginning of the SpikeTrain (in ms)
            t_stop  - the end of the SpikeTrain (in ms)
            rather than a SpikeTrain object.

        Examples:
            >> gen.poisson_generator(50, 0, 1000)
         
        See also:
            inh_poisson_generator, inh_adaptingmarkov_generator
        """

        # number = int((t_stop-t_start)/1000.0*2.0*rate)

        # less wasteful than double length method above
        n = (t_stop - t_start) / 1000.0 * rate
        number = np.ceil(n + 3 * np.sqrt(n))
        if number < 100:
            number = min(5 + np.ceil(2 * n), 100)

        if number > 0:
            isi = self.rng.exponential(1.0 / rate, int(number)) * 1000.0
            if number > 1:
                spikes = np.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = np.array([])

        spikes += t_start
        i = np.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i == len(spikes):
            # ISI buf overrun

            t_last = spikes[-1] + self.rng.exponential(1.0 / rate, 1)[0] * 1000.0

            while (t_last < t_stop):
                extra_spikes.append(t_last)
                t_last += self.rng.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))

            if debug:
                print("ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (
                len(spikes), len(extra_spikes)))


        else:
            spikes = np.resize(spikes, (i,))

        if debug:
            return spikes, extra_spikes
        else:
            return spikes

    def inh_poisson_generator(self, rate, t, t_stop):
        """
        Returns a SpikeTrain whose spikes are a realization of an inhomogeneous 
        poisson process (dynamic rate). The implementation uses the thinning 
        method, as presented in the references.

        Inputs:
            rate   - an array of the rates (Hz) where rate[i] is active on interval 
            [t[i],t[i+1]]
            t      - an array specifying the time bins (in milliseconds) at which to 
            specify the rate
            t_stop - length of time to simulate process (in ms)

        Note:
            t_start=t[0]

        References:

        Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier 
        Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
        Neural Comput. 2007 19: 2958-3010.
        
        Devroye, L. (1986). Non-uniform random variate generation. New York: Springer-Verlag.

        Examples:
            >> time = arange(0,1000)
            >> stgen.inh_poisson_generator(time,sin(time), 1000)

        See also:
            poisson_generator, inh_adaptingmarkov_generator
        """

        if np.shape(t) != np.shape(rate):
            raise ValueError('shape mismatch: t,rate must be of the same shape')

        # get max rate and generate poisson process to be thinned
        rmax = np.max(rate)
        ps = self.poisson_generator(rmax, t_start=t[0], t_stop=t_stop)

        # return empty if no spikes
        if len(ps) == 0:
            return np.array([])

        # gen uniform rand on 0,1 for each spike
        rn = np.array(self.rng.uniform(0, 1, len(ps)))

        # instantaneous rate for each spike

        idx = np.searchsorted(t, ps) - 1
        spike_rate = rate[idx]

        # thin and return spikes
        spike_train = ps[rn < (spike_rate / rmax)]

        return spike_train

    def _inh_adaptingmarkov_generator_python(self, a, bq, tau, t, t_stop):

        """
        Returns a SpikeList whose spikes are an inhomogeneous
        realization (dynamic rate) of the so-called adapting markov
        process (see references). The implementation uses the thinning
        method, as presented in the references.

        This is the 1d implementation, with no relative refractoriness.
        For the 2d implementation with relative refractoriness, 
        see the inh_2dadaptingmarkov_generator.

        Inputs:
            a,bq    - arrays of the parameters of the hazard function where a[i] and bq[i] 
            will be active on interval [t[i],t[i+1]]
            tau    - the time constant of adaptation (in milliseconds).
            t      - an array specifying the time bins (in milliseconds) at which to 
            specify the rate
            t_stop - length of time to simulate process (in ms)

        Note: 
            - t_start=t[0]

            - a is in units of Hz.  Typical values are available 
              in Fig. 1 of Muller et al 2007, a~5-80Hz (low to high stimulus)

            - bq here is taken to be the quantity b*q_s in Muller et al 2007, is thus
              dimensionless, and has typical values bq~3.0-1.0 (low to high stimulus)

            - tau_s has typical values on the order of 100 ms


        References:

        Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier 
        Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
        Neural Comput. 2007 19: 2958-3010.
        
        Devroye, L. (1986). Non-uniform random variate generation. New York: Springer-Verlag.

        Examples:
            See source:trunk/examples/stgen/inh_2Dmarkov_psth.py

        
        See also:
            inh_poisson_generator, inh_gamma_generator, inh_2dadaptingmarkov_generator

        """

        from numpy import shape

        if shape(t) != shape(a) or shape(a) != shape(bq):
            raise ValueError('shape mismatch: t,a,b must be of the same shape')

        # get max rate and generate poisson process to be thinned
        rmax = np.max(a)
        ps = self.poisson_generator(rmax, t_start=t[0], t_stop=t_stop)

        isi = np.zeros_like(ps)
        isi[1:] = ps[1:] - ps[:-1]
        isi[0] = ps[0]  # -0.0 # assume spike at 0.0

        # return empty if no spikes
        if len(ps) == 0:
            return SpikeTrain(np.array([]), t_start=t[0], t_stop=t_stop)

        # gen uniform rand on 0,1 for each spike
        rn = np.array(self.rng.uniform(0, 1, len(ps)))

        # instantaneous a,bq for each spike

        idx = np.searchsorted(t, ps) - 1
        spike_a = a[idx]
        spike_bq = bq[idx]

        keep = np.zeros(shape(ps), bool)

        # thin spikes

        i = 0
        t_last = 0.0
        t_i = 0
        # initial adaptation state is unadapted, i.e. large t_s
        t_s = 1000 * tau

        while (i < len(ps)):
            # find index in "t" time, without searching whole array each time
            t_i = np.searchsorted(t[t_i:], ps[i], 'right') - 1 + t_i

            # evolve adaptation state
            t_s += isi[i]

            if rn[i] < (a[t_i] * np.exp(-bq[t_i] * np.exp(old_div(-t_s, tau))) / rmax):
                # keep spike
                keep[i] = True
                # remap t_s state
                t_s = -tau * np.log(np.exp((-t_s / tau)) + 1)
            i += 1

        spike_train = ps[keep]

        return spike_train

    # use slow python implementation for the time being
    # TODO: provide optimized C/weave implementation if possible

    inh_adaptingmarkov_generator = _inh_adaptingmarkov_generator_python

    def _inh_2Dadaptingmarkov_generator_python(self, a, bq, tau_s, tau_r, qrqs, t, t_stop):

        """
        Returns a SpikeList whose spikes are an inhomogeneous
        realization (dynamic rate) of the so-called 2D adapting markov
        process (see references).  2D implies the process has two
        states, an adaptation state, and a refractory state, both of
        which affect its probability to spike.  The implementation
        uses the thinning method, as presented in the references.

        For the 1d implementation, with no relative refractoriness,
        see the inh_adaptingmarkov_generator.

        Inputs:
            a,bq    - arrays of the parameters of the hazard function where a[i] and bq[i] 
            will be active on interval [t[i],t[i+1]]
            tau_s    - the time constant of adaptation (in milliseconds).
            tau_r    - the time constant of refractoriness (in milliseconds).
            qrqs     - the ratio of refractoriness conductance to adaptation conductance.
            typically on the order of 200.
            t      - an array specifying the time bins (in milliseconds) at which to 
            specify the rate
            t_stop - length of time to simulate process (in ms)

        Note: 
            - t_start=t[0]

            - a is in units of Hz.  Typical values are available 
              in Fig. 1 of Muller et al 2007, a~5-80Hz (low to high stimulus)

            - bq here is taken to be the quantity b*q_s in Muller et al 2007, is thus
              dimensionless, and has typical values bq~3.0-1.0 (low to high stimulus)

            - qrqs is the quantity q_r/q_s in Muller et al 2007, 
              where a value of qrqs = 3124.0nS/14.48nS = 221.96 was used.

            - tau_s has typical values on the order of 100 ms
            - tau_r has typical values on the order of 2 ms


        References:

        Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier 
        Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
        Neural Comput. 2007 19: 2958-3010.
        
        Devroye, L. (1986). Non-uniform random variate generation. New York: Springer-Verlag.

        Examples:
            See source:trunk/examples/stgen/inh_2Dmarkov_psth.py
        
        See also:
            inh_poisson_generator, inh_adaptingmarkov_generator

        """

        from numpy import shape

        if shape(t) != shape(a) or shape(a) != shape(bq):
            raise ValueError('shape mismatch: t,a,b must be of the same shape')

        # get max rate and generate poisson process to be thinned
        rmax = np.max(a)
        ps = self.poisson_generator(rmax, t_start=t[0], t_stop=t_stop)

        isi = np.zeros_like(ps)
        isi[1:] = ps[1:] - ps[:-1]
        isi[0] = ps[0]  # -0.0 # assume spike at 0.0

        # return empty if no spikes
        if len(ps) == 0:
            return np.array([])

        # gen uniform rand on 0,1 for each spike
        rn = np.array(self.rng.uniform(0, 1, len(ps)))

        # instantaneous a,bq for each spike

        idx = np.searchsorted(t, ps) - 1
        spike_a = a[idx]
        spike_bq = bq[idx]

        keep = np.zeros(shape(ps), bool)

        # thin spikes

        i = 0
        t_last = 0.0
        t_i = 0
        # initial adaptation state is unadapted, i.e. large t_s
        t_s = 1000 * tau_s
        t_r = 1000 * tau_s

        while (i < len(ps)):
            # find index in "t" time, without searching whole array each time
            t_i = np.searchsorted(t[t_i:], ps[i], 'right') - 1 + t_i

            # evolve adaptation state
            t_s += isi[i]
            t_r += isi[i]

            if rn[i] < (a[t_i] * np.exp(-bq[t_i] * (np.exp((-t_s / tau_s)) + qrqs * np.exp((-t_r / tau_r)))) / rmax):
                # keep spike
                keep[i] = True
                # remap t_s state
                t_s = -tau_s * np.log(np.exp((-t_s / tau_s)) + 1)
                t_r = -tau_r * np.log(np.exp((-t_r / tau_r)) + 1)
            i += 1

        spike_train = ps[keep]

        return spike_train

    # use slow python implementation for the time being
    # TODO: provide optimized C/weave implementation if possible

    inh_2Dadaptingmarkov_generator = _inh_2Dadaptingmarkov_generator_python

    def _OU_generator_python(self, dt, tau, sigma, y0, t_start=0.0, t_stop=1000.0, time_it=False):
        """ 
        Generates an Orstein Ulbeck process using the forward euler method. The function returns
        an AnalogSignal object.
        
        Inputs:
            dt      - the time resolution in milliseconds of th signal
            tau     - the correlation time in milliseconds
            sigma   - std dev of the process
            y0      - initial value of the process, at t_start
            t_start - start time in milliseconds
            t_stop  - end time in milliseconds
        
        Examples:
            >> stgen.OU_generator(0.1, 2, 3, 0, 0, 10000)

        See also:
            OU_generator_weave1
        """

        import time

        if time_it:
            t1 = time.time()

        t = np.arange(t_start, t_stop, dt)
        N = len(t)
        y = np.zeros(N, float)
        gauss = self.rng.standard_normal(N - 1)
        y[0] = y0
        fac = (dt / tau)
        noise = np.sqrt(2 * fac) * sigma

        # python loop... bad+slow!
        for i in range(1, N):
            y[i] = y[i - 1] + fac * (y0 - y[i - 1]) + noise * gauss[i - 1]

        if time_it:
            print(time.time() - 1)

        return (y, t)

    # use slow python implementation for the time being
    # TODO: provide optimized C/weave implementation if possible

    def _OU_generator_python2(self, dt, tau, sigma, y0, t_start=0.0, t_stop=1000.0, time_it=False):
        """ 
        Generates an Orstein Ulbeck process using the forward euler method. The function returns
        an AnalogSignal object.
        
        Inputs:
            dt      - the time resolution in milliseconds of th signal
            tau     - the correlation time in milliseconds
            sigma   - std dev of the process
            y0      - initial value of the process, at t_start
            t_start - start time in milliseconds
            t_stop  - end time in milliseconds
        
        Examples:
            >> stgen.OU_generator(0.1, 2, 3, 0, 0, 10000)

        See also:
            OU_generator_weave1
        """

        import time

        if time_it:
            t1 = time.time()

        t = np.arange(t_start, t_stop, dt)
        N = len(t)
        y = np.zeros(N, float)
        y[0] = y0
        fac = (dt / tau)
        gauss = fac * y0 + np.sqrt(2 * fac) * sigma * self.rng.standard_normal(N - 1)
        mfac = 1 - fac

        # python loop... bad+slow!
        for i in range(1, N):
            idx = i - 1
            y[i] = y[idx] * mfac + gauss[idx]

        if time_it:
            print(time.time() - t1)

        return (y, t)

    # use slow python implementation for the time being
    # TODO: provide optimized C/weave implementation if possible

    def OU_generator_weave1(self, dt, tau, sigma, y0, t_start=0.0, t_stop=1000.0, time_it=False):
        """ 
        Generates an Orstein Ulbeck process using the forward euler method. The function returns
        an AnalogSignal object.

        OU_generator_weave1, as opposed to OU_generator, uses scipy.weave
        and is thus much faster.
        
        Inputs:
            dt      - the time resolution in milliseconds of th signal
            tau     - the correlation time in milliseconds
            sigma   - std dev of the process
            y0      - initial value of the process, at t_start
            t_start - start time in milliseconds
            t_stop  - end time in milliseconds

        
        Examples:
            >> stgen.OU_generator_weave1(0.1, 2, 3, 0, 0, 10000)

        See also:
            OU_generator
        """
        import scipy.weave

        import time

        if time_it:
            t1 = time.time()

        t = np.arange(t_start, t_stop, dt)
        N = len(t)
        y = np.zeros(N, float)
        y[0] = y0
        fac = (dt / tau)
        gauss = fac * y0 + np.sqrt(2 * fac) * sigma * self.rng.standard_normal(N - 1)

        # python loop... bad+slow!
        # for i in xrange(1,len(t)):
        #    y[i] = y[i-1]+dt/tau*(y0-y[i-1])+np.sqrt(2*dt/tau)*sigma*np.random.normal()

        # use weave instead

        code = """

        double f = 1.0-fac;

        for(int i=1;i<Ny[0];i++) {
          y(i) = y(i-1)*f + gauss(i-1);
        }
        """

        scipy.weave.inline(code, ['y', 'gauss', 'fac'],
                           type_converters=scipy.weave.converters.blitz)

        if time_it:
            print('Elapsed %.3f seconds.' % (time.time() - t1))

        return (y, t)

    OU_generator = _OU_generator_python2

    # TODO: optimized inhomogeneous OU generator


# TODO: have a array generator with spatio-temporal correlations

# TODO fix shotnoise stuff below  ... and write tests

# Operations on spike trains


def shotnoise_fromspikes(spike_train, q, tau, dt=0.1, t_start=None, t_stop=None, eps=1.0e-8):
    """ 
    Convolves the provided spike train with shot decaying exponentials
    yielding so called shot noise if the spike train is Poisson-like.  
    Returns (shotnoise,t) as numpy arrays. 

   Inputs:
      spike_train - a SpikeTrain object
      q - the shot jump for each spike
      tau - the shot decay time constant in milliseconds
      dt - the resolution of the resulting shotnoise in milliseconds
      t_start - start time of the resulting AnalogSignal
      If unspecified, t_start of spike_train is used
      t_stop  - stop time of the resulting AnalogSignal
      If unspecified, t_stop of spike_train is used
      eps - a numerical parameter indicating at what value of 
      the shot kernal the tail is cut.  The default is usually fine.

   Note:
      Spikes in spike_train before t_start are taken into account in the convolution.

   Examples:
      >> stg = stgen.StGen()
      >> st = stg.poisson_generator(10.0,0.0,1000.0)
      >> g_e = shotnoise_fromspikes(st,2.0,10.0,dt=0.1)


   See also:
      poisson_generator, inh_adaptingmarkov_generator, OU_generator ...
   """

    st = spike_train

    if t_start is not None and t_stop is not None:
        assert t_stop > t_start

    # time of vanishing significance
    vs_t = -tau * np.log((eps / q))

    if t_stop == None:
        t_stop = st.t_stop

    # need to be clever with start time
    # because we want to take spikes into
    # account which occured in spikes_times
    # before t_start
    if t_start == None:
        t_start = st.t_start
        window_start = st.t_start
    else:
        window_start = t_start
        if t_start > st.t_start:
            t_start = st.t_start

    t = np.arange(t_start, t_stop, dt)

    kern = q * np.exp((-np.arange(0.0, vs_t, dt) / tau))

    idx = np.clip(np.searchsorted(t, st.spike_times, 'right') - 1, 0, len(t) - 1)

    a = np.zeros(np.shape(t), float)

    a[idx] = 1.0

    y = np.convolve(a, kern)[0:len(t)]

    signal_t = np.arange(window_start, t_stop, dt)
    signal_y = y[-len(t):]
    return (signal_y, signal_t)


def _gen_g_add(spikes, q, tau, t, eps=1.0e-8):
    """

    spikes is a SpikeTrain object

    """

    # spikes = poisson_generator(rate,t[-1])

    gd_s = np.zeros(t.shape, float)

    dt = t[1] - t[0]

    # time of vanishing significance
    vs_t = -tau * np.log((eps / q))
    kern = q * np.exp((-np.arange(0.0, vs_t, dt) / tau))

    vs_idx = len(kern)

    idx = np.clip(np.searchsorted(t, spikes.spike_times), 0, len(t) - 1)
    idx2 = np.clip(idx + vs_idx, 0, len(gd_s))
    idx3 = idx2 - idx

    for i in range(len(idx)):
        gd_s[idx[i]:idx2[i]] += kern[0:idx3[i]]

    return gd_s
