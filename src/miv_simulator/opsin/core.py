"""Simulation of opsin receptors at the channel, cell, and network level. 
Based on code from the PyRhO: A Multiscale Optogenetics Simulation Platform
https://github.com/ProjectPyRhO/PyRhO.git
"""

from typing import TYPE_CHECKING, Dict, Set, Tuple
from collections import OrderedDict, defaultdict
import abc
import copy
import logging
import math
import numpy as np
from neuron import h
from miv_simulator.utils import (
    Struct,
    get_module_logger,
)

if TYPE_CHECKING:
    from neuron.hoc import HocObject

logger = get_module_logger(__name__)

def cycles2times(cycles, Dt_delay):
    r"""
    Convert pulse cycles (durations) to times (absolute events).

    Parameters
    ----------
    cycles : list or array
        Array of cycles and delay duration i.e.:
        :math:`[[\Delta t_{on,0}, \Delta t_{off,0}],
                ...,
                [\Delta t_{on,N-1}, \Delta t_{off,N-1}]]`.
    Dt_delay : float
        Delay duration, :math:`\Delta t_{delay}`

    Returns
    -------
    tuple
        List of pulse times and total protocol time aligned at t0 = 0 i.e.:
        :math:`([[t_{on,0}, t_{off,0}], ..., [t_{on,N-1}, t_{off,N-1}]], \Delta t_{total})`.
    """

    # TODO: Generalise to Dt_delays c.f. recovery
    cycles = np.array(cycles)
    nPulses = cycles.shape[0]
    assert(cycles.shape[1] <= 2)
    times = np.cumsum(np.r_[Dt_delay, cycles.ravel()])
    Dt_total = times[-1]
    times = times[:-1].reshape((nPulses, 2))  # Trim the final Dt_off & reshape
    #times = np.zeros((nPulses, 2)) #[Dt_delay,Dt_delay+cycles[row,0] for row in pulses]
    #Dt_total = Dt_delay
    #for p in range(nPulses):
    #    times[p, 0] = Dt_total
    #    times[p, 1] = Dt_total+cycles[p, 0]
    #    Dt_total += sum(cycles[p, :])
    return (times, Dt_total)


modelParams = OrderedDict([(3, Struct()), (4, Struct()), (6, Struct())])    

default_opsin_type = 'ChR2'
rho_type = default_opsin_type


### Move somewhere else e.g. base.py
class PyRhOobject(object):
    """Common base class for all PyRhO objects."""

    __metaclass__ = abc.ABCMeta
    # https://docs.python.org/3/reference/datamodel.html#special-method-names

    #def __new__(self):
    #    pass
    @abc.abstractmethod
    def __init__(self):
        pass

    def __del__(self):
        pass

    def __repr__(self):
        return str(self.__class__)

    def __str__(self):
        print("PyRhO object: ", self.__class__.__name__)

    def __call__(self):
        return

    def setParams(self, params):
        """Set all model parameters from a Struct() object."""
        #for param, value in params.items():
        #for p in params.keys():
        #    self.__dict__[p] = params[p].value #vars(self())[p]
        #for p in params.keys():
        #    setattr(self, p, params[p].value)
        for name, value in params.valuesdict().items():
            setattr(self, name, value)
        #for name, value in params.items():
        #    setattr(self, name, value)

    def updateParams(self, params):
        """Update model parameters which already exist."""
        pDict = params.valuesdict()
        count = 0
        for name, value in pDict.items():
            if hasattr(self, name):
                setattr(self, name, value)
                count += 1
        #for p, v in pDict.items():
        #    if p in self.__dict__: # Added to allow dummy variables in fitting parameters
        #        self.__dict__[p] = v #vars(self())[p]
        #        count += 1
        return count

    def getParams(self, params):
        """Export parameters to lmfit dictionary."""
        for p in self.__dict__.keys():
            params[p].value = self.__dict__[p]

    def exportParams(self, params):
        """Export parameters which are already in lmfit dictionary."""
        count = 0
        for p, v in self.__dict__.items():
            if p in params:
                params[p].value = v  # self.__dict__[p]
                count += 1
        return count

    def printParams(self):
        for p in self.__dict__.keys():
            print(p, ' = ', self.__dict__[p])

    def logParams(self):
        """Log parameters."""
        logger.info('Parameters for ' + self.__class__.__name__)
        for p in self.__dict__.keys():
            logger.info(' '.join([p, ' = ', str(self.__dict__[p])]))

    def printParamsWithLabels(self):
        for p in self.__dict__.keys():
            if p in unitLabels:
                print(p, ' = ', self.__dict__[p], ' [', unitLabels[p], ']')
            else:
                print(p, ' = ', self.__dict__[p])

    def printParamsWithUnits(self):
        for p in self.__dict__.keys():
            if p in modelUnits:
                print(p, ' = ', self.__dict__[p], ' * ', modelUnits[p])
            else:
                print(p, ' = ', self.__dict__[p])

    def getExt(self, var, ext='max'):
        if ext == 'max':
            mVal = max(self.__dict__[var])
        elif ext == 'min':
            mVal = min(self.__dict__[var])
        mInd = np.searchsorted(self.__dict__[var], mVal)
        return mVal, mInd
