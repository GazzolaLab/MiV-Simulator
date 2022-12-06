"""Routines to keep track of simulation computation time and terminate the simulation if not enough time has been allocated."""

import time
import datetime

from neuron.hoc import HocObject
from miv_simulator.utils import get_module_logger
from neuron import h

# This logger will inherit its settings from the root logger, created in miv_simulator.env
logger = get_module_logger(__name__)
if hasattr(h, "nrnmpi_init"):
    h.nrnmpi_init()


class SimTimeEvent:
    def __init__(
        self,
        pc: HocObject,
        tstop: float,
        max_walltime_hours: float,
        results_write_time: float,
        setup_time: float,
        dt_status: float = 1.0,
        dt_checksimtime: float = 10.0,
    ) -> None:
        if int(pc.id()) == 0:
            logger.info(
                f"*** allocated wall time is {max_walltime_hours:.2f} hours"
            )
        wt = time.time()
        self.pc = pc
        self.tstop = tstop
        self.walltime_status = wt
        self.walltime_checksimtime = wt
        self.dt_status = dt_status
        self.tcsum = 0.0
        self.tcma = 0.0
        self.nsimsteps = 0
        self.walltime_max = max_walltime_hours * 3600.0 - setup_time
        self.results_write_time = results_write_time
        self.dt_checksimtime = dt_checksimtime
        self.fih_checksimtime = h.FInitializeHandler(1, self.checksimtime)
        self.fih_simstatus = h.FInitializeHandler(1, self.simstatus)
        if int(self.pc.id()) == 0:
            logger.info(
                f"*** max wall time is {self.walltime_max:.2f} s; max setup time was {setup_time:.2f} s"
            )

    def reset(self) -> None:
        wt = time.time()
        self.walltime_max = self.walltime_max - self.tcsum
        self.tcsum = 0.0
        self.tcma = 0.0
        self.nsimsteps = 0
        self.walltime_status = wt
        self.walltime_checksimtime = wt
        self.fih_checksimtime = h.FInitializeHandler(1, self.checksimtime)
        self.fih_simstatus = h.FInitializeHandler(1, self.simstatus)

    def simstatus(self) -> None:
        wt = time.time()
        if h.t > 0.0:
            if int(self.pc.id()) == 0:
                logger.info(
                    f"*** rank 0 computation time at t={h.t:.2f} ms was {(wt - self.walltime_status):.2f} s"
                )
        self.walltime_status = wt
        if (h.t + self.dt_status) < self.tstop:
            h.cvode.event(h.t + self.dt_status, self.simstatus)

    def checksimtime(self) -> None:
        wt = time.time()
        if h.t > 0:
            tt = wt - self.walltime_checksimtime
            ## cumulative moving average wall time time per dt_checksimtime
            self.tcma = self.tcma + ((tt - self.tcma) / (self.nsimsteps + 1))
            self.tcsum = self.tcsum + tt
            ## remaining physical time
            trem = self.tstop - h.t
            ## remaining wall time
            walltime_rem = self.walltime_max - self.tcsum
            walltime_rem_min = self.pc.allreduce(
                walltime_rem, 3
            )  ## minimum value
            ## wall time necessary to complete the simulation
            walltime_needed = (
                trem / self.dt_checksimtime
            ) * self.tcma + self.results_write_time
            walltime_needed_max = self.pc.allreduce(
                walltime_needed, 2
            )  ## maximum value
            if int(self.pc.id()) == 0:
                logger.info(
                    f"*** remaining computation time is {walltime_rem:.2f} s ({str(datetime.timedelta(seconds=walltime_rem))}) and remaining simulation time is {trem:.2f} ms"
                )
                logger.info(
                    f"*** estimated computation time to completion is {walltime_needed_max:.2f} s ({time.ctime(time.time() + walltime_needed_max)})"
                )
                logger.info(
                    f"*** computation time so far is {self.tcsum:.2f} s ({str(datetime.timedelta(seconds=self.tcsum))})"
                )
            ## if not enough time, reduce tstop and perform collective operations to set minimum (earliest) tstop across all ranks
            if walltime_needed_max > walltime_rem_min:
                tstop1 = (
                    int(
                        (walltime_rem - self.results_write_time)
                        / (self.tcma / self.dt_checksimtime)
                    )
                    + h.t
                )
                min_tstop = self.pc.allreduce(tstop1, 3)  ## minimum value
                if int(self.pc.id()) == 0:
                    logger.info(
                        f"*** not enough time to complete {self.tstop:.2f} ms simulation, simulation will likely stop around {min_tstop:.2f} ms"
                    )
                if min_tstop <= h.t:
                    self.tstop = h.t + h.dt
                else:
                    self.tstop = min_tstop
                    h.cvode.event(self.tstop)
                if self.tstop < h.tstop:
                    h.tstop = self.tstop
            self.nsimsteps = self.nsimsteps + 1
        else:
            init_time = wt - self.walltime_checksimtime
            max_init_time = self.pc.allreduce(init_time, 2)  ## maximum value
            self.tcsum += max_init_time
            if int(self.pc.id()) == 0:
                logger.info(
                    f"*** max init time at t={h.t:.2f} ms was {max_init_time:.2f} s"
                )
                logger.info(
                    f"*** computation time so far is {self.tcsum:.2f} and total computation time is {self.walltime_max:.2f} s"
                )
        self.walltime_checksimtime = wt
        if h.t + self.dt_checksimtime < self.tstop:
            h.cvode.event(h.t + self.dt_checksimtime, self.checksimtime)
