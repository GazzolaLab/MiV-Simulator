import itertools
import os
import os.path
import pprint
import random
import sys
import uuid

import click
import numpy as np
from MiV import cells, io_utils, neuron_utils, synapses, utils
from MiV.neuron_utils import configure_hoc_env, h, make_rec, run_iclamp
from MiV.synapses import get_syn_filter_dict
from MiV.utils import (
    AbstractEnv,
    Context,
    config_logging,
    get_module_logger,
    is_interactive,
)
from MiV.cells import BiophysCell
from MiV.env import Env
from mpi4py import MPI  # Must come before importing NEURON
from neuroh5.io import append_cell_attributes
from neuron import h
from scipy import signal
from scipy.optimize import curve_fit
from numpy import ndarray
from typing import Dict, List, Optional, Tuple, Union

# This logger will inherit its settings from the root logger, created in MiV.env
logger = get_module_logger(__name__)

context = Context()


def init_biophys_cell(
    env: AbstractEnv,
    pop_name: str,
    gid: int,
    load_weights: bool = True,
    load_connections: bool = True,
    register_cell: bool = True,
    write_cell: bool = False,
    validate_tree: bool = True,
    cell_dict: Dict[
        str,
        Optional[
            Union[
                Dict[
                    str,
                    Union[
                        ndarray,
                        Dict[str, Union[int, Dict[int, ndarray], ndarray]],
                    ],
                ],
                Dict[str, ndarray],
                Tuple[
                    Dict[
                        str,
                        Dict[
                            str,
                            List[
                                Tuple[
                                    int,
                                    Tuple[ndarray, Dict[str, List[ndarray]]],
                                ]
                            ],
                        ],
                    ],
                    Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
                ],
            ]
        ],
    ] = {},
) -> BiophysCell:
    """
    Instantiates a BiophysCell instance and all its synapses.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    :param load_connections: bool
    :param register_cell: bool
    :param validate_tree: bool
    :param write_cell: bool
    :param cell_dict: dict

    Environment can be instantiated as:
    env = Env(config_file, template_paths, dataset_prefix, config_prefix)
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    :param config_prefix: str; path to directory containing network and cell mechanism config files
    """

    rank = int(env.pc.id())

    ## Determine template name for this cell type
    template_name = env.celltypes[pop_name]["template"]

    ## Determine if a mechanism configuration file exists for this cell type
    if "mech_file_path" in env.celltypes[pop_name]:
        mech_dict = env.celltypes[pop_name]["mech_dict"]
    else:
        mech_dict = None

    ## Determine if correct_for_spines flag has been specified for this cell type
    synapse_config = env.celltypes[pop_name]["synapses"]
    if "correct_for_spines" in synapse_config:
        correct_for_spines_flag = synapse_config["correct_for_spines"]
    else:
        correct_for_spines_flag = False

    ## Load cell gid and its synaptic attributes and connection data
    if template_name.lower() == "pr_nrn":
        cell = cells.make_PR_cell(
            env,
            pop_name,
            gid,
            tree_dict=cell_dict.get("morph", None),
            synapses_dict=cell_dict.get("synapse", None),
            connection_graph=cell_dict.get("connectivity", None),
            weight_dict=cell_dict.get("weight", None),
            mech_dict=mech_dict,
            load_synapses=True,
            load_weights=load_weights,
            load_edges=load_connections,
        )
    elif template_name.lower() == "sc_nrn":
        cell = cells.make_SC_cell(
            env,
            pop_name,
            gid,
            tree_dict=cell_dict.get("morph", None),
            synapses_dict=cell_dict.get("synapse", None),
            connection_graph=cell_dict.get("connectivity", None),
            weight_dict=cell_dict.get("weight", None),
            mech_dict=mech_dict,
            load_synapses=True,
            load_weights=load_weights,
            load_edges=load_connections,
        )
    else:
        cell = cells.make_biophys_cell(
            env,
            pop_name,
            gid,
            tree_dict=cell_dict.get("morph", None),
            synapses_dict=cell_dict.get("synapse", None),
            connection_graph=cell_dict.get("connectivity", None),
            weight_dict=cell_dict.get("weight", None),
            mech_dict=mech_dict,
            load_synapses=True,
            load_weights=load_weights,
            load_edges=load_connections,
            validate_tree=validate_tree,
        )

    cells.init_biophysics(
        cell,
        reset_cable=True,
        correct_cm=correct_for_spines_flag,
        correct_g_pas=correct_for_spines_flag,
        env=env,
    )
    synapses.init_syn_mech_attrs(cell, env)

    if register_cell:
        cells.register_cell(env, pop_name, gid, cell)

    is_reduced = False
    if hasattr(cell, "is_reduced"):
        is_reduced = cell.is_reduced
    if not is_reduced:
        cells.report_topology(env, cell)

    env.cell_selection[pop_name] = [gid]

    if is_interactive:
        context.update(locals())

    if write_cell:
        write_selection_file_path = "%s/%s_%d.h5" % (
            env.results_path,
            env.modelName,
            gid,
        )
        if rank == 0:
            io_utils.mkout(env, write_selection_file_path)
        env.comm.barrier()
        io_utils.write_cell_selection(env, write_selection_file_path)
        if load_connections:
            io_utils.write_connection_selection(env, write_selection_file_path)

    return cell


def measure_deflection(t, v, t0, t1, stim_amp=None):
    """Measure voltage deflection (min or max, between start and end)."""

    start_index = int(np.argwhere(t >= t0 * 0.999)[0])
    end_index = int(np.argwhere(t >= t1 * 0.999)[0])

    deflect_fn = np.argmin
    if stim_amp is not None and (stim_amp > 0):
        deflect_fn = np.argmax

    v_window = v[start_index:end_index]
    peak_index = deflect_fn(v_window) + start_index

    return {
        "t_peak": t[peak_index],
        "v_peak": v[peak_index],
        "peak_index": peak_index,
        "t_baseline": t[start_index],
        "v_baseline": v[start_index],
        "baseline_index": start_index,
        "stim_amp": stim_amp,
    }


##
## Code based on https://www.github.com/AllenInstitute/ipfx/ipfx/subthresh_features.py
##


def fit_membrane_time_constant(t, v, t0, t1, rmse_max_tol=1.0):
    """Fit an exponential to estimate membrane time constant between start and end

    Parameters
    ----------
    v : numpy array of voltages in mV
    t : numpy array of times in ms
    t0 : start of time window for exponential fit
    t1 : end of time window for exponential fit
    rsme_max_tol: minimal acceptable root mean square error (default 1e-4)

    Returns
    -------
    a, inv_tau, y0 : Coefficients of equation y0 + a * exp(-inv_tau * x)

    returns np.nan for values if fit fails
    """

    def exp_curve(x, a, inv_tau, y0):
        return y0 + a * np.exp(-inv_tau * x)

    start_index = int(np.argwhere(t >= t0 * 0.999)[0])
    end_index = int(np.argwhere(t >= t1 * 0.999)[0])

    p0 = (v[start_index] - v[end_index], 0.1, v[end_index])
    t_window = (t[start_index:end_index] - t[start_index]).astype(np.float64)
    v_window = v[start_index:end_index].astype(np.float64)
    try:
        popt, pcov = curve_fit(exp_curve, t_window, v_window, p0=p0)
    except RuntimeError:
        logging.info("Curve fit for membrane time constant failed")
        return np.nan, np.nan, np.nan

    pred = exp_curve(t_window, *popt)

    rmse = np.sqrt(np.mean((pred - v_window) ** 2))

    if rmse > rmse_max_tol:
        logging.debug(
            "RMSE %f for the Curve fit for membrane time constant exceeded the maximum tolerance of %f"
            % (rmse, rmse_max_tol)
        )
        return np.nan, np.nan, np.nan

    return popt


def measure_time_constant(
    t, v, t0, t1, stim_amp, frac=0.1, baseline_interval=100.0, min_snr=20.0
):
    """Calculate the membrane time constant by fitting the voltage response with a
    single exponential.

    Parameters
    ----------
    v : numpy array of voltages in mV
    t : numpy array of times in ms
    t0 : start of stimulus interval in ms
    t1 : end of stimulus interval in ms
    stim_amp : stimulus amplitude
    frac : fraction of peak deflection to find to determine start of fit window. (default 0.1)
    baseline_interval : duration before `start` for baseline Vm calculation
    min_snr : minimum signal-to-noise ratio (SNR) to allow calculation of time constant.
        If SNR is too low, np.nan will be returned. (default 20)

    Returns
    -------
    tau : membrane time constant in ms
    """

    if np.max(t) < t0 or np.max(t) < t1:
        logging.debug(
            "measure_time_constant: time series ends before t0 = {t0} or t1 = {t1}"
        )
        return np.nan

    # Assumes this is being done on a hyperpolarizing step
    deflection_results = measure_deflection(t, v, t0, t1, stim_amp)
    v_peak = deflection_results["v_peak"]
    peak_index = deflection_results["peak_index"]
    v_baseline = deflection_results["v_baseline"]
    start_index = deflection_results["baseline_index"]

    # Check that SNR is high enough to proceed
    signal = np.abs(v_baseline - v_peak)
    noise_interval_start_index = int(
        np.argwhere(t >= (t0 - baseline_interval) * 0.999)[0]
    )
    noise = np.std(v[noise_interval_start_index:start_index])
    t_noise_start = t[noise_interval_start_index]

    if noise == 0:  # noiseless - likely a deterministic model
        snr = np.inf
    else:
        snr = signal / noise
    if snr < min_snr:
        logging.debug(
            "measure_time_constant: signal-to-noise ratio too low for time constant estimate ({:g} < {:g})".format(
                snr, min_snr
            )
        )
        return np.nan

    search_result = np.flatnonzero(
        v[start_index:] <= frac * (v_peak - v_baseline) + v_baseline
    )

    if not search_result.size:
        logger.debug(
            "measure_time_constant: could not find interval for time constant estimate"
        )
        return np.nan

    fit_start_index = search_result[0] + start_index
    fit_end_index = peak_index
    fit_start = t[fit_start_index]
    fit_end = t[fit_end_index]

    a, inv_tau, y0 = fit_membrane_time_constant(t, v, fit_start, fit_end)

    return 1.0 / inv_tau


def measure_passive(
    gid,
    pop_name,
    v_init,
    env: AbstractEnv,
    prelength=1000.0,
    mainlength=3000.0,
    stimdur=1000.0,
    stim_amp=-0.1,
    cell_dict={},
):

    biophys_cell = init_biophys_cell(
        env, pop_name, gid, register_cell=False, cell_dict=cell_dict
    )
    hoc_cell = biophys_cell.hoc_cell

    iclamp_res = run_iclamp(
        hoc_cell,
        prelength=prelength,
        mainlength=mainlength,
        stimdur=stimdur,
        stim_amp=stim_amp,
    )

    t = iclamp_res["t"]
    v = iclamp_res["v"]
    t0 = iclamp_res["t0"]
    t1 = iclamp_res["t1"]

    if np.max(t) < t0 or np.max(t) < t1:
        logging.debug(
            "measure_passive: time series ends before t0 = {t0} or t1 = {t1}"
        )
        return {"Rinp": np.nan, "tau": np.nan}

    deflection_results = measure_deflection(t, v, t0, t1, stim_amp=stim_amp)
    v_peak = deflection_results["v_peak"]
    v_baseline = deflection_results["v_baseline"]

    Rin = (v_peak - v_baseline) / stim_amp
    tau0 = measure_time_constant(t, v, t0, t1, stim_amp)

    results = {
        "Rin": np.asarray([Rin], dtype=np.float32),
        "tau0": np.asarray([tau0], dtype=np.float32),
    }

    logger.info(f"results = {results}")
    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]

    return results


def measure_ap(gid, pop_name, v_init, env: AbstractEnv, cell_dict={}):

    biophys_cell = init_biophys_cell(
        env, pop_name, gid, register_cell=False, cell_dict=cell_dict
    )
    hoc_cell = biophys_cell.hoc_cell

    h.dt = env.dt

    prelength = 100.0
    stimdur = 10.0

    soma = list(hoc_cell.soma)[0]
    initial_amp = 0.05

    h.tlog = h.Vector()
    h.tlog.record(h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record(soma(0.5)._ref_v)

    thr = cells.find_spike_threshold_minimum(
        hoc_cell, loc=0.5, sec=soma, duration=stimdur, initial_amp=initial_amp
    )

    results = {
        "spike threshold current": np.asarray([thr], dtype=np.float32),
        "spike threshold trace t": np.asarray(
            h.tlog.to_python(), dtype=np.float32
        ),
        "spike threshold trace v": np.asarray(
            h.Vlog.to_python(), dtype=np.float32
        ),
    }

    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]

    return results


def measure_ap_rate(
    gid,
    pop_name,
    v_init,
    env: AbstractEnv,
    prelength=1000.0,
    mainlength=3000.0,
    stimdur=1000.0,
    stim_amp=0.2,
    minspikes=50,
    maxit=5,
    cell_dict={},
):

    biophys_cell = init_biophys_cell(
        env, pop_name, gid, register_cell=False, cell_dict=cell_dict
    )

    hoc_cell = biophys_cell.hoc_cell

    tstop = prelength + mainlength

    soma = list(hoc_cell.soma)[0]
    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur = stimdur
    stim1.amp = stim_amp

    h("objref nil, tlog, Vlog, spikelog")

    h.tlog = h.Vector()
    h.tlog.record(h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record(soma(0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = biophys_cell.spike_detector
    nc.record(h.spikelog)
    logger.info(f"ap_rate_test: spike threshold is {nc.threshold}")

    h.tstop = tstop

    it = 1
    ## Increase the injected current until at least maxspikes spikes occur
    ## or up to maxit steps
    while h.spikelog.size() < minspikes:

        logger.info(f"ap_rate_test: iteration {it}")

        h.dt = env.dt

        neuron_utils.simulate(v_init, prelength, mainlength)

        if (h.spikelog.size() < minspikes) & (it < maxit):
            logger.info(
                f"ap_rate_test: stim1.amp = {stim1.amp:.2f} spikelog.size = {h.spikelog.size()}"
            )
            stim1.amp = stim1.amp + 0.1
            h.spikelog.clear()
            h.tlog.clear()
            h.Vlog.clear()
            it += 1
        else:
            break

    logger.info(
        f"ap_rate_test: stim1.amp = {stim1.amp:.2f} spikelog.size = {h.spikelog.size()}"
    )

    isivect = h.Vector(h.spikelog.size() - 1, 0.0)
    tspike = h.spikelog.x[0]
    for i in range(1, int(h.spikelog.size())):
        isivect.x[i - 1] = h.spikelog.x[i] - tspike
        tspike = h.spikelog.x[i]

    isimean = isivect.mean()
    isivar = isivect.var()
    isistdev = isivect.stdev()

    isilast = int(isivect.size()) - 1
    if isivect.size() > 10:
        isi10th = 10
    else:
        isi10th = isilast

    ## Compute the last spike that is largest than the first one.
    ## This is necessary because some models generate spike doublets,
    ## (i.e. spike with very short distance between them, which confuse the ISI statistics.
    isilastgt = int(isivect.size()) - 1
    while isivect.x[isilastgt] < isivect.x[1]:
        isilastgt = isilastgt - 1

    if not (isilastgt > 0):
        isivect.printf()
        raise RuntimeError("Unable to find ISI greater than first ISI")

    results = {
        "spike_count": np.asarray([h.spikelog.size()], dtype=np.uint32),
        "FR_mean": np.asarray([1.0 / isimean], dtype=np.float32),
        "ISI_mean": np.asarray([isimean], dtype=np.float32),
        "ISI_var": np.asarray([isivar], dtype=np.float32),
        "ISI_stdev": np.asarray([isistdev], dtype=np.float32),
        "ISI_adaptation_1": np.asarray(
            [isivect.x[0] / isimean], dtype=np.float32
        ),
        "ISI_adaptation_2": np.asarray(
            [isivect.x[0] / isivect.x[isilast]], dtype=np.float32
        ),
        "ISI_adaptation_3": np.asarray(
            [isivect.x[0] / isivect.x[isi10th]], dtype=np.float32
        ),
        "ISI_adaptation_4": np.asarray(
            [isivect.x[0] / isivect.x[isilastgt]], dtype=np.float32
        ),
    }

    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]

    return results


def measure_fi(gid, pop_name, v_init, env: AbstractEnv, cell_dict={}):

    biophys_cell = init_biophys_cell(
        env, pop_name, gid, register_cell=False, cell_dict=cell_dict
    )
    hoc_cell = biophys_cell.hoc_cell

    soma = list(hoc_cell.soma)[0]
    h.dt = 0.025

    prelength = 1000.0
    mainlength = 2000.0

    tstop = prelength + mainlength

    stimdur = 1000.0

    stim1 = h.IClamp(soma(0.5))
    stim1.delay = prelength
    stim1.dur = stimdur
    stim1.amp = 0.2

    h("objref tlog, Vlog, spikelog")

    h.tlog = h.Vector()
    h.tlog.record(h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record(soma(0.5)._ref_v)

    h.spikelog = h.Vector()
    nc = biophys_cell.spike_detector
    nc.record(h.spikelog)

    h.tstop = tstop

    frs = []
    stim_amps = [stim1.amp]
    for it in range(1, 9):

        neuron_utils.simulate(v_init, prelength, mainlength)

        logger.info(
            "fi_test: stim1.amp = %g spikelog.size = %d\n"
            % (stim1.amp, h.spikelog.size())
        )
        stim1.amp = stim1.amp + 0.1
        stim_amps.append(stim1.amp)
        frs.append(h.spikelog.size())
        h.spikelog.clear()
        h.tlog.clear()
        h.Vlog.clear()

    results = {
        "FI_curve_amplitude": np.asarray(stim_amps, dtype=np.float32),
        "FI_curve_frequency": np.asarray(frs, dtype=np.float32),
    }

    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]

    return results


def measure_gap_junction_coupling(gid, population, v_init, env: AbstractEnv):

    h("objref gjlist, cells, Vlog1, Vlog2")

    pc = env.pc
    h.cells = h.List()
    h.gjlist = h.List()

    biophys_cell1 = init_biophys_cell(
        env, population, gid, register_cell=False, cell_dict=cell_dict
    )
    hoc_cell1 = biophys_cell1.hoc_cell

    cell1 = cells.make_neurotree_hoc_cell(template_class, neurotree_dict=tree)
    cell2 = cells.make_neurotree_hoc_cell(template_class, neurotree_dict=tree)

    h.cells.append(cell1)
    h.cells.append(cell2)

    ggid = 20000000
    source = 10422930
    destination = 10422670
    weight = 5.4e-4
    srcsec = int(cell1.somaidx.x[0])
    dstsec = int(cell2.somaidx.x[0])

    stimdur = 500
    tstop = 2000

    pc.set_gid2node(source, int(pc.id()))
    nc = cell1.connect2target(h.nil)
    pc.cell(source, nc, 1)
    soma1 = list(cell1.soma)[0]

    pc.set_gid2node(destination, int(pc.id()))
    nc = cell2.connect2target(h.nil)
    pc.cell(destination, nc, 1)
    soma2 = list(cell2.soma)[0]

    stim1 = h.IClamp(soma1(0.5))
    stim1.delay = 250
    stim1.dur = stimdur
    stim1.amp = -0.1

    stim2 = h.IClamp(soma2(0.5))
    stim2.delay = 500 + stimdur
    stim2.dur = stimdur
    stim2.amp = -0.1

    log_size = old_div(tstop, h.dt) + 1

    h.tlog = h.Vector(log_size, 0)
    h.tlog.record(h._ref_t)

    h.Vlog1 = h.Vector(log_size)
    h.Vlog1.record(soma1(0.5)._ref_v)

    h.Vlog2 = h.Vector(log_size)
    h.Vlog2.record(soma2(0.5)._ref_v)

    gjpos = 0.5
    neuron_utils.mkgap(
        env, cell1, source, gjpos, srcsec, ggid, ggid + 1, weight
    )
    neuron_utils.mkgap(
        env, cell2, destination, gjpos, dstsec, ggid + 1, ggid, weight
    )

    pc.setup_transfer()
    pc.set_maxstep(10.0)

    h.stdinit()
    h.finitialize(v_init)
    pc.barrier()

    h.tstop = tstop
    pc.psolve(h.tstop)


def measure_psc(
    gid,
    pop_name,
    presyn_name,
    env: AbstractEnv,
    v_init,
    v_holding,
    load_weights=False,
    cell_dict={},
):

    biophys_cell = init_biophys_cell(
        env,
        pop_name,
        gid,
        register_cell=False,
        load_weights=load_weights,
        cell_dict=cell_dict,
    )
    hoc_cell = biophys_cell.hoc_cell

    h.dt = env.dt

    stimdur = 1000.0
    tstop = stimdur
    tstart = 0.0

    soma = list(hoc_cell.soma)[0]
    se = h.SEClamp(soma(0.5))
    se.rs = 10
    se.dur = stimdur
    se.amp1 = v_holding

    h("objref nil, tlog, ilog, Vlog")

    h.tlog = h.Vector()
    h.tlog.record(h._ref_t)

    h.Vlog = h.Vector()
    h.Vlog.record(soma(0.5)._ref_v)

    h.ilog = h.Vector()
    ilog.record(se._ref_i)

    h.tstop = tstop

    neuron_utils.simulate(v_init, 0.0, stimdur)

    vec_i = h.ilog.to_python()
    vec_v = h.Vlog.to_python()
    vec_t = h.tlog.to_python()

    idx = np.where(vec_t > tstart)[0]
    vec_i = vec_i[idx]
    vec_v = vec_v[idx]
    vec_t = vec_t[idx]

    t_holding = vec_t[0]
    i_holding = vec_i[0]

    i_peak = np.max(np.abs(vec_i[1:]))
    peak_index = np.where(np.abs(vec_i) == i_peak)[0][0]
    t_peak = vec_t[peak_index]

    logger.info(
        "measure_psc: t_peak = %f i_holding = %f i_peak = %f"
        % (t_peak, i_holding, i_peak)
    )

    amp_i = abs(i_peak - i_holding) * 1000

    logger.info("measure_psc: amp_i = %f" % amp_i)

    return amp_i


def measure_psp(
    gid,
    pop_name,
    presyn_name,
    syn_mech_name,
    swc_type,
    env: AbstractEnv,
    v_init,
    erev,
    syn_layer=None,
    weight=1,
    syn_count=1,
    load_weights=False,
    cell_dict={},
):

    biophys_cell = init_biophys_cell(
        env,
        pop_name,
        gid,
        register_cell=False,
        load_weights=load_weights,
        cell_dict=cell_dict,
    )
    synapses.config_biophys_cell_syns(
        env,
        gid,
        pop_name,
        insert=True,
        insert_netcons=True,
        insert_vecstims=True,
    )

    hoc_cell = biophys_cell.hoc_cell

    h.dt = env.dt

    prelength = 200.0
    mainlength = 50.0

    rules = {"sources": [presyn_name]}
    if swc_type is not None:
        rules["swc_types"] = [swc_type]
    if syn_layer is not None:
        rules["layers"] = [syn_layer]
    syn_attrs = env.synapse_attributes
    syn_filters = get_syn_filter_dict(env, rules=rules, convert=True)
    syns = syn_attrs.filter_synapses(biophys_cell.gid, **syn_filters)

    print(
        "total number of %s %s synapses: %d"
        % (presyn_name, swc_type if swc_type is not None else "", len(syns))
    )
    stimvec = h.Vector()
    stimvec.append(prelength + 1.0)
    count = 0
    target_syn_pps = None
    for target_syn_id, target_syn in iter(syns.items()):

        target_syn_pps = syn_attrs.get_pps(gid, target_syn_id, syn_mech_name)
        target_syn_nc = syn_attrs.get_netcon(gid, target_syn_id, syn_mech_name)
        target_syn_nc.weight[0] = weight
        setattr(target_syn_pps, "e", erev)
        vs = target_syn_nc.pre()
        vs.play(stimvec)
        if syn_count <= count:
            break
        count += 1

    if target_syn_pps is None:
        raise RuntimeError(
            "measure_psp: Unable to find %s %s synaptic point process"
            % (presyn_name, swc_type)
        )

    sec = target_syn_pps.get_segment().sec

    v_rec = make_rec(
        "psp",
        pop_name,
        gid,
        biophys_cell.hoc_cell,
        sec=sec,
        dt=env.dt,
        loc=0.5,
        param="v",
    )

    h.tstop = mainlength + prelength
    h("objref nil, tlog, ilog")

    h.tlog = h.Vector()
    h.tlog.record(h._ref_t)

    h.ilog = h.Vector()
    h.ilog.record(target_syn_pps._ref_i)

    neuron_utils.simulate(v_init, prelength, mainlength)

    vec_v = np.asarray(v_rec["vec"].to_python())
    vec_i = np.asarray(h.ilog.to_python())
    vec_t = np.asarray(h.tlog.to_python())
    idx = np.where(vec_t >= prelength)[0]
    vec_v = vec_v[idx][1:]
    vec_t = vec_t[idx][1:]
    vec_i = vec_i[idx][1:]

    i_peak_index = np.argmax(np.abs(vec_i))
    i_peak = vec_i[i_peak_index]
    v_peak = vec_v[i_peak_index]

    amp_v = abs(v_peak - vec_v[0])
    amp_i = abs(i_peak - vec_i[0])

    print(
        "measure_psp: v0 = %f v_peak = %f (at t %f)"
        % (vec_v[0], v_peak, vec_t[i_peak_index])
    )
    print("measure_psp: i_peak = %f (at t %f)" % (i_peak, vec_t[i_peak_index]))
    print("measure_psp: amp_v = %f amp_i = %f" % (amp_v, amp_i))

    results = {
        "%s %s PSP"
        % (presyn_name, syn_mech_name): np.asarray([amp_v], dtype=np.float32),
        "%s %s PSP i"
        % (presyn_name, syn_mech_name): np.asarray(vec_i, dtype=np.float32),
        "%s %s PSP v"
        % (presyn_name, syn_mech_name): np.asarray(vec_v, dtype=np.float32),
        "%s %s PSP t"
        % (presyn_name, syn_mech_name): np.asarray(vec_t, dtype=np.float32),
    }

    env.synapse_attributes.del_syn_id_attr_dict(gid)
    if gid in env.biophys_cells[pop_name]:
        del env.biophys_cells[pop_name][gid]

    return results


@click.command()
@click.option(
    "--config-file",
    "-c",
    required=True,
    type=str,
    help="model configuration file name",
)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
    help="path to directory containing network and cell mechanism config files",
)
@click.option("--erev", type=float, help="synaptic reversal potential")
@click.option(
    "--population",
    "-p",
    required=True,
    type=str,
    default="GC",
    help="target population",
)
@click.option("--presyn-name", type=str, help="presynaptic population")
@click.option(
    "--gid", "-g", required=True, type=int, default=0, help="target cell gid"
)
@click.option("--load-weights", "-w", is_flag=True)
@click.option(
    "--measurements",
    "-m",
    type=str,
    default="passive,fi,ap,ap_rate",
    help="measurements to perform",
)
@click.option(
    "--template-paths",
    type=str,
    required=True,
    help="colon-separated list of paths to directories containing hoc cell templates",
)
@click.option(
    "--dataset-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory containing required neuroh5 data files",
)
@click.option(
    "--results-path",
    required=False,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory where output files will be written",
)
@click.option(
    "--results-file-id",
    type=str,
    required=False,
    default=None,
    help="identifier that is used to name neuroh5 files that contain output spike and intracellular trace data",
)
@click.option(
    "--results-namespace-id",
    type=str,
    required=False,
    default=None,
    help="identifier that is used to name neuroh5 namespaces that contain output spike and intracellular trace data",
)
@click.option("--syn-mech-name", type=str, help="synaptic mechanism name")
@click.option("--syn-weight", type=float, help="synaptic weight")
@click.option("--syn-count", type=int, default=1, help="synaptic count")
@click.option("--swc-type", type=str, help="synaptic swc type")
@click.option("--syn-layer", type=str, help="synaptic layer name")
@click.option(
    "--stim-amp",
    type=float,
    default=0.1,
    help="current stimulus amplitude (nA)",
)
@click.option(
    "--v-init",
    type=float,
    default=-75.0,
    help="initialization membrane potential (mV)",
)
@click.option(
    "--dt", type=float, default=0.025, help="simulation timestep (ms)"
)
@click.option("--use-cvode", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def main(
    config_file,
    config_prefix,
    erev,
    population,
    presyn_name,
    gid,
    load_weights,
    measurements,
    template_paths,
    dataset_prefix,
    results_path,
    results_file_id,
    results_namespace_id,
    syn_mech_name,
    syn_weight,
    syn_count,
    syn_layer,
    swc_type,
    stim_amp,
    v_init,
    dt,
    use_cvode,
    verbose,
):

    config_logging(verbose)

    if results_file_id is None:
        results_file_id = uuid.uuid4()
    if results_namespace_id is None:
        results_namespace_id = "Cell Clamp Results"
    comm = MPI.COMM_WORLD
    np.seterr(all="raise")
    params = dict(locals())
    env = Env(**params)
    configure_hoc_env(env)
    io_utils.mkout(env, env.results_file_path)
    env.cell_selection = {}

    if measurements is not None:
        measurements = [x.strip() for x in measurements.split(",")]

    attr_dict = {}
    attr_dict[gid] = {}
    if "passive" in measurements:
        attr_dict[gid].update(measure_passive(gid, population, v_init, env))
    if "ap" in measurements:
        attr_dict[gid].update(measure_ap(gid, population, v_init, env))
    if "ap_rate" in measurements:
        logger.info("ap_rate")
        attr_dict[gid].update(
            measure_ap_rate(gid, population, v_init, env, stim_amp=stim_amp)
        )
    if "fi" in measurements:
        attr_dict[gid].update(measure_fi(gid, population, v_init, env))
    if "gap" in measurements:
        measure_gap_junction_coupling(gid, population, v_init, env)
    if "psp" in measurements:
        assert presyn_name is not None
        assert syn_mech_name is not None
        assert erev is not None
        assert syn_weight is not None
        attr_dict[gid].update(
            measure_psp(
                gid,
                population,
                presyn_name,
                syn_mech_name,
                swc_type,
                env,
                v_init,
                erev,
                syn_layer=syn_layer,
                syn_count=syn_count,
                weight=syn_weight,
                load_weights=load_weights,
            )
        )

    if results_path is not None:
        append_cell_attributes(
            env.results_file_path,
            population,
            attr_dict,
            namespace=env.results_namespace_id,
            comm=env.comm,
            io_size=env.io_size,
        )


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                utils.list_find(
                    lambda x: os.path.basename(x) == os.path.basename(__file__),
                    sys.argv,
                )
                + 1
            ) :
        ]
    )
