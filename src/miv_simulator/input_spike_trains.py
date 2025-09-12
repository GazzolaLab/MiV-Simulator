import gc
import math
import os
import sys
import time
from typing import Dict, Union, List, Optional

import h5py
import numpy as np
from miv_simulator.utils.io import make_h5types
from miv_simulator.utils import (
    config_logging,
    get_script_logger,
)
from mpi4py import MPI
from neuroh5.io import (
    append_cell_attributes,
    bcast_cell_attributes,
    read_population_ranges,
)
from miv_simulator.input_features import EncoderTimeConfig, InputFeaturePopulation

logger = get_script_logger(os.path.basename(__file__))

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


def generate_input_spike_trains(
    env,
    population: InputFeaturePopulation,
    output_path: str,
    output_spikes_namespace: str,
    output_spike_train_attr_name: str,
    signal_id: str,
    signal: Optional[np.ndarray] = None,
    signal_path: Optional[str] = None,
    signal_namespace: Optional[str] = None,
    coords_path: Optional[str] = None,
    distances_namespace: Optional[str] = None,
    io_size: int = 1,
    chunk_size: int = 1000,
    value_chunk_size: int = 1000,
    cache_size: int = 1,
    write_size: int = 1,
    phase_mod: bool = False,
    debug: bool = False,
    debug_count: int = 10,
    verbose: bool = False,
    dry_run: bool = False,
):
    """
    :param env: env.Env
    :population: InputFeaturePopulation,
    :param signal_path: str (path to file)
    :param signal_namespace: str
    :param signal_id: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param write_size: int
    :param output_path: str (path to file)
    :param spikes_namespace: str
    :param spike_train_attr_name: str
    :param debug: bool
    :param verbose: bool
    :param dry_run: bool
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    config_logging(verbose)

    if phase_mod and (coords_path is None):
        raise RuntimeError(
            "generate_input_spike_trains: when phase_mod is True, coords_path is required"
        )

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info(f"{comm.size} ranks have been allocated")

    population_name = population.name

    soma_positions_dict = None
    if coords_path is not None:
        population_ranges = read_population_ranges(coords_path, comm)[0]
        populations = sorted(population_ranges.keys())
        soma_positions_dict = {}
        for population in populations:
            reference_u_arc_distance_bounds = None
            if rank == 0:
                with h5py.File(coords_path, "r") as coords_f:
                    reference_u_arc_distance_bounds = (
                        coords_f["Populations"][population][distances_namespace].attrs[
                            "Reference U Min"
                        ],
                        coords_f["Populations"][population][distances_namespace].attrs[
                            "Reference U Max"
                        ],
                    )
            comm.barrier()
            reference_u_arc_distance_bounds = comm.bcast(
                reference_u_arc_distance_bounds, root=0
            )
            distances = bcast_cell_attributes(
                coords_path, population, namespace=distances_namespace, root=0
            )
            abs_positions = {
                k: v["U Distance"][0] - reference_u_arc_distance_bounds[0]
                for (k, v) in distances
            }
            soma_positions_dict[population] = abs_positions
            del distances

    # TODO: equilibration support
    # equilibrate = stimulus.get_equilibration(env)

    if signal is None:
        if signal_path is None:
            raise RuntimeError(
                "generate_input_spike_trains: neither signal array nor signal file path is provided"
            )
        if rank == 0:
            with h5py.File(signal_path, "r") as signal_f:
                signal_ns = signal_f[signal_namespace]
                if signal_id not in signal_ns:
                    raise RuntimeError(
                        f"generate_input_spike_trains: no signal {signal_id} in namespace {signal_namespace} found "
                        f"for specified signal_path: {signal_path}"
                    )
                signal = signal_ns[signal_id]
        comm.barrier()
        signal = comm.bcast(signal, root=0)

    output_namespace = f"{output_spikes_namespace} {signal_id}"

    write_every = max(1, int(math.floor(write_size / comm.size)))
    req = comm.Ibarrier()
    gc.collect()
    req.wait()

    # TODO: phase modulation
    # phase_mod_config_dict = None
    # if phase_mod:
    #    phase_mod_config_dict = stimulus.oscillation_phase_mod_config(
    #        env, population, soma_positions_dict[population]
    #    )

    # Obtain feature modality
    feature_modality = population.modality

    # Process the signal using the modality
    processed_signal = feature_modality.preprocess_signal(signal)

    # Prepare encoder time configuration
    dt_ms = 1.0  # TODO: Encoder timestep [ms]
    sample_duration_ms = dt_ms  # TODO: Duration of one sample [ms]

    # Initialize the encoders with appropriate time config
    time_config = EncoderTimeConfig(duration_ms=sample_duration_ms, dt_ms=dt_ms)

    # Get responses and generate spike times
    process_time = time.time()
    if rank == 0:
        logger.info(
            f"Generating input source spike trains for population {population}..."
        )

    start_time = time.time()
    spikes_attr_dict = dict()
    gid_count = 0

    feature_items = list(population.features.items())
    n_iter = comm.allreduce(len(feature_items), op=MPI.MAX)

    if not dry_run and rank == 0:
        if output_path is None:
            raise RuntimeError("generate_input_spike_trains: missing output_path")
        if not os.path.isfile(output_path):
            make_h5types(env, output_path)
    comm.barrier()

    local_random = np.random.RandomState()

    for iter_count in range(n_iter):
        if iter_count < len(feature_items):
            gid, input_feature = feature_items[iter_count]
        else:
            gid, input_feature = None, None
        if gid is not None:
            if rank == 0:
                logger.info(f"Rank {rank}: generating spike trains for gid {gid}...")

            # TODO: phase modulation configuration
            # phase_mod_config = None
            # if phase_mod_config_dict is not None:
            #    phase_mod_config = phase_mod_config_dict[gid]

            # TODO: get the filtered signal from the input filter
            #       when debug mode is enabled
            # activation = feature.input_filter(processed_signal)

            # Initialize feature encoder
            input_feature.initialize_encoder(time_config, local_random)

            # Get spike response
            response = input_feature.get_response(processed_signal)
            if isinstance(response, list):
                response_length = 0
                for x in response:
                    response_length += len(x)
                if response_length > 0:
                    try:
                        response = np.concatenate(np.concatenate(response, dtype=np.float32))
                    except Exception as e:
                        logger.error(f"error concatenating response: {response}")
                        raise e
                else:
                    response = np.asarray([], dtype=np.float32)
            else:
                response = response.reshape((-1,)).astype(np.float32)

            if len(response) > 0:
                spikes_attr_dict[gid] = {output_spike_train_attr_name: response}

            gid_count += 1
        if (iter_count > 0 and iter_count % write_every == 0) or (
            debug and iter_count == debug_count
        ):
            req = comm.Ibarrier()
            total_gid_count = comm.reduce(gid_count, root=0, op=MPI.SUM)
            if rank == 0:
                logger.info(
                    f"generated spike trains for {total_gid_count} {population_name} cells"
                )
            req.wait()

            req = comm.Ibarrier()
            if not dry_run:
                append_cell_attributes(
                    output_path,
                    population_name,
                    spikes_attr_dict,
                    namespace=output_namespace,
                    comm=comm,
                    io_size=io_size,
                    chunk_size=chunk_size,
                    value_chunk_size=value_chunk_size,
                )
            req.wait()
            req = comm.Ibarrier()
            del spikes_attr_dict
            spikes_attr_dict = dict()
            gc.collect()
            req.wait()
            if debug and iter_count == debug_count:
                break

    if not dry_run:
        req = comm.Ibarrier()
        append_cell_attributes(
            output_path,
            population_name,
            spikes_attr_dict,
            namespace=output_namespace,
            comm=comm,
            io_size=io_size,
            chunk_size=chunk_size,
            value_chunk_size=value_chunk_size,
        )
        req.wait()
        req = comm.Ibarrier()
        del spikes_attr_dict
        spikes_attr_dict = dict()
        req.wait()
    process_time = time.time() - start_time

    req = comm.Ibarrier()
    total_gid_count = comm.reduce(gid_count, root=0, op=MPI.SUM)
    if rank == 0:
        logger.info(
            f"generated spike trains for {total_gid_count} {population} cells in {process_time:.2f} s"
        )
    req.wait()


def import_input_spike_train(
    data: Dict[int, Union[List, np.ndarray]],
    output_filepath: str,
    namespace: str = "Custom",
    attr_name: str = "Input Spikes",
    population_name: str = "STIM",
) -> None:
    """Takes data representing spike trains and writes it to a input spike HDF5 output file

    :param data: A dictionary where keys represent the global ID of input neurons and value are array-likes with spike times in seconds
    :param output_filepath: Output HDF5 file path
    :param namespace: HDF5 target namespace
    :param attr_name: HDF5 attribute name
    :param population: Neuron population
    """

    def _validate_key(_key):
        try:
            return int(_key)
        except Exception as _e:
            raise ValueError(
                f"Spike train data contains invalid neuron GID. Expected int key but found '{_key}'"
            ) from _e

    output_spike_attr_dict = dict(
        {
            _validate_key(k): {
                attr_name: np.array(v, dtype=np.float32) * 1000  # to miliseconds
            }
            for k, v in data.items()
        }
    )

    append_cell_attributes(
        output_filepath,
        population_name,
        output_spike_attr_dict,
        namespace=namespace,
    )
