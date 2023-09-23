from typing import Optional
from miv_simulator.utils import AbstractEnv
from mpi4py import MPI
from collections import defaultdict
from neuron import h
import logging
from miv_simulator.network import make_cells, connect_gjs, connect_cells
import time
import random
from miv_simulator import config
from miv_simulator.synapses import SynapseAttributes

from neuroh5.io import (
    read_cell_attribute_info,
    read_population_names,
    read_population_ranges,
    read_projection_names,
)

logger = logging.getLogger(__name__)


class Env(AbstractEnv):
    """Manages the global state within a simulation rank"""

    def __init__(
        self,
        comm: Optional[MPI.Intracomm] = None,
        seed: Optional[int] = None,
    ):
        self.seed = random.Random(seed).randint(1, 2**16 - 1)

        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm

        # --- Resources

        self.gidset = set()
        self.node_allocation = None  # node rank map

        # --- Statistics

        self.mkcellstime = -0.0
        self.connectgjstime = -0.0
        self.connectcellstime = -0.0

        # --- Graph

        self.cells = defaultdict(lambda: dict())
        self.artificial_cells = defaultdict(lambda: dict())
        self.biophys_cells = defaultdict(lambda: dict())
        self.spike_onset_delay = {}
        self.recording_sets = {}
        self.synapse_attributes = None
        self.edge_count = defaultdict(dict)
        self.syns_set = defaultdict(set)

        # --- Compat

        self.template_dict = {}

        # --- Simulator

        self.pc = h.pc
        self.rank = int(self.pc.id())

        # Spike time of all cells on this host
        self.t_vec = h.Vector()
        # Ids of spike times on this host
        self.id_vec = h.Vector()
        # Timestamps of intracellular traces on this host
        self.t_rec = h.Vector()

    # --- miv_simulator.network.init equivalent

    def load_cells(
        self,
        filepath: str,
        templates: str,
        cell_types: config.CellTypes,
        io_size: int = 0,
    ):
        if self.rank == 0:
            logger.info("*** Creating cells...")
        st = time.time()

        rank = self.comm.Get_rank()
        if rank == 0:
            color = 1
        else:
            color = 0
        ## comm0 includes only rank 0
        comm0 = self.comm.Split(color, 0)

        cell_attribute_info = None
        population_ranges = None
        population_names = None
        if rank == 0:
            population_names = read_population_names(filepath, comm0)
            (population_ranges, _) = read_population_ranges(filepath, comm0)
            cell_attribute_info = read_cell_attribute_info(
                filepath, population_names, comm=comm0
            )
            logger.info(f"population_names = {str(population_names)}")
            logger.info(f"population_ranges = {str(population_ranges)}")
            logger.info(f"attribute info: {str(cell_attribute_info)}")
        population_ranges = self.comm.bcast(population_ranges, root=0)
        population_names = self.comm.bcast(population_names, root=0)
        cell_attribute_info = self.comm.bcast(cell_attribute_info, root=0)
        comm0.Free()

        class _binding:
            pass

        this = _binding()
        this.__dict__.update(
            {
                # bound
                "pc": self.pc,
                "data_file_path": filepath,
                "io_size": io_size,
                "comm": self.comm,
                "node_allocation": self.node_allocation,
                "cells": self.cells,
                "artificial_cells": self.artificial_cells,
                "biophys_cells": self.biophys_cells,
                "spike_onset_delay": self.spike_onset_delay,
                "recording_sets": self.recording_sets,
                "t_vec": self.t_vec,
                "id_vec": self.id_vec,
                "t_rec": self.t_rec,
                # compat
                "gapjunctions_file_path": None,  # TODO
                "gapjunctions": None,  # TODO
                "recording_profile": None,  # TODO
                "dt": 0.025,  # TODO: understand the implications of this
                "gidset": self.gidset,
                "SWC_Types": config.SWCTypesDef.__members__,
                "template_paths": [templates],
                "dataset_path": None,
                "dataset_prefix": "",
                "template_dict": self.template_dict,
                "cell_attribute_info": cell_attribute_info,
                "celltypes": cell_types,
                "model_config": {
                    "Random Seeds": {
                        "Intracellular Recording Sample": self.seed
                    }
                },
            }
        )

        make_cells(this)

        self.mkcellstime = time.time() - st
        if self.rank == 0:
            logger.info(f"*** Cells created in {self.mkcellstime:.02f} s")
        local_num_cells = sum(len(cells) for cells in self.cells.values())

        logger.info(f"*** Rank {self.rank} created {local_num_cells} cells")

        st = time.time()

        connect_gjs(this)

        self.pc.setup_transfer()
        self.connectgjstime = time.time() - st
        if rank == 0:
            logger.info(
                f"*** Gap junctions created in {self.connectgjstime:.02f} s"
            )

    # -- user-space OptoStim and LFP etc.

    def load_connections(
        self,
        filepath: str,
        cell_filepath: str,
        cell_types: config.CellTypes,
        io_size: int = 0,
    ):
        st = time.time()
        if self.rank == 0:
            logger.info(f"*** Creating connections:")

        rank = self.comm.Get_rank()
        if rank == 0:
            color = 1
        else:
            color = 0
        ## comm0 includes only rank 0
        comm0 = self.comm.Split(color, 0)

        projection_dict = None
        if rank == 0:
            projection_dict = defaultdict(list)
            for src, dst in read_projection_names(filepath, comm=comm0):
                projection_dict[dst].append(src)
            projection_dict = dict(projection_dict)
            logger.info(f"projection_dict = {str(projection_dict)}")
        projection_dict = self.comm.bcast(projection_dict, root=0)
        comm0.Free()

        class _binding:
            pass

        this = _binding()
        this.__dict__.update(
            {
                # bound
                "pc": self.pc,
                "connectivity_file_path": filepath,
                "forest_file_path": cell_filepath,
                "io_size": io_size,
                "comm": self.comm,
                "node_allocation": self.node_allocation,
                "edge_count": self.edge_count,
                "microcircuit_inputs": False,
                "use_cell_attr_gen": False,  # TODO
                "projection_dict": projection_dict,
                "Populations": config.PopulationsDef.__members__,
                "connection_velocity": {  # TODO config
                    "PYR": 250,
                    "STIM": 250,
                    "PVBC": 250,
                    "OLM": 250,
                },
                "celltypes": cell_types,
            }
        )
        self.synapse_attributes = SynapseAttributes(
            this,
            # TODO: expose config through H5?
            {
                "AMPA": "LinExp2Syn",
                "NMDA": "LinExp2SynNMDA",
                "GABA_A": "LinExp2Syn",
                "GABA_B": "LinExp2Syn",
            },
            {
                "Exp2Syn": {
                    "mech_file": "exp2syn.mod",
                    "mech_params": ["tau1", "tau2", "e"],
                    "netcon_params": {"weight": 0},
                    "netcon_state": {},
                },
                "LinExp2Syn": {
                    "mech_file": "lin_exp2syn.mod",
                    "mech_params": ["tau_rise", "tau_decay", "e"],
                    "netcon_params": {"weight": 0, "g_unit": 1},
                    "netcon_state": {},
                },
                "LinExp2SynNMDA": {
                    "mech_file": "lin_exp2synNMDA.mod",
                    "mech_params": [
                        "tau_rise",
                        "tau_decay",
                        "e",
                        "mg",
                        "Kd",
                        "gamma",
                        "vshift",
                    ],
                    "netcon_params": {"weight": 0, "g_unit": 1},
                    "netcon_state": {},
                },
            },
        )
        this.__dict__["synapse_attributes"] = self.synapse_attributes

        connect_cells(this)

        self.pc.set_maxstep(10.0)

        self.connectcellstime = time.time() - st

        if self.rank == 0:
            logger.info(
                f"*** Done creating connections: time = {self.connectcellstime:.02f} s"
            )
        edge_count = int(sum(self.edge_count[dest] for dest in self.edge_count))
        logger.info(f"*** Rank {rank} created {edge_count} connections")
