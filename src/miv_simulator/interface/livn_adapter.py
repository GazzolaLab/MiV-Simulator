import gc
import logging
import math
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union, Self

import numpy as np
from machinable.config import to_dict
from miv_simulator import config
from miv_simulator.network import (
    connect_cell_selection,
    connect_cells,
    connect_gjs,
)
from miv_simulator.optimization import update_network_params
from miv_simulator.synapses import SynapseManager
from miv_simulator.utils import ExprClosure, from_yaml
from miv_simulator.utils import neuron as neuron_utils
from miv_simulator.utils.neuron import configure_hoc, HocObject
from miv_simulator import cells
from mpi4py import MPI
from neuroh5.io import (
    read_cell_attribute_selection,
    read_projection_names,
    read_cell_attribute_info,
    read_tree_selection,
    scatter_read_cell_attributes,
    scatter_read_cell_attribute_selection,
    scatter_read_trees,
)
from neuron import h

from livn.run import Run
from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol
from livn.types import SynapticParam
from livn.utils import DotDict, import_object_by_path

if TYPE_CHECKING:
    from livn.io import IO
    from livn.system import System
    from livn.types import Model


if hasattr(h, "nrnmpi_init"):
    h.nrnmpi_init()


logger = logging.getLogger(__name__)

logger.setLevel(os.getenv("LIVN_NEURON_LOGGING", "WARNING"))
logging.getLogger("miv_simulator").setLevel(
    os.getenv("LIVN_MIV_SIMULATOR_LOGGING", "WARNING")
)


def _cell_types(system) -> dict:
    value = getattr(system, "cell_types", None)
    if value is not None:
        return value
    legacy = getattr(system, "synapses_config", None)
    if legacy is not None and "cell_types" in legacy:
        return legacy["cell_types"]
    raise AttributeError(
        f"{type(system).__name__} exposes no cell types; livn's NeuroH5System "
        "has `cell_types` since 2026-10-08"
    )


class _Compat:
    def __init__(self, env):
        self.__dict__.update(
            {
                "_env": env,
                "cells": env._registry,
                "dt": 0.025,
                "dataset_path": None,
                "dataset_prefix": "",
                "datasetName": "",
                "recording_profile": None,
                "model_config": {
                    "Random Seeds": {
                        "Intracellular Recording Sample": env.seed,
                    }
                },
                "gapjunctions_file_path": None,
                "gapjunctions": None,
            }
        )

    def __getattr__(self, name):
        return getattr(self.__dict__["_env"], name)

    def __setattr__(self, name, value):
        if name in (
            "dt",
            "dataset_path",
            "dataset_prefix",
            "datasetName",
            "recording_profile",
            "model_config",
            "gapjunctions_file_path",
            "gapjunctions",
        ):
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["_env"], name, value)


class _Cell(cells.BiophysCell):
    def position(self, x: float, y: float, z: float) -> None:
        target = self.hoc_cell if self.hoc_cell is not None else self.cell_obj
        if target is None or not hasattr(target, "position"):
            raise RuntimeError("cell has no position()")
        target.position(x, y, z)


class Env(EnvProtocol):
    def __init__(
        self,
        system: Union["System", str],
        model: Union["Model", None] = None,
        io: Union["IO"] = None,
        seed: int | None = 123,
        comm: MPI.Intracomm | None = None,
        subworld_size: int | None = None,
    ):
        from livn.system import resolve

        self.seed = seed

        self.system = resolve(system, comm=comm)
        if model is None:
            model = self.system.default_model()
        self.model = model
        if io is None:
            io = self.system.default_io()
        self.io = io

        self._closed = False

        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.subworld_size = subworld_size

        self.duration = None
        self.encoding = None
        self.decoding = None

        # --- Resources

        self.gidset = set()
        self.node_allocation = None  # node rank map

        # --- Statistics

        self.mkcellstime = -0.0
        self.connectgjstime = -0.0
        self.connectcellstime = -0.0
        self.psolvetime = -0.0

        # --- Graph

        self._registry = defaultdict(lambda: dict())
        self.artificial_cells = defaultdict(lambda: dict())
        self.biophys_cells = defaultdict(lambda: dict())
        self.spike_onset_delay = {}
        self.recording_sets = {}
        self.synapse_manager = None
        self.edge_count = defaultdict(dict)
        self.syns_set = defaultdict(set)

        # --- State
        self.cells_meta_data = None
        self.connections_meta_data = None
        self.input_sources = None

        # --- Compat

        self.template_dict = {}

        self._flucts = {}
        self._noise_state: dict = {}

        # --- Simulator
        self.template_directory = self.model.neuron_template_directory()
        self.mechanisms_directory = self.model.neuron_mechanisms_directory()

        from livn.backend.neuron import mechanisms as livn_mechanisms

        livn_mechanisms.load_mechanisms(self.mechanisms_directory)
        configure_hoc(
            template_directory=self.template_directory,
            mechanisms_directory=None,
        )

        self.pc = h.pc
        # `rank` is livn's read-only protocol property over `comm`

        if self.subworld_size is not None:
            self.pc.subworlds(subworld_size)

        # Spike time of all cells on this host
        self.t_vec = h.Vector()
        # Ids of spike times on this host
        self.id_vec = h.Vector()
        # Timestamps of intracellular traces on this host
        self.t_rec = h.Vector()

        self.v_recs = {}
        self.v_recs_dt = {}

        # Membrane current recordings per gid
        self.i_recs = {}
        self.i_recs_dt = {}
        self.i_area = {}  # membrane surface area (cm^2) for absolute current conversion

        # Synaptic weight recordings (STDP)
        self.w_recs = {}  # (gid, syn_id, mech_name) -> h.Vector
        self.w_recs_dt = {}  # population -> dt
        self._plasticity_enabled = False
        self._weight_rec_dt = 0.1  # default recording dt
        self._weight_nc_refs = {}  # (gid, syn_id, mech_name) -> NetCon
        self._weight_recording_active = False
        self._w_rec_times = None  # h.Vector of timestamps

        self.t = 0
        self._dt: float | None = None
        self._stimulus_vectors: dict[tuple[int, int], dict] = {}
        self._stimulus_callback_registered = False
        self._stimulus_step_index = 0

        # Opsin state
        self._opsin_refs: dict[tuple[int, int], object] = {}
        self._opsin_stimulus_vectors: dict[tuple[int, int], dict] = {}
        self._opsin_callback_registered = False
        self._opsin_step_index = 0

        # Refractory spike-source filters (gid -> dict with 'filter', 'in_nc', 'out_nc')
        self._spike_filter_refs: dict[int, dict] = {}

        if hasattr(self.model, "neuron_refractory_period"):
            self._refractory_period = float(self.model.neuron_refractory_period())
        else:
            self._refractory_period = 0.0

        self._select_spec = None
        self._select_method = "first"
        self._select_bounds = None
        self._selection = None
        self._cell_selection = None  # {pop: sorted gids} once resolved
        self.v_init = -75.0

    @property
    def cells(self) -> dict:
        built = {}
        for pop, pop_cells in self._registry.items():
            keep = self.biophys_cells.get(pop)
            if not keep:
                continue
            built[pop] = {gid: obj for gid, obj in pop_cells.items() if gid in keep}
        return built

    @property
    def voltage_recording_dt(self) -> float:
        if self.v_recs_dt:
            return next(iter(self.v_recs_dt.values()))
        return super().voltage_recording_dt

    @property
    def membrane_current_recording_dt(self) -> float:
        """Recording time step for membrane current traces in ms"""
        if self.i_recs_dt:
            return next(iter(self.i_recs_dt.values()))
        return super().membrane_current_recording_dt

    def selection(self, select, method: str = "first", bounds=None) -> Self:
        if self._registry:
            raise RuntimeError("selection() must be called before init()")
        self._select_spec = select
        self._select_method = method
        self._select_bounds = bounds
        return self

    @property
    def selection_name(self) -> str | None:
        spec = self._select_spec
        return spec if isinstance(spec, str) else None

    def _apply_celsius(self) -> None:
        """The model's temperature, when it declares one. A template may set
        `h.celsius` while it builds (PyramidalCell: 35), so this runs after
        the cells exist and again at the first run, as livn does."""
        if hasattr(self.model, "neuron_celsius"):
            h.celsius = float(self.model.neuron_celsius())

    def init(self):
        self._load_cells()
        self._load_connections()
        self._insert_opsins()

        if hasattr(h, "nrn_netrec_state_adjust"):
            h.nrn_netrec_state_adjust = 1
        if hasattr(h, "nrn_sparse_partrans"):
            h.nrn_sparse_partrans = 1
        self._apply_celsius()

        # disable defaultdicts
        self._registry = dict(self._registry)
        self.artificial_cells = dict(self.artificial_cells)
        self.biophys_cells = dict(self.biophys_cells)
        self.edge_count = dict(self.edge_count)
        self.syns_set = dict(self.syns_set)

        self.pc.set_maxstep(10)

        return self

    def _insert_opsins(self):
        if not hasattr(self.model, "neuron_opsin_config"):
            return

        opsin_cfg = self.model.neuron_opsin_config()
        if opsin_cfg is None:
            return

        mech_name = opsin_cfg.get("mechanism", "RhO3c")
        populations = opsin_cfg.get("populations", list(self.cells.keys()))
        target_sections = opsin_cfg.get("sections", ["soma"])
        params = opsin_cfg.get("params", {})

        for pop in populations:
            for gid, cell in self.cells.get(pop, {}).items():
                if not self.pc.gid_exists(gid):
                    continue
                if not hasattr(cell, "sections"):
                    continue
                for sec_id, sec in enumerate(cell.sections):
                    sec_name = sec.name().split(".")[-1]
                    if sec_name not in target_sections:
                        continue

                    pp = getattr(h, mech_name)(sec(0.5))
                    for param_name, value in params.items():
                        setattr(pp, param_name, value)

                    self._opsin_refs[(int(gid), sec_id)] = pp

    def _load_cells(self):
        filepath = self.system.files["cells"]
        io_size: int = 1

        if self.rank == 0:
            logger.info("*** Creating cells...")
        st = time.time()

        try:
            self.pc.gid_clear()
        except Exception:
            logger.debug(
                "ParallelContext gid_clear before cell load failed", exc_info=True
            )

        rank = self.comm.Get_rank()

        population_ranges = self.system.cells_meta_data.population_ranges

        celltypes = to_dict(_cell_types(self.system))

        self.model.neuron_celltypes(celltypes)

        ignored = set()
        if hasattr(self.model, "ignored_populations"):
            ignored = set(self.model.ignored_populations())
        for k in list(celltypes.keys()):
            if k in ignored:
                del celltypes[k]

        typenames = sorted(celltypes.keys())
        for k in typenames:
            population_range = population_ranges.get(k, None)
            if population_range is not None:
                celltypes[k]["start"] = population_ranges[k][0]
                celltypes[k]["num"] = population_ranges[k][1]

                if "mechanism" in celltypes[k]:
                    mech_dict = celltypes[k]["mechanism"]
                    if isinstance(mech_dict, str):
                        if rank == 0:
                            mech_dict = from_yaml(mech_dict)
                        mech_dict = self.comm.bcast(mech_dict, root=0)
                    celltypes[k]["mech_dict"] = mech_dict
                    celltypes[k]["mech_file_path"] = "$mechanism"

                if "synapses" in celltypes[k]:
                    synapses_dict = celltypes[k]["synapses"]
                    if "weights" in synapses_dict:
                        weights_config = synapses_dict["weights"]
                        if isinstance(weights_config, list):
                            weights_dicts = weights_config
                        else:
                            weights_dicts = [weights_config]
                        for weights_dict in weights_dicts:
                            if "expr" in weights_dict:
                                expr = weights_dict["expr"]
                                parameter = weights_dict["parameter"]
                                const = weights_dict.get("const", {})
                                clos = ExprClosure(parameter, expr, const)
                                weights_dict["closure"] = clos
                        synapses_dict["weights"] = weights_dicts

        self.cells_meta_data = {
            "source": filepath,
            "cell_attribute_info": self.system.cells_meta_data.cell_attribute_info,
            "population_ranges": population_ranges,
            "population_names": self.system.cells_meta_data.population_names,
            "celltypes": celltypes,
        }

        self.celltypes = celltypes
        self.cell_attribute_info = self.system.cells_meta_data.cell_attribute_info
        self.data_file_path = filepath
        self.coordinates_ns = "Generated Coordinates"
        self.SWC_Types = config.SWCTypesDef.__members__
        self.io_size = io_size
        self.template_paths = [self.template_directory]

        buildable = [
            k for k in typenames if celltypes[k].get("template") != "VecStim"
        ]
        self._selection = self.system.selection(
            self._select_spec,
            populations=buildable,
            seed=self.seed,
            method=self._select_method,
            bounds=self._select_bounds,
        )
        if self._selection is None:
            self._make_cells()
        else:
            self._cell_selection = {
                pop: sorted(int(g) for g in gids)
                for pop, gids in self._selection.items()
                if pop in buildable and len(gids) > 0
            }
            self._make_cells(cell_selection=self._cell_selection)

        self.mkcellstime = time.time() - st
        if self.rank == 0:
            logger.info(f"*** Cells created in {self.mkcellstime:.02f} s")
        local_num_cells = sum(len(cells) for cells in self._registry.values())

        logger.info(f"*** Rank {self.rank} created {local_num_cells} cells")

        st = time.time()

        compat = _Compat(self)
        connect_gjs(compat)

        self.pc.setup_transfer()
        self.connectgjstime = time.time() - st
        if rank == 0:
            logger.info(f"*** Gap junctions created in {self.connectgjstime:.02f} s")

    def _apply_fitted_cell_params(self, cell, mech_dict) -> None:
        params = None
        for block in (mech_dict or {}).values():
            if isinstance(block, dict) and (
                "ic_constant" in block or "V_threshold" in block
            ):
                params = block
                break
        if params is None:
            return
        target = (
            getattr(cell, "hoc_cell", None) or getattr(cell, "cell_obj", None) or cell
        )
        soma = getattr(target, "soma", None)
        if isinstance(soma, list):
            soma = soma[0] if soma else None
        soma = getattr(soma, "section", soma)
        if "ic_constant" in params and soma is not None:
            try:
                soma.ic_constant = float(params["ic_constant"])
            except (AttributeError, LookupError):
                pass  # no `constant` mechanism on this soma
        detector = getattr(cell, "spike_detector", None)
        if "V_threshold" in params and detector is not None:
            detector.threshold = float(params["V_threshold"])

    def _install_spike_filter(self, cell) -> None:
        src_nc = getattr(cell, "spike_detector", None)
        if src_nc is None:
            return
        if self._refractory_period <= 0.0:
            return
        if not hasattr(h, "SpikeFilter"):
            raise RuntimeError(
                f"a {self._refractory_period} ms refractory period needs the "
                "`SpikeFilter` mechanism"
            )

        threshold = float(src_nc.threshold)
        delay = float(src_nc.delay)
        weight = float(src_nc.weight[0])

        # Locate the soma section the original detector watched
        soma_sec = None
        soma_attr = getattr(cell, "soma", None)
        if soma_attr:
            soma_node = soma_attr[0] if isinstance(soma_attr, list) else soma_attr
            soma_sec = getattr(soma_node, "section", soma_node)
        if soma_sec is None:
            hoc_cell = getattr(cell, "hoc_cell", None) or getattr(
                cell, "cell_obj", None
            )
            soma_sec = getattr(hoc_cell, "soma", None)
            if isinstance(soma_sec, list):
                soma_sec = soma_sec[0]
        if soma_sec is None:
            raise RuntimeError(
                f"_install_spike_filter: cannot find soma section for gid {cell.gid}"
            )

        spike_filter = h.SpikeFilter()
        spike_filter.tref = float(self._refractory_period)

        in_nc = h.NetCon(soma_sec(0.5)._ref_v, spike_filter, sec=soma_sec)
        in_nc.threshold = threshold
        in_nc.delay = 0.0  # the filter adds no latency; `out_nc` carries 2 dt
        in_nc.weight[0] = weight
        del delay

        out_nc = h.NetCon(spike_filter, None)
        out_nc.delay = max(2.0 * 0.025, 1e-3)
        out_nc.weight[0] = 1.0

        # Replace the cell's detector so register_cell binds pc.cell /
        # pc.spike_record to the filtered output instead of the raw threshold
        # NetCon. The original src_nc loses its only Python ref and is GC'd,
        # which detaches its watch on _ref_v
        cell.spike_detector = out_nc

        self._spike_filter_refs[int(cell.gid)] = {
            "filter": spike_filter,
            "in_nc": in_nc,
            "out_nc": out_nc,
        }

    def _make_cells(self, cell_selection=None):
        rank = self.rank
        nhosts = int(self.pc.nhost())
        compat = _Compat(self)

        if cell_selection is not None:
            pop_names = sorted(cell_selection.keys())
        else:
            pop_names = sorted(self.celltypes.keys())

        for pop_name in pop_names:
            if cell_selection is None:
                self.pc.barrier()

            if rank == 0:
                logger.info(f"*** Creating population {pop_name}")

            template_name = self.celltypes[pop_name].get("template", None)

            mech_dict = None
            if "mech_file_path" in self.celltypes[pop_name]:
                mech_dict = self.celltypes[pop_name]["mech_dict"]

            # resolve cell constructor
            reduced_cons = None
            if template_name and template_name.startswith("@"):
                # custom @-template constructor
                self.template_dict[pop_name] = None

                def _make_custom_cell(target):
                    def _cell(gid, pop_name, env, mech_dict):
                        cell = env.biophys_cells[pop_name][gid] = _Cell(
                            gid=gid,
                            population_name=pop_name,
                            cell_obj=import_object_by_path(target[1:])(),
                            mech_dict=mech_dict,
                            env=env,
                        )
                        return cell

                    return _cell

                reduced_cons = _make_custom_cell(template_name)
            else:
                # legacy HOC/template path
                template_class = neuron_utils.load_cell_template(
                    compat, pop_name, bcast_template=True
                )
                if cell_selection is None:
                    reduced_cons = cells.get_reduced_cell_constructor(template_name)

            num_cells = 0

            if (pop_name in self.cell_attribute_info) and (
                "Trees" in self.cell_attribute_info[pop_name]
            ):
                if rank == 0:
                    logger.info(f"*** Reading trees for population {pop_name}")

                if cell_selection is not None:
                    gid_range = [
                        gid for gid in cell_selection[pop_name] if gid % nhosts == rank
                    ]
                    (trees, _) = read_tree_selection(
                        self.data_file_path, pop_name, gid_range, comm=self.comm
                    )
                else:
                    if self.node_allocation is None:
                        (trees, _) = scatter_read_trees(
                            self.data_file_path,
                            pop_name,
                            comm=self.comm,
                            io_size=self.io_size,
                        )
                    else:
                        (trees, _) = scatter_read_trees(
                            self.data_file_path,
                            pop_name,
                            comm=self.comm,
                            io_size=self.io_size,
                            node_allocation=self.node_allocation,
                        )

                if rank == 0:
                    logger.info(f"*** Done reading trees for population {pop_name}")

                first_gid = None
                for i, (gid, tree) in enumerate(trees):
                    if rank == 0:
                        logger.info(f"*** Creating {pop_name} gid {gid}")
                    if first_gid is None:
                        first_gid = gid

                    if reduced_cons:
                        cell = reduced_cons(
                            gid=gid, pop_name=pop_name, env=compat, mech_dict=mech_dict
                        )
                    elif cell_selection is not None and template_class is not None:
                        if isinstance(template_class, HocObject):
                            hoc_cell = cells.make_hoc_cell(
                                compat, pop_name, gid, neurotree_dict=tree
                            )
                            if mech_dict is None:
                                cell = hoc_cell
                            else:
                                cell = cells.make_biophys_cell(
                                    gid=gid,
                                    population_name=pop_name,
                                    hoc_cell=hoc_cell,
                                    env=compat,
                                    tree_dict=tree,
                                    mech_dict=mech_dict,
                                )
                        else:
                            # the same constructor as without a selection,
                            # so the template receives its fitted parameters
                            cell = cells.make_biophys_cell(
                                gid=gid,
                                population_name=pop_name,
                                env=compat,
                                tree_dict=tree,
                                mech_dict=mech_dict,
                            )
                    else:
                        cell = cells.make_biophys_cell(
                            gid=gid,
                            population_name=pop_name,
                            env=compat,
                            tree_dict=tree,
                            mech_dict=mech_dict,
                        )

                    soma_xyz = cells.get_soma_xyz(tree, self.SWC_Types)
                    cell.position(soma_xyz[0], soma_xyz[1], soma_xyz[2])

                    self._apply_fitted_cell_params(cell, mech_dict)
                    self._install_spike_filter(cell)
                    cells.register_cell(compat, pop_name, gid, cell)
                    num_cells += 1
                del trees

            elif (pop_name in self.cell_attribute_info) and (
                self.coordinates_ns in self.cell_attribute_info[pop_name]
            ):
                if rank == 0:
                    logger.info(f"*** Reading coordinates for population {pop_name}")

                if cell_selection is not None:
                    gid_range = [
                        gid for gid in cell_selection[pop_name] if gid % nhosts == rank
                    ]
                    coords_iter, coords_attr_info = read_cell_attribute_selection(
                        self.data_file_path,
                        pop_name,
                        selection=gid_range,
                        namespace=self.coordinates_ns,
                        comm=self.comm,
                        return_type="tuple",
                    )
                else:
                    if self.node_allocation is None:
                        cell_attr_dict = scatter_read_cell_attributes(
                            self.data_file_path,
                            pop_name,
                            namespaces=[self.coordinates_ns],
                            comm=self.comm,
                            io_size=self.io_size,
                            return_type="tuple",
                        )
                    else:
                        cell_attr_dict = scatter_read_cell_attributes(
                            self.data_file_path,
                            pop_name,
                            namespaces=[self.coordinates_ns],
                            node_allocation=self.node_allocation,
                            comm=self.comm,
                            io_size=self.io_size,
                            return_type="tuple",
                        )
                    coords_iter, coords_attr_info = cell_attr_dict[self.coordinates_ns]

                if rank == 0:
                    logger.info(
                        f"*** Done reading coordinates for population {pop_name}"
                    )

                x_index = coords_attr_info.get("X Coordinate", None)
                y_index = coords_attr_info.get("Y Coordinate", None)
                z_index = coords_attr_info.get("Z Coordinate", None)

                for i, (gid, cell_coords) in enumerate(coords_iter):
                    if rank == 0:
                        logger.info(f"*** Creating {pop_name} gid {gid}")

                    cell_x = cell_coords[x_index][0]
                    cell_y = cell_coords[y_index][0]
                    cell_z = cell_coords[z_index][0]

                    if reduced_cons:
                        cell = reduced_cons(
                            gid=gid, pop_name=pop_name, env=compat, mech_dict=mech_dict
                        )
                    else:
                        cell = cells.make_hoc_cell(compat, pop_name, gid)
                    cell.position(cell_x, cell_y, cell_z)
                    self._apply_fitted_cell_params(cell, mech_dict)
                    self._install_spike_filter(cell)
                    cells.register_cell(compat, pop_name, gid, cell)
                    num_cells += 1
            else:
                raise RuntimeError(
                    f"_make_cells: unknown cell configuration type "
                    f"for cell type {pop_name}"
                )

            h.define_shape()

            self.recording_sets[pop_name] = set()

            logger.info(
                f"*** Rank {rank}: Created {num_cells} cells from population {pop_name}"
            )

        self.node_allocation = set(self.gidset)

        if self.rank == 0 and self._spike_filter_refs:
            logger.info(
                f"*** SpikeFilter installed on {len(self._spike_filter_refs)} cells "
                f"(tref = {self._refractory_period:.2f} ms)"
            )

    def _load_connections(self):
        synapses = self.system.connections_config["synapses"]
        filepath = self.system.files["connections"]
        cell_filepath = self.system.files["cells"]
        io_size: int = 1

        microcircuit_inputs = False
        if hasattr(self.model, "neuron_microcircuit_inputs"):
            microcircuit_inputs = self.model.neuron_microcircuit_inputs()
        if not self.cells_meta_data:
            raise RuntimeError("Please load the cells first using load_cells()")

        st = time.time()
        if self.rank == 0:
            logger.info("*** Creating connections:")

        rank = self.comm.Get_rank()
        if rank == 0:
            color = 1
        else:
            color = 0
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

        ignored = set()
        if hasattr(self.model, "ignored_populations"):
            ignored = set(self.model.ignored_populations())
        if ignored:
            projection_dict = {
                dst: [src for src in srcs if src not in ignored]
                for dst, srcs in projection_dict.items()
                if dst not in ignored
            }

        self.input_sources = {
            pop_name: set() for pop_name in self.cells_meta_data["celltypes"].keys()
        }

        class _binding:
            pass

        self.this = this = _binding()
        this.__dict__.update(
            {
                "pc": self.pc,
                "connectivity_file_path": filepath,
                "forest_file_path": cell_filepath,
                "io_size": io_size,
                "comm": self.comm,
                "node_allocation": self.node_allocation,
                "edge_count": self.edge_count,
                "biophys_cells": self.biophys_cells,
                "gidset": self.gidset,
                "recording_sets": self.recording_sets,
                "microcircuit_inputs": microcircuit_inputs,
                "microcircuit_input_sources": self.input_sources,
                "spike_input_attribute_info": None,
                "cell_selection": self._cell_selection,
                "netclamp_config": None,
                "use_cell_attr_gen": True,
                "cell_attr_gen_cache_size": 4,
                "cleanup": False,
                "projection_dict": projection_dict,
                "Populations": self.system.connections_config["population_definitions"],
                "layers": self.system.connections_config["layer_definitions"],
                "connection_config": DotDict.create(synapses),
                "connection_velocity": defaultdict(lambda: 250),
                "SWC_Types": config.SWCTypesDef.__members__,
                "celltypes": self.cells_meta_data["celltypes"],
                "cells": self._registry,
                "artificial_cells": self.artificial_cells,
                "dt": 0.025,  # TODO: hoist into run
                "t_vec": self.t_vec,
                "id_vec": self.id_vec,
                "t_rec": self.t_rec,
            }
        )
        self.synapse_manager = SynapseManager(
            this,
            self.model.neuron_synapse_mechanisms(),
            self.model.neuron_synapse_rules(),
        )
        this.__dict__["synapse_manager"] = self.synapse_manager

        if self._cell_selection is not None:
            connect_cell_selection(this)
        else:
            connect_cells(this)

        self.input_sources = this.microcircuit_input_sources
        self.node_allocation = this.node_allocation

        self.pc.set_maxstep(10.0)

        self.connectcellstime = time.time() - st

        if self.rank == 0:
            logger.info(
                f"*** Done creating connections: time = {self.connectcellstime:.02f} s"
            )
        edge_count = int(sum(self.edge_count[dest] for dest in self.edge_count))
        logger.info(f"*** Rank {rank} created {edge_count} connections")

    def run(
        self,
        duration,
        stimulus: Stimulus | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.duration = duration
        current_time = self.t

        if stimulus is not None:
            if not isinstance(stimulus, Stimulus):
                stimulus = Stimulus.from_arg(stimulus)
            stimulus = self.model.prepare_stimulus(stimulus)

        is_irradiance = stimulus is not None and stimulus.input_mode == "irradiance"

        if stimulus is not None and not is_irradiance:
            if stimulus.gids is None:
                stimulus.gids = self.active_gids()

            sections_per_neuron = len(stimulus) // len(self.active_neuron_coordinates())

            start_step = int(round(current_time / stimulus.dt))
            if not math.isclose(
                start_step * stimulus.dt, current_time, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    "Stimulus duration must align with dt when continuing runs"
                )

            for count, (gid, st) in enumerate(stimulus):
                if not (self.pc.gid_exists(gid)):
                    continue

                section_id = count % sections_per_neuron

                # pc.gid2cell returns the source object of the NetCon
                # registered with pc.cell. For biophysical cells we wrap
                # the somatic detector in a SpikeFilter ARTIFICIAL_CELL
                # (see _install_spike_filter), so gid2cell would return
                # the filter rather than the biophys cell. Look up the
                # underlying cell via self.cells instead
                cell = None
                for pop_cells in self._registry.values():
                    if gid in pop_cells:
                        cell = pop_cells[gid]
                        break
                if cell is None:
                    cell = self.pc.gid2cell(gid)
                if "VecStim" in getattr(cell, "hname", lambda: "")():
                    # artificial STIM cell
                    print(
                        f"Warning: Cell {gid} is a VecStim cell; use play() to induce spikes. Skipping stimulus ..."
                    )
                    continue

                sec = cell.sections[section_id]
                sec.push()
                has_extracellular = h.ismembrane("extracellular")
                h.pop_section()

                if not has_extracellular:
                    continue

                key = (int(gid), section_id)
                state = self._stimulus_vectors.get(key)
                target = sec(0.5)

                if state is None:
                    state = {
                        "segment": target,
                        "dt": stimulus.dt,
                        "buffer": np.zeros(0, dtype=np.float64),
                        "length": 0,
                    }
                    self._stimulus_vectors[key] = state
                else:
                    if not math.isclose(
                        state["dt"], stimulus.dt, rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(
                            f"Stimulus dt mismatch for gid {gid}; call clear() before rerunning"
                        )

                scheduled = state["length"]
                if start_step < scheduled:
                    raise ValueError(
                        f"Stimulus for gid {gid} overlaps existing schedule; call clear() first"
                    )

                if start_step > scheduled:
                    self._ensure_stimulus_buffer(state, start_step)
                    state["length"] = start_step

                if len(st) > 0:
                    values = np.asarray(st, dtype=np.float64)
                    end_step = start_step + len(values)
                    self._ensure_stimulus_buffer(state, end_step)
                    state["buffer"][start_step:end_step] = values
                    state["length"] = end_step
        if stimulus is not None and not is_irradiance and self._stimulus_vectors:
            self._ensure_stimulus_callback()

        if is_irradiance:
            self._setup_opsin_stimulus(stimulus, current_time)

        first_run = self.t == 0

        stored_dt = self._dt
        requested_dt = dt if dt is not None else stored_dt
        if requested_dt is None:
            requested_dt = 0.025

        if not first_run and stored_dt is not None:
            if abs(requested_dt - stored_dt) > 1e-12:
                raise ValueError(
                    "Cannot change dt during a running simulation; call clear() first."
                )
            requested_dt = stored_dt

        if first_run:
            self._dt = requested_dt

        if self._dt is not None:
            self._stimulus_step_index = int(round(current_time / self._dt))

        verbose = self.rank == 0 and kwargs.get("verbose", True)

        if verbose:
            if first_run:
                logger.info("*** finitialize")
            else:
                logger.info(f"*** Continuing run from t={self.t:.2f} ms")

        self.t_rec.record(h._ref_t)

        self.clear_recordings()

        if first_run:
            h.v_init = float(self.v_init)
            h.stdinit()
            h.secondorder = 2  # crank-nicholson
            h.dt = requested_dt
            self.pc.timeout(600.0)
            self._apply_celsius()
            h.finitialize(h.v_init)
            h.finitialize(h.v_init)
            if verbose:
                logger.info("*** Completed finitialize")
        else:
            self.pc.timeout(600.0)

        target_time = self.t + duration

        if verbose:
            logger.info(f"*** Simulating {duration} ms (target t={target_time:.2f} ms)")

        q = time.time()
        if self._weight_recording_active and self._weight_nc_refs:
            w_dt = self._weight_rec_dt
            while h.t < target_time - w_dt / 2:
                next_t = min(h.t + w_dt, target_time)
                self.pc.psolve(next_t)
                self._sample_weights()
            if h.t < target_time - 1e-6:
                self.pc.psolve(target_time)
        else:
            self.pc.psolve(target_time)
        self.psolvetime = time.time() - q

        self.t = target_time

        if verbose:
            logger.info(f"*** Done simulating within {self.psolvetime:.2f} s")

        # collect spikes
        tt = np.array(self.t_vec.as_numpy(), copy=True)
        ii = np.asarray(self.id_vec.as_numpy(), dtype=np.uint32)

        built = np.fromiter(
            (int(gid) for pop in self.biophys_cells.values() for gid in pop),
            dtype=np.int64,
        )
        if tt.size > 0:
            keep = np.isin(ii.astype(np.int64), built)
            tt, ii = tt[keep], ii[keep]
        if current_time != 0.0 and tt.size > 0:
            tt -= current_time
            tt[tt < 0.0] = 0.0

        # collect voltages
        if len(self.v_recs) > 0:
            iv = []
            v = []
            for (gid, sec_id), rec in self.v_recs.items():
                iv.append(gid)
                v.append(rec.as_numpy())
            iv = np.asarray(iv, dtype=np.uint32)
            v = np.array(v, dtype=np.float32)
        else:
            iv = None
            v = None

        # collect membrane currents
        if len(self.i_recs) == 0:
            self.duration = None
            return self._result(current_time, duration, ii, tt, iv, v, None, None)

        gids = self.active_gids()
        sections_per_neuron = len(self.i_recs) // len(gids)
        gid_to_index = {int(g): idx for idx, g in enumerate(gids)}
        any_rec = next(iter(self.i_recs.values()))
        T = len(any_rec)
        currents = np.zeros((len(self.i_recs), T), dtype=np.float32)
        im = np.ones([len(self.i_recs)], dtype=np.int32) * -1

        for (gid, sec_id), rec in self.i_recs.items():
            idx = gid_to_index.get(int(gid))
            if idx is None:
                continue
            arr = rec.as_numpy()
            # i_membrane_ with cvode.use_fast_imem(1) is the absolute
            # transmembrane current per segment in nA, not a current
            # density so we convert to microamperes
            arr = arr * 1e-3
            if len(arr) != T:
                # pad or truncate to T
                if len(arr) < T:
                    pad = np.zeros(T, dtype=np.float32)
                    pad[: len(arr)] = arr
                    arr = pad
                else:
                    arr = arr[:T]
            currents[idx * sections_per_neuron + sec_id, :] = arr
            im[idx * sections_per_neuron + sec_id] = gid

        self.duration = None

        return self._result(current_time, duration, ii, tt, iv, v, im, currents)

    def _result(self, current_time, duration, ii, tt, iv, v, im, currents):
        sections = None
        if iv is not None:
            sections = np.asarray(
                [self._section_name(gid, sec_id) for (gid, sec_id) in self.v_recs]
            )
        return (
            Run(t0=current_time, duration=duration)
            .add_spikes(ii, tt)
            .add_voltage(iv, v, dt=self.voltage_recording_dt, sections=sections)
            .add_current(im, currents, dt=self.membrane_current_recording_dt)
        )

    def _section_name(self, gid: int, sec_id: int) -> str:
        for pop_cells in self._registry.values():
            cell = pop_cells.get(int(gid))
            if cell is not None:
                try:
                    return str(cell.sections[sec_id].name()).split(".")[-1]
                except (AttributeError, IndexError, TypeError):
                    break
        return str(sec_id)

    def _ensure_stimulus_buffer(self, state: dict, target_length: int) -> None:
        current_length = len(state.get("buffer", []))
        if target_length <= current_length:
            return

        extra = target_length - current_length
        if current_length == 0:
            state["buffer"] = np.zeros(target_length, dtype=np.float64)
        else:
            pad = np.zeros(extra, dtype=np.float64)
            state["buffer"] = np.concatenate([state["buffer"], pad])

    def _ensure_stimulus_callback(self) -> None:
        if self._stimulus_callback_registered:
            return

        cvode = h.CVode()
        cvode.extra_scatter_gather(0, self._update_stimulus_values)
        self._stimulus_callback_registered = True

    def _update_stimulus_values(self) -> None:
        if not self._stimulus_vectors:
            return

        if self._dt is not None:
            current_time = self._stimulus_step_index * self._dt
        else:
            current_time = float(h.t)
        for state in self._stimulus_vectors.values():
            buffer = state["buffer"]
            if buffer.size == 0:
                value = 0.0
            else:
                dt = state["dt"]
                if dt <= 0:
                    value = buffer[-1]
                else:
                    idx = int(current_time / dt)
                    if idx < buffer.size:
                        value = buffer[idx]
                    else:
                        value = buffer[-1]
            state["segment"].e_extracellular = float(value)
        if self._dt is not None:
            self._stimulus_step_index += 1

    def _setup_opsin_stimulus(self, stimulus, current_time):
        if not self._opsin_refs:
            raise ValueError(
                "Stimulus has input_mode='irradiance' but no opsins are "
                "attached. Configure opsin in the model."
            )

        phi_stimulus = stimulus.convert_to("photon_flux")

        if phi_stimulus.gids is None:
            phi_stimulus.gids = self.active_gids()

        sections_per_neuron = len(phi_stimulus) // len(self.active_neuron_coordinates())

        start_step = int(round(current_time / phi_stimulus.dt))

        for count, (gid, st) in enumerate(phi_stimulus):
            if not self.pc.gid_exists(gid):
                continue

            section_id = count % sections_per_neuron
            key = (int(gid), section_id)

            if key not in self._opsin_refs:
                continue

            phi_array = np.asarray(st, dtype=np.float64)

            state = self._opsin_stimulus_vectors.get(key)
            if state is None:
                state = {
                    "pp": self._opsin_refs[key],
                    "dt": phi_stimulus.dt,
                    "buffer": np.zeros(0, dtype=np.float64),
                    "length": 0,
                }
                self._opsin_stimulus_vectors[key] = state

            scheduled = state["length"]
            if start_step > scheduled:
                self._ensure_stimulus_buffer(state, start_step)
                state["length"] = start_step

            if len(phi_array) > 0:
                end_step = start_step + len(phi_array)
                self._ensure_stimulus_buffer(state, end_step)
                state["buffer"][start_step:end_step] = phi_array
                state["length"] = end_step

        if self._opsin_stimulus_vectors:
            self._ensure_opsin_callback()

    def _ensure_opsin_callback(self) -> None:
        if self._opsin_callback_registered:
            return

        cvode = h.CVode()
        cvode.extra_scatter_gather(0, self._update_opsin_phi)
        self._opsin_callback_registered = True

    def _update_opsin_phi(self) -> None:
        if not self._opsin_stimulus_vectors:
            return

        if self._dt is not None:
            current_time = self._opsin_step_index * self._dt
        else:
            current_time = float(h.t)

        for state in self._opsin_stimulus_vectors.values():
            buffer = state["buffer"]
            if buffer.size == 0:
                phi = 0.0
            else:
                dt = state["dt"]
                if dt <= 0:
                    phi = buffer[-1]
                else:
                    idx = int(current_time / dt)
                    if idx < buffer.size:
                        phi = buffer[idx]
                    else:
                        phi = buffer[-1]
            state["pp"].phi = float(phi)

        if self._dt is not None:
            self._opsin_step_index += 1

    def clear_recordings(self) -> Self:
        self.t_vec.resize(0)
        self.id_vec.resize(0)
        self.t_rec.resize(0)
        for v_rec in self.v_recs.values():
            v_rec.resize(0)
        for i_rec in self.i_recs.values():
            i_rec.resize(0)
        # w_recs need to accumulate across consecutive run() calls so they are only freed in clear()

        return self

    def clear(self):
        self.clear_recordings()
        for w_rec in self.w_recs.values():
            w_rec.resize(0)
        if self._w_rec_times is not None:
            self._w_rec_times.resize(0)
        self.t = 0
        self._dt = None
        self._stimulus_step_index = 0
        self._stimulus_vectors.clear()
        self._opsin_step_index = 0
        self._opsin_stimulus_vectors.clear()

        return self

    def set_weights(self, weights):
        params = []
        dropped = []
        for k, v in weights.items():
            try:
                params.append((SynapticParam.from_string(k), v))
            except ValueError as e:
                dropped.append((k, str(e)))
        if dropped and self.rank == 0:
            print(
                f"[set_weights] dropped {len(dropped)} keys: {dropped[:3]}", flush=True
            )

        self.this.__dict__.update({"phenotype_dict": {}, "cache_queries": True})
        update_network_params(self.this, params)

        return self

    def set_noise(self, noise: dict):
        if not hasattr(self.model, "neuron_noise_mechanism"):
            if self.rank == 0:
                print(
                    f"[set_noise] Model {self.model} does not support noise setter, skipping {len(noise)} params",
                    flush=True,
                )
            return self

        # Merge with previously-applied noise so partial updates do not
        # silently reset unspecified parameters to function defaults
        self._noise_state.update(noise)
        merged = dict(self._noise_state)

        for population, pop_cells in self.cells.items():
            for gid, cell in pop_cells.items():
                if not (self.pc.gid_exists(gid)):
                    continue
                secs = []
                if hasattr(cell, "sections"):
                    secs = cell.sections
                for idx, sec in enumerate(secs):
                    sec.push()
                    fluct, state = self._flucts.get(f"{gid}-{idx}", (None, None))
                    if fluct is None:
                        fluct, state = self.model.neuron_noise_mechanism(sec(0.5))
                        self._flucts[f"{gid}-{idx}"] = (fluct, state)

                    self.model.neuron_noise_configure(
                        population, fluct, state, **merged
                    )

                    h.pop_section()

        return self

    def _iter_stdp_point_processes(self):  # -> (gid, syn_id, mech_name, point_process)
        """Iterates over all StdpLinExp2Syn/StdpLinExp2SynNMDA point processes"""
        if self.synapse_manager is None:
            return

        for gid, syn_dict in self.synapse_manager.pps_dict.items():
            for syn_id, pps in syn_dict.items():
                for mech_key, pp in pps.mech.items():
                    if pp is not None and hasattr(pp, "plasticity_on"):
                        try:
                            name = pp.hname().split("[")[0]
                        except Exception:
                            name = str(mech_key)
                        yield gid, syn_id, name, pp

    def _iter_stdp_connections(
        self,
    ):  # ->  (gid, syn_id, mech_name, point_process, netcon)
        """Iterates over every STDP connection in the network"""
        if self.synapse_manager is None:
            return
        for gid, syn_dict in self.synapse_manager.pps_dict.items():
            for syn_id, pps in syn_dict.items():
                for mech_key, pp in pps.mech.items():
                    if pp is not None and hasattr(pp, "plasticity_on"):
                        nc = pps.netcon.get(mech_key)
                        if nc is not None:
                            try:
                                name = pp.hname().split("[")[0]
                            except Exception:
                                name = str(mech_key)
                            yield gid, syn_id, name, pp, nc

    def enable_plasticity(self, config=None) -> Self:
        """Enable spike-timing dependent plasticity on synapses

        Parameters
        ----------
        config : dict, optional
            Either a flat ``{param: value}`` dict applied to all synapses,
            or a nested ``{population_name: {param: value}}`` dict where
            each group targets specific mechanism types defined by
            ``model.neuron_plasticity_mechanism_groups()``

            When None, uses ``neuron_plasticity_defaults()``
        """
        if config is None:
            if hasattr(self.model, "neuron_plasticity_defaults"):
                config = self.model.neuron_plasticity_defaults()
            else:
                config = {}

        per_population = config and isinstance(next(iter(config.values())), dict)

        mech_to_group = {}
        if per_population and hasattr(self.model, "neuron_plasticity_mechanism_groups"):
            for group, mechs in self.model.neuron_plasticity_mechanism_groups().items():
                for m in mechs:
                    mech_to_group[m] = group

        count = 0
        for gid, syn_id, mech_name, pp in self._iter_stdp_point_processes():
            if per_population:
                group = mech_to_group.get(mech_name)
                group_config = config.get(group, {}) if group else {}
            else:
                group_config = config

            for param, value in group_config.items():
                if hasattr(pp, param):
                    setattr(pp, param, value)
            pp.plasticity_on = 1
            count += 1

        self._plasticity_enabled = True

        if self.rank == 0:
            logger.info(f"*** Enabled STDP on {count} synapses")

        return self

    def disable_plasticity(self):
        """Disable STDP on synapses / freeze weights"""
        for gid, syn_id, mech_name, pp in self._iter_stdp_point_processes():
            pp.plasticity_on = 0

        self._plasticity_enabled = False

        if self.rank == 0:
            logger.info("*** Disabled STDP (weights frozen)")

        return self

    def get_weights(self) -> dict[tuple, float]:  # gid, syn_id, mech_name -> weight
        """Returns current synaptic weights of all plastic synapses"""
        weights = {}
        for gid, syn_id, mech_name, pp, nc in self._iter_stdp_connections():
            weights[(int(gid), int(syn_id), mech_name)] = float(nc.weight[2])
        return weights

    def normalize_weights(self, target: float | None = None) -> Self:
        """Synaptic scaling to normalize incoming weights per neuron (Turrigiano, 2008)

        Parameters
        ----------
        target : float, optional
            Desired sum of incoming weights for each neuron. When ``None``, the target
            is set to the number of incoming STDP connections for that neuron
            (i.e. the sum that would result if every weight were 1.0)
        """
        per_neuron: dict[int, list] = defaultdict(list)  # (nc, w_min, w_max)
        for gid, syn_id, mech_name, pp, nc in self._iter_stdp_connections():
            per_neuron[int(gid)].append((nc, float(pp.w_min), float(pp.w_max)))

        scaled_neurons = 0
        for gid, conns in per_neuron.items():
            t = target if target is not None else float(len(conns))

            # Iterative normalization where clamped weights are fixed,
            # and the remaining budget is redistributed among free
            free = list(conns)
            clamped_sum = 0.0
            for _ in range(20):  # should converge quickly
                free_sum = sum(nc.weight[2] for nc, _, _ in free)
                remaining = t - clamped_sum
                if free_sum <= 0 or abs(free_sum - remaining) < 1e-12:
                    break
                scale = remaining / free_sum
                next_free = []
                for nc, w_min, w_max in free:
                    new_w = nc.weight[2] * scale
                    if new_w >= w_max:
                        nc.weight[2] = w_max
                        clamped_sum += w_max
                    elif new_w <= w_min:
                        nc.weight[2] = w_min
                        clamped_sum += w_min
                    else:
                        nc.weight[2] = new_w
                        next_free.append((nc, w_min, w_max))
                if len(next_free) == len(free):
                    break
                free = next_free
            scaled_neurons += 1

        if self.rank == 0:
            logger.info(
                f"*** Synaptic scaling applied to {scaled_neurons} neurons "
                f"({sum(len(v) for v in per_neuron.values())} connections)"
            )

        return self

    def record_weights(self, dt: float = 0.1) -> Self:
        """Record weight evolution of all plastic connections."""
        self._weight_rec_dt = dt
        self._weight_nc_refs = {}

        for gid, syn_id, mech_name, pp, nc in self._iter_stdp_connections():
            key = (int(gid), int(syn_id), mech_name)
            self._weight_nc_refs[key] = nc
            if key not in self.w_recs:
                self.w_recs[key] = h.Vector()

        if self._w_rec_times is None:
            self._w_rec_times = h.Vector()

        self._weight_recording_active = True

        if self.rank == 0:
            logger.info(
                f"*** Recording weights for {len(self._weight_nc_refs)} STDP connections (dt={dt} ms)"
            )

        return self

    def _sample_weights(self):
        t = float(h.t)
        if self._w_rec_times is not None:
            self._w_rec_times.append(t)
        for key, nc in self._weight_nc_refs.items():
            self.w_recs[key].append(float(nc.weight[2]))

    def _record_voltage(
        self, population: str, dt: float, gids=None, sections=None
    ) -> "Env":
        if population not in self.cells:
            logger.info(
                f"Rank {self.rank} has no cells; try reducing the number of ranks."
            )
            return self

        self.v_recs_dt[population] = dt
        wanted = None if gids is None else {int(g) for g in gids}
        names = None if sections is None else {str(n) for n in sections}

        for gid, cell in self.cells[population].items():
            if wanted is not None and int(gid) not in wanted:
                continue
            if not (self.pc.gid_exists(gid)):
                continue

            if "VecStim" in getattr(cell, "hname", lambda: "")():
                continue

            for sec_id, sec in enumerate(cell.sections):
                if names is not None and str(sec.name()).split(".")[-1] not in names:
                    continue
                self.v_recs[(int(gid), sec_id)] = h.Vector()
                self.v_recs[(int(gid), sec_id)].record(sec(0.5)._ref_v, dt)

        return self

    def _record_membrane_current(self, population: str, dt: float) -> "Env":
        """Record transmembrane current (i_membrane) at soma midpoint for each cell"""
        if population not in self.cells:
            logger.info(
                f"Rank {self.rank} has no cells; try reducing the number of ranks."
            )
            return

        try:
            h.cvode.use_fast_imem(1)
        except Exception:
            pass

        self.i_recs_dt[population] = dt

        for gid, cell in self.cells[population].items():
            if not (self.pc.gid_exists(gid)):
                continue

            if "VecStim" in getattr(cell, "hname", lambda: "")():
                continue

            for sec_id, sec in enumerate(cell.sections):
                self.i_recs[(int(gid), sec_id)] = h.Vector()
                self.i_recs[(int(gid), sec_id)].record(sec(0.5)._ref_i_membrane_, dt)
                area_um2 = h.area(0.5, sec=sec)
                self.i_area[(int(gid), sec_id)] = float(area_um2) * 1e-8

        return self

    def apply_stimulus_from_h5(
        self,
        filepath: str,
        namespace: str,
        attribute: str = "Spike Train",
        onset: int = 0,
        io_size: int = 1,
        microcircuit_inputs: bool = True,
        n_trials: int = 1,
        equilibration_duration: float = 250.0,
    ):
        rank = self.comm.Get_rank()

        if rank == 0:
            logger.info(f"*** Stimulus onset is {onset} ms")

        if rank == 0:
            color = 1
        else:
            color = 0
        comm0 = self.comm.Split(color, 0)

        spike_input_attribute_info = None
        if rank == 0:
            spike_input_attribute_info = read_cell_attribute_info(
                filepath,
                sorted(self.system.connections_config["population_definitions"].keys()),
                comm=comm0,
            )

        spike_input_attribute_info = self.comm.bcast(spike_input_attribute_info, root=0)
        comm0.Free()

        celltypes = self.cells_meta_data["celltypes"]
        input_file_path = self.system.files["cells"]

        trial_index_attr = "Trial Index"
        trial_dur_attr = "Trial Duration"

        # VecStim populations
        for pop_name in sorted(celltypes.keys()):
            if "spike train" not in celltypes[pop_name]:
                continue

            vecstim_namespace = celltypes[pop_name]["spike train"].get("namespace")
            vecstim_attr = celltypes[pop_name]["spike train"].get("attribute")

            has_vecstim = False
            vecstim_source_loc = []

            if spike_input_attribute_info is not None:
                if pop_name in spike_input_attribute_info:
                    if namespace in spike_input_attribute_info[pop_name]:
                        has_vecstim = True
                        vecstim_source_loc.append((filepath, namespace, attribute))

            if self.system.cells_meta_data.cell_attribute_info is not None:
                if vecstim_namespace is not None:
                    if pop_name in self.system.cells_meta_data.cell_attribute_info:
                        if (
                            vecstim_namespace
                            in self.system.cells_meta_data.cell_attribute_info[pop_name]
                        ):
                            has_vecstim = True
                            vecstim_source_loc.append(
                                (input_file_path, vecstim_namespace, vecstim_attr)
                            )

            if not has_vecstim:
                continue

            for input_path, input_ns, input_attr in vecstim_source_loc:
                if rank == 0:
                    logger.info(
                        f"*** Initializing stimulus population {pop_name} from input path {input_path} namespace {input_ns}"
                    )

                if self.node_allocation is None:
                    cell_vecstim_dict = scatter_read_cell_attributes(
                        input_path,
                        pop_name,
                        namespaces=[input_ns],
                        mask={
                            input_attr,
                            vecstim_attr,
                            trial_index_attr,
                            trial_dur_attr,
                        },
                        comm=self.comm,
                        io_size=io_size,
                        return_type="tuple",
                    )
                    vecstim_iter, vecstim_attr_info = cell_vecstim_dict[input_ns]
                else:
                    selection = list(self.artificial_cells.get(pop_name, {}).keys())
                    (
                        vecstim_iter,
                        vecstim_attr_info,
                    ) = scatter_read_cell_attribute_selection(
                        input_path,
                        pop_name,
                        selection,
                        namespace=input_ns,
                        mask={
                            input_attr,
                            vecstim_attr,
                            trial_index_attr,
                            trial_dur_attr,
                        },
                        comm=self.comm,
                        io_size=io_size,
                        return_type="tuple",
                    )

                vecstim_attr_index = vecstim_attr_info.get(vecstim_attr, None)
                trial_index_attr_index = vecstim_attr_info.get(trial_index_attr, None)
                trial_dur_attr_index = vecstim_attr_info.get(trial_dur_attr, None)

                for gid, vecstim_tuple in vecstim_iter:
                    if not self.pc.gid_exists(gid):
                        continue

                    cell = self.artificial_cells[pop_name][gid]

                    spiketrain = vecstim_tuple[vecstim_attr_index]
                    trial_duration = None
                    trial_index = None
                    if trial_index_attr_index is not None:
                        trial_index = vecstim_tuple[trial_index_attr_index]
                        trial_duration = vecstim_tuple[trial_dur_attr_index]

                    if len(spiketrain) > 0:
                        spiketrain = self._merge_spiketrain_trials(
                            spiketrain,
                            trial_index,
                            trial_duration,
                            n_trials,
                        )
                        spiketrain += equilibration_duration + onset

                        if len(spiketrain) > 0:
                            cell.play(h.Vector(spiketrain.astype(np.float64)))
                            if rank == 0:
                                logger.info(
                                    f"*** Spike train for {pop_name} gid {gid} is of length {len(spiketrain)} ({spiketrain[0]} : {spiketrain[-1]} ms)"
                                )

        gc.collect()

        if microcircuit_inputs and self.input_sources is not None:
            for pop_name in sorted(self.input_sources.keys()):
                gid_range = self.input_sources.get(pop_name, set())
                this_gid_range = gid_range

                has_spike_train = False
                spike_input_source_loc = []

                if spike_input_attribute_info is not None:
                    if pop_name in spike_input_attribute_info:
                        if namespace in spike_input_attribute_info[pop_name]:
                            has_spike_train = True
                            spike_input_source_loc.append((filepath, namespace))

                if self.system.cells_meta_data.cell_attribute_info is not None:
                    if pop_name in self.system.cells_meta_data.cell_attribute_info:
                        if (
                            namespace
                            in self.system.cells_meta_data.cell_attribute_info[pop_name]
                        ):
                            has_spike_train = True
                            spike_input_source_loc.append((input_file_path, namespace))

                if rank == 0:
                    logger.info(
                        f"*** Initializing input source {pop_name} from locations {spike_input_source_loc}"
                    )

                if has_spike_train:
                    vecstim_attr_set = {"t", trial_index_attr, trial_dur_attr}
                    if attribute is not None:
                        vecstim_attr_set.add(attribute)
                    if pop_name in celltypes:
                        if "spike train" in celltypes[pop_name]:
                            vecstim_attr_set.add(
                                celltypes[pop_name]["spike train"]["attribute"]
                            )

                    cell_spikes_items = []
                    for input_path, input_ns in spike_input_source_loc:
                        item = scatter_read_cell_attribute_selection(
                            input_path,
                            pop_name,
                            list(this_gid_range),
                            namespace=input_ns,
                            mask=vecstim_attr_set,
                            comm=self.comm,
                            io_size=io_size,
                            return_type="tuple",
                        )
                        cell_spikes_items.append(item)

                    for cell_spikes_iter, cell_spikes_attr_info in cell_spikes_items:
                        if len(cell_spikes_attr_info) == 0:
                            continue

                        trial_index_attr_index = cell_spikes_attr_info.get(
                            trial_index_attr, None
                        )
                        trial_dur_attr_index = cell_spikes_attr_info.get(
                            trial_dur_attr, None
                        )

                        if (attribute is not None) and (
                            attribute in cell_spikes_attr_info
                        ):
                            spike_train_attr_index = cell_spikes_attr_info.get(
                                attribute, None
                            )
                        elif "t" in cell_spikes_attr_info.keys():
                            spike_train_attr_index = cell_spikes_attr_info.get(
                                "t", None
                            )
                        elif "Spike Train" in cell_spikes_attr_info.keys():
                            spike_train_attr_index = cell_spikes_attr_info.get(
                                "Spike Train", None
                            )
                        elif len(this_gid_range) > 0:
                            raise RuntimeError(
                                f"apply_stimulus_from_h5: unable to determine spike train attribute for population {pop_name} in spike input file {filepath}; "
                                f"namespace {namespace}; attr keys {list(cell_spikes_attr_info.keys())}"
                            )
                        else:
                            continue

                        for gid, cell_spikes_tuple in cell_spikes_iter:
                            if not self.pc.gid_exists(gid):
                                continue
                            if gid not in self.artificial_cells[pop_name]:
                                logger.warning(
                                    f"apply_stimulus_from_h5: Rank {rank}: gid {gid} not in artificial_cells[{pop_name}]"
                                )
                                continue

                            input_cell = self.artificial_cells[pop_name][gid]

                            spiketrain = cell_spikes_tuple[spike_train_attr_index]
                            trial_index = None
                            trial_duration = None
                            if trial_index_attr_index is not None:
                                trial_index = cell_spikes_tuple[trial_index_attr_index]
                                trial_duration = cell_spikes_tuple[trial_dur_attr_index]

                            if len(spiketrain) > 0:
                                spiketrain = self._merge_spiketrain_trials(
                                    spiketrain,
                                    trial_index,
                                    trial_duration,
                                    n_trials,
                                )
                                spiketrain += equilibration_duration + onset

                                if len(spiketrain) > 0:
                                    input_cell.play(
                                        h.Vector(spiketrain.astype(np.float64))
                                    )
                                    if rank == 0:
                                        logger.info(
                                            f"*** Spike train for {pop_name} gid {gid} is of length {len(spiketrain)} ({spiketrain[0]} : {spiketrain[-1]} ms)"
                                        )
                else:
                    if rank == 0 and len(this_gid_range) > 0:
                        logger.warning(
                            f"No spike train data found for population {pop_name} in spike input file {filepath}; "
                            f"namespace: {namespace}"
                        )

        gc.collect()

        return self

    @staticmethod
    def _merge_spiketrain_trials(
        spiketrain: np.ndarray,
        trial_index: np.ndarray,
        trial_duration: np.ndarray,
        n_trials: int,
    ) -> np.ndarray:
        if (trial_index is not None) and (trial_duration is not None):
            trial_spiketrains = []
            for trial_i in range(n_trials):
                trial_spiketrain_i = spiketrain[np.where(trial_index == trial_i)[0]]
                trial_spiketrain_i += np.sum(trial_duration[:trial_i])
                trial_spiketrains.append(trial_spiketrain_i)
            spiketrain = np.concatenate(trial_spiketrains)
        spiketrain.sort()
        return spiketrain

    def legacy(self, **kwargs):
        class _binding:
            pass

        this = _binding()

        attrs = {
            "_wrapped": self,
            "pc": self.pc,
            "comm": self.comm,
            "node_allocation": self.node_allocation,
            "cells": self._registry,
            "artificial_cells": self.artificial_cells,
            "biophys_cells": self.biophys_cells,
            "recording_sets": self.recording_sets,
            "t_vec": self.t_vec,
            "id_vec": self.id_vec,
            "t_rec": self.t_rec,
            "gidset": self.gidset,
            "spike_onset_delay": self.spike_onset_delay,
            "template_dict": self.template_dict,
            "SWC_Types": config.SWCTypesDef.__members__,
            "template_paths": [self.template_directory],
            "model_config": {
                "Random Seeds": {"Intracellular Recording Sample": self.seed}
            },
            "coordinates_ns": "Generated Coordinates",
            "gapjunctions_file_path": None,
            "gapjunctions": None,
            "recording_profile": None,
            "datasetName": "",
            "dataset_path": None,
            "dataset_prefix": "",
            "edge_count": self.edge_count,
            "microcircuit_input_sources": self.input_sources,
            "synapse_manager": self.synapse_manager,
            "phenotype_dict": {},
            "max_walltime_hours": 0.5,
            "results_write_time": 0,
            "cache_queries": False,
            "globals": {},
            "optlptbal": None,
            "optldbal": None,
            "profile_memory": False,
            "LFP_config": {},
            "simtime": None,
            "use_coreneuron": False,
            "v_init": -75.0,
            "checkpoint_interval": None,
            "nrn_timeout": 600.0,
            "mkcellstime": self.mkcellstime,
            "connectgjstime": self.connectgjstime,
            "connectcellstime": self.connectcellstime,
            "psolvetime": self.psolvetime,
            "stimulus_config": {
                "Equilibration Duration": 250.0,
                "Temporal Resolution": 50.0,
            },
            "stimulus_onset": 0,
            "spike_input_namespaces": [],
            "spike_input_path": None,
            "n_trials": 1,
            "spike_input_attr": None,
            "io_size": 1,
            "dt": 0.025,
            "spike_input_attribute_info": None,
            "cell_selection": None,
            "analysis_config": {
                "Firing Rate Inference": {
                    "BAKS Beta": 0.4196905892734352,
                    "Temporal Resolution": 10.0,
                    "Pad Duration": 1000.0,
                    "BAKS Alpha": 4.7725100028345535,
                },
                "Mutual Information": {"Spatial Resolution": 5.0},
                "Place Fields": {"Minimum Width": 10.0, "Minimum Rate": 1.0},
            },
            "netclamp_config": None,
            "use_cell_attr_gen": True,
            "cell_attr_gen_cache_size": 4,
            "cleanup": False,
        }

        if self.system:
            if hasattr(self.system, "files"):
                attrs["data_file_path"] = self.system.files.get("cells")
                attrs["connectivity_file_path"] = self.system.files.get("connections")
                attrs["forest_file_path"] = self.system.files.get("cells")

            if hasattr(self.system, "cells_meta_data"):
                attrs["cell_attribute_info"] = (
                    self.system.cells_meta_data.cell_attribute_info
                )

            if hasattr(self.system, "connections_config"):
                attrs["Populations"] = self.system.connections_config.get(
                    "population_definitions"
                )
                attrs["layers"] = self.system.connections_config.get(
                    "layer_definitions"
                )
                if "synapses" in self.system.connections_config:
                    attrs["connection_config"] = DotDict.create(
                        self.system.connections_config["synapses"]
                    )

        if self.cells_meta_data:
            attrs["celltypes"] = self.cells_meta_data.get("celltypes")

        if "microcircuit_inputs" not in kwargs:
            microcircuit_inputs = False
            if hasattr(self.model, "neuron_microcircuit_inputs"):
                microcircuit_inputs = self.model.neuron_microcircuit_inputs()
            attrs["microcircuit_inputs"] = microcircuit_inputs

        if "projection_dict" not in kwargs:
            attrs["projection_dict"] = None

        if "connection_velocity" not in kwargs:
            attrs["connection_velocity"] = defaultdict(lambda: 250)

        attrs.update(kwargs)

        this.__dict__.update(attrs)

        return this

    def _release_cell_container(self, container):
        if not container:
            return
        for pop_name, gid_map in list(container.items()):
            for gid, cell in list(gid_map.items()):
                try:
                    if hasattr(cell, "spike_detector"):
                        cell.spike_detector = None
                    if hasattr(cell, "hoc_cell"):
                        cell.hoc_cell = None
                    if hasattr(cell, "cell_obj"):
                        cell.cell_obj = None
                    if hasattr(cell, "sections"):
                        cell.sections = None
                except Exception:
                    logger.debug(
                        "Failed to release NEURON references for %s gid %s",
                        pop_name,
                        gid,
                        exc_info=True,
                    )
            gid_map.clear()
        container.clear()

    def close(self):
        if getattr(self, "_closed", False):
            return

        self._closed = True

        syn_manager = getattr(self, "synapse_manager", None)
        if syn_manager is not None and hasattr(syn_manager, "clear"):
            try:
                syn_manager.clear()
            except Exception:
                logger.debug("Failed to clear synapse manager", exc_info=True)
        self.synapse_manager = None

        self._release_cell_container(getattr(self, "cells", None))
        self._release_cell_container(getattr(self, "biophys_cells", None))
        self._release_cell_container(getattr(self, "artificial_cells", None))

        for attr_name in (
            "syns_set",
            "edge_count",
            "recording_sets",
            "v_recs",
            "v_recs_dt",
            "i_recs",
            "i_recs_dt",
            "i_area",
        ):
            mapping = getattr(self, attr_name, None)
            if mapping is not None:
                mapping.clear()

        stim_vectors = getattr(self, "_stimulus_vectors", None)
        if stim_vectors is not None:
            stim_vectors.clear()

        spike_filter_refs = getattr(self, "_spike_filter_refs", None)
        if spike_filter_refs is not None:
            spike_filter_refs.clear()

        gidset = getattr(self, "gidset", None)
        if gidset is not None:
            gidset.clear()
        flucts = getattr(self, "_flucts", None)
        if flucts is not None:
            flucts.clear()
        template_dict = getattr(self, "template_dict", None)
        if template_dict is not None:
            template_dict.clear()

        for vec_name in ("t_vec", "id_vec", "t_rec"):
            vec = getattr(self, vec_name, None)
            if vec is not None:
                try:
                    vec.resize(0)
                except Exception:
                    logger.debug("Failed to resize %s", vec_name, exc_info=True)

        if hasattr(self, "cells_meta_data"):
            self.cells_meta_data = None
        if hasattr(self, "connections_meta_data"):
            self.connections_meta_data = None
        if hasattr(self, "input_sources"):
            self.input_sources = None
        if hasattr(self, "this"):
            self.this = None
        if hasattr(self, "node_allocation"):
            self.node_allocation = None

        pc = getattr(self, "pc", None)
        if pc is not None:
            try:
                pc.gid_clear()
            except Exception:
                logger.debug("ParallelContext gid_clear failure", exc_info=True)
        self.pc = None

        try:
            secs = list(h.allsec())
        except Exception:
            secs = []
        for sec in reversed(secs):
            try:
                h.delete_section(sec=sec)
            except Exception:
                sec_name = "<unknown>"
                try:
                    sec_name = sec.hname()
                except Exception:
                    pass
                logger.debug(
                    "Unable to delete section %s during teardown",
                    sec_name,
                    exc_info=True,
                )

        gc.collect()

        return self

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
