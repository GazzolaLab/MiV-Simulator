from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import copy
import itertools
import math
import sys
import time
import traceback
import uuid
from collections import defaultdict, namedtuple
from functools import reduce

import numpy as np
from miv_simulator.cells import (
    BiophysCell,
    make_section_graph,
)
from miv_simulator.utils import (
    AbstractEnv,
    ExprClosure,
    NamedTupleWithDocstring,
    Promise,
    generator_ifempty,
    get_module_logger,
)
from miv_simulator.utils.neuron import (
    default_ordered_sec_types,
    interplocs,
    mknetcon,
    mknetcon_vecstim,
)
from neuroh5.io import write_cell_attributes
from neuron import h
from nrn import Section, Segment
from numpy import float64, ndarray, uint32

if TYPE_CHECKING:
    from neuron.hoc import HocObject

# This logger will inherit its settings from the root logger, created in env
logger = get_module_logger(__name__)
if hasattr(h, "nrnmpi_init"):
    h.nrnmpi_init()

# Array dtype for core synapse properties
SYNAPSE_CORE_DTYPE = np.dtype(
    [
        ("syn_id", np.uint32),  # Synapse ID (for indexing)
        ("syn_type", np.uint8),  # Synapse type
        ("swc_type", np.uint8),  # SWC type
        ("syn_layer", np.int8),  # Layer
        ("syn_loc", np.float32),  # Location in section
        ("syn_section", np.uint32),  # Section index
        ("source_gid", np.int32),  # Source cell GID
        ("source_population", np.int8),  # Source population index
        ("delay", np.float32),  # Connection delay
    ]
)


SynParam = namedtuple(
    "SynParam",
    (
        "population",
        "source",
        "sec_type",
        "syn_name",
        "param_path",
        "param_range",
        "phenotype",
    ),
    defaults=(None, None, None, None, None, None, None),
)


def syn_param_from_dict(d):
    return SynParam(*[d[key] for key in SynParam._fields])


def get_mech_rules_dict(cell, **rules):
    """
    Used by modify_syn_param. Takes in a series of arguments and
    constructs a validated rules dictionary that will be used to
    update a cell's mechanism dictionary.

    :param cell: :class:'BiophysCell'
    :param rules: dict
    :return: dict

    """
    rules_dict = {
        name: rules[name]
        for name in (
            name
            for name in ["value", "origin"]
            if name in rules and rules[name] is not None
        )
    }
    if "origin" in rules_dict:
        origin_type = rules_dict["origin"]
        valid_sec_types = [
            sec_type for sec_type in cell.nodes if len(cell.nodes[sec_type]) > 0
        ]
        if origin_type not in valid_sec_types + ["parent", "branch_origin"]:
            raise ValueError(
                f"get_mech_rules_dict: cannot inherit from invalid origin type: {origin_type}"
            )
    return rules_dict


def get_node_attribute(name, content, sec, secnodes, x=None):
    """

    :param name:
    :param content:
    :param sec:
    :param secnodes:
    :param x:
    :return:
    """
    if name in content:
        if x is None:
            return content[name]
        elif sec.n3d() == 0:
            return content[name][0]
        else:
            prev = None
            for i in range(sec.n3d()):
                pos = sec.arc3d(i) / sec.L
                if pos >= x:
                    if (prev is None) or (abs(pos - x) < abs(prev - x)):
                        return content[name][secnodes[i]]
                    else:
                        return content[name][secnodes[i - 1]]
                else:
                    prev = pos
    else:
        return None


def synapse_seg_density(
    syn_type_dict,
    layer_dict,
    layer_density_dicts,
    seg_dict,
    ran,
    neurotree_dict=None,
):
    """
    Computes per-segment density of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param seg_dict:
    :param ran:
    :param neurotree_dict:
    :return:
    """
    segdensity_dict = {}
    layers_dict = {}

    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict["section_topology"]["nodes"]
    else:
        secnodes_dict = None
    for syn_type_label, layer_density_dict in layer_density_dicts.items():
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for layer_label, density_dict in layer_density_dict.items():
            if layer_label == "default" or layer_label == -1:
                layer = "default"
            else:
                layer = int(layer_dict[layer_label])
            rans[layer] = ran
        segdensity = defaultdict(list)
        layers = defaultdict(list)
        total_seg_density = 0.0
        for sec_index, seg_list in seg_dict.items():
            for seg in seg_list:
                if neurotree_dict is not None:
                    secnodes = secnodes_dict[sec_index]
                    layer = get_node_attribute(
                        "layer", neurotree_dict, seg.sec, secnodes, seg.x
                    )
                else:
                    layer = -1
                layers[sec_index].append(layer)

                this_ran = None

                if layer > -1:
                    if layer in rans:
                        this_ran = rans[layer]
                    elif "default" in rans:
                        this_ran = rans["default"]
                    else:
                        this_ran = None
                elif "default" in rans:
                    this_ran = rans["default"]
                else:
                    this_ran = None
                if this_ran is not None:
                    while True:
                        dens = this_ran.normal(
                            density_dict["mean"], density_dict["variance"]
                        )
                        if dens > 0.0:
                            break
                else:
                    dens = 0.0
                total_seg_density += dens
                segdensity[sec_index].append(dens)

        if total_seg_density < 1e-6:
            logger.warning(
                f"sections with zero {syn_type_label} "
                f"synapse density: {segdensity}; rans: {rans}; "
                f"density_dict: {density_dict}; layers: {layers} "
                f"morphology: {neurotree_dict}"
            )

        segdensity_dict[syn_type] = segdensity

        layers_dict[syn_type] = layers
    return (segdensity_dict, layers_dict)


def synapse_seg_counts(
    syn_type_dict,
    layer_dict,
    layer_density_dicts,
    sec_index_dict,
    seg_dict,
    ran,
    neurotree_dict=None,
):
    """
    Computes per-segment relative counts of synapse placement.
    :param syn_type_dict:
    :param layer_dict:
    :param layer_density_dicts:
    :param sec_index_dict:
    :param seg_dict:
    :param seed:
    :param neurotree_dict:
    :return:
    """
    segcounts_dict = {}
    layers_dict = {}
    segcount_total = 0
    if neurotree_dict is not None:
        secnodes_dict = neurotree_dict["section_topology"]["nodes"]
    else:
        secnodes_dict = None
    for syn_type_label, layer_density_dict in layer_density_dicts.items():
        syn_type = syn_type_dict[syn_type_label]
        rans = {}
        for layer_label, density_dict in layer_density_dict.items():
            if layer_label == "default" or layer_label == -1:
                layer = "default"
            else:
                layer = layer_dict[layer_label]

            rans[layer] = ran
        segcounts = []
        layers = []
        for sec_index, seg_list in seg_dict.items():
            for seg in seg_list:
                L = seg.sec.L
                nseg = seg.sec.nseg
                if neurotree_dict is not None:
                    secnodes = secnodes_dict[sec_index]
                    layer = get_node_attribute(
                        "layer", neurotree_dict, seg.sec, secnodes, seg.x
                    )
                else:
                    layer = -1
                layers.append(layer)

                ran = None

                if layer > -1:
                    if layer in rans:
                        ran = rans[layer]
                    elif "default" in rans:
                        ran = rans["default"]
                    else:
                        ran = None
                elif "default" in rans:
                    ran = rans["default"]
                else:
                    ran = None
                if ran is not None:
                    pos = L / nseg
                    dens = ran.normal(density_dict["mean"], density_dict["variance"])
                    rc = dens * pos
                    segcount_total += rc
                    segcounts.append(rc)
                else:
                    segcounts.append(0)

            segcounts_dict[syn_type] = segcounts
            layers_dict[syn_type] = layers
    return (segcounts_dict, segcount_total, layers_dict)


def distribute_uniform_synapses(
    density_seed,
    syn_type_dict,
    swc_type_dict,
    layer_dict,
    sec_layer_density_dict,
    neurotree_dict,
    cell_sec_dict,
    cell_secidx_dict,
):
    """
    Computes uniformly-spaced synapse locations.
    :param density_seed:
    :param syn_type_dict:
    :param swc_type_dict:
    :param layer_dict:
    :param sec_layer_density_dict:
    :param neurotree_dict:
    :param sec_dict:
    :param secidx_dict:
    :return:
    """
    syn_ids = []
    syn_locs = []
    syn_cdists = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    r = np.random.RandomState()

    sec_interp_loc_dict = {}
    segcounts_per_sec = {}
    for sec_name, layer_density_dict in sec_layer_density_dict.items():
        sec_index_dict = cell_secidx_dict[sec_name]
        swc_type = swc_type_dict[sec_name]
        seg_list = []
        L_total = 0
        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        for sec, idx in zip(seclst, secidxlst):
            sec_interp_loc_dict[idx] = interplocs(sec)
        sec_dict = {int(idx): sec for sec, idx in zip(seclst, secidxlst)}
        seg_dict = {}
        for sec_index, sec in sec_dict.items():
            seg_list = []
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if (
                        seg.x < 1.0
                        and seg.x > 0.0
                        and ((L_total + sec.L * seg.x) <= maxdist)
                    ):
                        seg_list.append(seg)
            L_total += sec.L
            seg_dict[sec_index] = seg_list
        segcounts_dict, total, layers_dict = synapse_seg_counts(
            syn_type_dict,
            layer_dict,
            layer_density_dict,
            sec_index_dict=sec_index_dict,
            seg_dict=seg_dict,
            ran=r,
            neurotree_dict=neurotree_dict,
        )
        segcounts_per_sec[sec_name] = segcounts_dict
        for syn_type_label, _ in layer_density_dict.items():
            syn_type = syn_type_dict[syn_type_label]
            segcounts = segcounts_dict[syn_type]
            layers = layers_dict[syn_type]
            for sec_index, seg_list in seg_dict.items():
                interp_loc = sec_interp_loc_dict[sec_index]
                for seg, layer, seg_count in zip(seg_list, layers, segcounts):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end = seg.x + (0.5 / seg.sec.nseg)
                    seg_range = seg_end - seg_start
                    int_seg_count = math.floor(seg_count)
                    syn_count = 0
                    while syn_count < int_seg_count:
                        syn_loc = seg_start + seg_range * (syn_count + 1) / math.ceil(
                            seg_count
                        )
                        assert (syn_loc <= 1) & (syn_loc >= 0)
                        if syn_loc < 1.0:
                            syn_cdist = math.sqrt(
                                reduce(
                                    lambda a, b: a + b,
                                    (interp_loc[i](syn_loc) ** 2 for i in range(3)),
                                )
                            )
                            syn_cdists.append(syn_cdist)
                            syn_locs.append(syn_loc)
                            syn_ids.append(syn_index)
                            syn_secs.append(sec_index_dict[seg.sec])
                            syn_layers.append(layer)
                            syn_types.append(syn_type)
                            swc_types.append(swc_type)
                            syn_index += 1
                            syn_count += 1

    assert len(syn_ids) > 0
    syn_dict = {
        "syn_ids": np.asarray(syn_ids, dtype="uint32"),
        "syn_cdists": np.asarray(syn_cdists, dtype="float32"),
        "syn_locs": np.asarray(syn_locs, dtype="float32"),
        "syn_secs": np.asarray(syn_secs, dtype="uint32"),
        "syn_layers": np.asarray(syn_layers, dtype="int8"),
        "syn_types": np.asarray(syn_types, dtype="uint8"),
        "swc_types": np.asarray(swc_types, dtype="uint8"),
    }

    return (syn_dict, segcounts_per_sec)


def distribute_poisson_synapses(
    density_seed,
    syn_type_dict,
    swc_type_dict,
    layer_dict,
    sec_layer_density_dict,
    neurotree_dict,
    cell_sec_dict,
    cell_secidx_dict,
):
    """
    Computes synapse locations distributed according to a Poisson distribution.
    :param density_seed:
    :param syn_type_dict:
    :param swc_type_dict:
    :param layer_dict:
    :param sec_layer_density_dict:
    :param neurotree_dict:
    :param cell_sec_dict:
    :param cell_secidx_dict:
    :param verbose:
    :return:
    """
    import networkx as nx

    syn_ids = []
    syn_cdists = []
    syn_locs = []
    syn_secs = []
    syn_layers = []
    syn_types = []
    swc_types = []
    syn_index = 0

    sec_graph = make_section_graph(neurotree_dict)

    debug_flag = False
    secnodes_dict = neurotree_dict["section_topology"]["nodes"]
    for sec, secnodes in secnodes_dict.items():
        if len(secnodes) < 2:
            debug_flag = True

    if debug_flag:
        logger.debug(f"sec_graph: {str(list(sec_graph.edges))}")
        logger.debug(f"neurotree_dict: {str(neurotree_dict)}")

    sec_interp_loc_dict = {}
    seg_density_per_sec = {}
    r = np.random.RandomState()
    r.seed(int(density_seed))
    for sec_name, layer_density_dict in sec_layer_density_dict.items():
        swc_type = swc_type_dict[sec_name]
        seg_dict = {}
        L_total = 0

        (seclst, maxdist) = cell_sec_dict[sec_name]
        secidxlst = cell_secidx_dict[sec_name]
        for sec, idx in zip(seclst, secidxlst):
            sec_interp_loc_dict[idx] = interplocs(sec)
        sec_dict = {int(idx): sec for sec, idx in zip(seclst, secidxlst)}
        if len(sec_dict) > 1:
            sec_subgraph = sec_graph.subgraph(list(sec_dict.keys()))
            if len(sec_subgraph.edges()) > 0:
                sec_roots = [n for n, d in sec_subgraph.in_degree() if d == 0]
                sec_edges = []
                for sec_root in sec_roots:
                    sec_edges.append(list(nx.dfs_edges(sec_subgraph, sec_root)))
                    sec_edges.append([(None, sec_root)])
                sec_edges = [val for sublist in sec_edges for val in sublist]
            else:
                sec_edges = [(None, idx) for idx in list(sec_dict.keys())]
        else:
            sec_edges = [(None, idx) for idx in list(sec_dict.keys())]
        for sec_index, sec in sec_dict.items():
            seg_list = []
            if maxdist is None:
                for seg in sec:
                    if seg.x < 1.0 and seg.x > 0.0:
                        seg_list.append(seg)
            else:
                for seg in sec:
                    if (
                        seg.x < 1.0
                        and seg.x > 0.0
                        and ((L_total + sec.L * seg.x) <= maxdist)
                    ):
                        seg_list.append(seg)
            seg_dict[sec_index] = seg_list
            L_total += sec.L
        seg_density_dict, layers_dict = synapse_seg_density(
            syn_type_dict,
            layer_dict,
            layer_density_dict,
            seg_dict,
            r,
            neurotree_dict=neurotree_dict,
        )
        seg_density_per_sec[sec_name] = seg_density_dict
        for syn_type_label, _ in layer_density_dict.items():
            syn_type = syn_type_dict[syn_type_label]
            seg_density = seg_density_dict[syn_type]
            layers = layers_dict[syn_type]
            end_distance = {}
            for sec_parent, sec_index in sec_edges:
                interp_loc = sec_interp_loc_dict[sec_index]
                seg_list = seg_dict[sec_index]
                sec_seg_layers = layers[sec_index]
                sec_seg_density = seg_density[sec_index]
                interval = 0.0
                syn_loc = 0.0
                for seg, layer, density in zip(
                    seg_list, sec_seg_layers, sec_seg_density
                ):
                    seg_start = seg.x - (0.5 / seg.sec.nseg)
                    seg_end = seg.x + (0.5 / seg.sec.nseg)
                    L = seg.sec.L
                    L_seg_start = seg_start * L
                    L_seg_end = seg_end * L
                    if density > 0.0:
                        beta = 1.0 / density
                        if interval > 0.0:
                            sample = r.exponential(beta)
                        else:
                            while True:
                                sample = r.exponential(beta)
                                if (sample >= L_seg_start) and (sample < L_seg_end):
                                    break
                        interval += sample
                        while interval < L_seg_end:
                            if interval >= L_seg_start:
                                syn_loc = interval / L
                                assert (syn_loc <= 1) and (syn_loc >= seg_start)
                                if syn_loc < 1.0:
                                    syn_cdist = math.sqrt(
                                        reduce(
                                            lambda a, b: a + b,
                                            (
                                                interp_loc[i](syn_loc) ** 2
                                                for i in range(3)
                                            ),
                                        )
                                    )
                                    syn_cdists.append(syn_cdist)
                                    syn_locs.append(syn_loc)
                                    syn_ids.append(syn_index)
                                    syn_secs.append(sec_index)
                                    syn_layers.append(layer)
                                    syn_types.append(syn_type)
                                    swc_types.append(swc_type)
                                    syn_index += 1
                            interval += r.exponential(beta)
                    else:
                        interval = seg_end * L
                end_distance[sec_index] = (1.0 - syn_loc) * L

    assert len(syn_ids) > 0
    syn_dict = {
        "syn_ids": np.asarray(syn_ids, dtype="uint32"),
        "syn_cdists": np.asarray(syn_cdists, dtype="float32"),
        "syn_locs": np.asarray(syn_locs, dtype="float32"),
        "syn_secs": np.asarray(syn_secs, dtype="uint32"),
        "syn_layers": np.asarray(syn_layers, dtype="int8"),
        "syn_types": np.asarray(syn_types, dtype="uint8"),
        "swc_types": np.asarray(swc_types, dtype="uint8"),
    }

    return (syn_dict, seg_density_per_sec)


SynapsePointProcess = NamedTupleWithDocstring(
    """This class provides information about the point processes associated with a synapse.
      - mech - dictionary of synapse mechanisms
      - netcon - dictionary of netcons
      - vecstim - dictionary of vecstims
    """,
    "SynapsePointProcess",
    ["mech", "netcon", "vecstim"],
)


class SynapseMechanismParameterStore:
    """Storage for synapse mechanism parameters with selective defaults"""

    def __init__(self, mech_param_specs):
        """
        Initialize with mechanism parameter specifications

        Args:
            mech_param_specs: Dict mapping mechanism names to parameter specifications
        """
        self.mech_param_specs = mech_param_specs
        self.mech_param_index = {name: idx for idx, name in enumerate(mech_param_specs)}

        # Storage for per-mechanism parameters (will be created on demand)
        self.gid_data = {}

        # Keep track of selector size -- to be used as hint in setting parameter array sizes
        self.selector_sizes = {}

        # Track which parameters have arrays allocated
        self.allocated_params = defaultdict(lambda: defaultdict(set))

        # Default parameter values with selective application
        # Format: {mech_name: {param_name: [(synapse_selector, value), ...]}
        self.default_values = defaultdict(lambda: defaultdict(list))

        # Storage for complex objects that can't be stored in arrays
        self.complex_params = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

    def _ensure_gid_storage(self, gid, synapse_count=0):
        """Create basic data structures for a gid if not already present"""
        if gid not in self.gid_data:
            self.gid_data[gid] = {
                "arrays": {},  # Will hold parameter arrays
                "has_mech": {},  # Will track which synapses have each mechanism
                "synapse_count": synapse_count,  # Track active synapse count
            }

    def _ensure_mech_storage(self, gid, mech_name):
        """Create storage for a mechanism if not already present"""
        self._ensure_gid_storage(gid)

        if mech_name not in self.gid_data[gid]["has_mech"]:
            synapse_count = self.gid_data[gid]["synapse_count"]
            selector_size = self.selector_sizes.get(gid, 1000)
            # Create array to track which synapses have this mechanism
            self.gid_data[gid]["has_mech"][mech_name] = np.zeros(
                max(synapse_count, selector_size), dtype=bool
            )
            # Initialize empty dict for parameter arrays
            self.gid_data[gid]["arrays"][mech_name] = {}

    def _ensure_param_array(self, gid, mech_name, param_name):
        """Create parameter array if not already created"""
        self._ensure_mech_storage(gid, mech_name)

        # Check if already allocated
        if param_name in self.gid_data[gid]["arrays"][mech_name]:
            return

        # Get array size based on synapse count
        synapse_count = self.gid_data[gid]["synapse_count"]
        selector_size = self.selector_sizes.get(gid, 1000)
        array_size = max(synapse_count, selector_size)  # Minimum size for efficiency

        # Create array filled with NaN to indicate "use default value"
        self.gid_data[gid]["arrays"][mech_name][param_name] = np.full(
            array_size, np.nan, dtype=np.float32
        )

        # Mark parameter as allocated
        self.allocated_params[gid][mech_name].add(param_name)

    def _resize_arrays(self, gid, new_size):
        """Resize all arrays for a gid to accommodate more synapses"""
        if gid not in self.gid_data:
            return

        # Update synapse count
        self.gid_data[gid]["synapse_count"] = new_size

        # Resize all mechanism arrays
        for mech_name in self.gid_data[gid]["has_mech"]:
            # Resize has_mech array
            old_has_mech = self.gid_data[gid]["has_mech"][mech_name]
            if new_size > len(old_has_mech):
                new_has_mech = np.zeros(new_size, dtype=bool)
                new_has_mech[: len(old_has_mech)] = old_has_mech
                self.gid_data[gid]["has_mech"][mech_name] = new_has_mech

            # Resize each parameter array
            for param_name in self.gid_data[gid]["arrays"][mech_name]:
                old_array = self.gid_data[gid]["arrays"][mech_name][param_name]
                if new_size > len(old_array):
                    new_array = np.full(new_size, np.nan, dtype=np.float32)
                    new_array[: len(old_array)] = old_array
                    self.gid_data[gid]["arrays"][mech_name][param_name] = new_array

    def _create_selector(self, gid, selector_spec):
        """
        Create a selector function from the given specification

        Returns a function that takes (syn_id, syn_index) and returns True/False
        """
        if selector_spec is None:
            # Global default - applies to all synapses
            return lambda syn_id, idx: True

        elif isinstance(selector_spec, (list, tuple, set, np.ndarray)):
            # Set of specific synapse IDs
            syn_id_set = set(selector_spec)
            if gid in self.selector_sizes:
                selector_size = self.selector_sizes[gid]
                self.selector_sizes[gid] = max(len(syn_id_set), selector_size)
            else:
                self.selector_sizes[gid] = len(selector_spec)
            return lambda syn_id, idx: syn_id in syn_id_set

        elif callable(selector_spec):
            # Custom function
            return lambda syn_id, idx: selector_spec(syn_id)

        elif isinstance(selector_spec, dict):
            # Filter criteria - similar to filter_synapses method
            # This needs access to synapse properties
            props = self.parent_storage.properties[gid]
            if gid in self.selector_sizes:
                selector_size = self.selector_sizes[gid]
                self.selector_sizes[gid] = max(len(selector_spec), selector_size)
            else:
                self.selector_sizes[gid] = len(selector_spec)

            # Create mask based on filter criteria
            def filter_selector(syn_id, syn_index):
                # Apply each filter criterion
                if (
                    "syn_types" in selector_spec
                    and props["syn_type"][syn_index] not in selector_spec["syn_types"]
                ):
                    return False
                if (
                    "layers" in selector_spec
                    and props["syn_layer"][syn_index] not in selector_spec["layers"]
                ):
                    return False
                if (
                    "sources" in selector_spec
                    and props["source_population"][syn_index]
                    not in selector_spec["sources"]
                ):
                    return False
                if (
                    "swc_types" in selector_spec
                    and props["swc_type"][syn_index] not in selector_spec["swc_types"]
                ):
                    return False
                return True

            return filter_selector

        else:
            # Default to global selector if type is unrecognized
            return lambda syn_id, idx: True

    def set_default_value(
        self, gid, mech_name, param_name, value, synapse_selector=None
    ):
        """
        Set default value for a parameter that applies to a subset of synapses

        Args:
            gid: Cell GID
            mech_name: Mechanism name
            param_name: Parameter name
            value: Parameter value
            synapse_selector: Selection criteria for which synapses this default applies to:
                - None: Global default for all synapses
                - tuple/list of synapse IDs: Applies to specific synapse IDs
                - callable: Function that takes synapse ID and returns True/False
                - dict: Filter criteria like {'syn_types': [...], 'layers': [...]}
        """
        if mech_name not in self.mech_param_specs:
            return False

        if param_name not in self.mech_param_specs[mech_name]:
            return False

        # Create selector object based on input
        selector = self._create_selector(gid, synapse_selector)

        # Store complex objects with a type marker
        if not isinstance(value, (int, float)) or np.isnan(value):
            value = ("complex", value)

        # Add default to the list for this parameter
        self.default_values[gid][mech_name][param_name].append((selector, value))

        return True

    def get_default_value(self, gid, mech_name, param_name, syn_id, syn_index):
        """
        Get default value for a parameter, considering selective defaults

        Returns the value from the most specific matching default, or None if no default applies
        """
        if gid not in self.default_values or mech_name not in self.default_values[gid]:
            return None

        if param_name not in self.default_values[gid][mech_name]:
            return None

        # Get all defaults for this parameter
        defaults = self.default_values[gid][mech_name][param_name]

        # Find the first matching default (most recently added takes precedence)
        for selector, value in reversed(defaults):
            if selector(syn_id, syn_index):
                # Check if complex type
                if isinstance(value, tuple) and value[0] == "complex":
                    return value[1]
                return value

        return None

    def has_default_params(self, gid, syn_id, syn_index, mech_name):
        """
        Check if a synapse has default parameters for a specific mechanism

        Args:
        gid: Cell GID
        syn_id: Synapse ID
        syn_index: Synapse array index
        mech_name: Mechanism name

        Returns:
        bool: Whether the synapse has any default parameters for this mechanism
        """
        # Check if there are any defaults for this gid and mechanism
        if gid not in self.default_values or mech_name not in self.default_values[gid]:
            return False

        # Check each parameter to see if any defaults apply to this synapse
        for param_name, defaults in self.default_values[gid][mech_name].items():
            for selector, value in defaults:
                if selector(syn_id, syn_index):
                    return True

        # No defaults apply to this synapse
        return False

    def get_default_params(self, gid, syn_id, syn_index, mech_name):
        """
        Get the names of parameters that have defaults for a specific synapse and mechanism

        Args:
        gid: Cell GID
        syn_id: Synapse ID
        syn_index: Synapse array index
        mech_name: Mechanism name

        Returns:
        set: Set of parameter names that have defaults for this synapse
        """
        result = set()

        # Check if there are any defaults for this gid and mechanism
        if gid not in self.default_values or mech_name not in self.default_values[gid]:
            return result

        # Find all parameters with defaults that apply to this synapse
        for param_name, defaults in self.default_values[gid][mech_name].items():
            for selector, value in defaults:
                if selector(syn_id, syn_index):
                    result.add(param_name)
                    break  # Only need to find one matching default per parameter

        return result

    def set_synapse_parameter(
        self, gid, syn_index, mech_name, param_name, value, syn_id=None
    ):
        """Set parameter value for a specific synapse"""
        if mech_name not in self.mech_param_specs:
            return False

        if param_name not in self.mech_param_specs[mech_name]:
            return False

        # Ensure mechanism storage exists
        self._ensure_mech_storage(gid, mech_name)

        # Resize arrays if necessary
        if syn_index >= self.gid_data[gid]["synapse_count"]:
            new_size = max(
                syn_index + 1, int(self.gid_data[gid]["synapse_count"] * 1.1)
            )
            self._resize_arrays(gid, new_size)

        # Mark synapse as having this mechanism
        self.gid_data[gid]["has_mech"][mech_name][syn_index] = True

        # Handle different value types
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            # For numeric values, create/use array storage
            self._ensure_param_array(gid, mech_name, param_name)
            self.gid_data[gid]["arrays"][mech_name][param_name][syn_index] = value

            # Remove any complex parameter if it exists
            if (
                syn_index in self.complex_params[gid][mech_name]
                and param_name in self.complex_params[gid][mech_name][syn_index]
            ):
                del self.complex_params[gid][mech_name][syn_index][param_name]
        else:
            # For complex values, store in separate structure
            self.complex_params[gid][mech_name][syn_index][param_name] = value

            # If parameter array exists, set special value
            if param_name in self.gid_data[gid]["arrays"][mech_name]:
                self.gid_data[gid]["arrays"][mech_name][param_name][syn_index] = -np.inf

        return True

    def get_synapse_parameter(self, gid, syn_index, mech_name, param_name, syn_id=None):
        """Get parameter value for a specific synapse with proper fallback to defaults"""
        # Use parameter value hierarchy to get value with fallback
        value, source = self.get_parameter_value_hierarchy(
            gid, syn_index, mech_name, param_name, syn_id
        )
        return value

    def get_parameter_value_hierarchy(
        self, gid, syn_index, mech_name, param_name, syn_id=None
    ):
        """
        Get parameter value with complete hierarchy:
        1. Specifically set value for this synapse
        2. Default value that applies to this synapse
        3. None if not found

        Args:
            gid: Cell GID
            syn_index: Synapse array index
            mech_name: Mechanism name
            param_name: Parameter name
            syn_id: Synapse ID for default lookup

        Returns:
            Tuple of (value, source) where source is 'specific', 'default', or None
        """
        # Check for specific value first
        if gid in self.gid_data and mech_name in self.gid_data[gid]["arrays"]:
            # Check if synapse has this mechanism
            if (
                syn_index < len(self.gid_data[gid]["has_mech"][mech_name])
                and self.gid_data[gid]["has_mech"][mech_name][syn_index]
            ):
                # Check for complex parameter
                if (
                    syn_index in self.complex_params[gid][mech_name]
                    and param_name in self.complex_params[gid][mech_name][syn_index]
                ):
                    return (
                        self.complex_params[gid][mech_name][syn_index][param_name],
                        "specific",
                    )

                # Check for array parameter
                if param_name in self.gid_data[gid]["arrays"][mech_name]:
                    value = self.gid_data[gid]["arrays"][mech_name][param_name][
                        syn_index
                    ]
                    if not np.isnan(value) and value != -np.inf:
                        return (value, "specific")

        # Try to get default value
        default_value = self.get_default_value(
            gid, mech_name, param_name, syn_id, syn_index
        )
        if default_value is not None:
            return (default_value, "default")

        # No value found
        return (None, None)

    def has_mechanism(self, gid, syn_index, mech_name):
        """Check if a synapse has a specific mechanism"""
        if gid not in self.gid_data:
            return False

        if mech_name not in self.gid_data[gid]["has_mech"]:
            return False

        if syn_index >= len(self.gid_data[gid]["has_mech"][mech_name]):
            return False

        return self.gid_data[gid]["has_mech"][mech_name][syn_index]

    def get_mechanism_parameters(
        self,
        gid,
        syn_id,
        syn_name,
        throw_error_on_missing_id=True,
        throw_error_on_missing_param=False,
    ):
        """
        Get mechanism parameters with fallback to defaults

        Args:
            gid: Cell GID
            syn_id: Synapse ID
            syn_name: Synapse mechanism name
            throw_error_on_missing_id: Whether to throw error if gid/syn_id not found
            throw_error_on_missing_param: Whether to throw error if parameter missing

        Returns:
            Dict of parameter values
        """
        # Convert syn_name to mechanism name
        mech_name = self.syn_mech_names[syn_name]

        # Check if gid and syn_id exist
        if gid not in self.syn_store.id_to_index:
            if throw_error_on_missing_id:
                raise RuntimeError(f"Gid {gid} not found")
            return {}

        if syn_id not in self.syn_store.id_to_index[gid]:
            if throw_error_on_missing_id:
                raise RuntimeError(f"Synapse {syn_id} not found for gid {gid}")
            return {}

        # Get array index
        syn_index = self.syn_store.id_to_index[gid][syn_id]

        # Start with default parameters
        result = {}
        default_params = self.syn_store.param_store.get_default_params(
            gid, syn_id, syn_index, mech_name
        )
        if default_params:
            result.update(default_params)

        # Override with specific parameters if available
        if self.syn_store.param_store.has_mechanism_parameters(
            gid, syn_index, mech_name
        ):
            specific_params = self.syn_store.param_store.get_mechanism_parameters(
                gid, syn_index, mech_name
            )
            if specific_params:
                result.update(specific_params)

        # Check if any parameters found
        if not result and throw_error_on_missing_param:
            raise RuntimeError(
                f"No parameters found for gid {gid} synapse {syn_id} mechanism {syn_name}"
            )

        return result

    def set_mechanism_parameters(self, gid, syn_index, mech_name, attrs, syn_id=None):
        """Set multiple attributes for a mechanism at once"""
        if mech_name not in self.mech_param_specs:
            return False

        changed = False
        for param_name, value in attrs.items():
            if param_name in self.mech_param_specs[mech_name]:
                if self.set_param(gid, syn_index, mech_name, param_name, value, syn_id):
                    changed = True

        return changed

    def set_mechanism_parameters_for_synapses(
        self, gid, mech_name, attrs, synapse_selector
    ):
        """
        Set mechanism attributes for a subset of synapses

        Args:
            gid: Cell GID
            mech_name: Mechanism name
            attrs: Dict of parameter values
            synapse_selector: Selection criteria (same as set_default_value)
        """
        # If we're setting parameters for all synapses with the same value,
        # use default values instead of individual settings
        if synapse_selector is None:
            # Global defaults for all synapses
            for param_name, value in attrs.items():
                self.set_default_value(gid, mech_name, param_name, value)
            return True

        # Get matching synapse IDs
        matching_synapses = self._get_matching_synapses(gid, synapse_selector)

        # Set parameters for each matching synapse
        for syn_id, syn_index in matching_synapses:
            self.set_mech_attrs(gid, syn_index, mech_name, attrs, syn_id)

        return True

    def _get_matching_synapses(self, gid, selector):
        """
        Get list of (syn_id, syn_index) pairs matching the selector
        """
        # Need access to parent storage to get synapse ID mapping
        if not hasattr(self, "parent_storage"):
            return []

        result = []

        # Handle different selector types
        if selector is None:
            # All synapses
            return list(self.parent_storage.id_to_index[gid].items())

        elif isinstance(selector, (list, tuple, set, np.ndarray)):
            # Specific synapse IDs
            for syn_id in selector:
                if syn_id in self.parent_storage.id_to_index[gid]:
                    result.append(
                        (syn_id, self.parent_storage.id_to_index[gid][syn_id])
                    )

        elif callable(selector):
            # Custom function
            for syn_id, syn_index in self.parent_storage.id_to_index[gid].items():
                if selector(syn_id):
                    result.append((syn_id, syn_index))

        elif isinstance(selector, dict):
            # Filter criteria
            props = self.parent_storage.properties[gid]
            for syn_id, syn_index in self.parent_storage.id_to_index[gid].items():
                # Apply each filter criterion
                matches = True
                if (
                    "syn_types" in selector
                    and props["syn_type"][syn_index] not in selector["syn_types"]
                ):
                    matches = False
                if (
                    matches
                    and "layers" in selector
                    and props["syn_layer"][syn_index] not in selector["layers"]
                ):
                    matches = False
                if (
                    matches
                    and "sources" in selector
                    and props["source_population"][syn_index] not in selector["sources"]
                ):
                    matches = False
                if (
                    matches
                    and "swc_types" in selector
                    and props["swc_type"][syn_index] not in selector["swc_types"]
                ):
                    matches = False

                if matches:
                    result.append((syn_id, syn_index))

        return result


class SynapseStore:
    """Memory-efficient storage for synapses using NumPy arrays"""

    def __init__(self, mech_param_specs, initial_size=1000):
        """
        Initialize synapse storage

        Args:
            mech_param_specs: Dict mapping mechanism names to parameter specifications
            initial_size: minimum initial number of synapses per gid
        """
        self.mech_param_specs = mech_param_specs
        self.param_store = SynapseMechanismParameterStore(mech_param_specs)
        self.initial_size = initial_size

        # Storage for core synapse properties
        self.attrs = {}  # gid to synapse array

        # Maps syn_id to array index for each gid
        self.id_to_index = defaultdict(dict)

        # Next available index for each gid
        self.next_index = defaultdict(int)

    def add_synapses(self, gid, attr_items):
        """
        Add multiple synapses at once

        Args:
            gid: Cell GID
            syn_ids: List of synapse IDs
            attr_items: Iterator of pairs (syn_id, syn_attr_dict)
        """

        # Create or resize array if needed
        self._ensure_capacity(gid)

        # Store properties
        count = 0
        for i, (syn_id, attrs) in enumerate(attr_items):
            idx = self.next_index[gid] + i
            self.id_to_index[gid][syn_id] = idx

            # Set core properties
            for field in attrs:
                if field in SYNAPSE_CORE_DTYPE.names:
                    self.attrs[gid][idx][field] = getattr(attrs, field)

            # Set syn_id field
            self.attrs[gid][idx]["syn_id"] = syn_id

            count += 1

        self.next_index[gid] += count

    def add_synapses_from_arrays(
        self, gid, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs
    ):
        """
        Add synapses efficiently from column arrays

        Args:
        gid: Cell GID
        syn_ids: Array of synapse IDs
        syn_layers: Array of layer indices
        syn_types: Array of synapse types
        swc_types: Array of SWC types
        syn_secs: Array of section indices
        syn_locs: Array of section locations
        """
        count = len(syn_ids)
        if count == 0:
            return

        # Create or resize property array if needed
        self._ensure_capacity(gid, count)

        # Get array slice for new synapses
        start_idx = self.next_index[gid]
        end_idx = start_idx + count
        array_slice = slice(start_idx, end_idx)

        # Create index mapping
        for i, syn_id in enumerate(syn_ids):
            self.id_to_index[gid][syn_id] = start_idx + i

        # Set core attributes
        attrs = self.attrs[gid]
        attrs["syn_id"][array_slice] = syn_ids
        attrs["syn_type"][array_slice] = syn_types
        attrs["swc_type"][array_slice] = swc_types
        attrs["syn_layer"][array_slice] = syn_layers
        attrs["syn_section"][array_slice] = syn_secs
        attrs["syn_loc"][array_slice] = syn_locs

        # Initialize source attributes with default values
        attrs["source_gid"][array_slice] = -1  # Invalid GID
        attrs["source_population"][array_slice] = -1  # Invalid population
        attrs["delay"][array_slice] = 0.0  # Default delay

        # Update next available index
        self.next_index[gid] += count

    def get_synapse(self, gid, syn_id):
        """Get synapse wrapper for a specific ID"""
        if gid not in self.attrs:
            return None

        if syn_id not in self.id_to_index[gid]:
            return None

        idx = self.id_to_index[gid][syn_id]
        return SynapseView(self, gid, idx, syn_id)

    def get_synapses_by_filter(self, gid, **filters):
        """Get synapses matching filters (e.g., syn_type, layer, etc.)"""
        if gid not in self.attrs:
            return []

        # Map of filter names to synapse field names
        filter_name_map = {
            "syn_types": "syn_type",
            "layers": "syn_layer",
            "sources": "source_population",
            "swc_types": "swc_type",
        }

        # Start with all valid indices
        valid_indices = np.arange(self.next_index[gid])

        # Apply each filter
        for field, value in filters.items():
            dtype_field = filter_name_map[field]
            if dtype_field in SYNAPSE_CORE_DTYPE.names:
                if isinstance(value, (ndarray, list, set)):
                    # Multiple allowed values
                    mask = np.zeros(len(valid_indices), dtype=bool)
                    for val in value:
                        mask |= self.attrs[gid][dtype_field][valid_indices] == val
                    valid_indices = valid_indices[mask]
                else:
                    # Single value
                    valid_indices = valid_indices[
                        self.attrs[gid][dtype_field][valid_indices] == value
                    ]

        # Convert indices to SynapseView objects
        result = []
        for idx in valid_indices:
            syn_id = self.attrs[gid][idx]["syn_id"]
            result.append(SynapseView(self, gid, idx, syn_id))
        return result

    def _ensure_capacity(self, gid, needed_size=0):
        """Ensure arrays have enough capacity, resizing if necessary"""
        if gid not in self.attrs:
            # Initial creation - allocate with buffer
            initial_size = max(needed_size, self.initial_size)
            self.attrs[gid] = np.zeros(initial_size, dtype=SYNAPSE_CORE_DTYPE)

        elif self.next_index[gid] + needed_size > len(self.attrs[gid]):
            # Required to resize - allocate additional space
            new_size = max(
                int((self.next_index[gid] + needed_size) * 1.5),
                len(self.attrs[gid]) * 2,
            )

            # Resize core attrs
            new_array = np.zeros(new_size, dtype=SYNAPSE_CORE_DTYPE)
            new_array[: self.next_index[gid]] = self.attrs[gid][: self.next_index[gid]]
            self.attrs[gid] = new_array

            # Create new parameter arrays and copy data
            old_data = self.param_store.gid_data[gid]

            # Copy parameter data
            for mech_name in self.mech_param_specs:
                self.param_store.gid_data[gid]["arrays"][mech_name][
                    : self.next_index[gid]
                ] = old_data["arrays"][mech_name][: self.next_index[gid]]
                self.param_store.gid_data[gid]["has_mech"][mech_name][
                    : self.next_index[gid]
                ] = old_data["has_mech"][mech_name][: self.next_index[gid]]

    def keys(self, gid):
        """Iterate over syn_id keys for a gid"""
        if gid not in self.attrs:
            return

        for syn_id in self.id_to_index[gid].keys():
            yield syn_id

    def items(self, gid):
        """Iterate over (syn_id, synapse) pairs for a gid"""
        if gid not in self.attrs:
            return

        for syn_id, idx in self.id_to_index[gid].items():
            yield syn_id, SynapseView(self, gid, idx, syn_id)


class SynapseView:
    """View class that provides single synapse interface"""

    def __init__(self, storage, gid, index, syn_id):
        self.storage = storage
        self.gid = gid
        self.index = index
        self.syn_id = syn_id
        self._row = storage.attrs[gid][index]
        self.source = SynapseSourceView(storage, gid, index)
        self.mech_params = SynapseMechanismParameterView(storage, gid, index)

    @property
    def syn_type(self):
        return self._row["syn_type"]

    @syn_type.setter
    def syn_type(self, value):
        self._row["syn_type"] = value

    @property
    def swc_type(self):
        return self._row["swc_type"]

    @swc_type.setter
    def swc_type(self, value):
        self._row["swc_type"] = value

    @property
    def syn_layer(self):
        return self._row["syn_layer"]

    @syn_layer.setter
    def syn_layer(self, value):
        self._row["syn_layer"] = value

    @property
    def syn_loc(self):
        return self._row["syn_loc"]

    @syn_loc.setter
    def syn_loc(self, value):
        self._row["syn_loc"] = value

    @property
    def syn_section(self):
        return self._row["syn_section"]

    @syn_section.setter
    def syn_section(self, value):
        self._row["syn_section"] = value


class SynapseSourceView:
    """View class for synapse source properties"""

    def __init__(self, storage, gid, index):
        self.storage = storage
        self._cell_gid = gid
        self.index = index
        self._row = storage.attrs[gid][index]

    @property
    def gid(self):
        return self._row["source_gid"]

    @gid.setter
    def gid(self, value):
        self._row["source_gid"] = value

    @property
    def population(self):
        return self._row["source_population"]

    @population.setter
    def population(self, value):
        self._row["source_population"] = value

    @property
    def delay(self):
        return self._row["delay"]

    @delay.setter
    def delay(self, value):
        self._row["delay"] = value

    def __repr__(self):
        if self.delay is None:
            repr_delay = "None"
        else:
            repr_delay = f"{self.delay:.02f}"
        return f"SynapseSource({self.gid}, {self.population}, {repr_delay})"


class SynapseMechanismParameterView:
    """View class that provides interface to synapse mechanism parameters"""

    def __init__(self, storage, gid, index):
        self.storage = storage
        self.gid = gid
        self.index = index
        self.param_store = storage.param_store

    def __getitem__(self, mech_index):
        """Get attributes for a mechanism"""
        # Convert string keys to int if needed
        if isinstance(mech_index, str):
            try:
                mech_index = int(mech_index)
            except ValueError:
                return {}

        # Find mechanism name from index
        mech_name = None
        for name, idx in self.storage.param_store.mech_param_index.items():
            if idx == mech_index:
                mech_name = name
                break

        if mech_name is None:
            return {}

        # Get mechanism attributes
        return self.param_store.get_parameter_values(self.gid, self.index, mech_name)

    def __setitem__(self, mech_index, attrs):
        """Set attributes for a mechanism"""
        # Convert string keys to int if needed
        if isinstance(mech_index, str):
            try:
                mech_index = int(mech_index)
            except ValueError:
                return

        # Find mechanism name from index
        mech_name = None
        for name, idx in self.storage.param_store.mech_param_index.items():
            if idx == mech_index:
                mech_name = name
                break

        if mech_name is None:
            return

        # Set mechanism attributes
        self.param_store.set_parameter_values(self.gid, self.index, mech_name, attrs)

    def __contains__(self, mech_index):
        """Check if mechanism has any attributes set"""
        # Convert string keys to int if needed
        if isinstance(mech_index, str):
            try:
                mech_index = int(mech_index)
            except ValueError:
                return False

        # Find mechanism name from index
        mech_name = None
        for name, idx in self.storage.param_store.mech_param_index.items():
            if idx == mech_index:
                mech_name = name
                break

        if mech_name is None:
            return False

        return self.param_store.has_mechanism_parameters(
            self.gid, self.index, mech_name
        )


class SynapseManager:
    """This class provides an interface to store, retrieve, and modify
    attributes of synaptic mechanisms.
    """

    def __init__(
        self,
        env: AbstractEnv,
        syn_mech_names: Dict[str, str],
        syn_param_rules: Dict[str, Dict[str, Union[str, List[str], Dict[str, int]]]],
    ) -> None:
        """An Env object containing imported network configuration metadata
        uses an instance of SynapseManager to track all metadata
        related to the identity, location, and configuration of all
        synaptic connections in the network.

        :param env: :class:'Env'
        :param syn_mech_names: dict of the form: { label: mechanism name }
        :param syn_param_rules: dict of the form:
               { mechanism name:
                    mech_file: path.mod
                    mech_params: list of parameter names
                    netcon_params: dictionary { parameter name: index }
                }
        """
        self.env = env
        self.syn_mech_names = syn_mech_names
        self.syn_config = {
            k: v["synapses"] for k, v in env.celltypes.items() if "synapses" in v
        }
        self.syn_param_rules = syn_param_rules
        self.syn_name_index_dict = {
            label: index for index, label in enumerate(syn_mech_names)
        }  # int : mech_name dict

        # Define parameter specifications for each mechanism
        mech_param_specs = {}
        for mech_name, rule in syn_param_rules.items():
            params = {}
            idx = 0
            # Add mechanism parameters
            for param in rule.get("mech_params", []):
                params[param] = idx
                idx += 1
            # Add netcon parameters
            for param in rule.get("netcon_params", {}).keys():
                params[param] = idx
                idx += 1
            mech_param_specs[mech_name] = params

        # Synapse attribute storage
        self.syn_store = SynapseStore(mech_param_specs)
        self.syn_id_attr_backup_dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: None))
        )

        self.pps_dict = defaultdict(
            lambda: defaultdict(
                lambda: SynapsePointProcess(mech={}, netcon={}, vecstim={})
            )
        )
        self.presyn_names = {id: name for name, id in env.Populations.items()}
        self.filter_cache = {}
        self.filter_id_cache = {}

    def init_syn_id_attrs_from_iter(
        self, cell_iter, attr_type="dict", attr_tuple_index=None, debug=False
    ):
        """
        Initializes synaptic attributes given an iterator that returns (gid, attr_dict).
        See `init_syn_id_attrs` for details on the format of the input dictionary.
        """

        if attr_type == "dict":
            for gid, attr_dict in cell_iter:
                syn_ids = attr_dict["syn_ids"]
                syn_layers = attr_dict["syn_layers"]
                syn_types = attr_dict["syn_types"]
                swc_types = attr_dict["swc_types"]
                syn_secs = attr_dict["syn_secs"]
                syn_locs = attr_dict["syn_locs"]
                self.init_syn_id_attrs(
                    gid,
                    syn_ids,
                    syn_layers,
                    syn_types,
                    swc_types,
                    syn_secs,
                    syn_locs,
                )
        elif attr_type == "tuple":
            syn_ids_ind = attr_tuple_index.get("syn_ids", None)
            syn_locs_ind = attr_tuple_index.get("syn_locs", None)
            syn_layers_ind = attr_tuple_index.get("syn_layers", None)
            syn_types_ind = attr_tuple_index.get("syn_types", None)
            swc_types_ind = attr_tuple_index.get("swc_types", None)
            syn_secs_ind = attr_tuple_index.get("syn_secs", None)
            syn_locs_ind = attr_tuple_index.get("syn_locs", None)
            for gid, attr_tuple in cell_iter:
                syn_ids = attr_tuple[syn_ids_ind]
                syn_layers = attr_tuple[syn_layers_ind]
                syn_types = attr_tuple[syn_types_ind]
                swc_types = attr_tuple[swc_types_ind]
                syn_secs = attr_tuple[syn_secs_ind]
                syn_locs = attr_tuple[syn_locs_ind]
                self.init_syn_id_attrs(
                    gid,
                    syn_ids,
                    syn_layers,
                    syn_types,
                    swc_types,
                    syn_secs,
                    syn_locs,
                )
        else:
            raise RuntimeError(
                f"init_syn_id_attrs_from_iter: unrecognized input attribute type {attr_type}"
            )

    def init_syn_id_attrs(
        self,
        gid: int,
        syn_ids: ndarray,
        syn_layers: ndarray,
        syn_types: ndarray,
        swc_types: ndarray,
        syn_secs: ndarray,
        syn_locs: ndarray,
    ) -> None:
        """
        Initializes synaptic attributes for the given cell gid.
        Only the intrinsic properties of a synapse, such as type, layer, location are set.

        Connection edge attributes such as source gid, point process
        parameters, and netcon/vecstim objects are initialized to None
        or empty dictionaries.

          - syn_ids: synapse ids
          - syn_layers: layer index for each synapse id
          - syn_types: synapse type for each synapse id
          - swc_types: swc type for each synapse id
          - syn_secs: section index for each synapse id
          - syn_locs: section location for each synapse id

        """
        if gid in self.syn_store.attrs:
            raise RuntimeError(f"Entry {gid} exists in synapse attribute dictionary")

        synapse_count = len(syn_ids)
        if synapse_count == 0:
            return

        # Add the synapses to storage
        self.syn_store.add_synapses_from_arrays(
            gid, syn_ids, syn_layers, syn_types, swc_types, syn_secs, syn_locs
        )

    def init_edge_attrs(
        self,
        gid: int,
        presyn_name: str,
        presyn_gids: ndarray,
        edge_syn_ids: ndarray,
        delays: Optional[List[Union[float64, float]]] = None,
    ) -> None:
        """
        Sets connection edge attributes for the specified synapse ids.

        :param gid: gid for post-synaptic (target) cell (int)
        :param presyn_name: name of presynaptic (source) population (string)
        :param presyn_ids: gids for presynaptic (source) cells (array of int)
        :param edge_syn_ids: synapse ids on target cells to be used for connections (array of int)
        :param delays: axon conduction (netcon) delays (array of float)
        """

        presyn_index = int(self.env.Populations[presyn_name])

        if delays is None:
            delays = [2.0 * h.dt] * len(edge_syn_ids)

        # Get storage for this gid
        if gid not in self.syn_store.attrs:
            raise RuntimeError(f"init_edge_attrs: gid {gid} has not been initialized")

        # Determine synapse array indices
        indices = []

        for edge_syn_id in edge_syn_ids:
            if edge_syn_id not in self.syn_store.id_to_index[gid]:
                raise RuntimeError(
                    f"init_edge_attrs: gid {gid}: synapse id {edge_syn_id} has not been initialized"
                )

            idx = self.syn_store.id_to_index[gid][edge_syn_id]

            # Ensure source info is not already initialized
            if self.syn_store.attrs[gid]["source_gid"][idx] != -1:
                raise RuntimeError(
                    f"init_edge_attrs: gid {gid}: synapse id {edge_syn_id} has already been initialized with edge attributes"
                )

            indices.append(idx)

        indices = np.asarray(indices, dtype=np.int32)

        attrs = self.syn_store.attrs[gid]

        # Set source GIDs, presyn population index, delay
        attrs["source_gid"][indices] = presyn_gids
        attrs["source_population"][indices] = presyn_index
        attrs["delay"][indices] = delays

    def init_edge_attrs_from_iter(
        self,
        pop_name: str,
        presyn_name: str,
        attr_info: Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
        edge_iter: List[Tuple[int, Tuple[ndarray, Dict[str, List[ndarray]]]]],
        set_edge_delays: bool = True,
    ) -> None:
        """
        Initializes edge attributes for all cell gids returned by iterator.

        :param pop_name: name of postsynaptic (target) population (string)
        :param source_name: name of presynaptic (source) population (string)
        :param attr_info: dictionary mapping attribute name to indices in iterator tuple
        :param edge_iter: edge attribute iterator
        :param set_edge_delays: bool
        """
        connection_velocity = float(self.env.connection_velocity[presyn_name])
        if pop_name in attr_info and presyn_name in attr_info[pop_name]:
            edge_attr_info = attr_info[pop_name][presyn_name]
        else:
            raise RuntimeError(
                f"init_edge_attrs_from_iter: missing edge attributes for projection {presyn_name} -> {pop_name}"
            )

        if (
            "Synapses" in edge_attr_info
            and "syn_id" in edge_attr_info["Synapses"]
            and "Connections" in edge_attr_info
            and "distance" in edge_attr_info["Connections"]
        ):
            syn_id_attr_index = edge_attr_info["Synapses"]["syn_id"]
            distance_attr_index = edge_attr_info["Connections"]["distance"]
        else:
            raise RuntimeError(
                f"init_edge_attrs_from_iter: missing edge attributes for projection {presyn_name} -> {pop_name}"
            )

        for postsyn_gid, edges in edge_iter:
            presyn_gids, edge_attrs = edges
            edge_syn_ids = edge_attrs["Synapses"][syn_id_attr_index]
            edge_dists = edge_attrs["Connections"][distance_attr_index]

            if set_edge_delays:
                delays = np.asarray(
                    [
                        max((distance / connection_velocity), 2.0 * h.dt)
                        for distance in edge_dists
                    ],
                    dtype=np.float32,
                )
            else:
                delays = None

            self.init_edge_attrs(
                postsyn_gid,
                presyn_name,
                presyn_gids,
                edge_syn_ids,
                delays=delays,
            )

    def get_synapse(self, gid: int, syn_id: uint32):
        return self.syn_store.get_synapse(gid, syn_id)

    def add_point_process(
        self, gid: int, syn_id: uint32, syn_name: str, pps: "HocObject"
    ) -> "HocObject":
        """
        Adds mechanism point process for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param pps: hoc point process
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.mech:
            raise RuntimeError(
                f"add_point_process: gid {gid} Synapse id {syn_id} already has mechanism {syn_name}"
            )
        else:
            pps_dict.mech[syn_index] = pps
        return pps

    def has_point_process(self, gid, syn_id, syn_name):
        """
        Returns True if the given synapse id already has the named mechanism, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        return syn_index in pps_dict.mech

    def get_point_process(
        self, gid: int, syn_id: uint32, syn_name: str, throw_error: bool = True
    ) -> "HocObject":
        """
        Returns the mechanism for the given synapse id on the given cell.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: mechanism name
        :return: hoc point process
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.mech:
            return pps_dict.mech[syn_index]
        else:
            if throw_error:
                raise RuntimeError(
                    f"get_point_process: gid {gid} synapse id {syn_id} has no point process for mechanism {syn_name}"
                )
            else:
                return None

    def add_netcon(
        self, gid: int, syn_id: uint32, syn_name: str, nc: "HocObject"
    ) -> "HocObject":
        """
        Adds a NetCon object for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param nc: :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.netcon:
            raise RuntimeError(
                f"add_netcon: gid {gid} Synapse id {syn_id} mechanism {syn_name} already has netcon"
            )
        else:
            pps_dict.netcon[syn_index] = nc
        return nc

    def has_netcon(self, gid, syn_id, syn_name):
        """
        Returns True if a netcon exists for the specified cell/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        return syn_index in pps_dict.netcon

    def get_netcon(self, gid, syn_id, syn_name, throw_error=True):
        """
        Returns the NetCon object associated with the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.netcon:
            return pps_dict.netcon[syn_index]
        else:
            if throw_error:
                raise RuntimeError(
                    f"get_netcon: gid {gid} synapse id {syn_id} has no netcon for mechanism {syn_name}"
                )
            else:
                return None

    def del_netcon(self, gid, syn_id, syn_name, throw_error=True):
        """
        Removes a NetCon object for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.netcon:
            del pps_dict.netcon[syn_index]
        else:
            if throw_error:
                raise RuntimeError(
                    f"del_netcon: gid {gid} synapse id {syn_id} has no netcon for mechanism {syn_name}"
                )

    def add_vecstim(self, gid, syn_id, syn_name, vs, nc):
        """
        Adds a VecStim object and associated NetCon for the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param vs: :class:'h.VecStim'
        :param nc: :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.vecstim:
            raise RuntimeError(
                f"add_vecstim: gid {gid} synapse id {syn_id} mechanism {syn_name} already has vecstim"
            )
        else:
            pps_dict.vecstim[syn_index] = vs, nc
        return vs

    def has_vecstim(self, gid, syn_id, syn_name):
        """
        Returns True if a vecstim exists for the specified cell/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        return syn_index in pps_dict.vecstim

    def get_vecstim(self, gid, syn_id, syn_name, throw_error=True):
        """
        Returns the VecStim and NetCon objects associated with the specified cell/synapse id/mechanism name.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: tuple of :class:'h.VecStim' :class:'h.NetCon'
        """
        syn_index = self.syn_name_index_dict[syn_name]
        gid_pps_dict = self.pps_dict[gid]
        pps_dict = gid_pps_dict[syn_id]
        if syn_index in pps_dict.vecstim:
            return pps_dict.vecstim[syn_index]
        else:
            if throw_error:
                raise RuntimeError(
                    f"get_vecstim: gid {gid} synapse {syn_id}: vecstim for mechanism {syn_name} not found"
                )
            else:
                return None

    def has_mechanism_parameters(self, gid, syn_id, syn_name):
        """
        Returns True if mechanism attributes have been specified for the given cell id/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :return: bool
        """
        # Convert syn_name to mechanism name used in parameter store
        mech_name = self.syn_mech_names[syn_name]

        # Get synapse array index
        if (
            gid not in self.syn_store.id_to_index
            or syn_id not in self.syn_store.id_to_index[gid]
        ):
            return False

        syn_index = self.syn_store.id_to_index[gid][syn_id]

        # Check if mechanism exists for this synapse
        return self.syn_store.param_store.has_mechanism_parameters(
            gid, syn_index, mech_name
        )

    def get_mechanism_parameters(
        self,
        gid,
        syn_id,
        syn_name,
        throw_error_on_missing_id=True,
        throw_error_on_missing_param=True,
    ):
        """
        Returns mechanism attribute dictionary associated with the given cell id/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param throw_error_on_missing_id: Whether to raise error if gid or syn_id not found
        :param throw_error_on_missing_param: Whether to raise error if parameter value is not defined
        :return: dict
        """
        # Convert syn_name to mechanism name used in parameter store
        mech_name = self.syn_mech_names[syn_name]

        # Get synapse array index
        if gid not in self.syn_store.id_to_index:
            if throw_error_on_missing_id:
                raise RuntimeError(f"get_mechanism_parameters: gid {gid} not found")
            return None

        if syn_id not in self.syn_store.id_to_index[gid]:
            if throw_error_on_missing_id:
                raise RuntimeError(
                    f"get_mechanism_parameters: gid {gid} synapse {syn_id} not found"
                )
            return None

        syn_index = self.syn_store.id_to_index[gid][syn_id]

        # Check if mechanism exists
        if not self.syn_store.param_store.has_mechanism_params(
            gid, syn_index, mech_name
        ):
            if throw_error_on_missing_param:
                raise RuntimeError(
                    f"get_mechanism_parameters: gid {gid} synapse {syn_id}: attributes for synapse {syn_name} mechanism {mech_name} not found"
                )
            return None

        return self.syn_store.param_store.get_parameter_values(
            gid, syn_index, mech_name
        )

    def get_default_mechanism_parameters(
        self,
        gid,
        syn_id,
        syn_name,
        throw_error_on_missing_id=True,
        throw_error_on_missing_param=True,
    ):
        """
        Returns default mechanism attribute dictionary associated with the given cell id/synapse id/mechanism name, False otherwise.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param throw_error_on_missing_id: Whether to raise error if gid or syn_id not found
        :param throw_error_on_missing_param: Whether to raise error if parameter value is not defined
        :return: dict
        """
        # Convert syn_name to mechanism name used in parameter store
        mech_name = self.syn_mech_names[syn_name]

        # Get synapse array index
        if gid not in self.syn_store.id_to_index:
            if throw_error_on_missing_id:
                raise RuntimeError(f"get_mechanism_parameters: gid {gid} not found")
            return None

        if syn_id not in self.syn_store.id_to_index[gid]:
            if throw_error_on_missing_id:
                raise RuntimeError(
                    f"get_mechanism_parameters: gid {gid} synapse {syn_id} not found"
                )
            return None

        syn_index = self.syn_store.id_to_index[gid][syn_id]

        return self.syn_store.param_store.get_default_params(gid, syn_index, mech_name)

    def get_effective_mechanism_parameters(
        self, gid, syn_id, syn_name, throw_error_on_missing=True
    ):
        """
        Get all effective parameters for a mechanism with proper fallback to defaults

        Args:
            gid: Cell GID
            syn_id: Synapse ID
            syn_name: Synapse mechanism name
            throw_error_on_missing: Whether to throw error if no parameters found

        Returns:
            Dict of resolved parameter values (with expressions evaluated)

        Raises:
            RuntimeError: If parameters cannot be found and throw_error_on_missing is True
        """
        # Convert syn_name to mechanism name
        mech_name = self.syn_mech_names[syn_name]

        # Check if synapse exists
        if gid not in self.syn_store.id_to_index:
            if throw_error_on_missing:
                raise RuntimeError(
                    f"get_effective_mechanism_parameters: gid {gid} not found"
                )
            return {}

        if syn_id not in self.syn_store.id_to_index[gid]:
            if throw_error_on_missing:
                raise RuntimeError(
                    f"get_effective_mechanism_parameters: gid {gid} synapse {syn_id} not found"
                )
            return {}

        # Get array index
        syn_index = self.syn_store.id_to_index[gid][syn_id]

        # Get synapse properties for expression evaluation
        syn_props = self.syn_store.attrs[gid][syn_index]

        # Get all parameter names for this mechanism
        all_params = set(self.syn_param_rules[mech_name].get("mech_params", []))
        all_params.update(
            self.syn_param_rules[mech_name].get("netcon_params", {}).keys()
        )

        # Get parameter values with fallback using the hierarchy
        resolved_params = {}
        for param_name in all_params:
            value, source = self.syn_store.param_store.get_parameter_value_hierarchy(
                gid, syn_index, mech_name, param_name, syn_id
            )

            if value is not None:
                # Resolve any Promise or ExprClosure values
                if isinstance(value, Promise):
                    resolved_params[param_name] = value.clos(*value.args)
                elif isinstance(value, ExprClosure):
                    if value.parameters[0] == "delay":
                        delay = syn_props["delay"]
                        resolved_params[param_name] = value(delay)
                    else:
                        if throw_error_on_missing:
                            raise RuntimeError(
                                f"get_effective_mechanism_parameters: Unknown expression parameter "
                                f"{value.parameters} for {param_name}"
                            )
                else:
                    resolved_params[param_name] = value

        # If no parameters found, raise error if requested
        if not resolved_params and throw_error_on_missing:
            raise RuntimeError(
                f"get_effective_mechanism_parameters: No parameters found for gid {gid} "
                f"synapse {syn_id} mechanism {syn_name}"
            )

        return resolved_params

    def add_mechanism_parameters(self, gid, syn_id, syn_name, params, append=False):
        """
        Specifies mechanism attribute dictionary for the given cell id/synapse id/mechanism name. Assumes mechanism
        attributes have not been set yet for this synapse mechanism.

        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        :param append: whether to append attribute values with the same attribute name
        """
        self.add_mechanism_parameters_from_iter(
            gid,
            syn_name,
            iter({syn_id: params}),
            multiple="error",
            append=append,
        )

    def add_default_mechanism_parameters(self, gid, syn_name, params, syn_ids):
        """
        Specifies default mechanism parameters for the given cell
        id/mechanism name.

        :param gid: cell id
        :param syn_name: synapse mechanism name
        :param params: dict
        :param syn_ids: tuple/list of synapse ids

        """
        for param_name, param_value in params.items():
            self.syn_store.param_store.set_default_value(
                gid, syn_name, param_name, param_value, syn_ids
            )

    def add_mechanism_parameters_from_iter(
        self, gid, syn_name, params_iter, multiple="error", append=False
    ):
        """
        Adds mechanism attributes for the given cell id/synapse id/synapse mechanism.

        Args:
        gid: cell id
        syn_name: synapse mechanism name
        params_iter: iterator that yields (syn_id, params_dict) pairs
        multiple: behavior when an attribute value is provided for a synapse that already has attributes:
               - 'error' (default): raise an error
               - 'skip': do not update attribute value
               - 'overwrite': overwrite value
        append: whether to append attribute values with the same attribute name
        """
        # Convert syn_name to the mechanism name used in parameter store
        mech_name = self.syn_mech_names[syn_name]

        # Check if gid exists
        if gid not in self.syn_store.attrs:
            raise RuntimeError(
                f"add_mechanism_parameters_from_iter: gid {gid} not found"
            )

        # Process the iterator in batches for better performance
        batch_size = 1000
        batch = []

        # Process each (syn_id, params_dict) pair
        for syn_id, params_dict in params_iter:
            batch.append((syn_id, params_dict))

            # Process in batches to avoid memory issues with large inputs
            if len(batch) >= batch_size:
                self._process_mech_attrs_batch(gid, mech_name, batch, multiple, append)
                batch = []

        # Process any remaining items
        if batch:
            self._process_mech_attrs_batch(gid, mech_name, batch, multiple, append)

    def _add_mechanism_parameters_batch(self, gid, mech_name, batch, multiple, append):
        """Helper method to add a batch of mechanism attributes."""
        for syn_id, params_dict in batch:
            # Check if synapse exists
            if syn_id not in self.syn_store.id_to_index[gid]:
                raise RuntimeError(
                    f"add_mechanism_parameters_from_iter: gid {gid} synapse id {syn_id} has not been created yet"
                )

            # Get array index for this synapse
            syn_index = self.syn_store.id_to_index[gid][syn_id]

            # Check if mechanism already has attributes
            has_mech = self.syn_store.param_store.has_mechanism_parameters(
                gid, syn_index, mech_name
            )
            if has_mech:
                if multiple == "error":
                    raise RuntimeError(
                        f"add_mechanism_parameters_from_iter: gid {gid} synapse id {syn_id} mechanism {mech_name} already has parameters"
                    )
                elif multiple == "skip":
                    continue
                # For "overwrite", continue processing

            # Process each parameter
            for param_name, param_value in params_dict.items():
                if param_value is None:
                    raise RuntimeError(
                        f"add_mechanism_parameters_from_iter: gid {gid} synapse id {syn_id} mechanism {mech_name} parameter {param_name} has no value"
                    )

                # Get current value if appending
                if append and has_mech:
                    current_value = self.syn_store.param_store.get_param(
                        gid, syn_index, mech_name, param_name
                    )

                    if current_value is not None:
                        # Convert to list if not already
                        if not isinstance(current_value, list):
                            current_value = [current_value]
                        # Append new value
                        param_value = current_value + [param_value]

                # Set parameter value
                self.syn_store.param_store.set_parameter_value(
                    gid, syn_index, mech_name, param_name, param_value
                )

    def modify_mechanism_parameters(
        self, pop_name, gid, syn_id, syn_name, params, expr_param_check="ignore"
    ):
        """
        Modifies mechanism attributes for the given cell id/synapse id/mechanism name.

        :param pop_name: population name
        :param gid: cell id
        :param syn_id: synapse id
        :param syn_name: synapse mechanism name
        :param params: dict
        :param expr_param_check: how to handle expression parameters
        """

        rules = self.syn_param_rules
        mech_name = self.syn_mech_names[syn_name]

        # Get synapse array index
        if (
            gid not in self.syn_store.id_to_index
            or syn_id not in self.syn_store.id_to_index[gid]
        ):
            raise RuntimeError(
                f"modify_mechanism_parameters: gid {gid} synapse id {syn_id} not found"
            )

        array_index = self.syn_store.id_to_index[gid][syn_id]

        # Get synapse attrs
        attrs = self.syn_store.attrs[gid]

        # Get source population information
        source_population = attrs["source_population"][array_index]
        presyn_name = self.presyn_names.get(source_population, None)

        # Get connection parameters
        connection_syn_params = None
        if presyn_name:
            connection_syn_params = self.env.connection_config[pop_name][
                presyn_name
            ].mechanisms

        # Determine mechanism parameters
        mech_params = {}
        if connection_syn_params is not None:
            swc_type = attrs["swc_type"][array_index]
            if "default" in connection_syn_params:
                section_syn_params = connection_syn_params["default"]
            else:
                section_syn_params = connection_syn_params[swc_type]
            mech_params = section_syn_params.get(syn_name, {})

        # Process parameters
        for k, v in params.items():
            if not (
                (k in rules[mech_name]["mech_params"])
                or (k in rules[mech_name]["netcon_params"])
            ):
                raise RuntimeError(
                    f"modify_mechanism_parameters: unknown parameter type {k}"
                )

            # Get or evaluate parameter
            mech_param = mech_params.get(k, None)
            if isinstance(mech_param, ExprClosure):
                if mech_param.parameters[0] == "delay":
                    delay = attrs["delay"][array_index]
                    new_val = mech_param(delay)
                else:
                    raise RuntimeError(
                        f"modify_mechanism_parameters: unknown expression parameter {mech_param.parameters}"
                    )
            else:
                new_val = v

            assert new_val is not None

            (
                value_result,
                source,
            ) = self.syn_store.param_store.get_parameter_value_hierarchy(
                gid, array_index, mech_name, k, syn_id
            )
            old_val = value_result if value_result is not None else mech_param

            if isinstance(new_val, ExprClosure):
                if isinstance(old_val, Promise):
                    old_val.clos = new_val
                    value_to_store = old_val
                else:
                    value_to_store = Promise(new_val, old_val)

            elif isinstance(new_val, dict):
                # Dictionary value handling for expressions
                if isinstance(old_val, Promise):
                    for sk, sv in new_val.items():
                        old_val.clos[sk] = sv
                    value_to_store = old_val
                elif isinstance(old_val, ExprClosure):
                    for sk, sv in new_val.items():
                        old_val[sk] = sv
                    value_to_store = old_val
                else:
                    if expr_param_check != "ignore":
                        raise RuntimeError(
                            f"modify_mechanism_parameters: dictionary for non-expression parameter {k}"
                        )
                    continue
            else:
                value_to_store = new_val

            self.syn_store.param_store.set_synapse_parameter(
                gid, array_index, mech_name, k, value_to_store, syn_id
            )

    def stash_mech_attrs(self, pop_name, gid):
        """
        Preserves mechanism attributes for the given cell id.

        Args:
            pop_name: population name
            gid: cell gid

        Returns:
            stash_id: Unique identifier for this stash
        """
        # Create a unique ID for this stash
        stash_id = uuid.uuid4()

        # Check if the gid exists in our storage
        if gid not in self.syn_store.attrs:
            return stash_id  # Return ID even if no data to stash

        # Create backup structures
        backup = {"param_arrays": {}, "complex_params": {}}

        valid_count = self.syn_store.next_index[gid]

        # Back up parameter arrays
        if gid in self.syn_store.param_store.gid_data:
            param_data = self.syn_store.param_store.gid_data[gid]

            # Copy mechanism arrays
            for mech_name, array in param_data["arrays"].items():
                backup["param_arrays"][mech_name] = {
                    "data": np.copy(array[:valid_count]),
                    "has_mech": np.copy(
                        param_data["has_mech"][mech_name][:valid_count]
                    ),
                }

        # Back up complex parameters (deep copy)
        if gid in self.syn_store.param_store.complex_params:
            backup["complex_params"] = copy.deepcopy(
                self.syn_store.param_store.complex_params[gid]
            )

        # Store backup in the backup dictionary
        self.syn_id_attr_backup_dict[gid][stash_id] = backup

        return stash_id

    def restore_mech_attrs(self, pop_name, gid, stash_id):
        """
        Restores mechanism attributes for the given cell id from a previous stash.

        Args:
            pop_name: population name
            gid: cell gid
            stash_id: Unique identifier returned by stash_mech_attrs
        """
        # Check if stash exists
        if (
            gid not in self.syn_id_attr_backup_dict
            or stash_id not in self.syn_id_attr_backup_dict[gid]
        ):
            raise RuntimeError(
                f"restore_mech_attrs: No stash found for gid {gid} with id {stash_id}"
            )

        # Get backup data
        backup = self.syn_id_attr_backup_dict[gid][stash_id]
        valid_count = self.syn_store.next_index[gid]

        # Restore parameter arrays
        if backup["param_arrays"]:
            for mech_name, mech_data in backup["param_arrays"].items():
                if mech_name in self.syn_store.param_store.mech_param_specs:
                    # Restore array data
                    self.syn_store.param_store.gid_data[gid]["arrays"][mech_name][
                        :valid_count
                    ] = mech_data["data"]

                    # Restore mechanism flags
                    self.syn_store.param_store.gid_data[gid]["has_mech"][mech_name][
                        :valid_count
                    ] = mech_data["has_mech"]

        # Restore complex parameters
        if backup["complex_params"]:
            # Clear existing complex params
            if gid in self.syn_store.param_store.complex_params:
                self.syn_store.param_store.complex_params[gid].clear()

            # Set complex params from backup
            self.syn_store.param_store.complex_params[gid] = copy.deepcopy(
                backup["complex_params"]
            )

        # Remove the stash to free memory
        del self.syn_id_attr_backup_dict[gid][stash_id]

        # Clear any caches that might have stale data
        self.clear_filter_cache()

    def filter_synapses(
        self,
        gid: int,
        syn_sections: Optional[List[int]] = None,
        syn_indexes: None = None,
        syn_types: Optional[List[int]] = None,
        layers: None = None,
        sources: None = None,
        swc_types: None = None,
        cache: bool = False,
        max_cache_items: int = 10000,
    ) -> Dict[Any, Any]:
        """
        Returns a subset of the synapses of the given cell according to the given criteria.

        :param gid: int
        :param syn_sections: array of int
        :param syn_indexes: array of int: syn_ids
        :param syn_types: list of enumerated type: synapse category
        :param layers: list of enumerated type: layer
        :param sources: list of enumerated type: population names of source projections
        :param swc_types: list of enumerated type: swc_type
        :param cache: bool
        :return: iterator ( syn_id, synapse_view )
        """

        if cache:
            cache_args = tuple(
                tuple(x) if isinstance(x, list) else x
                for x in [
                    gid,
                    syn_sections,
                    syn_indexes,
                    syn_types,
                    layers,
                    sources,
                    swc_types,
                ]
            )
            if cache_args in self.filter_cache:
                return self.filter_cache[cache_args]

        # Check if GID exists
        if gid not in self.syn_store.attrs:
            return []

        # Get attr array for this GID
        attrs = self.syn_store.attrs[gid]

        # Start with all valid indices
        valid_count = self.syn_store.next_index[gid]
        mask = np.ones(valid_count, dtype=bool)

        # Apply section filter (fast path)
        if syn_sections is not None:
            # Convert to set for fast lookup
            section_set = set(syn_sections)
            section_mask = np.isin(
                attrs["syn_section"][:valid_count], list(section_set)
            )
            mask &= section_mask

        # Apply synapse ID filter
        if syn_indexes is not None:
            # Convert to set for fast lookup
            syn_id_set = set(syn_indexes)
            syn_id_mask = np.isin(attrs["syn_id"][:valid_count], list(syn_id_set))
            mask &= syn_id_mask

        # Apply synapse type filter
        if syn_types is not None:
            # Convert to set for fast lookup
            type_set = set(syn_types)
            type_mask = np.isin(attrs["syn_type"][:valid_count], list(type_set))
            mask &= type_mask

        # Apply layer filter
        if layers is not None:
            # Convert to set for fast lookup
            layer_set = set(layers)
            layer_mask = np.isin(attrs["syn_layer"][:valid_count], list(layer_set))
            mask &= layer_mask

        # Apply source population filter
        if sources is not None:
            # Convert to set for fast lookup
            source_set = set(sources)
            source_mask = np.isin(
                attrs["source_population"][:valid_count], list(source_set)
            )
            mask &= source_mask

        # Apply SWC type filter
        if swc_types is not None:
            # Convert to set for fast lookup
            swc_set = set(swc_types)
            swc_mask = np.isin(attrs["swc_type"][:valid_count], list(swc_set))
            mask &= swc_mask

        # Get array indices of matching synapses
        matching_indices = np.where(mask)[0]

        # For cache optimization, limit results
        if cache:
            # If too many items, don't cache
            if len(matching_indices) > max_cache_items:
                result_generator = self._generate_synapse_views(
                    gid, matching_indices, attrs
                )
                return result_generator

            # Otherwise, instantiate limited list for cache
            result_items = []
            for idx in matching_indices:
                syn_id = attrs["syn_id"][idx]
                syn_view = self.syn_store.get_synapse(gid, syn_id)
                result_items.append((syn_id, syn_view))

            # Cache the result
            self.filter_cache[cache_args] = result_items
            return result_items

        # Return generator to avoid instantiating all synapses at once
        return self._generate_synapse_views(gid, matching_indices, attrs)

    def _generate_synapse_views(self, gid, indices, attrs):
        """Helper to generate synapse views from array indices"""
        for idx in indices:
            syn_id = attrs["syn_id"][idx]
            syn_view = self.syn_store.get_synapse(gid, syn_id)
            yield (syn_id, syn_view)

    def filter_synapse_ids(
        self,
        gid: int,
        syn_sections: Optional[List[int]] = None,
        syn_indexes: Optional[List[int]] = None,
        syn_types: Optional[List[int]] = None,
        layers: Optional[List[int]] = None,
        sources: Optional[List[int]] = None,
        swc_types: Optional[List[int]] = None,
        cache: bool = False,
    ) -> np.ndarray:
        """
        Returns a subset of the synapse ids of the given cell according to the given criteria.
        Uses efficient array operations with the new SynapseStorage backend.

        Args:
            gid: Cell GID
            syn_sections: List of section indices
            syn_indexes: List of synapse IDs
            syn_types: List of synapse types
            layers: List of layer indices
            sources: List of source population indices
            swc_types: List of SWC types
            cache: Whether to cache results

        Returns:
            array of synapse IDs
        """
        # Check cache first
        if cache:
            cache_args = tuple(
                tuple(x) if isinstance(x, list) else x
                for x in [
                    gid,
                    syn_sections,
                    syn_indexes,
                    syn_types,
                    layers,
                    sources,
                    swc_types,
                ]
            )
            if cache_args in self.filter_id_cache:
                return self.filter_id_cache[cache_args]

        # Check if GID exists
        if gid not in self.syn_store.attrs:
            return np.array([], dtype=np.uint32)

        # Get properties array for this GID
        attrs = self.syn_store.attrs[gid]

        # Start with all valid indices
        valid_count = self.syn_store.next_index[gid]
        mask = np.ones(valid_count, dtype=bool)

        # Apply section filter
        if syn_sections is not None:
            section_set = set(syn_sections)
            section_mask = np.isin(
                attrs["syn_section"][:valid_count], list(section_set)
            )
            mask &= section_mask

        # Apply synapse ID filter
        if syn_indexes is not None:
            syn_id_set = set(syn_indexes)
            syn_id_mask = np.isin(attrs["syn_id"][:valid_count], list(syn_id_set))
            mask &= syn_id_mask

        # Apply synapse type filter
        if syn_types is not None:
            type_set = set(syn_types)
            type_mask = np.isin(attrs["syn_type"][:valid_count], list(type_set))
            mask &= type_mask

        # Apply layer filter
        if layers is not None:
            layer_set = set(layers)
            layer_mask = np.isin(attrs["syn_layer"][:valid_count], list(layer_set))
            mask &= layer_mask

        # Apply source population filter
        if sources is not None:
            source_set = set(sources)
            source_mask = np.isin(
                attrs["source_population"][:valid_count], list(source_set)
            )
            mask &= source_mask

        # Apply SWC type filter
        if swc_types is not None:
            swc_set = set(swc_types)
            swc_mask = np.isin(attrs["swc_type"][:valid_count], list(swc_set))
            mask &= swc_mask

        # Get synapse IDs of matching synapses
        result = attrs["syn_id"][mask]

        # Cache if requested
        if cache:
            self.filter_id_cache[cache_args] = result

        return result

    def get_filtered_syn_ids(
        self,
        gid,
        syn_sections=None,
        syn_indexes=None,
        syn_types=None,
        layers=None,
        sources=None,
        swc_types=None,
        cache=False,
    ):
        """
        Returns a subset of the synapse ids of the given cell according to the given criteria.
        :param gid:
        :param syn_sections:
        :param syn_indexes:
        :param syn_types:
        :param layers:
        :param sources:
        :param swc_types:
        :param cache:
        :return: sequence
        """
        return self.filter_synapse_ids(
            gid,
            syn_sections=syn_sections,
            syn_indexes=syn_indexes,
            syn_types=syn_types,
            layers=layers,
            sources=sources,
            swc_types=swc_types,
            cache=cache,
        )

    def partition_synapses_by_source(
        self, gid: int, syn_ids: Optional[List[uint32]] = None
    ) -> Dict[str, Optional[itertools.chain]]:
        """
        Partitions the synapse objects for the given cell based on the
        presynaptic (source) population index.

        Args:
            gid: cell id
            syn_ids: optional list of synapse ids to partition (if None, use all)

        Returns:
            dict mapping source population names to iterators of (syn_id, synapse) pairs
        """
        # Get population names and ordering
        source_names = {id: name for name, id in self.env.Populations.items()}

        # Check if gid exists
        if gid not in self.syn_store.attrs:
            # Return empty iterators for all populations
            return {name: itertools.chain() for name in source_names.values()}

        # Get properties array for this gid
        attrs = self.syn_store.attrs[gid]
        valid_count = self.syn_store.next_index[gid]

        # Filter by synapse IDs if specified
        if syn_ids is not None:
            # Create mask for specified synapse IDs
            syn_id_set = set(syn_ids)
            syn_mask = np.isin(attrs["syn_id"][:valid_count], list(syn_id_set))
            valid_indices = np.arange(valid_count)[syn_mask]
        else:
            # Use all valid indices
            valid_indices = np.arange(valid_count)

        # Create result dictionary with generators for each population
        result = {}

        # For each population, create a generator that yields matching synapses
        for pop_id, pop_name in source_names.items():
            # Create mask for this population
            pop_mask = attrs["source_population"][valid_indices] == pop_id
            pop_indices = valid_indices[pop_mask]

            # Create generator for this population's synapses
            def create_population_generator(indices):
                for idx in indices:
                    syn_id = attrs["syn_id"][idx]
                    syn_view = self.syn_store.get_synapse(gid, syn_id)
                    yield (syn_id, syn_view)

            # Use generator_ifempty to handle empty cases
            result[pop_name] = generator_ifempty(
                create_population_generator(pop_indices)
            )

        return result

    def partition_syn_ids_by_source(
        self, gid: int, syn_ids: Optional[List[uint32]] = None
    ) -> Dict[str, itertools.chain]:
        """
        Partitions the synapse ids for the given cell based on the
        presynaptic (source) population index.

        Args:
        gid: cell id
        syn_ids: optional list of synapse ids to partition (if None, use all)

        Returns:
        dict mapping source population names to iterators of synapse ids
        """
        # Get population names and ordering
        source_names = {id: name for name, id in self.env.Populations.items()}

        # Check if GID exists
        if gid not in self.syn_store.attrs:
            # Return empty iterators for all populations
            return {name: itertools.chain() for name in source_names.values()}

        # Get properties array for this GID
        attrs = self.syn_store.attrs[gid]
        valid_count = self.syn_store.next_index[gid]

        # Filter by synapse IDs if specified
        if syn_ids is not None:
            # Create mask for specified synapse IDs
            syn_id_set = set(syn_ids)
            mask = np.isin(attrs["syn_id"][:valid_count], list(syn_id_set))
            working_indices = np.arange(valid_count)[mask]
        else:
            # Use all valid indices
            working_indices = np.arange(valid_count)

        # Create result dictionary with generators for each population
        result = {}

        # For each population, create a generator that yields matching synapse IDs
        for pop_id, pop_name in source_names.items():
            # Create mask for this population
            pop_mask = attrs["source_population"][working_indices] == pop_id
            pop_indices = working_indices[pop_mask]

            # Get synapse IDs for this population
            pop_syn_ids = attrs["syn_id"][pop_indices]

            # Create generator from array
            def create_id_generator(ids):
                # Convert to Python list so we can iterate
                id_list = ids.tolist()
                return iter(id_list)

            # Use generator_ifempty to handle empty cases
            # This will return an empty iterator if there are no matches
            result[pop_name] = generator_ifempty(create_id_generator(pop_syn_ids))

        return result

    def del_syn_id_attr_dict(self, gid: int) -> None:
        """
        Removes the synapse attributes associated with the given cell gid.

        Args:
            gid: Cell gid to remove
        """
        # Remove from core properties storage
        if gid in self.syn_store.attrs:
            del self.syn_store.attrs[gid]

        # Remove from index mapping
        if gid in self.syn_store.id_to_index:
            del self.syn_store.id_to_index[gid]

        # Reset next index counter
        if gid in self.syn_store.next_index:
            del self.syn_store.next_index[gid]

        # Clean up parameter storage
        if gid in self.syn_store.param_store.gid_data:
            del self.syn_store.param_store.gid_data[gid]

        # Clean up complex parameter storage
        if gid in self.syn_store.param_store.complex_params:
            del self.syn_store.param_store.complex_params[gid]

        # Clean up point process structures
        if gid in self.pps_dict:
            del self.pps_dict[gid]

        # Clear any cached results that might reference this gid
        self.clear_filter_cache()

    def clear(self):
        """
        Clears all synapse data structures.
        """
        # Clear core storage structures
        self.syn_store.attrs.clear()
        self.syn_store.id_to_index.clear()
        self.syn_store.next_index.clear()

        # Clear parameter store
        self.syn_store.param_store.gid_data.clear()
        self.syn_store.param_store.complex_params.clear()

        # Clear point process structures
        self.pps_dict.clear()

        # Clear caches
        self.filter_cache.clear()
        self.filter_id_cache.clear()

    def clear_filter_cache(self):
        self.filter_id_cache.clear()
        self.filter_cache.clear()

    def has_gid(self, gid):
        return gid in self.syn_store.attrs

    def gids(self):
        return self.syn_store.attrs.keys()

    def synapse_ids(self, gid):
        return self.syn_store.keys(gid)

    def items(self, gid=None):
        if gid is None:
            return self.syn_store.attrs.items()
        else:
            return self.syn_store.items(gid)


def insert_cell_syns(
    env: AbstractEnv,
    gid: int,
    postsyn_name: str,
    presyn_name: str,
    syn_ids: Union[List[uint32], itertools.chain],
    unique: bool = False,
    insert_netcons: bool = False,
    insert_vecstims: bool = False,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """
    Insert mechanisms into given cell according to the synapse objects created in env.synapse_manager.
    Configures mechanisms according to parameter values specified in syn_params.

    :param env: :class:'Env'
    :param gid: cell id (int)
    :param postsyn_name: str
    :param presyn_name: str
    :param syn_ids: synapse ids (array of int)
    :param unique: True, if unique mechanisms are to be inserted for each synapse; False, if synapse mechanisms within
            a compartment will be shared.
    :param insert_netcons: bool; whether to build new netcons for newly constructed synapses
    :param insert_vecstims: bool; whether to build new vecstims for newly constructed netcons
    :param verbose: bool
    :return: number of inserted mechanisms

    """

    if gid not in env.biophys_cells[postsyn_name]:
        raise KeyError(
            f"insert_cell_syns: biophysical cell with gid {gid} does not exist"
        )

    cell = env.biophys_cells[postsyn_name][gid]

    syn_params = env.connection_config[postsyn_name][presyn_name].mechanisms

    synapse_config = env.celltypes[postsyn_name]["synapses"]

    if unique is None:
        if "unique" in synapse_config:
            unique = synapse_config["unique"]
        else:
            unique = False

    assert cell is not None

    swc_type_apical = env.SWC_Types["apical"]
    swc_type_basal = env.SWC_Types["basal"]
    swc_type_soma = env.SWC_Types["soma"]
    swc_type_axon = env.SWC_Types["axon"]
    swc_type_ais = env.SWC_Types["ais"]
    swc_type_hill = env.SWC_Types["hillock"]

    syns_dict_dend = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_axon = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_ais = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_hill = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    syns_dict_soma = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    layer_name_dict = dict({v: k for k, v in env.layers.items()})
    swc_name_dict = dict({v: k for k, v in env.SWC_Types.items()})

    syns_dict_by_type = {
        swc_type_apical: syns_dict_dend,
        swc_type_basal: syns_dict_dend,
        swc_type_axon: syns_dict_axon,
        swc_type_ais: syns_dict_ais,
        swc_type_hill: syns_dict_hill,
        swc_type_soma: syns_dict_soma,
    }
    py_sections = None
    if hasattr(cell, "hoc_cell"):
        if cell.hoc_cell is not None:
            cell = cell.hoc_cell
    if hasattr(cell, "cell_obj"):
        if cell.cell_obj is not None:
            cell = cell.cell_obj

    if hasattr(cell, "sections"):
        if cell.sections is not None:
            py_sections = [sec for sec in cell.sections]
    is_reduced = False
    if hasattr(cell, "is_reduced"):
        is_reduced = cell.is_reduced
        if callable(is_reduced):
            is_reduced = is_reduced()
        if isinstance(is_reduced, float):
            is_reduced = is_reduced > 0.0

    cell_soma = None
    cell_dendrite = None
    reduced_section_dict = {}
    if is_reduced:
        for swc_type_name in env.SWC_Types:
            for layer_name in env.layers:
                swc_layer_key = f"{swc_type_name}_{layer_name}_list"
                swc_layer_index_key = f"{swc_type_name}_{layer_name}_index"
                sec_list = getattr(cell, swc_layer_key, None)
                sec_index = getattr(cell, swc_layer_index_key, None)
                if sec_list is not None:
                    reduced_section_dict[swc_layer_key] = list(
                        zip(
                            np.asarray(sec_index, dtype=np.uint16),
                            list(sec_list),
                        )
                    )
        if len(reduced_section_dict) == 0:
            if hasattr(cell, "soma"):
                cell_soma = cell.soma
                if isinstance(cell_soma, list):
                    cell_soma = cell_soma[0]
            if hasattr(cell, "dend"):
                cell_dendrite = cell.dend

    syn_manager = env.synapse_manager

    make_syn_mech = make_unique_synapse_mech if unique else make_shared_synapse_mech

    mech_params = None
    if "default" in syn_params:
        mech_params = syn_params["default"]
        for mech_name, mech_param_config in mech_params.items():
            syn_manager.add_default_mechanism_parameters(
                gid, mech_name, mech_param_config, syn_ids
            )

    syn_count = 0
    nc_count = 0
    mech_count = 0
    sec_dx = 0.0
    sec_pos = 0.0
    current_sec_list = None
    current_sec_list_key = None
    for syn_id in syn_ids:
        syn = syn_manager.get_synapse(gid, syn_id)
        swc_type = syn.swc_type
        swc_type_name = swc_name_dict[swc_type]
        syn_loc = np.clip(syn.syn_loc, 0.05, 0.95)
        syn_section = syn.syn_section
        syn_layer = syn.syn_layer
        syn_layer_name = layer_name_dict[syn_layer]

        if swc_type in syn_params:
            mech_params = syn_params[swc_type]
        elif str(swc_type) in syn_params:
            mech_params = syn_params[str(swc_type)]
        elif "default" in syn_params:
            mech_params = syn_params["default"]

        if is_reduced:
            sec_index = 0
            sec_list_key = f"{swc_type_name}_{syn_layer_name}_list"

            if sec_list_key != current_sec_list_key:
                current_sec_list_key = sec_list_key
                current_sec_list = reduced_section_dict.get(sec_list_key, None)
                sec_pos = 0.0
                sec_dx = 0.0
            if current_sec_list is not None:
                if sec_pos >= 1.0:
                    sec_index, sec = current_sec_list.pop(0)
                    current_sec_list.append((sec_index, sec))
                    sec_pos = 0.0
                    sec_dx = 0.0
                sec_index, sec = current_sec_list[0]
                sec_dx = syn_loc - sec_dx
                sec_pos += sec_dx
            elif (swc_type == swc_type_soma) and (cell_soma is not None):
                sec = cell_soma
            elif (swc_type == swc_type_axon) and (cell_soma is not None):
                sec = cell_soma
            elif (swc_type == swc_type_ais) and (cell_soma is not None):
                sec = cell_soma
            elif (swc_type == swc_type_hill) and (cell_soma is not None):
                sec = cell_soma
            elif (swc_type == swc_type_apical) and (cell_dendrite is not None):
                sec = cell_dendrite
            elif (swc_type == swc_type_basal) and (cell_dendrite is not None):
                sec = cell_dendrite
            else:
                sec = py_sections[0]
        else:
            sec = py_sections[syn_section]

        if swc_type in syns_dict_by_type:
            syns_dict = syns_dict_by_type[swc_type]
        else:
            raise RuntimeError(
                f"insert_cell_syns: unsupported synapse SWC type {swc_type} for synapse {syn_id}"
            )

        for syn_name, params in mech_params.items():
            syn_mech = make_syn_mech(
                syn_name=syn_name,
                seg=sec(syn_loc),
                syns_dict=syns_dict,
                mech_names=syn_manager.syn_mech_names,
            )

            syn_manager.add_point_process(gid, syn_id, syn_name, syn_mech)

            mech_count += 1

            if insert_netcons or insert_vecstims:
                syn_pps = syn_manager.get_point_process(gid, syn_id, syn_name)
                this_vecstim = None
                this_nc = None
                if insert_vecstims:
                    this_nc, this_vecstim = mknetcon_vecstim(
                        syn_pps, delay=syn.source.delay
                    )
                    syn_manager.add_vecstim(
                        gid, syn_id, syn_name, this_vecstim, this_nc
                    )
                if insert_netcons:
                    if this_nc is None:
                        this_nc = mknetcon(
                            env.pc,
                            syn.source.gid,
                            syn_pps,
                            delay=syn.source.delay,
                        )
                    syn_manager.add_netcon(gid, syn_id, syn_name, this_nc)
                config_syn(
                    syn_name=syn_name,
                    rules=syn_manager.syn_param_rules,
                    mech_names=syn_manager.syn_mech_names,
                    syn=syn_mech,
                    nc=this_nc,
                    **params,
                )
                nc_count += 1
            else:
                config_syn(
                    syn_name=syn_name,
                    rules=syn_manager.syn_param_rules,
                    mech_names=syn_manager.syn_mech_names,
                    syn=syn_mech,
                    **params,
                )

        syn_count += 1

    if verbose:
        logger.info(
            f"insert_cell_syns: source: {presyn_name} target: {postsyn_name} cell {gid}: created {mech_count} mechanisms and {nc_count} "
            f"netcons for {syn_count} syn_ids"
        )

    return syn_count, mech_count, nc_count


def config_cell_syns(
    env: AbstractEnv,
    gid: int,
    postsyn_name: str,
    cell: Optional["HocObject"] = None,
    syn_ids: Optional[List[uint32]] = None,
    unique: Optional[bool] = None,
    insert: bool = False,
    insert_netcons: bool = False,
    insert_vecstims: bool = False,
    verbose: bool = False,
    throw_error: bool = False,
) -> Tuple[int, int, int]:
    """
    Configures synapses for a cell with parameter values from synapse storage.  If syn_ids=None, configures all synapses for the cell
    with the given gid.  If insert=True, iterate over sources and call
    insert_cell_syns.

    Args:
        env: Environment containing configuration
        gid: Cell GID
        postsyn_name: Name of postsynaptic population
        cell: Cell object (optional)
        syn_ids: List of synapse IDs to configure (if None, configure all)
        unique: Whether to use unique or shared synapses when inserting
        insert: Whether to insert synapses if not present
        insert_netcons: Whether to insert NetCons for new synapses
        insert_vecstims: Whether to insert VecStims for new NetCons
        verbose: Whether to print verbose output
        throw_error: Whether to throw errors for missing synapses/parameters

    Returns:
        Tuple of (syn_count, mech_count, nc_count)
    """
    rank = int(env.pc.id())
    syn_manager = env.synapse_manager

    # Get synapse configuration for this population
    synapse_config = env.celltypes[postsyn_name]["synapses"]

    # Determine if unique synapses are required
    if unique is None:
        unique = synapse_config.get("unique", False)

    # Get all synapse IDs if not specified
    if syn_ids is None:
        syn_ids = np.fromiter(syn_manager.synapse_ids(gid), dtype=np.uint32)

    # Handle synapse insertion if requested
    if insert:
        insert_start_time = time.time()

        # Verify cell access
        if (cell is None) and (not env.pc.gid_exists(gid)):
            raise RuntimeError(
                f"config_cell_syns: insert: cell with gid {gid} does not exist on rank {rank}"
            )

        if cell is None:
            cell = env.pc.gid2cell(gid)

        # Group synapses by source and insert
        source_syn_dict = syn_manager.partition_synapses_by_source(gid, syn_ids)

        for presyn_name, source_syns in source_syn_dict.items():
            if (presyn_name is not None) and (source_syns is not None):
                source_syn_ids = [x[0] for x in source_syns]

                # Insert synapses for this source
                syn_count, mech_count, nc_count = insert_cell_syns(
                    env,
                    gid,
                    postsyn_name,
                    presyn_name,
                    source_syn_ids,
                    unique=unique,
                    insert_netcons=insert_netcons,
                    insert_vecstims=insert_vecstims,
                    verbose=verbose,
                )

                if verbose:
                    logger.info(
                        f"config_cell_syns: population: {postsyn_name}; cell {gid}: "
                        f"inserted {mech_count} mechanisms for source {presyn_name}"
                    )

        if verbose:
            logger.info(
                f"config_cell_syns: population: {postsyn_name}; cell {gid}: "
                f"inserted mechanisms in {time.time() - insert_start_time:.2f} s"
            )

    # Configure existing synapses
    total_nc_count = 0
    total_mech_count = 0
    total_syn_count = 0

    # Group synapses by source for configuration
    source_syn_dict = syn_manager.partition_synapses_by_source(gid, syn_ids)

    # Get available synapse mechanism names
    syn_names = set(syn_manager.syn_mech_names.keys())

    for presyn_name, source_syns in source_syn_dict.items():
        if source_syns is None:
            continue

        # Process each synapse from this source
        for syn_id, syn in source_syns:
            total_syn_count += 1

            # Configure each mechanism type
            for syn_name in syn_names:
                # Check if point process exists
                if not syn_manager.has_point_process(gid, syn_id, syn_name):
                    if throw_error:
                        raise RuntimeError(
                            f"config_cell_syns: cell gid {gid} synapse {syn_id} does not have "
                            f"a point process for mechanism {syn_name}"
                        )
                    continue

                # Get point process and netcon
                point_process = syn_manager.get_point_process(
                    gid, syn_id, syn_name, throw_error=False
                )

                netcon = syn_manager.get_netcon(
                    gid, syn_id, syn_name, throw_error=False
                )

                # Get effective parameters with proper fallback
                try:
                    params = syn_manager.get_effective_mechanism_parameters(
                        gid, syn_id, syn_name, throw_error_on_missing=throw_error
                    )
                except RuntimeError as e:
                    if throw_error:
                        raise
                    logger.warning(
                        f"config_cell_syns: {str(e)}, skipping mechanism {syn_name} "
                        f"for synapse {syn_id}"
                    )
                    continue

                # Configure the mechanism with the parameters
                try:
                    (mech_set, nc_set) = config_syn(
                        syn_name=syn_name,
                        rules=syn_manager.syn_param_rules,
                        mech_names=syn_manager.syn_mech_names,
                        syn=point_process,
                        nc=netcon,
                        **params,
                    )

                    if mech_set:
                        total_mech_count += 1
                    if nc_set:
                        total_nc_count += 1

                except Exception as e:
                    if throw_error:
                        raise
                    logger.warning(
                        f"config_cell_syns: Error configuring mechanism {syn_name} "
                        f"for synapse {syn_id}: {str(e)}"
                    )

    if verbose:
        logger.info(
            f"config_cell_syns: target: {postsyn_name}; cell {gid}: "
            f"set parameters for {total_mech_count} syns and {total_nc_count} netcons "
            f"for {total_syn_count} syn_ids"
        )

    return total_syn_count, total_mech_count, total_nc_count


def config_syn(
    syn_name: str,
    rules: Dict[str, Dict[str, Union[str, List[str], Dict[str, int]]]],
    mech_names: Optional[Dict[str, str]] = None,
    syn: Optional["HocObject"] = None,
    nc: Optional["HocObject"] = None,
    **params,
) -> Tuple[bool, bool]:
    """
    Initializes synaptic and connection mechanisms with parameters.

    Args:
        syn_name: Synapse mechanism name
        rules: Dict to correctly parse params for specified mechanism
        mech_names: Dict to convert syn_name to mechanism name
        syn: Synaptic mechanism object
        nc: NetCon object
        **params: Parameter values

    Returns:
        Tuple of (mech_params_set, nc_params_set)

    Raises:
        RuntimeError: If required mechanism or parameter is missing
    """
    # Get mechanism name
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name

    # Check if mechanism rules exist
    if mech_name not in rules:
        raise RuntimeError(f"Mechanism rules not found for {mech_name}")

    mech_rules = rules[mech_name]

    # Check required components for parameters
    if syn is None and any(param in mech_rules["mech_params"] for param in params):
        raise RuntimeError(f"Synapse mechanism object required for {syn_name}")

    if nc is None and any(param in mech_rules["netcon_params"] for param in params):
        raise RuntimeError(f"NetCon object required for {syn_name}")

    nc_param = False
    mech_param = False

    # Apply parameters
    for param, val in params.items():
        # Skip None values
        if val is None:
            continue

        # Check parameter type
        if param in mech_rules["mech_params"]:
            if syn is not None:
                if isinstance(val, ExprClosure) and nc is not None:
                    # Handle expression closures
                    param_vals = []
                    for clos_param in val.parameters:
                        if hasattr(nc, clos_param):
                            param_vals.append(getattr(nc, clos_param))
                        else:
                            raise RuntimeError(
                                f"NetCon missing required attribute {clos_param} for {param}"
                            )
                    setattr(syn, param, val(*param_vals))
                else:
                    setattr(syn, param, val)
                mech_param = True
        elif param in mech_rules["netcon_params"]:
            if nc is not None:
                i = mech_rules["netcon_params"][param]
                if int(nc.wcnt()) <= i:
                    raise RuntimeError(
                        f"NetCon weight count ({nc.wcnt()}) too small for parameter {param}"
                    )

                if isinstance(val, ExprClosure):
                    # Handle expression closures
                    param_vals = []
                    for clos_param in val.parameters:
                        if hasattr(nc, clos_param):
                            param_vals.append(getattr(nc, clos_param))
                        else:
                            raise RuntimeError(
                                f"NetCon missing required attribute {clos_param} for {param}"
                            )
                    nc.weight[i] = val(*param_vals)
                else:
                    if isinstance(val, list):
                        if len(val) > 1:
                            raise RuntimeError(
                                f"NetCon attribute {param} has list of length > 1"
                            )
                        new_val = val[0]
                    else:
                        new_val = val
                    nc.weight[i] = new_val
                nc_param = True

        else:
            raise RuntimeError(f"Unknown parameter {param} for mechanism {syn_name}")

    return (mech_param, nc_param)


def syn_in_seg(
    syn_name: str,
    seg: Segment,
    syns_dict: DefaultDict[Section, DefaultDict[float, DefaultDict[str, "HocObject"]]],
) -> Optional["HocObject"]:
    """
    If a synaptic mechanism of the specified type already exists in the specified segment, it is returned. Otherwise,
    it returns None.
    :param syn_name: str
    :param seg: hoc segment
    :param syns_dict: nested defaultdict
    :return: hoc point process or None
    """
    sec = seg.sec
    for x in syns_dict[sec]:
        if sec(x) == seg:
            if syn_name in syns_dict[sec][x]:
                syn = syns_dict[sec][x][syn_name]
                return syn
    return None


def make_syn_mech(mech_name: str, seg: Segment) -> "HocObject":
    """
    TODO: Why was the hasattr(h, mech_name) check removed?
    :param mech_name: str (name of the point_process, specified by Env.synapse_manager.syn_mech_names)
    :param seg: hoc segment
    :return: hoc point process
    """
    syn = getattr(h, mech_name)(seg)
    return syn


def make_shared_synapse_mech(
    syn_name: str,
    seg: Segment,
    syns_dict: DefaultDict[Section, DefaultDict[float, DefaultDict[str, "HocObject"]]],
    mech_names: Optional[Dict[str, str]] = None,
) -> "HocObject":
    """
    If a synaptic mechanism of the specified type already exists in the specified segment, it is returned.
    Otherwise, this method creates one in the provided segment and adds it to the provided syns_dict before it is
    returned.

    :param syn_name: str
    :param seg: hoc segment
    :param syns_dict: nested defaultdict
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :return: hoc point process
    """
    syn_mech = syn_in_seg(syn_name, seg, syns_dict)
    if syn_mech is None:
        if mech_names is not None:
            mech_name = mech_names[syn_name]
        else:
            mech_name = syn_name
        syn_mech = make_syn_mech(mech_name, seg)
        syns_dict[seg.sec][seg.x][syn_name] = syn_mech
    return syn_mech


def make_unique_synapse_mech(syn_name, seg, syns_dict=None, mech_names=None):
    """
    Creates a new synapse in the provided segment, and returns it.

    :param syn_name: str
    :param seg: hoc segment
    :param syns_dict: nested defaultdict
    :param mech_names: map of synapse name to hoc mechanism name
    :return: hoc point process
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    syn_mech = make_syn_mech(mech_name, seg)
    return syn_mech


# ------------------------------- Methods to specify synaptic mechanisms  -------------------------------------------- #


def get_syn_mech_param(syn_name, rules, param_name, mech_names=None, nc=None):
    """

    :param syn_name: str
    :param rules: dict to correctly parse params for specified hoc mechanism
    :param param_name: str
    :param mech_names: dict to convert syn_name to hoc mechanism name
    :param nc: :class:'h.NetCon'
    """
    if mech_names is not None:
        mech_name = mech_names[syn_name]
    else:
        mech_name = syn_name
    if nc is not None:
        syn = nc.syn()
        if param_name in rules[mech_name]["mech_params"]:
            if syn is not None and hasattr(syn, param_name):
                return getattr(syn, param_name)
        elif param_name in rules[mech_name]["netcon_params"]:
            i = rules[mech_name]["netcon_params"][param_name]
            if nc.wcnt() >= i:
                return nc.weight[i]
    raise AttributeError(
        "get_syn_mech_param: problem setting attribute: {param_name} for synaptic mechanism: {mech_name}"
    )


def get_syn_filter_dict(
    env: AbstractEnv,
    rules: Dict[str, List[str]],
    convert: bool = False,
    check_valid: bool = True,
) -> Dict[str, List[int]]:
    """Used by modify_syn_param. Takes in a series of arguments and
    constructs a validated rules dictionary that specifies to which
    sets of synapses a rule applies. Values of filter queries are
    validated by the provided Env.

    :param env: :class:'Env'
    :param rules: dict
    :param convert: bool; whether to convert string values to enumerated type
    :return: dict

    """
    valid_filter_names = ["syn_types", "layers", "sources", "swc_types"]
    if check_valid:
        for name in rules:
            if name not in valid_filter_names:
                raise ValueError(
                    f"get_syn_filter_dict: unrecognized filter category: {name}"
                )
    rules_dict = copy.deepcopy(rules)
    syn_types = rules_dict.get("syn_types", None)
    swc_types = rules_dict.get("swc_types", None)
    layers = rules_dict.get("layers", None)
    sources = rules_dict.get("sources", None)
    if syn_types is not None:
        for i, syn_type in enumerate(syn_types):
            if syn_type not in env.Synapse_Types:
                raise ValueError(
                    f"get_syn_filter_dict: syn_type: {syn_type} not recognized by network configuration"
                )
            if convert:
                rules_dict["syn_types"][i] = env.Synapse_Types[syn_type]
    if swc_types is not None:
        for i, swc_type in enumerate(swc_types):
            if swc_type not in env.SWC_Types:
                raise ValueError(
                    f"get_syn_filter_dict: swc_type: {swc_type} not recognized by network configuration"
                )
            if convert:
                rules_dict["swc_types"][i] = env.SWC_Types[swc_type]
    if layers is not None:
        for i, layer in enumerate(layers):
            if layer not in env.layers:
                raise ValueError(
                    f"get_syn_filter_dict: layer: {layer} not recognized by network configuration"
                )
            if convert:
                rules_dict["layers"][i] = env.layers[layer]
    if sources is not None:
        source_idxs = []
        for i, source in enumerate(sources):
            if source not in env.Populations:
                raise ValueError(
                    f"get_syn_filter_dict: presynaptic population: {source} not recognized by network "
                    "configuration"
                )
            source_idxs.append(env.Populations[source])
        if convert:
            rules_dict["sources"] = source_idxs
    return rules_dict


def validate_syn_mech_param(env: AbstractEnv, syn_name, param_name):
    """

    :param env: :class:'Env'
    :param syn_name: str
    :param param_name: str
    :return: bool
    """
    syn_mech_names = env.synapse_manager.syn_mech_names
    if syn_name not in syn_mech_names:
        return False
    syn_param_rules = env.synapse_manager.syn_param_rules
    mech_name = syn_mech_names[syn_name]
    if mech_name not in syn_param_rules:
        return False
    if (
        "mech_params" in syn_param_rules[mech_name]
        and param_name in syn_param_rules[mech_name]["mech_params"]
    ):
        return True
    if (
        "netcon_params" in syn_param_rules[mech_name]
        and param_name in syn_param_rules[mech_name]["netcon_params"]
    ):
        return True
    return False


def modify_syn_param(
    cell,
    env: AbstractEnv,
    sec_type,
    syn_name,
    param_name=None,
    value=None,
    append=False,
    filters=None,
    update_targets=False,
    verbose=False,
):
    """Modifies a cell's mechanism dictionary to specify attributes of a
    synaptic mechanism by sec_type. This method is meant to be called
    manually during initial model specification, or during parameter
    optimization.

    Calls update_syn_mech_by_sec_type to set placeholder values in the
    syn_mech_attrs_dict of a SynapseManager object. If
    update_targets flag is True, the attributes of any target synaptic
    point_process and netcon objects that have been inserted will also
    be updated. Otherwise, they can be updated separately by calling

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param sec_type: str
    :param syn_name: str
    :param param_name: str
    :param value: float
    :param append: bool
    :param filters: dict
    :param update_targets: bool
    :param verbose: bool
    """
    if sec_type not in cell.nodes:
        raise ValueError(f"modify_syn_mech_param: sec_type: {sec_type} not in cell")
    if param_name is None:
        raise ValueError(
            f"modify_syn_mech_param: missing required parameter to modify synaptic mechanism: {syn_name} "
            f"in sec_type: {sec_type}"
        )
    if not validate_syn_mech_param(env, syn_name, param_name):
        raise ValueError(
            "modify_syn_mech_param: synaptic mechanism: "
            f"{syn_name} or parameter: {param_name} not recognized by network configuration"
        )
    if value is None:
        raise ValueError(
            f"modify_syn_mech_param: mechanism: {syn_name}; parameter: {param_name}; missing value for "
            "sec_type: {sec_type}"
        )

    rules = get_mech_rules_dict(cell, value=value)
    if filters is not None:
        syn_filters = get_syn_filter_dict(env, filters)
        rules["filters"] = syn_filters

    backup_mech_dict = copy.deepcopy(cell.mech_dict)

    mech_content = {param_name: rules}
    # No mechanisms have been specified in this type of section yet
    if sec_type not in cell.mech_dict:
        cell.mech_dict[sec_type] = {"synapses": {syn_name: mech_content}}
    # No synaptic mechanisms have been specified in this type of section yet
    elif "synapses" not in cell.mech_dict[sec_type]:
        cell.mech_dict[sec_type]["synapses"] = {syn_name: mech_content}
    # Synaptic mechanisms have been specified in this type of section, but not this syn_name
    elif syn_name not in cell.mech_dict[sec_type]["synapses"]:
        cell.mech_dict[sec_type]["synapses"][syn_name] = mech_content
    # This parameter of this syn_name has already been specified in this type of section, and the user wants to append
    # a new rule set
    elif param_name in cell.mech_dict[sec_type]["synapses"][syn_name] and append:
        cell.mech_dict[sec_type]["synapses"][syn_name][param_name].append(rules)
    # This syn_name has been specified, but not this parameter, or the user wants to replace an existing rule set
    else:
        cell.mech_dict[sec_type]["synapses"][syn_name][param_name] = rules

    try:
        update_syn_mech_by_sec_type(
            cell, env, sec_type, syn_name, mech_content, update_targets, verbose
        )
    except Exception as e:
        cell.mech_dict = copy.deepcopy(backup_mech_dict)
        traceback.print_exc(file=sys.stderr)
        logger.error(
            f"modify_syn_mech_param: gid {cell.gid}: "
            f"problem updating mechanism: {syn_name}; parameter: {param_name}; in sec_type: {sec_type}"
        )
        raise e


def update_syn_mech_by_sec_type(
    cell,
    env: AbstractEnv,
    sec_type,
    syn_name,
    mech_content,
    update_targets=False,
    verbose=False,
):
    """For the provided sec_type and synaptic mechanism, this method
    loops through the parameters specified in the mechanism
    dictionary, interprets the rules, and sets placeholder values in
    the syn_mech_attr_dict of a SynapseManager object.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param sec_type: str
    :param syn_name: str
    :param mech_content: dict
    :param update_targets: bool
    :param verbose: bool
    """
    for param_name, param_content in mech_content.items():
        update_syn_mech_param_by_sec_type(
            cell,
            env,
            sec_type,
            syn_name,
            param_name,
            param_content,
            update_targets,
            verbose,
        )


def update_syn_mech_param_by_sec_type(
    cell,
    env: AbstractEnv,
    sec_type,
    syn_name,
    param_name,
    rules,
    update_targets=False,
    verbose=False,
):
    """For the provided synaptic mechanism and parameter, this method
    loops through nodes of the provided sec_type, interprets the
    provided rules, and sets placeholder values in the
    syn_mech_attr_dict of a SynapseManager object.  If filter
    queries are provided, their values are converted to enumerated
    types.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param sec_type: str
    :param syn_name: str
    :param param_name: str
    :param rules: dict
    :param update_targets: bool
    :param verbose: bool
    """
    new_rules = copy.deepcopy(rules)
    if "filters" in new_rules:
        synapse_filters = get_syn_filter_dict(env, new_rules["filters"], convert=True)
        del new_rules["filters"]
    else:
        synapse_filters = {}

    is_reduced = False
    if hasattr(cell, "is_reduced"):
        is_reduced = cell.is_reduced
        if callable(is_reduced):
            is_reduced = is_reduced()
        if isinstance(is_reduced, float):
            is_reduced = is_reduced > 0.0

    if is_reduced:
        synapse_filters["swc_types"] = [env.SWC_Types[sec_type]]
        apply_syn_mech_rules(
            cell,
            env,
            syn_name,
            param_name,
            new_rules,
            synapse_filters=synapse_filters,
            update_targets=update_targets,
            verbose=verbose,
        )
    elif sec_type in cell.nodes:
        for node in cell.nodes[sec_type]:
            apply_syn_mech_rules(
                cell,
                env,
                syn_name,
                param_name,
                new_rules,
                node=node,
                synapse_filters=synapse_filters,
                update_targets=update_targets,
                verbose=verbose,
            )


def apply_syn_mech_rules(
    cell,
    env: AbstractEnv,
    syn_name,
    param_name,
    rules,
    node=None,
    syn_ids=None,
    synapse_filters=None,
    update_targets=False,
    verbose=False,
):
    """Provided a synaptic mechanism, a parameter, a node, a list of
    syn_ids, and a dict of rules. Interprets the provided rules and updates synaptic mechanisms.
    Calls set_syn_mech_param to sets parameter values in the syn_mech_attr_dict of a SynapseManager object.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param node: :class:'SectionNode'
    :param syn_ids: sequence of int
    :param syn_name: str
    :param param_name: str
    :param rules: dict
    :param update_targets: bool
    :param verbose: bool
    """
    if syn_ids is None:
        syn_manager = env.synapse_manager
        if synapse_filters is None:
            synapse_filters = {}
        if node is None:
            syn_ids = syn_manager.filter_synapse_ids(
                cell.gid, cache=env.cache_queries, **synapse_filters
            )
        else:
            syn_ids = syn_manager.filter_synapse_ids(
                cell.gid,
                syn_sections=[node.index],
                cache=env.cache_queries,
                **synapse_filters,
            )

    if "value" in rules:
        baseline = rules["value"]
    else:
        raise RuntimeError(
            "apply_syn_mech_rules: cannot set value of synaptic mechanism: "
            f"{syn_name} parameter: {param_name} in "
            f"sec_type: {node.type if node is not None else None}"
        )

    set_syn_mech_param(
        cell,
        env,
        node,
        syn_ids,
        syn_name,
        param_name,
        baseline,
        rules,
        update_targets,
        verbose,
    )


def set_syn_mech_param(
    cell,
    env: AbstractEnv,
    node,
    syn_ids,
    syn_name,
    param_name,
    baseline,
    rules,
    update_targets=False,
    verbose=False,
    batch_size=1000,
):
    """Provided a synaptic mechanism, a parameter, a node, a list of
    syn_ids, and a dict of rules. Sets placeholder values for each
    provided syn_id in the syn_mech_attr_dict of a SynapseManager
    object. If update_targets flag is True, the attributes
    of any target synaptic point_process and netcon objects that have
    been inserted will also be updated. Otherwise, they can be updated
    separately by calling config_syns.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param node: :class:'SectionNode'
    :param syn_ids: array of int
    :param syn_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    :param update_targets: bool
    :param verbose: bool
    """
    syn_manager = env.synapse_manager

    syn_id_iter = iter(syn_ids)
    while True:
        batch = list(itertools.islice(syn_id_iter, batch_size))
        if not batch:
            break

        for syn_id in batch:
            syn_manager.modify_mechanism_parameters(
                cell.population_name, cell.gid, syn_id, syn_name, {param_name: baseline}
            )

    if update_targets:
        syn_id_iter = iter(syn_ids)
        while True:
            batch_iterator = itertools.islice(syn_id_iter, batch_size)
            try:
                first_id = next(batch_iterator)
            except StopIteration:
                break

            batch_syn_ids = itertools.chain([first_id], batch_iterator)

            config_cell_syns(
                env,
                cell.gid,
                cell.population_name,
                syn_ids=batch_syn_ids,
                insert=False,
                verbose=verbose,
            )


def init_syn_mech_attrs(
    cell: BiophysCell,
    env: Optional[AbstractEnv] = None,
    reset_mech_dict: bool = False,
    update_targets: bool = False,
) -> None:
    """Consults a dictionary specifying parameters of NEURON synaptic mechanisms (point processes) for each type of
    section in a BiophysCell. Traverses through the tree of SHocNode nodes following order of inheritance. Calls
    update_syn_mech_by_sec_type to set placeholder values in the syn_mech_attrs_dict of a SynapseManager object. If
    update_targets flag is True, the attributes of any target synaptic point_process and netcon objects that have been
    inserted will also be updated. Otherwise, they can be updated separately by calling config_syns.

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param reset_mech_dict: bool
    :param update_targets: bool

    """
    if reset_mech_dict:
        cell.mech_dict = copy.deepcopy(cell.init_mech_dict)
    for sec_type in default_ordered_sec_types:
        if sec_type in cell.mech_dict and sec_type in cell.nodes:
            if cell.nodes[sec_type] and "synapses" in cell.mech_dict[sec_type]:
                for syn_name in cell.mech_dict[sec_type]["synapses"]:
                    update_syn_mech_by_sec_type(
                        cell,
                        env,
                        sec_type,
                        syn_name,
                        cell.mech_dict[sec_type]["synapses"][syn_name],
                        update_targets=update_targets,
                    )


def write_syn_spike_count(
    env: AbstractEnv,
    pop_name: str,
    output_path: str,
    filters: Optional[Dict[str, List[str]]] = None,
    syn_names: None = None,
    write_kwds: Dict[str, int] = {},
) -> None:
    """
    Writes spike counts per presynaptic source for each cell in the given population to a NeuroH5 file.
    Assumes that attributes have been set via config_syn.

    :param env: instance of env.Env
    :param pop_name: population name
    :param output_path: path to NeuroH5 file
    :param filters: optional filter for synapses
    """

    rank = int(env.pc.id())

    syn_manager = env.synapse_manager
    rules = syn_manager.syn_param_rules

    filters_dict = None
    if filters is not None:
        filters_dict = get_syn_filter_dict(env, filters, convert=True)

    if syn_names is None:
        syn_names = list(syn_manager.syn_name_index_dict.keys())

    output_dict = {
        syn_name: defaultdict(lambda: defaultdict(int)) for syn_name in syn_names
    }

    gids = []
    if pop_name in env.biophys_cells:
        gids = list(env.biophys_cells[pop_name].keys())

    for gid in gids:
        if filters_dict is None:
            syn_items = syn_manager.get_synapses(gid)
        else:
            syn_items = syn_manager.filter_synapses(gid, **filters_dict)
        logger.info(
            f"write_syn_mech_spike_counts: rank {rank}: population {pop_name}: gid {gid}"
        )

        for syn_id, syn in syn_items:
            source_population = syn.source.population
            syn_netcon_dict = syn_manager.pps_dict[gid][syn_id].netcon
            for syn_name in syn_names:
                mech_name = syn_manager.syn_mech_names[syn_name]
                syn_index = syn_manager.syn_name_index_dict[syn_name]
                if (
                    syn_index in syn_netcon_dict
                    and "count" in rules[mech_name]["netcon_state"]
                ):
                    count_index = rules[mech_name]["netcon_state"]["count"]
                    nc = syn_netcon_dict[syn_index]
                    spike_count = nc.weight[count_index]
                    output_dict[syn_name][gid][source_population] += spike_count

    for syn_name in sorted(output_dict):
        syn_attrs_dict = output_dict[syn_name]
        attr_dict = defaultdict(lambda: dict())

        for gid, gid_syn_spk_count_dict in syn_attrs_dict.items():
            for source_index, source_count in gid_syn_spk_count_dict.items():
                source_pop_name = syn_manager.presyn_names[source_index]
                attr_dict[gid][source_pop_name] = np.asarray(
                    [source_count], dtype="uint32"
                )

        logger.info(
            f"write_syn_mech_spike_counts: rank {rank}: population {pop_name}: writing mechanism {syn_name} spike counts for {len(attr_dict)} gids"
        )
        write_cell_attributes(
            output_path,
            pop_name,
            attr_dict,
            namespace=f"{syn_name} Spike Counts",
            **write_kwds,
        )
