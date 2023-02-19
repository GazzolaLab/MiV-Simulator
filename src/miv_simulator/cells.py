from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import collections
import copy
import math
import os

import networkx as nx
import numpy as np
from miv_simulator.utils import (
    AbstractEnv,
    Promise,
    get_module_logger,
    read_from_yaml,
    zip_longest,
)
from miv_simulator.utils.neuron import (
    BRKconfig,
    PRconfig,
    default_hoc_sec_lists,
    default_ordered_sec_types,
    h,
    init_nseg,
    load_cell_template,
    make_rec,
    reinit_diam,
)
from networkx.classes.digraph import DiGraph
from neuroh5.io import (
    read_cell_attribute_selection,
    read_graph_selection,
    read_tree_selection,
)
from nrn import Section
from numpy import ndarray, uint32

if TYPE_CHECKING:
    from neuron.hoc import HocObject

# This logger will inherit its settings from the root logger, created in env
logger = get_module_logger(__name__)


class SectionNode:
    def __init__(
        self,
        section_type: str,
        index: int,
        section: Section,
        content: Optional[Dict[str, ndarray]] = None,
    ) -> None:
        self.name = f"{section_type}{index}"
        self.section = section
        self.index = index
        self.section_type = section_type
        if content is None:
            content = dict()
        self.content = content

    @property
    def diam_bounds(self) -> None:
        return self.content.get("diam_bounds", None)

    def get_layer(self, x: None = None) -> ndarray:
        """
        NEURON sections can be assigned a layer type for convenience in order to later specify synaptic mechanisms and
        properties for each layer. If 3D points are used to specify cell morphology, each element in the list
        corresponds to the layer of the 3D point with the same index.
        :param x: float in [0, 1] : optional relative location in section
        :return: list or float or None
        """
        layer = self.content.get("layer", None)
        if x is None:
            result = layer
        else:
            for i in range(self.sec.n3d()):
                result = layer[i]
                if (self.sec.arc3d(i) / self.sec.L) >= x:
                    break
        return result

    @property
    def sec(self) -> Section:
        return self.section

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def make_neurotree_hoc_cell(
    template_class: "HocObject",
    gid: int = 0,
    neurotree_dict: Dict[
        str, Union[ndarray, Dict[str, Union[int, Dict[int, ndarray], ndarray]]]
    ] = {},
    section_content: Optional[bool] = None,
) -> Union["HocObject", Tuple["HocObject", Dict[int, Dict[str, ndarray]]]]:
    """
    :param template_class:
    :param local_id:
    :param gid:
    :param dataset_path:
    :param neurotree_dict:
    :return: hoc cell object
    """
    vx = neurotree_dict["x"]
    vy = neurotree_dict["y"]
    vz = neurotree_dict["z"]
    vradius = neurotree_dict["radius"]
    vlayer = neurotree_dict["layer"]
    vsection = neurotree_dict["section"]
    secnodes = neurotree_dict["section_topology"]["nodes"]
    vsrc = neurotree_dict["section_topology"]["src"]
    vdst = neurotree_dict["section_topology"]["dst"]
    vloc = neurotree_dict["section_topology"]["loc"]
    swc_type = neurotree_dict["swc_type"]
    cell = template_class(
        gid, secnodes, vlayer, vsrc, vdst, vloc, vx, vy, vz, vradius, swc_type
    )

    section_content_dict = dict()
    if section_content:
        if isinstance(section_content, dict):
            section_content_dict = section_content
        for section_index in secnodes:
            nodes = secnodes[section_index]
            node_layers = np.asarray([vlayer[n] for n in nodes], dtype=np.uint8)
            if not section_index in section_content_dict:
                section_content_dict[section_index] = dict()
            section_content_dict[section_index]["layer"] = node_layers

    if section_content:
        return cell, section_content_dict
    else:
        return cell


def make_hoc_cell(
    env: AbstractEnv,
    pop_name: str,
    gid: int,
    neurotree_dict: Union[
        bool,
        Dict[
            str,
            Union[ndarray, Dict[str, Union[int, Dict[int, ndarray], ndarray]]],
        ],
    ] = False,
) -> "HocObject":
    """

    :param env:
    :param gid:
    :param pop_name:
    :return:
    """
    dataset_path = env.dataset_path if env.dataset_path is not None else ""
    data_file_path = env.data_file_path
    template_name = env.celltypes[pop_name]["template"]
    assert hasattr(h, template_name)
    template_class = getattr(h, template_name)

    if neurotree_dict:
        hoc_cell = make_neurotree_hoc_cell(
            template_class, neurotree_dict=neurotree_dict, gid=gid
        )
    else:
        if (
            pop_name in env.cell_attribute_info
            and "Trees" in env.cell_attribute_info[pop_name]
        ):
            raise Exception(
                "make_hoc_cell: morphology for population %s gid: %i is not provided"
                % data_file_path,
                pop_name,
                gid,
            )
        else:
            hoc_cell = template_class(gid, dataset_path)

    env.biophys_cells[pop_name][gid] = hoc_cell

    return hoc_cell


def make_input_cell(
    env: AbstractEnv,
    gid: uint32,
    pop_id: int,
    input_source_dict: Dict[
        int, Dict[str, Union[Dict[Any, Any], Dict[int, Dict[str, ndarray]]]]
    ],
    spike_train_attr_name: str = "t",
) -> "HocObject":
    """
    Instantiates an input generator according to the given cell template.
    """

    input_sources = input_source_dict[pop_id]
    if "spiketrains" in input_sources:
        cell = h.VecStim()
        spk_attr_dict = input_sources["spiketrains"].get(gid, None)
        if spk_attr_dict is not None:
            spk_ts = spk_attr_dict[spike_train_attr_name]
            if len(spk_ts) > 0:
                cell.play(h.Vector(spk_ts))
    elif "generator" in input_sources:
        input_gen = input_sources["generator"]
        template_name = input_gen["template"]
        param_values = input_gen["params"]
        template = getattr(h, template_name)
        params = [
            param_values[p]
            for p in env.netclamp_config.template_params[template_name]
        ]
        cell = template(gid, *params)
    else:
        raise RuntimeError(
            "cells.make_input_cell: unrecognized input cell configuration"
        )

    return cell


def make_section_graph(neurotree_dict):
    """
    Creates a graph of sections that follows the topological organization of the given neuron.
    :param neurotree_dict:
    :return: NetworkX.DiGraph
    """
    import networkx as nx

    if "section_topology" in neurotree_dict:
        sec_src = neurotree_dict["section_topology"]["src"]
        sec_dst = neurotree_dict["section_topology"]["dst"]
        sec_loc = neurotree_dict["section_topology"]["loc"]
    else:
        sec_src = neurotree_dict["src"]
        sec_dst = neurotree_dict["dst"]
        sec_loc = []
        sec_nodes = {}
        pt_sections = neurotree_dict["sections"]
        pt_parents = neurotree_dict["parent"]
        sec_nodes = make_section_node_dict(neurotree_dict)
        for src, dst in zip_longest(sec_src, sec_dst):
            src_pts = sec_nodes[src]
            dst_pts = sec_nodes[dst]
            dst_parent = pt_parents[dst_pts[0]]
            loc = np.argwhere(src_pts == dst_parent)[0]
            sec_loc.append(loc)

    sec_graph = nx.DiGraph()
    for i, j, loc in zip(sec_src, sec_dst, sec_loc):
        sec_graph.add_edge(i, j, loc=loc)

    return sec_graph


class BRKneuron:
    """
    An implementation of a Booth-Rinzel-Kiehn-type reduced biophysical
    neuron model for simulation in NEURON.  Conforms to the same API
    as BiophysCell.

    """

    def __init__(
        self, gid, pop_name, env=None, cell_config=None, mech_dict=None
    ):
        """

        :param gid: int
        :param pop_name: str
        :param env: :class:'Env'
        :param cell_config: :namedtuple:'BRKconfig'
        """
        self._gid = gid
        self._pop_name = pop_name
        self.tree = nx.DiGraph()
        self.count = 0  # Keep track of number of nodes
        if env is not None:
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError(
                        "Warning! unexpected SWC Type definitions found in Env"
                    )
        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = None
        self.init_mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.mech_dict = dict(mech_dict) if mech_dict is not None else None

        self.random = np.random.RandomState()
        self.random.seed(self.gid)
        self.spike_detector = None
        self.spike_onset_delay = 0.0
        self.is_reduced = True
        if not isinstance(cell_config, BRKconfig):
            raise RuntimeError(
                "BRKneuron: argument cell_attrs must be of type BRKconfig"
            )

        param_dict = {
            "pp": cell_config.pp,
            "Ltotal": cell_config.Ltotal,
            "gc": cell_config.gc,
            "soma_gmax_Na": cell_config.soma_gmax_Na,
            "soma_gmax_K": cell_config.soma_gmax_K,
            "soma_gmax_KCa": cell_config.soma_gmax_KCa,
            "soma_gmax_CaN": cell_config.soma_gmax_CaN,
            "soma_f_Caconc": cell_config.soma_f_Caconc,
            "soma_kCa_Caconc": cell_config.soma_kCa_Caconc,
            "soma_alpha_Caconc": cell_config.soma_alpha_Caconc,
            "soma_g_pas": cell_config.soma_g_pas,
            "dend_gmax_CaL": cell_config.dend_gmax_CaL,
            "dend_gmax_CaN": cell_config.dend_gmax_CaN,
            "dend_gmax_KCa": cell_config.dend_gmax_KCa,
            "dend_g_pas": cell_config.dend_g_pas,
            "dend_f_Caconc": cell_config.dend_f_Caconc,
            "dend_kCa_Caconc": cell_config.dend_kCa_Caconc,
            "dend_alpha_Caconc": cell_config.dend_alpha_Caconc,
            "cm_ratio": cell_config.cm_ratio,
            "global_cm": cell_config.global_cm,
            "global_diam": cell_config.global_diam,
            "e_pas": cell_config.e_pas,
        }

        BRK_nrn = h.BRK_nrn(param_dict)
        BRK_nrn.soma.ic_constant = cell_config.ic_constant

        self.hoc_cell = BRK_nrn

        soma_node = insert_section_node(self, "soma", index=0, sec=BRK_nrn.soma)
        apical_node = insert_section_node(
            self, "apical", index=1, sec=BRK_nrn.dend
        )
        connect_nodes(
            self.tree, self.soma[0], self.apical[0], connect_hoc_sections=False
        )

        init_spike_detector(self, threshold=cell_config.V_threshold)

    def update_cell_attrs(self, **kwargs):
        for attr_name, attr_val in kwargs.items():
            if attr_name in BRKconfig._fields:
                setattr(self.hoc_cell, attr_name, attr_val)

    @property
    def gid(self):
        return self._gid

    @property
    def pop_name(self):
        return self._pop_name

    @property
    def soma(self):
        return self.nodes["soma"]

    @property
    def axon(self):
        return self.nodes["axon"]

    @property
    def basal(self):
        return self.nodes["basal"]

    @property
    def apical(self):
        return self.nodes["apical"]

    @property
    def trunk(self):
        return self.nodes["trunk"]

    @property
    def tuft(self):
        return self.nodes["tuft"]

    @property
    def spine(self):
        return self.nodes["spine_head"]

    @property
    def spine_head(self):
        return self.nodes["spine_head"]

    @property
    def spine_neck(self):
        return self.nodes["spine_neck"]

    @property
    def ais(self):
        return self.nodes["ais"]

    @property
    def hillock(self):
        return self.nodes["hillock"]


class PRneuron:
    """
    An implementation of a Pinsky-Rinzel-type reduced biophysical neuron model for simulation in NEURON.
    Conforms to the same API as BiophysCell.
    """

    def __init__(
        self, gid, pop_name, env=None, cell_config=None, mech_dict=None
    ):
        """

        :param gid: int
        :param pop_name: str
        :param env: :class:'Env'
        :param cell_config: :namedtuple:'PRconfig'
        """
        self._gid = gid
        self._pop_name = pop_name
        self.tree = nx.DiGraph()
        self.count = 0  # Keep track of number of nodes
        if env is not None:
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError(
                        "Warning! unexpected SWC Type definitions found in Env"
                    )
        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = None
        self.init_mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.mech_dict = dict(mech_dict) if mech_dict is not None else None

        self.random = np.random.RandomState()
        self.random.seed(self.gid)
        self.spike_detector = None
        self.spike_onset_delay = 0.0
        self.is_reduced = True
        if not isinstance(cell_config, PRconfig):
            raise RuntimeError(
                "PRneuron: argument cell_attrs must be of type PRconfig"
            )

        param_dict = {
            "pp": cell_config.pp,
            "Ltotal": cell_config.Ltotal,
            "gc": cell_config.gc,
            "soma_gmax_Na": cell_config.soma_gmax_Na,
            "soma_gmax_K": cell_config.soma_gmax_K,
            "soma_g_pas": cell_config.soma_g_pas,
            "dend_gmax_Ca": cell_config.dend_gmax_Ca,
            "dend_gmax_KCa": cell_config.dend_gmax_KCa,
            "dend_gmax_KAHP": cell_config.dend_gmax_KAHP,
            "dend_g_pas": cell_config.dend_g_pas,
            "dend_d_Caconc": cell_config.dend_d_Caconc,
            "cm_ratio": cell_config.cm_ratio,
            "global_cm": cell_config.global_cm,
            "global_diam": cell_config.global_diam,
            "e_pas": cell_config.e_pas,
        }

        PR_nrn = h.PR_nrn(param_dict)
        PR_nrn.soma.ic_constant = cell_config.ic_constant

        self.hoc_cell = PR_nrn

        soma_node = insert_section_node(self, "soma", index=0, sec=PR_nrn.soma)
        apical_node = insert_section_node(
            self, "apical", index=1, sec=PR_nrn.dend
        )
        connect_nodes(
            self.tree, self.soma[0], self.apical[0], connect_hoc_sections=False
        )

        init_spike_detector(self, threshold=cell_config.V_threshold)

    def update_cell_attrs(self, **kwargs):
        for attr_name, attr_val in kwargs.items():
            if attr_name in PRconfig._fields:
                setattr(self.hoc_cell, attr_name, attr_val)

    @property
    def gid(self):
        return self._gid

    @property
    def pop_name(self):
        return self._pop_name

    @property
    def soma(self):
        return self.nodes["soma"]

    @property
    def axon(self):
        return self.nodes["axon"]

    @property
    def basal(self):
        return self.nodes["basal"]

    @property
    def apical(self):
        return self.nodes["apical"]

    @property
    def trunk(self):
        return self.nodes["trunk"]

    @property
    def tuft(self):
        return self.nodes["tuft"]

    @property
    def spine(self):
        return self.nodes["spine_head"]

    @property
    def spine_head(self):
        return self.nodes["spine_head"]

    @property
    def spine_neck(self):
        return self.nodes["spine_neck"]

    @property
    def ais(self):
        return self.nodes["ais"]

    @property
    def hillock(self):
        return self.nodes["hillock"]


class SCneuron:
    """
    Single-compartment biophysical neuron model for simulation in NEURON.
    Conforms to the same API as BiophysCell.
    """

    def __init__(
        self,
        gid: int,
        pop_name: str,
        env: Optional[AbstractEnv] = None,
        mech_dict: Optional[
            Dict[
                str,
                Dict[str, Dict[str, Union[Dict[str, float], Dict[str, int]]]],
            ]
        ] = None,
        mech_file_path: None = None,
    ) -> None:
        """

        :param gid: int
        :param pop_name: str
        :param env: :class:'Env'
        :param mech_dict: dict
        """
        self._gid = gid
        self._pop_name = pop_name
        self.tree = nx.DiGraph()
        self.count = 0  # Keep track of number of nodes
        if env is not None:
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError(
                        "Warning! unexpected SWC Type definitions found in Env"
                    )
        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = mech_file_path
        self.init_mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.mech_dict = dict(mech_dict) if mech_dict is not None else None

        if (mech_dict is None) and (mech_file_path is not None):
            import_mech_dict_from_file(self, self.mech_file_path)
        elif mech_dict is None:
            # Allows for a cell to be created and for a new mech_dict to be constructed programmatically from scratch
            self.init_mech_dict = dict()
            self.mech_dict = dict()

        self.random = np.random.RandomState()
        self.random.seed(self.gid)
        self.spike_detector = None
        self.spike_onset_delay = 0.0
        self.is_reduced = True

        SC_nrn = h.SC_nrn()

        self.hoc_cell = SC_nrn
        soma_node = insert_section_node(self, "soma", index=0, sec=SC_nrn.soma)

        init_cable(self)
        init_spike_detector(self)

    @property
    def gid(self) -> int:
        return self._gid

    @property
    def pop_name(self):
        return self._pop_name

    @property
    def soma(self) -> List[SectionNode]:
        return self.nodes["soma"]

    @property
    def axon(self) -> List[Any]:
        return self.nodes["axon"]

    @property
    def basal(self):
        return self.nodes["basal"]

    @property
    def apical(self):
        return self.nodes["apical"]

    @property
    def trunk(self):
        return self.nodes["trunk"]

    @property
    def tuft(self):
        return self.nodes["tuft"]

    @property
    def spine(self):
        return self.nodes["spine_head"]

    @property
    def spine_head(self):
        return self.nodes["spine_head"]

    @property
    def spine_neck(self):
        return self.nodes["spine_neck"]

    @property
    def ais(self) -> List[Any]:
        return self.nodes["ais"]

    @property
    def hillock(self):
        return self.nodes["hillock"]


class BiophysCell:
    """
    A Python wrapper for neuronal cell objects specified in the NEURON language hoc.
    """

    def __init__(
        self,
        gid: int,
        population_name: str,
        hoc_cell: None = None,
        neurotree_dict: Optional[
            Dict[
                str,
                Union[
                    ndarray, Dict[str, Union[int, Dict[int, ndarray], ndarray]]
                ],
            ]
        ] = None,
        mech_file_path: None = None,
        mech_dict: None = None,
        env: Optional[AbstractEnv] = None,
    ) -> None:
        """

        :param gid: int
        :param population_name: str
        :param hoc_cell: :class:'h.hocObject': instance of a NEURON cell template
        :param mech_file_path: str (path)
        :param env: :class:'Env'
        """

        self._gid = gid
        self._population_name = population_name
        self.tree = nx.DiGraph()
        self.template_class = None

        if env is not None:
            self.template_class = env.template_dict[population_name]
            for sec_type in env.SWC_Types:
                if sec_type not in default_ordered_sec_types:
                    raise AttributeError(
                        "Unexpected SWC Type definitions found in Env"
                    )

        self.nodes = {key: [] for key in default_ordered_sec_types}
        self.mech_file_path = mech_file_path
        self.init_mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.mech_dict = dict(mech_dict) if mech_dict is not None else None
        self.spike_detector = None
        self.spike_onset_delay = 0.0
        if hoc_cell is not None:
            import_morphology_from_hoc(self, hoc_cell)
        elif neurotree_dict is not None:
            hoc_cell, section_content = make_neurotree_hoc_cell(
                self.template_class, gid, neurotree_dict, section_content=True
            )
            import_morphology_from_hoc(
                self, hoc_cell, section_content=section_content
            )
        if (mech_dict is None) and (mech_file_path is not None):
            import_mech_dict_from_file(self, self.mech_file_path)
        elif mech_dict is None:
            # Allows for a cell to be created and for a new mech_dict to be constructed programmatically from scratch
            self.init_mech_dict = dict()
            self.mech_dict = dict()
        self.hoc_cell = hoc_cell
        self.root = None
        sorted_nodes = list(nx.topological_sort(self.tree))
        if len(sorted_nodes) > 0:
            self.root = sorted_nodes[0]

        init_cable(self)
        init_spike_detector(self)

    @property
    def gid(self) -> int:
        return self._gid

    @property
    def population_name(self):
        return self._population_name

    @property
    def soma(self) -> List[SectionNode]:
        return self.nodes["soma"]

    @property
    def axon(self) -> List[Union[Any, SectionNode]]:
        return self.nodes["axon"]

    @property
    def basal(self):
        return self.nodes["basal"]

    @property
    def apical(self):
        return self.nodes["apical"]

    @property
    def trunk(self):
        return self.nodes["trunk"]

    @property
    def tuft(self):
        return self.nodes["tuft"]

    @property
    def spine(self):
        return self.nodes["spine"]

    @property
    def ais(self) -> List[Union[Any, SectionNode]]:
        return self.nodes["ais"]

    @property
    def hillock(self):
        return self.nodes["hillock"]


def get_distance_to_node(
    cell: BiophysCell,
    node: SectionNode,
    root: Optional[SectionNode] = None,
    loc: Optional[float] = None,
) -> float:
    """
    Returns the distance from the given location on the given node to its connection with a root node.
    :param node: int
    :param loc: float
    :return: int or float
    """
    if root is None:
        root = cell.root

    length = 0.0
    if (node is root) or (root is None) or (node is None):
        return length
    if loc is not None:
        length += loc * node.section.L
    rpath = list(
        reversed(nx.shortest_path(cell.tree, source=root, target=node))
    )
    while not len(rpath) == 0:
        node = rpath.pop()
        if not len(rpath) == 0:
            parent = rpath[-1]
            e = cell.tree.get_edge_data(node, parent)
            loc = e["parent_loc"]
            length += loc * parent.section.L
    return length


def get_node_parent(
    cell: BiophysCell, node: SectionNode, return_edge_data: bool = False
) -> Union[Tuple[None, None], Tuple[SectionNode, Dict[str, float]]]:
    predecessors = list(cell.tree.predecessors(node))
    if len(predecessors) > 1:
        raise RuntimeError(
            f"get_node_parent: node {node.name} {node.sec.hname()} has more than one parent"
        )
    parent = None
    edge_data = None
    if len(predecessors) == 1:
        parent = next(iter(predecessors))
        edge_data = cell.tree.get_edge_data(parent, node)
    if return_edge_data:
        return parent, edge_data
    else:
        return parent


def get_node_children(
    cell: BiophysCell, node: SectionNode, return_edge_data: bool = False
) -> List[Union[Any, SectionNode]]:
    successors = cell.tree.successors(node)
    edge_data = []
    children = []
    for d in successors:
        children.append(d)
        edge_data.append(cell.tree.get_edge_data(node, d))
    if return_edge_data:
        return children, edge_data
    else:
        return children


def insert_section_node(
    cell: Union[BiophysCell, SCneuron, PRneuron, BRKneuron],
    section_type: str,
    index: int,
    sec: Section,
    content: Optional[Dict[str, ndarray]] = None,
) -> SectionNode:
    node = SectionNode(section_type, index, sec, content=content)
    if cell.tree.has_node(node) or node in cell.nodes[section_type]:
        raise RuntimeError(
            f"insert_section: section index {index} already exists in cell {self.gid}"
        )
    cell.tree.add_node(node)
    cell.nodes[section_type].append(node)
    return node


def insert_section_tree(
    cell: BiophysCell,
    sec_list: List[Section],
    sec_dict: Dict[Section, Dict[str, Union[str, int, Dict[str, ndarray]]]],
    parent: None = None,
    connect_hoc_sections: bool = False,
) -> None:
    sec_stack = []
    for sec in sec_list:
        sec_stack.append((parent, sec))
    while not len(sec_stack) == 0:
        sec_parent, sec = sec_stack.pop()
        sec_info = sec_dict[sec]
        sec_children = sec.children()
        sec_node = insert_section_node(
            cell,
            sec_info["section_type"],
            sec_info["section_index"],
            sec,
            content=sec_info.get("section_content", None),
        )
        for child in sec_children:
            sec_stack.append((sec_node, child))
        if sec_parent is not None:
            cell.tree = connect_nodes(
                cell.tree,
                sec_parent,
                sec_node,
                connect_hoc_sections=connect_hoc_sections,
            )


def connect_nodes(
    tree: DiGraph,
    parent: SectionNode,
    child: SectionNode,
    parent_loc: float = 1.0,
    child_loc: float = 0.0,
    connect_hoc_sections: bool = False,
) -> DiGraph:
    """
    Connects the given section node to a parent node, and if specified, establishes a connection between their associated
    hoc sections.
    :param parent: SectionNode
    :param child: SectionNode
    :param parent_loc: float in [0,1] : connect to this end of the parent hoc section
    :param child_loc: float in [0,1] : connect this end of the child hoc section
    :param connect_hoc_sections: bool
    """
    tree.add_edge(parent, child, parent_loc=parent_loc, child_loc=child_loc)
    if connect_hoc_sections:
        child.section.connect(parent.section, parent_loc, child_loc)
    return tree


def import_morphology_from_hoc(
    cell: BiophysCell,
    hoc_cell: "HocObject",
    section_content: Optional[Dict[int, Dict[str, ndarray]]] = None,
) -> None:
    """
    Append sections from an existing instance of a NEURON cell template to a Python cell wrapper.
    :param cell: :class:'BiophysCell'
    :param hoc_cell: :class:'h.hocObject': instance of a NEURON cell template

    """
    sec_info_dict = {}
    root_sec = None
    for sec_type, sec_index_list in default_hoc_sec_lists.items():
        hoc_sec_attr_name = sec_type
        if not hasattr(hoc_cell, hoc_sec_attr_name):
            hoc_sec_attr_name = f"{sec_type}_list"
        if hasattr(hoc_cell, hoc_sec_attr_name) and (
            getattr(hoc_cell, hoc_sec_attr_name) is not None
        ):
            sec_list = list(getattr(hoc_cell, hoc_sec_attr_name))
            if hasattr(hoc_cell, sec_index_list):
                sec_indexes = list(getattr(hoc_cell, sec_index_list))
            else:
                raise AttributeError(
                    "import_morphology_from_hoc: %s is not an attribute of the hoc cell"
                    % sec_index_list
                )
            if sec_type == "soma":
                root_sec = sec_list[0]
            for sec, index in zip(sec_list, sec_indexes):
                if section_content is not None:
                    sec_info_dict[sec] = {
                        "section_type": sec_type,
                        "section_index": int(index),
                        "section_content": section_content[index],
                    }
                else:
                    sec_info_dict[sec] = {
                        "section_type": sec_type,
                        "section_index": int(index),
                    }
    if root_sec:
        insert_section_tree(cell, [root_sec], sec_info_dict)
    else:
        raise RuntimeError(
            f"import_morphology_from_hoc: unable to locate root section"
        )


def import_mech_dict_from_file(cell, mech_file_path=None):
    """
    Imports from a .yaml file a dictionary specifying parameters of NEURON cable properties, density mechanisms, and
    point processes for each type of section in a BiophysCell.
    :param cell: :class:'BiophysCell'
    :param mech_file_path: str (path)
    """
    if mech_file_path is None:
        if cell.mech_file_path is None:
            raise ValueError(
                "import_mech_dict_from_file: missing mech_file_path"
            )
        elif not os.path.isfile(cell.mech_file_path):
            raise OSError(
                "import_mech_dict_from_file: invalid mech_file_path: %s"
                % cell.mech_file_path
            )
    elif not os.path.isfile(mech_file_path):
        raise OSError(
            f"import_mech_dict_from_file: invalid mech_file_path: {mech_file_path}"
        )
    else:
        cell.mech_file_path = mech_file_path
    cell.init_mech_dict = read_from_yaml(cell.mech_file_path)
    cell.mech_dict = copy.deepcopy(cell.init_mech_dict)


def init_cable(
    cell: Union[BiophysCell, SCneuron], verbose: bool = False
) -> None:
    for sec_type in cell.nodes:
        for node in cell.nodes[sec_type]:
            reset_cable_by_node(cell, node, verbose=verbose)


def reset_cable_by_node(
    cell: Union[BiophysCell, SCneuron], node: SectionNode, verbose: bool = True
) -> None:
    """
    Consults a dictionary specifying parameters of NEURON cable properties such as axial resistance ('Ra'),
    membrane specific capacitance ('cm'), and a spatial resolution parameter to specify the number of separate
    segments per section in a BiophysCell
    :param cell: :class:'BiophysCell'
    :param node_index: int
    :param verbose: bool
    """
    sec_type = node.section_type
    if sec_type in cell.mech_dict and "cable" in cell.mech_dict[sec_type]:
        mech_content = cell.mech_dict[sec_type]["cable"]
        if mech_content is not None:
            update_mechanism_by_node(
                cell, node, "cable", mech_content, verbose=verbose
            )
    else:
        init_nseg(node.section, verbose=verbose)
        reinit_diam(node.section, node.diam_bounds)


def connect2target(
    cell: Union[BiophysCell, SCneuron],
    sec: Section,
    loc: float = 1.0,
    param: str = "_ref_v",
    delay: Optional[float] = None,
    weight: None = None,
    threshold: Optional[int] = None,
    target: None = None,
) -> "HocObject":
    """
    Converts analog voltage in the specified section to digital spike output. Initializes and returns an h.NetCon
    object with voltage as a reference parameter connected to the specified target.
    :param cell: :class:'BiophysCell'
    :param sec: :class:'h.Section'
    :param loc: float
    :param param: str
    :param delay: float
    :param weight: float
    :param threshold: float
    :param target: object that can receive spikes
    :return: :class:'h.NetCon'
    """
    if cell.spike_detector is not None:
        if delay is None:
            delay = cell.spike_detector.delay
        if weight is None:
            weight = cell.spike_detector.weight[0]
        if threshold is None:
            threshold = cell.spike_detector.threshold
    else:
        if delay is None:
            delay = 0.0
        if weight is None:
            weight = 1.0
        if threshold is None:
            threshold = -30.0
    ps = getattr(sec(loc), param)
    this_netcon = h.NetCon(ps, target, sec=sec)
    this_netcon.delay = delay
    this_netcon.weight[0] = weight
    this_netcon.threshold = threshold
    return this_netcon


def init_spike_detector(
    cell: Union[BiophysCell, SCneuron],
    node: None = None,
    distance: float = 100.0,
    threshold: int = -30,
    delay: float = 0.05,
    onset_delay: float = 0.0,
    loc: float = 0.5,
) -> "HocObject":
    """
    Initializes the spike detector in the given cell according to the
    given arguments or a spike detector configuration of the mechanism
    dictionary of the cell, if one exists.

    :param cell: :class:'BiophysCell'
    :param node: :class:'SectionNode'
    :param distance: float
    :param threshold: float
    :param delay: float
    :param onset_delay: float
    :param loc: float
    """
    if cell.mech_dict is not None:
        if "spike detector" in cell.mech_dict:
            config = cell.mech_dict["spike detector"]
            node = getattr(cell, config["section"])[0]
            loc = config["loc"]
            distance = config["distance"]
            threshold = config["threshold"]
            delay = config["delay"]
            onset_delay = config["onset delay"]

    if node is None:
        if cell.axon:
            for node in cell.axon:
                sec_seg_locs = [seg.x for seg in node.sec]
                for loc in sec_seg_locs:
                    if (
                        get_distance_to_node(
                            cell, node, root=cell.root, loc=loc
                        )
                        >= distance
                    ):
                        break
                else:
                    continue
                break
            else:
                node = cell.axon[-1]
                loc = 1.0
        elif cell.ais:
            node = cell.ais[0]
        elif cell.soma:
            node = cell.soma[-1]
        else:
            raise RuntimeError(
                "init_spike_detector: cell has neither soma nor axon compartment"
            )

    cell.spike_detector = connect2target(
        cell, node.section, loc=loc, delay=delay, threshold=threshold
    )

    cell.onset_delay = onset_delay

    return cell.spike_detector


def update_biophysics_by_sec_type(
    cell: BiophysCell,
    sec_type: str,
    reset_cable: bool = False,
    verbose: bool = False,
) -> None:
    """
    This method loops through all sections of the specified type, and consults the mechanism dictionary to update
    mechanism properties. If the reset_cable flag is True, cable parameters are re-initialized first, then the
    ion channel mechanisms are updated.

    :param cell: :class:'BiophysCell'
    :param sec_type: str
    :param reset_cable: bool
    """
    if sec_type in cell.nodes:
        if reset_cable:
            # cable properties must be set first, as they can change nseg, which will affect insertion of membrane
            # mechanism gradients
            for node in cell.nodes[sec_type]:
                reset_cable_by_node(cell, node, verbose=verbose)
        if sec_type in cell.mech_dict:
            for node in cell.nodes[sec_type]:
                for mech_name in (
                    mech_name
                    for mech_name in cell.mech_dict[sec_type]
                    if mech_name not in ["cable", "synapses"]
                ):
                    update_mechanism_by_node(
                        cell,
                        node,
                        mech_name,
                        cell.mech_dict[sec_type][mech_name],
                    )


def update_mechanism_by_node(
    cell: BiophysCell,
    node: SectionNode,
    mech_name: str,
    mech_content: Optional[
        Dict[str, Union[Dict[str, float], Dict[str, int]]]
    ] = None,
    verbose: bool = True,
) -> None:
    """
    This method loops through all the parameters for a single mechanism specified in the mechanism dictionary and
    calls apply_mech_rules to interpret the rules and set the values for the given node.
    :param cell: :class:'BiophysCell'
    :param node: :class:'SectionNode'
    :param mech_name: str
    :param mech_content: list of dict
    :param verbose: bool
    """
    if mech_content is not None:
        # process either a dict specifying rules for a single parameter
        if isinstance(mech_content, dict):
            point_process_loc = mech_content.get("loc", None)
            for param_name in mech_content:
                if param_name != "loc":
                    apply_mech_rules(
                        cell,
                        node,
                        mech_name,
                        param_name,
                        mech_content[param_name],
                        point_process_loc=point_process_loc,
                        verbose=verbose,
                    )
        else:
            raise RuntimeError(
                "update_mechanism_by_node: unknown mechanism rule structure"
            )

    else:
        node.section.insert(mech_name)


def apply_mech_rules(
    cell: BiophysCell,
    node: SectionNode,
    mech_name: str,
    param_name: str,
    rules: Dict[str, Union[int, float]],
    point_process_loc: Optional[float] = None,
    verbose: bool = True,
) -> None:
    """
    Provided a membrane density mechanism, a parameter, a node, and a dict of rules, interprets the provided rules,
    and applies resulting parameter values to mechanisms in the corresponding section.

    :param cell: :class:'BiophysCell'
    :param node: :class:'SectionNode'
    :param mech_name: str
    :param param_name: str
    :param rules: dict
    :param point_process_loc: location in section if mechanism is a point process
    :param verbose: bool
    """
    baseline = rules.get("value", None)
    if mech_name == "cable":
        setattr(node.sec, param_name, baseline)
        init_nseg(node.section, verbose=verbose)
        reinit_diam(node.section, node.diam_bounds)
    else:
        set_mech_param(
            cell,
            node,
            mech_name,
            param_name,
            baseline,
            rules,
            point_process_loc=point_process_loc,
        )


def set_mech_param(
    cell: BiophysCell,
    node: SectionNode,
    mech_name: str,
    param_name: str,
    baseline: Union[int, float],
    rules: Dict[str, Union[int, float]],
    point_process_loc: Optional[float] = None,
) -> None:
    """

    :param node: :class:'SectionNode'
    :param mech_name: str
    :param param_name: str
    :param baseline: float
    :param rules: dict
    """
    if point_process_loc is not None:
        pp_dict = node.content.get(mech_name, None)
        if pp_dict is None:
            pp_dict = {}
            node.content[mech_name] = pp_dict
        pp = pp_dict.get(point_process_loc, None)
        if pp is None:
            pp = getattr(h, mech_name)(node.sec(point_process_loc))
            pp_dict[point_process_loc] = pp
            hoc_cell = getattr(cell, "hoc_cell", None)
            if hoc_cell is not None:
                pps = getattr(hoc_cell, "pps", None)
                if pps is not None:
                    pps.append(pp)
        setattr(pp, param_name, baseline)
    else:
        try:
            node.sec.insert(mech_name)
        except Exception:
            raise RuntimeError(
                f"set_mech_param: unable to insert mechanism: {mech_name} cell: {cell} "
                f"in sec_type: {node.section_type}"
            )
        setattr(node.sec, f"{param_name}_{mech_name}", baseline)


def filter_nodes(
    cell: BiophysCell,
    sections: None = None,
    layers: None = None,
    swc_types: Optional[List[str]] = None,
) -> List[SectionNode]:
    """
    Returns a subset of the nodes of the given cell according to the given criteria.

    :param cell:
    :param sections: sequence of int
    :param layers: list of enumerated type: layer
    :param swc_types: list of enumerated type: swc_type
    :return: list of nodes
    """
    matches = lambda items: all(
        map(
            lambda query_item: (query_item[0] is None)
            or (query_item[1] in query_item[0]),
            items,
        )
    )

    nodes = []
    if swc_types is None:
        sections = sorted(cell.nodes.keys())
    for swc_type in swc_types:
        nodes.extend(cell.nodes[swc_type])

    result = [
        v
        for v in nodes
        if matches([(layers, v.get_layer()), (sections, v.sec)])
    ]

    return result


def report_topology(
    env: AbstractEnv, cell: BiophysCell, node: Optional[SectionNode] = None
) -> None:
    """
    Traverse a cell and report topology and number of synapses.
    :param cell:
    :param env:
    :param node:
    """
    if node is None:
        node = cell.root
    syn_attrs = env.synapse_attributes
    num_exc_syns = len(
        syn_attrs.filter_synapses(
            cell.gid,
            syn_sections=[node.index],
            syn_types=[env.Synapse_Types["excitatory"]],
        )
    )
    num_inh_syns = len(
        syn_attrs.filter_synapses(
            cell.gid,
            syn_sections=[node.index],
            syn_types=[env.Synapse_Types["inhibitory"]],
        )
    )

    diams_str = ", ".join(
        f"{node.sec.diam3d(i):.2f}" for i in range(node.sec.n3d())
    )
    report = (
        f"node: {node.name}, L: {node.sec.L:.1f}, diams: [{diams_str}], nseg: {node.sec.nseg}, "
        f"children: {len(node.sec.children())}, exc_syns: {num_exc_syns}, inh_syns: {num_inh_syns}"
    )
    parent, edge_data = get_node_parent(cell, node, return_edge_data=True)
    if parent is not None:
        report += f", parent: {parent.name}; connection_loc: {edge_data['parent_loc']:.1f}"
    logger.info(report)
    children = get_node_children(cell, node)
    for child in children:
        report_topology(env, cell, child)


def make_morph_graph(biophys_cell, node_filters={}):
    """
    Creates a graph of 3d points that follows the morphological organization of the given neuron.
    :param neurotree_dict:
    :return: NetworkX.DiGraph
    """
    import networkx as nx

    nodes = filter_nodes(biophys_cell, **node_filters)
    tree = biophys_cell.tree

    sec_layers = {}
    src_sec = []
    dst_sec = []
    connection_locs = []
    pt_xs = []
    pt_ys = []
    pt_zs = []
    pt_locs = []
    pt_idxs = []
    pt_layers = []
    pt_idx = 0
    sec_pts = collections.defaultdict(list)

    for node in nodes:
        sec = node.sec
        nn = sec.n3d()
        L = sec.L
        for i in range(nn):
            pt_xs.append(sec.x3d(i))
            pt_ys.append(sec.y3d(i))
            pt_zs.append(sec.z3d(i))
            loc = sec.arc3d(i) / L
            pt_locs.append(loc)
            pt_layers.append(node.get_layer(loc))
            pt_idxs.append(pt_idx)
            sec_pts[node.index].append(pt_idx)
            pt_idx += 1

        for child in tree.successors(node):
            src_sec.append(node.index)
            dst_sec.append(child.index)
            connection_locs.append(h.parent_connection(sec=child.sec))

    sec_pt_idxs = {}
    edges = []
    for sec, pts in sec_pts.items():
        sec_pt_idxs[pts[0]] = sec
        for i in range(1, len(pts)):
            sec_pt_idxs[pts[i]] = sec
            src_pt = pts[i - 1]
            dst_pt = pts[i]
            edges.append((src_pt, dst_pt))

    for s, d, parent_loc in zip(src_sec, dst_sec, connection_locs):
        for src_pt in sec_pts[s]:
            if pt_locs[src_pt] >= parent_loc:
                break
        dst_pt = sec_pts[d][0]
        edges.append((src_pt, dst_pt))

    morph_graph = nx.Graph()
    morph_graph.add_nodes_from(
        [
            (
                i,
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "sec": sec_pt_idxs[i],
                    "loc": loc,
                    "layer": layer,
                },
            )
            for (i, x, y, z, loc, layer) in zip(
                range(len(pt_idxs)), pt_xs, pt_ys, pt_zs, pt_locs, pt_layers
            )
        ]
    )
    for i, j in edges:
        morph_graph.add_edge(i, j)

    return morph_graph


def load_biophys_cell_dicts(
    env: AbstractEnv,
    pop_name: str,
    gid_set: Set[int],
    data_file_path: None = None,
    load_connections: bool = True,
    validate_tree: bool = True,
) -> Dict[
    int,
    Dict[
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
    ],
]:
    """
    Loads the data necessary to instantiate BiophysCell into the given dictionary.

    :param env: an instance of env.Env
    :param pop_name: population name
    :param gid: gid
    :param data_file_path: str or None
    :param load_connections: bool
    :param validate_tree: bool

    Environment can be instantiated as:
    env = Env(config_file, template_paths, dataset_prefix)
    :param template_paths: str; colon-separated list of paths to directories containing hoc cell templates
    :param dataset_prefix: str; path to directory containing required neuroh5 data files
    """

    synapse_config = env.celltypes[pop_name]["synapses"]

    has_weights = False
    weights_config = None
    if "weights" in synapse_config:
        has_weights = True
        weights_config = synapse_config["weights"]

    ## Loads cell morphological data, synaptic attributes and connection data

    tree_dicts = {}
    synapses_dicts = {}
    weight_dicts = {}
    connection_graphs = {gid: {pop_name: {}} for gid in gid_set}
    graph_attr_info = None

    gid_list = list(gid_set)
    tree_attr_iter, _ = read_tree_selection(
        env.data_file_path,
        pop_name,
        gid_list,
        comm=env.comm,
        topology=True,
        validate=validate_tree,
    )
    for gid, tree_dict in tree_attr_iter:
        tree_dicts[gid] = tree_dict

    if load_connections:
        synapses_iter = read_cell_attribute_selection(
            env.data_file_path,
            pop_name,
            gid_list,
            "Synapse Attributes",
            mask={
                "syn_ids",
                "syn_locs",
                "syn_secs",
                "syn_layers",
                "syn_types",
                "swc_types",
            },
            comm=env.comm,
        )
        for gid, attr_dict in synapses_iter:
            synapses_dicts[gid] = attr_dict

        if has_weights:
            for config in weights_config:
                weights_namespaces = config["namespace"]
                cell_weights_iters = [
                    read_cell_attribute_selection(
                        env.data_file_path,
                        pop_name,
                        gid_list,
                        weights_namespace,
                        comm=env.comm,
                    )
                    for weights_namespace in weights_namespaces
                ]
                for weights_namespace, cell_weights_iter in zip_longest(
                    weights_namespaces, cell_weights_iters
                ):
                    for gid, cell_weights_dict in cell_weights_iter:
                        this_weights_dict = weight_dicts.get(gid, {})
                        this_weights_dict[weights_namespace] = cell_weights_dict
                        weight_dicts[gid] = this_weights_dict

        graph, graph_attr_info = read_graph_selection(
            file_name=env.connectivity_file_path,
            selection=gid_list,
            namespaces=["Synapses", "Connections"],
            comm=env.comm,
        )
        if pop_name in graph:
            for presyn_name in graph[pop_name].keys():
                edge_iter = graph[pop_name][presyn_name]
                for postsyn_gid, edges in edge_iter:
                    connection_graphs[postsyn_gid][pop_name][presyn_name] = [
                        (postsyn_gid, edges)
                    ]

    cell_dicts = {}
    for gid in gid_set:
        this_cell_dict = {}

        tree_dict = tree_dicts[gid]
        this_cell_dict["morph"] = tree_dict

        if load_connections:
            synapses_dict = synapses_dicts[gid]
            weight_dict = weight_dicts.get(gid, None)
            connection_graph = connection_graphs[gid]
            this_cell_dict["synapse"] = synapses_dict
            this_cell_dict["connectivity"] = connection_graph, graph_attr_info
            this_cell_dict["weight"] = weight_dict
        cell_dicts[gid] = this_cell_dict

    return cell_dicts


def init_circuit_context(
    env: AbstractEnv,
    pop_name: str,
    gid: int,
    load_edges: bool = False,
    connection_graph: Optional[
        Tuple[
            Dict[
                str,
                Dict[
                    str,
                    List[Tuple[int, Tuple[ndarray, Dict[str, List[ndarray]]]]],
                ],
            ],
            Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
        ]
    ] = None,
    load_weights: bool = False,
    weight_dict: None = None,
    load_synapses: bool = False,
    synapses_dict: Optional[Dict[str, ndarray]] = None,
    set_edge_delays: bool = True,
    **kwargs,
) -> None:
    syn_attrs = env.synapse_attributes
    synapse_config = env.celltypes[pop_name]["synapses"]

    has_weights = False
    weight_config = []
    if "weights" in synapse_config:
        has_weights = True
        weight_config = synapse_config["weights"]

    init_synapses = False
    init_weights = False
    init_edges = False
    if load_edges or (connection_graph is not None):
        init_synapses = True
        init_edges = True
    if has_weights and (load_weights or (weight_dict is not None)):
        init_synapses = True
        init_weights = True
    if load_synapses or (synapses_dict is not None):
        init_synapses = True

    if init_synapses:
        if synapses_dict is not None:
            syn_attrs.init_syn_id_attrs(gid, **synapses_dict)
        elif load_synapses or load_edges:
            if (pop_name in env.cell_attribute_info) and (
                "Synapse Attributes" in env.cell_attribute_info[pop_name]
            ):
                synapses_iter = read_cell_attribute_selection(
                    env.data_file_path,
                    pop_name,
                    [gid],
                    "Synapse Attributes",
                    mask={
                        "syn_ids",
                        "syn_locs",
                        "syn_secs",
                        "syn_layers",
                        "syn_types",
                        "swc_types",
                    },
                    comm=env.comm,
                )
                syn_attrs.init_syn_id_attrs_from_iter(synapses_iter)
            else:
                raise RuntimeError(
                    "init_circuit_context: synapse attributes not found for %s: gid: %i"
                    % (pop_name, gid)
                )
        else:
            raise RuntimeError(
                "init_circuit_context: invalid synapses parameters"
            )

    if init_weights and has_weights:
        for weight_config_dict in weight_config:
            expr_closure = weight_config_dict.get("closure", None)
            weights_namespaces = weight_config_dict["namespace"]

            cell_weights_dicts = {}
            if weight_dict is not None:
                for weights_namespace in weights_namespaces:
                    if weights_namespace in weight_dict:
                        cell_weights_dicts[weights_namespace] = weight_dict[
                            weights_namespace
                        ]

            elif load_weights:
                if env.data_file_path is None:
                    raise RuntimeError(
                        "init_circuit_context: load_weights=True but data file path is not specified "
                    )

                for weights_namespace in weights_namespaces:
                    cell_weights_iter = read_cell_attribute_selection(
                        env.data_file_path,
                        pop_name,
                        selection=[gid],
                        namespace=weights_namespace,
                        comm=env.comm,
                    )
                    for (
                        cell_weights_gid,
                        cell_weights_dict,
                    ) in cell_weights_iter:
                        assert cell_weights_gid == gid
                        cell_weights_dicts[
                            weights_namespace
                        ] = cell_weights_dict

            else:
                raise RuntimeError(
                    "init_circuit_context: invalid weights parameters"
                )
            if len(weights_namespaces) != len(cell_weights_dicts):
                logger.warning(
                    "init_circuit_context: Unable to load all weights namespaces: %s"
                    % str(weights_namespaces)
                )

            multiple_weights = "error"
            append_weights = False
            for weights_namespace in weights_namespaces:
                if weights_namespace in cell_weights_dicts:
                    cell_weights_dict = cell_weights_dicts[weights_namespace]
                    weights_syn_ids = cell_weights_dict["syn_id"]
                    for syn_name in (
                        syn_name
                        for syn_name in cell_weights_dict
                        if syn_name != "syn_id"
                    ):
                        weights_values = cell_weights_dict[syn_name]
                        syn_attrs.add_mech_attrs_from_iter(
                            gid,
                            syn_name,
                            zip_longest(
                                weights_syn_ids,
                                [
                                    {"weight": Promise(expr_closure, [x])}
                                    for x in weights_values
                                ]
                                if expr_closure
                                else [{"weight": x} for x in weights_values],
                            ),
                            multiple=multiple_weights,
                            append=append_weights,
                        )
                        logger.info(
                            "init_circuit_context: gid: %i; found %i %s synaptic weights in namespace %s"
                            % (
                                gid,
                                len(cell_weights_dict[syn_name]),
                                syn_name,
                                weights_namespace,
                            )
                        )
                        logger.info(
                            "weight_values min/max/mean: %.02f / %.02f / %.02f"
                            % (
                                np.min(weights_values),
                                np.max(weights_values),
                                np.mean(weights_values),
                            )
                        )
                expr_closure = None
                append_weights = True
                multiple_weights = "overwrite"

    if init_edges:
        if connection_graph is not None:
            (graph, a) = connection_graph
        elif load_edges:
            if env.connectivity_file_path is None:
                raise RuntimeError(
                    "init_circuit_context: load_edges=True but connectivity file path is not specified "
                )
            elif os.path.isfile(env.connectivity_file_path):
                (graph, a) = read_graph_selection(
                    file_name=env.connectivity_file_path,
                    selection=[gid],
                    namespaces=["Synapses", "Connections"],
                    comm=env.comm,
                )
        else:
            raise RuntimeError(
                "init_circuit_context: connection file %s not found"
                % env.connectivity_file_path
            )
    else:
        (graph, a) = None, None

    if graph is not None:
        if pop_name in graph:
            for presyn_name in graph[pop_name].keys():
                edge_iter = graph[pop_name][presyn_name]
                syn_attrs.init_edge_attrs_from_iter(
                    pop_name, presyn_name, a, edge_iter, set_edge_delays
                )
        else:
            logger.error(
                "init_circuit_context: connection attributes not found for %s: gid: %i"
                % (pop_name, gid)
            )
            raise Exception


def init_biophysics(
    cell: Union[BiophysCell, SCneuron],
    env: Optional[AbstractEnv] = None,
    reset_cable: bool = True,
    correct_cm: bool = False,
    correct_g_pas: bool = False,
    reset_mech_dict: bool = False,
    verbose: bool = True,
) -> None:
    """
    Consults a dictionary specifying parameters of NEURON cable properties, density mechanisms, and point processes for
    each type of section in a BiophysCell. Traverses through the tree of SHocNode nodes following order of inheritance.
    Sets membrane mechanism parameters, including gradients and inheritance of parameters from nodes along the path from
    root. Warning! Do not reset cable after inserting synapses!
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param reset_cable: bool
    :param correct_cm: bool
    :param correct_g_pas: bool
    :param reset_mech_dict: bool
    :param verbose: bool
    """

    if (correct_cm or correct_g_pas) and env is None:
        raise ValueError(
            "init_biophysics: missing Env object; required to parse network configuration and count "
            "synapses."
        )
    if reset_mech_dict:
        cell.mech_dict = copy.deepcopy(cell.init_mech_dict)
    if reset_cable:
        for sec_type in default_ordered_sec_types:
            if sec_type in cell.mech_dict and sec_type in cell.nodes:
                for node in cell.nodes[sec_type]:
                    reset_cable_by_node(cell, node, verbose=verbose)
    if correct_cm:
        correct_cell_for_spines_cm(cell, env, verbose=verbose)
    else:
        for sec_type in default_ordered_sec_types:
            if sec_type in cell.mech_dict and sec_type in cell.nodes:
                if cell.nodes[sec_type]:
                    update_biophysics_by_sec_type(cell, sec_type)
    if correct_g_pas:
        correct_cell_for_spines_g_pas(cell, env, verbose=verbose)


def correct_node_for_spines_g_pas(
    node, env: AbstractEnv, gid, soma_g_pas, verbose=True
):
    """
    If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in this
    dendritic section proportional to the number of excitatory synapses contained in the section.
    :param node: :class:'SHocNode'
    :param env: :class:'Env'
    :param gid: int
    :param soma_g_pas: float
    :param verbose: bool
    """
    SA_spine = math.pi * (1.58 * 0.077 + 0.5 * 0.5)
    if len(node.spine_count) != node.sec.nseg:
        count_spines_per_seg(node, env, gid)
    for i, segment in enumerate(node.sec):
        SA_seg = segment.area()
        num_spines = node.spine_count[i]

        g_pas_correction_factor = (
            SA_seg * node.sec(segment.x).g_pas
            + num_spines * SA_spine * soma_g_pas
        ) / (SA_seg * node.sec(segment.x).g_pas)
        node.sec(segment.x).g_pas *= g_pas_correction_factor
        if verbose:
            logger.info(
                "g_pas_correction_factor for gid: %i; %s seg %i: %.3f"
                % (gid, node.name, i, g_pas_correction_factor)
            )


def correct_node_for_spines_cm(node, env: AbstractEnv, gid, verbose=True):
    """
    If not explicitly modeling spine compartments for excitatory synapses, this method scales cm in this
    dendritic section proportional to the number of excitatory synapses contained in the section.
    :param node: :class:'SHocNode'
    :param env:  :class:'Env'
    :param gid: int
    :param verbose: bool
    """
    # arrived at via optimization. spine neck appears to shield dendrite from spine head contribution to membrane
    # capacitance and time constant:
    cm_fraction = 0.40
    SA_spine = math.pi * (1.58 * 0.077 + 0.5 * 0.5)
    if len(node.spine_count) != node.sec.nseg:
        count_spines_per_seg(node, env, gid)
    for i, segment in enumerate(node.sec):
        SA_seg = segment.area()
        num_spines = node.spine_count[i]
        cm_correction_factor = (
            SA_seg + cm_fraction * num_spines * SA_spine
        ) / SA_seg
        node.sec(segment.x).cm *= cm_correction_factor
        if verbose:
            logger.info(
                "cm_correction_factor for gid: %i; %s seg %i: %.3f"
                % (gid, node.name, i, cm_correction_factor)
            )


def correct_cell_for_spines_g_pas(cell, env: AbstractEnv, verbose=False):
    """
    If not explicitly modeling spine compartments for excitatory synapses, this method scales g_pas in all
    dendritic sections proportional to the number of excitatory synapses contained in each section.
    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param verbose: bool
    """
    if "soma" in cell.mech_dict:
        soma_g_pas = cell.mech_dict["soma"]["pas"]["g"]["value"]
    elif hasattr(cell, "hoc_cell"):
        soma_g_pas = getattr(list(cell.hoc_cell.soma_list)[0], "g_pas")
    else:
        raise RuntimeError("unable to determine soma g_pas")
    for sec_type in ["basal", "trunk", "apical", "tuft"]:
        for node in cell.nodes[sec_type]:
            correct_node_for_spines_g_pas(
                node, env, cell.gid, soma_g_pas, verbose=verbose
            )


def correct_cell_for_spines_cm(cell, env: AbstractEnv, verbose=False):
    """

    :param cell: :class:'BiophysCell'
    :param env: :class:'Env'
    :param verbose: bool
    """
    loop = 0
    while loop < 2:
        for sec_type in ["basal", "trunk", "apical", "tuft"]:
            for node in cell.nodes[sec_type]:
                correct_node_for_spines_cm(node, env, cell.gid, verbose=verbose)
                if loop == 0:
                    init_nseg(node.sec, verbose=verbose)
                    node.reinit_diam()
        loop += 1
    init_biophysics(cell, env, reset_cable=False, verbose=verbose)


def make_biophys_cell(
    env: AbstractEnv,
    population_name: str,
    gid: int,
    mech_file_path: None = None,
    mech_dict: None = None,
    tree_dict: Optional[
        Dict[
            str,
            Union[ndarray, Dict[str, Union[int, Dict[int, ndarray], ndarray]]],
        ]
    ] = None,
    load_synapses: bool = False,
    synapses_dict: Optional[Dict[str, ndarray]] = None,
    load_edges: bool = False,
    connection_graph: Optional[
        Tuple[
            Dict[
                str,
                Dict[
                    str,
                    List[Tuple[int, Tuple[ndarray, Dict[str, List[ndarray]]]]],
                ],
            ],
            Dict[str, Dict[str, Dict[str, Dict[str, int]]]],
        ]
    ] = None,
    load_weights: bool = False,
    weight_dict: None = None,
    set_edge_delays: bool = True,
    bcast_template: bool = True,
    validate_tree: bool = True,
    **kwargs,
) -> BiophysCell:
    """
    :param env: :class:'Env'
    :param population_name: str
    :param gid: int
    :param tree_dict: dict
    :param synapses_dict: dict
    :param weight_dict: list of dict
    :param load_synapses: bool
    :param load_edges: bool
    :param load_weights: bool
    :param set_edge_delays: bool
    :param mech_file_path: str (path)
    :return: :class:'BiophysCell'
    """
    load_cell_template(env, population_name, bcast_template=bcast_template)

    if tree_dict is None:
        tree_attr_iter, _ = read_tree_selection(
            env.data_file_path,
            population_name,
            [gid],
            comm=env.comm,
            topology=True,
            validate=validate_tree,
        )
        _, tree_dict = next(tree_attr_iter)

    cell = BiophysCell(
        gid=gid,
        population_name=population_name,
        neurotree_dict=tree_dict,
        env=env,
        mech_file_path=mech_file_path,
        mech_dict=mech_dict,
    )
    circuit_flag = (
        load_edges
        or load_weights
        or load_synapses
        or synapses_dict
        or weight_dict
        or connection_graph
    )
    if circuit_flag:
        init_circuit_context(
            env,
            population_name,
            gid,
            load_synapses=load_synapses,
            synapses_dict=synapses_dict,
            load_edges=load_edges,
            connection_graph=connection_graph,
            load_weights=load_weights,
            weight_dict=weight_dict,
            set_edge_delays=set_edge_delays,
            **kwargs,
        )

    env.biophys_cells[population_name][gid] = cell

    return cell


def make_BRK_cell(
    env: AbstractEnv,
    pop_name,
    gid,
    mech_file_path=None,
    mech_dict=None,
    tree_dict=None,
    load_synapses=False,
    synapses_dict=None,
    load_edges=False,
    connection_graph=None,
    load_weights=False,
    weight_dict=None,
    set_edge_delays=True,
    bcast_template=True,
    **kwargs,
):
    """
    :param env: :class:'Env'
    :param pop_name: str
    :param gid: int
    :param mech_file_path: str (path)
    :param mech_dict: dict
    :param synapses_dict: dict
    :param weight_dicts: list of dict
    :param load_synapses: bool
    :param load_edges: bool
    :param load_weights: bool
    :param set_edge_delays: bool
    :return: :class:'BRKneuron'
    """
    load_cell_template(env, pop_name, bcast_template=bcast_template)

    if mech_dict is None and mech_file_path is None:
        raise RuntimeError(
            "make_BRK_cell: mech_dict or mech_file_path must be specified"
        )

    if mech_dict is None and mech_file_path is not None:
        mech_dict = read_from_yaml(mech_file_path)

    cell = BRKneuron(
        gid=gid,
        pop_name=pop_name,
        env=env,
        cell_config=BRKconfig(**mech_dict["BoothRinzelKiehn"]),
        mech_dict={
            k: mech_dict[k] for k in mech_dict if k != "BoothRinzelKiehn"
        },
    )

    circuit_flag = (
        load_edges
        or load_weights
        or load_synapses
        or synapses_dict
        or weight_dict
        or connection_graph
    )
    if circuit_flag:
        init_circuit_context(
            env,
            pop_name,
            gid,
            load_synapses=load_synapses,
            synapses_dict=synapses_dict,
            load_edges=load_edges,
            connection_graph=connection_graph,
            load_weights=load_weights,
            weight_dict=weight_dict,
            set_edge_delays=set_edge_delays,
            **kwargs,
        )

    env.biophys_cells[pop_name][gid] = cell
    return cell


def make_PR_cell(
    env: AbstractEnv,
    pop_name,
    gid,
    mech_file_path=None,
    mech_dict=None,
    tree_dict=None,
    load_synapses=False,
    synapses_dict=None,
    load_edges=False,
    connection_graph=None,
    load_weights=False,
    weight_dict=None,
    set_edge_delays=True,
    bcast_template=True,
    **kwargs,
):
    """
    :param env: :class:'Env'
    :param pop_name: str
    :param gid: int
    :param mech_file_path: str (path)
    :param mech_dict: dict
    :param synapses_dict: dict
    :param weight_dicts: list of dict
    :param load_synapses: bool
    :param load_edges: bool
    :param load_weights: bool
    :param set_edge_delays: bool
    :return: :class:'PRneuron'
    """
    load_cell_template(env, pop_name, bcast_template=bcast_template)

    if mech_dict is None and mech_file_path is None:
        raise RuntimeError(
            "make_PR_cell: mech_dict or mech_file_path must be specified"
        )

    if mech_dict is None and mech_file_path is not None:
        mech_dict = read_from_yaml(mech_file_path)

    cell = PRneuron(
        gid=gid,
        pop_name=pop_name,
        env=env,
        cell_config=PRconfig(**mech_dict["PinskyRinzel"]),
        mech_dict={k: mech_dict[k] for k in mech_dict if k != "PinskyRinzel"},
    )

    circuit_flag = (
        load_edges
        or load_weights
        or load_synapses
        or synapses_dict
        or weight_dict
        or connection_graph
    )
    if circuit_flag:
        init_circuit_context(
            env,
            pop_name,
            gid,
            load_synapses=load_synapses,
            synapses_dict=synapses_dict,
            load_edges=load_edges,
            connection_graph=connection_graph,
            load_weights=load_weights,
            weight_dict=weight_dict,
            set_edge_delays=set_edge_delays,
            **kwargs,
        )

    env.biophys_cells[pop_name][gid] = cell
    return cell


def make_SC_cell(
    env: AbstractEnv,
    pop_name: str,
    gid: int,
    mech_file_path: None = None,
    mech_dict: Optional[
        Dict[str, Dict[str, Dict[str, Union[Dict[str, float], Dict[str, int]]]]]
    ] = None,
    tree_dict: None = None,
    load_synapses: bool = False,
    synapses_dict: None = None,
    load_edges: bool = False,
    connection_graph: None = None,
    load_weights: bool = False,
    weight_dict: None = None,
    set_edge_delays: bool = True,
    bcast_template: bool = True,
    **kwargs,
) -> SCneuron:
    """
    :param env: :class:'Env'
    :param pop_name: str
    :param gid: int
    :param mech_file_path: str (path)
    :param mech_dict: dict
    :param synapses_dict: dict
    :param weight_dicts: list of dict
    :param load_synapses: bool
    :param load_edges: bool
    :param load_weights: bool
    :param set_edge_delays: bool
    :return: :class:'SCneuron'
    """
    load_cell_template(env, pop_name, bcast_template=bcast_template)

    if mech_dict is None and mech_file_path is None:
        raise RuntimeError(
            "make_SC_cell: mech_dict or mech_file_path must be specified"
        )

    if mech_dict is None and mech_file_path is not None:
        mech_dict = read_from_yaml(mech_file_path)

    cell = SCneuron(gid=gid, pop_name=pop_name, env=env, mech_dict=mech_dict)

    circuit_flag = (
        load_edges
        or load_weights
        or load_synapses
        or synapses_dict
        or weight_dict
        or connection_graph
    )
    if circuit_flag:
        init_circuit_context(
            env,
            pop_name,
            gid,
            load_synapses=load_synapses,
            synapses_dict=synapses_dict,
            load_edges=load_edges,
            connection_graph=connection_graph,
            load_weights=load_weights,
            weight_dict=weight_dict,
            set_edge_delays=set_edge_delays,
            **kwargs,
        )

    env.biophys_cells[pop_name][gid] = cell
    return cell


def register_cell(
    env: AbstractEnv,
    pop_name: str,
    gid: Union[uint32, int],
    cell: Union["HocObject", BiophysCell, SCneuron],
) -> None:
    """
    Registers a cell in a network environment.

    :param env: an instance of the `Env` class
    :param pop_name: population name
    :param gid: gid
    :param cell: cell instance
    """
    rank = env.comm.rank
    env.gidset.add(gid)
    env.pc.set_gid2node(gid, rank)
    hoc_cell = getattr(cell, "hoc_cell", cell)
    env.cells[pop_name][gid] = hoc_cell
    if hoc_cell.is_art() > 0:
        env.artificial_cells[pop_name][gid] = hoc_cell
    # Tell the ParallelContext that this cell is a spike source
    # for all other hosts. NetCon is temporary.
    nc = getattr(cell, "spike_detector", None)
    if nc is None:
        if hasattr(cell, "connect2target"):
            nc = hoc_cell.connect2target(h.nil)
        elif cell.is_art() > 0:
            nc = h.NetCon(cell, None)
        else:
            raise RuntimeError("register_cell: unknown cell type")
    nc.delay = max(2 * env.dt, nc.delay)
    env.pc.cell(gid, nc, 1)
    env.pc.outputcell(gid)
    # Record spikes of this cell
    env.pc.spike_record(gid, env.t_vec, env.id_vec)
    # if the spike detector is located in a compartment other than soma,
    # record the spike time delay relative to soma
    if hasattr(cell, "spike_onset_delay"):
        env.spike_onset_delay[gid] = cell.spike_onset_delay


def is_cell_registered(env: AbstractEnv, gid: Union[uint32, int]) -> int:
    """
    Returns True if cell gid has already been registered, False otherwise.
    """
    return env.pc.gid_exists(gid)


def record_cell(
    env: AbstractEnv, pop_name: str, gid: int, recording_profile: None = None
) -> List[Dict[str, Union[str, int, "HocObject", float]]]:
    """
    Creates a recording object for the given cell, according to configuration in env.recording_profile.
    """
    recs = []
    if recording_profile is None:
        recording_profile = env.recording_profile
    if recording_profile is not None:
        syn_attrs = env.synapse_attributes
        cell = env.biophys_cells[pop_name].get(gid, None)
        if cell is not None:
            label = recording_profile["label"]
            dt = recording_profile.get("dt", None)
            for reclab, recdict in recording_profile.get(
                "section quantity", {}
            ).items():
                recvar = recdict.get("variable", reclab)
                loc = recdict.get("loc", None)
                swc_types = recdict.get("swc_types", None)
                locdict = collections.defaultdict(lambda: 0.5)
                if (loc is not None) and (swc_types is not None):
                    for s, l in zip(swc_types, loc):
                        locdict[s] = l

                nodes = filter_nodes(
                    cell,
                    layers=recdict.get("layers", None),
                    swc_types=recdict.get("swc types", None),
                )
                node_type_count = collections.defaultdict(int)
                for node in nodes:
                    node_type_count[node.section_type] += 1
                visited = set()
                for node in nodes:
                    sec = node.sec
                    if str(sec) not in visited:
                        if node_type_count[node.section_type] == 1:
                            rec_id = f"{node.section_type}"
                        else:
                            rec_id = "%s.%i" % (node.section_type, node.index)
                        rec = make_rec(
                            rec_id,
                            pop_name,
                            gid,
                            cell.hoc_cell,
                            sec=sec,
                            dt=dt,
                            loc=locdict[node.section_type],
                            param=recvar,
                            label=reclab,
                            description=node.name,
                        )
                        recs.append(rec)
                        env.recs_dict[pop_name][rec_id].append(rec)
                        env.recs_count += 1
                        visited.add(str(sec))
            for recvar, recdict in recording_profile.get(
                "synaptic quantity", {}
            ).items():
                syn_filters = recdict.get("syn_filters", {})
                syn_sections = recdict.get("sections", None)
                synapses = syn_attrs.filter_synapses(
                    gid, syn_sections=syn_sections, **syn_filters
                )
                syn_names = recdict.get(
                    "syn names", syn_attrs.syn_name_index_dict.keys()
                )
                for syn_id, syn in synapses.items():
                    syn_swc_type_name = env.SWC_Type_index[syn.swc_type]
                    syn_section = syn.syn_section
                    for syn_name in syn_names:
                        pps = syn_attrs.get_pps(
                            gid, syn_id, syn_name, throw_error=False
                        )
                        if (pps is not None) and (pps not in env.recs_pps_set):
                            rec_id = "%d.%s.%s" % (
                                syn_id,
                                syn_name,
                                str(recvar),
                            )
                            label = f"{str(recvar)}"
                            rec = make_rec(
                                rec_id,
                                pop_name,
                                gid,
                                cell.hoc_cell,
                                ps=pps,
                                dt=dt,
                                param=recvar,
                                label=label,
                                description=f"{label}",
                            )
                            ns = "%s%d.%s" % (
                                syn_swc_type_name,
                                syn_section,
                                syn_name,
                            )
                            env.recs_dict[pop_name][ns].append(rec)
                            env.recs_count += 1
                            env.recs_pps_set.add(pps)
                            recs.append(rec)

    return recs
