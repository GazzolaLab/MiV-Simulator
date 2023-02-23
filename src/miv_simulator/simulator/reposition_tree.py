import gc
import os
import copy
import sys
import h5py
import numpy as np
from miv_simulator import utils
from scipy.spatial.distance import cdist
from miv_simulator.utils import config_logging, get_module_logger

logger = get_module_logger("reposition_tree")


def reposition_tree(neurotree_dict, cell_coords, swc_type_defs):
    """
    Given a neurotree dictionary, relocates all point coordinates to
    the positions indicated by cell_coords.  The delta distance
    necessary to reposition the cells is calculated as the smallest
    distances between soma points and cell_coords.

    Note: This procedure does not recalculate layer information.

    :param neurotree_dict:
    :param cell_coords:
    :param swc_type_defs:
    :return: neurotree dict

    """

    cell_coords = np.asarray(cell_coords).reshape((1, -1))

    pt_xs = copy.deepcopy(neurotree_dict["x"])
    pt_ys = copy.deepcopy(neurotree_dict["y"])
    pt_zs = copy.deepcopy(neurotree_dict["z"])
    pt_radius = copy.deepcopy(neurotree_dict["radius"])
    pt_layers = copy.deepcopy(neurotree_dict["layer"])
    pt_parents = copy.deepcopy(neurotree_dict["parent"])
    pt_swc_types = copy.deepcopy(neurotree_dict["swc_type"])
    pt_sections = copy.deepcopy(neurotree_dict["sections"])
    sec_src = copy.deepcopy(neurotree_dict["src"])
    sec_dst = copy.deepcopy(neurotree_dict["dst"])
    soma_pts = np.where(pt_swc_types == swc_type_defs["soma"])[0]

    soma_coords = np.column_stack(
        (pt_xs[soma_pts], pt_ys[soma_pts], pt_zs[soma_pts])
    )
    pos_delta = (
        soma_coords[np.argmin(cdist(soma_coords, cell_coords))] - cell_coords
    ).reshape((-1,))

    new_tree_dict = {
        "x": pt_xs - pos_delta[0],
        "y": pt_ys - pos_delta[1],
        "z": pt_zs - pos_delta[2],
        "radius": pt_radius,
        "layer": pt_layers,
        "parent": pt_parents,
        "swc_type": pt_swc_types,
        "sections": pt_sections,
        "src": sec_src,
        "dst": sec_dst,
    }

    return new_tree_dict
