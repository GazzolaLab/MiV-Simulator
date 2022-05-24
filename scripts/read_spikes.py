import pprint

import h5py
import numpy as np

grp_h5types = "H5Types"
grp_populations = "Populations"
dset_populations = "Populations"
dtype_population_labels = "Population labels"
cell_index_name = "Cell Index"
cell_attr_pointer_name = "Attribute Pointer"
cell_attr_value_name = "Attribute Value"


def read_population_ranges(file_path):
    with h5py.File(file_path, "r") as f:
        population_labels = f[grp_h5types][
            dtype_population_labels
        ].dtype.metadata["enum"]
        population_inds = {x[1]: x[0] for x in population_labels.items()}
        population_spec = f[grp_h5types][dset_populations][:]
        return {population_inds[x[2]]: (x[0], x[1]) for x in population_spec}


def read_cell_attributes(file_path, population, namespace):
    """
    :param file_path: str (path to neuroh5 file)
    :param population: str
    :param namespace: str
    :return: dict
    """

    attr_map = {}

    with h5py.File(file_path, "r") as f:
        if grp_populations not in f:
            raise RuntimeError(
                f"Populations group not found in file {file_path}"
            )
        if population not in f[grp_populations]:
            raise RuntimeError(
                f"Population {population} not found in Populations group in file {file_path}"
            )
        if namespace not in f[grp_populations][population]:
            raise RuntimeError(
                f"Namespace {namespace} for population {population} not found in file {file_path}"
            )

        population_ranges = read_population_ranges(file_path)
        this_pop_start = population_ranges[population][0]

        for attribute, group in f[grp_populations][population][
            namespace
        ].items():

            cell_index = group[cell_index_name]
            attr_pointer = group[cell_attr_pointer_name]
            attr_value = group[cell_attr_value_name]

            attr_values = np.split(attr_value, attr_pointer)

            attr_map[attribute] = dict(
                zip(cell_index[:] + this_pop_start, attr_values)
            )

    return attr_map


def read_spikes(file_path, population, attr_name="t", namespace="Spike Events"):

    spike_attrs = read_cell_attributes(file_path, population, namespace)

    return spike_attrs[attr_name]


pprint.pprint(read_spikes("Microcircuit_PYR_spikes.h5", "PYR"))
