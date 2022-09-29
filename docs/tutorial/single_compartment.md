---
file_format: mystnb
kernelspec:
  name: python3
  display_name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
    jupytext_version: 1.13.8
---

# 2. Single Compartment

In this tutorial, we will demonstrate how to re-use the previously constructed simulation and convert to **single-compartment** model for computational efficiency.

As shown in the previous [case](constructing_a_network_model.md), `MiV-Simulator` requires full details of dendritic morphology with multiple dendritic compartments. This realistic neuron-network models is useful to study the case when characterizing morphology is essential, but some studies only requires **single-compartment** models without all morphological details.

To reduce the computational expanses, one can sacrifice some realism by using `SC_nrn` mechanism instead.

:::{note}
The network and morphology cannot be constructed without the synpatic model: synapse network cannot be created with single-compartment model. The compromise is to re-use the same set of synaptic objects and coreresponding connections, but to map them onto a single compartment (or reduced 2-3 compartments) instead of the original morphology for which they were created.
:::

The general workflow is to use a complete dendritic morphology to create the synapses, and use `SC_nrn` for efficient simulation. `SC_nrn` will collapse all synapses onto a single compartment, while those connections still exists.

## Replace cell types to use `SC_nrn` template

`SC_nrn.mod` file is located within `mechanisms` directory. Neuron mechanism files (*.mod) provides a way to configure hoc templates via yaml.

## Run simulation
