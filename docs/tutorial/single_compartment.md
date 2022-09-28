---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 2. Single Compartment

In this tutorial, we will demonstrate how to re-use the previously constructed simulation and convert to **single-compartment** model for computational efficiency.

As shown in the previous [case](constructing_a_network_model.md), `MiV-Simulator` requires full details of dendritic morphology with multiple dendritic compartments. This realistic neuron-network models is useful to study the case when characterizing morphology is essential, but some studies only requires **single-compartment** models without all morphological details.

To reduce the computational expanses, one can sacrifice some realism by using `SC_nrn` mechanism instead.

> Note, the network and morphology cannot be constructed without the synpatic model: synapse network cannot be created with single-compartment model. The compromise is to re-use the same set of synaptic objects and coreresponding connections, but to map them onto a single compartment (or reduced 2-3 compartments) instead of the original morphology for which they were created.

The general workflow is to use a complete dendritic morphology to create the synapses, and use `SC_nrn` for efficient simulation. `SC_nrn` will collapse all synapses onto a single compartment, while those connections still exists.

## Replace cell types to use `SC_nrn` template

## Run simulation
