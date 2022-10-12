***************************************
Parallel HDF5 Configuration and Storage
***************************************

We extensively utilize Parallel-HDF5 data structure for configuration and simulation. We mainly use `NeuroH5 <https://github.com/iraikov/neuroh5>`_ package. Most of the descriptions/figures in the followingdiscussions are excerpted from [1]_. For the full details of the tool, please refer the paper.

- `GitHub <https://github.com/iraikov/neuroh5>`_

NeuroH5 Structure
=================

`NeuroH5` implements data structure specialized for scalable and high-performance neuronal simulation. Based on parallel HDF5 format, the structure incorporates morphological, synaptic, and connectivity information from large neuronal network.

The `NeuroH5` format focuses on two principal data structures:

    1. **graph projections**: specify connections between two populations of cells
    2. **cell attributess**: specify numerical attributes associated with individual cells

Hierarchical structure:

    1. **neuronal populations**: a particular kind of neural species
    2. **attribute namespaces**: logical grouping of attributes
    3. **global cell identifiers**: individual cells within a population
    4. **cell attribute**: homogeneous numerical vector that is contained iwthin a particular namespaces and associtated with a particular gid in a population
    5. **graph projection**: collection of numerical vectors that specify the osurce and destination node indices of each edge, where each node index corresponds to a gid

Individual Cell Attribute
-------------------------

The NeuroH5 data format for per-cell morphological, synaptic, connectivity, and other information of large neuronal network models, designed for further extension, construction, simulation, and analysis.

Each cell attributes contains:

    - cell index:
    - attribute pointer:
    - attribute value:

.. image:: https://raw.githubusercontent.com/iraikov/neuroh5/master/doc/Cell%20Attribute%20and%20Morphology%20Structure.png
   :width: 800
   :alt: Cell Attribute and Morphology Structure

Dendritic Connectivity (Graph)
------------------------------

Destination Block Sparse

.. image:: https://raw.githubusercontent.com/iraikov/neuroh5/master/doc/Destination%20Block%20Sparse%20Structure.png
   :width: 800
   :alt: Destination Block Sparse Structure

.. image:: https://raw.githubusercontent.com/iraikov/neuroh5/master/doc/Destination%20Block%20Storage%20Example.png
   :width: 800
   :alt: Destination Block Storage Example

.. image:: https://raw.githubusercontent.com/iraikov/neuroh5/master/doc/Destination%20Blocks.1.png
   :width: 800
   :alt: Destination Blocks


Neuro IO
--------

.. image:: https://raw.githubusercontent.com/iraikov/neuroh5/master/doc/NeuroIO%20Structure.png
   :width: 800
   :alt: NeuroIO Structure

Sample Structure
----------------

.. image:: https://raw.githubusercontent.com/iraikov/neuroh5/master/doc/sample.png
   :width: 800
   :alt: Sample Structure

Commonly Used Functions
=======================

- neuroh5.io.bcast_cell_attributes: query cell attribute and broadcast

.. code-block:: python

   from neuroh5.io import bcast_call_cell_attributes

    coords = bcast_cell_attributes(
        "coords.h5",                       # file_name
        "PYR",                             # pop_name
        0,                                 # root
        namespace="Generated Coordinates", # namespace
        comm=comm                          # comm
                                           # mask
    )
    soma_coords = {
        k: (
            v["U Coordinate"][0],
            v["V Coordinate"][0],
            v["L Coordinate"][0],
        )
        for (k, v) in coords
    }

Commonly Used CLI Tools
=======================

- neurotrees_import: Read SWC morphology files and convert as trees in h5 format.


References
==========

.. [1] I. G. Raikov, A. Milstein, G. G. Moolchand Prannath and Szabo, C. Schneider, A. Hadjiabadi Darian and Chatzikalymniou, and I. Soltesz, “Towards a general framework for modeling large-scale biophysical neuronal networks: a full-scale computational model of the rat dentate gyrus,” bioRxiv, p. 2021.11.02.466940, 2021.
