****************************
HDF5-Based Storage Structure
****************************

We extensively utilize HDF5 data structure to store simulation configuration and setup. We use same data structure illustrated [1]_ by using the package `NeuroH5 <https://github.com/iraikov/neuroh5>`_.

NeuroH5 Structure
=================

Morphology Structure
--------------------

.. image:: https://raw.githubusercontent.com/iraikov/neuroh5/master/doc/Cell%20Attribute%20and%20Morphology%20Structure.png
   :width: 800
   :alt: Cell Attribute and Morphology Structure

Dendritic Connectivity
----------------------

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

References
==========

.. [1] I. G. Raikov, A. Milstein, G. G. Moolchand Prannath and Szabo, C. Schneider, A. Hadjiabadi Darian and Chatzikalymniou, and I. Soltesz, “Towards a general framework for modeling large-scale biophysical neuronal networks: a full-scale computational model of the rat dentate gyrus,” bioRxiv, p. 2021.11.02.466940, 2021.
