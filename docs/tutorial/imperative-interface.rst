Imperative interface
====================

MiV-Simulator includes an imperative interface built
on `machinable <https://machinable.org>`__ that can be useful in more
custom use cases that are not supported out-of-the-box.

The interface API works different from the declarative mode described in
prior sections where the entire simulation was specified using YAML.
Instead, the API gives you full control to construct the simulation
imperatively using Python.

Generally, the process breaks down into (1) generating the
``cells.h5``/``connections.h5`` that represent the neural systems, and
(2) loading the cells and connection into the execution runtime to
simulate the dynamics.

Constructing the neural H5
--------------------------

To construct the neuro H5 files that describe the specific network
instantiation, the ``miv_simulator.interface`` module provides
configurable `machinable
components <https://machinable.org/guide/component.html>`__ that can be
chained as follows:

.. literalinclude:: ../MiV-Simulator-Cases/3-interface-api/rc-separation.py
   :language: python
   :linenos:
   :lines: 14-112

The ``graph.files()`` method in the last line returns the filepaths of the generated H5 files that are required to launch a simulation. 

Launching the simulation
------------------------

To simulate the network, the generated H5 files can be loaded into the execution environment, for example:

.. literalinclude:: ../MiV-Simulator-Cases/3-interface-api/interface/experiment/rc.py
   :language: python
   :linenos:
   :lines: 4,16,54-71

This will load and construct all cells and connections that will be part of the simulation. You are free to add additional elements to the system, such as an LFP to read out signals, for example:

.. literalinclude:: ../MiV-Simulator-Cases/3-interface-api/interface/experiment/rc.py
   :language: python
   :linenos:
   :lines: 5,16,73-88

The execution environment provides access to all NEURON objects, allowing to set up and modify elements as needed. For example, you can use NEURON usual ``cell.play`` function to generate stimulation patterns:


.. literalinclude:: ../MiV-Simulator-Cases/3-interface-api/interface/experiment/rc.py
   :language: python
   :linenos:
   :lines: 96-101

Finally, you can launch and control the simulation using the standard NEURON API:

.. literalinclude:: ../MiV-Simulator-Cases/3-interface-api/interface/experiment/rc.py
   :language: python
   :linenos:
   :lines: 103-119

The full code example using the above code for reservoir computing can be found `here <https://github.com/GazzolaLab/MiV-Simulator-Cases/blob/main/3-interface-api/rc-separation.py>`__.
