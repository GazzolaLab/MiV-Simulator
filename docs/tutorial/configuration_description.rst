******************
YAML Configuration
******************

The MiV-Simulator leverages a YAML-based configuration format to declaratively describe complex neural systems. The user-specified configuration files are used to generate concrete instantiations of the system used in the simulation.
For example, a YAML user configuration may describe the number and distribution of neurons which is used during network construction to determine the exact sampled soma positions.

The different configuration templates that are used for simulation construction are described below.


Simulation Configuration
========================

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Microcircuit_Small.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Connection_Velocity.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Definitions.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Geometry_Small.yaml
   :linenos:
   :language: yaml
   :caption:

Synapse / Dentrites Configuration
=================================

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Microcircuit_Connections.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Axon_Extent.yaml
   :linenos:
   :language: yaml
   :caption:


.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/OLM_synapse_density.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/PVBC_synapse_density.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/PYR_synapse_density.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/PYR_SoldadoMagraner.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Synapse_Parameter_Rules.yaml
   :linenos:
   :language: yaml
   :caption:

Input / Stimulation Configuration
=================================

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Input_Configuration.yaml
   :linenos:
   :language: yaml
   :caption:


Post-Process Configuration
==========================

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Recording.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Analysis_Configuration.yaml
   :linenos:
   :language: yaml
   :caption:

Miscellaneous Configuration
===========================

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Random.yaml
   :linenos:
   :language: yaml
   :caption:

.. literalinclude:: ../MiV-Simulator-Cases/1-construction/config/Global.yaml
   :linenos:
   :language: yaml
   :caption:
