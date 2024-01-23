Simulating Multielectrode Arrays
================================================

A common system of interest in neural simulation are Multielectrode Arrays (MEA) which integrate multiple microelectrodes to obtain or deliver neural signals to the culture.

To construct MEAs in simulation, you can position artificial STIM(umlus) cells that emulate delivering electrodes and/or position LFP readouts to emulate obtaining electrodes.

Positioning 'electrode' STIM cells
----------------------------------

Given the x,y,z coordinates of your MEA electrodes, you can insert them into the culture via a callback.

.. code:: python

   # define callable that returns MEA coordinates
   def callable(n, extents): # -> (n, 3)
       ...
       return xyz_coordinate_array


   "cell_distributions": {
     "STIM": {     # generate STIM(ulus) cells
       "@callable": 64   # @ to invoke callable during generation
     },
     ...
   }

   "layer_extents": { 
      "@callable": [      # extents definition
          [0.0, 0.0, 0.0],
          [200.0, 200.0, 150.0],
      ],
      ...
   }


Positioning 'electrode' LFPs
----------------------------

To emulate the readout feature of the electrodes, you can leverage the ``miv_simulator.lfp.LFP`` class.


.. literalinclude:: ../MiV-Simulator-Cases/3-interface-api/interface/experiment/rc.py
   :language: python
   :linenos:
   :lines: 5,73-88
