CLI Commands
============

For key routines, we also provides CLI-executable commands, such that one can manipulate
the simulation from the shell.

Simulation Builder
------------------

.. click:: scripts.make_h5types:main
   :prog: make-h5types
   :nested: full

.. click:: scripts.measure_distances:main
   :prog: measure-distances
   :nested: full

.. click:: scripts.generate_distance_connections:main
   :prog: generate-distance-connections
   :nested: full

.. click:: scripts.generate_input_features:main
   :prog: generate-input-features
   :nested: full

.. click:: scripts.generate_input_spike_trains:main
   :prog: generate-input-spike-trains
   :nested: full

.. click:: scripts.generate_soma_coordinates:main
   :prog: generate-soma-coordinates
   :nested: full

.. click:: scripts.distribute_synapse_locs:main
   :prog: distribute-synapse-locs
   :nested: full

Simulation Runner
-----------------

.. click:: scripts.run_network:main
   :prog: run-network
   :nested: full

Analysis scripts
----------------

.. click:: scripts.analysis.plot_biophys_cell_tree:main
   :prog: plot-biophys-cell-tree
   :nested: full

.. click:: scripts.analysis.plot_cell_tree:main
   :prog: plot-cell-tree
   :nested: full

.. click:: scripts.analysis.plot_coords_in_volume:main
   :prog: plot-coords-in-volume
   :nested: full

.. click:: scripts.analysis.plot_network_clamp:main
   :prog: plot-network-clamp
   :nested: full

.. click:: scripts.analysis.plot_spike_raster:main
   :prog: plot-spike-raster
   :nested: full

.. click:: scripts.analysis.plot_state:main
   :prog: plot-state
   :nested: full

.. click:: scripts.analysis.plot_single_vertex_dist:main
   :prog: plot-single-vertex-dist
   :nested: full

.. click:: scripts.analysis.plot_spatial_spike_raster:main
   :prog: plot-spatial-spike-raster
   :nested: full

.. click:: scripts.analysis.plot_lfp:main
   :prog: plot-lfp
   :nested: full

.. click:: scripts.analysis.plot_lfp_spectrogram:main
   :prog: plot-lfp-spectrogram
   :nested: full

Tools
-----

.. click:: scripts.tools.show_h5types:main
   :prog: show-h5types
   :nested: full

.. click:: scripts.tools.query_cell_attrs:main
   :prog: query-cell-attrs
   :nested: full

.. click:: scripts.tools.check_config:main
   :prog: check-config
   :nested: full

.. click:: scripts.tools.cut_slice:main
   :prog: cut-slice
   :nested: full

.. click:: scripts.tools.sample_cells:main
   :prog: sample-cells
   :nested: full
