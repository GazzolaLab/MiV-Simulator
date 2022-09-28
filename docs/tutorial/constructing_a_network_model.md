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

# 1. Constructing Simulation

:::{note}
> This tool extensively use NeuroH5 for simulation data structure. It is recommended to check this [discussion](../discussion/neuroh5.rst).
:::

```{mermaid}
graph LR
D(Design</br>Experiment) -->|yaml| C(Construct</br>Simulation) -->|h5| R(Run) -->|h5| P(Post Process)
```

The experiment is designed within `YAML` configurations. To run the first example simulation, we provide the set of configuration files below:

- Download: [link](https://uofi.box.com/shared/static/a88dy7muglte90hklryw0xskv7ne13j0.zip).
  1. __config__: configuration YAML files to construct simulation
  2. __mechanisms__: collection of NEURON mechanism files (.mod files)
  3. __templates__: collection of cell parameters (.hoc files)
  4. __datasets__: directory to construct simulation in h5 format. It also contains (.swc) files.

In the remaining, we demonstrate how to construct the simulation and how to run the simulation.

:::{note}
The detail description and configurability of each file is included [here](basic_configuration_yaml.md).
:::

:::{note}
> Add the option `--use-hwthread-cpus` for `mpirun` to use thread-wise MPI instead of core.
:::

:::{note}
> Run each cells only once.
:::

## Reset

The configuration of the simuulation environment (soma coordinate, dendrite connection, cell parameters, etc.) are built in NeuroH5 format in `datasets` directory. To reset the configuration steps in this tutorial, simply remove all `*.h5` files inside the directory.

```{code-cell} python
:tags: [hide-cell]
!rm -rf datasets/*.h5
```

## Creating H5Types definitions

```{code-cell} python
import os
import matplotlib.pyplot as plt

datapath = "datasets"
os.makedirs(datapath, exist_ok=True)
```

```{code-cell} python
#!make-h5types --output-path datasets/MiV_Small_h5types.h5  # If config path is not specified, use default.
!make-h5types --config-prefix config -c Microcircuit_Small.yaml --output-path datasets/MiV_Small_h5types.h5
```

You can use HDF5 utilities `h5ls` and `h5dump` to examine the contents of an HDF5 file as we build the case study.

- h5ls: List each objects of an HDF5 file name and objects within the file. Apply method recursively with `-r` flag.
- h5dump: Display h5 contents in dictionary format, in human readable form.

For more detail, checkout [this page](https://www.asc.ohio-state.edu/wilkins.5/computing/HDF/hdf5tutorial/util.html).

```{code-cell} python
!h5ls -r ./datasets/MiV_Small_h5types.h5
```

```{code-cell} python
!h5dump -d /H5Types/Populations ./datasets/MiV_Small_h5types.h5
```

## Copying and compiling NMODL mechanisms

For more detail of NEURON NMODL, checkout [this page](http://web.mit.edu/neuron_v7.4/nrntuthtml/tutorial/tutD.html)

```{code-cell} python
!nrnivmodl mechanisms/**/*.mod .
```

## Generating soma coordinates and measuring distances

Here, we create `Microcircuit_Small_coords.h5` file that stores soma coordinate information. To see the contents in the file, try to use `h5dump` like above.

```{code-cell} python
!generate-soma-coordinates -v \
    --config=Microcircuit_Small.yaml \
    --config-prefix=config \
    --types-path=datasets/MiV_Small_h5types.h5 \
    --output-path=datasets/Microcircuit_Small_coords.h5 \
    --output-namespace='Generated Coordinates'
```

```{code-cell} python
!mpirun -np 1 measure-distances -v \
             -i PYR -i PVBC -i OLM -i STIM \
             --config=Microcircuit_Small.yaml \
             --config-prefix=config \
             --coords-path=datasets/Microcircuit_Small_coords.h5
```

### Visualize (Soma Location)

```{code-cell} python
from miv_simulator import plotting as plot
from miv_simulator import utils
import matplotlib.pyplot as plt

%matplotlib inline
```

```{code-cell} python
utils.config_logging(True)
fig = plot.plot_coords_in_volume(
    populations=("PYR", "PVBC", "OLM"),
    coords_path="datasets/Microcircuit_Small_coords.h5",
    config="config_first_case/Microcircuit_Small.yaml",
    coords_namespace="Generated Coordinates",
    scale=25.0,
)
```

```{code-cell} python
utils.config_logging(True)
fig = plot.plot_coords_in_volume(
    populations=("STIM",),
    coords_path="datasets/Microcircuit_Small_coords.h5",
    config="config_first_case/Microcircuit_Small.yaml",
    coords_namespace="Generated Coordinates",
    scale=25.0,
)
```

## Creating dendritic trees in NeuroH5 format

`*.swc` file contains 3D point structure of the cell model. The tree model `*_tree.h5` can be created using `neurotree_import` feature from `neuroh5`.

```{code-cell} python
!~/github/neuroh5/bin/neurotrees_import PVBC datasets/PVBC_tree.h5 datasets/PVBC.swc

!~/github/neuroh5/bin/neurotrees_import PYR datasets/PYR_tree.h5 datasets/PYR.swc

!~/github/neuroh5/bin/neurotrees_import OLM datasets/OLM_tree.h5 datasets/OLM.swc

!h5copy -p -s '/H5Types' -d '/H5Types' -i datasets/MiV_Small_h5types.h5 -o datasets/PVBC_tree.h5
!h5copy -p -s '/H5Types' -d '/H5Types' -i datasets/MiV_Small_h5types.h5 -o datasets/PYR_tree.h5
!h5copy -p -s '/H5Types' -d '/H5Types' -i datasets/MiV_Small_h5types.h5 -o datasets/OLM_tree.h5
```

## Distributing synapses along dendritic trees

```{code-cell} python
!~/github/neuroh5/bin/neurotrees_copy --fill --output datasets/PYR_forest_Small.h5 datasets/PYR_tree.h5 PYR 10
```

```{code-cell} python
!~/github/neuroh5/bin/neurotrees_copy --fill --output datasets/PVBC_forest_Small.h5 datasets/PVBC_tree.h5 PVBC 90
```

```{code-cell} python
!~/github/neuroh5/bin/neurotrees_copy --fill --output datasets/OLM_forest_Small.h5 datasets/OLM_tree.h5 OLM 143
```

```{code-cell} python
!mpirun -np 1 distribute-synapse-locs \
              --template-path ../templates \
              --config=Microcircuit_Small.yaml \
              --config-prefix=config_first_case \
              --populations PYR \
              --forest-path=./datasets/PYR_forest_Small.h5 \
              --output-path=./datasets/PYR_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

```{code-cell} python
!mpirun -np 1 distribute-synapse-locs \
              --template-path ../templates \
              --config=Microcircuit_Small.yaml \
              --config-prefix=config_first_case \
              --populations PVBC \
              --forest-path=./datasets/PVBC_forest_Small.h5 \
              --output-path=./datasets/PVBC_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

```{code-cell} python
!mpirun -np 1 distribute-synapse-locs \
             --template-path ../templates \
              --config=Microcircuit_Small.yaml \
              --config-prefix=config_first_case \
              --populations OLM \
              --forest-path=./datasets/OLM_forest_Small.h5 \
              --output-path=./datasets/OLM_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

## Generating connections

Here, we generate distance connection network and store it in `Microcircuit_Small_connections.h5` file.

> The schematic of the data structure can be found [here](../discussion/neuroh5.rst).

```{code-cell} python
!mpirun -np 8 generate-distance-connections \
    --config=Microcircuit_Small.yaml \
    --config-prefix=config_first_case \
    --forest-path=datasets/PYR_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

```{code-cell} python
!mpirun -np 8 generate-distance-connections \
    --config=Microcircuit_Small.yaml \
    --config-prefix=config_first_case \
    --forest-path=datasets/PVBC_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

```{code-cell} python
!mpirun -np 8 generate-distance-connections \
    --config=Microcircuit_Small.yaml \
    --config-prefix=config_first_case \
    --forest-path=datasets/OLM_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

## Creating input features and spike trains

```{code-cell} python
!mpirun -np 1 generate-input-features \
        -p STIM \
        --config=Microcircuit_Small.yaml \
        --config-prefix=config_first_case \
        --coords-path=datasets/Microcircuit_Small_coords.h5 \
        --output-path=datasets/Microcircuit_Small_input_features.h5 \
        -v
```

```{code-cell} python
!mpirun -np 2 generate-input-spike-trains \
             --config=Microcircuit_Small.yaml \
             --config-prefix=config_first_case \
             --selectivity-path=datasets/Microcircuit_Small_input_features.h5 \
             --output-path=datasets/Microcircuit_Small_input_spikes.h5 \
             --n-trials=3 -p STIM -v
```

# Finalizing

In the following steps, we collapse all H5 files into two files: __cell__ configuration, and __connectivity__ configuration. The simulator takes these two files to run the experiment.

## Define path and variable names

```{code-cell} python
import os, sys
import h5py, pathlib


def h5_copy_dataset(f_src, f_dst, dset_path):
    print(f"Copying {dset_path} from {f_src} to {f_dst} ...")
    target_path = str(pathlib.Path(dset_path).parent)
    f_src.copy(f_src[dset_path], f_dst[target_path])


h5types_file = "MiV_Small_h5types.h5"

MiV_populations = ["PYR", "OLM", "PVBC", "STIM"]
MiV_IN_populations = ["OLM", "PVBC"]
MiV_EXT_populations = ["STIM"]

MiV_cells_file = "MiV_Cells_Microcircuit_Small_20220410.h5"
MiV_connections_file = "MiV_Connections_Microcircuit_Small_20220410.h5"

MiV_coordinate_file = "Microcircuit_Small_coords.h5"

MiV_PYR_forest_file = "PYR_forest_Small.h5"
MiV_PVBC_forest_file = "PVBC_forest_Small.h5"
MiV_OLM_forest_file = "OLM_forest_Small.h5"

MiV_PYR_forest_syns_file = "PYR_forest_Small.h5"
MiV_PVBC_forest_syns_file = "PVBC_forest_Small.h5"
MiV_OLM_forest_syns_file = "OLM_forest_Small.h5"

MiV_PYR_connectivity_file = "Microcircuit_Small_connections.h5"
MiV_PVBC_connectivity_file = "Microcircuit_Small_connections.h5"
MiV_OLM_connectivity_file = "Microcircuit_Small_connections.h5"
```

```{code-cell} python
connectivity_files = {
    "PYR": MiV_PYR_connectivity_file,
    "PVBC": MiV_PVBC_connectivity_file,
    "OLM": MiV_OLM_connectivity_file,
}


coordinate_files = {
    "PYR": MiV_coordinate_file,
    "PVBC": MiV_coordinate_file,
    "OLM": MiV_coordinate_file,
    "STIM": MiV_coordinate_file,
}

distances_ns = "Arc Distances"
input_coordinate_ns = "Generated Coordinates"
coordinate_ns = "Coordinates"
coordinate_namespaces = {
    "PYR": input_coordinate_ns,
    "OLM": input_coordinate_ns,
    "PVBC": input_coordinate_ns,
    "STIM": input_coordinate_ns,
}


forest_files = {
    "PYR": MiV_PYR_forest_file,
    "PVBC": MiV_PVBC_forest_file,
    "OLM": MiV_OLM_forest_file,
}

forest_syns_files = {
    "PYR": MiV_PYR_forest_syns_file,
    "PVBC": MiV_PVBC_forest_syns_file,
    "OLM": MiV_OLM_forest_syns_file,
}


vecstim_file_dict = {"A Diag": "Microcircuit_Small_input_spikes.h5"}

vecstim_dict = {
    f"Input Spikes {stim_id}": stim_file
    for stim_id, stim_file in vecstim_file_dict.items()
}
```

## Collapse files

```{code-cell} python
%cd datasets
```

```{code-cell} python
## Creates H5Types entries
with h5py.File(MiV_cells_file, "w") as f:
    input_file = h5py.File(h5types_file, "r")
    h5_copy_dataset(input_file, f, "/H5Types")
    input_file.close()
```

```{code-cell} python
## Creates coordinates entries
with h5py.File(MiV_cells_file, "a") as f_dst:

    grp = f_dst.create_group("Populations")

    for p in MiV_populations:
        grp.create_group(p)

    for p in MiV_populations:
        coords_file = coordinate_files[p]
        coords_ns = coordinate_namespaces[p]
        coords_dset_path = f"/Populations/{p}/{coords_ns}"
        coords_output_path = f"/Populations/{p}/Coordinates"
        distances_dset_path = f"/Populations/{p}/Arc Distances"
        with h5py.File(coords_file, "r") as f_src:
            h5_copy_dataset(f_src, f_dst, coords_dset_path)
            h5_copy_dataset(f_src, f_dst, distances_dset_path)
```

```{code-cell} python
## Creates forest entries and synapse attributes
for p in MiV_populations:
    if p in forest_files:
        forest_file = forest_files[p]
        forest_syns_file = forest_syns_files[p]
        forest_dset_path = f"/Populations/{p}/Trees"
        forest_syns_dset_path = f"/Populations/{p}/Synapse Attributes"
        cmd = (
            f"h5copy -p -s '{forest_dset_path}' -d '{forest_dset_path}' "
            f"-i {forest_file} -o {MiV_cells_file}"
        )
        print(cmd)
        os.system(cmd)
        cmd = (
            f"h5copy -p -s '{forest_syns_dset_path}' -d '{forest_syns_dset_path}' "
            f"-i {forest_syns_file} -o {MiV_cells_file}"
        )
        print(cmd)
        os.system(cmd)
```

```{code-cell} python
## Creates vector stimulus entries
for (vecstim_ns, vecstim_file) in vecstim_dict.items():
    for p in MiV_EXT_populations:
        vecstim_dset_path = f"/Populations/{p}/{vecstim_ns}"
        cmd = (
            f"h5copy -p -s '{vecstim_dset_path}' -d '{vecstim_dset_path}' "
            f"-i {vecstim_file} -o {MiV_cells_file}"
        )
        print(cmd)
        os.system(cmd)

## Copy coordinates for STIM cells
p = "STIM"
cmd = f"h5copy -p -s '/Populations/{p}/Generated Coordinates' -d '/Populations/{p}/Coordinates' -i {MiV_cells_file} -o {MiV_cells_file}"
print(cmd)
os.system(cmd)
```

```{code-cell} python
with h5py.File(MiV_connections_file, "w") as f:
    input_file = h5py.File(h5types_file, "r")
    h5_copy_dataset(input_file, f, "/H5Types")
    input_file.close()

## Creates connectivity entries
for p in MiV_populations:
    if p in connectivity_files:
        connectivity_file = connectivity_files[p]
        projection_dset_path = f"/Projections/{p}"
        cmd = (
            f"h5copy -p -s {projection_dset_path} -d {projection_dset_path} "
            f"-i {connectivity_file} -o {MiV_connections_file}"
        )
        print(cmd)
        os.system(cmd)
```

```{code-cell} ipython3
%cd ..
```

# Run Network

```{code-cell} ipython3

!mpirun -np 8 run-network \
    --config-file=Microcircuit_Small.yaml  \
    --config-prefix=config_first_case \
    --arena-id=A \
    --stimulus-id=Diag \
    --template-paths=templates \
    --dataset-prefix="./datasets" \
    --results-path=results \
    --io-size=1 \
    --tstop=500 \
    --v-init=-75 \
    --results-write-time=60 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=0.49 \
    --dt 0.025 \
    --verbose
```
