---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Simulation Notebook

> This tool extensively use NeuroH5 for simulation data structure. It is recommended to check this [discussion](../discussion/neuroh5.rst).

```{mermaid}
graph LR
D(Design</br>Experiment) -->|yaml| C(Construct</br>Simulation) -->|h5| R(Run) -->|h5| P(Post Process)
```

To run the first example simulation, first download the configuration files for this experiment:
- Download: [link](https://uofi.box.com/shared/static/a88dy7muglte90hklryw0xskv7ne13j0.zip).

> The detail description and configurability of each file is included [here](basic_configuration_yaml.md).

## Creating H5Types definitions

```python
import os
import matplotlib.pyplot as plt

datapath = "datasets"
os.makedirs(datapath, exist_ok=True)
```

```python
#!make-h5types --output-path datasets/MiV_Small_h5types.h5  # If config path is not specified, use default.
!make-h5types --config-prefix config -c Microcircuit_Small.yaml --output-path datasets/MiV_Small_h5types.h5
```

You can use HDF5 utilities `h5ls` and `h5dump` to examine the contents of an HDF5 file as we build the case study.

- h5ls: List each objects of an HDF5 file name and objects within the file. Apply method recursively with `-r` flag.
- h5dump: Display h5 contents in dictionary format, in human readable form.

For more detail, checkout [this page](https://www.asc.ohio-state.edu/wilkins.5/computing/HDF/hdf5tutorial/util.html).

```python
!h5ls -r ./datasets/MiV_Small_h5types.h5
```

```python
!h5dump -d /H5Types/Populations ./datasets/MiV_Small_h5types.h5
```

# Copying and compiling NMODL mechanisms

For more detail of NEURON NMODL, checkout [this page](http://web.mit.edu/neuron_v7.4/nrntuthtml/tutorial/tutD.html)

```python
!nrnivmodl .
```

# Generating soma coordinates and measuring distances

Here, we create `Microcircuit_Small_coords.h5` file that stores soma coordinate information. To see the contents in the file, try to use `h5dump` like above.

```python
!generate-soma-coordinates -v \
    --config=Microcircuit_Small.yaml \
    --config-prefix=config \
    --types-path=datasets/MiV_Small_h5types.h5 \
    --output-path=datasets/Microcircuit_Small_coords.h5 \
    --output-namespace='Generated Coordinates'
```

```python
!mpiexec -n 1 measure-distances -v \
             -i PYR -i PVBC -i OLM -i STIM \
             --config=Microcircuit_Small.yaml \
             --config-prefix=config \
             --coords-path=datasets/Microcircuit_Small_coords.h5
```

## Visualize

```python
%matplotlib inline
```

```python
!plot-coords-in-volume \
    --config config/Microcircuit_Small.yaml \
    --coords-path datasets/Microcircuit_Small_coords.h5 \
    -i PYR -i PVBC -i OLM
```

```python
!plot-coords-in-volume \
    --config config/Microcircuit_Small.yaml \
    --coords-path datasets/Microcircuit_Small_coords.h5 \
    -i STIM


```

# Creating dendritic trees in NeuroH5 format

`*.swc` file contains 3D point structure of the cell model. The tree model `*_tree.h5` can be created using `neurotree_import` feature from `neuroh5`.

```python
!~/github/neuroh5/bin/neurotrees_import PVBC datasets/PVBC_tree.h5 datasets/PVBC.swc

!~/github/neuroh5/bin/neurotrees_import PYR datasets/PYR_tree.h5 datasets/PYR.swc

!~/github/neuroh5/bin/neurotrees_import OLM datasets/OLM_tree.h5 datasets/OLM.swc

!h5copy -p -s '/H5Types' -d '/H5Types' -i datasets/MiV_Small_h5types.h5 -o datasets/PVBC_tree.h5
!h5copy -p -s '/H5Types' -d '/H5Types' -i datasets/MiV_Small_h5types.h5 -o datasets/PYR_tree.h5
!h5copy -p -s '/H5Types' -d '/H5Types' -i datasets/MiV_Small_h5types.h5 -o datasets/OLM_tree.h5
```

# Distributing synapses along dendritic trees

```python
!~/github/neuroh5/bin/neurotrees_copy --fill --output datasets/PYR_forest_Small.h5 datasets/PYR_tree.h5 PYR 10
```

```python
!~/github/neuroh5/bin/neurotrees_copy --fill --output datasets/PVBC_forest_Small.h5 datasets/PVBC_tree.h5 PVBC 90
```

```python
!~/github/neuroh5/bin/neurotrees_copy --fill --output datasets/OLM_forest_Small.h5 datasets/OLM_tree.h5 OLM 143
```

```python
!mpiexec -n 1 distribute-synapse-locs \
              --template-path ../templates \
              --config=config/Microcircuit_Small.yaml \
              --populations PYR \
              --forest-path=./datasets/PYR_forest_Small.h5 \
              --output-path=./datasets/PYR_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

```python
!mpiexec -n 1 distribute-synapse-locs \
             --template-path ../templates \
              --config=config/Microcircuit_Small.yaml \
              --populations PVBC \
              --forest-path=./datasets/PVBC_forest_Small.h5 \
              --output-path=./datasets/PVBC_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

```python
!mpiexec -n 1 distribute-synapse-locs \
             --template-path ../templates \
              --config=config/Microcircuit_Small.yaml \
              --populations OLM \
              --forest-path=./datasets/OLM_forest_Small.h5 \
              --output-path=./datasets/OLM_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

# Generating connections

Here, we generate distance connection network and store it in `Microcircuit_Small_connections.h5` file.

> The schematic of the data structure can be found [here](../discussion/neuroh5.rst).

```python
!mpiexec -n 8 generate-distance-connections \
    --config=config/Microcircuit_Small.yaml \
    --forest-path=datasets/PYR_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

```python
!mpiexec -n 8 generate-distance-connections \
    --config=config/Microcircuit_Small.yaml \
    --forest-path=datasets/PVBC_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

```python
!mpiexec -n 8 generate-distance-connections \
    --config=config/Microcircuit_Small.yaml \
    --forest-path=datasets/OLM_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

# Creating input features and spike trains

```python
!mpiexec -n 1 generate-input-features \
        -p STIM \
        --config=config/Microcircuit_Small.yaml \
        --coords-path=datasets/Microcircuit_Small_coords.h5 \
        --output-path=datasets/Microcircuit_Small_input_features.h5 \
        -v
```

```python
!mpiexec -np 2 generate-input-spike-trains \
             --config=config/Microcircuit_Small.yaml \
             --selectivity-path=datasets/Microcircuit_Small_input_features.h5 \
             --output-path=datasets/Microcircuit_Small_input_spikes.h5 \
             --n-trials=3 -p STIM -v
```

# Creating data files

```python
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

```python
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


vecstim_file_dict = {"A Diag": "MiV_input_spikes.h5"}

vecstim_dict = {
    f"Input Spikes {stim_id}": stim_file
    for stim_id, stim_file in vecstim_file_dict.items()
}
```

```python
%cd datasets
```

```python
## Creates H5Types entries
with h5py.File(MiV_cells_file, "w") as f:
    input_file = h5py.File(h5types_file, "r")
    h5_copy_dataset(input_file, f, "/H5Types")
    input_file.close()
```

```python
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

```python
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

```python
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
```

```python
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

```python
%cd ..
```

```python

```
