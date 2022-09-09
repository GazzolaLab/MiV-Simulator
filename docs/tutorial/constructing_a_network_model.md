---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
file_format: mystnb
---

# Simulation Notebook

```{mermaid}
graph LR
D(Design</br>Experiment) -->|yaml| C(Construct</br>Simulation) -->|h5| R(Run) -->|h5| P(Post Process)
```

To run the first example simulation, first download the configuration files for this experiment:
- Download: [link](https://uofi.box.com/shared/static/a88dy7muglte90hklryw0xskv7ne13j0.zip).

> The detail description and configurability of each file is included [here](basic_configuration_yaml.md).

## Creating H5Types definitions

```sh
python ../scripts/make_h5types.py -c ../config/Microcircuit_Small.yaml --output-path datasets/MiV_Small_h5types.h5
```

```sh
h5ls -r ./datasets/MiV_Small_h5types.h5
```

```sh
h5dump -d /H5Types/Populations ./datasets/MiV_Small_h5types.h5
```

# Copying and compiling NMODL mechanisms

```sh
cp ../mechanisms/*.mod .
```

```sh
~/bin/nrnpython3/bin/nrnivmodl .
```

# Generating soma coordinates and measuring distances

```sh
python3 ../scripts/generate_soma_coordinates.py -v \
    --config=Microcircuit_Small.yaml \
    --types-path=./datasets/MiV_Small_h5types.h5 \
    --output-path=./datasets/Microcircuit_Small_coords.h5 \
    --output-namespace='Generated Coordinates'
```

```sh
mpirun.mpich -n 8 python3 ../scripts/measure_distances.py -v \
             -i PYR -i PVBC -i OLM -i STIM \
             --config=../config/Microcircuit_Small.yaml \
             --coords-path=./datasets/Microcircuit_Small_coords.h5
```

```sh
python ../scripts/plot_coords_in_volume.py \
--config ../config/Microcircuit_Small.yaml \
--coords-path datasets/Microcircuit_Small_coords.h5 \
-i PYR -i PVBC -i OLM
```

```sh
python ../scripts/plot_coords_in_volume.py \
--config ../config/Microcircuit_Small.yaml \
--coords-path datasets/Microcircuit_Small_coords.h5 \
-i STIM
```

# Distributing synapses along dendritic trees

```sh
~/src/neuroh5/bin/neurotrees_copy --fill --output datasets/PYR_forest_Small.h5 datasets/PYR_tree.h5 PYR 10
```

```sh
~/src/neuroh5/bin/neurotrees_copy --fill --output datasets/PVBC_forest_Small.h5 datasets/PVBC_tree.h5 PVBC 90
```

```sh
~/src/neuroh5/bin/neurotrees_copy --fill --output datasets/OLM_forest_Small.h5 datasets/OLM_tree.h5 OLM 143
```

```sh
mpirun.mpich -n 1 python3 ../scripts/distribute_synapse_locs.py \
             --template-path ../templates \
              --config=Microcircuit.yaml \
              --populations PYR \
              --forest-path=./datasets/PYR_forest_Small.h5 \
              --output-path=./datasets/PYR_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

```sh
mpirun.mpich -n 1 python3 ../scripts/distribute_synapse_locs.py \
             --template-path ../templates \
              --config=Microcircuit.yaml \
              --populations PVBC \
              --forest-path=./datasets/PVBC_forest_Small.h5 \
              --output-path=./datasets/PVBC_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

```sh
mpirun.mpich -n 1 python3 ../scripts/distribute_synapse_locs.py \
             --template-path ../templates \
              --config=Microcircuit.yaml \
              --populations OLM \
              --forest-path=./datasets/OLM_forest_Small.h5 \
              --output-path=./datasets/OLM_forest_Small.h5 \
              --distribution=poisson \
              --io-size=1 --write-size=0 -v
```

# Generating connections

```sh
mpirun.mpich -n 8 python3 ../scripts/generate_distance_connections.py \
    --config=Microcircuit_Small.yaml \
    --forest-path=datasets/PYR_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

```sh
mpirun.mpich -n 8 python3 ../scripts/generate_distance_connections.py \
    --config=Microcircuit_Small.yaml \
    --forest-path=datasets/PVBC_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

```sh

```

```sh
mpirun.mpich -n 8 python3 ../scripts/generate_distance_connections.py \
    --config=Microcircuit_Small.yaml \
    --forest-path=datasets/OLM_forest_Small.h5 \
    --connectivity-path=datasets/Microcircuit_Small_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_Small_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=100 -v
```

# Creating input features and spike trains

```sh
mpirun.mpich -n 1 python3 ../scripts/generate_input_features.py \
        -p STIM \
        --config=Microcircuit_Small.yaml \
        --coords-path=datasets/Microcircuit_Small_coords.h5 \
        --output-path=datasets/Microcircuit_Small_input_features.h5 \
        -v
```

```sh
mpirun.mpich -np 2 python3 ../scripts/generate_input_spike_trains.py \
             --config=Microcircuit_Small.yaml \
             --selectivity-path=datasets/Microcircuit_Small_input_features.h5 \
             --output-path=datasets/Microcircuit_Small_input_spikes.h5 \
             --n-trials=3 -p STIM -v
```

# Creating data files

```{code-cell} ipython3
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

```{code-cell} ipython3
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

```{code-cell} ipython3
%cd datasets
```

```{code-cell} ipython3
## Creates H5Types entries
with h5py.File(MiV_cells_file, "w") as f:
    input_file = h5py.File(h5types_file, "r")
    h5_copy_dataset(input_file, f, "/H5Types")
    input_file.close()
```

```{code-cell} ipython3
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

```{code-cell} ipython3
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

```{code-cell} ipython3
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

```{code-cell} ipython3
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

```{code-cell} ipython3

```
