# MiV

Simulation and Analysis Code for Mind in Vitro

# Installation

## Generic installation

1. Core software libraries

- HDF5 (parallel build)
- MPICH

2. Python package prerequisites

```
pip install numpy scipy mpi4py h5py matplotlib Cython pyyaml sympy
```

3. Building and installing NEURON

```
git clone https://github.com/neuronsimulator/nrn.git
cd nrn
mkdir build
cd build
cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_RX3D=ON -DNRN_ENABLE_CORENEURON=ON -DNRN_ENABLE_PYTHON=ON -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
make install
```

4. Building and installing NeuroH5 

The NeuroH5 build system requires cmake.

```
git clone https://github.com/soltesz-lab/neuroh5.git
cd neuroh5
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install .
```

5. Fetching source code

```
# Main MiV repository
git clone https://github.com/GazzolaLab/MiV.git
```

## UIUC Campus Cluster installation

Make sure to run on a worker node, not the login-node!

### Dependencies

```sh
module load python/3
module load cmake/3.18.4
module load gcc/7.2.0
module load openmpi/4.1.0-gcc-7.2.0-pmi2
```

### Python environment

```python
python3 -m venv venv
. venv/bin/activate
pip install -U pip

pip install -I numpy scipy mpi4py h5py matplotlib Cython pyyaml sympy click
```

### Building HDF5

Download the source from [here](https://www.hdfgroup.org/downloads/hdf5/).

```
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz
tar -xzf hdf5-1.12.1.tar.gz
cd hdf5-1.12.1
CC=mpicc ./configure --prefix=$PWD/build --enable-parallel
make && make check
make install && make check-install
```

### Building NEURON simulator

```shell
git clone https://github.com/neuronsimulator/nrn.git
cd nrn
mkdir build
cd build
cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_RX3D=ON -DNRN_ENABLE_CORENEURON=ON -DNRN_ENABLE_PYTHON=ON -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_INSTALL_PREFIX=../install
make install

# in venv
cd ..
pip install .
```

### Install NeuroH5

(Optional if [GTest](https://github.com/google/googletest/releases) not available) 

```shell
wget https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
unzip release-1.11.0.zip
cd googletest-release-1.11.0
cmake -DBUILD_SHARED_LIBS=ON . -DCMAKE_INSTALL_PREFIX=./build
make
make install
```

**Install package**

```
# get the source
git clone https://github.com/soltesz-lab/neuroh5.git
cd neuroh5

# add HDF5 build to PATH
export HDF5_SOURCE=!!! fill hdf5-1.12.1/ directory here !!!
export PATH=$PATH:$HDF5_SOURCE/build

# in venv 
(make sure the node has enough RAM and cores, otherwise the compilation will fail)
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install .
```

# Running the main network simulation script

```
export PYTHONPATH=$PWD:$PYTHONPATH # Must include directory containing MiV repository


results_path=./results
export results_path

mkdir -p $results_path

cd MiV
mpirun python ./scripts/main.py \ # Main network simulation script
    --config-file=Test_Slice_10um.yaml  \ # Configuration file
    --arena-id=A --trajectory-id=Diag \ # Arena and trajectory identifier for simulated spatial input
    --template-paths=../dgc/Mateos-Aparicio2014:templates \ # Must include directory with DGC template
    --dataset-prefix="datasets" \ # Directory with HDF5 datasets
    --results-path=$results_path \
    --io-size=4 \ # Number of ranks performing I/O operations
    --tstop=50 \ # Simulation end time
    --v-init=-75 \
    --checkpoint-interval=10 \ # Simuation time interval for saving simulation outputs
    --checkpoint-clear-data \ # Clear data from memory after saving
    --max-walltime-hours=1 \ # Maximum walltime allotted 
    --verbose
```

# Example Configuration

## File Test_Microcircuit.yaml

```YAML
## Sample Model configuration of MiV network
Model Name: MiV
Dataset Name: Microcircuit
Definitions: !include Definitions.yaml
Global Parameters: !include Global.yaml
Geometry: !include Geometry.yaml
Random Seeds: !include Random.yaml
Cell Data: Test_Microcircuit.h5
Connection Data: Test_Microcircuit.h5
Connection Generator: !include Microcircuit_Connections.yaml
Stimulus: !include Input_Configuration.yaml
## Cell types for MiV model
Cell Types:
Cell Types:
  PYR:
    template: PoolosPyramidalCell
    synapses:
      density: !include PYR_synapse_density.yaml
  OLM:
    template: OLMCell
    synapses:
      density: !include OLM_synapse_density.yaml
  PVBC:
    template: PVBasketCell
    synapses:
      density: !include PVBC_synapse_density.yaml

```
