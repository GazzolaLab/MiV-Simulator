<div align='center'>
<h1>MiV - Simulator</h1>
</div>

Bio-physical neural network simulator for Mind-in-Vitro in context of processing and computation.

## Generic installation

The installation consist two part: python-dependencies and external libraries.

### Core Software/Libraries

Before the installation, following software/libraries should be installed on the system.

- Python 3.8+
- GCC 7.8+, cmake 3.18+
- HDF5 (parallel build)
- MPICH
- OpenMPI (for cluster)

### Python Dependencies

The python-dependencies are managed through [`poetry`][link-poetry-website]. We provide important alias in `makefile`.

```sh
make poetry-downloads  # Install poetry on the system
make install           # Install MiV-Simulator
```

### External Libraries

1. Building and installing NEURON

```
git clone https://github.com/neuronsimulator/nrn.git
cd nrn
mkdir build
cd build
cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_RX3D=ON -DNRN_ENABLE_CORENEURON=ON -DNRN_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
make install
```

2. Building and installing NeuroH5

The NeuroH5 build system requires cmake.

```
git clone https://github.com/soltesz-lab/neuroh5.git
cd neuroh5
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install .
```

## Cluster Usage

For architecture-level optimization, some of the core libraries should be built within the worker node.

> **NOTE:** Make sure to run on a worker node, not the login-node!

### Necessary Modules

```sh
module load python/3
module load cmake
module load gcc
module load openmpi
```

### Building HDF5 (Parallel)

Download the source from [here][source-hdf5].

```sh
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz
tar -xzf hdf5-1.12.1.tar.gz
cd hdf5-1.12.1
CC=mpicc ./configure --prefix=$PWD/build --enable-parallel
make && make check
make install && make check-install
```

### Install/Build Python Dependencies

```sh
make poetry-download
make install
```

### Building NEURON simulator

```sh
git clone https://github.com/neuronsimulator/nrn.git
cd nrn
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_RX3D=ON -DNRN_ENABLE_CORENEURON=ON -DPYTHON_EXECUTABLE=$(which python3) -DNRN_ENABLE_PYTHON=ON -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --parallel 8 --target install

export PATH=<Installation path>/bin:$PATH
export PYTHONPATH=<Installation path>/lib/python:$PYTHONPATH
```

<details>
  <summary>Possible Issues</summary>
  
- `Readline` cannot be found:
    - Try to install `Readline` using `apt` or `yum`. It can also be installed using `conda`.
    - `Readline` might already exist on the system. Search in `/usr/lib` or `/usr/lib64`.
    - Pass environment variable directly: `cmake -DReadline_INCLUDE_DIR=/usr/lib64 -DReadline_LIBRARY=/usr/lib64/libreadline.so.7 ....`
  
</details>
<br/>

### Install NeuroH5

(Optional if [GTest](https://github.com/google/googletest/releases) not available)
```sh
wget https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
unzip release-1.11.0.zip
cd googletest-release-1.11.0
cmake -DBUILD_SHARED_LIBS=ON . -DCMAKE_INSTALL_PREFIX=./build
make
make install
```

**Install package**

```sh
# get the source
git clone https://github.com/soltesz-lab/neuroh5.git
cd neuroh5

# add HDF5 build to PATH
export HDF5_SOURCE=<HDF5 installation directory>
export PATH=$PATH:$HDF5_SOURCE/build

(make sure the node has enough RAM and cores, otherwise the compilation will fail)
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install .

export PATH=<NeuroH5 installation path>/bin:$PATH
```

# Running the main network simulation script

```sh
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

[//]: # (Collection of URLs)

[link-poetry-website]: https://python-poetry.org/

[source-hdf5]: https://www.hdfgroup.org/downloads/hdf5/
