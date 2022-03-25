# MiV
Simulation and Analysis Code for Mind in Vitro

# Installation

1. Core software libraries

HDF5 (parallel build)
MPICH

2. Python package prerequisites

```
pip install numpy scipy mpi4py h5py matplotlib 
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

# Running the main network simulation script

```
export PYTHONPATH=$PWD;$PYTHONPATH # Must include directory containing MiV repository


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
  PYR:
    template: DGC
    template file: DGC_Template_minimal.hoc
    mechanism file: 20200219_DG_GC_excitability_synint_combined_gid_0_mech.yaml
    synapses:
      correct_for_spines: True
      density: !include GC_synapse_density.yaml
  PVBC:
    template: BasketCell
    synapses:
      density: !include BC_synapse_density.yaml
  EC:
    template: MPPCell
    spike train:
      namespace: Input Spikes
      attribute: Spike Train

```
