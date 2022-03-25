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
# Main dentate repository
git clone https://github.com/soltesz-lab/dentate.git

```

# Running the main network simulation script

```
export PYTHONPATH=$PWD;$PYTHONPATH # Must include directory containing dentate repository


results_path=./results
export results_path

mkdir -p $results_path

cd dentate
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

## File Test_Slice_10um.yaml

```YAML
## Sample Model configuration of dentate gyrus network
Model Name: dentatenet
Dataset Name: Slice
Definitions: !include Definitions.yaml
Global Parameters: !include Global.yaml
Geometry: !include Geometry.yaml
Random Seeds: !include Random.yaml
Cell Data: DG_Test_Slice_10um_20200628.h5
Connection Data: DG_Test_Slice_10um_20200628.h5
#Gap Junction Data: DG_gapjunctions_20181228.h5
Connection Generator: !include Full_Scale_Connections_GC_Exc_Sat_DD.yaml
Stimulus: !include Input_Configuration.yaml
## Cell types for dentate gyrus model
Cell Types:
  GC:
    template: DGC
    template file: DGC_Template_minimal.hoc
    mechanism file: 20200219_DG_GC_excitability_synint_combined_gid_0_mech.yaml
    synapses:
      correct_for_spines: True
      density: !include GC_synapse_density.yaml
  MC:
    template: MossyCell
    synapses:
      density: !include MC_synapse_density.yaml
  HC:
    template: HIPPCell
    synapses:
      density: !include HC_synapse_density.yaml
  BC:
    template: BasketCell
    synapses:
      density: !include BC_synapse_density.yaml
  AAC:
    template: AxoAxonicCell
    synapses:
      density: !include AAC_synapse_density.yaml
  HCC:
    template: HICAPCell
    synapses: 
      density: !include HCC_synapse_density.yaml
  NGFC:
    template: NGFCell
    synapses:
      density: !include NGFC_synapse_density.yaml
  MOPP:
    template: MOPPCell
    synapses:
      density: !include NGFC_synapse_density.yaml
  IS:
    template: ISCell
    synapses:
      density: !include IS_synapse_density.yaml
  MPP:
    template: MPPCell
    spike train:
      namespace: Input Spikes
      attribute: Spike Train
  LPP:
    template: LPPCell
    spike train:
      namespace: Input Spikes
      attribute: Spike Train
  CA3c:
    template: CA3Cell
    spike train:
      namespace: Input Spikes
      attribute: Spike Train

```
