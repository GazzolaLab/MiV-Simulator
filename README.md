<div align='center'>
<h1>MiV - Simulator</h1>

[![Documentation Status][badge-documentation]][link-documentation]
</div>

Bio-physical neural network simulator for Mind-in-Vitro in context of processing and computation.

## Documentation

The installation guideline and detailed documentation is available [here][link-documentation].

## Running the main network simulation script

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

## Example Configuration

### File Test_Microcircuit.yaml

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
[link-documentation]: https://miv-simulator.readthedocs.io/en/latest/?badge=latest

[badge-documentation]: https://readthedocs.org/projects/miv-simulator/badge/?version=latest

[source-hdf5]: https://www.hdfgroup.org/downloads/hdf5/
