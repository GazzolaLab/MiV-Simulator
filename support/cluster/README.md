# Tools and scripts to aid cluster usage

## Module installation

### Installation scripts

Here we provide example scripts to setup modules for the cluster environment.

```bash
# e.g. `bash <installation script> <path to install>
bash expanse.sh ${HOME}/workspace/MiV
```

1. [Expanse(UCSD)](expanse.sh)

> Setting up dependencies might take some time (0.5-1 hr).
> Depending on the user allocation and type of available nodes, some modules within the script might needs adjustments. Here, the scripts are intended to serve as a template. If you have successful setup on any other cluster or environment, please feel free to make PR and append the setup scripts here.

### Module usage

```bash
module use ${HOME}/workspace/MiV/modules
module load miv-simulator
conda activate ${HOME}/workspace/conda_env/miv
```

```bash
$ module avail
------------------------------------------------- test/modules -------------------------------------------------
   miv-simulator (L)
```

### Module help strings

```bash
$ module help miv-simulator

----------------------------------- Module Specific Help for "miv-simulator" -----------------------------------
This module is expected to be used with python-environment conda_env/miv.
We recommend to make a clone environment to protect original environment setup.
```

```bash
$ module whatis miv-simulator
miv-simulator       : Require C/C++ libraries and Python modules for MiV-Simulator.
miv-simulator       : The module includes phdf5-1.12.1, NEURON-8.2.1, and NeuroH5
miv-simulator       : Loading this module will additionally load openmpi, cmake, and anaconda.
```
