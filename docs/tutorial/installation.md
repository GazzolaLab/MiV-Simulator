# Installation

MiV-Simulator depends on MPI and parallel HDF5. The build system auto-detects
your MPI/HDF5 setup and forces source builds of critical packages (`mpi4py`,
`h5py`, `neuroh5`) so they link against the correct libraries.

## Prerequisites

| Dependency | Minimum version | Notes |
|---|---|---|
| Python | 3.9+ | |
| GCC | 7.8+ | C/C++ compiler |
| cmake | 3.18+ | Required by neuroh5 |
| MPI | — | MPICH **or** OpenMPI |
| Parallel HDF5 | 1.10+ | Must be built with MPI support |

### Ubuntu / Debian

```sh
sudo apt install build-essential cmake pkg-config
sudo apt install libmpich-dev          # or: libopenmpi-dev
sudo apt install libhdf5-mpich-dev     # or: libhdf5-openmpi-dev
```

### Fedora / RHEL

```sh
sudo dnf install gcc gcc-c++ cmake pkgconfig
sudo dnf install mpich-devel           # or: openmpi-devel
sudo dnf install hdf5-mpich-devel      # or: hdf5-openmpi-devel
```

### macOS (Homebrew)

```sh
brew install cmake open-mpi hdf5-mpi
```

## Quick Install

The recommended way to install is with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/GazzolaLab/MiV-Simulator.git
cd MiV-Simulator
make install
```

This runs `configure_mpi.py` to auto-detect your MPI and parallel HDF5, then
calls `uv sync` with the correct environment variables.

### What `make install` does

1. **`configure_mpi.py`** probes your system for `mpicc`, parallel HDF5 (via
   `pkg-config`, `h5pcc`, or common paths), and outputs shell `export` commands.
2. **`uv sync`** installs all dependencies. The `no-binary-package` setting in
   `pyproject.toml` forces `mpi4py`, `h5py`, and `neuroh5` to be compiled from
   source, linking against your system MPI and HDF5.
3. A **build-time check** validates the MPI environment when building
   miv-simulator's wheel.
4. A **runtime check** re-validates on every `import miv_simulator`, catching
   broken environments before they cause subtle errors.

### Manual Install (without Make)

If you prefer to run the steps yourself:

```sh
# 1. Auto-detect and export MPI/HDF5 environment
eval $(python configure_mpi.py)

# 2. Install
uv sync
```

Or set the variables explicitly:

```sh
export MPICC=/usr/bin/mpicc
export CC=/usr/bin/mpicc
export HDF5_MPI=ON
export HDF5_PKGCONFIG_NAME=hdf5-mpich   # Debian/Ubuntu
uv sync
```

## Environment Variables

The following environment variables control how MPI-dependent packages are built.
`configure_mpi.py` auto-detects all of them, but you can override any by
exporting before install.

| Variable | Purpose | Example |
|---|---|---|
| `MPICC` | MPI C compiler wrapper | `/usr/bin/mpicc` |
| `CC` | C compiler for extensions | `/usr/bin/mpicc` |
| `HDF5_MPI` | Enable parallel HDF5 in h5py | `ON` |
| `HDF5_PKGCONFIG_NAME` | pkg-config name for parallel HDF5 | `hdf5-mpich`, `hdf5-openmpi` |
| `HDF5_DIR` | HDF5 installation prefix | `/opt/hdf5-1.12.1/build` |
| `HDF5_INCLUDEDIR` | HDF5 include path (non-standard layouts) | `/usr/include/hdf5/mpich` |
| `HDF5_LIBDIR` | HDF5 library path (non-standard layouts) | `/usr/lib/x86_64-linux-gnu/hdf5/mpich` |
| `MIV_SKIP_MPI_CHECK` | Skip all MPI validation (`1` to disable) | `1` |

## Verifying the Installation

```sh
uv run python -c "import miv_simulator; print(miv_simulator.version)"
```

If the MPI environment is misconfigured, you will see a clear error message at
import time, for example:

> `miv_simulator.mpi_env.MPIEnvError: h5py is installed WITHOUT parallel-HDF5
> (MPI) support. Reinstall from source: CC=mpicc HDF5_MPI="ON" pip install
> --no-binary=h5py h5py`

## Docker

A Docker image is provided for quick experimentation (not recommended for
cluster/HPC use):

```sh
docker compose up
```

See `Dockerfile` and `compose.yaml` for details.

## Cluster / HPC Usage

For architecture-level optimization, build on a **worker node**, not the login
node.

### Load Modules

```sh
module load python/3
module load cmake
module load gcc
module load openmpi    # or: mpich
```

### Building Parallel HDF5 from Source

If your cluster does not provide a parallel HDF5 module:

```sh
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz
tar -xzf hdf5-1.12.1.tar.gz
cd hdf5-1.12.1
CC=mpicc ./configure --prefix=$PWD/build --enable-parallel --enable-shared
make -j$(nproc) && make install
```

Then point the installer to it:

```sh
export HDF5_DIR=$PWD/build
export HDF5_MPI=ON
export CC=mpicc
```

### Install MiV-Simulator

```sh
git clone https://github.com/GazzolaLab/MiV-Simulator.git
cd MiV-Simulator

# Auto-detect (or set HDF5_DIR/CC/MPICC manually as above)
eval $(python configure_mpi.py)
uv sync
```

### Building NEURON from Source (optional)

NEURON is installed automatically via `pip`, but for MPI-optimized builds:

```sh
git clone https://github.com/neuronsimulator/nrn.git
cd nrn
git submodule update --init --recursive
mkdir build && cd build
cmake .. \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_MPI=ON \
  -DNRN_ENABLE_RX3D=ON \
  -DNRN_ENABLE_CORENEURON=ON \
  -DNRN_ENABLE_PYTHON=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --parallel $(nproc) --target install

export PATH=$(pwd)/../install/bin:$PATH
export PYTHONPATH=$(pwd)/../install/lib/python:$PYTHONPATH
```

## Troubleshooting

**`Readline` not found (NEURON source build)**
: Install Readline (`apt install libreadline-dev`) or pass the path manually:
  `cmake -DReadline_INCLUDE_DIR=/usr/include -DReadline_LIBRARY=/usr/lib/x86_64-linux-gnu/libreadline.so ...`

> **Note**: If you encounter issues, please report them at
> [GitHub Issues](https://github.com/GazzolaLab/MiV-Simulator/issues).

## Additional Resources

- [Cluster module configuration](https://github.com/GazzolaLab/MiV-Simulator/tree/latest/support/cluster)

