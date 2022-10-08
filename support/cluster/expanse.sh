#-- CLI Prompts --#

read -rd '' globalhelp <<-EOF
    how to use setup.sh
    -------------------
    ./expanse.sh <options>

    options and explanations
    ---------------------------
      help : Print this help message

      path : Required. Path to install dependencies/modules (created if it does not exist).
EOF
if [[ $1 =~ ^([hH][eE][lL][pP]|[hH])$ ]]; then
    echo "${globalhelp}"
    exit 1
fi

if [ $# != 1 ]; then
    echo "You must pass one argument to setup modules, which"
    echo "is the directory where you want the dependencies to be built."
    exit 1
fi

set -e
source ~/.bashrc

#-- Set Modules --#

module purge

# SDSC Expanse - specific load
module load shared
module load cpu/0.15.4
module load slurm/expanse/21.08.8
module load sdsc/1.0

# Dependencies
module load gcc/10.2.0
module load openmpi cmake anaconda3

echo "DEBUG: Check cmake version:"
cmake --version

#-- Compiler Setup

echo "DEBUG: Check compiler versions:"
CC=mpicc
CXX=mpicxx
MPICC=mpicc
MPICXX=mpicxx
echo "C Compiler: " $CC
echo "C++ Compiler: " $CXX
echo "MPI-C Compiler: " $MPICC
echo "MPI-C++ Compiler: " $MPICXX
$MPICC --version
$MPICXX --version

#-- Installation --#

START_DIR=`pwd`
dep_dir=`realpath $1`
mkdir -p $dep_dir
cd $dep_dir

# Configuration
HDF5_MAJOR_VER=1
HDF5_MINOR_VER=12
HDF5_PATCH_VER=1
export HDF5_VER=${HDF5_MAJOR_VER}.${HDF5_MINOR_VER}.${HDF5_PATCH_VER}
export NEURON_VER=8.2.1

# Setup Python

PYTHON_VER=3.10
conda create -y -p $dep_dir/conda_env/miv python==$PYTHON_VER
PYTHON=$dep_dir/conda_env/miv/bin/python
PIP=$dep_dir/conda_env/miv/bin/pip
conda activate $dep_dir/conda_env/miv
echo "DEBUG: Check python/pip version and path:"
echo "   (Make sure python/pip from conda environment is being used.)"
which $PYTHON
$PYTHON --version
which $PIP
$PIP --version

# Install Poetry
#pip install poetry
#poetry --version

# Install Parallel HDF5 (source)
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_MAJOR_VER}.${HDF5_MINOR_VER}/hdf5-${HDF5_VER}/src/hdf5-${HDF5_VER}.tar.gz
tar -xzf hdf5-${HDF5_VER}.tar.gz
cd hdf5-${HDF5_VER}
./configure --prefix=$PWD/build --enable-parallel --enable-shared
make
make install
export HDF5_SOURCE=$PWD
export HDF5_ROOT=$PWD
export PATH=$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin:$PATH
# export LD_LIBRARY_PATH=$HDF5_SOURCE/build/lib:$LD_LIBRARY_PATH
# HDF5 test:
# NPROCS=4 make check-p
cd $dep_dir

# Install MPI4PY (pip)
env MPICC="$MPICC --shared" $PIP install --no-cache-dir mpi4py
$PIP install numpy

# Install h5py (pip)
HDF5_MPI="ON" HDF5_DIR=${HDF5_SOURCE}/build $PIP install --no-binary=h5py --no-deps h5py

# Install NeuroH5 (pip, source)
git clone https://github.com/iraikov/neuroh5.git
$PIP install --no-deps ./neuroh5
cd neuroh5
cmake .
make
export NEUROH5_SOURCE=$dep_dir/neuroh5
#export PATH=$NEUROH5_SOURCE/bin:$PATH
#which h5copy
#which neurotrees_import
cd $dep_dir

# Install Neuron (pip)
$PIP install --no-cache-dir neuron==${NEURON_VER}

# Install MiV Simulator (pip)
git clone https://github.com/GazzolaLab/MiV-Simulator
$PIP install --no-cache-dir ./MiV-Simulator

#-- Setup Module --#
mkdir -p $dep_dir/modules
cat >$dep_dir/modules/miv-simulator <<EOF
#%Module1.0
#

proc ModulesHelp { } {
    puts stdout "This module is expected to be used with python-environment conda_env/miv."
    puts stdout "We recommend to make a clone environment to protect original environment setup."
}

module-whatis "Require C/C++ libraries and Python modules for MiV-Simulator."
module-whatis "The module includes phdf5-$HDF5_VER, NEURON-$NEURON_VER, and NeuroH5"
module-whatis "Loading this module will additionally load openmpi, cmake, and anaconda."

prepend-path PATH "$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin"
prepend-path PATH "$NEUROH5_SOURCE/bin"
EOF

#-- Terminate --#
# Remove Configuration
conda deactivate

echo "MiV Dependency setup completed."
cd $START_DIR
