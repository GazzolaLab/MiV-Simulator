#!/bin/bash
#SBATCH -J distribute_synapses_MiV_PYR
#SBATCH -o ./results/distribute_synapses_MiV_PYR.%j.o
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=56
#SBATCH -t 1:00:00
#SBATCH -p development      # Queue (partition) name
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

module load python3
module load phdf5

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH

export I_MPI_ADJUST_ALLTOALL=4
export I_MPI_ADJUST_ALLTOALLV=2
export I_MPI_ADJUST_ALLREDUCE=6

export DATA_PREFIX=$SCRATCH/striped2/MiV

ibrun python3 ./scripts/distribute_synapse_locs.py  -v \
    --template-path templates \
    --config=Microcircuit.yaml \
    --config-prefix=./config \
    --populations=PYR \
    --forest-path=$DATA_PREFIX/Microcircuit/PYR_forest_compressed.h5 \
    --output-path=$DATA_PREFIX/Microcircuit/PYR_forest_syns.h5 \
    --distribution=poisson \
    --io-size=4 --write-size=0 \
    --chunk-size=10000 --value-chunk-size=10000
