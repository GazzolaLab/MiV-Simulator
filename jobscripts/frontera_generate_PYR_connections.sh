#!/bin/bash
#SBATCH -J generate_connections_MiV_PYR
#SBATCH -o ./results/generate_connections_MiV_PYR.%j.o
#SBATCH --nodes=8
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

ibrun python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Microcircuit.yaml \
    --forest-path=$DATA_PREFIX/Microcircuit/PYR_forest_syns_compressed.h5 \
    --connectivity-path=$DATA_PREFIX/Microcircuit/Microcircuit_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=$DATA_PREFIX/Microcircuit/Microcircuit_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=20 --cache-size=20 --write-size=0 -v

