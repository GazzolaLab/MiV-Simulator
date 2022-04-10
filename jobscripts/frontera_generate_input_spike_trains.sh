#!/bin/bash
#SBATCH -J generate_input_spike_trains_MiV
#SBATCH -o ./results/generate_input_spike_trains_MiV.%j.o
#SBATCH --nodes=1
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

ibrun -n 2 python3 ./scripts/generate_input_spike_trains.py \
    -p STIM \
    --config=Microcircuit.yaml \
    --config-prefix=./config \
    --selectivity-path=${DATA_PREFIX}/Microcircuit/MiV_input_features.h5 \
    --output-path=${DATA_PREFIX}/Microcircuit/MiV_input_spikes.h5 \
    --n-trials=3 -v


