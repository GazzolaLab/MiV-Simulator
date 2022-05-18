#!/bin/bash
#SBATCH -J run_microcircuit_mnist
#SBATCH -o ./results/run_microcircuit_mnist.%j.o
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=56
#SBATCH -t 2:00:00
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

results_path=$SCRATCH/MiV/results/Microcircuit_MNIST_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

ibrun python3 ./scripts/run_network.py  \
    --config-file=Microcircuit_MNIST_20220516.yaml  \
    --template-paths=templates \
    --dataset-prefix=${DATA_PREFIX} \
    --spike-input-path=${DATA_PREFIX}/Microcircuit/spiking_mnist_input.h5 \
    --spike-input-namespace='MNIST' \
    --spike-input-attr='Spike Train' \
    --results-path=${results_path} \
    --io-size=40 \
    --tstop=10000 \
    --v-init=-65 \
    --results-write-time=120 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=1.95 \
    --dt 0.025 \
    --verbose

