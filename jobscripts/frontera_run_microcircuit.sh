#!/bin/bash
#SBATCH -J run_microcircuit
#SBATCH -o ./results/run_microcircuit.%j.o
#SBATCH --nodes=40
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

results_path=$SCRATCH/MiV/results/Microcircuit_$SLURM_JOB_ID
export results_path

mkdir -p ${results_path}

ibrun python3 ./scripts/run_network.py  \
    --config-file=Microcircuit_20220412.yaml  \
    --arena-id=A \
    --stimulus-id=Diag \
    --template-paths=templates \
    --dataset-prefix=${DATA_PREFIX} \
    --results-path=${results_path} \
    --io-size=40 \
    --tstop=1000 \
    --v-init=-65 \
    --results-write-time=120 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=0.95 \
    --dt 0.025 \
    --verbose

