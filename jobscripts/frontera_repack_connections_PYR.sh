#!/bin/bash

#SBATCH -J repack_connections_PYR        # Job name
#SBATCH -o ./results/repack_connections_PYR.o%j       # Name of stdout output file
#SBATCH -e ./results/repack_connections_PYR.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH -n 56            # Total # of mpi tasks
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load phdf5

set -x

export prefix=$SCRATCH/striped2/MiV/Microcircuit
export input=$prefix/Microcircuit_connections.h5
export output=$prefix/PYR_connections_compressed.h5

export H5TOOLS_BUFSIZE=$(( 96 * 1024 * 1024 * 1024))

h5repack -v -f SHUF -f GZIP=9 -i $input -o $output
