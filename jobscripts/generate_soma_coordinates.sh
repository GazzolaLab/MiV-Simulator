#!/bin/bash
#
#SBATCH -J generate_soma_coordinates
#SBATCH -o ./results/generate_soma_coordinates.%j.o
#SBATCH -N 1
#SBATCH --ntasks-per-node=56
#SBATCH -p normal      # Queue (partition) name
#SBATCH -t 1:30:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


mpirun.mpich -n 4 python3 ./scripts/generate_soma_coordinates.py -v \
    --config=Microcircuit.yaml \
    --types-path=./datasets/MiV_h5types.h5 \
    --output-path=./datasets/Microcircuit_coords.h5 \
    --output-namespace='Generated Coordinates' 




