#!/bin/bash
mpirun.mpich -np 1 python3 ./scripts/distribute_synapse_locs.py \
             --template-path templates \
              --config=Microcircuit.yaml \
              --populations PC \
              --forest-path=./datasets/PC.h5 \
              --output-path=./datasets/PC.h5 \
              --distribution=poisson \
              --io-size=1 -v
