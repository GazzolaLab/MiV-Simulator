#!/bin/bash
mpirun.mpich -np 1 python3 ./scripts/distribute_synapse_locs.py \
             --template-path templates \
              --config=Microcircuit.yaml \
              --populations PYR \
              --forest-path=./datasets/PYR_forest.h5 \
              --output-path=./datasets/PYR_forest.h5 \
              --distribution=poisson \
              --io-size=1 -v --dry-run
