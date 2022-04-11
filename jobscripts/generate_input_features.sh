#!/bin/bash

dataset_prefix=./datasets/Microcircuit

mpirun.mpich -n 12 python3 ./scripts/generate_input_features.py \
        -p STIM \
        --config=Microcircuit.yaml \
        --config-prefix=./config \
        --coords-path=${dataset_prefix}/Microcircuit_coords.h5 \
        --output-path=${dataset_prefix}/MiV_input_features.h5 \
        -v

