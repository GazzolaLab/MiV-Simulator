#!/bin/bash

dataset_prefix=./datasets

mpirun.mpich -np 2 python3 ./scripts/generate_input_spike_trains.py \
             --config=Microcircuit.yaml \
             --config-prefix=./config \
             --selectivity-path=${dataset_prefix}/Microcircuit/MiV_input_features.h5 \
             --output-path=${dataset_prefix}/Microcircuit/MiV_input_spikes.h5 \
             --n-trials=1 -p STIM -v


