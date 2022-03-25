#!/bin/bash

mpirun -n 1 python3 ./scripts/generate_distance_connections.py \
    --config-prefix=./config \
    --config=Full_Scale.yaml \
    --forest-path=datasets/PC_forest_syns.h5 \
    --connectivity-path=datasets/Microcircuit_connections.h5 \
    --connectivity-namespace=Connections \
    --coords-path=datasets/Microcircuit_coords.h5 \
    --coords-namespace='Generated Coordinates' \
    --io-size=1 --cache-size=20 --write-size=1 -v --dry-run 

