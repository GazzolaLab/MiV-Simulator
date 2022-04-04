#!/bin/bash

mpirun.mpich -n 4 python3 ./scripts/run_network.py  \
    --config-file=Microcircuit_PYR.yaml  \
    --arena-id=A \
    --template-paths=templates \
    --dataset-prefix="datasets" \
    --results-path=results \
    --io-size=1 \
    --tstop=1000 \
    --v-init=-75 \
    --results-write-time=600 \
    --stimulus-onset=0.0 \
    --max-walltime-hours=0.49 \
    --dt 0.025 --use-coreneuron \
    --verbose

