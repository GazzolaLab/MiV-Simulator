
mpirun.mpich -n 8 python3 ./scripts/measure_distances.py -v \
             -i PYR -i PVBC -i OLM \
             --config=config/Microcircuit.yaml \
             --coords-path=./datasets/Microcircuit_coords.h5
