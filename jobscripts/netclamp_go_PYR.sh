export DATA_PREFIX=./datasets

mpirun -n 1 python3  network_clamp.py go \
       -c Network_Clamp_PYR_gid_48041.yaml \
       --template-paths templates --dt 0.01 \
       -p PYR -g 48041  -t 9500 \
       --dataset-prefix $DATA_PREFIX \
       --config-prefix config \
       --arena-id A --trajectory-id Diag \
       --input-features-path $DATA_PREFIX/Microcircuit/MiV_input_features.h5 \
       --input-features-namespaces 'Constant Selectivity' \
       --results-path results/netclamp



