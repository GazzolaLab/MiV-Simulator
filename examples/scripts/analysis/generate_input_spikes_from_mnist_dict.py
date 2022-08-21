import numpy as np
import pickle as pkl
from collections import defaultdict
from neuroh5.io import append_cell_attributes

chunk_size = 1000
value_chunk_size = 1000

with open("datasets/spiking_mnist_dictionary.pkl", "rb") as f:
    spiking_mnist_dictionary = pkl.load(f)


spiking_mnist_dictionary[0]["SPIKE_TRAIN"]


spike_train_namespace = "MNIST"
spike_train_attr_name = "Spike Train"

population = "STIM"

output_path = "spiking_mnist_input.h5"

trial_duration = 500.0

spike_attr_dict = defaultdict(list)
for trial in sorted(spiking_mnist_dictionary.keys()):
    for gid in spiking_mnist_dictionary[trial]["SPIKE_TRAIN"]:
        spiketrain = (
            spiking_mnist_dictionary[trial]["SPIKE_TRAIN"][gid] * 1000.0
            + trial * trial_duration
        )
        if len(spiketrain) > 0:
            spike_attr_dict[gid].append(spiketrain)

output_spike_attr_dict = dict(
    {
        k: {
            spike_train_attr_name: np.concatenate(
                spike_attr_dict[k], dtype=np.float32
            )
        }
        for k in spike_attr_dict
    }
)

print(output_spike_attr_dict)

append_cell_attributes(
    output_path,
    population,
    output_spike_attr_dict,
    namespace=spike_train_namespace,
)
