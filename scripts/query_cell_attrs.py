import os, sys
import click
from biophys_microcircuit import env, utils, io_utils
from biophys_microcircuit.env import Env
import numpy as np

@click.command()
@click.option("--input-path", '-p', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', type=str, multiple=True)
def main(input_path, populations):

    namespace_id_lst, attr_info_dict = io_utils.query_cell_attributes(input_path, list(populations))

    for population in populations:
        for this_namespace_id in sorted(namespace_id_lst):
            if this_namespace_id not in attr_info_dict[population]:
                continue
            print(f"Population {population}; Namespace: {this_namespace_id}")
            for attr_name, attr_cell_index in attr_info_dict[population][this_namespace_id]:
                print(f"\tAttribute: {attr_name}")
                print(f"\tIndex: {np.array2string(np.asarray(attr_cell_index))}")

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
