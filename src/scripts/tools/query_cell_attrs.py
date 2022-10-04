import os
import sys

import click
import numpy as np
from miv_simulator import utils
from miv_simulator.utils import io as io_utils


@click.command()
@click.option(
    "--input-path",
    "-p",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--populations", "-i", type=str, multiple=True)
def main(input_path, populations):
    """
    Query and show cell attributes in NeuroH5 file.

    .. code-block:: bash

       # Example Run
       % query-cell-attrs -p datasets/Microcircuit_Small_coords.h5 -i PYR

       numprocs=1
       Population PYR; Namespace: Arc Distances
          Attribute: U Distance
          Index: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
       34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
       58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
       82 83 84 85 86 87 88 89]
          Attribute: V Distance
          Index: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
       34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
       58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
       82 83 84 85 86 87 88 89]
       Population PYR; Namespace: Generated Coordinates
          Attribute: L Coordinate
          Index: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
       34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
       58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
       82 83 84 85 86 87 88 89]
          Attribute: U Coordinate
          Index: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
       34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
       58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
       82 83 84 85 86 87 88 89]
          Attribute: V Coordinate
          Index: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
       34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
       58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
       82 83 84 85 86 87 88 89]
          Attribute: X Coordinate
          Index: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
       34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
       58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
       82 83 84 85 86 87 88 89]
          Attribute: Y Coordinate
          Index: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
       34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
       58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
       82 83 84 85 86 87 88 89]
          Attribute: Z Coordinate
          Index: [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
       34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
       58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
       82 83 84 85 86 87 88 89]
    """

    namespace_id_lst, attr_info_dict = io_utils.query_cell_attributes(
        input_path, list(populations)
    )

    for population in populations:
        for this_namespace_id in sorted(namespace_id_lst):
            if this_namespace_id not in attr_info_dict[population]:
                continue
            print(f"Population {population}; Namespace: {this_namespace_id}")
            for attr_name, attr_cell_index in attr_info_dict[population][
                this_namespace_id
            ]:
                print(f"\tAttribute: {attr_name}")
                print(
                    f"\tIndex: {np.array2string(np.asarray(attr_cell_index))}"
                )
