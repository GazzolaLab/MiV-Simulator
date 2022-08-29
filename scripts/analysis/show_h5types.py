#!/usr/bin/env python3
import os
import sys

import click
from miv_simulator import utils
from miv_simulator.utils import io as io_utils


@click.command()
@click.option(
    "--input-path",
    "-p",
    default="MiV_h5types.h5",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
def main(input_path):

    io_utils.show_celltypes(input_path)


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                utils.list_find(
                    lambda x: os.path.basename(x) == os.path.basename(__file__),
                    sys.argv,
                )
                + 1
            ) :
        ]
    )
