#!/usr/bin/env python3
import os
import sys

import click
from miv_simulator import utils
from miv_simulator.simulator import make_h5types


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--output-path",
    default="MiV_h5types.h5",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option("--gap-junctions", is_flag=True)
def main(config, output_path, gap_junctions):
    make_h5types(config, output_path, gap_junctions)


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
