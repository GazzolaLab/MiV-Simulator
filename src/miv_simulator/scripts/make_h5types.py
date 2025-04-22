#!/usr/bin/env python3
import os
import sys

import click
from miv_simulator import utils
from miv_simulator.utils import io as io_utils
from miv_simulator.env import Env


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.option(
    "--config-prefix",
    type=click.Path(file_okay=False, dir_okay=True),
    default="",
    help="path to directory containing network and cell mechanism config files",
)
@click.option(
    "--output-path",
    default="MiV_h5types.h5",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option("--gap-junctions", is_flag=True)
def main(config, config_prefix, output_path, gap_junctions):
    env = Env(config=config, config_prefix=config_prefix)
    io_utils.make_h5types(env, output_path, gap_junctions, config_prefix=config_prefix)


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
