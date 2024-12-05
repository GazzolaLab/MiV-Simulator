#!/usr/bin/env python3
import os
import sys

import click
from miv_simulator import utils
from miv_simulator.env import Env
from miv_simulator.utils import io as io_utils


def make_h5types(
    config: str, output_file: str, gap_junctions: bool = False, config_prefix=""
):
    env = Env(config=config, config_prefix=config_prefix)
    return io_utils.create_neural_h5(
        output_file,
        env.geometry["Cell Distribution"],
        env.connection_config,
        env.Populations,
        env.gapjunctions if gap_junctions else None,
    )


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
    make_h5types(config, output_path, gap_junctions, config_prefix=config_prefix)


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
