#!/usr/bin/env python3

import os
import sys

import click
from miv_simulator import utils
from miv_simulator.simulator import distribute_synapse_locations


@click.command()
@click.option("--config", required=True, type=str)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
)
@click.option("--template-path", type=str)
@click.option(
    "--output-path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option(
    "--forest-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--populations", "-i", required=True, multiple=True, type=str)
@click.option("--distribution", type=str, default="uniform")
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--debug", is_flag=True)
def main(
    config,
    config_prefix,
    template_path,
    output_path,
    forest_path,
    populations,
    distribution,
    io_size,
    chunk_size,
    value_chunk_size,
    write_size,
    verbose,
    dry_run,
    debug,
):
    distribute_synapse_locations(
        config,
        template_path,
        output_path,
        forest_path,
        populations,
        distribution,
        io_size,
        chunk_size,
        value_chunk_size,
        write_size,
        verbose,
        dry_run,
        debug,
        config_prefix=config_prefix,
    )


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
