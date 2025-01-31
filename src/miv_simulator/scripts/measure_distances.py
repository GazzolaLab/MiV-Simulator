#!/usr/bin/env python3

import os
import sys

import click
from miv_simulator import utils
from miv_simulator.simulator.measure_distances import (
    measure_distances_ as measure_distances,
)


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory containing network and cell mechanism config files",
)
@click.option(
    "--coords-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--coords-namespace", type=str, default="Generated Coordinates")
@click.option(
    "--geometry-path",
    required=False,
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option("--populations", "-i", required=True, multiple=True, type=str)
@click.option("--resolution", type=(int, int, int), default=(30, 30, 10))
@click.option("--nsample", type=int, default=1000)
@click.option("--alpha-radius", type=float, default=100)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--verbose", "-v", is_flag=True)
def main(
    config,
    config_prefix,
    coords_path,
    coords_namespace,
    geometry_path,
    populations,
    resolution,
    nsample,
    alpha_radius,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    verbose,
):
    measure_distances(
        config,
        coords_path,
        coords_namespace,
        geometry_path,
        populations,
        resolution,
        nsample,
        alpha_radius,
        io_size,
        chunk_size,
        value_chunk_size,
        cache_size,
        verbose,
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
