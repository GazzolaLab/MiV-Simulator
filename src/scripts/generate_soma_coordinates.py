#!/usr/bin/env python3

##
## Generate soma coordinates within layer-specific volume.
##
import os
import sys

import click
from miv_simulator.simulator import generate_soma_coordinates
from miv_simulator.utils import list_find


@click.command()
@click.option("--config", required=True, type=str)
@click.option(
    "--config-prefix",
    required=False,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--types-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--geometry-path",
    required=False,
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option("--output-namespace", type=str, default="Generated Coordinates")
@click.option("--populations", "-i", type=str, multiple=True)
@click.option("--resolution", type=(int, int, int), default=(3, 3, 3))
@click.option("--alpha-radius", type=float, default=2500.0)
@click.option("--nodeiter", type=int, default=10)
@click.option("--dispersion-delta", type=float, default=0.1)
@click.option("--snap-delta", type=float, default=0.01)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--verbose", "-v", type=bool, default=False, is_flag=True)
def main(
    config,
    config_prefix,
    types_path,
    geometry_path,
    output_path,
    output_namespace,
    populations,
    resolution,
    alpha_radius,
    nodeiter,
    dispersion_delta,
    snap_delta,
    io_size,
    chunk_size,
    value_chunk_size,
    verbose,
):
    generate_soma_coordinates(
        config=config,
        types_path=types_path,
        output_path=output_path,
        geometry_path=geometry_path,
        output_namespace=output_namespace,
        populations=populations,
        resolution=resolution,
        alpha_radius=alpha_radius,
        nodeiter=nodeiter,
        dispersion_delta=dispersion_delta,
        snap_delta=snap_delta,
        io_size=io_size,
        chunk_size=chunk_size,
        value_chunk_size=value_chunk_size,
        verbose=verbose,
        config_prefix=config_prefix,
    )


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                list_find(
                    lambda x: os.path.basename(x) == os.path.basename(__file__),
                    sys.argv,
                )
                + 1
            ) :
        ]
    )
