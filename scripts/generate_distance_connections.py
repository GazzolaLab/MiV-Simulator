#!/usr/bin/env python3
##
## Generates distance-weighted random connectivity between the specified populations.
##

import os
import os.path
import sys

import click
from miv_simulator import utils
from miv_simulator.simulator import generate_distance_connections


@click.command()
@click.option("--config", required=True, type=str)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
)
@click.option("--include", "-i", type=str, multiple=True)
@click.option(
    "--forest-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--connectivity-path", required=True, type=click.Path())
@click.option("--connectivity-namespace", type=str, default="Connectivity")
@click.option(
    "--coords-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--coords-namespace", type=str, default="Coordinates")
@click.option("--synapses-namespace", type=str, default="Synapse Attributes")
@click.option("--distances-namespace", type=str, default="Arc Distances")
@click.option("--resolution", type=(int, int, int), default=(30, 30, 10))
@click.option("--interp-chunk-size", type=int, default=1000)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=1)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--debug", is_flag=True)
def main(
    config,
    config_prefix,
    include,
    forest_path,
    connectivity_path,
    connectivity_namespace,
    coords_path,
    coords_namespace,
    synapses_namespace,
    distances_namespace,
    resolution,
    interp_chunk_size,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    write_size,
    verbose,
    dry_run,
    debug,
):
    generate_distance_connections(
        config,
        include,
        forest_path,
        connectivity_path,
        connectivity_namespace,
        coords_path,
        coords_namespace,
        synapses_namespace,
        distances_namespace,
        resolution,
        interp_chunk_size,
        io_size,
        chunk_size,
        value_chunk_size,
        cache_size,
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
