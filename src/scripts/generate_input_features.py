#!/usr/bin/env python3
import os
import sys

import click
from miv_simulator.simulator import generate_input_features
from miv_simulator.utils import list_find


@click.command()
@click.option("--config", required=True, type=str)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
)
@click.option(
    "--coords-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--distances-namespace", "-n", type=str, default="Arc Distances")
@click.option(
    "--output-path",
    type=click.Path(file_okay=True, dir_okay=False),
    default=None,
)
@click.option("--arena-id", type=str, default="A")
@click.option("--populations", "-p", type=str, multiple=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=10000)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--gather", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--debug-count", type=int, default=10)
@click.option("--plot", is_flag=True)
@click.option("--show-fig", is_flag=True)
@click.option("--save-fig", required=False, type=str, default=None)
@click.option(
    "--save-fig-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
)
@click.option("--font-size", type=float, default=14)
@click.option("--fig-format", required=False, type=str, default="svg")
@click.option("--dry-run", is_flag=True)
def main(
    config,
    config_prefix,
    coords_path,
    distances_namespace,
    output_path,
    arena_id,
    populations,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    write_size,
    verbose,
    gather,
    interactive,
    debug,
    debug_count,
    plot,
    show_fig,
    save_fig,
    save_fig_dir,
    font_size,
    fig_format,
    dry_run,
):
    generate_input_features(
        config,
        coords_path,
        distances_namespace,
        output_path,
        arena_id,
        populations,
        io_size,
        chunk_size,
        value_chunk_size,
        cache_size,
        write_size,
        verbose,
        gather,
        interactive,
        debug,
        debug_count,
        plot,
        show_fig,
        save_fig,
        save_fig_dir,
        font_size,
        fig_format,
        dry_run,
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
        ],
        standalone_mode=False,
    )
