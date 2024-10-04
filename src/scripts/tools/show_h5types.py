
import click
from miv_simulator.utils import io as io_utils


@click.command()
@click.option(
    "--input-path",
    "-p",
    default="MiV_h5types.h5",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
def main(input_path):
    """
    CLI alias for :func:`show_celltypes <miv_simulator.utils.io.show_celltypes>`.

    .. code-block:: bash

       # Example Run
       % !show-h5types -p datasets/MiV_Small_h5types.h5

       numprocs=1
       Name       Start    Count
       ====       =====    =====
       STIM       0        10
       PYR        10       80
       PVBC       90       53
       OLM        143      44
    """
    io_utils.show_celltypes(input_path)
