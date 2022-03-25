import os, sys
import click
from biophys_microcircuit import env, utils, io_utils
from biophys_microcircuit.env import Env

@click.command()
@click.option("--config", '-c', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", default='biophys_microcircuit_h5types.h5', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--gap-junctions', is_flag=True)
def main(config, output_path, gap_junctions):

    env = Env(config_file=config)
    io_utils.make_h5types(env, output_path, gap_junctions=gap_junctions)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
