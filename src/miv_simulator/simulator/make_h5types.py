from miv_simulator.env import Env
from miv_simulator.utils import io as io_utils


def make_h5types(config: str, output_file: str, gap_junctions: bool = False):
    env = Env(config=config)
    io_utils.make_h5types(env, output_file, gap_junctions=gap_junctions)
