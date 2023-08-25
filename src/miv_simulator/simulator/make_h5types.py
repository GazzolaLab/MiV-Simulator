from miv_simulator.env import Env
from miv_simulator.utils import io as io_utils


# !deprecated, call io_utils.create_h5types directly instead
def make_h5types(
    config: str, output_file: str, gap_junctions: bool = False, config_prefix=""
):
    env = Env(config=config, config_prefix=config_prefix)
    return io_utils.create_h5types(
        output_file,
        env.geometry["Cell Distribution"],
        env.connection_config,
        env.gapjunctions if gap_junctions else None,
        env.Populations,
    )
