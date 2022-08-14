import yaml
from miv_simulator.env import Env


def test_env_config(tmp_path):
    # loads default config
    assert Env(config=None).model_config["Global Parameters"]["celsius"] == 35.0

    # loads custom config from file
    c = str(tmp_path / "custom.yaml")
    with open(c, "w") as f:
        yaml.dump(
            {
                "Test": 1,
                "Cell Types": {},
                "Definitions": {
                    "Populations": {},
                    "SWC Types": {},
                    "Synapse Types": {},
                    "Layers": {},
                    "Input Selectivity Types": {},
                },
            },
            f,
        )
    assert Env(config=c).model_config["Test"] == 1

    # applies patches
    assert (
        Env(config={"Global Parameters": {"celsius": 1}}).model_config[
            "Global Parameters"
        ]["celsius"]
        == 1
    )
