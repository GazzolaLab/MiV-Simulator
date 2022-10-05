from dataclasses import dataclass
from typing import Dict, Optional

import yaml
from miv_simulator.config import config_path
from miv_simulator.utils import IncludeLoader


@dataclass
class FromYAMLConfig:
    network: Optional[Dict] = None


class HandlesYAMLConfig:
    def on_before_create(self):
        # copy base configuration
        with open(config_path("default.yaml")) as fp:
            config = yaml.load(fp, IncludeLoader)

        self.save_data("network_config.json", config)
