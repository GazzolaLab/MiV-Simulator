from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BaseConfig:
    blueprint: Optional[Dict] = None
