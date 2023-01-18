import os

# forward-compatible structured config alias - to be replaced with a dataclass
Blueprint = dict


def config_path(*append) -> str:
    return os.path.join(os.path.dirname(__file__), *append)
