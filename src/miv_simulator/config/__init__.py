import os


def config_path(*append) -> str:
    return os.path.join(os.path.dirname(__file__), *append)
