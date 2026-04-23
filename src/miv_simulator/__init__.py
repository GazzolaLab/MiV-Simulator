__doc__ = """Python simulation software used in MiV project."""

from importlib import metadata as importlib_metadata

# Validate MPI environment on import (set MIV_SKIP_MPI_CHECK=1 to disable)
from miv_simulator.mpi_env import check_mpi_env as _check_mpi_env

_check_mpi_env()


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
