"""Runtime MPI environment validation for miv-simulator.

Detects misconfigured MPI / parallel HDF5 setups that would otherwise
cause subtle, hard-to-diagnose errors at runtime.

Usage::

    from miv_simulator.mpi_env import check_mpi_env
    check_mpi_env()          # raises on hard errors, warns otherwise

Set the environment variable ``MIV_SKIP_MPI_CHECK=1`` to silence all checks.
"""

import os
import platform
import shutil
import subprocess
import warnings


class MPIEnvError(RuntimeError):
    """Raised when the MPI environment is fatally misconfigured."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _shared_lib_deps(path):
    system = platform.system()
    try:
        if system == "Linux":
            r = subprocess.run(
                ["ldd", path],
                capture_output=True,
                text=True,
                timeout=10,
            )
        elif system == "Darwin":
            r = subprocess.run(
                ["otool", "-L", path],
                capture_output=True,
                text=True,
                timeout=10,
            )
        else:
            return ""
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _mpi_lib_from_ldd(text):
    for line in text.splitlines():
        line = line.strip()
        if "libmpi" not in line:
            continue
        if "=>" in line:
            p = line.split("=>")[1].strip().split("(")[0].strip()
            if p:
                return p
        elif line.startswith("/"):
            return line.split("(")[0].strip()
    return None


def _mpicc_libdir():
    mpicc = shutil.which("mpicc")
    if not mpicc:
        return None
    try:
        r = subprocess.run(
            [mpicc, "--showme:libdirs"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split()[0]
    except Exception:
        pass
    try:
        r = subprocess.run(
            [mpicc, "-show"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            for tok in r.stdout.split():
                if tok.startswith("-L"):
                    return tok[2:]
    except Exception:
        pass
    return None


def _module_so(import_path):
    try:
        parts = import_path.split(".")
        mod = __import__(import_path)
        for p in parts[1:]:
            mod = getattr(mod, p)
        f = getattr(mod, "__file__", None)
        if f and (f.endswith(".so") or f.endswith(".pyd") or ".cpython-" in f):
            return f
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _same_mpi_library(lib_path, mpicc_libdir):
    """Check whether *lib_path* (from ldd) belongs to the same MPI as *mpicc_libdir*.

    On many Linux distros the runtime linker resolves the MPI soname to a
    path outside the ``-L`` directory advertised by ``mpicc``.  For example,
    MPICH on Ubuntu::

        ldd  -> /lib/x86_64-linux-gnu/libmpich.so.12
        mpicc -> -L/usr/lib/x86_64-linux-gnu/mpich/lib

    Both refer to the same library.  We therefore check:
    1. Whether realpath(lib_path) lives under realpath(mpicc_libdir), OR
    2. Whether the library *basename* pattern (e.g. libmpich*) exists in
       mpicc_libdir (possibly via symlinks resolving to the same file).
    """
    real_lib = os.path.realpath(lib_path)
    real_dir = os.path.realpath(mpicc_libdir)

    # Direct containment
    if real_lib.startswith(real_dir + os.sep) or real_lib.startswith(real_dir):
        return True

    # Check if any libmpi* in mpicc_libdir resolves to the same file
    basename = os.path.basename(real_lib)
    # Extract the core library name (e.g. "libmpich" from "libmpich.so.12")
    core_name = basename.split(".")[0]
    if os.path.isdir(mpicc_libdir):
        import glob

        for candidate in glob.glob(os.path.join(mpicc_libdir, core_name + "*")):
            if os.path.realpath(candidate) == real_lib:
                return True

    return False


def check_mpi_env(*, strict=False):
    """Validate the MPI environment.

    Parameters
    ----------
    strict : bool
        If *True*, missing packages (mpi4py / h5py) are treated as errors
        rather than warnings.

    Raises
    ------
    MPIEnvError
        If a fatal misconfiguration is detected.
    """
    if os.environ.get("MIV_SKIP_MPI_CHECK", "0") == "1":
        return

    mpicc = shutil.which("mpicc")
    if not mpicc:
        raise MPIEnvError(
            "'mpicc' not found on PATH. "
            "Install an MPI implementation (OpenMPI / MPICH) "
            "and make sure 'mpicc' is available."
        )

    mpi_libdir = _mpicc_libdir()

    # -- mpi4py --
    mpi4py_lib = None
    try:
        so = _module_so("mpi4py.MPI")
        if so and os.path.isfile(so):
            mpi4py_lib = _mpi_lib_from_ldd(_shared_lib_deps(so))
            if mpi4py_lib and mpi_libdir:
                if not _same_mpi_library(mpi4py_lib, mpi_libdir):
                    raise MPIEnvError(
                        f"mpi4py links against {mpi4py_lib} but mpicc uses "
                        f"{mpi_libdir}. mpi4py was likely installed from a "
                        "pre-built wheel. Reinstall from source: "
                        "pip install --no-binary=mpi4py mpi4py"
                    )
    except ImportError:
        msg = (
            "mpi4py is not installed. Install from source: "
            'env MPICC="mpicc --shared" pip install --no-binary=mpi4py --force-reinstall --no-cache-dir mpi4py'
        )
        if strict:
            raise MPIEnvError(msg)
        warnings.warn(msg, stacklevel=2)

    # -- h5py --
    h5py_lib = None
    try:
        import h5py

        if not getattr(h5py.get_config(), "mpi", False):
            raise MPIEnvError(
                "h5py is installed WITHOUT parallel-HDF5 (MPI) support. "
                "Reinstall from source: "
                'CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py --force-reinstall --no-cache-dir h5py'
            )
        for sub in ("h5py.h5", "h5py._conv", "h5py._errors"):
            so = _module_so(sub)
            if so:
                h5py_lib = _mpi_lib_from_ldd(_shared_lib_deps(so))
                if h5py_lib:
                    break
    except ImportError:
        msg = (
            "h5py is not installed. Install with MPI support: "
            'CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py --force-reinstall --no-cache-dir h5py'
        )
        if strict:
            raise MPIEnvError(msg)
        warnings.warn(msg, stacklevel=2)

    # -- cross-library consistency --
    if mpi4py_lib and h5py_lib:
        if os.path.realpath(mpi4py_lib) != os.path.realpath(h5py_lib):
            raise MPIEnvError(
                "mpi4py and h5py link against DIFFERENT MPI libraries:\n"
                f"  mpi4py -> {os.path.realpath(mpi4py_lib)}\n"
                f"  h5py   -> {os.path.realpath(h5py_lib)}\n"
                "Reinstall both from source against the same MPI."
            )
