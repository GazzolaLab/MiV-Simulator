"""Custom PEP 517 build backend for miv-simulator.

Wraps hatchling and adds MPI environment detection and validation to
prevent misconfigured MPI/HDF5 setups from causing subtle runtime errors.

Environment variables (set before ``uv sync`` / ``pip install``):

  MPICC       Path to the MPI C compiler wrapper.
              Default: auto-detected from PATH.

  HDF5_DIR    Prefix of a parallel-HDF5 installation.
              Default: auto-detected via ``h5pcc`` / ``h5cc`` / pkg-config.

  HDF5_MPI    Set to "ON" to tell h5py to enable MPI I/O.
              Default: auto-set to "ON" when an MPI-enabled HDF5 is found.

  HDF5_INCLUDEDIR
              Override the HDF5 include directory (for non-standard layouts).

  HDF5_LIBDIR
              Override the HDF5 library directory (for non-standard layouts).

  HDF5_PKGCONFIG_NAME
              pkg-config package name for HDF5 (e.g. "hdf5-mpich").
              If set, h5py uses pkg-config instead of HDF5_DIR.

  CC          C compiler used when building extension modules.
              Default: set to the value of MPICC (so h5py/neuroh5 use it).

  MIV_SKIP_MPI_CHECK
              Set to "1" to bypass all detection and validation.

Checks performed during wheel / editable builds:
  1. MPI compiler wrapper (mpicc) is on PATH
  2. mpi4py links against the detected system MPI (not a pre-built wheel)
  3. h5py has parallel HDF5 / MPI support enabled
  4. mpi4py and h5py link against the same MPI library
"""

import os
import platform
import shutil
import subprocess
import sys

from hatchling.build import (
    build_editable as _build_editable,
    build_sdist as _build_sdist,
    build_wheel as _build_wheel,
    get_requires_for_build_editable as _get_requires_for_build_editable,
    get_requires_for_build_sdist as _get_requires_for_build_sdist,
    get_requires_for_build_wheel as _get_requires_for_build_wheel,
)

try:
    from hatchling.build import (
        prepare_metadata_for_build_editable as _prepare_metadata_for_build_editable,
        prepare_metadata_for_build_wheel as _prepare_metadata_for_build_wheel,
    )
except ImportError:
    _prepare_metadata_for_build_wheel = None
    _prepare_metadata_for_build_editable = None


# ---------------------------------------------------------------------------
# MPI / HDF5 environment detection helpers
# ---------------------------------------------------------------------------


def _get_shared_lib_deps(path):
    """Return shared-library dependency listing for a binary/shared object."""
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


def _extract_mpi_lib(ldd_output):
    """Extract the resolved path to libmpi from ldd/otool output."""
    for line in ldd_output.splitlines():
        line = line.strip()
        if "libmpi" not in line:
            continue
        # Linux ldd format:  libmpi.so.40 => /usr/lib/libmpi.so.40 (0x...)
        if "=>" in line:
            path = line.split("=>")[1].strip().split("(")[0].strip()
            if path:
                return path
        # macOS otool format:  /usr/local/lib/libmpi.40.dylib (...)
        elif line.startswith("/"):
            path = line.split("(")[0].strip()
            if path:
                return path
    return None


def _find_mpicc():
    """Return the resolved path to mpicc, honouring the MPICC env var."""
    mpicc = os.environ.get("MPICC")
    if mpicc:
        path = shutil.which(mpicc.split()[0])  # handle "mpicc --shared"
        if path:
            return path
    return shutil.which("mpicc")


def _find_mpicc_libdir():
    """Detect the MPI library directory via mpicc."""
    mpicc = _find_mpicc()
    if not mpicc:
        return None
    # OpenMPI
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
    # MPICH
    try:
        r = subprocess.run(
            [mpicc, "-show"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            for token in r.stdout.split():
                if token.startswith("-L"):
                    return token[2:]
    except Exception:
        pass
    return None


def _find_hdf5_dir():
    """Auto-detect a parallel HDF5 installation prefix.

    Resolution order:
      1. ``HDF5_DIR`` environment variable (explicit user override)
      2. ``h5pcc`` on PATH  (parallel HDF5 compiler wrapper)
      3. ``h5cc``  on PATH  (may or may not be parallel)
      4. Common system prefixes with ``lib/libhdf5.so``

    Returns ``(prefix, includedir, libdir)`` — any element may be *None*.
    """
    # 1. Explicit
    hdf5_dir = os.environ.get("HDF5_DIR")
    if hdf5_dir and os.path.isdir(hdf5_dir):
        return hdf5_dir, None, None

    # 2/3. h5pcc or h5cc
    for wrapper in ("h5pcc", "h5cc"):
        w = shutil.which(wrapper)
        if not w:
            continue
        try:
            r = subprocess.run(
                [w, "-showconfig"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "Installation point:" in line:
                        prefix = line.split(":", 1)[1].strip()
                        if os.path.isdir(prefix):
                            return prefix, None, None
        except Exception:
            pass
        # h5pcc -show gives us include/lib dirs directly
        try:
            r = subprocess.run(
                [w, "-show"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                incdir = libdir = None
                for token in r.stdout.split():
                    if token.startswith("-I") and "hdf5" in token.lower():
                        incdir = token[2:]
                    if token.startswith("-L") and "hdf5" in token.lower():
                        libdir = token[2:]
                if libdir:
                    return None, incdir, libdir
        except Exception:
            pass
        # Fall back: assume <prefix>/bin/<wrapper>
        bindir = os.path.dirname(os.path.realpath(w))
        prefix = os.path.dirname(bindir)
        if os.path.isdir(os.path.join(prefix, "lib")):
            return prefix, None, None

    # 4. Common system paths
    for prefix in ("/usr", "/usr/local", "/opt/local", "/opt/homebrew"):
        for lib in ("lib", "lib64", "lib/x86_64-linux-gnu"):
            if os.path.exists(os.path.join(prefix, lib, "libhdf5.so")):
                return prefix, None, None
            if os.path.exists(os.path.join(prefix, lib, "libhdf5.dylib")):
                return prefix, None, None

    return None, None, None


def _find_hdf5_pkgconfig():
    """Find a pkg-config package name for parallel HDF5.

    Returns the package name (e.g. ``"hdf5-mpich"``) or *None*.
    """
    # User override
    pkg = os.environ.get("HDF5_PKGCONFIG_NAME")
    if pkg:
        return pkg

    # Check for common parallel-HDF5 pkg-config names
    for name in ("hdf5-mpich", "hdf5-openmpi", "hdf5"):
        try:
            r = subprocess.run(
                ["pkg-config", "--exists", name],
                capture_output=True,
                timeout=5,
            )
            if r.returncode == 0:
                # Verify it's actually parallel
                r2 = subprocess.run(
                    ["pkg-config", "--libs", name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if r2.returncode == 0 and ("mpi" in r2.stdout.lower()):
                    return name
        except Exception:
            pass
    return None


def _hdf5_has_parallel(hdf5_dir):
    """Best-effort check whether the HDF5 at *hdf5_dir* was built with MPI."""
    if not hdf5_dir:
        return None  # unknown

    # Check via h5pcc (only exists for parallel builds)
    h5pcc = shutil.which("h5pcc")
    if h5pcc and os.path.realpath(h5pcc).startswith(os.path.realpath(hdf5_dir)):
        return True

    # Check libhdf5.settings if present
    for subdir in ("lib", "lib64", "share/hdf5", "lib/x86_64-linux-gnu"):
        settings = os.path.join(hdf5_dir, subdir, "libhdf5.settings")
        if os.path.isfile(settings):
            try:
                with open(settings) as f:
                    for line in f:
                        if "Parallel HDF5" in line and "yes" in line.lower():
                            return True
                        if "Parallel HDF5" in line and "no" in line.lower():
                            return False
            except OSError:
                pass

    return None  # unknown


def _find_module_so(import_path):
    """Return the file-system path of a compiled extension module."""
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
# Environment propagation
# ---------------------------------------------------------------------------


def _setup_env():
    """Auto-detect MPI/HDF5 and set environment variables for dependency builds.

    This is called *before* hatchling resolves build dependencies so that
    packages like mpi4py, h5py and neuroh5 pick up the right compilers and
    library paths when they are built from source.
    """
    if os.environ.get("MIV_SKIP_MPI_CHECK", "0") == "1":
        return

    mpicc = _find_mpicc()
    changed = {}

    # -- MPICC -----------------------------------------------------------
    if mpicc and not os.environ.get("MPICC"):
        os.environ["MPICC"] = mpicc
        changed["MPICC"] = mpicc

    # -- CC (h5py, neuroh5 honour CC) ------------------------------------
    if mpicc and not os.environ.get("CC"):
        os.environ["CC"] = mpicc
        changed["CC"] = mpicc

    # -- HDF5 detection --------------------------------------------------
    # Prefer pkg-config (works reliably on Debian/Ubuntu where parallel HDF5
    # lives in a non-standard subdirectory like /usr/lib/.../hdf5/mpich/).
    hdf5_pkg = _find_hdf5_pkgconfig()
    if hdf5_pkg and not os.environ.get("HDF5_PKGCONFIG_NAME"):
        os.environ["HDF5_PKGCONFIG_NAME"] = hdf5_pkg
        changed["HDF5_PKGCONFIG_NAME"] = hdf5_pkg

    # If pkg-config isn't available, fall back to HDF5_DIR / explicit dirs
    if not hdf5_pkg:
        hdf5_dir, hdf5_incdir, hdf5_libdir = _find_hdf5_dir()
        if hdf5_dir and not os.environ.get("HDF5_DIR"):
            os.environ["HDF5_DIR"] = hdf5_dir
            changed["HDF5_DIR"] = hdf5_dir
        if hdf5_incdir and not os.environ.get("HDF5_INCLUDEDIR"):
            os.environ["HDF5_INCLUDEDIR"] = hdf5_incdir
            changed["HDF5_INCLUDEDIR"] = hdf5_incdir
        if hdf5_libdir and not os.environ.get("HDF5_LIBDIR"):
            os.environ["HDF5_LIBDIR"] = hdf5_libdir
            changed["HDF5_LIBDIR"] = hdf5_libdir
    else:
        hdf5_dir = None

    # -- HDF5_MPI --------------------------------------------------------
    if not os.environ.get("HDF5_MPI"):
        if hdf5_pkg:
            os.environ["HDF5_MPI"] = "ON"
            changed["HDF5_MPI"] = "ON"
        elif hdf5_dir and _hdf5_has_parallel(hdf5_dir) is True:
            os.environ["HDF5_MPI"] = "ON"
            changed["HDF5_MPI"] = "ON"
        elif mpicc:
            # Optimistic: if the user has MPI, they likely want parallel h5py
            os.environ["HDF5_MPI"] = "ON"
            changed["HDF5_MPI"] = "ON"

    if changed:
        summary = ", ".join(f"{k}={v}" for k, v in changed.items())
        print(
            f"  [miv-simulator] auto-configured build environment: {summary}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_mpi_environment():
    """Validate MPI environment.  Returns ``(errors, warnings)`` string lists."""
    errors = []
    warns = []

    # -- 1. MPI compiler wrapper --
    mpicc = _find_mpicc()
    if not mpicc:
        errors.append(
            "'mpicc' not found on PATH (and MPICC is not set).\n"
            "  Install an MPI implementation and make sure 'mpicc' is available,\n"
            "  or point MPICC to your MPI compiler wrapper.\n"
            "  Ubuntu/Debian : apt install libopenmpi-dev\n"
            "  Fedora/RHEL   : dnf install openmpi-devel\n"
            "  macOS (brew)  : brew install open-mpi"
        )
        return errors, warns  # remaining checks need mpicc

    mpi_libdir = _find_mpicc_libdir()

    # -- 2. mpi4py linkage --
    mpi4py_mpi_lib = None
    try:
        mpi4py_so = _find_module_so("mpi4py.MPI")
        if mpi4py_so and os.path.isfile(mpi4py_so):
            ldd = _get_shared_lib_deps(mpi4py_so)
            mpi4py_mpi_lib = _extract_mpi_lib(ldd)
            if mpi4py_mpi_lib and mpi_libdir:
                real_lib = os.path.realpath(mpi4py_mpi_lib)
                real_dir = os.path.realpath(mpi_libdir)
                if not real_lib.startswith(real_dir):
                    errors.append(
                        "mpi4py links against a DIFFERENT MPI than the system mpicc.\n"
                        f"  mpi4py  -> {mpi4py_mpi_lib}  (resolves to {real_lib})\n"
                        f"  mpicc   -> {mpi_libdir}  (resolves to {real_dir})\n"
                        "  This usually means mpi4py was installed from a pre-built wheel.\n"
                        "  Reinstall from source:\n"
                        "    pip install --no-binary=mpi4py --no-cache-dir mpi4py"
                    )
    except ImportError:
        warns.append(
            "mpi4py is not installed yet. Make sure to build it from source:\n"
            '  env MPICC="mpicc --shared" pip install --no-binary=mpi4py mpi4py'
        )

    # -- 3. h5py MPI support --
    h5py_mpi_lib = None
    try:
        import h5py

        h5py_cfg = h5py.get_config()
        if not getattr(h5py_cfg, "mpi", False):
            errors.append(
                "h5py is installed WITHOUT MPI (parallel HDF5) support.\n"
                "  miv-simulator requires h5py built against parallel HDF5.\n"
                "  Reinstall from source:\n"
                '    CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py h5py'
            )
        else:
            # try to check linkage via the low-level C extension
            for sub in ("h5py._conv", "h5py._errors", "h5py.h5"):
                h5py_so = _find_module_so(sub)
                if h5py_so:
                    ldd = _get_shared_lib_deps(h5py_so)
                    h5py_mpi_lib = _extract_mpi_lib(ldd)
                    if h5py_mpi_lib:
                        break
    except ImportError:
        warns.append(
            "h5py is not installed yet. Make sure to build it with MPI support:\n"
            '  CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py h5py'
        )

    # -- 4. Cross-library MPI consistency --
    if mpi4py_mpi_lib and h5py_mpi_lib:
        real_mpi4py = os.path.realpath(mpi4py_mpi_lib)
        real_h5py = os.path.realpath(h5py_mpi_lib)
        if real_mpi4py != real_h5py:
            errors.append(
                "mpi4py and h5py link against DIFFERENT MPI libraries.\n"
                f"  mpi4py -> {real_mpi4py}\n"
                f"  h5py   -> {real_h5py}\n"
                "  Both must use the same MPI.  Reinstall both from source\n"
                "  against the same MPI installation."
            )

    return errors, warns


def _run_checks():
    """Run MPI validation; raise on errors unless skipped via env var."""
    if os.environ.get("MIV_SKIP_MPI_CHECK", "0") == "1":
        return

    errors, warns = validate_mpi_environment()

    for w in warns:
        print(f"  WARNING [miv-simulator build]: {w}", file=sys.stderr)

    if errors:
        sep = "=" * 64
        body = "\n\n".join(f"  [{i+1}/{len(errors)}] {e}" for i, e in enumerate(errors))
        raise SystemExit(
            f"\n{sep}\n"
            f"  miv-simulator: MPI environment validation FAILED\n"
            f"{sep}\n\n"
            f"{body}\n\n"
            f"  Set MIV_SKIP_MPI_CHECK=1 to skip this check (not recommended).\n"
            f"{sep}\n"
        )


# ---------------------------------------------------------------------------
# PEP 517 / PEP 660 build backend interface  (delegates to hatchling)
# ---------------------------------------------------------------------------

# Run auto-detection at import time so env vars are set before the installer
# builds any dependency (mpi4py, h5py, neuroh5) from source.
_setup_env()


def get_requires_for_build_wheel(config_settings=None):
    return _get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _get_requires_for_build_editable(config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _run_checks()
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    # sdist is just packaging source; skip MPI checks
    return _build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _run_checks()
    return _build_editable(wheel_directory, config_settings, metadata_directory)


if _prepare_metadata_for_build_wheel is not None:

    def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
        return _prepare_metadata_for_build_wheel(metadata_directory, config_settings)


if _prepare_metadata_for_build_editable is not None:

    def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
        return _prepare_metadata_for_build_editable(metadata_directory, config_settings)
