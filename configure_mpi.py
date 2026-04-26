#!/usr/bin/env python3
"""Detect MPI and parallel-HDF5 settings for miv-simulator.

Usage:
    # Print export commands to eval in your shell:
    eval $(python configure_mpi.py)

    # Then install:
    uv sync

    # Or as a one-liner:
    eval $(python configure_mpi.py) && uv sync

The script auto-detects:
  - MPICC         MPI C compiler wrapper
  - CC            C compiler (set to mpicc for h5py/neuroh5 builds)
  - HDF5_MPI      Enables parallel HDF5 in h5py
  - HDF5_DIR      HDF5 installation prefix (if standard layout)
  - HDF5_PKGCONFIG_NAME  pkg-config package for parallel HDF5
  - HDF5_INCLUDEDIR      HDF5 include path (non-standard layouts)
  - HDF5_LIBDIR          HDF5 library path (non-standard layouts)

Override any variable by exporting it before running this script.
"""

import os
import shutil
import subprocess
import sys


# Substrings in `pkg-config --libs`, prefix paths, or lib/ filenames that
# indicate a parallel HDF5 build. Expanded for Cray PE (whose libraries are
# named `libhdf5_parallel*` and whose prefix path contains `/hdf5-parallel/`)
# in addition to the common MPICH / OpenMPI naming.
_PARALLEL_MARKERS = ("mpi", "parallel", "gtl")


def _find_mpicc():
    mpicc = os.environ.get("MPICC")
    if mpicc:
        path = shutil.which(mpicc.split()[0])
        if path:
            return path
    return shutil.which("mpicc")


def _find_mpicxx():
    mpicxx = os.environ.get("MPICXX")
    if mpicxx:
        path = shutil.which(mpicxx.split()[0])
        if path:
            return path
    for name in ("mpicxx", "mpic++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _find_hdf5_pkgconfig():
    pkg = os.environ.get("HDF5_PKGCONFIG_NAME")
    if pkg:
        return pkg
    for name in ("hdf5-mpich", "hdf5-openmpi", "hdf5"):
        try:
            r = subprocess.run(
                ["pkg-config", "--exists", name],
                capture_output=True,
                timeout=5,
            )
            if r.returncode == 0:
                r2 = subprocess.run(
                    ["pkg-config", "--libs", name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # Also query the prefix so we can recognize Cray's path-based
                # "parallel" naming (Cray's hdf5.pc has no 'mpi' substring in
                # the libs line but has /hdf5-parallel/ in the prefix).
                r_prefix = subprocess.run(
                    ["pkg-config", "--variable=prefix", name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                blob = (r2.stdout + " " + r_prefix.stdout).lower()
                if r2.returncode == 0 and any(m in blob for m in _PARALLEL_MARKERS):
                    return name
        except Exception:
            pass
    return None


def _find_hdf5_via_wrapper():
    """Detect HDF5 prefix/includedir/libdir from h5pcc or h5cc."""
    for wrapper in ("h5pcc", "h5cc"):
        w = shutil.which(wrapper)
        if not w:
            continue
        # Try -showconfig for prefix
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
        # Try -show for -I/-L flags
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
    return None, None, None


def _looks_parallel(hdf5_dir):
    """Return True if hdf5_dir appears to be a parallel HDF5 install."""
    if not hdf5_dir:
        return False
    # Path-based check (Cray: /opt/cray/pe/hdf5-parallel/...)
    if any(m in hdf5_dir.lower() for m in _PARALLEL_MARKERS):
        return True
    # File-based check: look for libhdf5_parallel*, libhdf5_mpi*, etc.
    libdir = os.path.join(hdf5_dir, "lib")
    if os.path.isdir(libdir):
        try:
            libs = os.listdir(libdir)
            for n in libs:
                n_low = n.lower()
                if any(m in n_low for m in _PARALLEL_MARKERS):
                    return True
        except OSError:
            pass
    return False


def detect():
    """Detect MPI/HDF5 environment and return dict of env vars to set."""
    env = {}

    mpicc = _find_mpicc()
    if not mpicc:
        print(
            "# ERROR: mpicc not found on PATH.\n"
            "# Install an MPI implementation first:\n"
            "#   Ubuntu/Debian: apt install libopenmpi-dev (or libmpich-dev)\n"
            "#   Fedora/RHEL:   dnf install openmpi-devel\n"
            "#   macOS (brew):  brew install open-mpi",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("MPICC"):
        env["MPICC"] = mpicc
    if not os.environ.get("CC"):
        env["CC"] = mpicc

    mpicxx = _find_mpicxx()
    if mpicxx:
        if not os.environ.get("MPICXX"):
            env["MPICXX"] = mpicxx
        if not os.environ.get("CXX"):
            env["CXX"] = mpicxx

    # Prefer pkg-config (handles Debian/Ubuntu non-standard paths)
    hdf5_pkg = _find_hdf5_pkgconfig()
    if hdf5_pkg and not os.environ.get("HDF5_PKGCONFIG_NAME"):
        env["HDF5_PKGCONFIG_NAME"] = hdf5_pkg
    elif not hdf5_pkg:
        hdf5_dir = os.environ.get("HDF5_DIR")
        if hdf5_dir:
            # Re-export HDF5_DIR so child build processes (e.g. uv's isolated
            # PEP 517 build environments for h5py / neuroh5) reliably inherit
            # it, and warn if the path doesn't look like a parallel build.
            if _looks_parallel(hdf5_dir):
                env["HDF5_DIR"] = hdf5_dir
            else:
                print(
                    f"# WARNING: HDF5_DIR={hdf5_dir} does not look like a "
                    f"parallel HDF5 install. h5py may build without MPI "
                    f"support. Expected path or lib/ to contain one of: "
                    f"{', '.join(_PARALLEL_MARKERS)}.",
                    file=sys.stderr,
                )
                env["HDF5_DIR"] = hdf5_dir
        else:
            prefix, incdir, libdir = _find_hdf5_via_wrapper()
            if prefix:
                env["HDF5_DIR"] = prefix
            if incdir and not os.environ.get("HDF5_INCLUDEDIR"):
                env["HDF5_INCLUDEDIR"] = incdir
            if libdir and not os.environ.get("HDF5_LIBDIR"):
                env["HDF5_LIBDIR"] = libdir

    if not os.environ.get("HDF5_MPI"):
        env["HDF5_MPI"] = "ON"

    # neuroh5's setup.py / CMake reads HDF5_ROOT (not HDF5_DIR) to locate the
    # parallel HDF5 install. Without it, CMake's FindHDF5 may fall back to a
    # serial HDF5 (or fail to detect the parallel header), causing
    # HDF5_IS_PARALLEL=FALSE in the build cache. The result is a runtime
    # failure: "collective access for MPI-based drivers only" because the
    # H5Pset_fapl_mpio calls were #ifdef'd out at build time.
    detected_hdf5 = env.get("HDF5_DIR") or os.environ.get("HDF5_DIR")
    if detected_hdf5 and not os.environ.get("HDF5_ROOT"):
        env["HDF5_ROOT"] = detected_hdf5

    is_cray_pe = bool(
        os.environ.get("CRAY_MPICH_DIR")
        or os.environ.get("CRAY_HDF5_PARALLEL_DIR")
        or os.environ.get("PE_ENV")
    )

    # On Cray PE with GPU acceleration (e.g. DeltaAI / Cray EX), h5py /
    # mpi4py also transitively link Cray's CUDA-aware MPI GTL plugin
    # (libmpi_gtl_cuda.so) alongside the actual MPI library
    # (libmpi_gnu_*.so)
    has_cray_gpu = bool(
        os.environ.get("CRAY_ACCEL_TARGET") or os.environ.get("CRAY_ACCEL_VENDOR")
    )
    if is_cray_pe and has_cray_gpu and "MIV_SKIP_MPI_CHECK" not in os.environ:
        env["MIV_SKIP_MPI_CHECK"] = "1"

    # On Cray PE the default `cc` / `CC` compiler wrappers cause CMake's
    # FindHDF5 to take the "no interrogate" path (HDF5 is already provided
    # by the wrapper) and never populate HDF5_INCLUDE_DIRS. neuroh5's
    # find_hdf5() macro then iterates an empty list, leaves
    # HDF5_IS_PARALLEL=FALSE, and #ifdef's out every H5Pset_fapl_mpio call —
    # producing a runtime error "collective access for MPI-based drivers
    # only" the first time scatter_read_cell_attributes runs.
    #
    # We can't use a different CC (the gnu mpicc wrappers link libhugetlbfs
    # without proper init and segfault on import outside srun, and miss the
    # CUDA-aware GTL plugin). Instead, install a tiny `cmake` shim that
    # patches neuroh5's find_hdf5() macro before invoking cmake. This is a
    # no-op against neuroh5 versions where the macro already probes the
    # Cray HDF5 install (newer iohk-ipls/neuroh5).
    if is_cray_pe and detected_hdf5 and not os.environ.get("CMAKE_PATH"):
        import tempfile

        shim_dir = os.path.join(
            tempfile.gettempdir(), f"miv_cmake_shim_{os.getuid()}"
        )
        os.makedirs(shim_dir, exist_ok=True)
        shim = os.path.join(shim_dir, "cmake")
        real_cmake = shutil.which("cmake") or "cmake"
        with open(shim, "w") as f:
            f.write(
                "#!/bin/sh\n"
                "# Auto-generated by configure_mpi.py for Cray PE.\n"
                "# Patch neuroh5's find_hdf5() macro before invoking cmake so\n"
                "# that HDF5_IS_PARALLEL is set unconditionally — find_package\n"
                "# (HDF5) takes the NO_INTERROGATE path on Cray and never\n"
                "# populates HDF5_INCLUDE_DIRS, defeating the macro's probe.\n"
                "for arg in \"$@\"; do\n"
                "    if [ -d \"$arg\" ] && [ -f \"$arg/cmake/neuroh5_utils.cmake\" ]; then\n"
                "        f=\"$arg/cmake/neuroh5_utils.cmake\"\n"
                "        if ! grep -q MIV_PATCHED \"$f\"; then\n"
                "            sed -i 's|set( HDF5_IS_PARALLEL FALSE )|set( HDF5_IS_PARALLEL TRUE ) # MIV_PATCHED|' \"$f\"\n"
                "        fi\n"
                "        break\n"
                "    fi\n"
                "done\n"
                f"exec {real_cmake} \"$@\"\n"
            )
        os.chmod(shim, 0o755)
        # neuroh5's setup.py hardcodes the literal string "cmake" when
        # invoking the build (CMAKE_PATH is only used for an existence
        # probe), so prepend the shim directory to PATH so our wrapper is
        # picked up instead of the real cmake.
        env["CMAKE_PATH"] = shim
        env["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")

    # On Cray PE the `cc` / `CC` / `mpicc` wrappers do not auto-add the
    # Python include directory the way stock `gcc` does, so packages with
    # C extensions whose setup paths don't explicitly inject -I<sysconfig
    # include> (e.g. `treverhines-rbf`, `mpi4py` --no-binary) fail with
    # `fatal error: Python.h: No such file or directory`. Inject CPPFLAGS
    # pointing at the *project* interpreter's headers (uv's managed CPython
    # in .venv, NOT cray-python which has a different ABI version).
    if is_cray_pe and "CPPFLAGS" not in os.environ:
        venv_py = os.path.join(os.getcwd(), ".venv", "bin", "python")
        if os.path.isfile(venv_py) and os.access(venv_py, os.X_OK):
            try:
                inc = subprocess.check_output(
                    [
                        venv_py,
                        "-c",
                        "import sysconfig; print(sysconfig.get_path('include'))",
                    ],
                    text=True,
                ).strip()
            except subprocess.CalledProcessError:
                inc = ""
            if inc and os.path.isfile(os.path.join(inc, "Python.h")):
                env["CPPFLAGS"] = f"-I{inc}"

    return env


def main():
    env = detect()
    if not env:
        print("# MPI/HDF5 environment already configured.", file=sys.stderr)
        return

    print(
        "# Auto-detected MPI/HDF5 build environment for miv-simulator", file=sys.stderr
    )
    for key, val in env.items():
        print(f"#   {key}={val}", file=sys.stderr)
    print(file=sys.stderr)

    # Output export commands for eval
    for key, val in env.items():
        print(f"export {key}={val!r}")


if __name__ == "__main__":
    main()
