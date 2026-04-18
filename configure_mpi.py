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
                if r2.returncode == 0 and "mpi" in r2.stdout.lower():
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
        if not hdf5_dir:
            prefix, incdir, libdir = _find_hdf5_via_wrapper()
            if prefix:
                env["HDF5_DIR"] = prefix
            if incdir and not os.environ.get("HDF5_INCLUDEDIR"):
                env["HDF5_INCLUDEDIR"] = incdir
            if libdir and not os.environ.get("HDF5_LIBDIR"):
                env["HDF5_LIBDIR"] = libdir

    if not os.environ.get("HDF5_MPI"):
        env["HDF5_MPI"] = "ON"

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
