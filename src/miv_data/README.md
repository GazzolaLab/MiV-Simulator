# MiV Data

> Incubator stage -- not stable! This should eventually become its own package that will be required back into miv_simulator

Common data I/O for the Mind-In-Vitro project based on `HDF5` and `neuroh5` in particular.

## Installation

To install `miv_data` without installing `MiV-Simulator` use the following:
```sh
pip install -e "git+https://github.com/GazzolaLab/MiV-Simulator.git#egg=miv_data&subdirectory=src"
```

## Rationale

High-level interface to read and write
- `cells.h5` and `connections.h5` format used by the MiV-Simulator
- simulation inputs and results like spike trains etc.
- experimental recordings (?)

## Non-goals

No data processing or plotting algorithms that go beyond what is necessary to read and write from H5 source files. The aim is to be a thin I/O library, nothing more; anything more sophisticated should go into `MiV-OS`.
