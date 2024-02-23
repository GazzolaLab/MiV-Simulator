# MiV Data

> Incubator stage -- not stable! This should eventually become its own package that will be required back into miv_simulator

Common data I/O for the Mind-In-Vitro project based on `HDF5` and `neuroh5` in particular.

Rationale:

High-level interface to read and write
- `cells.h5` and `connections.h5` format used by the MiV-Simulator
- simulation inputs and results like spike trains etc.
- experimental recordings (?)

Non-goals:

No data processing or plotting algorithms that go beyond what is necessary to read and write from H5 source files. The aim is to be file I/O libary, nothing more; anything more sophisticated should go into `MiV-OS`.