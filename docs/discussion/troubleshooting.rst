*****************************
Troubleshooting Common Issues
*****************************

If you have any problem that is not covered in this article, please report it on the [GitHub issue](https://github.com/GazzolaLab/MiV-Simulator/issues).

Runtime issues during construction
==================================

```sh
HDF5-DIAG: Error detected in HDF5 (1.10.4) thread 140353498163008:
#000: H5F.c line 444 in H5Fcreate(): unable to create file
major: File accessibilty
minor: Unable to open file
...
```

- Check if directory exists, or if file already exists.
- Check if `mpi4py` and `NeuroH5` are installed with the **same** `MPICC`.

---
