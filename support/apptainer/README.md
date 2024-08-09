# Apptainer (singularity) container

- Frontera Base Container: https://containers-at-tacc.readthedocs.io/en/latest/singularity/03.mpi_and_gpus.html
- Vista Base Container: https://hub.docker.com/r/tacc/tacc-base 

## How to build

Download `.def` file in scratch, and run:

```bash
module load tacc-apptainer
apptainer build miv-simulator-container.sif <container name>.def
```

## MPI

On TACC system, make sure to use `ibrun`.

```bash
ibrun apptainer exec miv-simulator-container.sif python ...
```