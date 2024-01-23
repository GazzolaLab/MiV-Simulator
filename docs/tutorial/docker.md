# Docker Image

:::{note}
The docker image is not optimized for high-performance computing. The image file only provides necessary installation for users to quickly try-run a simple model.
:::

We provide `Dockerfile` to quick-install docker image. The image is based on Ubuntu, and it contains necessary dependencies and libraries to run the simulation. We use `juypter-lab` as a recommended interface.

> _Docker image is not yet uploaded to docker-hub._

## How To Use

Below, we provide basic set of docker instructions on how to build container and open the `jupyter-lab` environment.

:::{note}
The detail instructions and standard practice can be found in the [`docker` documentation][url-docker-docs].
:::

```{mermaid}
flowchart LR
    DF[Dockerfile]
    CI[(Image)]
    C1(Container)
    C2(Container)
    C3(Container)
    J{Jupyter-Lab}
    U((User))

    DF -->|build| CI --> C1
    CI --> C2
    CI --> C3
    C1 --> J
    U --> J
    J --> U
```
- Basic Terminology
  - Dockerfile: Instruction set to create image.
  - Image: _Template_ by which a container will be initialized at runtime.  To create image, use the command `docker build`.
  - Container: Isolated computing environment. Each container can be created from image by the command `docker run`.


### Quick start

The quickest way to get started is to use the preconfigured setup via docker-compose:

```bash
# If you haven't already, clone the MiV-Simulator source into a local directory
git clone https://github.com/GazzolaLab/MiV-Simulator  
cd MiV-Simulator
# Then launch docker-compose
docker-compose up
```

This will build and launch the `jupyter-lab` environment.

### Build Image

The command below builds `image` tagged `miv_env:01`, based on the instruction set provided in [Dockerfile][url-mivsim-dockerfile]. To see the available images, run `docker images`.

```bash
cd MiV-Simulator                         # Change directory to repository
docker build . --tag miv_env:0.1         # Build image
```

### Create Container From Image

The command below both _initialize_ a container `miv_env:0.1` and _launch_ that container.

```bash
# -p: <local port>:<container port>
docker run -p 18888:8888 -it miv_env:0.1  # access with localhost:18888
```

:::{note}
To see the running containers, run `docker ps` or `docker container ls`. Too show all containers including the one stopped, pass `-a` or `-all`.
:::

### Re-start and Stop

Once the container is created, try to run `docker ps --all` to check which name is assigned to the container.
Later on, the container can be restarted and stopped using

```bash
docker start <container name>
docker stop <container name>
```

## Advanced Capability

### Mount Local Directory

The docker container is isolated computation environment, which means the external folder/directory structure is natively inaccessible.
To mount the external volume, pass `-v` or `--volume` to bind extra volume driver inside the container.
The string to pass is in the form `<local directory>:<container directory>`.
For examples, the command below will mount a local workspace directory `$(pwd)/workspace` to container directory `/home/user/workspace`.

```bash
docker run -p 8888:8888 -v "$(pwd)/workspace:/home/user/workspace" -it miv_env:0.1
```

The argument `-v, --volume` can also take multiple volumes if one wants to mount multiple volumes.

```bash
docker run -p 8888:8888 \
    -v "$(pwd)/workspace-A:/home/user/workspace-A" \
    -v "$(pwd)/workspace-B:/home/user/workspace-B" \
    -v "$(pwd)/workspace-C:/home/user/workspace-C" \
    -it miv_env:0.1
```

Notice, mounting the volume is only possible when the container is initialized.

To organize and manage the local volumes for multiple docker-containers, one can assign `Docker Volumes`: [here][url-docker-docs-volume]

:::{note}
It is typically recommanded to save simulation results outside the docker-container to keep the container size small.
:::

### Transfer Files from/to Container

To copy files between container and local directory, one can use `docker cp` command: [doc][url-docker-docs-copy].

### Developing the MiV-Simulator

Note that by default, the source code of the simulator is part of the image and changes are lost as soon as the container terminates. To develop the MiV-Simulator package source code, you have to update the `miv_simulator` package installation inside the container (e.g. via Jupyter lab) to use the mounted repo that is persistent outside the container:

```python
# from within the container
import sys
!{sys.executable} -m pip install --no-cache-dir --no-deps -e /home/user/MiV-Simulator  # location of repo mount point
```

## Tutorial Cases

We provide a [starter-repository][url-repo-cases] that includes notebooks in [tutorials][url-tutorial].


<!-- Links -->

[url-tutorial]: https://miv-simulator.readthedocs.io/en/latest/tutorial/index.html
[url-docker-docs]: https://docs.docker.com/get-started://docs.docker.com/get-started/
[url-docker-docs-volume]: https://docs.docker.com/storage/volumes/#create-and-manage-volumes
[url-docker-docs-copy]: https://docs.docker.com/engine/reference/commandline/cp/

[url-repo-cases]: https://github.com/GazzolaLab/MiV-Simulator-Cases
[url-mivsim-dockerfile]: https://github.com/GazzolaLab/MiV-Simulator/blob/main/Dockerfile
