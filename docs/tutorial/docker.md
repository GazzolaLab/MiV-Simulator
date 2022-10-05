# Installation: Docker Image

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

### Build Image

The command below builds `image` tagged `miv_env:01`, based on the instruction set provided in [Dockerfile][url-mivsim-dockerfile]. To see the available images, run `docker images`.

```bash
cd MiV-Simulator                         # Change directory to repository
docker build . --tag miv_env:0.1         # Build image
```

### Create Container From Image

The command below both _initialize_ a container `miv_env:0.1` and _launch_ that container. By default, the container will run the `jupyter-lab` in the background on the port `8888`. User can alther the destination port by setting `<destination port>:8888`.

```bash
docker run -p 8888:8888 -it miv_env:0.1
```

To see the running containers, run `docker ps` or `docker container ls`. Too show all containers including the one stopped, pass `-a` or `-all`.

### Re-start and Stop

Once the container is created, try to run `docker ps --all` to check which name is assigned to the container.
Later on, the container can be restarted and stopped using

```bash
docker start <container name>
docker stop <container name>
```

### Mount Local Directory

The docker container is isolated computation environment, which means the external folder/directory structure is natively inaccessible.
To mount the external volume, pass `-v` or `--volume` to bind extra volume driver for the container.
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

:::{note}
It is typically recommanded to save simulation results outside the docker-container to keep the container size small.
:::

## Tutorial Cases

We provide a [starter-repository][url-repo-cases] that includes notebooks in [tutorials][url-tutorial].


<!-- Links -->

[url-tutorial]: https://miv-simulator.readthedocs.io/en/latest/tutorial/index.html
[url-docker-docs]: https://docs.docker.com/get-started://docs.docker.com/get-started/

[url-repo-cases]: https://github.com/GazzolaLab/MiV-Simulator-Cases
[url-mivsim-dockerfile]: https://github.com/GazzolaLab/MiV-Simulator/blob/main/Dockerfile
