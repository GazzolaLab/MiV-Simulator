# Installation: Docker Image

> Note: The docker image is not optimized for high-performance computing. The image file only provides necessary installation for users to quickly try-run small model.

We provide `Dockerfile` to quick-install docker image. The image contains necessary dependencies to run the simulation. We use `juypter-lab` as an recommended-interface.

## Building the image

> The docker image is not yet uploaded to docker-hub, so you need to build it locally

```bash
cd MiV-Simulator                         # Change directory to repository
docker build . --tag miv_env:0.1         # Build image
```

## Tutorial Cases

To get started quickly, we provide an optional [starter-repository](https://github.com/GazzolaLab/MiV-Simulator-Cases) that provides examples and tutorial cases.

```
git clone https://github.com/GazzolaLab/MiV-Simulator-Cases.git
```

## Running the image

By default, the image runs `jupyter-lab` on port `8888`. You may alter the destination port by setting `<destination port>:8888`.

```bash
# run the image with the examples and tutorial cases (recommended for beginners)
docker run -p 8888:8888 -v ./MiV-Simulator-Cases:/home/user/workspace -it miv_env:0.1

# just run the environment to run your own code (advanced)
docker run -p 8888:8888 -it miv_env:0.1
```

To develop the MiV-Simulator source code, you may mount the repository via `-v ./MiV-Simulator:/home/user` so that changes are persistet when your container stops running.
