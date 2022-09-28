# Docker Image

> Note: The docker image is not optimized for high-performance computing. The image file only provides necessary installation for users to quickly try-run small model.

We provide `Dockerfile` to quick-install docker image. The image contains necessary dependencies to run the simulation. We use `juypter-lab` as an recommended-interface.

```bash
cd MiV-Simulator                         # Change directory to repository
docker build . --tag miv_env:0.1         # Build image
docker run -p 8888:8888 -it miv_env:0.1  # Launch jupyter-lab
```

> Docker image is not yet uploaded to docker-hub.
