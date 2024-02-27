# Docker image file for running light simulation using MiV-Simulator
# Note: This docker image should not be used for cluster/HPC setting.

FROM ubuntu

SHELL ["/bin/bash", "-c"]
ENV SHELL /bin/bash

# File Maintainer
LABEL org.opencontainers.image.authors="skim0119@gmail.com"

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

# Configuration
ENV TZ=Etc/UTC
ENV OPENMPI_MAJOR_VER 4
ENV OPENMPI_MINOR_VER 1
ENV OPENMPI_PATCH_VER 4
ENV OPENMPI_VER=${OPENMPI_MAJOR_VER}.${OPENMPI_MINOR_VER}.${OPENMPI_PATCH_VER}
ENV HDF5_MAJOR_VER 1
ENV HDF5_MINOR_VER 12
ENV HDF5_PATCH_VER 1
ENV HDF5_VER=${HDF5_MAJOR_VER}.${HDF5_MINOR_VER}.${HDF5_PATCH_VER}
ENV NEURON_VER 8.2.1


# Install base utilities   #libopenmpi-dev
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3 python3-pip python3-dev python-is-python3 \
    && apt-get install -y --no-install-recommends ssh wget curl vim unzip git-all cmake apt-utils sudo \
    && apt-get install -y --no-install-recommends libgl1-mesa-glx \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
#&& apt-get update
#&& apt-get install -y --no-install-recommends software-properties-common \
#    && add-apt-repository ppa:deadsnakes/ppa \

# Install Python
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && echo "export PATH="/root/.local/bin:$PATH"" >> ~/.bashrc
ENV PATH /root/.local/bin:$PATH

# Check require builders
RUN gcc --version \
    && python3 --version \
    && pip3 --version \
    && poetry --version \
    && cmake --version # \
#&& mpicc --version \
#&& mpicxx --version \

# MiV-Simulator Dependencies
RUN mkdir -p /opt
WORKDIR /opt

# Install Openmpi
RUN wget https://download.open-mpi.org/release/open-mpi/v${OPENMPI_MAJOR_VER}.${OPENMPI_MINOR_VER}/openmpi-${OPENMPI_VER}.tar.bz2 \
    && tar xf openmpi-${OPENMPI_VER}.tar.bz2 \
    && cd openmpi-${OPENMPI_VER} \
    && ./configure --prefix=$PWD/build 2>&1 | tee openmpi_config.out \
    && make all 2>&1 | tee openmpi_make.out \
    && make install 2>&1 | tee openmpi_install.out
ENV MPICC_SOURCE /opt/openmpi-${OPENMPI_VER}
RUN echo "export PATH=\$PATH:$MPICC_SOURCE/build/bin" >> ~/.bashrc
ENV PATH $PATH:$MPICC_SOURCE/build/bin

# Parallel HDF5 (long)
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_MAJOR_VER}.${HDF5_MINOR_VER}/hdf5-${HDF5_VER}/src/hdf5-${HDF5_VER}.tar.gz \
    && tar -xzf hdf5-${HDF5_VER}.tar.gz \
    && cd hdf5-${HDF5_VER} \
    && CC=mpicc ./configure --prefix=$PWD/build --enable-parallel --enable-shared \
    && make && make install
ENV HDF5_SOURCE /opt/hdf5-${HDF5_VER}
RUN echo "export PATH=\$PATH:$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin" >> ~/.bashrc
ENV PATH $PATH:$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin

# MPI4PY
RUN env MPICC="mpicc --shared" pip install --no-cache-dir mpi4py
RUN pip install --no-cache-dir numpy

# h5py
RUN CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/hdf5-${HDF5_VER}/build pip install --no-binary=h5py --no-deps h5py==3.7.0

# NeuroH5
RUN git clone https://github.com/iraikov/neuroh5.git
RUN CC=mpicc CXX=mpicxx pip install --no-deps ./neuroh5
RUN cd neuroh5 \
    && CC=mpicc CXX=mpicxx cmake . && make -j4
RUN echo "export PATH=/opt/neuroh5/bin:\$PATH" >> ~/.bashrc
ENV PATH /opt/neuroh5/bin:$PATH
RUN which h5copy \
    && which neurotrees_import

# Neuron (nrn)
RUN pip install --no-cache-dir neuron==${NEURON_VER}

# MiV Simulator
RUN git clone https://github.com/GazzolaLab/MiV-Simulator \
    && pip install --no-cache-dir ./MiV-Simulator

# MiV Packages
#RUN useradd --create-home --shell /bin/bash --no-log-init --gid root -G sudo user
#USER user
RUN mkdir -p /home/user
WORKDIR /home/user

# Other Utilities
RUN pip install --no-cache-dir jupyter jupyterlab jupytext miv-os

# Prepare example cases
RUN git clone https://github.com/GazzolaLab/MiV-Simulator-Cases Tutorial

# Clean up
RUN pip cache purge

# Allow run as root
## TODO: Consider adding user
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT='1'" >> ~/.bashrc
ENV OMPI_ALLOW_RUN_AS_ROOT 1
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1'" >> ~/.bashrc
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1

# Launch jupyter lab
CMD ["jupyter", "lab", "--app_dir=/home/user", \
        "--port=8888", "--allow-root", "--ip", "0.0.0.0", \
        "--NotebookApp.token=''", "--NotebookApp.password=''"]

# HDF5 test:
# NPROCS=4 make check-p
