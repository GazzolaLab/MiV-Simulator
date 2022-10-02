# Docker image file for running light simulation using MiV-Simulator
# Note: This docker image should not be used for cluster/HPC setting.

FROM ubuntu

SHELL ["/bin/bash", "-c"]
ENV SHELL /bin/bash

# File Maintainer
MAINTAINER skim0119

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install base utilities   #libopenmpi-dev
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3 python3-pip python3-dev python-is-python3 \
    && apt-get install -y --no-install-recommends ssh wget curl vim unzip git-all cmake apt-utils sudo \
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
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.bz2 \
    && tar xf openmpi-4.1.4.tar.bz2 \
    && cd openmpi-4.1.4 \
    && ./configure --prefix=$PWD/build 2>&1 | tee openmpi_config.out \
    && make all 2>&1 | tee openmpi_make.out \
    && make install 2>&1 | tee openmpi_install.out
ENV MPICC_SOURCE /opt/openmpi-4.1.4
RUN echo "export PATH=\$PATH:$MPICC_SOURCE/build/bin" >> ~/.bashrc
ENV PATH $PATH:$MPICC_SOURCE/build/bin

# Parallel HDF5 (long)
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz \
    && tar -xzf hdf5-1.12.1.tar.gz \
    && cd hdf5-1.12.1 \
    && CC=mpicc ./configure --prefix=$PWD/build --enable-parallel --enable-shared \
    && make && make install
ENV HDF5_SOURCE /opt/hdf5-1.12.1
RUN echo "export PATH=\$PATH:$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin" >> ~/.bashrc
ENV PATH $PATH:$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin

# MPI4PY
RUN env MPICC="mpicc --shared" pip install --no-cache-dir mpi4py
RUN pip install --no-cache-dir numpy

# h5py
RUN CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/opt/hdf5-1.12.1/build pip install --no-binary=h5py --no-deps h5py

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
RUN pip install --no-cache-dir neuron==8.2.1

# MiV Packages
#RUN useradd --create-home --shell /bin/bash --no-log-init --gid root -G sudo user
#USER user
RUN mkdir -p /home/user
WORKDIR /home/user

RUN git clone https://github.com/GazzolaLab/MiV-Simulator \
    && pip install --no-cache-dir ./MiV-Simulator

# Other Utilities
RUN pip install --no-cache-dir jupyter jupyterlab jupytext miv-os

# Prepare example cases
RUN git clone https://github.com/GazzolaLab/MiV-Simulator-Cases
WORKDIR /home/user/MiV-Simulator-Cases
RUN rm -rf .git  # Remove git connection

# Clean up
RUN pip cache purge

# Allow run as root
## TODO: Consider adding user
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT='1'" >> ~/.bashrc
ENV OMPI_ALLOW_RUN_AS_ROOT 1
RUN echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1'" >> ~/.bashrc
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM 1

# Launch jupyter lab
WORKDIR /home/user/MiV-Simulator-Cases


CMD ["jupyter", "lab", "--app_dir=/home/user/MiV-Simulator-Cases", "--port=8888", "--allow-root", "--ip", "0.0.0.0"]

# HDF5 test:
# NPROCS=4 make check-p
