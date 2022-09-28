# Docker image file for running light simulation using MiV-Simulator
# Note: This docker image should not be used for cluster/HPC setting.

FROM ubuntu

# File Maintainer
MAINTAINER skim0119

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update the repository source list
RUN apt-get update;

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends apt-utils

# Install Git
RUN apt-get install -y --no-install-recommends git-all

# Install C/C++ Modules
RUN apt-get install -y --no-install-recommends cmake libopenmpi-dev

# Install Python
RUN apt-get install --no-install-recommends -y \
    python3 python3-pip python3-dev
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN echo "export PATH="/root/.local/bin:$PATH"" >> ~/.bashrc
ENV PATH /root/.local/bin:$PATH

# Check require builders
RUN gcc --version \
    && python3 --version \
    && pip3 --version \
    && poetry --version \
    && cmake --version \
    && mpicc --version

# MiV-Simulator Dependencies
WORKDIR /opt/

# Parallel HDF5 (long)
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz && \
    tar -xzf hdf5-1.12.1.tar.gz && \
    cd hdf5-1.12.1 && \
    CC=mpicc ./configure --prefix=$PWD/build --enable-parallel && \
    make -j4 && make install
ENV HDF5_SOURCE /opt/hdf5-1.12.1
RUN echo "export PATH=$PATH:$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin" >> ~/.bashrc
ENV PATH $PATH:$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin
RUN rm -f hdf5-1.12.1.tar.gz

# MPI4PY
RUN pip install --no-cache-dir mpi4py
#RUN apt-get install -y --no-install-recommends python3-mpi4py

# NeuroH5
RUN git clone https://github.com/iraikov/neuroh5.git
RUN pip install numpy
RUN pip install ./neuroh5
RUN cd neuroh5 \
    && cmake . && make -j4
RUN echo "export PATH=/opt/neuroh5/bin:$PATH" >> ~/.bashrc
ENV PATH /opt/neuroh5/bin:$PATH
RUN which h5copy \
    && which neurotrees_import

# Neuron (nrn)
RUN pip install --no-cache-dir neuron==8.2.1

# MiV Packages
RUN mkdir -p /home/shared
WORKDIR /home/shared

RUN git clone https://github.com/GazzolaLab/MiV-Simulator \
    && pip install -e ./MiV-Simulator

# Other Utilities
RUN apt-get install -y --no-install-recommends vim unzip
RUN apt-get install -y --no-install-recommends python-is-python3
RUN pip install jupyter jupyterlab jupytext
RUN pip install miv-os

# Prepare example cases
## TODO: Later, replace to example repository.
ARG DIRNAME=sample_case_1
RUN mkdir -p $DIRNAME
WORKDIR /home/shared/$DIRNAME
RUN wget https://uofi.box.com/shared/static/a88dy7muglte90hklryw0xskv7ne13j0.zip -O files.zip \
    && unzip files.zip \
    && rm files.zip
COPY docs/tutorial/constructing_a_network_model.md .
RUN jupytext constructing_a_network_model.md --to ipynb \
    && rm constructing_a_network_model.md

# Launch jupyter lab
WORKDIR /home/shared
CMD ["jupyter", "lab", "--app_dir=/home/shared", "--port=8888", "--allow-root", "--ip", "0.0.0.0"]
