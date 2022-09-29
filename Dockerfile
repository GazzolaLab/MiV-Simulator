# Docker image file for running light simulation using MiV-Simulator
# Note: This docker image should not be used for cluster/HPC setting.

FROM ubuntu

# File Maintainer
MAINTAINER skim0119

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install base utilities
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3 python3-pip python3-dev python-is-python3 \
    wget curl \
    && apt-get install -y --no-install-recommends curl vim unzip git-all cmake libopenmpi-dev apt-utils \
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
    && cmake --version \
    && mpicc --version

# MiV-Simulator Dependencies
WORKDIR /opt/

# Parallel HDF5 (long)
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz \
    && tar -xzf hdf5-1.12.1.tar.gz \
    && cd hdf5-1.12.1 \
    && CC=mpicc ./configure --prefix=$PWD/build --enable-parallel \
    && make -j4 && make install \
    && rm -f hdf5-1.12.1.tar.gz
ENV HDF5_SOURCE /opt/hdf5-1.12.1
RUN echo "export PATH=$PATH:$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin" >> ~/.bashrc
ENV PATH $PATH:$HDF5_SOURCE/build:$HDF5_SOURCE/build/bin

# MPI4PY
RUN pip install --no-cache-dir mpi4py numpy
#RUN apt-get install -y --no-install-recommends python3-mpi4py

# NeuroH5
RUN git clone https://github.com/iraikov/neuroh5.git
RUN pip install --no-cache-dir ./neuroh5
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
    && pip install --no-cache-dir ./MiV-Simulator

# Other Utilities
RUN pip install --no-cache-dir jupyter jupyterlab jupytext miv-os

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

# Clean up
RUN pip cache purge

# Launch jupyter lab
WORKDIR /home/shared
CMD ["jupyter", "lab", "--app_dir=/home/shared", "--port=8888", "--allow-root", "--ip", "0.0.0.0"]
