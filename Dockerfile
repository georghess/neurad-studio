ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
# Define base image.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION
ARG USER_ID

# metainformation
LABEL org.opencontainers.image.licenses = "Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

# Variables used at build time.
## CUDA architectures, required by Colmap and tiny-cuda-nn.
## NOTE: Most commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
ARG CUDA_ARCHITECTURES=90;89;86;80;75

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    wget \
    ffmpeg \
    git \
    vim-tiny \
    libatlas-base-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libmetis-dev \
    libqt5opengl5-dev \
    libsuitesparse-dev \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install packages.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip "setuptools<70.0" pathtools promise pybind11
SHELL ["/bin/bash", "-c"]
RUN python3.10 -m pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} python3.10 -m pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# Install waymo-open-dataset
RUN python3.10 -m pip install --no-cache-dir waymo-open-dataset-tf-2-11-0==1.6.1

# Install tzdata
RUN python3.10 -m pip install --no-cache-dir tzdata

# Change working directory
WORKDIR /workspace

RUN git clone https://github.com/georghess/neurad-studio.git
WORKDIR /workspace/neurad-studio
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \ 
    python3.10 -m pip install -e .[dev]
WORKDIR /workspace

RUN git clone --recurse-submodules https://github.com/carlinds/splatad.git
WORKDIR /workspace/splatad
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    BUILD_NO_CUDA=1 python3.10 -m pip install -e .[dev]

# Make sure viser client is built
RUN python -c "import viser; viser.ViserServer()"

# Install nerfstudio cli auto completion
RUN ns-install-cli --mode install

# Bash as default entrypoint.
CMD /bin/bash -l
