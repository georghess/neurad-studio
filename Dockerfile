# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

LABEL org.opencontainers.image.licenses="Apache-2.0"

# Override to build only for the GPUs you will use (e.g. 80 for A100, 120 for RTX 5090).
ARG TCNN_CUDA_ARCHITECTURES="70;75;80;86;89;90;100;120"
ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    NEURAD_VENV=/opt/neurad-venv \
    PATH=/opt/neurad-venv/bin:/usr/local/cuda/bin:${PATH} \
    TORCH_EXTENSIONS_DIR=/opt/neurad-venv/torch_extensions \
    MAX_JOBS=${MAX_JOBS}
ENV CPATH=${NEURAD_VENV}/lib/python3.12/site-packages/nvidia/cublas/include:${NEURAD_VENV}/lib/python3.12/site-packages/nvidia/cusolver/include:${NEURAD_VENV}/lib/python3.12/site-packages/nvidia/cusparse/include
# Link against the driver stub during builds without GPU passthrough.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git ffmpeg libgl1 libegl1 libglib2.0-0 libgomp1 \
    cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 libcurand-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /workspace/neurad-studio
COPY pyproject.toml ./
COPY scripts/setup-env.sh scripts/requirements-*.txt scripts/prepare-container.py ./scripts/
# Keep dependency compilation cached across source edits. Separate uv caches by
# target architecture because tiny-cuda-nn wheels contain architecture-specific code.
RUN --mount=type=cache,id=neurad-uv-${TCNN_CUDA_ARCHITECTURES},target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/yarn \
    --mount=type=cache,target=/root/.npm \
    TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES}" \
    bash scripts/setup-env.sh --skip-gpu-check --dependencies-only \
    && python scripts/prepare-container.py

COPY . .
RUN uv --no-cache pip install --python /opt/neurad-venv/bin/python --no-build-isolation --no-deps -e . \
    && uv --no-cache pip check --python /opt/neurad-venv/bin/python \
    && ns-train --help > /dev/null

EXPOSE 7007
CMD ["bash"]
