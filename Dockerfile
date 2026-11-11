# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

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

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /workspace/neurad-studio
COPY . .
# Docker builds do not need GPU passthrough. gsplat compiles its CUDA kernels on first use.
RUN --mount=type=cache,target=/root/.cache/uv TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES}" \
    bash scripts/setup-env.sh --skip-gpu-check \
    && ns-train --help > /dev/null

EXPOSE 7007
CMD ["bash"]
