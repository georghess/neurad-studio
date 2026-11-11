#!/usr/bin/env bash
# Set up a Linux CUDA 12.8 development environment without Conda.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: bash scripts/setup-env.sh [--skip-gpu-check]

Requires uv, git, gcc/g++, and a CUDA 12.8 toolkit. Set CUDA_HOME if the toolkit
is not in /usr/local/cuda. A toolkit at .venv/cuda is detected automatically.
Creates .venv with managed Python 3.12, installs development dependencies and
tiny-cuda-nn, and writes .venv/activate-neurad for subsequent shells.
Set NEURAD_VENV to create a separate environment when upgrading Python.
For container builds without a GPU, pass --skip-gpu-check and set
TCNN_CUDA_ARCHITECTURES to the target compute capabilities (e.g. "86;89;120").
HELP
    exit 0
fi
skip_gpu_check=false
if [[ "${1:-}" == "--skip-gpu-check" ]]; then
    skip_gpu_check=true
    shift
    : "${TCNN_CUDA_ARCHITECTURES:?Set TCNN_CUDA_ARCHITECTURES for a build without GPU access}"
fi
if [[ $# -ne 0 ]]; then
    echo "Unknown argument. Run with --help for usage." >&2
    exit 2
fi

for command in uv git gcc g++; do
    command -v "$command" >/dev/null || { echo "Missing prerequisite: $command" >&2; exit 1; }
done

if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -x "$repo_dir/.venv/cuda/bin/nvcc" ]]; then
        export CUDA_HOME="$repo_dir/.venv/cuda"
    else
        export CUDA_HOME=/usr/local/cuda
    fi
fi
if ! "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.8,'; then
    echo "This profile requires CUDA 12.8. Set CUDA_HOME to its toolkit directory." >&2
    exit 1
fi

# Use Python with headers so building extensions does not require python-dev.
venv_dir="${NEURAD_VENV:-$repo_dir/.venv}"
mkdir -p "$venv_dir"
venv_dir="$(cd "$venv_dir" && pwd)"
export UV_PYTHON_INSTALL_DIR="$venv_dir/python"
if [[ ! -x "$venv_dir/bin/python" ]]; then
    uv python install 3.12 --no-bin
    uv venv --allow-existing --managed-python --python 3.12 "$venv_dir"
fi
"$venv_dir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12; set NEURAD_VENV to create a separate environment"'

export PATH="$CUDA_HOME/bin:$venv_dir/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_EXTENSIONS_DIR="$venv_dir/torch_extensions"

# Do not inherit compiler paths or flags from an unrelated Conda/Pixi shell.
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CC="$(command -v gcc)" CXX="$(command -v g++)"

uv pip install --python "$venv_dir/bin/python" 'setuptools<70' wheel ninja pip
uv pip install --python "$venv_dir/bin/python" \
    'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
    --index-url https://download.pytorch.org/whl/cu128

# gsplat compiles its kernels on first use, after torch has been installed.
BUILD_NO_CUDA=1 uv pip install --python "$venv_dir/bin/python" --no-build-isolation \
    --overrides scripts/requirements-cu128.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match -e '.[dev]'
uv pip install --python "$venv_dir/bin/python" --no-build-isolation \
    'git+https://github.com/NVlabs/tiny-cuda-nn.git@3daa6e5b8b8e0fa17971a1a20acfbb3754494b38#subdirectory=bindings/torch'

# Keep CUDA visible to gsplat's lazy compiler in subsequent shells, too.
{
    printf 'source %q\n' "$venv_dir/bin/activate"
    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    printf 'export PATH=%q:"$PATH"\n' "$CUDA_HOME/bin"
    printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$TORCH_EXTENSIONS_DIR"
    printf 'export MAX_JOBS=%q\n' "$MAX_JOBS"
    printf 'export CC=%q CXX=%q\n' "$CC" "$CXX"
    printf 'unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS\n'
} > "$venv_dir/activate-neurad"

uv pip check --python "$venv_dir/bin/python"
"$venv_dir/bin/python" -c 'from gsplat import map_points_to_lidar_tiles, points_mapping_offset_encode, populate_image_from_points; from gsplat.rendering import lidar_rasterization'
if [[ "$skip_gpu_check" == false ]]; then
    "$venv_dir/bin/python" -c 'import torch, tinycudann; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
fi
echo "Ready. Activate with: source $venv_dir/activate-neurad"
