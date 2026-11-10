#!/usr/bin/env bash
# Set up a Linux CUDA 12.8 development environment without Conda.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: bash scripts/setup-env.sh

Requires uv, git, gcc/g++, and a CUDA 12.8 toolkit. Set CUDA_HOME if the toolkit
is not in /usr/local/cuda. A toolkit at .venv/cuda is detected automatically.
Creates .venv with managed Python 3.10, installs development dependencies and
tiny-cuda-nn, and writes .venv/activate-neurad for subsequent shells.
HELP
    exit 0
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
export UV_PYTHON_INSTALL_DIR="$repo_dir/.venv/python"
if [[ ! -x .venv/bin/python ]]; then
    uv python install 3.10 --no-bin
    uv venv --allow-existing --managed-python --python 3.10 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use a Python 3.10 virtual environment"'

export PATH="$CUDA_HOME/bin:$repo_dir/.venv/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_EXTENSIONS_DIR="$repo_dir/.venv/torch_extensions"

# Do not inherit compiler paths or flags from an unrelated Conda/Pixi shell.
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CC="$(command -v gcc)" CXX="$(command -v g++)"

uv pip install --python .venv/bin/python 'setuptools<70' wheel ninja pip
uv pip install --python .venv/bin/python \
    'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
    --index-url https://download.pytorch.org/whl/cu128

# gsplat compiles its kernels on first use, after torch has been installed.
BUILD_NO_CUDA=1 uv pip install --python .venv/bin/python --no-build-isolation \
    --overrides scripts/requirements-cu128.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match -e '.[dev]'
uv pip install --python .venv/bin/python --no-build-isolation \
    'git+https://github.com/NVlabs/tiny-cuda-nn.git@3daa6e5b8b8e0fa17971a1a20acfbb3754494b38#subdirectory=bindings/torch'

# Keep CUDA visible to gsplat's lazy compiler in subsequent shells, too.
{
    printf 'source %q\n' "$repo_dir/.venv/bin/activate"
    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    printf 'export PATH=%q:"$PATH"\n' "$CUDA_HOME/bin"
    printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$TORCH_EXTENSIONS_DIR"
    printf 'export MAX_JOBS=%q\n' "$MAX_JOBS"
    printf 'export CC=%q CXX=%q\n' "$CC" "$CXX"
    printf 'unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS\n'
} > .venv/activate-neurad

.venv/bin/python -c 'import torch, tinycudann; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
echo "Ready. Activate with: source .venv/activate-neurad"
