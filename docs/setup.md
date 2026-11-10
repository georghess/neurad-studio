# Local development with uv and CUDA 12.8

The CUDA 12.8 profile uses Python 3.12, PyTorch 2.7.1, and torchvision 0.22.1.
It supports RTX 50-series GPUs, which cannot use the original CUDA 11.8 setup.
The original Conda/container instructions remain available in the README.

## Install

Install `uv`, Git, GCC/G++, and the CUDA 12.8 development toolkit. The toolkit
must include `nvcc`; the CUDA runtime included in PyTorch is not sufficient to
build tiny-cuda-nn or SplatAD's custom gsplat kernels. A working NVIDIA driver
and a GPU visible to the shell are also required.

From the repository root:

```bash
CUDA_HOME=/usr/local/cuda-12.8 bash scripts/setup-env.sh
source .venv/activate-neurad
ns-train --help
```

Set `CUDA_HOME` to your toolkit's actual location. The script also detects a
toolkit installed under `.venv/cuda`. It creates a managed Python installation
with development headers and a virtual environment under `.venv`, installs the
repository in editable mode with development tools, and builds tiny-cuda-nn.
SplatAD's gsplat kernels compile on first use. Build parallelism defaults to four
jobs; set `MAX_JOBS` before setup to change it.

Use `source .venv/activate-neurad` in each new shell. This activates Python and
sets the CUDA/compiler paths needed for compilation. It clears inherited
Conda/Pixi compiler flags so another project's environment cannot affect builds.
The script can be rerun against an existing Python 3.12 environment.

To upgrade an existing Python 3.10 setup, create a separate environment:

```bash
NEURAD_VENV=.venv/py312-gpu CUDA_HOME=/usr/local/cuda-12.8 bash scripts/setup-env.sh
source .venv/py312-gpu/activate-neurad
```

Set `CUDA_HOME` to the existing toolkit path when reusing it. The script checks
the interpreter version before installing packages into an existing environment.

## Compatibility choices

`scripts/requirements-cu128.txt` selects the CUDA build of PyTorch for this
profile. Both CPU and CUDA profiles include
`scripts/requirements-common.txt`, which preserves the APIs used by the legacy
viewer (`dataclass-wizard`) and optional generative code (`huggingface-hub`).
These are compatibility constraints, not a lockfile for every transitive package.
The setup pins tiny-cuda-nn, gsplat, Viser, and PandaSet to specific revisions.

Training, evaluation, and rendering explicitly load full checkpoints with
`weights_only=False`. This preserves compatibility with optimizer/scheduler
state containing NumPy scalars under PyTorch 2.6 and later. Only load checkpoints
from trusted sources: this format uses Python pickle deserialization.

Waymo's TensorFlow packages and py123d are separate, optional dataset installs.
In particular, py123d 0.6.0 pins upstream `viser==1.0.24`, while this repository
uses a custom older Viser fork. Installing py123d normally into this environment
cannot satisfy both requirements. Do not replace the repository's Viser fork to
silence that conflict; use a separate environment for py123d conversion tools.

## Validate

```bash
python -c 'import torch, tinycudann; print(torch.__version__, torch.cuda.get_device_name())'
pytest tests/pipelines/test_vanilla_pipeline.py
pytest tests/engine/test_checkpoint_serialization.py
```

Run dataset-specific tests after installing their optional dependencies. Training
requires a prepared dataset; environment smoke tests do not validate a training run.

## CPU checks and GitHub Actions

The core workflow uses Ubuntu 24.04, managed Python 3.12, and CPU builds of the
same PyTorch/torchvision versions as the CUDA profile. It checks dependency
compatibility, notebook metadata, Ruff, and the test suite. Optional py123d tests
skip when that dataset package is absent; CUDA kernels and training need separate
GPU validation. Waymo's TensorFlow dependency is not needed for these core checks.
Runtime type checks retain Typeguard 2 for compatibility with jaxtyping's tensor
annotations. A test-only adapter uses `typing_extensions.evaluate_forward_ref`
because Python 3.12 changed the private API used by that older checker.

To reproduce the workflow in a separate environment from the repository root:

```bash
uv venv --managed-python --python 3.12 .venv/ci-env
source .venv/ci-env/bin/activate
uv pip install 'setuptools<70' wheel ninja
uv pip install 'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' \
    --index-url https://download.pytorch.org/whl/cpu
BUILD_NO_CUDA=1 uv pip install --no-build-isolation \
    --overrides scripts/requirements-cpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match -e '.[dev]'
uv pip check
python nerfstudio/scripts/docs/add_nb_tags.py --check
ruff check docs/ nerfstudio/ tests/
ruff format docs/ nerfstudio/ tests/ --check
pytest
```
