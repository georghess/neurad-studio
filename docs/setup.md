# Local development with uv and CUDA 12.8

The CUDA 12.8 profile uses Python 3.10, PyTorch 2.7.1, and torchvision 0.22.1.
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
The script can be rerun against an existing Python 3.10 environment.

## Compatibility choices

`scripts/requirements-cu128.txt` overrides the legacy development extra's
PyTorch upper bound for this profile. It also preserves the APIs used by the
legacy viewer (`dataclass-wizard`) and optional generative code (`huggingface-hub`).
These are compatibility constraints, not a lockfile for every transitive package.
The setup pins tiny-cuda-nn, gsplat, Viser, and PandaSet to specific revisions.

Waymo's TensorFlow packages and py123d are separate, optional dataset installs.
In particular, py123d 0.6.0 pins upstream `viser==1.0.24`, while this repository
uses a custom older Viser fork. Installing py123d normally into this environment
cannot satisfy both requirements. Do not replace the repository's Viser fork to
silence that conflict; use a separate environment for py123d conversion tools.

## Validate

```bash
python -c 'import torch, tinycudann; print(torch.__version__, torch.cuda.get_device_name())'
pytest tests/pipelines/test_vanilla_pipeline.py
```

Run dataset-specific tests after installing their optional dependencies. Training
requires a prepared dataset; environment smoke tests do not validate a training run.
