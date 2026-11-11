# Local development with uv and CUDA 12.8

The CUDA 12.8 profile uses Python 3.12, PyTorch 2.7.1, and torchvision 0.22.1.
It supports RTX 50-series GPUs, which cannot use the original CUDA 11.8 setup.
The [archived Python 3.10 / CUDA 11.8 instructions](https://github.com/georghess/neurad-studio/blob/550e6fb40e2baafc11448750b3e2574618c3572a/README.md#1-installation-setup-the-environment)
are retained for existing legacy environments; use this profile for new installs.

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

## Docker

The Dockerfile uses the same setup script and dependency constraints as the local
CUDA 12.8 environment. It installs the CUDA compiler and headers needed by the
custom kernels; PyTorch supplies its own CUDA math libraries and headers. The
container environment exposes those headers through `CPATH` for extension builds.
It builds the source in your checkout. The host needs an
NVIDIA driver supporting CUDA 12.8 and NVIDIA Container Toolkit for GPU access;
a host CUDA toolkit or Python installation is not required.

```bash
docker build -t neurad-studio .
mkdir -p outputs
docker run --rm -it --gpus all --shm-size=8g --ulimit nofile=65536:65536 \
    -p 127.0.0.1:7007:7007 \
    -v /absolute/path/to/pandaset:/data/pandaset:ro \
    -v "$PWD/outputs:/workspace/neurad-studio/outputs" \
    neurad-studio
# Inside the container:
ns-train neurad pandaset-data --data /data/pandaset
```

The default build includes tiny-cuda-nn for several GPU generations. To reduce
build time, select only your GPU's compute capability (70 for V100, 80 for A100,
86 for RTX 3090, 89 for RTX 4090, or 120 for RTX 5090):

```bash
docker build --build-arg TCNN_CUDA_ARCHITECTURES=120 -t neurad-studio .
```

The image can be built without a GPU. Its setup skips the runtime GPU check but
still checks dependency compatibility, custom gsplat imports, and CLI startup.
At runtime, verify GPU access with:

```bash
docker run --rm --gpus all neurad-studio \
    python -c 'import torch, tinycudann; print(torch.cuda.get_device_name())'
```

Docker caches dependency installation and viewer compilation separately from the
source code. Editing Python files or documentation only rebuilds the final
checkout-installation layer. Changes to dependency files, setup scripts, or GPU
architectures rebuild the dependency layer. Build caches are kept on the build
host, and viewer Node tooling is removed from the finished image.

Python and the virtual environment live under `/opt/neurad-venv`; gsplat compiles
its CUDA kernels there on first use. Waymo's legacy TensorFlow packages are not
included: their Python 3.10 constraints require a separate environment. This
image replaces the old CUDA 11.8 Docker recipe.

## Apptainer

The [Apptainer recipe](../apptainer_recipe) uses the same Python 3.12 / CUDA 12.8
setup script and dependency constraints as Docker. With Apptainer 1.3 or newer,
build from the repository root so the recipe can copy your checked-out source:

```bash
apptainer build --fakeroot neurad-studio.sif apptainer_recipe
# Optional: target only your GPU and reduce build parallelism.
apptainer build --fakeroot --build-arg TCNN_CUDA_ARCHITECTURES=120 \
    --build-arg MAX_JOBS=2 neurad-studio-rtx5090.sif apptainer_recipe
```

The build needs internet access but no GPU. It prebuilds the viewer assets and
checks dependencies, custom gsplat imports, and CLI startup. If your cluster
does not support fakeroot builds, ask its administrator to build the image or
use a build host with that support. See the
[Apptainer build documentation](https://apptainer.org/docs/user/1.3/build_a_container.html).

Use `--nv` for GPU access and `--cleanenv` to avoid inheriting another Python or
Conda environment. The host needs an NVIDIA driver supporting CUDA 12.8.

```bash
apptainer exec --cleanenv --nv neurad-studio.sif \
    python -c 'import torch, tinycudann; print(torch.cuda.get_device_name())'
mkdir -p outputs
apptainer run --cleanenv --nv \
    --bind /absolute/path/to/pandaset:/data/pandaset:ro \
    --bind "$PWD/outputs:/outputs" neurad-studio.sif \
    ns-train splatad --output-dir /outputs pandaset-data --data /data/pandaset
```

`apptainer run` without a command opens Bash. Commands keep the caller's working
directory. The source and virtual environment inside the SIF are read-only;
gsplat's runtime CUDA cache defaults to `$HOME/.cache/neurad/torch_extensions`
(or under `XDG_CACHE_HOME` when set). To use writable scratch space instead,
bind it and pass `--env TORCH_EXTENSIONS_DIR=/scratch/neurad/torch_extensions`.
Viewer Node tooling and download caches are removed after the frontend is built.
For faster SIF compression with a recent Apptainer/mksquashfs installation, add
`--mksquashfs-args '-comp zstd -Xcompression-level 3 -processors 4 -mem 2G'` to
`apptainer build`. This changes image compression, not the installed environment.
The default gzip compression remains available when compatibility requires it.
Waymo's legacy TensorFlow environment remains a separate install.

## Missing gsplat lidar functions or mismatched CUDA versions

SplatAD requires `carlinds/splatad`, not the PyPI/upstream gsplat package. Errors
about `map_points_to_lidar_tiles`, `points_mapping_offset_encode`, or
`populate_image_from_points` usually indicate that the wrong package is installed.
Rerun `scripts/setup-env.sh` in the supported environment, or rebuild the Docker
image. Both use the pinned fork in `scripts/requirements-common.txt`; do not
replace it with `pip install gsplat`.

The CUDA toolkit used to build extensions must match the PyTorch CUDA build.
The setup script checks for toolkit 12.8 and constrains PyTorch to 2.7.1+cu128
throughout dependency installation, so a later package cannot silently upgrade
it. `uv pip check` and imports of the required gsplat APIs run at the end of setup.

## NeuRAD loading memory

NeuRAD caches training images as uint8 by default and converts sampled pixels or
patches to float32 in [0, 1]. This uses one quarter of the float32 image-cache
storage. The default is one loader process with a queue of two batches. To retain
float32 caching, set `--pipeline.datamanager.cache-images-type float32`.

Caching still needs system RAM; reducing rays per batch only reduces the sampled
batches, not the full image cache. On machines with limited RAM, start with:

```bash
ns-train neurad \
    --pipeline.datamanager.num-processes 0 \
    --pipeline.datamanager.camera-res-scale-factor 0.5 \
    --pipeline.datamanager.eval-num-images-to-sample-from 16 \
    --pipeline.datamanager.eval-num-times-to-repeat-images 100 \
    pandaset-data --data /data/pandaset
```

This halves image dimensions (one quarter of the pixels), bounds the sampled
image-evaluation cache, and runs batch generation in the main process. Complete
image evaluation still visits the evaluation images. For container runs, allocate
shared memory and file descriptors as shown above; default Docker shared memory
can be too small for multiprocessing. Check host/container RAM limits if the OS
prints `Killed`. An unexpectedly exited loader worker now raises an error instead
of leaving training blocked on its queue.
