<p align="center">
    <!-- project badges -->
    <a href="https://research.zenseact.com/publications/neurad/"><img src="https://img.shields.io/badge/Project-Page-ffa"/></a>
    <!-- paper badges -->
    <a href="https://arxiv.org/abs/2311.15260">
        <img src='https://img.shields.io/badge/arXiv-Page-aff'>
    </a>
</p>


<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/neurad_logo_with_text_dark.png" />
    <img alt="tyro logo" src="docs/_static/imgs/neurad_logo_with_text.png" width="80%"/>
</picture>
</div>

<div align="center">
<h3 style="font-size:2.0em;">Neural Rendering for Autonomous Driving</h3>
<h4>CVPR 2024 highlight</h4>
</div>
<div align="center">

[Quickstart](#quickstart) ·
[Learn more](#learn-more) ·
[Planned Features](#planned-featurestodos)

</div>

# About

This is the official code release of the CVPR 2024 paper _NeuRAD: Neural Rendering for Autonomous Driving_, building on top of [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). Besides releasing the code for our NeuRAD model, we hope that this can lay the ground-work for research on applying neural rendering methods in autonomous driving.

In line with Nerfstudio's mission, this is a contributor-friendly repo with the goal of building a community where users can more easily build upon each other's contributions.

Do you have feature requests or want to add **your** new AD-NeRF model? Or maybe provide structures for a new dataset? **We welcome contributions!**

<div align="center">
<a href="https://zenseact.com/">
<picture style="padding-left: 10px; padding-right: 10px;">
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/ZEN_Vertical_logo_white.svg" />
    <img alt="zenseact logo" src="docs/_static/imgs/ZEN_Vertical_logo_black.svg" height="100px" />
</picture>
</a>
<a href="https://www.chalmers.se/en/">
<picture style="padding-left: 10px; padding-right: 10px; padding-bottom: 10px;">
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/EN_Avancez_CH_white.png" />
    <img alt="chalmers logo" src="docs/_static/imgs/EN_Avancez_CH_black.png" height="90px" />
</picture>
</a>
<a href="https://www.lunduniversity.lu.se/">
<picture style="padding-left: 10px; padding-right: 10px;">
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/LundUniversity_C2line_NEG.png" />
    <img alt="lund logo" src="docs/_static/imgs/LundUniversity_C2line_BLACK.png" height="100px" />
</picture>
</a>
<a href="https://liu.se/en">
<picture style="padding-left: 10px; padding-right: 10px;">
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/LiU_secondary_1_white-PNG.png" />
    <img alt="liu logo" src="docs/_static/imgs/LiU_secondary_1_black-PNG.png" height="100px" />
</picture>
</a>
<a href="https://wasp-sweden.org/">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/imgs/WASP-logotype-white.png" />
    <img alt="wasp logo" src="docs/_static/imgs/WASP_logotyp_grey_180116.png" height="80px" />
</picture>
</a>
</div>

# Quickstart

The quickstart will help you get started with the NeuRAD model on a PandaSet sequence.
For more complex changes (e.g., running with your own data/setting up a new NeRF graph), please refer to our [references](#learn-more).

## 1. Installation: Setup the environment

### Prerequisites

Our installation steps largely follow Nerfstudio, with some added dataset-specific dependencies. You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

### Create environment

NeuRAD requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name neurad -y python=3.8
conda activate neurad
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

We refer to [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/blob/v1.0.3/docs/quickstart/installation.md) for more installation support.

### Installing NeuRAD
```bash
git clone https://github.com/georghess/neurad-studio.git
cd neurad-studio
pip install --upgrade pip setuptools
pip install -e .
```

**OR** if you want to skip all installation steps and directly start using NeuRAD, use the provided docker image or apptainer recipe:

[Dockerfile](Dockerfile) or [Apptainer recipe](apptainer_recipe).

## 2. Training your first model!

The following will train a _NeuRAD_ model, our recommended model for real world AD scenes.

### Data preparation

Begin by downloading [PandaSet](https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset/data) and place it under ```data/pandaset```.

### Training

Training models is done the same way as in nerfstudio, i.e.,

```bash
# Train model
python nerfstudio/scripts/train.py neurad pandaset-data
```

If everything works, you should see training progress like the following:

<p align="center">
    <img width="800" alt="image" src="docs/_static/imgs/readme_training_progress_example.png">
</p>

Navigating to the link at the end of the terminal will load the webviewer. If you are running on a remote machine, you will need to port forward the websocket port (defaults to 7007).

<p align="center">
    <img width="800" alt="image" src="docs/_static/imgs/readme_viewer_neurad.png">
</p>

### Resume from checkpoint / visualize existing run

It is possible to load a pretrained model by running

```bash
pyhton nerfstudio/scripts/train.py neurad pandaset-data --load-dir {outputs/.../nerfstudio_models}
```

## Visualize existing run

Given a pretrained model checkpoint, you can start the viewer by running

```bash
python nerfstudio/scripts/viewer/run_viewer.py --load-config {outputs/.../config.yml}
```

## 3. Exporting Results

Once you have a NeRF model you can render its output. There are multiple different renders, more info available using

```bash
python nerfstudio/scripts/render.py --help
```


## 4. Advanced Options

### Training models other than NeuRAD

Besides NeuRAD, we will provide a reimplementation of [UniSim](https://arxiv.org/abs/2308.01898) as well. Once this is released it can be trained using

```bash
# Train model
python nerfstudio/scripts/train.py unisim pandaset-data
```

Further, as we build on top of nerfstudio, models such as _nerfacto_ or _splatfacto_ are available as well, see nerfstudio for details. However, note that these are made for static scenes.

For a full list of included models run `python nerfstudio/scripts/train.py --help`.

### Modify Configuration

Each model contains many parameters that can be changed, too many to list here. Use the `--help` command to see the full list of configuration options.

```bash
python nerfstudio/scripts/train.py neurad --help
```

### Tensorboard / WandB / Comet / Viewer

There are four different methods to track training progress, using the viewer, [tensorboard](https://www.tensorflow.org/tensorboard), [Weights and Biases](https://wandb.ai/site), and [Comet](https://comet.com/?utm_source=nerf&utm_medium=referral&utm_content=github). You can specify which visualizer to use by appending `--vis {viewer, tensorboard, wandb, comet viewer+wandb, viewer+tensorboard, viewer+comet}` to the training command. Simultaneously utilizing the viewer alongside wandb or tensorboard may cause stuttering issues during evaluation steps.

# Learn More

And that's it for getting started with the basics of NeuRAD. If you are missing some features, have a look at [Planned Features](#planned-featurestodos) to see if we have plans on implementing this. Otherwise, feel free to open an issue, or even better implement it yourself and open a PR!

If you want to add a dataset, look [here](#adding-datasets). If you want to add a method, have a look [here](#adding-methods).

## Adding Datasets

We have provided dataparsers for multiple autonomous driving dataset, see below for a complete list. However, your favorite AD dataset might still be missing.

To add a dataset, create `nerfstudio/data/dataparsers/mydataset.py` containing one dataparsers config class `MyADDataParserConfig` and one dataparser class `MyADData`. Preferrably, these inherit from `ADDataParserConfig` and `ADDataParser`, as these provide common functionality and streamline the expected format of AD data. For most datasets, it should then be sufficient to overwrite `_get_cameras`, `_get_lidars`, `_read_lidars`, `_get_actor_trajectories`, and `_generate_dataparser_outputs`.

| Data                                                                                          | Cameras | Lidars                                                      |
| --------------------------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------------- |
| 🚗 [nuScenes](https://www.nuscenes.org/)          | 6 cameras            |  32-beam lidar                   |
| 🚗 [ZOD](https://zod.zenseact.com/)           | 1 camera            | 128-beam + 2 x 16-beam lidars                   |
| 🚗 [Argoverse 2](https://www.argoverse.org/av2.html)   | 7 ring cameras + 2 stereo cameras            | 2 x 32-beam lidars                   |
| 🚗 [PandaSet](https://pandaset.org/)         | 6 cameras | 64-beam lidar                                  |
| 🚗 [KITTIMOT](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) | 2 stereo cameras | 64-beam lidar



## Adding Methods

Nerfstudio has made it easy to add new methods, see [here](https://docs.nerf.studio/developer_guides/new_methods.html) for details. We plan to examplify this using our UniSim reimplementation, to be released soon.

# Key features
- Dataparser for multiple autonomous driving datasets including
    - Dataparsing of lidar data (3D+intensity+time)
    - Dataparsing of annotations
- Datamanager for lidar+image data
- Rolling shutter handling for ray generation
- Viewer improvements
    - Lidar rendering
    - Dynamic actor modifications
- NeuRAD - SOTA neural rendering method for dynamic AD scenes


# Planned Features/TODOs

- [ ] UniSim plug-in
- [x] Release code

# Built On

<a href="https://github.com/nerfstudio-project/nerfstudio">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/docs/_static/imgs/logo-dark.png" />
    <img alt="tyro logo" src="https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/docs/_static/imgs/logo.png" width="150px" />
</picture>
</a>

- Collaboration friendly studio for NeRFs

# Citation

You can find our paper on [arXiv](https://arxiv.org/abs/2311.15260).

If you use this code or find our paper useful, please consider citing:

```bibtex
@article{neurad,
  title={NeuRAD: Neural Rendering for Autonomous Driving},
  author={Tonderski, Adam and Lindstr{\"o}m, Carl and Hess, Georg and Ljungbergh, William and Svensson, Lennart and Petersson, Christoffer},
  journal={arXiv preprint arXiv:2311.15260},
  year={2023}
}
```

# Contributors

<a href="https://github.com/georghess">
    <img src="https://github.com/georghess.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/carlinds">
    <img src="https://github.com/carlinds.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/atonderski">
    <img src="https://github.com/atonderski.png" width="60px;" style="border-radius: 50%;"/>
</a>
<a href="https://github.com/wljungbergh">
    <img src="https://github.com/wljungbergh.png" width="60px;" style="border-radius: 50%;"/>
</a> 

\+ [nerfstudio contributors](https://github.com/nerfstudio-project/nerfstudio/graphs/contributors)