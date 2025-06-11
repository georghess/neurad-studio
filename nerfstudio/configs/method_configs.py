# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Union

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig, ScaledCameraOptimizerConfig
from nerfstudio.configs.base_config import LoggingConfig, ViewerConfig
from nerfstudio.configs.external_methods import ExternalMethodDummyTrainerConfig, get_external_methods
from nerfstudio.data.datamanagers.ad_datamanager import ADDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import FullImageLidarDatamanagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSetDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, AdamWOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.lidar_nerfacto import LidarNerfactoModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neurad import NeuRADModelConfig
from nerfstudio.models.splatad import SplatADModelConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.ad_pipeline import ADPipelineConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.splatad_pipeline import SplatADPipelineConfig
from nerfstudio.plugins.registry import discover_methods

method_configs: Dict[str, Union[TrainerConfig, ExternalMethodDummyTrainerConfig]] = {}
descriptions = {
    "nerfacto": "Nerfstudio's default model.",
    "nerfacto-lidar": "Nerfacto with lidar supervision.",
    "splatfacto": "Gaussian Splatting model for static scenes",
    "neurad": "Continuously improving version of NeuRAD.",
    "neurad-paper": "NeuRAD with settings matching the paper.",
    "splatad": "Gaussian Splatting model for autonomous driving",
}

method_configs["nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=ParallelDataManagerConfig(
            dataparser=PandaSetDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            average_init_density=0.01,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nerfacto-big"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=100000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=ParallelDataManagerConfig(
            dataparser=PandaSetDataParserConfig(),
            train_num_rays_per_batch=8192,
            eval_num_rays_per_batch=4096,
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            num_nerf_samples_per_ray=128,
            num_proposal_samples_per_ray=(512, 256),
            hidden_dim=128,
            hidden_dim_color=128,
            appearance_embed_dim=128,
            max_res=4096,
            proposal_weights_anneal_max_num_iters=5000,
            log2_hashmap_size=21,
            average_init_density=0.01,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nerfacto-huge"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=100000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=ParallelDataManagerConfig(
            dataparser=PandaSetDataParserConfig(),
            train_num_rays_per_batch=16384,
            eval_num_rays_per_batch=4096,
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            num_nerf_samples_per_ray=64,
            num_proposal_samples_per_ray=(512, 512),
            proposal_net_args_list=[
                {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False},
                {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False},
            ],
            hidden_dim=256,
            hidden_dim_color=256,
            appearance_embed_dim=32,
            max_res=8192,
            proposal_weights_anneal_max_num_iters=5000,
            log2_hashmap_size=21,
            average_init_density=0.01,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nerfacto-lidar"] = TrainerConfig(
    method_name="nerfacto-lidar",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=ADPipelineConfig(
        datamanager=ADDataManagerConfig(dataparser=PandaSetDataParserConfig()),
        calc_fid_steps=(99999999,),
        model=LidarNerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "cam_opt": {
            "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["splatfacto"] = TrainerConfig(
    method_name="splatfacto",
    steps_per_eval_image=100,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=1000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=PandaSetDataParserConfig(sequence="028"),  # use static sequence
            cache_images_type="uint8",
        ),
        model=SplatfactoModelConfig(),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-7, max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["splatfacto-big"] = TrainerConfig(
    method_name="splatfacto",
    steps_per_eval_image=100,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=1000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=PandaSetDataParserConfig(sequence="028"),  # use static sequence
            cache_images_type="uint8",
        ),
        model=SplatfactoModelConfig(
            cull_alpha_thresh=0.005,
        ),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["splatad"] = TrainerConfig(
    method_name="splatad",
    steps_per_eval_image=500,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=2500,
    max_num_iterations=30001,
    mixed_precision=False,
    pipeline=SplatADPipelineConfig(
        calc_fid_steps=(30000,),
        datamanager=FullImageLidarDatamanagerConfig(
            dataparser=PandaSetDataParserConfig(add_missing_points=True),
            cache_images_type="uint8",
        ),
        model=SplatADModelConfig(max_steps=30001),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-7, max_steps=30000),
        },
        "camera_velocity_opt_linear": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-6, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "camera_velocity_opt_angular": {
            "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "camera_velocity_opt_time_to_center_pixel": {
            "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-7, max_steps=30000, warmup_steps=10000, lr_pre_warmup=0
            ),
        },
        "trajectory_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001, warmup_steps=2500),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20001, warmup_steps=500),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["neurad"] = TrainerConfig(
    method_name="neurad",
    steps_per_eval_batch=500,
    steps_per_eval_all_images=5000,
    steps_per_save=2000,
    max_num_iterations=20001,
    mixed_precision=True,
    pipeline=ADPipelineConfig(
        calc_fid_steps=(99999999,),
        datamanager=ADDataManagerConfig(dataparser=PandaSetDataParserConfig(add_missing_points=True)),
        model=NeuRADModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            camera_optimizer=CameraOptimizerConfig(mode="off"),  # SO3xR3
        ),
    ),
    optimizers={
        "trajectory_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001, warmup_steps=2500),
        },
        "cnn": {
            "optimizer": AdamWOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=20001, warmup_steps=2500),
        },
        "fields": {
            "optimizer": AdamWOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-7),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20001, warmup_steps=500),
        },
        "hashgrids": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=20001, warmup_steps=500),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=20001, warmup_steps=2500),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
    logging=LoggingConfig(steps_per_log=100),
)

# With scaled camera optimizer (tuned for nuscenes)
method_configs["neurad-scaleopt"] = deepcopy(method_configs["neurad"])
method_configs["neurad-scaleopt"].method_name = "neurad-scaleopt"
method_configs["neurad-scaleopt"].pipeline.model.camera_optimizer = ScaledCameraOptimizerConfig(
    weights=(1.0, 1.0, 0.01, 0.01, 0.01, 1.0),  # xrot, yrot, zrot, xtrans, ytrans, ztrans
    trans_l2_penalty=(
        1e-2,
        1e-2,
        1e-3,
    ),  # x, y, z
    mode="SO3xR3",
)


def _scaled_neurad_training(config: TrainerConfig, scale: float, newname: str) -> TrainerConfig:
    config = deepcopy(config)
    config.method_name = newname
    config.max_num_iterations = int((config.max_num_iterations - 1) * scale + 1)
    config.steps_per_eval_batch = int(config.steps_per_eval_batch * scale)
    config.steps_per_eval_image = int(config.steps_per_eval_image * scale)
    config.steps_per_eval_all_images = int(config.steps_per_eval_all_images * scale)
    config.steps_per_save = int(config.steps_per_save * scale)
    assert isinstance(config.pipeline, ADPipelineConfig)
    config.pipeline.calc_fid_steps = tuple(int(scale * s) for s in config.pipeline.calc_fid_steps)
    for optimizer in config.optimizers.values():
        optimizer["scheduler"].max_steps = int(optimizer["scheduler"].max_steps * scale)
        optimizer["scheduler"].warmup_steps = int(optimizer["scheduler"].warmup_steps * scale)
    return config


# Bigger, better, longer, stronger
method_configs["neurader"] = _scaled_neurad_training(method_configs["neurad"], 2.5, "neurader")
for optimizer in method_configs["neurader"].optimizers.values():
    optimizer["optimizer"].lr *= 0.5
    optimizer["scheduler"].lr_final *= 0.5
model: NeuRADModelConfig = method_configs["neurader"].pipeline.model
for field in (model.field, model.sampling.proposal_field_1, model.sampling.proposal_field_2):
    field.grid.static.max_res *= 2
    field.grid.static.base_res *= 2
    field.grid.static.log2_hashmap_size += 1
    field.grid.actor.log2_hashmap_size += 1

method_configs["neurader-scaleopt"] = deepcopy(method_configs["neurader"])
method_configs["neurader-scaleopt"].method_name = "neurader-scaleopt"
method_configs["neurader-scaleopt"].pipeline.model.camera_optimizer = ScaledCameraOptimizerConfig(
    weights=(1.0, 1.0, 0.01, 0.01, 0.01, 1.0),  # xrot, yrot, -zrot-, -xtrans-, -ytrans-, ztrans
    trans_l2_penalty=(
        1e-2,
        1e-2,
        1e-3,
    ),  # x, y, z
    mode="SO3xR3",
)

# Even longer training
method_configs["neuradest"] = _scaled_neurad_training(method_configs["neurader"], 3, "neuradest")
method_configs["neuradest-scaleopt"] = _scaled_neurad_training(
    method_configs["neurader-scaleopt"], 3, "neuradest-scaleopt"
)

# Configurations matching the paper (disable temporal appearance and actor flip)
method_configs["neurad-paper"] = deepcopy(method_configs["neurad"])
method_configs["neurad-paper"].method_name = "neurad-paper"
method_configs["neurad-paper"].pipeline.model.use_temporal_appearance = False  # type: ignore
for f in method_configs["neurad-paper"].pipeline.model.fields:  # type: ignore
    f.flip_prob = 0.0
method_configs["neurad-2x-paper"] = deepcopy(method_configs["neurader"])
method_configs["neurad-2x-paper"].method_name = "neurad-paper"
method_configs["neurad-2x-paper"].pipeline.model.use_temporal_appearance = False  # type: ignore
for f in method_configs["neurad-2x-paper"].pipeline.model.fields:  # type: ignore
    f.flip_prob = 0.0


def merge_methods(methods, method_descriptions, new_methods, new_descriptions, overwrite=True):
    """Merge new methods and descriptions into existing methods and descriptions.
    Args:
        methods: Existing methods.
        method_descriptions: Existing descriptions.
        new_methods: New methods to merge in.
        new_descriptions: New descriptions to merge in.
    Returns:
        Merged methods and descriptions.
    """
    methods = OrderedDict(**methods)
    method_descriptions = OrderedDict(**method_descriptions)
    for k, v in new_methods.items():
        if overwrite or k not in methods:
            methods[k] = v
            method_descriptions[k] = new_descriptions.get(k, "")
    return methods, method_descriptions


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


all_methods, all_descriptions = method_configs, descriptions
# Add discovered external methods
all_methods, all_descriptions = merge_methods(all_methods, all_descriptions, *discover_methods())
all_methods, all_descriptions = sort_methods(all_methods, all_descriptions)

# Register all possible external methods which can be installed with Nerfstudio
all_methods, all_descriptions = merge_methods(
    all_methods, all_descriptions, *sort_methods(*get_external_methods()), overwrite=False
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
