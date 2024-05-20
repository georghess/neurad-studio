# Copyright 2024 the authors of NeuRAD and contributors.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import torch
from rich.console import Console

from nerfstudio.cameras.lidars import transform_points
from nerfstudio.data.datamanagers.image_lidar_datamanager import ImageLidarDataManager, ImageLidarDataManagerConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import ScaledPatchSamplerConfig
from nerfstudio.data.utils.data_utils import remove_dynamic_points

CONSOLE = Console(width=120)


@dataclass
class ADDataManagerConfig(ImageLidarDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: ADDataManager)
    """Target class to instantiate."""
    train_num_lidar_rays_per_batch: int = 16384
    """Number of lidar rays per batch to use per training iteration."""
    train_num_rays_per_batch: int = 40960
    """Number of camera rays per batch to use per training iteration (equals 40 32x32 patches)."""
    eval_num_lidar_rays_per_batch: int = 8192
    """Number of lidar rays per batch to use per eval iteration."""
    eval_num_rays_per_batch: int = 40960
    """Number of camera rays per batch to use per eval iteration (equals 40 32x32 patches)."""
    downsample_factor: float = 1
    """Downsample factor for the lidar. If <1, downsample will be used."""
    image_divisible_by: int = 1
    """If >1, images will be cropped to be divisible by this number."""
    pixel_sampler: ScaledPatchSamplerConfig = field(default_factory=ScaledPatchSamplerConfig)
    """AD models default to path-based pixel sampler."""


class ADDataManager(ImageLidarDataManager):
    """This extends the VanillaDataManager to support lidar data.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ADDataManagerConfig

    def create_eval_dataset(self) -> InputDataset:
        dataset = super().create_eval_dataset()
        # Maybe crop images
        cams = dataset.cameras
        for width in cams.width.unique().tolist():
            width_crop = _find_smallest_crop(width, self.config.image_divisible_by)
            cams.width[cams.width == width] = width - width_crop
        for height in cams.height.unique().tolist():
            height_crop = _find_smallest_crop(height, self.config.image_divisible_by)
            cams.height[cams.height == height] = height - height_crop
        return dataset

    def change_patch_sampler(self, patch_scale: int, patch_size: int):
        """Change the camera sample to sample rays in NxN patches."""
        if self.config.train_num_rays_per_batch % (patch_size**2) != 0:
            CONSOLE.print("[bold yellow]WARNING: num_rays should be divisible by patch_size^2.")
        if patch_scale == self.eval_pixel_sampler.patch_scale and patch_size == self.eval_pixel_sampler.patch_size:
            return

        # Change train
        if self.use_mp:
            for func_queue in self.func_queues:
                func_queue.put((_worker_change_patch_sampler, (patch_scale, patch_size), {}))
            self.clear_data_queue()  # remove any old, invalid, batch
        else:
            _worker_change_patch_sampler(self.data_procs[0], patch_scale, patch_size)

        # Change eval
        self.eval_pixel_sampler.patch_scale = patch_scale
        self.eval_pixel_sampler.patch_size = patch_size
        if patch_scale % 2 == 0 and self.eval_ray_generator.image_coords[0, 0, 0] == 0.5:
            self.eval_ray_generator.image_coords = self.eval_ray_generator.image_coords - 0.5

    def get_accumulated_lidar_points(self, remove_dynamic: bool = False):
        """Get the lidar points for the current batch."""
        lidars = self.train_lidar_dataset.lidars
        point_clouds = self.train_lidar_dataset.point_clouds
        if remove_dynamic:
            assert "trajectories" in self.train_lidar_dataset.metadata, "No trajectories found in dataset."

            point_clouds = remove_dynamic_points(
                point_clouds,
                lidars.lidar_to_worlds,
                lidars.times,
                self.train_lidar_dataset.metadata["trajectories"],
            )

        return torch.cat(
            [transform_points(pc[:, :3], l2w) for pc, l2w in zip(point_clouds, lidars.lidar_to_worlds)], dim=0
        )


def _find_smallest_crop(dim: int, divider: int):
    crop_amount = 0
    while dim % divider:
        crop_amount += 1
        dim -= 1
    return crop_amount


def _worker_change_patch_sampler(worker, patch_scale, patch_size):
    worker.pixel_sampler.patch_scale = patch_scale
    worker.pixel_sampler.patch_size = patch_size
    # Ensure ray is generated from patch center (for odd scales center of pixel == center of patch)
    if patch_scale % 2 == 0 and worker.ray_generator.image_coords[0, 0, 0] == 0.5:
        worker.ray_generator.image_coords -= 0.5
