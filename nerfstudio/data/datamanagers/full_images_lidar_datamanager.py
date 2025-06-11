# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Data manager that outputs cameras / images and lidars / point clouds instead of raybundles

Good for things like gaussian splatting which require full sensors instead of the standard ray
paradigm
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin

import torch
from gsplat import map_points_to_lidar_tiles, points_mapping_offset_encode, populate_image_from_points
from rich.progress import track
from typing_extensions import assert_never

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import (
    Lidars,
    LidarType,
    get_lidar_azimuth_resolution,
    get_lidar_elevation_mapping,
    transform_points,
)
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.lidar_dataset import LidarDataset
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.poses import inverse
from nerfstudio.utils.rich_utils import CONSOLE

AZIM_CHANNELS_PER_TILE = 32
ELEV_CHANNELS_PER_TILE = 8


@dataclass
class FullImageLidarDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: FullImageLidarDatamanager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=DataParserConfig)
    eval_num_lidars_to_sample_from: int = -1
    """Number of lidars to sample during eval iteration."""
    eval_num_times_to_repeat_lidars: int = -1
    """When not evaluating on all lidars, number of iterations before picking
    new lidars. If -1, never pick new lidars."""
    eval_lidar_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the lidar indices to use during eval; if None, uses all."""
    cache_lidars: Literal["cpu", "gpu"] = "gpu"
    """Whether to cache lidars in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images and lidars. If None, uses all available threads."""
    downsample_factor: float = 1
    """Downsample factor for the lidar. If <1, downsample will be used."""
    paint_points: bool = True
    """Whether to project points into image and store their RGB values."""
    paint_points_topk: int = 2
    """Number of top cameras to use for painting points.
    For example, 2 means the two closest of every camera (2 front, 2 left, 2 right, etc)."""
    train_lidar_only: bool = False
    """Whether to only train on lidar data."""
    train_image_only: bool = False
    """Whether to only train on image data."""


class FullImageLidarDatamanager(FullImageDatamanager, Generic[TDataset]):
    """
    A datamanager that outputs full images and cameras instead of raybundles. This makes the
    datamanager more lightweight since we don't have to do generate rays. Useful for full-image
    training e.g. rasterization pipelines
    """

    config: FullImageLidarDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_lidar_dataset: LidarDataset
    eval_lidar_dataset: LidarDataset

    def __init__(
        self,
        config: FullImageLidarDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

        if len(self.train_lidar_dataset) > 500 and self.config.cache_lidars == "gpu":
            CONSOLE.print(
                "Lidar train dataset has over 500 point clouds, overriding cache_lidars to cpu",
                style="bold yellow",
            )
            self.config.cache_lidars = "cpu"

        if self.config.paint_points:
            CONSOLE.log("Painting lidar points")
            self.paint_points()

        # Some logic to make sure we sample every camera in equal amounts
        self.train_unseen_lidars = [i for i in range(len(self.train_lidar_dataset))]
        self.eval_unseen_lidars = [i for i in range(len(self.eval_lidar_dataset))]
        assert len(self.train_unseen_lidars) > 0, "No data found in dataset"

    @cached_property
    def cached_lidar_train(self) -> List[Dict[str, torch.Tensor]]:
        """Get the training images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_lidars("train", cache_lidars_device=self.config.cache_lidars)

    @cached_property
    def cached_lidar_eval(self) -> List[Dict[str, torch.Tensor]]:
        """Get the eval images. Will load and undistort the images the
        first time this (cached) property is accessed."""
        return self._load_lidars("eval", cache_lidars_device=self.config.cache_lidars)

    def _lidar_to_raster_pts(
        self,
        point_cloud,
        lidar,
        elevation_boundaries,
        elevation_mapping,
        azimuth_resolution,
        did_return_threshold,
        is_eval,
    ):
        if not is_eval:
            # shuffle points
            point_cloud = point_cloud[torch.randperm(point_cloud.shape[0])]

        # Remove ego motion
        rs_adjusted_point_cloud = point_cloud[:, :3] - lidar.metadata["linear_velocities_local"] * point_cloud[..., 4:5]

        azimuth = torch.rad2deg(torch.atan2(rs_adjusted_point_cloud[:, 1], rs_adjusted_point_cloud[:, 0]))
        distance = torch.linalg.vector_norm(rs_adjusted_point_cloud[:, :3], dim=1)
        elevation = torch.rad2deg(torch.asin(rs_adjusted_point_cloud[:, 2] / distance))

        intensity = point_cloud[:, 3]
        point_cloud_time = point_cloud[:, 4]
        spherical_coords_time_intensity = torch.stack(
            [azimuth, elevation, distance, point_cloud_time, intensity], dim=1
        ).cuda()
        points_tile_ids, flatten_ids = map_points_to_lidar_tiles(
            spherical_coords_time_intensity[None, :, :2],
            elevation_boundaries,
            azimuth_resolution * AZIM_CHANNELS_PER_TILE,
            -180.0,
        )
        tile_width = math.ceil(360 / (azimuth_resolution * AZIM_CHANNELS_PER_TILE))
        tile_height = len(elevation_boundaries) - 1
        tile_offsets = points_mapping_offset_encode(
            points_tile_ids,
            1,
            tile_width,
            tile_height,
        )

        image_width = tile_width * AZIM_CHANNELS_PER_TILE
        image_height = len(elevation_mapping)

        if is_eval:
            points_per_tile = torch.cat(
                [tile_offsets.flatten(), torch.tensor([point_cloud.shape[0]], device=tile_offsets.device)]
            ).diff()
            max_points_per_tile = ELEV_CHANNELS_PER_TILE * AZIM_CHANNELS_PER_TILE
            n_batches = (points_per_tile // (max_points_per_tile + 1)).max() + 1
            raster_pts_image = torch.zeros((n_batches, image_height, image_width, 5), device=point_cloud.device)
            for batch_idx in range(n_batches):
                flatten_ids_batch = torch.cat(
                    [
                        flatten_ids[s : (s + n)]
                        for s, n in zip(
                            (tile_offsets.flatten() + max_points_per_tile * batch_idx),
                            points_per_tile.clamp_max(max_points_per_tile),
                        )
                    ]
                )
                tile_offsets_batch = (
                    torch.cat(
                        [
                            torch.tensor([0], device=points_per_tile.device),
                            points_per_tile.clamp_max(max_points_per_tile).cumsum(dim=0)[:-1],
                        ]
                    )
                    .view(tile_offsets.shape)
                    .int()
                )
                points_per_tile = (points_per_tile - max_points_per_tile).clamp_min(0)

                raster_pts_image[batch_idx] = populate_image_from_points(  # (azimuth, elev, depth, time, intensity)
                    spherical_coords_time_intensity[None],
                    image_width=image_width,
                    image_height=image_height,
                    tile_width=AZIM_CHANNELS_PER_TILE,
                    tile_height=ELEV_CHANNELS_PER_TILE,
                    tile_offsets=tile_offsets_batch,
                    flatten_id=flatten_ids_batch,
                )

        else:
            raster_pts_image = populate_image_from_points(  # (azimuth, elev, depth, time, intensity)
                spherical_coords_time_intensity[None],
                image_width=image_width,
                image_height=image_height,
                tile_width=AZIM_CHANNELS_PER_TILE,
                tile_height=ELEV_CHANNELS_PER_TILE,
                tile_offsets=tile_offsets,
                flatten_id=flatten_ids,
            )

        return raster_pts_image

    def _add_metadata(self, lidar, data, num_cameras):
        data["lidar"] = data["lidar"].to(self.device)
        data["elevation_boundaries"] = data["elevation_boundaries"].to(self.device)
        data["elevation_mapping"] = data["elevation_mapping"].to(self.device)
        lidar.metadata["elevation_boundaries"] = data["elevation_boundaries"]
        lidar.metadata["azimuth_resolution"] = data["azimuth_resolution"]
        lidar.metadata["cam_idx"] = lidar.metadata["lidar_idx"] + num_cameras
        raster_pts_image = self._lidar_to_raster_pts(
            data["lidar"],
            lidar,
            data["elevation_boundaries"],
            data["elevation_mapping"],
            data["azimuth_resolution"],
            lidar.valid_lidar_distance_threshold,
            data["is_eval"],
        )
        data["raster_pts"] = raster_pts_image
        data["raster_pts_did_return"] = raster_pts_image[..., 2] <= lidar.valid_lidar_distance_threshold
        data["raster_pts_valid_depth_and_did_return"] = (
            (data["raster_pts_did_return"] & (raster_pts_image[..., 2] > 0)).flatten().nonzero().squeeze()
        )
        data["raster_pts_valid_depth_and_did_not_return"] = (
            (~data["raster_pts_did_return"] & (raster_pts_image[..., 2] > 0)).flatten().nonzero().squeeze()
        )
        lidar.metadata["raster_pts"] = raster_pts_image
        data["lidar_pts_did_return"] = data["lidar"].norm(dim=-1) <= lidar.valid_lidar_distance_threshold
        data["linear_velocities_local"] = lidar.metadata["linear_velocities_local"]

    def _load_lidars(self, split, cache_lidars_device):
        # Which dataset?
        if split == "train":
            dataset = self.train_lidar_dataset
        elif split == "eval":
            dataset = self.eval_lidar_dataset
        else:
            assert_never(split)

        def process_data(idx):
            data = dataset.get_data(idx)
            lidar = dataset.lidars[idx : idx + 1]
            lidar_type = LidarType(lidar.lidar_type.item())
            elevation_mapping = get_lidar_elevation_mapping(lidar_type)
            elevation_mapping = torch.tensor(sorted(elevation_mapping.values())).float()
            elevation_boundaries = torch.cat(
                [
                    elevation_mapping[0:1] - 1.0,
                    (
                        elevation_mapping[ELEV_CHANNELS_PER_TILE::ELEV_CHANNELS_PER_TILE]
                        + elevation_mapping[ELEV_CHANNELS_PER_TILE - 1 : -1 : ELEV_CHANNELS_PER_TILE]
                    )
                    / 2,
                    elevation_mapping[-1:] + 1.0,
                ]
            )
            azimuth_resolution = get_lidar_azimuth_resolution(lidar_type)
            data["elevation_boundaries"] = elevation_boundaries.cpu()
            data["elevation_mapping"] = elevation_mapping.cpu()
            data["azimuth_resolution"] = azimuth_resolution
            data["is_eval"] = split == "eval"
            return data

        CONSOLE.log(f"Caching {split} lidars")
        with ThreadPoolExecutor(max_workers=2) as executor:
            cached_data = list(
                track(
                    executor.map(
                        process_data,
                        range(len(dataset)),
                    ),
                    description=f"Caching {split} lidars",
                    transient=True,
                    total=len(dataset),
                )
            )

        if cache_lidars_device == "gpu":
            for cache in cached_data:
                cache["lidar"] = cache["lidar"].to(self.device)
                cache["elevation_boundaries"] = cache["elevation_boundaries"].to(self.device)
                cache["elevation_mapping"] = cache["elevation_mapping"].to(self.device)
                self.train_lidars = self.train_lidar_dataset.lidars.to(self.device)
        else:
            for cache in cached_data:
                cache["lidar"] = cache["lidar"].pin_memory()
                cache["elevation_boundaries"] = cache["elevation_boundaries"].pin_memory()
                cache["elevation_mapping"] = cache["elevation_mapping"].pin_memory()
                self.train_lidars = self.train_lidar_dataset.lidars

        return cached_data

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        self.train_lidar_dataset = LidarDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            downsample_factor=self.config.downsample_factor,
        )
        return super().dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        self.eval_lidar_dataset = LidarDataset(
            dataparser_outputs=eval_dataparser_outputs,
            downsample_factor=self.config.downsample_factor,
        )
        return super().dataset_type(
            dataparser_outputs=eval_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[FullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is FullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is FullImageDatamanager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is FullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        """Sets up the data loaders for training"""

    def setup_eval(self):
        """Sets up the data loader for evaluation"""

    @property
    def fixed_indices_eval_lidar_dataloader(self) -> List[Tuple[Lidars, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (lidar, data) tuples
        """
        lidar_indices = [i for i in range(len(self.eval_lidar_dataset))]
        data = [d.copy() for d in self.cached_lidar_eval]
        _lidars = deepcopy(self.eval_lidar_dataset.lidars).to(self.device)
        lidars = []
        for i in lidar_indices:
            data[i]["lidar"] = data[i]["lidar"].to(self.device)
            _lidar = _lidars[i : i + 1]
            _lidar.metadata["lidar_idx"] = i
            self._add_metadata(_lidar, data[i], len(self.eval_dataset))
            lidars.append(_lidar)
        assert len(self.eval_lidar_dataset.lidars.shape) == 1, "Assumes single batch dimension"
        return list(zip(lidars, data))

    def next_train_lidar(self, step: int) -> Tuple[Lidars, Dict]:
        """Returns the next training batch"""
        lidar_idx = self.train_unseen_lidars.pop(random.randint(0, len(self.train_unseen_lidars) - 1))

        assert len(self.train_lidars.shape) == 1, "Assumes single batch dimension"
        lidar = self.train_lidars[lidar_idx : lidar_idx + 1].to(self.device)
        if lidar.metadata is None:
            lidar.metadata = {}
        lidar.metadata["lidar_idx"] = lidar_idx

        data = self.cached_lidar_train[lidar_idx]
        data = data.copy()

        self._add_metadata(lidar, data, len(self.train_dataset))

        return lidar, data

    def next_train_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))

        data = self.cached_train[image_idx]
        # We're going to copy to make sure we don't mutate the cached dictionary.
        # This can cause a memory leak: https://github.com/nerfstudio-project/nerfstudio/issues/3335
        data = data.copy()
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_train(self, step: int) -> Tuple[Union[Cameras, Lidars], Dict]:
        """Returns the next training batch

        Returns a Camera or Lidar instead of raybundle"""
        if (len(self.train_unseen_cameras) + len(self.train_unseen_lidars)) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
            self.train_unseen_lidars = [i for i in range(len(self.train_lidar_dataset))]

        if self.config.train_lidar_only:
            self.train_unseen_cameras = []
        if self.config.train_image_only:
            self.train_unseen_lidars = []

        if random.randint(0, len(self.train_unseen_cameras) + len(self.train_unseen_lidars) - 1) < len(
            self.train_unseen_cameras
        ):
            return self.next_train_image(step)
        else:
            return self.next_train_lidar(step)

    def next_eval(self, step: int) -> Tuple[Union[Cameras, Lidars], Dict]:
        """Returns the next evaluation batch

        Returns a Camera or Lidar instead of raybundle"""
        # repopulate unseen cameras and lidars if they are empty
        if (len(self.eval_unseen_cameras) + len(self.eval_unseen_lidars)) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
            self.eval_unseen_lidars = [i for i in range(len(self.eval_lidar_dataset))]

        if random.randint(0, len(self.eval_unseen_cameras) + len(self.eval_unseen_lidars) - 1) < len(
            self.eval_unseen_cameras
        ):
            return self.next_eval_image(step)
        else:
            return self.next_eval_lidar(step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        data = self.cached_eval[image_idx]
        data = data.copy()
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval_lidar(self, step: int) -> Tuple[Lidars, Dict]:
        """Returns the next evaluation batch

        Returns a Lidar instead of raybundle"""
        if len(self.eval_unseen_lidars) == 0:
            self.eval_unseen_lidars = [i for i in range(len(self.eval_lidar_dataset))]
        lidar_idx = self.eval_unseen_lidars.pop(random.randint(0, len(self.eval_unseen_lidars) - 1))
        assert len(self.eval_lidar_dataset.lidars.shape) == 1, "Assumes single batch dimension"
        lidar = self.eval_lidar_dataset.lidars[lidar_idx : lidar_idx + 1].to(self.device)
        if lidar.metadata is None:
            lidar.metadata = {}
        lidar.metadata["lidar_idx"] = lidar_idx

        data = self.cached_lidar_eval[lidar_idx]
        data = data.copy()

        self._add_metadata(lidar, data, len(self.eval_dataset))

        return lidar, data

    def paint_points(self):
        cameras = self.train_dataset.cameras
        lidars = self.train_lidar_dataset.lidars
        image_cache = self.cached_train
        lidar_cache = self.cached_lidar_train
        point_clouds_rgb = []
        topk = len(cameras.metadata["sensor_idxs"].unique()) * self.config.paint_points_topk
        topk = min(topk, len(cameras))

        for lidar_i, lidar_data in enumerate(lidar_cache):
            pc = lidar_data["lidar"].to("cpu")
            lidar = lidars[lidar_i]
            pc_in_world = transform_points(pc[:, :3], lidar.lidar_to_worlds)
            point_cloud_rgb = torch.rand_like(pc[:, :3]) * 255
            lidar_time = lidar.times.squeeze(-1)
            top_k_cam_idx = torch.topk(
                (cameras.times - lidar_time).abs().squeeze(),
                topk,
                largest=False,
            ).indices
            for cam_idx in top_k_cam_idx.flip(0):
                camera = cameras[cam_idx]
                pc_in_camera = transform_points(pc_in_world, inverse(camera.camera_to_worlds.squeeze(0)))
                # Flip the y and z axis because of nerfstudio conventions
                pc_in_camera[:, 1] = -pc_in_camera[:, 1]
                pc_in_camera[:, 2] = -pc_in_camera[:, 2]
                # Only paint points in front of the camera
                valid_points = pc_in_camera[:, 2] > 0
                # Normalize the points
                pc_in_camera = pc_in_camera / pc_in_camera[:, 2:]

                intrinsics = camera.get_intrinsics_matrices().squeeze(0)
                pc_in_image = (torch.matmul(intrinsics, pc_in_camera[:, :3].T).T).to(torch.int64)

                # Only paint points that are within the image
                valid_points = (
                    valid_points
                    & (pc_in_image[:, 0] >= 0)
                    & (pc_in_image[:, 0] < camera.width)
                    & (pc_in_image[:, 1] >= 0)
                    & (pc_in_image[:, 1] < camera.height)
                )

                image = image_cache[cam_idx]["image"].to("cpu")
                point_cloud_rgb[valid_points] = (
                    image[pc_in_image[valid_points, 1], pc_in_image[valid_points, 0]]
                ).float()
            point_clouds_rgb.append(point_cloud_rgb)

        self.train_dataparser_outputs.metadata["point_clouds_rgb"] = point_clouds_rgb

    def get_num_train_data(self) -> int:
        return len(self.train_dataset) + len(self.train_lidar_dataset)
