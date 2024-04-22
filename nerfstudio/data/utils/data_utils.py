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

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.lidars import transform_points


def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.Resampling.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype=torch.int64).view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.Resampling.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])


def points_in_box(points, box2world, size):
    """Return a mask of points that are inside a box"""
    # points is [N, 3]

    mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

    # transform points to box frame
    world2box = torch.inverse(box2world)
    points = transform_points(points, world2box)

    # check if points are in box
    mask = mask & (points[:, 0] > -size[0] / 2) & (points[:, 0] < size[0] / 2)
    mask = mask & (points[:, 1] > -size[1] / 2) & (points[:, 1] < size[1] / 2)
    mask = mask & (points[:, 2] > -size[2] / 2) & (points[:, 2] < size[2] / 2)

    return mask


def remove_dynamic_points(point_clouds, l2w, timestamps, trajectories, extra_padding=0.15):
    """Remove dynamic points from point clouds.

    point_clouds - List of point clouds, assumed to be in sensor coordinates.
    l2w - List of sensor-to-world transforms.
    trajectories - List of object trajectories, assumed to be in in world coordinates.
    extra_padding - Extra padding around the object bounding box.

    Returns:
        List of pruned point clouds, in sensor coordinates.
    """
    pruned_point_clouds = []

    dynamic_trajs = [traj for traj in trajectories if not traj["stationary"]]

    for i, pc in enumerate(point_clouds):
        points_in_box_mask = torch.zeros(pc.shape[0], dtype=torch.bool, device=pc.device)
        pc_in_world = transform_points(pc[:, :3], l2w[i])
        for traj in dynamic_trajs:
            time_index = ((traj["timestamps"] - timestamps[i]).abs() < 1e-3).nonzero()
            if len(time_index) == 0:  # no matching timestamp
                continue
            time_index = time_index[0]
            pose = traj["poses"][time_index]
            size = traj["dims"] * (1 + extra_padding)
            points_in_box_mask = points_in_box_mask | points_in_box(pc_in_world, pose, size)

        pruned_point_clouds.append(pc[~points_in_box_mask])

    return pruned_point_clouds
