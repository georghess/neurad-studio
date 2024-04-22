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
Ray generator.
"""
from jaxtyping import Float, Int
from torch import Tensor, nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars
from nerfstudio.cameras.rays import RayBundle


class RayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class is the interface between the scene's cameras/camera optimizer and the ray sampler.

    Args:
        cameras: Camera objects containing camera info.
    """

    image_coords: Tensor

    def __init__(self, cameras: Cameras) -> None:
        super().__init__()
        self.cameras = cameras
        self.register_buffer("image_coords", cameras.get_image_coords(), persistent=False)

    def forward(self, ray_indices: Int[Tensor, "num_rays 3"]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            ray_indices: Contains camera, row, and col indices for target rays.
        """
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        coords = self.image_coords[y, x]

        ray_bundle = self.cameras.generate_rays(
            camera_indices=c.unsqueeze(-1),
            coords=coords,
        )
        return ray_bundle


class LidarRayGenerator(nn.Module):
    """torch.nn Module for generating rays for Lidars.
    This class is the interface between the scene's lidars/lidar optimizer and the ray sampler.

    Args:
        lidars: Lidar objects containing lidar info.
        pose_optimizer: pose optimization module, for optimizing noisy lidar intrinsics/extrinsics.
    """

    def __init__(self, lidars: Lidars) -> None:
        super().__init__()
        self.lidars = lidars

    def forward(self, ray_indices: Int[Tensor, "num_rays 2"], points: Float[Tensor, "num_rays 3"] = None) -> RayBundle:
        """Index into the lidars to generate the rays.

        Args:
            ray_indices: Contains (lidar, point) indices for target rays.
        """
        c = ray_indices[:, 0]  # lidar (scan) indices
        n = ray_indices[:, 1]  # point indices

        if points is not None:
            ray_bundle = self.lidars.generate_rays(
                lidar_indices=c.unsqueeze(-1),
                points=points,
            )
        else:
            ray_bundle = self.lidars.generate_rays(
                lidar_indices=c.unsqueeze(-1),
                coords=n,
            )

        return ray_bundle
