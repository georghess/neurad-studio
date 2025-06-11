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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.lidars import Lidars
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.dynamic_actors import DynamicActors, DynamicActorsConfig
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.poses import inverse as pose_inverse


@dataclass
class ADModelConfig(ModelConfig):
    """Base model config for all AD models."""

    _target: Type = field(default_factory=lambda: ADModel)

    rgb_upsample_factor: int = 1
    """Upsample factor for RGB images. For vanilla NeRFs this is 1 (1 pixel per ray)."""

    dynamic_actors: DynamicActorsConfig = field(default_factory=DynamicActorsConfig)
    """Dynamic actors configuration."""

    camera_optimizer: CameraOptimizerConfig = field(
        default_factory=lambda: CameraOptimizerConfig(mode="off", trans_l2_penalty=0.1)
    )
    """Config of the camera optimizer to use"""

    use_camopt_in_eval: bool = False
    """Use result of camera optimization also during evaluation. Only makes sense if trained with train_eval_split=1.0."""


class ADModel(Model):
    """Base model for all AD models."""

    config: ADModelConfig

    def forward(
        self, ray_bundle: RayBundle, patch_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Forward pass of the model."""
        outputs = super().forward(ray_bundle)
        if patch_size is not None and "rgb" in outputs:
            if isinstance(outputs["rgb"], torch.Tensor):
                # if patch size is given, we must reshape the output to respect it
                outputs["rgb"] = outputs["rgb"].reshape(-1, *patch_size, 3)
        return outputs

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        trajectories = self.kwargs["metadata"].get("trajectories")
        self.dynamic_actors: DynamicActors = self.config.dynamic_actors.setup(trajectories=trajectories)
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Get the parameter groups for the optimizer."""
        param_groups = defaultdict(list)
        self.camera_optimizer.get_param_groups(param_groups)
        self.dynamic_actors.get_param_groups(param_groups)
        return param_groups

    def disable_ray_drop(self):
        """Disables ray drop for the model."""
        pass

    @torch.no_grad()
    def get_outputs_for_lidar(
        self, lidar: Lidars, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Takes in a lidar, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        points = batch["lidar"]
        assert isinstance(batch["lidar_idx"], int), "All lidar points are assumed to be from the same scan."
        lidar_indices = torch.zeros_like(points[:, 0:1]).long()
        ray_bundle = lidar.generate_rays(lidar_indices=lidar_indices, points=points, keep_shape=True)
        ray_bundle.camera_indices = (torch.ones_like(points[:, 0:1]) * batch["lidar_idx"]).long()
        # TODO: Can we avoid needing to pass this from the raybundle to the batch?
        batch["is_lidar"] = ray_bundle.metadata["is_lidar"]
        batch["distance"] = ray_bundle.metadata["directions_norm"]
        batch["did_return"] = ray_bundle.metadata["did_return"]

        outputs = self.get_outputs_for_camera_ray_bundle(ray_bundle)

        # add points in local coords to model outputs
        l2w = lidar.lidar_to_worlds[0].to(self.device)
        w2l = pose_inverse(l2w)
        points = ray_bundle.origins + ray_bundle.directions * outputs["depth"]
        outputs["points"] = (w2l @ torch.cat([points, torch.ones_like(points[..., :1])], dim=-1).unsqueeze(-1)).squeeze(
            -1
        )
        return outputs, batch

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        if len(camera_ray_bundle.shape) == 1:  # lidar
            return self.forward(ray_bundle=camera_ray_bundle)  # type: ignore
        else:  # camera
            return super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
