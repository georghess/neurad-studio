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
Nerfacto augmented with depth supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.losses import DepthLossType, depth_loss
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.ad_model import ADModel, ADModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig


@dataclass
class LidarNerfactoModelConfig(NerfactoModelConfig, ADModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: LidarNerfactoModel)
    depth_loss_mult: float = 1e-3
    """Lambda of the depth loss."""
    is_euclidean_depth: bool = True
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.01
    """Uncertainty around depth values in meters (defaults to 1cm)."""
    should_decay_sigma: bool = True
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""
    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    """Depth loss type."""
    use_gradient_scaling: bool = True
    """Use gradient scaler where the gradients are lower for points closer to the camera."""


class LidarNerfactoModel(NerfactoModel, ADModel):
    """Depth loss augmented nerfacto model.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: LidarNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

        self.renderer_depth = DepthRenderer(method="expected")

        self.median_l2 = lambda pred, gt: torch.median((pred - gt) ** 2)

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        if "is_lidar" in ray_bundle.metadata:
            outputs["rgb"] = outputs["rgb"][~ray_bundle.metadata["is_lidar"][:, 0]]
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training and "lidar" in batch:
            metrics_dict["depth_loss"] = 0.0
            is_lidar = batch["is_lidar"][:, 0]
            sigma = self._get_sigma().to(self.device)
            pred_depths = [outputs[f"prop_depth_{i}"] for i in range(len(outputs["weights_list"]) - 1)]
            pred_depths.append(outputs["depth"])
            lidar_scale = batch.get("lidar_scale", [1.0])[0]
            for i in range(len(outputs["weights_list"])):
                metrics_dict["depth_loss"] += depth_loss(
                    weights=outputs["weights_list"][i][is_lidar],
                    ray_samples=outputs["ray_samples_list"][i][is_lidar],
                    termination_depth=batch["distance"],
                    predicted_depth=pred_depths[i][is_lidar],
                    sigma=sigma * lidar_scale,
                    directions_norm=outputs["directions_norm"][is_lidar],
                    is_euclidean=self.config.is_euclidean_depth,
                    depth_loss_type=self.config.depth_loss_type,
                    scaling_factor=lidar_scale,
                ) / len(outputs["weights_list"])

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training:
            assert metrics_dict
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics, images = {}, {}
        if "image" in batch:
            metrics, images = super().get_image_metrics_and_images(outputs, batch)
        if "lidar" in batch:
            metrics["depth_median_l2"] = float(self.median_l2(outputs["depth"], batch["distance"]))
        return metrics, images

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma

        self.depth_sigma = torch.maximum(  # pylint: disable=attribute-defined-outside-init
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma
