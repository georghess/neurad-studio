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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type, Union

import numpy
import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.utils import poses as pose_utils


@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    trans_l2_penalty: Union[Tuple, float] = 1e-2
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""

    # tyro.conf.Suppress prevents us from creating CLI arguments for these fields.
    optimizer: tyro.conf.Suppress[Optional[OptimizerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    scheduler: tyro.conf.Suppress[Optional[SchedulerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    def __post_init__(self):
        if self.optimizer is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\noptimizer is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)

        if self.scheduler is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\nscheduler is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


@dataclass
class CameraVelocityOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera velocities."""

    _target: Type = field(default_factory=lambda: CameraVelocityOptimizer)

    enabled: bool = False
    """Optimize velocities"""

    zero_initial_velocities: bool = False
    """Do not use initial velocities in cameras as a starting point"""

    linear_l2_penalty: float = 1e-6
    """L2 penalty on linear velocity"""

    angular_l2_penalty: float = 1e-5
    """L2 penalty on angular velocity"""


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        else:
            assert_never(self.config.mode)

    def _get_pose_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the pose adjustment."""
        return self.pose_adjustment

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
    ) -> Float[Tensor, "camera_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self._get_pose_adjustment()[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self._get_pose_adjustment()[indices, :]))
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            if self.non_trainable_camera_indices.device != self.pose_adjustment.device:
                self.non_trainable_camera_indices = self.non_trainable_camera_indices.to(self.pose_adjustment.device)
            outputs[0][self.non_trainable_camera_indices] = torch.eye(4, device=self.pose_adjustment.device)[:3, :4]

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)

    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            correction_matrices = self(raybundle.camera_indices.squeeze())  # type: ignore
            raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
            raybundle.directions = (
                torch.bmm(correction_matrices[:, :3, :3], raybundle.directions[..., None])
                .squeeze()
                .to(raybundle.origins)
            )

    def apply_to_camera(self, camera: Union[Cameras, Lidars]) -> torch.Tensor:
        """Apply the pose correction to the world-to-camera matrix in a Camera object"""
        sensor_to_world = camera.camera_to_worlds if isinstance(camera, Cameras) else camera.lidar_to_worlds
        if self.config.mode == "off":
            return sensor_to_world

        if camera.metadata is None or "cam_idx" not in camera.metadata:
            # Viser cameras
            return sensor_to_world

        camera_idx = camera.metadata["cam_idx"]
        adj = self(torch.tensor([camera_idx], dtype=torch.long, device=camera.device))  # type: ignore

        return torch.cat(
            [
                # Apply rotation to directions in world coordinates, without touching the origin.
                # Equivalent to: directions -> correction[:3,:3] @ directions
                torch.bmm(adj[..., :3, :3], sensor_to_world[..., :3, :3]),
                # Apply translation in world coordinate, independently of rotation.
                # Equivalent to: origins -> origins + correction[:3,3]
                sensor_to_world[..., :3, 3:] + adj[..., :3, 3:],
            ],
            dim=-1,
        )

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            pose_adjustment = self._get_pose_adjustment()
            loss_dict["camera_opt_regularizer"] = (
                pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.trans_l2_penalty
                + pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
            )

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            trans = self.pose_adjustment[:, :3].detach().norm(dim=-1)
            rot = self.pose_adjustment[:, 3:].detach().norm(dim=-1)
            metrics_dict["camera_opt_translation_max"] = trans.max()
            metrics_dict["camera_opt_translation_mean"] = trans.mean()
            metrics_dict["camera_opt_rotation_mean"] = numpy.rad2deg(rot.mean().cpu())
            metrics_dict["camera_opt_rotation_max"] = numpy.rad2deg(rot.max().cpu())

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0


class CameraVelocityOptimizer(nn.Module):
    """Layer that modifies camera velocities during training."""

    config: CameraVelocityOptimizerConfig

    def __init__(
        self,
        config: CameraVelocityOptimizerConfig,
        num_cameras: int,
        num_unique_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.num_unique_cameras = num_unique_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.enabled:
            self.linear_velocity_adjustment = torch.nn.Parameter(
                ((torch.rand((num_cameras, 3), device=device) - 0.5) * 1e-1)
            )
            self.angular_velocity_adjustment = torch.nn.Parameter(
                ((torch.rand((num_cameras, 3), device=device) - 0.5) * 1e-4)
            )
            self.time_to_center_pixel_adjustment = torch.nn.Parameter(
                ((torch.rand((num_unique_cameras), device=device) - 0.5) * 1e-6)
            )

    def get_time_to_center_pixel_adjustment(self, camera: Union[Cameras, Lidars]) -> Float[Tensor, "num_cameras 1"]:
        """Get the time to center pixel adjustment."""
        sensor_idx = camera.metadata["sensor_idxs"].view(-1)
        if self.config.enabled:
            return self.time_to_center_pixel_adjustment[sensor_idx]
        return torch.zeros_like(sensor_idx, device=camera.device)

    def apply_to_camera_velocity(self, camera: Union[Cameras, Lidars], return_init_only) -> torch.Tensor:
        init_velocities = None
        sensor_to_world = camera.camera_to_worlds if isinstance(camera, Cameras) else camera.lidar_to_worlds
        if self.config.zero_initial_velocities:
            init_velocities = torch.zeros((len(camera), 6), device=sensor_to_world.device)
        else:
            assert camera.metadata["linear_velocities_local"] is not None
            init_velocities = torch.hstack(
                [camera.metadata["linear_velocities_local"], camera.metadata["angular_velocities_local"]]
            )

        if not self.config.enabled or return_init_only:  # or not self.training:
            return init_velocities

        if camera.metadata is None or "cam_idx" not in camera.metadata:
            return init_velocities

        cam_idx = camera.metadata["cam_idx"]
        adj = torch.cat([self.linear_velocity_adjustment[cam_idx, :], self.angular_velocity_adjustment[cam_idx, :]])[
            None
        ]
        return init_velocities + adj

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.enabled:
            loss_dict["camera_velocity_regularizer"] = (
                self.linear_velocity_adjustment.norm(dim=-1).mean() * self.config.linear_l2_penalty
                + self.angular_velocity_adjustment.norm(dim=-1).mean() * self.config.angular_l2_penalty
            )

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera velocity optimizer metrics"""
        if self.config.enabled:
            lin = self.linear_velocity_adjustment.detach().norm(dim=-1)
            ang = self.angular_velocity_adjustment.detach().norm(dim=-1)
            metrics_dict["camera_opt_vel_max"] = lin.max()
            metrics_dict["camera_opt_vel_mean"] = lin.mean()
            metrics_dict["camera_opt_ang_vel_max"] = ang.max()
            metrics_dict["camera_opt_ang_vel_mean"] = ang.mean()
            for i in range(self.num_unique_cameras):
                metrics_dict[f"camera_opt_ttc_pixel_adjustment_{i}"] = self.time_to_center_pixel_adjustment[i].detach()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        vel_opt_params = list(self.parameters())
        if self.config.enabled:
            assert len(vel_opt_params) > 0
            param_groups["camera_velocity_opt_linear"] = vel_opt_params[0:1]
            param_groups["camera_velocity_opt_angular"] = vel_opt_params[1:2]
            param_groups["camera_velocity_opt_time_to_center_pixel"] = vel_opt_params[2:3]
        else:
            assert len(vel_opt_params) == 0


@dataclass
class ScaledCameraOptimizerConfig(CameraOptimizerConfig):
    """Configuration of axis-masked optimization for camera poses."""

    _target: Type = field(default_factory=lambda: ScaledCameraOptimizer)

    weights: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )

    trans_l2_penalty: Union[Tuple[float, float, float], float] = (
        1e-2,
        1e-2,
        1e-2,
    )  # TODO: this is l1


class ScaledCameraOptimizer(CameraOptimizer):
    """Camera optimizer that masks which components can be optimized."""

    def __init__(self, config: ScaledCameraOptimizerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.config: ScaledCameraOptimizerConfig = self.config
        self.register_buffer("weights", torch.tensor(self.config.weights, dtype=torch.float32))
        self.trans_penalty = torch.tensor(self.config.trans_l2_penalty, dtype=torch.float32, device=self.device)

    def _get_pose_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the pose adjustment."""
        return self.pose_adjustment * self.weights

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            pose_adjustment = self._get_pose_adjustment()
            self.trans_penalty = self.trans_penalty.to(pose_adjustment.device)
            loss_dict["camera_opt_regularizer"] = (
                pose_adjustment[:, :3].abs() * self.trans_penalty
            ).mean() + pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
