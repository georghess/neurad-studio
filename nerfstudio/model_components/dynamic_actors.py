# Copyright 2025 the authors of NeuRAD and contributors.
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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.camera_utils import matrix_to_rotation_6d, rotation_6d_to_matrix
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils import poses as pose_utils
from nerfstudio.utils.poses import interpolate_trajectories_6d, interpolate_velocities
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import ViewerSlider


@dataclass
class DynamicActorsConfig(InstantiateConfig):
    """Configuration of dynamic actors."""

    _target: Type = field(default_factory=lambda: DynamicActors)

    optimize_trajectories: bool = True
    """Whether to optimize the trajectories or not."""
    actor_bbox_padding: Tuple[float, float, float] = (0.25, 0.25, 0.1)
    """Padding to add to the bounding boxes of the actors (wlh order, in meters)."""


class DynamicActors(nn.Module):
    config: DynamicActorsConfig

    def __init__(self, config: DynamicActorsConfig, trajectories: List[dict]):
        super().__init__()
        self.config = config

        self._populate_actors(trajectories)
        self.requires_grad_(self.config.optimize_trajectories)

        self.actor_editing = {
            "lateral": 0.0,
            "longitudinal": 0.0,
            "rotation": 0.0,
            "index": -1.0,
            "height": 0.0,
        }
        self.actor_lateral_shift = ViewerSlider(
            name="Actor lateral shift (m)",
            default_value=self.actor_editing["lateral"],
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            cb_hook=lambda obj: self.actor_editing.update({"lateral": obj.value}),
        )

        self.actor_longitudinal_shift = ViewerSlider(
            name="Actor longitudinal shift (m)",
            default_value=self.actor_editing["longitudinal"],
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            cb_hook=lambda obj: self.actor_editing.update({"longitudinal": obj.value}),
        )

        self.actor_rotation_shift = ViewerSlider(
            name="Actor rotation shift (m)",
            default_value=self.actor_editing["rotation"],
            min_value=-3.14,
            max_value=3.14,
            step=0.1,
            cb_hook=lambda obj: self.actor_editing.update({"rotation": obj.value}),
        )

        self.actor_index_to_edit = ViewerSlider(
            name="Actor index to edit",
            default_value=self.actor_editing["index"],
            min_value=-1.0,
            max_value=len(trajectories),
            step=1.0,
            cb_hook=lambda obj: self.actor_editing.update({"index": obj.value}),
        )
        self._register_load_state_dict_pre_hook(self._overwrite_shapes_hook, with_module=False)

        self.actor_height_shift = ViewerSlider(
            name="Actor height shift (m)",
            default_value=self.actor_editing["height"],
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            cb_hook=lambda obj: self.actor_editing.update({"height": obj.value}),
        )

    def actor_bounds(self):
        return self.actor_sizes / 2 + self.actor_padding

    def _populate_actors(self, trajectories: List[dict]) -> None:
        unique_timestamps = torch.tensor(sorted(list({t.item() for traj in trajectories for t in traj["timestamps"]})))
        self.n_actors = len(trajectories)
        self.n_times = len(unique_timestamps)

        actor_poses_at_time = (
            torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(self.n_times, self.n_actors, 1, 1)
        )
        actor_present_at_time = torch.zeros((self.n_times, self.n_actors), dtype=torch.bool)
        actor_sizes = torch.zeros((self.n_actors, 3), dtype=torch.float32)
        actor_symmetric = torch.zeros((self.n_actors,), dtype=torch.bool)
        actor_deformable = torch.zeros((self.n_actors,), dtype=torch.bool)
        actor_vel_linear = torch.zeros((self.n_times, self.n_actors, 3), dtype=torch.float32)
        actor_vel_angular = torch.zeros((self.n_times, self.n_actors, 3), dtype=torch.float32)

        for actor_index, traj in enumerate(trajectories):
            actor_sizes[actor_index] = traj["dims"]
            actor_symmetric[actor_index] = traj["symmetric"]
            actor_deformable[actor_index] = traj["deformable"]

            for time_index, t in enumerate(unique_timestamps):
                time_diff = (traj["timestamps"] - t).abs()
                traj_time_index = time_diff.argmin(dim=0)
                if time_diff[traj_time_index] < 1e-4:
                    actor_present_at_time[time_index, actor_index] = True
                    actor_poses_at_time[time_index, actor_index] = traj["poses"][traj_time_index]
                    if "linear_velocities_global" in traj and "angular_velocities_local" in traj:
                        actor_vel_linear[time_index, actor_index] = traj["linear_velocities_global"][
                            traj_time_index, :3
                        ]
                        actor_vel_angular[time_index, actor_index] = traj["angular_velocities_local"][
                            traj_time_index, :3
                        ]
                else:
                    # TODO(carlinds): Is this hack needed anymore? Don't we already do the interpolation for all
                    # timesteps in the dataparser?
                    # TODO: This is a hack to make sure that all timestamps are present. This is needed because the
                    #  trajectory optimizer assumes that all timestamps are present.
                    # we duplicate the closest timestamp
                    actor_poses_at_time[time_index, actor_index] = traj["poses"][traj_time_index]
            assert actor_present_at_time[:, actor_index].sum() == len(
                traj["timestamps"].unique()
            ), "Failed to populate all timestamps for actor!"

        self.register_buffer("unique_timestamps", unique_timestamps)
        self.register_buffer("actor_poses_at_time", actor_poses_at_time)
        self.register_buffer("actor_present_at_time", actor_present_at_time)
        self.register_buffer("actor_sizes", actor_sizes)  # wlh
        self.register_buffer("actor_symmetric", actor_symmetric)
        self.register_buffer("actor_deformable", actor_deformable)
        self.register_buffer("actor_padding", torch.tensor(self.config.actor_bbox_padding))
        # this seems silly but allows us to duplicate actors during rendering
        self.register_buffer("actor_to_id", torch.arange(self.n_actors, dtype=torch.int64))

        # optimizable parameters
        self.actor_positions = nn.Parameter(self.actor_poses_at_time[..., :3, 3])
        self.actor_rotations_6d = nn.Parameter(matrix_to_rotation_6d(self.actor_poses_at_time[..., :3, :3]))
        self.register_buffer("initial_positions", self.actor_positions.clone())
        self.register_buffer("initial_rotations_6d", self.actor_rotations_6d.clone())

        self.actor_vel_linear = nn.Parameter(actor_vel_linear)
        self.actor_vel_angular = nn.Parameter(actor_vel_angular)

    def get_poses_3x4(self):
        rotations = rotation_6d_to_matrix(self.actor_rotations_6d)
        return torch.cat([rotations, self.actor_positions.unsqueeze(-1)], dim=-1)

    def get_world2boxes(self, query_times: Tensor, flatten: bool = True):
        boxes2world, *extra = self.get_boxes2world(query_times, flatten)
        world2boxes = pose_utils.inverse(boxes2world)
        return world2boxes, *extra

    def edit_boxes2world(self, boxes2world: Tensor, flattened_actor_indices: Optional[Tensor] = None):
        if (
            abs(self.actor_editing["longitudinal"]) == 0.0
            and abs(self.actor_editing["lateral"]) == 0.0
            and abs(self.actor_editing["rotation"]) == 0.0
        ):
            return boxes2world
        with torch.no_grad():
            if self.actor_editing["index"] == -1.0:
                indices = torch.arange(self.n_actors, device=boxes2world.device)
            else:
                index = min(self.actor_editing["index"], self.n_actors - 1)
                indices = torch.tensor([index], device=boxes2world.device, dtype=torch.int)

            if flattened_actor_indices is not None:
                actor_should_be_edited = torch.isin(flattened_actor_indices, indices)
            else:
                actor_should_be_edited = torch.ones_like(indices, dtype=torch.bool)

            if (
                abs(self.actor_editing["longitudinal"]) > 0.0
                or abs(self.actor_editing["lateral"]) > 0.0
                or abs(self.actor_editing["height"]) > 0.0
            ):
                if flattened_actor_indices is not None:
                    boxes2world[actor_should_be_edited, :, 3] = boxes2world[actor_should_be_edited] @ torch.tensor(
                        [
                            self.actor_editing["lateral"],
                            self.actor_editing["longitudinal"],
                            self.actor_editing["height"],
                            1.0,
                        ],
                        device=boxes2world.device,
                    )
                else:
                    boxes2world[:, indices, :, 3] = boxes2world[:, indices] @ torch.tensor(
                        [
                            self.actor_editing["lateral"],
                            self.actor_editing["longitudinal"],
                            self.actor_editing["height"],
                            1.0,
                        ],
                        device=boxes2world.device,
                    )

            if abs(self.actor_editing["rotation"]) > 0.0:
                if flattened_actor_indices is not None:
                    rotation_yaw = torch.tensor(
                        [
                            [math.cos(self.actor_editing["rotation"]), -math.sin(self.actor_editing["rotation"]), 0.0],
                            [math.sin(self.actor_editing["rotation"]), math.cos(self.actor_editing["rotation"]), 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        device=boxes2world.device,
                    )
                    boxes2world[actor_should_be_edited, :3, :3] = (
                        rotation_yaw @ boxes2world[actor_should_be_edited, :3, :3]
                    )
                else:
                    rotation_yaw = torch.tensor(
                        [
                            [math.cos(self.actor_editing["rotation"]), -math.sin(self.actor_editing["rotation"]), 0.0],
                            [math.sin(self.actor_editing["rotation"]), math.cos(self.actor_editing["rotation"]), 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        device=boxes2world.device,
                    )
                    boxes2world[:, indices, :3, :3] = rotation_yaw @ boxes2world[:, indices, :3, :3]
        return boxes2world

    def get_boxes2world(self, query_times: Tensor, flatten: bool = True):
        poses, *extra = interpolate_trajectories_6d(
            torch.cat([self.actor_rotations_6d, self.actor_positions], dim=-1),
            self.unique_timestamps,
            query_times,
            self.actor_present_at_time,
            flatten=flatten,
        )
        boxes2world = torch.cat([rotation_6d_to_matrix(poses[..., :6]), poses[..., 6:].unsqueeze(-1)], dim=-1)

        if not self.training:
            if flatten:
                boxes2world = self.edit_boxes2world(boxes2world, extra[1])
            else:
                boxes2world = self.edit_boxes2world(boxes2world)

        boxes2world = pose_utils.to4x4(boxes2world)
        return boxes2world, *extra

    def get_velocities(self, query_times: Tensor):
        return interpolate_velocities(
            torch.cat([self.actor_vel_linear, self.actor_vel_angular], dim=-1),
            self.unique_timestamps,
            query_times,
        )

    def requires_grad_(self, requires: bool) -> None:
        self.actor_positions.requires_grad_(requires)
        self.actor_rotations_6d.requires_grad_(requires)

    def get_param_groups(self, param_groups: Dict):
        if self.config.optimize_trajectories:
            param_groups["trajectory_opt"] = param_groups.get("trajectory_opt", []) + list(self.parameters())

    def _overwrite_shapes_hook(self, state_dict, prefix, *_, **__):
        state = self.state_dict()
        print_warning = False

        for name, param in state_dict.items():
            name = name.removeprefix(prefix)
            if name in state and state[name].shape != param.shape:
                param = param.to(state[name].device)
                if name in self._parameters:
                    self.register_parameter(name, nn.Parameter(param, requires_grad=state[name].requires_grad))
                else:
                    self.register_buffer(name, param)
                print_warning = True

        if print_warning:
            CONSOLE.print("Warning: found shape mismatches in dynamic actors, overwriting", style="bold red")
