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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.models.ad_model import ADModel
from nerfstudio.pipelines.ad_pipeline import ADPipeline
from nerfstudio.scripts.closed_loop.models import TrajectoryDict
from nerfstudio.scripts.render import streamline_ad_config
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.poses import to4x4


@dataclass
class ClosedLoopServer:
    """Configuration for closed loop renderer instantiation."""

    load_config: Optional[Path] = None
    """Path to config YAML file."""

    port: int = 8000
    """Port to run the server on."""
    host: str = "0.0.0.0"
    """Host to run the server on."""

    pad_cameras: bool = True
    """Whether to require all cameras to have the same size."""
    adjust_pose: bool = False
    """Whether to adjust the height of rendered cameras to match the nearest training camera."""

    @torch.no_grad()
    def main(self):
        assert self.load_config is not None, "Must specify a config file to load"
        _, self.pipeline, _, _ = eval_setup(
            self.load_config,
            test_mode="inference",
            update_config_callback=streamline_ad_config,
            strict_load=False,
        )
        assert isinstance(self.pipeline, ADPipeline)
        dataparser_outputs: DataparserOutputs = self.pipeline.datamanager.eval_dataparser_outputs
        self.model: ADModel = self.pipeline.model
        assert isinstance(self.model, ADModel), "Closed loop server only works with an ADModel"
        self.model.eval()

        assert self.pipeline.datamanager.eval_dataset is not None
        self._train_cams = self.pipeline.datamanager.train_dataset.cameras
        self._unique_cameras = _find_unique_cameras(self.pipeline.datamanager.eval_dataset.cameras)
        if self.pad_cameras:
            self._unique_cameras.height[:] = self._unique_cameras.height.max()
            self._unique_cameras.width[:] = self._unique_cameras.width.max()

        self.min_time = dataparser_outputs.time_offset
        self.world_transform = to4x4(dataparser_outputs.dataparser_transform).to(self.model.device)
        self.actor_transform = to4x4(dataparser_outputs.actor_transform).to(self.model.device)

        sensor_idx_to_name: Dict[int, str] = dataparser_outputs.metadata["sensor_idx_to_name"]
        self.sensor_name_to_idx = {v: k for k, v in sensor_idx_to_name.items()}
        assert len(self.sensor_name_to_idx) == len(sensor_idx_to_name), "Sensor names must be unique"
        self.actor_uuids = [traj["uuid"] for traj in dataparser_outputs.metadata["trajectories"]]
        self.actor_uuid_to_idx = {uuid: idx for idx, uuid in enumerate(self.actor_uuids)}
        assert len(self.actor_uuid_to_idx) == len(self.actor_uuids), "Actor UUIDs must be unique"

    @torch.no_grad()
    def get_image(self, pose: Tensor, timestamp: int, camera_name: str) -> Tensor:
        """Render an image for the given pose and time.

        Args:
            pose: 4x4 camera pose matrix
            timestamp: Timestamp in microseconds
            camera_name: Camera name

        """
        camera_name = camera_name.lstrip("CAM_")
        cam_idx = self.sensor_name_to_idx[camera_name]
        cam = self._unique_cameras[cam_idx : cam_idx + 1].to(self.model.device)

        pose = self.world_transform.cpu() @ pose
        pose[:3, :3] = pose[:3, :3] @ OPENCV_TO_NERFSTUDIO

        cam.camera_to_worlds = pose[:3, :4].unsqueeze(0).to(self.model.device)
        assert cam.times is not None
        cam.times[...] = (timestamp / 1e6) - self.min_time
        ray_bundle = cam.generate_rays(camera_indices=0)  # cam is already the desired camera

        if self.adjust_pose:
            pos = self._train_cams.camera_to_worlds[:, :3, 3]
            mask = (self._train_cams.metadata["sensor_idxs"] == cam_idx).squeeze(-1)
            distances = (pose[None, :3, 3] - pos).norm(dim=-1).squeeze(-1)
            nearest_train_cam_idx = torch.argmin(
                torch.where(mask, distances, torch.tensor(float("inf"), device=distances.device))
            )
            correction_matrices = self.model.camera_optimizer(
                torch.tensor([nearest_train_cam_idx], device=self.model.device)
            )
            ray_bundle.origins = ray_bundle.origins + correction_matrices[:, :3, 3]
            ray_bundle.directions = torch.einsum("ij,...j->...i", correction_matrices[0, :3, :3], ray_bundle.directions)

        return self.model.get_outputs_for_camera_ray_bundle(ray_bundle)["rgb"]

    @torch.no_grad()
    def update_actor_trajectories(self, new_trajectories: list[TrajectoryDict]):
        """Update actor trajectories."""
        device = self.model.device
        modified_trajectories = []
        actor_ids = []
        actor_transform = self.actor_transform.to(device)
        for traj in new_trajectories:
            timestamps_in_seconds = traj["timestamps"].to(torch.float64) / 1e6
            modified_trajectories.append(
                {
                    "poses": self.world_transform @ traj["poses"].to(device) @ actor_transform,
                    "timestamps": (timestamps_in_seconds - self.min_time).to(device),
                    "dims": traj["dims"].to(device),
                    "symmetric": True,  # TODO
                    "deformable": False,  # TODO
                    "uuid": traj["uuid"],
                }
            )
            actor_ids.append(self.actor_uuid_to_idx[traj["uuid"]])
        old_n_actors = self.model.dynamic_actors.n_actors
        self.model.dynamic_actors._populate_actors(modified_trajectories)
        self.model.dynamic_actors.actor_to_id[:] = torch.tensor(actor_ids, device=device)
        self.model.dynamic_actors.n_actors = old_n_actors  # used in hash encoding
        self.model.dynamic_actors.to(device)  # need to move again as some tensors are created inside _populate_actors

        self.actor_uuids = [traj["uuid"] for traj in new_trajectories]

    @torch.no_grad()
    def get_actor_trajectories(self) -> list[TrajectoryDict]:
        """Get actor trajectories."""
        poses_3x4 = self.model.dynamic_actors.get_poses_3x4()
        poses = to4x4(poses_3x4)
        world_inverse = self.world_transform.inverse().view(1, 4, 4)
        timestamps = (self.model.dynamic_actors.unique_timestamps.to(torch.float64) + self.min_time) * 1e6
        actor_transform_inv = self.actor_transform.inverse()
        trajs = []
        for actor_idx in range(poses.shape[1]):
            trajs.append(
                {
                    "poses": (world_inverse @ poses[:, actor_idx]).cpu() @ actor_transform_inv,
                    "timestamps": timestamps.cpu().long(),
                    "dims": self.model.dynamic_actors.actor_sizes[actor_idx].cpu(),
                    "uuid": self.actor_uuids[actor_idx],
                }
            )
        return trajs


class DummyClosedLoopServer(ClosedLoopServer):
    """Dummy closed loop server for testing."""

    def main(self):
        self.min_time = 0.0

    def get_image(self, pose: Tensor, timestamp: int, camera_name: str) -> Tensor:
        return torch.zeros((512, 512, 3), dtype=torch.uint8)

    def update_actor_trajectories(self, new_trajectories: list[TrajectoryDict]):
        print("Updated actor trajectories")

    def get_actor_trajectories(self) -> list[TrajectoryDict]:
        return [
            {
                "poses": torch.empty((0, 4, 4)),
                "timestamps": torch.empty(0),
                "dims": torch.empty((0, 3)),
                "uuid": "uuid1",
            }
        ]


def _find_unique_cameras(cameras: Cameras):
    assert "sensor_idxs" in cameras.metadata, "Cameras must have sensor_idx metadata"
    sensor_idxs = cameras.metadata["sensor_idxs"]
    print(f"Found {len(sensor_idxs)} cameras, with unique indices: {torch.unique(sensor_idxs).tolist()}")
    # need to go through numpy as torch.unique doesn't support return_index
    first_unique_idx = torch.from_numpy(np.unique(sensor_idxs.numpy(), return_index=True)[1])
    return cameras[first_unique_idx]
