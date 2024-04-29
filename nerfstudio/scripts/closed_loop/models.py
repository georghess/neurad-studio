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

from enum import Enum
from typing import List, TypedDict

import torch
from pydantic import BaseModel
from torch import Tensor


class ImageFormat(str, Enum):
    raw = "raw"  # will return a raw tensor, works good when sending across same machine
    png = "png"  # more suitable if sent over network
    jpg = "jpg"  # more suitable if sent over network, pseudo for jpeg
    jpeg = "jpeg"  # more suitable if sent over network


class TrajectoryDict(TypedDict):
    uuid: str
    poses: Tensor
    timestamps: Tensor
    dims: Tensor


class ActorTrajectory(BaseModel):
    """Trajectory of an actor."""

    uuid: str
    """Actor UUID."""
    poses: List[List[List[float]]]
    """List of 4x4 actor poses."""
    timestamps: List[int]
    """List of timestamps in microseconds."""
    dims: List[float]
    """Dimensions of the actor."""

    def to_torch(self) -> TrajectoryDict:
        return {
            "uuid": self.uuid,
            "poses": torch.tensor(self.poses, dtype=torch.float32),
            "timestamps": torch.tensor(self.timestamps, dtype=torch.int64),
            "dims": torch.tensor(self.dims, dtype=torch.float32),
        }

    @classmethod
    def from_torch(cls, data: TrajectoryDict) -> ActorTrajectory:
        return cls(
            uuid=data["uuid"],
            poses=data["poses"].tolist(),
            timestamps=data["timestamps"].tolist(),
            dims=data["dims"].tolist(),
        )


class RenderInput(BaseModel):
    """Input to the render_image endpoint."""

    pose: List[List[float]]
    """4x4 camera pose matrix"""
    timestamp: int
    """Timestamp in microseconds"""
    camera_name: str
    """Camera name"""
    image_format: ImageFormat = ImageFormat.raw
    """What format to return the image in. Defaults to raw tensor."""
