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

import io

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, Response
from PIL import Image
from torch import Tensor

from nerfstudio.scripts.closed_loop.models import ActorTrajectory, RenderInput
from nerfstudio.scripts.closed_loop.server import ClosedLoopServer

app = FastAPI()


@app.get("/alive")
async def alive() -> bool:
    return True


@app.get("/get_actors")
async def get_actors() -> list[ActorTrajectory]:
    """Get actor trajectories."""
    actor_trajectories = cl_server.get_actor_trajectories()
    actor_trajectories = [ActorTrajectory.from_torch(act_traj) for act_traj in actor_trajectories]
    return actor_trajectories


@app.post("/update_actors")
async def update_actors(actor_trajectories: list[ActorTrajectory]) -> None:
    """Update actor trajectories (keys correspond to actor uuids)."""
    torch_actor_trajectories = [act_traj.to_torch() for act_traj in actor_trajectories]
    cl_server.update_actor_trajectories(torch_actor_trajectories)


@app.post("/render_image", response_class=Response, responses={200: {"content": {"image/png": {}}}})
async def render_image(data: RenderInput) -> Response:
    torch_pose = torch.tensor(data.pose, dtype=torch.float32)
    render = cl_server.get_image(torch_pose, data.timestamp, data.camera_name)
    return Response(content=_torch_to_png(render), media_type="image/png")


@app.get("/start_time")
async def get_start_time() -> int:
    return int(cl_server.min_time * 1e6)


def _torch_to_png(render: Tensor) -> bytes:
    """Convert a torch tensor to a PNG image."""
    img = Image.fromarray((render * 255).cpu().numpy().astype(np.uint8))
    image_stream = io.BytesIO()
    img.save(image_stream, format="PNG")
    image_bytes = image_stream.getvalue()
    return image_bytes


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    cl_server = tyro.cli(ClosedLoopServer)
    cl_server.main()
    # little hacky to place host and port on the cl_server, but it makes it easier to use tyro
    uvicorn.run(app, host=cl_server.host, port=cl_server.port)
