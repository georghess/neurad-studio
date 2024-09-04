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

import base64
import io
import threading
from typing import Literal, Union

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from PIL import Image
from torch import Tensor

from nerfstudio.scripts.closed_loop.models import ActorTrajectory, ImageFormat, RenderInput
from nerfstudio.scripts.closed_loop.server import ClosedLoopServer

app = FastAPI()
render_lock = threading.Lock()


@app.get("/alive")
def alive() -> bool:
    return True


@app.get("/get_actors")
def get_actors() -> list[ActorTrajectory]:
    """Get actor trajectories."""
    actor_trajectories = cl_server.get_actor_trajectories()
    actor_trajectories = [ActorTrajectory.from_torch(act_traj) for act_traj in actor_trajectories]
    return actor_trajectories


@app.post("/update_actors")
def update_actors(actor_trajectories: list[ActorTrajectory]) -> None:
    """Update actor trajectories (keys correspond to actor uuids)."""
    torch_actor_trajectories = [act_traj.to_torch() for act_traj in actor_trajectories]
    cl_server.update_actor_trajectories(torch_actor_trajectories)


@app.post(
    "/render_image",
    response_class=Response,
    responses={200: {"content": {"text/plain": {}, "image/png": {}, "image/jpeg": {}}}},
)
def get_image(data: RenderInput) -> Response:
    torch_pose = torch.tensor(data.pose, dtype=torch.float32)

    with render_lock:
        render = cl_server.get_image(torch_pose, data.timestamp, data.camera_name)

    if data.image_format == ImageFormat.raw:
        return Response(content=_torch_to_bytestr(render), media_type="text/plain")
    elif data.image_format == ImageFormat.png:
        return Response(content=_torch_to_img(render, "png"), media_type="image/png")
    elif data.image_format in (ImageFormat.jpg, ImageFormat.jpeg):
        return Response(content=_torch_to_img(render, "jpeg"), media_type="text/jpeg")
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid image format: {data.image_format}, must be 'raw', 'png', 'jpg', or 'jpeg'"
        )


@app.get("/start_time")
def get_start_time() -> int:
    return int(cl_server.min_time * 1e6)


def _torch_to_bytestr(render: Tensor) -> bytes:
    """Convert a torch tensor to a base64 encoded bytestring."""
    buff = io.BytesIO()
    img = (render * 255).to(torch.uint8).cpu()
    torch.save(img, buff)
    return base64.b64encode(buff.getvalue())


def _torch_to_img(render: Tensor, format: Union[Literal["jpeg"], Literal["png"]]) -> bytes:
    """Convert a torch tensor to a PNG or JPG image."""
    if format not in ("jpeg", "png"):
        raise ValueError(f"Invalid format: {format}")

    img = Image.fromarray((render * 255).cpu().numpy().astype(np.uint8))
    buff = io.BytesIO()
    img.save(buff, format=format.upper())
    return buff.getvalue()


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    cl_server = tyro.cli(ClosedLoopServer)
    cl_server.main()
    # little hacky to place host and port on the cl_server, but it makes it easier to use tyro
    uvicorn.run(app, host=cl_server.host, port=cl_server.port)
