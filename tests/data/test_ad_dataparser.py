"""Sensor-motion regressions that do not require an optional dataset SDK."""

from types import SimpleNamespace

import pytest
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars
from nerfstudio.data.dataparsers.ad_dataparser import ADDataParser, _get_world_to_mean_transform


def make_sensors(poses, times, sensor_idxs):
    cameras = Cameras(
        camera_to_worlds=poses.clone(),
        fx=1.0,
        fy=1.0,
        cx=1.0,
        cy=1.0,
        times=times,
        metadata={"sensor_idxs": sensor_idxs},
    )
    lidars = Lidars(lidar_to_worlds=poses.clone(), times=times, metadata={"sensor_idxs": sensor_idxs})
    return cameras, lidars


def test_single_frame_sensor_does_not_prevent_other_sensor_velocities():
    poses = torch.eye(4)[:3].repeat(3, 1, 1)
    poses[:, 0, 3] = torch.tensor([8.0, 1.0, 3.0])
    cameras, lidars = make_sensors(poses, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([[0], [1], [1]]))
    parser = SimpleNamespace(config=SimpleNamespace(rolling_shutter_time=0.0, time_to_center_pixel=0.0))

    ADDataParser._add_sensor_velocities(parser, cameras, lidars)

    expected = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [2.0, 0, 0]])
    for sensor in (cameras, lidars):
        torch.testing.assert_close(sensor.metadata["velocities"], expected)
        torch.testing.assert_close(sensor.metadata["linear_velocities_local"], expected)
        torch.testing.assert_close(sensor.metadata["angular_velocities_local"], torch.zeros(3, 3))


@pytest.mark.parametrize("use_lidar", [True, False])
def test_single_frame_centering_maps_sensor_pose_to_origin(use_lidar):
    poses = torch.eye(4)[:3][None].clone()
    poses[0, :, 3] = torch.tensor([8.0, -2.0, 1.0])
    cameras, lidars = make_sensors(poses, torch.tensor([0.0]), torch.tensor([[0]]))
    if not use_lidar:
        _, lidars = make_sensors(poses[:0], torch.empty(0), torch.empty((0, 1), dtype=torch.long))

    transform = _get_world_to_mean_transform(cameras, lidars)

    torch.testing.assert_close(transform[:, :3], torch.eye(3))
    torch.testing.assert_close(transform[:, :3] @ poses[0, :, 3] + transform[:, 3], torch.zeros(3))
