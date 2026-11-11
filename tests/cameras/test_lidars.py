"""Preserve lidar motion conventions through TensorDataclass operations."""

from dataclasses import replace

import pytest
import torch

from nerfstudio.cameras.lidars import Lidars


@pytest.mark.parametrize("compensated", [False, True])
def test_lidar_operations_preserve_ego_compensation(compensated):
    lidars = Lidars(lidar_to_worlds=torch.eye(4)[:3].repeat(2, 1, 1), assume_ego_compensated=compensated)
    transformed = (
        lidars[:1],
        lidars[torch.tensor([False, True])],
        lidars.to("cpu"),
        lidars.reshape((1, 2)),
        lidars.broadcast_to((3, 2)),
        lidars.cat([lidars]),
        replace(lidars),
    )
    for result in transformed:
        assert result.assume_ego_compensated is compensated
