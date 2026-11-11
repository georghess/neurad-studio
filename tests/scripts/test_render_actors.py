"""Actor removal must select every requested actor at every selected timestamp."""

from types import SimpleNamespace

import pytest
import torch

from nerfstudio.scripts.render import modify_actors


@pytest.mark.parametrize("actor_indices", [None, [0, 2], []])
@pytest.mark.parametrize("removal_time", [-1.0, 1.0, 4.0])
def test_actor_removal_uses_cartesian_time_and_actor_selection(actor_indices, removal_time):
    actors = SimpleNamespace(
        actor_positions=torch.zeros(4, 3, 3),
        unique_timestamps=torch.arange(4, dtype=torch.float32),
        actor_present_at_time=torch.ones(4, 3, dtype=torch.bool),
    )
    pipeline = SimpleNamespace(model=SimpleNamespace(device="cpu", dynamic_actors=actors))

    modify_actors(pipeline, (0.0, 0.0, 0.0), removal_time, None, actor_indices)

    expected = torch.ones(4, 3, dtype=torch.bool)
    for time in range(4):
        for actor in range(3):
            if time > removal_time and (actor_indices is None or actor in actor_indices):
                expected[time, actor] = False
    torch.testing.assert_close(actors.actor_present_at_time, expected)
