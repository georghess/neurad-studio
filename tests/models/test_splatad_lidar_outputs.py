"""Regression tests for SplatAD lidar rendering through the sensor API (issues #75, #79)."""

from types import SimpleNamespace

import pytest
import torch

from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import get_lidar_raster_params
from nerfstudio.models.ad_model import ADModel
from nerfstudio.models.splatad import ELEV_CHANNELS_PER_TILE, SplatADModel, grid_tile_elevation_boundaries


def test_splatad_overrides_ray_based_lidar_rendering():
    """The inherited ADModel path builds a RayBundle, which SplatADModel.get_outputs rejects."""
    assert SplatADModel.get_outputs_for_lidar is not ADModel.get_outputs_for_lidar


def test_lidar_raster_params_bracket_the_elevation_channels():
    for lidar_type in LidarType:
        try:
            boundaries, mapping, azimuth_resolution = get_lidar_raster_params(lidar_type)
        except (KeyError, NotImplementedError, ValueError):
            continue  # lidar types without a registered elevation mapping
        assert azimuth_resolution > 0
        assert torch.all(boundaries[1:] > boundaries[:-1]), "tile boundaries must be increasing"
        assert boundaries[0] < mapping.min() and boundaries[-1] > mapping.max()


def test_splatad_advertises_rasterized_lidar_to_the_viewer():
    assert SplatADModel.renders_lidar_by_rasterization is True
    assert not getattr(ADModel, "renders_lidar_by_rasterization", False)


def test_lidars_accept_azimuth_elevation_grids():
    """The viewer builds a Lidars on a regular grid; azimuths/elevations are [*num_lidars, n, 1] fields."""

    lidar = Lidars(
        lidar_to_worlds=torch.eye(4)[:3][None],
        times=torch.tensor([0.0]),
        azimuths=torch.linspace(0, 6.28, 720)[None, :, None],
        elevations=torch.linspace(-0.3, 0.3, 24)[None, :, None],
    )
    assert lidar.shape == (1,)
    assert lidar.azimuths.shape == (1, 720, 1) and lidar.elevations.shape == (1, 24, 1)


def test_grid_tile_boundaries_bracket_every_tile():
    """Viewer default: 24 beams over -15..15 deg. The last boundary must lie above the top beam
    (the earlier construction stopped at 6.87 deg and dropped the upper tiles)."""
    elevations = torch.linspace(-15.0, 15.0, 24)
    b = grid_tile_elevation_boundaries(elevations)
    assert b.numel() == 24 // ELEV_CHANNELS_PER_TILE + 1
    assert torch.all(b[1:] > b[:-1])
    assert b[0] < elevations[0] and b[-1] > elevations[-1]
    # every channel falls inside exactly one tile
    tile = torch.bucketize(elevations, b) - 1
    assert tile.min() == 0 and tile.max() == 24 // ELEV_CHANNELS_PER_TILE - 1
    assert torch.equal(tile, torch.arange(24) // ELEV_CHANNELS_PER_TILE)


@pytest.mark.parametrize("raster_source", ["batch", "metadata"])
def test_repeated_lidar_render_preserves_cached_scan(monkeypatch, raster_source):
    """Exercise real rolling-shutter compensation with a deterministic CPU rasterizer."""
    actor_times = []

    def adjusted_means(means, times, ids):
        actor_times.append(times.clone())
        return means, torch.zeros_like(means)

    def rasterize(**kwargs):
        shape = kwargs["raster_pts"].shape[:-1]
        depth = torch.ones(*shape, 1)
        return torch.cat([torch.zeros_like(depth), depth], dim=-1), depth, None, {"median_depths": depth}

    monkeypatch.setattr("nerfstudio.models.splatad.lidar_rasterization", rasterize)
    decoder = torch.nn.Linear(4, 2)
    decoder.out_dim = 2
    model = SimpleNamespace(
        device=torch.device("cpu"),
        training=False,
        config=SimpleNamespace(
            use_camopt_in_eval=False,
            rasterize_mode="classic",
            compensate_rs_lidar=True,
            radius_clip_lidar=0.0,
            line_of_sight_lambda=0.0,
            line_of_sight_dist=1.0,
            use_absgrad=False,
        ),
        camera_velocity_optimizer=SimpleNamespace(
            apply_to_camera_velocity=lambda *args, **kwargs: torch.tensor([[10.0, 0, 0, 0, 0, 0]])
        ),
        features_rest=torch.zeros(1, 1),
        means=torch.zeros(1, 3),
        id=torch.zeros(1),
        quats=torch.tensor([[1.0, 0, 0, 0]]),
        scales=torch.zeros(1, 3),
        opacities=torch.zeros(1, 1),
        last_size=(1, 1, -1),
        lidar_decoder=decoder,
        _get_actor_adjusted_means=adjusted_means,
        _get_appearance_embedding=lambda sensor, features: features[..., :0],
    )
    model.get_lidar_outputs = lambda lidar: SplatADModel.get_lidar_outputs(model, lidar)
    model.get_outputs = lambda lidar: SplatADModel.get_outputs(model, lidar)
    raster = torch.tensor([[[[0.0, 0.0, 1.0, -0.1, 0.5], [0.0, 0.0, 1.0, 0.0, 0.5]]]])
    original = raster.clone()
    raster_metadata = {
        "raster_pts": raster,
        "elevation_boundaries": torch.tensor([-1.0, 1.0]),
        "azimuth_resolution": 1.0,
    }
    metadata = {"linear_velocities_local": torch.tensor([[10.0, 0, 0]])}
    batch = raster_metadata if raster_source == "batch" else None
    lidar = Lidars(lidar_to_worlds=torch.eye(4)[:3][None], times=torch.tensor([1.0]), metadata=metadata)
    if raster_source == "metadata":
        lidar.metadata.update(raster_metadata)

    first, _ = SplatADModel.get_outputs_for_lidar(model, lidar, batch)
    second, _ = SplatADModel.get_outputs_for_lidar(model, lidar, batch)

    torch.testing.assert_close(raster, original)
    torch.testing.assert_close(first["points"], second["points"])
    torch.testing.assert_close(first["points"][..., 0], torch.tensor([[[0.0, 1.0]]]))
    torch.testing.assert_close(actor_times[0], torch.full_like(actor_times[0], 0.95))
    torch.testing.assert_close(actor_times[1], actor_times[0])
