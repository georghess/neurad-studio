"""Regression tests for SplatAD lidar rendering through the sensor API (issues #75, #79)."""

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
