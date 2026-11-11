"""Byte image caching preserves NeuRAD samples while reducing resident image storage."""

import queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import variable_res_collate
from nerfstudio.data.datamanagers.image_lidar_datamanager import (
    ImageLidarDataManager,
    ImageLidarDataProcessor,
    _cache_images,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import PixelSamplerConfig, ScaledPatchSamplerConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator


@pytest.mark.parametrize("variable_resolution", [False, True])
@pytest.mark.parametrize("patches", [False, True])
def test_byte_cache_preserves_crops_and_sampled_pixels(tmp_path, variable_resolution, patches):
    paths = []
    for idx in range(2):
        path = tmp_path / f"{idx}.png"
        Image.fromarray(np.random.default_rng(idx).integers(0, 256, (8, 10, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    cameras = Cameras(
        camera_to_worlds=torch.eye(4)[:3].repeat(2, 1, 1),
        fx=10.0,
        fy=10.0,
        cx=4.0,
        cy=3.0,
        height=torch.tensor([[6], [4 if variable_resolution else 6]]),
        width=8,
    )
    dataset = InputDataset(DataparserOutputs(image_filenames=paths, cameras=cameras))
    collate = variable_res_collate if variable_resolution else nerfstudio_collate
    byte_cache = _cache_images(dataset, 1, collate, "uint8")
    float_cache = _cache_images(dataset, 1, collate, "float32")
    for byte_image, float_image in zip(byte_cache["image"], float_cache["image"]):
        assert byte_image.dtype == torch.uint8
        torch.testing.assert_close(byte_image.float() / 255, float_image)
        assert byte_image.numel() * byte_image.element_size() * 4 == float_image.numel() * float_image.element_size()
    sampler_config = ScaledPatchSamplerConfig(patch_size=2) if patches else PixelSamplerConfig()
    processor = SimpleNamespace(
        image_dataset=dataset,
        pixel_sampler=sampler_config.setup(num_rays_per_batch=8),
        ray_generator=RayGenerator(cameras),
        cached_images=byte_cache,
    )
    torch.manual_seed(42)
    byte_batch, byte_rays = ImageLidarDataProcessor.get_image_batch_and_ray_bundle(processor)
    processor.cached_images = float_cache
    torch.manual_seed(42)
    float_batch, float_rays = ImageLidarDataProcessor.get_image_batch_and_ray_bundle(processor)
    assert byte_batch["image"].dtype == torch.float32
    torch.testing.assert_close(byte_batch["image"], float_batch["image"])
    torch.testing.assert_close(byte_rays.directions, float_rays.directions)
    torch.testing.assert_close(byte_batch["indices"], float_batch["indices"])


def test_byte_mask_color_uses_byte_scale(tmp_path):
    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(image)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(mask)
    cameras = Cameras(camera_to_worlds=torch.eye(4)[:3][None], fx=4.0, fy=4.0, cx=2.0, cy=2.0, height=4, width=4)
    dataset = InputDataset(
        DataparserOutputs(
            image_filenames=[image],
            mask_filenames=[mask],
            cameras=cameras,
            metadata={"mask_color": [1.0, 0.5, 0.0]},
        )
    )
    byte_image = dataset.get_data(0, image_type="uint8")["image"]
    assert byte_image.dtype == torch.uint8
    torch.testing.assert_close(byte_image[0, 0], torch.tensor([255, 127, 0], dtype=torch.uint8))
    torch.testing.assert_close(byte_image.float() / 255, dataset[0]["image"], rtol=0, atol=1 / 255)


def test_dead_loading_worker_raises_instead_of_waiting_forever():
    class EmptyQueue:
        def get(self, timeout):
            assert timeout > 0
            raise queue.Empty

    manager = SimpleNamespace(use_mp=True, data_queue=EmptyQueue(), data_procs=[SimpleNamespace(exitcode=-9)])
    with pytest.raises(RuntimeError, match="Data-loading worker exited.*-9.*RAM"):
        ImageLidarDataManager._get_from_queue(manager)
