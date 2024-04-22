# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Code for sampling pixels.
"""

import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.utils.pixel_sampling_utils import erode_mask


@dataclass
class PixelSamplerConfig(InstantiateConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""
    ignore_mask: bool = False
    """Whether to ignore the masks when sampling."""
    fisheye_crop_radius: Optional[float] = None
    """Set to the radius (in pixels) for fisheye cameras."""
    rejection_sample_mask: bool = True
    """Whether or not to use rejection sampling when sampling images with masks"""
    max_num_iterations: int = 100
    """If rejection sampling masks, the maximum number of times to sample"""


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PixelSamplerConfig

    def __init__(self, config: PixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        # Possibly override some values if they are present in the kwargs dictionary
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.config.is_equirectangular = self.kwargs.get("is_equirectangular", self.config.is_equirectangular)
        self.config.fisheye_crop_radius = self.kwargs.get("fisheye_crop_radius", self.config.fisheye_crop_radius)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        """
        Naive pixel sampler, uniformly samples across all possible pixels of all possible images.

        Args:
            batch_size: number of samples in a batch
            num_images: number of images to sample over
            mask: mask of possible pixels in an image to sample from.
        """
        indices = (
            torch.rand((batch_size, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            if self.config.rejection_sample_mask:
                num_valid = 0
                for _ in range(self.config.max_num_iterations):
                    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
                    chosen_indices_validity = mask[..., 0][c, y, x].bool()
                    num_valid = int(torch.sum(chosen_indices_validity).item())
                    if num_valid == batch_size:
                        break
                    else:
                        replacement_indices = (
                            torch.rand((batch_size - num_valid, 3), device=device)
                            * torch.tensor([num_images, image_height, image_width], device=device)
                        ).long()
                        indices[~chosen_indices_validity] = replacement_indices

                if num_valid != batch_size:
                    warnings.warn(
                        """
                        Masked sampling failed, mask is either empty or mostly empty.
                        Reverting behavior to non-rejection sampling. Consider setting
                        pipeline.datamanager.pixel-sampler.rejection-sample-mask to False
                        or increasing pipeline.datamanager.pixel-sampler.max-num-iterations
                        """
                    )
                    self.config.rejection_sample_mask = False
                    nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                    chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                    indices = nonzero_indices[chosen_indices]
            else:
                nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=batch_size)
                indices = nonzero_indices[chosen_indices]

        return indices

    def sample_method_equirectangular(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            # Note: if there is a mask, sampling reduces back to uniform sampling, which gives more
            # sampling weight to the poles of the image than the equators.
            # TODO(kevinddchen): implement the correct mask-sampling method.

            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # We sample theta uniformly in [0, 2*pi]
            # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
            # This is done by inverse transform sampling.
            # http://corysimon.github.io/articles/uniformdistn-on-sphere/
            num_images_rand = torch.rand(batch_size, device=device)
            phi_rand = torch.acos(1 - 2 * torch.rand(batch_size, device=device)) / torch.pi
            theta_rand = torch.rand(batch_size, device=device)
            indices = torch.floor(
                torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
                * torch.tensor([num_images, image_height, image_width], device=device)
            ).long()

        return indices

    def sample_method_fisheye(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, torch.Tensor) and not self.config.ignore_mask:
            indices = self.sample_method(batch_size, num_images, image_height, image_width, mask=mask, device=device)
        else:
            # Rejection sampling.
            valid: Optional[torch.Tensor] = None
            indices = None
            while True:
                samples_needed = batch_size if valid is None else int(batch_size - torch.sum(valid).item())

                # Check if done!
                if samples_needed == 0:
                    break

                rand_samples = torch.rand((samples_needed, 2), device=device)
                # Convert random samples to radius and theta.
                radii = self.config.fisheye_crop_radius * torch.sqrt(rand_samples[:, 0])
                theta = 2.0 * torch.pi * rand_samples[:, 1]

                # Convert radius and theta to x and y.
                x = (radii * torch.cos(theta) + image_width // 2).long()
                y = (radii * torch.sin(theta) + image_height // 2).long()
                sampled_indices = torch.stack(
                    [torch.randint(0, num_images, size=(samples_needed,), device=device), y, x], dim=-1
                )

                # Update indices.
                if valid is None:
                    indices = sampled_indices
                    valid = (
                        (sampled_indices[:, 1] >= 0)
                        & (sampled_indices[:, 1] < image_height)
                        & (sampled_indices[:, 2] >= 0)
                        & (sampled_indices[:, 2] < image_width)
                    )
                else:
                    assert indices is not None
                    not_valid = ~valid
                    indices[not_valid, :] = sampled_indices
                    valid[not_valid] = (
                        (sampled_indices[:, 1] >= 0)
                        & (sampled_indices[:, 1] < image_height)
                        & (sampled_indices[:, 2] >= 0)
                        & (sampled_indices[:, 2] < image_width)
                    )
            assert indices is not None

        assert indices.shape == (batch_size, 3)
        return indices

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        if "mask" in batch:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
            else:
                indices = self.sample_method(
                    num_rays_per_batch, num_images, image_height, image_width, mask=batch["mask"], device=device
                )
        else:
            if self.config.is_equirectangular:
                indices = self.sample_method_equirectangular(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            elif self.config.fisheye_crop_radius is not None:
                indices = self.sample_method_fisheye(
                    num_rays_per_batch, num_images, image_height, image_width, device=device
                )
            else:
                indices = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []
        all_depth_images = []

        # find the optimal number of rays per image such that it is divisible by 2 and sums to the total number of rays
        num_rays_per_image = num_rays_per_batch / num_images
        residual = num_rays_per_image % 2
        num_rays_per_image_under = int(num_rays_per_image - residual)
        num_rays_per_image_over = int(num_rays_per_image_under + 2)
        num_images_under = math.ceil(num_images * (1 - residual / 2))
        num_images_over = num_images - num_images_under
        num_rays_per_image = num_images_under * [num_rays_per_image_under] + num_images_over * [num_rays_per_image_over]
        num_rays_per_image[-1] += num_rays_per_batch - sum(num_rays_per_image)

        if "mask" in batch:
            for i, num_rays in enumerate(num_rays_per_image):
                image_height, image_width, _ = batch["image"][i].shape

                indices = self.sample_method(
                    num_rays, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                if "depth_image" in batch:
                    all_depth_images.append(batch["depth_image"][i][indices[:, 1], indices[:, 2]])

        else:
            for i, num_rays in enumerate(num_rays_per_image):
                image_height, image_width, _ = batch["image"][i].shape

                if self.config.is_equirectangular:
                    indices = self.sample_method_equirectangular(num_rays, 1, image_height, image_width, device=device)
                else:
                    indices = self.sample_method(num_rays, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                if "depth_image" in batch:
                    all_depth_images.append(batch["depth_image"][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key != "image_idx" and key != "image" and key != "mask" and key != "depth_image" and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)
        if "depth_image" in batch:
            collated_batch["depth_image"] = torch.cat(all_depth_images, dim=0)

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            pixel_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


@dataclass
class PatchPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: PatchPixelSampler)
    """Target class to instantiate."""
    patch_size: int = 32
    """Side length of patch. This must be consistent in the method
    config in order for samples to be reshaped into patches correctly."""


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: PatchPixelSamplerConfig

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)

    # overrides base method
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            sub_bs = batch_size // (self.config.patch_size**2)
            half_patch_size = int(self.config.patch_size / 2)
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=half_patch_size)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=sub_bs)
            indices = nonzero_indices[chosen_indices]

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys - half_patch_size
            indices[:, ..., 2] += xxs - half_patch_size

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)
        else:
            sub_bs = batch_size // (self.config.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
                device=device,
            )

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices


@dataclass
class LidarPointSamplerConfig(PixelSamplerConfig):
    """Config dataclass for LidarPointSampler."""

    _target: Type = field(default_factory=lambda: LidarPointSampler)
    """Target class to instantiate."""


class LidarPointSampler(PixelSampler):
    """Samples point from a point cloud."""

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
        a list.

        We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
        The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
        since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """

        device = batch["lidar"][0].device
        num_lidars = len(batch["lidar"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_lidars = []

        num_rays_in_batch = num_rays_per_batch // num_lidars
        for i in range(num_lidars):
            if i == num_lidars - 1:
                num_rays_in_batch = num_rays_per_batch - (num_lidars - 1) * num_rays_in_batch
            num_points, _ = batch["lidar"][i].shape
            # Use pixel sampler method with a "height" of num_points and a "width" of 1
            indices = self.sample_method(num_rays_in_batch, 1, num_points, 1, device=device)
            indices = indices[:, :-1]
            indices[:, 0] = i
            all_indices.append(indices)
            all_lidars.append(batch["lidar"][i][indices[:, 1]])

        indices = torch.cat(all_indices, dim=0)

        c, n = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, n] for key, value in batch.items() if key not in ("lidar", "lidar_idx") and value is not None
        }
        collated_batch["lidar"] = torch.cat(all_lidars, dim=0)

        # Needed to correct the random indices to their actual lidar idx locations.
        indices[:, 0] = batch["lidar_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        if keep_full_image:
            raise NotImplementedError("keep_full_image not implemented for lidar")

        return collated_batch

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """
        device = batch["lidar"].device
        num_lidars = len(batch["points_per_lidar"])
        shuffle_lidars = torch.randperm(num_lidars, device=device)

        num_rays_per_lidar = math.ceil(num_rays_per_batch / num_lidars)

        n_points_per_lidar = batch["points_per_lidar"].to(device).to(torch.int64)
        cum_points_per_lidar = torch.zeros((num_lidars,), device=device, dtype=torch.int64)
        cum_points_per_lidar[1:] = torch.cumsum(n_points_per_lidar, dim=0)[:-1]
        cum_points_per_lidar = cum_points_per_lidar.view(num_lidars, 1)
        point_indices = torch.rand((num_lidars, num_rays_per_lidar), device=device, dtype=torch.float64)
        point_indices = torch.floor(point_indices * n_points_per_lidar.view(num_lidars, 1)).long()
        lidar_indices = torch.arange(num_lidars, device=device).unsqueeze(1).repeat(1, num_rays_per_lidar)

        # shuffle the lidars to avoid dropping the same each time
        lidar_indices = lidar_indices[shuffle_lidars]
        point_indices = point_indices[shuffle_lidars]
        cum_points_per_lidar = cum_points_per_lidar[shuffle_lidars]

        indices = torch.stack((lidar_indices.flatten(), point_indices.flatten()), dim=-1)[:num_rays_per_batch]

        flat_indices = (point_indices + cum_points_per_lidar).flatten()
        all_lidars = batch["lidar"][flat_indices][:num_rays_per_batch]

        c, n = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, n]
            for key, value in batch.items()
            if key not in ("lidar", "lidar_idx", "points_per_lidar") and value is not None
        }
        collated_batch["lidar"] = all_lidars

        # Needed to correct the random indices to their actual lidar idx locations.
        indices[:, 0] = batch["lidar_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        if keep_full_image:
            raise NotImplementedError("keep_full_image not implemented for lidar")

        return collated_batch

    def sample(self, image_batch: Dict):
        """Sample an -image- lidar batch and return a -pixel- point batch.

        Args:
            image_batch: batch of -images- lidars to sample from
        """
        if isinstance(image_batch["lidar"], list):
            image_batch = dict(image_batch.items())  # copy the dictionary so we don't modify the original
            point_batch = self.collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        elif isinstance(image_batch["lidar"], torch.Tensor):
            point_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['lidar'] must be a list or torch.Tensor")
        return point_batch


@dataclass
class ScaledPatchSamplerConfig(PixelSamplerConfig):
    """Config dataclass for ScaledPatchSampler."""

    _target: Type = field(default_factory=lambda: ScaledPatchSampler)
    """Target class to instantiate."""
    patch_scale: int = 1
    """The upsampling ratio between sampled rays and pixel ground truths."""
    patch_size: int = 1
    """The size of sampled patches."""


class ScaledPatchSampler(PixelSampler):
    def __init__(
        self,
        config: ScaledPatchSamplerConfig,
        num_rays_per_batch: int,
        keep_full_image: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(config, num_rays_per_batch=num_rays_per_batch, keep_full_image=keep_full_image, **kwargs)
        self.config: ScaledPatchSamplerConfig
        self.patch_scale = self.config.patch_scale
        self.patch_size = self.config.patch_size
        self.sampling_weights: Optional[Tensor] = None
        self.sampling_scale: int = 1

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """
        self._check_extra_keys(batch)
        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape
        image = batch["image"]
        rgb_size = self.patch_size * self.patch_scale
        num_patches = num_rays_per_batch // (self.patch_size**2)

        patch_center_indices = self.sample_method(
            num_patches, num_images, image_height, image_width, rgb_size, self.sampling_weights, device
        )
        ray_indices, img_patches = self._patches_from_centers(image, patch_center_indices, rgb_size, device)
        ray_indices[:, 0] = batch["image_idx"][ray_indices[:, 0]]

        collated_batch = {
            "indices": ray_indices,
            "image": img_patches,
        }
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        self._check_extra_keys(batch)
        device = batch["image"][0].device
        num_images = len(batch["image"])
        rgb_size = self.patch_size * self.patch_scale
        num_patches = num_rays_per_batch // (self.patch_size**2)

        assert self.sampling_weights is None, "sampling_weights not supported for ScaledPatchSampler in list mode"
        img_indices = torch.randint(0, num_images, (num_patches,), device=device)  # TODO: use sampling weights
        img_indices, img_counts = torch.unique(img_indices, return_counts=True)

        all_img_patches = []
        all_ray_indices = []
        for img_idx, patches_in_img in zip(img_indices, img_counts):
            image = batch["image"][img_idx].unsqueeze(0)
            _, height, width, _ = image.shape
            patch_center_indices = self.sample_method(
                patches_in_img, 1, height, width, rgb_size, sampling_weights=None, device=device
            )
            ray_indices, img_patches = self._patches_from_centers(image, patch_center_indices, rgb_size, device)
            all_img_patches.append(img_patches)
            ray_indices[:, 0] = img_idx
            all_ray_indices.append(ray_indices)
        ray_indices = torch.cat(all_ray_indices, dim=0)
        img_patches = torch.cat(all_img_patches, dim=0)
        ray_indices[:, 0] = batch["image_idx"][ray_indices[:, 0]]  # use the "global" image idx

        collated_batch = {"indices": ray_indices, "image": img_patches}
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]
        return collated_batch

    def _patches_from_centers(
        self,
        image: torch.Tensor,
        patch_center_indices: torch.Tensor,
        rgb_size: int,
        device: Union[torch.device, str] = "cpu",
    ):
        """Convert patch center coordinates to the full set of ray indices and image patches."""
        offsets = torch.arange(-(rgb_size // 2), (rgb_size // 2) + rgb_size % 2, device=device)
        zeros = offsets.new_zeros((rgb_size, rgb_size))
        relative_indices = torch.stack((zeros, *torch.meshgrid(offsets, offsets, indexing="ij")), dim=-1)[
            None
        ]  # 1xKxKx3
        rgb_indices = patch_center_indices[:, None, None] + relative_indices  # NxKxKx3
        ray_indices = rgb_indices[
            :, self.patch_scale // 2 :: self.patch_scale, self.patch_scale // 2 :: self.patch_scale
        ]  # NxKfxKfx3
        ray_indices = ray_indices.reshape(-1, 3)  # (N*Kf*Kf)x3
        img_patches = image[rgb_indices[..., 0], rgb_indices[..., 1], rgb_indices[..., 2]]
        return ray_indices, img_patches

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        rgb_size: int,
        sampling_weights: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ):
        """Samples (img, y, x) indices from the image batch, optionally according to per-pixel weights."""
        if sampling_weights is None:
            cropped_h, cropped_w = image_height - rgb_size + 1, image_width - rgb_size + 1
            crop = rgb_size // 2
            centers = super().sample_method(batch_size, num_images, cropped_h, cropped_w, device=device)
            centers[:, 1:] = centers[:, 1:] + crop
        else:
            sampled_flat_indices = torch.multinomial(sampling_weights, batch_size, replacement=True)
            scaled_h, scaled_w = self.sampling_shape[-2:]
            image_indices = sampled_flat_indices // (scaled_h * scaled_w)
            h_indices = (sampled_flat_indices % (scaled_h * scaled_w)) // scaled_w
            w_indices = (sampled_flat_indices % (scaled_h * scaled_w)) % scaled_w
            # Apply random jitter to the centers to compensate for sampling at a lower scale
            if self.sampling_scale > 1:
                jitter = torch.randint(0, self.sampling_scale, (batch_size, 2), device=device)
                h_indices = h_indices * self.sampling_scale + jitter[:, 0]
                w_indices = w_indices * self.sampling_scale + jitter[:, 1]
            # move the centers if they are outside the crop range
            h_indices = h_indices.clip(min=rgb_size // 2, max=image_height - rgb_size // 2 - 1)
            w_indices = w_indices.clip(min=rgb_size // 2, max=image_width - rgb_size // 2 - 1)
            centers = torch.stack((image_indices, h_indices, w_indices), dim=-1)
        return centers  # img_idx, pixel_h, pixel_w

    def update_sampling_weights(
        self, scores: Float[Tensor, "n h w"], ratio_uniform: float = 0.5, sampling_scale: int = 1
    ):
        """Update the sampling weights based on the scores."""
        # Downsample scores for efficient sampling
        scores = torch.nn.functional.avg_pool2d(scores, sampling_scale, sampling_scale, ceil_mode=True)
        self.sampling_scale = sampling_scale
        self.sampling_shape = scores.shape
        scores = scores.flatten()
        score_probs = scores / scores.sum()
        self.sampling_weights = ratio_uniform / scores.numel() + (1 - ratio_uniform) * score_probs

    def _check_extra_keys(self, batch):
        """Check for extra keys in the batch."""
        extra_keys = set(batch.keys()) - {"image", "image_idx"}
        if extra_keys:
            raise NotImplementedError("Patch sampler not implemented for extra_keys")


@dataclass
class PairPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PairPixelSampler."""

    _target: Type = field(default_factory=lambda: PairPixelSampler)
    """Target class to instantiate."""
    radius: int = 2
    """max distance between pairs of pixels."""


class PairPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples pair of pixels from 'image_batch's. Samples pairs of pixels from
        from the images randomly within a 'radius' distance apart. Useful for pair-based losses.

    Args:
        config: the PairPixelSamplerConfig used to instantiate class
    """

    def __init__(self, config: PairPixelSamplerConfig, **kwargs) -> None:
        self.config = config
        self.radius = self.config.radius
        super().__init__(self.config, **kwargs)
        self.rays_to_sample = self.config.num_rays_per_batch // 2

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: Optional[int],
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        rays_to_sample = self.rays_to_sample
        if batch_size is not None:
            assert (
                int(batch_size) % 2 == 0
            ), f"PairPixelSampler can only return batch sizes in multiples of two (got {batch_size})"
            rays_to_sample = batch_size // 2

        if isinstance(mask, Tensor) and not self.config.ignore_mask:
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=self.radius)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=rays_to_sample)
            indices = nonzero_indices[chosen_indices]
        else:
            s = (rays_to_sample, 1)
            ns = torch.randint(0, num_images, s, dtype=torch.long, device=device)
            hs = torch.randint(self.radius, image_height - self.radius, s, dtype=torch.long, device=device)
            ws = torch.randint(self.radius, image_width - self.radius, s, dtype=torch.long, device=device)
            indices = torch.concat((ns, hs, ws), dim=1)

        pair_indices = torch.hstack(
            (
                torch.zeros(rays_to_sample, 1, device=device, dtype=torch.long),
                torch.randint(-self.radius, self.radius, (rays_to_sample, 2), device=device, dtype=torch.long),
            )
        )
        pair_indices += indices
        indices = torch.hstack((indices, pair_indices)).view(rays_to_sample * 2, 3)
        return indices
